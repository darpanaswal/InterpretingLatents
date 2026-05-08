"""
Gradient-subspace ablation + amplification interventions.

Drop-in replacement for the INLP-based interventions.py, using the
gradient-weighted causal subspace B_t produced by gradient_subspace.py
instead of INLP's label-decoded subspace.

Why this script exists
----------------------
INLP assumes the model's causal directions are linearly recoverable from
a concept label. When that assumption is violated (nonlinear features,
distributed encoding), INLP's nullspace is not the model's nullspace and
the resulting ablation is uninformative. The gradient subspace
    g_{i,t} = ∂L_i / ∂h_{i,t}
    G_t    = stack g_{i,t}          (N, D)
    G_t    = U S V^T                 (SVD)
    B_t    = top-k columns of V      (D, k_t)
makes no linearity assumption: it is the local first-order direction
that actually moves the loss. Projecting onto / off span(B_t) tests the
*architecture's* causal directions directly.

Interventions
-------------
ABLATION  (nullspace projection):
    # P_t = I - B_t B_t^T
    # h_t  <-  P_t h_t
Zeros out the gradient subspace and keeps everything else.

AMPLIFICATION  (subspace-restricted multiplicative scaling):
    # C_t = B_t B_t^T                    projector onto causal subspace
    # h^c_t   = C_t h_t                  causal component
    # h^null_t = h_t - h^c_t             nullspace component
    # h'_t    = h^null_t + alpha * h^c_t  scaled reconstruction

    At alpha=0 this is ablation (remove causal component entirely).
    At alpha=1 this is identity (h' = h).
    At alpha>1 this amplifies the existing causal signal.

    Unlike additive steering (h + alpha * d), this stays on the data
    manifold: both h^c and h^null are components of the original h,
    so scaling h^c by a moderate factor (1.5x, 2x, 3x) produces a
    vector that is still "in distribution" in direction. This
    eliminates the magnitude confound where random directions cause
    more damage than gradient directions simply because they push h
    off-manifold.

    The question this answers: "Is the downstream circuitry wired to
    read the gradient subspace?" If amplifying h^c flips predictions
    while amplifying a random-subspace component does not, the model's
    downstream layers are specifically coupled to the gradient
    directions — even if those directions don't carry enough signal at
    baseline to affect the output (causal wiring without current use).

Random control
--------------
    # B_t^rand = QR(randn(D, k_t)).Q[:, :k_t]
    # C_t^rand = B_t^rand (B_t^rand)^T
Same amplification protocol applied to a rank-matched random subspace.
Isolates "is the effect from the *gradient* directions specifically"
from "is it from any rank-k subspace at all".

Multi-seed random control (--n_random_seeds K)
----------------------------------------------
With K > 1 (default 3), K independent random subspaces are drawn
(seeds = args.seed, args.seed+1, ..., args.seed+K-1). Each is
evaluated independently, then aggregated two ways:
  (i)  Pooled: concat K per-instance vectors → bootstrap over K*N.
  (ii) Per-seed: individual CIs + a cross-seed summary record with
       seed_mean and seed_std.
Paired tests use the pooled random vector against baseline REPEATED
K times (preserving instance pairing across draws).
To reproduce old single-seed behavior, pass --n_random_seeds=1.

Bases handling for rank-0 timesteps
-----------------------------------
gradient_subspace.py emits k_t = 0 at any t whose post-mask gradient
matrix had < 2 nonzero rows (typically t = K because the gold answer
sits at K and there is no h_K to take a gradient for). When k_t = 0
we skip the intervention at that t entirely (pass-through).
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from src.config import BASE_DIR, THOUGHTS
from src.bootstrap_stats import (
    bootstrap_mean,
    paired_bootstrap_diff,
    mcnemar_test,
    report_mean_with_ci,
    save_record,
    BootstrapResult,
)
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    run_intervened_inference_pauseaware,
    load_data,
    deep_convert,
    normalize_text_for_flip,
    make_projection_intervention,
    run_codi_single_alpha,
    _shard_indices,
    _merge_shards,
)


# ═══════════════════════════════════════════════════════════════════
# Building projectors from gradient bases
# ═══════════════════════════════════════════════════════════════════

def load_bases(bases_path):
    """
    Load per-timestep gradient bases saved by gradient_subspace.py.

    bases.npz keys are 'B_t{t}' for t = 0..T-1, each an (D, k_t) array.
    A timestep with rank 0 is stored as an (D, 0) array.

    Returns: dict {t: B_t as np.ndarray (D, k_t)}
    """
    blob = np.load(bases_path)
    bases = {}
    for key in blob.files:
        if not key.startswith("B_t"):
            continue
        t = int(key[len("B_t"):])
        bases[t] = blob[key]
    return bases


def build_subspace_projectors(bases):
    """
    From {t: B_t} build the projectors:
      - nullspace projector P_t = I - B_t B_t^T  (used for ABLATION)
      - subspace projector  C_t = B_t B_t^T      (used for AMPLIFICATION)

    Math:
        # B_t has orthonormal columns (output of np.linalg.svd Vh.T)
        # B_t B_t^T projects onto span(B_t)
        # I - B_t B_t^T projects onto its orthogonal complement

    Rank-0 handling: timesteps with k_t = 0 are OMITTED from the
    returned dicts so that closures pass through h unchanged.

    Returns:
        nullspace: dict {t: P_t (D, D)}        only t with k_t > 0
        subspace : dict {t: C_t (D, D)}        only t with k_t > 0
        ranks    : dict {t: k_t}                all timesteps
    """
    nullspace = {}
    subspace = {}
    ranks = {}
    for t, B_t in bases.items():
        D, k_t = B_t.shape
        ranks[t] = int(k_t)
        if k_t == 0:
            continue
        BBt = B_t @ B_t.T
        subspace[t] = BBt
        nullspace[t] = np.eye(D) - BBt
    return nullspace, subspace, ranks


def build_random_subspace_projectors(ranks, D, seed):
    """
    Build a rank-matched random orthonormal subspace at each timestep.

    For each t with k_t > 0:
        # M = randn(D, k_t)
        # Q, _ = qr(M)               Q has orthonormal columns
        # B_t^rand = Q[:, :k_t]
        # C_t^rand = B_t^rand B_t^rand^T
        # P_t^rand = I - C_t^rand

    Rank-0 timesteps are skipped to match the gradient side exactly.

    Returns: (nullspace_rand, subspace_rand) dicts.
    """
    rng = np.random.default_rng(seed)
    nullspace_rand = {}
    subspace_rand = {}
    eye = np.eye(D)
    for t, k_t in ranks.items():
        if k_t == 0:
            continue
        M = rng.standard_normal((D, k_t))
        Q, _ = np.linalg.qr(M)
        BBt = Q @ Q.T
        subspace_rand[t] = BBt
        nullspace_rand[t] = eye - BBt
    return nullspace_rand, subspace_rand


# ═══════════════════════════════════════════════════════════════════
# Amplification intervention closure
# ═══════════════════════════════════════════════════════════════════

def make_amplification_intervention(concept_projectors, alpha, device):
    """
    Subspace-restricted multiplicative scaling.

    For each thought vector h_t at timestep t:
        # h^c_t    = C_t @ h_t             (causal component)
        # h^null_t = h_t - h^c_t           (nullspace component)
        # h'_t     = h^null_t + alpha * h^c_t

    At alpha=0: h' = h^null (ablation).
    At alpha=1: h' = h      (identity).
    At alpha>1: h^c amplified, h^null preserved.

    Dtype contract: incoming h may be bfloat16 (CODI) or float32.
    We upcast to float32 for the projection, then cast back.
    """
    C_tensors = {
        t: torch.tensor(C, dtype=torch.float32, device=device)
        for t, C in concept_projectors.items()
    }

    def intervention_fn(h, t):
        if t not in C_tensors:
            return h
        C = C_tensors[t]
        orig_dtype = h.dtype

        h_f32 = h.to(torch.float32)

        # h^c = C_t @ h   (project onto causal subspace)
        if h_f32.dim() == 1:
            h_c = C @ h_f32
        else:
            # Batched: h is (B, D), C is (D, D)
            h_c = h_f32 @ C.T

        # h^null = h - h^c
        h_null = h_f32 - h_c

        # h' = h^null + alpha * h^c
        out = h_null + alpha * h_c
        return out.to(orig_dtype)

    return intervention_fn


# ═══════════════════════════════════════════════════════════════════
# Amplification sweep (unbatched, per-alpha)
# ═══════════════════════════════════════════════════════════════════

def run_amplification_sweep(
    ctx, data, concept_projectors, alphas, baseline_texts,
    label_name, instance_indices=None,
):
    """
    Per-alpha unbatched sweep. For each instance and each alpha, run
    inference with the amplification closure and check for flips vs
    the baseline text.

    Uses the ctx-dispatch pattern from temporal_mean_ablation.py:
    CODI goes through run_codi_single_alpha, Coconut/Pause through
    run_intervened_inference_pauseaware. Both accept any (h, t) -> h'
    closure.

    Returns: {alpha: {"n_flipped", "n_total", "flip_rate", "flipped_indices"}}
    """
    if instance_indices is None:
        instance_indices = list(range(len(data)))

    n_total = len(instance_indices)
    flipped_indices = {alpha: [] for alpha in alphas}

    for count, idx in enumerate(instance_indices):
        if count % 100 == 0:
            print(f"    [{label_name}] {count}/{n_total}")

        sample = data[idx]
        base_norm = normalize_text_for_flip(baseline_texts[idx])

        for alpha in alphas:
            intervention_fn = make_amplification_intervention(
                concept_projectors, alpha, ctx["device"],
            )
            r = _run_intervened(ctx, sample, intervention_fn)
            text = r.get("text", r.get("predicted", ""))
            if normalize_text_for_flip(text) != base_norm:
                flipped_indices[alpha].append(idx)

    results = {}
    for alpha in alphas:
        n_flipped = len(flipped_indices[alpha])
        results[alpha] = {
            "n_flipped": n_flipped,
            "n_total": n_total,
            "flip_rate": n_flipped / max(n_total, 1),
            "flipped_indices": flipped_indices[alpha],
        }
        print(f"    [{label_name}] Alpha {alpha}: "
              f"{n_flipped}/{n_total} flipped ({results[alpha]['flip_rate']:.1%})")
    return results


def _run_intervened(ctx, sample, intervention_fn):
    """Dispatch to CODI or Coconut/Pause inference."""
    if ctx["is_codi"]:
        return run_codi_single_alpha(
            ctx["codi_dict"], sample, ctx["n_thoughts"],
            ctx["device"], intervention_fn, task=ctx["task"],
        )
    return run_intervened_inference_pauseaware(
        ctx["coconut_model"], ctx["base_model"], ctx["tokenizer"],
        ctx["end_id"], sample, ctx["n_thoughts"], ctx["device"],
        intervention_fn,
        start_id=ctx["start_id"], latent_id=ctx["latent_id"],
        task=ctx["task"],
    )


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU: amplification-aware worker + orchestrator
# ═══════════════════════════════════════════════════════════════════

def _amplification_worker(
    rank, world_size, task, model_name, n_thoughts,
    data, alphas, baseline_texts,
    concept_projectors_grad_np, concept_projectors_rand_np,
    return_queue,
):
    """
    Per-GPU worker for the amplification sweep. Loads its own model on
    cuda:{rank}, runs the unbatched amplification sweep on its shard,
    sends partial flip counts back via queue.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    is_codi = (model_name == "codi")
    if is_codi:
        codi_dict = setup_codi_model(task, device)
        ctx = {
            "is_codi": True, "codi_dict": codi_dict,
            "coconut_model": None, "base_model": None, "tokenizer": None,
            "start_id": None, "latent_id": None, "end_id": None,
            "n_thoughts": n_thoughts, "device": device, "task": task,
        }
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device)
        ctx = {
            "is_codi": False, "codi_dict": None,
            "coconut_model": coconut_model, "base_model": base_model,
            "tokenizer": tokenizer,
            "start_id": start_id, "latent_id": latent_id, "end_id": end_id,
            "n_thoughts": n_thoughts, "device": device, "task": task,
        }

    indices = _shard_indices(len(data), world_size, rank)
    print(f"[rank {rank}] processing {len(indices)} instances on {device}")

    grad_part = run_amplification_sweep(
        ctx, data, concept_projectors_grad_np, alphas,
        baseline_texts, label_name=f"GRAD r{rank}",
        instance_indices=indices,
    )
    rand_part = run_amplification_sweep(
        ctx, data, concept_projectors_rand_np, alphas,
        baseline_texts, label_name=f"Rand r{rank}",
        instance_indices=indices,
    )
    return_queue.put({"rank": rank, "grad": grad_part, "rand": rand_part})


def run_amplification_multigpu(
    task, model_name, n_thoughts, data, alphas, baseline_texts,
    concept_projectors_grad, concept_projectors_rand, n_gpus,
):
    """Spawn n_gpus workers for the amplification sweep."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_amplification_worker,
            args=(rank, n_gpus, task, model_name, n_thoughts,
                  data, alphas, baseline_texts,
                  concept_projectors_grad, concept_projectors_rand, q),
        )
        p.start()
        procs.append(p)

    shards = []
    for _ in range(n_gpus):
        shards.append(q.get())
    for p in procs:
        p.join()

    shards.sort(key=lambda s: s["rank"])
    grad_merged = _merge_shards(
        [s["grad"] for s in shards], alphas, len(data))
    rand_merged = _merge_shards(
        [s["rand"] for s in shards], alphas, len(data))
    return grad_merged, rand_merged


# ═══════════════════════════════════════════════════════════════════
# Pretty-printing the rank table
# ═══════════════════════════════════════════════════════════════════

def print_rank_table(ranks, T):
    """Show k_t per t and flag any rank-0 timesteps as pass-through."""
    print("\n" + "=" * 60)
    print("GRADIENT SUBSPACE RANKS")
    print("=" * 60)
    print(f"  {'t':>3}  {'k_t':>5}  status")
    print(f"  {'-'*3}  {'-'*5}  {'-'*30}")
    n_skipped = 0
    for t in range(T):
        k_t = ranks.get(t, 0)
        status = "intervene" if k_t > 0 else "PASS-THROUGH (rank 0)"
        if k_t == 0:
            n_skipped += 1
        print(f"  {t:>3}  {k_t:>5}  {status}")
    if n_skipped:
        print(f"  ({n_skipped} timestep(s) will be left untouched)")
    print()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-subspace ablation + amplification. Replaces "
                    "INLP-based interventions with bases from "
                    "gradient_subspace.py."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        default="coconut")
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=1000,
                    help="Bootstrap iterations for all CIs.")
    parser.add_argument("--alpha_sweep", type=str,
                        default="0,0.5,1,1.5,2,5,10,25,50,100",
                        help="Comma-separated scaling factors. "
                             "alpha=0 is ablation, alpha=1 is identity, "
                             "alpha>1 amplifies the causal component.")
    parser.add_argument("--bases_path", type=str, default=None,
                        help="Path to bases.npz produced by "
                             "gradient_subspace.py. Default: "
                             "BASE_DIR/outputs/gradient_geometry/"
                             "<task>/<model>/bases.npz")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="If >1, shard instances across GPUs 0..n_gpus-1.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the rank-matched random control basis.")
    parser.add_argument("--n_random_seeds", type=int, default=3,
                        help="Number of independent random subspace draws. "
                             "Each uses seed+k for k=0..K-1. Default 3. "
                             "Use 1 to reproduce old single-seed behavior.")
    args = parser.parse_args()

    if args.n_random_seeds < 1:
        parser.error("--n_random_seeds must be >= 1")
    K = args.n_random_seeds

    is_codi = (args.model == "codi")

    # ── Output dir ─────────────────────────────────────────────────
    output_dir = (Path(args.output_dir) if args.output_dir else
                  BASE_DIR / "outputs" / "grad_subspace" / args.task / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Task: {args.task}  Model: {args.model}")
    print(f"[INFO] Output dir: {output_dir}")

    # ── Load data + thoughts cache (for regime diagnostic) ──────────
    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Instances: {len(data)}")

    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    thoughts = torch.load(thoughts_path, map_location="cpu",
                          weights_only=False)["thoughts"]
    D = thoughts.shape[2]
    T = thoughts.shape[1]

    # ── Load gradient bases ────────────────────────────────────────
    bases_path = (Path(args.bases_path) if args.bases_path else
                  BASE_DIR / "outputs" / "gradient_geometry"
                  / args.task / args.model / "bases.npz")
    if not bases_path.exists():
        raise FileNotFoundError(
            f"bases.npz not found at {bases_path}.\n"
            f"Run gradient_subspace.py for {args.task}/{args.model} first."
        )
    bases = load_bases(bases_path)
    print(f"[INFO] Loaded bases from {bases_path}")

    for t, B_t in bases.items():
        if B_t.shape[0] != D:
            raise ValueError(
                f"Basis dim mismatch at t={t}: B_t has D={B_t.shape[0]} "
                f"but thoughts have D={D}."
            )

    # Build projectors (gradient + matched-rank random control).
    nullspace_grad, subspace_grad, ranks = build_subspace_projectors(bases)
    # Random projectors are built per-seed inside the eval loops below.
    print_rank_table(ranks, T)
    print(f"[INFO] n_random_seeds: {K}  (seeds {args.seed}..{args.seed+K-1})")

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # ── Load model ─────────────────────────────────────────────────
    if is_codi:
        codi_dict = setup_codi_model(args.task, args.device)
        coconut_model = base_model = tokenizer = None
        latent_id = start_id = end_id = None
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device)

    # Build the ctx dict for dispatch.
    ctx = {
        "is_codi": is_codi,
        "codi_dict": codi_dict,
        "coconut_model": coconut_model,
        "base_model": base_model,
        "tokenizer": tokenizer,
        "start_id": start_id,
        "latent_id": latent_id,
        "end_id": end_id,
        "n_thoughts": args.n_thoughts,
        "device": args.device,
        "task": args.task,
    }

    # CI output path (JSONL, one record per metric).
    cis_jsonl = str(output_dir / "bootstrap_cis.jsonl")
    ci_ctx = {"task": args.task, "model": args.model}

    # ── Helper: run eval loop, return per-instance correctness vector ─
    def _eval_correctness(intervention_fn, label):
        """Run inference with `intervention_fn`, return list[int] (0/1)."""
        correct_vec = []
        for idx, sample in enumerate(data):
            if idx % 100 == 0:
                print(f"    [{label}] {idx}/{len(data)}")
            r = _run_intervened(ctx, sample, intervention_fn)
            correct_vec.append(int(r["is_correct"]))
        acc = sum(correct_vec) / max(len(correct_vec), 1)
        print(f"    [{label}] Accuracy: {sum(correct_vec)}/{len(data)}"
              f" = {acc:.1%}")
        return correct_vec

    # ── Baseline ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    identity_fn = lambda h, t: h
    baseline_texts = []
    baseline_correct = []
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [Baseline] {idx}/{len(data)}")
        r = _run_intervened(ctx, sample, identity_fn)
        baseline_texts.append(r.get("text", r.get("predicted", "")))
        baseline_correct.append(int(r["is_correct"]))
    baseline_acc = sum(baseline_correct) / max(len(data), 1)

    baseline_ci = report_mean_with_ci(
        baseline_correct, metric="baseline_accuracy",
        context={**ci_ctx, "condition": "baseline"},
        cis_jsonl=cis_jsonl,
        n_boot=args.n_boot,
    )

    # ════════════════════════════════════════════════════════════════
    # PART 1: ABLATION — h <- (I - B_t B_t^T) h
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GRADIENT-SUBSPACE ABLATION")
    print("=" * 60)

    # ── Gradient ablation (deterministic, single run) ─────────────
    grad_fn = make_projection_intervention(nullspace_grad, args.device)
    print("  Running gradient-nullspace ablation...")
    grad_correct = _eval_correctness(grad_fn, "GRAD")
    grad_acc = sum(grad_correct) / max(len(data), 1)

    grad_ci = report_mean_with_ci(
        grad_correct, metric="grad_ablation_accuracy",
        context={**ci_ctx, "condition": "grad_nullspace"},
        cis_jsonl=cis_jsonl,
        n_boot=args.n_boot,
    )

    # ── Random ablation: K independent seeds ─────────────────────
    #
    # For each seed s_k = args.seed + k:
    #   1. Build random subspace projectors (rank-matched to gradient)
    #   2. Run ablation eval → per-instance correctness vector (length N)
    #   3. Save per-seed CI record
    #   4. Accumulate into pooled vector, then discard per-seed outputs
    #
    # rand_correct_pooled = concat of K correctness vectors (length K*N)
    # rand_correct_per_seed_accs[k] = mean accuracy for seed k

    N = len(data)
    rand_correct_pooled = []       # will be length K*N
    rand_correct_per_seed_accs = []

    for k in range(K):
        s_k = args.seed + k
        print(f"\n  Running rand-nullspace control (seed {s_k}, "
              f"{k+1}/{K})...")

        nullspace_rand_k, subspace_rand_k = build_random_subspace_projectors(
            ranks, D, seed=s_k,
        )
        rand_fn_k = make_projection_intervention(nullspace_rand_k, args.device)
        rand_correct_k = _eval_correctness(rand_fn_k, f"Rand s{s_k}")

        # Per-seed CI record
        seed_acc = sum(rand_correct_k) / max(N, 1)
        rand_correct_per_seed_accs.append(seed_acc)
        # seed_ci = report_mean_with_ci(
        #     rand_correct_k, metric="rand_ablation_accuracy",
        #     context={**ci_ctx, "condition": "rand_nullspace",
        #              "seed": s_k, "seed_index": k,
        #              "n_random_seeds": K},
        #     cis_jsonl=cis_jsonl,
        #     n_boot=args.n_boot,
        # )

        # Accumulate into pooled vector, then free per-seed data
        rand_correct_pooled.extend(rand_correct_k)
        del nullspace_rand_k, subspace_rand_k, rand_correct_k

    rand_correct_per_seed_accs = np.array(rand_correct_per_seed_accs)
    rand_correct_pooled = np.array(rand_correct_pooled, dtype=np.float64)

    # (ii) Cross-seed summary record:
    #   point = mean of per-seed accuracies
    #   extra.seed_std = std of per-seed accuracies
    seed_mean_rec = BootstrapResult(
        metric="rand_ablation_accuracy_seed_mean",
        point=float(rand_correct_per_seed_accs.mean()),
        ci_low=float("nan"), ci_high=float("nan"),
        ci_level=95.0, n=K, n_boot=0, seed=args.seed,
        method="seed_summary",
        extra={"seed_std": float(rand_correct_per_seed_accs.std()),
               "n_random_seeds": K,
               "per_seed_accs": rand_correct_per_seed_accs.tolist()},
    )
    save_record(cis_jsonl, seed_mean_rec,
                context={**ci_ctx, "condition": "rand_nullspace_seed_summary"})
    print(f"  [CI] rand_ablation seed_mean: "
          f"{seed_mean_rec.point:.3f} ± {seed_mean_rec.extra['seed_std']:.4f} "
          f"(K={K})")

    # (i) Pooled CI: bootstrap over K*N entries
    rand_ci = bootstrap_mean(
        rand_correct_pooled,
        metric="rand_ablation_accuracy",
        n_boot=args.n_boot,
    )
    save_record(cis_jsonl, rand_ci,
                context={**ci_ctx, "condition": "rand_nullspace_pooled",
                         "seed_pooling": "concat", "n_random_seeds": K})
    print(f"  [CI] rand_ablation (pooled): {rand_ci.to_short_str()}")

    rand_acc = float(rand_correct_pooled.mean())

    # ── Paired tests: baseline vs ablated ─────────────────────────
    #   diff = baseline_correct_i - ablated_correct_i
    #   positive diff = baseline was correct, ablated was wrong
    grad_drop_ci = paired_bootstrap_diff(
        baseline_correct, grad_correct,
        metric="accuracy_drop_grad",
        n_boot=args.n_boot,
    )
    save_record(cis_jsonl, grad_drop_ci,
                context={**ci_ctx, "condition": "baseline_minus_grad"})
    print(f"  [CI] accuracy_drop_grad: {grad_drop_ci.to_short_str()}")

    # Paired test for rand: repeat baseline K times to match pooled rand
    #   baseline_rep = tile(baseline_correct, K)  → length K*N
    #   rand_correct_pooled[k*N + i] is paired with baseline_rep[k*N + i]
    baseline_rep = np.tile(np.array(baseline_correct, dtype=np.float64), K)
    rand_drop_ci = paired_bootstrap_diff(
        baseline_rep, rand_correct_pooled,
        metric="accuracy_drop_rand",
        n_boot=args.n_boot,
    )
    save_record(cis_jsonl, rand_drop_ci,
                context={**ci_ctx, "condition": "baseline_minus_rand",
                         "seed_pooling": "concat", "n_random_seeds": K})
    print(f"  [CI] accuracy_drop_rand (pooled): {rand_drop_ci.to_short_str()}")

    # McNemar: gradient vs baseline (deterministic, unpooled)
    mc_grad = mcnemar_test(baseline_correct, grad_correct,
                           metric="mcnemar_baseline_vs_grad")
    save_record(cis_jsonl, mc_grad, context={**ci_ctx})
    print(f"  [CI] McNemar baseline vs grad: p={mc_grad['p_value']:.4g}")

    # McNemar: rand pooled vs baseline repeated
    mc_rand = mcnemar_test(
        baseline_rep.astype(int).tolist(),
        rand_correct_pooled.astype(int).tolist(),
        metric="mcnemar_baseline_vs_rand",
    )
    save_record(cis_jsonl, mc_rand,
                context={**ci_ctx, "seed_pooling": "concat",
                         "n_random_seeds": K})
    print(f"  [CI] McNemar baseline vs rand (pooled): "
          f"p={mc_rand['p_value']:.4g}")

    print(f"\n  {'='*50}")
    print(f"  ABLATION SUMMARY")
    print(f"  {'='*50}")
    print(f"  Baseline:        {baseline_ci.to_short_str()}")
    print(f"  Grad-nullspace:  {grad_ci.to_short_str()}  "
          f"(drop: {grad_drop_ci.to_short_str()})")
    print(f"  Rand-nullspace:  {rand_ci.to_short_str()}  "
          f"(drop: {rand_drop_ci.to_short_str()})  [K={K} seeds pooled]")

    ablation_results = {
        "task": args.task, "model": args.model,
        "baseline_accuracy": baseline_acc,
        "baseline_ci": [baseline_ci.ci_low, baseline_ci.ci_high],
        "grad_accuracy": grad_acc,
        "grad_ci": [grad_ci.ci_low, grad_ci.ci_high],
        "rand_accuracy": rand_acc,
        "rand_ci": [rand_ci.ci_low, rand_ci.ci_high],
        "accuracy_drop_grad": baseline_acc - grad_acc,
        "accuracy_drop_grad_ci": [grad_drop_ci.ci_low, grad_drop_ci.ci_high],
        "accuracy_drop_rand": float(np.mean(baseline_rep) - rand_acc),
        "accuracy_drop_rand_ci": [rand_drop_ci.ci_low, rand_drop_ci.ci_high],
        "mcnemar_grad_p": mc_grad["p_value"],
        "mcnemar_rand_p": mc_rand["p_value"],
        "ranks_per_t": ranks,
        "n_random_seeds": K,
        "rand_seed_mean": float(rand_correct_per_seed_accs.mean()),
        "rand_seed_std": float(rand_correct_per_seed_accs.std()),
    }
    abl_path = output_dir / "ablation_results.json"
    with open(abl_path, "w") as f:
        json.dump(deep_convert(ablation_results), f, indent=2)
    print(f"  Saved -> {abl_path}")

    # ════════════════════════════════════════════════════════════════
    # PART 2: AMPLIFICATION — h' = h^null + alpha * h^c
    #
    # Math per timestep t:
    #   h^c    = C_t @ h        (causal component, C_t = B_t B_t^T)
    #   h^null = h - h^c        (nullspace component)
    #   h'     = h^null + alpha * h^c
    #
    # alpha=0: ablation (h' = h^null)
    # alpha=1: identity (h' = h)
    # alpha>1: amplify the causal component while preserving nullspace
    #
    # Unlike additive steering (h + alpha * d/||d||), this scales an
    # existing component of h rather than injecting an external vector.
    # Both grad and rand controls stay on the data manifold, so the
    # magnitude confound (random dirs being more off-manifold than
    # gradient dirs) is eliminated.
    #
    # Random control: K independent random subspaces, same pooling
    # scheme as the ablation section above.
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUBSPACE AMPLIFICATION")
    print("=" * 60)

    # ── Gradient amplification (deterministic, single run) ────────
    # Multi-GPU path only runs gradient once; rand loops below.
    if args.n_gpus > 1:
        print(f"  Multi-GPU: sharding {len(data)} instances across "
              f"{args.n_gpus} GPUs")
        # Free single-GPU model before spawning workers.
        if is_codi:
            del codi_dict
            ctx["codi_dict"] = None
        else:
            del coconut_model, base_model, tokenizer
            ctx["coconut_model"] = ctx["base_model"] = ctx["tokenizer"] = None
        torch.cuda.empty_cache()

        # For multi-GPU, we need a single subspace_rand for each seed.
        # Run gradient once, then loop K rand seeds.
        # First: gradient only (pass dummy rand to get grad results).
        dummy_null_rand, dummy_sub_rand = build_random_subspace_projectors(
            ranks, D, seed=args.seed,
        )
        grad_amplification, _ = run_amplification_multigpu(
            args.task, args.model, args.n_thoughts, data, alphas,
            baseline_texts, subspace_grad, dummy_sub_rand, args.n_gpus,
        )
        del dummy_null_rand, dummy_sub_rand
    else:
        print("\n  Amplifying along gradient subspace:")
        grad_amplification = run_amplification_sweep(
            ctx, data, subspace_grad, alphas,
            baseline_texts, label_name="GRAD",
        )

    # ── Random amplification: K independent seeds ─────────────────
    #
    # For each alpha, accumulate K per-instance flip vectors.
    # rand_flip_pooled[alpha] = concat of K flip vectors (length K*N)
    # rand_flip_per_seed_rates[alpha][k] = flip rate for seed k

    rand_flip_pooled = {alpha: [] for alpha in alphas}
    rand_flip_per_seed_rates = {alpha: [] for alpha in alphas}

    for k in range(K):
        s_k = args.seed + k
        print(f"\n  Rand amplification (seed {s_k}, {k+1}/{K}):")

        _, subspace_rand_k = build_random_subspace_projectors(
            ranks, D, seed=s_k,
        )

        if args.n_gpus > 1:
            # Re-run multi-GPU for this rand seed only.
            # Pass subspace_grad as dummy for grad slot (we discard it).
            _, rand_amp_k = run_amplification_multigpu(
                args.task, args.model, args.n_thoughts, data, alphas,
                baseline_texts, subspace_grad, subspace_rand_k, args.n_gpus,
            )
        else:
            rand_amp_k = run_amplification_sweep(
                ctx, data, subspace_rand_k, alphas,
                baseline_texts, label_name=f"Rand s{s_k}",
            )

        # Extract per-instance flip vectors, accumulate, then discard
        for alpha in alphas:
            flip_k = np.zeros(N, dtype=np.float64)
            for i in rand_amp_k[alpha]["flipped_indices"]:
                flip_k[i] = 1.0
            rand_flip_pooled[alpha].extend(flip_k.tolist())
            rand_flip_per_seed_rates[alpha].append(
                rand_amp_k[alpha]["flip_rate"]
            )

            # # Per-seed CI record
            # seed_ci = bootstrap_mean(
            #     flip_k,
            #     metric=f"flip_rate_rand_a{alpha}",
            #     n_boot=args.n_boot,
            # )
            # save_record(cis_jsonl, seed_ci,
            #             context={**ci_ctx, "condition": "rand_amp",
            #                      "alpha": alpha, "seed": s_k,
            #                      "seed_index": k, "n_random_seeds": K})

        del subspace_rand_k, rand_amp_k

    # ── Summary with bootstrap CIs ──────────────────────────────────
    #
    # For each alpha:
    #   grad_flip: length N (deterministic)
    #   rand_flip_pooled[alpha]: length K*N (pooled across seeds)
    #
    #   flip_rate   = mean(flip_vec)                        (Pattern A)
    #   flip_diff   = mean(grad_flip_rep - rand_flip_pooled) (Pattern C)
    #     where grad_flip_rep = tile(grad_flip, K) for pairing

    amp_ci_records = {}

    for alpha in alphas:
        # Gradient flip vector (single deterministic run)
        grad_flip = np.zeros(N, dtype=np.float64)
        for i in grad_amplification[alpha]["flipped_indices"]:
            grad_flip[i] = 1.0

        rand_flip_pool_arr = np.array(
            rand_flip_pooled[alpha], dtype=np.float64,
        )

        g_ci = bootstrap_mean(grad_flip, metric=f"flip_rate_grad_a{alpha}", n_boot=args.n_boot,)
        save_record(cis_jsonl, g_ci,
                    context={**ci_ctx, "condition": "grad_amp", "alpha": alpha})

        # (i) Pooled rand CI
        r_ci = bootstrap_mean(
            rand_flip_pool_arr,
            metric=f"flip_rate_rand_a{alpha}",
            n_boot=args.n_boot,
        )
        save_record(cis_jsonl, r_ci,
                    context={**ci_ctx, "condition": "rand_amp_pooled",
                             "alpha": alpha, "seed_pooling": "concat",
                             "n_random_seeds": K})

        # (ii) Cross-seed summary for this alpha
        per_seed_rates = np.array(rand_flip_per_seed_rates[alpha])
        seed_mean_amp = BootstrapResult(
            metric=f"flip_rate_rand_a{alpha}_seed_mean",
            point=float(per_seed_rates.mean()),
            ci_low=float("nan"), ci_high=float("nan"),
            ci_level=95.0, n=K, n_boot=0, seed=args.seed,
            method="seed_summary",
            extra={"seed_std": float(per_seed_rates.std()),
                   "n_random_seeds": K,
                   "per_seed_rates": per_seed_rates.tolist()},
        )
        save_record(cis_jsonl, seed_mean_amp,
                    context={**ci_ctx, "condition": "rand_amp_seed_summary",
                             "alpha": alpha})

        # Paired diff: tile grad to match pooled rand length
        grad_flip_rep = np.tile(grad_flip, K)
        d_ci = paired_bootstrap_diff(
            grad_flip_rep, rand_flip_pool_arr,
            metric=f"flip_diff_grad_minus_rand_a{alpha}",
            n_boot=args.n_boot,
        )
        save_record(cis_jsonl, d_ci,
                    context={**ci_ctx, "condition": "grad_minus_rand_amp",
                             "alpha": alpha, "seed_pooling": "concat",
                             "n_random_seeds": K})

        mc = mcnemar_test(
            grad_flip_rep.astype(int), rand_flip_pool_arr.astype(int),
            metric=f"mcnemar_grad_vs_rand_a{alpha}",
        )
        save_record(cis_jsonl, mc,
                    context={**ci_ctx, "alpha": alpha,
                             "seed_pooling": "concat", "n_random_seeds": K})

        amp_ci_records[alpha] = {
            "grad_ci": [g_ci.ci_low, g_ci.ci_high],
            "rand_ci": [r_ci.ci_low, r_ci.ci_high],
            "diff_ci": [d_ci.ci_low, d_ci.ci_high],
            "diff_point": d_ci.point,
            "mcnemar_p": mc["p_value"],
            "rand_seed_mean": float(per_seed_rates.mean()),
            "rand_seed_std": float(per_seed_rates.std()),
        }

    # Use pooled rand flip rate for the summary table
    print(f"\n  {'='*72}")
    print(f"  AMPLIFICATION SUMMARY (subspace-restricted scaling, flip rate)"
          f"  [K={K} rand seeds pooled]")
    print(f"  {'='*72}")
    print(f"  {'Alpha':>8}  {'GRAD Flip':>22}  "
          f"{'Rand Flip':>22}  {'diff p':>8}  Interpretation")
    print(f"  {'-'*8}  {'-'*22}  {'-'*22}  {'-'*8}  {'-'*30}")
    for alpha in alphas:
        grad_fr = grad_amplification[alpha]["flip_rate"]
        # Pooled rand flip rate = mean of pooled vector
        rand_fr = float(np.mean(rand_flip_pooled[alpha]))
        ci_rec = amp_ci_records[alpha]

        if abs(alpha - 1.0) < 1e-9:
            interp = "Identity (sanity check)"
        elif grad_fr < 0.02 and rand_fr < 0.02:
            interp = "No effect"
        elif grad_fr > 0.05 and rand_fr < 0.02:
            interp = "Grad dirs are causal"
        elif grad_fr > 0.05 and rand_fr > 0.05:
            if grad_fr > rand_fr * 1.5:
                interp = "Grad dirs preferentially causal"
            else:
                interp = "Generally fragile"
        else:
            interp = "Ambiguous"

        g_lo, g_hi = ci_rec["grad_ci"]
        r_lo, r_hi = ci_rec["rand_ci"]
        print(f"  {alpha:>8g}  {grad_fr:.1%} [{g_lo:.3f},{g_hi:.3f}]  "
              f"{rand_fr:.1%} [{r_lo:.3f},{r_hi:.3f}]  "
              f"{ci_rec['mcnemar_p']:>8.3g}  {interp}")

    amplification_results = {
        "task": args.task, "model": args.model,
        "alphas": alphas,
        "grad_amplification": deep_convert(grad_amplification),
        "rand_flip_pooled_rates": {
            alpha: float(np.mean(rand_flip_pooled[alpha]))
            for alpha in alphas
        },
        "rand_flip_per_seed_rates": deep_convert(rand_flip_per_seed_rates),
        "amplification_cis": deep_convert(amp_ci_records),
        "ranks_per_t": ranks,
        "n_gpus": args.n_gpus,
        "seed": args.seed,
        "n_random_seeds": K,
    }
    amp_path = output_dir / "amplification_results.json"
    with open(amp_path, "w") as f:
        json.dump(deep_convert(amplification_results), f, indent=2)
    print(f"\n  Saved -> {amp_path}")
    print(f"  Bootstrap CIs -> {cis_jsonl}")


if __name__ == "__main__":
    main()