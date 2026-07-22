"""
Mean ablation: causal test of variance decomposition components.

Hypothesis
----------
Variance decomposition:
    # h_{i,t} = mu + (mu_t - mu) + (mu_i_hat - mu) + eps_{i,t}
    #          (global)  (temporal)   (instance)      (residual)

Combinatorial Interventions:
    # ABLATIONS (Remove 1 component)
    # 1. timestep_ablation:   h'_{i,t} = h_{i,t} - mu_t + mu
    # 2. instance_ablation:   h'_{i,t} = h_{i,t} - mu_i_hat + mu
    # 3. residual_ablation:   h'_{i,t} = mu_t + mu_i_hat - mu  (Note: Time-travel leak via mu_i_hat)

    # ISOLATIONS (Keep 1 component)
    # 4. timestep_isolation:  h'_{i,t} = mu_t  (AKA The Hollow Clock)
    # 5. instance_isolation:  h'_{i,t} = mu_i_hat
    # 6. residual_isolation:  h'_{i,t} = h_{i,t} - mu_t - mu_i_hat + 2*mu

    # CONTROL (matched-norm random replacement)
    # 7. random_matched_norm: r_t ~ N(0, I_D), ||r_t|| = ||mu_t - mu||
    #                         h'_{i,t} = h_{i,t} + r_t

Multi-GPU
---------
Uses torch.multiprocessing.spawn-style sharding (matches inlp_interventions.py):
contiguous shards across cuda:0..cuda:{n_gpus-1}, results merged in main proc.

Usage
-----
    # Single-GPU
    python -m experiments.geometry.mean_ablation \
        --task prosqa --model coconut_u --intervention all
    # Skip baseline + control
    python -m experiments.geometry.mean_ablation \
        --task prosqa --model coconut_u --intervention core
    # Multi-GPU
    python -m experiments.geometry.mean_ablation \
        --task prosqa --model coconut_u --intervention core --n_gpus 4
    # Llama
    python -m experiments.geometry.mean_ablation \
        --task prosqa --model coconut_u --intervention core --model_family llama
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, THOUGHTS, set_seed
from src.utils import (
    run_codi_single_alpha,
    setup_model_and_tokenizer,
    setup_codi_model,
    run_intervened_inference_pauseaware,
)
from src.bootstrap_stats import (
    bootstrap_mean, paired_bootstrap_diff, mcnemar_test,
    save_record,
)

# All 6 decomposition interventions (excludes baseline + control)
CORE_INTERVENTIONS = [
    "timestep_ablation", "instance_ablation", "residual_ablation",
    "timestep_isolation", "instance_isolation", "residual_isolation",
]

# ═══════════════════════════════════════════════════════════════════
# Inference dispatch
# ═══════════════════════════════════════════════════════════════════

def _run_intervened(ctx, sample, intervention_fn):
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

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data

def deep_convert(obj):
    if isinstance(obj, dict):
        return {str(k): deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

# ═══════════════════════════════════════════════════════════════════
# Train-split statistics
# ═══════════════════════════════════════════════════════════════════

def load_train_thoughts(task, model, family="gpt2"):
    # Layout matches extract_thoughts.py / markovianity_test.py:
    # THOUGHTS/<family>/<task>/thoughts_<model>_train.pt
    path = THOUGHTS / family / task / f"thoughts_{model}_train.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Train thoughts not found at {path}. "
            f"Run extract_thoughts.py --split train for "
            f"--task {task} --model {model} --model_family {family}."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["thoughts"]

def compute_temporal_stats(train_thoughts):
    train = train_thoughts.float()
    mu_t = train.mean(dim=0)                    
    mu = train.mean(dim=(0, 1))                 
    # delta_norms[t] = ||mu_t[t] - mu||_2  -- target norm for matched-norm control
    delta_norms = (mu_t - mu.unsqueeze(0)).norm(dim=-1)
    return mu_t, mu, delta_norms

# ═══════════════════════════════════════════════════════════════════
# Intervention factories
# ═══════════════════════════════════════════════════════════════════

def make_identity_intervention():
    def intervention_fn(h, t): return h
    return intervention_fn

# --- ABLATIONS (Remove 1) ---

def make_timestep_ablation_intervention(mu_t, mu, device):
    mu_t_dev = mu_t.to(device)
    mu_dev = mu.to(device)
    T = mu_t_dev.shape[0]
    def intervention_fn(h, t):
        if t < 0 or t >= T: return h
        return (h.to(torch.float32) - mu_t_dev[t] + mu_dev).to(h.dtype)
    return intervention_fn

def make_instance_ablation_intervention(mu_i_hat, mu, device):
    mu_i_dev, mu_dev = mu_i_hat.to(device), mu.to(device)
    def intervention_fn(h, t):
        return (h.to(torch.float32) - mu_i_dev + mu_dev).to(h.dtype)
    return intervention_fn

def make_residual_ablation_intervention(mu_i_hat, mu_t, mu, device):
    mu_i_dev, mu_t_dev, mu_dev = mu_i_hat.to(device), mu_t.to(device), mu.to(device)
    T = mu_t_dev.shape[0]
    def intervention_fn(h, t):
        if t < 0 or t >= T: return h
        return (mu_t_dev[t] + mu_i_dev - mu_dev).to(h.dtype)
    return intervention_fn

# --- ISOLATIONS (Keep 1) ---

def make_timestep_isolation_intervention(mu_t, device):
    mu_t_dev = mu_t.to(device)
    T = mu_t_dev.shape[0]
    def intervention_fn(h, t):
        if t < 0 or t >= T: return h
        return mu_t_dev[t].clone().to(h.dtype)
    return intervention_fn

def make_instance_isolation_intervention(mu_i_hat, device):
    mu_i_dev = mu_i_hat.to(device)
    def intervention_fn(h, t):
        return mu_i_dev.clone().to(h.dtype)
    return intervention_fn

def make_residual_isolation_intervention(mu_i_hat, mu_t, mu, device):
    mu_i_dev, mu_t_dev, mu_dev = mu_i_hat.to(device), mu_t.to(device), mu.to(device)
    T = mu_t_dev.shape[0]
    def intervention_fn(h, t):
        if t < 0 or t >= T: return h
        return (h.to(torch.float32) - mu_t_dev[t] - mu_i_dev + (2 * mu_dev)).to(h.dtype)
    return intervention_fn

# --- CONTROL (matched-norm random) ---
# Per-step seeding via _seed_for ensures reproducibility regardless of which
# rank processes the instance. Always pass GLOBAL instance idx.
#     # seed(run, idx, t) = run * 10^9 + idx * 10^4 + t + 1

def _seed_for(run_seed, idx, t):
    return run_seed * 1_000_000_000 + idx * 10_000 + t + 1

def make_random_matched_norm_intervention(run_seed, idx, delta_norms, device):
    """
    # r_t ~ N(0, I_D),  r_t <- r_t * (||mu_t - mu|| / ||r_t||)
    # h'_{i,t} = h_{i,t} + r_t
    #
    # Additive perturbation with the SAME L2 norm as the temporal
    # deviation mu_t - mu, but a random direction. Isolates the
    # "temporal axis" claim from a "perturbation of this magnitude" claim.
    """
    delta_norms_dev = delta_norms.to(device)
    T = delta_norms_dev.shape[0]

    def intervention_fn(h, t):
        if t < 0 or t >= T: return h
        torch.manual_seed(_seed_for(run_seed, idx, t))
        r = torch.randn_like(h, dtype=torch.float32)
        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-12)
        r = r * (delta_norms_dev[t] / r_norm)
        return (h.to(torch.float32) + r).to(h.dtype)
    return intervention_fn

# ═══════════════════════════════════════════════════════════════════
# Per-shard evaluation
# ═══════════════════════════════════════════════════════════════════
# Each eval_* returns (n_correct, n_total) over the assigned shard.
# Main process sums these across ranks → final accuracy.

def collect_clean_trajectory(ctx, sample):
    captured = []
    def recording_fn(h, t):
        captured.append(h.detach().to(torch.float32).cpu().clone())
        return h
    result = _run_intervened(ctx, sample, recording_fn)
    stacked = torch.stack(captured, dim=0)       
    mu_i_hat = stacked.mean(dim=0)               
    return mu_i_hat, result["is_correct"], result.get("text", "")

def eval_single_pass(ctx, data, indices, intervention_fn, label=""):
    """For interventions whose fn is identical across instances.
    Returns list of (global_idx, is_correct) pairs."""
    results = []
    N = len(indices)
    n_correct = 0
    for j, idx in enumerate(indices):
        if j % 100 == 0 and j > 0:
            print(f"    [r{ctx['rank']}][{label}] {j}/{N}  running acc: {n_correct/j:.1%}")
        r = _run_intervened(ctx, data[idx], intervention_fn)
        correct = int(r["is_correct"])
        results.append((idx, correct))
        n_correct += correct
    print(f"  [r{ctx['rank']}][{label}] shard acc: {n_correct}/{N}")
    return results

def eval_per_instance(ctx, data, indices, fn_builder, label=""):
    """
    For interventions where the fn depends on the GLOBAL instance idx
    (random control). fn_builder: global_idx -> intervention_fn.
    Returns list of (global_idx, is_correct) pairs.
    """
    results = []
    N = len(indices)
    n_correct = 0
    for j, idx in enumerate(indices):
        if j % 100 == 0 and j > 0:
            print(f"    [r{ctx['rank']}][{label}] {j}/{N}  running acc: {n_correct/j:.1%}")
        r = _run_intervened(ctx, data[idx], fn_builder(idx))
        correct = int(r["is_correct"])
        results.append((idx, correct))
        n_correct += correct
    print(f"  [r{ctx['rank']}][{label}] shard acc: {n_correct}/{N}")
    return results

def eval_two_pass(ctx, data, indices, mu_t, mu, intervention_type):
    """For interventions needing per-instance mu_i_hat from a clean pre-pass.
    Returns list of (global_idx, is_correct) pairs."""
    results = []
    N = len(indices)
    n_correct = 0
    for j, idx in enumerate(indices):
        if j % 100 == 0 and j > 0:
            print(f"    [r{ctx['rank']}][{intervention_type}] {j}/{N}  running acc: {n_correct/j:.1%}")

        mu_i_hat, _, _ = collect_clean_trajectory(ctx, data[idx])

        if intervention_type == "instance_ablation":
            fn = make_instance_ablation_intervention(mu_i_hat, mu, ctx["device"])
        elif intervention_type == "residual_ablation":
            fn = make_residual_ablation_intervention(mu_i_hat, mu_t, mu, ctx["device"])
        elif intervention_type == "instance_isolation":
            fn = make_instance_isolation_intervention(mu_i_hat, ctx["device"])
        elif intervention_type == "residual_isolation":
            fn = make_residual_isolation_intervention(mu_i_hat, mu_t, mu, ctx["device"])

        r = _run_intervened(ctx, data[idx], fn)
        correct = int(r["is_correct"])
        results.append((idx, correct))
        n_correct += correct

    print(f"  [r{ctx['rank']}][{intervention_type}] shard acc: {n_correct}/{N}")
    return results

# ═══════════════════════════════════════════════════════════════════
# Run-all-selected on a shard (used by both single-GPU and worker)
# ═══════════════════════════════════════════════════════════════════

def run_selected_on_shard(ctx, data, indices, mu_t, mu, delta_norms, selection):
    """
    selection: dict with bool keys:
        baseline, random_matched_norm, and each name in CORE_INTERVENTIONS.
    Returns: dict {name: (n_correct, n_total)}
    """
    out = {}

    if selection["baseline"]:
        out["baseline"] = eval_single_pass(
            ctx, data, indices, make_identity_intervention(), "baseline")

    if selection["timestep_ablation"]:
        out["timestep_ablation"] = eval_single_pass(
            ctx, data, indices,
            make_timestep_ablation_intervention(mu_t, mu, ctx["device"]),
            "timestep_ablation")

    if selection["instance_ablation"]:
        out["instance_ablation"] = eval_two_pass(
            ctx, data, indices, mu_t, mu, "instance_ablation")

    if selection["residual_ablation"]:
        out["residual_ablation"] = eval_two_pass(
            ctx, data, indices, mu_t, mu, "residual_ablation")

    if selection["timestep_isolation"]:
        out["timestep_isolation"] = eval_single_pass(
            ctx, data, indices,
            make_timestep_isolation_intervention(mu_t, ctx["device"]),
            "timestep_isolation")

    if selection["instance_isolation"]:
        out["instance_isolation"] = eval_two_pass(
            ctx, data, indices, mu_t, mu, "instance_isolation")

    if selection["residual_isolation"]:
        out["residual_isolation"] = eval_two_pass(
            ctx, data, indices, mu_t, mu, "residual_isolation")

    if selection["random_matched_norm"]:
        run_seed = ctx["random_seed"]
        out["random_matched_norm"] = eval_per_instance(
            ctx, data, indices,
            lambda idx: make_random_matched_norm_intervention(
                run_seed, idx, delta_norms, ctx["device"]),
            "random_matched_norm")

    return out

# ═══════════════════════════════════════════════════════════════════
# Multi-GPU: shard instances across ranks (matches inlp_interventions.py)
# ═══════════════════════════════════════════════════════════════════

def _shard_indices(n_items, world_size, rank):
    """Contiguous shard assignment: rank r handles items [r*chunk, (r+1)*chunk)."""
    # chunk = ceil(n_items / world_size); last rank may be shorter
    chunk = (n_items + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_items)
    return list(range(start, end))

def _build_ctx(rank, task, model_name, n_thoughts, random_seed, family="gpt2"):
    """Per-worker model load on cuda:{rank}."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    is_codi = (model_name == "codi")
    if is_codi:
        ctx = {
            "is_codi": True,
            "codi_dict": setup_codi_model(task, device, family=family),
            "n_thoughts": n_thoughts, "device": device, "task": task,
            "rank": rank, "random_seed": random_seed,
        }
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device, family=family)
        ctx = {
            "is_codi": False, "coconut_model": coconut_model,
            "base_model": base_model, "tokenizer": tokenizer,
            "start_id": start_id, "latent_id": latent_id, "end_id": end_id,
            "n_thoughts": n_thoughts, "device": device, "task": task,
            "rank": rank, "random_seed": random_seed,
        }
    return ctx

def _worker(rank, world_size, task, model_name, n_thoughts,
            data, mu_t, mu, delta_norms, selection, random_seed, return_queue,
            family="gpt2"):
    """Per-GPU worker. Processes its shard and returns partial counts."""
    ctx = _build_ctx(rank, task, model_name, n_thoughts, random_seed, family=family)
    indices = _shard_indices(len(data), world_size, rank)
    print(f"[rank {rank}] processing {len(indices)} instances on {ctx['device']}")
    partial = run_selected_on_shard(ctx, data, indices, mu_t, mu, delta_norms, selection)
    return_queue.put({"rank": rank, "partial": partial})

def run_multigpu(task, model_name, n_thoughts, data, mu_t, mu, delta_norms,
                 selection, random_seed, n_gpus, family="gpt2"):
    """Spawn n_gpus workers, each handling a contiguous shard."""
    ctx_mp = mp.get_context("spawn")
    q = ctx_mp.Queue()
    procs = []
    for rank in range(n_gpus):
        p = ctx_mp.Process(
            target=_worker,
            args=(rank, n_gpus, task, model_name, n_thoughts,
                  data, mu_t, mu, delta_norms, selection, random_seed, q,
                  family),
        )
        p.start()
        procs.append(p)

    shards = []
    for _ in range(n_gpus):
        shards.append(q.get())
    for p in procs:
        p.join()

    # Merge per-instance results across ranks, sorted by global index.
    # Each shard value is a list of (global_idx, is_correct) pairs.
    merged = {}
    all_keys = set()
    for s in shards:
        all_keys.update(s["partial"].keys())
    for key in all_keys:
        combined = []
        for s in shards:
            if key in s["partial"]:
                combined.extend(s["partial"][key])
        # Sort by global instance index to get a deterministic ordering
        combined.sort(key=lambda x: x[0])
        merged[key] = [c for _, c in combined]
    return merged

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument("--model", choices=["coconut", "coconut_u", "pause", "codi"], default="coconut_u")
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Threads through model loading and "
             "namespaces the train-thoughts load path and outputs.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--intervention", type=str, default="all",
        choices=[
            "all", "core", "baseline", "random_matched_norm",
            "timestep_ablation", "instance_ablation", "residual_ablation",
            "timestep_isolation", "instance_isolation", "residual_isolation",
        ],
    )
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Seed for random_matched_norm control.")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="If >1, shard instances across GPUs 0..n_gpus-1.")
    args = parser.parse_args()
    set_seed(0)

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "variance_matrix" / args.model_family / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.task, args.max_instances)
    train_thoughts = load_train_thoughts(args.task, args.model, family=args.model_family)
    mu_t, mu, delta_norms = compute_temporal_stats(train_thoughts)

    # Build selection dict from --intervention.
    sel = args.intervention
    selection = {
        "baseline":            sel in ("all", "baseline"),
        "random_matched_norm": sel in ("all", "random_matched_norm"),
    }
    for name in CORE_INTERVENTIONS:
        selection[name] = sel in ("all", "core", name)

    print("\n" + "="*50 + "\nVARIANCE DECOMPOSITION MATRIX\n" + "="*50)
    print(f"  task={args.task}  model={args.model}  family={args.model_family}  "
          f"n_instances={len(data)}  "
          f"n_gpus={args.n_gpus}  intervention={args.intervention}")

    if args.n_gpus > 1:
        # per_instance_vectors: {intervention_name: [is_correct_0, ..., is_correct_N]}
        per_instance_vectors = run_multigpu(
            args.task, args.model, args.n_thoughts, data, mu_t, mu, delta_norms,
            selection, args.random_seed, args.n_gpus, family=args.model_family,
        )
    else:
        # Single-GPU path
        is_codi = (args.model == "codi")
        if is_codi:
            ctx = {"is_codi": True,
                   "codi_dict": setup_codi_model(args.task, args.device,
                                                 family=args.model_family),
                   "n_thoughts": args.n_thoughts, "device": args.device, "task": args.task,
                   "rank": 0, "random_seed": args.random_seed}
        else:
            coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
                setup_model_and_tokenizer(args.task, args.model, args.device,
                                          family=args.model_family)
            ctx = {"is_codi": False, "coconut_model": coconut_model,
                   "base_model": base_model, "tokenizer": tokenizer,
                   "start_id": start_id, "latent_id": latent_id, "end_id": end_id,
                   "n_thoughts": args.n_thoughts, "device": args.device, "task": args.task,
                   "rank": 0, "random_seed": args.random_seed}
        indices = list(range(len(data)))
        partial = run_selected_on_shard(ctx, data, indices, mu_t, mu, delta_norms, selection)
        # Convert list of (global_idx, is_correct) → sorted vector
        per_instance_vectors = {}
        for k, pairs in partial.items():
            pairs.sort(key=lambda x: x[0])
            per_instance_vectors[k] = [c for _, c in pairs]

    # ─── Bootstrap CIs (Pattern A: per-instance accuracy) ────────
    cis_jsonl = str(output_dir / f"cis_{args.intervention}.jsonl")
    # baseline uses no train-derived reference; all others use mu_t/mu/delta_norms
    # computed from train_thoughts
    _ref_split = lambda name: "none" if name == "baseline" else "train"
    ci_ctx = {"task": args.task, "model": args.model,
              "model_family": args.model_family,
              "intervention_run": args.intervention,
              "eval_split": "test",
              "n_train_for_reference": int(train_thoughts.shape[0]),
              "n_test_eval": int(len(data))}

    results_acc = {}
    ci_results = {}
    for name, vec in per_instance_vectors.items():
        ci = bootstrap_mean(vec, metric=f"acc_{name}")
        save_record(cis_jsonl, ci, context={
            **ci_ctx, "intervention": name, "reference_split": _ref_split(name)})
        results_acc[name] = ci.point
        ci_results[name] = ci

    # ─── Paired comparisons against baseline (Pattern C) ─────────
    baseline_vec = per_instance_vectors.get("baseline")
    if baseline_vec is not None:
        for name, vec in per_instance_vectors.items():
            if name == "baseline":
                continue
            # paired_bootstrap_diff: E[baseline_i - intervention_i]
            diff_ci = paired_bootstrap_diff(
                baseline_vec, vec, metric=f"diff_baseline_vs_{name}")
            save_record(cis_jsonl, diff_ci, context={
                **ci_ctx, "intervention": name, "reference_split": _ref_split(name)})

            mc = mcnemar_test(baseline_vec, vec, metric=f"mcnemar_baseline_vs_{name}")
            save_record(cis_jsonl, mc, context={
                **ci_ctx, "intervention": name, "reference_split": _ref_split(name)})

    results = {"task": args.task, "model": args.model,
               "model_family": args.model_family,
               "intervention_run": args.intervention,
               "n_instances": len(data), "n_gpus": args.n_gpus,
               **results_acc}

    print("\n" + "=" * 50 + "\nSUMMARY\n" + "=" * 50)
    for name, ci in ci_results.items():
        print(f"  {name:<22}: {ci.to_short_str()}")

    with open(output_dir / f"summary_{args.intervention}.json", "w") as f:
        json.dump(deep_convert(results), f, indent=2)

if __name__ == "__main__":
    main()