"""
Test 2: Noise-input subspace comparison.

Falsifies: "the LDA temporal subspace is determined by position, not content —
any input of the right length would yield the same scaffold."

Procedure:
    1. Partition real task instances into disjoint subsets A, B (same task,
       same model, no overlap).
    2. Build a noise-input set N: sequences whose question tokens are random
       token IDs drawn uniformly from the base vocabulary, with lengths
       sampled from the empirical length distribution of the real questions.
       The thought-region layout is identical to real instances, so
       position_ids at thought positions cover the same range.
    3. Extract thoughts for A, B, N using the same model and pipeline.
    4. Fit LDA temporal subspace independently on each: Q_A, Q_B, Q_N.
    5. Compare pairwise via principal angles.

Interpretation:
    H0 (PE / length-driven):
        Q_A ~ Q_B ~ Q_N                 (all three subspaces aligned)
        scaffold is a function of position index, not input content.

    H1 (emergent, content-driven):
        Q_A ~ Q_B    (same task, same model => same scaffold)
        Q_A ≠ Q_N    (noise drives different dynamics => different scaffold)

Additional cross-transfer check:
    Use Q_A to classify timestep on B  -> expect near-perfect if same subspace
    Use Q_A to classify timestep on N  -> expect chance if subspaces differ
    Use Q_N to classify timestep on A  -> expect chance if subspaces differ

Hypothesis controls (base / cot):
    base / cot are NOT recursion-trained. We force them to recurse at
    inference time by wrapping in Coconut (feedback_mode='continuous')
    and feeding last-layer hidden states back as input embeddings,
    exactly like coconut/coconut_u. This separates two sources of any
    A vs N gap:

        - "recursion training" : content-vs-position discrimination is
          a property of optimization end-to-end through the recurrence.
        - "process of recursion": the gap emerges merely from feeding
          hidden states back through a transformer with per-position PE,
          regardless of training.

    Predictions:
        - If base + forced recursion already shows |Q_A - Q_N| (i.e. a
          content-driven gap), the gap is mechanistic, not a product of
          recursion training.
        - If only coconut/coconut_u/pause/codi show the gap, it is
          training-acquired.
        - cot vs base disambiguates further (token-level CoT training
          without continuous-recurrence training).

Usage:
    python -m experiments.probe_thoughts.noise_input_test \
        --task gsm --model codi

    # Forced-recursion controls:
    python -m experiments.probe_thoughts.noise_input_test \
        --task prosqa --model base
    python -m experiments.probe_thoughts.noise_input_test \
        --task prosqa --model cot
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from src.config import THOUGHTS, PROSQA_TEST, GSM_TEST
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
)
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )
from experiments.extract_thoughts import (
    load_data,
    extract_thoughts_single_instance,
    extract_thoughts_codi_batch,
)
from deprecated_experiments.temporal_scaffold.scaffolding_metrics import (
    variance_decomposition,
    fit_lda_subspace,
    principal_angles,
    held_out_timestep_accuracy,
    lda_cluster_separation,
)


# ═══════════════════════════════════════════════════════════════════
# Noise-input generator
# ═══════════════════════════════════════════════════════════════════

def build_noise_samples(real_samples, tokenizer, seed=0,
                        exclude_special=True):
    """
    For each real instance, build a matched noise instance:
      - same number of question tokens (so thought positions match absolutely)
      - token ids sampled uniformly at random from the base vocabulary
      - special tokens excluded (to avoid triggering control behaviour)

    The 'answer' field is preserved but will never be consumed since we only
    extract thoughts, not evaluate correctness.

    # length_i = |tokenize(q_real_i)|
    # noise_tokens_i = uniform sample of length_i from vocab \ special
    # noise_q_i      = tokenizer.decode(noise_tokens_i)
    # The decoded string is re-tokenized during extraction, yielding
    # approximately length_i tokens (BPE is not strictly round-trip).
    # We accept this small mismatch — it is the same for all four models.
    """
    rng = np.random.default_rng(seed)

    vocab_size = tokenizer.vocab_size
    if exclude_special:
        special_ids = set(tokenizer.all_special_ids or [])
        # Also exclude tokens added for this model
        added = getattr(tokenizer, "added_tokens_encoder", {}) or {}
        special_ids |= set(added.values())
    else:
        special_ids = set()

    allowed = np.array([i for i in range(vocab_size)
                        if i not in special_ids], dtype=np.int64)

    noise_samples = []
    for s in real_samples:
        real_ids = tokenizer.encode(s["question"], add_special_tokens=False)
        length = max(len(real_ids), 1)
        # Uniform-at-random token ids
        pick = rng.choice(len(allowed), size=length, replace=True)
        token_ids = allowed[pick].tolist()
        noise_q = tokenizer.decode(token_ids, skip_special_tokens=True)
        noise_samples.append({
            "question": noise_q,
            "answer": s.get("answer", ""),
        })
    return noise_samples


# ═══════════════════════════════════════════════════════════════════
# Bulk thought extraction (mirrors extract_thoughts.py)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_thoughts_bulk(samples, task, model_name, n_thoughts, device,
                          coconut_model=None, base_model=None, tokenizer=None,
                          latent_id=None, start_id=None, end_id=None,
                          codi_dict=None):
    """
    Extract thoughts for a list of samples. Model must already be loaded.
    Returns tensor of shape (len(samples), n_thoughts+1, D).
    """
    is_codi = (model_name == "codi")

    if is_codi:
        return extract_thoughts_codi_batch(
            codi_dict, samples, n_thoughts, device,
            batch_size=32,
        )

    N = len(samples)
    hidden_dim = base_model.config.n_embd
    out = torch.zeros(N, n_thoughts + 1, hidden_dim)

    for idx, sample in enumerate(samples):
        if idx % 100 == 0:
            print(f"    extracting {idx}/{N}")
        t = extract_thoughts_single_instance(
            coconut_model, base_model, tokenizer, sample,
            n_thoughts, device, start_id, latent_id, end_id,
        )
        out[idx] = t
    return out


# ═══════════════════════════════════════════════════════════════════
# Pairwise subspace comparison
# ═══════════════════════════════════════════════════════════════════

def pairwise_comparison(name_a, thoughts_a, name_b, thoughts_b):
    """
    Compare two thought tensors by fitting LDA on each and reporting
    principal angles + cross-transfer classification accuracy.
    """
    Q_a, _ = fit_lda_subspace(thoughts_a)
    Q_b, _ = fit_lda_subspace(thoughts_b)
    cos_ang, rad_ang = principal_angles(Q_a, Q_b)

    acc_a_on_b = held_out_timestep_accuracy(thoughts_a, thoughts_b, Q_a)
    acc_b_on_a = held_out_timestep_accuracy(thoughts_b, thoughts_a, Q_b)

    return {
        "pair": f"{name_a} vs {name_b}",
        "principal_angles_cos": cos_ang.tolist(),
        "principal_angles_rad": rad_ang.tolist(),
        "mean_cos": float(cos_ang.mean()),
        "min_cos": float(cos_ang.min()),
        f"Q_{name_a}_classifies_{name_b}": acc_a_on_b,
        f"Q_{name_b}_classifies_{name_a}": acc_b_on_a,
    }


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def _fmt_var(v):
    return (f"timestep={v['pct_timestep']:.2f}%  "
            f"instance={v['pct_instance']:.2f}%  "
            f"residual={v['pct_residual']:.2f}%")


def print_report(report, task, model_name):
    line = "=" * 72
    print(f"\n{line}")
    print(f"NOISE-INPUT REPORT  —  task={task}  model={model_name}")
    print(line)

    print("\n  Variance decomposition:")
    for split in ("A", "B", "N"):
        v = report[f"variance_{split}"]
        print(f"    set {split}:  {_fmt_var(v)}")

    print("\n  LDA Fisher separation (higher = cleaner timestep clusters):")
    for split in ("A", "B", "N"):
        print(f"    set {split}:  {report[f'lda_separation_{split}']:.3f}")

    print("\n  Pairwise subspace comparisons:")
    for pair_key in ("AB", "AN", "BN"):
        pc = report[f"pair_{pair_key}"]
        print(f"\n    {pc['pair']}:")
        print(f"      mean cos(theta):  {pc['mean_cos']:.3f}")
        print(f"      min  cos(theta):  {pc['min_cos']:.3f}")
        # cross-transfer
        for k, v in pc.items():
            if k.startswith("Q_"):
                print(f"      {k}:  {v:.3f}")

    # Interpretation hints
    mean_AB = report["pair_AB"]["mean_cos"]
    mean_AN = report["pair_AN"]["mean_cos"]
    print(f"\n  Hypothesis verdict (content vs position):")
    print(f"    mean cos(A,B) = {mean_AB:.3f}   (same task, disjoint instances)")
    print(f"    mean cos(A,N) = {mean_AN:.3f}   (task vs noise)")
    gap = mean_AB - mean_AN
    print(f"    gap (A,B) - (A,N) = {gap:+.3f}")
    if gap > 0.15:
        print(f"    → Subspace for real inputs is substantially more aligned with")
        print(f"      itself than with noise inputs — scaffold is CONTENT-DRIVEN.")
    elif gap < 0.05:
        print(f"    → Subspaces for task and noise are similarly aligned — scaffold")
        print(f"      is likely POSITION-DRIVEN (artifact of PE).")
    else:
        print(f"    → Inconclusive gap; inspect per-dimension cosines.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi",
                                 "base", "cot"],
                        required=True,
                        help="Recursion-trained models: coconut, coconut_u, "
                             "pause, codi. Forced-recursion controls (no "
                             "recursion training): base, cot.")
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Cap total real instances used (split A+B).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        THOUGHTS / args.task / args.model / "noise_input_test"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load model once (A, B, N all use the same model) --------
    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(args.task, args.device)
        tokenizer = codi_dict["tokenizer"]
        coconut_model = None
        base_model = codi_dict["model"]
        latent_id = start_id = end_id = None
    else:
        (coconut_model, base_model, tokenizer, latent_id,
         start_id, end_id, _) = setup_model_and_tokenizer(
            args.task, args.model, args.device,
        )
        codi_dict = None

    # -------- Load real task data, split into A and B --------
    real = load_data(args.task, args.max_instances)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(real))
    rng.shuffle(idx)
    half = len(idx) // 2
    idx_a, idx_b = idx[:half], idx[half:2 * half]
    samples_a = [real[i] for i in idx_a]
    samples_b = [real[i] for i in idx_b]
    samples_n = build_noise_samples(samples_a, tokenizer, seed=args.seed)

    print(f"[INFO] |A|={len(samples_a)}  |B|={len(samples_b)}  |N|={len(samples_n)}")

    # -------- Extract thoughts for each split --------
    kwargs = dict(
        task=args.task, model_name=args.model,
        n_thoughts=args.n_thoughts, device=args.device,
        coconut_model=coconut_model, base_model=base_model,
        tokenizer=tokenizer, latent_id=latent_id,
        start_id=start_id, end_id=end_id, codi_dict=codi_dict,
    )
    print("[INFO] Extracting thoughts for set A (real)...")
    thoughts_A = extract_thoughts_bulk(samples_a, **kwargs)
    print("[INFO] Extracting thoughts for set B (real, disjoint)...")
    thoughts_B = extract_thoughts_bulk(samples_b, **kwargs)
    print("[INFO] Extracting thoughts for set N (noise inputs)...")
    thoughts_N = extract_thoughts_bulk(samples_n, **kwargs)

    # Save raw tensors for downstream reuse / sanity-check plotting
    torch.save({
        "thoughts_A": thoughts_A,
        "thoughts_B": thoughts_B,
        "thoughts_N": thoughts_N,
        "idx_A": idx_a.tolist(),
        "idx_B": idx_b.tolist(),
    }, out_dir / "thoughts_ABN.pt")

    # -------- Metrics --------
    report = {
        "task": args.task,
        "model": args.model,
        "n_thoughts": int(thoughts_A.shape[1]) - 1,
        "n_A": int(thoughts_A.shape[0]),
        "n_B": int(thoughts_B.shape[0]),
        "n_N": int(thoughts_N.shape[0]),
    }

    # Per-split single-number stats
    for name, t in [("A", thoughts_A), ("B", thoughts_B), ("N", thoughts_N)]:
        report[f"variance_{name}"] = variance_decomposition(t)
        report[f"lda_separation_{name}"] = lda_cluster_separation(t)

    # Pairwise principal angles + cross-transfer accuracies
    report["pair_AB"] = pairwise_comparison("A", thoughts_A, "B", thoughts_B)
    report["pair_AN"] = pairwise_comparison("A", thoughts_A, "N", thoughts_N)
    report["pair_BN"] = pairwise_comparison("B", thoughts_B, "N", thoughts_N)

    print_report(report, args.task, args.model)

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()