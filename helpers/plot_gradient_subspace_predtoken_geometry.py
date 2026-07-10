"""
Geometry diagnostic for the PREDICTED-TOKEN gradient subspace.

Identical analysis to gradient_subspace_geometry.py (principal-angle
stability of B_t across timesteps, with bootstrap CIs by resampling the
gradient-matrix rows and re-running the per-timestep SVD), but consuming
the predicted-token artefacts and writing back into the predtoken tree:

    outputs/gradient_geometry_predtoken/<family>/<task>/<model>/
        bases.npz, gradients.npz        (inputs)
        diagnosis.json, bootstrap_cis.jsonl   (outputs)

WHY A SEPARATE ENTRY POINT
--------------------------
The gold geometry script hardcodes the gradient_geometry/ root for both
input and output. Rather than fork its compute, we import its functions
unchanged (load_bases, bootstrap_stability, print_stability, deep_convert)
and only redirect the root to gradient_geometry_predtoken/. Every number is
produced by the exact same code path as the gold diagnostic.

USAGE
-----
    python -u -m experiments.geometry.gradient_subspace_predtoken_geometry \
        --task prosqa --model coconut
    python -u -m experiments.geometry.gradient_subspace_predtoken_geometry \
        --task gsm --model codi --model_family gpt2
"""

import json
import argparse
from pathlib import Path

import numpy as np

from src.config import BASE_DIR, set_seed
from src.bootstrap_stats import save_record

# Reuse the gold geometry module's compute + I/O helpers unchanged. Only the
# input/output ROOT differs (predtoken tree instead of gold tree).
from experiments.geometry.gradient_subspace_geometry import (
    load_bases,
    bootstrap_stability,
    print_stability,
    deep_convert,
)


# Predtoken tree root, mirroring gradient_geometry/ one level over.
PREDTOKEN_ROOT = BASE_DIR / "outputs" / "gradient_geometry_predtoken"


def main():
    parser = argparse.ArgumentParser(
        description="Subspace-stability diagnostic on the PREDICTED-TOKEN "
                    "gradient subspace produced by "
                    "gradient_subspace_predtoken.py."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument(
        "--model",
        choices=["coconut", "coconut_u", "pause", "codi"],
        required=True,
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family; selects the namespaced predtoken input/"
             "output dir.",
    )
    parser.add_argument(
        "--bases_path", type=str, default=None,
        help="Override path to the predicted-token bases.npz. Default: "
             "outputs/gradient_geometry_predtoken/<family>/<task>/<model>/"
             "bases.npz",
    )
    parser.add_argument("--explained_variance", type=float, default=0.95,
                        help="Cumulative energy threshold for SVD rank "
                             "selection during bootstrap re-SVD.")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Number of bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    # Predtoken input/output dir (same layout as gold, different root).
    output_dir = PREDTOKEN_ROOT / args.model_family / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    bases_path = (Path(args.bases_path) if args.bases_path
                  else output_dir / "bases.npz")
    grads_path = output_dir / "gradients.npz"

    if not bases_path.exists():
        raise FileNotFoundError(
            f"predicted-token bases.npz not found at {bases_path}. "
            f"Run gradient_subspace_predtoken.py first."
        )
    if not grads_path.exists():
        raise FileNotFoundError(
            f"predicted-token gradients.npz not found at {grads_path}. "
            f"Run gradient_subspace_predtoken.py first (needed for bootstrap)."
        )

    print(f"[main] PREDICTED-TOKEN geometry")
    print(f"[main] task={args.task}  model={args.model}  "
          f"family={args.model_family}")
    print(f"[main] bases_path: {bases_path}")
    print(f"[main] grads_path: {grads_path}")
    print(f"[main] output_dir: {output_dir}")

    # ── Load bases (for ranks display) and raw gradients ──────────
    # Same body as gold gradient_subspace_geometry.main(), only the paths
    # and the ci_ctx source-tag differ.
    bases = load_bases(bases_path)
    T = max(bases.keys()) + 1
    ranks = [int(bases[t].shape[1]) for t in range(T)]
    print(f"[main] subspace ranks per t: {ranks} (mean {np.mean(ranks):.1f})")

    blob = np.load(grads_path)
    G = blob["G"]  # (N, T, D)
    N = G.shape[0]
    print(f"[main] gradient matrix G: {G.shape}  "
          f"(N={N}, T={G.shape[1]}, D={G.shape[2]})")

    # ── Point estimates + bootstrap CIs ───────────────────────────
    print(f"\n[main] Running bootstrap (n_boot={args.n_boot})...")
    adj_cis, offdiag_ci = bootstrap_stability(
        G, T,
        explained_variance=args.explained_variance,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print()
    print("=" * 70)
    print(f"  PREDTOKEN GRADIENT-SUBSPACE STABILITY  "
          f"({args.task}/{args.model})")
    print("=" * 70)
    print_stability(adj_cis, offdiag_ci)

    # ── Save CI records (JSONL) ───────────────────────────────────
    # Tag records with subspace_source so downstream plotting can tell gold
    # and predtoken geometry apart if both land in the same collation.
    cis_jsonl = str(output_dir / "bootstrap_cis.jsonl")
    ci_ctx = {"task": args.task, "model": args.model,
              "model_family": args.model_family,
              "subspace_source": "pred"}
    for r in adj_cis:
        save_record(cis_jsonl, r, context=ci_ctx)
    save_record(cis_jsonl, offdiag_ci, context=ci_ctx)
    print(f"\n[main] Bootstrap CIs -> {cis_jsonl}")

    # ── Save summary JSON ─────────────────────────────────────────
    summary = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "subspace_source": "pred",
        "T": int(T),
        "subspace_ranks": ranks,
        "q3_adjacent_per_t": [r.point for r in adj_cis],
        "q3_adjacent_ci_per_t": [[r.ci_low, r.ci_high] for r in adj_cis],
        "q3_offdiag_mean": offdiag_ci.point,
        "q3_offdiag_ci": [offdiag_ci.ci_low, offdiag_ci.ci_high],
        "n_boot": args.n_boot,
        "explained_variance": args.explained_variance,
    }
    out_path = output_dir / "diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(deep_convert(summary), f, indent=2)
    print(f"[main] Saved diagnosis -> {out_path}")


if __name__ == "__main__":
    main()