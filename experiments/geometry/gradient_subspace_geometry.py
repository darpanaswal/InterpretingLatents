"""
Geometry diagnostic for the gradient-weighted causal subspace.

Consumes bases.npz and gradients.npz produced by gradient_subspace.py.
Computes the principal-angle stability of B_t between timesteps, with
bootstrap CIs obtained by resampling instance rows of the gradient
matrix G, re-running SVD per timestep, and recomputing stability on
the resampled bases.

What is reported
----------------
For each pair (t, t'), the cosine-squared of the principal angles
between B_t and B_{t'} characterises how aligned the two subspaces are.
1 = same subspace, 0 = orthogonal. We report:

    - q3_adjacent_per_t  : mean cos^2 between B_t and B_{t+1},
                           one value per t in [0, T-2].
    - q3_offdiag_mean    : mean over all unordered pairs (t, t')
                           with t != t'. A global stability summary.

Both point estimates and bootstrap 95% CIs are produced.

Math summary
------------
For each pair of orthonormal bases B1 in R^{D x k1}, B2 in R^{D x k2}:

    # M             = B1^T B2                  shape (k1, k2)
    # cos(theta_i)  = i-th singular value of M, clipped to [0, 1]
    # mean_cos_sq   = mean(cos(theta_i)^2)     in [0, 1]

Bootstrap procedure (Pattern B — function of all instances):
    # For b = 1..B:
    #   idx = sample N indices with replacement
    #   For each t: G_t^b = G[idx, t, :]; SVD -> B_t^b
    #   Recompute adjacent/offdiag stability from {B_t^b}
    # CI = percentiles of the B bootstrap statistics

Output
------
    BASE_DIR / outputs / gradient_geometry / <family> / <task> / <model> /
        diagnosis.json          point estimates + CIs
        bootstrap_cis.jsonl     machine-readable CI records

Usage
-----
    python -u -m experiments.geometry.gradient_subspace_diagnosis \
        --task prosqa --model coconut_u
    # Llama:
    python -u -m experiments.geometry.gradient_subspace_diagnosis \
        --task prosqa --model coconut_u --model_family llama
"""

import json
import argparse
import numpy as np
from pathlib import Path
from src.config import BASE_DIR, set_seed
from src.bootstrap_stats import (
    save_record,
    BootstrapResult,
    DEFAULT_N_BOOT,
    DEFAULT_CI,
    DEFAULT_SEED,
)


# ═══════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════

def load_bases(bases_path):
    """Load per-timestep bases from npz (keys: 'B_t0', 'B_t1', ...)."""
    blob = np.load(bases_path)
    bases = {}
    for key in blob.files:
        if not key.startswith("B_t"):
            continue
        t = int(key[len("B_t"):])
        bases[t] = blob[key]
    return bases


# ═══════════════════════════════════════════════════════════════════
# SVD helper — reused for both point estimates and bootstrap draws
# ═══════════════════════════════════════════════════════════════════
#
# Given G_t of shape (n, D), compute the top-k right singular vectors
# (orthonormal basis for the row span) where k satisfies the cumulative
# energy threshold.
#
# When n < D, eigendecomposing the (n, n) Gram matrix G G^T is faster
# than computing the full (n, D) SVD: one O(n^2 D) gemm + one O(n^3)
# eigh, vs O(n D min(n, D)) for full SVD. For n=500, D=768 this is
# ~3x faster. Gives full-precision results, no truncation, no
# randomization, no fallback path.
#
# Math:
#   # Thin SVD:  G = U diag(S) V^T,    U:(n,r), V:(D,r), r = min(n,D)
#   # Gram:      G G^T = U diag(S^2) U^T   shape (n, n)
#   # eigh of G G^T gives (eigvals, U) with eigvals_i = S_i^2 >= 0
#   # Recover:   V[:, :k] = G^T U[:, :k] / S[:k]   (only top-k)
#   # Energy:    total = sum(S^2) = ||G||_F^2 = sum(eigvals)
#   # Rank:      k = min { r : cumsum(eigvals)_r / total >= rho }

def _bases_from_G(G, T, explained_variance=0.95):
    """Compute per-timestep bases from G (N, T, D) via Gram-matrix eigh."""
    bases = {}
    D = G.shape[2]
    for t in range(T):
        G_t = G[:, t, :]
        row_norms = np.linalg.norm(G_t, axis=1)
        G_t_eff = G_t[row_norms > 0].astype(np.float64)
        if G_t_eff.shape[0] < 2:
            bases[t] = np.zeros((D, 0))
            continue

        # Gram matrix and its eigendecomposition.
        # eigh returns eigenvalues in ASCENDING order — reverse them.
        GGt = G_t_eff @ G_t_eff.T                       # (n, n)
        eigvals, U = np.linalg.eigh(GGt)                # (n,), (n, n)
        eigvals = eigvals[::-1]
        U = U[:, ::-1]
        eigvals = np.maximum(eigvals, 0.0)              # numerical safety

        total = float(eigvals.sum())                    # = ||G_t_eff||_F^2
        if total <= 0:
            bases[t] = np.zeros((D, 0))
            continue

        cum_var = np.cumsum(eigvals) / total
        k = int(np.searchsorted(cum_var, explained_variance) + 1)
        k = min(k, len(eigvals))

        # Recover top-k right singular vectors:  V = G^T U / S
        S_top = np.sqrt(eigvals[:k])
        # Guard against tiny S that would blow up the division
        nz = S_top > 1e-12
        Vt_top = np.zeros((k, D), dtype=np.float64)
        if nz.any():
            Vt_top[nz] = (U[:, :k][:, nz].T @ G_t_eff) / S_top[nz, None]
        bases[t] = Vt_top.T                             # (D, k)
    return bases


# ═══════════════════════════════════════════════════════════════════
# Principal-angle stability between subspaces
# ═══════════════════════════════════════════════════════════════════
# Math (per pair):
#   # M             = B1^T B2           shape (k1, k2)
#   # cos(theta_i)  = i-th singular value of M
#   # mean_cos_sq   = mean(cos^2)       in [0, 1]
# ═══════════════════════════════════════════════════════════════════

def mean_cos_sq(B1, B2):
    if B1.shape[1] == 0 or B2.shape[1] == 0:
        return 0.0
    M = B1.T @ B2
    cos_sv = np.linalg.svd(M, compute_uv=False)
    cos_sv = np.clip(cos_sv, 0.0, 1.0)
    return float(np.mean(cos_sv ** 2))


def _compute_stability(bases, T):
    """Return (adjacent list, offdiag_mean float) from bases dict."""
    adjacent = [mean_cos_sq(bases[t], bases[t + 1]) for t in range(T - 1)]
    pairs = []
    for t in range(T):
        for tp in range(t + 1, T):
            pairs.append(mean_cos_sq(bases[t], bases[tp]))
    offdiag = float(np.mean(pairs)) if pairs else float("nan")
    return adjacent, offdiag


# ═══════════════════════════════════════════════════════════════════
# Bootstrap CIs on stability metrics
# ═══════════════════════════════════════════════════════════════════
#
# Instance axis is 0 of G (N, T, D). Each bootstrap draw resamples
# N rows with replacement, re-runs SVD per t, then recomputes both
# the adjacent cos^2 vector and the off-diagonal mean.

def bootstrap_stability(
    G, T, explained_variance=0.95,
    n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI, seed=DEFAULT_SEED,
):
    """
    Returns:
        adj_cis    : list of BootstrapResult, one per adjacent pair
        offdiag_ci : BootstrapResult for the global off-diagonal mean
    """
    N = G.shape[0]
    rng = np.random.default_rng(seed)

    # Point estimates from the full dataset.
    bases_full = _bases_from_G(G, T, explained_variance)
    adj_point, offdiag_point = _compute_stability(bases_full, T)

    # Bootstrap draws.
    n_adj = T - 1
    boot_adj = np.empty((n_boot, n_adj), dtype=np.float64)
    boot_offdiag = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        G_b = G[idx]
        bases_b = _bases_from_G(G_b, T, explained_variance)
        adj_b, off_b = _compute_stability(bases_b, T)
        boot_adj[b, :] = adj_b
        boot_offdiag[b] = off_b

    # Build CI results.
    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct

    adj_cis = []
    for t in range(n_adj):
        lo, hi = np.percentile(boot_adj[:, t], [lo_pct, hi_pct])
        adj_cis.append(BootstrapResult(
            metric=f"q3_adjacent_t{t}_t{t+1}",
            point=adj_point[t],
            ci_low=float(lo), ci_high=float(hi),
            ci_level=ci, n=N, n_boot=n_boot, seed=seed,
            method="function",
        ))

    lo, hi = np.percentile(boot_offdiag, [lo_pct, hi_pct])
    offdiag_ci = BootstrapResult(
        metric="q3_offdiag_mean",
        point=offdiag_point,
        ci_low=float(lo), ci_high=float(hi),
        ci_level=ci, n=N, n_boot=n_boot, seed=seed,
        method="function",
    )

    return adj_cis, offdiag_ci


# ═══════════════════════════════════════════════════════════════════
# Pretty-printing
# ═══════════════════════════════════════════════════════════════════

def print_stability(adj_cis, offdiag_ci):
    print("\n  Adjacent-step subspace stability (mean cos^2)")
    print(f"  {'(t, t+1)':>10}  {'point':>8}  {'95% CI':>20}")
    for t, r in enumerate(adj_cis):
        print(f"  ({t:>2}, {t + 1:>2})  {r.point:>8.4f}  "
              f"[{r.ci_low:.4f}, {r.ci_high:.4f}]")
    print(f"  Off-diagonal mean: {offdiag_ci.to_short_str()}  "
          f"(1 = same subspace, 0 = orthogonal)")


# ═══════════════════════════════════════════════════════════════════
# JSON-serialisable conversion
# ═══════════════════════════════════════════════════════════════════

def deep_convert(obj):
    if isinstance(obj, dict):
        return {str(k): deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Subspace-stability diagnostic on the gradient "
                    "subspace produced by gradient_subspace.py."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument(
        "--model",
        choices=["coconut", "coconut_u", "pause", "codi"],
        required=True,
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Must match the family used by "
             "gradient_subspace.py; selects the namespaced input/output dir.",
    )
    parser.add_argument("--bases_path", type=str, default=None,
                        help="Override path to bases.npz. Default: "
                             "BASE_DIR/outputs/gradient_geometry/"
                             "<family>/<task>/<model>/bases.npz")
    parser.add_argument("--explained_variance", type=float, default=0.95,
                        help="Cumulative energy threshold for SVD rank "
                             "selection during bootstrap re-SVD.")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Number of bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    # Namespaced by family to match gradient_subspace.py's output layout.
    output_dir = (BASE_DIR / "outputs" / "gradient_geometry"
                  / args.model_family / args.task / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases_path = (Path(args.bases_path) if args.bases_path
                  else output_dir / "bases.npz")
    grads_path = output_dir / "gradients.npz"

    if not bases_path.exists():
        raise FileNotFoundError(
            f"bases.npz not found at {bases_path}. "
            f"Run gradient_subspace.py first."
        )
    if not grads_path.exists():
        raise FileNotFoundError(
            f"gradients.npz not found at {grads_path}. "
            f"Run gradient_subspace.py first (needed for bootstrap)."
        )

    print(f"[main] task={args.task}  model={args.model}  family={args.model_family}")
    print(f"[main] bases_path: {bases_path}")
    print(f"[main] grads_path: {grads_path}")
    print(f"[main] output_dir: {output_dir}")

    # ── Load bases (for ranks display) and raw gradients ──────────
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

    # ── Print ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  GRADIENT-SUBSPACE STABILITY  ({args.task}/{args.model})")
    print("=" * 70)
    print_stability(adj_cis, offdiag_ci)

    # ── Save CI records (JSONL) ───────────────────────────────────
    cis_jsonl = str(output_dir / "bootstrap_cis.jsonl")
    ci_ctx = {"task": args.task, "model": args.model,
              "model_family": args.model_family}
    for r in adj_cis:
        save_record(cis_jsonl, r, context=ci_ctx)
    save_record(cis_jsonl, offdiag_ci, context=ci_ctx)
    print(f"\n[main] Bootstrap CIs -> {cis_jsonl}")

    # ── Save summary JSON ─────────────────────────────────────────
    summary = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "T": int(T),
        "subspace_ranks": ranks,
        "q3_adjacent_per_t": [r.point for r in adj_cis],
        "q3_adjacent_ci_per_t": [
            [r.ci_low, r.ci_high] for r in adj_cis
        ],
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