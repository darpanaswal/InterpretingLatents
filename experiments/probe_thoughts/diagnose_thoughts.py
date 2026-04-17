"""
Diagnostic: Is the thought vector space dominated by timestep or instance
identity?

Runs on the original thoughts. Three core checks:
  1. Variance decomposition: between-timestep vs between-instance variance
  2. CKA across timesteps: instance-level similarity structure
  3. PCA visualization: trajectories in 2D
  4. Per-timestep instance variance: where instance info concentrates

Temporal-scaffolding extension (Reverse INLP)
─────────────────────────────────────────────
Hypothesis: thoughts = gamma(alpha * temporal + beta * instance), where
the temporal-scaffolding component lives in a low-dimensional subspace
of the 768-dim thought space.

Test via reverse INLP: fit a linear SVM that predicts timestep from
thought vectors (pooled across instances). Its weight matrix W has rows
= hyperplane normals that define the subspace used to decode timestep.
Projecting ONTO span(W) keeps only what discriminates timesteps, killing
everything orthogonal — which, under the hypothesis, is instance info.

# W: (T, D) SVM weight matrix, rows = hyperplane normals
# QR: W^T = Q R,  Q (D, k) orthonormal, k = rank(W)
# P_temporal = Q Q^T      (projects onto row-span(W))
# X_temporal = X @ P_temporal

Optionally iterates: residualize X, refit SVM on the residual, accumulate
directions until timestep becomes unpredictable. The accumulated subspace
is the full temporal scaffolding.

If the hypothesis holds, PCA on X_temporal will show clean temporal
clustering, and a classifier for the INSTANCE answer trained on
X_temporal will perform at chance — demonstrating that temporal
scaffolding is orthogonal to reasoning semantics.

Usage:
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model pause
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model coconut
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model coconut_u
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model pause
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model coconut
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model coconut_u
    python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model codi
"""

import sys
import torch
import argparse
import matplotlib
import cupy as cp
import numpy as np
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cuml.svm import LinearSVC as cuLinearSVC

from src.config import THOUGHTS


# --- Helper for logging ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_thoughts(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = data["thoughts"]  # (N, T, D)
    print(f"[INFO] Loaded thoughts: {thoughts.shape}")
    return thoughts


# ═══════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════

def variance_decomposition(thoughts):
    """
    # var_total    = E[||h_it - mu||^2]
    # var_timestep = E[||mu_t - mu||^2]     (between-timestep)
    # var_instance = E[||mu_i - mu||^2]     (between-instance)
    # var_residual = var_total - var_timestep - var_instance
    """
    N, T, D = thoughts.shape

    mu = thoughts.mean(dim=(0, 1))
    mu_t = thoughts.mean(dim=0)
    mu_i = thoughts.mean(dim=1)

    var_total = ((thoughts - mu) ** 2).sum(dim=2).mean().item()
    var_timestep = ((mu_t - mu) ** 2).sum(dim=1).mean().item()
    var_instance = ((mu_i - mu) ** 2).sum(dim=1).mean().item()
    var_residual = var_total - var_timestep - var_instance

    print(f"\n{'='*60}")
    print("VARIANCE DECOMPOSITION")
    print(f"{'='*60}")
    print(f"  Total variance:       {var_total:.4f}")
    print(f"  Timestep component:   {var_timestep:.4f}  "
          f"({var_timestep/var_total*100:.1f}%)")
    print(f"  Instance component:   {var_instance:.4f}  "
          f"({var_instance/var_total*100:.1f}%)")
    print(f"  Residual (interact.): {var_residual:.4f}  "
          f"({var_residual/var_total*100:.1f}%)")

    if var_timestep > var_instance:
        ratio = var_timestep / max(var_instance, 1e-8)
        print(f"\n  Timestep variance is {ratio:.1f}x larger than instance "
              f"variance.")
        print(f"  → Thought vectors are primarily organized by WHEN, "
              f"not WHAT.")
    else:
        ratio = var_instance / max(var_timestep, 1e-8)
        print(f"\n  Instance variance is {ratio:.1f}x larger than timestep "
              f"variance.")
        print(f"  → Thought vectors are primarily organized by WHAT, "
              f"not WHEN.")

    return var_total, var_timestep, var_instance, var_residual


def per_timestep_instance_variance(thoughts, output_dir, tag):
    """
    # Var_inst(t) = (1/N) * sum_i || h_{i,t} - mu_t ||^2
    # where mu_t = (1/N) * sum_i h_{i,t}
    """
    N, T, D = thoughts.shape

    mu_t = thoughts.mean(dim=0)
    deviations = thoughts - mu_t.unsqueeze(0)
    per_step_var = (deviations ** 2).sum(dim=2).mean(dim=0)  # (T,)

    mu_global = thoughts.mean(dim=(0, 1))
    centroid_offset = ((mu_t - mu_global) ** 2).sum(dim=1)  # (T,)

    print(f"\n{'='*60}")
    print("PER-TIMESTEP INSTANCE VARIANCE")
    print(f"{'='*60}")
    print(f"  {'Step':>6}  {'Var_inst(t)':>14}  {'% of max':>10}  "
          f"{'Centroid offset':>16}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*10}  {'-'*16}")

    max_var = per_step_var.max().item()
    for t in range(T):
        v = per_step_var[t].item()
        pct = v / max_var * 100 if max_var > 0 else 0
        co = centroid_offset[t].item()
        print(f"  {t:>6}  {v:>14.2f}  {pct:>9.1f}%  {co:>16.2f}")

    min_var = per_step_var.min().item()
    ratio = max_var / max(min_var, 1e-8)
    print(f"\n  Max/min ratio: {ratio:.1f}x")

    # Check for alternating pattern
    if T >= 4:
        even_mean = per_step_var[0::2].mean().item()
        odd_mean = per_step_var[1::2].mean().item()
        print(f"  Even-step mean: {even_mean:.2f}")
        print(f"  Odd-step mean:  {odd_mean:.2f}")
        if even_mean > odd_mean * 1.5:
            print(f"  → Even steps carry "
                  f"{even_mean/max(odd_mean,1e-8):.1f}x more instance info "
                  f"(alternating pattern)")
        elif odd_mean > even_mean * 1.5:
            print(f"  → Odd steps carry "
                  f"{odd_mean/max(even_mean,1e-8):.1f}x more instance info "
                  f"(alternating pattern)")
        else:
            print(f"  → No strong alternating pattern")

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(T)
    ax.bar(steps, per_step_var.numpy(), color='steelblue', edgecolor='white')
    ax.set_xlabel("Thought step $t$")
    ax.set_ylabel("Instance variance $\\mathrm{Var}_{\\mathrm{inst}}(t)$")
    ax.set_title(f"Per-timestep instance variance ({tag})")
    ax.set_xticks(steps)
    fig.tight_layout()
    path = output_dir / f"per_timestep_instance_var_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved to {path}")

    return per_step_var


def cka_across_timesteps(thoughts, output_dir, tag):
    """
    Linear CKA (Kornblith et al., 2019) between all pairs of timesteps,
    with each timestep centered so only instance-level structure
    contributes.

    # X_t centered: X_t[i,:] <- h_{i,t} - mu_t
    # CKA(s, t) = ||X_t^T X_s||_F^2 / (||X_s^T X_s||_F * ||X_t^T X_t||_F)
    """
    N, T, D = thoughts.shape

    mu_t = thoughts.mean(dim=0)
    centered = (thoughts - mu_t.unsqueeze(0)).numpy()

    # hsic_self[t] = ||X_t^T X_t||_F
    hsic_self = np.zeros(T)
    for t in range(T):
        XtX = centered[:, t, :].T @ centered[:, t, :]
        hsic_self[t] = np.linalg.norm(XtX, 'fro')

    cka_matrix = np.zeros((T, T))
    for s in range(T):
        for t in range(T):
            YtX = centered[:, t, :].T @ centered[:, s, :]
            cross = np.linalg.norm(YtX, 'fro') ** 2
            denom = hsic_self[s] * hsic_self[t]
            cka_matrix[s, t] = cross / max(denom, 1e-12)

    print(f"\n{'='*60}")
    print("CKA ACROSS TIMESTEPS (instance-level similarity)")
    print(f"{'='*60}")

    header = "      " + "".join(f"  t={t:d}  " for t in range(T))
    print(header)
    for s in range(T):
        row = f"  t={s:d} "
        for t in range(T):
            row += f" {cka_matrix[s, t]:6.3f} "
        print(row)

    # ── Pattern classification (checkerboard / drift / fragmented / uniform) ──
    if T >= 4:
        within_parity = []
        cross_parity = []
        for s in range(T):
            for t in range(s + 1, T):
                if s % 2 == t % 2:
                    within_parity.append(cka_matrix[s, t])
                else:
                    cross_parity.append(cka_matrix[s, t])
        wp_mean = np.mean(within_parity)
        cp_mean = np.mean(cross_parity)

        adjacent = [cka_matrix[t, t+1] for t in range(T-1)]
        distant = [cka_matrix[0, t] for t in range(2, T)]
        adj_mean = np.mean(adjacent)
        dist_mean = np.mean(distant)

        drift_ratio = adj_mean / max(dist_mean, 1e-8)
        checker_ratio = wp_mean / max(cp_mean, 1e-8)

        print(f"\n  Within-parity CKA (even-even, odd-odd): {wp_mean:.3f}")
        print(f"  Cross-parity CKA (even-odd):            {cp_mean:.3f}")
        print(f"  Checkerboard ratio (within/cross):       "
              f"{checker_ratio:.2f}x")
        print(f"\n  Adjacent-step CKA (t, t+1): {adj_mean:.3f}")
        print(f"  Distant-step CKA (0, t>=2):  {dist_mean:.3f}")
        print(f"  Drift ratio (adjacent/distant):          "
              f"{drift_ratio:.2f}x")

        if checker_ratio > 2.0 and cp_mean < 0.15:
            print(f"\n  → CHECKERBOARD: even/odd steps encode structurally")
            print(f"    different instance info (within/cross = "
                  f"{checker_ratio:.1f}x)")
        elif drift_ratio > 1.5:
            print(f"\n  → DRIFT: instance info gradually transforms across "
                  f"steps (adjacent/distant = {drift_ratio:.1f}x)")
        elif wp_mean < 0.25 and cp_mean < 0.25:
            print(f"\n  → FRAGMENTED: each step encodes different instance "
                  f"info (all pairwise CKA < 0.25)")
        else:
            print(f"\n  → UNIFORM: instance info similarly distributed "
                  f"across timesteps")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    fig.colorbar(im, ax=ax, label='Linear CKA')
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"t={t}" for t in range(T)])
    ax.set_yticklabels([f"t={t}" for t in range(T)])
    ax.set_title(f"CKA: instance-level similarity ({tag})")
    for s in range(T):
        for t in range(T):
            color = 'white' if cka_matrix[s, t] < 0.5 else 'black'
            ax.text(t, s, f"{cka_matrix[s, t]:.2f}", ha='center', va='center',
                    fontsize=7, color=color)
    fig.tight_layout()
    path = output_dir / f"cka_heatmap_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Heatmap saved to {path}")

    return cka_matrix


def pca_visualization(thoughts, output_dir, tag):
    """2D PCA: timestep-colored scatter + per-instance trajectories."""
    N, T, D = thoughts.shape
    flat = thoughts.reshape(-1, D).numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    coords_by_instance = coords.reshape(N, T, 2)
    timestep_colors = np.tile(np.arange(T), N)

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=timestep_colors,
                          cmap="viridis", s=3, alpha=0.5)
    fig1.colorbar(scatter, ax=ax1).set_label("Timestep $t$")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title(f"PCA ({tag} - colored by timestep)")
    path1 = output_dir / f"thought_pca_timestep_{tag}.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab20
    for i in range(min(N, 100)):
        ax2.plot(coords_by_instance[i, :, 0], coords_by_instance[i, :, 1],
                 '-o', color=cmap(i % 20 / 20.0),
                 markersize=2, linewidth=0.5, alpha=0.3)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title(f"PCA Trajectories ({tag})")
    path2 = output_dir / f"thought_pca_trajectories_{tag}.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    print(f"\n  Timestep PCA plot saved to {path1}")
    print(f"  Trajectory PCA plot saved to {path2}")
    return pca


def run_diagnostic_suite(thoughts, output_dir, tag):
    """All core diagnostics on a single (N, T, D) tensor."""
    variance_decomposition(thoughts)
    cka_across_timesteps(thoughts, output_dir, tag)
    per_timestep_instance_variance(thoughts, output_dir, tag)
    pca_visualization(thoughts, output_dir, tag)


# ═══════════════════════════════════════════════════════════════════
# Reverse INLP: project ONTO the timestep-decodable subspace
# ═══════════════════════════════════════════════════════════════════

def fit_timestep_probe(X_pooled, t_pooled, C=0.1, max_iter=5000):
    """
    Fit a one-vs-rest linear SVM (GPU-accelerated via cuML) to predict
    timestep from thought vectors.
    """
    # 1. Force C-contiguity on the host before sending to GPU
    X_contig = np.ascontiguousarray(X_pooled, dtype=np.float32)
    
    # 2. Use int32 for class labels (safer for cuML multi-class)
    y_contig = np.ascontiguousarray(t_pooled, dtype=np.int32) 

    # 3. Transfer to GPU
    X_gpu = cp.asarray(X_contig)
    y_gpu = cp.asarray(y_contig)

    # 4. Fit model
    clf = cuLinearSVC(C=C, max_iter=max_iter, output_type='cupy')
    clf.fit(X_gpu, y_gpu)
    
    W = cp.asnumpy(clf.coef_).astype(np.float32)  # (T, D) for T > 2
    preds = clf.predict(X_gpu)
    acc = float(cp.mean(preds == y_gpu))
    
    return W, acc


def temporal_subspace_projector(W_stack):
    """
    Build the orthogonal projector onto row-span(W_stack) via QR.

    # W_stack: (k, D), rows = accumulated hyperplane normals
    # W_stack^T = Q R,   Q (D, k_eff) orthonormal, k_eff = rank(W_stack)
    # P_temporal = Q Q^T
    # Equivalent to Gemini's W^T (W W^T)^{-1} W but numerically stable
    # when rows are near-collinear.
    """
    Q, _ = np.linalg.qr(W_stack.T.astype(np.float64))
    col_norms = np.linalg.norm(Q, axis=0)
    Q = Q[:, col_norms > 1e-8]
    D = W_stack.shape[1]
    P = (Q @ Q.T).astype(np.float32)
    return P, int(Q.shape[1])


def reverse_inlp(thoughts, n_iter=1, svm_C=0.1, seed=0):
    """
    Iteratively decode timestep, accumulate hyperplane normals, build
    the projector onto the full temporal-decodable subspace.

    # Loop for i = 0 .. n_iter-1:
    #   W_i = SVM(X_residual, t_pooled).coef_
    #   Accumulate W_i into W_all
    #   Q_i, _ = QR(W_i^T); residualize: X_residual <- X_residual (I - Q_i Q_i^T)
    # P_temporal = Q_all Q_all^T  (orthonormalized basis of W_all)

    Returns:
        P_temporal: (D, D) projector onto the temporal subspace
        stats: dict with per-iteration probe accuracies and rank info
    """
    N, T, D = thoughts.shape
    X_pooled = thoughts.reshape(N * T, D).numpy().astype(np.float32)
    t_pooled = np.tile(np.arange(T, dtype=np.int64), N)

    # Baseline: chance = 1/T, probe accuracy tells us how decodable t is
    majority_acc = 1.0 / T

    W_accumulated = []
    probe_accs = []
    residual = X_pooled.copy()

    for i in range(n_iter):
        W_i, acc = fit_timestep_probe(residual, t_pooled, C=svm_C)
        probe_accs.append(acc)
        W_accumulated.append(W_i)

        # Residualize: remove directions we just captured
        Q_i, _ = np.linalg.qr(W_i.T.astype(np.float64))
        col_norms = np.linalg.norm(Q_i, axis=0)
        Q_i = Q_i[:, col_norms > 1e-8]
        P_nullspace_step = (
            np.eye(D, dtype=np.float32)
            - (Q_i @ Q_i.T).astype(np.float32)
        )
        residual = residual @ P_nullspace_step.T

        # Early stop if probe is already near chance on the residual
        if acc < majority_acc + 0.02:
            break

    W_all = np.concatenate(W_accumulated, axis=0)
    P_temporal, rank = temporal_subspace_projector(W_all)

    # Final probe on the projected data: should achieve ~original accuracy
    # (we kept exactly the directions the probe uses)
    X_projected = X_pooled @ P_temporal.T
    _, final_acc = fit_timestep_probe(X_projected, t_pooled, C=svm_C)

    stats = {
        "n_iterations": len(W_accumulated),
        "rank_temporal_subspace": rank,
        "probe_acc_per_iter": probe_accs,
        "probe_acc_on_temporal_projection": final_acc,
        "chance_level": majority_acc,
    }
    return P_temporal, stats


def run_reverse_inlp_analysis(thoughts, output_dir, n_iter, seed=0):
    """
    Run reverse INLP, project thoughts onto the temporal subspace,
    run the full diagnostic suite on the projection.
    """
    N, T, D = thoughts.shape
    rev_dir = output_dir / "reverse_inlp"
    rev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("REVERSE INLP: FITTING TEMPORAL SUBSPACE")
    print(f"{'='*60}")
    print(f"  Thoughts: ({N}, {T}, {D}), n_iter={n_iter}")

    P_temporal, stats = reverse_inlp(thoughts, n_iter=n_iter, seed=seed)

    print(f"\n  Iterations run:              {stats['n_iterations']}")
    print(f"  Temporal subspace rank:      {stats['rank_temporal_subspace']} "
          f"/ {D}")
    print(f"  Probe accuracy per iter:     "
          f"{['%.3f' % a for a in stats['probe_acc_per_iter']]}")
    print(f"  Chance level (1/T):          {stats['chance_level']:.3f}")
    print(f"  Probe acc on projection:     "
          f"{stats['probe_acc_on_temporal_projection']:.3f}")
    print(f"  Fraction of D kept:          "
          f"{stats['rank_temporal_subspace'] / D * 100:.1f}%")

    # Apply projection: (N, T, D) @ P_temporal^T = (N, T, D)
    # P_temporal is symmetric (P = Q Q^T), so P^T = P
    thoughts_temporal = torch.from_numpy(
        thoughts.numpy() @ P_temporal.T
    ).to(thoughts.dtype)

    # Compute how much variance survived the projection
    var_before = (thoughts ** 2).sum(dim=2).mean().item()
    var_after = (thoughts_temporal ** 2).sum(dim=2).mean().item()
    print(f"\n  Mean squared norm before:    {var_before:.4f}")
    print(f"  Mean squared norm after:     {var_after:.4f}")
    print(f"  Fraction of norm^2 retained: {var_after / var_before * 100:.1f}%")

    # Full diagnostic suite on the projected thoughts
    print(f"\n{'#'*70}")
    print(f"# DIAGNOSTICS ON TEMPORAL-ONLY PROJECTION")
    print(f"{'#'*70}")
    run_diagnostic_suite(thoughts_temporal, rev_dir, "temporal_only")

    # Save the projected tensor so downstream scripts can consume it
    save_path = rev_dir / "thoughts_temporal_only.pt"
    torch.save({
        "thoughts": thoughts_temporal,
        "P_temporal": P_temporal,
        "stats": stats,
    }, save_path)
    print(f"\n  Projected thoughts saved to {save_path}")

    return thoughts_temporal, stats


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose thought vector structure on original thoughts."
    )
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"],
                        default="prosqa")
    parser.add_argument("--model", type=str,
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        default="coconut")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--reverse_inlp_iter", type=int, default=1,
                        help="Iterations of reverse INLP. 1 = single SVM "
                             "pass (rank-T temporal subspace). Higher = "
                             "iterate to capture the full decodable "
                             "temporal subspace (stops at chance).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model == "codi" and args.task != "gsm":
        parser.error("CODI is only available for --task gsm")

    model_dir = Path(args.output_dir) if args.output_dir else \
        THOUGHTS / args.task / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_dir / "diagnostics.txt"
    sys.stdout = Logger(log_path)

    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        print(f"[ERROR] Thoughts not found at {thoughts_path}. "
              f"Run extract_thoughts.py first.")
        return

    print(f"DIAGNOSTIC LOG: Task={args.task}, Model={args.model}")

    # ── Standard diagnostics on original thoughts ──────────────────
    print(f"\n{'#'*70}")
    print(f"# STANDARD DIAGNOSTICS (original thoughts)")
    print(f"{'#'*70}")
    thoughts = load_thoughts(thoughts_path)
    orig_dir = model_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)
    run_diagnostic_suite(thoughts, orig_dir, "original")

    # ── Reverse INLP: isolate the temporal subspace ────────────────
    print(f"\n{'#'*70}")
    print(f"# REVERSE INLP: TEMPORAL-SUBSPACE ISOLATION")
    print(f"{'#'*70}")
    run_reverse_inlp_analysis(
        thoughts, model_dir, n_iter=args.reverse_inlp_iter, seed=args.seed,
    )

    print(f"\n[COMPLETE] All output saved to {model_dir}")


if __name__ == "__main__":
    main()