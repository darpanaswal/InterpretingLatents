"""
Diagnostic: Is the thought vector space dominated by timestep or instance identity?

Three checks:
1. Variance decomposition: between-timestep vs between-instance variance
2. Cosine similarity: within-timestep (across instances) vs within-instance (across steps)
3. PCA visualization: visualizes the trajectories in 2D space

Usage:
    python -m experiments.probe_thoughts.diagnose_thoughts --model pause
    python -m experiments.probe_thoughts.diagnose_thoughts --model coconut
    python -m experiments.probe_thoughts.diagnose_thoughts --model coconut_u
"""

import torch
import argparse
import matplotlib
import numpy as np
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
from utils.config import THOUGHTS
from sklearn.decomposition import PCA


def load_thoughts(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = data["thoughts"]  # (N, T, D)
    print(f"[INFO] Loaded thoughts: {thoughts.shape}")
    return thoughts


def variance_decomposition(thoughts):
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
    print(f"  Timestep component:   {var_timestep:.4f}  ({var_timestep/var_total*100:.1f}%)")
    print(f"  Instance component:   {var_instance:.4f}  ({var_instance/var_total*100:.1f}%)")
    print(f"  Residual (interact.): {var_residual:.4f}  ({var_residual/var_total*100:.1f}%)")

    if var_timestep > var_instance:
        ratio = var_timestep / max(var_instance, 1e-8)
        print(f"\n  Timestep variance is {ratio:.1f}x larger than instance variance.")
        print(f"  → Thought vectors are primarily organized by WHEN, not WHAT.")
    else:
        ratio = var_instance / max(var_timestep, 1e-8)
        print(f"\n  Instance variance is {ratio:.1f}x larger than timestep variance.")
        print(f"  → Thought vectors are primarily organized by WHAT, not WHEN.")

    return var_total, var_timestep, var_instance, var_residual


def cosine_similarity_analysis(thoughts):
    N, T, D = thoughts.shape
    n_pairs = min(5000, N * (N - 1) // 2)

    within_timestep_sims = []
    for _ in range(n_pairs):
        t = np.random.randint(T)
        i, j = np.random.choice(N, 2, replace=False)
        a = thoughts[i, t]
        b = thoughts[j, t]
        sim = (a @ b) / (a.norm() * b.norm() + 1e-8)
        within_timestep_sims.append(sim.item())

    within_instance_sims = []
    for _ in range(n_pairs):
        i = np.random.randint(N)
        s, t = np.random.choice(T, 2, replace=False)
        a = thoughts[i, s]
        b = thoughts[i, t]
        sim = (a @ b) / (a.norm() * b.norm() + 1e-8)
        within_instance_sims.append(sim.item())

    wt_mean = np.mean(within_timestep_sims)
    wt_std = np.std(within_timestep_sims)
    wi_mean = np.mean(within_instance_sims)
    wi_std = np.std(within_instance_sims)

    print(f"\n{'='*60}")
    print("COSINE SIMILARITY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Within-timestep (same step, diff instances): {wt_mean:.4f} ± {wt_std:.4f}")
    print(f"  Within-instance (same instance, diff steps): {wi_mean:.4f} ± {wi_std:.4f}")

    if wt_mean > wi_mean:
        print(f"\n  Vectors at the same timestep are MORE similar to each other")
        print(f"  than vectors from the same instance at different steps.")
        print(f"  → Timestep identity dominates the representation.")
    else:
        print(f"\n  Vectors from the same instance are MORE similar to each other")
        print(f"  than vectors at the same timestep from different instances.")
        print(f"  → Instance identity dominates the representation.")

    return wt_mean, wt_std, wi_mean, wi_std


def pca_visualization(thoughts, output_dir, model):
    N, T, D = thoughts.shape
    flat = thoughts.reshape(-1, D).numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)

    coords_by_instance = coords.reshape(N, T, 2)
    timestep_colors = np.tile(np.arange(T), N)

    # =========================
    # (1) TIMESTEP SCATTER PLOT
    # =========================
    fig1, ax1 = plt.subplots(figsize=(7, 6))

    scatter = ax1.scatter(
        coords[:, 0], coords[:, 1],
        c=timestep_colors,
        cmap="viridis",
        s=3,
        alpha=0.5
    )

    cbar = fig1.colorbar(scatter, ax=ax1)
    cbar.set_label("Timestep $t$")

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("PCA (colored by timestep)")

    fig1.tight_layout()
    path1 = output_dir / f"thought_pca_timestep_{model}.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # =========================
    # (2) TRAJECTORY PLOT (ALL INSTANCES)
    # =========================
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    cmap = plt.cm.tab20

    for i in range(N):  # ALL 500
        color = cmap(i % 20 / 20.0)  # cycle colors
        ax2.plot(
            coords_by_instance[i, :, 0],
            coords_by_instance[i, :, 1],
            '-o',
            color=color,
            markersize=2,
            linewidth=0.5,
            alpha=0.3
        )

    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title(f"PCA Trajectories ({N} instances)")

    fig2.tight_layout()
    path2 = output_dir / f"thought_pca_trajectories_{model}.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    print(f"\n  Timestep PCA plot saved to {path1}")
    print(f"  Trajectory PCA plot saved to {path2}")
    print(f"\n  PCA explained variance: "
          f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    return pca


def main():
    parser = argparse.ArgumentParser(description="Diagnose thought vector structure.")
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    thoughts_path = THOUGHTS / f"thoughts_{args.model}.pt"
    thoughts = load_thoughts(thoughts_path)
    output_dir = Path(args.output_dir) if args.output_dir else thoughts_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_decomposition(thoughts)
    cosine_similarity_analysis(thoughts)
    pca_visualization(thoughts, output_dir, args.model)


if __name__ == "__main__":
    main()