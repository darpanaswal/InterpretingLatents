"""
Diagnostic: Is the thought vector space dominated by timestep or instance identity?

Three checks:
1. Variance decomposition: between-timestep vs between-instance variance
2. Cosine similarity: within-timestep (across instances) vs within-instance (across steps)
3. Codebook assignment breakdown: which timesteps map to which codebook entries

Usage:
    python diagnose_thoughts.py --thoughts_path outputs/vqvae/coconut_u/thoughts_coconut_u.pt
    python diagnose_thoughts.py --thoughts_path outputs/vqvae/coconut_u/thoughts_coconut_u.pt \
        --codebook_path outputs/vqvae/coconut_u/vqvae_results/codebook_K4.pt
"""

import torch
import argparse
import matplotlib
import numpy as np
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
from utils.config import BASE_DIR
from sklearn.decomposition import PCA


def load_thoughts(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = data["thoughts"]  # (N, T, D)
    print(f"[INFO] Loaded thoughts: {thoughts.shape}")
    return thoughts


def variance_decomposition(thoughts):
    """
    Decompose total variance into timestep and instance components.

    Total variance of a thought vector h_{i,t}:
        Var_total = E[ || h_{i,t} - mu ||^2 ]

    Decompose via:
        h_{i,t} = mu + (mu_t - mu) + (mu_i - mu) + residual

    where:
        mu = grand mean over all (i, t)
        mu_t = mean over all instances at timestep t
        mu_i = mean over all timesteps for instance i

    Between-timestep variance:
        Var_timestep = E_t[ || mu_t - mu ||^2 ]
        (how much do timestep means differ from grand mean)

    Between-instance variance:
        Var_instance = E_i[ || mu_i - mu ||^2 ]
        (how much do instance means differ from grand mean)

    Residual = Var_total - Var_timestep - Var_instance
        (interaction + noise)
    """
    N, T, D = thoughts.shape

    # Grand mean: (D,)
    mu = thoughts.mean(dim=(0, 1))

    # Timestep means: (T, D) — average across instances for each step
    mu_t = thoughts.mean(dim=0)

    # Instance means: (N, D) — average across steps for each instance
    mu_i = thoughts.mean(dim=1)

    # Var_total = (1/NT) * sum_{i,t} || h_{i,t} - mu ||^2
    var_total = ((thoughts - mu) ** 2).sum(dim=2).mean().item()

    # Var_timestep = (1/T) * sum_t || mu_t - mu ||^2
    var_timestep = ((mu_t - mu) ** 2).sum(dim=1).mean().item()

    # Var_instance = (1/N) * sum_i || mu_i - mu ||^2
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
    """
    Compare:
        - Within-timestep similarity: cosine(h_{i,t}, h_{j,t}) for random i,j pairs
          (same step, different instances — should be high if timestep dominates)
        - Within-instance similarity: cosine(h_{i,s}, h_{i,t}) for random s,t pairs
          (same instance, different steps — should be high if instance dominates)
    """
    N, T, D = thoughts.shape
    n_pairs = min(5000, N * (N - 1) // 2)

    # Within-timestep: same t, different instances
    within_timestep_sims = []
    for _ in range(n_pairs):
        t = np.random.randint(T)
        i, j = np.random.choice(N, 2, replace=False)
        a = thoughts[i, t]
        b = thoughts[j, t]
        # cos(a, b) = (a . b) / (||a|| * ||b||)
        sim = (a @ b) / (a.norm() * b.norm() + 1e-8)
        within_timestep_sims.append(sim.item())

    # Within-instance: same instance, different steps
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


def pca_visualization(thoughts, output_dir):
    """
    PCA on all thought vectors, colored by timestep and by instance.
    """
    N, T, D = thoughts.shape
    flat = thoughts.reshape(-1, D).numpy()  # (N*T, D)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)  # (N*T, 2)

    # Reshape back
    coords_by_instance = coords.reshape(N, T, 2)

    # Timestep labels: 0,0,...,0, 1,1,...,1, etc.
    timestep_labels = np.repeat(np.arange(T), N)
    # For coloring by timestep, we need (N*T,) in instance-major order
    # flat is (N*T,) in instance-major: [inst0_t0, inst0_t1, ..., inst0_t6, inst1_t0, ...]
    timestep_colors = np.tile(np.arange(T), N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: colored by timestep
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=timestep_colors,
                          cmap="viridis", s=3, alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax1, label="Timestep $t$")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("(a) Colored by timestep")

    # Panel B: colored by instance (subsample for visibility)
    n_show = min(50, N)
    cmap = plt.cm.tab20
    for i in range(n_show):
        color = cmap(i / n_show)
        ax2.plot(coords_by_instance[i, :, 0], coords_by_instance[i, :, 1],
                 '-o', color=color, markersize=3, linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title(f"(b) Trajectories for {n_show} instances")

    fig.suptitle("PCA of Continuous Thought Vectors", fontsize=14)
    fig.tight_layout()

    path = output_dir / "thought_pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  PCA plot saved to {path}")

    print(f"\n  PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    return pca


def codebook_timestep_breakdown(thoughts, codebook_path):
    """
    For each codebook entry, show which timesteps get assigned to it.
    """
    cb_data = torch.load(codebook_path, map_location="cpu", weights_only=False)
    codebook = cb_data["codebook"]  # (K, D)
    num_codes = cb_data["num_codes"]
    N, T, D = thoughts.shape

    print(f"\n{'='*60}")
    print(f"CODEBOOK ASSIGNMENT BREAKDOWN (K={num_codes})")
    print(f"{'='*60}")

    # Assign every thought vector to nearest codebook entry
    flat = thoughts.reshape(-1, D)  # (N*T, D)
    # dists[i, j] = || flat[i] - codebook[j] ||^2
    dists = (
        (flat ** 2).sum(dim=1, keepdim=True)
        - 2 * flat @ codebook.T
        + (codebook ** 2).sum(dim=1, keepdim=True).T
    )
    assignments = dists.argmin(dim=1).reshape(N, T)  # (N, T)

    # For each codebook entry, count how many assignments come from each timestep
    print(f"\n  Codebook entry → timestep distribution:")
    print(f"  {'Entry':>6}  {'Total':>6}  " + "  ".join([f"t={t}" for t in range(T)]))
    print(f"  {'-'*6}  {'-'*6}  " + "  ".join([f"{'---':>4}"] * T))

    for c in range(num_codes):
        counts_by_t = []
        total = 0
        for t in range(T):
            count = (assignments[:, t] == c).sum().item()
            counts_by_t.append(count)
            total += count
        counts_str = "  ".join([f"{ct:>4}" for ct in counts_by_t])
        print(f"  {c:>6}  {total:>6}  {counts_str}")

    # Also show: for each timestep, which entry gets the majority
    print(f"\n  Timestep → majority codebook entry:")
    for t in range(T):
        col = assignments[:, t]
        counts = torch.bincount(col, minlength=num_codes)
        majority = counts.argmax().item()
        majority_frac = counts[majority].item() / N
        print(f"    t={t}: entry {majority} ({majority_frac:.0%} of instances)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose thought vector structure.")
    parser.add_argument("--thoughts_path", type=str, required=True)
    parser.add_argument("--codebook_path", type=str, default=None,
                        help="Optional: codebook .pt file for assignment breakdown.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    thoughts = load_thoughts(args.thoughts_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.thoughts_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_decomposition(thoughts)
    cosine_similarity_analysis(thoughts)
    pca_visualization(thoughts, output_dir)

    if args.codebook_path:
        codebook_timestep_breakdown(thoughts, args.codebook_path)


if __name__ == "__main__":
    main()