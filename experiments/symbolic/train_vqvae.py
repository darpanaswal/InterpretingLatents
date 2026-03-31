"""
Train a VQ-VAE codebook on extracted Coconut thought vectors.

The goal: determine whether continuous thought vectors cluster into a
small number of discrete states. We learn a codebook C = {c_1, ..., c_K}
and measure how well the quantized vectors preserve downstream structure.

This script trains codebooks for multiple K values and saves:
    - The learned codebook vectors
    - Per-vector codebook assignments
    - Reconstruction error statistics
    - Codebook utilization metrics
"""

import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from utils.config import VQVAE
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer (van den Oord et al., 2017).

    Given a set of input vectors z_e, snaps each to its nearest codebook entry:

        z_q = c_j   where   j = argmin_i || z_e - c_i ||_2

    Loss has three terms:

        L_total = L_reconstruct + L_codebook + beta * L_commit

        L_reconstruct = || x - decode(z_q) ||^2
            (not computed here — we measure it via downstream task accuracy)

        L_codebook = || sg[z_e] - z_q ||^2
            (pull codebook entries toward encoder outputs; sg = stop gradient)

        L_commit = || z_e - sg[z_q] ||^2
            (pull encoder outputs toward codebook entries)

    Since we have no encoder/decoder — our "inputs" are frozen thought
    vectors — we only optimize the codebook. The loss reduces to:

        L = || z_e - z_q ||^2
            (pure reconstruction: how well does the nearest codebook entry
             approximate each thought vector)

    We use exponential moving average (EMA) updates for the codebook,
    which is more stable than gradient descent for VQ:

        N_i(t) = gamma * N_i(t-1) + (1 - gamma) * n_i
        m_i(t) = gamma * m_i(t-1) + (1 - gamma) * sum(z_e assigned to i)
        c_i(t) = m_i(t) / N_i(t)

    where:
        N_i = running count of vectors assigned to codebook entry i
        m_i = running sum of vectors assigned to codebook entry i
        n_i = number of vectors assigned to entry i in this batch
        gamma = EMA decay (typically 0.99)
    """

    def __init__(self, num_codes: int, code_dim: int, gamma: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.gamma = gamma

        # Initialize codebook from N(0, 1) — will be overwritten by
        # k-means init in practice
        self.register_buffer("codebook", torch.randn(num_codes, code_dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.zeros(num_codes, code_dim))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize_from_data(self, data: torch.Tensor):
        """
        Initialize codebook entries via k-means++ seeding on a sample of data.

        data: (M, D) tensor of thought vectors.
        """
        M = data.shape[0]
        K = self.num_codes

        if M < K:
            raise ValueError(
                f"Not enough data points ({M}) to initialize {K} codebook entries. "
                f"Need at least K={K} data points."
            )

        # k-means++ initialization:
        # 1. Pick first center uniformly at random
        # 2. For each subsequent center, pick with probability proportional
        #    to D(x)^2, where D(x) = distance to nearest existing center
        indices = []
        idx = torch.randint(0, M, (1,)).item()
        indices.append(idx)

        for _ in range(1, K):
            # dists[j] = min distance from data[j] to any chosen center
            # Shape: (M, len(indices), D) -> (M, len(indices)) -> (M,)
            centers = data[indices]  # (len(indices), D)
            # || data[j] - centers[i] ||^2 for all j, i
            dists = torch.cdist(data, centers)  # (M, len(indices))
            min_dists = dists.min(dim=1).values  # (M,)
            # p(j) = D(j)^2 / sum(D^2)
            probs = min_dists ** 2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)

        self.codebook.copy_(data[indices])
        self.ema_sum.copy_(self.codebook.clone())
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)
        print(f"[INFO] Codebook initialized with k-means++ from {M} vectors")

    def quantize(self, z_e: torch.Tensor):
        """
        Snap each input vector to its nearest codebook entry.

        z_e: (B, D) input vectors.

        Returns:
            z_q: (B, D) quantized vectors
            indices: (B,) codebook entry indices
            commit_loss: scalar, || z_e - sg[z_q] ||^2
        """
        # || z_e - c_i ||^2 = ||z_e||^2 - 2 * z_e . c_i + ||c_i||^2
        # Expanding the L2 distance for efficient batched computation:
        #   dist(j, i) = sum(z_e[j]^2) - 2 * z_e[j] . c_i + sum(c_i^2)
        dists = (
            (z_e ** 2).sum(dim=1, keepdim=True)         # (B, 1)
            - 2 * z_e @ self.codebook.T                   # (B, K)
            + (self.codebook ** 2).sum(dim=1, keepdim=True).T  # (1, K)
        )  # (B, K)

        indices = dists.argmin(dim=1)  # (B,)
        z_q = self.codebook[indices]   # (B, D)

        # Commitment loss: || z_e - sg[z_q] ||^2
        commit_loss = F.mse_loss(z_e, z_q.detach())

        return z_q, indices, commit_loss

    def ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        """
        Update codebook via exponential moving average.

        EMA update rules:
            N_i(t) = gamma * N_i(t-1) + (1 - gamma) * n_i
            m_i(t) = gamma * m_i(t-1) + (1 - gamma) * sum_{j: assign(j)=i} z_e[j]
            c_i(t) = m_i(t) / N_i(t)

        Laplace smoothing to prevent dead codes:
            N_i(t) = (N_i(t) + epsilon) / (sum(N) + K * epsilon) * sum(N)
        """
        device = z_e.device
        K = self.num_codes
        epsilon = 1e-5

        # Count assignments per codebook entry: n_i
        one_hot = F.one_hot(indices, K).float()  # (B, K)
        n_i = one_hot.sum(dim=0)  # (K,)

        # Sum of assigned vectors per entry: sum_{j: assign(j)=i} z_e[j]
        sum_i = one_hot.T @ z_e  # (K, D)

        # EMA updates
        self.ema_count = self.gamma * self.ema_count + (1 - self.gamma) * n_i
        self.ema_sum = self.gamma * self.ema_sum + (1 - self.gamma) * sum_i

        # Laplace smoothing to prevent division by zero / dead codes
        N_total = self.ema_count.sum()
        count_smoothed = (
            (self.ema_count + epsilon)
            / (N_total + K * epsilon)
            * N_total
        )

        # Update codebook: c_i = m_i / N_i
        self.codebook = self.ema_sum / count_smoothed.unsqueeze(1)


def compute_metrics(
    z_e: torch.Tensor,
    z_q: torch.Tensor,
    indices: torch.Tensor,
    num_codes: int,
) -> dict:
    """
    Compute quality metrics for the quantization.

    Metrics:
        mse: mean squared reconstruction error = (1/N) * sum || z_e - z_q ||^2
        codebook_usage: fraction of codebook entries that are actually used
        perplexity: exp(H) where H = -sum p_i log p_i is the entropy of
                    the assignment distribution. Perfect uniform usage gives
                    perplexity = K; collapsed usage gives perplexity ≈ 1.
    """
    mse = F.mse_loss(z_e, z_q).item()

    counts = torch.bincount(indices, minlength=num_codes).float()
    usage = (counts > 0).float().mean().item()

    # Perplexity = exp(H)
    # H = -sum_i p_i * log(p_i), where p_i = count_i / total
    probs = counts / counts.sum()
    # Filter zeros to avoid log(0)
    probs_nonzero = probs[probs > 0]
    entropy = -(probs_nonzero * probs_nonzero.log()).sum().item()
    perplexity = float(np.exp(entropy))

    return {
        "mse": mse,
        "codebook_usage": usage,
        "perplexity": perplexity,
        "assignment_counts": counts.cpu().numpy().tolist(),
    }


def train_codebook(
    thoughts: torch.Tensor,
    num_codes: int,
    n_epochs: int = 50,
    batch_size: int = 256,
    gamma: float = 0.99,
    seed: int = 42,
) -> dict:
    """
    Train a VQ codebook on flattened thought vectors via EMA updates.

    thoughts: (N, K+1, D) tensor of thought vectors
    num_codes: codebook size to train

    Returns dict with codebook, assignments, metrics.
    """
    torch.manual_seed(seed)

    N, T, D = thoughts.shape  # T = K+1 (including step 0)
    # Flatten: every thought vector is a training point
    # Shape: (N * T, D)
    z_all = thoughts.reshape(-1, D)
    M = z_all.shape[0]

    print(f"\n{'='*60}")
    print(f"Training codebook: K={num_codes}, data={M} vectors ({N} instances × {T} steps), D={D}")
    print(f"{'='*60}")

    vq = VectorQuantizer(num_codes, D, gamma)
    vq.initialize_from_data(z_all)

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(M)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, M, batch_size):
            batch_idx = perm[start : start + batch_size]
            z_e = z_all[batch_idx]

            z_q, indices, commit_loss = vq.quantize(z_e)
            vq.ema_update(z_e, indices)

            total_loss += commit_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Quick metrics on full data
            z_q_all, idx_all, _ = vq.quantize(z_all)
            metrics = compute_metrics(z_all, z_q_all, idx_all, num_codes)
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}: "
                f"MSE={metrics['mse']:.4f}  "
                f"Usage={metrics['codebook_usage']:.2%}  "
                f"Perplexity={metrics['perplexity']:.1f}/{num_codes}"
            )

    # ── Final quantization on all data ──────────────────────────────
    z_q_all, idx_all, _ = vq.quantize(z_all)
    final_metrics = compute_metrics(z_all, z_q_all, idx_all, num_codes)

    # Reshape assignments back to (N, T)
    assignments = idx_all.reshape(N, T)

    return {
        "codebook": vq.codebook.cpu(),         # (num_codes, D)
        "assignments": assignments.cpu(),       # (N, T)
        "metrics": final_metrics,
        "num_codes": num_codes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train VQ-VAE codebooks on Coconut thought vectors."
    )
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument(
        "--thoughts_path",
        type=str,
        required=True,
        help="Path to .pt file from extract_thoughts.py",
    )
    parser.add_argument(
        "--codebook_sizes",
        type=str,
        default="1,2,3,4,8,16,32,64,128,256",
        help="Comma-separated codebook sizes K to sweep (default: 1,2,3,4,8,...,256).",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Training epochs per codebook (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for EMA updates (default: 256).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: VQVAE / model / vqvae_results/",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else VQVAE / f"{args.model}/vqvae_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load thought vectors ────────────────────────────────────────
    print(f"[INFO] Loading thoughts from {args.thoughts_path}")
    data = torch.load(args.thoughts_path, map_location="cpu")
    thoughts = data["thoughts"]  # (N, K+1, D)
    N, T, D = thoughts.shape
    print(f"[INFO] Shape: {N} instances × {T} steps × {D} dims")

    # ── Sweep codebook sizes ────────────────────────────────────────
    codebook_sizes = [int(k) for k in args.codebook_sizes.split(",")]
    all_results = {}

    for K in codebook_sizes:
        result = train_codebook(
            thoughts, K,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Save individual codebook
        save_path = output_dir / f"codebook_K{K}.pt"
        torch.save(result, save_path)
        print(f"  → Saved to {save_path}")

        all_results[K] = {
            "mse": result["metrics"]["mse"],
            "codebook_usage": result["metrics"]["codebook_usage"],
            "perplexity": result["metrics"]["perplexity"],
        }

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary: Codebook size vs. reconstruction quality")
    print(f"{'='*60}")
    print(f"{'K':>6}  {'MSE':>10}  {'Usage':>8}  {'Perplexity':>12}")
    print(f"{'-'*6}  {'-'*10}  {'-'*8}  {'-'*12}")
    for K in codebook_sizes:
        r = all_results[K]
        print(f"{K:>6}  {r['mse']:>10.4f}  {r['codebook_usage']:>7.1%}  {r['perplexity']:>11.1f}/{K}")

    # Save summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Sweep summary saved to {summary_path}")


if __name__ == "__main__":
    main()