"""
Diagnostic: Codebook assignment breakdown for VQ-VAE on continuous thoughts.

Analyzes which timesteps map to which codebook entries to determine 
if the learned discrete states are temporal (step-specific) or spatial/concept-based.

Usage:
    python -m experiments.probe_thoughts.diagnose_codebook --model coconut --k "4,8,16"
"""

import torch
import argparse
from pathlib import Path
from src.config import THOUGHTS, VQVAE


def load_thoughts(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = data["thoughts"]  # (N, T, D)
    print(f"[INFO] Loaded thoughts: {thoughts.shape}")
    return thoughts


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
    parser = argparse.ArgumentParser(description="Diagnose VQ-VAE codebook assignments.")
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument(
        "--k", type=str, required=True,
        help="Comma-separated list of K values to analyze (e.g., '4,8,16')."
    )
    args = parser.parse_args()

    thoughts_path = THOUGHTS / f"thoughts_{args.model}.pt"
    
    if not thoughts_path.exists():
        print(f"[ERROR] Thoughts file not found at {thoughts_path}")
        return
        
    thoughts = load_thoughts(thoughts_path)

    # Parse the comma-separated string into a list of integers
    try:
        k_values = [int(k.strip()) for k in args.k.split(",")]
    except ValueError:
        print("[ERROR] The --k argument must be a comma-separated list of integers.")
        return

    # Loop through the parsed K values
    for k in k_values:
        codebook_path = VQVAE / f"{args.model}/vqvae_results/codebook_K{k}.pt"
        
        if not codebook_path.exists():
            print(f"\n[INFO] Codebook K={k} not found at {codebook_path}. Skipping.")
            continue
            
        codebook_timestep_breakdown(thoughts, codebook_path)


if __name__ == "__main__":
    main()