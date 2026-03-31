"""
Quick check: are extracted thought vectors actually different across timesteps?

Prints norms, first 5 values, and pairwise cosine similarities for a few
instances across all timesteps. Also checks whether vectors at the same
timestep are identical across instances (which would indicate a bug).

Usage:
    python verify_thoughts.py --thoughts_path outputs/vqvae/coconut_u/thoughts_coconut_u.pt
    python verify_thoughts.py --thoughts_path outputs/vqvae/coconut/thoughts_coconut.pt
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thoughts_path", type=str, required=True)
    parser.add_argument("--n_instances", type=int, default=3)
    args = parser.parse_args()

    data = torch.load(args.thoughts_path, map_location="cpu", weights_only=False)
    thoughts = data["thoughts"]  # (N, T, D)
    N, T, D = thoughts.shape
    print(f"Shape: {N} instances x {T} timesteps x {D} dims\n")

    n_show = min(args.n_instances, N)

    for i in range(n_show):
        print(f"{'='*70}")
        print(f"Instance {i}")
        print(f"{'='*70}")

        for t in range(T):
            h = thoughts[i, t]
            norm = h.norm().item()
            first5 = h[:5].tolist()
            print(f"  t={t}: norm={norm:>10.4f}  first5={[f'{v:.4f}' for v in first5]}")

        # Pairwise cosine similarity across timesteps for this instance
        print(f"\n  Pairwise cosine similarity (instance {i}):")
        header = "       " + "  ".join([f"  t={t}" for t in range(T)])
        print(header)
        for s in range(T):
            row = f"  t={s}  "
            for t in range(T):
                cos = (thoughts[i, s] @ thoughts[i, t]) / (
                    thoughts[i, s].norm() * thoughts[i, t].norm() + 1e-8
                )
                row += f" {cos.item():>5.3f} "
            print(row)
        print()

    # Cross-instance check: are vectors at the same timestep identical?
    print(f"{'='*70}")
    print(f"Cross-instance check: are vectors at the same timestep identical?")
    print(f"{'='*70}")
    for t in range(T):
        ref = thoughts[0, t]
        max_diff = 0.0
        for i in range(1, min(N, 50)):
            diff = (thoughts[i, t] - ref).abs().max().item()
            max_diff = max(max_diff, diff)
        print(f"  t={t}: max abs diff across first 50 instances = {max_diff:.6f}  "
              f"({'IDENTICAL — BUG!' if max_diff == 0 else 'DISTINCT — OK'})")


if __name__ == "__main__":
    main()