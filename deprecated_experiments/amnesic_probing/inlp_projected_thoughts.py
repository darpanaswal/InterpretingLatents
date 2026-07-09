"""
Apply INLP and random nullspace projections to extracted thought vectors
and save the projected thoughts for downstream geometric analysis.

Pure CPU/numpy computation. No model loading, no inference. Reads:
    - thoughts_{model}.pt  (from extract_thoughts.py)
    - inlp_results.pt      (from inlp.py)
And writes:
    - thoughts_{model}_inlp.pt  — concept-ablated thoughts
    - thoughts_{model}_rand.pt  — random-control-ablated thoughts

These files are consumed by diagnose_thoughts.py, which will iterate over
all three variants (original, inlp, rand) automatically.

Usage:
    python -m experiments.amnesic_probing.projected_thoughts --task prosqa --model coconut
    python -m experiments.amnesic_probing.projected_thoughts --task prosqa --model coconut_u
    python -m experiments.amnesic_probing.projected_thoughts --task prosqa --model pause

    python -m experiments.amnesic_probing.projected_thoughts --task gsm --model pause
    python -m experiments.amnesic_probing.projected_thoughts --task gsm --model coconut
    python -m experiments.amnesic_probing.projected_thoughts --task gsm --model coconut_u
    python -m experiments.amnesic_probing.projected_thoughts --task gsm --model codi
"""

import torch
import argparse
from src.config import THOUGHTS, BASE_DIR
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )


def apply_projection(thoughts, proj_dict):
    """
    Apply per-timestep nullspace projection to thought vectors.

    For each timestep t:
        # h'_{n,t} = P_t @ h_{n,t}
        # where P_t is the D x D nullspace projection matrix at timestep t.

    Timesteps without a projection entry are passed through unchanged.
    """
    N, T, D = thoughts.shape
    projected = torch.zeros_like(thoughts)
    for t in range(T):
        P = proj_dict.get(t)
        if P is not None:
            # h'_{n,t} = P_t @ h_{n,t}  (applied as right-mult with P^T for batched rows)
            P_tensor = torch.tensor(P, dtype=torch.float32)
            projected[:, t, :] = thoughts[:, t, :] @ P_tensor.T
        else:
            projected[:, t, :] = thoughts[:, t, :]
    return projected


def main():
    parser = argparse.ArgumentParser(
        description="Save INLP- and rand-projected thought vectors."
    )
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    args = parser.parse_args()

    # ── Load original thoughts ─────────────────────────────────────
    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        raise FileNotFoundError(
            f"Original thoughts not found at {thoughts_path}. "
            f"Run extract_thoughts.py first."
        )
    print(f"[INFO] Loading thoughts from {thoughts_path}")
    thought_data = torch.load(thoughts_path, map_location="cpu", weights_only=False)
    thoughts = thought_data["thoughts"]
    N, T, D = thoughts.shape
    print(f"[INFO] Thoughts shape: {thoughts.shape}")

    # ── Load INLP projections ──────────────────────────────────────
    inlp_path = BASE_DIR / f"outputs/inlp/{args.task}/{args.model}/inlp_results.pt"
    if not inlp_path.exists():
        raise FileNotFoundError(f"Please run inlp.py to extract inlp_results for {args.model}--{args.task}")
    
    print(f"[INFO] Loading INLP from {inlp_path}")
    inlp_data = torch.load(inlp_path, map_location="cpu", weights_only=False)
    projections = {int(k) if isinstance(k, str) else k: v
                   for k, v in inlp_data["projections"].items()}
    rand_projections = {int(k) if isinstance(k, str) else k: v
                        for k, v in inlp_data["rand_projections"].items()}

    # ── Apply and save each variant ────────────────────────────────
    thoughts_dir = THOUGHTS / args.task

    for suffix, proj_dict in [("inlp", projections), ("rand", rand_projections)]:
        projected = apply_projection(thoughts, proj_dict)
        save_dict = {
            "thoughts": projected,
            "instance_indices": list(range(N)),
            "n_thoughts": T - 1,
            "model": args.model,
            "projection_type": suffix,
        }
        save_path = thoughts_dir / f"thoughts_{args.model}_{suffix}.pt"
        torch.save(save_dict, save_path)
        print(f"[INFO] Saved {suffix}-projected thoughts to {save_path}")


if __name__ == "__main__":
    main()