"""
Principal angles between the gold-label subspace B_t^{gold} and the
predicted-token subspace B_t^{pred}, per timestep.

Motivation
----------
B_t is built from gradients of the supervised loss, so the gold-label
basis mechanically encodes the answer. The predicted-token control
rebuilds the subspace from gradients of the model's own greedy decode,
carrying no gold information. If the two bases span (nearly) the same
subspace at each t, the label never did the selecting: the gradient
structure — not the specific supervising target — fixes the directions.

This is the geometric companion to the interventional finding that the
two bases yield the same flip-rate results. Small principal angles here
+ matched interventions there = "gradient structure of almost any kind
is enough, and the specific directions are interchangeable."

What is reported (per timestep t)
---------------------------------
    principal_angles_deg : theta_i = arccos(sigma_i(B_gold^T B_pred)),
                           ascending, one per shared dimension
                           min(k_gold, k_pred).
    mean_cos_sq          : mean(cos^2 theta_i) in [0, 1].
                           1 = same subspace, 0 = orthogonal.
    max_angle_deg        : largest principal angle (the worst-aligned
                           direction; the honest single number).
    grassmann_dist       : sqrt(sum theta_i^2), the geodesic distance
                           on the Grassmannian (0 = identical subspace).

A random rank-matched control angle is also reported per t: two Haar-
random orthonormal bases of the same ranks, to calibrate what "diffuse /
unaligned" looks like at this D and k.

Math
----
For orthonormal B1 in R^{D x k1}, B2 in R^{D x k2}:
    # M            = B1^T B2                         (k1, k2)
    # cos(theta_i) = singular values of M, clipped to [0, 1]
    # theta_i      = arccos(cos(theta_i))            radians, ascending
Number of angles = min(k1, k2). Ordering: numpy SVD returns singular
values descending, so cos descending => theta ascending after we flip.

Output
------
    <out_dir> / gold_predtoken_angles.json      per-t summary
    <out_dir> / gold_predtoken_angles.jsonl     one record per t (tables)

Usage
-----
    python -u -m experiments.geometry.gold_predtoken_angles \
        --task prosqa --model pause
    # Llama:
    python -u -m experiments.geometry.gold_predtoken_angles \
        --task gsm --model coconut --model_family llama
    # Explicit overrides:
    python -u -m experiments.geometry.gold_predtoken_angles \
        --task prosqa --model pause \
        --gold_bases  .../gradient_geometry/gpt2/prosqa/pause/bases.npz \
        --pred_bases  .../gradient_geometry_predtoken/gpt2/prosqa/pause/bases.npz
"""

import json
import argparse
import numpy as np
from pathlib import Path
from src.config import BASE_DIR, set_seed


# ═══════════════════════════════════════════════════════════════════
# Loading  (matches gradient_subspace.py's savez layout: keys "B_t{t}")
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
# Principal angles between two subspaces
# ═══════════════════════════════════════════════════════════════════

def principal_angles(B1, B2):
    """
    Principal angles (radians, ascending) between the column spans of
    two orthonormal bases B1 (D, k1) and B2 (D, k2).
    Returns a length-min(k1, k2) array. Empty if either is rank 0.
    """
    k1, k2 = B1.shape[1], B2.shape[1]
    if k1 == 0 or k2 == 0:
        return np.zeros(0)
    M = B1.T @ B2                                   # (k1, k2)
    cos_sv = np.linalg.svd(M, compute_uv=False)     # descending
    cos_sv = np.clip(cos_sv, 0.0, 1.0)
    theta = np.arccos(cos_sv)                        # ascending in theta
    return theta                                     # radians


def angle_summary(B1, B2):
    """Scalar summaries of the subspace alignment between B1 and B2."""
    theta = principal_angles(B1, B2)
    if theta.size == 0:
        return {
            "n_angles": 0,
            "principal_angles_deg": [],
            "mean_cos_sq": 0.0,
            "max_angle_deg": float("nan"),
            "grassmann_dist": float("nan"),
        }
    cos_sq = np.cos(theta) ** 2
    return {
        "n_angles": int(theta.size),
        "principal_angles_deg": np.degrees(theta).tolist(),
        "mean_cos_sq": float(np.mean(cos_sq)),
        "max_angle_deg": float(np.degrees(theta.max())),
        "grassmann_dist": float(np.sqrt(np.sum(theta ** 2))),
    }


# ═══════════════════════════════════════════════════════════════════
# Rank-matched random control
# ═══════════════════════════════════════════════════════════════════
# Two Haar-random orthonormal bases of the given ranks in R^D, averaged
# over a few draws, to show what unaligned subspaces score at this
# (D, k1, k2). QR of a Gaussian gives a Haar-random orthonormal frame.

def random_control_summary(D, k1, k2, rng, n_draws=20):
    if k1 == 0 or k2 == 0:
        return {"mean_cos_sq": 0.0, "max_angle_deg": float("nan")}
    mcs, maxang = [], []
    for _ in range(n_draws):
        Q1, _ = np.linalg.qr(rng.standard_normal((D, k1)))
        Q2, _ = np.linalg.qr(rng.standard_normal((D, k2)))
        s = angle_summary(Q1[:, :k1], Q2[:, :k2])
        mcs.append(s["mean_cos_sq"])
        maxang.append(s["max_angle_deg"])
    return {
        "mean_cos_sq": float(np.mean(mcs)),
        "max_angle_deg": float(np.mean(maxang)),
    }


# ═══════════════════════════════════════════════════════════════════
# Default path resolution
# ═══════════════════════════════════════════════════════════════════

def default_bases_path(tree, family, task, model):
    return BASE_DIR / "outputs" / tree / family / task / model / "bases.npz"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Per-timestep principal angles between the gold-label "
                    "and predicted-token gradient subspaces."
    )
    p.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    p.add_argument(
        "--model",
        choices=["coconut", "coconut_u", "pause", "codi"],
        required=True,
    )
    p.add_argument("--model_family", choices=["gpt2", "llama"], default="gpt2")
    p.add_argument("--gold_bases", type=str, default=None,
                   help="Override path to gold bases.npz. Default: "
                        "outputs/gradient_geometry/<family>/<task>/<model>/bases.npz")
    p.add_argument("--pred_bases", type=str, default=None,
                   help="Override path to predtoken bases.npz. Default: "
                        "outputs/gradient_geometry_predtoken/<family>/<task>/<model>/bases.npz")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory. Default: "
                        "outputs/gradient_geometry_predtoken/<family>/<task>/<model>")
    p.add_argument("--n_control_draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    gold_path = (Path(args.gold_bases) if args.gold_bases
                 else default_bases_path("gradient_geometry",
                                         args.model_family, args.task, args.model))
    pred_path = (Path(args.pred_bases) if args.pred_bases
                 else default_bases_path("gradient_geometry_predtoken",
                                         args.model_family, args.task, args.model))
    out_dir = (Path(args.out_dir) if args.out_dir
               else default_bases_path("gradient_geometry_predtoken",
                                       args.model_family, args.task, args.model).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, pth in [("gold", gold_path), ("pred", pred_path)]:
        if not pth.exists():
            raise FileNotFoundError(
                f"{label} bases.npz not found at {pth}. "
                f"Run gradient_subspace{'_predtoken' if label == 'pred' else ''}.py first."
            )

    print(f"[main] task={args.task}  model={args.model}  family={args.model_family}")
    print(f"[main] gold bases: {gold_path}")
    print(f"[main] pred bases: {pred_path}")
    print(f"[main] out_dir   : {out_dir}")

    gold = load_bases(gold_path)
    pred = load_bases(pred_path)

    T = min(max(gold.keys()), max(pred.keys())) + 1
    if set(range(T)) - set(gold.keys()) or set(range(T)) - set(pred.keys()):
        print(f"[warn] using T={T}; timesteps present "
              f"gold={sorted(gold.keys())} pred={sorted(pred.keys())}")

    D = gold[0].shape[0]
    if pred[0].shape[0] != D:
        raise ValueError(
            f"Dimension mismatch: gold D={D}, pred D={pred[0].shape[0]}. "
            f"Are these the same model family?"
        )

    # ── Per-timestep angles ───────────────────────────────────────
    records = []
    for t in range(T):
        summ = angle_summary(gold[t], pred[t])
        ctrl = random_control_summary(
            D, gold[t].shape[1], pred[t].shape[1], rng,
            n_draws=args.n_control_draws,
        )
        rec = {
            "task": args.task,
            "model": args.model,
            "model_family": args.model_family,
            "t": t,
            "k_gold": int(gold[t].shape[1]),
            "k_pred": int(pred[t].shape[1]),
            **summ,
            "control_mean_cos_sq": ctrl["mean_cos_sq"],
            "control_max_angle_deg": ctrl["max_angle_deg"],
        }
        records.append(rec)

    # ── Print ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"  GOLD vs PREDICTED-TOKEN SUBSPACE ANGLES  ({args.task}/{args.model})")
    print("=" * 78)
    print(f"  {'t':>2}  {'k_g':>3} {'k_p':>3}  {'mean cos^2':>10}  "
          f"{'max angle':>9}  {'Grassmann':>9}  {'ctrl cos^2':>10}")
    for r in records:
        print(f"  {r['t']:>2}  {r['k_gold']:>3} {r['k_pred']:>3}  "
              f"{r['mean_cos_sq']:>10.4f}  "
              f"{r['max_angle_deg']:>8.2f}°  "
              f"{r['grassmann_dist']:>9.4f}  "
              f"{r['control_mean_cos_sq']:>10.4f}")
    mcs_all = np.mean([r["mean_cos_sq"] for r in records])
    ctrl_all = np.mean([r["control_mean_cos_sq"] for r in records])
    print("-" * 78)
    print(f"  mean cos^2 over t: {mcs_all:.4f}   "
          f"(rank-matched random control: {ctrl_all:.4f})")
    print(f"  1 = same subspace (label did not select), 0 = orthogonal")

    # ── Save ───────────────────────────────────────────────────────
    jsonl_path = out_dir / "gold_predtoken_angles.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[main] Per-t records -> {jsonl_path}")

    summary = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "T": int(T),
        "D": int(D),
        "mean_cos_sq_over_t": float(mcs_all),
        "control_mean_cos_sq_over_t": float(ctrl_all),
        "per_t": records,
    }
    json_path = out_dir / "gold_predtoken_angles.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] Summary      -> {json_path}")


if __name__ == "__main__":
    main()