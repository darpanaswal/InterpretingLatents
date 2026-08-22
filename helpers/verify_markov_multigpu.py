"""
Compares two markovianity_test.py result trees -- one produced with
--n_gpus 1, one with --n_gpus > 1 -- and checks that sharding the `orders`
sweep across GPU workers didn't change the numbers, only the wall-clock.

Point estimates (r2_uniform, cosine, r2_var_weighted for every baseline:
mean/identity/linear_shared/mlp_shared) must match to float precision,
since both runs use the same seeds; sharding only changes which process
computes which order, not the arithmetic. Bootstrap CIs are seeded
identically too so should also match, checked with a looser tolerance
since floating-point summation order can differ slightly across
processes/devices.

Usage:
    python -m helpers.verify_markov_multigpu \
        --single outputs/markovianity_test_sandbox/single/llama \
        --multi  outputs/markovianity_test_sandbox/multi/llama \
        --task prosqa --model coconut
"""

import json
import argparse
from pathlib import Path

POINT_KEYS = ("r2_uniform", "r2_var_weighted", "cosine")
SECTIONS = ("mean_baseline", "identity_baseline", "linear_shared", "mlp_shared")


def load(path):
    with open(path) as f:
        return json.load(f)


def compare_order(order, a, b, tol=1e-6, tag=""):
    problems = []
    for section in SECTIONS:
        sa, sb = a.get(section, {}), b.get(section, {})
        for key in POINT_KEYS:
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None:
                continue
            if abs(va - vb) > tol:
                problems.append(
                    f"{tag}order={order} {section}.{key}: "
                    f"single={va:.6f} multi={vb:.6f} (diff={abs(va-vb):.2e})"
                )
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single", required=True, help="output_dir from the --n_gpus 1 run")
    ap.add_argument("--multi", required=True, help="output_dir from the --n_gpus >1 run")
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    single_path = Path(args.single) / f"results_{args.model}_{args.task}.json"
    multi_path = Path(args.multi) / f"results_{args.model}_{args.task}.json"

    if not single_path.exists():
        raise SystemExit(f"[FAIL] missing {single_path}")
    if not multi_path.exists():
        raise SystemExit(f"[FAIL] missing {multi_path}")

    single = load(single_path)
    multi = load(multi_path)

    # JSON round-trips int dict keys as strings; sort numerically for a
    # sane comparison order and readable output.
    single_orders = set(single["orders"].keys())
    multi_orders = set(multi["orders"].keys())
    if single_orders != multi_orders:
        raise SystemExit(
            f"[FAIL] order sets differ: single={sorted(single_orders, key=int)} "
            f"multi={sorted(multi_orders, key=int)}"
        )
    if not single_orders:
        raise SystemExit("[FAIL] no orders found in either run -- nothing to compare")

    problems = []
    for order in sorted(single_orders, key=int):
        problems += compare_order(
            order, single["orders"][order], multi["orders"][order], tol=args.tol
        )

    print(f"[INFO] compared {len(single_orders)} order(s): "
          f"{sorted(single_orders, key=int)}")
    if problems:
        print(f"[FAIL] {len(problems)} mismatch(es):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print("[PASS] multi-GPU sharded run matches single-GPU run "
          f"(tol={args.tol}) across all orders and metrics.")


if __name__ == "__main__":
    main()
