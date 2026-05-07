"""
variance_decomposition.py — ANOVA-style variance decomposition for continuous
thought vectors.

Given a (task, model) pair, loads the original thoughts and all available
PE-ablation variants, computes per-variant variance decomposition, and
writes a JSON report + text log.

Variance decomposition (per variant):
    var_total    = E_{i,t}[ || h_{i,t} - mu ||^2 ]
    var_timestep = E_t[ || mu_t - mu ||^2 ]          (between-timestep)
    var_instance = E_i[ || mu_i - mu ||^2 ]          (between-instance)
    var_residual = var_total - var_timestep - var_instance

Output files (written to THOUGHTS/<task>/<model>/diagnose/):
    - variance_report.json
    - variance_decomposition.txt

Usage:
    python -m experiments.probe_thoughts.variance_decomposition --task prosqa --model coconut
    python -m experiments.probe_thoughts.variance_decomposition --task prosqa --model coconut_u
    python -m experiments.probe_thoughts.variance_decomposition --task gsm    --model codi
"""

import re
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from src.config import THOUGHTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════

def load_thoughts(path):
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    return data["thoughts"]


def try_load(path):
    """Returns torch.Tensor or None if the file does not exist."""
    if not Path(path).exists():
        return None
    return load_thoughts(path)


# ═══════════════════════════════════════════════════════════════════
# Variance decomposition
# ═══════════════════════════════════════════════════════════════════

def variance_decomposition(thoughts):
    """
    # var_total    = E_{i,t}[ || h_{i,t} - mu ||^2 ]
    # var_timestep = E_t [ || mu_t - mu ||^2 ]     (between-timestep)
    # var_instance = E_i [ || mu_i - mu ||^2 ]     (between-instance)
    # var_residual = var_total - var_timestep - var_instance
    """
    if isinstance(thoughts, np.ndarray):
        thoughts = torch.from_numpy(thoughts).to(DEVICE)
    
    mu = thoughts.mean(dim=(0, 1))
    mu_t = thoughts.mean(dim=0)
    mu_i = thoughts.mean(dim=1)

    var_total = ((thoughts - mu) ** 2).sum(dim=2).mean().item()
    var_timestep = ((mu_t - mu) ** 2).sum(dim=1).mean().item()
    var_instance = ((mu_i - mu) ** 2).sum(dim=1).mean().item()
    var_residual = var_total - var_timestep - var_instance

    def pct(x):
        return 100.0 * x / max(var_total, 1e-12)

    return {
        "var_total": var_total,
        "var_timestep": var_timestep,
        "var_instance": var_instance,
        "var_residual": var_residual,
        "pct_timestep": pct(var_timestep),
        "pct_instance": pct(var_instance),
        "pct_residual": pct(var_residual),
    }


# ═══════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════

def bootstrap_variance_decomposition(thoughts, n_bootstrap=1000, ci=95, seed=0):
    """
    Bootstrap CIs for pct_timestep, pct_instance, pct_residual.
    Resamples over the instance axis (dim 0) with replacement.

    # For each resample b = 1..B:
    #   thoughts_b = thoughts[idx_b, :, :]   where idx_b ~ Uniform({0..N-1})^N
    #   compute variance_decomposition(thoughts_b)
    # CI_alpha = [percentile((100-alpha)/2), percentile((100+alpha)/2)]
    """
    rng = np.random.default_rng(seed)
    N = thoughts.shape[0]

    pct_ts, pct_in, pct_re = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        sample = thoughts[idx]
        v = variance_decomposition(sample)
        pct_ts.append(v["pct_timestep"])
        pct_in.append(v["pct_instance"])
        pct_re.append(v["pct_residual"])

    lo = (100 - ci) / 2
    hi = 100 - lo

    def interval(arr):
        return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

    return {
        "ci_pct": ci,
        "n_bootstrap": n_bootstrap,
        "pct_timestep_ci": interval(pct_ts),
        "pct_instance_ci": interval(pct_in),
        "pct_residual_ci": interval(pct_re),
    }


# ═══════════════════════════════════════════════════════════════════
# Variant discovery & loading
# ═══════════════════════════════════════════════════════════════════

def _discover_random_seeds(pe_dir, mode):
    if not pe_dir.exists():
        return []
    pattern = re.compile(rf"^thoughts_ablated_{re.escape(mode)}_seed(\d+)\.pt$")
    seeds = []
    for p in pe_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def build_variants(task, model_name):
    """Load original + all available PE-ablation variants."""
    base_dir = THOUGHTS / task
    pe_dir = base_dir / model_name / "pe_ablation"

    variants = []

    orig_path = base_dir / f"thoughts_{model_name}.pt"
    orig = try_load(orig_path)
    if orig is None:
        raise FileNotFoundError(
            f"Original thoughts missing at {orig_path}. "
            f"Run extract_thoughts.py first."
        )
    variants.append(("original", orig))

    abl_zero = try_load(pe_dir / "thoughts_ablated_zero.pt")
    if abl_zero is not None:
        variants.append(("ablated_zero", abl_zero))
    else:
        print(f"[WARN] ablated_zero missing; skipping.")

    abl_const = try_load(pe_dir / "thoughts_ablated_constant.pt")
    if abl_const is not None:
        variants.append(("ablated_constant", abl_const))
    else:
        print(f"[WARN] ablated_constant missing; skipping.")

    for s in _discover_random_seeds(pe_dir, "random_gaussian"):
        t = try_load(pe_dir / f"thoughts_ablated_random_gaussian_seed{s}.pt")
        if t is not None:
            variants.append((f"ablated_random_gaussian_seed{s}", t))

    for s in _discover_random_seeds(pe_dir, "random_shuffle"):
        t = try_load(pe_dir / f"thoughts_ablated_random_shuffle_seed{s}.pt")
        if t is not None:
            variants.append((f"ablated_random_shuffle_seed{s}", t))

    return variants


# ═══════════════════════════════════════════════════════════════════
# Pretty labels
# ═══════════════════════════════════════════════════════════════════

_ROW_LABELS = {
    "original":          "Original",
    "ablated_zero":      "PE ablated (zero)",
    "ablated_constant":  "PE ablated (constant)",
}

def pretty_row_label(row_label):
    if row_label in _ROW_LABELS:
        return _ROW_LABELS[row_label]
    m = re.match(r"^ablated_random_gaussian_seed(\d+)$", row_label)
    if m:
        return f"PE rand-gauss (s={m.group(1)})"
    m = re.match(r"^ablated_random_shuffle_seed(\d+)$", row_label)
    if m:
        return f"PE rand-shuf (s={m.group(1)})"
    return row_label


# ═══════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════

def print_report(report, task, model_name):
    line = "=" * 70
    print(f"\n{line}")
    print(f"VARIANCE DECOMPOSITION  —  task={task}  model={model_name}  split={report['split']}")
    print(line)
    print(f"\n  T = {report['T']},  D = {report['D']},  "
          f"N = {report['n_instances']}")

    print(f"\n  Per-variant variance decomposition (95% bootstrap CI):\n")
    header = (f"    {'Variant':<26} {'N':>6} {'timestep%':>22} "
              f"{'instance%':>22} {'residual%':>22} {'total':>12}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in report["rows"]:
        v = row["variance"]
        ci = row["bootstrap_ci"]

        def fmt(pct, interval):
            lo, hi = interval
            return f"{pct:6.2f} [{lo:5.2f},{hi:5.2f}]"

        print(f"    {pretty_row_label(row['row_label']):<26} "
              f"{row['n_instances']:>6} "
              f"{fmt(v['pct_timestep'],  ci['pct_timestep_ci']):>22} "
              f"{fmt(v['pct_instance'],  ci['pct_instance_ci']):>22} "
              f"{fmt(v['pct_residual'],  ci['pct_residual_ci']):>22} "
              f"{v['var_total']:>12.2f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ANOVA-style variance decomposition for thought vectors.",
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CI estimation (default: 1000)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        THOUGHTS / args.task / args.model / "diagnose"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(out_dir / "variance_decomposition.txt")

    print(f"[INFO] task={args.task}  model={args.model}")
    print(f"[INFO] output_dir={out_dir}")

    variants = build_variants(args.task, args.model)
    print(f"[INFO] Variants: {[v[0] for v in variants]}")

    T = variants[0][1].shape[1]
    D = int(variants[0][1].shape[2])
    N = int(variants[0][1].shape[0])

    rows_report = []
    for row_label, thoughts in variants:
        n_instances_row = int(thoughts.shape[0])
        print(f"\n[INFO] Processing: {row_label}  shape={tuple(thoughts.shape)}")
        var = variance_decomposition(thoughts)
        print(f"       timestep={var['pct_timestep']:.2f}%  "
              f"instance={var['pct_instance']:.2f}%  "
              f"residual={var['pct_residual']:.2f}%")
        print(f"       Running bootstrap (n={args.n_bootstrap})...")
        ci = bootstrap_variance_decomposition(thoughts, n_bootstrap=args.n_bootstrap)
        print(f"       timestep_ci={ci['pct_timestep_ci']}  "
              f"instance_ci={ci['pct_instance_ci']}")
        rows_report.append({
            "row_label": row_label,
            "n_instances": n_instances_row,
            "variance": var,
            "bootstrap_ci": ci,
        })

    report = {
        "task": args.task,
        "model": args.model,
        "split": "test",
        "T": T,
        "D": D,
        "n_instances": N,
        "rows": rows_report,
    }
    print_report(report, args.task, args.model)

    report_path = out_dir / "variance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to {report_path}")


if __name__ == "__main__":
    main()