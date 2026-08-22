"""
variance_decomposition.py - ANOVA-style variance decomposition for continuous
thought vectors.

Given a (task, model) pair, loads the original thoughts, computes the 
variance decomposition, and writes a JSON report, compact PCA plot, 
text log, and aggregate TeX tables.

Variance decomposition:
    var_total    = E_{i,t}[ || h_{i,t} - mu ||^2 ]
    var_timestep = E_t[ || mu_t - mu ||^2 ]          (between-timestep)
    var_instance = E_i[ || mu_i - mu ||^2 ]          (between-instance)
    var_residual = var_total - var_timestep - var_instance

Output logs and reports (written to VARIANCE_DECOMPOSITION/<family>/<task>/<model>/):
    - variance_report.json
    - variance_decomposition.txt

Plots (written to Plots/variance_decomposition/pca_<model>/<task>/):
    - <family>.pdf

Aggregate paper files:
    - Tables/main/variance_decomposition_<family>.tex
    - Tables/statistical/variance_decomposition_<family>.tex

Usage:
    python -m experiments.geometry.variance_decomposition --task prosqa --model coconut
    python -m experiments.geometry.variance_decomposition --all
    # Llama:
    python -m experiments.geometry.variance_decomposition --all --model_family llama
    # Both families (reuses existing outputs, writes tables for each):
    python -m experiments.geometry.variance_decomposition --all --model_family both
    # Force recompute:
    python -m experiments.geometry.variance_decomposition --all --model_family both --force
"""

import sys
import json
import os
import torch
import argparse
import numpy as np
from pathlib import Path
from dataclasses import asdict
from safetensors.torch import load_file as load_safetensors

from src.config import BASE_DIR, THOUGHTS, VARIANCE_DECOMPOSITION
from src.bootstrap_stats import bootstrap_variance_decomposition as shared_bootstrap_variance_decomposition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm", "Arithmetic-Reasoning"),
]

MODELS = [
    ("pause", "PaT"),
    ("coconut", "C"),
    ("coconut_u", r"C$_u$"),
    ("codi", "CODI"),
]


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

    def close(self):
        self.log.close()


# ═══════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════

def load_thoughts(path):
    return load_safetensors(str(path), device=str(DEVICE))["thoughts"]

def try_load(path):
    """Returns torch.Tensor or None if the file does not exist."""
    if not Path(path).exists():
        return None
    return load_thoughts(path)

def load_original_thoughts(task, model_name, family="gpt2"):
    """Load only the original thoughts."""
    base_dir = THOUGHTS / family / task
    orig_path = base_dir / f"thoughts_{model_name}.safetensors"
    orig = try_load(orig_path)
    if orig is None:
        raise FileNotFoundError(
            f"Original thoughts missing at {orig_path}. "
            f"Run extract_thoughts.py first (--model_family {family})."
        )
    return orig


# ═══════════════════════════════════════════════════════════════════
# Variance decomposition
# ═══════════════════════════════════════════════════════════════════

def variance_decomposition(thoughts):
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
    if isinstance(thoughts, torch.Tensor):
        thoughts_np = thoughts.detach().cpu().numpy()
    else:
        thoughts_np = np.asarray(thoughts)

    records = shared_bootstrap_variance_decomposition(
        thoughts_np,
        n_boot=n_bootstrap,
        ci=ci,
        seed=seed,
    )

    return {
        "ci_pct": ci,
        "n_bootstrap": n_bootstrap,
        "pct_timestep_ci": (
            records["pct_timestep"].ci_low,
            records["pct_timestep"].ci_high,
        ),
        "pct_instance_ci": (
            records["pct_instance"].ci_low,
            records["pct_instance"].ci_high,
        ),
        "pct_residual_ci": (
            records["pct_residual"].ci_low,
            records["pct_residual"].ci_high,
        ),
        "records": {k: asdict(v) for k, v in records.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════

def print_report(report, task, model_name):
    line = "=" * 70
    print(f"\n{line}")
    print(f"VARIANCE DECOMPOSITION  —  task={task}  model={model_name}  split={report['split']}")
    print(line)
    print(f"\n  T = {report['T']},  D = {report['D']},  N = {report['n_instances']}")

    v = report["variance"]
    ci = report["bootstrap_ci"]

    def fmt(pct, interval):
        lo, hi = interval
        return f"{pct:6.2f} [{lo:5.2f},{hi:5.2f}]"

    print(f"\n  Variance decomposition (95% bootstrap CI):\n")
    header = (f"    {'Component':<16} {'% Variance':>22} {'Abs Total':>12}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    print(f"    {'Timestep':<16} {fmt(v['pct_timestep'], ci['pct_timestep_ci']):>22} {'--':>12}")
    print(f"    {'Instance':<16} {fmt(v['pct_instance'], ci['pct_instance_ci']):>22} {'--':>12}")
    print(f"    {'Residual':<16} {fmt(v['pct_residual'], ci['pct_residual_ci']):>22} {'--':>12}")
    print(f"    {'Total':<16} {'100.00':>22} {v['var_total']:>12.2f}\n")


# ═══════════════════════════════════════════════════════════════════
# Compact PCA plot
# ═══════════════════════════════════════════════════════════════════

def _pca_2d(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[:2].T
    var = (x ** 2).sum(axis=0).sum()
    comp_var = ((z - z.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    explained = 100.0 * comp_var / max(var, 1e-12)
    return z, explained

def _stratified_flatten(thoughts, max_points=6000, seed=0):
    if isinstance(thoughts, torch.Tensor):
        h = thoughts.detach().cpu().numpy()
    else:
        h = np.asarray(thoughts)

    n, t, d = h.shape
    flat = h.reshape(n * t, d)
    labels = np.tile(np.arange(t), n)

    if flat.shape[0] <= max_points:
        return flat, labels

    rng = np.random.default_rng(seed)
    per_t = max(1, max_points // t)
    keep = []
    for step in range(t):
        idx = np.flatnonzero(labels == step)
        take = min(per_t, len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    keep = np.concatenate(keep)
    keep.sort()
    return flat[keep], labels[keep]

def plot_pca(thoughts, out_pdf, max_points=6000, seed=0):
    out_dir = out_pdf.parent
    mpl_config = out_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache = out_dir / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, labels = _stratified_flatten(thoughts, max_points=max_points, seed=seed)
    z, explained = _pca_2d(x)

    fig, ax = plt.subplots(figsize=(3.0, 2.45), dpi=250)
    sc = ax.scatter(
        z[:, 0], z[:, 1],
        c=labels,
        cmap="viridis",
        s=2.6,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=7)
    ax.tick_params(labelsize=6, length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"Timestep $t$", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    ax.margins(x=0.035, y=0.06)
    fig.tight_layout(pad=0.25)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_pdf


# ═══════════════════════════════════════════════════════════════════
# TeX tables
# ═══════════════════════════════════════════════════════════════════

def load_report(task, model, family="gpt2"):
    path = VARIANCE_DECOMPOSITION / family / task / model / "variance_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def collect_reports(family="gpt2"):
    data = {}
    for task, _ in TASKS:
        data[task] = {}
        for model, _ in MODELS:
            data[task][model] = load_report(task, model, family=family)
    return data

def pm_pct(report, key, decimals=1):
    if report is None or "variance" not in report:
        return "--"
    point = report["variance"][key]
    ci_key = key + "_ci"
    lo, hi = report["bootstrap_ci"][ci_key]
    half = (hi - lo) / 2.0
    return rf"${point:.{decimals}f} {{\scriptstyle \pm {half:.{decimals}f}}}$"

def ci_pct(report, key, decimals=1):
    if report is None or "variance" not in report:
        return "--"
    point = report["variance"][key]
    lo, hi = report["bootstrap_ci"][key + "_ci"]
    return f"{point:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"

def build_main_table(data):
    vals = {}
    for task, _ in TASKS:
        for model, _ in MODELS:
            report = data.get(task, {}).get(model)
            vals[f"{task}_{model}_time"] = pm_pct(report, "pct_timestep")
            vals[f"{task}_{model}_inst"] = pm_pct(report, "pct_instance")

    return r"""\begin{table}[!h]
\centering
\tiny
\setlength{\tabcolsep}{3pt}
\caption{Variance decomposition of thought tokens into temporal-- and instance--specific components. Residual variance = 100 - (Var$_\text{time}$ + Var$_\text{inst}$) is the interactive variance. All reported scores are percentages with 95\%% bootstrap CIs.}
\label{tab:variance_decomposition}
\begin{tabular}{l cc | cc}
\toprule
& \multicolumn{2}{c}{\textbf{Graph-Hopping}} & \multicolumn{2}{c}{\textbf{Arithmetic-Reasoning}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Model} & \textbf{Var}$_\textbf{time}$ & \textbf{Var}$_\textbf{inst}$ & \textbf{Var}$_\textbf{time}$ & \textbf{Var}$_\textbf{inst}$ \\
\midrule
PaT & %(prosqa_pause_time)s & %(prosqa_pause_inst)s & %(gsm_pause_time)s & %(gsm_pause_inst)s \\
\midrule
C & %(prosqa_coconut_time)s & %(prosqa_coconut_inst)s & %(gsm_coconut_time)s & %(gsm_coconut_inst)s \\
\midrule
C$_u$ & %(prosqa_coconut_u_time)s & %(prosqa_coconut_u_inst)s & %(gsm_coconut_u_time)s & %(gsm_coconut_u_inst)s \\
\midrule
CODI & %(prosqa_codi_time)s & %(prosqa_codi_inst)s & %(gsm_codi_time)s & %(gsm_codi_inst)s \\
\bottomrule
\end{tabular}
\end{table}
""" % vals

def build_appendix_table(data):
    lines = [
        r"\subsection{Variance Decomposition}",
        r"\label{app:variance_decomposition_stats}",
        "",
        r"Point estimates and 95\% percentile bootstrap CIs for the ANOVA-style decomposition of thought-token variance. "
        r"Rows use the original, unablated thoughts. "
        r"Var$_\text{resid}$ is the interaction/residual component.",
        "",
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Variance-decomposition statistical summary across tasks.}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        (
            r"Model & Var$_\text{time}$ [\% CI] "
            r"& Var$_\text{inst}$ [\% CI] "
            r"& Var$_\text{resid}$ [\% CI] "
            r"& $n$ \\"
        ),
    ]

    for task, task_label in TASKS:
        
        # Add a midrule and a centered sub-header for the task
        lines += [
            r"\midrule",
            rf"\multicolumn{{5}}{{c}}{{\textbf{{{task_label}}}}} \\",
            r"\midrule",
        ]
        
        for model, label in MODELS:
            report = data.get(task, {}).get(model)
            n = report["n_instances"] if report is not None else "--"
            lines.append(
                f"{label} & "
                f"{ci_pct(report, 'pct_timestep')} & "
                f"{ci_pct(report, 'pct_instance')} & "
                f"{ci_pct(report, 'pct_residual')} & "
                f"{n} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:variance_decomposition_stats_combined}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines).strip()

def write_tex_tables(out_main, out_appendix, family="gpt2"):
    data = collect_reports(family=family)
    out_main = Path(out_main)
    out_appendix = Path(out_appendix)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    out_appendix.parent.mkdir(parents=True, exist_ok=True)
    out_main.write_text(build_main_table(data))
    out_appendix.write_text(build_appendix_table(data))
    return out_main, out_appendix


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_one(task, model, family, args):
    out_dir = Path(args.output_dir) if args.output_dir else (
        VARIANCE_DECOMPOSITION / family / task / model
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "variance_report.json"

    needs_compute = args.force or not report_path.exists()
    
    # If we don't need to compute AND plotting is disabled, exit entirely.
    if not needs_compute and not args.plot_pca:
        print(f"[SKIP] task={task} model={model} family={family} -> reusing {report_path}")
        return report_path

    # Load thoughts once, as both plotting and computing require them.
    thoughts = load_original_thoughts(task, model, family=family)

    # 1. Always plot if requested, regardless of computation status
    if args.plot_pca:
        pdf_path = BASE_DIR / "Plots" / "variance_decomposition" / f"pca_{model}_{task}_{family}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = plot_pca(
            thoughts,
            pdf_path,
            max_points=args.max_pca_points,
            seed=args.seed,
        )
        print(f"[INFO] PCA PDF saved to {pdf}")

    # 2. Skip the heavy bootstrapping/variance math if the report exists
    if not needs_compute:
        print(f"[SKIP] Computation task={task} model={model} family={family} -> reusing {report_path}")
        return report_path

    # 3. Otherwise, run the full computation
    previous_stdout = sys.stdout
    logger = Logger(out_dir / "variance_decomposition.txt")
    sys.stdout = logger

    try:
        print(f"[INFO] task={task}  model={model}  family={family}")
        print(f"[INFO] output_dir={out_dir}")

        T = thoughts.shape[1]
        D = int(thoughts.shape[2])
        N = int(thoughts.shape[0])

        print(f"\n[INFO] Processing original thoughts  shape={tuple(thoughts.shape)}")
        var = variance_decomposition(thoughts)
        
        print(f"       timestep={var['pct_timestep']:.2f}%  "
              f"instance={var['pct_instance']:.2f}%  "
              f"residual={var['pct_residual']:.2f}%")
              
        print(f"       Running bootstrap (n={args.n_bootstrap})...")
        ci = bootstrap_variance_decomposition(
            thoughts,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(f"       timestep_ci={ci['pct_timestep_ci']}  "
              f"instance_ci={ci['pct_instance_ci']}")

        report = {
            "task": task,
            "model": model,
            "split": "test",
            "T": T,
            "D": D,
            "n_instances": N,
            "variance": var,
            "bootstrap_ci": ci,
        }
        
        print_report(report, task, model)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Report saved to {report_path}")

        return report_path
    finally:
        logger.close()
        sys.stdout = previous_stdout


def main():
    parser = argparse.ArgumentParser(
        description="ANOVA-style variance decomposition for thought vectors.",
    )
    parser.add_argument("--task", choices=[t for t, _ in TASKS])
    parser.add_argument("--model", choices=[m for m, _ in MODELS])
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama", "both"],
        default="gpt2",
        help="Base model family. Determines the thoughts load path "
             "(THOUGHTS/<family>/...) and namespaces all outputs. "
             "'both' runs gpt2 and llama and writes tables for each.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if variance_report.json exists "
                             "(default: reuse existing outputs).")
    parser.add_argument("--all", action="store_true",
                        help="Run all task/model combinations.")
    parser.add_argument("--output_dir", default=None,
                        help="Only valid for a single --task/--model run.")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CI estimation (default: 1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_pca", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Write a compact PCA plot for original thoughts.")
    parser.add_argument("--max_pca_points", type=int, default=6000)
    parser.add_argument("--write_tables", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Write aggregate main/appendix TeX tables.")
    parser.add_argument("--out_main", default=None,
                        help="Override main TeX path. Default: "
                             "Tables/main/variance_decomposition_<family>.tex")
    parser.add_argument("--out_appendix", default=None,
                        help="Override appendix TeX path. Default: "
                             "Tables/statistical/variance_decomposition_<family>.tex")
    args = parser.parse_args()

    families = ["gpt2", "llama"] if args.model_family == "both" else [args.model_family]

    if len(families) > 1 and (args.out_main or args.out_appendix):
        parser.error("--out_main/--out_appendix cannot be combined with "
                     "--model_family both (paths would collide)")
    if len(families) > 1 and args.output_dir:
        parser.error("--output_dir is only supported for a single family run")

    if args.all:
        if args.output_dir:
            raise ValueError("--output_dir is only supported for a single --task/--model run")
        for family in families:
            for task, _ in TASKS:
                for model, _ in MODELS:
                    run_one(task, model, family, args)
    else:
        if not args.task or not args.model:
            parser.error("provide --task and --model, or use --all")
        for family in families:
            run_one(args.task, args.model, family, args)

    if args.write_tables:
        for family in families:
            out_main = args.out_main or str(
                BASE_DIR / "Tables" / "main" / f"variance_decomposition_{family}.tex"
            )
            out_appendix = args.out_appendix or str(
                BASE_DIR / "Tables" / "statistical" / f"variance_decomposition_{family}.tex"
            )
            out_main, out_appendix = write_tex_tables(
                out_main, out_appendix, family=family)
            print(f"[OK] {family} main table -> {out_main.resolve()}")
            print(f"[OK] {family} appendix   -> {out_appendix.resolve()}")


if __name__ == "__main__":
    main()