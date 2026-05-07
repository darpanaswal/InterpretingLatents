"""
Standalone plotting for gradient_subspace_diagnosis.py + bases.npz.

Reads
    OUT_DIR / <task> / <model> / diagnosis.json   (from diagnosis script)
    OUT_DIR / <task> / <model> / bases.npz        (from gradient_subspace.py)

Writes
    OUT_DIR / summary.csv
    OUT_DIR / summary.tex
    OUT_DIR / figures/
        cos_panels.png         -- subspace stability cos^2(B_t, B_{t+1})

The Q1 / Q2 alignment-of-variance-component panels were removed: they
conflate magnitude, rank, and per-sample vs population alignment, and
therefore cannot be used to argue causal relevance.

All input and output paths are hardcoded; no CLI flags.
"""

import csv
import json
import numpy as np
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
from src.config import BASE_DIR


# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "gradient_geometry"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_TEX = OUT_DIR / "summary.tex"

MODEL_LABELS = {
    "coconut":   "C",
    "coconut_u": r"C$_u$",
    "pause":     "PaT",
    "codi":      "CODI",
}

MODEL_ORDER = ["pause", "coconut", "coconut_u", "codi"]
TASK_ORDER = ["prosqa", "gsm"]

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}

MODEL_COLOR = {
    "pause":     "tab:green",
    "coconut":   "tab:blue",
    "coconut_u": "tab:orange",
    "codi":      "tab:red",
}


def sort_model_key(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return 99


def sort_task_key(task):
    try:
        return TASK_ORDER.index(task)
    except ValueError:
        return 99


# ═══════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════

def discover_results():
    """
    Walk OUT_DIR / <task> / <model> / and load both diagnosis.json and
    bases.npz when both are present. Returns a list of dicts with keys
        task, model, diagnosis (dict), bases (dict {t: ndarray (D, k_t)})
    """
    results = []
    for diag_path in sorted(OUT_DIR.glob("*/*/diagnosis.json")):
        with open(diag_path, "r") as f:
            r = json.load(f)
        bases_path = diag_path.parent / "bases.npz"
        bases = None
        if bases_path.exists():
            blob = np.load(bases_path)
            bases = {}
            for key in blob.files:
                if not key.startswith("B_t"):
                    continue
                t = int(key[len("B_t"):])
                bases[t] = blob[key]
        results.append({
            "task":      r["task"],
            "model":     r["model"],
            "diagnosis": r,
            "bases":     bases,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Subspace-stability cos^2 panels
# ═══════════════════════════════════════════════════════════════════

def plot_cos_panels(results, out_path):
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in results)]
    if not tasks:
        return

    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(4.0 * len(tasks), 2.8),
        sharex=False, sharey=True, squeeze=False,
    )

    by_task = {t: [r for r in results if r["task"] == t] for t in tasks}

    for j, task in enumerate(tasks):
        task_results = sorted(by_task[task], key=lambda r: sort_model_key(r["model"]))
        ax = axes[0][j]

        for r in task_results:
            ys = r["diagnosis"]["q3_adjacent_per_t"]
            xs = np.arange(len(ys)) + 0.5
            ax.plot(xs, ys,
                    color=MODEL_COLOR.get(r["model"], "black"),
                    marker="o", linewidth=1.6,
                    label=MODEL_LABELS.get(r["model"], r["model"]))

        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
        ax.grid(alpha=0.25)

        ax.set_title(f"({chr(97 + j)}) {TASK_LABELS.get(task, task)}",
                     fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("timestep $t$")
        if j == 0:
            ax.set_ylabel(r"mean $\cos^2(B_t, B_{t+1})$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.91), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Summary table (Q3 only)
# ═══════════════════════════════════════════════════════════════════

def _fmt_signed(x, decimals=3):
    s = f"{x:+.{decimals}f}"
    return s.replace("+", r"\phantom{-}")


def write_summary(results, out_csv, out_tex):
    rows = []
    for r in sorted(results, key=lambda r: (sort_task_key(r["task"]),
                                            sort_model_key(r["model"]))):
        d = r["diagnosis"]
        rows.append({
            "task":            TASK_LABELS.get(r["task"], r["task"]),
            "model":           r["model"],
            "T":               d["T"],
            "mean_k":          float(np.mean(d["subspace_ranks"])),
            "Q3_adj_mean":     float(np.mean(d["q3_adjacent_per_t"]))
                               if d["q3_adjacent_per_t"] else float("nan"),
            "Q3_offdiag_mean": d["q3_offdiag_mean"],
        })
    if not rows:
        return

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv ] {out_csv}")

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Gradient-subspace stability summary. $\bar k$ is "
        r"the mean per-timestep subspace rank. We report mean $\cos^2$ "
        r"between adjacent subspaces $(B_t, B_{t+1})$ and over all "
        r"unordered pairs $(t, t')$ with $t \neq t'$. "
        r"$1$ = same subspace, $0$ = orthogonal.}",
        r"\label{tab:gradient_geometry}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"task & model & $T$ & $\bar k$ & Q3 adj. & Q3 off-diag \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [
            r["task"],
            r["model"].replace("_", r"\_"),
            f"{r['T']}",
            f"{r['mean_k']:.1f}",
            _fmt_signed(r["Q3_adj_mean"]),
            _fmt_signed(r["Q3_offdiag_mean"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex ] {out_tex}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    if not OUT_DIR.exists():
        raise SystemExit(f"OUT_DIR does not exist: {OUT_DIR}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = discover_results()
    if not results:
        raise SystemExit(f"No diagnosis.json found under {OUT_DIR}/*/*/.")

    plot_cos_panels(results, FIG_DIR / "cos_panels.png")
    write_summary(results, SUMMARY_CSV, SUMMARY_TEX)


if __name__ == "__main__":
    main()