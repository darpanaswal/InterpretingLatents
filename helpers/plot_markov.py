"""
Standalone plotting for markovianity_test.py results.

Reads ``out_dir/results_{model}_{task}.json`` (the flat layout produced
by ``markovianity_test.py``) and writes:

  out_dir/
      summary.csv
      summary.tex
      figures/
          order_curves_r2_uniform.png        -- main paper figure
          order_curves_r2_var_weighted.png  -- appendix
          order_curves_cosine.png            -- appendix
          regime_scatter.png                 -- bigram identity vs linear-gain
          per_step_heatmap_{task}_{model}.png   -- one per (task, model)

No fitting, no extraction -- pure replotter. Run after markovianity_test.py.

All input and output paths are hardcoded; no CLI flags.
"""

import csv
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.config import BASE_DIR
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths (must match markovianity_test.py default out_dir)
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "markovianity"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_TEX = OUT_DIR / "summary.tex"

SUMMARY_ORDER = 1

MODEL_LABELS = {
    "coconut":   "C",
    "coconut_u": r"C$_u$",
    "pause":     "PaT",
    "codi":      "CODI",
}

# Required rendering orders
MODEL_ORDER = ["pause", "coconut", "coconut_u", "codi"]
TASK_ORDER = ["prosqa", "gsm"]

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}


def fig_label(task, model):
    """Format used in figure titles, annotations: model (task)."""
    return f"{MODEL_LABELS.get(model, model)} ({TASK_LABELS.get(task, task)})"


def sort_model_key(model_name):
    """Helper to enforce strict model ordering."""
    try:
        return MODEL_ORDER.index(model_name)
    except ValueError:
        return 99


def sort_task_key(task_name):
    """Helper to enforce strict task ordering."""
    try:
        return TASK_ORDER.index(task_name)
    except ValueError:
        return 99


def discover_results():
    results = []
    for path in sorted(OUT_DIR.glob("results_*.json")):
        with open(path, "r") as f:
            r = json.load(f)
        r["orders"] = {int(k): v for k, v in r["orders"].items()}
        for o in r["orders"].values():
            o["linear_per_step"] = {int(k): v for k, v in o["linear_per_step"].items()}
        results.append(r)
    return results


def _plot_order_curves_metric(results, out_path, metric_key, ylabel, ylim):
    """
    Grid of (task x model) panels showing `metric_key` vs Markov order
    for the four predictors (mean, identity, linear-shared, MLP-shared).
    """
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in results)]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in results)]

    fig, axes = plt.subplots(len(tasks), len(models),
                             figsize=(3.5 * len(models), 2.8 * len(tasks)),
                             sharex=True, sharey=True, squeeze=False)
    series = [
        ("mean_baseline", "mean", "tab:gray", ":"),
        ("identity_baseline", "identity", "tab:red", "--"),
        ("linear_shared", "linear (shared)", "tab:blue", "-"),
        ("mlp_shared", "MLP (shared)", "tab:green", "-"),
    ]
    for r in results:
        i = tasks.index(r["task"]); j = models.index(r["model"])
        ax = axes[i][j]
        orders = sorted(r["orders"].keys())
        for key, label, color, ls in series:
            ys = [r["orders"][o][key][metric_key] for o in orders]
            ax.plot(orders, ys, color=color, linestyle=ls,
                    marker="o", label=label, linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)

        ax.set_title(MODEL_LABELS.get(r["model"], r["model"]), fontsize=10, pad=8)
        ax.set_xticks(orders)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Markov order")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=0.5)

    fig.canvas.draw()
    for i, task in enumerate(tasks):
        bbox_left = axes[i][0].get_position()
        bbox_right = axes[i][-1].get_position()
        x_center = (bbox_left.x0 + bbox_right.x1) / 2.0
        y_top = bbox_left.y1 + 0.03
        letter = chr(97 + i)
        fig.text(x_center, y_top, f"({letter}) {TASK_LABELS.get(task, task)}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# (metric_key, file suffix, ylabel, ylim)
ORDER_CURVE_METRICS = [
    ("r2_uniform",      "r2_uniform",      r"$R^2$ (uniform avg)",      (-1.1, 1.1)),
    ("r2_var_weighted", "r2_var_weighted", r"$R^2$ (variance-weighted)", (-1.1, 1.1)),
    ("cosine",          "cosine",          r"cosine similarity",         (-1.1, 1.1)),
]


def plot_order_curves(results, fig_dir, suffix=""):
    for metric_key, met_suffix, ylabel, ylim in ORDER_CURVE_METRICS:
        _plot_order_curves_metric(
            results, fig_dir / f"order_curves_{met_suffix}{suffix}.png",
            metric_key=metric_key, ylabel=ylabel, ylim=ylim,
        )


def plot_regime_scatter(results, out_path, order=SUMMARY_ORDER):
    """
    Scatter of identity-baseline R^2 vs linear-gain-over-identity at the
    given Markov order. Each point is one (task, model) pair: color
    encodes model, marker shape encodes task. Labels are lifted out
    of the plot into a legend on the right.
    """
    task_marker = {"prosqa": "o", "gsm": "s"}
    model_color = {"coconut": "tab:blue", "coconut_u": "tab:orange",
                   "pause": "tab:green", "codi": "tab:red"}

    fig, ax = plt.subplots(figsize=(8.0, 3.5))

    legend_handles = []
    # Iterate in canonical (task, model) order so the legend reads cleanly.
    sorted_results = sorted(
        results,
        key=lambda r: (sort_task_key(r["task"]), sort_model_key(r["model"])),
    )
    for r in sorted_results:
        if order not in r["orders"]:
            continue
        # x = identity-baseline R^2;  y = R^2(linear) - R^2(identity)
        x = r["orders"][order]["identity_baseline"]["r2_uniform"]
        y = r["orders"][order]["linear_shared"]["r2_uniform"] - x

        marker = task_marker.get(r["task"], "x")
        color = model_color.get(r["model"], "black")
        ax.scatter(x, y, marker=marker, color=color,
                   s=110, edgecolor="black", linewidth=0.7)

        # Proxy handle for the external legend: same marker + color.
        legend_handles.append(plt.Line2D(
            [0], [0], marker=marker, color="white",
            markerfacecolor=color, markeredgecolor="black",
            markeredgewidth=0.7, markersize=10,
            label=fig_label(r["task"], r["model"]),
        ))

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"Identity baseline $R^2$ (test)")
    ax.set_ylabel(r"Linear gain over identity (test)")
    ax.grid(alpha=0.25)

    # Anchor the legend to the right edge of the *axes* (not the figure),
    # so the gap between plot area and legend is fixed regardless of
    # figure size. bbox_inches="tight" then trims any margin past it.
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=9, ncol=2,
        handletextpad=0.4, columnspacing=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_per_step_heatmap(result, out_path):
    orders = sorted(result["orders"].keys())
    all_ts = sorted({t for o in orders
                     for t in result["orders"][o]["linear_per_step"].keys()})
    M = np.full((len(orders), len(all_ts)), np.nan)
    for i, o in enumerate(orders):
        per = result["orders"][o]["linear_per_step"]
        for j, t in enumerate(all_ts):
            if t in per:
                M[i, j] = per[t]["r2_uniform"]

    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(all_ts), 1.0 + 0.5 * len(orders)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(all_ts))); ax.set_xticklabels(all_ts)
    ax.set_yticks(range(len(orders))); ax.set_yticklabels(orders)
    ax.set_xlabel("target timestep t")
    ax.set_ylabel("Markov order")
    ax.set_title(f"{fig_label(result['task'], result['model'])}:\nper-step linear R² (test)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def _fmt_signed(x, decimals=3):
    s = f"{x:+.{decimals}f}"
    return s.replace("+", r"\phantom{-}")


def write_summary(results, out_csv, out_tex, order=SUMMARY_ORDER):
    rows = []
    for r in sorted(results, key=lambda r: (sort_task_key(r["task"]), sort_model_key(r["model"]))):
        if order not in r["orders"]:
            continue
        o = r["orders"][order]
        rows.append({
            "task": TASK_LABELS.get(r["task"], r["task"]),
            "model": r["model"],
            "n_train": r.get("n_train"),
            "n_test": r.get("n_test"),
            "identity_R2": o["identity_baseline"]["r2_uniform"],
            "linear_R2": o["linear_shared"]["r2_uniform"],
            "mlp_R2": o["mlp_shared"]["r2_uniform"],
            "linear_gain": (o["linear_shared"]["r2_uniform"] - o["identity_baseline"]["r2_uniform"]),
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
        r"\caption{Bigram (order=1) markovianity summary: test-set $R^2$ for predicting $h_t$ from $h_{t-1}$. The linear transition is a single shared ridge regression. Rows with positive \emph{linear gain} (linear $R^2$ minus identity $R^2$) indicate that fitting a transition adds information beyond just copying the previous thought.}",
        r"\label{tab:markovianity_bigram}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"task & model & $N_\mathrm{tr}$ & $N_\mathrm{te}$ & identity $R^2$ & linear $R^2$ & MLP $R^2$ & linear gain \\",
        r"\midrule"
    ]
    for r in rows:
        cells = [
            r["task"],
            r["model"].replace("_", r"\_"),
            f"{r['n_train']}",
            f"{r['n_test']}",
            _fmt_signed(r["identity_R2"]),
            _fmt_signed(r["linear_R2"]),
            _fmt_signed(r["mlp_R2"]),
            _fmt_signed(r["linear_gain"]),
        ]
        if r["linear_gain"] > 0:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex ] {out_tex}")


def main():
    if not OUT_DIR.exists():
        raise SystemExit(f"OUT_DIR does not exist: {OUT_DIR}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = discover_results()
    if not all_results:
        raise SystemExit(f"No results_*.json found under {OUT_DIR}.")
    
    # Dynamically group results by their projection mode
    grouped_results = defaultdict(list)
    for r in all_results:
        grouped_results[r.get("projected", False)].append(r)

    # Plot separately for each mode, appending suffixes as needed
    for is_projected, results in grouped_results.items():
        suffix = "_subspace" if is_projected else ""
        
        plot_order_curves(results, FIG_DIR, suffix=suffix)
        plot_regime_scatter(results, FIG_DIR / f"regime_scatter{suffix}.png", order=SUMMARY_ORDER)

        for r in results:
            out_path = FIG_DIR / f"per_step_heatmap_{r['task']}_{r['model']}{suffix}.png"
            plot_per_step_heatmap(r, out_path)

        out_csv = OUT_DIR / f"summary{suffix}.csv"
        out_tex = OUT_DIR / f"summary{suffix}.tex"
        write_summary(results, out_csv, out_tex, order=SUMMARY_ORDER)

if __name__ == "__main__":
    main()