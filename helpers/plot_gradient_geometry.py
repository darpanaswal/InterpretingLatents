"""
Standalone plotting for gradient_subspace_diagnosis.py + bases.npz.

Reads, for each subspace source (gold and pred, independently):
    OUT_DIR / <family> / <task> / <model> / diagnosis.json   (from gradient_subspace_geometry.py)
    OUT_DIR / <family> / <task> / <model> / summary.json     (from gradient_subspace.py / _predtoken.py)
    OUT_DIR / <family> / <task> / <model> / bases.npz        (from gradient_subspace.py / _predtoken.py)
where OUT_DIR is outputs/gradient_geometry (gold) or
outputs/gradient_geometry_predtoken (pred). A source is skipped entirely
if its directory has no diagnosis.json files (e.g. gradient_subspace_geometry.py
was never run with --subspace_source pred).

Writes
    Plots/gradient_geometry/
        cos_panels_{model_family}.pdf             -- gold subspace stability cos^2(B_t, B_{t+1})
        cos_panels_{model_family}_pred.pdf         -- pred subspace, same plot
    Tables/statistical/
        subspace_geometry_{family}.tex            -- gold per-family appendix tables
        (pred subspace has plots only; no LaTeX table is produced for it)

The Q1 / Q2 alignment-of-variance-component panels were removed: they
conflate magnitude, rank, and per-sample vs population alignment, and
therefore cannot be used to argue causal relevance.

All input and output paths are hardcoded; no CLI flags.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import BASE_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "text.usetex": False,
})

# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "gradient_geometry"
OUT_DIR_PRED = BASE_DIR / "outputs" / "gradient_geometry_predtoken"
PLOT_DIR = Path("Plots/gradient_geometry")
TABLE_DIR = Path("Tables/statistical")

# Subspace sources to discover + plot: (label, source dir, filename suffix).
# Gold keeps the legacy unsuffixed filenames; pred sits alongside with
# "_pred" so nothing gets overwritten.
SOURCES = [
    ("gold", OUT_DIR, ""),
    ("pred", OUT_DIR_PRED, "_pred"),
]

# Recurrent models: their 7th timestep (index 6) has rank 0 and
# must be excluded when computing averages.
RECURRENT_MODELS = {"coconut", "coconut_u", "codi"}

MODEL_LABELS = {
    "coconut":   "C",
    "coconut_u": r"C$_u$",
    "pause":     "PaT",
    "codi":      "CODI",
}

MODEL_ORDER = ["pause", "coconut", "coconut_u", "codi"]
TASK_ORDER = ["prosqa", "gsm"]
FAMILY_ORDER = ["gpt2", "llama"]

FAMILY_LABELS = {
    "gpt2":  "GPT-2",
    "llama": "Llama",
}


def sort_family_key(family):
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return 99

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}

MODEL_STYLE = {
    "base":      {"color": "#999999", "marker": "x",  "ls": ":"},
    "cot":       {"color": "#E69F00", "marker": "^",  "ls": "--"},
    "pause":     {"color": "#0072B2", "marker": "o",  "ls": "-"},
    "coconut":   {"color": "#D55E00", "marker": "s",  "ls": "-"},
    "coconut_u": {"color": "#009E73", "marker": "D",  "ls": "-"},
    "codi":      {"color": "#CC79A7", "marker": "v",  "ls": "-."},
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

def discover_results(out_dir):
    """
    Walk out_dir / <family> / <task> / <model> / and load both
    diagnosis.json and bases.npz when both are present.  Also loads
    summary.json (full per-timestep Q1-Q4 statistics) when available.
    """
    results = []
    for diag_path in sorted(out_dir.glob("*/*/*/diagnosis.json")):
        with open(diag_path, "r") as f:
            r = json.load(f)
        
        family = r.get("model_family")
        if family is None:
            family = diag_path.parent.parent.parent.name
            
        summary_path = diag_path.parent / "summary.json"
        summary = None
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
                
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
            "family":    family,
            "task":      r["task"],
            "model":     r["model"],
            "diagnosis": r,
            "summary":   summary,
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
        figsize=(4.75, 1.1),
        sharex=False, sharey=True, squeeze=False,
        gridspec_kw={"wspace": 0.25}
    )

    by_task = {t: [r for r in results if r["task"] == t] for t in tasks}

    for j, task in enumerate(tasks):
        task_results = sorted(by_task[task], key=lambda r: sort_model_key(r["model"]))
        ax = axes[0][j]
        
        # Determine all x points
        all_xs = set()
        for r in task_results:
            model = r["model"]
            ys = r["diagnosis"]["q3_adjacent_per_t"]
            xs = np.arange(len(ys)) + 0.5
            if model in ["coconut", "coconut_u", "codi"] and len(ys) > 0:
                xs = xs[:-1]
            all_xs.update(xs)
        if not all_xs:
            continue
        all_xs = sorted(list(all_xs))
        
        endpoints = []
        
        for r in task_results:
            model = r["model"]
            ys = np.array(r["diagnosis"]["q3_adjacent_per_t"])
            xs = np.arange(len(ys)) + 0.5
            
            if model in ["coconut", "coconut_u", "codi"] and len(ys) > 0:
                ys = ys[:-1]
                xs = xs[:-1]
            
            style = MODEL_STYLE.get(model, {"color": "black", "marker": "o", "ls": "-"})
            
            ax.plot(xs, ys,
                    color=style["color"],
                    marker=style["marker"],
                    ls=style["ls"],
                    label=MODEL_LABELS.get(model, model),
                    markeredgewidth=0.3,
                    markeredgecolor="black",
                    zorder=3)
                    
            if len(ys) > 0:
                endpoints.append((ys[0], model, xs[0], style["color"]))

        ylim = (-0.05, 1.05)
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.5, ls="--", zorder=1)
        ax.axhline(0.0, color="#bbbbbb", linewidth=0.5, ls="--", zorder=1)

        y_range = ylim[1] - ylim[0]
        min_gap = 0.10 * y_range

        endpoints.sort(key=lambda item: item[0])
        placed = []
        staggered = []
        
        bottom_threshold = ylim[0] + 0.08 * y_range 

        for item in endpoints:
            y_val = item[0]
            label_y = max(y_val, bottom_threshold)
            
            for py in placed:
                if abs(label_y - py) < min_gap:
                    label_y = py + min_gap
            
            label_y = min(label_y, ylim[1] - 0.03 * y_range)
            placed.append(label_y)
            staggered.append((*item, label_y))

        line_extend = 0.5
        x_right_edge = max(all_xs)
        for y_val, mode, x_start, color, label_y in staggered:
            x_pts = [x_start, x_right_edge + 0.15, x_right_edge + 0.4, x_right_edge + line_extend]
            y_pts = [y_val, y_val, label_y, label_y]
            ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)

            ax.annotate(
                f"{y_val:.2f}",
                xy=(x_right_edge + line_extend, label_y),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                weight="bold",
                ha="left",
                va="center",
                clip_on=False
            )

        ax.set_title(f"({chr(97 + j)}) {TASK_LABELS.get(task, task)}", fontsize=8, pad=6)
        ax.set_xlabel("Timestep $t$", fontsize=7)
        if j == 0:
            ax.set_ylabel(r"Mean $\cos^2(B_t, B_{t+1})$", fontsize=7)
            
        left_pad = -0.2
        right_pad = 2.0
        ax.set_xlim(min(all_xs) + left_pad, max(all_xs) + right_pad)
        ax.set_xticks(all_xs)
        ax.set_xticklabels([f"{t:.1f}" for t in all_xs])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(handles),
        fontsize=6,
        frameon=True,
        framealpha=0.8,
        edgecolor="#cccccc",
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def _active_ranks(ranks, model):
    """Return ranks for active timesteps only (skip t=6 for recurrent models)."""
    if model in RECURRENT_MODELS:
        return [r for r in ranks[:-1] if r > 0]
    return [r for r in ranks if r > 0]


# ═══════════════════════════════════════════════════════════════════
# Per-family appendix table
# ═══════════════════════════════════════════════════════════════════

def build_per_family_geometry_table(family, family_results) -> str:
    """Build one appendix table for a single model family."""
    sorted_results = sorted(
        family_results,
        key=lambda r: (sort_task_key(r["task"]), sort_model_key(r["model"])),
    )
    if not sorted_results:
        return ""

    T = sorted_results[0]["diagnosis"]["T"]
    family_label = FAMILY_LABELS.get(family, family)

    t_headers = " & ".join([f"$k_{{{t}}}$" for t in range(T)])
    col_spec = "ll" + "r" * T + "r" + "c" + "c" + "r"

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        (rf"\caption{{Gradient-subspace diagnostics for the "
         rf"{family_label} family. Per-timestep subspace rank $k_t$ at "
         r"$\rho{=}0.95$, mean rank $\bar k$ (degenerate final step "
         r"excluded for recurrent models), adjacent-step similarity "
         r"$\overline{\cos^2}(B_t, B_{t+1})$, off-diagonal mean "
         r"similarity $\overline{\cos^2}(B_t, B_{t'})$ with 95\% "
         r"bootstrap CIs, and the pooled norm fraction "
         r"$\|h^c\|/\|h\|$ retained in the causal subspace.}"),
        rf"\label{{tab:subspace_geometry_{family}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (f"Task & Model & {t_headers} & $\\bar k$ "
         r"& Adj.\ $\overline{\cos^2}$ "
         r"& Off-diag.\ $\overline{\cos^2}$ "
         r"& $\|h^c\|/\|h\|$" + r" \\"),
        r"\midrule",
    ]

    for t_idx, task in enumerate(TASK_ORDER):
        task_results = [r for r in sorted_results if r["task"] == task]
        if not task_results:
            continue
        if t_idx > 0:
            lines.append(r"\midrule")

        n_models = len(task_results)
        task_str = TASK_LABELS.get(task, task).replace("-", r"-\\")
        task_cell = (rf"\multirow{{{n_models}}}{{*}}"
                     rf"{{\makecell[l]{{{task_str}}}}}")

        for m_idx, r in enumerate(task_results):
            model = r["model"]
            d = r["diagnosis"]
            ranks = d["subspace_ranks"]

            active = _active_ranks(ranks, model)
            mean_k = float(np.mean(active)) if active else 0.0

            q3_adj = d.get("q3_adjacent_per_t", [])
            q3_adj_active = (q3_adj[:-1]
                             if model in RECURRENT_MODELS and q3_adj
                             else q3_adj)
            q3_adj_mean = (float(np.mean(q3_adj_active))
                           if q3_adj_active else float("nan"))

            q3_offdiag_mean = d.get("q3_offdiag_mean", float("nan"))
            q3_offdiag_ci = d.get("q3_offdiag_ci", [None, None])

            summary = r.get("summary")
            if summary and "grad" in summary:
                q4_pooled = summary["grad"].get("q4_norm_budget_pooled", {})
                q4_mean = q4_pooled.get("mean", float("nan"))
            else:
                q4_mean = float("nan")

            rank_cells = [
                (r"\textcolor{gray}{--}" if rk == 0 else str(rk))
                for rk in ranks
            ]

            q3_adj_str = (f"${q3_adj_mean:.3f}$"
                          if not np.isnan(q3_adj_mean) else "--")

            if q3_offdiag_ci[0] is not None and q3_offdiag_ci[1] is not None:
                hw = (q3_offdiag_ci[1] - q3_offdiag_ci[0]) / 2
                q3_od_str = (rf"${q3_offdiag_mean:.3f}"
                             rf" {{\scriptstyle \pm {hw:.3f}}}$")
            else:
                q3_od_str = f"${q3_offdiag_mean:.3f}$"

            q4_str = (f"${q4_mean:.3f}$"
                      if not np.isnan(q4_mean) else "--")

            first_col = task_cell if m_idx == 0 else ""
            cells = [
                first_col,
                MODEL_LABELS.get(model, model),
                *rank_cells,
                f"{mean_k:.1f}",
                q3_adj_str,
                q3_od_str,
                q4_str,
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_all_family_geometry_tables(results) -> dict:
    """Return {family: latex_str} for every family present in results."""
    families = sorted(
        {r["family"] for r in results}, key=sort_family_key
    )
    out = {}
    for family in families:
        fam_results = [r for r in results if r["family"] == family]
        out[family] = build_per_family_geometry_table(family, fam_results)
    return out


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    any_found = False
    for source, out_dir, suffix in SOURCES:
        if not out_dir.exists():
            print(f"[skip] {source}: {out_dir} does not exist")
            continue
        results = discover_results(out_dir)
        if not results:
            print(f"[skip] {source}: no diagnosis.json found under "
                  f"{out_dir}/*/*/*/")
            continue
        any_found = True

        # ── Plots by family ─────────────────────────────────────────
        families = sorted({r["family"] for r in results}, key=sort_family_key)
        for family in families:
            fam_results = [r for r in results if r["family"] == family]
            plot_path = PLOT_DIR / f"cos_panels_{family}{suffix}.pdf"
            plot_cos_panels(fam_results, plot_path)

        # ── Per-family geometry tables (gold subspace only) ─────────────
        if source == "gold":
            family_tables = build_all_family_geometry_tables(results)
            for family, tex in family_tables.items():
                if not tex:
                    continue
                p = TABLE_DIR / f"subspace_geometry_{family}{suffix}.tex"
                p.write_text(tex)
                print(f"[tex ] Geometry table [{source}/{family}] -> {p.resolve()}")

    if not any_found:
        raise SystemExit(
            f"No diagnosis.json found under {OUT_DIR}/*/*/*/ or "
            f"{OUT_DIR_PRED}/*/*/*/."
        )


if __name__ == "__main__":
    main()