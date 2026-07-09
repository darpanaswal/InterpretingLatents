"""
Compact print-ready heatmap of steering flip rates.
Two panels side-by-side (ProsQA | GSM8k), shared horizontal colorbar
and threshold-marker legend below.

Loads data from steering_results_fast.json files via the same conventions
as steering_tables.py — single source of truth for table + figure.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from src.config import INLP

# ─── Configuration (mirrors steering_tables.py) ──────────────────

TASKS = ["prosqa", "gsm"]
MODELS = ["pause", "coconut", "coconut_u", "codi"]
MODEL_LABELS = {
    "pause": "Pause",
    "coconut": "Coconut",
    "coconut_u": r"Coconut-$u$",
    "codi": "CODI",
}
TASK_LABELS = {"prosqa": "ProsQA", "gsm": "GSM8k"}

GENUINE_MAX = 0.1
MAGNITUDE_MIN = 5.0


def regime(alpha, median_norm):
    r = alpha / median_norm
    if r <= GENUINE_MAX:
        return "G"
    if r <= MAGNITUDE_MIN:
        return "T"
    return "M"


def load_steering_results(task, model):
    path = INLP / task / model / "steering_results_fast.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_flip(steering_dict, a):
    """Look up flip_rate by alpha across possible key formats."""
    for key in [a, str(a), str(float(a))]:
        if str(key) in steering_dict:
            return steering_dict[str(key)]["flip_rate"]
        if key in steering_dict:
            return steering_dict[key]["flip_rate"]
    return None


# ─── Data assembly ───────────────────────────────────────────────

def assemble_data():
    """
    Returns:
      alphas:        list of α values
      panels:        {task: ndarray (n_alphas, 2*n_models)} with flip rates in %
                     (NaN where data missing)
      dag_cells:     {task: list of (alpha_idx, col_idx) for † markers}
      ddag_cells:    {task: list of (alpha_idx, col_idx) for ‡ markers}
      median_norms:  {task: {model: float or None}}
    """
    results = {t: {m: load_steering_results(t, m) for m in MODELS} for t in TASKS}

    # Discover alpha list from any available result
    alphas = None
    for t in TASKS:
        for m in MODELS:
            if results[t][m] is not None:
                alphas = results[t][m]["alphas"]
                break
        if alphas is not None:
            break
    if alphas is None:
        raise RuntimeError("No steering_results_fast.json found anywhere")

    # Median pooled norms
    median_norms = {t: {} for t in TASKS}
    for t in TASKS:
        for m in MODELS:
            r = results[t][m]
            median_norms[t][m] = (
                r["regime_info"]["median_pooled"] if r is not None else None
            )

    # First-T / first-M index per (task, model) — mirrors steering_tables.py exactly:
    # only marks G→T and T→M transitions, not the starting regime.
    first_T = {t: {} for t in TASKS}
    first_M = {t: {} for t in TASKS}
    for t in TASKS:
        for m in MODELS:
            first_T[t][m] = None
            first_M[t][m] = None
            mn = median_norms[t][m]
            if mn is None:
                continue
            prev = None
            for i, a in enumerate(alphas):
                cur = regime(a, mn)
                if prev == "G" and cur == "T":
                    first_T[t][m] = i
                if prev == "T" and cur == "M":
                    first_M[t][m] = i
                prev = cur

    # Build per-task data matrices: rows=alpha, cols=(model × {INLP, Rand})
    n_alphas = len(alphas)
    n_cols = 2 * len(MODELS)
    panels = {}
    dag_cells = {t: [] for t in TASKS}
    ddag_cells = {t: [] for t in TASKS}

    for t in TASKS:
        mat = np.full((n_alphas, n_cols), np.nan)
        for m_idx, m in enumerate(MODELS):
            r = results[t][m]
            inlp_col = 2 * m_idx
            rand_col = 2 * m_idx + 1
            if r is None:
                continue
            for a_idx, a in enumerate(alphas):
                inlp_fr = get_flip(r["inlp_steering"], a)
                rand_fr = get_flip(r["rand_steering"], a)
                if inlp_fr is not None:
                    mat[a_idx, inlp_col] = inlp_fr * 100.0
                if rand_fr is not None:
                    mat[a_idx, rand_col] = rand_fr * 100.0

            # Markers placed on the INLP column only (matches table convention)
            if first_T[t][m] is not None:
                dag_cells[t].append((first_T[t][m], inlp_col))
            if first_M[t][m] is not None:
                ddag_cells[t].append((first_M[t][m], inlp_col))

        panels[t] = mat

    return alphas, panels, dag_cells, ddag_cells, median_norms


# ─── Plotting ────────────────────────────────────────────────────

def format_alpha_tick(a):
    return str(int(a)) if a == int(a) else str(a)


def cell_text(v):
    if np.isnan(v):
        return "—"
    if v == 0.0:
        return "0"
    if v >= 99.95:
        return "100"
    return f"{v:.1f}"


def setup_rcparams():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'


def make_figure(alphas, panels, dag_cells, ddag_cells, median_norms,
                out_pdf, out_png):
    setup_rcparams()

    pink_stops = ['#FBEAF0', '#F4C0D1', '#ED93B1', '#D4537E',
                  '#993556', '#72243E', '#4B1528']
    cmap = LinearSegmentedColormap.from_list('pink_ramp', pink_stops, N=256)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    FIG_W = 7.0
    FIG_H = 3.2
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={'wspace': 0.18},
    )

    model_display = [MODEL_LABELS[m] for m in MODELS]

    def draw_panel(ax, data, title, dags, ddags, show_ylabel):
        n_rows, n_cols = data.shape

        masked = np.ma.masked_invalid(data)
        cmap_local = cmap.copy()
        cmap_local.set_bad(color='#F2F2F2')

        im = ax.imshow(masked, aspect='auto', cmap=cmap_local,
                       norm=norm, interpolation='nearest')

        # Cell text
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                label = cell_text(v)
                if np.isnan(v):
                    tcolor = '#888888'
                else:
                    tcolor = 'white' if v > 55 else '#3A0E1F'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=6.0, color=tcolor)

        # Threshold markers — upper-left of cell
        for (i, j) in dags:
            v = data[i, j]
            tcolor = 'white' if (not np.isnan(v) and v > 55) else '#3A0E1F'
            ax.text(j - 0.42, i - 0.30, r'$\cdot$', ha='left', va='top',
                    fontsize=10, color=tcolor)
        for (i, j) in ddags:
            v = data[i, j]
            tcolor = 'white' if (not np.isnan(v) and v > 55) else '#3A0E1F'
            ax.text(j - 0.42, i - 0.30, r'$\circ$', ha='left', va='top',
                    fontsize=6, color=tcolor)

        # Y axis: alpha values
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([format_alpha_tick(a) for a in alphas], fontsize=7)
        if show_ylabel:
            ax.set_ylabel(r'$\alpha$', fontsize=9, rotation=0,
                          labelpad=8, va='center')

        # X axis: INLP / Rand
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(['INLP', 'Rand'] * len(MODELS), fontsize=6.5)
        ax.tick_params(axis='both', length=0, pad=2)

        # Top axis: model labels (centered over each pair of columns)
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks([2 * k + 0.5 for k in range(len(MODELS))])
        ax2.set_xticklabels(model_display, fontsize=8)
        ax2.tick_params(axis='x', length=0, pad=3)

        # White dividers between models
        for k in range(1, len(MODELS)):
            ax.axvline(2 * k - 0.5, color='white', lw=1.2)

        ax.set_title(title, fontsize=9, color='#888888', loc='left', pad=14)

        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(0.4)

        return im

    im_last = None
    for idx, t in enumerate(TASKS):
        im_last = draw_panel(
            axes[idx], panels[t], TASK_LABELS[t],
            dag_cells[t], ddag_cells[t],
            show_ylabel=(idx == 0),
        )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.20)

    cbar_ax = fig.add_axes([0.20, 0.06, 0.30, 0.025])
    cb = fig.colorbar(im_last, cax=cbar_ax, orientation='horizontal')
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=7, length=0, pad=2)
    cb.outline.set_linewidth(0.4)

    fig.text(0.18, 0.072, 'Flip rate (%)', ha='right', va='center',
             fontsize=8, color='#444444')

    fig.text(0.55, 0.072,
             r'$\cdot$ first $\alpha$ with $r > 0.1$' + '     ' +
             r'$\circ$ first $\alpha$ with $r > 5$',
             ha='left', va='center', fontsize=7.5, color='#444444')

    out_pdf = Path(out_pdf)
    out_png = Path(out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


# ─── Entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=INLP / "figures",
        help="Output directory (default: INLP/figures)",
    )
    args = parser.parse_args()

    alphas, panels, dag_cells, ddag_cells, median_norms = assemble_data()

    out_pdf = args.out_dir / "steering_flip_rates_heatmap.pdf"
    out_png = args.out_dir / "steering_flip_rates_heatmap.png"
    make_figure(alphas, panels, dag_cells, ddag_cells, median_norms,
                out_pdf, out_png)

    print("\nMedian pooled norms (for caption):")
    for t in TASKS:
        parts = []
        for m in MODELS:
            mn = median_norms[t][m]
            parts.append(
                f"{MODEL_LABELS[m]}={mn:.1f}" if mn is not None
                else f"{MODEL_LABELS[m]}=n/a"
            )
        print(f"  {TASK_LABELS[t]}: " + ", ".join(parts))


if __name__ == "__main__":
    main()