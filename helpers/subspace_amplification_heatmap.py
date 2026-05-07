"""
Compact print-ready heatmap of subspace-amplification flip rates.
Two panels side-by-side (ProsQA | GSM8k), shared horizontal colorbar.

Loads data from amplification_results.json files written by
gradient_subspace_interventions.py, and computes pooled median ||h_t||
per (task, model) directly from the cached thought vectors written by
extract_thoughts.py.

Two marker glyphs are added next to cells:
  '·'  first alpha (smallest in sweep) with ratio r = alpha/median > 0.1
  '°'  first alpha (smallest in sweep) with ratio r > 5
where median is the per-(task, model) pooled median ||h_t||.
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib as mpl
from src.config import OUTPUTS, THOUGHTS
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ─── Configuration ───────────────────────────────────────────────

GRAD_SUBSPACE = OUTPUTS / "grad_subspace"

TASKS = ["prosqa", "gsm"]
MODELS = ["pause", "coconut", "coconut_u", "codi"]
MODEL_LABELS = {
    "pause": "Pause",
    "coconut": "Coconut",
    "coconut_u": r"Coconut-$u$",
    "codi": "CODI",
}
TASK_LABELS = {"prosqa": "Graph-Hopping", "gsm": "Arithmetic-Reasoning"}

MODEL_TO_THOUGHT_FILE = {
    "pause": "pause",
    "coconut": "coconut",
    "coconut_u": "coconut_u",
    "codi": "codi",
}

# r-thresholds and the glyph drawn at the smallest alpha that crosses each.
# Order matters: if a single alpha crosses both thresholds, we draw the
# higher-threshold glyph (the more meaningful event).
R_THRESHOLDS = [
    (0.1, "·"),   # first alpha with r > 0.1
    (5.0, "°"),   # first alpha with r > 5
]


def load_amplification_results(task, model):
    path = GRAD_SUBSPACE / task / model / "amplification_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_flip(steering_dict, a):
    for key in [a, str(a), str(float(a))]:
        if str(key) in steering_dict:
            return steering_dict[str(key)]["flip_rate"]
        if key in steering_dict:
            return steering_dict[key]["flip_rate"]
    return None


# ─── Pooled median ||h_t|| ───────────────────────────────────────
#
# Math (matches compute_alpha_regimes in interventions.py):
#   thoughts ∈ R^{N x T x D}        (T = K+1, all cached timesteps)
#   h_norms[n, t] = || thoughts[n, t, :] ||_2
#   median_pooled = median over all (n, t) of h_norms

def compute_pooled_median_norm(task, model):
    fname = f"thoughts_{MODEL_TO_THOUGHT_FILE[model]}.pt"
    path = THOUGHTS / task / fname
    if not path.exists():
        return None

    obj = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = obj["thoughts"]
    if not torch.is_tensor(thoughts):
        thoughts = torch.as_tensor(thoughts)
    h_norms = thoughts.float().norm(dim=-1)
    return float(h_norms.median().item())


def compute_all_median_norms():
    return {t: {m: compute_pooled_median_norm(t, m) for m in MODELS}
            for t in TASKS}


def format_median_caption_line(task, medians_for_task):
    short = {"pause": "PaT", "coconut": "C",
             "coconut_u": "Cu", "codi": "CODI"}
    task_short = {"prosqa": "ProsQA", "gsm": "GSM8k"}
    parts = []
    for m in MODELS:
        v = medians_for_task[m]
        parts.append(f"{short[m]}=N/A" if v is None else f"{short[m]}={v:.1f}")
    return f"{task_short[task]}: " + ", ".join(parts)


# ─── r-threshold markers ─────────────────────────────────────────
#
# For each (task, model), find the first alpha (in displayed order)
# whose ratio r = alpha / median crosses each R_THRESHOLD.
# If the same alpha is the crossing point for multiple thresholds,
# the higher-threshold glyph wins.
#
# Returns: marker_rows[task][model] = {alpha_index: glyph}

def compute_marker_rows(alphas, medians):
    marker_rows = {}
    for t in TASKS:
        marker_rows[t] = {}
        for m in MODELS:
            med = medians[t][m]
            row_to_glyph = {}
            if med is None or med <= 0:
                marker_rows[t][m] = row_to_glyph
                continue
            ratios = [float(a) / med for a in alphas]
            for thr, glyph in R_THRESHOLDS:
                # First (smallest-α) row index with r > thr.
                idx = next((i for i, r in enumerate(ratios) if r > thr), None)
                if idx is not None:
                    # Higher threshold overwrites lower if they coincide.
                    row_to_glyph[idx] = glyph
            marker_rows[t][m] = row_to_glyph
    return marker_rows


# ─── Data assembly ───────────────────────────────────────────────

def assemble_data():
    results = {t: {m: load_amplification_results(t, m) for m in MODELS}
               for t in TASKS}

    alphas = None
    for t in TASKS:
        for m in MODELS:
            if results[t][m] is not None:
                alphas = results[t][m]["alphas"]
                break
        if alphas is not None:
            break
    if alphas is None:
        raise RuntimeError(
            f"No amplification_results.json found under {GRAD_SUBSPACE}"
        )

    alphas = [a for a in alphas if float(a) > 1.0]

    n_alphas = len(alphas)
    n_cols = 2 * len(MODELS)
    panels = {}

    for t in TASKS:
        mat = np.full((n_alphas, n_cols), np.nan)
        for m_idx, m in enumerate(MODELS):
            r = results[t][m]
            grad_col, rand_col = 2 * m_idx, 2 * m_idx + 1
            if r is None:
                continue
            for a_idx, a in enumerate(alphas):
                grad_fr = get_flip(r["grad_amplification"], a)
                rand_fr = get_flip(r["rand_amplification"], a)
                if grad_fr is not None:
                    mat[a_idx, grad_col] = grad_fr * 100.0
                if rand_fr is not None:
                    mat[a_idx, rand_col] = rand_fr * 100.0
        panels[t] = mat

    return alphas, panels


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


# Horizontal offset (in axis data units, where each cell spans 1.0)
# from the cell centre to where the marker glyph is drawn.
MARKER_X_OFFSET = -0.38


def draw_panel(ax, data, title, show_ylabel, alphas,
               marker_rows_for_task):
    n_rows, n_cols = data.shape

    masked = np.ma.masked_invalid(data)
    pink_stops = ['#FBEAF0', '#F4C0D1', '#ED93B1', '#D4537E',
                  '#993556', '#72243E', '#4B1528']
    cmap = LinearSegmentedColormap.from_list('pink_ramp', pink_stops, N=256)
    cmap_local = cmap.copy()
    cmap_local.set_bad(color='#F2F2F2')
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    im = ax.imshow(masked, aspect='auto', cmap=cmap_local,
                   norm=norm, interpolation='nearest')

    # Cell text + marker glyphs.
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

            # Marker: column j belongs to model j // 2.
            model_idx = j // 2
            model_key = MODELS[model_idx]
            glyph = marker_rows_for_task.get(model_key, {}).get(i)
            if glyph is not None:
                # Larger glyph for the open circle so it reads at print size.
                gsize = 9.0 if glyph == "°" else 11.0
                ax.text(j + MARKER_X_OFFSET, i, glyph,
                        ha='center', va='center',
                        fontsize=gsize, color=tcolor)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([format_alpha_tick(a) for a in alphas], fontsize=7)
    if show_ylabel:
        ax.set_ylabel(r'$\alpha$', fontsize=9, rotation=0,
                      labelpad=8, va='center')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(['GRAD', 'Rand'] * len(MODELS), fontsize=6.5)
    ax.tick_params(axis='both', length=0, pad=2)

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks([2 * k + 0.5 for k in range(len(MODELS))])
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8)
    ax2.tick_params(axis='x', length=0, pad=3)

    for k in range(1, len(MODELS)):
        ax.axvline(2 * k - 0.5, color='white', lw=1.2)

    ax.set_title(title, fontsize=9, color='#888888', loc='left', pad=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.4)
    return im


def make_figure(alphas, panels, marker_rows, out_pdf, out_png):
    setup_rcparams()

    FIG_W, FIG_H = 7.0, 3.4
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={'wspace': 0.18},
    )

    im_last = None
    for idx, t in enumerate(TASKS):
        im_last = draw_panel(
            axes[idx], panels[t], TASK_LABELS[t],
            show_ylabel=(idx == 0),
            alphas=alphas,
            marker_rows_for_task=marker_rows[t],
        )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22)

    # Colorbar.
    cbar_ax = fig.add_axes([0.20, 0.07, 0.30, 0.025])
    cb = fig.colorbar(im_last, cax=cbar_ax, orientation='horizontal')
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=7, length=0, pad=2)
    cb.outline.set_linewidth(0.4)
    fig.text(0.18, 0.082, 'Flip rate (%)', ha='right', va='center',
             fontsize=8, color='#444444')

    # Marker legend.
    fig.text(0.56, 0.082, r'$\cdot$  first $\alpha$ with $r>0.1$',
             ha='left', va='center', fontsize=8, color='#444444')
    fig.text(0.78, 0.082, r'$\circ$  first $\alpha$ with $r>5$',
             ha='left', va='center', fontsize=8, color='#444444')

    out_pdf, out_png = Path(out_pdf), Path(out_png)
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
        "--out-dir", type=Path, default=GRAD_SUBSPACE / "figures",
        help="Output directory (default: outputs/grad_subspace/figures)",
    )
    args = parser.parse_args()

    alphas, panels = assemble_data()
    medians = compute_all_median_norms()
    marker_rows = compute_marker_rows(alphas, medians)

    print("\nPooled median ||h_t|| (for figure caption):")
    for t in TASKS:
        print("  " + format_median_caption_line(t, medians[t]))
    print()

    out_pdf = args.out_dir / "amplification_flip_rates_heatmap.pdf"
    out_png = args.out_dir / "amplification_flip_rates_heatmap.png"
    make_figure(alphas, panels, marker_rows, out_pdf, out_png)


if __name__ == "__main__":
    main()