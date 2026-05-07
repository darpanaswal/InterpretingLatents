"""
gradient_subspace_tables.py
============================

Reads artefacts produced by gradient_subspace_interventions.py and emits:

  Main text
  ---------
  1. tab_gradsubspace_main.tex      — ablation performance table
                                      (Original | Grad-Ablated | Rand-Control)
  2. fig_amplification_heatmap.pdf  — flip-rate heatmap (GRAD vs Rand, both tasks)
  3. fig_amplification_heatmap.png

  Appendix
  --------
  4. tab_gradsubspace_ablation_stats.tex   — bootstrap CIs + McNemar for ablation
  5. tab_gradsubspace_amp_stats.tex        — per-alpha CIs + McNemar for amplification

Usage
-----
    python gradient_subspace_tables.py

    # Discover paths and loaded values without writing any files:
    python gradient_subspace_tables.py --debug

    # Override output directory:
    python gradient_subspace_tables.py --out-dir results/grad_subspace
"""

import json
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from src.config import OUTPUTS, THOUGHTS

# ─── Canonical layout ────────────────────────────────────────────────────────

GRAD_SUBSPACE = OUTPUTS / "grad_subspace"

MODELS = [
    ("pause",      "PaT"),
    ("coconut",    "C"),
    ("coconut_u",  r"C$_u$"),
    ("codi",       "CODI"),
]

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm",    "Arithmetic-Reasoning"),
]

# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load_json(path: Path, debug: bool = False):
    if not path.exists():
        if debug:
            print(f"  [MISS] {path}")
        return None
    if debug:
        print(f"  [LOAD] {path}")
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path, debug: bool = False) -> list:
    if not path.exists():
        if debug:
            print(f"  [MISS] {path}")
        return []
    if debug:
        print(f"  [LOAD] {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if debug:
        phases = list({r.get("_context", {}).get("phase", "?") for r in records})
        print(f"         -> {len(records)} records; phases: {phases}")
    return records


def find_record(records: list, metric: str, phase: str = None):
    """Return the first record matching metric and (optionally) phase."""
    for r in records:
        if r.get("metric") != metric:
            continue
        if phase is not None and r.get("_context", {}).get("phase") != phase:
            continue
        return r
    return None


def find_amp_record(records: list, metric: str, alpha, condition: str = None):
    """Return the first amplification record matching metric, alpha, condition."""
    alpha_f = float(alpha)
    for r in records:
        ctx = r.get("_context", {})
        if r.get("metric") != metric:
            continue
        if abs(float(ctx.get("alpha", -1)) - alpha_f) > 1e-6:
            continue
        if condition is not None and ctx.get("condition") != condition:
            continue
        return r
    return None


# ─── Formatting ──────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=1) -> str:
    # 0-1 float -> percentage string, e.g. 0.954 -> "95.4"
    return f"{v * 100:.{decimals}f}" if v is not None else "--"


def ci_pct(r) -> str:
    """BootstrapResult dict -> 'point [lo, hi]' as percentages."""
    if r is None:
        return "--"
    p, lo, hi = r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def diff_pct(r) -> str:
    """Signed paired-diff CI as percentages (positive = first condition wins)."""
    if r is None:
        return "--"
    p, lo, hi = r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f} [{lo:.1f}, {hi:.1f}]"


def fmt_mcnemar(r) -> str:
    if r is None or r.get("p_value") is None:
        return "--"
    pv = r["p_value"]
    b  = r.get("b_a_only", "?")
    c  = r.get("c_b_only", "?")
    stars = (
        r"$^{***}$" if pv < 0.001 else
        r"$^{**}$"  if pv < 0.01  else
        r"$^{*}$"   if pv < 0.05  else ""
    )
    return f"{pv:.4f}{stars} (b={b}, c={c})"


def fmt_flip_ci(r) -> str:
    """Flip-rate bootstrap CI stored as 0-1; convert to %."""
    if r is None:
        return "--"
    p, lo, hi = r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def fmt_alpha(a) -> str:
    return str(int(a)) if a == int(a) else str(a)


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_ablation(debug: bool = False) -> dict:
    """
    For each (task, model) load ablation_results.json + cis.jsonl (phase=ablation).
    Returns nested dict: data[task][model] = entry dict or None.
    """
    data = {}
    for task, task_label in TASKS:
        data[task] = {}
        if debug:
            print(f"\n-- ABLATION  {task_label} ({task}) --")
        for dir_name, _ in MODELS:
            base = GRAD_SUBSPACE / task / dir_name
            if debug:
                print(f"  model={dir_name}  base={base}")
            summary = load_json(base / "ablation_results.json", debug)
            if summary is None:
                data[task][dir_name] = None
                continue
            data[task][dir_name] = {
                "n":         summary.get("n_instances"),
                "orig":      summary.get("baseline_accuracy"),
                "grad":      summary.get("grad_accuracy"),
                "rand":      summary.get("rand_accuracy"),
                # CIs stored as 0-1 probabilities
                "ci_orig":   summary.get("ci_baseline_accuracy"),
                "ci_grad":   summary.get("ci_grad_accuracy"),
                "ci_rand":   summary.get("ci_rand_accuracy"),
                # Paired diffs (baseline - ablated; positive = ablation hurts)
                "diff_grad": summary.get("paired_diff_baseline_minus_grad"),
                "diff_rand": summary.get("paired_diff_baseline_minus_rand"),
                # McNemar exact tests
                "mc_grad":   summary.get("mcnemar_baseline_vs_grad"),
                "mc_rand":   summary.get("mcnemar_baseline_vs_rand"),
            }
    return data


def collect_amplification(debug: bool = False) -> dict:
    """
    For each (task, model) load amplification_results.json + cis.jsonl
    (phase=amplification).
    Returns: amp[task][model] = {"alphas", "n", "raw", "records"} or None.
    """
    amp = {}
    for task, task_label in TASKS:
        amp[task] = {}
        if debug:
            print(f"\n-- AMPLIFICATION  {task_label} ({task}) --")
        for dir_name, _ in MODELS:
            base = GRAD_SUBSPACE / task / dir_name
            summary = load_json(base / "amplification_results.json", debug)
            records = load_jsonl(base / "cis.jsonl", debug)
            if summary is None:
                amp[task][dir_name] = None
                continue
            amp[task][dir_name] = {
                "n":       summary.get("n_instances"),
                "alphas":  [float(a) for a in summary.get("alphas", [])],
                "raw":     summary,
                "records": records,
            }
    return amp


# ─── Pooled median ||h_t|| (for amplification heatmap markers) ───────────────
#
# Math:
#   thoughts in R^{N x T x D}
#   h_norms[n, t] = || thoughts[n, t, :] ||_2
#   median_pooled = median over all (n, t) of h_norms

def compute_pooled_median_norm(task: str, model: str):
    fname = f"thoughts_{model}.pt"
    path  = THOUGHTS / task / fname
    if not path.exists():
        return None
    obj      = torch.load(path, map_location="cpu", weights_only=False)
    thoughts = obj["thoughts"]
    if not torch.is_tensor(thoughts):
        thoughts = torch.as_tensor(thoughts)
    h_norms = thoughts.float().norm(dim=-1)
    return float(h_norms.median().item())


# r-threshold markers for the heatmap.
# For each (task, model) and alpha index i (into the display_alphas list),
# annotate the FIRST alpha whose ratio r = alpha / median_norm crosses the
# threshold. Higher-threshold glyph overwrites lower when they share a row.
R_THRESHOLDS = [
    (0.1, r"$\cdot$"),
    (5.0, r"$\circ$"),
]


def compute_marker_rows(alphas: list, medians: dict) -> dict:
    """Returns marker_rows[task][model] = {alpha_index: glyph_string}."""
    marker_rows = {}
    for task, _ in TASKS:
        marker_rows[task] = {}
        for dir_name, _ in MODELS:
            med = medians[task][dir_name]
            row_to_glyph = {}
            if med is None or med <= 0:
                marker_rows[task][dir_name] = row_to_glyph
                continue
            ratios = [float(a) / med for a in alphas]
            for thr, glyph in R_THRESHOLDS:
                idx = next((i for i, r in enumerate(ratios) if r > thr), None)
                if idx is not None:
                    row_to_glyph[idx] = glyph   # higher threshold wins on collision
            marker_rows[task][dir_name] = row_to_glyph
    return marker_rows


# ─── Main text: ablation performance table ───────────────────────────────────

ABLATION_MAIN_TEMPLATE = r"""\begin{{table}}[!h]
\centering
\tiny
\caption{{Accuracy scores with original thought vectors, gradient-subspace ablations,
and random-subspace controls. Percentage scores are reported.}}
\label{{tab:subspace_ablation}}
\begin{{tabular}}{{lccc|ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{Graph-Hopping}} & \multicolumn{{3}}{{c}}{{Arithmetic-Reasoning}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
Model & Original & \makecell{{Gradient \\ Subspace \\ Ablated}} & \makecell{{Rand \\ Control}}
      & Original & \makecell{{Gradient \\ Subspace \\ Ablated}} & \makecell{{Rand \\ Control}}\\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def build_ablation_main_table(data: dict) -> str:
    rows = []
    for dir_name, col_label in MODELS:
        cells = [col_label]
        for task, _ in TASKS:
            entry = data[task].get(dir_name)
            if entry is None:
                cells += ["--", "--", "--"]
            else:
                cells += [
                    fmt_pct(entry.get("orig")),
                    fmt_pct(entry.get("grad")),
                    fmt_pct(entry.get("rand")),
                ]
        rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    return ABLATION_MAIN_TEMPLATE.format(rows="\n".join(rows))


# ─── Appendix: ablation statistical tests ────────────────────────────────────

def build_ablation_stats_table(data: dict) -> str:
    lines = []
    lines += [
        r"\subsection{Gradient-Subspace Ablation (Experiment~2, Part~1)}",
        r"\label{app:gradsubspace_ablation_stats}",
        "",
        r"Point estimates and 95\% bootstrap CIs (10{,}000 percentile resamples) "
        r"for accuracy under three conditions. "
        r"$\Delta_{\text{grad}} = \text{Acc}_{\text{orig}} - \text{Acc}_{\text{grad}}$ "
        r"(positive = gradient ablation hurts). "
        r"McNemar $p$ is exact two-sided binomial on discordant pairs "
        r"($b$: correct only under original; $c$: correct only under ablation).",
        "",
    ]

    for task, task_label in TASKS:
        lines += [
            r"\begin{table}[h!]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Gradient-subspace ablation: " + task_label + r".}",
            r"\begin{tabular}{l ccc cc cc c}",
            r"\toprule",
            (r"Model & Acc$_{\text{orig}}$ [\% CI] "
             r"& Acc$_{\text{grad}}$ [\% CI] "
             r"& Acc$_{\text{rand}}$ [\% CI] "
             r"& $\Delta_{\text{grad}}$ [\% CI] "
             r"& McNemar$_{\text{grad}}$ $p$ "
             r"& $\Delta_{\text{rand}}$ [\% CI] "
             r"& McNemar$_{\text{rand}}$ $p$ "
             r"& $n$ \\"),
            r"\midrule",
        ]

        for dir_name, col_label in MODELS:
            entry = data[task].get(dir_name)
            if entry is None:
                lines.append(
                    f"{col_label} & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
                )
            else:
                lines.append(
                    f"{col_label} & "
                    f"{ci_pct(entry['ci_orig'])} & "
                    f"{ci_pct(entry['ci_grad'])} & "
                    f"{ci_pct(entry['ci_rand'])} & "
                    f"{diff_pct(entry['diff_grad'])} & "
                    f"{fmt_mcnemar(entry['mc_grad'])} & "
                    f"{diff_pct(entry['diff_rand'])} & "
                    f"{fmt_mcnemar(entry['mc_rand'])} & "
                    f"{entry['n'] or '--'} \\\\"
                )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:gradsubspace_ablation_stats_" + task + r"}",
            r"\end{table}",
            "",
        ]

    lines.append(
        r"\noindent$^{*}p{<}0.05$,\quad $^{**}p{<}0.01$,\quad "
        r"$^{***}p{<}0.001$ (exact two-sided McNemar)."
    )
    return "\n".join(lines).strip()


# ─── Appendix: amplification statistical tests ───────────────────────────────

def build_amp_stats_table(amp: dict) -> str:
    lines = []
    lines += [
        r"\subsection{Gradient-Subspace Amplification (Experiment~2, Part~2)}",
        r"\label{app:gradsubspace_amp_stats}",
        "",
        r"Per-$\alpha$ flip-rate 95\% bootstrap CIs (10{,}000 resamples). "
        r"A \emph{flip} occurs when the model's normalised output text changes "
        r"relative to the unintervened baseline. "
        r"$\Delta = \text{flip}_{\text{grad}} - \text{flip}_{\text{rand}}$ "
        r"(positive = gradient subspace causes more flips than a rank-matched "
        r"random control). "
        r"McNemar $p$ tests whether grad and rand flip the \emph{same instances}.",
        "",
    ]

    for task, task_label in TASKS:
        lines += [
            r"\begin{table}[h!]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Gradient-subspace amplification flip-rate statistics: "
            + task_label + r".}",
            r"\begin{tabular}{l l cccc c}",
            r"\toprule",
            (r"Model & $\alpha$ "
             r"& Flip$_{\text{grad}}$ [\% CI] "
             r"& Flip$_{\text{rand}}$ [\% CI] "
             r"& $\Delta$ [\% CI] "
             r"& McNemar $p$ "
             r"& $n$ \\"),
            r"\midrule",
        ]

        for dir_name, col_label in MODELS:
            entry = amp[task].get(dir_name)
            if entry is None:
                lines.append(f"{col_label} & -- & -- & -- & -- & -- & -- \\\\")
                lines.append(r"\midrule")
                continue

            records = entry["records"]
            # Exclude alpha=1.0 (identity sanity check, not a result)
            display_alphas = [a for a in entry["alphas"] if abs(a - 1.0) > 1e-6]
            first_row = True
            for a in display_alphas:
                r_grad = find_amp_record(records, "grad_flip_rate",                    a, "grad")
                r_rand = find_amp_record(records, "rand_flip_rate",                    a, "rand")
                r_diff = find_amp_record(records, "paired_diff_grad_minus_rand_flip",  a)
                r_mcn  = find_amp_record(records, "mcnemar_grad_vs_rand_flip",         a)

                model_cell = col_label if first_row else ""
                n_cell     = str(entry["n"]) if (first_row and entry["n"]) else ""
                first_row  = False

                lines.append(
                    f"{model_cell} & {fmt_alpha(a)} & "
                    f"{fmt_flip_ci(r_grad)} & "
                    f"{fmt_flip_ci(r_rand)} & "
                    f"{diff_pct(r_diff)} & "
                    f"{fmt_mcnemar(r_mcn)} & "
                    f"{n_cell} \\\\"
                )

            lines.append(r"\midrule")

        if lines[-1] == r"\midrule":
            lines.pop()

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:gradsubspace_amp_stats_" + task + r"}",
            r"\end{table}",
            "",
        ]

    lines.append(
        r"\noindent$^{*}p{<}0.05$,\quad $^{**}p{<}0.01$,\quad "
        r"$^{***}p{<}0.001$ (exact two-sided McNemar on per-instance flip indicators)."
    )
    return "\n".join(lines).strip()


# ─── Amplification heatmap (main text figure) ────────────────────────────────

PINK_STOPS = [
    "#FBEAF0", "#F4C0D1", "#ED93B1",
    "#D4537E", "#993556", "#72243E", "#4B1528",
]
MARKER_X_OFFSET = -0.38


def _setup_rcparams():
    mpl.rcParams["pdf.fonttype"]     = 42
    mpl.rcParams["ps.fonttype"]      = 42
    mpl.rcParams["font.family"]      = "serif"
    mpl.rcParams["font.serif"]       = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"


def _cell_text(v: float) -> str:
    if np.isnan(v):
        return "—"
    if v == 0.0:
        return "0"
    if v >= 99.95:
        return "100"
    return f"{v:.1f}"


def _get_flip(steering_dict: dict, a) -> float:
    """Flexible key lookup covering int / float / str variants."""
    for key in [a, str(a), str(float(a))]:
        val = steering_dict.get(str(key)) or steering_dict.get(key)
        if val is not None:
            return val.get("flip_rate")
    return None


def _assemble_heatmap_panels(amp: dict, display_alphas: list) -> dict:
    """
    Build float matrix [n_alphas x (2 * n_models)] per task.
    Column layout per model pair: [GRAD | Rand].  Values in percent.
    """
    n_alphas = len(display_alphas)
    n_cols   = 2 * len(MODELS)
    panels   = {}
    for task, _ in TASKS:
        mat = np.full((n_alphas, n_cols), np.nan)
        for m_idx, (dir_name, _) in enumerate(MODELS):
            entry = amp[task].get(dir_name)
            if entry is None:
                continue
            raw_grad = entry["raw"].get("grad_amplification", {})
            raw_rand = entry["raw"].get("rand_amplification", {})
            for a_idx, a in enumerate(display_alphas):
                g = _get_flip(raw_grad, a)
                r = _get_flip(raw_rand, a)
                if g is not None:
                    mat[a_idx, 2 * m_idx]     = g * 100.0
                if r is not None:
                    mat[a_idx, 2 * m_idx + 1] = r * 100.0
        panels[task] = mat
    return panels


def _draw_panel(ax, data, title, show_ylabel, alphas, marker_rows_for_task):
    n_rows, n_cols = data.shape
    cmap = LinearSegmentedColormap.from_list("pink_ramp", PINK_STOPS, N=256)
    cmap.set_bad(color="#F2F2F2")
    norm   = mpl.colors.Normalize(vmin=0, vmax=100)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    model_names      = [d for d, _ in MODELS]
    model_labels_map = dict(MODELS)

    for i in range(n_rows):
        for j in range(n_cols):
            v      = data[i, j]
            label  = _cell_text(v)
            tcolor = ("white" if (not np.isnan(v) and v > 55)
                      else ("#888888" if np.isnan(v) else "#3A0E1F"))
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=6.0, color=tcolor)

            # Marker glyph: column j belongs to model j // 2
            model_key = model_names[j // 2]
            glyph = marker_rows_for_task.get(model_key, {}).get(i)
            if glyph is not None:
                gsize = 9.0 if "circ" in glyph else 11.0
                ax.text(j + MARKER_X_OFFSET, i, glyph,
                        ha="center", va="center", fontsize=gsize, color=tcolor)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([fmt_alpha(a) for a in alphas], fontsize=7)
    if show_ylabel:
        ax.set_ylabel(r"$\alpha$", fontsize=9, rotation=0, labelpad=8, va="center")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["GRAD", "Rand"] * len(MODELS), fontsize=6.5)
    ax.tick_params(axis="both", length=0, pad=2)

    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([2 * k + 0.5 for k in range(len(MODELS))])
    ax2.set_xticklabels([model_labels_map[d] for d, _ in MODELS], fontsize=8)
    ax2.tick_params(axis="x", length=0, pad=3)

    for k in range(1, len(MODELS)):
        ax.axvline(2 * k - 0.5, color="white", lw=1.2)

    ax.set_title(title, fontsize=9, color="#888888", loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.4)
    return im


def build_amplification_heatmap(amp: dict, out_pdf: Path, out_png: Path):
    _setup_rcparams()

    # Union of display alphas (alpha != 1.0) across all loaded models.
    all_alphas = set()
    for task, _ in TASKS:
        for dir_name, _ in MODELS:
            entry = amp[task].get(dir_name)
            if entry:
                all_alphas.update(a for a in entry["alphas"] if abs(a - 1.0) > 1e-6)
    display_alphas = sorted(all_alphas)

    medians = {
        task: {dir_name: compute_pooled_median_norm(task, dir_name)
               for dir_name, _ in MODELS}
        for task, _ in TASKS
    }
    marker_rows = compute_marker_rows(display_alphas, medians)
    panels      = _assemble_heatmap_panels(amp, display_alphas)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.4),
        gridspec_kw={"wspace": 0.18},
    )
    im_last = None
    for idx, (task, task_label) in enumerate(TASKS):
        im_last = _draw_panel(
            axes[idx], panels[task], task_label,
            show_ylabel=(idx == 0),
            alphas=display_alphas,
            marker_rows_for_task=marker_rows[task],
        )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22)

    cbar_ax = fig.add_axes([0.20, 0.07, 0.30, 0.025])
    cb = fig.colorbar(im_last, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=7, length=0, pad=2)
    cb.outline.set_linewidth(0.4)
    fig.text(0.18, 0.082, "Flip rate (%)", ha="right", va="center",
             fontsize=8, color="#444444")

    fig.text(0.56, 0.082, r"$\cdot$  first $\alpha$ with $r>0.1$",
             ha="left", va="center", fontsize=8, color="#444444")
    fig.text(0.78, 0.082, r"$\circ$  first $\alpha$ with $r>5$",
             ha="left", va="center", fontsize=8, color="#444444")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[OK] Heatmap PDF -> {out_pdf.resolve()}")
    print(f"[OK] Heatmap PNG -> {out_png.resolve()}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path,
        default=GRAD_SUBSPACE / "tables",
        help="Output directory for all artefacts (default: outputs/grad_subspace/tables)",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print every path probed and loaded values, then exit.",
    )
    args = ap.parse_args()

    abl_data = collect_ablation(debug=args.debug)
    amp_data = collect_amplification(debug=args.debug)

    if args.debug:
        print("\n-- Ablation data summary --")
        for task, models in abl_data.items():
            for m, e in models.items():
                print(f"  {task}/{m}: {e}")
        print("\n-- Amplification data summary --")
        for task, models in amp_data.items():
            for m, e in models.items():
                n_alphas = len(e["alphas"]) if e else 0
                print(f"  {task}/{m}: {'None' if e is None else f'{n_alphas} alphas'}")
        return

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Main text
    abl_main = build_ablation_main_table(abl_data)
    p = out / "tab_gradsubspace_main.tex"
    p.write_text(abl_main)
    print(f"[OK] Ablation main table       -> {p.resolve()}")

    build_amplification_heatmap(
        amp_data,
        out / "fig_amplification_heatmap.pdf",
        out / "fig_amplification_heatmap.png",
    )

    # Appendix
    abl_stats = build_ablation_stats_table(abl_data)
    p = out / "tab_gradsubspace_ablation_stats.tex"
    p.write_text(abl_stats)
    print(f"[OK] Ablation stats table      -> {p.resolve()}")

    amp_stats = build_amp_stats_table(amp_data)
    p = out / "tab_gradsubspace_amp_stats.tex"
    p.write_text(amp_stats)
    print(f"[OK] Amplification stats table -> {p.resolve()}")

    sep = "=" * 70
    print(f"\n{sep}\nABLATION MAIN TABLE\n{sep}\n{abl_main}")
    print(f"\n{sep}\nABLATION STATS TABLE\n{sep}\n{abl_stats}")
    print(f"\n{sep}\nAMPLIFICATION STATS TABLE\n{sep}\n{amp_stats}")


if __name__ == "__main__":
    main()