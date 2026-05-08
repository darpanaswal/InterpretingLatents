"""
gradient_subspace_tables.py
============================

Reads artefacts produced by gradient_subspace_interventions.py and emits:

  Main-text figure
  ----------------
  1. fig_amplification_lineplot.{pdf,png}
       Four-panel lineplot of flip rate vs alpha, styled like
       plot_superposition.py: dotted leader lines + per-model
       "point ± half-CI" endpoint annotations.
       Layout: (Graph-Hopping | Arithmetic-Reasoning) x (GRAD | Rand)
       One line per model {Pause, Coconut, Coconut_u, CODI}.

  Appendix tables
  ---------------
  2. tab_gradsubspace_ablation_stats.tex
       Per-(task, model) bootstrap CIs + paired diffs + McNemar
       for the ablation phase.
  3. tab_gradsubspace_amp_stats.tex
       Per-(task, model, alpha) bootstrap flip-rate CIs + paired
       diffs + McNemar for the amplification phase.

Source artefacts (per task/model under outputs/grad_subspace/):
  - ablation_results.json         (summary point estimates + CIs + McNemar p)
  - amplification_results.json    (alphas, grad/rand flip rates, amp_cis)
  - bootstrap_cis.jsonl           (full CI records with _context tags)

Usage
-----
    python gradient_subspace_intervention_tables.py

    # Dry-run: print every path probed and what was loaded.
    python gradient_subspace_intervention_tables.py --debug

    # Override output directory:
    python gradient_subspace_intervention_tables.py --out-dir results/grad_subspace
"""

import json
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUTS


# ── Canonical layout ────────────────────────────────────────────────────────

GRAD_SUBSPACE = OUTPUTS / "grad_subspace"

# (dir name on disk, LaTeX column label, plot label)
MODELS = [
    ("pause",      "PaT",       "PaT"),
    ("coconut",    "C",         "C"),
    ("coconut_u",  r"C$_u$",    r"$C_u$"),
    ("codi",       "CODI",      "CODI"),
]

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm",    "Arithmetic-Reasoning"),
]

# Per-model line/marker style (mirrors plot_superposition.py for consistency).
MODEL_STYLE = {
    "pause":     {"color": "#0072B2", "marker": "o", "ls": "-"},
    "coconut":   {"color": "#D55E00", "marker": "s", "ls": "-"},
    "coconut_u": {"color": "#009E73", "marker": "D", "ls": "-"},
    "codi":      {"color": "#CC79A7", "marker": "v", "ls": "-."},
}


# ── I/O helpers ─────────────────────────────────────────────────────────────

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
        phases = sorted({r.get("metric") for r in records if r.get("metric")})
        print(f"         -> {len(records)} records; metrics: {phases}")
    return records


# ── Record lookup ───────────────────────────────────────────────────────────

def find_record(records: list, metric: str, context_match: dict = None):
    """Return first record whose metric matches and (optional) _context fields agree."""
    context_match = context_match or {}
    for r in records:
        if r.get("metric") != metric:
            continue
        ctx = r.get("_context", {})
        if all(ctx.get(k) == v for k, v in context_match.items()):
            return r
    return None


# ── Formatting ──────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=1) -> str:
    # 0-1 float -> "%.1f" string in percent.
    return f"{v * 100:.{decimals}f}" if v is not None else "--"


def ci_pct(r) -> str:
    """BootstrapResult dict -> 'point [lo, hi]' as percentages."""
    if r is None:
        return "--"
    p, lo, hi = r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def diff_pct(r) -> str:
    """Signed paired-diff CI as percentages."""
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


def fmt_alpha(a) -> str:
    return str(int(a)) if float(a) == int(a) else str(a)


# ── Endpoint annotation helpers (lifted from plot_superposition.py) ─────────
#
# Math:
#   point      = r["point"]                                    in [0, 1]
#   half_width = (r["ci_high"] - r["ci_low"]) / 2              in [0, 1]
#   We display both *100 so the panel y-axis reads as a percentage.

def _ci_half_width_pct(r):
    if r is None:
        return None
    return (r["ci_high"] - r["ci_low"]) * 100 / 2


def _fmt_endpoint_pct(point_pct, ci_record, decimals=1):
    """'point ± half_CI', percentage units. Falls back to point only if no CI."""
    half = _ci_half_width_pct(ci_record)
    if half is None:
        return f"{point_pct:.{decimals}f}"
    return f"{point_pct:.{decimals}f}±{half:.{decimals}f}"


# ── Data collection ─────────────────────────────────────────────────────────

def collect_ablation(debug: bool = False) -> dict:
    """
    For each (task, model) load ablation_results.json and bootstrap_cis.jsonl.
    Returns: data[task][dir_name] = entry dict or None.

    Note: ablation_results.json (written by gradient_subspace_interventions.py)
    contains the point estimates and the CIs as [low, high] lists; the McNemar
    p-values are present but b/c counts live only in the JSONL.
    """
    data = {}
    for task, task_label in TASKS:
        data[task] = {}
        if debug:
            print(f"\n-- ABLATION  {task_label} ({task}) --")
        for dir_name, _, _ in MODELS:
            base = GRAD_SUBSPACE / task / dir_name
            if debug:
                print(f"  model={dir_name}  base={base}")

            summary = load_json(base / "ablation_results.json", debug)
            records = load_jsonl(base / "bootstrap_cis.jsonl", debug)

            if summary is None:
                data[task][dir_name] = None
                continue

            # Pull richer CI records (with point + ci) directly from JSONL so
            # we can present them uniformly via ci_pct / diff_pct / fmt_mcnemar.
            ci_orig = find_record(records, "baseline_accuracy",
                                  {"condition": "baseline"})
            ci_grad = find_record(records, "grad_ablation_accuracy",
                                  {"condition": "grad_ablation"})
            # Rand: prefer the pooled record; fall back to any rand_ablation.
            ci_rand = (find_record(records, "rand_ablation_accuracy",
                                   {"condition": "rand_ablation_pooled"})
                       or find_record(records, "rand_ablation_accuracy"))
            diff_grad = find_record(records, "accuracy_drop_grad",
                                    {"condition": "baseline_minus_grad"})
            diff_rand = find_record(records, "accuracy_drop_rand",
                                    {"condition": "baseline_minus_rand"})
            mc_grad   = find_record(records, "mcnemar_baseline_vs_grad")
            mc_rand   = find_record(records, "mcnemar_baseline_vs_rand")

            data[task][dir_name] = {
                "n":         summary.get("n_instances"),
                "orig":      summary.get("baseline_accuracy"),
                "grad":      summary.get("grad_accuracy"),
                "rand":      summary.get("rand_accuracy"),
                "ci_orig":   ci_orig,
                "ci_grad":   ci_grad,
                "ci_rand":   ci_rand,
                "diff_grad": diff_grad,
                "diff_rand": diff_rand,
                "mc_grad":   mc_grad,
                "mc_rand":   mc_rand,
            }
    return data


def collect_amplification(debug: bool = False) -> dict:
    """
    For each (task, model) load amplification_results.json and bootstrap_cis.jsonl.
    Returns: amp[task][dir_name] = {"alphas", "raw", "records", "n"} or None.

    amp_results.json layout (written by gradient_subspace_interventions.py):
      - alphas:                  list[float]
      - grad_amplification:      dict[alpha_str -> {"flip_rate", "flipped_indices", ...}]
      - rand_flip_pooled_rates:  dict[alpha_str -> float]      (already pooled)
      - amplification_cis:       dict[alpha_str -> {"grad_ci", "rand_ci",
                                                    "diff_ci", "diff_point",
                                                    "mcnemar_p", ...}]
    """
    amp = {}
    for task, task_label in TASKS:
        amp[task] = {}
        if debug:
            print(f"\n-- AMPLIFICATION  {task_label} ({task}) --")
        for dir_name, _, _ in MODELS:
            base = GRAD_SUBSPACE / task / dir_name
            summary = load_json(base / "amplification_results.json", debug)
            records = load_jsonl(base / "bootstrap_cis.jsonl", debug)
            if summary is None:
                amp[task][dir_name] = None
                continue

            n_inst = (
                summary.get("n_instances")
                or len(summary.get("grad_amplification", {})
                              .get(str(summary.get("alphas", [None])[0]), {})
                              .get("flipped_indices", []))
                or None
            )

            amp[task][dir_name] = {
                "alphas":  [float(a) for a in summary.get("alphas", [])],
                "raw":     summary,
                "records": records,
                "n":       n_inst,
            }
    return amp


# ── Flexible-key lookup for alpha-indexed dicts ─────────────────────────────
#
# amp_results.json encodes alpha keys as strings like "1.0", "2.0", ... but
# the python loader may surface them as either str or float depending on path.

def _lookup_alpha(d: dict, alpha):
    if d is None:
        return None
    for key in (alpha, str(alpha), str(float(alpha)), str(int(alpha)) if float(alpha) == int(alpha) else None):
        if key is None:
            continue
        if key in d:
            return d[key]
    return None


def get_grad_flip_rate(raw: dict, alpha) -> float:
    block = _lookup_alpha(raw.get("grad_amplification", {}), alpha)
    return None if block is None else block.get("flip_rate")


def get_rand_flip_rate(raw: dict, alpha) -> float:
    """Pooled rand flip rate (already aggregated across K random seeds)."""
    return _lookup_alpha(raw.get("rand_flip_pooled_rates", {}), alpha)


def get_amp_ci(raw: dict, alpha):
    """
    Returns three CI-ish records reconstructed in BootstrapResult shape so the
    table formatters (ci_pct / diff_pct) can consume them uniformly:
      grad_ci_rec, rand_ci_rec, diff_rec, mcnemar_rec
    """
    cis = _lookup_alpha(raw.get("amplification_cis", {}), alpha)
    if cis is None:
        return None, None, None, None

    grad_pt = get_grad_flip_rate(raw, alpha)
    rand_pt = get_rand_flip_rate(raw, alpha)
    g_lo, g_hi = cis.get("grad_ci", [None, None])
    r_lo, r_hi = cis.get("rand_ci", [None, None])
    d_lo, d_hi = cis.get("diff_ci", [None, None])

    grad_rec = ({"point": grad_pt, "ci_low": g_lo, "ci_high": g_hi}
                if None not in (grad_pt, g_lo, g_hi) else None)
    rand_rec = ({"point": rand_pt, "ci_low": r_lo, "ci_high": r_hi}
                if None not in (rand_pt, r_lo, r_hi) else None)
    diff_rec = ({"point": cis.get("diff_point"), "ci_low": d_lo, "ci_high": d_hi}
                if None not in (cis.get("diff_point"), d_lo, d_hi) else None)
    mcn_rec = ({"p_value": cis.get("mcnemar_p"),
                "b_a_only": cis.get("mcnemar_b", "?"),
                "c_b_only": cis.get("mcnemar_c", "?")}
               if cis.get("mcnemar_p") is not None else None)
    return grad_rec, rand_rec, diff_rec, mcn_rec


# Use JSONL records too — they carry b/c counts that amp_results.json drops.
def get_amp_jsonl_records(records: list, alpha):
    """Pull the per-alpha JSONL CI records that include McNemar b/c counts."""
    metric_grad = f"flip_rate_grad_a{alpha}"
    metric_rand = f"flip_rate_rand_a{alpha}"
    metric_diff = f"flip_diff_grad_minus_rand_a{alpha}"
    metric_mcn  = f"mcnemar_grad_vs_rand_a{alpha}"

    g = find_record(records, metric_grad, {"condition": "grad_amp"})
    # Pooled rand wins if present, else first match.
    r = (find_record(records, metric_rand, {"condition": "rand_amp_pooled"})
         or find_record(records, metric_rand))
    d = find_record(records, metric_diff, {"condition": "grad_minus_rand_amp"})
    m = find_record(records, metric_mcn)
    return g, r, d, m


# ────────────────────────────────────────────────────────────────────────────
# Lineplot (replaces the heatmap)
# ────────────────────────────────────────────────────────────────────────────

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _panel_ylim(panel_data):
    """panel_data: list of (xs, ys_pct) per model. Returns (ymin, ymax)."""
    vals = [y for _, ys in panel_data for y in ys if y is not None and not math.isnan(y)]
    if not vals:
        return (0, 100)
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    # Extra headroom on top so staggered ± endpoint labels don't collide.
    return (max(-2, lo - 0.08 * span), min(108, hi + 0.32 * span))


def _draw_panel(ax, panel_data_per_model, panel_title, alphas_grid,
                show_ylabel: bool):
    """
    panel_data_per_model: dict[dir_name -> {
        "xs": list[float],            # alphas where flip rate is available
        "ys": list[float],            # flip rates in PERCENT
        "ci_recs": list[dict|None],   # one CI record per (xs, ys) point
    }]
    alphas_grid: union of alphas across models, used for x ticks.
    """

    # Compute y-limits from all models in this panel.
    flat = [(d["xs"], d["ys"]) for d in panel_data_per_model.values()]
    ymin, ymax = _panel_ylim(flat)
    y_range = ymax - ymin
    min_gap = 0.13 * y_range
    bottom_threshold = ymin + 0.08 * y_range

    # ── Plot lines ──────────────────────────────────────────────────────────
    endpoints = []  # (y_at_last_x, dir_name, x_last, color, ci_rec_at_last)
    for dir_name, _, plot_label in MODELS:
        d = panel_data_per_model.get(dir_name)
        if d is None or not d["xs"]:
            continue
        style = MODEL_STYLE[dir_name]
        ax.plot(
            d["xs"], d["ys"],
            color=style["color"],
            marker=style["marker"],
            ls=style["ls"],
            label=plot_label,
            markeredgewidth=0.3,
            markeredgecolor="black",
        )
        endpoints.append(
            (d["ys"][-1], dir_name, d["xs"][-1], style["color"], d["ci_recs"][-1])
        )

    # ── Stagger endpoint labels vertically ──────────────────────────────────
    # Build a cap that always has room for n stacked labels at min_gap apart,
    # even if the cluster sits right at the top of the panel.
    n_endpoints = max(len(endpoints), 1)
    label_cap = max(ymax - 0.03 * y_range,
                    max((y for y, *_ in endpoints), default=ymin)
                    + (n_endpoints - 1) * min_gap + 0.03 * y_range)

    endpoints.sort(key=lambda item: item[0])
    placed = []
    staggered = []
    for item in endpoints:
        y_val = item[0]
        label_y = max(y_val, bottom_threshold)
        for py in placed:
            if abs(label_y - py) < min_gap:
                label_y = py + min_gap
        label_y = min(label_y, label_cap)
        placed.append(label_y)
        staggered.append((*item, label_y))

    # ── Dotted leader lines + ± annotations ─────────────────────────────────
    # X span used for the leader line, in α units.
    if len(alphas_grid) >= 2:
        x_span = alphas_grid[-1] - alphas_grid[0]
    else:
        x_span = 1.0
    line_extend = 0.18 * x_span    # length of leader line in data units
    bend_short  = 0.05 * x_span    # short stub before the bend
    bend_mid    = 0.13 * x_span    # midpoint of bend

    for y_val, _dir, x_val, color, ci_rec, label_y in staggered:
        x_pts = [x_val, x_val + bend_short, x_val + bend_mid, x_val + line_extend]
        y_pts = [y_val, y_val, label_y, label_y]
        ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)),
                linewidth=0.65, alpha=0.75, zorder=0)
        ax.annotate(
            _fmt_endpoint_pct(y_val, ci_rec, decimals=1),
            xy=(x_val + line_extend, label_y),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=5.5,
            color=color,
            weight="bold",
            ha="left",
            va="center",
            clip_on=False,
        )

    # Extend ymax so the stacked label tower is fully inside the axes.
    if staggered:
        max_label_y = max(item[-1] for item in staggered)
        ymax = max(ymax, max_label_y + 0.05 * y_range)
    ax.set_ylim(ymin, ymax)

    # X-axis: extend right so endpoint labels have room; left a touch in.
    if alphas_grid:
        ax.set_xlim(alphas_grid[0] - 0.04 * x_span,
                    alphas_grid[-1] + line_extend + 0.30 * x_span)
        ax.set_xticks(alphas_grid)
        ax.set_xticklabels([fmt_alpha(a) for a in alphas_grid])

    ax.set_xlabel(r"$\alpha$")
    if show_ylabel:
        ax.set_ylabel("Flip rate (%)")
    ax.set_title(panel_title, fontsize=8, pad=6)


def build_amplification_lineplot(amp: dict, out_pdf: Path, out_png: Path):
    """
    Four panels:
      (a) Graph-Hopping  / GRAD     (b) Graph-Hopping  / Rand
      (c) Arithmetic-Reasoning / GRAD   (d) Arithmetic-Reasoning / Rand
    """
    # Union of alphas (excluding the alpha=1.0 identity sanity check).
    all_alphas = set()
    for task, _ in TASKS:
        for dir_name, _, _ in MODELS:
            entry = amp[task].get(dir_name)
            if entry is None:
                continue
            for a in entry["alphas"]:
                if abs(a - 1.0) > 1e-6:
                    all_alphas.add(a)
    alphas_grid = sorted(all_alphas)
    if not alphas_grid:
        print("[WARN] No alphas (other than 1.0) found; skipping lineplot.")
        return

    # Assemble panel data: for each (task, condition), build per-model xs/ys/ci.
    panels = {}      # (task, condition) -> dict[dir_name -> {xs, ys, ci_recs}]
    for task, _ in TASKS:
        for cond in ("grad", "rand"):
            panel = {}
            for dir_name, _, _ in MODELS:
                entry = amp[task].get(dir_name)
                if entry is None:
                    continue
                xs, ys, ci_recs = [], [], []
                for a in alphas_grid:
                    if a not in entry["alphas"]:
                        continue
                    if cond == "grad":
                        rate = get_grad_flip_rate(entry["raw"], a)
                    else:
                        rate = get_rand_flip_rate(entry["raw"], a)
                    if rate is None:
                        continue
                    g_rec, r_rec, _d, _m = get_amp_ci(entry["raw"], a)
                    ci_rec = g_rec if cond == "grad" else r_rec
                    xs.append(a)
                    ys.append(rate * 100.0)
                    ci_recs.append(ci_rec)
                panel[dir_name] = {"xs": xs, "ys": ys, "ci_recs": ci_recs}
            panels[(task, cond)] = panel

    # ── Figure layout: 2 rows (tasks) x 2 cols (conditions) ─────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(9.5, 4.0),
        gridspec_kw={"wspace": 0.42, "hspace": 0.55},
    )

    panel_specs = [
        (axes[0, 0], ("prosqa", "grad"), "(a) Graph-Hopping — GRAD",  True),
        (axes[0, 1], ("prosqa", "rand"), "(b) Graph-Hopping — Rand",  False),
        (axes[1, 0], ("gsm",    "grad"), "(c) Arithmetic-Reasoning — GRAD", True),
        (axes[1, 1], ("gsm",    "rand"), "(d) Arithmetic-Reasoning — Rand", False),
    ]

    for ax, key, title, show_yl in panel_specs:
        _draw_panel(ax, panels[key], title, alphas_grid, show_ylabel=show_yl)

    # Single legend below all four panels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels),
            fontsize=6,
            frameon=True,
            framealpha=0.8,
            edgecolor="#cccccc",
            handlelength=2.0,
            columnspacing=1.2,
            handletextpad=0.5,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Lineplot PDF -> {out_pdf.resolve()}")
    print(f"[OK] Lineplot PNG -> {out_png.resolve()}")


# ────────────────────────────────────────────────────────────────────────────
# Appendix table 1: Ablation statistical tests
# ────────────────────────────────────────────────────────────────────────────

def build_ablation_stats_table(data: dict) -> str:
    lines = [
        r"\subsection{Gradient-Subspace Ablation (Experiment~2, Part~1)}",
        r"\label{app:gradsubspace_ablation_stats}",
        "",
        r"Point estimates and 95\% bootstrap CIs (10{,}000 percentile resamples) "
        r"for accuracy under three conditions: original thought vectors "
        r"(\textsc{Orig}), gradient-subspace ablated (\textsc{Grad}), and "
        r"random-subspace ablated (\textsc{Rand}, pooled across $K$ random "
        r"seeds). "
        r"$\Delta_{\text{grad}} = \text{Acc}_{\text{orig}} - "
        r"\text{Acc}_{\text{grad}}$ "
        r"(positive = gradient ablation hurts). "
        r"McNemar $p$ is exact two-sided binomial on discordant pairs "
        r"($b$: correct only under \textsc{Orig}; $c$: correct only under the "
        r"ablated condition).",
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
            (r"Model "
             r"& Acc$_{\text{orig}}$ [\% CI] "
             r"& Acc$_{\text{grad}}$ [\% CI] "
             r"& Acc$_{\text{rand}}$ [\% CI] "
             r"& $\Delta_{\text{grad}}$ [\% CI] "
             r"& McNemar$_{\text{grad}}$ $p$ "
             r"& $\Delta_{\text{rand}}$ [\% CI] "
             r"& McNemar$_{\text{rand}}$ $p$ "
             r"& $n$ \\"),
            r"\midrule",
        ]

        for dir_name, col_label, _ in MODELS:
            entry = data[task].get(dir_name)
            if entry is None:
                lines.append(
                    f"{col_label} & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
                )
                continue
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


# ────────────────────────────────────────────────────────────────────────────
# Appendix table 2: Amplification statistical tests (full per-α grid)
# ────────────────────────────────────────────────────────────────────────────

def build_amp_stats_table(amp: dict) -> str:
    lines = [
        r"\subsection{Gradient-Subspace Amplification (Experiment~2, Part~2)}",
        r"\label{app:gradsubspace_amp_stats}",
        "",
        r"Per-$\alpha$ flip-rate 95\% bootstrap CIs (10{,}000 resamples). "
        r"A \emph{flip} is recorded when the model's normalised output text "
        r"changes relative to the unintervened baseline. "
        r"$\Delta = \text{flip}_{\text{grad}} - \text{flip}_{\text{rand}}$ "
        r"(positive = the gradient subspace causes more flips than a "
        r"rank-matched random control, pooled across $K$ random seeds). "
        r"McNemar $p$ tests whether \textsc{Grad} and \textsc{Rand} flip the "
        r"\emph{same instances}. "
        r"The $\alpha=1$ identity sanity check is omitted.",
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

        for dir_name, col_label, _ in MODELS:
            entry = amp[task].get(dir_name)
            if entry is None:
                lines.append(f"{col_label} & -- & -- & -- & -- & -- & -- \\\\")
                lines.append(r"\midrule")
                continue

            display_alphas = [a for a in entry["alphas"] if abs(a - 1.0) > 1e-6]
            first_row = True
            for a in display_alphas:
                # Prefer JSONL records (carry McNemar b/c counts).
                g_jl, r_jl, d_jl, m_jl = get_amp_jsonl_records(entry["records"], a)
                # Fall back to amp_results.json reconstructions.
                g_fb, r_fb, d_fb, m_fb = get_amp_ci(entry["raw"], a)

                g = g_jl or g_fb
                r = r_jl or r_fb
                d = d_jl or d_fb
                m = m_jl or m_fb

                model_cell = col_label if first_row else ""
                n_cell     = (str(entry["n"])
                              if (first_row and entry["n"] is not None) else "")
                first_row  = False

                lines.append(
                    f"{model_cell} & {fmt_alpha(a)} & "
                    f"{ci_pct(g)} & "
                    f"{ci_pct(r)} & "
                    f"{diff_pct(d)} & "
                    f"{fmt_mcnemar(m)} & "
                    f"{n_cell} \\\\"
                )
            lines.append(r"\midrule")

        if lines and lines[-1] == r"\midrule":
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
        r"$^{***}p{<}0.001$ (exact two-sided McNemar on per-instance flip "
        r"indicators)."
    )
    return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path,
        default=GRAD_SUBSPACE / "tables",
        help="Output directory for figure + appendix tables "
             "(default: outputs/grad_subspace/tables)",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print every path probed and what was loaded; do not write files.",
    )
    args = ap.parse_args()

    abl_data = collect_ablation(debug=args.debug)
    amp_data = collect_amplification(debug=args.debug)

    if args.debug:
        print("\n-- Ablation summary --")
        for task, models in abl_data.items():
            for m, e in models.items():
                ok = "None" if e is None else (
                    f"orig={e['orig']}, grad={e['grad']}, rand={e['rand']}, n={e['n']}"
                )
                print(f"  {task}/{m}: {ok}")
        print("\n-- Amplification summary --")
        for task, models in amp_data.items():
            for m, e in models.items():
                ok = "None" if e is None else f"{len(e['alphas'])} alphas, n={e['n']}"
                print(f"  {task}/{m}: {ok}")
        return

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Main-text figure.
    build_amplification_lineplot(
        amp_data,
        out / "fig_amplification_lineplot.pdf",
        out / "fig_amplification_lineplot.png",
    )

    # Appendix tables.
    abl_stats = build_ablation_stats_table(abl_data)
    p = out / "tab_gradsubspace_ablation_stats.tex"
    p.write_text(abl_stats)
    print(f"[OK] Ablation appendix table   -> {p.resolve()}")

    amp_stats = build_amp_stats_table(amp_data)
    p = out / "tab_gradsubspace_amp_stats.tex"
    p.write_text(amp_stats)
    print(f"[OK] Amplification appendix    -> {p.resolve()}")

    sep = "=" * 70
    print(f"\n{sep}\nABLATION APPENDIX TABLE\n{sep}\n{abl_stats}")
    print(f"\n{sep}\nAMPLIFICATION APPENDIX TABLE\n{sep}\n{amp_stats}")


if __name__ == "__main__":
    main()