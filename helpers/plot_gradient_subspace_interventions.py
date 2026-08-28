"""
gradient_subspace_tables.py
============================

Reads artefacts produced by gradient_subspace_interventions.py (gold,
gradient of gold-answer NLL) and gradient_subspace_predtoken_interventions.py
(pred, gradient of the model's own predicted-token NLL) and emits, for each
source independently:

  Main-text figure
  ----------------
  1. fig_amplification_lineplot.{pdf,png}
       Four-panel lineplot of flip rate vs alpha, styled like
       plot_superposition.py: dotted leader lines + per-model
       "point ± half-CI" endpoint annotations.
       Layout: (Graph-Hopping | Arithmetic-Reasoning) x (GRAD | Rand)
       One line per model {Pause, Coconut, Coconut_u, CODI}.
       The full alpha sweep is shown, including the low-alpha points:
         alpha=0   ablation (causal component removed) -- labelled
         alpha=0.5 continuity point                    -- unlabelled
         alpha=1   identity / no change                -- unlabelled
       so the line is continuous from ablation through amplification.

  Appendix tables
  ---------------
  2. tab_gradsubspace_stats.tex
       Combined single file containing:
       - Per-(task, model) bootstrap CIs + paired diffs + McNemar
         for the ablation phase.
       - Per-(task, model, alpha) bootstrap flip-rate CIs + paired
         diffs + McNemar for the amplification phase.

Source artefacts (per family/task/model, root differs by source):
  gold: outputs/grad_subspace/<family>/<task>/<model>/
  pred: outputs/grad_subspace_predtoken/<family>/<task>/<model>/
    - ablation_results.json         (summary point estimates + CIs + McNemar p)
    - amplification_results.json    (alphas, grad/rand flip rates, amp_cis)
    - bootstrap_cis.jsonl           (full CI records with _context tags)

Writes, for every model family found on disk for each source:
    Plots/gradient_subspace_interventions/amplification_{family}.pdf            (gold)
    Plots/gradient_subspace_predtoken_interventions/amplification_predtoken_{family}.pdf (pred)
    Tables/statistical/gradient_subspace_interventions_{family}.tex             (gold)
    Tables/statistical/gradient_subspace_predtoken_interventions_{family}.tex   (pred)

A source is skipped entirely if its root directory has no families on disk
(e.g. gradient_subspace_predtoken_interventions.py was never run).

Usage
-----
    python -m helpers.plot_gradient_subspace_interventions

    # Dry-run: print every path probed and what was loaded.
    python -m helpers.plot_gradient_subspace_interventions --debug

    # Override the tables output directory:
    python -m helpers.plot_gradient_subspace_interventions --out_tables_dir results/tables
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

# Subspace sources to discover + plot: gold keeps the legacy paths, pred
# sits alongside in its own namespaced tree so nothing gets overwritten.
SOURCES = [
    {
        "name": "gold",
        "root": OUTPUTS / "grad_subspace",
        "fig_dir": Path("Plots/gradient_subspace_interventions"),
        "fig_name": "amplification_{family}.pdf",
        "table_name": "gradient_subspace_interventions_{family}.tex",
        "relabel": False,
    },
    {
        "name": "pred",
        "root": OUTPUTS / "grad_subspace_predtoken",
        "fig_dir": Path("Plots/gradient_subspace_predtoken_interventions"),
        "fig_name": "amplification_predtoken_{family}.pdf",
        "table_name": "gradient_subspace_predtoken_interventions_{family}.tex",
        "relabel": True,
    },
]

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
    """Return the last (most recent) record whose metric matches and (optional) _context fields agree."""
    context_match = context_match or {}
    for r in reversed(records):
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
    """BootstrapResult dict -> 'point \\pm half_ci' as percentages."""
    if r is None:
        return "--"
    p = r["point"] * 100
    hw = (r["ci_high"] - r["ci_low"]) * 100 / 2
    return rf"${p:.1f} {{\scriptstyle \pm {hw:.1f}}}$"


def diff_pct(r) -> str:
    """Signed paired-diff CI as percentages."""
    if r is None:
        return "--"
    p = r["point"] * 100
    hw = (r["ci_high"] - r["ci_low"]) * 100 / 2
    sign = "+" if p >= 0 else ""
    return rf"${sign}{p:.1f} {{\scriptstyle \pm {hw:.1f}}}$"


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

def collect_ablation(root: Path, family: str = "gpt2", debug: bool = False) -> dict:
    """
    For each (task, model) load ablation_results.json and bootstrap_cis.jsonl.
    Returns: data[task][dir_name] = entry dict or None.

    root is either outputs/grad_subspace (gold) or
    outputs/grad_subspace_predtoken (pred); both are written with the same
    <family>/<task>/<model>/ layout.

    Note: ablation_results.json (written by gradient_subspace_interventions.py)
    contains the point estimates and the CIs as [low, high] lists; the McNemar
    p-values are present but b/c counts live only in the JSONL.
    """
    data = {}
    for task, task_label in TASKS:
        data[task] = {}
        if debug:
            print(f"\n-- ABLATION  {task_label} ({task}) [{family}] --")
        for dir_name, _, _ in MODELS:
            base = root / family / task / dir_name
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
            ci_grad = (find_record(records, "grad_ablation_accuracy",
                                   {"condition": "grad_nullspace"})
                       or find_record(records, "grad_ablation_accuracy"))
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


def collect_amplification(root: Path, family: str = "gpt2", debug: bool = False) -> dict:
    """
    For each (task, model) load amplification_results.json and bootstrap_cis.jsonl.
    Returns: amp[task][dir_name] = {"alphas", "raw", "records", "n"} or None.

    root is either outputs/grad_subspace (gold) or
    outputs/grad_subspace_predtoken (pred); both are written with the same
    <family>/<task>/<model>/ layout.

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
            print(f"\n-- AMPLIFICATION  {task_label} ({task}) [{family}] --")
        for dir_name, _, _ in MODELS:
            base = root / family / task / dir_name
            summary = load_json(base / "amplification_results.json", debug)
            records = load_jsonl(base / "bootstrap_cis.jsonl", debug)
            if summary is None:
                amp[task][dir_name] = None
                continue

            # "n_total" (dataset size) lives per-alpha inside grad_amplification;
            # it's constant across alphas, so any entry gives the true n.
            # (Previously this fell back to len(flipped_indices) -- the count
            # of instances that flipped at the first alpha, not the dataset
            # size -- which is why displayed n fluctuated with alpha/model.)
            grad_amp = summary.get("grad_amplification", {})
            n_inst = summary.get("n_instances")
            if n_inst is None:
                for entry in grad_amp.values():
                    if entry.get("n_total") is not None:
                        n_inst = entry["n_total"]
                        break

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


def _draw_panel(ax, panel_data_per_model, panel_title, alphas_grid, alpha_labels,
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
    min_gap = 0.20 * y_range
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
    line_extend = 0.15 * x_span    # length of leader line in data units
    bend_short  = 0.05 * x_span    # short stub before the bend
    bend_mid    = 0.11 * x_span    # midpoint of bend

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

    # ── Ablation (alpha=0) labels on the LEFT of each line ──────────────────
    # alpha=0 fully removes the causal component, so its flip rate is the
    # gradient-/random-subspace ABLATION flip rate. We label it explicitly
    # (mirroring the right-side amplification endpoint annotations) while
    # leaving the alpha=0.5 and alpha=1 continuity points unlabelled.
    abl_idx = next((i for i, lbl in enumerate(alpha_labels) if lbl == "0"), None)
    abl_endpoints = []  # (y_at_abl, color, ci_rec_at_abl)
    if abl_idx is not None:
        for dir_name, _, _ in MODELS:
            d = panel_data_per_model.get(dir_name)
            if d is None or not d["xs"]:
                continue
            if abl_idx not in d["xs"]:
                continue
            j = d["xs"].index(abl_idx)
            style = MODEL_STYLE[dir_name]
            abl_endpoints.append((d["ys"][j], style["color"], d["ci_recs"][j]))

    # Stagger ablation labels vertically, same scheme as the right endpoints.
    abl_staggered = []
    if abl_endpoints:
        abl_endpoints.sort(key=lambda item: item[0])
        placed_abl = []
        for y_val, color, ci_rec in abl_endpoints:
            label_y = max(y_val, bottom_threshold)
            for py in placed_abl:
                if abs(label_y - py) < min_gap:
                    label_y = py + min_gap
            label_y = min(label_y, label_cap)
            placed_abl.append(label_y)
            abl_staggered.append((y_val, color, ci_rec, label_y))

        # Leader line bends to the LEFT of the alpha=0 marker.
        x0 = float(abl_idx)
        for y_val, color, ci_rec, label_y in abl_staggered:
            x_pts = [x0, x0 - bend_short, x0 - bend_mid, x0 - line_extend]
            y_pts = [y_val, y_val, label_y, label_y]
            ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)),
                    linewidth=0.65, alpha=0.75, zorder=0)
            ax.annotate(
                _fmt_endpoint_pct(y_val, ci_rec, decimals=1),
                xy=(x0 - line_extend, label_y),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                weight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )

    # Extend ymax so the stacked label tower is fully inside the axes.
    label_towers = list(staggered) + [
        (yv, None, None, ly) for (yv, _c, _ci, ly) in abl_staggered
    ]
    if label_towers:
        max_label_y = max(item[-1] for item in label_towers)
        ymax = max(ymax, max_label_y + 0.05 * y_range)
    ax.set_ylim(ymin, ymax)

    # X-axis: extend right for amplification endpoint labels and LEFT for the
    # ablation labels.
    if alphas_grid:
        left_pad = (line_extend + 0.36 * x_span) if abl_staggered else (0.04 * x_span)
        ax.set_xlim(alphas_grid[0] - left_pad,
                    alphas_grid[-1] + line_extend + 0.40 * x_span)
        ax.set_xticks(alphas_grid)
        ax.set_xticklabels(alpha_labels)

    ax.set_xlabel(r"$\alpha$ (amplification constant)")
    if show_ylabel:
        ax.set_ylabel("Flip rate (%)")
    ax.set_title(panel_title, fontsize=8, pad=6)


def build_amplification_lineplot(amp: dict, out_pdf: Path):
    """
    Four panels:
      (a) Graph-Hopping  / GRAD     (b) Graph-Hopping  / Rand
      (c) Arithmetic-Reasoning / GRAD   (d) Arithmetic-Reasoning / Rand
    """
    # Union of all swept alphas, including the low-alpha continuity points:
    #   alpha=0   -> ablation (causal component fully removed)
    #   alpha=0.5 -> continuity point between ablation and identity
    #   alpha=1   -> identity / no-change
    #   alpha>1   -> amplification
    all_alphas = set()
    for task, _ in TASKS:
        for dir_name, _, _ in MODELS:
            entry = amp[task].get(dir_name)
            if entry is None:
                continue
            for a in entry["alphas"]:
                all_alphas.add(a)
    alphas_grid = sorted(all_alphas)
    if not alphas_grid:
        print("[WARN] No alphas found; skipping lineplot.")
        return

    alpha_labels = [fmt_alpha(a) for a in alphas_grid]
    alpha_indices = list(range(len(alphas_grid)))

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
                    xs.append(alphas_grid.index(a))
                    ys.append(rate * 100.0)
                    ci_recs.append(ci_rec)
                panel[dir_name] = {"xs": xs, "ys": ys, "ci_recs": ci_recs}
            panels[(task, cond)] = panel

    # ── Figure layout: 2 rows (tasks) x 2 cols (conditions) ─────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 3.2),
        gridspec_kw={"wspace": 0.15, "hspace": 0.60},
    )

    panel_specs = [
        (axes[0, 0], ("prosqa", "grad"), "(a) Graph-Hopping — GRAD",  True),
        (axes[0, 1], ("prosqa", "rand"), "(b) Graph-Hopping — Rand",  False),
        (axes[1, 0], ("gsm",    "grad"), "(c) Arithmetic-Reasoning — GRAD", True),
        (axes[1, 1], ("gsm",    "rand"), "(d) Arithmetic-Reasoning — Rand", False),
    ]

    for ax, key, title, show_yl in panel_specs:
        _draw_panel(ax, panels[key], title, alpha_indices, alpha_labels, show_ylabel=show_yl)

    # Single legend in the right margin (vertical), reclaiming the vertical
    # space the old below-panel bar consumed.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            ncol=1,
            fontsize=6,
            frameon=True,
            framealpha=0.8,
            edgecolor="#cccccc",
            handlelength=2.0,
            labelspacing=0.8,
            handletextpad=0.5,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Lineplot PDF -> {out_pdf.resolve()}")


# ────────────────────────────────────────────────────────────────────────────
# Appendix table 1: Ablation statistical tests
# ────────────────────────────────────────────────────────────────────────────

FAMILY_LABELS = {"gpt2": "GPT-2", "llama": "Llama-3.2"}

# (dict key, subspace section label)
SUBSPACE_SOURCES = [
    ("gold", "Gold Subspace"),
    ("pred", "Predicted-Token Subspace"),
]

def _ablation_caption(family: str) -> str:
    return (rf"Full gradient-subspace ablation results with statistical "
             rf"testing for the {FAMILY_LABELS.get(family, family)} model family.")

def _amp_caption(family: str) -> str:
    return (rf"Full gradient-subspace amplification results with statistical "
             rf"testing for the {FAMILY_LABELS.get(family, family)} model family.")

def _ablation_body_lines(data: dict) -> list:
    lines = []
    for task, task_label in TASKS:
        lines += [
            r"\midrule",
            rf"\multicolumn{{8}}{{c}}{{\textbf{{{task_label}}}}} \\",
            r"\midrule",
        ]

        for dir_name, col_label, _ in MODELS:
            entry = data[task].get(dir_name)
            if entry is None:
                lines.append(
                    f"{col_label} & -- & -- & -- & -- & -- & -- & -- \\\\"
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
                f"{fmt_mcnemar(entry['mc_rand'])} \\\\"
            )
    return lines

def build_ablation_stats_table(data_by_source: dict, family: str = "") -> str:
    # No "n" column: ablation_results.json carries no instance-count field
    # for this table (unlike the amplification table), so it was always "--".
    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l ccc cc cc}",
        r"\toprule",
        (r"Model "
         r"& Acc$_{\text{orig}}$ [\% CI] "
         r"& Acc$_{\text{grad}}$ [\% CI] "
         r"& Acc$_{\text{rand}}$ [\% CI] "
         r"& $\Delta_{\text{grad}}$ [\% CI] "
         r"& McNemar$_{\text{grad}}$ $p$ "
         r"& $\Delta_{\text{rand}}$ [\% CI] "
         r"& McNemar$_{\text{rand}}$ $p$ \\"),
    ]

    for key, subspace_label in SUBSPACE_SOURCES:
        data = data_by_source.get(key)
        if data is None:
            continue
        lines += [
            r"\midrule",
            rf"\multicolumn{{8}}{{c}}{{\textbf{{{subspace_label}}}}} \\",
        ]
        lines += _ablation_body_lines(data)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_ablation_caption(family)}}}",
        rf"\label{{tab:gradsubspace_ablation_stats_{family}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────────────────────
# Appendix table 2: Amplification statistical tests (full per-α grid)
# ────────────────────────────────────────────────────────────────────────────

def _amp_body_lines(amp: dict) -> list:
    lines = []
    for task, task_label in TASKS:
        lines += [
            r"\midrule",
            rf"\multicolumn{{7}}{{c}}{{\textbf{{{task_label}}}}} \\",
            r"\midrule",
        ]

        for i, (dir_name, col_label, _) in enumerate(MODELS):
            entry = amp[task].get(dir_name)
            display_alphas = sorted(a for a in entry["alphas"] if a >= 1.5) if entry is not None else []

            if not display_alphas:
                lines.append(f"{col_label} & -- & -- & -- & -- & -- & -- \\\\")
                if i < len(MODELS) - 1:
                    lines.append(r"\midrule")
                continue

            first_row = True
            for a in display_alphas:
                row_parts = []

                model_cell = col_label if first_row else ""
                row_parts.append(model_cell)
                row_parts.append(fmt_alpha(a))

                g_jl, r_jl, d_jl, m_jl = get_amp_jsonl_records(entry["records"], a)
                g_fb, r_fb, d_fb, m_fb = get_amp_ci(entry["raw"], a)

                g = g_jl or g_fb
                r = r_jl or r_fb
                d = d_jl or d_fb
                m = m_jl or m_fb

                n_val = entry["n"] if first_row else ""

                row_parts.append(ci_pct(g))
                row_parts.append(ci_pct(r))
                row_parts.append(diff_pct(d))
                row_parts.append(fmt_mcnemar(m))
                row_parts.append(str(n_val))

                lines.append(" & ".join(row_parts) + r" \\")
                first_row = False

            if i < len(MODELS) - 1:
                lines.append(r"\midrule")
    return lines

def build_amp_stats_tables(amp_by_source: dict, family: str = "") -> list:
    """One \\begin{table*}...\\end{table*} per available subspace source.

    The merged (gold+pred) amplification table is too tall to fit a page
    even at \\tiny (4 models x 7 alphas x 2 tasks x 2 subspaces), so each
    subspace gets its own table block instead of being folded into rows."""
    tag_by_key = {"gold": "gold", "pred": "predtoken"}
    word_by_key = {"gold": "Gold", "pred": "Predicted-token"}

    tables = []
    for key, _ in SUBSPACE_SOURCES:
        amp = amp_by_source.get(key)
        if amp is None:
            continue
        lines = [
            r"\begin{table*}[h!]",
            r"\centering",
            r"\tiny",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l l cccc c}",
            r"\toprule",
            (r"Model & $\alpha$ "
             r"& Flip$_{\text{grad}}$ [\% CI] "
             r"& Flip$_{\text{rand}}$ [\% CI] "
             r"& $\Delta$ [\% CI] "
             r"& McNemar $p$ "
             r"& $n$ \\"),
        ]
        lines += _amp_body_lines(amp)
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{_amp_caption(family)} ({word_by_key[key]} subspace.)}}",
            rf"\label{{tab:gradsubspace_amp_stats_{tag_by_key[key]}_{family}}}",
            r"\end{table*}",
        ]
        tables.append("\n".join(lines).strip())
    return tables


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def discover_families(root: Path) -> list:
    """Families present on disk under root/<family>/."""
    found = set()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name != "figures":
                found.add(child.name)
    known = [f for f in ("gpt2", "llama") if f in found]
    extra = sorted(found - set(known))
    return known + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_tables_dir", type=Path,
        default=Path("Tables/statistical"),
        help="Output directory for appendix tables (default: Tables/statistical/)",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print every path probed and what was loaded; do not write files.",
    )
    args = ap.parse_args()

    out_tables = args.out_tables_dir
    out_tables.mkdir(parents=True, exist_ok=True)

    # Union of families found across both sources.
    families_by_source = {}
    for source in SOURCES:
        families = discover_families(source["root"])
        families_by_source[source["name"]] = families
        if families:
            print(f"[INFO] ({source['name']}) Families: {families}")
        else:
            print(f"[skip] {source['name']}: no families found under {source['root']}")

    all_families = sorted(set().union(*families_by_source.values()))
    if not all_families:
        raise SystemExit(
            f"No families found under {SOURCES[0]['root']} or {SOURCES[1]['root']}."
        )

    for family in all_families:
        abl_by_source = {}
        amp_by_source = {}

        for source in SOURCES:
            if family not in families_by_source[source["name"]]:
                continue
            root = source["root"]
            abl_data = collect_ablation(root, family=family, debug=args.debug)
            amp_data = collect_amplification(root, family=family, debug=args.debug)

            if args.debug:
                print(f"\n-- ({source['name']}) Ablation summary [{family}] --")
                for task, models in abl_data.items():
                    for m, e in models.items():
                        ok = "None" if e is None else (
                            f"orig={e['orig']}, grad={e['grad']}, rand={e['rand']}, n={e['n']}"
                        )
                        print(f"  {task}/{m}: {ok}")
                print(f"\n-- ({source['name']}) Amplification summary [{family}] --")
                for task, models in amp_data.items():
                    for m, e in models.items():
                        ok = "None" if e is None else f"{len(e['alphas'])} alphas, n={e['n']}"
                        print(f"  {task}/{m}: {ok}")
                continue

            abl_by_source[source["name"]] = abl_data
            amp_by_source[source["name"]] = amp_data

            # Main-text figure (namespaced by family and source; unchanged, one per source).
            fig_dir = source["fig_dir"]
            fig_dir.mkdir(parents=True, exist_ok=True)
            out_fig = fig_dir / source["fig_name"].format(family=family)
            build_amplification_lineplot(amp_data, out_fig)

        if args.debug:
            continue

        # Appendix tables — one merged file per family. The (short) ablation
        # table folds gold subspace rows on top of predicted-token subspace
        # rows; the amplification table is too tall for that fold to fit a
        # page even at \tiny, so it gets one table* block per subspace instead.
        abl_stats = build_ablation_stats_table(abl_by_source, family=family)
        amp_tables = build_amp_stats_tables(amp_by_source, family=family)
        combined_stats = "\n\n".join([abl_stats] + amp_tables)
        p = out_tables / f"gradient_subspace_interventions_{family}.tex"
        p.write_text(combined_stats)
        print(f"[OK] Appendix tables -> {p.resolve()}")


if __name__ == "__main__":
    main()