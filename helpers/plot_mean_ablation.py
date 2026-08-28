"""
plot_mean_ablation.py
=====================

Reads output artefacts from mean_ablation.py and emits:

  1. Main-text bar figure (mean_ablation.pdf)
       - Two panels: Graph-Hopping (left), Arithmetic-Reasoning (right)
       - x-axis slots: Orig | Ablate(μt, μ̂i, r) | Isolate(μt, μ̂i, r) | [ε_rand]
       - Grouped bars: one bar per model per slot; 95% bootstrap CIs as
         asymmetric error bars
       - Style matches plot_superposition.py / plot_logit_lens.py

  2. Appendix statistical-test table (tab_mean_ablation_stats.tex)
       - Full bootstrap CIs for all 6 decomposition interventions
       - Both tasks side-by-side in a single table
       - Paired Δ vs. original and McNemar p (if "all"-run data exist)
       - Subsection-formatted, ready to \input into the appendix

Data sources
------------
Core interventions  outputs/variance_matrix/{family}/{task}/{model}/summary_core.json
                                                                    cis_core.jsonl
Baseline ("Orig.")  outputs/remove_thoughts/{task}/{family}/{model}/removeThoughts.json
                                                                    removeThoughts_cis.jsonl
Random control      outputs/variance_matrix/{family}/{task}/{model}/summary_all.json  (optional)
                                                                    cis_all.jsonl     (optional)
Paired Δ / McNemar  outputs/variance_matrix/{family}/{task}/{model}/cis_all.jsonl     (optional)

Note: the two writers use different component orders (family/task/model vs
task/family/model) — collect() preserves each writer's actual layout.

Usage
-----
    python helpers/mean_ablation_tables.py \
        --out_fig outputs/variance_matrix/mean_ablation.pdf \
        --out_appendix tables/tab_mean_ablation_stats.tex

    # Debug mode (print every path probed, then exit):
    python helpers/mean_ablation_tables.py --debug
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import BASE_DIR, THOUGHT_ABLATION

# ── Shared figure style (matches plot_superposition.py / plot_logit_lens.py) ──

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "lines.linewidth":   1.0,
    "lines.markersize":  6,
    "text.usetex":       False,
})

MODEL_STYLE = {
    "pause":     {"color": "#0072B2", "marker": "o",  "ls": "-",   "label": "PaT"},
    "coconut":   {"color": "#D55E00", "marker": "s",  "ls": "-",   "label": "C"},
    "coconut_u": {"color": "#009E73", "marker": "D",  "ls": "-",   "label": r"C$_u$"},
    "codi":      {"color": "#CC79A7", "marker": "v",  "ls": "-.",  "label": "CODI"},
}

# Grouped bar layout (see plot_mean_ablation for slot constants):
#   slot 0       = Orig.
#   slots 1,2,3  = Ablate  μ_t / μ̂_i / r
#   slots 4,5,6  = Isolate μ_t / μ̂_i / r
#   slot 7       = ε_rand ctrl. (only when random_matched_norm data exist)
ABLATE_KEYS  = ["timestep_ablation",  "instance_ablation",  "residual_ablation"]
ISOLATE_KEYS = ["timestep_isolation", "instance_isolation", "residual_isolation"]

# ── Canonical model / task order ─────────────────────────────────────────────

MODELS = [
    ("pause",     "PaT"),
    ("coconut",   "C"),
    ("coconut_u", r"C$_u$"),
    ("codi",      "CODI"),
]

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm",    "Arithmetic-Reasoning"),
]

# Base model families. Both are looped; the figure is faceted by family (rows).
FAMILIES = [
    ("gpt2",  "GPT-2"),
    ("llama", "Llama"),
]

TASK_MAKECELL = {
    "prosqa": r"\makecell[l]{Graph-\\Hopping}",
    "gsm":    r"\makecell[l]{Arithmetic-\\Reasoning}",
}

VARIANCE_MATRIX = BASE_DIR / "outputs" / "variance_matrix"

# 6 core decomposition interventions (excludes baseline + random control).
# Each entry: (summary_key, ci_metric_name, short_col_label, long_display_name)
INTERVENTIONS = [
    ("timestep_ablation",  "acc_timestep_ablation",  r"$\mu_t$",        r"Ablate $\mu_t$"),
    ("instance_ablation",  "acc_instance_ablation",  r"$\mu_i$",        r"Ablate $\mu_i$"),
    ("residual_ablation",  "acc_residual_ablation",  r"$\varepsilon$",  r"Ablate $\varepsilon$"),
    ("timestep_isolation", "acc_timestep_isolation", r"$\mu_t$",        r"Isolate $\mu_t$"),
    ("instance_isolation", "acc_instance_isolation", r"$\mu_i$",        r"Isolate $\mu_i$"),
    ("residual_isolation", "acc_residual_isolation", r"$\varepsilon$",  r"Isolate $\varepsilon$"),
]

# ── I/O helpers ──────────────────────────────────────────────────────────────

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
        print(f"         -> {len(records)} records")
    return records


def find_record(records: list, metric: str):
    """Return the last record whose 'metric' field equals `metric` (most recent run)."""
    match = None
    for r in records:
        if r.get("metric") == metric:
            match = r
    return match


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=1) -> str:
    return f"{v * 100:.{decimals}f}" if v is not None else "--"


def pm_pct(r, decimals=1):
    """
    BootstrapResult dict →  '$X.X {\scriptstyle \pm Y.Y}$'
    Returns None when record is missing so the caller can fall back.
    """
    if r is None:
        return None
    p  = r["point"] * 100
    hw = (r["ci_high"] - r["ci_low"]) * 100 / 2
    return rf"${p:.{decimals}f} {{\scriptstyle \pm {hw:.{decimals}f}}}$"


def ci_pct(r) -> str:
    """BootstrapResult dict → 'X.X [lo, hi]' as percentages."""
    if r is None:
        return "--"
    p  = r["point"] * 100
    lo = r["ci_low"]  * 100
    hi = r["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def diff_pct(r) -> str:
    """Signed paired-diff CI as percentages."""
    if r is None:
        return "--"
    p  = r["point"] * 100
    lo = r["ci_low"]  * 100
    hi = r["ci_high"] * 100
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
        r"$^{*}$"   if pv < 0.05  else
        ""
    )
    return f"{pv:.4f}{stars} (b={b}, c={c})"


# ── Data collection ───────────────────────────────────────────────────────────

def collect(debug: bool = False) -> dict:
    """
    Returns:
        data[family][task][model] = { ... }  # see body
        or None when no data is available for that (family, task, model) triple.

    Path layout (the two writers disagree on component order — preserved):
        variance_matrix : {family}/{task}/{model}   (mean_ablation.py)
        remove_thoughts : {task}/{family}/{model}    (remove_thoughts.py)
    """
    data = {}

    for family, family_label in FAMILIES:
        data[family] = {}

        if debug:
            print(f"\n{'#'*60}\n# FAMILY: {family_label} ({family})\n{'#'*60}")

        for task, task_label in TASKS:
            data[family][task] = {}

            if debug:
                print(f"\n{'='*60}\n{family_label} / {task_label} ({family}/{task})\n{'='*60}")

            for dir_name, _ in MODELS:
                vm_base = VARIANCE_MATRIX  / family / task / dir_name
                rt_base = THOUGHT_ABLATION / task / family / dir_name

                if debug:
                    print(f"\n  model={dir_name}")

                summary_core = load_json(vm_base / "summary_core.json", debug)
                ci_core      = load_jsonl(vm_base / "cis_core.jsonl",   debug)

                summary_rt   = load_json(rt_base / "removeThoughts.json",      debug)
                ci_rt        = load_jsonl(rt_base / "removeThoughts_cis.jsonl", debug)

                # Optional "all"-run artefacts (baseline + random control + paired tests).
                summary_all  = load_json(vm_base / "summary_all.json", debug)
                ci_all       = load_jsonl(vm_base / "cis_all.jsonl",   debug)

                # Runs invoked with `--intervention all` write everything into
                # summary_all.json / cis_all.jsonl instead of *_core artefacts
                # (mean_ablation.py keys output files by args.intervention). The
                # 6 core point estimates live as top-level keys in summary_all
                # (results = {..., **results_acc}) and the acc_* CI records live
                # in cis_all.jsonl. Fall back to those when *_core is absent.
                core_summary = summary_core if summary_core is not None else summary_all
                core_cis     = ci_core      if ci_core            else ci_all

                if core_summary is None and summary_rt is None:
                    data[family][task][dir_name] = None
                    continue

                # --- Baseline ("Orig.") from remove_thoughts output -----------
                orig_point = summary_rt.get("unquantized_accuracy") if summary_rt else None
                n_th       = summary_rt.get("n_thoughts", 6)        if summary_rt else 6
                ci_orig    = find_record(ci_rt, f"accuracy_K{n_th}")

                n = (summary_rt.get("n_instances") if summary_rt
                     else core_summary.get("n_instances") if core_summary
                     else None)

                # --- Random matched-norm control (from "all" run, optional) ---
                random_point = summary_all.get("random_matched_norm") if summary_all else None
                ci_random    = find_record(ci_all, "acc_random_matched_norm")

                # --- 6 core decomposition interventions -----------------------
                interventions = {}
                diffs         = {}

                for sum_key, ci_metric, _, _ in INTERVENTIONS:
                    point = core_summary.get(sum_key) if core_summary else None
                    ci    = find_record(core_cis, ci_metric)
                    interventions[sum_key] = {"point": point, "ci": ci}

                    diff_ci = find_record(ci_all, f"diff_baseline_vs_{sum_key}")
                    mc      = find_record(ci_all, f"mcnemar_baseline_vs_{sum_key}")
                    diffs[sum_key] = {"diff_ci": diff_ci, "mc": mc}

                rand_diff_ci = find_record(ci_all, "diff_baseline_vs_random_matched_norm")
                rand_mc      = find_record(ci_all, "mcnemar_baseline_vs_random_matched_norm")

                data[family][task][dir_name] = {
                    "orig":          orig_point,
                    "ci_orig":       ci_orig,
                    "random":        random_point,
                    "ci_random":     ci_random,
                    "rand_diff_ci":  rand_diff_ci,
                    "rand_mc":       rand_mc,
                    "n":             n,
                    "interventions": interventions,
                    "diffs":         diffs,
                }

    return data


# ── Helpers for table cells ───────────────────────────────────────────────────

def _cell_orig(entry) -> str:
    if entry is None:
        return "--"
    return pm_pct(entry.get("ci_orig")) or fmt_pct(entry.get("orig"))


def _cell_random(entry) -> str:
    if entry is None:
        return "--"
    return pm_pct(entry.get("ci_random")) or fmt_pct(entry.get("random"))


def _cell_iv(entry, sum_key) -> str:
    if entry is None:
        return "--"
    iv = entry["interventions"].get(sum_key, {})
    return pm_pct(iv.get("ci")) or fmt_pct(iv.get("point"))


# ── Table 1: main performance table ──────────────────────────────────────────

def build_main_table(family_data: dict) -> str:
    header = r"""\begin{table}[!h]
\centering
\tiny
\setlength{\tabcolsep}{3pt}
\caption{Variance-decomposition interventions across model--task pairs.
Ablations remove one mean component; isolations keep only one.
Matched-norm random control ($\varepsilon_{\text{rand}}$) adds isotropic noise
scaled to $|\mu_t - \mu|$.
Values are accuracy (\%) with 95\% bootstrap half-widths.}
\label{tab:mean_ablation}
\begin{tabular}{ll c ccc ccc c}
\toprule
 & & & \multicolumn{3}{c}{\textbf{Ablate}} & \multicolumn{3}{c}{\textbf{Isolate}} & \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
\textbf{Task} & \textbf{Model} & \textbf{Orig.}
  & $\mu_t$ & $\mu_i$ & $\varepsilon$
  & $\mu_t$ & $\mu_i$ & $\varepsilon$
  & \makecell{$\varepsilon_{\text{rand}}$ \\ \textbf{ctrl.}} \\
\midrule"""

    footer = r"""\bottomrule
\end{tabular}
\end{table}"""

    blocks = []
    for task, _ in TASKS:
        rows = []
        for i, (dir_name, model_label) in enumerate(MODELS):
            entry = family_data[task].get(dir_name)

            task_col = (
                rf"\multirow{{{len(MODELS)}}}{{*}}{{{TASK_MAKECELL[task]}}}"
                if i == 0 else ""
            )

            cells = [
                task_col,
                model_label,
                _cell_orig(entry),
                _cell_iv(entry, "timestep_ablation"),
                _cell_iv(entry, "instance_ablation"),
                _cell_iv(entry, "residual_ablation"),
                _cell_iv(entry, "timestep_isolation"),
                _cell_iv(entry, "instance_isolation"),
                _cell_iv(entry, "residual_isolation"),
                _cell_random(entry),
            ]
            rows.append(" & ".join(cells) + r" \\")

        blocks.append("\n".join(rows))

    body = "\n\\midrule\n".join(blocks)
    return "\n".join([header, body, footer])


# ── Figure: main-text line plot ───────────────────────────────────────────────

# Slot layout for the grouped bar chart. Each "slot" is one intervention
# column; within a slot, one bar per model is drawn side by side.
#   slot 0       = Orig.
#   slots 1,2,3  = Ablate   μ_t / μ̂_i / r
#   slots 4,5,6  = Isolate   μ_t / μ̂_i / r
#   slot 7       = ε_rand ctrl. (only when random data exist)
SLOT_ORIG    = 0
SLOT_ABLATE  = [1, 2, 3]
SLOT_ISOLATE = [4, 5, 6]
SLOT_RAND    = 7


def _bar_value_and_err(point, ci):
    """
    Resolve a bar's height + asymmetric error-bar arms (all in %).

    Prefer the bootstrap CI's own point estimate so the bar height and the
    error bar share the same resampling. Returns (height_pct, lo_arm, hi_arm)
    or None when no value is available.

    Asymmetric error arms (matplotlib yerr convention):
        lo_arm = point - ci_low
        hi_arm = ci_high - point
    """
    if ci is not None:
        p_pct  = ci["point"]    * 100
        lo_arm = (ci["point"] - ci["ci_low"])  * 100
        hi_arm = (ci["ci_high"] - ci["point"]) * 100
        return p_pct, max(lo_arm, 0.0), max(hi_arm, 0.0)
    if point is not None:
        return point * 100, 0.0, 0.0
    return None


def _draw_panel(ax, panel_data, present_models, n_models, bar_w):
    """
    Draw one family×task panel of grouped bars onto `ax`.

    panel_data: dict[model_dir] -> entry | None   (entries for ONE family+task)
    Returns (handles, labels, has_rand) for legend assembly by the caller.
    Handles/labels are returned per-call; caller dedups across panels.
    """
    has_rand = any(
        panel_data.get(m) is not None and panel_data[m].get("random") is not None
        for m, _ in MODELS
    )
    slots = [SLOT_ORIG] + SLOT_ABLATE + SLOT_ISOLATE + ([SLOT_RAND] if has_rand else [])

    # ── Group label annotations above the bars ──────────────────────────────
    for xc, txt in [(2, "Ablate"), (5, "Isolate")]:
        ax.annotate(
            txt,
            xy=(xc, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=8,
            style="italic", color="#444444",
        )
    if has_rand:
        ax.annotate(
            r"$\varepsilon_\mathrm{rand}$",
            xy=(SLOT_RAND, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=8,
            style="italic", color="#444444",
        )

    # ── Thin separator lines between groups ─────────────────────────────────
    for xv in (0.5, 3.5):
        ax.axvline(xv, color="#bbbbbb", lw=0.5, ls="--", zorder=1)
    if has_rand:
        ax.axvline(6.5, color="#bbbbbb", lw=0.5, ls="--", zorder=1)

    # ── Per-model grouped bars ──────────────────────────────────────────────
    handles, labels = [], []
    y_max_seen = 0.0
    for m_idx, (dir_name, style) in enumerate(present_models):
        entry = panel_data.get(dir_name)
        if entry is None:
            continue

        color  = style["color"]
        offset = (m_idx - (n_models - 1) / 2) * bar_w   # center bars on each slot

        xs, heights, err_lo, err_hi = [], [], [], []

        def _push(slot, point, ci):
            resolved = _bar_value_and_err(point, ci)
            if resolved is None:
                return
            h, lo, hi = resolved
            xs.append(slot + offset)
            heights.append(h)
            err_lo.append(lo)
            err_hi.append(hi)

        _push(SLOT_ORIG, entry.get("orig"), entry.get("ci_orig"))
        for slot, key in zip(SLOT_ABLATE, ABLATE_KEYS):
            iv = entry["interventions"].get(key, {})
            _push(slot, iv.get("point"), iv.get("ci"))
        for slot, key in zip(SLOT_ISOLATE, ISOLATE_KEYS):
            iv = entry["interventions"].get(key, {})
            _push(slot, iv.get("point"), iv.get("ci"))
        if has_rand:
            _push(SLOT_RAND, entry.get("random"), entry.get("ci_random"))

        if not xs:
            continue

        bars = ax.bar(
            xs, heights, width=bar_w * 0.9,
            color=color, edgecolor="black", linewidth=0.3,
            zorder=2, label=style["label"],
        )
        ax.errorbar(
            xs, heights,
            yerr=[err_lo, err_hi],
            fmt="none", ecolor="#222222", elinewidth=0.6,
            capsize=1.8, capthick=0.6, zorder=3,
        )
        y_max_seen = max(y_max_seen, *(h + e for h, e in zip(heights, err_hi)))
        handles.append(bars[0])
        labels.append(style["label"])

    # ── Axes cosmetics ──────────────────────────────────────────────────────
    x_ticks  = [SLOT_ORIG] + SLOT_ABLATE + SLOT_ISOLATE
    x_labels = ["Orig.", r"$\mu_t$", r"$\hat{\mu}_i$", r"$r$",
                r"$\mu_t$", r"$\hat{\mu}_i$", r"$r$"]
    if has_rand:
        x_ticks.append(SLOT_RAND)
        x_labels.append(r"$\varepsilon_\mathrm{rand}$")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.tick_params(axis="x", pad=3, length=0)
    ax.set_ylim(0, max(y_max_seen * 1.12, 1.0))
    ax.set_xlim(slots[0] - 0.6, slots[-1] + 0.6)
    ax.grid(axis="y", color="#e2e2e2", lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    return handles, labels, has_rand


def plot_mean_ablation(data: dict, out_dir: str):
    """
    One grouped-bar figure per model family: cols = task (Graph-Hopping left,
    Arithmetic-Reasoning right).

    Each x-slot is a categorical intervention; bars within a slot are the
    models, drawn side by side. Error bars are 95% bootstrap CIs (asymmetric).

    Slot layout (per panel):
      0        = Orig.
      1, 2, 3  = Ablate  μ_t / μ̂_i / r
      4, 5, 6  = Isolate μ_t / μ̂_i / r
      7        = ε_rand ctrl. (only when data exist)
    """
    families_present = [
        (fam, lbl) for fam, lbl in FAMILIES
        if any(
            data.get(fam, {}).get(task, {}).get(m) is not None
            for task, _ in TASKS
            for m, _ in MODELS
        )
    ]
    if not families_present:
        print("[WARN] No model data for any family — no figure written. "
              "Run with --debug to see which artefacts are missing.")
        return

    present_models = [(m, MODEL_STYLE[m]) for m, _ in MODELS if m in MODEL_STYLE]
    n_models = len(present_models)
    bar_w    = 0.8 / max(n_models, 1)

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    for fam, fam_label in families_present:
        n_cols = len(TASKS)
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(8.0, 2.8),
            squeeze=False,
            gridspec_kw={"wspace": 0.22},
        )

        all_handles, all_labels = [], []
        panel_letters = iter("ab")

        for c, (task, task_label) in enumerate(TASKS):
            ax = axes[0][c]
            panel_data = data.get(fam, {}).get(task, {})

            handles, labels, _ = _draw_panel(
                ax, panel_data, present_models, n_models, bar_w
            )
            for h, l in zip(handles, labels):
                if l not in all_labels:
                    all_handles.append(h)
                    all_labels.append(l)

            if c == 0:
                ax.set_ylabel("Accuracy (%)")
            letter = next(panel_letters)
            ax.set_title(f"({letter}) {task_label}",
                         fontsize=11, pad=14)

        if all_handles:
            fig.legend(
                all_handles, all_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(all_handles),
                fontsize=9,
                frameon=True,
                framealpha=0.8,
                edgecolor="#cccccc",
                handlelength=2.0,
                columnspacing=1.2,
                handletextpad=0.5,
            )

        out_pdf = out_base / f"mean_ablation_{fam}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure       -> {out_pdf.resolve()}")


# ── Table 2: appendix statistical tests ──────────────────────────────────────

def build_appendix_table(family_data: dict, family: str = "", family_label: str = "") -> str:
    has_diffs = any(
        family_data[task].get(dir_name) is not None
        and any(
            family_data[task][dir_name]["diffs"][k]["diff_ci"] is not None
            for k, _, _, _ in INTERVENTIONS
        )
        for task, _ in TASKS
        for dir_name, _ in MODELS
    )

    has_rand = any(
        family_data[task].get(dir_name) is not None
        and family_data[task][dir_name].get("random") is not None
        for task, _ in TASKS
        for dir_name, _ in MODELS
    )

    # Both tasks appear as column groups side by side.
    # Per-task columns: (Acc, n) or (Acc, Δ, McNemar, n) when all-run data exist.
    task_ncols   = 4 if has_diffs else 2
    total_ncols  = 2 + len(TASKS) * task_ncols

    # Center the n-columns explicitly.
    # Acc / Δ / McNemar remain left-aligned for readability.
    task_col_spec = "lllc" if has_diffs else "lc"
    col_spec = "ll " + " ".join([task_col_spec] * len(TASKS))

    # cmidrule ranges for the two task column groups
    cmidrules = []
    start = 3
    for _ in TASKS:
        end = start + task_ncols - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    if has_diffs:
        task_subheader = r"Acc [\% CI] & $\Delta$ vs.\ Orig.\ [\% CI] & McNemar $p$ & $n$"
    else:
        task_subheader = r"Acc [\% CI] & $n$"

    task_multicols = " & ".join(
        rf"\multicolumn{{{task_ncols}}}{{c}}{{\textbf{{{lbl}}}}}"
        for _, lbl in TASKS
    )

    fam_suffix = f" ({family})" if family else ""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & & {task_multicols} \\",
        " ".join(cmidrules),
        rf"Model & Intervention & {(' & '.join([task_subheader] * len(TASKS)))} \\",
        r"\midrule",
    ]

    for m_idx, (dir_name, model_label) in enumerate(MODELS):
        n_rows = len(INTERVENTIONS) + 1 + (1 if has_rand else 0)
        orig_row_label = rf"\multirow{{{n_rows}}}{{*}}{{{model_label}}}"

        # Orig. row — shows n; remaining rows leave n blank.
        orig_cells = [orig_row_label, r"Orig.\ (identity)"]
        for task, _ in TASKS:
            entry = family_data[task].get(dir_name)
            n_str   = str(entry["n"]) if entry and entry.get("n") else "--"
            acc_str = ci_pct(entry["ci_orig"]) if entry and entry.get("ci_orig") else (
                      fmt_pct(entry.get("orig")) if entry else "--")
            if has_diffs:
                orig_cells += [acc_str, "---", "---", n_str]
            else:
                orig_cells += [acc_str, n_str]
        lines.append(" & ".join(orig_cells) + r" \\")

        # One row per intervention.
        for sum_key, _, _, display_name in INTERVENTIONS:
            iv_cells = ["", display_name]
            for task, _ in TASKS:
                entry = family_data[task].get(dir_name)
                if entry is None:
                    iv_cells += ["--", "--", "--", ""] if has_diffs else ["--", ""]
                    continue
                iv     = entry["interventions"].get(sum_key, {})
                diff_d = entry["diffs"].get(sum_key, {})
                acc_str = ci_pct(iv.get("ci")) if iv.get("ci") else fmt_pct(iv.get("point"))
                if has_diffs:
                    iv_cells += [acc_str, diff_pct(diff_d.get("diff_ci")),
                                 fmt_mcnemar(diff_d.get("mc")), ""]
                else:
                    iv_cells += [acc_str, ""]
            lines.append(" & ".join(iv_cells) + r" \\")

        # ε_rand control row (only when random-matched-norm data exist).
        if has_rand:
            rand_cells = ["", r"$\varepsilon_\mathrm{rand}$ ctrl."]
            for task, _ in TASKS:
                entry = family_data[task].get(dir_name)
                if entry is None or entry.get("random") is None:
                    rand_cells += ["--", "--", "--", ""] if has_diffs else ["--", ""]
                else:
                    acc_str = (ci_pct(entry["ci_random"]) if entry.get("ci_random")
                               else fmt_pct(entry.get("random")))
                    if has_diffs:
                        rand_cells += [acc_str, diff_pct(entry.get("rand_diff_ci")),
                                       fmt_mcnemar(entry.get("rand_mc")), ""]
                    else:
                        rand_cells += [acc_str, ""]
            lines.append(" & ".join(rand_cells) + r" \\")

        # Thin rule between model blocks (not after the last).
        if m_idx < len(MODELS) - 1:
            lines.append(rf"\cmidrule(lr){{2-{total_ncols}}}")

    p_note = (r" $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ "
              r"(exact two-sided McNemar).") if has_diffs else ""
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Mean-ablation accuracy and statistical tests{fam_suffix}.{p_note}}}",
        rf"\label{{tab:mean_ablation_stats_{family or 'combined'}}}",
        r"\end{table}",
    ]

    return "\n".join(lines).strip()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_fig_dir",   default="Plots/mean_ablation",
                    help="Output directory for per-family bar figures.")
    ap.add_argument("--out_main_dir",  default="Tables/main",
                    help="Output directory for main-text LaTeX tables.")
    ap.add_argument("--out_stats_dir", default="Tables/extended",
                    help="Output directory for appendix statistical LaTeX tables.")
    ap.add_argument("--debug", action="store_true",
                    help="Print every path probed and collected values, then exit.")
    args = ap.parse_args()

    data = collect(debug=args.debug)

    if args.debug:
        print("\n\n-- Collected data summary --")
        for family, tasks in data.items():
            for task, models in tasks.items():
                for m, entry in models.items():
                    if entry is None:
                        print(f"  {family}/{task}/{m}: None")
                    else:
                        iv_pts = {k: f"{v['point']*100:.1f}" if v.get("point") else "--"
                                  for k, v in entry["interventions"].items()}
                        print(f"  {family}/{task}/{m}: orig={entry['orig']}, "
                              f"n={entry['n']}, ivs={iv_pts}")
        return

    print("=" * 60)
    print("Figure: mean-ablation bar plot (one per family)")
    print("=" * 60)
    plot_mean_ablation(data, args.out_fig_dir)

    print("=" * 60)
    print("Main-text tables (one per family)")
    print("=" * 60)
    main_dir = Path(args.out_main_dir)
    main_dir.mkdir(parents=True, exist_ok=True)
    for family, family_label in FAMILIES:
        family_data = data.get(family)
        if not family_data:
            continue
        has_any = any(
            family_data[task].get(m) is not None
            for task, _ in TASKS
            for m, _ in MODELS
        )
        if not has_any:
            print(f"[skip] {family_label}: no data")
            continue
        main_tex = build_main_table(family_data)
        p = main_dir / f"mean_ablation_{family}.tex"
        p.write_text(main_tex)
        print(f"[OK] Main table   -> {p.resolve()}")

    print("=" * 60)
    print("Appendix: statistical-test tables (one per family)")
    print("=" * 60)
    stats_dir = Path(args.out_stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    for family, family_label in FAMILIES:
        family_data = data.get(family)
        if not family_data:
            continue
        has_any = any(
            family_data[task].get(m) is not None
            for task, _ in TASKS
            for m, _ in MODELS
        )
        if not has_any:
            print(f"[skip] {family_label}: no data")
            continue
        app_tex = build_appendix_table(family_data, family, family_label)
        p = stats_dir / f"mean_ablation_{family}.tex"
        p.write_text(app_tex)
        print(f"[OK] Stats table  -> {p.resolve()}")


if __name__ == "__main__":
    main()