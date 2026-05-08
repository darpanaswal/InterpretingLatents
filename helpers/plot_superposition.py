# plot_superposition.py
import json
import argparse
import math
from pathlib import Path
from src.config import CONTROL_EXPT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

_MODES_ORDER = ["base", "cot", "pause", "coconut", "coconut_u", "codi"]
_LABELS_SHORT = {
    "base": "B", "cot": "CoT", "pause": "PaT", "coconut": "C",
    "coconut_u": "C$_u$", "codi": "CODI",
}
_LABELS_LONG = {
    "base": "Base", "cot": "CoT", "pause": "Pause", "coconut": "Coconut",
    "coconut_u": "Coconut$_{u}$", "codi": "CODI",
}
_PLOT_LABELS = {
    "base": "B",
    "cot": "CoT",
    "pause": "PaT",
    "coconut": "C",
    "coconut_u": r"$C_u$",
    "codi": "CODI",
}
_MODEL_STYLE = {
    "base":      {"color": "#999999", "marker": "x",  "ls": ":"},
    "cot":       {"color": "#E69F00", "marker": "^",  "ls": "--"},
    "pause":     {"color": "#0072B2", "marker": "o",  "ls": "-"},
    "coconut":   {"color": "#D55E00", "marker": "s",  "ls": "-"},
    "coconut_u": {"color": "#009E73", "marker": "D",  "ls": "-"},
    "codi":      {"color": "#CC79A7", "marker": "v",  "ls": "-."},
}

_PLOT_METRICS = [
    ("mean_normalized_entropy", "normalized_entropy", "Entropy", r"$H/\log_2 N$", False, 2),
    ("top1_correct_frac", "top1_is_correct", "Top-1 Correct", "Top-1 correct (%)", True, 1),
    ("mean_value_correct", "value_correct", r"$P(\mathrm{correct})$", r"$P(\mathrm{correct})$", False, 2),
    ("mean_candidate_mass", "candidate_mass", "Cand. Mass", "Cand. mass", False, 2),
]

def _load_all_results(task):
    results_dir = CONTROL_EXPT
    results = {}
    if not results_dir.exists():
        return results, results_dir
    for f in sorted(results_dir.glob("results_*.json")):
        mode = f.stem.replace("results_", "")
        with open(f) as fh:
            results[mode] = json.load(fh)
    return results, results_dir

def _load_jsonl(path):
    if not Path(path).exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _load_ci_records(results_dir, modes):
    ci_dir = results_dir / "ci"
    return {m: _load_jsonl(ci_dir / f"superposition_{m}.jsonl") for m in modes}

def _context_matches(record, expected):
    ctx = record.get("_context", {})
    return all(ctx.get(key) == value for key, value in expected.items())

def _find_ci_record(ci_records, metric, context=None):
    context = context or {}
    for record in reversed(ci_records):
        if record.get("metric") == metric and _context_matches(record, context):
            return record
    return None

def _ci_half_width(record, scale=1.0):
    if record is None:
        return None
    return (record["ci_high"] - record["ci_low"]) * scale / 2

def _fmt_endpoint(value, record=None, pct=False, decimals=1):
    scale = 100 if pct else 1
    point = value * scale
    if record is None:
        return f"{point:.{decimals}f}"
    half = _ci_half_width(record, scale)
    return f"{point:.{decimals}f}±{half:.{decimals}f}"

def fmt_float(val, decimals=2):
    if math.isnan(val): return "-"
    # Mathematically rounds to specified decimal places
    s = f"{val:.{decimals}f}"
    # Remove leading zero for values < 1 and > -1
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s

def fmt_pct(val):
    if math.isnan(val): return "-"
    # Rounds to the nearest integer and removes decimals
    return f"{round(val * 100)}"

def _fmt_with_ci(value, ci_record, pct=False, decimals=2):
    """
    Point estimate plus bootstrap half-width, in the same '$p {\\scriptstyle \\pm hw}$'
    style used by remove_thoughts_tables.py. Falls back to point-only if no CI record.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    scale = 100 if pct else 1
    p = value * scale
    if ci_record is None:
        return f"${p:.{decimals}f}$"
    hw = (ci_record["ci_high"] - ci_record["ci_low"]) * scale / 2
    return rf"${p:.{decimals}f} {{\scriptstyle \pm {hw:.{decimals}f}}}$"


def _bfs_header_block(modes, metric_specs):
    """
    Emit one header block (multicol row + cmidrules + per-model $k$ row)
    covering the metrics in `metric_specs`. Each spec is (column_label,).
    Returns a list of LaTeX lines.
    """
    num_m = len(modes)
    multicol = ["", *[
        f"\\multicolumn{{{num_m}}}{{c}}{{{label}}}"
        for (label,) in metric_specs
    ]]
    cmidrules = []
    start = 2
    for _ in metric_specs:
        end = start + num_m - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    headers = ["$k$"] + [_LABELS_SHORT[m] for m in modes] * len(metric_specs)
    return [
        " & ".join(multicol) + " \\\\",
        " ".join(cmidrules),
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]


def _bfs_data_rows(results, modes, all_k, ci_data, blocks):
    """
    One row per k, with one cell per (metric, model). `blocks` is a list of
    (summary_key, ci_metric_key, pct_flag, decimals) tuples.
    """
    rows = []
    for k in all_k:
        row = [str(k)]
        for summary_key, ci_key, pct, decimals in blocks:
            for m in modes:
                v = results[m]["summary"].get(str(k), {}).get(summary_key, float("nan"))
                rec = _find_ci_record(ci_data.get(m, []), ci_key, {"t": int(k)})
                row.append(_fmt_with_ci(v, rec, pct=pct, decimals=decimals))
        rows.append(" & ".join(row) + " \\\\")
    return rows


def generate_appendix_main_table(results, modes, all_k, ci_data=None):
    """
    Full standard-probing table over all (model, k), with bootstrap CIs.
    Layout: a single table with two stacked halves (separated by \\midrule)
    to halve the horizontal footprint:

        Top half:    $H/\\log_2 N$, Top-1 correct (%)
        Bottom half: $P(\\text{correct})$, Cand. mass

    Each half repeats the per-model column headers and the $k$ index column.
    Column count goes from 1 + 4M to 1 + 2M (M = number of models).
    """
    ci_data = ci_data or {}
    num_m = len(modes)
    cols = "l " + " ".join(["c" * num_m] * 2)

    # (summary_key, ci_metric_key, pct_flag, decimals)
    top_blocks = [
        ("mean_normalized_entropy", "normalized_entropy", False, 2),
        ("top1_correct_frac",       "top1_is_correct",    True,  1),
    ]
    bot_blocks = [
        ("mean_value_correct",  "value_correct",  False, 2),
        ("mean_candidate_mass", "candidate_mass", False, 2),
    ]
    top_specs = [(r"$H/\log_2 N$",), (r"Top-1 correct (\%)",)]
    bot_specs = [(r"$P(\text{correct})$",), (r"Cand.\ mass",)]

    latex = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Full standard-probing results across timesteps and models. "
        "Each cell is the per-timestep mean with its 95\\% bootstrap half-width "
        "(10{,}000 percentile resamples). The table is split into two stacked "
        "halves to fit the page width: the top half reports normalized entropy "
        "and Top-1 correctness; the bottom half reports $P(\\text{correct})$ "
        "(the raw, unnormalized joint probability summed over correct candidates) "
        "and candidate mass (total raw probability assigned to all candidates at "
        "the depth frontier). Values $<0.01$ for PaT and C at $k{=}0{-}2$ reflect "
        "that these models were trained to reason through latent recurrence and "
        "assign near-zero probability to any concept before the recurrence has begun.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
    ]
    latex += _bfs_header_block(modes, top_specs)
    latex += _bfs_data_rows(results, modes, all_k, ci_data, top_blocks)
    latex += ["\\midrule"]
    latex += _bfs_header_block(modes, bot_specs)
    latex += _bfs_data_rows(results, modes, all_k, ci_data, bot_blocks)
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:appendix_bfs_probing}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(latex)

def _metric_ylim(results, modes, all_k, summary_key, pct):
    vals = []
    for mode in modes:
        for k in all_k:
            value = results[mode]["summary"].get(str(k), {}).get(summary_key)
            if value is None or math.isnan(value):
                continue
            vals.append(value * 100 if pct else value)
    if not vals:
        return (0, 1)
    lo = min(vals)
    hi = max(vals)
    if pct:
        return (max(-2, lo - 0.08 * max(hi - lo, 1)), min(105, hi + 0.18 * max(hi - lo, 1)))
    if summary_key == "mean_normalized_entropy":
        return (-0.02, 1.05)
    pad = 0.18 * max(hi - lo, 0.01)
    return (max(-0.01, lo - pad), hi + pad)

def plot_superposition_metrics(results, modes, all_k, results_dir):
    plot_k = [k for k in all_k if k <= 4]
    if not plot_k:
        print("No k<=4 timesteps available for superposition plot.")
        return

    ci_data = _load_ci_records(results_dir, modes)
    
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 1.75), gridspec_kw={"wspace": 0.38})
    axes_flat = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (summary_key, ci_key, title, ylabel, pct, decimals), panel_label in zip(
        axes_flat, _PLOT_METRICS, panel_labels
    ):
        ylim = _metric_ylim(results, modes, plot_k, summary_key, pct)
        endpoints = []

        for mode in modes:
            xs = []
            ys = []
            for k in plot_k:
                value = results[mode]["summary"].get(str(k), {}).get(summary_key)
                if value is None or math.isnan(value):
                    continue
                xs.append(k)
                ys.append(value * 100 if pct else value)
            if not xs:
                continue
            style = _MODEL_STYLE[mode]
            ax.plot(
                xs, ys,
                color=style["color"],
                marker=style["marker"],
                ls=style["ls"],
                label=_PLOT_LABELS[mode],
                markeredgewidth=0.3,
                markeredgecolor="black",
            )
            endpoints.append((ys[-1], mode, xs[-1], style["color"]))

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

        for y_val, mode, x_val, color, label_y in staggered:
            record = _find_ci_record(ci_data.get(mode, []), ci_key, {"t": int(x_val)})
            raw_value = y_val / 100 if pct else y_val

            x_pts = [x_val, x_val + 0.15, x_val + 0.4, x_val + line_extend]
            y_pts = [y_val, y_val, label_y, label_y]
            ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)

            ax.annotate(
                _fmt_endpoint(raw_value, record, pct=pct, decimals=decimals),
                xy=(x_val + line_extend, label_y),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                weight="bold",
                ha="left",
                va="center",
                clip_on=False
            )

        ax.set_xlabel("Thought step $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {title}", fontsize=8, pad=6)
        ax.set_ylim(*ylim)
        
        left_pad = -0.2
        right_pad = 3
        ax.set_xlim(min(plot_k) + left_pad, max(plot_k) + right_pad)
        ax.set_xticks(plot_k)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(labels),
        fontsize=6,
        frameon=True,
        framealpha=0.8,
        edgecolor="#cccccc",
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    out_path = results_dir / "superposition_metrics.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Superposition Figure to: {out_path}")

def generate_table_candidate_parallelism(results, modes, all_k):
    num_m = len(modes)
    cols = "l " + " ".join(["ccc"] * num_m)
    
    latex = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Cumulative top-$k$ raw probabilities. T1 = top-1, T3 = top-3, $\\Delta$ = T3 $-$ T1. Coconut $u{=}0.0$ and Pause-as-thought show negligible gaps throughout, concentrating mass on a single candidate. Coconut $u{=}0.3$ shows substantial gaps at shallow $k$ (broad exploration) that narrow with depth.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
    ]
    
    h1 = [""] + [f"\\multicolumn{{3}}{{c}}{{{_LABELS_LONG[m]}}}" for m in modes]
    latex.append(" & ".join(h1) + " \\\\")
    
    cmidrules = []
    start = 2
    for _ in range(num_m):
        end = start + 2
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    latex.append(" ".join(cmidrules))
    
    h2 = ["$k$"] + ["T1 & T3 & $\\Delta$"] * num_m
    latex.append(" & ".join(h2) + " \\\\")
    latex.append("\\midrule")
    
    for k in all_k[:5]:
        row = [str(k)]
        for m in modes:
            s = results[m]["summary"].get(str(k), {})
            t1 = s.get("mean_top1_prob", float("nan"))
            t3 = s.get("mean_top3_cumul", float("nan"))
            if math.isnan(t1):
                row.append("- & - & -")
            else:
                delta = max(0, t3 - t1)
                row.append(f"{fmt_float(t1, 3)} & {fmt_float(t3, 3)} & {fmt_float(delta, 3)}")
        latex.append(" & ".join(row) + " \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:candidate_parallelism}",
        "\\end{table*}",
        ""
    ])
    return "\n".join(latex)

def generate_table_convergence_trajectories(results, modes):
    latex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\caption{Convergence from $k{=}0$ to $k{=}4$. Pause-as-thought and Coconut $u{=}0.0$ show increasing P(correct) with decreasing entropy (exploration to convergence). Base, CoT, and Coconut $u{=}0.3$ show degrading or flat P(correct).}",
        "\\begin{tabular}{l cc cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{$P(\\text{correct})$} & \\multicolumn{2}{c}{$H/\\log_2 N$} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Condition & $k{=}0$ & $k{=}4$ & $k{=}0$ & $k{=}4$ \\\\",
        "\\midrule"
    ]
    
    for m in modes:
        s0 = results[m]["summary"].get("0", {})
        s4 = results[m]["summary"].get("4", {})
        
        pc0 = s0.get("mean_value_correct", float("nan"))
        pc4 = s4.get("mean_value_correct", float("nan"))
        h0 = s0.get("mean_normalized_entropy", float("nan"))
        h4 = s4.get("mean_normalized_entropy", float("nan"))
        
        row = [
            _LABELS_LONG[m],
            fmt_float(pc0, 2),
            fmt_float(pc4, 2),
            fmt_float(h0, 2),
            fmt_float(h4, 2)
        ]
        latex.append(" & ".join(row) + " \\\\")
        
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:convergence_trajectories}",
        "\\end{table}",
        ""
    ])
    return "\n".join(latex)

def generate_table_per_instance_statistics(results, modes):
    num_m = len(modes)
    cols = "l " + "c"*num_m
    
    latex = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\small",
        "\\caption{Per-instance analysis. ``P(correct) increases'' counts instances where the raw probability summed over correct candidates is higher at the final $k$ evaluated for that instance than at $k{=}0$. The close match between Coconut $u{=}0.0$ (61.5\\%) and Pause-as-thought (62.5\\%) reinforces that the curriculum drives the convergence pattern.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & " + " & ".join([_LABELS_LONG[m] for m in modes]) + " \\\\",
        "\\midrule"
    ]
    
    row_inc = ["P(correct) increases ($k{=}0 \\to k_{\\text{last}}$)"]
    row_top1 = ["Top-1 correct at final $k$"]
    row_ent = ["High entropy at $k{=}0$ ($H/\\log_2 N > 0.5$)"]
    row_n = ["$n$"]
    
    for m in modes:
        per_inst = results[m].get("per_instance", {})
        n_total = n_inc = n_top1 = n_ent = 0
        
        for si, k_data in per_inst.items():
            ks = sorted(int(k) for k in k_data)
            if len(ks) < 2: continue
            n_total += 1
            first = k_data[str(ks[0])]["metrics"]
            last = k_data[str(ks[-1])]["metrics"]
            
            if last.get("value_correct", 0) > first.get("value_correct", 0) + 0.01:
                n_inc += 1
            if last.get("top1_is_correct", False):
                n_top1 += 1
            if first.get("normalized_entropy", 0) > 0.5:
                n_ent += 1
                
        if n_total > 0:
            row_inc.append(f"{round((n_inc/n_total)*100)}\\%")
            row_top1.append(f"{round((n_top1/n_total)*100)}\\%")
            row_ent.append(f"{round((n_ent/n_total)*100)}\\%")
            row_n.append(str(n_total))
        else:
            row_inc.append("-")
            row_top1.append("-")
            row_ent.append("-")
            row_n.append("-")
            
    latex.append(" & ".join(row_inc) + " \\\\")
    latex.append(" & ".join(row_top1) + " \\\\")
    latex.append(" & ".join(row_ent) + " \\\\")
    latex.append("\\midrule")
    latex.append(" & ".join(row_n) + " \\\\")
    
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:per_instance_statistics}",
        "\\end{table*}"
    ])
    return "\n".join(latex)

def generate_stats_subsection(results, modes, all_k, ci_data):
    """
    Statistical-tests subsection: full standard-probing table with per-timestep
    bootstrap 95% CIs. Self-contained \\subsection-wrapped tex file.
    """
    lines = [
        r"\subsection{Superposition Probing (Experiment~2): Statistical Tests}",
        r"\label{app:superposition_stats}",
        "",
        r"Per-timestep point estimates and 95\% bootstrap CIs "
        r"(10{,}000 percentile resamples) for the four headline standard-probing "
        r"metrics: normalized entropy $H/\log_2 N$, top-1 correctness, raw "
        r"probability mass on correct candidates $P(\text{correct})$, and total "
        r"candidate mass. Half-widths are reported alongside each point estimate; "
        r"non-overlap of CIs across models at a given $k$ indicates a reliable "
        r"between-condition difference.",
        "",
        generate_appendix_main_table(results, modes, all_k, ci_data=ci_data),
    ]
    return "\n".join(lines).strip() + "\n"


def generate_extended_subsection(results, modes, all_k):
    """
    Extended-analysis subsection: qualitative / structural patterns
    (candidate parallelism, convergence trajectories, per-instance dynamics).
    Self-contained \\subsection-wrapped tex file.
    """
    lines = [
        r"\subsection{Superposition Probing (Experiment~2): Extended Analysis}",
        r"\label{app:superposition_extended}",
        "",
        r"Three complementary views of the probing signal beyond per-timestep "
        r"point estimates. Table~\ref{tab:candidate_parallelism} reports cumulative "
        r"top-$k$ raw probabilities (T1, T3, $\Delta = $ T3 $-$ T1) to quantify how "
        r"much mass each model spreads across competing candidates. "
        r"Table~\ref{tab:convergence_trajectories} contrasts $k{=}0$ and $k{=}4$ to "
        r"summarize whether each model converges (rising $P(\text{correct})$ with "
        r"falling entropy) or fails to. Table~\ref{tab:per_instance_statistics} "
        r"breaks the same patterns down per instance, isolating the fraction of "
        r"problems on which each behavior actually occurs.",
        "",
        generate_table_candidate_parallelism(results, modes, all_k),
        "",
        generate_table_convergence_trajectories(results, modes),
        "",
        generate_table_per_instance_statistics(results, modes),
    ]
    return "\n".join(lines).strip() + "\n"


def run_analysis(task):
    all_results, results_dir = _load_all_results(task)
    if not all_results:
        print(f"No results found in {results_dir}")
        return

    available = [m for m in _MODES_ORDER if m in all_results]
    all_k = sorted({
        int(k) for mode in available for k in all_results[mode]["summary"]
    })

    # CI records (loaded once; reused by main table and figure).
    ci_data = _load_ci_records(results_dir, available)

    # Replacement main-text figure.
    plot_superposition_metrics(all_results, available, all_k, results_dir)

    # Two subsection-wrapped appendix files, mirroring remove_thoughts_tables.py.
    subsections = [
        ("tab_superposition_stats.tex",
         generate_stats_subsection(all_results, available, all_k, ci_data),
         "Statistical-Tests Subsection"),
        ("tab_superposition_extended.tex",
         generate_extended_subsection(all_results, available, all_k),
         "Extended-Analysis Subsection"),
    ]
    for fname, tex, label in subsections:
        out = f"tables/{fname}"
        with open(out, "w") as f:
            f.write(tex)
        print(f"Saved {label} to: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    args = parser.parse_args()
    run_analysis(args.task)