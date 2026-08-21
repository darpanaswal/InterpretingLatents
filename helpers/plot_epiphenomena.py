# plot_epiphenomena
import json
import argparse
import math
import re
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# Style: Minimalist IEEE/CVPR standards (thin lines, serif)
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
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

# ═══════════════════════════════════════════════════════════════════
# Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════
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

def fmt_pm_pct(point_pct, record=None, decimals=1, latex=False):
    if record is None:
        return f"{point_pct:.{decimals}f}"
    half_width = _ci_half_width(record, 100.0)
    if latex:
        return rf"${point_pct:.{decimals}f} {{\scriptstyle \pm {half_width:.{decimals}f}}}$"
    return f"{point_pct:.{decimals}f}±{half_width:.{decimals}f}"

def _fmt_endpoint(value, record=None, pct=False, decimals=1):
    scale = 100 if pct else 1
    point = value * scale
    if record is None:
        return f"{point:.{decimals}f}"
    half = _ci_half_width(record, scale)
    return f"{point:.{decimals}f}±{half:.{decimals}f}"

def _fmt_with_ci(value, ci_record, pct=False, decimals=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    scale = 100 if pct else 1
    p = value * scale
    if ci_record is None:
        return f"${p:.{decimals}f}$"
    hw = _ci_half_width(ci_record, scale)
    return rf"${p:.{decimals}f} {{\scriptstyle \pm {hw:.{decimals}f}}}$"

def fmt_float(val, decimals=2):
    if math.isnan(val): return "-"
    s = f"{val:.{decimals}f}"
    if s.startswith("0."): return s[1:]
    if s.startswith("-0."): return "-" + s[2:]
    return s

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
    lo, hi = min(vals), max(vals)
    if pct:
        return (max(-2, lo - 0.08 * max(hi - lo, 1)), min(105, hi + 0.18 * max(hi - lo, 1)))
    if summary_key == "mean_normalized_entropy":
        return (-0.02, 1.05)
    pad = 0.18 * max(hi - lo, 0.01)
    return (max(-0.01, lo - pad), hi + pad)

# ═══════════════════════════════════════════════════════════════════
# Split Plotting: Superposition figure & Logit-Lens/Scratchpad figure
# (one standalone 1x3 figure per group, each with its own legend)
# ═══════════════════════════════════════════════════════════════════
def plot_superposition_metrics(sup_results, sup_cis, sup_k, ll_k, out_dir, model_family):
    """
    Creates a standalone 1x3 figure: Entropy, P(correct), Cand. Mass.
    Legend: Placed on the right side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 1.25), gridspec_kw={'wspace': 0.35})

    sup_metrics = [
        ("mean_normalized_entropy", "normalized_entropy", "Entropy", r"$H/\log_2 N$", False, 2),
        ("mean_value_correct", "value_correct", r"$P(\mathrm{correct})$", r"$P(\mathrm{correct})$", False, 2),
        ("mean_candidate_mass", "candidate_mass", "Cand. Mass", "Cand. mass", False, 2),
    ]
    panel_labels_r1 = ["(a)", "(b)", "(c)"]

    for ax, (summary_key, ci_key, title, ylabel, pct, decimals), panel_label in zip(axes, sup_metrics, panel_labels_r1):
        ylim = _metric_ylim(sup_results, _MODES_ORDER, sup_k, summary_key, pct)
        endpoints = []
        
        for mode in _MODES_ORDER:
            if mode not in sup_results: continue
            xs, ys = [], []
            for k in sup_k:
                value = sup_results[mode]["summary"].get(str(k), {}).get(summary_key)
                if value is None or math.isnan(value): continue
                xs.append(k)
                ys.append(value * 100 if pct else value)
            if not xs: continue
            
            s = _MODEL_STYLE[mode]
            ax.plot(xs, ys, color=s["color"], marker=s["marker"], ls=s["ls"], 
                    label=_PLOT_LABELS[mode], markeredgewidth=0.3, markeredgecolor="black")
            endpoints.append((ys[-1], ll_k, mode, s["color"]))

        y_range = ylim[1] - ylim[0]
        min_gap = 0.10 * y_range
        endpoints.sort(key=lambda item: item[0])
        placed, staggered = [], []
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
        for y_val, x_val, mode, color, label_y in staggered:
            # Overriding x_val to the max k present in sup_k for plotting
            x_val = max(sup_k)
            record = _find_ci_record(sup_cis.get(mode, []), ci_key, {"t": int(x_val)})
            raw_value = y_val / 100 if pct else y_val
            x_pts = [x_val, x_val + 0.15, x_val + 0.4, x_val + line_extend]
            y_pts = [y_val, y_val, label_y, label_y]
            ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)

            ax.annotate(_fmt_endpoint(raw_value, record, pct=pct, decimals=decimals),
                        xy=(x_val + line_extend, label_y), xytext=(3, 0), textcoords="offset points",
                        fontsize=5.5, color=color, weight="bold", ha="left", va="center", clip_on=False)

        ax.set_xlabel("Thought step $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {title}", fontsize=8, pad=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(min(sup_k) - 0.2, max(sup_k) + 2.5)
        ax.set_xticks(sup_k)

    # ─── Legend on the right ───
    fig.subplots_adjust(right=0.88)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.89, 0.5),
               ncol=1, fontsize=6, frameon=True, framealpha=0.8, edgecolor="#cccccc",
               handlelength=2.0, handletextpad=0.5)

    out_path = Path(out_dir) / f"epiphenomena_{model_family}_superposition.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Superposition Figure to: {out_path}")


def plot_scratchpad_metrics(ll_results, ll_cis, ll_k, out_dir, model_family):
    """
    Creates a standalone 1x3 figure: Hit Rate, Superposition, Step Alignment.
    Legend: Placed on the right side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 1.25), gridspec_kw={'wspace': 0.35})

    ll_metrics = [
        ("hit_rate", "Intermediate Hit Rate", "Hit rate (%)", "(d)", "endpoint_right", (-2, 82), None),
        ("superposition_rate", "Superposition Rate", "Superposition (%)", "(e)", "endpoint_right", (-1, 30), None),
        ("alignment_rate", "Step Alignment", "Step alignment (%)", "(f)", "start_right", (-2, 68), 1.0)
    ]
    steps = list(range(ll_k + 1))

    for ax, (key, title_str, ylabel, panel_label, annotate, ylim, skip_below) in zip(axes, ll_metrics):
        endpoints, peaks, starts = [], [], []
        
        for model in _MODES_ORDER:
            if model not in ll_results: continue
            vals = [ll_results[model]["summary"][str(t)][key] for t in steps]
            pcts = [v * 100 for v in vals]
            s = _MODEL_STYLE[model]
            
            ax.plot(steps, pcts, color=s["color"], marker=s["marker"], ls=s["ls"], 
                    label=_PLOT_LABELS[model], markeredgewidth=0.3, markeredgecolor="black")
            
            endpoints.append((pcts[-1], ll_k, model, s["color"]))
            peak_t = int(np.argmax(pcts))
            peaks.append((pcts[peak_t], peak_t, model, s["color"]))
            starts.append((pcts[0], 0, model, s["color"]))

        y_range = ylim[1] - ylim[0]
        min_gap = 0.10 * y_range  

        def _stagger(items, value_idx=0):
            items = sorted(items, key=lambda e: e[value_idx])
            placed, out = [], []
            for item in items:
                label_y = item[value_idx]
                for py in placed:
                    if abs(label_y - py) < min_gap:
                        label_y = py + min_gap
                label_y = min(label_y, ylim[1] - 0.03 * y_range)
                placed.append(label_y)
                out.append((*item, label_y))
            return out

        line_extend = 0.5  
        target_list = endpoints if annotate == "endpoint_right" else starts
        
        for y_val, t_val, model, color, label_y in _stagger(target_list, value_idx=0 if annotate=="endpoint_right" else 0):
            t_ref = ll_k if annotate == "endpoint_right" else t_val
            
            if model == "base": continue
            if skip_below is not None and y_val < skip_below: continue

            ci = _find_ci_record(ll_cis.get(model, []), key, {"condition": "per_step", "t": int(t_ref)})
            x_pts = [t_ref, ll_k + 0.15, ll_k + 0.4, ll_k + line_extend]
            y_pts = [y_val, y_val, label_y, label_y]
            ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)

            ax.annotate(fmt_pm_pct(y_val, ci), xy=(ll_k + line_extend, label_y), xytext=(3, 0),
                        textcoords="offset points", fontsize=5.5, color=color, weight="bold",
                        ha="left", va="center", clip_on=False)

        ax.set_xlabel("Thought step $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {title_str}", fontsize=8, pad=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.2, ll_k + 2.8)
        ax.set_xticks(steps)

    # ─── Legend on the right ───
    fig.subplots_adjust(right=0.88)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.89, 0.5),
               ncol=1, fontsize=6, frameon=True, framealpha=0.8, edgecolor="#cccccc",
               handlelength=2.0, handletextpad=0.5)

    out_path = Path(out_dir) / f"epiphenomena_{model_family}_scratchpad.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Scratchpad Figure to: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Logit-Lens qualitative figures (ported from plot_logit_lens.py)
# Split: one standalone figure per family (no combined 2-col panel).
# ═══════════════════════════════════════════════════════════════════
FAMILY_LABELS = {"gpt2": "GPT-2", "llama": "Llama"}
_LL_TASKS = [("prosqa", "Graph-Hopping"), ("gsm", "Arithmetic-Reasoning")]


def _ll_load_json(data_dir, task, model, k=6):
    path = Path(data_dir) / f"{task}_{model}_k{k}.json"
    if not path.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _ll_load_ci_records(data_dir, task, model, k=6):
    return _load_jsonl(Path(data_dir) / f"{task}_{model}_k{k}_cis.jsonl")


def _extract_top1_tokens_gsm(instance, k):
    """Top-1 decoded token at each step from a GSM per_instance entry.

    JSON format: per_step[t].decoded = [(token_str, prob_str), ...]
    """
    tokens = []
    for step_data in instance["per_step"]:
        top1 = step_data["decoded"][0][0].strip()
        tokens.append(top1)
    return tokens[:k + 1]


def _pick_gsm_example(instances, k):
    """Pick a GSM instance with the most varied logit-lens structure."""
    best, best_score = None, -1
    for inst in instances[:50]:
        n_hits = sum(1 for s in inst["per_step"] if s.get("has_hit", False))
        n_unique = len(set(_extract_top1_tokens_gsm(inst, k)))
        score = n_hits + n_unique
        if score > best_score:
            best_score = score
            best = inst
    return best


def plot_logit_lens_examples(family, data_dir, out_dir, k=6):
    """Logit-lens decoded-token grid for ONE family.

    1 row x 2 cols: tasks (left: Graph-Hopping, right: Arithmetic-Reasoning).
    rows within each panel = models, columns = thought steps 0..k.
    Cell background:
        light blue  -> first token (always new)
        light green -> changed from previous step
        light gray  -> same as previous step (static/degenerate)
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6))

    def _plot_example_panel(ax, task, title):
        rows, row_labels = [], []
        for model in _MODES_ORDER:
            d = _ll_load_json(data_dir, task, model, k)
            if d is None:
                continue
            instances = d["per_instance"]
            if not instances:
                continue
            if task == "prosqa":
                inst = instances[0]
                tokens = inst["top1_tokens"][:k + 1]
            else:
                inst = _pick_gsm_example(instances, k)
                if inst is None:
                    continue
                tokens = _extract_top1_tokens_gsm(inst, k)
            rows.append(tokens)
            row_labels.append(_LABELS_SHORT[model])

        if not rows:
            ax.set_visible(False)
            return

        n_rows, n_cols = len(rows), len(rows[0])
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_aspect("auto")

        for i, (tokens, label) in enumerate(zip(rows, row_labels)):
            for j, tok in enumerate(tokens):
                if j == 0:
                    color = "#e8f4fd"
                elif tok == tokens[j - 1]:
                    color = "#f0f0f0"
                else:
                    color = "#d4edda"
                rect = plt.Rectangle(
                    (j - 0.48, i - 0.42), 0.96, 0.84,
                    facecolor=color, edgecolor="#cccccc", linewidth=0.3,
                )
                ax.add_patch(rect)
                display = tok if len(tok) <= 6 else tok[:5] + "…"
                ax.text(j, i, display, ha="center", va="center",
                        fontsize=5, fontfamily="monospace")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"$t_{{{t}}}$" for t in range(n_cols)], fontsize=6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_xlabel("Thought step", fontsize=7)
        ax.set_title(title, fontsize=8, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    for col, (task, task_label) in enumerate(_LL_TASKS):
        _plot_example_panel(axes[col], task, task_label)

    fig.tight_layout(w_pad=1.5)
    out_path = Path(out_dir) / f"logit_lens_examples_{family}.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


def _load_family_attn(data_dir, k):
    """Load attention-mass JSON + CI records for one family directory."""
    task_data = {"prosqa": {}, "gsm": {}}
    ci_by_task_model = {}
    for task in ("prosqa", "gsm"):
        for m in _MODES_ORDER:
            d = _ll_load_json(data_dir, task, m, k)
            if d is not None:
                task_data[task][m] = d
                ci_by_task_model[(task, m)] = _ll_load_ci_records(data_dir, task, m, k)
    return task_data, ci_by_task_model


def plot_attention_bars(family, data_dir, out_dir, k=6):
    """Attention-mass stacked bars for ONE family.

    1 row x 2 cols: tasks (left: Graph-Hopping, right: Arithmetic-Reasoning).
    Stacks = prompt / latent / self attention mass (%), with in-bar ±CI labels.
    """
    task_data, ci_by_task_model = _load_family_attn(data_dir, k)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2),
                             gridspec_kw={'wspace': 0.25})

    def _plot_stacked_attn(ax, task, td, panel_label, show_ylabel):
        attn_models = []
        prompt_vals, latent_vals, self_vals = [], [], []
        prompt_cis, latent_cis, self_cis = [], [], []
        for model in _MODES_ORDER:
            if model not in td:
                continue
            am = td[model].get("attention_mass") or td[model].get("attention")
            if am is not None:
                attn_models.append(model)
                prompt_vals.append(am.get("mean_prompt", 0) * 100)
                latent_vals.append(am.get("mean_latent", 0) * 100)
                self_vals.append(am.get("mean_self", 0) * 100)
                records = ci_by_task_model.get((task, model), [])
                ctx = {"condition": "end_token"}
                prompt_cis.append(_find_ci_record(records, "attn_prompt", ctx))
                latent_cis.append(_find_ci_record(records, "attn_latent", ctx))
                self_cis.append(_find_ci_record(records, "attn_self", ctx))

        x_bar = np.arange(len(attn_models))
        w = 0.6
        ax.bar(x_bar, prompt_vals, width=w, label="Prompt", color="#4e79a7", edgecolor="black", linewidth=0.3)
        ax.bar(x_bar, latent_vals, width=w, bottom=prompt_vals, label="Latent", color="#e15759", edgecolor="black", linewidth=0.3)
        bottoms = [p + l for p, l in zip(prompt_vals, latent_vals)]
        ax.bar(x_bar, self_vals, width=w, bottom=bottoms, label="Self", color="#f28e2b", edgecolor="black", linewidth=0.3)

        for i in range(len(x_bar)):
            if prompt_vals[i] > 4:
                ax.text(x_bar[i], prompt_vals[i] / 2, fmt_pm_pct(prompt_vals[i], prompt_cis[i]),
                        ha="center", va="center", fontsize=3.8, color="white")
            if latent_vals[i] > 4:
                ax.text(x_bar[i], prompt_vals[i] + latent_vals[i] / 2, fmt_pm_pct(latent_vals[i], latent_cis[i]),
                        ha="center", va="center", fontsize=3.8, color="white")
            if self_vals[i] > 4:
                ax.text(x_bar[i], bottoms[i] + self_vals[i] / 2, fmt_pm_pct(self_vals[i], self_cis[i]),
                        ha="center", va="center", fontsize=3.8, color="black")

        ax.set_xticks(x_bar)
        ax.set_xticklabels([_LABELS_SHORT[m] for m in attn_models])
        if show_ylabel:
            ax.set_ylabel("Attention mass (%)")
        ax.set_title(panel_label, fontsize=8, pad=14)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.96),
                  ncol=3, fontsize=5.5, frameon=False, handletextpad=0.4, columnspacing=1.2)

    panel_letters = {"prosqa": "(a)", "gsm": "(b)"}
    for col, (task, task_label) in enumerate(_LL_TASKS):
        title = f"{panel_letters[task]} {task_label}"
        _plot_stacked_attn(axes[col], task, task_data[task], title, show_ylabel=(col == 0))

    out_path = Path(out_dir) / f"attention_bars_{family}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Table Generation Logic
# ═══════════════════════════════════════════════════════════════════
def _bfs_header_block(modes, metric_specs):
    num_m = len(modes)
    multicol = ["", *[f"\\multicolumn{{{num_m}}}{{c}}{{{label}}}" for (label,) in metric_specs]]
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

def generate_appendix_main_table(results, modes, all_k, ci_data=None, family=""):
    ci_data = ci_data or {}
    num_m = len(modes)
    cols = "l " + " ".join(["c" * num_m] * 2)

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

    fam_suffix = f" ({family})" if family else ""
    latex = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
    ]
    latex += _bfs_header_block(modes, top_specs)
    latex += _bfs_data_rows(results, modes, all_k, ci_data, top_blocks)
    latex += ["\\midrule"]
    latex += _bfs_header_block(modes, bot_specs)
    latex += _bfs_data_rows(results, modes, all_k, ci_data, bot_blocks)
    latex += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{Full standard-probing results across timesteps and models{fam_suffix}.}}",
              f"\\label{{tab:appendix_bfs_probing_{family or 'combined'}}}",
              "\\end{table*}", ""]
    return "\n".join(latex)

def generate_table_candidate_parallelism(results, modes, all_k):
    num_m = len(modes)
    cols = "l " + " ".join(["ccc"] * num_m)
    latex = [
        "\\begin{table*}[h!]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Cumulative top-$k$ raw probabilities. T1 = top-1, T3 = top-3, $\\Delta$ = T3 $-$ T1. Coconut $u{=}0.0$ and Pause-as-thought show negligible gaps throughout, concentrating mass on a single candidate. Coconut $u{=}0.3$ shows substantial gaps at shallow $k$ (broad exploration) that narrow with depth.}",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
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
            t1, t3 = s.get("mean_top1_prob", float("nan")), s.get("mean_top3_cumul", float("nan"))
            if math.isnan(t1):
                row.append("- & - & -")
            else:
                row.append(f"{fmt_float(t1, 3)} & {fmt_float(t3, 3)} & {fmt_float(max(0, t3 - t1), 3)}")
        latex.append(" & ".join(row) + " \\\\")

    latex.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:candidate_parallelism}", "\\end{table*}", ""])
    return "\n".join(latex)

def generate_table_convergence_trajectories(results, modes):
    latex = [
        "\\begin{table}[h!]", "\\centering", "\\small",
        "\\caption{Convergence from $k{=}0$ to $k{=}4$. Pause-as-thought and Coconut $u{=}0.0$ show increasing P(correct) with decreasing entropy (exploration to convergence). Base, CoT, and Coconut $u{=}0.3$ show degrading or flat P(correct).}",
        "\\begin{tabular}{l cc cc}", "\\toprule",
        "& \\multicolumn{2}{c}{$P(\\text{correct})$} & \\multicolumn{2}{c}{$H/\\log_2 N$} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Condition & $k{=}0$ & $k{=}4$ & $k{=}0$ & $k{=}4$ \\\\",
        "\\midrule"
    ]
    for m in modes:
        s0, s4 = results[m]["summary"].get("0", {}), results[m]["summary"].get("4", {})
        row = [
            _LABELS_LONG[m],
            fmt_float(s0.get("mean_value_correct", float("nan")), 2),
            fmt_float(s4.get("mean_value_correct", float("nan")), 2),
            fmt_float(s0.get("mean_normalized_entropy", float("nan")), 2),
            fmt_float(s4.get("mean_normalized_entropy", float("nan")), 2)
        ]
        latex.append(" & ".join(row) + " \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:convergence_trajectories}", "\\end{table}", ""])
    return "\n".join(latex)

def generate_table_per_instance_statistics(results, modes):
    num_m = len(modes)
    cols = "l " + "c"*num_m
    latex = [
        "\\begin{table*}[h!]", "\\centering", "\\small",
        "\\caption{Per-instance analysis. ``P(correct) increases'' counts instances where the raw probability summed over correct candidates is higher at the final $k$ evaluated for that instance than at $k{=}0$. The close match between Coconut $u{=}0.0$ (61.5\\%) and Pause-as-thought (62.5\\%) reinforces that the curriculum drives the convergence pattern.}",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
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
            first, last = k_data[str(ks[0])]["metrics"], k_data[str(ks[-1])]["metrics"]
            if last.get("value_correct", 0) > first.get("value_correct", 0) + 0.01: n_inc += 1
            if last.get("top1_is_correct", False): n_top1 += 1
            if first.get("normalized_entropy", 0) > 0.5: n_ent += 1
                
        if n_total > 0:
            row_inc.append(f"{round((n_inc/n_total)*100)}\\%")
            row_top1.append(f"{round((n_top1/n_total)*100)}\\%")
            row_ent.append(f"{round((n_ent/n_total)*100)}\\%")
            row_n.append(str(n_total))
        else:
            row_inc.append("-"); row_top1.append("-"); row_ent.append("-"); row_n.append("-")
            
    latex.extend([" & ".join(row_inc) + " \\\\", " & ".join(row_top1) + " \\\\", " & ".join(row_ent) + " \\\\", "\\midrule", " & ".join(row_n) + " \\\\"])
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:per_instance_statistics}", "\\end{table*}"])
    return "\n".join(latex)

def generate_table_scratchpad_probing(results, modes, all_k, ci_data=None, family=""):
    ci_data = ci_data or {}

    def fmt_bounds(val, rec):
        if val is None or math.isnan(val): return "-"
        p = val * 100
        if rec is None: return f"{p:.1f}"
        lo, hi = rec["ci_low"] * 100, rec["ci_high"] * 100
        return f"{p:.1f} \\textsubscript{{({lo:.1f}, {hi:.1f})}}"

    latex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll ccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Step} & \\textbf{Hit Rate} & \\textbf{Superpos.} & \\textbf{Step Align.} \\\\",
        "\\midrule"
    ]

    labels = {"base": "B", "cot": "CoT", "pause": "PaT", "coconut": "C", "coconut_u": "$C_u$", "codi": "CODI"}

    # ── Per-Step Analysis ──
    for m in modes:
        if m not in results: continue
        for i, k in enumerate(all_k):
            model_col = labels.get(m, m) if i == 0 else ""

            s = results[m]["summary"].get(str(k), {})
            hr = s.get("hit_rate", float("nan"))
            sr = s.get("superposition_rate", float("nan"))
            ar = s.get("alignment_rate", float("nan"))

            ci_hr = _find_ci_record(ci_data.get(m, []), "hit_rate", {"condition": "per_step", "t": k})
            ci_sr = _find_ci_record(ci_data.get(m, []), "superposition_rate", {"condition": "per_step", "t": k})
            ci_ar = _find_ci_record(ci_data.get(m, []), "alignment_rate", {"condition": "per_step", "t": k})

            row = [
                model_col,
                str(k),
                fmt_bounds(hr, ci_hr),
                fmt_bounds(sr, ci_sr),
                fmt_bounds(ar, ci_ar)
            ]
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")

    # ── Pooled Section ──
    latex.append("\\multicolumn{5}{l}{\\textbf{Pooled (all steps)}} \\\\")
    for m in modes:
        if m not in results: continue
        ov = results[m].get("overall", {})
        hr = ov.get("hit_rate", float("nan"))
        sr = ov.get("superposition_rate", float("nan"))

        ci_oh = _find_ci_record(ci_data.get(m, []), "overall_hit_rate", {"condition": "pooled"})
        ci_os = _find_ci_record(ci_data.get(m, []), "overall_superposition_rate", {"condition": "pooled"})

        row = [
            labels.get(m, m),
            "--",
            fmt_bounds(hr, ci_oh),
            fmt_bounds(sr, ci_os),
            "--"
        ]
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    fam_str = "GPT-2" if family.lower() == "gpt2" else "Llama"
    latex.append(f"\\caption{{Scratchpad-Thinking probing full statistical results ({fam_str}).}}")
    latex.append(f"\\label{{tab:scratchpad_{family}}}")
    latex.append("\\end{table}\n")

    return "\n".join(latex)

# ═══════════════════════════════════════════════════════════════════
# Execution Pipeline
# ═══════════════════════════════════════════════════════════════════
def _load_data(data_dir, file_pattern):
    results = {}
    if not Path(data_dir).exists(): return results
    for f in sorted(Path(data_dir).glob(file_pattern)):
        if "_k" in f.stem:
            # e.g. "gsm_coconut_u_k6" -> task prefix "gsm_", trailing "_k6": mode = "coconut_u"
            task_prefix, _, rest = f.stem.partition("_")
            mode = re.sub(r"_k\d+$", "", rest)
        else:
            mode = f.stem.replace("results_", "")
        if mode in _MODES_ORDER:
            with open(f) as fh:
                results[mode] = json.load(fh)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="Plots/epiphenomena", help="Output directory for merged PDF")
    parser.add_argument("--tables_dir", type=str, default="Tables/statistical", help="Directory for LaTeX tables")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)

    for model_family in ("gpt2", "llama"):
        # 1. Load Superposition Data
        sup_dir = f"outputs/superposition/{model_family}"
        sup_results = _load_data(sup_dir, "results_*.json")
        sup_cis = {m: _load_jsonl(Path(sup_dir) / "ci" / f"superposition_{m}.jsonl") for m in sup_results}
        sup_k = [k for k in sorted({int(k) for mode in sup_results for k in sup_results[mode]["summary"]}) if k <= 4]

        # 2. Load Logit-Lens Data
        ll_k = 6
        ll_dir = f"outputs/logit_lens/{model_family}"
        ll_results = _load_data(ll_dir, f"gsm_*_k{ll_k}.json")
        ll_cis = {m: _load_jsonl(Path(ll_dir) / f"gsm_{m}_k{ll_k}_cis.jsonl") for m in ll_results}

        if not sup_results and not ll_results:
            print(f"[skip] {model_family}: no superposition or logit-lens data found")
            continue

        # 3. Generate Superposition & Scratchpad Figures (standalone, own legend each)
        if sup_results:
            plot_superposition_metrics(sup_results, sup_cis, sup_k, ll_k, args.out_dir, model_family)
        else:
            print(f"Missing superposition data to generate figure [{model_family}].")

        if ll_results:
            plot_scratchpad_metrics(ll_results, ll_cis, ll_k, args.out_dir, model_family)
        else:
            print(f"Missing logit-lens data to generate scratchpad figure [{model_family}].")

        # 3b. Generate qualitative logit-lens figures for this family
        plot_logit_lens_examples(model_family, ll_dir, args.out_dir, ll_k)
        plot_attention_bars(model_family, ll_dir, args.out_dir, ll_k)

        # 4. Generate Tables
        if sup_results:
            with open(f"{args.tables_dir}/superposition_{model_family}.tex", "w") as f:
                f.write(generate_appendix_main_table(sup_results, _MODES_ORDER, sup_k, sup_cis, family=model_family))
            with open(f"{args.tables_dir}/superposition_extended_{model_family}.tex", "w") as f:
                f.write(generate_table_candidate_parallelism(sup_results, _MODES_ORDER, sup_k) + "\n\n")
                f.write(generate_table_convergence_trajectories(sup_results, _MODES_ORDER) + "\n\n")
                f.write(generate_table_per_instance_statistics(sup_results, _MODES_ORDER) + "\n")
            print(f"Saved LaTeX tables to {args.tables_dir}/")

        # 5. Generate Scratchpad Probing Table (Logit Lens GSM)
        if ll_results:
            out_table = Path(args.tables_dir) / f"scratchpad_{model_family}.tex"
            with open(out_table, "w") as f:
                f.write(generate_table_scratchpad_probing(
                    results=ll_results,
                    modes=_MODES_ORDER,
                    all_k=list(range(ll_k + 1)),
                    ci_data=ll_cis,
                    family=model_family
                ))
            print(f"Saved Scratchpad LaTeX table to {out_table}")

if __name__ == "__main__":
    main()