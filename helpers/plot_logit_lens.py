"""
Plot logit-lens and attention results from stored JSON outputs.

Figure 1 (GSM8k): 2x2 grid — step-wise metrics across models.
    Row 1: Thought Attention (bar), Hit Rate (line)
    Row 2: Superposition (line), Step Alignment (line)

Figure 2 (ProsQA): Split panel showing logit-lens decoded tokens
    for Pause (degenerate) vs CoT (evolving) as qualitative examples.

Usage:
    python -m experiments.dead_salmon.plot_logit_lens
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Style: ICML single-column (3.25in wide), tight, readable
# ═══════════════════════════════════════════════════════════════════

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
# Model styling
# ═══════════════════════════════════════════════════════════════════

# Model display names matching the paper
MODEL_LABELS = {
    "base": "B",
    "cot": "CoT",
    "pause": "PaT",
    "coconut": "C",
    "coconut_u": r"$C_u$",
    "codi": "CODI",
}

# Consistent colors and markers per model
MODEL_STYLE = {
    "base":      {"color": "#999999", "marker": "x",  "ls": ":"},
    "cot":       {"color": "#E69F00", "marker": "^",  "ls": "--"},
    "pause":     {"color": "#0072B2", "marker": "o",  "ls": "-"},
    "coconut":   {"color": "#D55E00", "marker": "s",  "ls": "-"},
    "coconut_u": {"color": "#009E73", "marker": "D",  "ls": "-"},
    "codi":      {"color": "#CC79A7", "marker": "v",  "ls": "-."},
}

# Which models to include in each figure
GSM_MODELS = ["base", "cot", "pause", "coconut", "coconut_u", "codi"]


def load_json(data_dir, task, model, k=6):
    path = Path(data_dir) / f"{task}_{model}_k{k}.json"
    if not path.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    if not Path(path).exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_ci_records(data_dir, task, model, k=6):
    return load_jsonl(Path(data_dir) / f"{task}_{model}_k{k}_cis.jsonl")


def _context_matches(record, expected):
    ctx = record.get("_context", {})
    return all(ctx.get(key) == value for key, value in expected.items())


def find_ci_record(ci_records, metric, context=None):
    """
    Return the most recent matching CI record.

    The experiment scripts append JSONL records, so the last match is the one
    corresponding to the most recent run when a file has been regenerated.
    """
    context = context or {}
    for record in reversed(ci_records):
        if record.get("metric") == metric and _context_matches(record, context):
            return record
    return None


def ci_half_width_pct(record):
    if record is None:
        return None
    return (record["ci_high"] - record["ci_low"]) * 100 / 2


def fmt_pm_pct(point_pct, record=None, decimals=1, latex=False):
    if record is None:
        return f"{point_pct:.{decimals}f}"
    half_width = ci_half_width_pct(record)
    if latex:
        return (
            rf"${point_pct:.{decimals}f}"
            rf" {{\scriptstyle \pm {half_width:.{decimals}f}}}$"
        )
    return f"{point_pct:.{decimals}f}±{half_width:.{decimals}f}"


def fmt_ci_pct(record, decimals=1):
    if record is None:
        return "--"
    p = record["point"] * 100
    lo = record["ci_low"] * 100
    hi = record["ci_high"] * 100
    return f"{p:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_n(record, fallback=None):
    if record is not None and record.get("n") is not None:
        return str(record["n"])
    return str(fallback) if fallback is not None else "--"


def fmt_point_pct(point, decimals=1):
    if point is None:
        return "--"
    return f"{point * 100:.{decimals}f}"


def fmt_point_or_ci_pct(point, record=None, decimals=1):
    if record is not None:
        return fmt_ci_pct(record, decimals)
    return fmt_point_pct(point, decimals)


def latex_escape(value):
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: GSM8k step-wise metrics (2 rows × 2 cols)
# ═══════════════════════════════════════════════════════════════════

def _annotate_endpoint(ax, x, y, fmt="{:.0f}", fontsize=5, color="black",
                       ha="left", offset=(2, 0)):
    """
    Place a small value label at (x, y) with pixel offset.

    Args:
        ax: matplotlib axes
        x, y: data coordinates
        fmt: format string for the value (applied to y)
        fontsize: label font size
        color: text color
        ha: horizontal alignment
        offset: (dx, dy) in points from the data point
    """
    ax.annotate(
        fmt.format(y), xy=(x, y),
        xytext=offset, textcoords="offset points",
        fontsize=fontsize, color=color, ha=ha, va="center",
    )


def plot_gsm_stepwise(data_dir, out_dir, k=6):
    """
    Layout (2 rows × 2 cols, full ICML width):
        [Thought Attention (bar)]  [Hit Rate (line)]
        ─────────── shared legend (horizontal, ncol=6) ───────────
        [Superposition (line)]     [Step Alignment (line)]

    Annotation strategy:
        - Bar chart: value label above each bar
        - Line plots: value label at the rightmost point (t=k),
          placed to the right of the marker so the line stays clean.
          For B (flat at 0) we skip the label to avoid clutter.
    """

    # ── Load data ──
    all_data = {}
    for model in GSM_MODELS:
        d = load_json(data_dir, "gsm", model, k)
        if d is not None:
            all_data[model] = d
    ci_data = {
        model: load_ci_records(data_dir, "gsm", model, k)
        for model in all_data
    }

    steps = list(range(k + 1))  # 0..6

    # summary[str(t)] = {hit_rate, superposition_rate, alignment_rate, n}
    def get_step_series(model, key):
        summary = all_data[model]["summary"]
        return [summary[str(t)][key] for t in steps]

    # ── Create figure ──
    # gridspec_kw.hspace leaves vertical room for the shared legend
    fig, axes = plt.subplots(
        2, 2, figsize=(6.75, 3.6),
        gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )

    # Collect one handle set for the shared legend (built from axes[0,1])
    legend_handles = []
    legend_labels = []

    # ────────────────────────────────────────────────────────
    # (0,0) Thought Attention — stacked bar chart with value labels
    # ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    attn_models = []
    prompt_vals, latent_vals, self_vals = [], [], []
    prompt_cis, latent_cis, self_cis = [], [], []
    
    for model in GSM_MODELS:
        if model not in all_data:
            continue
        # Fallback to check either "attention_mass" or "attention" key
        am = all_data[model].get("attention_mass") or all_data[model].get("attention")
        if am is not None:
            attn_models.append(model)
            prompt_vals.append(am.get("mean_prompt", 0) * 100)
            latent_vals.append(am.get("mean_latent", 0) * 100)
            self_vals.append(am.get("mean_self", 0) * 100)
            records = ci_data.get(model, [])
            ctx = {"condition": "end_token"}
            prompt_cis.append(find_ci_record(records, "attn_prompt", ctx))
            latent_cis.append(find_ci_record(records, "attn_latent", ctx))
            self_cis.append(find_ci_record(records, "attn_self", ctx))

    x_bar = np.arange(len(attn_models))
    w = 0.6

    ax.bar(x_bar, prompt_vals, width=w, label="Prompt", color="#4e79a7", edgecolor="black", linewidth=0.3)
    ax.bar(x_bar, latent_vals, width=w, bottom=prompt_vals, label="Latent", color="#e15759", edgecolor="black", linewidth=0.3)
    bottoms = [p + l for p, l in zip(prompt_vals, latent_vals)]
    ax.bar(x_bar, self_vals, width=w, bottom=bottoms, label="Self", color="#f28e2b", edgecolor="black", linewidth=0.3)

    # Value labels centered inside each segment (skip if segment is too small)
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
    ax.set_xticklabels([MODEL_LABELS[m] for m in attn_models])
    ax.set_ylabel("Attention mass (%)")
    ax.set_title("(a) Thought Attention", fontsize=8)
    ax.set_ylim(0, 105)
    
    # Add a small legend for the segments inside the subplot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=5.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2
    )
    ax.set_title("(a) Thought Attention", fontsize=8, pad=16)

    # ────────────────────────────────────────────────────────
    # Helper: plot a line-panel with right-endpoint annotation
    # ────────────────────────────────────────────────────────
    def _plot_line_panel(ax, key, title_str, ylim, ylabel, panel_label,
                         skip_below=None):
        """
        Plot one line-panel for all models.

        Annotates the value at t=k (rightmost point) to the right
        of the marker.  Skips annotation for B (flat near zero)
        to avoid clutter.

        Args:
            skip_below: if set, suppress endpoint labels for values
                        below this threshold (in %). Useful when many
                        models converge to near-zero and labels pile up.

        The annotation offset alternates vertically for models
        whose endpoints are close together, preventing overlap.
        """
        # Collect endpoint values for stagger calculation
        endpoints = []  # (y_val_pct, model)
        for model in GSM_MODELS:
            if model not in all_data:
                continue
            vals = get_step_series(model, key)
            pcts = [v * 100 for v in vals]
            s = MODEL_STYLE[model]
            line, = ax.plot(
                steps, pcts,
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=MODEL_LABELS[model], markeredgewidth=0.3,
                markeredgecolor="black",
            )
            endpoints.append((pcts[-1], model, s["color"]))

        # Sort endpoints by y-value to detect overlap
        endpoints.sort(key=lambda e: e[0])

        # Annotate right endpoint for each model
        # Use data-coordinate staggering: ensure labels are ≥ min_gap apart
        y_range = ylim[1] - ylim[0]
        min_gap = 0.08 * y_range  # minimum vertical gap between labels
        placed = []  # list of placed label y-positions (in data coords)
        for y_val, model, color in endpoints:
            if model == "base":
                continue  # skip flat-zero baseline label
            if skip_below is not None and y_val < skip_below:
                continue  # suppress near-zero labels to avoid pile-up

            # Find a label position that doesn't collide
            label_y = y_val
            for py in placed:
                if abs(label_y - py) < min_gap:
                    label_y = py + min_gap
            placed.append(label_y)

            # dy_offset in points: convert data offset to approximate points
            # At 7pt font, the axes height is ~1.2in ≈ 86pt for these panels
            ax_height_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
            pts_per_data = ax_height_pts / y_range
            dy_pts = (label_y - y_val) * pts_per_data
            ci = find_ci_record(
                ci_data.get(model, []),
                key,
                {"condition": "per_step", "t": int(k)},
            )

            ax.annotate(
                fmt_pm_pct(y_val, ci), xy=(k, y_val),
                xytext=(4, dy_pts),
                textcoords="offset points",
                fontsize=4.5, color=color, ha="left", va="center",
            )

        ax.set_xlabel("Thought step $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {title_str}", fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.2, k + 1.0)   # <-- Add this to add padding on the right
        ax.set_xticks(steps)

    # ────────────────────────────────────────────────────────
    # (0,1) Hit Rate
    # ────────────────────────────────────────────────────────
    _plot_line_panel(axes[0, 1], "hit_rate",
                     "Intermediate Hit Rate", (-2, 82),
                     "Hit rate (%)", "(b)")

    # ────────────────────────────────────────────────────────
    # (1,0) Superposition
    # ────────────────────────────────────────────────────────
    _plot_line_panel(axes[1, 0], "superposition_rate",
                     "Superposition Rate", (-1, 30),
                     "Superposition (%)", "(c)")

    # ────────────────────────────────────────────────────────
    # (1,1) Step Alignment
    # ────────────────────────────────────────────────────────
    _plot_line_panel(axes[1, 1], "alignment_rate",
                     "Step Alignment", (-2, 68),
                     "Step alignment (%)", "(d)",
                     skip_below=1.0)

    # ────────────────────────────────────────────────────────
    # Shared legend: centered between the two rows
    # ────────────────────────────────────────────────────────
    # Grab handles from axes[0,1] (the first line-plot panel)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.48),  # vertically between the two rows
        ncol=len(labels)/2,
        fontsize=6,
        frameon=True,
        framealpha=0.8,
        edgecolor="#cccccc",
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    out_path = Path(out_dir) / "gsm_stepwise.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 2: ProsQA logit-lens examples (qualitative)
#
# For each model, pick one representative instance and show
# the top-1 decoded token at each thought step as a heatmap row.
# Degenerate models show a single repeated token; evolving models
# show changing tokens.
# ═══════════════════════════════════════════════════════════════════

PROSQA_MODELS = ["base", "cot", "pause", "coconut", "coconut_u", "codi"]


def _extract_top1_tokens_gsm(instance, k):
    """
    Extract top-1 decoded token at each step from a GSM per_instance entry.

    JSON format: per_step[t].decoded = [(token_str, prob_str), ...]
    The token_str may have leading whitespace from the tokenizer.
    """
    tokens = []
    for step_data in instance["per_step"]:
        # decoded is list of [token_str, prob_str] pairs
        top1 = step_data["decoded"][0][0].strip()
        tokens.append(top1)
    return tokens[:k + 1]


def _pick_gsm_example(instances, k):
    """
    Pick a GSM instance that shows interesting logit-lens structure:
    prefer one with hits at multiple steps (shows the alternation pattern).
    """
    best = None
    best_score = -1
    for inst in instances[:50]:  # only scan first 50 saved
        n_hits = sum(1 for s in inst["per_step"] if s.get("has_hit", False))
        n_unique = len(set(_extract_top1_tokens_gsm(inst, k)))
        score = n_hits + n_unique
        if score > best_score:
            best_score = score
            best = inst
    return best


def plot_logit_lens_examples(data_dir, out_dir, k=6):
    """
    Grid: rows = models, columns = thought steps 0..k.
    Each cell shows the top-1 decoded token as text.
    Cell background:
        - light blue:  first token (always "new")
        - light green: changed from previous step
        - light gray:  same as previous step (static/degenerate)

    Two panels side by side: ProsQA (left) and GSM8k (right).
    """

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4))

    for panel_idx, (task, title) in enumerate([
        ("prosqa", "Graph-Hopping"),
        ("gsm", "Arithmetic-Reasoning"),
    ]):
        ax = axes[panel_idx]

        rows = []
        row_labels = []
        for model in PROSQA_MODELS:
            d = load_json(data_dir, task, model, k)
            if d is None:
                continue

            instances = d["per_instance"]
            if not instances:
                continue

            if task == "prosqa":
                # ProsQA stores top1_tokens directly
                inst = instances[0]
                tokens = inst["top1_tokens"][:k + 1]
            else:
                # GSM: pick an instance with varied decoded tokens
                inst = _pick_gsm_example(instances, k)
                if inst is None:
                    continue
                tokens = _extract_top1_tokens_gsm(inst, k)

            rows.append(tokens)
            row_labels.append(MODEL_LABELS[model])

        if not rows:
            ax.set_visible(False)
            continue

        n_rows = len(rows)
        n_cols = len(rows[0])

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_aspect("auto")

        for i, (tokens, label) in enumerate(zip(rows, row_labels)):
            for j, tok in enumerate(tokens):
                # Color logic:
                #   first step -> blue (always "new")
                #   same as previous -> gray (static/degenerate)
                #   different from previous -> green (evolving)
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

                # Truncate long tokens; use monospace for token text
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

    fig.tight_layout(w_pad=1.5)
    out_path = Path(out_dir) / "logit_lens_examples.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 3 (optional): ProsQA attention mass bar chart
# ═══════════════════════════════════════════════════════════════════

def plot_prosqa_attention(data_dir, out_dir, k=6):
    """
    Stacked bar chart: prompt vs latent vs self attention mass
    for each model on ProsQA. Single-column width.
    """
    fig, ax = plt.subplots(figsize=(3.25, 1.8))

    models_with_attn = []
    prompt_vals, latent_vals, self_vals = [], [], []
    prompt_cis, latent_cis, self_cis = [], [], []

    for model in PROSQA_MODELS:
        d = load_json(data_dir, "prosqa", model, k)
        if d is None or d.get("attention") is None:
            continue
        attn = d["attention"]
        records = load_ci_records(data_dir, "prosqa", model, k)
        ctx = {"condition": "end_token"}
        models_with_attn.append(model)
        prompt_vals.append(attn["mean_prompt"] * 100)
        latent_vals.append(attn["mean_latent"] * 100)
        self_vals.append(attn["mean_self"] * 100)
        prompt_cis.append(find_ci_record(records, "attn_prompt", ctx))
        latent_cis.append(find_ci_record(records, "attn_latent", ctx))
        self_cis.append(find_ci_record(records, "attn_self", ctx))

    x = np.arange(len(models_with_attn))
    w = 0.55

    ax.bar(x, prompt_vals, w, label="Prompt", color="#4e79a7", edgecolor="black", linewidth=0.3)
    ax.bar(x, latent_vals, w, bottom=prompt_vals, label="Latent",
           color="#e15759", edgecolor="black", linewidth=0.3)
    bottoms = [p + l for p, l in zip(prompt_vals, latent_vals)]
    ax.bar(x, self_vals, w, bottom=bottoms, label="Self",
           color="#f28e2b", edgecolor="black", linewidth=0.3)
    # --- NEW CODE: Add numbers inside segments ---
    for i in range(len(x)):
        if prompt_vals[i] > 4:
            ax.text(x[i], prompt_vals[i] / 2, fmt_pm_pct(prompt_vals[i], prompt_cis[i]),
                    ha="center", va="center", fontsize=3.8, color="white")
        if latent_vals[i] > 4:
            ax.text(x[i], prompt_vals[i] + latent_vals[i] / 2, fmt_pm_pct(latent_vals[i], latent_cis[i]),
                    ha="center", va="center", fontsize=3.8, color="white")
        if self_vals[i] > 4:
            ax.text(x[i], bottoms[i] + self_vals[i] / 2, fmt_pm_pct(self_vals[i], self_cis[i]),
                    ha="center", va="center", fontsize=3.8, color="black")
    # ---------------------------------------------

    ax.set_xticks(x)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_with_attn])
    ax.set_ylabel("Attention mass (%)")
    ax.set_title("ProsQA: Attention Distribution", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=5.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2
    )
    ax.set_title("ProsQA: Attention Distribution", fontsize=8, pad=16)

    fig.tight_layout()
    out_path = Path(out_dir) / "prosqa_attention.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)

def plot_line_metrics(data_dir, out_dir, k=6):
    """Figure 1: 3-panel line metrics."""
    gsm_data = {}
    for model in GSM_MODELS:
        d = load_json(data_dir, "gsm", model, k)
        if d is not None: gsm_data[model] = d
    gsm_cis = {
        model: load_ci_records(data_dir, "gsm", model, k)
        for model in gsm_data
    }

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 1.35), gridspec_kw={'wspace': 0.28})

    def _plot_line_panel(ax, key, title_str, ylim, ylabel, panel_label,
                         skip_below=None, annotate="endpoint_right"):
        steps = list(range(k + 1))
        endpoints = []
        peaks = []
        starts = []
        
        for model in GSM_MODELS:
            if model not in gsm_data: continue
            vals = [gsm_data[model]["summary"][str(t)][key] for t in steps]
            pcts = [v * 100 for v in vals]
            s = MODEL_STYLE[model]
            ax.plot(steps, pcts, color=s["color"], marker=s["marker"], ls=s["ls"], 
                    label=MODEL_LABELS[model], markeredgewidth=0.3, markeredgecolor="black")
            
            endpoints.append((pcts[-1], model, s["color"]))
            peak_t = int(np.argmax(pcts))
            peaks.append((pcts[peak_t], peak_t, model, s["color"]))
            starts.append((pcts[0], 0, model, s["color"]))

        y_range = ylim[1] - ylim[0]
        min_gap = 0.10 * y_range  

        def _stagger(items, value_idx=0):
            items = sorted(items, key=lambda e: e[value_idx])
            placed = []
            out = []
            for item in items:
                y_val = item[value_idx]
                label_y = y_val
                for py in placed:
                    if abs(label_y - py) < min_gap:
                        label_y = py + min_gap
                label_y = min(label_y, ylim[1] - 0.03 * y_range)
                placed.append(label_y)
                out.append((*item, label_y))
            return out

        line_extend = 0.5  
        
        if annotate == "endpoint_right":
            for y_val, model, color, label_y in _stagger(endpoints):
                if model == "base": continue
                if skip_below is not None and y_val < skip_below: continue

                ci = find_ci_record(
                    gsm_cis.get(model, []),
                    key,
                    {"condition": "per_step", "t": int(k)},
                )

                x_pts = [k, k + 0.15, k + 0.4, k + line_extend]
                y_pts = [y_val, y_val, label_y, label_y]
                ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)

                ax.annotate(fmt_pm_pct(y_val, ci), xy=(k + line_extend, label_y), xytext=(3, 0),
                            textcoords="offset points", fontsize=5.5, color=color, weight="bold",
                            ha="left", va="center", clip_on=False)
                            
        elif annotate == "peak_right":
            for y_val, peak_t, model, color, label_y in _stagger(peaks):
                if model == "base": continue
                if skip_below is not None and y_val < skip_below: continue
                
                ci = find_ci_record(
                    gsm_cis.get(model, []),
                    key,
                    {"condition": "per_step", "t": int(peak_t)},
                )
                
                x_pts = [peak_t, k + 0.15, k + 0.4, k + line_extend]
                y_pts = [y_val, y_val, label_y, label_y]
                ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)
                
                ax.annotate(
                    fmt_pm_pct(y_val, ci), xy=(k + line_extend, label_y), xytext=(3, 0),
                    textcoords="offset points", fontsize=5.5, color=color, weight="bold",
                    ha="left", va="center", clip_on=False,
                )

        elif annotate == "start_right":
            for y_val, start_t, model, color, label_y in _stagger(starts):
                if model == "base": continue
                if skip_below is not None and y_val < skip_below: continue
                
                ci = find_ci_record(
                    gsm_cis.get(model, []),
                    key,
                    {"condition": "per_step", "t": int(start_t)},
                )
                
                x_pts = [start_t, k + 0.15, k + 0.4, k + line_extend]
                y_pts = [y_val, y_val, label_y, label_y]
                ax.plot(x_pts, y_pts, color=color, ls=(0, (1.2, 1.8)), linewidth=0.65, alpha=0.75, zorder=0)
                
                ax.annotate(
                    fmt_pm_pct(y_val, ci), xy=(k + line_extend, label_y), xytext=(3, 0),
                    textcoords="offset points", fontsize=5.5, color=color, weight="bold",
                    ha="left", va="center", clip_on=False,
                )

        ax.set_xlabel("Thought step $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {title_str}", fontsize=8, pad=6)
        ax.set_ylim(*ylim)
        
        left_pad = -0.2 
        right_pad = 2.8 
        ax.set_xlim(left_pad, k + right_pad)
        ax.set_xticks(steps)

    _plot_line_panel(axes[0], "hit_rate", "Intermediate Hit Rate", (-2, 82), "Hit rate (%)", "(a)", annotate="endpoint_right")
    _plot_line_panel(axes[1], "superposition_rate", "Superposition Rate", (-1, 30), "Superposition (%)", "(b)", annotate="endpoint_right")
    _plot_line_panel(axes[2], "alignment_rate", "Step Alignment", (-2, 68),
                     "Step alignment (%)", "(c)", skip_below=1.0,
                     annotate="start_right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.25), 
               ncol=6, fontsize=6, frameon=True, framealpha=0.8, edgecolor="#cccccc", 
               handlelength=2.0, columnspacing=1.2, handletextpad=0.5)

    out_path = Path(out_dir) / "intermediate_decodability.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_attention_bars(data_dir, out_dir, k=6):
    """Figure 2: 2-panel attention mass (Vertically stacked)."""
    gsm_data = {}
    for m in GSM_MODELS:
        d = load_json(data_dir, "gsm", m, k)
        if d is not None:
            gsm_data[m] = d
    prosqa_data = {}
    for m in PROSQA_MODELS:
        d = load_json(data_dir, "prosqa", m, k)
        if d is not None:
            prosqa_data[m] = d
    ci_by_task_model = {
        ("gsm", m): load_ci_records(data_dir, "gsm", m, k)
        for m in gsm_data
    }
    ci_by_task_model.update({
        ("prosqa", m): load_ci_records(data_dir, "prosqa", m, k)
        for m in prosqa_data
    })

    # 2 rows, 1 column. Width adjusted to 3.25 (standard ICML single column)
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), gridspec_kw={'hspace': 1.0})

    def _plot_stacked_attn(ax, task, task_data, task_title, panel_label):
        attn_models = []
        prompt_vals, latent_vals, self_vals = [], [], []
        prompt_cis, latent_cis, self_cis = [], [], []
        for model in GSM_MODELS:  
            if model not in task_data: continue
            am = task_data[model].get("attention_mass") or task_data[model].get("attention")
            if am is not None:
                attn_models.append(model)
                prompt_vals.append(am.get("mean_prompt", 0) * 100)
                latent_vals.append(am.get("mean_latent", 0) * 100)
                self_vals.append(am.get("mean_self", 0) * 100)
                records = ci_by_task_model.get((task, model), [])
                ctx = {"condition": "end_token"}
                prompt_cis.append(find_ci_record(records, "attn_prompt", ctx))
                latent_cis.append(find_ci_record(records, "attn_latent", ctx))
                self_cis.append(find_ci_record(records, "attn_self", ctx))

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
        ax.set_xticklabels([MODEL_LABELS[m] for m in attn_models])
        ax.set_ylabel("Attention mass (%)")
        ax.set_title(f"{panel_label} {task_title}", fontsize=8, pad=14)
        ax.set_ylim(0, 105)
        
        # Legend applied directly to the axes, placed right above the title
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), 
                  ncol=3, fontsize=5.5, frameon=False, handletextpad=0.4, columnspacing=1.2)

    # ProsQA on top, GSM on bottom
    _plot_stacked_attn(axes[0], "prosqa", prosqa_data, "Graph-Hopping: Attention", "(a)")
    _plot_stacked_attn(axes[1], "gsm", gsm_data, "Arithmetic-Reasoning: Attention", "(b)")

    out_path = Path(out_dir) / "attention_bars.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Appendix tables
# ═══════════════════════════════════════════════════════════════════

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm": "Arithmetic-Reasoning",
}


def _collect_available(data_dir, k=6):
    data = {}
    cis = {}
    for task in ["prosqa", "gsm"]:
        models = PROSQA_MODELS if task == "prosqa" else GSM_MODELS
        data[task] = {}
        cis[task] = {}
        for model in models:
            d = load_json(data_dir, task, model, k)
            if d is None:
                continue
            data[task][model] = d
            cis[task][model] = load_ci_records(data_dir, task, model, k)
    return data, cis


def build_appendix_tables(data_dir, k=6):
    data, cis = _collect_available(data_dir, k)
    lines = [
        r"\subsection{Logit-Lens and Attention Analyses}",
        r"\label{app:logit_lens_stats}",
        "",
        r"Point estimates and 95\% bootstrap CIs "
        r"(1{,}000 percentile resamples) for the logit-lens and "
        r"attention diagnostics. Attention mass is measured at the "
        r"decoder-entry/end-latent token. GSM8k pooled metrics aggregate "
        r"over all thought steps.",
        "",
    ]

    # Attention mass table across both tasks.
    lines += [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Logit-lens attention-mass CIs.}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Model & Prompt [\% CI] & Latent [\% CI] & Self [\% CI] & $n$ \\",
        r"\midrule",
    ]
    
    for t_idx, task in enumerate(["prosqa", "gsm"]):
        models = PROSQA_MODELS if task == "prosqa" else GSM_MODELS
        
        # Pre-filter valid models to determine the multirow span
        valid_models = []
        for model in models:
            if model in data[task]:
                am = data[task][model].get("attention_mass") or data[task][model].get("attention")
                if am is not None:
                    valid_models.append(model)
        
        if not valid_models:
            continue
            
        # Insert a midrule between different task blocks
        if t_idx > 0:
            lines.append(r"\midrule")
            
        for m_idx, model in enumerate(valid_models):
            am = data[task][model].get("attention_mass") or data[task][model].get("attention")
            records = cis[task].get(model, [])
            ctx = {"condition": "end_token"}
            prompt = find_ci_record(records, "attn_prompt", ctx)
            latent = find_ci_record(records, "attn_latent", ctx)
            self_attn = find_ci_record(records, "attn_self", ctx)
            n_fallback = data[task][model].get("n") or data[task][model].get("n_instances")
            n = fmt_n(prompt or latent or self_attn, n_fallback)
            
            # Apply multirow to the first row of the block, leave blank for others
            if m_idx == 0:
                task_col = rf"\multirow{{{len(valid_models)}}}{{*}}{{{TASK_LABELS[task]}}}"
            else:
                task_col = ""
                
            lines.append(
                f"{task_col} & {MODEL_LABELS[model]} & "
                f"{fmt_point_or_ci_pct(am.get('mean_prompt'), prompt)} & "
                f"{fmt_point_or_ci_pct(am.get('mean_latent'), latent)} & "
                f"{fmt_point_or_ci_pct(am.get('mean_self'), self_attn)} & {n} \\\\"
            )
            
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:logit_lens_attention_stats}",
        r"\end{table}",
        "",
    ]

    if data.get("gsm"):
        # GSM pooled table.
        lines += [
            r"\begin{table}[h!]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\caption{Arithmetic-Reasoning pooled logit-lens CIs.}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & Any intermediate hit [\% CI] & Superposition [\% CI] & $n$ \\",
            r"\midrule",
        ]
        for model in GSM_MODELS:
            if model not in data["gsm"]:
                continue
            records = cis["gsm"].get(model, [])
            ctx = {"condition": "pooled"}
            hit = find_ci_record(records, "overall_hit_rate", ctx)
            sup = find_ci_record(records, "overall_superposition_rate", ctx)
            n = fmt_n(hit or sup, data["gsm"][model].get("n_instances"))
            overall = data["gsm"][model].get("overall", {})
            lines.append(
                f"{MODEL_LABELS[model]} & "
                f"{fmt_point_or_ci_pct(overall.get('hit_rate'), hit)} & "
                f"{fmt_point_or_ci_pct(overall.get('superposition_rate'), sup)} & {n} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:logit_lens_gsm_pooled_stats}",
            r"\end{table}",
            "",
        ]

    return "\n".join(lines).strip()

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs/logit_lens",
                        help="Path to outputs/logit_lens/ containing JSON files")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for PDFs (default: same as data_dir)")
    parser.add_argument("--out_appendix", type=str, default="tables/tab_logit_lens_stats.tex",
                        help="Output path for appendix LaTeX statistical tables")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Figure 1: CODI superposition control")
    print("=" * 60)
    plot_line_metrics(args.data_dir, out_dir, args.k)
    
    print("=" * 60)
    print("Figure 2: Attention-mass analyses")
    print("=" * 60)
    plot_attention_bars(args.data_dir, out_dir, args.k)

    print("=" * 60)
    print("Figure 3: Logit-lens decoded token examples")
    print("=" * 60)
    plot_logit_lens_examples(args.data_dir, out_dir, args.k)

    print("=" * 60)
    print("Appendix: Logit-lens statistical tables")
    print("=" * 60)
    appendix_tex = build_appendix_tables(args.data_dir, args.k)
    out_appendix = Path(args.out_appendix)
    out_appendix.parent.mkdir(parents=True, exist_ok=True)
    out_appendix.write_text(appendix_tex)
    print(f"Saved: {out_appendix.resolve()}")

    # print("\n" + "=" * 60)
    # print("Figure 3: ProsQA attention distribution")
    # print("=" * 60)
    # plot_prosqa_attention(args.data_dir, out_dir, args.k)


if __name__ == "__main__":
    main()
