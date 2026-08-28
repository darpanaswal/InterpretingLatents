"""
causal_trace_helper.py
======================

Reads the merged-trace artifacts produced by causal_trace.py or 
reverse_causal_trace.py depending on --mode:

  Restorative: outputs/causal_trace/<t>_<m>/trace_<t>_<m>_<corr>_<gran>.npz
  Disruptive:  outputs/reverse_causal_trace/<t>_<m>/rev_trace_<t>_<m>_<corr>_<gran>.npz

and produces:

  1. Three main-text heatmap figures (one per component channel):
       resid_pre.pdf, attn_out.pdf, mlp_out.pdf
     Each figure: 2 rows (Graph-Hopping on top, Arithmetic-Reasoning on
     bottom) × 4 columns (one per model).

  2. An appendix .tex file with full statistical tests per (task, model):
       Tables/statistical/causal_trace_<model_family>.tex

Usage:
    python causal_trace_helper.py --mode disruptive
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# ─── style (minimalist, publication-ready) ──────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.5,
    "text.usetex": False,
})

# ─── canonical orderings ─────────────────────────────────────────────
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
COMPONENT_NAMES = ("resid_pre", "attn_out", "mlp_out")

# 11-column display grid.
BUCKETS = [
    "P_full", "P_max", "P_b",
    "T_1", "T_2", "T_3", "T_4", "T_5", "T_6",
    "T_full", "A_b",
]
BUCKETS_TEX = [
    r"P$_{\text{full}}$", r"P$_{\max}$", r"P$_b$",
    "T$_1$", "T$_2$", "T$_3$", "T$_4$", "T$_5$", "T$_6$",
    r"T$_{\text{full}}$", r"A$_b$",
]


# ═════════════════════════════════════════════════════════════════════
# I/O
# ═════════════════════════════════════════════════════════════════════

def load_merged(in_dir: Path, task: str, model: str,
                corruption: str, granularity: str, mode: str):
    """Load merged trace_*.npz or rev_trace_*.npz. Returns NpzFile or None.

    causal_trace.py writes outputs/causal_trace/<family>/<task>_<model>/...
    so `in_dir` here is expected to be already family-scoped by the caller
    (i.e. <base>/<family>).
    """
    prefix = "trace" if mode == "restorative" else "rev_trace"
    path = (in_dir / f"{task}_{model}"
            / f"{prefix}_{task}_{model}_{corruption}_{granularity}.npz")
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


# ═════════════════════════════════════════════════════════════════════
# Position-grid resolution
# ═════════════════════════════════════════════════════════════════════

def _per_instance_grid(npz):
    labels_raw = npz["position_labels"]
    kinds_raw = npz["position_kinds"]

    shared = (isinstance(labels_raw, np.ndarray)
              and labels_raw.dtype != object
              and labels_raw.ndim == 1)

    N_inst = int(np.asarray(npz["clean_correct"]).shape[0])

    if shared:
        kinds = np.asarray(kinds_raw).tolist()
        labels = np.asarray(labels_raw).tolist()
        per_inst = list(zip(kinds, labels))
        return [per_inst] * N_inst

    out = []
    for i in range(N_inst):
        k_i = np.asarray(kinds_raw[i]).tolist()
        l_i = np.asarray(labels_raw[i]).tolist()
        out.append(list(zip(k_i, l_i)))
    return out


# ═════════════════════════════════════════════════════════════════════
# Per-instance content window & bucket tensors
# ═════════════════════════════════════════════════════════════════════

def _content_window(n_format_prefix, m, n_gen):
    start = int(n_format_prefix)
    end = min(start + int(m), int(n_gen))
    return start, end

def _bucket_position_indices(grid, bucket: str):
    if bucket == "P_full": return [j for j, (k, _) in enumerate(grid) if k == "joint_prompt"]
    if bucket == "T_full": return [j for j, (k, _) in enumerate(grid) if k == "joint_thought"]
    if bucket == "P_max":  return [j for j, (k, _) in enumerate(grid) if k == "prompt"]
    if bucket == "P_b":    return [j for j, (k, _) in enumerate(grid) if k == "prompt_boundary"]
    if bucket == "A_b":    return [j for j, (k, _) in enumerate(grid) if k == "answer_boundary"]
    if bucket.startswith("T_"):
        return [j for j, (k, lab) in enumerate(grid) if k == "thought" and lab == bucket]
    return []

def _per_instance_ie_per_bucket(npz, mode: str):
    clean_correct = np.asarray(npz["clean_correct"]).astype(bool)
    n_format_prefix = np.asarray(npz["n_format_prefix"]).astype(int)
    gold_tokens = npz["gold_tokens"]
    kl_clean_corr = np.asarray(npz["kl_clean_corr"], dtype=np.float64)
    instance_ids = np.asarray(npz["instance_ids"]).astype(int)
    grids = _per_instance_grid(npz)
    N_inst, N_gen = kl_clean_corr.shape

    kl_clean_patched_raw = npz["kl_clean_patched"]
    if kl_clean_patched_raw.dtype != object:
        kl_clean_patched_all = kl_clean_patched_raw.astype(np.float64)
    else:
        kl_clean_patched_all = kl_clean_patched_raw

    sample = np.asarray(kl_clean_patched_all[0], dtype=np.float64)
    L, C = sample.shape[0], sample.shape[1]

    out = {b: {"ie": [], "klp": []} for b in BUCKETS}
    klc_list = []
    iid_list = []

    n_total = N_inst
    n_drop_clean, n_drop_window, n_drop_klc, n_kept = 0, 0, 0, 0

    for i in range(N_inst):
        m_i = int(np.asarray(gold_tokens[i]).shape[0])
        s, e = _content_window(n_format_prefix[i], m_i, N_gen)
        if e <= s:
            n_drop_window += 1
            continue
        klc_i = float(kl_clean_corr[i, s:e].mean())
        if klc_i <= 0:
            n_drop_klc += 1
            continue

        n_kept += 1
        kl_p_i = np.asarray(kl_clean_patched_all[i], dtype=np.float64)
        kl_p_W = kl_p_i[..., s:e].mean(axis=-1)  

        for b in BUCKETS:
            idxs = _bucket_position_indices(grids[i], b)
            if not idxs:
                klp = np.full((L, C), np.nan, dtype=np.float64)
                ie = np.full((L, C), np.nan, dtype=np.float64)
            else:
                sub = kl_p_W[:, :, idxs]
                if b == "P_max":
                    klp = sub.min(axis=-1) if mode == "restorative" else sub.max(axis=-1)
                else:
                    klp = sub[:, :, 0] if sub.shape[-1] == 1 else sub.mean(axis=-1)
                
                if mode == "restorative":
                    ie = 1.0 - klp / klc_i
                else:
                    ie = np.clip(klp / klc_i, 0.0, 1.0)

            out[b]["ie"].append(ie)
            out[b]["klp"].append(klp)

        klc_list.append(klc_i)
        iid_list.append(int(instance_ids[i]))

    result = {}
    for b in BUCKETS:
        result[b] = {
            "ie":  np.stack(out[b]["ie"], axis=0) if out[b]["ie"] else np.zeros((0, L, C)),
            "klp": np.stack(out[b]["klp"], axis=0) if out[b]["klp"] else np.zeros((0, L, C)),
        }
    result["_meta"] = {
        "klc": np.array(klc_list, dtype=np.float64),
        "instance_ids": np.array(iid_list, dtype=np.int64),
        "n_kept": len(iid_list),
        "L": L, "C": C,
        "filter_breakdown": {
            "n_total": n_total, "n_drop_clean": n_drop_clean,
            "n_drop_window": n_drop_window, "n_drop_klc": n_drop_klc, "n_kept": n_kept,
        },
    }
    return result


# ═════════════════════════════════════════════════════════════════════
# Heatmap Aggregation & Stats
# ═════════════════════════════════════════════════════════════════════

def _build_heatmap_LCB(per_inst_bucket):
    L, C = per_inst_bucket["_meta"]["L"], per_inst_bucket["_meta"]["C"]
    B = len(BUCKETS)
    out = np.full((L, C, B), np.nan, dtype=np.float64)
    for bi, b in enumerate(BUCKETS):
        arr = per_inst_bucket[b]["ie"]
        if arr.size == 0: continue
        with np.errstate(invalid="ignore"):
            out[:, :, bi] = np.nanmedian(arr, axis=0)
    return out

def _bootstrap_mean_ci(values, n_boot=1000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0: return None
    point = float(v.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return point, float(np.percentile(means, 100*(alpha/2))), float(np.percentile(means, 100*(1-alpha/2)))

def _per_instance_bucket_max_klp(per_inst_bucket, bucket, mode):
    arr = per_inst_bucket[bucket]["klp"]
    if arr.size == 0: return np.array([], dtype=np.float64)
    return np.nanmin(arr.reshape(arr.shape[0], -1), axis=1) if mode == "restorative" else np.nanmax(arr.reshape(arr.shape[0], -1), axis=1)

def _per_instance_bucket_max_ie(per_inst_bucket, bucket):
    arr = per_inst_bucket[bucket]["ie"]
    if arr.size == 0: return np.array([], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", r"All-NaN slice encountered")
        return np.nanmax(arr.reshape(arr.shape[0], -1), axis=1)

def _paired_thought_prompt_ie(per_inst_bucket):
    THOUGHT_BUCKETS = ("T_full", "T_1", "T_2", "T_3", "T_4", "T_5", "T_6")
    PROMPT_BUCKETS = ("P_full", "P_max")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", r"All-NaN slice encountered")
        t_stack = np.stack([_per_instance_bucket_max_ie(per_inst_bucket, b) for b in THOUGHT_BUCKETS], axis=1)
        p_stack = np.stack([_per_instance_bucket_max_ie(per_inst_bucket, b) for b in PROMPT_BUCKETS], axis=1)
        return np.nanmax(t_stack, axis=1), np.nanmax(p_stack, axis=1)

def _wilcoxon_thought_vs_prompt(per_inst_bucket):
    thought, prompt = _paired_thought_prompt_ie(per_inst_bucket)
    mask = np.isfinite(thought) & np.isfinite(prompt)
    diff = thought[mask] - prompt[mask]
    if diff.size < 2 or np.allclose(diff, 0): return None
    try:
        res = stats.wilcoxon(thought[mask], prompt[mask], alternative="two-sided")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n_pairs": int(diff.size)}
    except ValueError: return None

def _mcnemar_at_tau(per_inst_bucket, tau):
    thought, prompt = _paired_thought_prompt_ie(per_inst_bucket)
    mask = np.isfinite(thought) & np.isfinite(prompt)
    t_succ, p_succ = (thought[mask] > tau).astype(int), (prompt[mask] > tau).astype(int)
    b, c = int(np.sum((t_succ == 1) & (p_succ == 0))), int(np.sum((t_succ == 0) & (p_succ == 1)))
    n_disc = b + c
    if n_disc == 0: return {"p_value": None, "b_a_only": b, "c_b_only": c, "n_pairs": int(mask.sum())}
    p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), n_disc, 0.5)))
    return {"p_value": p_value, "b_a_only": b, "c_b_only": c, "n_pairs": int(mask.sum())}


# ═════════════════════════════════════════════════════════════════════
# Main Heatmap Plotting
# ═════════════════════════════════════════════════════════════════════

def _panel_dynamic_range(mat, clip_pct):
    finite = mat[np.isfinite(mat)]
    if finite.size == 0: 
        return 0.0, 1.0
    
    real_min = float(np.nanmin(finite))
    # Clip outliers on both ends for robustness
    vmin = float(np.nanpercentile(finite, 100.0 - clip_pct))
    vmax = float(np.nanpercentile(finite, clip_pct))
    
    if not np.isfinite(vmax) or vmax == 0: 
        vmax = max(1e-5, float(np.nanmax(finite)))
        
    if not np.isfinite(vmin):
        vmin = real_min
        
    # Clamp insignificant negative lower bounds to 0 to prevent tick overlap
    if vmin >= -0.05:
        vmin = 0.0
        
    return vmin, vmax

def _panel_heatmap(ax, heat_LCB, model_label, *, component_idx, show_ylabel, show_xticks, clip_pct):
    mat = heat_LCB[:, component_idx, :]
    vmin, vmax = _panel_dynamic_range(mat, clip_pct)
    
    # Viridis is sequential, standard normalization maps directly to the data limits
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", norm=norm, interpolation="nearest")
    
    ax.set_title(model_label)
    ax.set_xticks(range(len(BUCKETS)))
    
    if show_xticks: 
        ax.set_xticklabels(BUCKETS_TEX, rotation=90, ha="center")
    else: 
        ax.set_xticklabels([])
        
    if show_ylabel: 
        ax.set_ylabel("Layer")
    
    n_L = mat.shape[0]
    ax.set_yticks([0, n_L // 2, n_L - 1])
    ax.set_yticklabels([0, n_L // 2, n_L - 1])
    return im, vmin, vmax

def build_main_figure(per_panel_data, out_pdf: Path, *, component_idx, clip_pct):
    n_models = len(MODELS)
    fig, axes = plt.subplots(nrows=2, ncols=n_models, figsize=(2.0 * n_models + 0.4, 3.8), sharey=False)
    if n_models == 1: axes = axes.reshape(2, 1)

    for r, (task, task_label) in enumerate(TASKS):
        for col, (model_dir, model_label) in enumerate(MODELS):
            ax = axes[r, col]
            entry = per_panel_data[task].get(model_dir)
            if entry is None:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=6, color="#888")
                ax.set_title(model_label)
                continue
            
            heat_LCB, n_kept = entry
            im, vmin, vmax = _panel_heatmap(
                ax, heat_LCB, model_label=f"{model_label}  (n={n_kept})",
                component_idx=component_idx, show_ylabel=(col == 0),
                show_xticks=(r == 1), clip_pct=clip_pct,
            )
            
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
            
            # Dynamic ticks based on clamped range
            if vmin < 0 < vmax:
                cb.set_ticks([vmin, 0.0, vmax])
                cb.set_ticklabels([f"{vmin:+.2g}", "0", f"{vmax:+.2g}"])
            else:
                mid = (vmin + vmax) / 2.0
                cb.set_ticks([vmin, mid, vmax])
                vmin_lbl = "0" if vmin == 0.0 else f"{vmin:.2g}"
                cb.set_ticklabels([vmin_lbl, f"{mid:.2g}", f"{vmax:.2g}"])

        axes[r, 0].annotate(task_label, xy=(-0.50, 0.5), xycoords="axes fraction", ha="center", va="center", rotation=90, fontsize=7)

    fig.subplots_adjust(wspace=0.55, hspace=0.45)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Main heatmap: {out_pdf}")


# ═════════════════════════════════════════════════════════════════════
# Appendix .tex
# ═════════════════════════════════════════════════════════════════════

def _ci_cell(ci_tuple) -> str: return f"{ci_tuple[0]:.3f} [{ci_tuple[1]:.3f}, {ci_tuple[2]:.3f}]" if ci_tuple else "--"
def _stars(pv): return "" if pv is None else (r"$^{***}$" if pv < 0.001 else (r"$^{**}$" if pv < 0.01 else (r"$^{*}$" if pv < 0.05 else "")))
def _wilcoxon_cell(r): return f"{r['p_value']:.4f}{_stars(r['p_value'])}" if r and r.get("p_value") else "--"
def _mcnemar_cell(r):
    if not r or not r.get("p_value"): return f"-- ({r['b_a_only'] if r else '?'},{r['c_b_only'] if r else '?'})"
    return f"{r['p_value']:.4f}{_stars(r['p_value'])} ({r['b_a_only']},{r['c_b_only']})"

PROMPT_BUCKETS = ["P_full", "P_max", "P_b", "A_b"]
THOUGHT_BUCKETS = ["T_1", "T_2", "T_3", "T_4", "T_5", "T_6", "T_full"]
PROMPT_BUCKETS_TEX = [t for b, t in zip(BUCKETS, BUCKETS_TEX) if b in PROMPT_BUCKETS]
THOUGHT_BUCKETS_TEX = [t for b, t in zip(BUCKETS, BUCKETS_TEX) if b in THOUGHT_BUCKETS]

FAMILY_LABELS = {"gpt2": "GPT-2", "llama": "Llama-3.2"}

def build_appendix_table(stats_per_panel, *, corruption, granularity, tau, mode, family):
    title_text = "Full causal tracing results" if mode == "restorative" else "Full reverse causal tracing results"
    lbl_prefix = "causal_trace" if mode == "restorative" else "rev_causal_trace"
    family_label = FAMILY_LABELS.get(family, family)

    # ── prompt-positions table ──────────────────────────────────────
    n_prompt = len(PROMPT_BUCKETS)
    prompt_lines = [
        r"\begin{table*}[h!]", r"\centering", r"\tiny", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{l ccc c cc cc r}", r"\toprule",
        ("Model & " + " & ".join(PROMPT_BUCKETS_TEX) + r" & $\overline{\text{KL}_{c\to c'}}$ "
         + r"& Wilc.\ $p$ & McN.\ $p$ ($\tau{=}" + f"{tau}" + r"$) & $n$ \\"),
    ]
    for task, task_label in TASKS:
        prompt_lines += [r"\midrule", rf"\multicolumn{{{n_prompt + 5}}}{{c}}{{\textbf{{{task_label}}}}} \\", r"\midrule"]
        for model_dir, model_label in MODELS:
            entry = stats_per_panel[task].get(model_dir)
            if entry is None:
                prompt_lines.append(f"{model_label} & " + " & ".join(["--"] * (n_prompt + 3)) + r" & -- \\")
                continue
            prompt_lines.append(
                f"{model_label} & " + " & ".join([_ci_cell(entry["bucket_ci"][b]) for b in PROMPT_BUCKETS])
                + f" & {_ci_cell(entry['klc_ref'])} & {_wilcoxon_cell(entry['wilcoxon'])} & {_mcnemar_cell(entry['mcnemar'])} & {entry['n_kept']} \\\\"
            )
    prompt_lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{title_text} with statistical testing (prompt-positions) for the {family_label} model family.}}",
        rf"\label{{tab:{lbl_prefix}_prompt_{family}}}",
        r"\end{table*}",
    ]

    # ── thought-positions table ─────────────────────────────────────
    n_thought = len(THOUGHT_BUCKETS)
    thought_lines = [
        r"\begin{table*}[h!]", r"\centering", r"\tiny", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{l " + "c" * n_thought + r"}", r"\toprule",
        ("Model & " + " & ".join(THOUGHT_BUCKETS_TEX) + r" \\"),
    ]
    for task, task_label in TASKS:
        thought_lines += [r"\midrule", rf"\multicolumn{{{n_thought + 1}}}{{c}}{{\textbf{{{task_label}}}}} \\", r"\midrule"]
        for model_dir, model_label in MODELS:
            entry = stats_per_panel[task].get(model_dir)
            if entry is None:
                thought_lines.append(f"{model_label} & " + " & ".join(["--"] * n_thought) + r" \\")
                continue
            thought_lines.append(
                f"{model_label} & " + " & ".join([_ci_cell(entry["bucket_ci"][b]) for b in THOUGHT_BUCKETS]) + r" \\"
            )
    thought_lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{title_text} with statistical testing (thought-positions) for the {family_label} model family.}}",
        rf"\label{{tab:{lbl_prefix}_thought_{family}}}",
        r"\end{table*}",
    ]

    return "\n".join(prompt_lines).strip() + "\n\n" + "\n".join(thought_lines).strip()


# ═════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════

def process_all(in_dir, corruption, granularity, tau, clip_pct, mode):
    plot_data, stats_data, filter_data, raw_data = ({t: {} for t, _ in TASKS} for _ in range(4))

    for task, _ in TASKS:
        for model_dir, _ in MODELS:
            npz = load_merged(in_dir, task, model_dir, corruption, granularity, mode)
            if npz is None:
                for d in (plot_data, stats_data, filter_data, raw_data): d[task][model_dir] = None
                continue

            per_inst = _per_instance_ie_per_bucket(npz, mode)
            raw_data[task][model_dir] = per_inst
            n_kept = per_inst["_meta"]["n_kept"]
            
            plot_data[task][model_dir] = (_build_heatmap_LCB(per_inst), n_kept)
            filter_data[task][model_dir] = per_inst["_meta"]["filter_breakdown"]

            bucket_ci = {b: _bootstrap_mean_ci(_per_instance_bucket_max_klp(per_inst, b, mode)) for b in BUCKETS}
            stats_data[task][model_dir] = {
                "bucket_ci": bucket_ci, "klc_ref": _bootstrap_mean_ci(per_inst["_meta"]["klc"]),
                "wilcoxon": _wilcoxon_thought_vs_prompt(per_inst), "mcnemar": _mcnemar_at_tau(per_inst, tau), "n_kept": n_kept,
            }

    return plot_data, stats_data, filter_data, raw_data

def print_filter_breakdown(filter_data):
    print("\n[filter breakdown]")
    print(f"  {'task':<8} {'model':<10} {'total':>6} {'kept':>6} {'!clean':>7} {'win=0':>6} {'KLc<=0':>7}")
    for task, _ in TASKS:
        for model_dir, _ in MODELS:
            fb = filter_data[task].get(model_dir)
            if not fb: print(f"  {task:<8} {model_dir:<10} {'--':>6} (no shard)")
            else: print(f"  {task:<8} {model_dir:<10} {fb['n_total']:>6} {fb['n_kept']:>6} {fb['n_drop_clean']:>7} {fb['n_drop_window']:>6} {fb['n_drop_klc']:>7}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def discover_families(base_dir: Path) -> list:
    """Families present on disk under <base>/<family>/<task>_<model>/."""
    found = set()
    if base_dir.is_dir():
        for child in base_dir.iterdir():
            if child.is_dir():
                found.add(child.name)
    known = [f for f in ("gpt2", "llama") if f in found]
    extra = sorted(found - set(known))
    return known + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["restorative", "disruptive"], default="restorative")
    ap.add_argument("--in_dir", type=str, default=None)
    ap.add_argument("--corruption", default="symbol_swap")
    ap.add_argument("--granularity", choices=["single", "window"], default="single")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--clip_pct", type=float, default=98.0)
    ap.add_argument("--out_fig_dir", type=str, default=None)
    ap.add_argument("--out_tex_dir", type=str, default="Tables/statistical")
    args = ap.parse_args()

    default_base_in = "outputs/causal_trace" if args.mode == "restorative" else "outputs/reverse_causal_trace"
    default_base_fig = "Plots/causal_trace" if args.mode == "restorative" else "Plots/reverse_causal_trace"
    
    base_in  = Path(args.in_dir) if args.in_dir else Path(default_base_in)
    base_fig = Path(args.out_fig_dir) if args.out_fig_dir else Path(default_base_fig)
    out_tex_dir = Path(args.out_tex_dir)
    out_tex_dir.mkdir(parents=True, exist_ok=True)
    prefix = "causal_trace" if args.mode == "restorative" else "rev_causal_trace"

    families = discover_families(base_in)
    if not families:
        print(f"[WARN] No families found under {base_in}")
        return
    print(f"[INFO] Families: {families}")

    for family in families:
        in_dir = base_in / family
        
        # Ensure the base figure directory exists (no family sub-directory)
        base_fig.mkdir(parents=True, exist_ok=True)

        plot_data, stats_data, filter_data, raw_data = process_all(
            in_dir, args.corruption, args.granularity, tau=args.tau, clip_pct=args.clip_pct, mode=args.mode
        )
        print(f"\n[{family}]")
        print_filter_breakdown(filter_data)

        # 1. Main Heatmaps (namespaced by family in the filename)
        for ci, comp in enumerate(COMPONENT_NAMES):
            out_pdf = base_fig / f"{comp}_{family}.pdf"
            build_main_figure(plot_data, out_pdf, component_idx=ci, clip_pct=args.clip_pct)

        # 2. Appendix LaTeX Table — one per family (both tasks): <prefix>_<family>.tex
        tex = build_appendix_table(
            stats_data, corruption=args.corruption, granularity=args.granularity,
            tau=args.tau, mode=args.mode, family=family,
        )
        tex_path = out_tex_dir / f"{prefix}_{family}.tex"
        tex_path.write_text(tex)
        print(f"[OK] LaTeX Table: {tex_path}")


if __name__ == "__main__":
    main()