"""
Standalone plotting for markovianity_test.py results.

Reads ``out_dir/results_{model}_{task}[_subspace|_subspace_pred].json`` and
CI logs. Produces, per family:
    Plots/markov/markov_heatmaps_<family>.pdf       -- full thoughts + GOLD
                                                        subspace (unchanged)
    Plots/markov/markov_heatmaps_<family>_pred.pdf  -- full thoughts + PRED
                                                        (predicted-token)
                                                        subspace
    Tables/statistical/markovianity_<family>.tex    -- appendix table, GOLD
                                                        subspace only (the
                                                        pred subspace does
                                                        NOT get its own
                                                        table)
"""

import csv
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "markovianity"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_TEX = "Tables/statistical/markovianity_stats.tex"

SUMMARY_ORDER = 1
METRIC = "r2_uniform"

MODEL_LABELS = {
    "pause":     "PaT",
    "coconut":   "C",
    "coconut_u": r"C$_u$",
    "codi":      "CODI",
}

MODEL_ORDER = ["pause", "coconut", "coconut_u", "codi"]
TASK_ORDER = ["prosqa", "gsm"]

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}

PREDICTOR_KEYS = [
    "identity_baseline",
    "mean_baseline",
    "linear_shared",
    "mlp_shared",
]

PREDICTOR_LABELS = {
    "mean_baseline": "Mean",
    "identity_baseline": "Identity",
    "linear_shared": "Linear",
    "mlp_shared": "MLP",
}

# The CI keys in jsonl corresponding to the above predictors at SUMMARY_ORDER
def _get_ci_key(predictor, metric, order):
    if predictor == "mean_baseline":
        return f"o{order}_mean_baseline_{metric}" # Doesn't exist currently but handled gracefully
    elif predictor == "identity_baseline":
        return f"o{order}_identity_{metric}"
    elif predictor == "linear_shared":
        return f"o{order}_linear_shared_{metric}"
    elif predictor == "mlp_shared":
        # Usually pooled variant is what we care about or seed0
        return f"o{order}_mlp_shared_{metric}_pooled"
    return f"o{order}_{predictor}_{metric}"

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _subspace_source_of(res: dict) -> str:
    """Which subspace a projected result used.

    New results carry an explicit "subspace_source" ("gold"/"pred").
    Legacy projected results predate that field and are always gold.
    """
    src = res.get("subspace_source")
    if src in ("gold", "pred"):
        return src
    return "gold"


def _ci_suffix(proj: bool, source: str) -> str:
    """Filename suffix mirroring markovianity_test._proj_suffix."""
    if not proj:
        return ""
    return "_subspace" if source == "gold" else "_subspace_pred"


def discover_results(out_dir: Path, subspace_source: str = "gold"):
    """
    Returns data[is_projected][task][model] = {
        "point": {predictor: value},
        "ci": {predictor: (low, high)}
    }

    The projected (True) slot is populated ONLY by results whose subspace
    source matches `subspace_source` ("gold" or "pred"), so the appendix
    table's "Subspace" row shows the requested subspace. The unprojected
    (False) slot is source-independent.

    markovianity_test.py writes outputs/markovianity/<family>/results_<model>_<task>[suffix].json,
    so `out_dir` is expected to be already family-scoped by the caller.
    """
    data = {False: defaultdict(dict), True: defaultdict(dict)}

    # Initialize all keys
    for proj in [False, True]:
        for t in TASK_ORDER:
            for m in MODEL_ORDER:
                data[proj][t][m] = {"point": {}, "ci": {}, "n_test": None}

    # Iterate over existing JSONs
    for path in out_dir.glob("results_*.json"):
        res = load_json(path)
        if res is None: continue

        task = res["task"]
        model = res["model"]
        proj = res.get("projected", False)

        # For projected results, keep only the requested subspace source.
        source = _subspace_source_of(res)
        if proj and source != subspace_source:
            continue

        # Suffix for CI file must match the source-aware results suffix.
        suffix = _ci_suffix(proj, source)
        ci_path = out_dir / f"cis_{model}_{task}{suffix}.jsonl"
        ci_records = load_jsonl(ci_path)
        
        order_data = res["orders"].get(str(SUMMARY_ORDER))
        if not order_data:
            order_data = res["orders"].get(SUMMARY_ORDER)
        
        if order_data:
            data[proj][task][model]["n_test"] = res.get("n_test")
            for p in PREDICTOR_KEYS:
                # Point estimate
                p_data = order_data.get(p)
                if p_data and METRIC in p_data:
                    data[proj][task][model]["point"][p] = p_data[METRIC]
                
                # CI estimate
                ci_key = _get_ci_key(p, METRIC, SUMMARY_ORDER)
                ci_match = None
                for rec in reversed(ci_records):
                    if rec.get("metric") == ci_key:
                        ci_match = rec
                        break
                
                # Fallback for MLP if pooled is missing (might use seed0)
                if ci_match is None and p == "mlp_shared":
                    ci_key_seed0 = _get_ci_key(p, METRIC, SUMMARY_ORDER).replace("_pooled", "_seed0")
                    for rec in reversed(ci_records):
                        if rec.get("metric") == ci_key_seed0:
                            ci_match = rec
                            break

                if ci_match:
                    data[proj][task][model]["ci"][p] = (ci_match["ci_low"], ci_match["ci_high"])

    return data


def plot_heatmaps(data, out_path, subspace_source="gold"):
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.0))

    sub_label = ("Gradient-Subspace" if subspace_source == "gold"
                 else "Gradient-Subspace (pred)")
    proj_rows = [(False, "Full Thoughts"), (True, sub_label)]
    
    vmin, vmax = -0.1, 1.0
    
    for row_idx, (proj, row_title) in enumerate(proj_rows):
        for col_idx, task in enumerate(TASK_ORDER):
            ax = axes[row_idx, col_idx]
            
            inert_models = []
            active_models = []
            
            for model in MODEL_ORDER:
                cell_data = data[proj][task].get(model, {})
                pts = {p: cell_data.get("point", {}).get(p) for p in PREDICTOR_KEYS}
                pts = {p: v for p, v in pts.items() if v is not None}
                if not pts:
                    continue
                
                # Account for MLP overfitting: if identity/mean is close to linear/mlp, prefer inert
                best_inert = max(pts.get("identity_baseline", -float('inf')), pts.get("mean_baseline", -float('inf')))
                best_active = max(pts.get("linear_shared", -float('inf')), pts.get("mlp_shared", -float('inf')))
                
                OVERFIT_MARGIN = 0.1
                if best_inert >= best_active - OVERFIT_MARGIN:
                    inert_models.append(model)
                else:
                    active_models.append(model)
                    
            plot_models = inert_models + active_models
            M = np.full((len(PREDICTOR_KEYS), len(plot_models)), np.nan)
            
            for m_idx, model in enumerate(plot_models):
                cell_data = data[proj][task].get(model, {})
                for p_idx, p in enumerate(PREDICTOR_KEYS):
                    val = cell_data.get("point", {}).get(p)
                    if val is not None:
                        M[p_idx, m_idx] = val

            im = ax.imshow(M, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
            
            ax.set_yticks(range(len(PREDICTOR_KEYS)))
            if col_idx == 0:
                ax.set_yticklabels([PREDICTOR_LABELS[p] for p in PREDICTOR_KEYS], fontsize=8)
                ax.set_ylabel(row_title, fontsize=9, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
                
            if row_idx == 0:
                ax.set_title(TASK_LABELS[task], fontsize=10, fontweight="bold")
                
            ax.set_xticks(range(len(plot_models)))
            ax.set_xticklabels([MODEL_LABELS[m] for m in plot_models], fontsize=8)
            
            # Add separating line and group labels
            if inert_models and active_models:
                sep_x = len(inert_models) - 0.5
                ax.axvline(sep_x, color='black', linewidth=1.5)
                
                ax.text((len(inert_models) - 1) / 2.0, len(PREDICTOR_KEYS) + 0.2, "Static", ha="center", va="top", fontsize=9, fontweight="bold", color="dimgrey")
                ax.text(len(inert_models) + (len(active_models) - 1) / 2.0, len(PREDICTOR_KEYS) + 0.2, "Evolving", ha="center", va="top", fontsize=9, fontweight="bold", color="dimgrey")
            elif inert_models:
                ax.text((len(inert_models) - 1) / 2.0, len(PREDICTOR_KEYS) + 0.2, "Static", ha="center", va="top", fontsize=9, fontweight="bold", color="dimgrey")
            elif active_models:
                ax.text((len(active_models) - 1) / 2.0, len(PREDICTOR_KEYS) + 0.2, "Evolving", ha="center", va="top", fontsize=9, fontweight="bold", color="dimgrey")

    plt.subplots_adjust(left=0.15, right=0.88, wspace=0.1, hspace=0.6, bottom=0.15)
    cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax, label=rf"$R^2$ (Test)")
    
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

def build_appendix_table(data, family, out_path, subspace_source="gold"):
    # LaTeX formatting functions
    def _fmt_ci(pt, ci):
        if pt is None: return "--"
        if ci is None: return f"{pt:.3f}"
        return f"{pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    sub_row_label = ("Subspace" if subspace_source == "gold"
                     else "Subspace(pred)")
    src_caption = ("gold-answer gradient subspace" if subspace_source == "gold"
                   else "predicted-token gradient subspace")
    src_tag = "" if subspace_source == "gold" else "_pred"

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Graph-Hopping}} & \multicolumn{4}{c}{\textbf{Arithmetic-Reasoning}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"Proj. & Model & " + " & ".join([PREDICTOR_LABELS[p] for p in PREDICTOR_KEYS]) + " & " + " & ".join([PREDICTOR_LABELS[p] for p in PREDICTOR_KEYS]) + r" \\",
        r"\midrule",
    ]

    proj_rows = [(False, "Full"), (True, sub_row_label)]

    for p_idx, (proj, proj_label) in enumerate(proj_rows):
        for m_idx, model in enumerate(MODEL_ORDER):
            row_cells = []
            if m_idx == 0:
                row_cells.append(rf"\multirow{{{len(MODEL_ORDER)}}}{{*}}{{{proj_label}}}")
            else:
                row_cells.append("")
            row_cells.append(MODEL_LABELS[model])

            for task in TASK_ORDER:
                cell_data = data[proj][task].get(model, {})
                for p in PREDICTOR_KEYS:
                    pt = cell_data.get("point", {}).get(p)
                    ci = cell_data.get("ci", {}).get(p)
                    row_cells.append(_fmt_ci(pt, ci))

            lines.append(" & ".join(row_cells) + r" \\")

        if p_idx < len(proj_rows) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Markovianity $R^2$ at order 1, test-set with 95\% "
        rf"bootstrap CIs ({family}). Projected row uses the {src_caption}.}}",
        rf"\label{{tab:markovianity_{family}{src_tag}}}",
        r"\end{table}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex ] {out_path}")


def discover_families() -> list:
    """Families present on disk under outputs/markovianity/<family>/."""
    found = set()
    if OUT_DIR.is_dir():
        for child in OUT_DIR.iterdir():
            if child.is_dir() and child.name != "figures":
                found.add(child.name)
    known = [f for f in ("gpt2", "llama") if f in found]
    extra = sorted(found - set(known))
    return known + extra


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables_dir", default="Tables/statistical",
                    help="Directory for markovianity_<family>.tex files.")
    args = ap.parse_args()

    families = discover_families()
    if not families:
        print(f"[WARN] No families found under {OUT_DIR}")
        return
    print(f"[INFO] Families: {families}")

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("Plots/markov")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        data_dir = OUT_DIR / family

        # Two heatmaps per family: full+gold (unchanged) and full+pred (new).
        # Each subspace source gets its own `data` pull since discover_results
        # only fills the projected slot for the requested source.
        for subspace_source in ("gold", "pred"):
            data = discover_results(data_dir, subspace_source=subspace_source)
            src_tag = "" if subspace_source == "gold" else "_pred"
            plot_heatmaps(
                data, plot_dir / f"markov_heatmaps_{family}{src_tag}.pdf",
                subspace_source=subspace_source,
            )

        # Appendix table: gold subspace only, unchanged from before (no
        # pred-subspace table is produced).
        data_gold = discover_results(data_dir, subspace_source="gold")
        out_path = tables_dir / f"markovianity_{family}.tex"
        build_appendix_table(data_gold, family, out_path,
                             subspace_source="gold")


if __name__ == "__main__":
    main()