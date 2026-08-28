"""
Standalone LaTeX-table generation for gradient_subspace_diagnosis.py + bases.npz.

Reads, for each subspace source (gold and pred, independently):
    OUT_DIR / <family> / <task> / <model> / diagnosis.json   (from gradient_subspace_geometry.py;
                                                                includes q4_norm_budget_pooled,
                                                                the ||h^c||/||h|| retained fraction)
    OUT_DIR / <family> / <task> / <model> / bases.npz        (from gradient_subspace.py / _predtoken.py)
where OUT_DIR is outputs/gradient_geometry (gold) or
outputs/gradient_geometry_predtoken (pred). A source is skipped entirely
if its directory has no diagnosis.json files (e.g. gradient_subspace_geometry.py
was never run with --subspace_source pred).

Writes
    Tables/statistical/
        subspace_geometry_{family}.tex             -- gold per-family appendix tables
        subspace_geometry_{family}_pred.tex         -- pred per-family appendix tables

The cos^2(B_t, B_{t+1}) subspace-stability panels are plotted by
plot_gradient_geometry.py, not here.

All input and output paths are hardcoded; no CLI flags.
"""

import json
import numpy as np
from pathlib import Path
from src.config import BASE_DIR

# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "gradient_geometry"
OUT_DIR_PRED = BASE_DIR / "outputs" / "gradient_geometry_predtoken"
TABLE_DIR = Path("Tables/statistical")

# Subspace sources to discover + tabulate: (label, source dir, filename suffix).
# Gold keeps the legacy unsuffixed filenames; pred sits alongside with
# "_pred" so nothing gets overwritten.
SOURCES = [
    ("gold", OUT_DIR, ""),
    ("pred", OUT_DIR_PRED, "_pred"),
]

# Recurrent models: their 7th timestep (index 6) has rank 0 and
# must be excluded when computing averages.
RECURRENT_MODELS = {"coconut", "coconut_u", "codi"}

MODEL_LABELS = {
    "coconut":   "C",
    "coconut_u": r"C$_u$",
    "pause":     "PaT",
    "codi":      "CODI",
}

MODEL_ORDER = ["pause", "coconut", "coconut_u", "codi"]
TASK_ORDER = ["prosqa", "gsm"]
FAMILY_ORDER = ["gpt2", "llama"]

FAMILY_LABELS = {
    "gpt2":  "GPT-2",
    "llama": "Llama-3.2",
}


def sort_family_key(family):
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return 99

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}

def sort_model_key(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return 99

def sort_task_key(task):
    try:
        return TASK_ORDER.index(task)
    except ValueError:
        return 99

# ═══════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════

def discover_results(out_dir):
    """
    Walk out_dir / <family> / <task> / <model> / and load both
    diagnosis.json and bases.npz when both are present. diagnosis.json
    (written by gradient_subspace_geometry.py) carries the q4 norm-budget
    fields directly, so no separate summary file is needed.
    """
    results = []
    for diag_path in sorted(out_dir.glob("*/*/*/diagnosis.json")):
        with open(diag_path, "r") as f:
            r = json.load(f)

        family = r.get("model_family")
        if family is None:
            family = diag_path.parent.parent.parent.name

        bases_path = diag_path.parent / "bases.npz"
        bases = None
        if bases_path.exists():
            blob = np.load(bases_path)
            bases = {}
            for key in blob.files:
                if not key.startswith("B_t"):
                    continue
                t = int(key[len("B_t"):])
                bases[t] = blob[key]

        results.append({
            "family":    family,
            "task":      r["task"],
            "model":     r["model"],
            "diagnosis": r,
            "bases":     bases,
        })
    return results


def _active_ranks(ranks, model):
    """Return ranks for active timesteps only (skip t=6 for recurrent models)."""
    if model in RECURRENT_MODELS:
        return [r for r in ranks[:-1] if r > 0]
    return [r for r in ranks if r > 0]


# ═══════════════════════════════════════════════════════════════════
# Per-family appendix table
# ═══════════════════════════════════════════════════════════════════

def build_per_family_geometry_table(family, family_results, source="gold", suffix="") -> str:
    """Build one appendix table for a single model family.

    `source`/`suffix` distinguish the gold-subspace table (default,
    unsuffixed, unchanged from before) from the predicted-token-subspace
    table (source="pred", suffix="_pred") so captions/labels don't collide
    when both are generated.
    """
    sorted_results = sorted(
        family_results,
        key=lambda r: (sort_task_key(r["task"]), sort_model_key(r["model"])),
    )
    if not sorted_results:
        return ""

    T = sorted_results[0]["diagnosis"]["T"]
    family_label = FAMILY_LABELS.get(family, family)
    subspace_label = ("gold-answer" if source == "gold"
                       else "model-predicted-token")

    t_headers = " & ".join([f"$k_{{{t}}}$" for t in range(T)])
    col_spec = "ll" + "r" * T + "r" + "c" + "c" + "r"

    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        (rf"\caption{{Gradient-subspace (predicted-token) diagnostics across "
         rf"tasks for the {family_label} model family.}}"
         if source == "pred" else
         rf"\caption{{Gradient-subspace diagnostics across tasks for the "
         rf"{family_label} model family.}}"),
        rf"\label{{tab:subspace_geometry_{family}{suffix}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (f"Task & Model & {t_headers} & $\\bar k$ "
         r"& Adj.\ $\overline{\cos^2}$ "
         r"& Off-diag.\ $\overline{\cos^2}$ "
         r"& $\|h^c\|/\|h\|$" + r" \\"),
        r"\midrule",
    ]

    for t_idx, task in enumerate(TASK_ORDER):
        task_results = [r for r in sorted_results if r["task"] == task]
        if not task_results:
            continue
        if t_idx > 0:
            lines.append(r"\midrule")

        n_models = len(task_results)
        task_str = TASK_LABELS.get(task, task).replace("-", r"-\\")
        task_cell = (rf"\multirow{{{n_models}}}{{*}}"
                     rf"{{\makecell[l]{{{task_str}}}}}")

        for m_idx, r in enumerate(task_results):
            model = r["model"]
            d = r["diagnosis"]
            ranks = d["subspace_ranks"]

            active = _active_ranks(ranks, model)
            mean_k = float(np.mean(active)) if active else 0.0

            q3_adj = d.get("q3_adjacent_per_t", [])
            q3_adj_active = (q3_adj[:-1]
                             if model in RECURRENT_MODELS and q3_adj
                             else q3_adj)
            q3_adj_mean = (float(np.mean(q3_adj_active))
                           if q3_adj_active else float("nan"))

            q3_offdiag_mean = d.get("q3_offdiag_mean", float("nan"))
            q3_offdiag_ci = d.get("q3_offdiag_ci", [None, None])

            q4_pooled = d.get("q4_norm_budget_pooled", {})
            q4_mean = q4_pooled.get("mean", float("nan"))

            rank_cells = [
                (r"\textcolor{gray}{--}" if rk == 0 else str(rk))
                for rk in ranks
            ]

            q3_adj_str = (f"${q3_adj_mean:.3f}$"
                          if not np.isnan(q3_adj_mean) else "--")

            if q3_offdiag_ci[0] is not None and q3_offdiag_ci[1] is not None:
                hw = (q3_offdiag_ci[1] - q3_offdiag_ci[0]) / 2
                q3_od_str = (rf"${q3_offdiag_mean:.3f}"
                             rf" {{\scriptstyle \pm {hw:.3f}}}$")
            else:
                q3_od_str = f"${q3_offdiag_mean:.3f}$"

            q4_str = (f"${q4_mean:.3f}$"
                      if not np.isnan(q4_mean) else "--")

            first_col = task_cell if m_idx == 0 else ""
            cells = [
                first_col,
                MODEL_LABELS.get(model, model),
                *rank_cells,
                f"{mean_k:.1f}",
                q3_adj_str,
                q3_od_str,
                q4_str,
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def build_all_family_geometry_tables(results, source="gold", suffix="") -> dict:
    """Return {family: latex_str} for every family present in results."""
    families = sorted(
        {r["family"] for r in results}, key=sort_family_key
    )
    out = {}
    for family in families:
        fam_results = [r for r in results if r["family"] == family]
        out[family] = build_per_family_geometry_table(
            family, fam_results, source=source, suffix=suffix)
    return out


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    any_found = False
    for source, out_dir, suffix in SOURCES:
        if not out_dir.exists():
            print(f"[skip] {source}: {out_dir} does not exist")
            continue
        results = discover_results(out_dir)
        if not results:
            print(f"[skip] {source}: no diagnosis.json found under "
                  f"{out_dir}/*/*/*/")
            continue
        any_found = True

        # ── Per-family geometry tables (both gold and pred subspaces) ────
        family_tables = build_all_family_geometry_tables(
            results, source=source, suffix=suffix)
        for family, tex in family_tables.items():
            if not tex:
                continue
            p = TABLE_DIR / f"subspace_ranks_{family}{suffix}.tex"
            p.write_text(tex)
            print(f"[tex ] Geometry table [{source}/{family}] -> {p.resolve()}")

    if not any_found:
        raise SystemExit(
            f"No diagnosis.json found under {OUT_DIR}/*/*/*/ or "
            f"{OUT_DIR_PRED}/*/*/*/."
        )


if __name__ == "__main__":
    main()
