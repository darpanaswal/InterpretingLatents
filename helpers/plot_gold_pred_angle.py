"""
Standalone LaTeX-table generation for gold_predtoken_angles.py.

Reads, for each (family, task, model):
    outputs/gradient_geometry_predtoken / <family> / <task> / <model> / gold_predtoken_angles.json

and reports the exact max principal angle at every timestep t, alongside
the rank-matched random control averaged (mean cos^2 theta) over t.

Writes
    Tables/extended/
        gold_pred_angle_{family}.tex

All input and output paths are hardcoded; no CLI flags.
"""

import json
import numpy as np
from pathlib import Path
from src.config import BASE_DIR

# ═══════════════════════════════════════════════════════════════════
# Hardcoded paths
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = BASE_DIR / "outputs" / "gradient_geometry_predtoken"
TABLE_DIR = Path("Tables/extended")

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

TASK_LABELS = {
    "prosqa": "Graph-Hopping",
    "gsm":    "Arithmetic-Reasoning",
}


def sort_family_key(family):
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return 99


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

def discover_results():
    """Walk OUT_DIR / <family> / <task> / <model> / gold_predtoken_angles.json."""
    results = []
    for path in sorted(OUT_DIR.glob("*/*/*/gold_predtoken_angles.json")):
        with open(path, "r") as f:
            r = json.load(f)
        family = path.parent.parent.parent.name
        results.append({
            "family": family,
            "task": r["task"],
            "model": r["model"],
            "summary": r,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Per-family table
# ═══════════════════════════════════════════════════════════════════

def build_per_family_angle_table(family, family_results) -> str:
    sorted_results = sorted(
        family_results,
        key=lambda r: (sort_task_key(r["task"]), sort_model_key(r["model"])),
    )
    if not sorted_results:
        return ""

    family_label = FAMILY_LABELS.get(family, family)
    T = sorted_results[0]["summary"]["T"]

    t_headers = " & ".join([f"$t_{{{t}}}$" for t in range(T)])
    col_spec = "ll" + "r" * T + "rr"

    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (rf"\caption{{Gold-label vs.\ predicted-token subspace alignment "
         rf"across tasks for the {family_label} model family. Each "
         rf"$t_i$ column reports $\overline{{\cos^2\theta}}$ between "
         rf"$B_t^{{\text{{gold}}}}$ and $B_t^{{\text{{pred}}}}$ at that "
         rf"timestep (1 = same subspace, 0 = orthogonal); the "
         rf"$\overline{{\cos^2\theta}}$ column is the mean over $t$, and "
         rf"the control column is the same statistic for rank-matched "
         rf"random subspaces.}}"),
        rf"\label{{tab:gold_pred_angle_{family}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (f"Task & Model & {t_headers} & "
         r"$\overline{\cos^2\theta}$ & Control $\overline{\cos^2\theta}$"
         r" \\"),
        r"\midrule",
    ]

    for t_idx, task in enumerate(TASK_ORDER):
        task_results = [r for r in sorted_results if r["task"] == task]
        if not task_results:
            continue
        if t_idx > 0:
            lines.append(r"\midrule")

        n_models = len(task_results)
        task_cell = (rf"\multirow{{{n_models}}}{{*}}"
                     rf"{{\makecell[l]{{{TASK_LABELS.get(task, task)}}}}}")

        for m_idx, r in enumerate(task_results):
            model = r["model"]
            s = r["summary"]
            per_t = s["per_t"]

            # A timestep with n_angles == 0 (e.g. a recurrent model's
            # rank-0 final step) has no defined cos^2 (mean_cos_sq is 0.0
            # by construction there, not a meaningful alignment score).
            cos_sq_cells = [
                ("--" if p["n_angles"] == 0 else f"{p['mean_cos_sq']:.3f}")
                for p in per_t
            ]
            active = [p for p in per_t if p["n_angles"] > 0]
            mean_cos_sq = (float(np.mean([p["mean_cos_sq"] for p in active]))
                           if active else float("nan"))
            control = float(np.mean([p["control_mean_cos_sq"] for p in per_t]))

            first_col = task_cell if m_idx == 0 else ""
            cells = [
                first_col,
                MODEL_LABELS.get(model, model),
                *cos_sq_cells,
                f"{mean_cos_sq:.3f}" if not np.isnan(mean_cos_sq) else "--",
                f"{control:.3f}",
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def build_all_family_angle_tables(results) -> dict:
    families = sorted({r["family"] for r in results}, key=sort_family_key)
    return {
        family: build_per_family_angle_table(
            family, [r for r in results if r["family"] == family])
        for family in families
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    results = discover_results()
    if not results:
        raise SystemExit(
            f"No gold_predtoken_angles.json found under {OUT_DIR}/*/*/*/."
        )

    family_tables = build_all_family_angle_tables(results)
    for family, tex in family_tables.items():
        if not tex:
            continue
        p = TABLE_DIR / f"gold_pred_angle_{family}.tex"
        p.write_text(tex)
        print(f"[tex ] Gold-pred angle table [{family}] -> {p.resolve()}")


if __name__ == "__main__":
    main()
