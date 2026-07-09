import json
import argparse
from pathlib import Path
from src.config import INLP

# ─── Configuration ───────────────────────────────────────────────

TASKS = ["prosqa", "gsm"]
MODELS = ["pause", "coconut", "coconut_u", "codi"]
MODEL_LABELS = {
    "pause": "Pause",
    "coconut": "Coconut",
    "coconut_u": "Coconut-$u$",
    "codi": "CODI",
}
TASK_LABELS = {"prosqa": "ProsQA", "gsm": "GSM8k"}

# Regime thresholds (must match interventions_fast.py)
GENUINE_MAX = 0.1
MAGNITUDE_MIN = 5.0


def regime(alpha, median_norm):
    """
    Classify alpha into a steering regime.
        r = alpha / median_norm
        r <= GENUINE_MAX           => "G" (Genuine)
        GENUINE_MAX < r <= MAG_MIN => "T" (Transition)
        r > MAG_MIN                => "M" (Magnitude)
    """
    r = alpha / median_norm
    if r <= GENUINE_MAX:
        return "G"
    if r <= MAGNITUDE_MIN:
        return "T"
    return "M"


def load_steering_results(task, model):
    """
    Load steering_results_fast.json for a (task, model) pair.

    Returns None if the file doesn't exist (e.g. CODI crashed).
    Otherwise returns the parsed JSON dict with keys:
        alphas, inlp_steering, rand_steering, regime_info
    """
    path = INLP / task / model / "steering_results_fast.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_flip(rate_fraction):
    """
    Format a flip rate (0.0–1.0 fraction) as a percentage string.
    0.0     => "0.0"
    0.002   => "0.2"
    0.349   => "34.9"
    1.0     => "100"
    """
    pct = rate_fraction * 100
    if pct == 0.0:
        return "0.0"
    if pct >= 99.95:
        return "100"
    return f"{pct:.1f}"


def build_table():
    """
    Build the full LaTeX table string.
    """

    # ── Load all data ────────────────────────────────────────────
    results = {}
    for task in TASKS:
        results[task] = {}
        for model in MODELS:
            results[task][model] = load_steering_results(task, model)

    # ── Determine alpha list from any available result ───────────
    alphas = None
    for task in TASKS:
        for model in MODELS:
            r = results[task][model]
            if r is not None:
                alphas = r["alphas"]
                break
        if alphas is not None:
            break
    if alphas is None:
        raise RuntimeError("No steering_results_fast.json found anywhere")

    # ── Compute regime per (task, model, alpha) ──────────────────
    median_norms = {}
    for task in TASKS:
        median_norms[task] = {}
        for model in MODELS:
            r = results[task][model]
            if r is not None:
                median_norms[task][model] = r["regime_info"]["median_pooled"]
            else:
                median_norms[task][model] = None

    # ── Find first-transition and first-magnitude alpha indices ──
    first_T = {t: {} for t in TASKS}
    first_M = {t: {} for t in TASKS}
    for task in TASKS:
        for model in MODELS:
            mn = median_norms[task][model]
            first_T[task][model] = None
            first_M[task][model] = None
            if mn is None:
                continue
            prev_regime = None
            for i, alpha in enumerate(alphas):
                cur = regime(alpha, mn)
                if prev_regime == "G" and cur == "T":
                    first_T[task][model] = i
                if prev_regime == "T" and cur == "M":
                    first_M[task][model] = i
                prev_regime = cur

    # ── Build LaTeX ──────────────────────────────────────────────
    n_models = len(MODELS)
    # Column spec: alpha + 8 ProsQA + 8 GSM = 17 columns
    col_spec = "r" + "|" + "rr" * n_models + "|" + "rr" * n_models

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # ── Header row 1: task names spanning model groups ───────────
    header1 = r" & \multicolumn{" + str(n_models * 2) + r"}{c|}{" + TASK_LABELS["prosqa"] + "}"
    header1 += r" & \multicolumn{" + str(n_models * 2) + r"}{c}{" + TASK_LABELS["gsm"] + "}"
    header1 += r" \\"
    lines.append(header1)

    # ── Header row 2: model names ────────────────────────────────
    header2 = r" "
    for t_idx, task in enumerate(TASKS):
        for m_idx, model in enumerate(MODELS):
            bar = "|" if t_idx == 0 and m_idx == len(MODELS) - 1 else ""
            header2 += r" & \multicolumn{2}{c" + bar + r"}{" + MODEL_LABELS[model] + "}"
    header2 += r" \\"
    lines.append(header2)

    # ── Header row 3: INLP and Rand ──────────────────────────────
    header3 = r"$\alpha$"
    for task in TASKS:
        for model in MODELS:
            header3 += r" & INLP & Rand"
    header3 += r" \\"
    lines.append(header3)
    lines.append(r"\midrule")

    # ── Data rows ────────────────────────────────────────────────
    for i, alpha in enumerate(alphas):
        if alpha == int(alpha):
            alpha_str = str(int(alpha))
        else:
            alpha_str = str(alpha)

        row = alpha_str

        for task in TASKS:
            for model in MODELS:
                r = results[task][model]
                if r is None:
                    inlp_cell = r"\textemdash"
                    rand_cell = r"\textemdash"
                else:
                    def get_flip(steering_dict, a):
                        for key in [a, str(a), str(float(a))]:
                            if str(key) in steering_dict:
                                return steering_dict[str(key)]["flip_rate"]
                            if key in steering_dict:
                                return steering_dict[key]["flip_rate"]
                        return None

                    inlp_fr = get_flip(r["inlp_steering"], alpha)
                    rand_fr = get_flip(r["rand_steering"], alpha)

                    if inlp_fr is None or rand_fr is None:
                        inlp_cell = r"\textemdash"
                        rand_cell = r"\textemdash"
                    else:
                        inlp_cell = format_flip(inlp_fr)
                        rand_cell = format_flip(rand_fr)

                # ── Regime transition markers ────────────────
                marker = ""
                if first_T[task][model] == i:
                    marker += r"$^{\dagger}$"
                if first_M[task][model] == i:
                    marker += r"$^{\ddagger}$"

                row += " & " + inlp_cell + marker + " & " + rand_cell

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # ── Caption with regime legend and median norms ──────────────
    caption_parts = []
    for task in TASKS:
        norms_str = ", ".join(
            f"{MODEL_LABELS[m]}={median_norms[task][m]:.1f}"
            if median_norms[task][m] is not None
            else f"{MODEL_LABELS[m]}=n/a"
            for m in MODELS
        )
        caption_parts.append(f"{TASK_LABELS[task]}: {norms_str}")

    lines.append(
        r"\caption{Direction-agnostic steering flip rates (\%) "
        r"across $\alpha$ values. "
        r"$^{\dagger}$\,marks the first $\alpha$ entering the Transition regime "
        r"($r > 0.1$); "
        r"$^{\ddagger}$\,marks Magnitude ($r > 5$), "
        r"where $r = \alpha / \mathrm{median}\,\|h_t\|$. "
        r"Pooled median norms: " + "; ".join(caption_parts) + r". "
        r"\textemdash\ indicates missing data (CODI dtype crash).}"
    )

    lines.append(r"\label{tab:steering_flip_rates}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    output_dir = INLP / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    table_tex = build_table()

    out_path = output_dir / "steering_flip_rates.tex"
    with open(out_path, "w") as f:
        f.write(table_tex + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()