"""
make_ablation_tables.py
=======================

Reads the output artefacts produced by remove_thoughts.py and emits:

  1. The main-text performance table  (tab_ablation_main.tex)
  2. An appendix statistical-test table (tab_ablation_stats.tex)

Changes from previous version
------------------------------
- Main table is now arranged vertically:
    * Graph-Hopping row block on top
    * Arithmetic-Reasoning row block on bottom
- Confidence intervals use a smaller font for the ± term to reduce density:
    e.g. $95.4 {\scriptsize \pm 1.2}$
- Task labels are vertically centered using \multirow

Usage
-----
    python make_ablation_tables.py \
        --out_main tables/tab_ablation_main.tex \
        --out_appendix tables/tab_ablation_stats.tex

    # Debug mode:
    python make_ablation_tables.py --debug
"""

import json
import argparse
from pathlib import Path
from src.config import THOUGHTS


# ── canonical column order ───────────────────────────────────────────────────

# Each entry:
# (dir_name_in_thoughts_dir, LaTeX_column_label, has_remove_thoughts_output)
MODELS = [
    ("b",         "B",       False),
    ("cot",       "CoT",     False),
    ("pause",     "PaT",     True),
    ("coconut",   "C",       True),
    ("coconut_u", "C$_u$",   True),
    ("codi",      "CODI",    True),
]

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm",    "Arithmetic-Reasoning"),
]

# Hard-coded cells for models that don't run through remove_thoughts.py
HARDCODED = {
    ("prosqa", "b"):   {"orig": "0",  "ablated": "--"},
    ("prosqa", "cot"): {"orig": "79", "ablated": r"--\inlinecomment{12}"},
    ("gsm",    "b"):   {"orig": "0",  "ablated": "--"},
    ("gsm",    "cot"): {"orig": "42", "ablated": r"--\inlinecomment{23}"},
}


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
        print(
            f"         -> {len(records)} records; "
            f"metrics: {[r.get('metric') for r in records]}"
        )

    return records


# ── JSONL record lookup ──────────────────────────────────────────────────────

def find_record(records: list, metric: str):
    """Return first record whose top-level 'metric' field equals `metric`."""
    for r in records:
        if r.get("metric") == metric:
            return r
    return None


# ── formatting ───────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=1) -> str:
    """0-1 float -> percentage string."""
    return f"{v * 100:.{decimals}f}" if v is not None else "--"


def pm_pct(r, decimals=1):
    """
    BootstrapResult dict ->
        '$point {\\scriptstyle \\pm half_width}$'

    Returns None when record missing.
    """
    if r is None:
        return None

    p = r["point"] * 100
    hw = (r["ci_high"] - r["ci_low"]) * 100 / 2

    return (
        rf"${p:.{decimals}f}"
        rf" {{\scriptstyle \pm {hw:.{decimals}f}}}$"
    )


def ci_pct(r) -> str:
    """BootstrapResult dict -> 'point [lo, hi]' as percentages."""
    if r is None:
        return "--"

    p = r["point"] * 100
    lo = r["ci_low"] * 100
    hi = r["ci_high"] * 100

    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def diff_pct(r) -> str:
    """Signed paired-diff CI as percentages."""
    if r is None:
        return "--"

    p = r["point"] * 100
    lo = r["ci_low"] * 100
    hi = r["ci_high"] * 100

    sign = "+" if p >= 0 else ""

    return f"{sign}{p:.1f} [{lo:.1f}, {hi:.1f}]"


def fmt_mcnemar(r) -> str:
    if r is None or r.get("p_value") is None:
        return "--"

    pv = r["p_value"]
    b = r.get("b_a_only", "?")
    c = r.get("c_b_only", "?")

    stars = (
        r"$^{***}$" if pv < 0.001 else
        r"$^{**}$"  if pv < 0.01  else
        r"$^{*}$"   if pv < 0.05  else
        ""
    )

    return f"{pv:.4f}{stars} (b={b}, c={c})"


# ── data collection ──────────────────────────────────────────────────────────

def collect(thoughts_dir: Path, debug: bool = False) -> dict:
    data = {}

    for task, task_label in TASKS:
        data[task] = {}

        if debug:
            print(f"\n-- {task_label} ({task}) --")

        for dir_name, _, has_output in MODELS:

            if not has_output:
                continue

            base = thoughts_dir / task / dir_name

            if debug:
                print(f"\n  model={dir_name}  base={base}")

            summary = load_json(base / "removeThoughts.json", debug)
            records = load_jsonl(base / "removeThoughts_cis.jsonl", debug)

            if summary is None:
                data[task][dir_name] = None
                continue

            n_th = summary.get("n_thoughts", 6)
            ci_orig_metric = f"accuracy_K{n_th}"

            data[task][dir_name] = {
                "orig":       summary.get("unquantized_accuracy"),
                "ablated":    summary.get("k0_accuracy"),
                "n":          summary.get("n_instances"),
                "n_thoughts": n_th,
                "ci_orig":    find_record(records, ci_orig_metric),
                "ci_ablated": find_record(records, "accuracy_K0"),
                "ci_diff":    find_record(records, "acc_diff_Kn_minus_K0"),
                "mc":         find_record(records, "mcnemar_Kn_vs_K0"),
            }

    return data


# ── Table 1: main performance table ──────────────────────────────────────────

MAIN_TEMPLATE = r"""\begin{table}[!t]
\centering
\tiny
\setlength{\tabcolsep}{3pt}
\caption{
Original (with thoughts for LRMs; standard for B and CoT)
vs.\ thought-ablated (\%%) scores across models and tasks.
}
\begin{tabular}{llcccccc}
\toprule
Task & Condition & B & CoT & PaT & C & C$_u$ & CODI \\
\midrule

\multirow{2}{*}{\makecell[l]{Graph-\\Hopping}}
& Original
& %(prosqa_b_o)s
& %(prosqa_cot_o)s
& %(prosqa_pause_o)s
& %(prosqa_coconut_o)s
& %(prosqa_coconut_u_o)s
& %(prosqa_codi_o)s \\

& \makecell[l]{Thoughts\\Removed}
& %(prosqa_b_a)s
& %(prosqa_cot_a)s
& %(prosqa_pause_a)s
& %(prosqa_coconut_a)s
& %(prosqa_coconut_u_a)s
& %(prosqa_codi_a)s \\

\midrule

\multirow{2}{*}{\makecell[l]{Arithmetic-\\Reasoning}}
& Original
& %(gsm_b_o)s
& %(gsm_cot_o)s
& %(gsm_pause_o)s
& %(gsm_coconut_o)s
& %(gsm_coconut_u_o)s
& %(gsm_codi_o)s \\

& \makecell[l]{Thoughts\\Removed}
& %(gsm_b_a)s
& %(gsm_cot_a)s
& %(gsm_pause_a)s
& %(gsm_coconut_a)s
& %(gsm_coconut_u_a)s
& %(gsm_codi_a)s \\

\bottomrule
\end{tabular}
\label{tab:model_performance}
\end{table}
"""


def build_main_table(data: dict) -> str:
    vals = {}

    for task, _ in TASKS:
        for dir_name, _, has_output in MODELS:

            k = f"{task}_{dir_name}"

            if not has_output:
                hc = HARDCODED.get((task, dir_name), {})

                vals[f"{k}_o"] = hc.get("orig", "--")
                vals[f"{k}_a"] = hc.get("ablated", "--")

            else:
                entry = data[task].get(dir_name)

                if entry is None:
                    vals[f"{k}_o"] = "--"
                    vals[f"{k}_a"] = "--"

                else:
                    vals[f"{k}_o"] = (
                        pm_pct(entry.get("ci_orig"))
                        or fmt_pct(entry.get("orig"))
                    )

                    vals[f"{k}_a"] = (
                        pm_pct(entry.get("ci_ablated"))
                        or fmt_pct(entry.get("ablated"))
                    )

    return MAIN_TEMPLATE % vals


# ── Table 2: appendix statistical tests ──────────────────────────────────────

def build_appendix_table(data: dict) -> str:
    lines = []

    lines += [
        r"\subsection{Thought-Token Ablation (Experiment~1)}",
        r"\label{app:ablation_stats}",
        "",
        r"Point estimates and 95\% bootstrap CIs "
        r"(10{,}000 percentile resamples) "
        r"for accuracy with thoughts ($K{=}K_{\max}$) "
        r"and without ($K{=}0$). "
        r"$\Delta = \text{Acc}_{K_{\max}} - \text{Acc}_{K{=}0}$ "
        r"(positive = thoughts help). "
        r"McNemar $p$ is the exact two-sided binomial on discordant pairs "
        r"($b$: correct only with thoughts; "
        r"$c$: correct only without).",
        "",
    ]

    for task, task_label in TASKS:

        lines += [
            r"\begin{table}[h!]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Thought-token ablation statistical tests: "
            + task_label + r".}",
            r"\begin{tabular}{l ccc cc}",
            r"\toprule",
            (
                r"Model & Acc$_{K_{\max}}$ [\% CI] "
                r"& Acc$_{K{=}0}$ [\% CI] "
                r"& $\Delta$ [\% CI] "
                r"& McNemar $p$ & $n$ \\"
            ),
            r"\midrule",
        ]

        for dir_name, col_label, has_output in MODELS:

            if not has_output:
                continue

            entry = data[task].get(dir_name)

            if entry is None:
                lines.append(
                    f"{col_label} & -- & -- & -- & -- & -- \\\\"
                )
                continue

            lines.append(
                f"{col_label} & "
                f"{ci_pct(entry['ci_orig'])} & "
                f"{ci_pct(entry['ci_ablated'])} & "
                f"{diff_pct(entry['ci_diff'])} & "
                f"{fmt_mcnemar(entry['mc'])} & "
                f"{entry['n'] or '--'} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:ablation_stats_" + task + r"}",
            r"\end{table}",
            "",
        ]

    lines.append(
        r"\noindent$^{*}p{<}0.05$,\quad "
        r"$^{**}p{<}0.01$,\quad "
        r"$^{***}p{<}0.001$ "
        r"(exact two-sided McNemar)."
    )

    return "\n".join(lines).strip()


# ── entry point ──────────────────────────────────────────────────────────────

def main():

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--out_main",
        default="tables/tab_ablation_main.tex"
    )

    ap.add_argument(
        "--out_appendix",
        default="tables/tab_ablation_stats.tex"
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print every path probed and loaded values, then exit."
    )

    args = ap.parse_args()

    data = collect(THOUGHTS, debug=args.debug)

    if args.debug:

        print("\n\n-- Collected data summary --")

        for task, models in data.items():
            for m, entry in models.items():
                print(f"  {task}/{m}: {entry}")

        return

    main_tex = build_main_table(data)

    out_main = Path(args.out_main)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    out_main.write_text(main_tex)

    print(f"[OK] Main table   -> {out_main.resolve()}")

    app_tex = build_appendix_table(data)

    out_app = Path(args.out_appendix)
    out_app.parent.mkdir(parents=True, exist_ok=True)
    out_app.write_text(app_tex)

    print(f"[OK] Appendix     -> {out_app.resolve()}")

    print("\n" + "=" * 70 + "\nMAIN TABLE\n" + "=" * 70)
    print(main_tex)

    print("\n" + "=" * 70 + "\nAPPENDIX TABLE\n" + "=" * 70)
    print(app_tex)


if __name__ == "__main__":
    main()