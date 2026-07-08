"""
plot_remove_thoughts.py
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
from src.config import THOUGHT_ABLATION


# ── canonical column order ───────────────────────────────────────────────────

# Each entry:
# (dir_name_in_thoughts_dir, LaTeX_column_label, has_remove_thoughts_output)
MODELS = [
    ("base",      "B",       True),
    ("cot",       "CoT",     True),
    ("pause",     "PaT",     True),
    ("coconut",   "C",       True),
    ("coconut_u", "C$_u$",   True),
    ("codi",      "CODI",    True),
]

TASKS = [
    ("prosqa", "Graph-Hopping"),
    ("gsm",    "Arithmetic-Reasoning"),
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
        print(
            f"         -> {len(records)} records; "
            f"metrics: {[r.get('metric') for r in records]}"
        )

    return records


# ── JSONL record lookup ──────────────────────────────────────────────────────

def find_record(records: list, metric: str):
    """Return the last (most recent) record whose top-level 'metric' field equals `metric`."""
    for r in reversed(records):
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

def collect(thoughts_dir: Path, family: str = "gpt2", debug: bool = False) -> dict:
    # remove_thoughts.py writes outputs/remove_thoughts/<task>/<family>/<model>/
    # (see remove_thoughts.py: output_base = THOUGHT_ABLATION / task / model_family).
    data = {}

    for task, task_label in TASKS:
        data[task] = {}

        if debug:
            print(f"\n-- {task_label} ({task}) [{family}] --")

        for dir_name, _, has_output in MODELS:

            if not has_output:
                continue

            base = thoughts_dir / task / family / dir_name

            if debug:
                print(f"\n  model={dir_name}  base={base}")

            summary = load_json(base / "removeThoughts.json", debug)
            records = load_jsonl(base / "removeThoughts_cis.jsonl", debug)

            if summary is None:
                data[task][dir_name] = None
                continue

            # Route extraction based on model type
            if dir_name in ["base", "cot"]:
                # Standard inference schema
                data[task][dir_name] = {
                    "orig":       summary.get("accuracy"),
                    "ablated":    None,  # No ablation possible
                    "n":          summary.get("n_instances"),
                    "n_thoughts": 0,
                    "ci_orig":    find_record(records, "accuracy"),
                    "ci_ablated": None,
                    "ci_diff":    None,
                    "mc":         None,
                }
            else:
                # LRM continuous thought schema
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

def build_main_table(data: dict, family: str) -> str:
    """Main performance table in compact horizontal task layout."""

    def _cell(task, model, key):
        entry = data[task].get(model)
        if entry is None:
            return "--"

        if key == "orig":
            return pm_pct(entry.get("ci_orig")) or fmt_pct(entry.get("orig"))

        if key == "ablated":
            if entry.get("ci_ablated") is None and entry.get("ablated") is None:
                return "--"
            return (
                pm_pct(entry.get("ci_ablated"))
                or fmt_pct(entry.get("ablated"))
            )

        return "--"

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcc|cc}",
        r"\toprule",
        (
            r"Model & \makecell{Graph- \\ Hopping} "
            r"& \makecell{Thoughts \\ Removed} "
            r"& \makecell{Arithmetic- \\ Reasoning} "
            r"& \makecell{Thoughts \\ Removed} \\"
        ),
        r"\midrule",
    ]

    for dir_name, label, has_output in MODELS:

        gh_orig = _cell("prosqa", dir_name, "orig")
        gh_abl  = _cell("prosqa", dir_name, "ablated")

        ar_orig = _cell("gsm", dir_name, "orig")
        ar_abl  = _cell("gsm", dir_name, "ablated")

        lines.append(
            f"{label} & "
            f"{gh_orig} & {gh_abl} & "
            f"{ar_orig} & {ar_abl} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Thought-token ablation ({family}).}}",
        rf"\label{{tab:ablation_main_{family}}}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ── Table 2: appendix statistical tests ──────────────────────────────────────

def build_appendix_table(data: dict, family: str) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc cc}",
        r"\toprule",
        (r"Model & Acc$_{K_{\max}}$ [\% CI] "
         r"& Acc$_{K{=}0}$ [\% CI] "
         r"& $\Delta$ [\% CI] "
         r"& McNemar $p$ & $n$ \\"),
    ]

    for task, task_label in TASKS:
        lines += [
            r"\midrule",
            rf"\multicolumn{{6}}{{c}}{{\textbf{{{task_label}}}}} \\",
            r"\midrule",
        ]
        for dir_name, col_label, has_output in MODELS:
            if not has_output:
                continue
            entry = data[task].get(dir_name)
            if entry is None:
                lines.append(f"{col_label} & -- & -- & -- & -- & -- \\\\")
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
        rf"\caption{{Thought-token ablation statistical tests ({family}). "
        r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ (exact two-sided McNemar).}",
        rf"\label{{tab:ablation_stats_{family}}}",
        r"\end{table}",
    ]
    return "\n".join(lines).strip()


# ── entry point ──────────────────────────────────────────────────────────────

def discover_families(thoughts_dir: Path) -> list:
    """Families present on disk under remove_thoughts/<task>/<family>/.

    Scans every task subdir and unions the family-level directory names.
    """
    found = set()
    for task, _ in TASKS:
        task_dir = thoughts_dir / task
        if not task_dir.is_dir():
            continue
        for child in task_dir.iterdir():
            if child.is_dir():
                found.add(child.name)
    # Stable order: known families first, then any extras alphabetically.
    known = [f for f in ("gpt2", "llama") if f in found]
    extra = sorted(found - set(known))
    return known + extra


def main():

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--model_family", choices=["gpt2", "llama"], default=None,
        help="Restrict to one family. Default: loop all families found on disk.",
    )
    ap.add_argument(
        "--out_main_dir", default="Tables/main",
        help="Directory for main-text tables.",
    )
    ap.add_argument(
        "--out_stats_dir", default="Tables/statistical",
        help="Directory for appendix statistical tables.",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print every path probed and loaded values, then exit.",
    )

    args = ap.parse_args()

    families = ([args.model_family] if args.model_family
                else discover_families(THOUGHT_ABLATION))
    if not families:
        print(f"[WARN] No families found under {THOUGHT_ABLATION}")
        return
    print(f"[INFO] Families: {families}")

    out_main_dir = Path(args.out_main_dir)
    out_stats_dir = Path(args.out_stats_dir)
    out_main_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        data = collect(THOUGHT_ABLATION, family=family, debug=args.debug)

        if args.debug:
            print(f"\n-- Collected data summary [{family}] --")
            for task, models in data.items():
                for m, entry in models.items():
                    print(f"  {task}/{m}: {entry}")
            continue

        main_tex = build_main_table(data, family)
        p_main = out_main_dir / f"remove_thoughts_{family}.tex"
        p_main.write_text(main_tex)
        print(f"[OK] Main  -> {p_main.resolve()}")

        app_tex = build_appendix_table(data, family)
        p_app = out_stats_dir / f"remove_thoughts_{family}.tex"
        p_app.write_text(app_tex)
        print(f"[OK] Stats -> {p_app.resolve()}")


if __name__ == "__main__":
    main()