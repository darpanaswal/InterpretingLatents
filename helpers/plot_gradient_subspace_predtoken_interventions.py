"""
Plot + tabulate the PREDICTED-TOKEN gradient-subspace intervention results.

Same figures and appendix tables as plot_gradient_subspace_interventions.py
(amplification lineplot + ablation/amplification stat tables), but reads the
predicted-token intervention artefacts:

    outputs/grad_subspace_predtoken/<family>/<task>/<model>/
        ablation_results.json
        amplification_results.json
        bootstrap_cis.jsonl

and writes to predtoken-namespaced figure/table paths so nothing overwrites
the gold outputs.

WHY A WRAPPER
-------------
The gold plotter keys every path off its module global GRAD_SUBSPACE
(= outputs/grad_subspace). We import that module, repoint GRAD_SUBSPACE at
the predtoken root, and reuse its collect_* / build_* functions verbatim so
the two figure sets are produced by identical code.

USAGE
-----
    python -m helpers.plot_gradient_subspace_predtoken_interventions
    python -m helpers.plot_gradient_subspace_predtoken_interventions --debug
    python -m helpers.plot_gradient_subspace_predtoken_interventions \
        --model_family gpt2
"""

import argparse
from pathlib import Path

from src.config import OUTPUTS

# Import the gold plotter module so we can (a) repoint its data root and
# (b) reuse its collect/build functions unchanged.
import helpers.plot_gradient_subspace_interventions as gold


# Predtoken intervention root (mirrors grad_subspace/ one level over).
PREDTOKEN_ROOT = OUTPUTS / "grad_subspace_predtoken"


def discover_families() -> list:
    """Families present on disk under grad_subspace_predtoken/<family>/."""
    found = set()
    if PREDTOKEN_ROOT.is_dir():
        for child in PREDTOKEN_ROOT.iterdir():
            if child.is_dir() and child.name != "figures":
                found.add(child.name)
    known = [f for f in ("gpt2", "llama") if f in found]
    extra = sorted(found - set(known))
    return known + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model_family", choices=["gpt2", "llama"], default=None,
        help="Restrict to one family. Default: loop all families found "
             "under outputs/grad_subspace_predtoken/.",
    )
    ap.add_argument(
        "--out_fig_dir", type=Path,
        default="Plots/gradient_subspace_predtoken_interventions",
        help="Directory for per-family line figures.",
    )
    ap.add_argument(
        "--out_tables_dir", type=Path,
        default=Path("Tables/statistical"),
        help="Output directory for appendix tables.",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print every path probed and what was loaded; do not write files.",
    )
    args = ap.parse_args()

    # ── Repoint the gold module's data root at the predtoken tree ──────
    # collect_ablation / collect_amplification read gold.GRAD_SUBSPACE, so
    # overriding this single global redirects all their I/O.
    gold.GRAD_SUBSPACE = PREDTOKEN_ROOT

    families = ([args.model_family] if args.model_family
                else discover_families())
    if not families:
        print(f"[WARN] No families found under {PREDTOKEN_ROOT}")
        return
    print(f"[INFO] (predtoken) Families: {families}")

    out_tables = args.out_tables_dir
    out_fig_dir = args.out_fig_dir
    out_tables.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        abl_data = gold.collect_ablation(family=family, debug=args.debug)
        amp_data = gold.collect_amplification(family=family, debug=args.debug)

        if args.debug:
            print(f"\n-- (predtoken) Ablation summary [{family}] --")
            for task, models in abl_data.items():
                for m, e in models.items():
                    ok = "None" if e is None else (
                        f"orig={e['orig']}, grad={e['grad']}, "
                        f"rand={e['rand']}, n={e['n']}"
                    )
                    print(f"  {task}/{m}: {ok}")
            print(f"\n-- (predtoken) Amplification summary [{family}] --")
            for task, models in amp_data.items():
                for m, e in models.items():
                    ok = ("None" if e is None
                          else f"{len(e['alphas'])} alphas, n={e['n']}")
                    print(f"  {task}/{m}: {ok}")
            continue

        # Main-text figure (predtoken-namespaced filename).
        out_fig = out_fig_dir / f"amplification_predtoken_{family}.pdf"
        gold.build_amplification_lineplot(amp_data, out_fig)

        # Appendix tables — predtoken-namespaced .tex.
        abl_stats = gold.build_ablation_stats_table(abl_data, family=family)
        amp_stats = gold.build_amp_stats_table(amp_data, family=family)
        # Re-label so LaTeX \label keys and captions don't collide with gold.
        abl_stats = abl_stats.replace(
            f"tab:gradsubspace_ablation_stats_{family}",
            f"tab:gradsubspace_predtoken_ablation_stats_{family}",
        ).replace("ablation statistical tests (",
                  "ablation statistical tests, predicted-token subspace (")
        amp_stats = amp_stats.replace(
            f"tab:gradsubspace_amp_stats_{family}",
            f"tab:gradsubspace_predtoken_amp_stats_{family}",
        ).replace("amplification flip-rate statistics (",
                  "amplification flip-rate statistics, predicted-token "
                  "subspace (")

        combined = abl_stats + "\n\n" + amp_stats
        p = out_tables / f"gradient_subspace_predtoken_interventions_{family}.tex"
        p.write_text(combined)
        print(f"[OK] (predtoken) Appendix tables -> {p.resolve()}")


if __name__ == "__main__":
    main()