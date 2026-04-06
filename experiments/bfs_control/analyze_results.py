"""
ANALYSIS: Interpreting latent reasoning as tree search
======================================================
1. Value function: p(concept) as implicit value estimate
2. Parallelism: cumulative top-k values
3. Height vs. value correlation

Usage:
    python analyze_results.py --results_dir experiments/control_expt/
"""

import json
import argparse
from pathlib import Path


def load_results(results_dir):
    """Load all results_*.json files from the directory."""
    results = {}
    for f in sorted(Path(results_dir).glob("results_*.json")):
        mode = f.stem.replace("results_", "")
        with open(f) as fh:
            results[mode] = json.load(fh)
    return results


def print_table_1(all_results):
    """
    Table matching the paper's aggregate metrics:
        H/log2(N), top-1 value, P(correct value), top-1 correct %

    These are the metrics from Table 1 of your Progress.pdf but now
    computed from raw (unnormalized) probabilities.
    """
    modes_order = ["base", "cot", "pause", "coconut", "coconut_u", ]
    mode_labels = {"base": "B", "cot": "CoT", "coconut": "C",
                   "coconut_u": "Cu", "pause": "Pa"}

    available = [m for m in modes_order if m in all_results]

    # Collect k values present across all modes
    all_k = set()
    for mode in available:
        all_k.update(int(k) for k in all_results[mode]["summary"].keys())
    all_k = sorted(all_k)

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS (raw probabilities, depth-frontier candidates)")
    print("=" * 80)

    # --- Normalized entropy ---
    print(f"\n{'H/log2(N)':<12}", end="")
    for m in available:
        print(f"  {mode_labels[m]:>6}", end="")
    print()
    for k in all_k:
        print(f"  k={k:<8}", end="")
        for m in available:
            s = all_results[m]["summary"].get(str(k), {})
            v = s.get("mean_normalized_entropy", float("nan"))
            print(f"  {v:>6.2f}", end="")
        print()

    # --- Top-1 correct fraction ---
    print(f"\n{'Top-1 corr%':<12}", end="")
    for m in available:
        print(f"  {mode_labels[m]:>6}", end="")
    print()
    for k in all_k:
        print(f"  k={k:<8}", end="")
        for m in available:
            s = all_results[m]["summary"].get(str(k), {})
            v = s.get("top1_correct_frac", float("nan"))
            print(f"  {v*100:>5.0f}%", end="")
        print()

    # --- P(correct) = sum of raw probs for correct candidates ---
    print(f"\n{'P(correct)':<12}", end="")
    for m in available:
        print(f"  {mode_labels[m]:>6}", end="")
    print()
    for k in all_k:
        print(f"  k={k:<8}", end="")
        for m in available:
            s = all_results[m]["summary"].get(str(k), {})
            v = s.get("mean_value_correct", float("nan"))
            print(f"  {v:>6.2f}", end="")
        print()

    # --- Candidate mass (total prob on graph-relevant candidates) ---
    print(f"\n{'Cand. mass':<12}", end="")
    for m in available:
        print(f"  {mode_labels[m]:>6}", end="")
    print()
    for k in all_k:
        print(f"  k={k:<8}", end="")
        for m in available:
            s = all_results[m]["summary"].get(str(k), {})
            v = s.get("mean_candidate_mass", float("nan"))
            print(f"  {v:>6.2f}", end="")
        print()


def print_parallelism_analysis(all_results):
    """
    Parallelism analysis matching Figure 6:
        Cumulative top-1, top-2, top-3 values across k.

    The gap between top-1 and top-3 indicates how much the model
    maintains multiple candidates (exploration) vs. concentrating
    on one (convergence).
    """
    modes_order = ["base", "cot", "coconut", "coconut_u", "pause"]
    mode_labels = {"base": "Base", "cot": "CoT", "coconut": "Coconut",
                   "coconut_u": "Coconut-u", "pause": "Pause"}

    available = [m for m in modes_order if m in all_results]

    print("\n" + "=" * 80)
    print("PARALLELISM ANALYSIS (Figure 6 — cumulative top-k values)")
    print("=" * 80)

    for mode in available:
        summary = all_results[mode]["summary"]
        ks = sorted(int(k) for k in summary.keys())

        print(f"\n  {mode_labels[mode]}:")
        print(f"  {'k':>3}  {'top-1':>7}  {'top-2':>7}  {'top-3':>7}  {'gap(3-1)':>8}")
        for k in ks:
            s = summary[str(k)]
            t1 = s.get("mean_top1_prob", 0)
            t2 = s.get("mean_top2_cumul", 0)
            t3 = s.get("mean_top3_cumul", 0)
            gap = t3 - t1
            print(f"  {k:>3}  {t1:>7.3f}  {t2:>7.3f}  {t3:>7.3f}  {gap:>8.3f}")


def print_convergence_analysis(all_results):
    """
    How the value function evolves across k.

    Reports the trajectory of P(correct) and normalized entropy
    from k=0 to k=max, which is the core of the BFS interpretation
    (Section 4.3-4.4): does the model explore early and converge late?
    """
    modes_order = ["base", "cot", "coconut", "coconut_u", "pause"]
    mode_labels = {"base": "Base", "cot": "CoT", "coconut": "Coconut",
                   "coconut_u": "Coconut-u", "pause": "Pause"}

    available = [m for m in modes_order if m in all_results]

    print("\n" + "=" * 80)
    print("CONVERGENCE TRAJECTORIES (Section 4.3-4.4)")
    print("  Does P(correct) increase and entropy decrease across k?")
    print("=" * 80)

    for mode in available:
        summary = all_results[mode]["summary"]
        ks = sorted(int(k) for k in summary.keys())

        if len(ks) < 2:
            continue

        first_k = str(ks[0])
        last_k = str(ks[-1])

        pcorr_first = summary[first_k].get("mean_value_correct", 0)
        pcorr_last = summary[last_k].get("mean_value_correct", 0)
        ent_first = summary[first_k].get("mean_normalized_entropy", 0)
        ent_last = summary[last_k].get("mean_normalized_entropy", 0)

        delta_pcorr = pcorr_last - pcorr_first
        delta_ent = ent_last - ent_first

        print(f"\n  {mode_labels[mode]}:")
        print(f"    P(correct): {pcorr_first:.3f} → {pcorr_last:.3f}  "
              f"(Δ = {delta_pcorr:+.3f})")
        print(f"    H/log2(N):  {ent_first:.3f} → {ent_last:.3f}  "
              f"(Δ = {delta_ent:+.3f})")

        # Characterize trajectory without hard categorical labels
        if delta_pcorr > 0.05 and delta_ent < -0.05:
            print(f"    → Exploration-to-convergence pattern (consistent with BFS)")
        elif delta_pcorr < -0.05:
            print(f"    → Correct signal degrades with more thoughts")
        elif abs(delta_ent) < 0.05 and abs(delta_pcorr) < 0.05:
            print(f"    → Stable across k (no significant trajectory)")
        else:
            print(f"    → Mixed trajectory")


def print_per_instance_summary(all_results):
    """
    Per-instance statistics: what fraction of instances show
    the exploration → convergence pattern at the instance level.

    Uses continuous thresholds rather than hard categorical labels.
    """
    modes_order = ["base", "cot", "coconut", "coconut_u", "pause"]
    mode_labels = {"base": "Base", "cot": "CoT", "coconut": "Coconut",
                   "coconut_u": "Coconut-u", "pause": "Pause"}

    available = [m for m in modes_order if m in all_results]

    print("\n" + "=" * 80)
    print("PER-INSTANCE ANALYSIS")
    print("  Fraction of instances where P(correct) increases from first to last k")
    print("=" * 80)

    for mode in available:
        per_inst = all_results[mode].get("per_instance", {})
        if not per_inst:
            continue

        n_total = 0
        n_pcorr_increases = 0
        n_top1_correct_at_end = 0
        n_high_entropy_early = 0  # normalized entropy > 0.5 at k=0 or k=1

        for si, k_data in per_inst.items():
            ks = sorted(int(k) for k in k_data.keys())
            if len(ks) < 2:
                continue

            n_total += 1

            first = k_data[str(ks[0])]["metrics"]
            last = k_data[str(ks[-1])]["metrics"]

            # Correct candidates
            correct_set = set(k_data[str(ks[0])].get("correct_candidates", []))

            # P(correct) trajectory
            pcorr_first = first.get("value_correct", 0)
            pcorr_last = last.get("value_correct", 0)
            if pcorr_last > pcorr_first + 0.01:
                n_pcorr_increases += 1

            # Top-1 correct at final k
            if last.get("top1_is_correct", False):
                n_top1_correct_at_end += 1

            # High entropy early (superposition)
            ent_early = first.get("normalized_entropy", 0)
            if ent_early > 0.5:
                n_high_entropy_early += 1

        if n_total == 0:
            continue

        print(f"\n  {mode_labels[mode]} (n={n_total}):")
        print(f"    P(correct) increases over k:     "
              f"{n_pcorr_increases/n_total:.1%}")
        print(f"    Top-1 correct at final k:        "
              f"{n_top1_correct_at_end/n_total:.1%}")
        print(f"    High entropy at first k (>0.5):  "
              f"{n_high_entropy_early/n_total:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing results_*.json files")
    args = parser.parse_args()

    all_results = load_results(args.results_dir)

    if not all_results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded results for: {', '.join(all_results.keys())}")

    print_table_1(all_results)
    print_parallelism_analysis(all_results)
    print_convergence_analysis(all_results)
    print_per_instance_summary(all_results)


if __name__ == "__main__":
    main()