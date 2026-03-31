"""
Export per-instance BFS categories from control experiment results.

For each model condition, classifies every instance into one of:
    no_superposition       top-1 >= 0.7 at every k
    transient              exactly 1 k value has top-1 < 0.7
    sustained_no_convergence   >= 2 k values with top-1 < 0.7, but
                               does not meet BFS or anti-BFS criteria
    bfs                    >= 2 k values with top-1 < 0.7, AND
                               P(correct) increases by > 0.05, AND
                               top-1 identity shifts, AND
                               final top-1 is correct
    anti_bfs               >= 2 k values with top-1 < 0.7, AND
                               P(correct) decreases by > 0.05

Classification logic:
    distributed_ks = count of k values where top-1 probability < 0.7
    P_corr_first   = sum of P(candidate) for correct candidates at first k
    P_corr_last    = same at last k
    top1_shifts    = whether the identity of the top-1 candidate changes across k

    if distributed_ks == 0:           -> no_superposition
    elif distributed_ks == 1:         -> transient
    elif P_corr_last > P_corr_first + 0.05
         AND top1_shifts
         AND final top-1 is correct:  -> bfs
    elif P_corr_last < P_corr_first - 0.05: -> anti_bfs
    else:                             -> sustained_no_convergence

Output format (per model):
    {"0": "bfs", "1": "no_superposition", "2": "transient", ...}

Usage:
    python export_bfs_categories.py
"""

import json
from pathlib import Path
from utils.config import CONTROL_EXPT


def classify_instance(inst):
    """Classify a single instance into one of 5 BFS categories."""
    ks = sorted(inst.keys(), key=int)
    if len(ks) < 3:
        return "insufficient_data"

    # top-1 probability at each k
    top1_probs = [max(inst[k]["concept_probs"].values()) for k in ks]

    # how many k values have distributed mass (top-1 < 0.7)?
    distributed_ks = sum(1 for p in top1_probs if p < 0.7)

    if distributed_ks == 0:
        return "no_superposition"
    if distributed_ks == 1:
        return "transient"

    # sustained superposition (distributed_ks >= 2) — check BFS / anti-BFS

    # P(correct) at first and last k
    first_k, last_k = ks[0], ks[-1]

    def p_correct(k):
        probs = inst[k]["concept_probs"]
        correct = inst[k]["correct_candidates"]
        return sum(probs.get(c, 0) for c in correct)

    p_corr_first = p_correct(first_k)
    p_corr_last = p_correct(last_k)

    # does the top-1 candidate identity change across k?
    top1_names = [
        max(inst[k]["concept_probs"], key=inst[k]["concept_probs"].get)
        for k in ks
    ]
    top1_shifts = len(set(top1_names)) > 1

    # is the final top-1 candidate correct?
    final_top1 = max(
        inst[last_k]["concept_probs"],
        key=inst[last_k]["concept_probs"].get,
    )
    final_correct = final_top1 in inst[last_k]["correct_candidates"]

    # BFS: P(correct) increases, top-1 shifts, final top-1 correct
    if (
        p_corr_last > p_corr_first + 0.05
        and top1_shifts
        and final_correct
    ):
        return "bfs"

    # anti-BFS: P(correct) decreases
    if p_corr_last < p_corr_first - 0.05:
        return "anti_bfs"

    return "sustained_no_convergence"


def classify_all(results_path):
    """Classify all instances in a results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    categories = {}
    for si, inst in data["per_instance"].items():
        categories[si] = classify_instance(inst)

    return categories


def main():
    result_files = {
        "base": "results_base.json",
        "cot": "results_cot.json",
        "coconut": "results_coconut.json",
        "coconut_u": "results_coconut_u.json",
        "pause": "results_pause.json",
    }

    for mode, fname in result_files.items():
        path = CONTROL_EXPT / fname
        if not path.exists():
            print(f"SKIP {fname} (not found)")
            continue

        categories = classify_all(path)

        # count summary
        from collections import Counter
        counts = Counter(categories.values())
        n = len(categories)
        print(f"{mode} (n={n}):")
        for cat in [
            "no_superposition",
            "transient",
            "sustained_no_convergence",
            "bfs",
            "anti_bfs",
        ]:
            c = counts.get(cat, 0)
            print(f"  {cat:30s} {c:4d} ({c/n*100:5.1f}%)")

        # save
        out_path = CONTROL_EXPT / f"bfs_categories_{mode}.json"
        with open(out_path, "w") as f:
            json.dump(categories, f, indent=2)
        print(f"  -> {out_path}\n")


if __name__ == "__main__":
    main()