"""
DIAGNOSTIC: Graph depth coverage in ProsQA
==========================================

Checks how many of the 500 test instances have >=2 candidate nodes
at each depth from the root. If most graphs don't have nodes at
depth 5 or 6, then the depth-frontier probing results at high k
are computed over a tiny, unrepresentative subset.

This explains anomalies like coconut_u showing P(correct)=0.00
at k=6 — if only 3 instances have depth-6 nodes, the aggregate
is meaningless.

Usage:
    python diagnose_depth_coverage.py
"""

import json
from collections import defaultdict


def build_children_map(edges):
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    return children


def get_frontier_size(instance, target_depth):
    """
    Returns the number of nodes at exactly `target_depth` hops
    from the root in this instance's graph.
    """
    root_idx = instance["root"]
    edges = instance["edges"]
    children_map = build_children_map(edges)

    depth_of = {root_idx: 0}
    queue = [root_idx]

    while queue:
        node = queue.pop(0)
        d = depth_of[node]
        if d >= target_depth:
            continue
        for child in children_map[node]:
            if child not in depth_of:
                depth_of[child] = d + 1
                queue.append(child)

    frontier = [idx for idx, d in depth_of.items() if d == target_depth]
    return len(frontier)


def get_max_depth(instance):
    """BFS to find the maximum depth reachable from root."""
    root_idx = instance["root"]
    edges = instance["edges"]
    children_map = build_children_map(edges)

    depth_of = {root_idx: 0}
    queue = [root_idx]

    while queue:
        node = queue.pop(0)
        d = depth_of[node]
        for child in children_map[node]:
            if child not in depth_of:
                depth_of[child] = d + 1
                queue.append(child)

    return max(depth_of.values()) if depth_of else 0


def get_shortest_path_length(instance):
    """BFS shortest path from root to target."""
    root_idx = instance["root"]
    target_idx = instance["target"]
    edges = instance["edges"]
    children_map = build_children_map(edges)

    depth_of = {root_idx: 0}
    queue = [root_idx]

    while queue:
        node = queue.pop(0)
        if node == target_idx:
            return depth_of[node]
        d = depth_of[node]
        for child in children_map[node]:
            if child not in depth_of:
                depth_of[child] = d + 1
                queue.append(child)

    return -1  # target not reachable


def main():
    # Update this path to your ProsQA test file
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default — adjust to your setup
        data_path = "data/prosqa_test.json"

    print(f"Loading: {data_path}")
    with open(data_path) as f:
        raw_data = json.load(f)

    n = len(raw_data)
    print(f"Total instances: {n}\n")

    # --- Depth coverage ---
    print("=" * 70)
    print("DEPTH FRONTIER COVERAGE")
    print("How many instances have >=2 candidates at each depth?")
    print("(This is the filter used in the probing experiment)")
    print("=" * 70)

    max_k = 7
    for k in range(0, max_k + 1):
        target_depth = max(k, 1)
        n_with_candidates = 0
        frontier_sizes = []

        for inst in raw_data:
            fs = get_frontier_size(inst, target_depth)
            frontier_sizes.append(fs)
            if fs >= 2:
                n_with_candidates += 1

        avg_fs = sum(frontier_sizes) / n if n > 0 else 0
        print(f"  k={k} (depth={target_depth}): "
              f"{n_with_candidates}/{n} instances have >=2 candidates "
              f"({n_with_candidates/n:.1%}), "
              f"avg frontier size={avg_fs:.1f}")

    # --- Shortest path distribution ---
    print("\n" + "=" * 70)
    print("SHORTEST PATH LENGTH DISTRIBUTION")
    print("=" * 70)

    path_lengths = [get_shortest_path_length(inst) for inst in raw_data]
    from collections import Counter
    pl_counts = Counter(path_lengths)
    for pl in sorted(pl_counts.keys()):
        print(f"  path_length={pl}: {pl_counts[pl]} instances ({pl_counts[pl]/n:.1%})")

    avg_pl = sum(path_lengths) / n
    print(f"  Mean: {avg_pl:.2f}")

    # --- Max depth distribution ---
    print("\n" + "=" * 70)
    print("MAX REACHABLE DEPTH FROM ROOT")
    print("=" * 70)

    max_depths = [get_max_depth(inst) for inst in raw_data]
    md_counts = Counter(max_depths)
    for md in sorted(md_counts.keys()):
        print(f"  max_depth={md}: {md_counts[md]} instances ({md_counts[md]/n:.1%})")

    avg_md = sum(max_depths) / n
    print(f"  Mean: {avg_md:.2f}")

    # --- Cross-check: instances surviving at each k ---
    print("\n" + "=" * 70)
    print("CRITICAL CHECK: SAMPLE SIZE AT EACH k")
    print("If n drops sharply, aggregate metrics are unreliable")
    print("=" * 70)

    for k in range(0, max_k + 1):
        target_depth = max(k, 1)
        surviving = sum(1 for inst in raw_data
                       if get_frontier_size(inst, target_depth) >= 2)
        flag = " *** UNRELIABLE" if surviving < 50 else ""
        print(f"  k={k}: n={surviving}{flag}")


if __name__ == "__main__":
    main()