"""
Generate plots from VQ-VAE experiment results.

Reads eval JSONs from outputs/vqvae/<model>/codebook_K{K}.eval.json
and produces publication-quality figures.

Plots:
    1. Accuracy vs codebook size K (all three models)
    2. Trajectory diversity vs K (all three models)
    3. Self-transition rate vs K, stratified by BFS category (per model)
    4. Combined 2-panel summary (accuracy + diversity)

Usage:
    python plot_vqvae.py
    python plot_vqvae.py --output_dir figures/
    python plot_vqvae.py --format pdf
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from src.config import BASE_DIR, THOUGHTS


# ── Configuration ───────────────────────────────────────────────────

MODELS = {
    "coconut": {"label": "Coconut $u=0.0$", "color": "#2563eb", "marker": "o"},
    "coconut_u": {"label": "Coconut $u=0.3$", "color": "#dc2626", "marker": "s"},
    "pause": {"label": "Pause", "color": "#059669", "marker": "^"},
}

# ── Baseline loading ────────────────────────────────────────────────
# Baselines are saved by sanity_check_vqvae.py as:
#   THOUGHTS/removeThoughts_<model>.json
# containing k0_accuracy and unquantized_accuracy.

def load_baselines() -> dict:
    """
    Load baseline accuracies from sanity check JSONs.

    Returns: {model_name: {"k0": float, "unquantized": float}, ...}
    """
    baselines = {}
    for model_name in MODELS:
        bl_path = THOUGHTS / f"removeThoughts_{model_name}.json"
        if not bl_path.exists():
            print(f"  [WARN] No baselines found at {bl_path}")
            continue
        with open(bl_path) as f:
            data = json.load(f)
        baselines[model_name] = {
            "k0": data["k0_accuracy"],
            "unquantized": data["unquantized_accuracy"],
        }
        print(f"  Baselines for {model_name}: "
              f"K=0={data['k0_accuracy']:.1%}, "
              f"unquantized={data['unquantized_accuracy']:.1%}")
    return baselines

K_VALUES = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]

BFS_CATEGORIES = [
    "no_superposition", "transient", "sustained_no_convergence", "bfs", "anti_bfs"
]

BFS_CATEGORY_STYLES = {
    "no_superposition": {"label": "No superposition", "color": "#6b7280", "ls": "-"},
    "transient": {"label": "Transient", "color": "#f59e0b", "ls": "--"},
    "sustained_no_convergence": {"label": "Sustained (no conv.)", "color": "#8b5cf6", "ls": "-."},
    "bfs": {"label": "BFS", "color": "#2563eb", "ls": "-"},
    "anti_bfs": {"label": "Anti-BFS", "color": "#dc2626", "ls": ":"},
}


# ── Data loading ────────────────────────────────────────────────────

def load_eval_results(base_path: Path) -> dict:
    """
    Load all eval JSONs for all models.

    Returns: {model_name: {K: eval_dict, ...}, ...}
    """
    all_results = {}

    for model_name in MODELS:
        model_results = {}
        results_dir = base_path / model_name

        for k in K_VALUES:
            eval_path = results_dir / f"codebook_K{k}.eval.json"
            if not eval_path.exists():
                print(f"  [WARN] Missing: {eval_path}")
                continue

            with open(eval_path) as f:
                data = json.load(f)
            model_results[k] = data

        if model_results:
            all_results[model_name] = model_results
            print(f"  Loaded {len(model_results)} results for {model_name}")
        else:
            print(f"  [WARN] No results found for {model_name} in {results_dir}")

    return all_results


def extract_accuracy(results: dict) -> tuple:
    """Extract (K_values, accuracies) from model results."""
    ks = sorted(results.keys())
    accs = [results[k]["intervention"]["accuracy"] for k in ks]
    return ks, accs


def extract_diversity(results: dict) -> tuple:
    """Extract (K_values, trajectory_diversity) from model results."""
    ks = sorted(results.keys())
    divs = [results[k]["trajectory"]["trajectory_diversity"] for k in ks]
    return ks, divs


def extract_self_transition_by_category(results: dict) -> dict:
    """
    Extract self-transition rates per BFS category across K values.

    Returns: {category: (K_values, rates)}
    """
    ks = sorted(results.keys())
    category_data = defaultdict(lambda: ([], []))

    for k in ks:
        per_cat = results[k].get("trajectory", {}).get("per_category", {})
        for cat in BFS_CATEGORIES:
            if cat in per_cat:
                category_data[cat][0].append(k)
                category_data[cat][1].append(per_cat[cat]["self_transition_rate"])

    return dict(category_data)


# ── Plotting ────────────────────────────────────────────────────────

def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def plot_accuracy_vs_k(all_results, baselines, output_dir, fmt):
    """
    Plot 1: Intervention accuracy vs codebook size K.
    Baselines annotated as labeled markers on the right margin.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_name, style in MODELS.items():
        if model_name not in all_results:
            continue
        ks, accs = extract_accuracy(all_results[model_name])
        accs_pct = [a * 100 for a in accs]
        ax.plot(ks, accs_pct, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Codebook size $K$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Quantized Recurrence: Accuracy vs Codebook Size")
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(0.8, 350)
    ax.set_ylim(65, 102)
    ax.legend(loc="lower right", framealpha=0.9)

    # ── Baseline annotations on the right margin ────────────────────
    # Place triangular markers and labels just outside the plot area.
    # K=0 = "▷" marker, unquantized = "◁" marker, offset vertically to avoid overlap.
    if baselines:
        x_annot = 310  # just past the last data point
        for model_name, style in MODELS.items():
            if model_name not in baselines:
                continue
            bl = baselines[model_name]
            k0_y = bl["k0"] * 100
            unq_y = bl["unquantized"] * 100

            # K=0 baseline: open triangle pointing right
            ax.annotate(
                f'K=0: {k0_y:.1f}%',
                xy=(256, k0_y), xytext=(x_annot, k0_y),
                fontsize=7, color=style["color"], va="center",
                arrowprops=dict(arrowstyle="-", color=style["color"],
                                alpha=0.3, linestyle=":"),
            )

    path = output_dir / f"vqvae_accuracy_vs_k.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_diversity_vs_k(all_results, output_dir, fmt):
    """
    Plot 2: Trajectory diversity vs codebook size K.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_name, style in MODELS.items():
        if model_name not in all_results:
            continue
        ks, divs = extract_diversity(all_results[model_name])
        # diversity as percentage
        divs_pct = [d * 100 for d in divs]
        ax.plot(ks, divs_pct, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Codebook size $K$")
    ax.set_ylabel("Trajectory diversity (%)")
    ax.set_title("Unique Codebook Trajectories vs Codebook Size")
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(0.8, 350)
    ax.set_ylim(-2, 105)
    ax.legend(loc="lower right", framealpha=0.9)

    path = output_dir / f"vqvae_diversity_vs_k.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_self_transition_per_model(all_results, output_dir, fmt):
    """
    Plot 3: Self-transition rate vs K, stratified by BFS category.
    One subplot per model.
    """
    n_models = sum(1 for m in MODELS if m in all_results)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), squeeze=False)
    axes = axes[0]

    plot_idx = 0
    for model_name, model_style in MODELS.items():
        if model_name not in all_results:
            continue

        ax = axes[plot_idx]
        cat_data = extract_self_transition_by_category(all_results[model_name])

        for cat, (ks, rates) in cat_data.items():
            if cat not in BFS_CATEGORY_STYLES:
                continue
            cs = BFS_CATEGORY_STYLES[cat]
            # rates as percentage
            rates_pct = [r * 100 for r in rates]
            ax.plot(ks, rates_pct, color=cs["color"], linestyle=cs["ls"],
                    label=cs["label"], linewidth=1.5, markersize=3, marker="o")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Codebook size $K$")
        if plot_idx == 0:
            ax.set_ylabel("Self-transition rate (%)")
        ax.set_title(model_style["label"])
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlim(0.8, 350)
        ax.set_ylim(-2, 105)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        plot_idx += 1

    fig.suptitle("Self-Transition Rate by BFS Category", fontsize=14, y=1.02)
    fig.tight_layout()

    path = output_dir / f"vqvae_self_transition_by_category.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_summary(all_results, baselines, output_dir, fmt):
    """
    Plot 4: Two-panel summary — accuracy (left) and diversity (right).
    This is the main figure for the paper.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: Accuracy ───────────────────────────────────────────
    for model_name, style in MODELS.items():
        if model_name not in all_results:
            continue
        ks, accs = extract_accuracy(all_results[model_name])
        accs_pct = [a * 100 for a in accs]
        ax1.plot(ks, accs_pct, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=1.8, markersize=5, zorder=3)

        # K=0 as diamond marker at x=0.7
        if model_name in baselines:
            bl = baselines[model_name]
            ax1.plot(0.7, bl["k0"] * 100, marker="D", color=style["color"],
                     markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                     zorder=4)

    ax1.plot([], [], marker="D", color="gray", linestyle="None",
             markersize=5, markeredgecolor="white", markeredgewidth=1,
             label="K=0 (no recurrence)")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Codebook size $K$")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Quantized Recurrence Accuracy")
    ax1.set_xticks(K_VALUES)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlim(0.5, 350)
    ax1.set_ylim(65, 102)
    ax1.legend(loc="lower right", framealpha=0.9)

    # ── Panel B: Diversity ──────────────────────────────────────────
    for model_name, style in MODELS.items():
        if model_name not in all_results:
            continue
        ks, divs = extract_diversity(all_results[model_name])
        divs_pct = [d * 100 for d in divs]
        ax2.plot(ks, divs_pct, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=1.8, markersize=5, zorder=3)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Codebook size $K$")
    ax2.set_ylabel("Trajectory diversity (%)")
    ax2.set_title("(b) Trajectory Diversity")
    ax2.set_xticks(K_VALUES)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlim(0.8, 350)
    ax2.set_ylim(-2, 105)
    ax2.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()

    path = output_dir / f"vqvae_summary.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_with_baselines_table(all_results, baselines, output_dir, fmt):
    """
    Plot 5: Accuracy plot with K=0 baselines plotted as distinct points
    and unquantized K=6 as dashed horizontal lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model_name, style in MODELS.items():
        if model_name not in all_results:
            continue
        ks, accs = extract_accuracy(all_results[model_name])
        accs_pct = [a * 100 for a in accs]

        # Main curve
        ax.plot(ks, accs_pct, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=6, zorder=3)

        if model_name in baselines:
            bl = baselines[model_name]

            # K=0 as a distinct diamond marker at x=0.7 (left of K=1)
            ax.plot(0.7, bl["k0"] * 100, marker="D", color=style["color"],
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    zorder=4)
            # Label it
            ax.annotate(
                f'{bl["k0"]*100:.1f}%',
                xy=(0.7, bl["k0"] * 100),
                xytext=(-15, 8), textcoords="offset points",
                fontsize=7, color=style["color"], fontweight="bold",
                ha="center",
            )

            # Unquantized K=6 as dashed horizontal line
            ax.axhline(bl["unquantized"] * 100, color=style["color"],
                       linestyle="--", linewidth=1.0, alpha=0.4)

    # Add a fake legend entry for K=0 markers
    ax.plot([], [], marker="D", color="gray", linestyle="None",
            markersize=6, markeredgecolor="white", markeredgewidth=1,
            label="K=0 (no recurrence)")
    ax.plot([], [], color="gray", linestyle="--", linewidth=1.0, alpha=0.5,
            label="Unquantized (K=6)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Codebook size $K$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("VQ-VAE Intervention: Accuracy vs Codebook Size")
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(0.5, 350)
    ax.set_ylim(65, 102)
    ax.legend(loc="center right", framealpha=0.9, fontsize=9)

    fig.tight_layout()

    path = output_dir / f"vqvae_accuracy_annotated.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot VQ-VAE experiment results.")
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Base directory containing model subdirs. Default: BASE_DIR/outputs/vqvae/",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures. Default: BASE_DIR/outputs/vqvae/figures/",
    )
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"],
        help="Output format (default: pdf).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else BASE_DIR / "outputs" / "vqvae"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"[INFO] Loading results from {results_dir}")
    all_results = load_eval_results(results_dir)

    if not all_results:
        print("[ERROR] No results found. Check the results directory.")
        return

    print(f"\n[INFO] Loading baselines...")
    baselines = load_baselines(results_dir)
    if not baselines:
        print("[WARN] No baselines found. K=0 and unquantized lines will be missing.")
        print("[WARN] Run remove_thoughts.py first to generate remove_thoughts_model.json files.")

    print(f"\n[INFO] Generating plots...")
    plot_accuracy_vs_k(all_results, baselines, output_dir, args.format)
    plot_diversity_vs_k(all_results, output_dir, args.format)
    plot_self_transition_per_model(all_results, output_dir, args.format)
    plot_combined_summary(all_results, baselines, output_dir, args.format)
    plot_accuracy_with_baselines_table(all_results, baselines, output_dir, args.format)

    print(f"\n[INFO] All figures saved to {output_dir}")


if __name__ == "__main__":
    main()