"""
INLP: Train classifiers and compute nullspace projections on ProsQA.

Accelerated with multiprocessing (CPU) and RAPIDS cuML + CuPy (GPU).
Outputs inlp_results.pt for use by interventions.py.

ProsQA labels are graph-concept strings, so the problem is genuinely
multi-class (n_classes per iteration of INLP). For scalar regression
targets (GSM), use inlp_regression.py instead.

Usage:
    python -m inlp_classification --model pause --max_workers 4
"""

import json
import torch
import argparse
import subprocess
import cupy as cp
import numpy as np
from pathlib import Path
import multiprocessing as mp
from cuml.svm import LinearSVC as cuLinearSVC
from src.config import BASE_DIR, PROSQA_TEST, THOUGHTS
from concurrent.futures import ProcessPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def load_data(max_instances=None):
    with open(PROSQA_TEST) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def extract_labels(data):
    """ProsQA label = the concept after 'is a' in the gold answer."""
    labels = []
    all_labels = set()
    for sample in data:
        answer_text = sample["answer"].strip().rstrip(".")
        correct_concept = answer_text.split(" is a ")[-1].strip()
        labels.append(correct_concept)
        all_labels.add(correct_concept)
        # Include every concept mentioned in reasoning steps so the
        # label set covers the full concept vocabulary
        for step in sample.get("steps", []):
            parts = step.split(" is a ")
            if len(parts) == 2:
                all_labels.add(parts[0].replace("Every ", "").strip())
                all_labels.add(parts[1].strip().rstrip("."))
    return labels, sorted(all_labels)


# ═══════════════════════════════════════════════════════════════════
# INLP (GPU accelerated)
# ═══════════════════════════════════════════════════════════════════

def compute_nullspace_projection_gpu(W):
    """
    # P = I - V^T V, where V = top right-singular vectors of W
    # rank(V) = rank(W), so P projects onto the orthogonal complement
    # of row-span(W) = span of SVM hyperplane normals
    """
    if W.ndim == 1:
        W = W.reshape(1, -1)
    U, S, Vt = cp.linalg.svd(W, full_matrices=False)
    rank = cp.sum(S > 1e-10)
    basis = Vt[:rank, :]
    return cp.eye(W.shape[1], dtype=W.dtype) - basis.T @ basis


def compute_random_projection_gpu(D, n_directions_removed, seed=42):
    """Matched-rank random-direction control."""
    cp.random.seed(seed)
    random_dirs = cp.random.randn(n_directions_removed, D)
    Q, _ = cp.linalg.qr(random_dirs.T)
    n_actual = min(n_directions_removed, D)
    basis = Q[:, :n_actual].T
    return cp.eye(D, dtype=random_dirs.dtype) - basis.T @ basis


def run_inlp_gpu(X_np, y_np, max_iter=300, convergence_threshold=1.0):
    """
    Iterative Nullspace Projection on GPU.

    Each iteration fits a one-vs-rest linear SVM, returning a
    (n_classes, D) weight matrix W. Removing row-span(W) nulls the
    entire set of class hyperplanes simultaneously.

    Stop when train accuracy falls within `convergence_threshold` of
    the majority-class baseline.
    """
    X = cp.asarray(X_np, dtype=cp.float32)
    y = cp.asarray(y_np, dtype=cp.float32)

    N, D = X.shape
    majority_acc = int(cp.max(cp.bincount(y.astype(cp.int32)))) / N * 100

    P_total = cp.eye(D, dtype=cp.float32)
    X_projected = X.copy()
    n_directions_removed = 0

    for iteration in range(max_iter):
        clf = cuLinearSVC(max_iter=5000, C=0.1, output_type='cupy')
        clf.fit(X_projected, y)
        preds = clf.predict(X_projected)
        acc = float(cp.mean(preds == y)) * 100

        if acc <= majority_acc + convergence_threshold:
            break

        W = clf.coef_
        P_i = compute_nullspace_projection_gpu(W)
        P_total = P_i @ P_total
        X_projected = X @ P_total.T
        n_directions_removed += W.shape[0]

    return cp.asnumpy(P_total), iteration + 1, n_directions_removed


# ═══════════════════════════════════════════════════════════════════
# Multiprocessing wrapper
# ═══════════════════════════════════════════════════════════════════

def _process_single_step(t, X_np, y_np, D):
    """Worker: run INLP for a single timestep on GPU."""
    try:
        P_np, n_iter, n_dirs = run_inlp_gpu(X_np, y_np)
        rand_P_np = cp.asnumpy(
            compute_random_projection_gpu(D, n_dirs, seed=42 + t)
        )

        # Verification: probe accuracy before and after INLP
        X_gpu = cp.asarray(X_np, dtype=cp.float32)
        y_gpu = cp.asarray(y_np, dtype=cp.float32)
        P_gpu = cp.asarray(P_np, dtype=cp.float32)
        X_clean = X_gpu @ P_gpu.T

        clf_v = cuLinearSVC(max_iter=5000, C=0.1, output_type='cupy')
        clf_v.fit(X_clean, y_gpu)
        verify_acc = float(cp.mean(clf_v.predict(X_clean) == y_gpu)) * 100

        clf_o = cuLinearSVC(max_iter=5000, C=0.1, output_type='cupy')
        clf_o.fit(X_gpu, y_gpu)
        orig_acc = float(cp.mean(clf_o.predict(X_gpu) == y_gpu)) * 100

        stats = {
            "original_probe_acc": orig_acc,
            "post_inlp_probe_acc": verify_acc,
            "n_iterations": n_iter,
            "n_directions_removed": n_dirs,
        }
        return t, P_np, rand_P_np, stats

    except Exception as e:
        print(f"Error in step {t}: {e}")
        raise e


def run_inlp_all_steps_parallel(thoughts, labels, all_labels, max_workers):
    label_to_idx = {c: i for i, c in enumerate(all_labels)}
    y_np = np.array([label_to_idx[l] for l in labels], dtype=np.float32)

    N, T, D = thoughts.shape
    projections = {}
    rand_projections = {}
    inlp_stats = {}

    print(f"\n  Running INLP for {T} steps, {len(all_labels)} unique "
          f"labels, {N} instances")
    print(f"  Using {max_workers} concurrent GPU workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_step, t, thoughts[:, t, :].numpy(), y_np, D,
            ): t
            for t in range(T)
        }
        for future in as_completed(futures):
            t, P, rand_P, stats = future.result()
            projections[t] = P
            rand_projections[t] = rand_P
            inlp_stats[t] = stats

            print(f"    Step {t:>2} completed: "
                  f"Orig {stats['original_probe_acc']:>5.1f}% -> "
                  f"Post {stats['post_inlp_probe_acc']:>5.1f}% "
                  f"({stats['n_directions_removed']} dirs removed)")

    return projections, rand_projections, inlp_stats, label_to_idx


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="INLP: train classifiers and save nullspace "
                    "projections for ProsQA (GPU)."
    )
    parser.add_argument(
        "--model", type=str,
        choices=["coconut", "coconut_u", "pause"],
        default="coconut",
        help="CODI is GSM-only and uses regression INLP.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    # Default 2 to avoid GPU OOM; increase with high VRAM.
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / "prosqa" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load thoughts ──────────────────────────────────────────────
    thoughts_path = THOUGHTS / "prosqa" / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        print("  [INFO] Triggering thought extraction...")
        cmd = ["python", "-u", "-m",
               "experiments.probe_thoughts.extract_thoughts",
               "--task", "prosqa", "--model", args.model,
               "--n_thoughts", str(args.n_thoughts)]
        if args.max_instances:
            cmd.extend(["--max_instances", str(args.max_instances)])
        subprocess.run(cmd, check=True)

    print(f"  Loading thoughts from {thoughts_path}")
    thoughts = torch.load(thoughts_path, map_location="cpu",
                          weights_only=False)["thoughts"]

    data = load_data(args.max_instances)
    labels, all_labels = extract_labels(data)
    print(f"[INFO] Model: {args.model}, instances: {len(data)}, "
          f"labels: {len(all_labels)} unique")

    # ── INLP ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INLP: COMPUTING NULLSPACE PROJECTIONS (GPU)")
    print("=" * 60)

    projections, rand_projections, inlp_stats, label_to_idx = \
        run_inlp_all_steps_parallel(
            thoughts, labels, all_labels, args.max_workers,
        )

    # ── Save ────────────────────────────────────────────────────────
    save_path = output_dir / "inlp_results.pt"
    torch.save({
        "projections": projections,
        "rand_projections": rand_projections,
        "inlp_stats": inlp_stats,
        "label_to_idx": label_to_idx,
        "all_labels": all_labels,
    }, save_path)
    print(f"\n  Saved to {save_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  INLP SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Step':>6}  {'Orig Acc':>10}  {'Post-INLP':>10}  "
          f"{'Dirs Removed':>14}")
    for t in sorted(inlp_stats.keys()):
        s = inlp_stats[t]
        print(f"  {t:>6}  {s['original_probe_acc']:>9.1f}%  "
              f"{s['post_inlp_probe_acc']:>9.1f}%  "
              f"{s['n_directions_removed']:>14}")


if __name__ == "__main__":
    # CRITICAL: 'spawn' required for CUDA + multiprocessing in Python
    mp.set_start_method('spawn', force=True)
    main()