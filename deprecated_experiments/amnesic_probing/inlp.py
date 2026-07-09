"""
Unified INLP: Iterative Nullspace Projection for ProsQA and GSM.

Dispatches on --task:
  prosqa  -> classification mode (one-vs-rest LinearSVC, GPU via cuML/CuPy,
             multiprocessing across timesteps). Labels are concept strings.
  gsm     -> regression mode (ridge regression, CPU/NumPy, sequential).
             Labels are scalar answers, transformed via signed_log.

The output schema (inlp_results.pt) is identical across modes so that
downstream consumers (interventions.py) work unchanged. Regression-only
fields (r2_*, regression_config) are absent for prosqa; classification-only
fields (label_to_idx, all_labels) are None for gsm.

Usage:
    python -m inlp --task prosqa --model pause --max_workers 4
    python -m inlp --task gsm    --model coconut --ridge_alpha 100
"""

import json
import torch
import argparse
import subprocess
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, THOUGHTS
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )

# ═══════════════════════════════════════════════════════════════════
# Shared: thoughts loading
# ═══════════════════════════════════════════════════════════════════

def load_thoughts(task, model, n_thoughts=None, max_instances=None,
                  auto_extract=False):
    """Load extracted thoughts; optionally trigger extraction if missing."""
    thoughts_path = THOUGHTS / task / f"thoughts_{model}.pt"
    if not thoughts_path.exists() and auto_extract:
        print("  [INFO] Triggering thought extraction...")
        cmd = ["python", "-u", "-m",
               "experiments.probe_thoughts.extract_thoughts",
               "--task", task, "--model", model]
        if n_thoughts is not None:
            cmd.extend(["--n_thoughts", str(n_thoughts)])
        if max_instances:
            cmd.extend(["--max_instances", str(max_instances)])
        subprocess.run(cmd, check=True)

    print(f"  Loading thoughts from {thoughts_path}")
    return torch.load(thoughts_path, map_location="cpu",
                      weights_only=False)["thoughts"]


# ╔═════════════════════════════════════════════════════════════════╗
# ║                   CLASSIFICATION MODE (ProsQA)                  ║
# ╚═════════════════════════════════════════════════════════════════╝

def _import_gpu_backend():
    """Lazy GPU imports — only when classification mode is active."""
    import cupy as cp
    try:
        from cuml.svm import LinearSVC as cuLinearSVC
        print("Using GPU (cuML)")
    except ImportError:
        from sklearn.svm import LinearSVC
        print("Using CPU (sklearn)")

        class cuLinearSVC:
            def __init__(self, **kwargs):
                kwargs.pop('output_type', None)
                self.clf = LinearSVC(**kwargs)

            def fit(self, X, y):
                self.clf.fit(cp.asnumpy(X), cp.asnumpy(y))
                self.coef_ = cp.asarray(self.clf.coef_)
                return self

            def predict(self, X):
                return cp.asarray(self.clf.predict(cp.asnumpy(X)))

    return cp, cuLinearSVC


# ── Data ──────────────────────────────────────────────────────────

def load_prosqa_labels(max_instances=None):
    """ProsQA label = concept after 'is a' in the gold answer."""
    with open(PROSQA_TEST) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]

    labels, all_labels = [], set()
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
    return labels, sorted(all_labels), len(data)


# ── INLP (GPU) ────────────────────────────────────────────────────

def _cls_nullspace_projection(W, cp):
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


def _cls_random_projection(D, n_directions_removed, cp, seed=42):
    """Matched-rank random-direction control."""
    cp.random.seed(seed)
    random_dirs = cp.random.randn(n_directions_removed, D)
    Q, _ = cp.linalg.qr(random_dirs.T)
    n_actual = min(n_directions_removed, D)
    basis = Q[:, :n_actual].T
    return cp.eye(D, dtype=random_dirs.dtype) - basis.T @ basis


def _cls_run_inlp_gpu(X_np, y_np, cp, cuLinearSVC,
                      max_iter=300, convergence_threshold=1.0):
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
        P_i = _cls_nullspace_projection(W, cp)
        P_total = P_i @ P_total
        X_projected = X @ P_total.T
        n_directions_removed += W.shape[0]

    return cp.asnumpy(P_total), iteration + 1, n_directions_removed


def _cls_process_single_step(t, X_np, y_np, D):
    """Worker: run INLP for a single timestep on GPU."""
    cp, cuLinearSVC = _import_gpu_backend()
    try:
        P_np, n_iter, n_dirs = _cls_run_inlp_gpu(X_np, y_np, cp, cuLinearSVC)
        rand_P_np = cp.asnumpy(
            _cls_random_projection(D, n_dirs, cp, seed=42 + t)
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


def _cls_run_all_steps(thoughts, labels, all_labels, max_workers):
    label_to_idx = {c: i for i, c in enumerate(all_labels)}
    y_np = np.array([label_to_idx[l] for l in labels], dtype=np.float32)

    N, T, D = thoughts.shape
    projections, rand_projections, inlp_stats = {}, {}, {}

    print(f"\n  Running INLP for {T} steps, {len(all_labels)} unique "
          f"labels, {N} instances")
    print(f"  Using {max_workers} concurrent GPU workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _cls_process_single_step,
                t, thoughts[:, t, :].numpy(), y_np, D,
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


def run_classification(args):
    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / "prosqa" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    thoughts = load_thoughts("prosqa", args.model,
                             n_thoughts=args.n_thoughts,
                             max_instances=args.max_instances,
                             auto_extract=True)
    labels, all_labels, n_data = load_prosqa_labels(args.max_instances)
    print(f"[INFO] Model: {args.model}, instances: {n_data}, "
          f"labels: {len(all_labels)} unique")

    print("\n" + "=" * 60)
    print("INLP: COMPUTING NULLSPACE PROJECTIONS (GPU)")
    print("=" * 60)

    projections, rand_projections, inlp_stats, label_to_idx = \
        _cls_run_all_steps(thoughts, labels, all_labels, args.max_workers)

    save_path = output_dir / "inlp_results.pt"
    torch.save({
        "projections": projections,
        "rand_projections": rand_projections,
        "inlp_stats": inlp_stats,
        "label_to_idx": label_to_idx,
        "all_labels": all_labels,
    }, save_path)
    print(f"\n  Saved to {save_path}")

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


# ╔═════════════════════════════════════════════════════════════════╗
# ║                    REGRESSION MODE (GSM)                        ║
# ╚═════════════════════════════════════════════════════════════════╝

# CPU-only deps, imported at module top would be fine but we keep them
# scoped to avoid sklearn import on classification-only runs.

def _import_cpu_backend():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from src.utils import extract_answer_number
    return Ridge, train_test_split, extract_answer_number


# ── Data ──────────────────────────────────────────────────────────

def load_gsm_labels(extract_answer_number):
    with open(GSM_TEST) as f:
        data = json.load(f)
    y = []
    for sample in data:
        gold = sample.get("answer", "").replace(",", "").strip()
        if "####" in gold:
            gold = gold.split("####")[-1].strip()
        val = extract_answer_number(gold, task="gsm")
        y.append(val if np.isfinite(val) else np.nan)
    return np.array(y, dtype=np.float32)


# signed_log: y' = sign(y) * log10(1 + |y|)
# Stabilizes the objective; GSM answers span ~5 orders of magnitude.
def signed_log(y):
    return np.sign(y) * np.log10(1.0 + np.abs(y))


# ── Projections (CPU) ─────────────────────────────────────────────

def _reg_nullspace_from_basis(W):
    """
    Build the nullspace projector from a set of (possibly non-orthogonal)
    direction vectors.

    # W: (k, D) matrix of removed directions
    # QR decomposition of W^T gives orthonormal columns spanning the same
    # subspace: W^T = Q R, with Q (D, k) having orthonormal columns
    # P = I - Q Q^T projects onto the orthogonal complement of span(W)
    """
    D = W.shape[1]
    Q, _ = np.linalg.qr(W.T.astype(np.float64))
    # Drop numerically-zero columns (e.g. if a later w_i was nearly in
    # span of earlier ones)
    col_norms = np.linalg.norm(Q, axis=0)
    Q = Q[:, col_norms > 1e-6]
    P = np.eye(D, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
    return P, int(Q.shape[1])


def _reg_random_projector(D, n_dirs, seed):
    """Random-direction control with matched rank."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_dirs, D))
    Q, _ = np.linalg.qr(R.T)
    Q = Q[:, :n_dirs]
    return (np.eye(D, dtype=np.float32) - Q @ Q.T).astype(np.float32)


# ── R^2 helper ────────────────────────────────────────────────────

def _ridge_test_r2(X_tr, y_tr, X_te, y_te, alpha, Ridge):
    """
    # R^2 = 1 - SS_res / SS_tot, on the held-out test split.
    """
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_tr, y_tr)
    w = reg.coef_.astype(np.float32)
    y_pred = reg.predict(X_te)
    ss_res = float(np.sum((y_te - y_pred) ** 2))
    ss_tot = float(np.sum(y_te ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return w, r2


# ── Iterative INLP ────────────────────────────────────────────────

def _reg_run_inlp_timestep(X, y, alpha, n_max, epsilon, test_size, seed,
                           Ridge, train_test_split):
    """
    Iteratively remove probe directions until the saved projector
    reduces test R^2 below epsilon (or n_max is reached).
    """
    N, D = X.shape
    idx_tr, idx_te = train_test_split(
        np.arange(N), test_size=test_size, random_state=seed,
    )
    X_tr = X[idx_tr].astype(np.float64)
    X_te = X[idx_te].astype(np.float64)
    y_tr = y[idx_tr].astype(np.float64)
    y_te = y[idx_te].astype(np.float64)

    # z-score y on train so R^2 is interpretable
    y_tr_z = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-8)
    y_te_z = (y_te - y_tr.mean()) / (y_tr.std() + 1e-8)

    # Initial probe on unprojected data
    _, r2_init = _ridge_test_r2(X_tr, y_tr_z, X_te, y_te_z, alpha, Ridge)

    W_removed = []
    r2_history = [r2_init]

    X_tr_cur = X_tr.copy()
    X_te_cur = X_te.copy()

    # Iterate: fit on residualized data, evaluate on projected data
    for i in range(n_max):
        w, _ = _ridge_test_r2(X_tr_cur, y_tr_z, X_te_cur, y_te_z, alpha, Ridge)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-8:
            break

        W_removed.append(w)
        P, n_eff = _reg_nullspace_from_basis(np.stack(W_removed, axis=0))

        # Evaluate on data passed through the orthogonalized P
        _, r2_proj = _ridge_test_r2(
            X_tr @ P.T, y_tr_z, X_te @ P.T, y_te_z, alpha, Ridge,
        )
        r2_history.append(r2_proj)

        if r2_proj < epsilon:
            break

        # Residualize for next iteration's fit
        P_i_step, _ = _reg_nullspace_from_basis((w / w_norm)[None, :])
        X_tr_cur = X_tr_cur @ P_i_step.T
        X_te_cur = X_te_cur @ P_i_step.T

    # Final projector
    if len(W_removed) == 0:
        P_total = np.eye(D, dtype=np.float32)
        n_dirs = 0
    else:
        P_total, n_dirs = _reg_nullspace_from_basis(
            np.stack(W_removed, axis=0)
        )

    # Matched-rank random control
    if n_dirs == 0:
        r2_rand = r2_init
        P_rand = np.eye(D, dtype=np.float32)
    else:
        P_rand = _reg_random_projector(D, n_dirs, seed=seed + 7919)
        _, r2_rand = _ridge_test_r2(
            X_tr @ P_rand.T, y_tr_z, X_te @ P_rand.T, y_te_z, alpha, Ridge,
        )

    # Null-labels control: run the same iterative procedure with y
    # shuffled, measure R^2 of the real probe on the resulting
    # null-projector. If real-INLP's R^2_proj and null-INLP's R^2_null
    # are similar, the intervention is non-specific — any n_dirs
    # directions fit to any scalar would kill R^2 equally well.
    #
    # # y_null = permutation of y_tr  (same distribution, no label info)
    # # Run INLP with y_null -> P_null (same n_dirs by construction
    # #                                  since epsilon drives stopping)
    # # Evaluate real ridge on X P_null^T
    r2_null = None
    if n_dirs > 0:
        rng = np.random.default_rng(seed + 31337)
        y_tr_null = y_tr_z.copy()
        rng.shuffle(y_tr_null)

        W_null = []
        X_tr_cur_n = X_tr.copy()
        # Fit exactly n_dirs null directions to match rank
        for _ in range(n_dirs):
            w_n, _ = _ridge_test_r2(
                X_tr_cur_n, y_tr_null, X_te, y_te_z, alpha, Ridge,
            )
            wn_norm = np.linalg.norm(w_n)
            if wn_norm < 1e-8:
                break
            W_null.append(w_n)
            P_step_n, _ = _reg_nullspace_from_basis(
                (w_n / wn_norm)[None, :]
            )
            X_tr_cur_n = X_tr_cur_n @ P_step_n.T

        if len(W_null) > 0:
            P_null, _ = _reg_nullspace_from_basis(np.stack(W_null, axis=0))
            _, r2_null = _ridge_test_r2(
                X_tr @ P_null.T, y_tr_z, X_te @ P_null.T, y_te_z, alpha, Ridge,
            )

    r2_proj_final = r2_history[-1] if len(W_removed) > 0 else r2_init

    stats = {
        "n_iterations": len(W_removed),
        "n_directions_removed": n_dirs,
        "r2_init": r2_init,
        "r2_proj": r2_proj_final,
        "r2_rand": r2_rand,
        "r2_null": r2_null,
        "r2_history": r2_history,
    }
    return P_total, P_rand, stats


def run_regression(args):
    Ridge, train_test_split, extract_answer_number = _import_cpu_backend()

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / "gsm" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    thoughts = load_thoughts("gsm", args.model, auto_extract=False)

    # Load + transform labels; drop rows whose gold parse failed
    y_raw = load_gsm_labels(extract_answer_number)
    keep = np.isfinite(y_raw)
    thoughts = thoughts[keep]
    y = signed_log(y_raw[keep]).astype(np.float32)
    N, T, D = thoughts.shape
    print(f"[INFO] Shape ({N}, {T}, {D}), alpha={args.ridge_alpha}, "
          f"epsilon={args.epsilon}")

    projections, rand_projections, inlp_stats = {}, {}, {}

    print(f"\n{'t':>3} {'iters':>6} {'n_dirs':>7} {'R^2_init':>9} "
          f"{'R^2_proj':>9} {'R^2_rand':>9} {'R^2_null':>9}")
    print(f"{'-'*3} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for t in range(T):
        X_t = thoughts[:, t, :].numpy()
        P, P_rand, stats = _reg_run_inlp_timestep(
            X_t, y,
            alpha=args.ridge_alpha, n_max=args.n_max, epsilon=args.epsilon,
            test_size=args.test_size, seed=args.seed,
            Ridge=Ridge, train_test_split=train_test_split,
        )
        projections[t] = P
        rand_projections[t] = P_rand
        inlp_stats[t] = stats
        r2_null_str = (f"{stats['r2_null']:>+9.3f}"
                       if stats['r2_null'] is not None else f"{'—':>9}")
        print(f"{t:>3} {stats['n_iterations']:>6} "
              f"{stats['n_directions_removed']:>7} "
              f"{stats['r2_init']:>+9.3f} {stats['r2_proj']:>+9.3f} "
              f"{stats['r2_rand']:>+9.3f} {r2_null_str}")

    save_path = output_dir / "inlp_results.pt"
    torch.save({
        "projections": projections,
        "rand_projections": rand_projections,
        "inlp_stats": inlp_stats,
        "label_to_idx": None,
        "all_labels": None,
        "regression_config": {
            "ridge_alpha": args.ridge_alpha,
            "epsilon": args.epsilon,
            "n_max": args.n_max,
            "target_transform": "signed_log",
            "test_size": args.test_size,
            "seed": args.seed,
        },
    }, save_path)
    print(f"\n[INFO] Saved to {save_path}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║                            CLI                                  ║
# ╚═════════════════════════════════════════════════════════════════╝

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified INLP for ProsQA (classification, GPU) and "
                    "GSM (regression, CPU).",
    )
    parser.add_argument(
        "--task", choices=["prosqa", "gsm"], required=True,
        help="prosqa -> classification mode; gsm -> regression mode.",
    )
    parser.add_argument(
        "--model", choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    # Classification-only
    cls = parser.add_argument_group("classification (--task prosqa)")
    cls.add_argument("--n_thoughts", type=int, default=6)
    cls.add_argument("--max_instances", type=int, default=None)
    # Default 1 to avoid GPU OOM; increase with high VRAM.
    cls.add_argument("--max_workers", type=int, default=1)

    # Regression-only
    reg = parser.add_argument_group("regression (--task gsm)")
    reg.add_argument("--ridge_alpha", type=float, default=100.0,
                     help="Tune via inlp_diagnostic.py; peak ~100 on GSM.")
    reg.add_argument("--epsilon", type=float, default=0.02,
                     help="Stop when saved-projector R^2 falls below this.")
    reg.add_argument("--n_max", type=int, default=50)
    reg.add_argument("--test_size", type=float, default=0.2)
    reg.add_argument("--seed", type=int, default=0)

    return parser


def main():
    args = build_parser().parse_args()
    if args.task == "prosqa":
        # CRITICAL: 'spawn' required for CUDA + multiprocessing in Python
        mp.set_start_method('spawn', force=True)
        run_classification(args)
    else:
        run_regression(args)


if __name__ == "__main__":
    main()