"""
Regression-based INLP for scalar (continuous) targets.

Method
──────
For a scalar target y, the gradient of a linear predictor f(x) = w^T x
is exactly w. So removing span(w) kills the one-dim direction of steepest
predictor increase. Iterate: fit ridge, remove w, repeat, until a probe on
the projected space can no longer decode y above noise.

Math (per timestep t, per iteration i)
──────────────────────────────────────
# Fit:          w_i = argmin_w ||X_tr w - y_tr||^2 + alpha ||w||^2
# Record:       W_i = [w_1, ..., w_i]
# Projector:    P_i = I - Q Q^T, where Q = orthonormal basis of W_i
# Eval:         r_i = test R^2 of a fresh ridge fit on X P_i^T vs y
# Stop if:      r_i < epsilon  OR  i >= n_max
# Residualize:  X <- X P_i^T  (for next iteration's fit)

The orthogonalized P_i (QR) is the saved projector. All reported R^2
values are evaluated on data passed through this projector — never on
the iteratively-residualized copy, which drifts.

Usage
─────
python -m inlp_regression --model coconut --ridge_alpha 100
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, GSM_TEST, THOUGHTS
from src.utils import extract_answer_number


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def load_labels():
    with open(GSM_TEST) as f:
        data = json.load(f)
    y = []
    for sample in data:
        gold = sample.get("answer", "").replace(",", "").strip()
        if "####" in gold:
            gold = gold.split("####")[-1].strip()
        val = extract_answer_number(gold)
        y.append(val if np.isfinite(val) else np.nan)
    return np.array(y, dtype=np.float32)


# signed_log: y' = sign(y) * log10(1 + |y|)
# Stabilizes the objective; GSM answers span ~5 orders of magnitude.
def signed_log(y):
    return np.sign(y) * np.log10(1.0 + np.abs(y))


# ═══════════════════════════════════════════════════════════════════
# Projections
# ═══════════════════════════════════════════════════════════════════

def nullspace_from_basis(W):
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


def random_projector(D, n_dirs, seed):
    """Random-direction control with matched rank."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_dirs, D))
    Q, _ = np.linalg.qr(R.T)
    Q = Q[:, :n_dirs]
    return (np.eye(D, dtype=np.float32) - Q @ Q.T).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# R^2 helper
# ═══════════════════════════════════════════════════════════════════

def ridge_test_r2(X_tr, y_tr, X_te, y_te, alpha):
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


# ═══════════════════════════════════════════════════════════════════
# Iterative INLP
# ═══════════════════════════════════════════════════════════════════

def run_inlp_timestep(X, y, alpha, n_max, epsilon, test_size, seed):
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
    _, r2_init = ridge_test_r2(X_tr, y_tr_z, X_te, y_te_z, alpha)

    W_removed = []
    r2_history = [r2_init]

    X_tr_cur = X_tr.copy()
    X_te_cur = X_te.copy()

    # Iterate: fit on residualized data, evaluate on projected data
    for i in range(n_max):
        w, _ = ridge_test_r2(X_tr_cur, y_tr_z, X_te_cur, y_te_z, alpha)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-8:
            break

        W_removed.append(w)
        P, n_eff = nullspace_from_basis(np.stack(W_removed, axis=0))

        # Evaluate on data passed through the orthogonalized P
        _, r2_proj = ridge_test_r2(
            X_tr @ P.T, y_tr_z, X_te @ P.T, y_te_z, alpha
        )
        r2_history.append(r2_proj)

        if r2_proj < epsilon:
            break

        # Residualize for next iteration's fit
        P_i_step, _ = nullspace_from_basis((w / w_norm)[None, :])
        X_tr_cur = X_tr_cur @ P_i_step.T
        X_te_cur = X_te_cur @ P_i_step.T

    # Final projector
    if len(W_removed) == 0:
        P_total = np.eye(D, dtype=np.float32)
        n_dirs = 0
    else:
        P_total, n_dirs = nullspace_from_basis(np.stack(W_removed, axis=0))

    # Matched-rank random control
    if n_dirs == 0:
        r2_rand = r2_init
        P_rand = np.eye(D, dtype=np.float32)
    else:
        P_rand = random_projector(D, n_dirs, seed=seed + 7919)
        _, r2_rand = ridge_test_r2(
            X_tr @ P_rand.T, y_tr_z, X_te @ P_rand.T, y_te_z, alpha
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
            w_n, _ = ridge_test_r2(
                X_tr_cur_n, y_tr_null, X_te, y_te_z, alpha,
            )
            wn_norm = np.linalg.norm(w_n)
            if wn_norm < 1e-8:
                break
            W_null.append(w_n)
            P_step_n, _ = nullspace_from_basis((w_n / wn_norm)[None, :])
            X_tr_cur_n = X_tr_cur_n @ P_step_n.T

        if len(W_null) > 0:
            P_null, _ = nullspace_from_basis(np.stack(W_null, axis=0))
            _, r2_null = ridge_test_r2(
                X_tr @ P_null.T, y_tr_z, X_te @ P_null.T, y_te_z, alpha,
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


# ═══════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        default="coconut")
    parser.add_argument("--ridge_alpha", type=float, default=100.0,
                        help="Tune via inlp_diagnostic.py; peak ~100 on GSM.")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Stop when saved-projector R^2 falls below this.")
    parser.add_argument("--n_max", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / "gsm" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load thoughts
    thoughts_path = THOUGHTS / "gsm" / f"thoughts_{args.model}.pt"
    print(f"[INFO] Loading {thoughts_path}")
    thoughts = torch.load(thoughts_path, map_location="cpu",
                          weights_only=False)["thoughts"]

    # Load + transform labels
    y_raw = load_labels()
    keep = np.isfinite(y_raw)
    thoughts = thoughts[keep]
    y = signed_log(y_raw[keep]).astype(np.float32)
    N, T, D = thoughts.shape
    print(f"[INFO] Shape ({N}, {T}, {D}), alpha={args.ridge_alpha}, "
          f"epsilon={args.epsilon}")

    # Run per timestep
    projections = {}
    rand_projections = {}
    inlp_stats = {}

    print(f"\n{'t':>3} {'iters':>6} {'n_dirs':>7} {'R^2_init':>9} "
          f"{'R^2_proj':>9} {'R^2_rand':>9} {'R^2_null':>9}")
    print(f"{'-'*3} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for t in range(T):
        X_t = thoughts[:, t, :].numpy()
        P, P_rand, stats = run_inlp_timestep(
            X_t, y,
            alpha=args.ridge_alpha, n_max=args.n_max, epsilon=args.epsilon,
            test_size=args.test_size, seed=args.seed,
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

    # Save (drop-in compatible with inlp_results.pt consumers)
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


if __name__ == "__main__":
    main()