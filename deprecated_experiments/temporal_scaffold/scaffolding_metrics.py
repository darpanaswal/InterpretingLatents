"""
scaffolding_metrics.py

Shared metric functions for the scaffolding / PE-ablation experiments.
These are the primitives used by:
  - pe_ablation.py
  - noise_input_test.py

Six public functions:

    variance_decomposition(thoughts)      -> dict
    fit_lda_subspace(thoughts)            -> (Q, lda)
    lda_cluster_separation(thoughts)      -> float
    principal_angles(Q_a, Q_b)            -> (cos_angles, rad_angles)
    held_out_timestep_accuracy(
        train_thoughts, test_thoughts, Q) -> float
    split_instances(thoughts, test_frac=0.3, seed=0)
                                          -> (train, test, idx_train, idx_test)

All functions accept either torch.Tensor or np.ndarray thought tensors of
shape (N, T, D) where:
    N = number of instances
    T = number of timesteps (thought positions)
    D = hidden dimension

Internally the functions reshape to (N*T, D) with a timestep-label vector
y = tile(arange(T), N). LDA-based functions use sklearn.
"""

import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )


# ═══════════════════════════════════════════════════════════════════
# Shared reshape helper
# ═══════════════════════════════════════════════════════════════════

def _as_numpy(thoughts):
    """torch.Tensor or ndarray -> float64 ndarray, shape preserved."""
    if isinstance(thoughts, torch.Tensor):
        thoughts = thoughts.detach().cpu().numpy()
    return np.asarray(thoughts, dtype=np.float64)


def _reshape_xy(thoughts):
    """
    Flatten (N, T, D) to (N*T, D) with timestep labels.

    # X in R^{(N*T) x D}, y in {0,...,T-1}^{N*T}
    # y is arranged so that row (i*T + t) has label t (trajectory-major order).
    """
    arr = _as_numpy(thoughts)
    N, T, D = arr.shape
    X = arr.reshape(N * T, D)
    y = np.tile(np.arange(T), N)
    return X, y, N, T, D


# ═══════════════════════════════════════════════════════════════════
# Variance decomposition
# ═══════════════════════════════════════════════════════════════════

def variance_decomposition(thoughts):
    """
    Additive decomposition of total variance into timestep, instance, and
    residual components.

    # mu      = E_{i,t}[ h_{i,t} ]                                 in R^D
    # mu_t    = E_i    [ h_{i,t} ]                                 in R^{T x D}
    # mu_i    = E_t    [ h_{i,t} ]                                 in R^{N x D}
    # var_total    = E_{i,t}[ ||h_{i,t} - mu||^2 ]
    # var_timestep = E_t [ ||mu_t - mu||^2 ]        (between-timestep)
    # var_instance = E_i [ ||mu_i - mu||^2 ]        (between-instance)
    # var_residual = var_total - var_timestep - var_instance
    #
    # pct_*        = 100 * var_* / var_total
    #
    # By definition pct_timestep + pct_instance + pct_residual = 100.
    # This decomposition is exact under the additive model
    #     h_{i,t} = mu + (mu_t - mu) + (mu_i - mu) + eps_{i,t}
    # and approximately additive otherwise; the residual absorbs cross-terms.
    """
    # Use torch for the reductions to match the existing diagnose output
    # exactly (bit-for-bit where the input tensor was already torch).
    if isinstance(thoughts, np.ndarray):
        thoughts = torch.from_numpy(thoughts)

    mu   = thoughts.mean(dim=(0, 1))                 # (D,)
    mu_t = thoughts.mean(dim=0)                      # (T, D)
    mu_i = thoughts.mean(dim=1)                      # (N, D)

    var_total    = ((thoughts - mu) ** 2).sum(dim=2).mean().item()
    var_timestep = ((mu_t     - mu) ** 2).sum(dim=1).mean().item()
    var_instance = ((mu_i     - mu) ** 2).sum(dim=1).mean().item()
    var_residual = var_total - var_timestep - var_instance

    denom = max(var_total, 1e-12)
    return {
        "var_total":    var_total,
        "var_timestep": var_timestep,
        "var_instance": var_instance,
        "var_residual": var_residual,
        "pct_timestep": 100.0 * var_timestep / denom,
        "pct_instance": 100.0 * var_instance / denom,
        "pct_residual": 100.0 * var_residual / denom,
    }


# ═══════════════════════════════════════════════════════════════════
# LDA subspace fit
# ═══════════════════════════════════════════════════════════════════

def fit_lda_subspace(thoughts):
    """
    Fit LDA discriminating timesteps pooled across instances; return an
    orthonormal basis of the discriminant subspace.

    # LDA.scalings_[:, :T-1]         -> W in R^{D x (T-1)}  (unnormalised)
    # Q, _ = QR(W)                   -> Q in R^{D x r},  Q^T Q = I_r
    # Columns with near-zero norm after QR are dropped (numerical guard).
    #
    # Return order is (Q, lda) so callers can do:  Q, _ = fit_lda_subspace(...)
    """
    X, y, N, T, D = _reshape_xy(thoughts)
    lda = LinearDiscriminantAnalysis(n_components=T - 1)
    lda.fit(X, y)
    W = lda.scalings_[:, :T - 1]
    Q, _ = np.linalg.qr(W)
    col_norms = np.linalg.norm(Q, axis=0)
    Q = Q[:, col_norms > 1e-8]
    return Q.astype(np.float32), lda


# ═══════════════════════════════════════════════════════════════════
# Fisher separation in LDA-projected space
# ═══════════════════════════════════════════════════════════════════

def lda_cluster_separation(thoughts):
    """
    Fisher ratio S_b / S_w in the LDA-projected space.  Large value means
    timestep clusters are tight around their centroids relative to how far
    apart the centroids are; small value means clusters overlap.

    # Let Z = X @ W  with Z in R^{(N*T) x (T-1)}  (sklearn LDA transform).
    # mu  = mean(Z)                                    overall centroid
    # mu_t= mean_{y_i=t}(z_i)                          per-class centroid
    # S_b = sum_t N_t * ||mu_t - mu||^2                between-class scatter
    # S_w = sum_t sum_{y_i=t} ||z_i - mu_t||^2         within-class scatter
    # separation = S_b / max(S_w, eps)
    """
    X, y, N, T, D = _reshape_xy(thoughts)
    lda = LinearDiscriminantAnalysis(n_components=T - 1)
    Z = lda.fit_transform(X, y)

    mu = Z.mean(axis=0)
    S_b, S_w = 0.0, 0.0
    for t in range(T):
        mask = (y == t)
        Zt = Z[mask]
        mu_t = Zt.mean(axis=0)
        S_b += mask.sum() * np.sum((mu_t - mu) ** 2)
        S_w += np.sum((Zt - mu_t) ** 2)
    return float(S_b / max(S_w, 1e-12))


# ═══════════════════════════════════════════════════════════════════
# Principal angles between two subspaces
# ═══════════════════════════════════════════════════════════════════

def principal_angles(Q_a, Q_b):
    """
    Principal angles between the column spans of two orthonormal bases.

    # Given Q_a in R^{D x r_a}, Q_b in R^{D x r_b} with orthonormal columns,
    # form M = Q_a^T @ Q_b  in R^{r_a x r_b}.
    # Singular values sigma_1 >= ... >= sigma_r  with  r = min(r_a, r_b)
    # are the cosines of the principal angles:
    #     cos(theta_i) = sigma_i
    #     theta_i      = arccos( clip(sigma_i, -1, 1) )
    #
    # Clipping is a numerical safeguard: Q_a^T Q_b can have singular values
    # slightly above 1 due to floating-point error, which would produce NaN
    # from arccos.
    #
    # Interpretation:
    #   cos = 1  -> the i-th most-aligned direction pair is identical
    #   cos = 0  -> the i-th pair is already orthogonal (and so are all pairs
    #               with index > i): the subspaces share at most an
    #               (i-1)-dimensional intersection.
    """
    Q_a = np.asarray(Q_a, dtype=np.float64)
    Q_b = np.asarray(Q_b, dtype=np.float64)
    M = Q_a.T @ Q_b
    sigmas = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(sigmas, -1.0, 1.0)
    rad_angles = np.arccos(cos_angles)
    return cos_angles, rad_angles


# ═══════════════════════════════════════════════════════════════════
# Instance-level train / test split
# ═══════════════════════════════════════════════════════════════════

def split_instances(thoughts, test_frac=0.3, seed=0):
    """
    Partition instances (not individual timesteps) into train / test.  Each
    instance's full trajectory of length T stays together in exactly one side
    of the split.

    # rng           = np.random.default_rng(seed)
    # idx           = rng.permutation(N)
    # n_test        = int(test_frac * N)
    # idx_test      = idx[:n_test]          (instance indices)
    # idx_train     = idx[n_test:]
    # train_thoughts= thoughts[idx_train]   in R^{(N - n_test) x T x D}
    # test_thoughts = thoughts[idx_test]    in R^{n_test       x T x D}
    #
    # Instance-level splitting is required because timesteps within the same
    # instance share instance identity (the mu_i component), and mixing them
    # across train/test would leak instance information into the classifier.
    """
    arr = _as_numpy(thoughts)
    N, T, D = arr.shape

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_test = int(test_frac * N)
    idx_test  = idx[:n_test]
    idx_train = idx[n_test:]

    train_thoughts = arr[idx_train]
    test_thoughts  = arr[idx_test]
    return train_thoughts, test_thoughts, idx_train, idx_test


# ═══════════════════════════════════════════════════════════════════
# Held-out timestep classification in a given subspace Q
# ═══════════════════════════════════════════════════════════════════

def held_out_timestep_accuracy(train_thoughts, test_thoughts, Q):
    """
    Fit an LDA classifier on train thoughts projected through Q; score on
    test thoughts projected through the same Q.

    # Q in R^{D x r}    (orthonormal basis of the scaffold subspace)
    # X_tr = train.reshape(-1, D) @ Q   in R^{(N_tr * T) x r}
    # X_te = test .reshape(-1, D) @ Q   in R^{(N_te * T) x r}
    # y_tr = tile(arange(T), N_tr)
    # y_te = tile(arange(T), N_te)
    # clf  = LDA().fit(X_tr, y_tr)
    # return  clf.score(X_te, y_te)     (fraction of test samples for which
    #                                    argmax prediction equals true timestep)
    #
    # Projecting through Q before fitting restricts the classifier to the
    # scaffold subspace.  This is the right evaluation when the question is
    # "does the subspace identified on the train data carry timestep info
    # that generalises to unseen instances?"  — which is the hypothesis the
    # pe_ablation pipeline is testing.
    """
    train = _as_numpy(train_thoughts)
    test  = _as_numpy(test_thoughts)
    Q_arr = np.asarray(Q, dtype=np.float64)

    N_tr, T, D = train.shape
    N_te, T_te, D_te = test.shape
    assert T == T_te and D == D_te, (
        f"train/test shape mismatch: train={train.shape}, test={test.shape}"
    )
    assert Q_arr.shape[0] == D, (
        f"Q has {Q_arr.shape[0]} rows but thoughts have D={D}"
    )

    # # Project into the Q-subspace, then fit LDA in r-dim space.
    X_tr = train.reshape(N_tr * T, D) @ Q_arr
    X_te = test .reshape(N_te * T, D) @ Q_arr
    y_tr = np.tile(np.arange(T), N_tr)
    y_te = np.tile(np.arange(T), N_te)

    # n_components for the classifier is capped at min(n_classes - 1, n_features).
    r = X_tr.shape[1]
    n_comp = min(T - 1, r)
    clf = LinearDiscriminantAnalysis(n_components=n_comp)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))