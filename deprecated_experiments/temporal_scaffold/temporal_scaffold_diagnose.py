"""
temporal_scaffold_diagnose.py — [ARCHIVED] temporal scaffold diagnostics for
continuous thought vectors.

This script is the LDA / scaffolding / PCA / cross-model side of the original
diagnose_thoughts.py, separated from variance_decomposition.py for clarity.

What this does (per (task, model)):
  - LDA Fisher separation of timestep classes
  - Held-out timestep classification accuracy (instance-level split)
  - Label-shuffle null control
  - Matched-Gaussian Monte Carlo null test (optional, --null_B)
  - PCA grid figures (raw + shared scaffolding-frame PCA) per variant
  - Controls summary figure (variance bar, Fisher+acc, principal angles)

Cross-model mode (--cross_model_only):
  - Loads existing per-(task, model) reports and produces rollup figures.

Output files (per-(task, model)):
    - comparative_pca_clusters.png
    - comparative_pca_trajectories.png
    - controls_summary.png
    - report.json
    - diagnose.txt

Output files (cross-model, THOUGHTS/<task>/cross_model/):
    - cross_model_variance_fisher.png
    - cross_model_principal_angles.png

Usage:
    python -m experiments.probe_thoughts.temporal_scaffold_diagnose --task prosqa --model coconut
    python -m experiments.probe_thoughts.temporal_scaffold_diagnose --task prosqa --cross_model_only
"""

import re
import sys
import json
import torch
import argparse
import matplotlib
import numpy as np
matplotlib.use("Agg")
from pathlib import Path
from src.config import THOUGHTS
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Reuse loading and variance from the clean module
from experiments.geometry.variance_decomposition import (
    Logger,
    load_thoughts,
    try_load,
    variance_decomposition,
    pretty_row_label,
    _discover_random_seeds,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# LDA: fit, Fisher separation, held-out accuracy
# ═══════════════════════════════════════════════════════════════════

def _reshape_xy(thoughts):
    """# X in R^{(N*T) x D}, y in {0,...,T-1}^{N*T}. Moves data to CPU for Sklearn."""
    if isinstance(thoughts, torch.Tensor):
        thoughts = thoughts.detach().cpu().numpy()
    N, T, D = thoughts.shape
    X = thoughts.reshape(N * T, D).astype(np.float64)
    y = np.tile(np.arange(T), N)
    return X, y, N, T, D


def fit_lda(thoughts):
    X, y, N, T, D = _reshape_xy(thoughts)
    lda = LinearDiscriminantAnalysis(n_components=T - 1)
    lda.fit(X, y)
    W = lda.scalings_[:, :T - 1]
    Q, _ = np.linalg.qr(W)
    col_norms = np.linalg.norm(Q, axis=0)
    Q = Q[:, col_norms > 1e-8]
    return lda, Q.astype(np.float32)


def lda_fisher_separation(thoughts):
    """
    # In LDA-projected space Z = X @ W  (shape (N*T, T-1)):
    """
    X, y, N, T, D = _reshape_xy(thoughts)
    lda = LinearDiscriminantAnalysis(n_components=T - 1)
    Z = lda.fit_transform(X, y)
    mu = Z.mean(axis=0)

    Sb, Sw = 0.0, 0.0
    for t in range(T):
        mask = (y == t)
        Zt = Z[mask]
        mu_t = Zt.mean(axis=0)
        Sb += mask.sum() * np.sum((mu_t - mu) ** 2)
        Sw += np.sum((Zt - mu_t) ** 2)
    return float(Sb / max(Sw, 1e-12))


def heldout_timestep_accuracy(thoughts, test_frac=0.3, seed=0):
    if isinstance(thoughts, torch.Tensor):
        thoughts = thoughts.detach().cpu().numpy()
    N, T, D = thoughts.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_te = int(test_frac * N)
    idx_te, idx_tr = idx[:n_te], idx[n_te:]

    X_tr = thoughts[idx_tr].reshape(-1, D)
    y_tr = np.tile(np.arange(T), len(idx_tr))
    X_te = thoughts[idx_te].reshape(-1, D)
    y_te = np.tile(np.arange(T), len(idx_te))

    clf = LinearDiscriminantAnalysis(n_components=T - 1)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def label_shuffle_accuracy(thoughts, test_frac=0.3, seed=0):
    if isinstance(thoughts, torch.Tensor):
        thoughts = thoughts.detach().cpu().numpy()
    N, T, D = thoughts.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_te = int(test_frac * N)
    idx_te, idx_tr = idx[:n_te], idx[n_te:]

    X_tr = thoughts[idx_tr].reshape(-1, D)
    y_tr_true = np.tile(np.arange(T), len(idx_tr))
    X_te = thoughts[idx_te].reshape(-1, D)
    y_te_true = np.tile(np.arange(T), len(idx_te))

    y_tr_shuf = y_tr_true.copy()
    rng.shuffle(y_tr_shuf)

    clf = LinearDiscriminantAnalysis(n_components=T - 1)
    clf.fit(X_tr, y_tr_shuf)
    return float(clf.score(X_te, y_te_true))


# ═══════════════════════════════════════════════════════════════════
# Matched-Gaussian resampler + Monte Carlo null
# ═══════════════════════════════════════════════════════════════════

def _estimate_matched_gaussian_params(thoughts):
    """
    # mu      = (1 / NT)  sum_{i, t} h_{i,t}        in R^D
    # Sigma   = (1 / (NT - 1))  sum_{i, t} (h_{i,t} - mu)(h_{i,t} - mu)^T   in R^{DxD}
    # Sigma   = L L^T   (Cholesky; eigendecomp fallback if Sigma is singular)
    """
    if isinstance(thoughts, np.ndarray):
        thoughts = torch.from_numpy(thoughts).to(DEVICE)
    N, T, D = thoughts.shape

    X = thoughts.reshape(N * T, D).to(torch.float64)
    mu = X.mean(dim=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / max(N * T - 1, 1)

    L = None
    jitter = 0.0
    for _ in range(5):
        try:
            L = torch.linalg.cholesky(Sigma + jitter * torch.eye(D, device=DEVICE, dtype=torch.float64))
            break
        except (torch.linalg.LinAlgError, RuntimeError):
            jitter = max(jitter * 10, 1e-6)

    if L is None:
        w, V = torch.linalg.eigh(Sigma)
        w = torch.clamp(w, min=0.0)
        L = V @ torch.diag(torch.sqrt(w))

    return mu, L, (N, T, D)


def _draw_matched_gaussian(mu, L, shape, seed):
    """
    # h_tilde_{i,t} = mu + L z,   z ~ N(0, I_D),  i.i.d. for each (i, t)
    """
    N, T, D = shape
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    Z = torch.randn((N * T, D), device=DEVICE, dtype=torch.float64, generator=gen)
    X_rand = mu + Z @ L.T
    return X_rand.reshape(N, T, D).to(torch.float32)


def matched_gaussian_sample(thoughts, seed=0):
    mu, L, shape = _estimate_matched_gaussian_params(thoughts)
    return _draw_matched_gaussian(mu, L, shape, seed)


def matched_gaussian_null_distribution(thoughts_original, B, seed=0,
                                       test_frac=0.3, log_every=100):
    """
    # for b = 1..B:
    #     h_tilde_b ~ N(mu_hat, L_hat L_hat^T)
    #     sep_b     = Fisher(LDA(h_tilde_b))
    #     acc_b     = HeldoutAcc(LDA(h_tilde_b), test_frac)
    """
    mu, L, shape = _estimate_matched_gaussian_params(thoughts_original)

    sep_null = np.empty(B, dtype=np.float64)
    acc_null = np.empty(B, dtype=np.float64)

    for b in range(B):
        h_tilde = _draw_matched_gaussian(mu, L, shape, seed=seed + b + 1)
        sep_null[b] = lda_fisher_separation(h_tilde)
        acc_null[b] = heldout_timestep_accuracy(h_tilde, test_frac=test_frac,
                                                seed=seed)
        if log_every and (b + 1) % log_every == 0:
            print(f"    [null]   b={b+1:>5}/{B}  "
                  f"sep_b={sep_null[b]:.4f}  acc_b={acc_null[b]:.4f}")

    return sep_null, acc_null


def null_test_summary(observed, null_array):
    """
    # p_value     = (1 + #{b : null_b >= observed}) / (B + 1)
    # log10_ratio = log10(observed / median(null))
    # z_score     = (observed - mean(null)) / std(null)
    """
    null = np.asarray(null_array, dtype=np.float64)
    B = null.size
    obs = float(observed)

    n_ge = int(np.sum(null >= obs))
    p_value = (1 + n_ge) / (B + 1)

    median_null = float(np.median(null))
    mean_null = float(np.mean(null))
    std_null = float(np.std(null, ddof=1)) if B > 1 else 0.0

    if median_null > 0 and obs > 0:
        log10_ratio = float(np.log10(obs / median_null))
    else:
        log10_ratio = float("nan")

    if std_null > 0:
        z_score = (obs - mean_null) / std_null
    else:
        z_score = float("nan")

    return {
        "observed": obs,
        "B": B,
        "median_null": median_null,
        "mean_null": mean_null,
        "std_null": std_null,
        "min_null": float(np.min(null)),
        "max_null": float(np.max(null)),
        "n_ge": n_ge,
        "p_value": float(p_value),
        "log10_ratio": log10_ratio,
        "z_score": float(z_score),
    }


# ═══════════════════════════════════════════════════════════════════
# Variant loading (includes matched_gaussian)
# ═══════════════════════════════════════════════════════════════════

def build_variants(task, model_name, seed=0):
    base_dir = THOUGHTS / task
    pe_dir = base_dir / model_name / "pe_ablation"

    variants = []

    orig_path = base_dir / f"thoughts_{model_name}.pt"
    orig = try_load(orig_path)
    if orig is None:
        raise FileNotFoundError(
            f"Original thoughts missing at {orig_path}. "
            f"Run extract_thoughts.py first."
        )
    variants.append(("original", orig))

    abl_zero = try_load(pe_dir / "thoughts_ablated_zero.pt")
    if abl_zero is not None:
        variants.append(("ablated_zero", abl_zero))
    else:
        print(f"[WARN] ablated_zero missing; skipping row.")

    abl_const = try_load(pe_dir / "thoughts_ablated_constant.pt")
    if abl_const is not None:
        variants.append(("ablated_constant", abl_const))
    else:
        print(f"[WARN] ablated_constant missing; skipping row.")

    for s in _discover_random_seeds(pe_dir, "random_gaussian"):
        t = try_load(pe_dir / f"thoughts_ablated_random_gaussian_seed{s}.pt")
        if t is not None:
            variants.append((f"ablated_random_gaussian_seed{s}", t))

    for s in _discover_random_seeds(pe_dir, "random_shuffle"):
        t = try_load(pe_dir / f"thoughts_ablated_random_shuffle_seed{s}.pt")
        if t is not None:
            variants.append((f"ablated_random_shuffle_seed{s}", t))

    mg = matched_gaussian_sample(orig, seed=seed)
    variants.append(("matched_gaussian", mg))

    return variants


# ═══════════════════════════════════════════════════════════════════
# GPU PCA coordinates
# ═══════════════════════════════════════════════════════════════════

def compute_pca_coords(thoughts):
    if isinstance(thoughts, np.ndarray):
        thoughts = torch.from_numpy(thoughts).to(DEVICE)
    N, T, D = thoughts.shape
    X = thoughts.reshape(N * T, D)

    X_c = X - X.mean(dim=0)
    _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    V = Vh.T[:, :2]
    Z = X_c @ V

    var_total = (S**2).sum()
    var_ratio_0 = ((S[0]**2) / var_total).item()
    var_ratio_1 = ((S[1]**2) / var_total).item()

    return Z.reshape(N, T, 2).cpu().numpy(), var_ratio_0, var_ratio_1


def fit_scaffold_frame(thoughts_original, target_instance_frac=0.008):
    if isinstance(thoughts_original, np.ndarray):
        thoughts_original = torch.from_numpy(thoughts_original).to(DEVICE)

    N, T, D = thoughts_original.shape

    X_flat_cpu = thoughts_original.reshape(N * T, D).cpu().numpy().astype(np.float64)
    y_flat = np.tile(np.arange(T), N)

    lda = LinearDiscriminantAnalysis(n_components=T - 1)
    lda.fit(X_flat_cpu, y_flat)
    Q_cpu, _ = np.linalg.qr(lda.scalings_)

    Q = torch.from_numpy(Q_cpu).to(device=DEVICE, dtype=thoughts_original.dtype)
    P = Q @ Q.T

    X_flat = thoughts_original.reshape(N * T, D)
    X_low_orig = (X_flat @ P).reshape(N, T, D)

    mu_orig = X_low_orig.mean(dim=(0, 1), keepdim=True)
    mu_t_orig = X_low_orig.mean(dim=0, keepdim=True)
    mu_i_orig = X_low_orig.mean(dim=1, keepdim=True)

    drift_orig = mu_t_orig - mu_orig
    identity_orig = mu_i_orig - mu_orig

    var_t_pop = (drift_orig.expand(N, T, D) ** 2).sum().item()
    var_i_pop = (identity_orig.expand(N, T, D) ** 2).sum().item()
    target_var_i = (target_instance_frac * var_t_pop) / (1 - target_instance_frac)
    scale = np.sqrt(target_var_i / max(var_i_pop, 1e-12))

    scaf_orig = mu_orig + drift_orig + identity_orig * scale
    scaf_flat = scaf_orig.reshape(N * T, D)

    scaf_mean = scaf_flat.mean(dim=0)
    scaf_c = scaf_flat - scaf_mean
    _, S, Vh = torch.linalg.svd(scaf_c, full_matrices=False)
    V = Vh.T[:, :2]

    var_total = (S**2).sum()
    vr0 = ((S[0]**2) / var_total).item()
    vr1 = ((S[1]**2) / var_total).item()

    return {
        "P": P,
        "pca_mean": scaf_mean,
        "pca_V": V,
        "instance_scale": float(scale),
        "instance_frac": float(target_instance_frac),
        "var_ratio": (vr0, vr1),
    }


def apply_scaffold_frame(thoughts, frame):
    if isinstance(thoughts, np.ndarray):
        thoughts = torch.from_numpy(thoughts).to(DEVICE)

    N, T, D = thoughts.shape
    P = frame["P"]
    scale = frame["instance_scale"]

    X_flat = thoughts.reshape(N * T, D)
    X_low = (X_flat @ P).reshape(N, T, D)

    mu = X_low.mean(dim=(0, 1), keepdim=True)
    mu_t = X_low.mean(dim=0, keepdim=True)
    mu_i = X_low.mean(dim=1, keepdim=True)

    drift = mu_t - mu
    identity = mu_i - mu
    scaf = mu + drift + identity * scale

    scaf_c = scaf.reshape(N * T, D) - frame["pca_mean"]
    coords = scaf_c @ frame["pca_V"]

    return coords.reshape(N, T, 2).cpu().numpy(), frame["var_ratio"][0], frame["var_ratio"][1]


# ═══════════════════════════════════════════════════════════════════
# Figure rendering
# ═══════════════════════════════════════════════════════════════════

ROW_LABELS = {
    "original":          "Original",
    "ablated_zero":      "PE ablated (zero)",
    "ablated_constant":  "PE ablated (constant)",
    "matched_gaussian":  "Matched Gaussian",
}


def render_cluster_figure(panels, task, model_name, out_path):
    n_rows = len(panels)
    fig, axes = plt.subplots(2, n_rows, figsize=(4 * n_rows, 12), squeeze=False)
    T_global = panels[0]["T"]

    orig_scaf = panels[0]["scaf_coords"].reshape(-1, 2)
    ox_min, ox_max = orig_scaf[:, 0].min(), orig_scaf[:, 0].max()
    oy_min, oy_max = orig_scaf[:, 1].min(), orig_scaf[:, 1].max()
    odx, ody = (ox_max - ox_min) * 0.15, (oy_max - oy_min) * 0.15
    orig_limits = (ox_min - odx, ox_max + odx, oy_min - ody, oy_max + ody)

    for i, p in enumerate(panels):
        T = p["T"]
        N = p["raw_coords"].shape[0]
        colors = np.tile(np.arange(T), N)

        ax_raw = axes[0, i]
        raw_data = p["raw_coords"].reshape(-1, 2)
        ax_raw.scatter(raw_data[:, 0], raw_data[:, 1],
                       c=colors, cmap="viridis", s=3, alpha=0.4,
                       vmin=0, vmax=T_global - 1)
        ax_raw.set_xlabel(f"PC1 ({p['raw_vr0']*100:.1f}%)")
        ax_raw.set_ylabel(f"PC2 ({p['raw_vr1']*100:.1f}%)")

        ax_sca = axes[1, i]
        scaf_data = p["scaf_coords"].reshape(-1, 2)

        is_collapsed = np.var(scaf_data, axis=0).max() < 1e-7
        point_size = 25 if (is_collapsed or p["row_label"] == "matched_gaussian") else 4

        sc = ax_sca.scatter(scaf_data[:, 0], scaf_data[:, 1],
                            c=colors, cmap="viridis", s=point_size, alpha=0.7,
                            vmin=0, vmax=T_global - 1, edgecolors='none')

        if p["row_label"] == "matched_gaussian":
            ax_sca.set_xlim(orig_limits[0], orig_limits[1])
            ax_sca.set_ylim(orig_limits[2], orig_limits[3])
            ax_sca.set_title("Scaffolding space\n(ANCHORED TO ORIGINAL SCALE)", fontsize=9, color='red')
        else:
            x_min, x_max = scaf_data[:, 0].min(), scaf_data[:, 0].max()
            y_min, y_max = scaf_data[:, 1].min(), scaf_data[:, 1].max()
            dx = (x_max - x_min) * 0.15 if x_max != x_min else 0.05
            dy = (y_max - y_min) * 0.15 if y_max != y_min else 0.05
            ax_sca.set_xlim(x_min - dx, x_max + dx)
            ax_sca.set_ylim(y_min - dy, y_max + dy)
            if i == 0: ax_sca.set_title("Scaffolding space\n(shared frame / dynamic scale)", fontsize=10)

        ax_raw.annotate(pretty_row_label(p["row_label"]),
                        xy=(-0.3, 0.5), xycoords="axes fraction", rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"model={model_name} task={task}")
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Timestep t")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_trajectory_figure(panels, task, model_name, out_path, n_traj=100):
    n_rows = len(panels)
    fig, axes = plt.subplots(2, n_rows, figsize=(4 * n_rows, 12), squeeze=False)
    cmap = plt.cm.tab20

    orig_scaf = panels[0]["scaf_coords"]
    ox_min, ox_max = orig_scaf[..., 0].min(), orig_scaf[..., 0].max()
    oy_min, oy_max = orig_scaf[..., 1].min(), orig_scaf[..., 1].max()
    odx, ody = (ox_max - ox_min) * 0.15, (oy_max - oy_min) * 0.15
    orig_limits = (ox_min - odx, ox_max + odx, oy_min - ody, oy_max + ody)

    for i, p in enumerate(panels):
        raw = p["raw_coords"]
        scaf = p["scaf_coords"]
        N, T, _ = raw.shape

        ax_raw = axes[0, i]
        for j in range(min(N, n_traj)):
            ax_raw.plot(raw[j, :, 0], raw[j, :, 1], "-o", color=cmap(j % 20 / 20.0), markersize=1.5, alpha=0.2)

        rx_min, rx_max = raw[..., 0].min(), raw[..., 0].max()
        ry_min, ry_max = raw[..., 1].min(), raw[..., 1].max()
        rdx, rdy = (rx_max - rx_min) * 0.1, (ry_max - ry_min) * 0.1
        ax_raw.set_xlim(rx_min - (rdx or 0.1), rx_max + (rdx or 0.1))
        ax_raw.set_ylim(ry_min - (rdy or 0.1), ry_max + (rdy or 0.1))

        ax_sca = axes[1, i]
        var_across_instances = np.var(scaf, axis=0).max()
        is_collapsed = var_across_instances < 1e-8

        if not is_collapsed:
            for j in range(min(N, n_traj)):
                ax_sca.plot(scaf[j, :, 0], scaf[j, :, 1], "-", color="gray", alpha=0.15, linewidth=0.5)

        mean_traj = scaf.mean(axis=0)
        ax_sca.plot(mean_traj[:, 0], mean_traj[:, 1], "-o", color="red", markersize=4, label="Mean", zorder=10)

        if p["row_label"] == "matched_gaussian":
            ax_sca.set_xlim(orig_limits[0], orig_limits[1])
            ax_sca.set_ylim(orig_limits[2], orig_limits[3])
            ax_sca.set_title("Scaffold: ANCHORED TO ORIGINAL", fontsize=9, color='red')
        else:
            sx_min, sx_max = scaf[..., 0].min(), scaf[..., 0].max()
            sy_min, sy_max = scaf[..., 1].min(), scaf[..., 1].max()
            sdx, sdy = (sx_max - sx_min) * 0.15, (sy_max - sy_min) * 0.15
            ax_sca.set_xlim(sx_min - (sdx or 0.05), sx_max + (sdx or 0.05))
            ax_sca.set_ylim(sy_min - (sdy or 0.05), sy_max + (sdy or 0.05))

        ax_raw.annotate(pretty_row_label(p["row_label"]),
                        xy=(-0.3, 0.5), xycoords="axes fraction", rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"model={model_name} task={task}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Controls summary (1x3 figure)
# ═══════════════════════════════════════════════════════════════════

_CONDITION_COLOR = {
    "original":         "#1f77b4",
    "ablated_zero":     "#ff7f0e",
    "ablated_constant": "#2ca02c",
    "matched_gaussian": "#d62728",
}
_RANDOM_GAUSSIAN_COLOR = "#9467bd"
_RANDOM_SHUFFLE_COLOR  = "#8c564b"
_COMPONENT_COLOR = {
    "timestep": "#4c72b0",
    "instance": "#dd8452",
    "residual": "#bcbcbc",
}


def _condition_color(condition):
    if condition in _CONDITION_COLOR:
        return _CONDITION_COLOR[condition]
    if condition.startswith("ablated_random_gaussian"):
        return _RANDOM_GAUSSIAN_COLOR
    if condition.startswith("ablated_random_shuffle"):
        return _RANDOM_SHUFFLE_COLOR
    return "#7f7f7f"


def _short_condition_label(condition):
    short_map = {
        "original":         "Original",
        "ablated_zero":     "PE zero",
        "ablated_constant": "PE const",
        "matched_gaussian": "Matched\nGaussian",
    }
    if condition in short_map:
        return short_map[condition]
    m = re.match(r"^ablated_random_gaussian_seed(\d+)$", condition)
    if m:
        return f"PE rand-g\n(s={m.group(1)})"
    m = re.match(r"^ablated_random_shuffle_seed(\d+)$", condition)
    if m:
        return f"PE rand-s\n(s={m.group(1)})"
    return condition


def discover_pe_ablation_reports(pe_dir):
    if not pe_dir.exists():
        return {}
    reports = {}
    name_map = [
        (re.compile(r"^report_zero\.json$"),                              "ablated_zero"),
        (re.compile(r"^report_constant\.json$"),                          "ablated_constant"),
        (re.compile(r"^report_random_gaussian_seed(\d+)\.json$"),         "ablated_random_gaussian_seed{S}"),
        (re.compile(r"^report_random_shuffle_seed(\d+)\.json$"),          "ablated_random_shuffle_seed{S}"),
    ]
    for p in pe_dir.iterdir():
        for pattern, tag_template in name_map:
            m = pattern.match(p.name)
            if m:
                tag = tag_template.replace("{S}", m.group(1)) if m.groups() else tag_template
                with open(p) as f:
                    reports[tag] = json.load(f)
                break
    return reports


def assemble_controls_data(rows_report, pe_reports, T):
    rows_by_label = {r["row_label"]: r for r in rows_report}

    random_g = sorted(
        [k for k in rows_by_label if k.startswith("ablated_random_gaussian")],
        key=lambda s: int(re.search(r"seed(\d+)", s).group(1)),
    )
    random_s = sorted(
        [k for k in rows_by_label if k.startswith("ablated_random_shuffle")],
        key=lambda s: int(re.search(r"seed(\d+)", s).group(1)),
    )

    condition_order = []
    for c in ["original", "ablated_zero", "ablated_constant"]:
        if c in rows_by_label:
            condition_order.append(c)
    condition_order.extend(random_g)
    condition_order.extend(random_s)
    if "matched_gaussian" in rows_by_label:
        condition_order.append("matched_gaussian")

    variance = {}
    fisher = {}
    heldout_acc = {}
    for c in condition_order:
        v = rows_by_label[c]["variance"]
        variance[c]    = (v["pct_timestep"], v["pct_instance"], v["pct_residual"])
        fisher[c]      = rows_by_label[c]["fisher_separation"]
        heldout_acc[c] = rows_by_label[c]["heldout_acc"]

    principal_angles = {}
    for tag, rpt in pe_reports.items():
        try:
            principal_angles[tag] = rpt["metrics"]["principal_angles_cos"]
        except KeyError:
            pass

    return {
        "condition_order":   condition_order,
        "variance":          variance,
        "fisher":            fisher,
        "heldout_acc":       heldout_acc,
        "principal_angles":  principal_angles,
        "T":                 T,
    }


def _plot_controls_variance(ax, data):
    conds = data["condition_order"]
    x = np.arange(len(conds))
    bottom = np.zeros(len(conds))

    for comp_idx, comp_name in enumerate(["timestep", "instance", "residual"]):
        vals = np.array([data["variance"][c][comp_idx] for c in conds])
        ax.bar(x, vals, bottom=bottom,
               color=_COMPONENT_COLOR[comp_name],
               edgecolor="white", linewidth=0.5,
               label=comp_name)
        if comp_name == "timestep":
            for xi, v in zip(x, vals):
                ax.text(xi, v / 2, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([_short_condition_label(c) for c in conds], fontsize=8)
    ax.set_ylabel("variance %")
    ax.set_ylim(0, 100)
    ax.set_title("Variance composition")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)


def _plot_controls_fisher_acc(ax, data):
    conds = data["condition_order"]
    x = np.arange(len(conds))
    width = 0.38

    fisher_vals = np.array([data["fisher"][c]      for c in conds])
    acc_vals    = np.array([data["heldout_acc"][c] for c in conds])
    colors      = [_condition_color(c) for c in conds]

    ax.bar(x - width / 2, fisher_vals, width,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Fisher separation (log)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_condition_label(c) for c in conds], fontsize=8)
    floor = max(fisher_vals.min() * 0.3, 1e-3)
    ax.set_ylim(floor, fisher_vals.max() * 3)
    for xi, v in zip(x - width / 2, fisher_vals):
        ax.text(xi, v * 1.3, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, acc_vals, width,
            color=colors, edgecolor="black", linewidth=0.5,
            hatch="///", alpha=0.7)
    ax2.set_ylabel("Held-out accuracy")
    ax2.set_ylim(0, 1)

    chance = 1.0 / data["T"]
    ax2.axhline(chance, color="gray", linestyle=":", linewidth=1)
    ax2.text(-0.4, chance + 0.04,
             f"chance = {chance:.3f}",
             fontsize=7, color="gray", ha="left")
    for xi, v in zip(x + width / 2, acc_vals):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Fisher separation  &  held-out accuracy\n"
                 "(solid = Fisher, hatched = accuracy)",
                 pad=18)


def _plot_controls_principal_angles(ax, data):
    pa = data["principal_angles"]
    if not pa:
        ax.set_axis_off()
        ax.set_title("Subspace preservation\n(no pe_ablation reports found)")
        return

    def _sort_key(tag):
        if tag == "ablated_zero":     return (0, 0)
        if tag == "ablated_constant": return (1, 0)
        m = re.match(r"^ablated_random_gaussian_seed(\d+)$", tag)
        if m: return (2, int(m.group(1)))
        m = re.match(r"^ablated_random_shuffle_seed(\d+)$", tag)
        if m: return (3, int(m.group(1)))
        return (9, 0)

    sorted_tags = sorted(pa.keys(), key=_sort_key)
    max_len = max(len(pa[t]) for t in sorted_tags)
    idx = np.arange(1, max_len + 1)

    for tag in sorted_tags:
        cosines = pa[tag]
        x = np.arange(1, len(cosines) + 1)
        ax.plot(x, cosines, "o-",
                color=_condition_color(tag),
                markersize=6, linewidth=1.6,
                label=_short_condition_label(tag).replace("\n", " "))

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle=":",  linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LDA component index  i")
    ax.set_ylabel(r"$\cos(\theta_i)$")
    ax.set_xticks(idx)
    ax.set_title("Subspace preservation\n(ablated vs original)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95, edgecolor="gray")


def render_controls_summary(controls_data, task, model_name, out_path):
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 4.5),
        gridspec_kw={"width_ratios": [1.1, 1.4, 1.1]},
    )
    _plot_controls_variance(axes[0], controls_data)
    _plot_controls_fisher_acc(axes[1], controls_data)
    _plot_controls_principal_angles(axes[2], controls_data)

    fig.suptitle(f"Control diagnostics — task={task}, model={model_name}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Cross-model comparison
# ═══════════════════════════════════════════════════════════════════

_ALL_MODELS = ["coconut", "coconut_u", "pause", "codi"]


def _models_for_task(task):
    if task == "prosqa":
        return [m for m in _ALL_MODELS if m != "codi"]
    return list(_ALL_MODELS)


def load_existing_diagnose_report(thoughts_root, task, model):
    p = Path(thoughts_root) / task / model / "diagnose" / "report.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def collect_cross_model_data(thoughts_root, task):
    out = {}
    for model in _models_for_task(task):
        diag = load_existing_diagnose_report(thoughts_root, task, model)
        if diag is None:
            print(f"[SKIP cross_model] {task}/{model}: no diagnose/report.json")
            continue
        pe_dir = Path(thoughts_root) / task / model / "pe_ablation"
        pe_reports = discover_pe_ablation_reports(pe_dir)
        controls = assemble_controls_data(
            rows_report=diag["rows"],
            pe_reports=pe_reports,
            T=int(diag["T"]),
        )
        out[model] = controls
    return out


def render_cross_model_variance_fisher(per_model, task, out_path):
    models = list(per_model.keys())
    M = len(models)
    if M == 0:
        print(f"[WARN cross_model_variance_fisher] no models for task={task}")
        return

    fig, axes = plt.subplots(
        nrows=2, ncols=M, figsize=(5.5 * M, 9.5), squeeze=False,
    )
    for j, model in enumerate(models):
        data = per_model[model]
        _plot_controls_variance(axes[0, j], data)
        _plot_controls_fisher_acc(axes[1, j], data)
        axes[0, j].set_title(f"{model}  —  variance composition",
                             fontsize=11, pad=12)
        axes[1, j].set_title(f"{model}  —  Fisher  &  held-out acc",
                             fontsize=11, pad=18)

    fig.suptitle(f"Cross-model diagnostics — task={task}",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_cross_model_principal_angles(per_model, task, out_path):
    models = list(per_model.keys())
    if not models:
        print(f"[WARN cross_model_principal_angles] no models for task={task}")
        return

    all_tags = set()
    for data in per_model.values():
        all_tags.update(data["principal_angles"].keys())

    def _sort_key(tag):
        if tag == "ablated_zero":     return (0, 0)
        if tag == "ablated_constant": return (1, 0)
        m = re.match(r"^ablated_random_gaussian_seed(\d+)$", tag)
        if m: return (2, int(m.group(1)))
        m = re.match(r"^ablated_random_shuffle_seed(\d+)$", tag)
        if m: return (3, int(m.group(1)))
        return (9, 0)

    tags = sorted(all_tags, key=_sort_key)
    if not tags:
        print(f"[WARN cross_model_principal_angles] no principal-angle data")
        return

    M = len(models)
    K = len(tags)
    width = 0.8 / max(K, 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * M * K), 5))
    x = np.arange(M)

    for ki, tag in enumerate(tags):
        heights = []
        for model in models:
            cosines = per_model[model]["principal_angles"].get(tag, None)
            heights.append(float(np.mean(cosines)) if cosines else np.nan)
        offset = (ki - (K - 1) / 2.0) * width
        bars = ax.bar(x + offset, heights, width,
                      color=_condition_color(tag),
                      edgecolor="black", linewidth=0.5,
                      label=_short_condition_label(tag).replace("\n", " "))
        for xi, h in zip(x + offset, heights):
            if not np.isnan(h):
                ax.text(xi, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle=":",  linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel(r"mean $\cos(\theta_i)$  over LDA components")
    ax.set_title(f"Cross-model subspace preservation — task={task}\n"
                 f"(higher = ablation left LDA subspace untouched)")
    ax.legend(fontsize=8, loc="lower right", ncol=min(K, 3))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════

def print_report(report, task, model_name):
    line = "=" * 76
    print(f"\n{line}")
    print(f"TEMPORAL SCAFFOLD DIAGNOSE  —  task={task}  model={model_name}")
    print(line)

    T = report["T"]
    chance = 1.0 / T
    print(f"\n  T = {T}, chance accuracy = {chance:.3f}")

    print(f"\n  Per-variant metrics:\n")
    header = (f"    {'Variant':<26} {'timestep%':>10} {'instance%':>10} "
              f"{'residual%':>10} {'Fisher':>10} {'held-out':>10}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in report["rows"]:
        v = row["variance"]
        print(f"    {pretty_row_label(row['row_label']):<26} "
              f"{v['pct_timestep']:>9.2f}% "
              f"{v['pct_instance']:>9.2f}% "
              f"{v['pct_residual']:>9.2f}% "
              f"{row['fisher_separation']:>10.2f} "
              f"{row['heldout_acc']:>10.3f}")

    print(f"\n  Label-shuffle control (on original thoughts):")
    print(f"    held-out acc = {report['label_shuffle_acc']:.3f}  "
          f"(chance = {chance:.3f})")

    null_block = report.get("matched_gaussian_null", None)
    if null_block is not None:
        B = null_block["B"]
        ns_sep = null_block["null_summary_sep"]
        ns_acc = null_block["null_summary_acc"]
        print(f"\n  Matched-Gaussian Monte Carlo null test  (B={B}):")
        print(f"    Null Fisher sep:   median={ns_sep['median']:.4f}  "
              f"mean={ns_sep['mean']:.4f}  std={ns_sep['std']:.4f}  "
              f"range=[{ns_sep['min']:.4f}, {ns_sep['max']:.4f}]")
        print(f"    Null held-out acc: median={ns_acc['median']:.4f}  "
              f"mean={ns_acc['mean']:.4f}  std={ns_acc['std']:.4f}  "
              f"range=[{ns_acc['min']:.4f}, {ns_acc['max']:.4f}]")

        print(f"\n    Per-variant tests against the matched-Gaussian null:")
        hdr = (f"    {'Variant':<26} "
               f"{'sep_obs':>9} {'log10_r':>8} {'z':>8} {'p_sep':>9}  "
               f"{'acc_obs':>9} {'log10_r':>8} {'z':>8} {'p_acc':>9}")
        print()
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for stat in null_block["per_variant"]:
            s = stat["fisher_separation"]
            a = stat["heldout_acc"]
            print(f"    {pretty_row_label(stat['row_label']):<26} "
                  f"{s['observed']:>9.3f} "
                  f"{s['log10_ratio']:>8.2f} "
                  f"{s['z_score']:>8.2f} "
                  f"{s['p_value']:>9.4f}  "
                  f"{a['observed']:>9.3f} "
                  f"{a['log10_ratio']:>8.2f} "
                  f"{a['z_score']:>8.2f} "
                  f"{a['p_value']:>9.4f}")


# ═══════════════════════════════════════════════════════════════════
# Per-(task, model) pipeline
# ═══════════════════════════════════════════════════════════════════

def run_per_model(args):
    out_dir = Path(args.output_dir) if args.output_dir else (
        THOUGHTS / args.task / args.model / "diagnose"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(out_dir / "diagnose.txt")

    print(f"[INFO] task={args.task}  model={args.model}")
    print(f"[INFO] output_dir={out_dir}")

    variants = build_variants(args.task, args.model, seed=args.seed)
    print(f"[INFO] Variants available: {[v[0] for v in variants]}")

    T = variants[0][1].shape[1]

    orig_label, orig_thoughts = variants[0]
    assert orig_label == "original"
    scaffold_frame = fit_scaffold_frame(
        orig_thoughts, target_instance_frac=args.instance_frac,
    )
    print(f"[INFO] Scaffold frame fit on original.  "
          f"PCA var_ratio = "
          f"{scaffold_frame['var_ratio'][0]:.3f}, "
          f"{scaffold_frame['var_ratio'][1]:.3f}")

    rows_report = []
    panels = []

    for row_label, thoughts in variants:
        print(f"\n[INFO] Processing variant: {row_label}  "
              f"shape={tuple(thoughts.shape)}")

        var = variance_decomposition(thoughts)
        sep = lda_fisher_separation(thoughts)
        acc = heldout_timestep_accuracy(thoughts, seed=args.seed)

        raw_coords, raw_vr0, raw_vr1 = compute_pca_coords(thoughts)
        scaf_coords, scaf_vr0, scaf_vr1 = apply_scaffold_frame(
            thoughts, scaffold_frame,
        )

        panels.append({
            "row_label": row_label,
            "raw_coords": raw_coords,
            "raw_vr0": raw_vr0,
            "raw_vr1": raw_vr1,
            "scaf_coords": scaf_coords,
            "scaf_vr0": scaf_vr0,
            "scaf_vr1": scaf_vr1,
            "T": T,
        })

        rows_report.append({
            "row_label": row_label,
            "variance": var,
            "fisher_separation": sep,
            "heldout_acc": acc,
        })

    # Label-shuffle control
    print(f"\n[INFO] Running label-shuffle control on original thoughts...")
    orig = variants[0][1]
    ls_acc = label_shuffle_accuracy(orig, seed=args.seed)

    # Matched-Gaussian Monte Carlo null test
    null_block = None
    if args.null_B > 0:
        print(f"\n[INFO] Running matched-Gaussian null test  "
              f"(B={args.null_B}, test_frac={args.null_test_frac}) ...")
        sep_null, acc_null = matched_gaussian_null_distribution(
            orig, B=args.null_B, seed=args.seed,
            test_frac=args.null_test_frac, log_every=max(1, args.null_B // 10),
        )

        per_variant_stats = []
        for row in rows_report:
            sep_obs = float(row["fisher_separation"])
            acc_obs = float(row["heldout_acc"])
            per_variant_stats.append({
                "row_label": row["row_label"],
                "fisher_separation": null_test_summary(sep_obs, sep_null),
                "heldout_acc":       null_test_summary(acc_obs, acc_null),
            })

        null_block = {
            "B": int(args.null_B),
            "test_frac": float(args.null_test_frac),
            "seed": int(args.seed),
            "null_summary_sep": {
                "median": float(np.median(sep_null)),
                "mean":   float(np.mean(sep_null)),
                "std":    float(np.std(sep_null, ddof=1)) if args.null_B > 1 else 0.0,
                "min":    float(np.min(sep_null)),
                "max":    float(np.max(sep_null)),
            },
            "null_summary_acc": {
                "median": float(np.median(acc_null)),
                "mean":   float(np.mean(acc_null)),
                "std":    float(np.std(acc_null, ddof=1)) if args.null_B > 1 else 0.0,
                "min":    float(np.min(acc_null)),
                "max":    float(np.max(acc_null)),
            },
            "per_variant": per_variant_stats,
        }

        null_draws_path = out_dir / "null_draws.npz"
        np.savez(null_draws_path,
                 sep_null=sep_null, acc_null=acc_null,
                 B=np.array([args.null_B]), seed=np.array([args.seed]),
                 test_frac=np.array([args.null_test_frac]))
        print(f"[INFO] Null draws saved to {null_draws_path}")

    # Render figures
    cluster_path = out_dir / "comparative_pca_clusters.png"
    traj_path = out_dir / "comparative_pca_trajectories.png"
    render_cluster_figure(panels, args.task, args.model, cluster_path)
    render_trajectory_figure(panels, args.task, args.model, traj_path,
                             n_traj=args.n_traj)
    print(f"\n[INFO] Cluster figure saved to    {cluster_path}")
    print(f"[INFO] Trajectory figure saved to {traj_path}")

    pe_dir = THOUGHTS / args.task / args.model / "pe_ablation"
    pe_reports = discover_pe_ablation_reports(pe_dir)
    print(f"\n[INFO] PE-ablation reports found: {sorted(pe_reports.keys())}")

    controls_data = assemble_controls_data(rows_report, pe_reports, T)
    controls_path = out_dir / "controls_summary.png"
    render_controls_summary(controls_data, args.task, args.model, controls_path)
    print(f"[INFO] Controls summary saved to {controls_path}")

    report = {
        "task": args.task,
        "model": args.model,
        "T": T,
        "D": int(variants[0][1].shape[2]),
        "n_instances": int(variants[0][1].shape[0]),
        "instance_frac": args.instance_frac,
        "rows": rows_report,
        "label_shuffle_acc": ls_acc,
        "matched_gaussian_null": null_block,
    }
    print_report(report, args.task, args.model)

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to {report_path}")


# ═══════════════════════════════════════════════════════════════════
# Cross-model rollup pipeline
# ═══════════════════════════════════════════════════════════════════

def run_cross_model(args):
    out_dir = Path(args.output_dir) if args.output_dir else (
        THOUGHTS / args.task / "cross_model"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(out_dir / "cross_model.txt")

    print(f"[INFO] task={args.task}  mode=cross_model_only")
    print(f"[INFO] output_dir={out_dir}")

    per_model = collect_cross_model_data(THOUGHTS, args.task)
    print(f"[INFO] Models loaded: {list(per_model.keys())}")
    if not per_model:
        print(f"[ERROR] No per-(task, model) reports found under "
              f"{THOUGHTS / args.task}.  Run the per-model pipeline first.")
        return

    vf_path = out_dir / "cross_model_variance_fisher.png"
    pa_path = out_dir / "cross_model_principal_angles.png"
    render_cross_model_variance_fisher(per_model, args.task, vf_path)
    render_cross_model_principal_angles(per_model, args.task, pa_path)
    print(f"[INFO] Variance/Fisher figure saved to    {vf_path}")
    print(f"[INFO] Principal-angles figure saved to   {pa_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="[ARCHIVED] Temporal scaffold diagnostics: LDA, "
                    "Fisher separation, scaffolding PCA, cross-model comparison.",
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        default=None,
                        help="Required unless --cross_model_only is set.")
    parser.add_argument("--cross_model_only", action="store_true")
    parser.add_argument("--instance_frac", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_traj", type=int, default=100)
    parser.add_argument("--null_B", type=int, default=0)
    parser.add_argument("--null_test_frac", type=float, default=0.3)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.cross_model_only:
        if args.model is not None:
            print(f"[WARN] --model={args.model} ignored under --cross_model_only.")
        run_cross_model(args)
    else:
        if args.model is None:
            parser.error("--model is required unless --cross_model_only is set.")
        run_per_model(args)


if __name__ == "__main__":
    main()