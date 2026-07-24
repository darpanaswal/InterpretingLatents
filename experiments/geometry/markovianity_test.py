"""
markovianity_test.py v3
Markovianity test for continuous thought vectors.

Question:
    Are thoughts h_1, ..., h_K Markovian, i.e. can h_t be predicted from a
    bounded window of preceding thoughts?

We sweep the order of dependence:

    n-gram (order n-1):  h_t = f(h_{t-1}, h_{t-2}, ..., h_{t-n+1})

Order=1 (bigram) is the strongest Markov claim:  h_t = f(h_{t-1}).
Larger orders weaken the claim.

Function classes:
    1. Linear:    h_t = W [h_{t-1}; ...; h_{t-n+1}] + b
    2. Non-linear: 2-layer MLP (small + early-stopped to avoid overfit).

Train on TRAIN-SPLIT thoughts; evaluate on TEST-SPLIT thoughts.
The test split is ONLY ever used for evaluation -- never for fitting,
model selection, early-stopping, or the mean baseline's reference mean.

Baselines that the model must beat to mean anything:
    - Mean baseline:      h_t_hat = mean of h_t over TRAIN
        => sets the denominator of R^2.
    - Identity baseline:  h_t_hat = h_{t-1}
        => if thoughts barely move step-to-step, identity already
           gives high R^2 and a "good" linear fit is meaningless.

Both TRAIN- and TEST-split thoughts must already exist on disk (see
extract_thoughts.py); this script never extracts them.

Subspace-projection mode (--project_to_subspace)
------------------------------------------------
Repeats the entire analysis but on h^c = B_t B_t^T h instead of h,
where B_t is the per-timestep gradient subspace produced by
gradient_subspace.py.  This answers a different question than the
default mode:

    Default mode:
        Does the full thought vector move between recurrence steps?
        (Identity baseline says PaT/C/CODI thoughts are static.)

    Subspace mode:
        Does the *causal-relevant* component of the thought vector
        move between recurrence steps?  If identity R^2 stays high
        in B_t too, the recurrence transports nothing the model uses.
        If identity R^2 drops in B_t (and a linear shift fits well),
        the recurrence carries causal content forward across steps.

Projection is per-timestep (B_t differs across t) and lives in the
same ambient R^D, so the existing fitting / baseline / evaluation
code runs unchanged.

Output files in subspace mode are suffixed with `_subspace` so the
two modes' results sit side-by-side.
"""

import gc
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score
from src.config import THOUGHTS, BASE_DIR, set_seed
from src.bootstrap_stats import (
    bootstrap_r2, bootstrap_mean, save_record, BootstrapResult,
)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_thoughts(task, model, family="gpt2"):
    # Layout matches extract_thoughts.py: THOUGHTS/<family>/<task>/...
    base = THOUGHTS / family / task
    train_path = base / f"thoughts_{model}_train.pt"
    test_path = base / f"thoughts_{model}.pt"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test thoughts not found at {test_path}. "
            f"Run extract_thoughts.py for --task {task} --model {model} "
            f"--model_family {family}."
        )
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train thoughts not found at {train_path}. "
            f"Run extract_thoughts.py --split train for --task {task} "
            f"--model {model} --model_family {family}."
        )

    train = torch.load(train_path, map_location="cpu",
                       weights_only=False)["thoughts"]
    test = torch.load(test_path, map_location="cpu",
                      weights_only=False)["thoughts"]
    print(f"[INFO] train={tuple(train.shape)}  test={tuple(test.shape)}")
    return train, test


# ═══════════════════════════════════════════════════════════════════
# Subspace projection
# ═══════════════════════════════════════════════════════════════════

# Subspace source -> on-disk tree.
#   "gold": gradient_geometry/            (gradient of gold-answer NLL)
#   "pred": gradient_geometry_predtoken/  (gradient of model's own argmax NLL)
_SUBSPACE_ROOT = {
    "gold": "gradient_geometry",
    "pred": "gradient_geometry_predtoken",
}


def _bases_path(task, model, family="gpt2", subspace_source="gold"):
    # Must match the layout written by gradient_subspace.py (gold) /
    # gradient_subspace_predtoken.py (pred).
    root = _SUBSPACE_ROOT[subspace_source]
    return (BASE_DIR / "outputs" / root
            / family / task / model / "bases.npz")


def load_bases(task, model, bases_path=None, family="gpt2",
               subspace_source="gold"):
    path = (Path(bases_path) if bases_path
            else _bases_path(task, model, family=family,
                             subspace_source=subspace_source))
    if not path.exists():
        gen = ("gradient_subspace.py" if subspace_source == "gold"
               else "gradient_subspace_predtoken.py")
        raise FileNotFoundError(
            f"bases.npz not found at {path}. "
            f"Run {gen} for --task {task} --model {model} "
            f"--model_family {family}."
        )
    blob = np.load(path)
    bases = {}
    for key in blob.files:
        if not key.startswith("B_t"):
            continue
        t = int(key[len("B_t"):])
        bases[t] = blob[key]
    print(f"[INFO] loaded bases from {path}: ranks "
          f"{[bases[t].shape[1] for t in sorted(bases)]}")
    return bases


def _proj_suffix(project_to_subspace, subspace_source):
    """Filename suffix for a run.

    off              -> ""
    gold projection  -> "_subspace"       (legacy name; back-compatible)
    pred projection  -> "_subspace_pred"
    """
    if not project_to_subspace:
        return ""
    return "_subspace" if subspace_source == "gold" else "_subspace_pred"


def project_thoughts_per_t(thoughts, bases):
    N, T, D = thoughts.shape
    out = torch.zeros_like(thoughts)
    for t in range(T):
        if t not in bases or bases[t].shape[1] == 0:
            continue
        B = torch.as_tensor(bases[t], dtype=thoughts.dtype)
        coords = thoughts[:, t, :] @ B
        out[:, t, :] = coords @ B.T
    return out


# ═══════════════════════════════════════════════════════════════════
# Build (X, Y) pairs
# ═══════════════════════════════════════════════════════════════════

def build_pairs(thoughts, order, drop_h0=True, skip_ts=None):
    if skip_ts is None: skip_ts = []
    N, Kp1, D = thoughts.shape
    start_pos = 1 if drop_h0 else 0
    first_target = start_pos + order
    if first_target >= Kp1:
        raise ValueError(
            f"order={order} too large for K+1={Kp1} drop_h0={drop_h0}."
        )
    target_ts = [t for t in range(first_target, Kp1) if t not in skip_ts]
    if not target_ts:
        return None, None, []

    # Fill preallocated buffers directly instead of building a list of
    # per-timestep chunks and torch.cat-ing them: cat needs the whole list
    # AND the freshly-copied output alive at once, transiently ~doubling
    # peak host memory for large training sets (e.g. many-thought models).
    n_t = len(target_ts)
    X = torch.empty(n_t * N, order * D, dtype=thoughts.dtype)
    Y = torch.empty(n_t * N, D, dtype=thoughts.dtype)
    for i, t in enumerate(target_ts):
        sl = slice(i * N, (i + 1) * N)
        for k in range(1, order + 1):
            X[sl, (k - 1) * D: k * D] = thoughts[:, t - k, :]
        Y[sl] = thoughts[:, t, :]
    return X, Y, target_ts


def build_pairs_per_step(thoughts, order, drop_h0=True, skip_ts=None):
    if skip_ts is None: skip_ts = []
    N, Kp1, _ = thoughts.shape
    start_pos = 1 if drop_h0 else 0
    out = {}
    for t in range(start_pos + order, Kp1):
        if t in skip_ts:
            continue
        ctx = [thoughts[:, t - k, :] for k in range(1, order + 1)]
        out[t] = (torch.cat(ctx, dim=-1), thoughts[:, t, :])
    return out


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════

def _row_chunk_size(p_aug, itemsize, target_bytes=512 * 1024 * 1024):
    """Rows per chunk so a [chunk, p_aug] buffer stays around target_bytes."""
    return max(1024, int(target_bytes // (p_aug * itemsize)))


def fit_linear_ridge(X_tr, Y_tr, ridge=1.0, device="cpu", chunk_size=None):
    n, p = X_tr.shape
    itemsize = X_tr.element_size()
    if chunk_size is None:
        chunk_size = _row_chunk_size(p + 1, itemsize)

    # Accumulate A = X_aug^T X_aug and B = X_aug^T Y in row chunks so we
    # never materialize the full [n, p+1] augmented matrix on `device`.
    A = torch.zeros(p + 1, p + 1, dtype=X_tr.dtype, device=device)
    B = torch.zeros(p + 1, Y_tr.shape[1], dtype=X_tr.dtype, device=device)
    for i in range(0, n, chunk_size):
        Xc = X_tr[i:i + chunk_size].to(device)
        Yc = Y_tr[i:i + chunk_size].to(device)
        ones = torch.ones(Xc.shape[0], 1, dtype=Xc.dtype, device=device)
        Xc_aug = torch.cat([ones, Xc], dim=1)
        A += Xc_aug.T @ Xc_aug
        B += Xc_aug.T @ Yc

    # Create a penalty vector that ONLY regularizes the features.
    # We multiply by n to keep regularization invariant to dataset size,
    # but we DO NOT penalize the bias (index 0).
    penalty = torch.full((p + 1,), ridge * n, dtype=A.dtype, device=device)
    penalty[0] = 0.0  # Leave the bias completely unpenalized

    A.diagonal().add_(penalty)

    return torch.linalg.solve(A, B)


def predict_linear(W_aug, X, chunk_size=None):
    device = W_aug.device
    n, p = X.shape
    if chunk_size is None:
        chunk_size = _row_chunk_size(p + 1, X.element_size())
    outs = []
    for i in range(0, n, chunk_size):
        Xc = X[i:i + chunk_size].to(device)
        ones = torch.ones(Xc.shape[0], 1, dtype=Xc.dtype, device=device)
        Xc_aug = torch.cat([ones, Xc], dim=1)
        outs.append((Xc_aug @ W_aug).cpu())
    return torch.cat(outs, dim=0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(X_tr, Y_tr, hidden=256, epochs=200, lr=1e-3, weight_decay=1e-4,
            batch_size=512, val_frac=0.1, patience=15, device="cpu", seed=0):
    set_seed(seed)
    n = X_tr.shape[0]
    n_val = max(1, int(val_frac * n))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Keep the splits on CPU; only per-batch slices are moved to `device`.
    # Avoids materializing the full [n_train, p] / [n_val, p] tensors on the
    # GPU at once, which for large p (higher orders) can itself OOM.
    X_in = X_tr[tr_idx]; Y_in = Y_tr[tr_idx]
    X_va = X_tr[val_idx]; Y_va = Y_tr[val_idx]
    eval_chunk = batch_size * 8

    model = MLP(X_tr.shape[1], Y_tr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        perm_ep = torch.randperm(X_in.shape[0], generator=g)
        for i in range(0, X_in.shape[0], batch_size):
            idx = perm_ep[i:i + batch_size]
            xb = X_in[idx].to(device); yb = Y_in[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            # Weighted mean over chunks reproduces the same value MSELoss
            # would give over the full X_va/Y_va in one shot (D is constant
            # across chunks, so a count-weighted average of per-chunk means
            # equals the mean over all elements).
            loss_sum, n_seen = 0.0, 0
            for i in range(0, X_va.shape[0], eval_chunk):
                xb = X_va[i:i + eval_chunk].to(device)
                yb = Y_va[i:i + eval_chunk].to(device)
                loss_sum += loss_fn(model(xb), yb).item() * xb.shape[0]
                n_seen += xb.shape[0]
            v = loss_sum / n_seen
        if v < best - 1e-6:
            best = v
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_mlp(model, X, device="cpu", batch_size=4096):
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def evaluate(Y_true, Y_pred):
    yt = Y_true.numpy(); yp = Y_pred.numpy()
    yt_t = Y_true / Y_true.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    yp_t = Y_pred / Y_pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # per-pair cosine similarities — kept for bootstrap_mean
    per_pair_cosine = (yt_t * yp_t).sum(dim=-1)  # [n_pairs]
    return {
        "r2_uniform": float(r2_score(yt, yp, multioutput="uniform_average")),
        "r2_var_weighted": float(r2_score(yt, yp,
                                          multioutput="variance_weighted")),
        "cosine": float(per_pair_cosine.mean()),
        "_Y_true": yt,
        "_Y_pred": yp,
        "_per_pair_cosine": per_pair_cosine.numpy(),
    }


def _strip_arrays(metrics_dict):
    """Remove internal numpy arrays before JSON serialization.
    Keeps scalar metadata (e.g. _seed_std, _n_seeds) — only strips
    keys whose values are numpy arrays."""
    return {k: v for k, v in metrics_dict.items()
            if not isinstance(v, np.ndarray)}


def _bootstrap_eval(metrics_dict, metric_prefix, cis_jsonl, context,
                    metric_suffix="", n_boot=1000):
    """
    Compute bootstrap CIs for r2_uniform and cosine from an evaluate() result.
    Appends records to cis_jsonl. Returns list of BootstrapResult for logging.

    metric_suffix: appended to each metric name, e.g. "_seed0" or "_pooled".
    """
    cis = []

    # R² CI via row-resampling (Pattern B)
    if "_Y_true" in metrics_dict and "_Y_pred" in metrics_dict:
        r2_ci = _bootstrap_r2_fast(
            metrics_dict["_Y_true"], metrics_dict["_Y_pred"],
            metric=f"{metric_prefix}_r2_uniform{metric_suffix}",
            n_boot=n_boot,
        )
        save_record(cis_jsonl, r2_ci, context=context)
        cis.append(r2_ci)

    # Cosine CI via per-pair mean (Pattern A)
    if "_per_pair_cosine" in metrics_dict:
        cos_ci = bootstrap_mean(
            metrics_dict["_per_pair_cosine"],
            metric=f"{metric_prefix}_cosine{metric_suffix}",
            n_boot=n_boot,
        )
        save_record(cis_jsonl, cos_ci, context=context)
        cis.append(cos_ci)

    return cis

def _r2_uniform_fast(Y, Yhat):
    # SSE per output: shape [D]
    diff = Y - Yhat
    sse = (diff * diff).sum(axis=0)
    # SST per output: centered Y squared, summed
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    sst = (Y_centered * Y_centered).sum(axis=0)
    # Per-dim R^2 with sklearn's SST==0 convention
    nonzero = sst > 0
    r2_per = np.zeros_like(sst)
    r2_per[nonzero] = 1.0 - sse[nonzero] / sst[nonzero]
    return float(r2_per.mean())


def _bootstrap_r2_fast(Y_true, Y_pred, n_boot=1000, ci=95.0,
                       seed=0, metric="r2"):
    # Pattern B with closed-form R^2 per draw.
    Y = np.asarray(Y_true)
    Yh = np.asarray(Y_pred)
    assert Y.shape == Yh.shape, "shape mismatch"
    n = Y.shape[0]
    rng = np.random.default_rng(seed)

    point = _r2_uniform_fast(Y, Yh)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _r2_uniform_fast(Y[idx], Yh[idx])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return BootstrapResult(
        metric=metric, point=float(point),
        ci_low=float(lo), ci_high=float(hi),
        ci_level=ci, n=n, n_boot=n_boot, seed=seed,
        method="function",
    )


def identity_prediction(thoughts_eval, order, drop_h0=True, skip_ts=None):
    if skip_ts is None: skip_ts = []
    N, Kp1, D = thoughts_eval.shape
    start_pos = 1 if drop_h0 else 0
    target_ts = [t for t in range(start_pos + order, Kp1) if t not in skip_ts]
    if not target_ts:
        return None, None

    n_t = len(target_ts)
    Y_pred = torch.empty(n_t * N, D, dtype=thoughts_eval.dtype)
    Y_true = torch.empty(n_t * N, D, dtype=thoughts_eval.dtype)
    for i, t in enumerate(target_ts):
        sl = slice(i * N, (i + 1) * N)
        Y_pred[sl] = thoughts_eval[:, t - 1, :]
        Y_true[sl] = thoughts_eval[:, t, :]
    return Y_pred, Y_true


# ═══════════════════════════════════════════════════════════════════
# Run a single (task, model)
# ═══════════════════════════════════════════════════════════════════

def run(task, model, orders, ridge, mlp_hidden, device, drop_h0=True,
        project_to_subspace=False, bases_path=None,
        out_dir=None, mlp_seeds=None, n_boot=1000, family="gpt2",
        subspace_source="gold"):

    if mlp_seeds is None:
        mlp_seeds = [0, 1, 2]

    proj_tag = (f"  [SUBSPACE-PROJECTED: {subspace_source}]"
                if project_to_subspace else "")
    print(f"\n{'='*64}\n  task={task}  model={model}  family={family}"
          + proj_tag
          + f"\n{'='*64}")

    train, test = load_thoughts(task, model, family=family)

    Kp1 = train.shape[1]
    skip_ts = []

    if project_to_subspace:
        bases = load_bases(task, model, bases_path=bases_path, family=family,
                           subspace_source=subspace_source)
        train = project_thoughts_per_t(train, bases)
        test = project_thoughts_per_t(test, bases)

        skip_ts = [t for t in range(Kp1)
                   if t not in bases or bases[t].shape[1] == 0]

        print(f"[INFO] projected to per-t gradient subspace "
              f"(ambient R^D preserved); train={tuple(train.shape)}  "
              f"test={tuple(test.shape)}")

        if skip_ts:
            print(f"[INFO] target timesteps with empty B_t (skipped): {skip_ts}")

    max_order = Kp1 - 1 - (1 if drop_h0 else 0)

    # Bootstrap CI output path (sibling to the results JSON).
    # Gold projection keeps the legacy "_subspace" suffix for back-compat;
    # predtoken projection uses "_subspace_pred" so the two sit side-by-side.
    suffix = _proj_suffix(project_to_subspace, subspace_source)
    ci_dir = out_dir if out_dir else BASE_DIR / "outputs" / "markovianity"
    cis_jsonl = str(Path(ci_dir) / f"cis_{model}_{task}{suffix}.jsonl")

    results = {
        "task": task,
        "model": model,
        "model_family": family,
        "drop_h0": drop_h0,
        "projected": bool(project_to_subspace),
        # Which subspace was projected onto (only meaningful when projected):
        #   "gold" = gradient of gold-answer NLL
        #   "pred" = gradient of model's own predicted-token NLL
        "subspace_source": (subspace_source if project_to_subspace else None),
        "K_plus_1": int(Kp1),
        "D": int(train.shape[2]),
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "orders": {}
    }

    for order in orders:
        if order > max_order:
            print(f"  [skip] order={order} > max_order={max_order}")
            continue

        print(f"\n  --- order={order} "
              f"({'bigram' if order == 1 else f'{order+1}-gram'}) ---")

        X_tr, Y_tr, target_ts = build_pairs(train, order, drop_h0=drop_h0, skip_ts=skip_ts)
        if X_tr is None:
            print(f"  [skip] order={order}: no valid target timesteps.")
            continue

        X_te, Y_te, _ = build_pairs(test, order, drop_h0=drop_h0, skip_ts=skip_ts)

        print(f"    train_pairs={X_tr.shape[0]} test_pairs={X_te.shape[0]} "
              f"in_dim={X_tr.shape[1]}")

        # ─── Mean baseline ───────────────────────────────────────────
        Y_mean = Y_tr.mean(dim=0, keepdim=True)
        m_mean_tr = evaluate(Y_tr, Y_mean.expand_as(Y_tr))
        m_mean_te = evaluate(Y_te, Y_mean.expand_as(Y_te))

        # ─── Identity baseline ───────────────────────────────────────
        Yp_id_tr, Yt_id_tr = identity_prediction(train, order, drop_h0=drop_h0, skip_ts=skip_ts)
        Yp_id_te, Yt_id_te = identity_prediction(test, order, drop_h0=drop_h0, skip_ts=skip_ts)

        m_id_tr = evaluate(Yt_id_tr, Yp_id_tr)
        m_id_te = evaluate(Yt_id_te, Yp_id_te)

        # ─── Linear shared ───────────────────────────────────────────
        W = fit_linear_ridge(X_tr, Y_tr, ridge=ridge, device=device)

        m_lin_tr = evaluate(Y_tr, predict_linear(W, X_tr))
        m_lin_te = evaluate(Y_te, predict_linear(W, X_te))

        # ─── MLP (repeated across seeds for training variance) ─────
        mlp_te_per_seed = []
        mlp_tr_per_seed = []
        for s in mlp_seeds:
            mlp_s = fit_mlp(X_tr, Y_tr, hidden=mlp_hidden, device=device, seed=s)
            mlp_tr_per_seed.append(evaluate(Y_tr, predict_mlp(mlp_s, X_tr, device=device)))
            mlp_te_per_seed.append(evaluate(Y_te, predict_mlp(mlp_s, X_te, device=device)))

        # Aggregate across seeds: report mean ± std of scalar metrics
        scalar_keys = ("r2_uniform", "r2_var_weighted", "cosine")
        m_mlp_tr = {k: float(np.mean([m[k] for m in mlp_tr_per_seed]))
                    for k in scalar_keys}
        m_mlp_te = {k: float(np.mean([m[k] for m in mlp_te_per_seed]))
                    for k in scalar_keys}
        m_mlp_te_std = {k: float(np.std([m[k] for m in mlp_te_per_seed]))
                        for k in scalar_keys}
        m_mlp_te["_seed_std"] = m_mlp_te_std
        m_mlp_te["_n_seeds"] = len(mlp_seeds)
        m_mlp_te["_seeds"] = list(mlp_seeds)
        # ─── Seed-0 arrays (for seed-0-only CI) ─────────────────────
        m_mlp_te_seed0 = {}
        for arr_key in ("_Y_true", "_Y_pred", "_per_pair_cosine"):
            if arr_key in mlp_te_per_seed[0]:
                m_mlp_te_seed0[arr_key] = mlp_te_per_seed[0][arr_key]

        # ─── Pooled arrays across seeds (for joint sampling⊗training CI)
        # Memory budget: K_seeds × n_test × D.  Typical: 3 × 5000 × 768
        # ≈ 11.5M floats ≈ 44 MB (float32).  Fine for any test set.
        #
        # Pool by concatenating (Y_true_repeated, Y_pred_k) along axis 0.
        # bootstrap_r2 resamples rows of the pooled pair, capturing both
        # which-test-instances and which-seed contribute to each draw.
        K_seeds = len(mlp_te_per_seed)
        if "_Y_pred" in mlp_te_per_seed[0]:
            Y_true_one = mlp_te_per_seed[0]["_Y_true"]          # [n_test, D]
            # tile Y_true K times to align with concat of per-seed Y_pred
            Y_true_pooled = np.tile(Y_true_one, (K_seeds, 1))   # [K*n_test, D]
            Y_pred_pooled = np.concatenate(
                [m["_Y_pred"] for m in mlp_te_per_seed], axis=0  # [K*n_test, D]
            )
            m_mlp_te_pooled_r2 = {
                "_Y_true": Y_true_pooled,
                "_Y_pred": Y_pred_pooled,
            }
        else:
            m_mlp_te_pooled_r2 = {}

        if "_per_pair_cosine" in mlp_te_per_seed[0]:
            cos_pooled = np.concatenate(
                [m["_per_pair_cosine"] for m in mlp_te_per_seed], axis=0
            )
            m_mlp_te_pooled_cos = {"_per_pair_cosine": cos_pooled}
        else:
            m_mlp_te_pooled_cos = {}

        # Also keep seed-0 arrays in m_mlp_te for downstream compatibility
        m_mlp_te.update(m_mlp_te_seed0)

        # ─── Per-step linear ─────────────────────────────────────────
        per_tr = build_pairs_per_step(train, order, drop_h0=drop_h0, skip_ts=skip_ts)
        per_te = build_pairs_per_step(test, order, drop_h0=drop_h0, skip_ts=skip_ts)

        per_step_tr = []
        per_step_te = {}

        for t in target_ts:
            X_tr_t, Y_tr_t = per_tr[t]
            X_te_t, Y_te_t = per_te[t]

            W_t = fit_linear_ridge(X_tr_t, Y_tr_t, ridge=ridge, device=device)

            per_step_tr.append(evaluate(Y_tr_t, predict_linear(W_t, X_tr_t)))
            per_step_te[t] = evaluate(Y_te_t, predict_linear(W_t, X_te_t))

        per_step_avg_tr = {k: float(np.mean([m[k] for m in per_step_tr]))
                           for k in ("r2_uniform", "r2_var_weighted", "cosine")}
        per_step_avg_te = {k: float(np.mean([m[k] for m in per_step_te.values()]))
                           for k in ("r2_uniform", "r2_var_weighted", "cosine")}

        # ─── Pretty print ────────────────────────────────────────────
        def fmt_both(tr, te):
            return (f"Train R2={tr['r2_uniform']:+.3f} | Test R2={te['r2_uniform']:+.3f}  "
                    f"(Train R2vw={tr['r2_var_weighted']:+.3f} | Test R2vw={te['r2_var_weighted']:+.3f})")

        print(f"    mean        : {fmt_both(m_mean_tr, m_mean_te)}")
        print(f"    identity    : {fmt_both(m_id_tr, m_id_te)}")
        print(f"    linear share: {fmt_both(m_lin_tr, m_lin_te)}")
        print(f"    linear/step : {fmt_both(per_step_avg_tr, per_step_avg_te)}")
        print(f"    mlp shared  : {fmt_both(m_mlp_tr, m_mlp_te)}"
              f"  [±std R2={m_mlp_te_std['r2_uniform']:.4f} over {len(mlp_seeds)} seeds]")

        # ─── Bootstrap CIs on test-split metrics ────────────────────
        ci_ctx = {"task": task, "model": model, "model_family": family,
                  "order": order,
                  "projected": bool(project_to_subspace),
                  "subspace_source": (subspace_source
                                      if project_to_subspace else None)}

        # Identity and linear_shared: deterministic, single CI as before
        for label, m_te in [("identity", m_id_te), ("linear_shared", m_lin_te)]:
            cis = _bootstrap_eval(m_te, f"o{order}_{label}", cis_jsonl, ci_ctx, n_boot=n_boot)
            for c in cis:
                print(f"    [CI] {c.metric}: {c.to_short_str()}")

        # MLP: seed-0 CI (sampling only)
        seed0_cis = _bootstrap_eval(
            m_mlp_te_seed0, f"o{order}_mlp_shared",
            cis_jsonl, {**ci_ctx, "variant": "seed0"},
            metric_suffix="_seed0", n_boot=n_boot
        )

        # MLP: pooled CI (sampling ⊗ training)
        pooled_input = {**m_mlp_te_pooled_r2, **m_mlp_te_pooled_cos}
        pooled_cis = _bootstrap_eval(
            pooled_input, f"o{order}_mlp_shared",
            cis_jsonl,
            {**ci_ctx, "variant": "pooled",
             "n_seeds": K_seeds, "seeds": list(mlp_seeds),
             "pooling": "concat"},
            metric_suffix="_pooled", n_boot=n_boot
        )

        # Print three-number summary per metric
        #   seed-0 CI  |  pooled CI  |  seed_mean ± seed_std
        seed0_by_metric = {c.metric: c for c in seed0_cis}
        pooled_by_metric = {c.metric: c for c in pooled_cis}
        for base_name, std_key in [("r2_uniform", "r2_uniform"),
                                   ("cosine", "cosine")]:
            s0_key = f"o{order}_mlp_shared_{base_name}_seed0"
            pl_key = f"o{order}_mlp_shared_{base_name}_pooled"
            s0 = seed0_by_metric.get(s0_key)
            pl = pooled_by_metric.get(pl_key)
            if s0 and pl:
                print(f"    [CI] mlp {base_name}:  "
                      f"seed0={s0.to_short_str()}  "
                      f"pooled={pl.to_short_str()}  "
                      f"seed_mean={m_mlp_te[std_key]:.3f}"
                      f"±{m_mlp_te_std[std_key]:.4f}")

        # ─── Save results ────────────────────────────────────────────
        results["orders"][order] = {
            "n_train_pairs": int(X_tr.shape[0]),
            "n_test_pairs": int(X_te.shape[0]),

            "mean_baseline": _strip_arrays(m_mean_te),
            "mean_baseline_train": _strip_arrays(m_mean_tr),

            "identity_baseline": _strip_arrays(m_id_te),
            "identity_baseline_train": _strip_arrays(m_id_tr),

            "linear_shared": _strip_arrays(m_lin_te),
            "linear_shared_train": _strip_arrays(m_lin_tr),

            "linear_per_step": {int(t): _strip_arrays(m) for t, m in per_step_te.items()},
            "linear_per_step_avg": per_step_avg_te,
            "linear_per_step_avg_train": per_step_avg_tr,

            "mlp_shared": _strip_arrays(m_mlp_te),
            "mlp_shared_train": _strip_arrays(m_mlp_tr),
        }

        # ─── Free this order's large pair tensors ───────────────────
        # `X_tr, Y_tr, ... = build_pairs(...)` at the top of the next
        # iteration evaluates build_pairs() (allocating the *new* order's
        # tensors) before rebinding the names -- so without this, the
        # previous order's and next order's multi-GB X_tr/Y_tr/identity
        # arrays briefly coexist, roughly doubling peak host RAM on
        # datasets with many training thoughts (e.g. `pause`).
        del X_tr, Y_tr, X_te, Y_te, Yp_id_tr, Yt_id_tr, Yp_id_te, Yt_id_te
        del per_tr, per_te
        gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test markovianity of continuous thought vectors."
    )
    parser.add_argument("--task", type=str,
                        choices=["prosqa", "gsm", "all"], required=True)
    parser.add_argument("--model", type=str,
                        choices=["coconut", "coconut_u", "pause", "codi",
                                 "all"],
                        required=True)
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Namespaces all thought/bases/output paths.",
    )
    parser.add_argument("--orders", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_seeds", type=int, nargs="+",
                        default=[0, 1, 2],
                        help="Seeds for MLP training (init + val split). "
                             "Reports mean ± std across seeds.")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ridge solves and MLP fitting.",
    )
    parser.add_argument("--include_h0", action="store_true",
                        help="Include h_0 (pre-recurrence) in inputs.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global reproducibility seed.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--project_to_subspace", type=str,
        choices=["off", "gold", "gold_only", "pred", "pred_only",
                 "pred_and_gold", "all"],
        default="off",
        help="Project thoughts onto the per-t gradient subspace B_t before "
             "fitting / baselines / evaluation. The subspace source can be "
             "the gold-answer gradient ('gold', from gradient_geometry/) or "
             "the model's own predicted-token gradient ('pred', from "
             "gradient_geometry_predtoken/). Each mode names the runs it "
             "produces, out of {full, gold, pred}:\n"
             "  off           -> [full]                 (no projection; default)\n"
             "  gold          -> [full, gold]            (full + gold subspace)\n"
             "  gold_only     -> [gold]                  (ONLY the gold subspace)\n"
             "  pred          -> [full, pred]            (full + predtoken subspace)\n"
             "  pred_only     -> [pred]                  (ONLY the predtoken subspace)\n"
             "  pred_and_gold -> [gold, pred]             (both subspaces, no full)\n"
             "  all           -> [full, gold, pred]       (everything)\n"
             "Projected runs get a `_subspace` (gold) or `_subspace_pred` "
             "(pred) filename suffix so all variants sit side-by-side.",
    )
    parser.add_argument(
        "--bases_path", type=str, default=None,
        help="Override path to bases.npz. Default: "
             "BASE_DIR/outputs/<gradient_geometry|gradient_geometry_predtoken>"
             "/<family>/<task>/<model>/bases.npz, chosen by the projection "
             "mode's subspace source. Applies to single-source modes only.",
    )
    parser.add_argument("--n_boot", type=int, default=1000,
                    help="Number of bootstrap iterations.")
    args = parser.parse_args()

    set_seed(args.seed)

    tasks = ["prosqa", "gsm"] if args.task == "all" else [args.task]
    models = (["pause", "coconut", "coconut_u", "codi"]
              if args.model == "all" else [args.model])

    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "markovianity" / args.model_family
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each plan is (project_to_subspace: bool, subspace_source: str).
    # subspace_source is ignored when project_to_subspace is False.
    PLANS = {
        "off":           [(False, "gold")],
        "gold":          [(False, "gold"), (True, "gold")],
        "gold_only":     [(True,  "gold")],
        "pred":          [(False, "gold"), (True, "pred")],
        "pred_only":     [(True,  "pred")],
        "pred_and_gold": [(True,  "gold"), (True, "pred")],
        "all":           [(False, "gold"), (True, "gold"), (True, "pred")],
    }
    run_plans = PLANS[args.project_to_subspace]

    # --bases_path is a single path; it can only safely apply when the mode
    # has exactly one projected source. For multi-source modes (gold/pred/
    # pred_and_gold/all) ignore it and let each source resolve its own
    # default tree.
    n_projected = sum(1 for p, _ in run_plans if p)
    if args.bases_path and n_projected != 1:
        print(f"[WARN] --bases_path ignored for mode "
              f"'{args.project_to_subspace}' (has {n_projected} projected "
              f"sources); each resolves its own default bases.npz.")
        effective_bases_path = None
    else:
        effective_bases_path = args.bases_path

    all_results = []
    for task in tasks:
        for model in models:
            for proj_mode, source in run_plans:
                suffix = _proj_suffix(proj_mode, source)
                res = run(
                    task=task, model=model,
                    orders=sorted(args.orders),
                    ridge=args.ridge, mlp_hidden=args.mlp_hidden,
                    device=args.device, drop_h0=(not args.include_h0),
                    project_to_subspace=proj_mode,
                    bases_path=(effective_bases_path if proj_mode else None),
                    out_dir=out_dir,
                    mlp_seeds=args.mlp_seeds,
                    n_boot=args.n_boot,
                    family=args.model_family,
                    subspace_source=source,
                )
                out_path = out_dir / f"results_{model}_{task}{suffix}.json"
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[saved] {out_path}")
                all_results.append(res)


if __name__ == "__main__":
    main()