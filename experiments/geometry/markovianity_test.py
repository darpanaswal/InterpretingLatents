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

Multi-GPU (--n_gpus)
---------------------
The `orders` sweep (default 1..5) is embarrassingly parallel: each order's
mean/identity/linear/MLP fit + bootstrap CIs is independent of every other
order given the (already-loaded) train/test thought tensors. With
--n_gpus > 1, orders are round-robin sharded across that many worker
processes (torch.multiprocessing, mirroring gradient_subspace_interventions.py),
each pinned to its own GPU. Each worker re-loads the (small) thought tensors
independently rather than passing them through IPC, and writes its own
bootstrap-CI shard file that gets concatenated into the shared .jsonl after
all workers finish. With --n_gpus 1 (default) the original single-process
path runs unchanged.
"""

import gc
import json
import queue
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from pathlib import Path
from safetensors import safe_open
from sklearn.metrics import r2_score
from src.config import THOUGHTS, BASE_DIR, set_seed
from src.bootstrap_stats import (
    bootstrap_r2, bootstrap_mean, save_record, BootstrapResult,
)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
}


def _load_tensor_chunked(path, key="thoughts", device="cpu", chunk_rows=8192):
    """Stream a safetensors tensor onto `device` in row-chunks instead of
    materializing the whole thing as one host-RAM buffer first.

    load_safetensors(..., device="cuda") has to stage the full tensor
    through host memory before the H2D copy (that's inherent to how a
    cudaMemcpy source works -- safetensors' mmap trick only avoids the
    copy for device="cpu"). For a ~20GB train tensor that staging buffer
    alone was enough to trip the SLURM cgroup's OOM killer even with
    plenty of *GPU* memory free and even at n_gpus=1. Reading via
    safe_open + get_slice keeps the file mmap'd and only pages in /
    transfers one chunk at a time, so peak host RAM is ~chunk_rows worth
    of rows rather than the full array.
    """
    with safe_open(str(path), framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        shape = tuple(sl.get_shape())
        dtype = _ST_DTYPES[sl.get_dtype()]
        out = torch.empty(shape, dtype=dtype, device=device)
        N = shape[0]
        for i in range(0, N, chunk_rows):
            j = min(i + chunk_rows, N)
            out[i:j] = sl[i:j].to(device)
    return out


def load_thoughts(task, model, family="gpt2", device="cpu"):
    # Layout matches extract_thoughts.py: THOUGHTS/<family>/<task>/...
    # safetensors only -- extract_thoughts.py no longer writes .pt; back
    # fill old extractions with helpers/convert_thoughts_to_safetensors.py.
    base = THOUGHTS / family / task
    train_path = base / f"thoughts_{model}_train.safetensors"
    test_path = base / f"thoughts_{model}.safetensors"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test thoughts not found at {test_path}. Run extract_thoughts.py "
            f"for --task {task} --model {model} --model_family {family}, "
            f"or if it was extracted before the safetensors migration, "
            f"backfill with helpers/convert_thoughts_to_safetensors.py."
        )
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train thoughts not found at {train_path}. Run extract_thoughts.py "
            f"--split train for --task {task} --model {model} "
            f"--model_family {family}, or backfill with "
            f"helpers/convert_thoughts_to_safetensors.py."
        )

    train = _load_tensor_chunked(train_path, device=device)
    test = _load_tensor_chunked(test_path, device=device)
    print(f"[INFO] train={tuple(train.shape)}  test={tuple(test.shape)}  "
          f"device={device}")
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
    # In-place: avoid allocating a second full-size (N, T, D) tensor, which
    # doubles peak GPU memory and OOMs on large N (e.g. ~20GB train tensors).
    N, T, D = thoughts.shape
    for t in range(T):
        if t not in bases or bases[t].shape[1] == 0:
            thoughts[:, t, :].zero_()
            continue
        B = torch.as_tensor(bases[t], dtype=thoughts.dtype, device=thoughts.device)
        coords = thoughts[:, t, :] @ B
        thoughts[:, t, :] = coords @ B.T
    return thoughts


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
    X = torch.empty(n_t * N, order * D, dtype=thoughts.dtype, device=thoughts.device)
    Y = torch.empty(n_t * N, D, dtype=thoughts.dtype, device=thoughts.device)
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
    # Stays on W_aug's device throughout (no forced .cpu() per chunk) so
    # the result is directly comparable to Y_true in evaluate() without a
    # device mismatch, whether that device is "cpu" or a GPU.
    device = W_aug.device
    n, p = X.shape
    if chunk_size is None:
        chunk_size = _row_chunk_size(p + 1, X.element_size())
    outs = []
    for i in range(0, n, chunk_size):
        Xc = X[i:i + chunk_size].to(device)
        ones = torch.ones(Xc.shape[0], 1, dtype=Xc.dtype, device=device)
        Xc_aug = torch.cat([ones, Xc], dim=1)
        outs.append(Xc_aug @ W_aug)
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

    # X_tr/Y_tr may already be GPU-resident (loaded there directly); the
    # per-batch .to(device) below is then a no-op. When they're CPU-resident
    # (device="cpu", or a caller passes CPU tensors), only per-batch slices
    # get moved, avoiding materializing the full [n_train, p] tensor on GPU.
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
    # Same reasoning as predict_linear: stay on `device` so the caller can
    # compare against a same-device Y_true without an implicit CPU hop.
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size].to(device)))
    return torch.cat(out, dim=0)


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def evaluate(Y_true, Y_pred):
    # sklearn's r2_score (and downstream JSON/bootstrap consumers) need
    # numpy/CPU arrays regardless of what device Y_true/Y_pred live on.
    yt = Y_true.detach().cpu().numpy(); yp = Y_pred.detach().cpu().numpy()
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
        "_per_pair_cosine": per_pair_cosine.detach().cpu().numpy(),
    }


def _strip_arrays(metrics_dict):
    """Remove internal numpy arrays before JSON serialization.
    Keeps scalar metadata (e.g. _seed_std, _n_seeds) — only strips
    keys whose values are numpy arrays."""
    return {k: v for k, v in metrics_dict.items()
            if not isinstance(v, np.ndarray)}


def _bootstrap_compute(metrics_dict, metric_prefix, metric_suffix="", n_boot=1000):
    """
    Compute bootstrap CIs for r2_uniform and cosine from an evaluate() result.
    Pure compute, no I/O -- this is the part that's expensive (n_boot
    resampling passes) and safe to run in a GPU worker process, unlike
    writing to a shared file from multiple concurrent processes. Returns
    list of BootstrapResult.

    metric_suffix: appended to each metric name, e.g. "_seed0" or "_pooled".
    """
    cis = []

    # R² CI via row-resampling (Pattern B)
    if "_Y_true" in metrics_dict and "_Y_pred" in metrics_dict:
        cis.append(_bootstrap_r2_fast(
            metrics_dict["_Y_true"], metrics_dict["_Y_pred"],
            metric=f"{metric_prefix}_r2_uniform{metric_suffix}",
            n_boot=n_boot,
        ))

    # Cosine CI via per-pair mean (Pattern A)
    if "_per_pair_cosine" in metrics_dict:
        cis.append(bootstrap_mean(
            metrics_dict["_per_pair_cosine"],
            metric=f"{metric_prefix}_cosine{metric_suffix}",
            n_boot=n_boot,
        ))

    return cis


def _bootstrap_eval(metrics_dict, metric_prefix, cis_jsonl, context,
                    metric_suffix="", n_boot=1000):
    """_bootstrap_compute() + persist each record to cis_jsonl. Used by the
    single-process path and by the parent process after multi-GPU merge --
    never by a worker process directly writing to a shared file."""
    cis = _bootstrap_compute(metrics_dict, metric_prefix,
                             metric_suffix=metric_suffix, n_boot=n_boot)
    for r in cis:
        save_record(cis_jsonl, r, context=context)
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
    Y_pred = torch.empty(n_t * N, D, dtype=thoughts_eval.dtype, device=thoughts_eval.device)
    Y_true = torch.empty(n_t * N, D, dtype=thoughts_eval.dtype, device=thoughts_eval.device)
    for i, t in enumerate(target_ts):
        sl = slice(i * N, (i + 1) * N)
        Y_pred[sl] = thoughts_eval[:, t - 1, :]
        Y_true[sl] = thoughts_eval[:, t, :]
    return Y_pred, Y_true


# ═══════════════════════════════════════════════════════════════════
# Run a single (task, model)
# ═══════════════════════════════════════════════════════════════════

def _prepare_thoughts(task, model, family, project_to_subspace, bases_path,
                       subspace_source, drop_h0, device="cpu"):
    """Load (and optionally subspace-project) train/test thought tensors
    directly onto `device`.

    Loading + subspace projection happen on `device` (GPU-direct via the
    .safetensors sidecar when available) so build_pairs / fit_linear_ridge
    / fit_mlp never have to shuttle the multi-GB per-order tensors between
    host and device -- that CPU-side data movement, not GPU compute, was
    the actual bottleneck before this. Cheap enough (small load + a matmul)
    that each multi-GPU worker just calls this independently rather than
    receiving train/test via IPC.
    """
    train, test = load_thoughts(task, model, family=family, device=device)
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
    return train, test, skip_ts, Kp1, max_order


# ═══════════════════════════════════════════════════════════════════
# Streaming train-side pooling
# ═══════════════════════════════════════════════════════════════════
#
# build_pairs()/build_pairs_per_step() materialize the full pooled
# (n_t * N, order * D) tensor before fitting anything. Fine for the test
# split (a few thousand rows) but not for train: GSM has 385,620 training
# instances (21.6x prosqa's 17,886), so a shared-fit pool at order=1 is
# already tens of GB and grows with order. These functions compute the
# IDENTICAL quantities (ridge solve, MLP fit, R2/cosine) by streaming
# (timestep, instance-chunk) blocks directly out of `thoughts` (N, T, D),
# so the full pool is never materialized at once. Row encoding matches
# build_pairs() exactly (global row g = ti*N + i <-> target_ts[ti],
# instance i), so results are numerically identical to the pooled path --
# only test-side code (small; also needed in full for CI bootstrap
# resampling) still uses the original build_pairs()/build_pairs_per_step().

def _stream_pooled_chunks(thoughts, order, target_ts, chunk_size=None):
    """Sequentially yields (Xc, Yc) chunks covering the whole train pool,
    one target timestep at a time, chunked over the instance axis."""
    N, _, D = thoughts.shape
    if chunk_size is None:
        chunk_size = _row_chunk_size(order * D + 1, thoughts.element_size())
    for t in target_ts:
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            Xc = torch.empty(j - i, order * D, dtype=thoughts.dtype, device=thoughts.device)
            for k in range(1, order + 1):
                Xc[:, (k - 1) * D: k * D] = thoughts[i:j, t - k, :]
            Yc = thoughts[i:j, t, :]
            yield Xc, Yc


def _gather_pooled_rows(thoughts, order, target_ts, N, idx):
    """Direct-gathers arbitrary (e.g. shuffled) global pooled row indices
    `idx` straight from `thoughts`, without materializing the full pool.
    Used for MLP minibatch sampling, where batches mix rows from different
    timesteps after a global shuffle."""
    D = thoughts.shape[2]
    ti = torch.div(idx, N, rounding_mode="floor")
    inst = idx % N
    Xc = torch.empty(len(idx), order * D, dtype=thoughts.dtype, device=thoughts.device)
    Yc = torch.empty(len(idx), D, dtype=thoughts.dtype, device=thoughts.device)
    for pos, t in enumerate(target_ts):
        mask = ti == pos
        if not mask.any():
            continue
        rows = inst[mask]
        for k in range(1, order + 1):
            Xc[mask, (k - 1) * D: k * D] = thoughts[rows, t - k, :]
        Yc[mask] = thoughts[rows, t, :]
    return Xc, Yc


def _fit_ridge_from_chunks(chunks, p, ridge, device, dtype):
    """Shared accumulation core for fit_linear_ridge (test-side, chunks an
    already-materialized X/Y) and fit_linear_ridge_stream (train-side,
    chunks straight from `thoughts`) -- same math either way."""
    A = torch.zeros(p + 1, p + 1, dtype=dtype, device=device)
    B = None
    n_total = 0
    for Xc, Yc in chunks:
        Xc = Xc.to(device); Yc = Yc.to(device)
        if B is None:
            B = torch.zeros(p + 1, Yc.shape[1], dtype=dtype, device=device)
        ones = torch.ones(Xc.shape[0], 1, dtype=Xc.dtype, device=device)
        Xc_aug = torch.cat([ones, Xc], dim=1)
        A += Xc_aug.T @ Xc_aug
        B += Xc_aug.T @ Yc
        n_total += Xc.shape[0]

    penalty = torch.full((p + 1,), ridge * n_total, dtype=A.dtype, device=device)
    penalty[0] = 0.0
    A.diagonal().add_(penalty)
    return torch.linalg.solve(A, B)


def fit_linear_ridge_stream(thoughts, order, target_ts, ridge=1.0, device="cpu",
                            chunk_size=None):
    """Train-side ridge fit: identical result to
    fit_linear_ridge(*build_pairs(thoughts, order, ...)[:2]), computed
    without ever materializing the pooled X/Y."""
    D = thoughts.shape[2]
    return _fit_ridge_from_chunks(
        _stream_pooled_chunks(thoughts, order, target_ts, chunk_size),
        order * D, ridge, device, thoughts.dtype,
    )


def evaluate_stream(chunks, D, device):
    """Streaming equivalent of evaluate(Y_true, Y_pred): consumes
    (Y_true_chunk, Y_pred_chunk) pairs and accumulates the same R2
    (uniform + variance-weighted) and mean-cosine metrics evaluate()
    computes from full arrays, without ever materializing them. No
    _Y_true/_Y_pred arrays in the result -- only needed for CI bootstrap
    resampling, which stays test-side (small) and uses evaluate() as-is.
    """
    sse = torch.zeros(D, dtype=torch.float64, device=device)
    sum_y = torch.zeros(D, dtype=torch.float64, device=device)
    sum_y2 = torch.zeros(D, dtype=torch.float64, device=device)
    cos_sum = 0.0
    n = 0
    for yt, yp in chunks:
        yt = yt.to(device); yp = yp.to(device)
        diff = (yt - yp).double()
        sse += (diff * diff).sum(dim=0)
        yt64 = yt.double()
        sum_y += yt64.sum(dim=0)
        sum_y2 += (yt64 * yt64).sum(dim=0)
        yt_n = yt / yt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        yp_n = yp / yp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_sum += (yt_n * yp_n).sum(dim=-1).sum().item()
        n += yt.shape[0]

    mean_y = sum_y / n
    sst = sum_y2 - n * mean_y * mean_y  # sum((y-mean)^2) = sum(y^2) - n*mean^2
    nonzero = sst > 0
    r2_per = torch.zeros_like(sst)
    r2_per[nonzero] = 1.0 - sse[nonzero] / sst[nonzero]

    total_sst = sst.sum()
    total_sse = sse.sum()
    r2_var_weighted = float((1.0 - total_sse / total_sst).item()) if total_sst > 0 else 0.0

    return {
        "r2_uniform": float(r2_per.mean().item()),
        "r2_var_weighted": r2_var_weighted,
        "cosine": (cos_sum / n) if n > 0 else 0.0,
    }


def _stream_eval_train(thoughts, order, target_ts, device, predict_fn, chunk_size=None):
    """predict_fn(Xc) -> Yc_pred for one chunk. Streams (Y_true, Y_pred)
    chunks into evaluate_stream()."""
    D = thoughts.shape[2]
    def gen():
        for Xc, Yc in _stream_pooled_chunks(thoughts, order, target_ts, chunk_size):
            Xc = Xc.to(device); Yc = Yc.to(device)
            yield Yc, predict_fn(Xc)
    return evaluate_stream(gen(), D, device)


def mean_baseline_stream(thoughts, target_ts, device, chunk_size=None):
    """Returns (Y_mean shape (1, D), eval metrics dict) computed via
    running sums -- identical to evaluate(Y_tr, Y_mean.expand_as(Y_tr))
    where Y_mean = Y_tr.mean(dim=0, keepdim=True) on the pooled tensor."""
    N, _, D = thoughts.shape
    if chunk_size is None:
        chunk_size = _row_chunk_size(D, thoughts.element_size())
    total_sum = torch.zeros(D, dtype=torch.float64, device=device)
    n_total = 0
    for t in target_ts:
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            total_sum += thoughts[i:j, t, :].to(device).double().sum(dim=0)
            n_total += (j - i)
    Y_mean = (total_sum / n_total).to(thoughts.dtype).unsqueeze(0)

    def gen():
        for t in target_ts:
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                yc = thoughts[i:j, t, :].to(device)
                yield yc, Y_mean.expand(j - i, -1)
    return Y_mean, evaluate_stream(gen(), D, device)


def identity_baseline_stream(thoughts, target_ts, device, chunk_size=None):
    """Train-side identity baseline (prediction = previous timestep),
    streamed. Identical result to evaluate(*identity_prediction(thoughts,
    order, ...)) but never materializes the pooled arrays."""
    N, _, D = thoughts.shape
    if chunk_size is None:
        chunk_size = _row_chunk_size(D, thoughts.element_size())
    def gen():
        for t in target_ts:
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                yield (thoughts[i:j, t, :].to(device),
                       thoughts[i:j, t - 1, :].to(device))
    return evaluate_stream(gen(), D, device)


def fit_mlp_stream(thoughts, order, target_ts, N, hidden=256, epochs=200, lr=1e-3,
                   weight_decay=1e-4, batch_size=512, val_frac=0.1, patience=15,
                   device="cpu", seed=0):
    """Train-side MLP fit: same training procedure as fit_mlp() (random
    train/val split, minibatch AdamW, early stopping on val loss) but
    minibatches are gathered directly from `thoughts` via
    _gather_pooled_rows() instead of indexing a pre-materialized pool."""
    set_seed(seed)
    D = thoughts.shape[2]
    n_total = len(target_ts) * N
    n_val = max(1, int(val_frac * n_total))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    eval_chunk = batch_size * 8

    model = MLP(order * D, D, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        perm_ep = tr_idx[torch.randperm(tr_idx.shape[0], generator=g)]
        for i in range(0, perm_ep.shape[0], batch_size):
            idx = perm_ep[i:i + batch_size]
            xb, yb = _gather_pooled_rows(thoughts, order, target_ts, N, idx)
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            loss_sum, n_seen = 0.0, 0
            for i in range(0, val_idx.shape[0], eval_chunk):
                idx = val_idx[i:i + eval_chunk]
                xb, yb = _gather_pooled_rows(thoughts, order, target_ts, N, idx)
                xb = xb.to(device); yb = yb.to(device)
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


_POOL_CAP_BYTES = 4 * 1024 ** 3  # 4 GiB
# ProsQA-scale (17,886 train instances) stays under this at every order
# 1-5 (~1GB worst case), so it uses the fast pooled path (a single big
# matmul-friendly tensor, same as before streaming existed) end to end.
# GSM-scale (385,620 instances, 21.6x bigger) blows past it even at
# order=1 (~19GB) and falls back to the streaming path, which is
# memory-safe but has real per-chunk/per-batch Python overhead -- the
# threshold exists specifically so that overhead is only paid where it's
# actually needed for correctness, not on every dataset regardless of size.

def _pool_fits(N_train, order, D, n_t, itemsize):
    """Would materializing this order's pooled train X+Y fit in _POOL_CAP_BYTES?"""
    pool_bytes = n_t * N_train * (order * D + D) * itemsize
    return pool_bytes <= _POOL_CAP_BYTES


def _order_setup(order, train, test, skip_ts, drop_h0):
    """
    Cheap per-order setup shared by the "core" unit (mean/identity/linear/
    per-step) and every MLP-seed unit: computes target_ts, decides the
    pooled-vs-streaming train strategy, and builds the small test pool
    (+ the train pool too, if it's decided small enough to fit).

    Returns None if this order has no valid target timesteps. Cheap
    enough that every worker that touches this order just recomputes it
    independently rather than sharing it via IPC.
    """
    N_train, Kp1, D = train.shape
    start_pos = 1 if drop_h0 else 0
    target_ts = [t for t in range(start_pos + order, Kp1) if t not in skip_ts]
    if not target_ts:
        return None
    n_train_pairs = len(target_ts) * N_train

    X_te, Y_te, _ = build_pairs(test, order, drop_h0=drop_h0, skip_ts=skip_ts)
    if X_te is None:
        return None

    use_pool = _pool_fits(N_train, order, D, len(target_ts), train.element_size())
    X_tr = Y_tr = None
    if use_pool:
        X_tr, Y_tr, _ = build_pairs(train, order, drop_h0=drop_h0, skip_ts=skip_ts)

    return {
        "target_ts": target_ts, "N_train": N_train, "D": D,
        "n_train_pairs": n_train_pairs,
        "X_te": X_te, "Y_te": Y_te,
        "use_pool": use_pool, "X_tr": X_tr, "Y_tr": Y_tr,
    }


def _order_core(order, setup, train, test, drop_h0, skip_ts, ridge, device,
                n_boot, task, model, family, project_to_subspace, subspace_source):
    """Mean / identity / linear-shared / per-step-linear for one order --
    everything except the MLP fit, which is repeated per seed and is the
    expensive part. Independent of mlp_seeds, so this is its own
    shardable unit alongside each ("mlp", order, seed) unit.

    Also computes (but does not persist) the identity + linear_shared
    bootstrap CIs right here, since m_id_te/m_lin_te are already at hand
    and n_boot=1000 resampling is real CPU work (~2s each on realistic
    data) -- doing it in whichever process/worker computed core, instead
    of always serially in the parent after every worker's units are
    merged, is what actually uses the GPU-worker parallelism for this
    part of the pipeline too."""
    target_ts = setup["target_ts"]
    X_te, Y_te = setup["X_te"], setup["Y_te"]
    use_pool = setup["use_pool"]

    # Train-side results never need _Y_true/_Y_pred/_per_pair_cosine
    # downstream (only their scalar r2/cosine keys feed the entry +
    # printing; CI bootstrap is test-side only) -- but evaluate() always
    # computes them. When use_pool=True those are TRAIN-sized arrays (up
    # to ~1.5GB apiece here), and every unit's core/seed result sits in a
    # worker's `results` list for its whole lifetime before being shipped
    # back (finalization is centralized in the parent now), so leaving
    # them attached is exactly what silently grew a worker's memory with
    # every order/seed it touched. Strip immediately; the streaming
    # variants never produce array keys in the first place.

    # ─── Mean baseline ───────────────────────────────────────────
    if use_pool:
        X_tr, Y_tr = setup["X_tr"], setup["Y_tr"]
        Y_mean = Y_tr.mean(dim=0, keepdim=True)
        m_mean_tr = _strip_arrays(evaluate(Y_tr, Y_mean.expand_as(Y_tr)))
    else:
        Y_mean, m_mean_tr = mean_baseline_stream(train, target_ts, device)
    m_mean_te = evaluate(Y_te, Y_mean.expand_as(Y_te))

    # ─── Identity baseline ───────────────────────────────────────
    if use_pool:
        Yp_id_tr, Yt_id_tr = identity_prediction(train, order, drop_h0=drop_h0, skip_ts=skip_ts)
        m_id_tr = _strip_arrays(evaluate(Yt_id_tr, Yp_id_tr))
    else:
        m_id_tr = identity_baseline_stream(train, target_ts, device)
    Yp_id_te, Yt_id_te = identity_prediction(test, order, drop_h0=drop_h0, skip_ts=skip_ts)
    m_id_te = evaluate(Yt_id_te, Yp_id_te)

    # ─── Linear shared ───────────────────────────────────────────
    if use_pool:
        X_tr, Y_tr = setup["X_tr"], setup["Y_tr"]
        W = fit_linear_ridge(X_tr, Y_tr, ridge=ridge, device=device)
        m_lin_tr = _strip_arrays(evaluate(Y_tr, predict_linear(W, X_tr)))
    else:
        W = fit_linear_ridge_stream(train, order, target_ts, ridge=ridge, device=device)
        m_lin_tr = _stream_eval_train(train, order, target_ts, device,
                                      predict_fn=lambda Xc: predict_linear(W, Xc))
    m_lin_te = evaluate(Y_te, predict_linear(W, X_te))

    # ─── Per-step linear ───────────────────────────────────────────
    per_te = build_pairs_per_step(test, order, drop_h0=drop_h0, skip_ts=skip_ts)
    per_step_tr = []
    per_step_te = {}
    for t in target_ts:
        if use_pool:
            X_tr_t = torch.cat([train[:, t - k, :] for k in range(1, order + 1)], dim=-1)
            Y_tr_t = train[:, t, :]
            W_t = fit_linear_ridge(X_tr_t, Y_tr_t, ridge=ridge, device=device)
            per_step_tr.append(_strip_arrays(evaluate(Y_tr_t, predict_linear(W_t, X_tr_t))))
        else:
            W_t = fit_linear_ridge_stream(train, order, [t], ridge=ridge, device=device)
            per_step_tr.append(
                _stream_eval_train(train, order, [t], device,
                                   predict_fn=lambda Xc, w=W_t: predict_linear(w, Xc)))
        X_te_t, Y_te_t = per_te[t]
        per_step_te[t] = evaluate(Y_te_t, predict_linear(W_t, X_te_t))

    ci_ctx = {"task": task, "model": model, "model_family": family,
              "order": order,
              "projected": bool(project_to_subspace),
              "subspace_source": (subspace_source
                                  if project_to_subspace else None)}
    identity_linear_cis = {
        "identity": _bootstrap_compute(m_id_te, f"o{order}_identity", n_boot=n_boot),
        "linear_shared": _bootstrap_compute(m_lin_te, f"o{order}_linear_shared", n_boot=n_boot),
    }

    return {
        "n_train_pairs": setup["n_train_pairs"],
        "n_test_pairs": int(X_te.shape[0]),
        "m_mean_tr": m_mean_tr, "m_mean_te": m_mean_te,
        "m_id_tr": m_id_tr, "m_id_te": m_id_te,
        "m_lin_tr": m_lin_tr, "m_lin_te": m_lin_te,
        "per_step_tr": per_step_tr, "per_step_te": per_step_te,
        "identity_linear_cis": identity_linear_cis,
    }


def _order_mlp_seed(order, setup, train, seed, mlp_hidden, device,
                    is_seed0=False, n_boot=1000):
    """One MLP seed's fit + train/test eval for one order -- the unit
    that gets sharded per-seed-per-order across GPUs, since it's the
    dominant cost (200-epoch training) and independent of every other
    seed given (train, setup).

    is_seed0: whether this is mlp_seeds[0] specifically -- if so, also
    computes (but doesn't persist) its seed0-only bootstrap CI here,
    since m_te is already at hand. The "pooled across all seeds" CI
    still has to wait for every seed to be merged, so that one stays in
    the parent regardless."""
    target_ts = setup["target_ts"]
    X_te, Y_te = setup["X_te"], setup["Y_te"]
    use_pool = setup["use_pool"]
    N_train = setup["N_train"]

    if use_pool:
        X_tr, Y_tr = setup["X_tr"], setup["Y_tr"]
        mlp_s = fit_mlp(X_tr, Y_tr, hidden=mlp_hidden, device=device, seed=seed)
        # Strip train-sized _Y_true/_Y_pred immediately -- see _order_core's
        # comment; never needed downstream and would otherwise sit in a
        # worker's results list for its whole remaining lifetime.
        m_tr = _strip_arrays(evaluate(Y_tr, predict_mlp(mlp_s, X_tr, device=device)))
    else:
        mlp_s = fit_mlp_stream(train, order, target_ts, N_train,
                               hidden=mlp_hidden, device=device, seed=seed)
        m_tr = _stream_eval_train(train, order, target_ts, device,
                                  predict_fn=lambda Xc, m=mlp_s: predict_mlp(m, Xc, device=device))
    m_te = evaluate(Y_te, predict_mlp(mlp_s, X_te, device=device))

    seed0_cis = None
    if is_seed0:
        seed0_cis = _bootstrap_compute(m_te, f"o{order}_mlp_shared",
                                       metric_suffix="_seed0", n_boot=n_boot)

    return m_tr, m_te, seed0_cis


def _finalize_order_entry(order, core, mlp_tr_per_seed, mlp_te_per_seed, mlp_seeds,
                          cis_jsonl, n_boot, task, model, family,
                          project_to_subspace, subspace_source,
                          identity_linear_cis, seed0_cis):
    """Aggregates one order's core result + every MLP seed's result into
    the final entry dict, printing the summary table and persisting
    bootstrap-CI records as a side effect. Runs once per order in the
    process that owns the merge (the single-process path, or the parent
    process after collecting every worker's units for multi-GPU).

    identity_linear_cis and seed0_cis are already-computed (not yet
    persisted) BootstrapResults from _order_core()/_order_mlp_seed() --
    the expensive n_boot=1000 resampling for those already happened
    wherever that unit ran (a GPU worker, in the multi-GPU path), so this
    function only needs to write + print them. Only the MLP "pooled
    across all seeds" CI is computed here, since it needs every seed's
    test predictions merged first."""
    m_mean_tr, m_mean_te = core["m_mean_tr"], core["m_mean_te"]
    m_id_tr, m_id_te = core["m_id_tr"], core["m_id_te"]
    m_lin_tr, m_lin_te = core["m_lin_tr"], core["m_lin_te"]
    per_step_tr, per_step_te = core["per_step_tr"], core["per_step_te"]

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

    # ─── Per-step linear averages ────────────────────────────────
    # per_step_tr/te themselves were already computed in _order_core().
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

    # Identity and linear_shared: already computed in _order_core() --
    # just persist + print here.
    for label in ("identity", "linear_shared"):
        for c in identity_linear_cis[label]:
            save_record(cis_jsonl, c, context=ci_ctx)
            print(f"    [CI] {c.metric}: {c.to_short_str()}")

    # MLP: seed-0 CI -- already computed in _order_mlp_seed() for
    # whichever unit was mlp_seeds[0]; just persist here.
    for c in seed0_cis:
        save_record(cis_jsonl, c, context={**ci_ctx, "variant": "seed0"})

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

    # ─── Assemble entry ──────────────────────────────────────────
    entry = {
        "n_train_pairs": core["n_train_pairs"],
        "n_test_pairs": core["n_test_pairs"],

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
    return entry


def _run_single_order(order, train, test, skip_ts, drop_h0, ridge, mlp_hidden,
                      mlp_seeds, device, cis_jsonl, n_boot, task, model, family,
                      project_to_subspace, subspace_source):
    """Single-process path: setup -> core -> each MLP seed -> finalize,
    all for one order. Returns None if this order has no valid target
    timesteps."""
    print(f"\n  --- order={order} "
          f"({'bigram' if order == 1 else f'{order+1}-gram'}) ---")

    setup = _order_setup(order, train, test, skip_ts, drop_h0)
    if setup is None:
        print(f"  [skip] order={order}: no valid target timesteps.")
        return None
    print(f"    train_pairs={setup['n_train_pairs']} "
          f"test_pairs={setup['X_te'].shape[0]} in_dim={order * setup['D']}"
          + ("" if setup["use_pool"] else "  [streamed: train pool too large to materialize]"))

    core = _order_core(order, setup, train, test, drop_h0, skip_ts, ridge, device,
                       n_boot, task, model, family, project_to_subspace, subspace_source)

    mlp_tr_per_seed = []
    mlp_te_per_seed = []
    seed0_cis = None
    for s in mlp_seeds:
        m_tr, m_te, s0_cis = _order_mlp_seed(
            order, setup, train, s, mlp_hidden, device,
            is_seed0=(s == mlp_seeds[0]), n_boot=n_boot)
        mlp_tr_per_seed.append(m_tr)
        mlp_te_per_seed.append(m_te)
        if s0_cis is not None:
            seed0_cis = s0_cis

    entry = _finalize_order_entry(
        order, core, mlp_tr_per_seed, mlp_te_per_seed, mlp_seeds,
        cis_jsonl, n_boot, task, model, family,
        project_to_subspace, subspace_source,
        identity_linear_cis=core["identity_linear_cis"], seed0_cis=seed0_cis)

    del setup, core
    gc.collect()
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return entry


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU sharding -- one order's "core" plus each ("order", seed) MLP
# fit are independent units, so both order-level and seed-level
# parallelism fall out of sharding the same flat unit list. With only 1
# valid order and 3 mlp_seeds, this still spreads across up to 4 GPUs
# (1 core unit + 3 mlp-seed units) instead of being stuck on 1.
# ═══════════════════════════════════════════════════════════════════

def _unit_worker(rank, unit_shard, task, model, family, ridge, mlp_hidden,
                 mlp_seeds, n_boot, drop_h0, project_to_subspace, bases_path,
                 subspace_source, return_queue):
    """One GPU's share of (order, "core"|seed) units. Re-loads (small)
    thought tensors independently rather than receiving them via IPC.
    Only returns metric dicts (small, at most test-set-sized arrays) --
    never the underlying train/test tensors -- so IPC stays cheap."""
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    train, test, skip_ts, _, _ = _prepare_thoughts(
        task, model, family, project_to_subspace, bases_path,
        subspace_source, drop_h0, device=device)

    # Process one order at a time (not the shard's arbitrary interleaved
    # order) so at most one order's setup -- including its pooled train
    # X/Y when small enough to fast-path -- is memory-resident at once.
    # A worker's shard can span several different orders (round-robin
    # sharding + orders not dividing evenly into n_gpus), and caching
    # every order's setup for the worker's whole lifetime without ever
    # freeing it is exactly what caused a real OOM here: each new order
    # added its own multi-hundred-MB-to-GB pooled tensor without the
    # previous orders' ever being released.
    by_order = {}
    for order, kind, seed in unit_shard:
        by_order.setdefault(order, []).append((kind, seed))

    results = []
    for order in sorted(by_order):
        setup = _order_setup(order, train, test, skip_ts, drop_h0)
        if setup is None:
            continue

        for kind, seed in by_order[order]:
            if kind == "core":
                core = _order_core(order, setup, train, test, drop_h0, skip_ts, ridge, device,
                                   n_boot, task, model, family, project_to_subspace, subspace_source)
                results.append((order, "core", None, core))
                print(f"[rank {rank}] order={order} core done on {device}", flush=True)
            else:
                m_tr, m_te, s0_cis = _order_mlp_seed(
                    order, setup, train, seed, mlp_hidden, device,
                    is_seed0=(seed == mlp_seeds[0]), n_boot=n_boot)
                results.append((order, "mlp", seed, (m_tr, m_te, s0_cis)))
                print(f"[rank {rank}] order={order} seed={seed} mlp done on {device}", flush=True)

        del setup
        gc.collect()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return_queue.put({"rank": rank, "results": results})


def run_orders_multigpu(valid_orders, n_gpus, task, model, family, ridge,
                        mlp_hidden, mlp_seeds, n_boot, drop_h0,
                        project_to_subspace, bases_path, subspace_source,
                        cis_jsonl):
    """
    Round-robin shards the flat (order, "core"|seed) unit list across
    n_gpus worker processes, one per GPU, collects every worker's partial
    results, then finalizes each order (aggregation + bootstrap CIs +
    printing) once in this (parent) process after merging its core +ll
    mlp-seed results -- no per-worker CI shard files needed since
    finalization is centralized here, unlike the old order-only sharding.

    Returns dict {order: entry}, matching what the single-process path
    would have produced.
    """
    units = []
    for order in valid_orders:
        units.append((order, "core", None))
        for s in mlp_seeds:
            units.append((order, "mlp", s))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    shards = [units[i::n_gpus] for i in range(n_gpus)]

    procs = []
    for rank, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(
            target=_unit_worker,
            args=(rank, shard, task, model, family, ridge, mlp_hidden,
                  mlp_seeds, n_boot, drop_h0, project_to_subspace,
                  bases_path, subspace_source, q),
        )
        p.start()
        procs.append(p)

    collected = []
    for _ in procs:
        while True:
            try:
                collected.append(q.get(timeout=5.0))
                break
            except queue.Empty:
                for p in procs:
                    if not p.is_alive() and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {p.pid} crashed with exit code "
                            f"{p.exitcode}. Check console for CUDA "
                            "out-of-memory or other runtime errors."
                        )
    for p in procs:
        p.join()

    # Group every worker's units back by order: one "core" result plus
    # one (m_tr, m_te) pair per seed.
    by_order = {order: {"core": None, "mlp": {}} for order in valid_orders}
    for res in collected:
        for order, kind, seed, payload in res["results"]:
            if kind == "core":
                by_order[order]["core"] = payload
            else:
                by_order[order]["mlp"][seed] = payload

    order_entries = {}
    for order in valid_orders:
        core = by_order[order]["core"]
        if core is None:
            continue  # order had no valid target timesteps
        mlp_by_seed = by_order[order]["mlp"]
        mlp_tr_per_seed = [mlp_by_seed[s][0] for s in mlp_seeds if s in mlp_by_seed]
        mlp_te_per_seed = [mlp_by_seed[s][1] for s in mlp_seeds if s in mlp_by_seed]
        if len(mlp_tr_per_seed) != len(mlp_seeds):
            raise RuntimeError(
                f"order={order}: expected {len(mlp_seeds)} mlp-seed results, "
                f"got {len(mlp_tr_per_seed)}. A worker unit may have been dropped."
            )
        # seed0's CI was computed by whichever worker drew that unit --
        # only the "pooled across all seeds" CI still has to wait until
        # here, since it needs every seed's test predictions merged first.
        seed0_cis = mlp_by_seed[mlp_seeds[0]][2]
        print(f"\n  --- order={order} "
              f"({'bigram' if order == 1 else f'{order+1}-gram'}) ---")
        entry = _finalize_order_entry(
            order, core, mlp_tr_per_seed, mlp_te_per_seed, mlp_seeds,
            cis_jsonl, n_boot, task, model, family,
            project_to_subspace, subspace_source,
            identity_linear_cis=core["identity_linear_cis"], seed0_cis=seed0_cis)
        order_entries[order] = entry

    return order_entries


# ═══════════════════════════════════════════════════════════════════
# Run a single (task, model)
# ═══════════════════════════════════════════════════════════════════

def run(task, model, orders, ridge, mlp_hidden, device, drop_h0=True,
        project_to_subspace=False, bases_path=None,
        out_dir=None, mlp_seeds=None, n_boot=1000, family="gpt2",
        subspace_source="gold", n_gpus=1):

    if mlp_seeds is None:
        mlp_seeds = [0, 1, 2]

    proj_tag = (f"  [SUBSPACE-PROJECTED: {subspace_source}]"
                if project_to_subspace else "")
    print(f"\n{'='*64}\n  task={task}  model={model}  family={family}"
          + proj_tag
          + f"\n{'='*64}")

    train, test, skip_ts, Kp1, max_order = _prepare_thoughts(
        task, model, family, project_to_subspace, bases_path,
        subspace_source, drop_h0, device=device)

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

    valid_orders = []
    for order in orders:
        if order > max_order:
            print(f"  [skip] order={order} > max_order={max_order}")
        else:
            valid_orders.append(order)

    # Sharding unit is (order, "core"|seed) -- an order's core analysis
    # plus one unit per mlp_seed -- so parallelism isn't capped by
    # len(valid_orders): e.g. 1 order x 3 seeds still gives 4 units to
    # spread across up to 4 GPUs.
    n_units = len(valid_orders) * (1 + len(mlp_seeds))
    n_workers = min(n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    n_workers = min(n_workers, n_units)
    if n_workers > 1:
        print(f"[INFO] sharding {len(valid_orders)} order(s) x "
              f"(1 core + {len(mlp_seeds)} mlp seeds) = {n_units} unit(s) "
              f"across {n_workers} GPU worker(s)")
        # Each worker reloads train/test onto its own GPU independently
        # (see _unit_worker), so this process's own device-resident copy
        # is dead weight from here on -- and rank 0 targets the same
        # physical device this process loaded onto, so holding both
        # copies at once needlessly doubles memory pressure on it.
        del train, test
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        order_entries = run_orders_multigpu(
            valid_orders, n_workers, task, model,
            family, ridge, mlp_hidden, mlp_seeds, n_boot, drop_h0,
            project_to_subspace, bases_path, subspace_source, cis_jsonl)
        results["orders"] = order_entries
    else:
        for order in valid_orders:
            entry = _run_single_order(
                order, train, test, skip_ts, drop_h0, ridge, mlp_hidden,
                mlp_seeds, device, cis_jsonl, n_boot, task, model, family,
                project_to_subspace, subspace_source)
            if entry is not None:
                results["orders"][order] = entry

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
    parser.add_argument(
        "--n_gpus", type=int, default=1,
        help="If >1, shard the --orders sweep across GPUs 0..n_gpus-1 "
             "(one worker process per GPU, round-robin over orders). "
             "With 1 (default), runs on a single --device as before.",
    )
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
                    n_gpus=args.n_gpus,
                )
                out_path = out_dir / f"results_{model}_{task}{suffix}.json"
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[saved] {out_path}")
                all_results.append(res)


if __name__ == "__main__":
    main()