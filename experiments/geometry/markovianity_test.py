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

If train-split thoughts do not exist on disk, we extract them now using
the same machinery as extract_thoughts.py, pointed at the TRAIN data
file. The script never silently splits the test file.

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

import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score
from src.config import (
    THOUGHTS, BASE_DIR, set_seed,
    PROSQA_TRAIN, GSM_TRAIN,
)
from src.bootstrap_stats import (
    bootstrap_r2, bootstrap_mean, save_record, BootstrapResult,
)


# ═══════════════════════════════════════════════════════════════════
# Train-split thought extraction (only run if file is missing)
# ═══════════════════════════════════════════════════════════════════

def _train_data_path(task):
    return PROSQA_TRAIN if task == "prosqa" else GSM_TRAIN


def _select_train_indices(task, max_instances, seed=0):
    train_path = _train_data_path(task)
    with open(train_path, "r") as f:
        n_total = len(json.load(f))
    if max_instances is None or max_instances >= n_total:
        return list(range(n_total))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)[:max_instances]
    return sorted(perm.tolist())


def _extract_indices(task, model, n_thoughts, device, indices,
                     batch_size=32, log_prefix="[extract]", family="gpt2"):
    from experiments.extract_thoughts import (
        extract_thoughts_single_instance,
        extract_thoughts_codi_batch,
    )
    from src.utils import setup_model_and_tokenizer, setup_codi_model

    train_path = _train_data_path(task)
    with open(train_path, "r") as f:
        full_data = json.load(f)
    data = [full_data[i] for i in indices]
    N = len(data)
    K = n_thoughts

    is_codi = (model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(task, device, family=family)
        D = codi_dict["hidden_size"]
        thoughts = extract_thoughts_codi_batch(
            codi_dict, data, K, device, batch_size=batch_size,
        )
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ \
            = setup_model_and_tokenizer(task, model, device, family=family)
        # GPT-2 exposes `n_embd`; Llama exposes `hidden_size`.
        D = getattr(base_model.config, "n_embd",
                    getattr(base_model.config, "hidden_size", None))
        thoughts = torch.zeros(N, K + 1, D)
        for j, sample in enumerate(data):
            if j % 100 == 0:
                print(f"  {log_prefix} {j}/{N}", flush=True)
            thoughts[j] = extract_thoughts_single_instance(
                coconut_model, base_model, tokenizer, sample, K, device,
                start_id, latent_id, end_id,
            )
    return thoughts


def extract_train_thoughts(task, model, n_thoughts, device,
                           batch_size=32, max_instances=None,
                           num_gpus=1, seed=0, family="gpt2"):
    indices = _select_train_indices(task, max_instances, seed=seed)
    N = len(indices)
    print(f"[extract] {model}/{task} ({family}) TRAIN: {N} instances "
          f"(max_instances={max_instances}, seed={seed}, num_gpus={num_gpus})")

    # Layout matches extract_thoughts.py: THOUGHTS/<family>/<task>/...
    out_path = THOUGHTS / family / task / f"thoughts_{model}_train.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if num_gpus <= 1:
        thoughts = _extract_indices(task, model, n_thoughts, device,
                                    indices, batch_size=batch_size,
                                    family=family)
    else:
        thoughts = _extract_with_workers(
            task, model, n_thoughts, indices,
            batch_size=batch_size, num_gpus=num_gpus, seed=seed,
            family=family,
        )

    torch.save({
        "thoughts": thoughts,
        "instance_indices": indices,
        "n_thoughts": n_thoughts,
        "model": model,
        "model_family": family,
        "data_path": str(_train_data_path(task)),
        "split": "train",
        "subsample_seed": seed,
        "max_instances": max_instances,
    }, out_path)
    print(f"[extract] saved {tuple(thoughts.shape)} -> {out_path}")
    return thoughts


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU orchestration
# ═══════════════════════════════════════════════════════════════════

def _shard_path(task, model, shard_id, family="gpt2"):
    return THOUGHTS / family / task / f"_thoughts_{model}_train_shard{shard_id}.pt"


def _extract_with_workers(task, model, n_thoughts, indices,
                          batch_size, num_gpus, seed, family="gpt2"):
    import os
    import subprocess
    import sys
    import tempfile

    module_name = None
    spec = globals().get("__spec__", None)
    if spec is not None and spec.name and spec.name != "__main__":
        module_name = spec.name

    project_root = str(Path(__file__).resolve().parents[1])
    chunks = np.array_split(np.array(indices), num_gpus)
    tmp_dir = Path(tempfile.mkdtemp(prefix="markov_shards_"))
    procs = []
    for gpu_id, chunk in enumerate(chunks):
        idx_file = tmp_dir / f"indices_gpu{gpu_id}.json"
        with open(idx_file, "w") as f:
            json.dump([int(i) for i in chunk], f)

        env = dict(**os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (project_root + os.pathsep + existing
                             if existing else project_root)

        if module_name is not None:
            launcher = [sys.executable, "-m", module_name]
        else:
            launcher = [sys.executable, str(Path(__file__).resolve())]

        # NOTE: family must be passed to BOTH --model_family (consumed by the
        # worker's _build_ctx path) and --_worker_family (used inside
        # _run_worker_mode), else the worker silently loads gpt2.
        cmd = launcher + [
            "--_worker_mode",
            "--_worker_task", task,
            "--_worker_model", model,
            "--_worker_n_thoughts", str(n_thoughts),
            "--_worker_indices_file", str(idx_file),
            "--_worker_shard_id", str(gpu_id),
            "--_worker_batch_size", str(batch_size),
            "--_worker_family", family,
            "--task", task, "--model", model, "--model_family", family,
        ]
        print(f"[orchestrator] launching worker {gpu_id} on GPU {gpu_id} "
              f"({family}) with {len(chunk)} instances", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    failed = []
    for gpu_id, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failed.append((gpu_id, rc))
    if failed:
        raise RuntimeError(f"Worker(s) failed: {failed}")

    parts = []
    for gpu_id in range(num_gpus):
        path = _shard_path(task, model, gpu_id, family=family)
        parts.append(torch.load(path, map_location="cpu",
                                weights_only=False)["thoughts"])
        path.unlink()
    for gpu_id in range(num_gpus):
        (tmp_dir / f"indices_gpu{gpu_id}.json").unlink(missing_ok=True)
    tmp_dir.rmdir()
    return torch.cat(parts, dim=0)


def _run_worker_mode(args):
    with open(args._worker_indices_file, "r") as f:
        indices = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = args._worker_family
    print(f"[worker {args._worker_shard_id}] device={device} "
          f"family={family} n_indices={len(indices)}", flush=True)
    thoughts = _extract_indices(
        args._worker_task, args._worker_model, args._worker_n_thoughts,
        device, indices, batch_size=args._worker_batch_size,
        log_prefix=f"[worker {args._worker_shard_id}]", family=family,
    )
    out = _shard_path(args._worker_task, args._worker_model,
                      args._worker_shard_id, family=family)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"thoughts": thoughts, "indices": indices}, out)
    print(f"[worker {args._worker_shard_id}] saved {tuple(thoughts.shape)} "
          f"-> {out}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_thoughts(task, model, n_thoughts, device,
                  batch_size=32, max_train_instances=None,
                  num_gpus=1, seed=0, family="gpt2"):
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
        print(f"[INFO] Train thoughts missing at {train_path}; extracting...")
        extract_train_thoughts(task, model, n_thoughts, device,
                               batch_size=batch_size,
                               max_instances=max_train_instances,
                               num_gpus=num_gpus, seed=seed, family=family)

    train = torch.load(train_path, map_location="cpu",
                       weights_only=False)["thoughts"]
    test = torch.load(test_path, map_location="cpu",
                      weights_only=False)["thoughts"]
    print(f"[INFO] train={tuple(train.shape)}  test={tuple(test.shape)}")
    return train, test


# ═══════════════════════════════════════════════════════════════════
# Subspace projection
# ═══════════════════════════════════════════════════════════════════

def _bases_path(task, model, family="gpt2"):
    # Must match gradient_subspace.py's output layout.
    return (BASE_DIR / "outputs" / "gradient_geometry"
            / family / task / model / "bases.npz")


def load_bases(task, model, bases_path=None, family="gpt2"):
    path = Path(bases_path) if bases_path else _bases_path(task, model, family=family)
    if not path.exists():
        raise FileNotFoundError(
            f"bases.npz not found at {path}. "
            f"Run gradient_subspace.py for --task {task} --model {model} "
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
    X_chunks, Y_chunks, target_ts = [], [], []
    for t in range(first_target, Kp1):
        if t in skip_ts:
            continue
        ctx = [thoughts[:, t - k, :] for k in range(1, order + 1)]
        X_chunks.append(torch.cat(ctx, dim=-1))
        Y_chunks.append(thoughts[:, t, :])
        target_ts.append(t)
    if not X_chunks:
        return None, None, []
    return (torch.cat(X_chunks, dim=0),
            torch.cat(Y_chunks, dim=0),
            target_ts)


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

def fit_linear_ridge(X_tr, Y_tr, ridge=1.0, device="cpu"):
    X_tr = X_tr.to(device)
    Y_tr = Y_tr.to(device)
    n, p = X_tr.shape
    
    X_aug = torch.cat([torch.ones(n, 1, dtype=X_tr.dtype, device=device),
                       X_tr], dim=1)
    A = X_aug.T @ X_aug
    
    # Create a penalty vector that ONLY regularizes the features.
    # We multiply by n to keep regularization invariant to dataset size, 
    # but we DO NOT penalize the bias (index 0).
    penalty = torch.full((p + 1,), ridge * n, dtype=A.dtype, device=device)
    penalty[0] = 0.0  # Leave the bias completely unpenalized
    
    A.diagonal().add_(penalty)
    
    B = X_aug.T @ Y_tr
    return torch.linalg.solve(A, B)


def predict_linear(W_aug, X):
    device = W_aug.device
    n = X.shape[0]
    X = X.to(device)
    X_aug = torch.cat([torch.ones(n, 1, dtype=X.dtype, device=device),
                       X], dim=1)
    return (X_aug @ W_aug).cpu()


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

    X_in = X_tr[tr_idx].to(device); Y_in = Y_tr[tr_idx].to(device)
    X_va = X_tr[val_idx].to(device); Y_va = Y_tr[val_idx].to(device)

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
            opt.zero_grad()
            loss_fn(model(X_in[idx]), Y_in[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_va), Y_va).item()
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
    N, Kp1, _ = thoughts_eval.shape
    start_pos = 1 if drop_h0 else 0
    Y_pred, Y_true = [], []
    for t in range(start_pos + order, Kp1):
        if t in skip_ts:
            continue
        Y_pred.append(thoughts_eval[:, t - 1, :])
        Y_true.append(thoughts_eval[:, t, :])
    if not Y_pred:
        return None, None
    return torch.cat(Y_pred, dim=0), torch.cat(Y_true, dim=0)


# ═══════════════════════════════════════════════════════════════════
# Run a single (task, model)
# ═══════════════════════════════════════════════════════════════════

# [ONLY showing the modified run() section — everything else remains EXACTLY your v3]

def run(task, model, orders, ridge, mlp_hidden, device, drop_h0=True,
        n_thoughts=6, batch_size=32, max_train_instances=None,
        num_gpus=1, seed=0, project_to_subspace=False, bases_path=None,
        out_dir=None, mlp_seeds=None, n_boot=1000, family="gpt2"):

    if mlp_seeds is None:
        mlp_seeds = [0, 1, 2]

    print(f"\n{'='*64}\n  task={task}  model={model}  family={family}"
          + ("  [SUBSPACE-PROJECTED]" if project_to_subspace else "")
          + f"\n{'='*64}")

    train, test = load_thoughts(task, model, n_thoughts=n_thoughts,
                                device=device, batch_size=batch_size,
                                max_train_instances=max_train_instances,
                                num_gpus=num_gpus, seed=seed, family=family)

    Kp1 = train.shape[1]
    skip_ts = []

    if project_to_subspace:
        bases = load_bases(task, model, bases_path=bases_path, family=family)
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

    # Bootstrap CI output path (sibling to the results JSON)
    suffix = "_subspace" if project_to_subspace else ""
    ci_dir = out_dir if out_dir else BASE_DIR / "outputs" / "markovianity"
    cis_jsonl = str(Path(ci_dir) / f"cis_{model}_{task}{suffix}.jsonl")

    results = {
        "task": task,
        "model": model,
        "model_family": family,
        "drop_h0": drop_h0,
        "projected": bool(project_to_subspace),
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
                  "projected": bool(project_to_subspace)}

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
        help="Base model family. Threads through train-thought extraction "
             "(model loading) and namespaces all thought/bases/output paths.",
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
        help="Device for thought extraction, ridge solves, and MLP fitting.",
    )
    parser.add_argument("--include_h0", action="store_true",
                        help="Include h_0 (pre-recurrence) in inputs.")
    parser.add_argument("--n_thoughts", type=int, default=6,
                        help="K, used only when extracting train thoughts.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Used only for CODI extraction.")
    parser.add_argument("--max_train_instances", type=int, default=100_000,
                        help="Cap on TRAIN instances to extract. None=all. "
                             "Subsampled with --seed; sufficient for "
                             "fitting up to ~4k-dim ridge regressions.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="If >1, shard TRAIN extraction across N GPUs "
                             "via subprocess workers.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Subsample + shuffle seed.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--project_to_subspace", type=str, choices=["off", "on", "both"],
        default="off",
        help="Project thoughts onto the per-t gradient subspace B_t "
             "(loaded from gradient_geometry/<task>/<model>/bases.npz) "
             "before fitting / baselines / evaluation. Modes: 'off' (default), "
             "'on', or 'both'. Output filenames get a `_subspace` suffix "
             "so projected and unprojected results sit side-by-side.",
    )
    parser.add_argument(
        "--bases_path", type=str, default=None,
        help="Override path to bases.npz. Default: "
             "BASE_DIR/outputs/gradient_geometry/<family>/<task>/<model>/bases.npz",
    )
    parser.add_argument("--n_boot", type=int, default=1000,
                    help="Number of bootstrap iterations.")

    parser.add_argument("--_worker_mode", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_task", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_model", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_n_thoughts", type=int, default=6,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_indices_file", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_shard_id", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_batch_size", type=int, default=32,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_family", type=str, default="gpt2",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker_mode:
        _run_worker_mode(args)
        return

    set_seed(args.seed)

    tasks = ["prosqa", "gsm"] if args.task == "all" else [args.task]
    models = (["pause", "coconut", "coconut_u", "codi"]
              if args.model == "all" else [args.model])

    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "markovianity" / args.model_family
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.project_to_subspace == "both":
        proj_modes = [False, True]
    elif args.project_to_subspace == "on":
        proj_modes = [True]
    else:
        proj_modes = [False]

    all_results = []
    for task in tasks:
        for model in models:
            for proj_mode in proj_modes:
                suffix = "_subspace" if proj_mode else ""
                res = run(
                    task=task, model=model,
                    orders=sorted(args.orders),
                    ridge=args.ridge, mlp_hidden=args.mlp_hidden,
                    device=args.device, drop_h0=(not args.include_h0),
                    n_thoughts=args.n_thoughts, batch_size=args.batch_size,
                    max_train_instances=args.max_train_instances,
                    num_gpus=args.num_gpus, seed=args.seed,
                    project_to_subspace=proj_mode,
                    bases_path=args.bases_path,
                    out_dir=out_dir,
                    mlp_seeds=args.mlp_seeds,
                    n_boot=args.n_boot,
                    family=args.model_family,
                )
                out_path = out_dir / f"results_{model}_{task}{suffix}.json"
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[saved] {out_path}")
                all_results.append(res)


if __name__ == "__main__":
    main()