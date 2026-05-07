"""
bootstrap_stats.py
==================

Shared statistical-uncertainty utilities for the LRM interpretability project.
Produces bootstrap CIs and paired tests over a finite test set, addressing
sampling uncertainty even when scripts evaluate deterministically on 100% of
the test split.

Three metric patterns are supported:

  A) per-instance scalar mean   (accuracy, flip rate, P(correct), hit rate,
                                 superposition rate, step alignment, ...)
  B) function-of-all-instances  (variance decomposition ratios, R^2,
                                 subspace stability — recomputed each draw)
  C) paired comparison          (original vs ablated on identical instances)

Every script saves a per-instance results vector (or aligned tensors for B/C)
to a standard JSON record so CIs can be recomputed offline without rerunning
inference.

Usage from an experiment script (minimal, two added lines at the aggregation
point) — see end of file for examples.
"""
import os
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, Sequence, Union

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

DEFAULT_N_BOOT = 10_000
DEFAULT_CI = 95.0
DEFAULT_SEED = 0


# ───────────────────────────────────────────────────────────────────────────
# Result record
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """A single CI record. Saved to disk; reloadable for re-aggregation."""
    metric: str                       # short metric name, e.g. "accuracy"
    point: float                      # observed estimate on full test set
    ci_low: float                     # lower CI bound
    ci_high: float                    # upper CI bound
    ci_level: float                   # e.g. 95.0
    n: int                            # sample size (instances)
    n_boot: int                       # bootstrap iterations used
    seed: int                         # RNG seed
    method: str                       # "mean" | "function" | "paired_diff" | "mcnemar"
    extra: dict = field(default_factory=dict)

    def to_short_str(self) -> str:
        # Compact display for table cells / logs.
        return (f"{self.point:.3f} "
                f"[{self.ci_low:.3f}, {self.ci_high:.3f}] "
                f"(n={self.n})")


# ───────────────────────────────────────────────────────────────────────────
# Pattern A — per-instance scalar mean
# ───────────────────────────────────────────────────────────────────────────
# point = mean(values)
# CI: for b = 1..B, sample N indices with replacement, take mean, then take
#     percentiles [(100-ci)/2, 100-(100-ci)/2] of the B bootstrap means.

def bootstrap_mean(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    metric: str = "mean",
) -> BootstrapResult:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return BootstrapResult(metric, float("nan"), float("nan"), float("nan"),
                               ci, 0, n_boot, seed, method="mean")
    rng = np.random.default_rng(seed)
    # Vectorized resampling: B x N index matrix, then mean along axis=1.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return BootstrapResult(
        metric=metric,
        point=float(arr.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=ci,
        n=n,
        n_boot=n_boot,
        seed=seed,
        method="mean",
    )


# ───────────────────────────────────────────────────────────────────────────
# Pattern B — function of all instances (R^2, variance ratios, subspace stats)
# ───────────────────────────────────────────────────────────────────────────
# point = f(payload)
# CI: for b = 1..B, sample N row indices with replacement, slice payload along
#     `instance_axis` for every array in payload, recompute f, take percentiles
#     of the B bootstrap statistics.
#
# `payload` is a dict of arrays sharing axis `instance_axis` (default 0). The
# user-supplied `func` receives a payload dict with the same keys and returns a
# single scalar.

def bootstrap_function(
    payload: dict,
    func: Callable[[dict], float],
    n: int,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    instance_axis: int = 0,
    metric: str = "function",
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    point = float(func(payload))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = {k: np.take(v, idx, axis=instance_axis) for k, v in payload.items()}
        boots[b] = float(func(sub))
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return BootstrapResult(
        metric=metric,
        point=point,
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=ci,
        n=n,
        n_boot=n_boot,
        seed=seed,
        method="function",
    )


# Convenience wrappers for the two B-pattern statistics that recur in the paper.

def bootstrap_r2(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    multioutput: str = "uniform_average",
    metric: str = "r2",
) -> BootstrapResult:
    # R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2), per output dim,
    # then averaged according to `multioutput`.
    from sklearn.metrics import r2_score
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    assert Y_true.shape == Y_pred.shape, "shape mismatch"
    n = Y_true.shape[0]

    def _r2(p):
        return float(r2_score(p["y"], p["yhat"], multioutput=multioutput))

    return bootstrap_function(
        payload={"y": Y_true, "yhat": Y_pred},
        func=_r2,
        n=n,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
        metric=metric,
    )


def bootstrap_variance_decomposition(
    thoughts: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Bootstrap the (pct_timestep, pct_instance, pct_residual) decomposition.
    `thoughts` has shape [N, T, D]; the instance axis is 0.

        var_total    = E_{i,t}[ || h_{i,t} - mu ||^2 ]
        var_timestep = E_t   [ || mu_t      - mu ||^2 ]
        var_instance = E_i   [ || mu_i      - mu ||^2 ]
        var_residual = var_total - var_timestep - var_instance
    """
    thoughts = np.asarray(thoughts)
    n = thoughts.shape[0]

    def decomp(p):
        h = p["h"]                                # [N, T, D]
        mu   = h.mean(axis=(0, 1))                # [D]
        mu_t = h.mean(axis=0)                     # [T, D]
        mu_i = h.mean(axis=1)                     # [N, D]
        v_total = ((h - mu) ** 2).sum(axis=2).mean()
        v_time  = ((mu_t - mu) ** 2).sum(axis=1).mean()
        v_inst  = ((mu_i - mu) ** 2).sum(axis=1).mean()
        return v_total, v_time, v_inst

    # Point estimate
    v_total, v_time, v_inst = decomp({"h": thoughts})
    v_resid = v_total - v_time - v_inst
    pct_time  = 100.0 * v_time  / max(v_total, 1e-12)
    pct_inst  = 100.0 * v_inst  / max(v_total, 1e-12)
    pct_resid = 100.0 * v_resid / max(v_total, 1e-12)

    # Bootstrap each ratio jointly via index resampling (same indices → all 3)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 3), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v_t, v_ti, v_in = decomp({"h": thoughts[idx]})
        v_re = v_t - v_ti - v_in
        denom = max(v_t, 1e-12)
        boot[b, 0] = 100.0 * v_ti / denom
        boot[b, 1] = 100.0 * v_in / denom
        boot[b, 2] = 100.0 * v_re / denom

    out = {}
    for k, point, col in [
        ("pct_timestep", pct_time,  0),
        ("pct_instance", pct_inst,  1),
        ("pct_residual", pct_resid, 2),
    ]:
        lo, hi = np.percentile(boot[:, col], [(100 - ci) / 2, 100 - (100 - ci) / 2])
        out[k] = BootstrapResult(
            metric=k, point=float(point),
            ci_low=float(lo), ci_high=float(hi),
            ci_level=ci, n=n, n_boot=n_boot, seed=seed,
            method="function",
        )
    return out


# ───────────────────────────────────────────────────────────────────────────
# Pattern C — paired comparisons (original vs ablated on identical instances)
# ───────────────────────────────────────────────────────────────────────────

def paired_bootstrap_diff(
    cond_a: Sequence[float],
    cond_b: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    metric: str = "paired_diff",
) -> BootstrapResult:
    # Bootstrap CI on E[a_i - b_i]. Pairing is preserved by sampling the SAME
    # index for both vectors each draw.
    a = np.asarray(cond_a, dtype=np.float64)
    b = np.asarray(cond_b, dtype=np.float64)
    assert a.shape == b.shape, "paired vectors must align"
    n = len(a)
    if n == 0:
        return BootstrapResult(metric, float("nan"), float("nan"), float("nan"),
                               ci, 0, n_boot, seed, method="paired_diff")
    diffs = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return BootstrapResult(
        metric=metric,
        point=float(diffs.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=ci,
        n=n,
        n_boot=n_boot,
        seed=seed,
        method="paired_diff",
    )


def mcnemar_test(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    metric: str = "mcnemar",
) -> dict:
    """
    Exact McNemar test for paired binary outcomes (each entry is 0 or 1).
    Discordant pairs:
        b = #{a=1, b=0}   (A correct, B wrong)
        c = #{a=0, b=1}   (A wrong,   B correct)
    Returns p-value (two-sided exact binomial on min(b, c) with n=b+c, p=0.5),
    plus the contingency counts.
    """
    from scipy.stats import binomtest
    a = np.asarray(correct_a, dtype=np.int64)
    bv = np.asarray(correct_b, dtype=np.int64)
    assert a.shape == bv.shape, "paired vectors must align"
    b = int(((a == 1) & (bv == 0)).sum())
    c = int(((a == 0) & (bv == 1)).sum())
    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(b, c), n_disc, 0.5,
                                  alternative="two-sided").pvalue)
    return {
        "metric": metric,
        "b_a_only": b,
        "c_b_only": c,
        "n_discordant": n_disc,
        "p_value": p_value,
    }


# ───────────────────────────────────────────────────────────────────────────
# Persistence — one JSON record per call site, append-friendly
# ───────────────────────────────────────────────────────────────────────────
# We use JSONL (one record per line) so multiple metrics from one script
# accumulate without rewriting the file. Loading reads them all back.

def save_record(
    out_path: str,
    record: Union[BootstrapResult, dict],
    context: Optional[dict] = None,
    overwrite: bool = False,
) -> None:
    # `context` carries identifying metadata (model, task, condition, alpha,
    # ...) so the JSONL is reusable across the whole project.
    payload = asdict(record) if isinstance(record, BootstrapResult) else dict(record)
    if context:
        payload["_context"] = context
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mode = "w" if overwrite else "a"
    with open(out_path, mode) as f:
        f.write(json.dumps(payload) + "\n")


def save_per_instance_vector(
    out_path: str,
    values: Sequence[float],
    context: dict,
    overwrite: bool = True,
) -> None:
    """
    Save the raw per-instance vector underlying a metric. Lets us recompute
    CIs offline, change n_boot, or do paired tests later — without rerunning
    inference. Uses .npz for arrays + a sibling .json for context.
    """
    arr = np.asarray(values)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(out_path)
    np.savez(out_path, values=arr)
    with open(out_path.replace(".npz", ".json"), "w") as f:
        json.dump(context, f, indent=2)


def load_records(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ───────────────────────────────────────────────────────────────────────────
# One-call helper for the dominant case (Pattern A + persistence)
# ───────────────────────────────────────────────────────────────────────────

def report_mean_with_ci(
    values: Sequence[float],
    metric: str,
    context: dict,
    cis_jsonl: str,
    vector_npz: Optional[str] = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    log: bool = True,
) -> BootstrapResult:
    """
    One-line replacement for `accuracy = n_correct / n` style aggregation.
    - Computes point estimate + bootstrap CI.
    - Appends a JSON record to `cis_jsonl`.
    - Optionally saves raw per-instance vector to `vector_npz` for re-use.
    - Optionally prints a one-line summary.
    """
    res = bootstrap_mean(values, n_boot=n_boot, ci=ci, seed=seed, metric=metric)
    save_record(cis_jsonl, res, context=context)
    if vector_npz is not None:
        save_per_instance_vector(vector_npz, values, context=context)
    if log:
        ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
        print(f"  [CI] {metric} {ctx_str}: {res.to_short_str()}")
    return res


# ───────────────────────────────────────────────────────────────────────────
# Self-test — sanity checks for the math
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) bootstrap_mean: known Bernoulli
    p_true = 0.7
    n = 500
    samples = rng.binomial(1, p_true, size=n)
    r = bootstrap_mean(samples, seed=1, metric="acc")
    print(f"[A] mean({p_true}) ≈ {r.to_short_str()}  "
          f"(should cover {p_true})")
    assert r.ci_low <= p_true <= r.ci_high, "CI failed to cover true mean"

    # 2) paired_bootstrap_diff: known constant gap
    a = rng.binomial(1, 0.8, size=300)
    b = a.copy()
    flip_idx = rng.choice(len(a), size=30, replace=False)
    b[flip_idx] = 1 - b[flip_idx]
    r2 = paired_bootstrap_diff(a, b, seed=1, metric="orig-ablated")
    print(f"[C] paired diff: {r2.to_short_str()}  (b is a-with-30-flipped)")

    # 3) mcnemar
    mc = mcnemar_test(a, b)
    print(f"[C] McNemar: {mc}")

    # 4) variance decomposition: synthetic [N,T,D] where T explains all variance
    N, T, D = 200, 5, 16
    t_offsets = rng.normal(size=(T, D)) * 5.0
    h = np.tile(t_offsets[None, :, :], (N, 1, 1)) + rng.normal(scale=0.01, size=(N, T, D))
    out = bootstrap_variance_decomposition(h, n_boot=500, seed=1)
    print(f"[B] var-decomp synthetic-time-dominated:")
    for k, v in out.items():
        print(f"    {k}: {v.to_short_str()}")

    # 5) R^2 sanity — perfect prediction should give R^2 = 1 with tight CI
    y = rng.normal(size=(400, 8))
    rr = bootstrap_r2(y, y.copy(), n_boot=500, seed=1, metric="r2_perfect")
    print(f"[B] R^2 perfect: {rr.to_short_str()}")

    # 6) Persistence round-trip
    tmp = "/tmp/_bs_test.jsonl"
    if os.path.exists(tmp):
        os.remove(tmp)
    save_record(tmp, r, context={"model": "test", "metric": "acc"})
    save_record(tmp, r2, context={"model": "test", "metric": "diff"})
    loaded = load_records(tmp)
    assert len(loaded) == 2 and loaded[0]["_context"]["metric"] == "acc"
    print(f"[persist] round-trip OK: {len(loaded)} records")

    print("\nAll self-tests passed.")