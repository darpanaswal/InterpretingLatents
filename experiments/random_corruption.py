"""
Hypothesis test: do latent thoughts carry useful information for generation?

═══════════════════════════════════════════════════════════════════════════════
Statistical framing
═══════════════════════════════════════════════════════════════════════════════

We test the candidate hypothesis (H1) that latent thoughts build up useful
representations for the model's predictions. The null hypothesis (H0) is that
they don't — replacing or perturbing them with noise should leave model
predictions unchanged in distribution.

Two complementary statistical lenses (same per-instance correctness vectors):

  1. Monte-Carlo p-value over N i.i.d. noisy runs (across-seed null):
     # n_ge   = #{i : A_null[i] >= A_obs}
     # p      = (1 + n_ge) / (N + 1)          (conservative MC p-value)
     This tests: is the *baseline accuracy* surprisingly high under the null
     distribution of noisy-run accuracies?

  2. Bootstrap CIs + paired tests over instances (within-seed):
     - Bootstrap CI on baseline accuracy.
     - Bootstrap CI on the across-seed mean accuracy for each perturbation,
       built from the (N × n_instances) flat pool of noisy outcomes.
     - For each (perturbation, seed) — paired bootstrap diff and exact McNemar
       on (baseline_i, null_i): does noise change *which instances* the model
       gets right, holding instance identity fixed?

Effect sizes:
    # delta_acc    = A_obs - mean(A_null)
    # log10_ratio  = log10(A_obs / median(A_null))      (when both > 0)
    # z            = (A_obs - mean(A_null)) / std(A_null)   (when std > 0)

Decision (MC):
    p small  → A_obs sits in the upper tail of the noisy-runs distribution
              → noise hurts performance → REJECT H0 → H1 supported.
    p large  → noise doesn't degrade accuracy
              → CANNOT REJECT H0 → evidence the thoughts are causally inert.

═══════════════════════════════════════════════════════════════════════════════
Perturbations
═══════════════════════════════════════════════════════════════════════════════

For thought vector h_t at step t:

    additive (mode = "noise"):    h'_t = h_t + λ σ_{h_t} ε,    ε ~ N(0, I)
    replacement (mode = "replace"): h'_t =       σ_{h_t} ε,    ε ~ N(0, I)

where σ_{h_t} = std(h_t) over feature dimensions (per-step, per-instance).

═══════════════════════════════════════════════════════════════════════════════
Models, tasks, and parallelism
═══════════════════════════════════════════════════════════════════════════════

Supports {coconut, coconut_u, pause, codi} × {prosqa, gsm}.

CODI inference always goes through a batched path (run_codi_batched_intervened
below), even at batch_size=1, per project convention. Coconut/Pause are
unbatched (their per-instance pause-aware recurrence does not pad cleanly),
so --codi_batch_size only affects CODI.

Multi-GPU sharding follows mean_ablation.py: contiguous shards across
cuda:0..cuda:{n_gpus-1}, results merged in main process. Per-instance kernels
are seeded by GLOBAL instance idx so the noise is shard-invariant.

Usage:
    python -m experiments.random_corruption --task prosqa --model pause
    python -m experiments.random_corruption --task gsm    --model codi \
        --codi_batch_size 16 --n_seeds 100 --n_gpus 4

    # Llama family (add --model_family llama):
    python -m experiments.random_corruption --task gsm --model codi \
        --model_family llama --codi_batch_size 16 --n_seeds 100 --n_gpus 4
"""

import json
import math
import torch
import argparse
import numpy as np
import sys
from pathlib import Path
import queue
import torch.multiprocessing as mp
import concurrent.futures

from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, set_seed
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    run_intervened_inference_pauseaware,
    _compare_answers,
)
from src.bootstrap_stats import (
    bootstrap_mean, paired_bootstrap_diff, mcnemar_test,
    save_record, save_per_instance_vector,
)


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def deep_convert(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════
# Perturbation kernels (Thread-safe and Picklable)
# ═══════════════════════════════════════════════════════════════════

def _seed_for(run_seed, idx, t):
    return run_seed * 1_000_000_000 + idx * 10_000 + t + 1


class AdditiveNoiseKernel:
    def __init__(self, noise_scale, run_seed, idx):
        self.noise_scale = noise_scale
        self.run_seed = run_seed
        self.idx = idx

    def __call__(self, h, t):
        # Local generator prevents race conditions when multi-threading
        gen = torch.Generator(device=h.device)
        gen.manual_seed(_seed_for(self.run_seed, self.idx, t))
        std_h = torch.std(h, dim=-1, keepdim=True)
        eps = torch.randn(*h.shape, generator=gen, device=h.device, dtype=h.dtype)
        return h + self.noise_scale * std_h * eps


class ReplacementNoiseKernel:
    def __init__(self, run_seed, idx):
        self.run_seed = run_seed
        self.idx = idx

    def __call__(self, h, t):
        gen = torch.Generator(device=h.device)
        gen.manual_seed(_seed_for(self.run_seed, self.idx, t))
        std_h = torch.std(h, dim=-1, keepdim=True)
        eps = torch.randn(*h.shape, generator=gen, device=h.device, dtype=h.dtype)
        return std_h * eps


class IdentityKernel:
    def __call__(self, h, t):
        return h


class AdditiveNoiseFactory:
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def __call__(self, run_seed, idx):
        return AdditiveNoiseKernel(self.noise_scale, run_seed, idx)


class ReplacementNoiseFactory:
    def __call__(self, run_seed, idx):
        return ReplacementNoiseKernel(run_seed, idx)


class IdentityFactory:
    def __call__(self, run_seed, idx):
        return IdentityKernel()


def make_additive_noise_factory(noise_scale):
    return AdditiveNoiseFactory(noise_scale)


def make_replacement_noise_factory():
    return ReplacementNoiseFactory()


def make_identity_factory():
    return IdentityFactory()


# ═══════════════════════════════════════════════════════════════════
# Batched CODI inference with per-step, per-instance intervention
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_codi_batched_intervened(
    codi_dict, batch, kernel_factory, run_seed, batch_global_indices,
    n_thoughts, device, task, max_decode_tokens=256,
):
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']
    vocab_size = base_model.config.vocab_size

    B = len(batch)
    assert len(batch_global_indices) == B, "batch and indices must align"

    interventions = [
        kernel_factory(run_seed, batch_global_indices[b]) for b in range(B)
    ]

    questions = [s["question"].strip().replace('  ', ' ') for s in batch]
    enc = tokenizer(questions, return_tensors="pt", padding="longest")
    input_ids = enc["input_ids"].to(device)            # (B, L)
    attention_mask = enc["attention_mask"].to(device)  # (B, L)

    if remove_eos:
        suffix = torch.tensor([[bot_id]] * B, dtype=torch.long, device=device)
    else:
        suffix = torch.tensor(
            [[tokenizer.eos_token_id, bot_id]] * B,
            dtype=torch.long, device=device,
        )
    input_ids = torch.cat([input_ids, suffix], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(suffix)], dim=1)

    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    real_len = attention_mask.sum(dim=1)               # (B,)

    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv = outputs.past_key_values
    h_rows = outputs.hidden_states[-1][:, -1, :]       # (B, D)

    h = torch.stack(
        [interventions[b](h_rows[b], 0) for b in range(B)], dim=0,
    )                                                  # (B, D)

    latent = h.unsqueeze(1)                            # (B, 1, D)
    if use_prj and prj is not None:
        latent = prj(latent)

    running_mask = attention_mask
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((B, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        pos_t = (real_len + (t - 1)).unsqueeze(1)      # (B, 1)

        outputs = base_model(
            inputs_embeds=latent,
            attention_mask=running_mask,
            position_ids=pos_t,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        h_rows = outputs.hidden_states[-1][:, -1, :]   # (B, D)

        h = torch.stack(
            [interventions[b](h_rows[b], t) for b in range(B)], dim=0,
        )
        latent = h.unsqueeze(1)
        if use_prj and prj is not None:
            latent = prj(latent)

    if remove_eos:
        eot_ids = torch.tensor([[eot_id]] * B, device=device)
    else:
        eot_ids = torch.tensor(
            [[eot_id, tokenizer.eos_token_id]] * B, device=device,
        )
    eot_emb = embedding_fn(eot_ids)                    # (B, eot_len, D)
    eot_len = eot_emb.size(1)

    eot_pos = (real_len + n_thoughts).unsqueeze(1) + torch.arange(
        eot_len, device=device,
    ).unsqueeze(0)                                     # (B, eot_len)
    running_mask = torch.cat(
        [running_mask, torch.ones((B, eot_len), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )

    outputs = base_model(
        inputs_embeds=eot_emb,
        attention_mask=running_mask,
        position_ids=eot_pos,
        use_cache=True,
        past_key_values=past_kv,
    )
    past_kv = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :vocab_size - 1]   # (B, V-1)

    current_pos = real_len + n_thoughts + eot_len

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pred_tokens = [[] for _ in range(B)]

    for _ in range(max_decode_tokens):
        next_token_ids = next_logits.argmax(dim=-1)    # (B,)

        for b in range(B):
            if not finished[b]:
                tok = next_token_ids[b].item()
                pred_tokens[b].append(tok)
                if tok == tokenizer.eos_token_id:
                    finished[b] = True

        if finished.all():
            break

        next_emb = embedding_fn(next_token_ids).unsqueeze(1)
        running_mask = torch.cat(
            [running_mask, torch.ones((B, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        decode_pos = current_pos.unsqueeze(1)
        out = base_model(
            inputs_embeds=next_emb,
            attention_mask=running_mask,
            position_ids=decode_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = out.logits[:, -1, :vocab_size - 1]
        past_kv = out.past_key_values
        current_pos = current_pos + 1

    results = []
    for b, sample in enumerate(batch):
        text = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
        answer, correct_answer, is_correct = _compare_answers(text, sample, task)
        results.append({
            "predicted": answer,
            "correct": correct_answer,
            "is_correct": is_correct,
            "text": text,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Per-shard evaluation
# ═══════════════════════════════════════════════════════════════════

def _eval_shard_coconut_pause_threaded(ctx, data, indices, jobs, max_workers=16):
    """
    Since Coconut is unbatched, we use a thread pool to launch multiple inference 
    passes concurrently. PyTorch executes them in parallel on the GPU.
    """
    results = {job["key"]: [] for job in jobs}
    
    def _worker_fn(job, idx):
        intervention_fn = job["kernel_factory"](job["run_seed"], idx)
        r = run_intervened_inference_pauseaware(
            coconut_model=ctx["coconut_model"],
            base_model=ctx["base_model"],
            tokenizer=ctx["tokenizer"],
            end_id=ctx["end_id"],
            sample=data[idx],
            n_thoughts=ctx["n_thoughts"],
            device=ctx["device"],
            intervention_fn=intervention_fn,
            start_id=ctx["start_id"],
            latent_id=ctx["latent_id"],
            task=ctx["task"],
        )
        return job["key"], idx, int(r["is_correct"])

    all_tasks = [(job, idx) for job in jobs for idx in indices]
    total_tasks = len(all_tasks)
    
    print(f"  [r{ctx['rank']}] Preparing to launch {total_tasks} inferences across {max_workers} threads...", flush=True)
    
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker_fn, j, i): (j, i) for j, i in all_tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            key, returned_idx, is_correct = future.result()
            results[key].append((returned_idx, is_correct))
            completed += 1
            
            # Print progress frequently with flush=True to bypass mp buffering
            if completed % 250 == 0 or completed == total_tasks:
                print(f"    [r{ctx['rank']}] completed {completed}/{total_tasks} inferences...", flush=True)
                
    return results


def _eval_shard_codi(ctx, data, indices, kernel_factory, run_seed, batch_size, label=""):
    results = []
    n_correct = 0
    N = len(indices)
    n_chunks = math.ceil(N / batch_size)
    for c in range(n_chunks):
        s = c * batch_size
        e = min(s + batch_size, N)
        chunk_idxs = indices[s:e]
        chunk_batch = [data[i] for i in chunk_idxs]

        if s % 100 == 0 and s > 0:
            print(f"    [r{ctx['rank']}][{label}] {s}/{N}  running acc: {n_correct/s:.1%}", flush=True)

        out = run_codi_batched_intervened(
            codi_dict=ctx["codi_dict"],
            batch=chunk_batch,
            kernel_factory=kernel_factory,
            run_seed=run_seed,
            batch_global_indices=chunk_idxs,
            n_thoughts=ctx["n_thoughts"],
            device=ctx["device"],
            task=ctx["task"],
        )
        for idx, r in zip(chunk_idxs, out):
            correct = int(r["is_correct"])
            results.append((idx, correct))
            n_correct += correct

    print(f"  [r{ctx['rank']}][{label}] shard acc: {n_correct}/{N}", flush=True)
    return results


def run_jobs_on_shard(ctx, data, indices, jobs, batch_size):
    """Dispatch execution based on model type."""
    if ctx["is_codi"]:
        out = {}
        for j_idx, job in enumerate(jobs):
            log = f"{job['label']} seed={job['run_seed']}  [job {j_idx+1}/{len(jobs)}]"
            out[job["key"]] = _eval_shard_codi(
                ctx, data, indices,
                kernel_factory=job["kernel_factory"],
                run_seed=job["run_seed"],
                batch_size=batch_size,
                label=log,
            )
        return out
    else:
        # Repurpose batch_size as thread count (min 8)
        n_threads = max(batch_size, 8)
        return _eval_shard_coconut_pause_threaded(ctx, data, indices, jobs, max_workers=n_threads)


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU: shard instances across ranks (matches mean_ablation.py)
# ═══════════════════════════════════════════════════════════════════

def _shard_indices(n_items, world_size, rank):
    chunk = (n_items + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_items)
    return list(range(start, end))


def _build_ctx(rank, task, model_name, n_thoughts, family="gpt2"):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    is_codi = (model_name == "codi")
    ctx = {
        "is_codi": is_codi, "n_thoughts": n_thoughts,
        "device": device, "task": task, "rank": rank,
    }
    if is_codi:
        ctx["codi_dict"] = setup_codi_model(task, device, family=family)
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device, family=family)
        ctx.update({
            "coconut_model": coconut_model, "base_model": base_model,
            "tokenizer": tokenizer, "start_id": start_id,
            "latent_id": latent_id, "end_id": end_id,
        })
    return ctx


def _worker(rank, world_size, task, model_name, n_thoughts,
            data, jobs, batch_size, return_queue, family="gpt2"):
    ctx = _build_ctx(rank, task, model_name, n_thoughts, family=family)
    indices = _shard_indices(len(data), world_size, rank)
    print(f"[rank {rank}] processing {len(indices)} instances on {ctx['device']}", flush=True)
    partial = run_jobs_on_shard(ctx, data, indices, jobs, batch_size)
    return_queue.put({"rank": rank, "partial": partial})


def run_multigpu(task, model_name, n_thoughts, data, jobs, batch_size, n_gpus,
                 family="gpt2"):
    ctx_mp = mp.get_context("spawn")
    q = ctx_mp.Queue()
    procs = []
    for rank in range(n_gpus):
        p = ctx_mp.Process(
            target=_worker,
            args=(rank, n_gpus, task, model_name, n_thoughts,
                  data, jobs, batch_size, q, family),
        )
        p.start()
        procs.append(p)

    shards = []
    for _ in range(n_gpus):
        while True:
            try:
                # 5-second poll so the main process can check for silent worker crashes
                res = q.get(timeout=5.0)
                shards.append(res)
                break
            except queue.Empty:
                for p in procs:
                    if not p.is_alive() and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {p.pid} crashed with exit code {p.exitcode}. "
                            "Check console for CUDA Out-Of-Memory or other runtime errors."
                        )

    for p in procs:
        p.join()

    merged = {}
    all_keys = set()
    for s in shards:
        all_keys.update(s["partial"].keys())
    for key in all_keys:
        combined = []
        for s in shards:
            if key in s["partial"]:
                combined.extend(s["partial"][key])
        combined.sort(key=lambda x: x[0])
        merged[key] = [c for _, c in combined]
    return merged


# ═══════════════════════════════════════════════════════════════════
# Hypothesis test stats
# ═══════════════════════════════════════════════════════════════════

def monte_carlo_p_and_effect(observed, null_array):
    null = np.asarray(null_array, dtype=np.float64)
    N = null.size
    obs = float(observed)

    n_ge = int(np.sum(null >= obs))
    p_value = (1 + n_ge) / (N + 1)

    mean_null = float(np.mean(null))
    median_null = float(np.median(null))
    std_null = float(np.std(null, ddof=1)) if N > 1 else 0.0
    delta_acc = obs - mean_null

    if median_null > 0 and obs > 0:
        log10_ratio = float(np.log10(obs / median_null))
    else:
        log10_ratio = float("nan")

    z_score = (obs - mean_null) / std_null if std_null > 0 else float("nan")

    return {
        "observed": obs,
        "N": N,
        "null_mean": mean_null,
        "null_median": median_null,
        "null_std": std_null,
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
        "n_ge": n_ge,
        "p_value": float(p_value),
        "delta_acc": float(delta_acc),
        "log10_ratio": log10_ratio,
        "z_score": float(z_score),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model", type=str,
        choices=["coconut", "coconut_u", "pause", "codi"], default="coconut",
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Determines checkpoint paths and dtype.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)

    parser.add_argument(
        "--mode", type=str, choices=["noise", "replace", "both"], default="both",
    )
    parser.add_argument(
        "--noise_sweep", type=str, default="0.1,0.5,1.0,5.0,10.0,25.0,50.0",
    )

    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument("--codi_batch_size", type=int, default=8)
    parser.add_argument("--n_gpus", type=int, default=1)

    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--ci_level", type=float, default=95.0)
    parser.add_argument("--bootstrap_seed", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _build_perturbation_specs(args):
    specs = []
    if args.mode in ("noise", "both"):
        for scale in [float(x) for x in args.noise_sweep.split(",")]:
            specs.append((
                f"noise_lambda={scale}",
                make_additive_noise_factory(scale),
                {"mode": "noise", "lambda": scale},
            ))
    if args.mode in ("replace", "both"):
        specs.append((
            "replace_with_scale_matched_gaussian",
            make_replacement_noise_factory(),
            {"mode": "replace"},
        ))
    return specs


def _build_jobs(args, perturbation_specs):
    jobs = [{
        "key": ("baseline", args.base_seed),
        "kernel_factory": make_identity_factory(),
        "run_seed": args.base_seed,
        "label": "baseline",
    }]
    for label, factory, _meta in perturbation_specs:
        for i in range(args.n_seeds):
            seed_i = args.base_seed + i + 1
            jobs.append({
                "key": (label, seed_i),
                "kernel_factory": factory,
                "run_seed": seed_i,
                "label": label,
            })
    return jobs


def _single_gpu_run(args, data, jobs):
    is_codi = (args.model == "codi")
    ctx = {
        "is_codi": is_codi, "n_thoughts": args.n_thoughts,
        "device": args.device, "task": args.task, "rank": 0,
    }
    if is_codi:
        ctx["codi_dict"] = setup_codi_model(args.task, args.device,
                                            family=args.model_family)
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device,
                                      family=args.model_family)
        ctx.update({
            "coconut_model": coconut_model, "base_model": base_model,
            "tokenizer": tokenizer, "start_id": start_id,
            "latent_id": latent_id, "end_id": end_id,
        })
    indices = list(range(len(data)))
    partial = run_jobs_on_shard(ctx, data, indices, jobs, args.codi_batch_size)
    return {k: [c for _, c in sorted(pairs, key=lambda x: x[0])]
            for k, pairs in partial.items()}


def main():
    args = parse_args()
    set_seed(0)

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "random_corruption" / args.model_family \
        / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Task: {args.task}, Model: {args.model}, "
          f"Family: {args.model_family}, "
          f"instances: {len(data)}, n_seeds: {args.n_seeds}, "
          f"mode: {args.mode}, n_gpus: {args.n_gpus}, "
          f"codi_batch_size: {args.codi_batch_size}")

    perturbation_specs = _build_perturbation_specs(args)
    jobs = _build_jobs(args, perturbation_specs)
    print(f"[INFO] {len(jobs)} jobs total "
          f"(1 baseline + {len(perturbation_specs)} pert × {args.n_seeds} seeds)")

    if args.n_gpus > 1:
        per_instance_vectors = run_multigpu(
            args.task, args.model, args.n_thoughts, data, jobs,
            args.codi_batch_size, args.n_gpus, family=args.model_family,
        )
    else:
        per_instance_vectors = _single_gpu_run(args, data, jobs)

    baseline_vec = per_instance_vectors[("baseline", args.base_seed)]
    A_obs = float(np.mean(baseline_vec))

    cis_jsonl = str(output_dir / f"cis_{args.mode}.jsonl")
    if Path(cis_jsonl).exists():
        Path(cis_jsonl).unlink()

    ci_ctx_base = {
        "task": args.task, "model": args.model,
        "n_thoughts": args.n_thoughts, "n_instances": len(data),
        "n_seeds": args.n_seeds, "base_seed": args.base_seed,
        "mode_run": args.mode,
    }
    baseline_ci = bootstrap_mean(
        baseline_vec, n_boot=args.n_boot, ci=args.ci_level,
        seed=args.bootstrap_seed, metric="acc_baseline",
    )
    save_record(cis_jsonl, baseline_ci, context={
        **ci_ctx_base, "perturbation": "baseline", "run_seed": args.base_seed,
    })
    save_per_instance_vector(
        str(output_dir / "vec_baseline.npz"),
        baseline_vec,
        context={**ci_ctx_base, "perturbation": "baseline",
                 "run_seed": args.base_seed},
    )

    print("\n" + "=" * 64)
    print("BASELINE (no perturbation)")
    print("=" * 64)
    print(f"  A_obs = {A_obs:.4f}   CI: {baseline_ci.to_short_str()}")

    print("\n" + "=" * 64)
    print(f"PERTURBATIONS (N = {args.n_seeds} seeds each)")
    print("=" * 64)

    results = []
    for label, _factory, meta in perturbation_specs:
        print(f"\n  [{label}]")

        seed_vectors = []
        for i in range(args.n_seeds):
            seed_i = args.base_seed + i + 1
            vec = per_instance_vectors[(label, seed_i)]
            seed_vectors.append(vec)

        null_accs = [float(np.mean(v)) for v in seed_vectors]
        mc_stats = monte_carlo_p_and_effect(A_obs, null_accs)

        pool = np.concatenate([np.asarray(v) for v in seed_vectors])
        pool_ci = bootstrap_mean(
            pool, n_boot=args.n_boot, ci=args.ci_level,
            seed=args.bootstrap_seed, metric=f"acc_pool_{label}",
        )
        save_record(cis_jsonl, pool_ci, context={
            **ci_ctx_base, "perturbation": label, "aggregation": "pool",
            **meta,
        })

        per_seed_diff_points = []
        per_seed_mcnemar_p = []
        for i, vec in enumerate(seed_vectors):
            seed_i = args.base_seed + i + 1
            diff_ci = paired_bootstrap_diff(
                baseline_vec, vec,
                n_boot=args.n_boot, ci=args.ci_level,
                seed=args.bootstrap_seed,
                metric=f"diff_baseline_vs_{label}",
            )
            save_record(cis_jsonl, diff_ci, context={
                **ci_ctx_base, "perturbation": label,
                "run_seed": seed_i, "aggregation": "per_seed_diff", **meta,
            })
            mc = mcnemar_test(baseline_vec, vec,
                              metric=f"mcnemar_baseline_vs_{label}")
            save_record(cis_jsonl, mc, context={
                **ci_ctx_base, "perturbation": label,
                "run_seed": seed_i, "aggregation": "per_seed_mcnemar", **meta,
            })
            per_seed_diff_points.append(diff_ci.point)
            per_seed_mcnemar_p.append(mc["p_value"])

        vec_arr = np.asarray(seed_vectors, dtype=np.int8)
        save_per_instance_vector(
            str(output_dir / f"vec_{label}.npz"),
            vec_arr,
            context={**ci_ctx_base, "perturbation": label,
                     "shape": list(vec_arr.shape), **meta},
        )

        stats = {
            "label": label, "meta": meta,
            "null_accs": null_accs,
            "mc_stats": mc_stats,
            "pool_ci": {"point": pool_ci.point,
                        "ci_low": pool_ci.ci_low,
                        "ci_high": pool_ci.ci_high,
                        "ci_level": pool_ci.ci_level,
                        "n": pool_ci.n, "n_boot": pool_ci.n_boot},
            "per_seed_diff_points": per_seed_diff_points,
            "per_seed_mcnemar_p":   per_seed_mcnemar_p,
        }
        results.append(stats)

        verdict = ("REJECT H0 (thoughts useful)" if mc_stats["p_value"] < 0.05
                   else "CANNOT REJECT H0")
        print(f"    A_obs={A_obs:.3f}  null_mean={mc_stats['null_mean']:.3f}  "
              f"Δ={mc_stats['delta_acc']:+.3f}  z={mc_stats['z_score']:.2f}  "
              f"p_MC={mc_stats['p_value']:.4f}  → {verdict}")
        print(f"    pool CI: {pool_ci.to_short_str()}")

    print("\n" + "=" * 64)
    print(f"SUMMARY ({args.task.upper()} / {args.model})")
    print("=" * 64)
    header = (
        f"  {'perturbation':<40}  {'A_obs':>6}  {'null_mean':>9}  "
        f"{'Δacc':>7}  {'z':>6}  {'p_MC':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in results:
        mc = s["mc_stats"]
        print(
            f"  {s['label']:<40}  "
            f"{mc['observed']:>6.3f}  "
            f"{mc['null_mean']:>9.3f}  "
            f"{mc['delta_acc']:>+7.3f}  "
            f"{mc['z_score']:>6.2f}  "
            f"{mc['p_value']:>7.4f}"
        )

    out = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "n_thoughts": args.n_thoughts,
        "n_instances": len(data),
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "mode": args.mode,
        "n_gpus": args.n_gpus,
        "codi_batch_size": args.codi_batch_size,
        "baseline_accuracy": A_obs,
        "baseline_ci": {"point": baseline_ci.point,
                        "ci_low": baseline_ci.ci_low,
                        "ci_high": baseline_ci.ci_high,
                        "ci_level": baseline_ci.ci_level,
                        "n": baseline_ci.n, "n_boot": baseline_ci.n_boot},
        "results": results,
    }
    path = output_dir / f"hypothesis_test_results_{args.mode}.json"
    with open(path, "w") as f:
        json.dump(deep_convert(out), f, indent=2)
    print(f"\n  Saved → {path}")
    print(f"  Bootstrap CIs (JSONL) → {cis_jsonl}")


if __name__ == "__main__":
    main()