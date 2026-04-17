"""
Fast Interventions: batched direction-agnostic steering.

Drop-in replacement for the steering sweep in interventions.py.

Speedup sources (per instance):

  1. Prefix forward pass is shared across alphas (coconut path).
  2. All alphas are stacked along the batch dimension — one forward pass
     per thought step / decode step instead of one per alpha.
  3. Concept projectors are materialized once, outside all loops.
  4. Optional multi-GPU sharding via torch.multiprocessing.spawn:
     instances are split across ranks; each rank has its own model
     replica; flip counts are aggregated at the end.

Correctness is validated by --sanity_check, which runs both the original
per-alpha path and the batched path on a small slice and asserts the
decoded texts match for every (instance, alpha) pair.

Intervention math (unchanged from interventions.py):

    # concept component: c_t = C_t @ h_t,  where C_t = I - P_t
    # direction:         d_t = c_t / ||c_t||
    # steered vector:    h'_t = h_t + alpha * d_t

In the batched path, h_t has shape (n_alphas, D) because prior-step
steering makes each row's trajectory diverge. C is applied row-wise:

    # c = h @ C^T          shape (n_alphas, D)
    # norm = ||c||_2       shape (n_alphas, 1)
    # d = c / norm         shape (n_alphas, D)
    # h' = h + alpha * d   alpha has shape (n_alphas, 1)
"""

import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, THOUGHTS
from src.utils import (
    setup_model_and_tokenizer,
    run_intervened_inference_pauseaware,
    is_pause_model,
    _compare_answers,
)

# Reuse helpers that already exist in interventions.py — no duplication.
from experiments.amnesic_probing.interventions import (
    load_data,
    normalize_text_for_flip,
    deep_convert,
    make_concept_steering_intervention,
    make_projection_intervention,
    run_eval_with_intervention,
    run_steering_sweep as run_steering_sweep_original,
)


# ═══════════════════════════════════════════════════════════════════
# Steering regime thresholds (see compute_alpha_regimes)
# ═══════════════════════════════════════════════════════════════════
#
# The steering perturbation is  h' = h + alpha * d,  where d is a unit
# vector along h's own concept-subspace projection. The relative
# perturbation magnitude is therefore
#
#     # r = ||alpha * d|| / ||h|| = alpha / ||h||
#
# r partitions the sweep into three regimes:
#
#   - GENUINE  (r <= REGIME_GENUINE_MAX):
#       Nudge is small relative to h; a flip here is evidence that the
#       concept direction is causally used by the model.
#
#   - TRANSITION (REGIME_GENUINE_MAX < r <= REGIME_MAGNITUDE_MIN):
#       Perturbation is comparable to h. Model is OOD in norm-space;
#       flips here are ambiguous between direction-specific causality
#       and generic OOD fragility.
#
#   - MAGNITUDE (r > REGIME_MAGNITUDE_MIN):
#       h' is dominated by alpha * d. Original h is rounding error.
#       A flip here is not evidence of direction causality — it's
#       evidence that any large vector in the concept subspace
#       (or a random subspace) breaks the forward pass. Equivalent in
#       interpretation to the random-corruption control: if INLP and
#       Rand flip rates match here, that's expected, not informative.
#
# Values are defensible eyeballed cutoffs. Report exact r values
# alongside regime labels in the paper; the labels are a reading aid.

REGIME_GENUINE_MAX = 0.1
REGIME_MAGNITUDE_MIN = 5.0


def _regime_label(ratio):
    """Map ||alpha*d|| / ||h|| to a three-way regime label."""
    if ratio <= REGIME_GENUINE_MAX:
        return "GENUINE"
    if ratio <= REGIME_MAGNITUDE_MIN:
        return "TRANSITION"
    return "MAGNITUDE"


def compute_alpha_regimes(thoughts, alphas):
    """
    For each alpha in the sweep, report what steering regime it falls
    into given the empirical distribution of thought-vector norms.

    Args:
        thoughts: tensor of shape (N, T, D) — cached hidden states at
            thought positions across N instances and T timesteps.
        alphas: list of float alpha values.

    Returns:
        dict with:
          - "norms_per_t": list[T] of dicts {median, p10, p90} of ||h_t||
          - "median_pooled": float, median over all (n, t)
          - "regimes_per_alpha": list, one entry per alpha:
                { "alpha": float,
                  "ratio_pooled": float,
                  "regime_pooled": str,
                  "ratio_per_t": list[T] of float,
                  "regime_per_t": list[T] of str }

    Math:
        # h_norms[n, t] = ||thoughts[n, t, :]||_2
        # median_t = median_n h_norms[n, t]
        # ratio[alpha, t] = alpha / median_t
    """
    # h_norms: (N, T)
    h_norms = thoughts.float().norm(dim=-1)
    T = h_norms.shape[1]

    median_per_t = h_norms.median(dim=0).values  # (T,)
    p10_per_t = h_norms.quantile(0.1, dim=0)     # (T,)
    p90_per_t = h_norms.quantile(0.9, dim=0)     # (T,)
    median_pooled = h_norms.median().item()

    norms_per_t = [
        {"median": median_per_t[t].item(),
         "p10": p10_per_t[t].item(),
         "p90": p90_per_t[t].item()}
        for t in range(T)
    ]

    regimes_per_alpha = []
    for alpha in alphas:
        ratio_pooled = alpha / median_pooled
        ratio_per_t = [alpha / median_per_t[t].item() for t in range(T)]
        regime_per_t = [_regime_label(r) for r in ratio_per_t]
        regimes_per_alpha.append({
            "alpha": alpha,
            "ratio_pooled": ratio_pooled,
            "regime_pooled": _regime_label(ratio_pooled),
            "ratio_per_t": ratio_per_t,
            "regime_per_t": regime_per_t,
        })

    return {
        "norms_per_t": norms_per_t,
        "median_pooled": median_pooled,
        "regimes_per_alpha": regimes_per_alpha,
    }


def print_alpha_regimes(regime_info, alphas):
    """Print the regime diagnostic table. Call this before the sweep runs."""
    print("\n" + "=" * 70)
    print("STEERING REGIME DIAGNOSTIC")
    print("=" * 70)
    print(f"  Model/task-specific regime boundaries depend on ||h||.")
    print(f"  Thresholds: r <= {REGIME_GENUINE_MAX:g}  -> GENUINE")
    print(f"              r <= {REGIME_MAGNITUDE_MIN:g}  -> TRANSITION")
    print(f"              r >  {REGIME_MAGNITUDE_MIN:g}  -> MAGNITUDE (corruption-equivalent)")
    print(f"    where r = alpha / median(||h_t||).")
    print()

    print(f"  Thought-vector norm ||h_t|| per timestep (median [p10, p90]):")
    for t, n in enumerate(regime_info["norms_per_t"]):
        print(f"    t={t}:  {n['median']:7.3f}   "
              f"[{n['p10']:7.3f}, {n['p90']:7.3f}]")
    print(f"    pooled median ||h|| = {regime_info['median_pooled']:.3f}")
    print()

    # Per-alpha regime (pooled across t for quick read, then per-t detail)
    print(f"  {'alpha':>10}  {'r (pooled)':>12}  {'regime (pooled)':>18}  "
          f"regime by t [t=0..T-1]")
    print(f"  {'-'*10}  {'-'*12}  {'-'*18}  {'-'*40}")
    for r in regime_info["regimes_per_alpha"]:
        per_t_str = " ".join(lab[0] for lab in r["regime_per_t"])
        # Legend: G=GENUINE, T=TRANSITION, M=MAGNITUDE
        print(f"  {r['alpha']:>10g}  {r['ratio_pooled']:>12.3f}  "
              f"{r['regime_pooled']:>18}  {per_t_str}")
    print(f"  (per-t legend: G=GENUINE, T=TRANSITION, M=MAGNITUDE)")
    print()


# ═══════════════════════════════════════════════════════════════════
# Batched concept-steering inference
# ═══════════════════════════════════════════════════════════════════

def _build_alpha_steer_fn(C_tensors, alpha_vec):
    """
    Build a batched concept-steering intervention closure.

    Shapes at timestep t:
        h:         (B, D)           B = n_alphas
        C_t:       (D, D)
        alpha_vec: (B, 1)

    Math (applied row-wise across the batch):
        # c    = h @ C_t^T        projection onto concept subspace
        # norm = ||c||_2          per-row L2 norm, shape (B, 1)
        # mask = norm > 1e-8      rows with vanishing concept component pass through
        # d    = c / norm         unit direction
        # h'   = h + alpha * d    where alpha broadcasts per-row
    """
    def steer_fn(h, t):
        if t not in C_tensors:
            return h
        C = C_tensors[t]
        # c = h @ C^T  ->  (B, D)
        c = h @ C.T
        # norm -> (B, 1), keepdim for broadcasting
        norm = c.norm(dim=-1, keepdim=True)
        # Prevent div-by-zero: rows with tiny norm get d=0 (pass-through)
        safe_norm = torch.where(norm < 1e-8, torch.ones_like(norm), norm)
        d = c / safe_norm
        d = torch.where(norm < 1e-8, torch.zeros_like(d), d)
        # h' = h + alpha * d, alpha shape (B, 1) broadcasts over D
        return h + alpha_vec * d
    return steer_fn


@torch.no_grad()
def _steering_sweep_coconut_batched(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas, task,
):
    """
    Coconut: one forward pass per thought step for all alphas at once.

    Pattern mirrors utils._alpha_sweep_coconut: prompt tokens are repeated
    n_alphas times along batch dim so the KV cache is shared-structurally
    but row-independent once interventions start diverging the rows.

    Returns: dict {alpha: decoded_text}
    """
    n_alphas = len(alphas)
    alpha_vec = torch.tensor(alphas, dtype=torch.float32,
                             device=device).view(n_alphas, 1)
    steer_fn = _build_alpha_steer_fn(C_tensors, alpha_vec)

    # Prompt forward (shared content, batched so KV has batch dim = n_alphas)
    input_ids = tokenizer.encode(
        sample["question"] + " <|start-latent|>", return_tensors="pt"
    ).to(device)
    input_ids = input_ids.repeat(n_alphas, 1)

    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    # h at <start_latent> position, per row
    h = outputs.hidden_states[-1][:, -1, :]     # (n_alphas, D)
    past_kv = outputs.past_key_values

    # t = 0: intervene at start_latent position
    h = steer_fn(h, 0)
    ct = h.unsqueeze(1)                         # (n_alphas, 1, D)

    # Thought recurrence, t = 1..K
    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=ct,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][:, 0, :]  # (n_alphas, D)
        h = steer_fn(h, t)
        ct = h.unsqueeze(1)
        past_kv = outputs.past_key_values

    # Feed <end_latent>
    end_input = torch.tensor([[end_id]] * n_alphas, device=device)
    outputs = base_model(
        input_ids=end_input, past_key_values=past_kv, use_cache=True,
    )
    past_kv = outputs.past_key_values

    # Batched greedy decode
    next_logits = outputs.logits[:, -1, :]      # (n_alphas, V)
    generated_ids = [[] for _ in range(n_alphas)]
    finished = [False] * n_alphas

    for _ in range(128):
        next_tokens = next_logits.argmax(dim=-1)   # (n_alphas,)
        for b in range(n_alphas):
            if not finished[b]:
                if next_tokens[b].item() == tokenizer.eos_token_id:
                    finished[b] = True
                else:
                    generated_ids[b].append(next_tokens[b].item())
        if all(finished):
            break
        outputs = base_model(
            input_ids=next_tokens.unsqueeze(1),
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

    return {
        alphas[b]: tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        for b in range(n_alphas)
    }


@torch.no_grad()
def _steering_sweep_pause_batched(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas, start_id, latent_id, task,
):
    """
    Pause model: single forward pass per alpha, but all alphas fused into
    the batch dimension. One forward pass + batched decode replaces the
    entire inner alpha loop.

    Embedding setup per row (identical across rows initially):
        [question_tokens] <start_latent> <pause>*K <end_latent>
    with <pause> positions replaced by the learned pause_embedding.

    Then for each row b and each thought position (including start_latent),
    apply the concept-steering intervention with that row's alpha. The
    forward pass consumes the modified embeddings, so steering propagates
    through every transformer layer — unlike a hook on layer L output,
    which nothing downstream reads (see _intervened_inference_pause).

    Returns: dict {alpha: decoded_text}
    """
    n_alphas = len(alphas)
    alpha_vec = torch.tensor(alphas, dtype=torch.float32,
                             device=device).view(n_alphas, 1)
    steer_fn = _build_alpha_steer_fn(C_tensors, alpha_vec)

    question_text = sample["question"]
    question_tokens = tokenizer.encode(question_text + "\n",
                                       add_special_tokens=True)
    input_ids_list = (
        question_tokens + [start_id] + [latent_id] * n_thoughts + [end_id]
    )
    # Repeat along batch so each row has its own copy to mutate
    input_ids = torch.tensor([input_ids_list] * n_alphas, device=device)
    L = input_ids.shape[1]

    # Initial embeddings, then substitute pause_embedding at thought positions
    embedding = coconut_model.embedding
    inputs_embeds = embedding(input_ids).clone()           # (n_alphas, L, D)
    pause_emb = coconut_model.pause_embedding               # (D,)

    start_of_latent = len(question_tokens) + 1              # first <latent>
    start_latent_pos = len(question_tokens)                 # <start_latent>

    for i in range(n_thoughts):
        pos = start_of_latent + i
        inputs_embeds[:, pos, :] = pause_emb

    # Thought positions: t=0 is <start_latent>, t=1..K are <latent> slots
    thought_positions = [start_latent_pos] + [
        start_of_latent + i for i in range(n_thoughts)
    ]

    # Apply steering at each thought position across all alphas at once.
    # h has shape (n_alphas, D); steer_fn returns (n_alphas, D) because
    # alpha_vec broadcasts row-wise.
    for t_idx, pos in enumerate(thought_positions):
        h = inputs_embeds[:, pos, :]            # (n_alphas, D)
        inputs_embeds[:, pos, :] = steer_fn(h, t_idx)

    # Single batched forward pass
    attention_mask = torch.ones((n_alphas, L), device=device)
    position_ids = (
        torch.arange(L, device=device).unsqueeze(0).expand(n_alphas, -1)
    )
    outputs = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]      # (n_alphas, V)

    # Batched greedy decode
    generated_ids = [[] for _ in range(n_alphas)]
    finished = [False] * n_alphas

    for _ in range(128):
        next_tokens = next_logits.argmax(dim=-1)
        for b in range(n_alphas):
            if not finished[b]:
                if next_tokens[b].item() == tokenizer.eos_token_id:
                    finished[b] = True
                else:
                    generated_ids[b].append(next_tokens[b].item())
        if all(finished):
            break
        outputs = base_model(
            input_ids=next_tokens.unsqueeze(1),
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

    return {
        alphas[b]: tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        for b in range(n_alphas)
    }


@torch.no_grad()
def steering_sweep_one_instance(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas,
    start_id=None, latent_id=None, task="prosqa",
):
    """Dispatch: pause vs coconut."""
    if is_pause_model(coconut_model):
        return _steering_sweep_pause_batched(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, C_tensors, alphas,
            start_id, latent_id, task,
        )
    return _steering_sweep_coconut_batched(
        base_model, tokenizer, end_id, sample,
        n_thoughts, device, C_tensors, alphas, task,
    )


# ═══════════════════════════════════════════════════════════════════
# Batched sweep + flip-rate aggregation
# ═══════════════════════════════════════════════════════════════════

def run_steering_sweep_batched(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, concept_projectors, alphas,
    baseline_texts, label_name,
    start_id=None, latent_id=None, task="prosqa",
    instance_indices=None,
):
    """
    Batched equivalent of interventions.run_steering_sweep.

    concept_projectors: dict {t: C_t as np.ndarray} where C_t = I - P_t

    instance_indices: optional list of ints — which rows of `data` to
      process (for multi-GPU sharding). If None, process all.

    Returns: {alpha: {"n_flipped": int, "n_total": int, "flip_rate": float,
                      "flipped_indices": [int, ...]}}

    flipped_indices is included to make aggregation across GPU shards
    unambiguous (you can union indices instead of summing counts, which
    lets the sanity check compare results shard-wise too).
    """
    # Materialize C tensors ONCE (was per-instance-per-alpha before)
    C_tensors = {
        t: torch.tensor(C, dtype=torch.float32, device=device)
        for t, C in concept_projectors.items()
    }

    if instance_indices is None:
        instance_indices = list(range(len(data)))

    n_total = len(instance_indices)
    flipped_indices = {alpha: [] for alpha in alphas}

    for count, idx in enumerate(instance_indices):
        if count % 100 == 0:
            print(f"    [{label_name}] {count}/{n_total}")

        sample = data[idx]
        base_text = baseline_texts[idx]
        base_norm = normalize_text_for_flip(base_text)

        alpha_to_text = steering_sweep_one_instance(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, C_tensors, alphas,
            start_id=start_id, latent_id=latent_id, task=task,
        )

        for alpha in alphas:
            if normalize_text_for_flip(alpha_to_text[alpha]) != base_norm:
                flipped_indices[alpha].append(idx)

    results = {}
    for alpha in alphas:
        n_flipped = len(flipped_indices[alpha])
        results[alpha] = {
            "n_flipped": n_flipped,
            "n_total": n_total,
            "flip_rate": n_flipped / max(n_total, 1),
            "flipped_indices": flipped_indices[alpha],
        }
        print(f"    [{label_name}] Alpha {alpha}: "
              f"{n_flipped}/{n_total} flipped ({results[alpha]['flip_rate']:.1%})")
    return results


# ═══════════════════════════════════════════════════════════════════
# Sanity check: batched vs original must produce identical decoded text
# ═══════════════════════════════════════════════════════════════════

def sanity_check(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, concept_projectors, alphas,
    start_id=None, latent_id=None, task="prosqa",
    n_check=5, verbose=True,
):
    """
    Verify that the batched steering produces the same decoded text as
    the original per-alpha path, for every (instance, alpha) pair.

    This is the most important correctness guarantee in this module.
    If this passes, the batched sweep's flip counts must match the
    original's exactly.

    Checks performed:
      (1) For each instance in data[:n_check], for each alpha:
            batched_text == original_text   (exact string equality)
      (2) Sum of mismatches across all (instance, alpha) pairs must be 0.

    Note on determinism: greedy argmax decoding is deterministic on the
    same device at the same dtype. Batching changes the matmul shapes,
    which on some GPUs can introduce ~ULP-scale differences in logits
    that *may* flip an argmax at ties. If this check fails on a handful
    of tokens, rerunning with `torch.use_deterministic_algorithms(True)`
    and float64 projectors will confirm whether the cause is numeric
    (benign) or logical (a real bug).
    """
    C_tensors = {
        t: torch.tensor(C, dtype=torch.float32, device=device)
        for t, C in concept_projectors.items()
    }
    n_check = min(n_check, len(data))
    n_mismatch = 0
    mismatches = []

    if verbose:
        print(f"  [SANITY] Checking {n_check} instances × {len(alphas)} alphas "
              f"= {n_check * len(alphas)} (instance, alpha) pairs")

    for idx in range(n_check):
        sample = data[idx]

        # --- batched ---
        batched = steering_sweep_one_instance(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, C_tensors, alphas,
            start_id=start_id, latent_id=latent_id, task=task,
        )

        # --- original (per-alpha loop, unbatched) ---
        original = {}
        for alpha in alphas:
            steer_fn = make_concept_steering_intervention(
                concept_projectors, alpha, device,
            )
            r = run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                n_thoughts, device, steer_fn,
                start_id=start_id, latent_id=latent_id, task=task,
            )
            original[alpha] = r.get("text", r.get("predicted", ""))

        # --- compare ---
        for alpha in alphas:
            if batched[alpha] != original[alpha]:
                n_mismatch += 1
                mismatches.append({
                    "idx": idx,
                    "alpha": alpha,
                    "batched": batched[alpha],
                    "original": original[alpha],
                })

    if verbose:
        total = n_check * len(alphas)
        print(f"  [SANITY] Mismatches: {n_mismatch}/{total}")
        for m in mismatches[:10]:
            print(f"    idx={m['idx']} alpha={m['alpha']}")
            print(f"      batched : {m['batched']!r}")
            print(f"      original: {m['original']!r}")
        if n_mismatch == 0:
            print("  [SANITY] PASS — batched and original produce identical text.")
        else:
            print(f"  [SANITY] FAIL — {n_mismatch} mismatches. Investigate before trusting results.")

    return {"n_mismatch": n_mismatch, "n_total": n_check * len(alphas),
            "mismatches": mismatches}


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU: shard instances across ranks
# ═══════════════════════════════════════════════════════════════════

def _shard_indices(n_items, world_size, rank):
    """Contiguous shard assignment: rank r handles items [r*chunk, (r+1)*chunk)."""
    # chunk = ceil(n_items / world_size); last rank may be shorter
    chunk = (n_items + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_items)
    return list(range(start, end))


def _worker(
    rank, world_size, task, model_name, n_thoughts,
    data, alphas, baseline_texts,
    concept_projectors_inlp_np, concept_projectors_rand_np,
    return_queue,
):
    """
    Per-GPU worker. Loads its own model on cuda:{rank}, processes its
    shard of data, sends partial flip counts back via queue.

    We ship numpy projectors (not torch tensors) over the queue because
    torch tensors across processes need careful sharing semantics; numpy
    is pickle-trivial and small.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
        setup_model_and_tokenizer(task, model_name, device)

    indices = _shard_indices(len(data), world_size, rank)
    print(f"[rank {rank}] processing {len(indices)} instances on {device}")

    inlp_part = run_steering_sweep_batched(
        coconut_model, base_model, tokenizer, end_id, data,
        n_thoughts, device, concept_projectors_inlp_np, alphas,
        baseline_texts, label_name=f"INLP r{rank}",
        start_id=start_id, latent_id=latent_id, task=task,
        instance_indices=indices,
    )
    rand_part = run_steering_sweep_batched(
        coconut_model, base_model, tokenizer, end_id, data,
        n_thoughts, device, concept_projectors_rand_np, alphas,
        baseline_texts, label_name=f"Rand r{rank}",
        start_id=start_id, latent_id=latent_id, task=task,
        instance_indices=indices,
    )
    return_queue.put({"rank": rank, "inlp": inlp_part, "rand": rand_part})


def _merge_shards(shards, alphas, n_total):
    """
    Merge per-rank partial results. flipped_indices are unioned (each
    index appears in exactly one shard), then flip_rate is recomputed
    against the full n_total.
    """
    merged = {}
    for alpha in alphas:
        all_indices = []
        for s in shards:
            all_indices.extend(s[alpha]["flipped_indices"])
        all_indices = sorted(set(all_indices))
        merged[alpha] = {
            "n_flipped": len(all_indices),
            "n_total": n_total,
            "flip_rate": len(all_indices) / max(n_total, 1),
            "flipped_indices": all_indices,
        }
    return merged


def run_multigpu(
    task, model_name, n_thoughts, data, alphas, baseline_texts,
    concept_projectors_inlp, concept_projectors_rand, n_gpus,
):
    """Spawn n_gpus workers, each handling a contiguous shard of data."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_worker,
            args=(rank, n_gpus, task, model_name, n_thoughts,
                  data, alphas, baseline_texts,
                  concept_projectors_inlp, concept_projectors_rand, q),
        )
        p.start()
        procs.append(p)

    shards = []
    for _ in range(n_gpus):
        shards.append(q.get())
    for p in procs:
        p.join()

    shards.sort(key=lambda s: s["rank"])
    inlp_merged = _merge_shards(
        [s["inlp"] for s in shards], alphas, len(data))
    rand_merged = _merge_shards(
        [s["rand"] for s in shards], alphas, len(data))
    return inlp_merged, rand_merged


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fast interventions: batched direction-agnostic steering."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi"],
                        default="coconut")
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--alpha_sweep", type=str,
                        default="0.1,0.5,1,5,10,25,50,100,250,500")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="If >1, shard instances across GPUs 0..n_gpus-1.")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run sanity check (batched vs original) and exit.")
    parser.add_argument("--sanity_n", type=int, default=5,
                        help="Number of instances to check in --sanity_check.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data + INLP ────────────────────────────────────────────
    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Task: {args.task}, Model: {args.model}, "
          f"instances: {len(data)}")

    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        print(f"  [INFO] Triggering thought extraction...")
        cmd = ["python", "-u", "-m",
               "experiments.probe_thoughts.extract_thoughts",
               "--task", args.task, "--model", args.model,
               "--n_thoughts", str(args.n_thoughts)]
        if args.max_instances:
            cmd.extend(["--max_instances", str(args.max_instances)])
        subprocess.run(cmd, check=True)

    thoughts = torch.load(thoughts_path, map_location="cpu",
                          weights_only=False)["thoughts"]

    inlp_path = BASE_DIR / f"outputs/inlp/{args.task}/{args.model}/inlp_results.pt"
    if not inlp_path.exists():
        raise FileNotFoundError(
            f"Run inlp.py first for {args.model}--{args.task}")

    inlp_data = torch.load(inlp_path, map_location="cpu", weights_only=False)
    projections = {int(k) if isinstance(k, str) else k: v
                   for k, v in inlp_data["projections"].items()}
    rand_projections = {int(k) if isinstance(k, str) else k: v
                        for k, v in inlp_data["rand_projections"].items()}

    # C_t = I - P_t  (projects onto concept-encoding subspace)
    D = thoughts.shape[2]
    I = np.eye(D)
    concept_projectors_inlp = {t: I - P for t, P in projections.items()}
    concept_projectors_rand = {t: I - P for t, P in rand_projections.items()}

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # ── Regime diagnostic (model/task-specific) ─────────────────────
    # Compute once from the cached thoughts tensor so we know which
    # alphas correspond to genuine steering vs magnitude attack.
    regime_info = compute_alpha_regimes(thoughts, alphas)
    print_alpha_regimes(regime_info, alphas)
    # Map alpha -> pooled regime label, used in the final summary table
    alpha_to_regime = {
        r["alpha"]: r["regime_pooled"]
        for r in regime_info["regimes_per_alpha"]
    }

    # ── Sanity-check path (single GPU, no multiprocessing) ─────────
    if args.sanity_check:
        print("\n" + "=" * 60)
        print("SANITY CHECK: batched vs original")
        print("=" * 60)
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device)

        print("\n  [INLP concept projectors]")
        res_inlp = sanity_check(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_inlp, alphas,
            start_id=start_id, latent_id=latent_id, task=args.task,
            n_check=args.sanity_n,
        )
        print("\n  [Rand control projectors]")
        res_rand = sanity_check(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_rand, alphas,
            start_id=start_id, latent_id=latent_id, task=args.task,
            n_check=args.sanity_n,
        )
        total_mis = res_inlp["n_mismatch"] + res_rand["n_mismatch"]
        if total_mis == 0:
            print("\n[SANITY] ALL CHECKS PASSED.")
        else:
            print(f"\n[SANITY] {total_mis} mismatches found — do not trust batched results.")
        return

    # ── Baseline (needed for flip comparison) ──────────────────────
    # Baseline is single-GPU: it's cheap (1 pass/instance, no alpha sweep)
    # and lets us keep the multi-GPU logic focused on the expensive part.
    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
        setup_model_and_tokenizer(args.task, args.model, args.device)

    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    identity_fn = lambda h, t: h
    baseline_texts = []
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [Baseline] {idx}/{len(data)}")
        r = run_intervened_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            args.n_thoughts, args.device, identity_fn,
            start_id=start_id, latent_id=latent_id, task=args.task,
        )
        baseline_texts.append(r.get("text", r.get("predicted", "")))
        if r["is_correct"]:
            n_correct += 1
    baseline_acc = n_correct / len(data)
    print(f"  Baseline accuracy: {baseline_acc:.1%}")

    # ════════════════════════════════════════════════════════════════
    # PART 1: INLP ABLATION
    #
    # Apply the INLP nullspace projection (removes label-encoding
    # directions) and a random control projection, measure accuracy
    # drop. This is unchanged from interventions.py — it's per-instance,
    # one pass per method, so batching across alphas doesn't apply.
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("INLP ABLATION")
    print("=" * 60)

    print("  Running INLP ablation...")
    inlp_fn = make_projection_intervention(projections, args.device)
    inlp_acc = run_eval_with_intervention(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, inlp_fn, label="INLP",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    print("  Running Rand control...")
    rand_fn = make_projection_intervention(rand_projections, args.device)
    rand_acc = run_eval_with_intervention(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, rand_fn, label="Rand",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    print(f"\n  {'='*50}")
    print(f"  ABLATION SUMMARY")
    print(f"  {'='*50}")
    print(f"  Baseline:     {baseline_acc:.1%}")
    print(f"  INLP:         {inlp_acc:.1%}  (drop: {baseline_acc - inlp_acc:.1%})")
    print(f"  Rand control: {rand_acc:.1%}  (drop: {baseline_acc - rand_acc:.1%})")

    ablation_results = {
        "task": args.task, "model": args.model,
        "baseline_accuracy": baseline_acc,
        "inlp_accuracy": inlp_acc, "rand_accuracy": rand_acc,
        "accuracy_drop_inlp": baseline_acc - inlp_acc,
        "accuracy_drop_rand": baseline_acc - rand_acc,
    }
    abl_path = output_dir / "ablation_results.json"
    with open(abl_path, "w") as f:
        json.dump(deep_convert(ablation_results), f, indent=2)
    print(f"  Saved to {abl_path}")

    # ── Steering sweep (single- or multi-GPU) ──────────────────────
    print("\n" + "=" * 60)
    print("DIRECTION-AGNOSTIC STEERING (batched)")
    print("=" * 60)

    if args.n_gpus > 1:
        print(f"  Multi-GPU: sharding {len(data)} instances across "
              f"{args.n_gpus} GPUs")
        # Free the single-GPU model before spawning (workers allocate their own)
        del coconut_model, base_model, tokenizer
        torch.cuda.empty_cache()

        inlp_steering, rand_steering = run_multigpu(
            args.task, args.model, args.n_thoughts, data, alphas,
            baseline_texts, concept_projectors_inlp, concept_projectors_rand,
            args.n_gpus,
        )
    else:
        print("\n  Steering along INLP concept directions:")
        inlp_steering = run_steering_sweep_batched(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_inlp, alphas,
            baseline_texts, label_name="INLP",
            start_id=start_id, latent_id=latent_id, task=args.task,
        )
        print("\n  Steering along random control directions:")
        rand_steering = run_steering_sweep_batched(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_rand, alphas,
            baseline_texts, label_name="Rand",
            start_id=start_id, latent_id=latent_id, task=args.task,
        )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  STEERING SUMMARY (direction-agnostic flip rate)")
    print(f"  {'='*72}")
    print(f"  {'Alpha':>8}  {'Regime':>11}  {'INLP Flip':>11}  {'Rand Flip':>11}  Interpretation")
    print(f"  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*30}")
    for alpha in alphas:
        inlp_fr = inlp_steering[alpha]["flip_rate"]
        rand_fr = rand_steering[alpha]["flip_rate"]
        regime = alpha_to_regime[alpha]
        if inlp_fr < 0.02 and rand_fr < 0.02:
            interp = "Robust (causal inertness)"
        elif inlp_fr > 0.05 and rand_fr < 0.02:
            interp = "Concept dirs are causal"
        elif inlp_fr > 0.05 and rand_fr > 0.05:
            interp = "Generally fragile"
        else:
            interp = "Ambiguous"
        print(f"  {alpha:>8g}  {regime:>11}  {inlp_fr:>10.1%}  "
              f"{rand_fr:>10.1%}  {interp}")

    steering_results = {
        "task": args.task, "model": args.model,
        "alphas": alphas,
        "inlp_steering": deep_convert(inlp_steering),
        "rand_steering": deep_convert(rand_steering),
        "regime_info": deep_convert(regime_info),
        "regime_thresholds": {
            "genuine_max": REGIME_GENUINE_MAX,
            "magnitude_min": REGIME_MAGNITUDE_MIN,
        },
        "n_gpus": args.n_gpus,
    }
    steer_path = output_dir / "steering_results_fast.json"
    with open(steer_path, "w") as f:
        json.dump(deep_convert(steering_results), f, indent=2)
    print(f"\n  Saved to {steer_path}")


if __name__ == "__main__":
    main()