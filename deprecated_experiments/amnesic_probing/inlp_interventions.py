"""
Fast Interventions: batched direction-agnostic steering.

Drop-in replacement for the steering sweep in interventions.py.

Speedup sources (per instance):

  1. Prefix forward pass is shared across alphas (coconut / codi paths).
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

CODI notes:
  - Hidden states are collected PRE-projection at t=0..K (same indexing
    as `thoughts` tensor and INLP projectors).
  - Steering acts on the pre-projection hidden state; prj(h_steered)
    is what gets fed back. This matches random_corruption.py.
  - The model and projection run in bfloat16; concept projectors live
    in float32. The steer closure casts internally so behaviour matches
    the per-alpha reference path.
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )

# CODI generates CoT reasoning before the answer on GSM8k, which can exceed
# 128 tokens.  256 matches remove_thoughts.py / the original CODI test.py.
# Coconut/Pause answers are short, so 256 is harmless overhead there.
MAX_DECODE_TOKENS = 256
from src.config import BASE_DIR, THOUGHTS
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    run_intervened_inference_pauseaware,
    is_pause_model,
    load_data,
    deep_convert,
    normalize_text_for_flip,
    make_projection_intervention,
    run_eval_with_intervention,
    run_codi_single_alpha,
    run_codi_eval_with_intervention,
    run_codi_baseline,
    compute_alpha_regimes,
    print_alpha_regimes,
    _shard_indices,
    _merge_shards,
    REGIME_GENUINE_MAX,
    REGIME_MAGNITUDE_MIN,
)
def make_concept_steering_intervention(concept_projectors, alpha, device):
    """
    Per-instance concept steering.

    For each thought vector h_t at timestep t:
        # concept component: c_t = C_t @ h_t   where C_t = I - P_t
        # direction: d_t = c_t / ||c_t||
        # steered: h'_t = h_t + alpha * d_t

    This pushes h_t further along its own concept direction.
    No target label required.
    """
    C_tensors = {t: torch.tensor(C, dtype=torch.float32, device=device)
                 for t, C in concept_projectors.items()}

    def intervention_fn(h, t):
        if t not in C_tensors:
            return h
        # c_t = C_t @ h_t  (project onto concept subspace)
        c = C_tensors[t] @ h
        norm = c.norm()
        if norm < 1e-8:
            return h
        # d_t = c_t / ||c_t||
        d = c / norm
        # h'_t = h_t + alpha * d_t
        return h + alpha * d
    return intervention_fn

# CODI reference path (per-alpha) uses run_codi_single_alpha from utils.


# ═══════════════════════════════════════════════════════════════════
# Batched concept-steering closure
# ═══════════════════════════════════════════════════════════════════

def _build_alpha_steer_fn(C_tensors, alpha_vec):
    """
    Build a batched concept-steering intervention closure.

    Shapes at timestep t:
        h:         (B, D)           B = n_alphas
        C_t:       (D, D)           stored in float32
        alpha_vec: (B, 1)           float32

    Math (applied row-wise across the batch):
        # c    = h @ C_t^T        projection onto concept subspace
        # norm = ||c||_2          per-row L2 norm, shape (B, 1)
        # mask = norm > 1e-8      rows with vanishing concept component pass through
        # d    = c / norm         unit direction
        # h'   = h + alpha * d    where alpha broadcasts per-row

    Dtype contract: the incoming h may be any float dtype (bfloat16 for
    CODI, float32 otherwise). We upcast to float32 for the projection
    math, then cast the result back to h's original dtype before
    returning. This guarantees the downstream forward pass receives
    the same dtype it would have in the unsteered reference path.
    """
    def steer_fn(h, t):
        if t not in C_tensors:
            return h
        C = C_tensors[t]
        orig_dtype = h.dtype

        # Upcast for the projection arithmetic
        h_f32 = h.to(torch.float32)
        # c = h @ C^T  ->  (B, D)
        c = h_f32 @ C.T
        # norm -> (B, 1), keepdim for broadcasting
        norm = c.norm(dim=-1, keepdim=True)
        # Prevent div-by-zero: rows with tiny norm get d=0 (pass-through)
        safe_norm = torch.where(norm < 1e-8, torch.ones_like(norm), norm)
        d = c / safe_norm
        d = torch.where(norm < 1e-8, torch.zeros_like(d), d)
        # h' = h + alpha * d, alpha shape (B, 1) broadcasts over D
        out = h_f32 + alpha_vec * d
        return out.to(orig_dtype)
    return steer_fn


# ═══════════════════════════════════════════════════════════════════
# Coconut: batched steering sweep (all alphas in one forward pass)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _steering_sweep_coconut_batched(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas,
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

    for _ in range(MAX_DECODE_TOKENS):
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


# ═══════════════════════════════════════════════════════════════════
# Pause: batched steering sweep (all alphas in one forward pass)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _steering_sweep_pause_batched(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas, start_id, latent_id,
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

    for _ in range(MAX_DECODE_TOKENS):
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


# ═══════════════════════════════════════════════════════════════════
# CODI: batched steering sweep + single-alpha reference (for sanity)
# ═══════════════════════════════════════════════════════════════════
#
# CODI inference differs from coconut in three ways that matter here:
#
#   1. Delimiters: [bot] before the thought region, [eot] after it,
#      instead of <start_latent> / <end_latent>. The prompt is
#         [question_tokens]  ([eos]?)  [bot]
#      and the post-thought delimiter is
#         [eot]  ([eos]?)
#      where the optional [eos] tokens depend on codi_dict['remove_eos'].
#
#   2. Projection: between recurrence steps, the steered hidden state
#      is passed through prj(·) before being fed back as inputs_embeds.
#      Steering acts on the pre-projection state, matching the
#      hidden-state cache used by INLP.
#
#   3. Decode: greedy-decode is done by feeding embeddings, and logits
#      are clipped to [:vocab_size - 1] to exclude [eot] from generation,
#      matching CODI's test.py --greedy True path.

def _codi_build_prompt_ids(codi_dict, sample, device, n_alphas):
    """Build the [question] ([eos]?) [bot] prompt batched along n_alphas.

    Returns: input_ids tensor of shape (n_alphas, L_prompt)
    """
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    remove_eos = codi_dict['remove_eos']

    question_tokens = tokenizer.encode(sample["question"].strip().replace('  ', ' '),
                                       add_special_tokens=True)
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    return torch.tensor([ids] * n_alphas, device=device)


@torch.no_grad()
def _steering_sweep_codi_batched(
    codi_dict, sample, n_thoughts, device, C_tensors, alphas,
):
    """
    CODI: one forward pass per thought step for all alphas at once.

    Pipeline per thought step t (batched across alphas):
        # h_t = last_hidden[:, -1, :]                       (B, D)
        # h_t' = steer(h_t, t)                              (B, D)   # pre-prj
        # feed_t = prj(h_t')  if use_prj else h_t'          (B, D)
        # next_step input_embeds = feed_t.unsqueeze(1)      (B, 1, D)

    Returns: dict {alpha: decoded_text}
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']

    n_alphas = len(alphas)
    alpha_vec = torch.tensor(alphas, dtype=torch.float32,
                             device=device).view(n_alphas, 1)
    steer_fn = _build_alpha_steer_fn(C_tensors, alpha_vec)

    # ── Step 0: encode prompt [question] ([eos]?) [bot] ─────────────
    input_ids = _codi_build_prompt_ids(codi_dict, sample, device, n_alphas)
    attention_mask = torch.ones_like(input_ids)           # (n_alphas, L)
    # No padding: all rows have identical content, so position_ids = arange.
    # pos_ids[b, j] = j,  real_len = L for all rows.
    L = input_ids.size(1)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(n_alphas, -1)

    outputs = base_model(
        input_ids=input_ids, use_cache=True, output_hidden_states=True,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][:, -1, :]     # (n_alphas, D)

    # Steer at t=0 (pre-projection)
    h = steer_fn(h, 0)
    latent = h.unsqueeze(1)                     # (n_alphas, 1, D)
    if use_prj and prj is not None:
        latent = prj(latent)

    # ── Steps 1..K: recurrence ──────────────────────────────────────
    # running_mask grows by 1 each step; position at step t = L + t - 1
    running_mask = attention_mask                         # (n_alphas, L)
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((n_alphas, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        # pos_t = L + t - 1  (same for all rows — no padding)
        pos_t = torch.full((n_alphas, 1), L + t - 1,
                           dtype=torch.long, device=device)

        outputs = base_model(
            inputs_embeds=latent, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
            attention_mask=running_mask,
            position_ids=pos_t,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][:, -1, :]  # (n_alphas, D)
        h = steer_fn(h, t)
        latent = h.unsqueeze(1)
        if use_prj and prj is not None:
            latent = prj(latent)

    # ── Feed [eot] ([eos]?) delimiter ───────────────────────────────
    if remove_eos:
        eot_ids_row = [eot_id]
    else:
        eot_ids_row = [eot_id, tokenizer.eos_token_id]
    eot_ids = torch.tensor([eot_ids_row] * n_alphas, device=device)
    eot_emb = embedding_fn(eot_ids)
    eot_len = eot_emb.size(1)                             # 1 or 2

    # eot positions: L + K, ..., L + K + eot_len - 1
    eot_pos = torch.arange(L + n_thoughts, L + n_thoughts + eot_len,
                           device=device).unsqueeze(0).expand(n_alphas, -1)
    running_mask = torch.cat(
        [running_mask, torch.ones((n_alphas, eot_len), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )

    outputs = base_model(
        inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
        attention_mask=running_mask,
        position_ids=eot_pos,
    )
    past_kv = outputs.past_key_values

    # Track the next decode position (same for all rows)
    current_pos = L + n_thoughts + eot_len

    # ── Batched greedy decode (exclude eot via vocab_size-1 clip) ──
    # Match CODI test.py --greedy True: logits[:, -1, :vocab_size - 1].
    vocab_size = base_model.config.vocab_size
    next_logits = outputs.logits[:, -1, :vocab_size - 1]  # (n_alphas, V-1)

    generated_ids = [[] for _ in range(n_alphas)]
    finished = [False] * n_alphas

    for _ in range(MAX_DECODE_TOKENS):
        next_tokens = next_logits.argmax(dim=-1)    # (n_alphas,)
        for b in range(n_alphas):
            if not finished[b]:
                if next_tokens[b].item() == tokenizer.eos_token_id:
                    finished[b] = True
                else:
                    generated_ids[b].append(next_tokens[b].item())
        if all(finished):
            break
        # CODI decodes by feeding embeddings, not input_ids
        next_emb = embedding_fn(next_tokens).unsqueeze(1)   # (n_alphas, 1, D)
        running_mask = torch.cat(
            [running_mask, torch.ones((n_alphas, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        decode_pos = torch.full((n_alphas, 1), current_pos,
                                dtype=torch.long, device=device)
        outputs = base_model(
            inputs_embeds=next_emb,
            past_key_values=past_kv,
            use_cache=True,
            attention_mask=running_mask,
            position_ids=decode_pos,
        )
        next_logits = outputs.logits[:, -1, :vocab_size - 1]
        past_kv = outputs.past_key_values
        current_pos += 1

    return {
        alphas[b]: tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        for b in range(n_alphas)
    }


# ═══════════════════════════════════════════════════════════════════
# Dispatch: batched one-instance sweep
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def steering_sweep_one_instance(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, C_tensors, alphas,
    start_id=None, latent_id=None, codi_dict=None,
):
    """
    Dispatch to the correct batched steering path:
      - codi_dict is not None  ->  CODI
      - coconut_model is pause ->  pause
      - otherwise               ->  coconut
    """
    if codi_dict is not None:
        return _steering_sweep_codi_batched(
            codi_dict, sample, n_thoughts, device, C_tensors, alphas,
        )
    if is_pause_model(coconut_model):
        return _steering_sweep_pause_batched(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, C_tensors, alphas,
            start_id, latent_id,
        )
    return _steering_sweep_coconut_batched(
        base_model, tokenizer, end_id, sample,
        n_thoughts, device, C_tensors, alphas,
    )


# ═══════════════════════════════════════════════════════════════════
# Batched sweep + flip-rate aggregation
# ═══════════════════════════════════════════════════════════════════

def run_steering_sweep_batched(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, concept_projectors, alphas,
    baseline_texts, label_name,
    start_id=None, latent_id=None,
    codi_dict=None,
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
    # Materialize C tensors ONCE (was per-instance-per-alpha before).
    # Kept in float32 regardless of model dtype; the steer closure
    # upcasts h to float32 for the projection arithmetic.
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
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
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
    start_id=None, latent_id=None, codi_dict=None,
    n_check=5, verbose=True, task="prosqa",
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
    (benign) or logical (a real bug). CODI runs in bfloat16, so a tiny
    number of benign mismatches is more likely there than for float32
    coconut/pause.
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
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
        )

        # --- original (per-alpha loop, unbatched) ---
        original = {}
        for alpha in alphas:
            steer_fn = make_concept_steering_intervention(
                concept_projectors, alpha, device,
            )
            if codi_dict is not None:
                r = run_codi_single_alpha(
                    codi_dict, sample, n_thoughts, device, steer_fn,
                    task=task,
                )
            else:
                r = run_intervened_inference_pauseaware(
                    coconut_model, base_model, tokenizer, end_id, sample,
                    n_thoughts, device, steer_fn,
                    start_id=start_id, latent_id=latent_id,
                    task=task,
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

    Dispatches on model_name == 'codi' to load via setup_codi_model;
    otherwise uses setup_model_and_tokenizer.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    is_codi = (model_name == "codi")
    if is_codi:
        codi_dict = setup_codi_model(task, device)
        coconut_model = base_model = tokenizer = None
        latent_id = start_id = end_id = None
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device)

    indices = _shard_indices(len(data), world_size, rank)
    print(f"[rank {rank}] processing {len(indices)} instances on {device}")

    inlp_part = run_steering_sweep_batched(
        coconut_model, base_model, tokenizer, end_id, data,
        n_thoughts, device, concept_projectors_inlp_np, alphas,
        baseline_texts, label_name=f"INLP r{rank}",
        start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
        instance_indices=indices,
    )
    rand_part = run_steering_sweep_batched(
        coconut_model, base_model, tokenizer, end_id, data,
        n_thoughts, device, concept_projectors_rand_np, alphas,
        baseline_texts, label_name=f"Rand r{rank}",
        start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
        instance_indices=indices,
    )
    return_queue.put({"rank": rank, "inlp": inlp_part, "rand": rand_part})


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

    is_codi = (args.model == "codi")

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data + INLP ────────────────────────────────────────────
    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Task: {args.task}, Model: {args.model}, "
          f"instances: {len(data)}")

    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        print(f"  [INFO] Extract thoughts first...")

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
    eye = np.eye(D)
    concept_projectors_inlp = {t: eye - P for t, P in projections.items()}
    concept_projectors_rand = {t: eye - P for t, P in rand_projections.items()}

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # ── Regime diagnostic (model/task-specific) ─────────────────────
    regime_info = compute_alpha_regimes(thoughts, alphas)
    print_alpha_regimes(regime_info, alphas)
    alpha_to_regime = {
        r["alpha"]: r["regime_pooled"]
        for r in regime_info["regimes_per_alpha"]
    }

    # ── Load model (single-GPU) ─────────────────────────────────────
    # Always loaded on main process for baseline + ablation + (optional)
    # single-GPU steering; multi-GPU workers load their own replicas.
    if is_codi:
        codi_dict = setup_codi_model(args.task, args.device)
        coconut_model = base_model = tokenizer = None
        latent_id = start_id = end_id = None
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device)

    # ── Sanity-check path (single GPU, no multiprocessing) ─────────
    if args.sanity_check:
        print("\n" + "=" * 60)
        print("SANITY CHECK: batched vs original")
        print("=" * 60)

        print("\n  [INLP concept projectors]")
        res_inlp = sanity_check(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_inlp, alphas,
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
            n_check=args.sanity_n, task=args.task,
        )
        print("\n  [Rand control projectors]")
        res_rand = sanity_check(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_rand, alphas,
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
            n_check=args.sanity_n, task=args.task,
        )
        total_mis = res_inlp["n_mismatch"] + res_rand["n_mismatch"]
        if total_mis == 0:
            print("\n[SANITY] ALL CHECKS PASSED.")
        else:
            print(f"\n[SANITY] {total_mis} mismatches found — do not trust batched results.")
        return

    # ── Baseline ───────────────────────────────────────────────────
    # Cheap (1 pass/instance, no alpha sweep). Run single-GPU so the
    # multi-GPU logic stays focused on the expensive steering sweep.
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    if is_codi:
        baseline_acc, baseline_texts = run_codi_baseline(
            codi_dict, data, args.n_thoughts, args.device,
            task=args.task,
        )
    else:
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
    # drop. Per-instance, one pass per method — no alpha axis to batch.
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("INLP ABLATION")
    print("=" * 60)

    if is_codi:
        print("  Running INLP ablation...")
        inlp_fn = make_projection_intervention(projections, args.device)
        inlp_acc = run_codi_eval_with_intervention(
            codi_dict, data, args.n_thoughts, args.device, inlp_fn,
            label="INLP", task=args.task,
        )

        print("  Running Rand control...")
        rand_fn = make_projection_intervention(rand_projections, args.device)
        rand_acc = run_codi_eval_with_intervention(
            codi_dict, data, args.n_thoughts, args.device, rand_fn,
            label="Rand", task=args.task,
        )
    else:
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
        if is_codi:
            del codi_dict
        else:
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
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
        )
        print("\n  Steering along random control directions:")
        rand_steering = run_steering_sweep_batched(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, concept_projectors_rand, alphas,
            baseline_texts, label_name="Rand",
            start_id=start_id, latent_id=latent_id, codi_dict=codi_dict,
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