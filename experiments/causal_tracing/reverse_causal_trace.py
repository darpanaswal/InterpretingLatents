"""
reverse_causal_trace.py
=======================

Disruptive (reverse) causal tracing for LRMs + PaT control.

Setup
-----
Symmetric to causal_trace.py, but with the patch direction flipped:

    causal_trace.py  (restorative):
        base run        = CORRUPTED (partner prompt)
        per-site patch  = inject CLEAN activation
        measures whether a single clean site can rescue the corrupted run

    reverse_causal_trace.py  (disruptive):
        base run        = CLEAN  (clean prompt)
        per-site patch  = inject PARTNER activation
        measures whether a single corrupted site can derail the clean run

This is the "noising" half of the standard ROME-style causal-trace pair.
A site is causally load-bearing iff corrupting it alone shifts the output
substantially toward the partner's distribution. The natural metric is

    IE_disrupt(L, c, p) = (avg KL_patched_disrupt) / (avg KL_corr)

clipped to [0, 1]: 0 = the site carries no corruption-sensitive info,
1 = corrupting this single site reproduces the full-corruption effect.

Per-position KL framework
-------------------------
For each test instance and each site (layer L, component c, role-mapped
position p), three forward passes + N-step greedy decode from A_b:

  1. CLEAN     : forward on the clean prompt x_i; cache the clean
                 activation at site (L, c, p); decode N tokens; save the
                 N next-token distributions P_clean^(j).
  2. CORR      : forward on the FULL partner prompt x~_i; cache the
                 partner activation at site (L, c, p_partner); decode
                 N tokens; save distributions P_corr^(j).
                 [This is the natural-grid partner forward, different
                  from causal_trace.py's symbol-swap forward.]
  3. PATCHED_d : forward on the CLEAN prompt with the PARTNER activation
                 injected at (L, c, p); decode N tokens; save
                 distributions P_patched^(j).

Per-position KL (fp32), computed inside the trace:

    KL_j^patched = KL(P_clean^(j) || P_patched^(j))
    KL_j^corr    = KL(P_clean^(j) || P_corr^(j))

Position-role mapping
---------------------
Because the partner forward lives on its own grid (different prompt
length, distinct abs positions), partner activations are looked up by
position ROLE, not absolute index. Mapping clean -> partner:

    prompt[k] from the right   ↔ partner prompt[k] from the right
    prompt_boundary            ↔ partner prompt_boundary
    T_i  (i = 1..K)            ↔ partner T_i
    answer_boundary            ↔ partner answer_boundary
    joint_prompt slab          ↔ corresponding right-aligned partner slab
    joint_thought slab         ↔ partner thought slab

Thoughts and boundaries map exactly (same n_thoughts on both sides).
Prompt tokens are right-aligned so the trailing last_n window (the
default coverage in causal_trace.py) remains semantically anchored at
the question→latent transition regardless of question length. If the
partner is shorter than the clean prompt at the same right-offset, that
prompt site is skipped (NaN in the output grid, mirroring the
unavailable-activation pattern in causal_trace.py).

Output
------
File: rev_trace_{task}_{model}_symbol_swap_{granularity}.npz

Schema mirrors causal_trace.py merged shards (same field names so
downstream score_trace.py can be reused with minimal changes — the only
semantic difference is the interpretation of kl_clean_patched, hence
IE_disrupt = avg(KL_patched) / avg(KL_corr) instead of
IE = 1 - avg(KL_patched) / avg(KL_corr)).
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path

from src.config import OUTPUTS
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    load_data,
    is_pause_model,
    _compare_answers,
    _shard_indices,
    run_normal_inference_pauseaware,
    run_codi_single_alpha,
)

# Reuse machinery already validated in causal_trace.py.
from experiments.causal_tracing.causal_trace import (
    COMPONENTS, N_LAYERS, N_FORMAT_PREFIX, FORMAT_PREFIX_TOKENS, N_GENERATE,
    ActivationRecorder,
    ActivationPatcher,
    MultiPositionActivationPatcher,
    _multi_ctx,
    find_corruption_partner,
    get_gold_tokens,
    greedy_decode,
    _per_position_kl,
    build_coconut_inputs,
    coconut_forward_with_recorder,
    codi_forward,
    build_position_registry_coconut,
    build_position_registry_codi,
    _get_blocks,
    _to_object_array,
    _stack_or_object,
)


# ═════════════════════════════════════════════════════════════════════
# Role-based position mapping: clean abs_pos -> partner abs_pos
# ═════════════════════════════════════════════════════════════════════
#
# Cleaner grid:        [q_1 ... q_Lc] [start] [T_1 ... T_K] [end]
# Partner grid:        [p_1 ... p_Lp] [start] [T_1 ... T_K] [end]
#
# Mapping by role, with right-alignment for prompt tokens:
#   clean abs_pos < Lc      (prompt token at offset k from right)
#       -> partner abs_pos = Lp - (Lc - clean_pos)
#       (skip if partner_abs_pos < 0)
#   clean prompt_boundary (Lc)             -> partner prompt_boundary (Lp)
#   clean thought T_i (Lc + 1 + (i-1))     -> partner thought T_i (Lp + 1 + (i-1))
#   clean answer_boundary (Lc + 1 + K)     -> partner answer_boundary (Lp + 1 + K)
#
# Returns (partner_abs_pos, ok). ok=False means this site has no partner
# counterpart and should be skipped.
# ═════════════════════════════════════════════════════════════════════

def _role_map_coconut(clean_abs_pos, kind, clean_lengths, partner_lengths, n_thoughts):
    """
    clean_lengths / partner_lengths: dict with keys
        'prompt_len'        Lc / Lp  (q_tokens length, before start-latent)
        'start_pos'         == prompt_len (where start-latent sits)
        'end_pos'           start_pos + 1 + n_thoughts
    """
    Lc = clean_lengths['prompt_len']
    Lp = partner_lengths['prompt_len']

    if kind == "prompt":
        # Right-aligned offset
        offset_from_right = Lc - clean_abs_pos  # >=1
        partner_pos = Lp - offset_from_right
        if partner_pos < 0:
            return None, False
        return partner_pos, True

    if kind == "prompt_boundary":
        return partner_lengths['start_pos'], True

    if kind == "thought":
        # clean_abs_pos = clean_lengths['start_pos'] + 1 + i
        i = clean_abs_pos - clean_lengths['start_pos'] - 1
        return partner_lengths['start_pos'] + 1 + i, True

    if kind == "answer_boundary":
        return partner_lengths['end_pos'], True

    return None, False


def _role_map_codi(clean_abs_pos, kind, clean_lengths, partner_lengths, n_thoughts):
    """CODI grid: [q_1..q_{L-2}] [bot]  positions 0..L-1
                  [T_1..T_K]              positions L..L+K-1
                  [eot]                   position L+K
    clean_lengths['L'] / partner_lengths['L'] = L (prompt incl. bot)."""
    Lc = clean_lengths['L']
    Lp = partner_lengths['L']
    K = n_thoughts

    if kind == "prompt":
        # bot sits at Lc-1; right-align relative to bot.
        offset_from_right = (Lc - 1) - clean_abs_pos      # >=1 for non-bot
        partner_pos = (Lp - 1) - offset_from_right
        if partner_pos < 0:
            return None, False
        return partner_pos, True
    if kind == "prompt_boundary":
        return Lp - 1, True
    if kind == "thought":
        # clean thought_i lives at abs_pos Lc + i - 1, i in 1..K  →  partner Lp + i - 1
        i = clean_abs_pos - Lc + 1
        return Lp + i - 1, True
    if kind == "answer_boundary":
        return Lp + K, True
    return None, False


# ═════════════════════════════════════════════════════════════════════
# Disrupt-mode patched forward (Unbatched)
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_disrupt_forward_coconut(
    coconut_model, base_model, tokenizer, sample, n_thoughts, device,
    start_id, latent_id, end_id, *, patchers,
):
    """Clean-prompt Coconut/PaT forward with patchers active."""
    input_ids, _ = build_coconut_inputs(
        coconut_model, tokenizer, sample, n_thoughts, device,
        start_id, latent_id, end_id,
    )
    embedding = coconut_model.embedding
    embeds = embedding(input_ids).clone()

    is_pause = is_pause_model(coconut_model)
    if is_pause:
        for p in patchers:
            p.set_pass_offset(0, input_ids.shape[1])
        pause_emb = coconut_model.pause_embedding
        latent_positions = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()
        for pos in latent_positions:
            embeds[0, pos, :] = pause_emb
        cms = [p.patching() for p in patchers]
        with _multi_ctx(*cms):
            outputs = base_model(
                inputs_embeds=embeds,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
                use_cache=True,
            )
        return outputs.logits[0, -1, :], outputs.past_key_values

    latent_indices = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()
    if len(latent_indices) == 0:
        for p in patchers:
            p.set_pass_offset(0, input_ids.shape[1])
        cms = [p.patching() for p in patchers]
        with _multi_ctx(*cms):
            outputs = base_model(
                inputs_embeds=embeds,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
                use_cache=True,
            )
        return outputs.logits[0, -1, :], outputs.past_key_values

    next_compute_range = (0, latent_indices[0])
    kv_cache = None
    last_step_logits = None
    max_n_latents = len(latent_indices)
    pass_idx = 0

    while True:
        cur_start, cur_end = next_compute_range
        cur_len = cur_end - cur_start
        for p in patchers:
            p.set_pass_offset(cur_start, cur_len)
        cms = [p.patching() for p in patchers]
        with _multi_ctx(*cms):
            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=embeds[:, cur_start:cur_end, :],
                    attention_mask=torch.ones((1, cur_end), device=device, dtype=torch.long),
                    position_ids=torch.arange(cur_start, cur_end, device=device).unsqueeze(0),
                    use_cache=True, output_hidden_states=True,
                )
            else:
                past_kv_trim = [
                    (k[:, :, :cur_start, :], v[:, :, :cur_start, :])
                    for k, v in kv_cache
                ]
                outputs = base_model(
                    inputs_embeds=embeds[:, cur_start:cur_end, :],
                    attention_mask=torch.ones((1, cur_end), device=device, dtype=torch.long),
                    position_ids=torch.arange(cur_start, cur_end, device=device).unsqueeze(0),
                    past_key_values=past_kv_trim, use_cache=True,
                    output_hidden_states=True,
                )
        kv_cache = outputs.past_key_values
        last_step_logits = outputs.logits[0, -1, :]

        if pass_idx + 1 >= max_n_latents:
            final_start = cur_end
            final_end = input_ids.shape[1]
            if final_end > final_start:
                for p in patchers:
                    p.set_pass_offset(final_start, final_end - final_start)
                past_kv_trim = [
                    (k[:, :, :final_start, :], v[:, :, :final_start, :])
                    for k, v in kv_cache
                ]
                cms = [p.patching() for p in patchers]
                with _multi_ctx(*cms):
                    outputs = base_model(
                        inputs_embeds=embeds[:, final_start:final_end, :],
                        attention_mask=torch.ones((1, final_end), device=device, dtype=torch.long),
                        position_ids=torch.arange(final_start, final_end, device=device).unsqueeze(0),
                        past_key_values=past_kv_trim, use_cache=True,
                        output_hidden_states=True,
                    )
                kv_cache = outputs.past_key_values
                last_step_logits = outputs.logits[0, -1, :]
            break

        # Recurrence (clean grid)
        next_latent_pos = latent_indices[pass_idx]
        hidden_states = outputs.hidden_states[-1]
        local_pos = next_latent_pos - 1 - cur_start
        recurrent_h = hidden_states[0, local_pos, :]
        embeds[0, next_latent_pos, :] = recurrent_h
        next_compute_range = (next_latent_pos, next_latent_pos + 1)
        pass_idx += 1

    return last_step_logits, kv_cache


@torch.no_grad()
def _run_disrupt_forward_codi(codi_dict, sample, n_thoughts, device, *, patchers):
    """Clean-prompt CODI forward with patchers active.

    Structurally identical to causal_trace._run_patched_forward_codi but
    WITHOUT the partner-prompt substitution: prompt embeddings stay clean.
    """
    model = codi_dict["model"]
    prj = codi_dict["prj"]
    tokenizer = codi_dict["tokenizer"]
    bot_id = codi_dict["bot_id"]
    eot_id = codi_dict["eot_id"]
    embedding_fn = codi_dict["embedding_fn"]
    use_prj = codi_dict["use_prj"]
    remove_eos = codi_dict["remove_eos"]

    question_tokens = tokenizer.encode(
        sample["question"].strip().replace("  ", " "), add_special_tokens=True,
    )
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L = input_ids.size(1)
    embeds = embedding_fn(input_ids).clone()

    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    for p in patchers:
        p.set_pass_offset(0, L)
    cms = [p.patching() for p in patchers]
    with _multi_ctx(*cms):
        outputs = model(
            inputs_embeds=embeds, use_cache=True, output_hidden_states=True,
            attention_mask=attention_mask, position_ids=position_ids,
        )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]
    latent = h.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)
    running_mask = attention_mask

    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype, device=device)], dim=1,
        )
        pos_t = torch.tensor([[L + t - 1]], device=device)
        for p in patchers:
            p.set_pass_offset(L + t - 1, 1)
        cms = [p.patching() for p in patchers]
        with _multi_ctx(*cms):
            outputs = model(
                inputs_embeds=latent, use_cache=True, output_hidden_states=True,
                past_key_values=past_kv, attention_mask=running_mask,
                position_ids=pos_t,
            )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][0, -1, :]
        latent = h.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    if remove_eos:
        eot_row = [eot_id]
    else:
        eot_row = [eot_id, tokenizer.eos_token_id]
    eot_ids = torch.tensor([eot_row], device=device)
    eot_emb = embedding_fn(eot_ids)
    eot_len = eot_emb.size(1)
    eot_pos = torch.arange(L + n_thoughts, L + n_thoughts + eot_len, device=device).unsqueeze(0)
    running_mask = torch.cat(
        [running_mask, torch.ones((1, eot_len), dtype=running_mask.dtype, device=device)], dim=1,
    )
    for p in patchers:
        p.set_pass_offset(L + n_thoughts, eot_len)
    cms = [p.patching() for p in patchers]
    with _multi_ctx(*cms):
        outputs = model(
            inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
            attention_mask=running_mask, position_ids=eot_pos,
            output_hidden_states=True,
        )
    return outputs.logits[0, -1, :], outputs.past_key_values


def _run_disrupt_forward(
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, n_thoughts, device, start_id, latent_id, end_id, *, patchers,
):
    if is_codi:
        return _run_disrupt_forward_codi(codi_dict, sample, n_thoughts, device, patchers=patchers)
    return _run_disrupt_forward_coconut(
        coconut_model, base_model, tokenizer, sample, n_thoughts, device,
        start_id, latent_id, end_id, patchers=patchers,
    )


# ═════════════════════════════════════════════════════════════════════
# Batched Disrupt-mode patched forward
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _batched_run_disrupt_forward_codi(codi_dict, sample, n_thoughts, device, *, batched_patcher):
    """Batched clean-prompt CODI forward with partner patches."""
    from experiments.causal_tracing.batched_patch import _tile_embeds
    model = codi_dict["model"]
    prj = codi_dict["prj"]
    tokenizer = codi_dict["tokenizer"]
    bot_id = codi_dict["bot_id"]
    eot_id = codi_dict["eot_id"]
    embedding_fn = codi_dict["embedding_fn"]
    use_prj = codi_dict["use_prj"]
    remove_eos = codi_dict["remove_eos"]
    B = batched_patcher.B

    question_tokens = tokenizer.encode(
        sample["question"].strip().replace("  ", " "), add_special_tokens=True,
    )
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L = input_ids.size(1)
    
    # Tile the CLEAN embeds (no partner substitution)
    embeds = embedding_fn(input_ids).clone()
    embeds_b = _tile_embeds(embeds, B)

    attention_mask = torch.ones((B, L), device=device, dtype=torch.long)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    with batched_patcher.patching():
        batched_patcher.set_pass_offset(0, L)
        outputs = model(
            inputs_embeds=embeds_b, use_cache=True, output_hidden_states=True,
            attention_mask=attention_mask, position_ids=position_ids,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][:, -1, :]            # (B, D)
        latent = h.unsqueeze(1)                            # (B, 1, D)
        if use_prj and prj is not None:
            latent = prj(latent)
        running_mask = attention_mask

        for t in range(1, n_thoughts + 1):
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=running_mask.dtype, device=device)], dim=1,
            )
            pos_t = torch.full((B, 1), L + t - 1, dtype=torch.long, device=device)
            batched_patcher.set_pass_offset(L + t - 1, 1)
            outputs = model(
                inputs_embeds=latent, use_cache=True, output_hidden_states=True,
                past_key_values=past_kv, attention_mask=running_mask,
                position_ids=pos_t,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][:, -1, :]
            latent = h.unsqueeze(1)
            if use_prj and prj is not None:
                latent = prj(latent)

        if remove_eos:
            eot_row = [eot_id]
        else:
            eot_row = [eot_id, tokenizer.eos_token_id]
        eot_ids = torch.tensor([eot_row], device=device)
        eot_emb = embedding_fn(eot_ids).expand(B, -1, -1).contiguous()
        eot_len = eot_emb.size(1)
        eot_pos = torch.arange(L + n_thoughts, L + n_thoughts + eot_len, device=device).unsqueeze(0).expand(B, -1)
        running_mask = torch.cat(
            [running_mask, torch.ones((B, eot_len), dtype=running_mask.dtype, device=device)], dim=1,
        )
        batched_patcher.set_pass_offset(L + n_thoughts, eot_len)
        outputs = model(
            inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
            attention_mask=running_mask, position_ids=eot_pos,
            output_hidden_states=True,
        )

    return outputs.logits[:, -1, :], outputs.past_key_values


@torch.no_grad()
def _batched_run_disrupt_forward(
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, n_thoughts, device, start_id, latent_id, end_id,
    *, batched_patcher
):
    """Dispatcher for batched disrupt trace."""
    if is_codi:
        return _batched_run_disrupt_forward_codi(
            codi_dict, sample, n_thoughts, device, batched_patcher=batched_patcher
        )

    from experiments.causal_tracing.batched_patch import _batched_coconut_patched_forward
    input_ids, _ = build_coconut_inputs(
        coconut_model, tokenizer, sample, n_thoughts, device,
        start_id, latent_id, end_id,
    )
    # Tile the CLEAN embeds (no partner substitution)
    embeds = coconut_model.embedding(input_ids).clone()
    
    return _batched_coconut_patched_forward(
        coconut_model, base_model, input_ids, embeds, n_thoughts, device,
        batched_patcher
    )


# ═════════════════════════════════════════════════════════════════════
# Trace a single instance (disruptive)
# ═════════════════════════════════════════════════════════════════════

def trace_instance_disrupt(
    *,
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, partner_sample,
    n_thoughts, device, task, model_name,
    start_id, latent_id, end_id,
    layers_to_trace, components_to_trace,
    granularity, window_size,
    prompt_coverage, last_n,
    instance_id=None, partner_id=None,
    batch_size=1, verify_batched=False,
):
    """Run CLEAN + partner-CORR + per-site DISRUPT forwards.

    Disrupt = clean-prompt forward with PARTNER activations injected at
    one site. Partner activations come from a NATIVE partner forward
    (not from the symbol-swap forward used in restorative tracing).
    """
    if is_codi:
        active_tokenizer = codi_dict["tokenizer"]
        blocks = _get_blocks(codi_dict["model"])
        decoder_model = codi_dict["model"]
        vocab_limit = decoder_model.config.vocab_size
    else:
        active_tokenizer = tokenizer
        blocks = _get_blocks(coconut_model)
        decoder_model = base_model
        vocab_limit = None

    gold_tokens = get_gold_tokens(active_tokenizer, sample, task)
    n_format_prefix = N_FORMAT_PREFIX[(task, model_name)]

    # ── (1) CLEAN forward, with recorder ──
    clean_recorder = ActivationRecorder(blocks)
    if is_codi:
        clean_sub_pass, clean_ans_logits, clean_past_kv, clean_L = codi_forward(
            codi_dict, sample, n_thoughts, device, recorder=clean_recorder,
        )
        clean_lengths = {"L": clean_L}
        clean_registry, clean_joint_positions = build_position_registry_codi(
            codi_dict, sample, n_thoughts, clean_sub_pass, clean_L,
            prompt_coverage, last_n,
        )
    else:
        clean_input_ids, clean_q_len = build_coconut_inputs(
            coconut_model, tokenizer, sample, n_thoughts, device,
            start_id, latent_id, end_id,
        )
        _, clean_sub_pass, clean_ans_logits, clean_past_kv = coconut_forward_with_recorder(
            coconut_model, base_model, clean_input_ids, n_thoughts, device,
            recorder=clean_recorder,
        )
        # In Coconut, start-latent sits right after the q-tokens, end-latent after the K latents.
        # prompt_len = position of start-latent in clean_input_ids.
        seq = clean_input_ids[0].tolist()
        clean_start_pos = seq.index(coconut_model.start_latent_id)
        clean_end_pos = seq.index(coconut_model.end_latent_id)
        clean_lengths = {
            "prompt_len": clean_start_pos,
            "start_pos":  clean_start_pos,
            "end_pos":    clean_end_pos,
        }
        clean_registry, clean_joint_positions = build_position_registry_coconut(
            coconut_model, clean_input_ids, n_thoughts, clean_sub_pass,
            prompt_coverage, last_n,
        )

    gen_clean_ids, dist_clean = greedy_decode(
        decoder_model, clean_ans_logits, clean_past_kv, N_GENERATE, device,
        vocab_limit=vocab_limit,
    )
    gen_clean_list = gen_clean_ids.detach().cpu().tolist()

    expected_prefix = FORMAT_PREFIX_TOKENS[model_name]
    observed_prefix = gen_clean_list[:n_format_prefix]
    if observed_prefix != expected_prefix:
        warnings.warn(
            f"[disrupt] format-prefix mismatch ({task},{model_name}): "
            f"expected {expected_prefix}, got {observed_prefix}",
            RuntimeWarning,
        )

    # Clean correctness via the standard inference path
    if is_codi:
        clean_decode = run_codi_single_alpha(
            codi_dict, sample, n_thoughts, device, lambda h, t: h, task=task,
        )
    else:
        clean_decode = run_normal_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device,
            start_id=start_id, latent_id=latent_id, task=task,
        )
    clean_correct = bool(clean_decode["is_correct"])

    # ── (2) PARTNER forward (native partner grid), with recorder ──
    partner_recorder = ActivationRecorder(blocks)
    if is_codi:
        partner_sub_pass, partner_ans_logits, partner_past_kv, partner_L = codi_forward(
            codi_dict, partner_sample, n_thoughts, device, recorder=partner_recorder,
        )
        partner_lengths = {"L": partner_L}
    else:
        partner_input_ids, _ = build_coconut_inputs(
            coconut_model, tokenizer, partner_sample, n_thoughts, device,
            start_id, latent_id, end_id,
        )
        _, partner_sub_pass, partner_ans_logits, partner_past_kv = coconut_forward_with_recorder(
            coconut_model, base_model, partner_input_ids, n_thoughts, device,
            recorder=partner_recorder,
        )
        partner_seq = partner_input_ids[0].tolist()
        partner_start_pos = partner_seq.index(coconut_model.start_latent_id)
        partner_end_pos = partner_seq.index(coconut_model.end_latent_id)
        partner_lengths = {
            "prompt_len": partner_start_pos,
            "start_pos":  partner_start_pos,
            "end_pos":    partner_end_pos,
        }

    gen_corr_ids, dist_corr = greedy_decode(
        decoder_model, partner_ans_logits, partner_past_kv, N_GENERATE, device,
        vocab_limit=vocab_limit,
    )
    gen_corr_list = gen_corr_ids.detach().cpu().tolist()

    # Corrupted correctness: does the partner's own decode emit the CLEAN gold?
    if is_codi:
        corr_decode = run_codi_single_alpha(
            codi_dict, partner_sample, n_thoughts, device, lambda h, t: h, task=task,
        )
    else:
        corr_decode = run_normal_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, partner_sample,
            n_thoughts, device,
            start_id=start_id, latent_id=latent_id, task=task,
        )
    _, _, corrupted_correct = _compare_answers(corr_decode["text"], sample, task)
    corrupted_correct = bool(corrupted_correct)

    # ── (3) Per-position KL(clean || corr) ──
    kl_clean_corr = _per_position_kl(dist_clean, dist_corr)

    # ── Sub-pass lookup on the PARTNER grid (for value lookup) ──
    def _partner_subpass(abs_pos_partner):
        for sp_idx, (offset, length) in enumerate(partner_sub_pass):
            if offset <= abs_pos_partner < offset + length:
                return sp_idx
        return len(partner_sub_pass) - 1

    role_map = _role_map_codi if is_codi else _role_map_coconut

    def _lookup_partner_value(tL, comp, clean_abs_pos, kind):
        """Resolve the partner-grid abs_pos for the role at (kind, clean_abs_pos),
        then read partner_recorder.cache[(tL, comp)][sp_idx][local_pos]."""
        partner_abs_pos, ok = role_map(
            clean_abs_pos, kind, clean_lengths, partner_lengths, n_thoughts,
        )
        if not ok:
            return None, None
        sp_idx = _partner_subpass(partner_abs_pos)
        k = (tL, comp)
        if k not in partner_recorder.cache or sp_idx >= len(partner_recorder.cache[k]):
            return None, None
        sub_act = partner_recorder.cache[k][sp_idx]
        local_pos = partner_abs_pos - partner_sub_pass[sp_idx][0]
        if not (0 <= local_pos < sub_act.shape[0]):
            return None, None
        return sub_act[local_pos], partner_abs_pos

    # ── Site iteration ──
    n_layers = len(layers_to_trace)
    n_components = len(components_to_trace)
    n_positions = len(clean_registry)

    kl_clean_patched = np.full(
        (n_layers, n_components, n_positions, N_GENERATE), np.nan, dtype=np.float32,
    )
    gen_patched = np.full(
        (n_layers, n_components, n_positions, N_GENERATE), -1, dtype=np.int32,
    )

    if granularity == "window":
        def layers_for_site(L_center):
            half = window_size // 2
            return [L for L in range(L_center - half, L_center + half + 1) if 0 <= L < N_LAYERS]
    else:
        def layers_for_site(L_center):
            return [L_center]

    if batch_size <= 1:
        for li, L_center in enumerate(layers_to_trace):
            target_layers = layers_for_site(L_center)
            for ci, comp in enumerate(components_to_trace):
                for pi, (clean_abs_pos, label, kind, _) in enumerate(clean_registry):
                    is_joint = kind in ("joint_prompt", "joint_thought")

                    if is_joint:
                        clean_slab_positions = clean_joint_positions[label]
                        if not clean_slab_positions:
                            continue
                        slab_kind = "prompt" if label == "joint_prompt" else "thought"

                        patchers = []
                        ok = True
                        for tL in target_layers:
                            values_per_abs = {}
                            for c_ap in clean_slab_positions:
                                v, _ = _lookup_partner_value(tL, comp, c_ap, slab_kind)
                                if v is None:
                                    continue
                                values_per_abs[c_ap] = v
                            if not values_per_abs:
                                ok = False
                                break
                            patchers.append(MultiPositionActivationPatcher(
                                blocks=blocks, layers=[tL], component=comp,
                                abs_positions=list(values_per_abs.keys()),
                                values_per_abs_pos=values_per_abs,
                            ))
                        if not ok:
                            continue
                    else:
                        values_per_layer = {}
                        ok = True
                        for tL in target_layers:
                            v, _ = _lookup_partner_value(tL, comp, clean_abs_pos, kind)
                            if v is None:
                                ok = False
                                break
                            values_per_layer[tL] = v
                        if not ok:
                            continue
                        patchers = [
                            ActivationPatcher(
                                blocks=blocks, layers=[tL], component=comp,
                                abs_pos=clean_abs_pos, value=values_per_layer[tL],
                            )
                            for tL in target_layers
                        ]

                    ans_logits_p, past_kv_p = _run_disrupt_forward(
                        is_codi, coconut_model, base_model, tokenizer, codi_dict,
                        sample, n_thoughts, device, start_id, latent_id, end_id,
                        patchers=patchers,
                    )
                    gen_p_ids, dist_p = greedy_decode(
                        decoder_model, ans_logits_p, past_kv_p, N_GENERATE, device,
                        vocab_limit=vocab_limit,
                    )
                    kl_clean_patched[li, ci, pi, :] = _per_position_kl(dist_clean, dist_p)
                    gen_patched[li, ci, pi, :] = gen_p_ids.detach().cpu().numpy().astype(np.int32)
    else:
        # ── Batched path ──
        from experiments.causal_tracing.batched_patch import (
            BatchedActivationPatcher, _batched_greedy_decode, _per_position_kl_batched
        )

        sites = [] 
        for li, L_center in enumerate(layers_to_trace):
            for ci, comp in enumerate(components_to_trace):
                for pi, (clean_abs_pos, label, kind, _) in enumerate(clean_registry):
                    sites.append((li, ci, pi, L_center, comp, clean_abs_pos, kind, label))

        specs_all = []
        kept_idx = []
        skipped_idx = []

        for si, (li, ci, pi, L_center, comp, clean_abs_pos, kind, label) in enumerate(sites):
            target_layers = layers_for_site(L_center)
            is_joint = kind in ("joint_prompt", "joint_thought")

            if is_joint:
                slab_positions = clean_joint_positions[label]
                if not slab_positions:
                    skipped_idx.append(si)
                    continue
                slab_kind = "prompt" if label == "joint_prompt" else "thought"

                values = {}
                ok = True
                for tL in target_layers:
                    for c_ap in slab_positions:
                        v, _ = _lookup_partner_value(tL, comp, c_ap, slab_kind)
                        if v is not None:
                            values[(tL, c_ap)] = v
                if not values:
                    skipped_idx.append(si)
                    continue
                
                specs_all.append({
                    "layers": list(target_layers),
                    "component": comp,
                    "abs_positions": list(set(k[1] for k in values.keys())),
                    "values_per_layer_abs": values,
                })
                kept_idx.append(si)
            else:
                values = {}
                ok = True
                for tL in target_layers:
                    v, _ = _lookup_partner_value(tL, comp, clean_abs_pos, kind)
                    if v is None:
                        ok = False
                        break
                    values[(tL, clean_abs_pos)] = v
                
                if not ok:
                    skipped_idx.append(si)
                    continue

                specs_all.append({
                    "layers": list(target_layers),
                    "component": comp,
                    "abs_positions": [clean_abs_pos],
                    "values_per_layer_abs": values,
                })
                kept_idx.append(si)

        # Chunk and run batched dispatching
        site_index = [(s[0], s[1], s[2]) for s in sites]
        for chunk_start in range(0, len(specs_all), batch_size):
            chunk_specs = specs_all[chunk_start:chunk_start + batch_size]
            chunk_site_idx = [site_index[kept_idx[chunk_start + i]] for i in range(len(chunk_specs))]
            B = len(chunk_specs)

            batched_patcher = BatchedActivationPatcher(
                blocks=blocks, batch_size=B, row_specs=chunk_specs,
            )

            ans_logits_b, past_kv_b = _batched_run_disrupt_forward(
                is_codi, coconut_model, base_model, tokenizer, codi_dict,
                sample, n_thoughts, device, start_id, latent_id, end_id,
                batched_patcher=batched_patcher
            )

            gen_ids_b, dist_b = _batched_greedy_decode(
                decoder_model, ans_logits_b, past_kv_b, N_GENERATE, device,
                vocab_limit=vocab_limit
            )

            kl_b = _per_position_kl_batched(dist_clean, dist_b)
            gen_b = gen_ids_b.detach().cpu().numpy().astype(np.int32)

            for bi, (li, ci, pi) in enumerate(chunk_site_idx):
                kl_clean_patched[li, ci, pi, :] = kl_b[bi]
                gen_patched[li, ci, pi, :] = gen_b[bi]

        if verify_batched:
            n_check = min(8, len(kept_idx))
            print(f"[verify_batched] checking {n_check} sites against unbatched reference", flush=True)
            max_kl_diff = 0.0
            max_gen_diff = 0
            for ki in range(n_check):
                si = kept_idx[ki]
                li, ci, pi, L_center, comp, clean_abs_pos, kind, label = sites[si]
                target_layers = layers_for_site(L_center)
                is_joint = kind in ("joint_prompt", "joint_thought")

                if is_joint:
                    slab_positions = clean_joint_positions[label]
                    slab_kind = "prompt" if label == "joint_prompt" else "thought"
                    patchers = []
                    for tL in target_layers:
                        vpa = {}
                        for c_ap in slab_positions:
                            v, _ = _lookup_partner_value(tL, comp, c_ap, slab_kind)
                            if v is not None:
                                vpa[c_ap] = v
                        patchers.append(MultiPositionActivationPatcher(
                            blocks=blocks, layers=[tL], component=comp,
                            abs_positions=list(vpa.keys()), values_per_abs_pos=vpa
                        ))
                else:
                    patchers = []
                    for tL in target_layers:
                        v, _ = _lookup_partner_value(tL, comp, clean_abs_pos, kind)
                        patchers.append(ActivationPatcher(
                            blocks=blocks, layers=[tL], component=comp,
                            abs_pos=clean_abs_pos, value=v
                        ))

                a_logits, pkv = _run_disrupt_forward(
                    is_codi, coconut_model, base_model, tokenizer, codi_dict,
                    sample, n_thoughts, device, start_id, latent_id, end_id,
                    patchers=patchers
                )
                gen_ref, dist_ref = greedy_decode(
                    decoder_model, a_logits, pkv, N_GENERATE, device, vocab_limit=vocab_limit
                )
                kl_ref = _per_position_kl(dist_clean, dist_ref)
                kl_diff = float(np.max(np.abs(kl_ref - kl_clean_patched[li, ci, pi, :])))
                gen_diff = int(np.sum(
                    gen_ref.detach().cpu().numpy().astype(np.int32) != gen_patched[li, ci, pi, :]
                ))
                max_kl_diff = max(max_kl_diff, kl_diff)
                max_gen_diff = max(max_gen_diff, gen_diff)
            print(f"[verify_batched] max |KL_batched - KL_unbatched| = {max_kl_diff:.3e}", flush=True)
            print(f"[verify_batched] max token mismatches = {max_gen_diff}", flush=True)
            assert max_kl_diff < 1e-4, f"batched/unbatched KL mismatch {max_kl_diff:.3e}"
            assert max_gen_diff == 0, f"batched/unbatched token mismatch on {max_gen_diff} positions"

    return {
        "position_labels": [r[1] for r in clean_registry],
        "position_kinds":  [r[2] for r in clean_registry],
        "position_abs":    [r[0] for r in clean_registry],
        "joint_positions": clean_joint_positions,
        "clean_correct":     clean_correct,
        "corrupted_correct": corrupted_correct,
        "gold_tokens":      list(gold_tokens),
        "n_format_prefix":  int(n_format_prefix),
        "kl_clean_corr":    kl_clean_corr,
        "kl_clean_patched": kl_clean_patched,
        "gen_clean":        gen_clean_list,
        "gen_corr":         gen_corr_list,
        "gen_patched":      gen_patched,
        "n_sub_passes_clean": len(clean_sub_pass),
    }


# ═════════════════════════════════════════════════════════════════════
# Shard save / merge (rev_trace_* naming)
# ═════════════════════════════════════════════════════════════════════

def _save_shard_rev(path, per_inst):
    """Same schema as causal_trace._save_shard, written under rev_trace_*."""
    nps = set(len(p["position_labels"]) for p in per_inst)
    uniform = (len(nps) == 1)

    gold_tokens_arr = _to_object_array(
        [np.asarray(p["gold_tokens"], dtype=np.int32) for p in per_inst]
    )
    gen_clean_arr = np.stack(
        [np.asarray(p["gen_clean"], dtype=np.int32) for p in per_inst], axis=0
    )
    gen_corr_arr = np.stack(
        [np.asarray(p["gen_corr"], dtype=np.int32) for p in per_inst], axis=0
    )
    kl_clean_corr_arr = np.stack(
        [np.asarray(p["kl_clean_corr"], dtype=np.float32) for p in per_inst], axis=0
    )
    joint_prompt_positions_arr = _to_object_array(
        [np.asarray(p["joint_positions"].get("joint_prompt", []), dtype=np.int32)
         for p in per_inst]
    )
    joint_thought_positions_arr = _to_object_array(
        [np.asarray(p["joint_positions"].get("joint_thought", []), dtype=np.int32)
         for p in per_inst]
    )

    if uniform:
        kl_clean_patched = np.stack([p["kl_clean_patched"] for p in per_inst], axis=0)
        gen_patched = np.stack([p["gen_patched"] for p in per_inst], axis=0)
        np.savez_compressed(
            path,
            kl_clean_corr=kl_clean_corr_arr,
            kl_clean_patched=kl_clean_patched,
            gen_clean=gen_clean_arr, gen_corr=gen_corr_arr, gen_patched=gen_patched,
            gold_tokens=gold_tokens_arr,
            n_format_prefix=np.array([p["n_format_prefix"] for p in per_inst], dtype=np.int32),
            position_labels=np.array(per_inst[0]["position_labels"]),
            position_kinds=np.array(per_inst[0]["position_kinds"]),
            position_abs=np.array(per_inst[0]["position_abs"], dtype=np.int32),
            joint_prompt_positions=joint_prompt_positions_arr,
            joint_thought_positions=joint_thought_positions_arr,
            clean_correct=np.array([p["clean_correct"] for p in per_inst]),
            corrupted_correct=np.array([p["corrupted_correct"] for p in per_inst]),
            instance_ids=np.array([p["instance_id"] for p in per_inst]),
            partner_ids=np.array([p["partner_id"] for p in per_inst]),
            n_sub_passes_clean=np.array([p["n_sub_passes_clean"] for p in per_inst]),
            uniform=np.array(True),
            mode=np.array("disrupt"),
        )
    else:
        kl_patched_obj = _to_object_array([p["kl_clean_patched"] for p in per_inst])
        gen_patched_obj = _to_object_array([p["gen_patched"] for p in per_inst])
        labels_obj = _to_object_array([np.asarray(p["position_labels"]) for p in per_inst])
        kinds_obj = _to_object_array([np.asarray(p["position_kinds"]) for p in per_inst])
        abs_obj = _to_object_array(
            [np.asarray(p["position_abs"], dtype=np.int32) for p in per_inst]
        )
        np.savez_compressed(
            path,
            kl_clean_corr=kl_clean_corr_arr,
            kl_clean_patched=kl_patched_obj,
            gen_clean=gen_clean_arr, gen_corr=gen_corr_arr, gen_patched=gen_patched_obj,
            gold_tokens=gold_tokens_arr,
            n_format_prefix=np.array([p["n_format_prefix"] for p in per_inst], dtype=np.int32),
            position_labels=labels_obj, position_kinds=kinds_obj, position_abs=abs_obj,
            joint_prompt_positions=joint_prompt_positions_arr,
            joint_thought_positions=joint_thought_positions_arr,
            clean_correct=np.array([p["clean_correct"] for p in per_inst]),
            corrupted_correct=np.array([p["corrupted_correct"] for p in per_inst]),
            instance_ids=np.array([p["instance_id"] for p in per_inst]),
            partner_ids=np.array([p["partner_id"] for p in per_inst]),
            n_sub_passes_clean=np.array([p["n_sub_passes_clean"] for p in per_inst]),
            uniform=np.array(False),
            mode=np.array("disrupt"),
        )


def merge_rev_shards(in_dir, task, model, granularity, delete_shards=True):
    """Merge rev_trace_*_rank*.npz into rev_trace_{task}_{model}_symbol_swap_{gran}.npz.

    Structurally identical to causal_trace.merge_rank_shards but reads/writes
    the rev_trace_* prefix. We reuse the underlying merge logic by glob-renaming
    in memory: load all shards, then call the same stack-or-object code path.
    """
    in_dir = Path(in_dir)
    pattern = f"rev_trace_{task}_{model}_symbol_swap_{granularity}_rank*.npz"
    paths = sorted(in_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards matching {pattern} under {in_dir}")

    var_keys = (
        "kl_clean_patched", "gen_patched",
        "position_labels", "position_kinds", "position_abs",
        "gold_tokens",
        "joint_prompt_positions", "joint_thought_positions",
    )
    concat_keys = (
        "kl_clean_corr", "gen_clean", "gen_corr",
        "n_format_prefix",
        "clean_correct", "corrupted_correct",
        "instance_ids", "partner_ids", "n_sub_passes_clean",
    )
    grid_keys = ("position_labels", "position_kinds", "position_abs")
    per_inst_keys = tuple(k for k in var_keys if k not in grid_keys)

    var_lists = {k: [] for k in per_inst_keys}
    grid_samples = {k: [] for k in grid_keys}
    concat_lists = {k: [] for k in concat_keys}
    shard_counts = []
    seen_instance_ids = {}

    for p in paths:
        z = np.load(p, allow_pickle=True)
        is_uniform_shard = bool(z["uniform"]) if "uniform" in z.files else True
        n_inst = int(np.atleast_1d(z["instance_ids"]).shape[0])
        shard_counts.append((p.name, n_inst))

        for k in per_inst_keys + tuple(k for k in concat_keys if k in z.files):
            arr = z[k]
            n_axis0 = int(np.atleast_1d(arr).shape[0])
            assert n_axis0 == n_inst, (
                f"Shard {p.name}: field '{k}' has {n_axis0} != n_inst={n_inst}"
            )

        for iid in np.asarray(z["instance_ids"]).tolist():
            iid = int(iid)
            if iid in seen_instance_ids:
                raise ValueError(
                    f"instance_id {iid} in both {seen_instance_ids[iid]} and {p.name}"
                )
            seen_instance_ids[iid] = p.name

        for k in per_inst_keys:
            arr = z[k]
            if is_uniform_shard and arr.dtype != object:
                var_lists[k].extend([arr[i] for i in range(n_inst)])
            else:
                var_lists[k].extend(list(arr))

        for k in grid_keys:
            arr = z[k]
            if is_uniform_shard and arr.dtype != object and arr.ndim == 1:
                grid_samples[k].append(arr)
            else:
                grid_samples[k].append(list(arr))

        for k in concat_keys:
            if k in z.files:
                concat_lists[k].append(np.atleast_1d(z[k]))

    out = {}
    uniform_global = True
    for k in per_inst_keys:
        items = var_lists[k]
        stacked, is_unif = _stack_or_object([np.asarray(x) for x in items])
        out[k] = stacked
        if k in ("kl_clean_patched", "gen_patched") and not is_unif:
            uniform_global = False

    for k in grid_keys:
        samples = grid_samples[k]
        all_uniform_shards = all(
            isinstance(s, np.ndarray) and s.ndim == 1 for s in samples
        )
        if uniform_global and all_uniform_shards and len(samples) > 0:
            ref = samples[0]
            all_same = all(
                s.shape == ref.shape and np.array_equal(s, ref) for s in samples
            )
            if all_same:
                out[k] = np.asarray(ref)
                continue
        per_inst_grids = []
        for p, s in zip(paths, samples):
            z = np.load(p, allow_pickle=True)
            n_i = int(np.atleast_1d(z["instance_ids"]).shape[0])
            if isinstance(s, np.ndarray) and s.ndim == 1:
                per_inst_grids.extend([s] * n_i)
            else:
                per_inst_grids.extend(list(s))
        out[k] = _to_object_array(per_inst_grids)
        uniform_global = False

    for k in concat_keys:
        if concat_lists[k]:
            out[k] = np.concatenate(concat_lists[k])

    out["uniform"] = np.array(uniform_global)
    out["mode"] = np.array("disrupt")

    expected_N = sum(n for _, n in shard_counts)
    audit_fields = list(per_inst_keys) + list(concat_keys)
    for k in audit_fields:
        if k not in out:
            continue
        n_have = int(np.atleast_1d(out[k]).shape[0])
        assert n_have == expected_N, (
            f"Post-merge audit failed: '{k}' has {n_have} != {expected_N}; "
            f"per-shard: {shard_counts}"
        )

    merged_path = in_dir / f"rev_trace_{task}_{model}_symbol_swap_{granularity}.npz"
    np.savez_compressed(merged_path, **out)
    print(f"  [merge] {merged_path.name}: N={expected_N} from {len(paths)} shard(s) "
          f"({', '.join(f'{n}:{c}' for n, c in shard_counts)})  "
          f"uniform={uniform_global}", flush=True)

    if delete_shards:
        for p in paths:
            p.unlink()
    return merged_path


# ═════════════════════════════════════════════════════════════════════
# Worker / main
# ═════════════════════════════════════════════════════════════════════

def _trace_worker(rank, world_size, args, return_dir):
    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)
        _sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = f"cuda:{rank}" if torch.cuda.is_available() and args.n_gpus > 0 else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    print(f"[rank {rank}] DISRUPT  device={device}  model={args.model}  task={args.task}",
          flush=True)

    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(args.task, device)
        coconut_model = base_model = tokenizer = None
        start_id = latent_id = end_id = None
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, device)
        codi_dict = None

    data = load_data(args.task, max_instances=args.max_instances)
    n_total = len(data)
    indices = _shard_indices(n_total, world_size, rank)
    print(f"[rank {rank}] {len(indices)}/{n_total} instances assigned", flush=True)

    rng = np.random.default_rng(args.seed + rank)
    partner_idx_map = {i: find_corruption_partner(data, i, args.task, rng) for i in indices}

    layers_to_trace = list(range(N_LAYERS))
    components_to_trace = list(COMPONENTS)

    grans = []
    if args.granularity in ("single", "both"):
        grans.append("single")
    if args.granularity in ("window", "both"):
        grans.append("window")

    for gran in grans:
        print(f"\n[rank {rank}] ▶ disrupt tracing: granularity={gran}", flush=True)
        per_inst = []
        t0 = time.time()
        for c, i in enumerate(indices):
            sample = data[i]
            partner_i = partner_idx_map[i]
            if partner_i is None:
                print(f"[rank {rank}] WARN: no partner for instance {i}; skipping", flush=True)
                continue
            partner_sample = data[partner_i]

            try:
                out = trace_instance_disrupt(
                    is_codi=is_codi, coconut_model=coconut_model, base_model=base_model,
                    tokenizer=tokenizer, codi_dict=codi_dict,
                    sample=sample, partner_sample=partner_sample,
                    n_thoughts=args.n_thoughts, device=device, task=args.task,
                    model_name=args.model,
                    start_id=start_id, latent_id=latent_id, end_id=end_id,
                    layers_to_trace=layers_to_trace,
                    components_to_trace=components_to_trace,
                    granularity=gran, window_size=args.window_size,
                    prompt_coverage=args.prompt_coverage,
                    last_n=args.last_n,
                    instance_id=i, partner_id=partner_i,
                    batch_size=args.batch_size,
                    verify_batched=(args.verify_batched and c == 0),
                )
                out["instance_id"] = i
                out["partner_id"] = partner_i
                per_inst.append(out)
            except Exception as e:
                print(f"[rank {rank}] ERROR instance {i}: {e!r}", flush=True)
                continue

            elapsed = time.time() - t0
            rate = (c + 1) / elapsed
            eta = (len(indices) - c - 1) / rate
            print(f"[rank {rank}]   {c+1}/{len(indices)}  rate={rate:.2f}/s  "
                  f"ETA={eta/60:.1f}m", flush=True)

        if not per_inst:
            print(f"[rank {rank}] no instances completed for {gran}; skipping save",
                  flush=True)
            continue

        save_path = return_dir / f"rev_trace_{args.task}_{args.model}_symbol_swap_{gran}_rank{rank}.npz"
        try:
            _save_shard_rev(save_path, per_inst)
            print(f"[rank {rank}] saved {len(per_inst)} -> {save_path}", flush=True)
        except Exception as e:
            import pickle
            fallback_path = save_path.with_suffix(".pkl")
            with open(fallback_path, "wb") as f:
                pickle.dump({"per_inst": per_inst, "args": vars(args)}, f)
            print(f"[rank {rank}] _save_shard_rev FAILED ({e!r}); "
                  f"raw pickle -> {fallback_path}", flush=True)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model", choices=["pause", "coconut", "coconut_u", "codi"],
                        required=True)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--granularity", choices=["single", "window", "both"],
                        default="single")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--prompt_coverage", choices=["all", "last_n"], default="last_n")
    parser.add_argument("--last_n", type=int, default=15)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default=str(OUTPUTS / "reverse_causal_trace"))
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of sites to batch into one patched forward.")
    parser.add_argument("--verify_batched", action="store_true",
                        help="When --batch_size > 1, re-run the first 8 sites unbatched and assert KL match.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"{args.task}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs -> {out_dir}", flush=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.n_gpus <= 1:
        _trace_worker(0, 1, args, out_dir)
    else:
        mp.spawn(_trace_worker, args=(args.n_gpus, args, out_dir),
                 nprocs=args.n_gpus, join=True)

    print("\n=== Merging rev_trace shards ===", flush=True)
    grans = []
    if args.granularity in ("single", "both"):
        grans.append("single")
    if args.granularity in ("window", "both"):
        grans.append("window")
    for gran in grans:
        try:
            merged = merge_rev_shards(out_dir, args.task, args.model, gran,
                                      delete_shards=True)
            print(f"  merged -> {merged.name}", flush=True)
        except FileNotFoundError as e:
            print(f"  skip {gran}: {e}", flush=True)
            continue

    print("\nDone. Disrupt-mode trace ready. "
          "Downstream: IE_disrupt = avg(KL_patched) / avg(KL_corr), "
          "clipped to [0, 1].", flush=True)


if __name__ == "__main__":
    main()