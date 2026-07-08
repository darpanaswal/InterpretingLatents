"""
causal_trace.py
================

Causal tracing for LRMs (Coconut family + CODI) and their non-recurrent
control (PaT-curriculum) on ProsQA and GSM8k.

Per-position KL framework
-------------------------
For each test instance and each site (layer L, component c, position p),
we run three forward passes and then greedy-decode N tokens starting at
the answer-boundary position A_b:

  1. CLEAN   : forward on the original prompt x_i; cache the activation
               at site (L, c, p); then greedy-decode N tokens from A_b,
               saving the N next-token distributions P_clean^(j) and
               token ids.
  2. CORR    : forward on the partner prompt x~_i (different gold answer);
               greedy-decode N tokens; save distributions P_corr^(j) and
               token ids.
  3. PATCHED : forward on x~_i with the CLEAN activation injected at site
               (L, c, p); greedy-decode N tokens; save distributions
               P_patched^(j) and token ids.

Per-position KL (in fp32) is computed inside the trace:

    KL_j^patched = KL(P_clean^(j) || P_patched^(j))
    KL_j^corr    = KL(P_clean^(j) || P_corr^(j))

These are saved per instance:
    kl_clean_corr     : shape (N,)
    kl_clean_patched  : shape (n_layers, n_components, n_positions, N)

Offline in score_trace.py: restrict to the content window
    j in [n_format_prefix, n_format_prefix + m_i)   (m_i = len(gold_tokens))
average each KL across that window, and compute
    IE(L, c, p) = 1 - (avg KL_patched) / (avg KL_corr)
Filter to clean_correct=True, aggregate across instances.

Causal stance: WRITE-TIME patching. The same patching machinery is used
for prompt tokens, thought positions, and answer-boundary; thought-pass
schedules drive the patcher's sub-pass offset.

Multi-GPU: instance-sharded data-parallel (mp.spawn); each rank writes
its own NPZ shard; --merge_shards combines them.
"""

import os
# Force unbuffered stdout/stderr for parent + spawned children. Without
# this, multi-GPU workers' print() calls sit in 64KB buffers and only
# flush on process exit, making long runs appear silently hung.
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
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from contextlib import contextmanager

from src.config import BASE_DIR, OUTPUTS
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    load_data,
    is_pause_model,
    extract_answer_number,
    _compare_answers,
    _shard_indices,
    run_normal_inference_pauseaware,
    run_codi_single_alpha,
    tokenize_question_for_recurrence,
)


# ═════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════

# Component names. For each layer we hook the residual stream (input to
# the block), the attention output, and the MLP output. All three are
# "write-time" sites. Component set is family-agnostic; the per-family
# submodule names that realize attn_out / mlp_out are resolved at runtime
# (see _resolve_submodule_names).
COMPONENTS = ("resid_pre", "attn_out", "mlp_out")

# Default layer count for GPT-2 small. The true count is read off the
# resolved transformer blocks at runtime (see _n_layers); this constant
# is only a fallback / documentation value.
N_LAYERS = 12  # GPT-2 small

# Format prefix = the fixed tokens the model emits between A_b and the
# gold answer. The TEXT of that prefix is fixed per model_name; the
# concrete token IDs depend on the tokenizer (GPT-2 BPE vs Llama BPE
# differ), so we tokenize the string at runtime rather than hardcode IDs.
#
#   PaT / Coconut family : '###'
#   CODI                 : 'The answer is:'
#
# Tokenized with add_special_tokens=False so no BOS leaks in.
FORMAT_PREFIX_TEXT = {
    ("prosqa", "pause"):     "###",
    ("prosqa", "coconut"):   "###",
    ("prosqa", "coconut_u"): "###",
    ("prosqa", "codi"):      "The answer is:",
    ("gsm",    "pause"):     "",
    ("gsm",    "coconut"):   "",
    ("gsm",    "coconut_u"): "",
    ("gsm",    "codi"):      "The answer is:",   # CODI keeps its prefix on GSM
}


def format_prefix_tokens(tokenizer, task, model_name):
    """Token IDs of the fixed format-prefix string for this (task, model_name)
    under the given tokenizer. Empty string -> empty list (no prefix)."""
    text = FORMAT_PREFIX_TEXT[(task, model_name)]
    if text == "":
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def n_format_prefix_tokens(tokenizer, task, model_name):
    """Length of the format-prefix token sequence (was the hardcoded
    N_FORMAT_PREFIX table; now derived so it stays correct across
    tokenizers)."""
    return len(format_prefix_tokens(tokenizer, task, model_name))

# Generation horizon: uniform across all (model, task). Covers
# max(n_format_prefix + max_gold_len + buffer) = 4 + 9 + 2 = 15.
N_GENERATE = 15


# ═════════════════════════════════════════════════════════════════════
# Architecture resolution (GPT-2 + Llama)
# ═════════════════════════════════════════════════════════════════════
#
# We attach forward hooks to the transformer blocks. Per block L, both
# supported families expose the same three "write-time" sites, but under
# different module names:
#
#   GPT-2 :  block.ln_1 -> block.attn      -> +resid -> block.ln_2 -> block.mlp -> +resid
#   Llama :  block.input_layernorm -> block.self_attn -> +resid
#            -> block.post_attention_layernorm -> block.mlp -> +resid
#
# Instrumented sites (identical semantics across families):
#   - resid_pre[L]: input to block L  (block forward-pre-hook)
#   - attn_out[L] : attention submodule output (post-proj, pre-residual-add)
#   - mlp_out[L]  : mlp submodule output       (post-proj, pre-residual-add)
#
# Submodule names:
#   attn submodule: "attn"  (GPT-2)  /  "self_attn" (Llama)
#   mlp  submodule: "mlp"   (both)
# resolved at runtime by _resolve_submodule_names from the block itself.
#
# Output shapes: GPT-2 submodules return tuples ((hidden, present, ...));
# Llama attn returns a tuple too (attn_output, attn_weights, ...) and
# Llama mlp returns the hidden tensor directly. The hooks below already
# handle both via `output[0] if isinstance(output, tuple) else output`,
# so name-resolution is the only family-specific piece.
# ═════════════════════════════════════════════════════════════════════

def _unwrap_to_hf(model):
    """Unwrap PEFT / Coconut wrappers down to the underlying HF
    CausalLM whose body is `.transformer` (GPT-2) or `.model` (Llama).

    Handles, in order:
      - Coconut wrapper        -> .base_causallm
      - PeftModel              -> .get_base_model()
    then returns the resulting module (still a *ForCausalLM).
    """
    m = model
    # Coconut wrapper exposes the HF/PEFT model as base_causallm.
    if hasattr(m, "base_causallm"):
        m = m.base_causallm
    # PEFT wrapper (LoRA): unwrap to the real base model.
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


def _get_blocks(model):
    """Return the list of transformer blocks for GPT-2 or Llama,
    transparently unwrapping Coconut and PEFT wrappers.

    GPT-2 : <hf>.transformer.h
    Llama : <hf>.model.layers
    """
    hf = _unwrap_to_hf(model)
    if hasattr(hf, "transformer") and hasattr(hf.transformer, "h"):
        return hf.transformer.h                       # GPT-2
    if hasattr(hf, "model") and hasattr(hf.model, "layers"):
        return hf.model.layers                        # Llama
    raise AttributeError(
        f"Cannot locate transformer blocks on {type(hf).__name__}; "
        f"expected .transformer.h (GPT-2) or .model.layers (Llama)."
    )


def _resolve_submodule_names(block):
    """Return (attn_name, mlp_name) for a single transformer block.

    GPT-2 : ("attn", "mlp")
    Llama : ("self_attn", "mlp")
    """
    if hasattr(block, "attn"):
        attn_name = "attn"
    elif hasattr(block, "self_attn"):
        attn_name = "self_attn"
    else:
        raise AttributeError(
            f"Block {type(block).__name__} has neither .attn nor .self_attn"
        )
    if not hasattr(block, "mlp"):
        raise AttributeError(f"Block {type(block).__name__} has no .mlp")
    return attn_name, "mlp"


def _n_layers(blocks):
    """True number of transformer blocks (replaces hardcoded N_LAYERS)."""
    return len(blocks)


# ─────────────────────────────────────────────────────────────────────
# KV-cache compatibility shim (legacy tuple cache vs transformers Cache)
# ─────────────────────────────────────────────────────────────────────
#
# GPT-2 (and Llama under some transformers configs) returns the legacy
# tuple cache: a tuple of per-layer (key, value) tensors, each shaped
# (batch, n_heads, seq_len, head_dim). Newer transformers wraps Llama's
# cache in a DynamicCache object that is NOT subscriptable as [layer][kv]
# and cannot be reconstructed by `for k, v in cache`.
#
# The trace code needs two operations on a KV cache:
#   1. read its current sequence length  (greedy_decode position bookkeeping)
#   2. trim every layer's K/V to the first `n` positions, then feed the
#      trimmed cache back in  (Coconut multi-pass recurrence)
#
# These helpers express both operations in a form that works on either
# representation. For Cache objects we trim the underlying per-layer
# tensors in place on a converted legacy view, then rebuild a fresh
# DynamicCache so the model sees the type it expects.
# ─────────────────────────────────────────────────────────────────────

def _kv_to_legacy(past_kv):
    """Return a list of (key, value) tensor pairs regardless of cache type."""
    if past_kv is None:
        return None
    # transformers Cache object
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
        return [(k, v) for (k, v) in legacy]
    # already legacy tuple/list of (k, v)
    return [(k, v) for (k, v) in past_kv]


def _kv_seq_len(past_kv):
    """Current cached sequence length (axis-2 of any layer's key tensor)."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    return int(past_kv[0][0].shape[2])


def _kv_trim(past_kv, n):
    """Trim every layer's K/V to the first `n` sequence positions and
    return a cache of the SAME type the model produced.

    For legacy tuple caches: return a list of trimmed (k, v) pairs (which
    HF accepts as past_key_values).
    For Cache objects: rebuild via DynamicCache.from_legacy_cache so the
    model receives a Cache, not a tuple.
    """
    legacy = _kv_to_legacy(past_kv)
    trimmed = tuple(
        (k[:, :, :n, :], v[:, :, :n, :]) for (k, v) in legacy
    )
    if hasattr(past_kv, "to_legacy_cache"):
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(trimmed)
    return list(trimmed)


class ActivationRecorder:
    """
    Captures clean activations across a forward pass / pass-sequence.

    Stores per (layer, component, sub_pass_idx) a list of tensors over
    sub-passes. Position indexing within each sub-pass is preserved.

    Stored layout:
        self.cache[(layer, component)] : list[Tensor], one entry per
            sub-pass; each Tensor is shape (seq_len_in_subpass, D).
    """
    def __init__(self, blocks):
        self.blocks = blocks
        self.handles = []
        self.cache = {}
        self.sub_pass_idx = -1  # incremented to 0 on first sub-pass

    def _block_pre_hook(self, layer):
        def hook(module, inputs):
            h = inputs[0]  # (batch, seq, D). batch is always 1 in tracing.
            key = (layer, "resid_pre")
            self.cache.setdefault(key, []).append(h[0].detach().clone())
        return hook

    def _attn_hook(self, layer):
        def hook(module, inputs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            key = (layer, "attn_out")
            self.cache.setdefault(key, []).append(attn_out[0].detach().clone())
        return hook

    def _mlp_hook(self, layer):
        def hook(module, inputs, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            key = (layer, "mlp_out")
            self.cache.setdefault(key, []).append(mlp_out[0].detach().clone())
        return hook

    def attach(self):
        for L, block in enumerate(self.blocks):
            attn_name, mlp_name = _resolve_submodule_names(block)
            self.handles.append(block.register_forward_pre_hook(self._block_pre_hook(L)))
            self.handles.append(getattr(block, attn_name).register_forward_hook(self._attn_hook(L)))
            self.handles.append(getattr(block, mlp_name).register_forward_hook(self._mlp_hook(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @contextmanager
    def recording(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()


class ActivationPatcher:
    """
    Applies a single-site patch during a forward pass.

    Constructor target: (layer, component, abs_pos, value)
        layer:      block index 0..L-1
        component:  one of COMPONENTS
        abs_pos:    absolute position in the FULL sequence (across sub-passes)
        value:      Tensor (D,) — the clean-cached activation to insert

    Sub-pass tracking
    -----------------
    We maintain a `current_offset` that tells us where in the absolute
    sequence the current sub-pass starts. Caller updates this before
    each sub-pass via `set_pass_offset(offset, length)`. Hooks then
    check whether `abs_pos` falls inside the current sub-pass's window
    and translate to a local position.
    """
    def __init__(self, blocks, layers, component, abs_pos, value):
        self.blocks = blocks
        self.layers = layers if isinstance(layers, (list, tuple)) else [layers]
        self.component = component
        self.abs_pos = abs_pos
        self.value = value  # (D,) tensor
        self.handles = []
        self.current_offset = 0
        self.current_length = 0

    def set_pass_offset(self, offset, length):
        """Tell the patcher where the current sub-pass sits in absolute coords."""
        self.current_offset = offset
        self.current_length = length

    def _local_pos(self):
        """Return local position within current sub-pass, or None if abs_pos is outside it."""
        lp = self.abs_pos - self.current_offset
        if 0 <= lp < self.current_length:
            return lp
        return None

    def _make_block_pre_hook(self, layer):
        def hook(module, inputs):
            if layer not in self.layers:
                return None
            if self.component != "resid_pre":
                return None
            lp = self._local_pos()
            if lp is None:
                return None
            h = inputs[0]
            new_h = h.clone()
            new_h[0, lp, :] = self.value.to(new_h.dtype).to(new_h.device)
            return (new_h,) + inputs[1:]
        return hook

    def _make_attn_hook(self, layer):
        def hook(module, inputs, output):
            if layer not in self.layers:
                return output
            if self.component != "attn_out":
                return output
            lp = self._local_pos()
            if lp is None:
                return output
            if isinstance(output, tuple):
                attn_out = output[0]
                new = attn_out.clone()
                new[0, lp, :] = self.value.to(new.dtype).to(new.device)
                return (new,) + output[1:]
            else:
                new = output.clone()
                new[0, lp, :] = self.value.to(new.dtype).to(new.device)
                return new
        return hook

    def _make_mlp_hook(self, layer):
        def hook(module, inputs, output):
            if layer not in self.layers:
                return output
            if self.component != "mlp_out":
                return output
            lp = self._local_pos()
            if lp is None:
                return output
            if isinstance(output, tuple):
                mlp_out = output[0]
                new = mlp_out.clone()
                new[0, lp, :] = self.value.to(new.dtype).to(new.device)
                return (new,) + output[1:]
            else:
                new = output.clone()
                new[0, lp, :] = self.value.to(new.dtype).to(new.device)
                return new
        return hook

    def attach(self):
        for L, block in enumerate(self.blocks):
            attn_name, mlp_name = _resolve_submodule_names(block)
            self.handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(L)))
            self.handles.append(getattr(block, attn_name).register_forward_hook(self._make_attn_hook(L)))
            self.handles.append(getattr(block, mlp_name).register_forward_hook(self._make_mlp_hook(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @contextmanager
    def patching(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()


# ═════════════════════════════════════════════════════════════════════
# MultiPositionActivationPatcher: joint-site patch over a set of positions
# ═════════════════════════════════════════════════════════════════════
#
# Used for the two "joint" rows in the heatmap:
#   - joint_prompt:  patch every prompt position (the slab covered by the
#                    last_n prompt window or full prompt) at a single
#                    (layer, component).
#   - joint_thought: patch every thought position (T_1..T_K) at a single
#                    (layer, component).
#
# Operationally identical to running K single-position ActivationPatchers
# simultaneously, but with one shared hook per layer to avoid registering
# 3*K hooks and to keep the in-place write paths deterministic.
#
# Each sub-pass only touches positions that fall inside its window, so the
# hook iterates the slab and writes any abs_pos in the slab whose local
# offset lands inside the current sub-pass.
# ═════════════════════════════════════════════════════════════════════

class MultiPositionActivationPatcher:
    """
    Joint-site patcher: substitutes clean activations at a SET of absolute
    positions, at one (layer, component).

    Constructor:
        layers:               list[int]  — block indices to patch at
        component:            str        — one of COMPONENTS
        abs_positions:        list[int]  — absolute positions to patch
        values_per_abs_pos:   dict[int -> Tensor(D,)] — clean activations
                              keyed by absolute position
    """
    def __init__(self, blocks, layers, component, abs_positions, values_per_abs_pos):
        self.blocks = blocks
        self.layers = layers if isinstance(layers, (list, tuple)) else [layers]
        self.component = component
        self.abs_positions = list(abs_positions)
        self.values_per_abs_pos = values_per_abs_pos
        self.handles = []
        self.current_offset = 0
        self.current_length = 0

    def set_pass_offset(self, offset, length):
        self.current_offset = offset
        self.current_length = length

    def _locals_in_pass(self):
        """Return list of (local_pos, abs_pos) for abs_positions inside the
        current sub-pass window. Empty list if none."""
        out = []
        for ap in self.abs_positions:
            lp = ap - self.current_offset
            if 0 <= lp < self.current_length:
                out.append((lp, ap))
        return out

    def _write_slab(self, tensor, locals_):
        """Clone tensor and write self.values_per_abs_pos[ap] at each local_pos."""
        new = tensor.clone()
        for lp, ap in locals_:
            v = self.values_per_abs_pos[ap]
            new[0, lp, :] = v.to(new.dtype).to(new.device)
        return new

    def _make_block_pre_hook(self, layer):
        def hook(module, inputs):
            if layer not in self.layers:
                return None
            if self.component != "resid_pre":
                return None
            locals_ = self._locals_in_pass()
            if not locals_:
                return None
            h = inputs[0]
            return (self._write_slab(h, locals_),) + inputs[1:]
        return hook

    def _make_attn_hook(self, layer):
        def hook(module, inputs, output):
            if layer not in self.layers:
                return output
            if self.component != "attn_out":
                return output
            locals_ = self._locals_in_pass()
            if not locals_:
                return output
            if isinstance(output, tuple):
                return (self._write_slab(output[0], locals_),) + output[1:]
            return self._write_slab(output, locals_)
        return hook

    def _make_mlp_hook(self, layer):
        def hook(module, inputs, output):
            if layer not in self.layers:
                return output
            if self.component != "mlp_out":
                return output
            locals_ = self._locals_in_pass()
            if not locals_:
                return output
            if isinstance(output, tuple):
                return (self._write_slab(output[0], locals_),) + output[1:]
            return self._write_slab(output, locals_)
        return hook

    def attach(self):
        for L, block in enumerate(self.blocks):
            attn_name, mlp_name = _resolve_submodule_names(block)
            self.handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(L)))
            self.handles.append(getattr(block, attn_name).register_forward_hook(self._make_attn_hook(L)))
            self.handles.append(getattr(block, mlp_name).register_forward_hook(self._make_mlp_hook(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @contextmanager
    def patching(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()


@contextmanager
def _multi_ctx(*ctxs):
    """Enter multiple context managers."""
    entered = []
    try:
        for c in ctxs:
            c.__enter__()
            entered.append(c)
        yield
    finally:
        for c in reversed(entered):
            c.__exit__(None, None, None)


# ═════════════════════════════════════════════════════════════════════
# Corruption: dataset-internal symbol-swap
# ═════════════════════════════════════════════════════════════════════
#
# symbol_swap: replace the entire prompt with another sample's prompt
#   whose ground-truth answer differs. The "clean answer" and
#   "corrupted answer" tokens then come from the two distinct samples.
#   This gives a well-formed prompt + a meaningful per-position KL
#   denominator (KL(P_clean || P_corr) at every generated step).
#
# Gaussian mode is NOT supported in this schema: the per-position KL
# framework requires a partner-based corrupted run as the denominator;
# gaussian (additive noise) doesn't produce a clean baseline
# distribution that's comparable position-by-position.
# ═════════════════════════════════════════════════════════════════════

def find_corruption_partner(data, clean_idx, task, rng):
    """
    For symbol_swap: find another sample with a DIFFERENT answer.
    Returns the partner sample's index, or None if no valid partner exists.
    """
    clean_sample = data[clean_idx]
    clean_gold = clean_sample.get("answer", "").strip()
    if task == "gsm":
        clean_ans_norm = _gsm_gold_number(clean_gold)
    else:
        clean_ans_norm = clean_gold.strip().lower()

    n = len(data)
    for _ in range(20):
        j = int(rng.integers(0, n))
        if j == clean_idx:
            continue
        other_gold = data[j].get("answer", "").strip()
        if task == "gsm":
            other_norm = _gsm_gold_number(other_gold)
        else:
            other_norm = other_gold.strip().lower()
        if other_norm != clean_ans_norm and other_norm != "":
            return j
    return None


def _gsm_gold_number(gold_text):
    """Extract the numeric gold answer from GSM8K format ('... #### 42')."""
    gold = gold_text.replace(",", "").strip()
    if "####" in gold:
        gold = gold.split("####")[-1].strip()
    try:
        return float(extract_answer_number(gold, task="gsm"))
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════
# Gold-answer tokens
# ═════════════════════════════════════════════════════════════════════

def get_first_answer_token(tokenizer, sample, task):
    """
    Return the FIRST gold-answer token id (kept for back-compat with
    score_trace.py). Format-aware: PaT/Coconut emit '### <answer>';
    CODI emits 'The answer is: <answer>'. By GPT-2 BPE convention we
    tokenize with a leading space so the answer's first token matches
    what the model would emit right after the format-prefix sequence.
    """
    tokens = get_gold_tokens(tokenizer, sample, task)
    return tokens[0] if tokens else tokenizer.eos_token_id


def get_gold_tokens(tokenizer, sample, task):
    """
    Return the FULL list of gold-answer token ids (no format prefix).

    For GSM: numeric answer string after '####'.
    For ProsQA: full gold sentence ('Sally is a sterpus.').

    Tokenized with a leading space (GPT-2 BPE convention) so that the
    first token aligns with what the model emits after the trailing
    format-prefix token (which itself does not end with a space).
    """
    gold = sample.get("answer", "").strip()
    if task == "gsm":
        if "####" in gold:
            ans_text = gold.split("####")[-1].strip().replace(",", "")
        else:
            ans_text = gold.replace(",", "").strip()
    else:
        ans_text = gold.strip()

    tokens = tokenizer.encode(" " + ans_text, add_special_tokens=False)
    if len(tokens) == 0:
        tokens = tokenizer.encode(ans_text, add_special_tokens=False)
    return list(tokens)


# ═════════════════════════════════════════════════════════════════════
# Greedy decode helper
# ═════════════════════════════════════════════════════════════════════
#
# Decode `n_steps` tokens greedily from a model state defined by
# `past_kv` (with the A_b token already processed). At each step j:
#
#   # Math:
#   #   logits_j        : input (`first_logits` at j=0, model output otherwise)
#   #   if vocab_limit is not None:
#   #       logits_j   <- logits_j[..., :vocab_limit - 1]
#   #   distribution_j  = softmax(logits_j.float())                # fp32
#   #   token_j         = argmax(logits_j).item()
#   #   pos_j           = past_kv seq-length  (read from KV cache)
#   #   model step:
#   #       outputs = model(
#   #           input_ids=tensor([[token_j]]),
#   #           past_key_values=past_kv,
#   #           attention_mask=ones((1, pos_j + 1)),
#   #           position_ids=tensor([[pos_j]]),
#   #           use_cache=True,
#   #       )
#   #       next_logits = outputs.logits[0, -1, :]
#   #       past_kv     = outputs.past_key_values
#
# `vocab_limit` (CODI only): the resized model has vocab size V_real + 3
# for [PAD]/[bot]/[eot]; slicing to `[..., :V_real + 2]` (i.e.
# vocab_limit - 1) excludes the [eot] token from argmax, matching the
# existing CODI decoding convention. Coconut/PaT does not need this.
#
# EOS handling: token 50256 (`<|endoftext|>`) is NOT a stop signal here.
# We always decode exactly n_steps so the per-position KL arrays have
# uniform shape (N,). If the model commits to EOS, the next-token
# distributions remain valid softmax outputs, and KL between two such
# distributions is meaningful for the per-position metric.
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(base_model, first_logits, past_kv, n_steps, device,
                  vocab_limit=None):
    """
    Returns:
        generated_ids : torch.LongTensor (n_steps,)
        distributions : torch.FloatTensor (n_steps, V_eff) in fp32
    """
    # Slice helper. Keep clean / corr / patched comparable: same slice every step.
    def _slice(logits):
        if vocab_limit is None:
            return logits
        return logits[..., :vocab_limit - 1]

    cur_logits = _slice(first_logits)
    V_eff = cur_logits.shape[-1]

    distributions = torch.empty((n_steps, V_eff), dtype=torch.float32, device=device)
    generated_ids = torch.empty((n_steps,), dtype=torch.long, device=device)

    for j in range(n_steps):
        # fp32 distribution
        dist_j = F.softmax(cur_logits.float(), dim=-1)
        distributions[j] = dist_j
        tok_j = int(cur_logits.argmax().item())
        generated_ids[j] = tok_j

        if j == n_steps - 1:
            break

        # Determine next position from KV cache length. Works for both the
        # legacy tuple cache (GPT-2: tensors (batch, n_heads, seq_len,
        # head_dim), seq_len at axis 2) and transformers Cache objects
        # (Llama), via _kv_seq_len.
        past_len = _kv_seq_len(past_kv)
        attn_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=device)
        input_ids = torch.tensor([[tok_j]], dtype=torch.long, device=device)

        outputs = base_model(
            input_ids=input_ids,
            past_key_values=past_kv,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        cur_logits = _slice(outputs.logits[0, -1, :])

    return generated_ids, distributions


# ═════════════════════════════════════════════════════════════════════
# Per-position KL
# ═════════════════════════════════════════════════════════════════════
#
# # Math:
# #   KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))    (>= 0)
# #
# # For numerical stability we keep both distributions in log space:
# #   log_p = log(P_clean)
# #   log_q = log(P_other)
# #   KL(P_clean || P_other) = sum_v exp(log_p) * (log_p - log_q)
# #
# # PyTorch's F.kl_div(input, target, log_target=True) computes
# #   sum: exp(target) * (target - input)
# # so to obtain KL(P_clean || P_other) we pass target=log_p (clean) and
# # input=log_q (other). This yields exp(log_p) * (log_p - log_q) = the
# # forward-KL with P_clean as the reference distribution.
# ═════════════════════════════════════════════════════════════════════

def _per_position_kl(p_clean_dists, p_other_dists):
    """
    Inputs are softmax distributions (fp32) of shape (N, V).

    Returns a numpy array of shape (N,) with KL(P_clean^(j) || P_other^(j))
    computed in fp32, per position j.
    """
    # Convert to log probabilities for the kl_div(log_target=True) form.
    # Add no epsilon: F.log on a valid softmax output is finite (down to ~-87
    # for fp32 minima). For exact zeros we clamp at -1e30 below.
    # clamp at 1e-30 (well above fp32 subnormal threshold ~1.18e-38) so
    # log() is finite on all backends; subnormal values may flush to 0.
    log_p = torch.log(p_clean_dists.clamp_min(1e-30))
    log_q = torch.log(p_other_dists.clamp_min(1e-30))
    # Per-row KL: sum_v exp(log_p) * (log_p - log_q)
    # Equivalent to F.kl_div(log_q, log_p, log_target=True, reduction='none').sum(-1)
    kl_per_row = F.kl_div(log_q, log_p, log_target=True, reduction='none').sum(dim=-1)
    return kl_per_row.detach().cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# Debug printer for --debug mode (single-instance trace inspection)
# ═════════════════════════════════════════════════════════════════════
#
# Prints, for one instance:
#   (A) sample / partner text + tokenization
#   (B) position registry (abs_pos, label, kind, sub_pass_idx)
#   (C) sub-pass schedule
#   (D) clean / corr generated tokens (decoded text + ids)
#   (E) format-prefix sanity check result
#   (F) kl_clean_corr per generated position
#   (G) summary of kl_clean_patched: mean/max KL per position-kind bucket
#       and a few representative (layer, component, position) sites
# Everything is plain text to stdout; no plotting.
# ═════════════════════════════════════════════════════════════════════

def _debug_print_instance(
    *,
    instance_id, partner_id, task, model_name,
    sample, partner_sample, tokenizer,
    input_ids_clean,                # may be None for CODI
    sub_pass_schedule_clean,
    registry,
    joint_positions,                # dict[label -> list[int]]
    gen_clean_list, gen_corr_list,
    expected_prefix, observed_prefix, n_format_prefix,
    gold_tokens,
    kl_clean_corr,
    kl_clean_patched,               # (L, C, P, N) np.float32
    layers_to_trace, components_to_trace,
    clean_correct, corrupted_correct,
):
    def _hr(c="─", n=72):
        print(c * n, flush=True)

    _hr("═")
    print(f"[DEBUG] instance_id={instance_id}  partner_id={partner_id}  "
          f"task={task}  model={model_name}", flush=True)
    _hr("═")

    # ── (A) Sample / partner text ──
    print("[A] CLEAN sample question:", flush=True)
    print(f"    {sample.get('question', '<no question>')[:300]}", flush=True)
    print(f"    gold(text)={sample.get('answer', sample.get('gold', '?'))}",
          flush=True)
    print("[A] PARTNER sample question:", flush=True)
    print(f"    {partner_sample.get('question', '<no question>')[:300]}",
          flush=True)
    print(f"    gold(text)={partner_sample.get('answer', partner_sample.get('gold', '?'))}",
          flush=True)
    print(f"[A] clean_correct={clean_correct}  corrupted_correct={corrupted_correct}",
          flush=True)

    # Show tokenized prompt (Coconut only — CODI tokenization is internal)
    if input_ids_clean is not None:
        ids = input_ids_clean[0].tolist()
        decoded = [tokenizer.decode([t]) for t in ids]
        print(f"[A] tokenized clean input ({len(ids)} tokens):", flush=True)
        for i, (tid, ts) in enumerate(zip(ids, decoded)):
            marker = ""
            for (abs_pos, label, kind, _) in registry:
                if abs_pos == i:
                    marker = f"  ◄ {label} ({kind})"
                    break
            print(f"    [{i:3d}] {tid:6d}  {ts!r:20s}{marker}", flush=True)

    # ── (B) Sub-pass schedule ──
    _hr()
    print(f"[B] sub_pass_schedule_clean ({len(sub_pass_schedule_clean)} passes):",
          flush=True)
    for idx, (off, length) in enumerate(sub_pass_schedule_clean):
        print(f"    pass {idx}: abs_positions [{off}, {off+length})  len={length}",
              flush=True)

    # ── (C) Position registry ──
    _hr()
    print(f"[C] position registry ({len(registry)} sites):", flush=True)
    for (abs_pos, label, kind, sp_idx) in registry:
        print(f"    abs_pos={abs_pos:3d}  kind={kind:18s}  label={label:14s}  "
              f"sp_idx={sp_idx}", flush=True)

    # ── (D) Generated tokens ──
    _hr()
    def _decode_seq(ids):
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return "<decode failed>"
    print(f"[D] gen_clean ids:  {gen_clean_list}", flush=True)
    print(f"[D] gen_clean text: {_decode_seq(gen_clean_list)!r}", flush=True)
    print(f"[D] gen_corr  ids:  {gen_corr_list}", flush=True)
    print(f"[D] gen_corr  text: {_decode_seq(gen_corr_list)!r}", flush=True)
    print(f"[D] gold_tokens ids:  {list(gold_tokens)}", flush=True)
    print(f"[D] gold_tokens text: {_decode_seq(list(gold_tokens))!r}", flush=True)

    # ── (E) Format-prefix sanity ──
    _hr()
    ok = (observed_prefix == expected_prefix)
    print(f"[E] format-prefix check ({'OK' if ok else 'MISMATCH'}):", flush=True)
    print(f"    n_format_prefix = {n_format_prefix}", flush=True)
    print(f"    expected = {expected_prefix} "
          f"({_decode_seq(expected_prefix)!r})", flush=True)
    print(f"    observed = {observed_prefix} "
          f"({_decode_seq(observed_prefix)!r})", flush=True)

    # ── (F) Per-position kl_clean_corr ──
    _hr()
    # The "content window" is [n_format_prefix, n_format_prefix + len(gold)).
    cw_start = n_format_prefix
    cw_end = min(n_format_prefix + len(gold_tokens), len(kl_clean_corr))
    print(f"[F] kl_clean_corr per generated position (content window = "
          f"[{cw_start}, {cw_end})):", flush=True)
    for j, kl in enumerate(kl_clean_corr):
        in_cw = cw_start <= j < cw_end
        tag = " ◄ content" if in_cw else ""
        print(f"    pos {j:2d}: KL={kl:8.4f}{tag}", flush=True)
    if cw_end > cw_start:
        avg = float(kl_clean_corr[cw_start:cw_end].mean())
        print(f"    avg KL over content window = {avg:.4f}", flush=True)

    # ── (G) Patched-KL summary ──
    _hr()
    # Average KL per site over the content window.
    if cw_end > cw_start:
        kl_site = kl_clean_patched[..., cw_start:cw_end].mean(axis=-1)  # (L,C,P)
    else:
        kl_site = kl_clean_patched.mean(axis=-1)
    # Mask out NaNs (sites that were skipped — e.g., abs_pos outside sub-pass).
    kl_site_masked = np.where(np.isnan(kl_site), np.nan, kl_site)

    print(f"[G] kl_clean_patched shape = {tuple(kl_clean_patched.shape)} "
          f"(layers={len(layers_to_trace)}, components={len(components_to_trace)}, "
          f"positions={len(registry)}, N={kl_clean_patched.shape[-1]})", flush=True)

    # Bucket by position kind
    kinds = [r[2] for r in registry]
    unique_kinds = []
    for k in kinds:
        if k not in unique_kinds:
            unique_kinds.append(k)
    print(f"[G] avg patched-KL by position-kind (averaged over content window, "
          f"layers, components):", flush=True)
    for k in unique_kinds:
        mask = np.array([kk == k for kk in kinds], dtype=bool)
        slab = kl_site_masked[:, :, mask]
        valid = slab[~np.isnan(slab)]
        if valid.size == 0:
            print(f"    {k:18s} : <no valid sites>", flush=True)
        else:
            print(f"    {k:18s} : mean={valid.mean():.4f}  "
                  f"median={np.median(valid):.4f}  "
                  f"max={valid.max():.4f}  n_sites={valid.size}", flush=True)

    # Top-5 highest-KL sites overall (most "shifted away from clean" patches)
    print(f"[G] top-5 highest patched-KL sites (most disruptive):", flush=True)
    flat = kl_site_masked.reshape(-1)
    finite = np.where(np.isfinite(flat))[0]
    if finite.size > 0:
        top = finite[np.argsort(-flat[finite])][:5]
        for idx_flat in top:
            li, ci, pi = np.unravel_index(idx_flat, kl_site_masked.shape)
            L = layers_to_trace[li]
            comp = components_to_trace[ci]
            label = registry[pi][1]
            kind = registry[pi][2]
            abs_pos = registry[pi][0]
            print(f"    L={L:2d} comp={comp:8s} site=({kind}, {label}, abs={abs_pos}) "
                  f"avg_KL={float(flat[idx_flat]):.4f}", flush=True)

    # ── (H) Joint-site report ──
    # The two joint sites are the headline test. Print a per-layer KL profile
    # for resid_pre at each joint site, alongside the corr baseline.
    _hr()
    baseline = float(kl_clean_corr[cw_start:cw_end].mean()) if cw_end > cw_start else float("nan")
    print(f"[H] joint-site KL profiles (resid_pre, content-window-averaged):",
          flush=True)
    print(f"    baseline KL(clean||corr) = {baseline:.4f}", flush=True)
    print(f"    joint_prompt covers {len(joint_positions.get('joint_prompt', []))} "
          f"positions: {joint_positions.get('joint_prompt', [])}", flush=True)
    print(f"    joint_thought covers {len(joint_positions.get('joint_thought', []))} "
          f"positions: {joint_positions.get('joint_thought', [])}", flush=True)
    # Find resid_pre component index
    try:
        ci_resid = list(components_to_trace).index("resid_pre")
    except ValueError:
        ci_resid = 0
    for joint_label in ("joint_prompt", "joint_thought"):
        # find pi for this label
        pi_joint = None
        for pi_, (_, lbl, _, _) in enumerate(registry):
            if lbl == joint_label:
                pi_joint = pi_
                break
        if pi_joint is None:
            print(f"    {joint_label:14s}: <not in registry>", flush=True)
            continue
        print(f"    {joint_label} (per-layer avg KL, resid_pre):", flush=True)
        for li, L in enumerate(layers_to_trace):
            val = kl_site_masked[li, ci_resid, pi_joint]
            if np.isnan(val):
                print(f"      L={L:2d}: <nan>", flush=True)
            else:
                delta = baseline - float(val)
                print(f"      L={L:2d}: KL={float(val):.4f}  "
                      f"(baseline - KL = {delta:+.4f}; "
                      f"+ = restorative, - = disruptive)", flush=True)
    _hr("═")


# ═════════════════════════════════════════════════════════════════════
# Coconut/PaT: input building + multi-pass forward
# ═════════════════════════════════════════════════════════════════════

def build_coconut_inputs(coconut_model, tokenizer, sample, n_thoughts, device,
                         start_id, latent_id, end_id):
    """Tokenize and build the [Q, start, latents*K, end] sequence.

    Question prefix tokenized via tokenize_question_for_recurrence so the
    format matches training: chat template for instruct Llama, raw "{q}\\n"
    for GPT-2 / non-instruct. Using raw encode for instruct Llama is
    format-OOD and produces spurious results, so this must stay centralized.
    """
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])

    ids = (
        question_tokens
        + [start_id]
        + [latent_id] * n_thoughts
        + [end_id]
    )
    input_ids = torch.tensor([ids], device=device)
    return input_ids, len(question_tokens)


@torch.no_grad()
def coconut_forward_with_recorder(coconut_model, base_model, input_ids, n_thoughts,
                                  device, recorder=None, patcher=None,
                                  patch_pass_idx_filter=None):
    """
    Run the coconut multi-pass forward. Returns:
        final_logits      : (1, seq_len, vocab)
        sub_pass_schedule : list of (offset, length)
        end_latent_logit  : (vocab,) — logits at the answer-boundary position
                            (the last position of the FINAL sub-pass, where
                             decoding resumes)
        final_past_kv     : KV cache after the final sub-pass

    If recorder is provided, attaches it for the duration.
    If patcher is provided, attaches it; patcher.set_pass_offset is
    advanced for each sub-pass.
    patch_pass_idx_filter: if not None, only allow patching during sub-passes
        whose index is in this set. None means "always".
    """
    is_pause = is_pause_model(coconut_model)

    sub_pass_schedule = []

    if is_pause:
        # Single forward pass. Replace latent tokens with pause embedding.
        embedding = coconut_model.embedding
        inputs_embeds = embedding(input_ids).clone()
        pause_emb = coconut_model.pause_embedding
        latent_positions = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()
        for pos in latent_positions:
            inputs_embeds[0, pos, :] = pause_emb

        attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        seq_len = input_ids.shape[1]
        sub_pass_schedule.append((0, seq_len))

        if patcher is not None:
            if patch_pass_idx_filter is None or 0 in patch_pass_idx_filter:
                patcher.set_pass_offset(0, seq_len)
            else:
                patcher.set_pass_offset(-10_000, 0)  # disable

        cms = []
        if recorder is not None: cms.append(recorder.recording())
        if patcher is not None:  cms.append(patcher.patching())
        with _multi_ctx(*cms):
            outputs = base_model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True,
            )
        return outputs.logits, sub_pass_schedule, outputs.logits[0, -1, :], outputs.past_key_values

    # ── Coconut continuous: multi-pass recurrence ──
    latent_indices = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()
    if len(latent_indices) == 0:
        # Degenerate: no thoughts
        seq_len = input_ids.shape[1]
        sub_pass_schedule.append((0, seq_len))
        if patcher is not None:
            patcher.set_pass_offset(0, seq_len)
        cms = []
        if recorder is not None: cms.append(recorder.recording())
        if patcher is not None:  cms.append(patcher.patching())
        with _multi_ctx(*cms):
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
                use_cache=True,
            )
        return outputs.logits, sub_pass_schedule, outputs.logits[0, -1, :], outputs.past_key_values

    max_n_latents = len(latent_indices)
    embedding = coconut_model.embedding
    inputs_embeds = embedding(input_ids).clone()

    next_compute_range = (0, latent_indices[0])
    kv_cache = None
    all_logits = []
    last_step_logits = None
    final_past_kv = None

    pass_idx = 0
    while True:
        cur_start, cur_end = next_compute_range
        cur_len = cur_end - cur_start
        sub_pass_schedule.append((cur_start, cur_len))

        if patcher is not None:
            if patch_pass_idx_filter is None or pass_idx in patch_pass_idx_filter:
                patcher.set_pass_offset(cur_start, cur_len)
            else:
                patcher.set_pass_offset(-10_000, 0)

        if kv_cache is None:
            cms = []
            if recorder is not None: cms.append(recorder.recording())
            if patcher is not None:  cms.append(patcher.patching())
            with _multi_ctx(*cms):
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, cur_start:cur_end, :],
                    attention_mask=torch.ones((1, cur_end), device=device, dtype=torch.long),
                    position_ids=torch.arange(cur_start, cur_end, device=device).unsqueeze(0),
                    use_cache=True,
                    output_hidden_states=True,
                )
        else:
            past_kv_trim = _kv_trim(kv_cache, cur_start)
            cms = []
            if recorder is not None: cms.append(recorder.recording())
            if patcher is not None:  cms.append(patcher.patching())
            with _multi_ctx(*cms):
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, cur_start:cur_end, :],
                    attention_mask=torch.ones((1, cur_end), device=device, dtype=torch.long),
                    position_ids=torch.arange(cur_start, cur_end, device=device).unsqueeze(0),
                    past_key_values=past_kv_trim,
                    use_cache=True,
                    output_hidden_states=True,
                )

        all_logits.append(outputs.logits)
        last_step_logits = outputs.logits[0, -1, :]
        kv_cache = outputs.past_key_values
        final_past_kv = kv_cache

        if pass_idx + 1 >= max_n_latents:
            # Final pass over the answer region
            final_start = cur_end
            final_end = input_ids.shape[1]
            if final_end > final_start:
                sub_pass_schedule.append((final_start, final_end - final_start))
                final_pass_idx = pass_idx + 1
                if patcher is not None:
                    if patch_pass_idx_filter is None or final_pass_idx in patch_pass_idx_filter:
                        patcher.set_pass_offset(final_start, final_end - final_start)
                    else:
                        patcher.set_pass_offset(-10_000, 0)
                past_kv_trim = _kv_trim(kv_cache, final_start)
                cms = []
                if recorder is not None: cms.append(recorder.recording())
                if patcher is not None:  cms.append(patcher.patching())
                with _multi_ctx(*cms):
                    outputs = base_model(
                        inputs_embeds=inputs_embeds[:, final_start:final_end, :],
                        attention_mask=torch.ones((1, final_end), device=device, dtype=torch.long),
                        position_ids=torch.arange(final_start, final_end, device=device).unsqueeze(0),
                        past_key_values=past_kv_trim,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                all_logits.append(outputs.logits)
                last_step_logits = outputs.logits[0, -1, :]
                final_past_kv = outputs.past_key_values
            break

        # Recurrence: fill the next latent slot with previous pass's last hidden state
        next_latent_pos = latent_indices[pass_idx]
        hidden_states = outputs.hidden_states[-1]
        hidden_states_offset = cur_start
        local_pos = next_latent_pos - 1 - hidden_states_offset
        recurrent_h = hidden_states[0, local_pos, :]
        inputs_embeds[0, next_latent_pos, :] = recurrent_h

        next_compute_range = (next_latent_pos, next_latent_pos + 1)
        pass_idx += 1

    full_logits = torch.cat(all_logits, dim=-2) if len(all_logits) > 1 else all_logits[0]
    return full_logits, sub_pass_schedule, last_step_logits, final_past_kv


# ═════════════════════════════════════════════════════════════════════
# CODI: multi-pass forward
# ═════════════════════════════════════════════════════════════════════
#
# CODI builds: [question_tokens] [bot] then loops K times feeding back
# the last hidden state, optionally through a projection layer.
#
# Sub-pass schedule:
#   pass 0       : prompt tokens including [bot]; positions 0..L-1
#                    (L = len(question + [bot]))
#   pass t (1..K): single position L+t-1 (one new latent)
#   pass K+1     : [eot] token at position L+K
#
# Answer-boundary = position L+K (the eot output is the next-token
# logit for the answer).
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def codi_forward(codi_dict, sample, n_thoughts, device, recorder=None, patcher=None,
                 patch_pass_idx_filter=None):
    model = codi_dict["model"]
    prj = codi_dict["prj"]
    tokenizer = codi_dict["tokenizer"]
    bot_id = codi_dict["bot_id"]
    eot_id = codi_dict["eot_id"]
    embedding_fn = codi_dict["embedding_fn"]
    use_prj = codi_dict["use_prj"]
    remove_eos = codi_dict["remove_eos"]

    # ── Sub-pass 0: prompt ──
    question_tokens = tokenizer.encode(
        sample["question"].strip().replace("  ", " "),
        add_special_tokens=True,
    )
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L = input_ids.size(1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    sub_pass_schedule = [(0, L)]
    if patcher is not None:
        if patch_pass_idx_filter is None or 0 in patch_pass_idx_filter:
            patcher.set_pass_offset(0, L)
        else:
            patcher.set_pass_offset(-10_000, 0)

    cms = []
    if recorder is not None: cms.append(recorder.recording())
    if patcher is not None:  cms.append(patcher.patching())
    with _multi_ctx(*cms):
        outputs = model(
            input_ids=input_ids, use_cache=True, output_hidden_states=True,
            attention_mask=attention_mask, position_ids=position_ids,
        )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]
    latent = h.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    running_mask = attention_mask
    # ── Sub-passes 1..K: latent recurrence ──
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype, device=device)], dim=1,
        )
        pos_t = torch.tensor([[L + t - 1]], device=device)

        sub_pass_schedule.append((L + t - 1, 1))
        if patcher is not None:
            if patch_pass_idx_filter is None or t in patch_pass_idx_filter:
                patcher.set_pass_offset(L + t - 1, 1)
            else:
                patcher.set_pass_offset(-10_000, 0)

        cms = []
        if recorder is not None: cms.append(recorder.recording())
        if patcher is not None:  cms.append(patcher.patching())
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

    # ── Sub-pass K+1: eot (answer boundary) ──
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

    sub_pass_schedule.append((L + n_thoughts, eot_len))
    final_pass_idx = n_thoughts + 1
    if patcher is not None:
        if patch_pass_idx_filter is None or final_pass_idx in patch_pass_idx_filter:
            patcher.set_pass_offset(L + n_thoughts, eot_len)
        else:
            patcher.set_pass_offset(-10_000, 0)

    cms = []
    if recorder is not None: cms.append(recorder.recording())
    if patcher is not None:  cms.append(patcher.patching())
    with _multi_ctx(*cms):
        outputs = model(
            inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
            attention_mask=running_mask, position_ids=eot_pos,
            output_hidden_states=True,
        )

    vocab_size = model.config.vocab_size
    # Note: full vocab returned here (no slicing). The greedy_decode helper
    # applies a uniform vocab_limit if provided, so callers get consistent
    # treatment across clean / corr / patched runs.
    answer_boundary_logits = outputs.logits[0, -1, :]
    return sub_pass_schedule, answer_boundary_logits, outputs.past_key_values, L


# ═════════════════════════════════════════════════════════════════════
# Position registries
# ═════════════════════════════════════════════════════════════════════
#
# For Coconut/PaT:
#   0..len(q_tokens)-1     : prompt tokens
#   len(q_tokens)          : <start-latent> (== prompt_boundary, P_b)
#   len(q_tokens)+1..      : <latent>*K     (T_1 .. T_K)
#   len(q_tokens)+K+1      : <end-latent>   (== answer_boundary, A_b)
#
# For CODI:
#   0..L-2                 : prompt tokens
#   L-1                    : [bot] (P_b)
#   L..L+K-1               : thoughts T_1..T_K
#   L+K                    : [eot] (A_b)
#
# Each registry entry: (abs_pos, label, kind, sub_pass_idx).
# ═════════════════════════════════════════════════════════════════════

def build_position_registry_coconut(coconut_model, input_ids, n_thoughts,
                                    sub_pass_schedule, prompt_coverage, last_n):
    """Returns (registry, joint_positions).

    registry: list of (abs_pos, label, kind, sub_pass_idx). For joint rows,
        abs_pos is -1 (sentinel) and sub_pass_idx is -1; the actual
        positions are in joint_positions[label].
    joint_positions: dict mapping joint-row label -> list[int] of absolute
        positions covered by that joint slab.
    """
    seq = input_ids[0].tolist()

    start_id = coconut_model.start_latent_id
    end_id = coconut_model.end_latent_id

    start_pos = seq.index(start_id)
    end_pos = seq.index(end_id)
    thought_positions = [start_pos + 1 + i for i in range(n_thoughts)]
    prompt_boundary = start_pos
    answer_boundary = end_pos
    prompt_token_positions = list(range(0, start_pos))

    if prompt_coverage == "last_n":
        prompt_token_positions = prompt_token_positions[-last_n:] if last_n > 0 else []

    def find_subpass(abs_pos):
        for sp_idx, (offset, length) in enumerate(sub_pass_schedule):
            if offset <= abs_pos < offset + length:
                return sp_idx
        return len(sub_pass_schedule) - 1

    registry = []
    for p in prompt_token_positions:
        registry.append((p, f"p_{p - start_pos}", "prompt", find_subpass(p)))
    registry.append((prompt_boundary, "prompt_boundary", "prompt_boundary", find_subpass(prompt_boundary)))
    for i, p in enumerate(thought_positions):
        registry.append((p, f"T_{i+1}", "thought", find_subpass(p)))
    registry.append((answer_boundary, "answer_boundary", "answer_boundary", find_subpass(answer_boundary)))

    # Joint sites: slab over multiple positions. abs_pos sentinel = -1.
    # joint_prompt covers the SAME prompt window the per-position rows
    # cover (i.e., last_n or all, controlled by prompt_coverage).
    joint_positions = {
        "joint_prompt":  list(prompt_token_positions),
        "joint_thought": list(thought_positions),
    }
    registry.append((-1, "joint_prompt",  "joint_prompt",  -1))
    registry.append((-1, "joint_thought", "joint_thought", -1))

    return registry, joint_positions


def build_position_registry_codi(codi_dict, sample, n_thoughts,
                                 sub_pass_schedule, prompt_len_L,
                                 prompt_coverage, last_n):
    """CODI position registry. sub_pass_schedule from codi_forward."""
    L = prompt_len_L
    K = n_thoughts
    bot_pos = L - 1
    thought_positions = [L + i - 1 for i in range(1, K + 1)]
    answer_boundary = L + K
    prompt_token_positions = list(range(0, bot_pos))
    if prompt_coverage == "last_n":
        prompt_token_positions = prompt_token_positions[-last_n:] if last_n > 0 else []

    def find_subpass(abs_pos):
        for sp_idx, (offset, length) in enumerate(sub_pass_schedule):
            if offset <= abs_pos < offset + length:
                return sp_idx
        return len(sub_pass_schedule) - 1

    registry = []
    for p in prompt_token_positions:
        registry.append((p, f"p_{p - bot_pos}", "prompt", find_subpass(p)))
    registry.append((bot_pos, "prompt_boundary", "prompt_boundary", find_subpass(bot_pos)))
    for i, p in enumerate(thought_positions):
        registry.append((p, f"T_{i+1}", "thought", find_subpass(p)))
    registry.append((answer_boundary, "answer_boundary", "answer_boundary", find_subpass(answer_boundary)))

    joint_positions = {
        "joint_prompt":  list(prompt_token_positions),
        "joint_thought": list(thought_positions),
    }
    registry.append((-1, "joint_prompt",  "joint_prompt",  -1))
    registry.append((-1, "joint_thought", "joint_thought", -1))

    return registry, joint_positions


# ═════════════════════════════════════════════════════════════════════
# Trace a single instance: clean + corrupted + per-site patched
# ═════════════════════════════════════════════════════════════════════

def trace_instance(
    *,
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, partner_sample,
    n_thoughts, device, task, model_name,
    start_id, latent_id, end_id,
    layers_to_trace,
    components_to_trace,
    granularity,                # 'single' or 'window'
    window_size,                # 3 for window
    rng,                        # numpy rng (unused under symbol_swap; kept for API)
    prompt_coverage,            # 'all' or 'last_n'
    last_n,                     # int
    debug=False,                # if True, print debug trace for this instance
    instance_id=None,           # for debug printout
    partner_id=None,            # for debug printout
    batch_size=1,               # if >1, batch sites within an instance
    verify_batched=False,       # if True, compare batched KL vs unbatched
):
    """
    Run CLEAN + CORR + per-site PATCHED forwards, greedy-decoding N
    tokens from A_b after each, and computing per-position KL.

    Returns a dict with the per-instance save schema:
        instance_id, partner_id           # set by caller
        position_labels, position_kinds, position_abs
        clean_correct, corrupted_correct
        gold_tokens, n_format_prefix
        kl_clean_corr      : (N,)                                 fp32
        kl_clean_patched   : (n_layers, n_components, n_positions, N)  fp32
        gen_clean          : list[int]  length N
        gen_corr           : list[int]  length N
        gen_patched        : (n_layers, n_components, n_positions, N) int32
        n_sub_passes_clean
    """
    # ── Resolve model handles ──
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

    # ── Gold answer tokens (no format prefix) and format-prefix length ──
    gold_tokens = get_gold_tokens(active_tokenizer, sample, task)
    # Format prefix is tokenizer-derived (GPT-2 vs Llama IDs differ).
    expected_prefix = format_prefix_tokens(active_tokenizer, task, model_name)
    n_format_prefix = len(expected_prefix)
    # True layer count for this model (was hardcoded N_LAYERS=12 GPT-2).
    n_model_layers = _n_layers(blocks)

    # ── CLEAN run: record activations, greedy-decode N tokens ──
    recorder = ActivationRecorder(blocks)
    input_ids_clean = None  # populated in Coconut branch; None for CODI
    if is_codi:
        sub_pass_schedule_clean, ans_logits_clean, past_kv_clean, prompt_len_L = codi_forward(
            codi_dict, sample, n_thoughts, device, recorder=recorder,
        )
        registry, joint_positions = build_position_registry_codi(
            codi_dict, sample, n_thoughts, sub_pass_schedule_clean,
            prompt_len_L, prompt_coverage, last_n,
        )
    else:
        input_ids_clean, _ = build_coconut_inputs(
            coconut_model, tokenizer, sample, n_thoughts, device,
            start_id, latent_id, end_id,
        )
        _, sub_pass_schedule_clean, ans_logits_clean, past_kv_clean = coconut_forward_with_recorder(
            coconut_model, base_model, input_ids_clean, n_thoughts, device, recorder=recorder,
        )
        registry, joint_positions = build_position_registry_coconut(
            coconut_model, input_ids_clean, n_thoughts, sub_pass_schedule_clean,
            prompt_coverage, last_n,
        )

    gen_clean_ids, dist_clean = greedy_decode(
        decoder_model, ans_logits_clean, past_kv_clean, N_GENERATE, device,
        vocab_limit=vocab_limit,
    )
    gen_clean_list = gen_clean_ids.detach().cpu().tolist()

    # Format-prefix sanity check: emit a warning (not an error) if the
    # model's first n_format_prefix generated tokens don't match the
    # expected fixed prefix. The KL math is still well-defined; this
    # only signals a regime mismatch (e.g., wrong checkpoint).
    # expected_prefix computed above (tokenizer-derived).
    observed_prefix = gen_clean_list[:n_format_prefix]
    if observed_prefix != expected_prefix:
        warnings.warn(
            f"format-prefix mismatch for ({task},{model_name}): "
            f"expected {expected_prefix}, got {observed_prefix}",
            RuntimeWarning,
        )

    # ── Correctness: full greedy decode via the standard inference path ──
    if is_codi:
        identity_fn = lambda h, t: h
        clean_decode = run_codi_single_alpha(
            codi_dict, sample, n_thoughts, device, identity_fn, task=task,
        )
    else:
        clean_decode = run_normal_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device,
            start_id=start_id, latent_id=latent_id, task=task,
        )
    clean_correct = bool(clean_decode["is_correct"])

    # ── CORRUPTED run (no patches), greedy-decode N tokens ──
    #
    # Driven by _run_patched_forward with patchers=[] for symbol_swap: this
    # uses the CLEAN grid with partner-prompt embeddings substituted into
    # the prompt-region positions, so abs_pos coordinates are stable
    # across clean / corr / patched.
    ans_logits_corr, past_kv_corr = _run_patched_forward(
        is_codi, coconut_model, base_model, tokenizer, codi_dict,
        sample, partner_sample, n_thoughts, device,
        start_id, latent_id, end_id,
        patchers=[],
    )
    gen_corr_ids, dist_corr = greedy_decode(
        decoder_model, ans_logits_corr, past_kv_corr, N_GENERATE, device,
        vocab_limit=vocab_limit,
    )
    gen_corr_list = gen_corr_ids.detach().cpu().tolist()

    # Corrupted correctness: does the corrupted (partner) prompt's full
    # decode still emit the CLEAN gold? If yes, corruption failed.
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

    # ── Per-position KL(clean || corr) ──
    kl_clean_corr = _per_position_kl(dist_clean, dist_corr)  # (N,) fp32

    # ── Iterate sites; per-site patched forward + decode + per-position KL ──
    n_layers = len(layers_to_trace)
    n_components = len(components_to_trace)
    n_positions = len(registry)

    kl_clean_patched = np.full(
        (n_layers, n_components, n_positions, N_GENERATE), np.nan, dtype=np.float32,
    )
    gen_patched = np.full(
        (n_layers, n_components, n_positions, N_GENERATE), -1, dtype=np.int32,
    )

    if granularity == "window":
        def layers_for_site(L_center):
            half = window_size // 2
            return [L for L in range(L_center - half, L_center + half + 1) if 0 <= L < n_model_layers]
    else:
        def layers_for_site(L_center):
            return [L_center]

    # ── Sub-pass lookup helper for joint sites ──
    # A joint slab spans many abs_positions; each lives in its own sub-pass.
    # Returns sp_idx for any abs_pos, mirroring find_subpass in the registry
    # builders. Used only for joint rows.
    def _find_subpass(abs_pos):
        for sp_idx, (offset, length) in enumerate(sub_pass_schedule_clean):
            if offset <= abs_pos < offset + length:
                return sp_idx
        return len(sub_pass_schedule_clean) - 1

    def _lookup_clean_value(tL, comp, abs_pos):
        """Look up the clean cached activation at (layer tL, component comp,
        abs_pos). Returns Tensor(D,) or None if unavailable."""
        sp_idx = _find_subpass(abs_pos)
        k = (tL, comp)
        if k not in recorder.cache or sp_idx >= len(recorder.cache[k]):
            return None
        sub_act = recorder.cache[k][sp_idx]
        local_pos = abs_pos - sub_pass_schedule_clean[sp_idx][0]
        if not (0 <= local_pos < sub_act.shape[0]):
            return None
        return sub_act[local_pos]

    # ── Site iteration: unbatched (per-site) or batched (chunked) ──
    #
    # Both paths share the same lookup_clean_value / layers_for_site
    # helpers and write into the same kl_clean_patched / gen_patched
    # arrays. The unbatched path is the reference; the batched path is
    # numerically equivalent (verified by --verify_batched, which runs
    # both and compares).
    if batch_size <= 1:
        for li, L_center in enumerate(layers_to_trace):
            target_layers = layers_for_site(L_center)
            for ci, comp in enumerate(components_to_trace):
                for pi, (abs_pos, label, kind, sp_idx) in enumerate(registry):
                    is_joint = kind in ("joint_prompt", "joint_thought")

                    if is_joint:
                        slab_positions = joint_positions[label]
                        if not slab_positions:
                            continue
                        patchers = []
                        ok = True
                        for tL in target_layers:
                            values_per_abs = {}
                            for ap in slab_positions:
                                v = _lookup_clean_value(tL, comp, ap)
                                if v is None:
                                    ok = False
                                    break
                                values_per_abs[ap] = v
                            if not ok:
                                break
                            patchers.append(MultiPositionActivationPatcher(
                                blocks=blocks, layers=[tL], component=comp,
                                abs_positions=slab_positions,
                                values_per_abs_pos=values_per_abs,
                            ))
                        if not ok:
                            continue
                    else:
                        values_per_layer = {}
                        ok = True
                        for tL in target_layers:
                            v = _lookup_clean_value(tL, comp, abs_pos)
                            if v is None:
                                ok = False
                                break
                            values_per_layer[tL] = v
                        if not ok:
                            continue
                        patchers = [
                            ActivationPatcher(
                                blocks=blocks, layers=[tL], component=comp,
                                abs_pos=abs_pos, value=values_per_layer[tL],
                            )
                            for tL in target_layers
                        ]

                    ans_logits_patched, past_kv_patched = _run_patched_forward(
                        is_codi, coconut_model, base_model, tokenizer, codi_dict,
                        sample, partner_sample, n_thoughts, device,
                        start_id, latent_id, end_id,
                        patchers=patchers,
                    )
                    gen_patched_ids, dist_patched = greedy_decode(
                        decoder_model, ans_logits_patched, past_kv_patched, N_GENERATE, device,
                        vocab_limit=vocab_limit,
                    )

                    kl_clean_patched[li, ci, pi, :] = _per_position_kl(dist_clean, dist_patched)
                    gen_patched[li, ci, pi, :] = gen_patched_ids.detach().cpu().numpy().astype(np.int32)
    else:
        # ── Batched path ──
        # Flatten sites into a list, chunk by batch_size, run one
        # batched forward+decode per chunk, scatter results back into
        # the (n_layers, n_components, n_positions, N) grids.
        from experiments.causal_tracing.batched_patch import (
            BatchedActivationPatcher, build_batched_row_specs,
            _batched_run_patched_forward, _batched_greedy_decode,
            _per_position_kl_batched,
        )

        sites = []  # (li, ci, pi, L_center, comp, abs_pos, kind, label)
        for li, L_center in enumerate(layers_to_trace):
            for ci, comp in enumerate(components_to_trace):
                for pi, (abs_pos, label, kind, sp_idx) in enumerate(registry):
                    sites.append((li, ci, pi, L_center, comp, abs_pos, kind, label))

        # Build a flat list of (L_center, comp, pi, abs_pos, kind, label)
        # tuples matching the builder's expected schema, plus a parallel
        # index list (li, ci, pi) for scattering results.
        site_descs = [(L_center, comp, pi, abs_pos, kind, label)
                      for (li, ci, pi, L_center, comp, abs_pos, kind, label) in sites]
        site_index = [(li, ci, pi)
                      for (li, ci, pi, L_center, comp, abs_pos, kind, label) in sites]

        specs_all, kept_idx, skipped_idx = build_batched_row_specs(
            site_descs, layers_for_site, _lookup_clean_value, joint_positions,
        )
        print(f"    [batched] {len(specs_all)} sites kept, "
              f"{len(skipped_idx)} skipped, batch_size={batch_size}", flush=True)
        # Skipped sites: leave their (kl, gen) entries as NaN/-1 default,
        # exactly as the unbatched path does (those sites hit `continue`).

        # Chunk and run.
        n_chunks = (len(specs_all) + batch_size - 1) // batch_size
        for chunk_i, chunk_start in enumerate(range(0, len(specs_all), batch_size)):
            chunk_specs = specs_all[chunk_start:chunk_start + batch_size]
            chunk_site_idx = [site_index[kept_idx[chunk_start + i]]
                              for i in range(len(chunk_specs))]
            B = len(chunk_specs)
            print(f"      [batched] chunk {chunk_i+1}/{n_chunks}  "
                  f"(B={B}, sites {chunk_start}..{chunk_start+B})", flush=True)
            batched_patcher = BatchedActivationPatcher(
                blocks=blocks, batch_size=B, row_specs=chunk_specs,
            )
            ans_logits_b, past_kv_b = _batched_run_patched_forward(
                is_codi, coconut_model, base_model, tokenizer, codi_dict,
                sample, partner_sample, n_thoughts, device,
                start_id, latent_id, end_id,
                batched_patcher=batched_patcher,
                build_coconut_inputs_fn=build_coconut_inputs,
            )
            gen_ids_b, dist_b = _batched_greedy_decode(
                decoder_model, ans_logits_b, past_kv_b, N_GENERATE, device,
                vocab_limit=vocab_limit,
            )
            kl_b = _per_position_kl_batched(dist_clean, dist_b)   # (B, N)
            gen_b = gen_ids_b.detach().cpu().numpy().astype(np.int32)  # (B, N)
            for bi, (li, ci, pi) in enumerate(chunk_site_idx):
                kl_clean_patched[li, ci, pi, :] = kl_b[bi]
                gen_patched[li, ci, pi, :] = gen_b[bi]

        if verify_batched:
            # Re-run a small sample of sites unbatched and compare.
            # Limited to first min(8, n_kept) kept sites to bound cost.
            #
            # Tolerance is dtype-aware. The batched (B>1) and unbatched
            # (B=1) forwards are algebraically identical but execute
            # different GEMM/attention kernels, so in bf16 (Llama) the
            # logits — hence the per-position KL — differ by kernel-level
            # rounding that does NOT indicate a logic bug. We therefore:
            #   - in fp32 (GPT-2): keep the strict absolute check (1e-4);
            #   - in bf16/fp16:    check a RELATIVE diff against the KL
            #                      scale, and treat token-identity (argmax
            #                      agreement) as the primary correctness
            #                      signal.
            n_check = min(8, len(kept_idx))
            print(f"[verify_batched] checking {n_check} sites against unbatched reference", flush=True)

            # Detect the compute dtype actually used by the model.
            try:
                model_dtype = next(decoder_model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float32
            is_low_precision = model_dtype in (torch.bfloat16, torch.float16)

            max_abs_diff = 0.0
            max_rel_diff = 0.0
            max_gen_diff = 0
            per_site = []
            for ki in range(n_check):
                si = kept_idx[ki]
                L_center, comp, pi, abs_pos, kind, label = site_descs[si]
                li, ci, _pi = site_index[si]
                target_layers = layers_for_site(L_center)
                is_joint = kind in ("joint_prompt", "joint_thought")
                if is_joint:
                    slab_positions = joint_positions[label]
                    patchers = []
                    for tL in target_layers:
                        vpa = {ap: _lookup_clean_value(tL, comp, ap) for ap in slab_positions}
                        patchers.append(MultiPositionActivationPatcher(
                            blocks=blocks, layers=[tL], component=comp,
                            abs_positions=slab_positions, values_per_abs_pos=vpa,
                        ))
                else:
                    patchers = [
                        ActivationPatcher(
                            blocks=blocks, layers=[tL], component=comp,
                            abs_pos=abs_pos,
                            value=_lookup_clean_value(tL, comp, abs_pos),
                        )
                        for tL in target_layers
                    ]
                a_logits, pkv = _run_patched_forward(
                    is_codi, coconut_model, base_model, tokenizer, codi_dict,
                    sample, partner_sample, n_thoughts, device,
                    start_id, latent_id, end_id, patchers=patchers,
                )
                gen_ref, dist_ref = greedy_decode(
                    decoder_model, a_logits, pkv, N_GENERATE, device,
                    vocab_limit=vocab_limit,
                )
                kl_ref = _per_position_kl(dist_clean, dist_ref)        # (N,)
                kl_bat = kl_clean_patched[li, ci, pi, :]               # (N,)
                abs_diff = float(np.max(np.abs(kl_ref - kl_bat)))
                # Relative diff against the local KL scale (avoid /0).
                scale = float(np.max(np.abs(kl_ref))) + 1e-6
                rel_diff = abs_diff / scale
                gen_diff = int(np.sum(
                    gen_ref.detach().cpu().numpy().astype(np.int32) != gen_patched[li, ci, pi, :]
                ))
                max_abs_diff = max(max_abs_diff, abs_diff)
                max_rel_diff = max(max_rel_diff, rel_diff)
                max_gen_diff = max(max_gen_diff, gen_diff)
                per_site.append((ki, comp, abs_diff, rel_diff, gen_diff))

            print(f"[verify_batched] dtype={model_dtype}  "
                  f"max|abs KL diff|={max_abs_diff:.3e}  "
                  f"max rel KL diff={max_rel_diff:.3e}  "
                  f"max token mismatches={max_gen_diff}", flush=True)
            for (ki, comp, ad, rd, gd) in per_site:
                print(f"    site {ki}: comp={comp:9s} abs={ad:.3e} rel={rd:.3e} tok_mismatch={gd}",
                      flush=True)

            # Primary correctness signal: the decoded tokens must match
            # exactly in BOTH precision regimes. A mislocated/incorrect
            # patch flips the argmax; rounding does not.
            assert max_gen_diff == 0, \
                f"batched/unbatched token mismatch on {max_gen_diff} positions"

            if is_low_precision:
                # bf16/fp16: accept kernel-level rounding. Flag only a
                # gross relative divergence that would signal a real bug
                # despite matching tokens.
                if max_rel_diff > 0.25:
                    warnings.warn(
                        f"batched/unbatched KL rel-diff {max_rel_diff:.3e} "
                        f"is large even for {model_dtype}; tokens still match. "
                        f"Inspect per-site output above.",
                        RuntimeWarning,
                    )
            else:
                # fp32: paths should be near-identical.
                assert max_abs_diff < 1e-4, \
                    f"batched/unbatched KL mismatch {max_abs_diff:.3e} (fp32)"

    # Free cache
    recorder.cache.clear()

    if debug:
        _debug_print_instance(
            instance_id=instance_id, partner_id=partner_id,
            task=task, model_name=model_name,
            sample=sample, partner_sample=partner_sample,
            tokenizer=(active_tokenizer if not is_codi else codi_dict["tokenizer"]),
            input_ids_clean=input_ids_clean,
            sub_pass_schedule_clean=sub_pass_schedule_clean,
            registry=registry,
            joint_positions=joint_positions,
            gen_clean_list=gen_clean_list,
            gen_corr_list=gen_corr_list,
            expected_prefix=expected_prefix,
            observed_prefix=observed_prefix,
            n_format_prefix=n_format_prefix,
            gold_tokens=list(gold_tokens),
            kl_clean_corr=kl_clean_corr,
            kl_clean_patched=kl_clean_patched,
            layers_to_trace=layers_to_trace,
            components_to_trace=components_to_trace,
            clean_correct=clean_correct,
            corrupted_correct=corrupted_correct,
        )

    return {
        "position_labels": [r[1] for r in registry],
        "position_kinds":  [r[2] for r in registry],
        "position_abs":    [r[0] for r in registry],
        "joint_positions": joint_positions,    # dict[label -> list[int]]
        "clean_correct":     clean_correct,
        "corrupted_correct": corrupted_correct,
        "gold_tokens":      list(gold_tokens),
        "n_format_prefix":  int(n_format_prefix),
        "kl_clean_corr":    kl_clean_corr,
        "kl_clean_patched": kl_clean_patched,
        "gen_clean":        gen_clean_list,
        "gen_corr":         gen_corr_list,
        "gen_patched":      gen_patched,
        "n_sub_passes_clean": len(sub_pass_schedule_clean),
    }


# ═════════════════════════════════════════════════════════════════════
# Patched forward: drive corrupted forward with active patchers
# ═════════════════════════════════════════════════════════════════════
#
# Returns (answer_boundary_logits, past_key_values_after_A_b) so the
# caller can immediately greedy-decode from A_b. The KV cache covers
# everything up to and including A_b.
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_patched_forward(
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, partner_sample, n_thoughts, device,
    start_id, latent_id, end_id,
    *, patchers,
):
    """
    Run a corrupted (symbol_swap) forward with one or more patchers active.

    Symbol_swap: build embeddings on the CLEAN sample's grid (preserving
    abs_pos meaning across runs) but substitute partner-prompt embeddings
    into the prompt-region positions.

    Returns:
        ans_logits     : (V,) — logits at A_b (full vocab, no slicing)
        past_kv        : KV cache up to and including A_b
    """
    if is_codi:
        return _run_patched_forward_codi(
            codi_dict, sample, partner_sample, n_thoughts, device,
            patchers=patchers,
        )

    # ── Coconut/PaT: build embeddings on the clean grid ──
    input_ids, q_len = build_coconut_inputs(
        coconut_model, tokenizer, sample, n_thoughts, device,
        start_id, latent_id, end_id,
    )
    embedding = coconut_model.embedding
    embeds = embedding(input_ids).clone()

    # Substitute partner prompt into the prompt-region positions.
    # Must use the SAME tokenization as the clean grid (build_coconut_inputs
    # -> tokenize_question_for_recurrence) so partner/clean prompt lengths
    # are comparable and abs_pos meaning is preserved across runs.
    partner_q_tokens = tokenize_question_for_recurrence(
        tokenizer, partner_sample["question"]
    )
    partner_ids = torch.tensor([partner_q_tokens], device=device)
    partner_embeds = embedding(partner_ids).clone()
    if partner_embeds.shape[1] >= q_len:
        embeds[0, :q_len, :] = partner_embeds[0, :q_len, :]
    else:
        # Right-pad with mean partner embedding
        mean_e = partner_embeds[0].mean(dim=0)
        embeds[0, :partner_embeds.shape[1], :] = partner_embeds[0]
        for p in range(partner_embeds.shape[1], q_len):
            embeds[0, p, :] = mean_e

    return _coconut_patched_forward_inline(
        coconut_model, base_model, input_ids, embeds, n_thoughts, device, patchers,
    )


@torch.no_grad()
def _coconut_patched_forward_inline(
    coconut_model, base_model, input_ids, embeds, n_thoughts, device, patchers,
):
    """
    Coconut multi-pass forward with patchers active.

    Returns (ans_logits, past_kv) where ans_logits is the logits at A_b
    and past_kv is the KV cache after the final sub-pass.
    """
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

    # Continuous Coconut
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
                past_kv_trim = _kv_trim(kv_cache, cur_start)
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
                past_kv_trim = _kv_trim(kv_cache, final_start)
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

        # Recurrence
        next_latent_pos = latent_indices[pass_idx]
        hidden_states = outputs.hidden_states[-1]
        local_pos = next_latent_pos - 1 - cur_start
        recurrent_h = hidden_states[0, local_pos, :]
        embeds[0, next_latent_pos, :] = recurrent_h
        next_compute_range = (next_latent_pos, next_latent_pos + 1)
        pass_idx += 1

    return last_step_logits, kv_cache


@torch.no_grad()
def _run_patched_forward_codi(codi_dict, sample, partner_sample, n_thoughts, device,
                              *, patchers):
    """CODI corrupted (symbol_swap) forward with patchers active.

    Returns (ans_logits, past_kv).
    """
    model = codi_dict["model"]
    prj = codi_dict["prj"]
    tokenizer = codi_dict["tokenizer"]
    bot_id = codi_dict["bot_id"]
    eot_id = codi_dict["eot_id"]
    embedding_fn = codi_dict["embedding_fn"]
    use_prj = codi_dict["use_prj"]
    remove_eos = codi_dict["remove_eos"]

    # Clean prompt grid
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

    # Symbol_swap: substitute partner prompt
    partner_q_tokens = tokenizer.encode(
        partner_sample["question"].strip().replace("  ", " "), add_special_tokens=True,
    )
    partner_ids = torch.tensor([partner_q_tokens], device=device)
    partner_embeds = embedding_fn(partner_ids).clone()
    q_len = L - 1  # not counting bot
    if partner_embeds.shape[1] >= q_len:
        embeds[0, :q_len, :] = partner_embeds[0, :q_len, :]
    else:
        mean_e = partner_embeds[0].mean(dim=0)
        embeds[0, :partner_embeds.shape[1], :] = partner_embeds[0]
        for p in range(partner_embeds.shape[1], q_len):
            embeds[0, p, :] = mean_e

    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    # Sub-pass 0
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

    # eot pass
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
    # Full vocab returned; greedy_decode applies the uniform vocab_limit slice.
    return outputs.logits[0, -1, :], outputs.past_key_values


# ═════════════════════════════════════════════════════════════════════
# Per-rank worker
# ═════════════════════════════════════════════════════════════════════

def _trace_worker(rank, world_size, args, return_dir):
    """Per-GPU worker: load model, trace shard, save shard NPZ."""
    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)
        _sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = f"cuda:{rank}" if torch.cuda.is_available() and args.n_gpus > 0 else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    print(f"[rank {rank}] device={device}  model={args.model}  task={args.task}", flush=True)

    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(args.task, device, family=args.model_family)
        coconut_model = base_model = tokenizer = None
        start_id = latent_id = end_id = None
        blocks_for_count = _get_blocks(codi_dict["model"])
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, device, family=args.model_family)
        codi_dict = None
        blocks_for_count = _get_blocks(coconut_model)

    data = load_data(args.task, max_instances=args.max_instances)
    n_total = len(data)
    indices = _shard_indices(n_total, world_size, rank)
    print(f"[rank {rank}] {len(indices)}/{n_total} instances assigned")

    rng = np.random.default_rng(args.seed + rank)

    # Pre-pick corruption partners
    partner_idx_map = {}
    for i in indices:
        partner_idx_map[i] = find_corruption_partner(data, i, args.task, rng)

    # Trace every layer of the loaded model (GPT-2: 12; Llama: 16).
    layers_to_trace = list(range(_n_layers(blocks_for_count)))
    components_to_trace = list(COMPONENTS)

    # Granularities to run (corruption is symbol_swap only)
    grans = []
    if args.granularity in ("single", "both"):
        grans.append("single")
    if args.granularity in ("window", "both"):
        grans.append("window")

    for gran in grans:
        print(f"\n[rank {rank}] ▶ tracing: granularity={gran}", flush=True)
        per_inst = []
        t0 = time.time()
        for c, i in enumerate(indices):
            sample = data[i]
            partner_i = partner_idx_map[i]
            if partner_i is None:
                print(f"[rank {rank}] WARN: no partner for instance {i}; skipping")
                continue
            partner_sample = data[partner_i]

            try:
                out = trace_instance(
                    is_codi=is_codi, coconut_model=coconut_model, base_model=base_model,
                    tokenizer=tokenizer, codi_dict=codi_dict,
                    sample=sample, partner_sample=partner_sample,
                    n_thoughts=args.n_thoughts, device=device, task=args.task,
                    model_name=args.model,
                    start_id=start_id, latent_id=latent_id, end_id=end_id,
                    layers_to_trace=layers_to_trace,
                    components_to_trace=components_to_trace,
                    granularity=gran, window_size=args.window_size,
                    rng=rng,
                    prompt_coverage=args.prompt_coverage,
                    last_n=args.last_n,
                    debug=(args.debug and c == 0),
                    instance_id=i,
                    partner_id=partner_i,
                    batch_size=args.batch_size,
                    verify_batched=(args.verify_batched and c == 0),
                )
                out["instance_id"] = i
                out["partner_id"] = partner_i
                per_inst.append(out)
            except Exception as e:
                print(f"[rank {rank}] ERROR instance {i}: {e}")
                continue

            elapsed = time.time() - t0
            done = c + 1
            sec_per_inst = elapsed / done
            eta = (len(indices) - done) * sec_per_inst
            print(f"[rank {rank}]   {done}/{len(indices)}  "
                  f"{sec_per_inst:.0f}s/instance  ETA={eta/60:.1f}m",
                  flush=True)

        if not per_inst:
            print(f"[rank {rank}] no instances completed for {gran}; skipping save")
            continue

        save_path = return_dir / f"trace_{args.task}_{args.model}_symbol_swap_{gran}_rank{rank}.npz"
        try:
            _save_shard(save_path, per_inst, args)
            print(f"[rank {rank}] saved {len(per_inst)} instances -> {save_path}")
        except Exception as e:
            import pickle
            fallback_path = save_path.with_suffix(".pkl")
            with open(fallback_path, "wb") as f:
                pickle.dump({"per_inst": per_inst, "args": vars(args)}, f)
            print(f"[rank {rank}] _save_shard FAILED ({e!r}); "
                  f"raw per_inst pickled -> {fallback_path}", flush=True)
            raise


def _to_object_array(items):
    """Build a 1-D object array of length len(items) holding arbitrary ndarrays.

    # Why this exists:
    # np.array(list_of_ndarrays, dtype=object) does NOT reliably produce a
    # 1-D object array when the input arrays share some leading dims but
    # differ on a trailing dim. Pre-allocating an empty(N, dtype=object) and
    # assigning element-wise forces NumPy to treat each entry as opaque and
    # skip the broadcast attempt.
    """
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


def _save_shard(path, per_inst, args):
    """Save shard NPZ. Uniform layout when all instances share n_positions
    (stackable along axis 0); object-array layout otherwise.
    """
    # Per-instance variable-shape fields:
    #   kl_clean_patched : (L, C, P_i, N)
    #   gen_patched      : (L, C, P_i, N)
    #   position_labels  : (P_i,)
    #   position_kinds   : (P_i,)
    #   position_abs     : (P_i,)
    # Per-instance fixed-shape fields:
    #   kl_clean_corr    : (N,)
    #   gen_clean        : (N,)
    #   gen_corr         : (N,)
    #   gold_tokens      : list (variable length)
    # Per-instance scalars:
    #   clean_correct, corrupted_correct, n_format_prefix,
    #   instance_id, partner_id, n_sub_passes_clean
    nps = set(len(p["position_labels"]) for p in per_inst)
    uniform = (len(nps) == 1)

    # Always store gold_tokens as object array (variable length)
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
    )  # (N_inst, N)

    # joint_positions: per-instance, list[int] for each joint label.
    # Lengths differ across instances when last_n is capped by short prompts,
    # so store as object arrays.
    joint_prompt_positions_arr = _to_object_array(
        [np.asarray(p["joint_positions"].get("joint_prompt", []), dtype=np.int32)
         for p in per_inst]
    )
    joint_thought_positions_arr = _to_object_array(
        [np.asarray(p["joint_positions"].get("joint_thought", []), dtype=np.int32)
         for p in per_inst]
    )

    if uniform:
        kl_clean_patched = np.stack(
            [p["kl_clean_patched"] for p in per_inst], axis=0
        )  # (N_inst, L, C, P, N)
        gen_patched = np.stack(
            [p["gen_patched"] for p in per_inst], axis=0
        )
        np.savez_compressed(
            path,
            kl_clean_corr=kl_clean_corr_arr,
            kl_clean_patched=kl_clean_patched,
            gen_clean=gen_clean_arr,
            gen_corr=gen_corr_arr,
            gen_patched=gen_patched,
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
        )
    else:
        kl_patched_obj = _to_object_array([p["kl_clean_patched"] for p in per_inst])
        gen_patched_obj = _to_object_array([p["gen_patched"] for p in per_inst])
        labels_obj = _to_object_array(
            [np.asarray(p["position_labels"]) for p in per_inst]
        )
        kinds_obj = _to_object_array(
            [np.asarray(p["position_kinds"]) for p in per_inst]
        )
        abs_obj = _to_object_array(
            [np.asarray(p["position_abs"], dtype=np.int32) for p in per_inst]
        )
        np.savez_compressed(
            path,
            kl_clean_corr=kl_clean_corr_arr,
            kl_clean_patched=kl_patched_obj,
            gen_clean=gen_clean_arr,
            gen_corr=gen_corr_arr,
            gen_patched=gen_patched_obj,
            gold_tokens=gold_tokens_arr,
            n_format_prefix=np.array([p["n_format_prefix"] for p in per_inst], dtype=np.int32),
            position_labels=labels_obj,
            position_kinds=kinds_obj,
            position_abs=abs_obj,
            joint_prompt_positions=joint_prompt_positions_arr,
            joint_thought_positions=joint_thought_positions_arr,
            clean_correct=np.array([p["clean_correct"] for p in per_inst]),
            corrupted_correct=np.array([p["corrupted_correct"] for p in per_inst]),
            instance_ids=np.array([p["instance_id"] for p in per_inst]),
            partner_ids=np.array([p["partner_id"] for p in per_inst]),
            n_sub_passes_clean=np.array([p["n_sub_passes_clean"] for p in per_inst]),
            uniform=np.array(False),
        )


# ═════════════════════════════════════════════════════════════════════
# Shard merging
# ═════════════════════════════════════════════════════════════════════
#
# After all rank-workers finish, concatenate per-rank shards into one
# merged trace file (rank prefix removed). The per-position KL framework
# does no in-file aggregation here; that happens in score_trace.py.
# ═════════════════════════════════════════════════════════════════════

def _stack_or_object(items):
    """Stack a list of ndarrays along axis 0 if shapes agree; else build a
    1-D object array.
    """
    shapes = {tuple(a.shape) for a in items}
    if len(shapes) == 1:
        return np.stack(items, axis=0), True
    return _to_object_array(items), False


def merge_rank_shards(in_dir, task, model, granularity, delete_shards=True):
    """Combine all rank shards for (task, model, symbol_swap, granularity)
    into one merged .npz with the rank prefix removed.

    Returns the merged file path. Per-shard layout (uniform vs object) is
    resolved into a single layout based on whether all shards have
    matching n_positions across all instances.
    """
    in_dir = Path(in_dir)
    pattern = f"trace_{task}_{model}_symbol_swap_{granularity}_rank*.npz"
    paths = sorted(in_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards matching {pattern} under {in_dir}")

    # Per-instance variable-shape fields (need stack/object handling)
    var_keys = (
        "kl_clean_patched", "gen_patched",
        "position_labels", "position_kinds", "position_abs",
        "gold_tokens",
        "joint_prompt_positions", "joint_thought_positions",
    )
    # Per-instance fixed-shape-along-instance fields (concatenate along axis 0)
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

        # Integrity check: every per-instance field's axis-0 length must equal n_inst.
        for k in per_inst_keys + tuple(k for k in concat_keys if k in z.files):
            arr = z[k]
            n_axis0 = int(np.atleast_1d(arr).shape[0])
            assert n_axis0 == n_inst, (
                f"Shard {p.name}: field '{k}' has {n_axis0} entries but "
                f"instance_ids has {n_inst}. Data alignment would be "
                f"corrupted by merging — aborting."
            )

        # Duplicate-instance-id detection across shards
        for iid in np.asarray(z["instance_ids"]).tolist():
            iid = int(iid)
            if iid in seen_instance_ids:
                raise ValueError(
                    f"instance_id {iid} appears in both "
                    f"{seen_instance_ids[iid]} and {p.name}."
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
                # Shared (P,) grid for all instances in this shard
                grid_samples[k].append(arr)
            else:
                # Per-instance grids in object array
                grid_samples[k].append(list(arr))

        for k in concat_keys:
            if k in z.files:
                concat_lists[k].append(np.atleast_1d(z[k]))

    out = {}
    uniform_global = True

    # Stack per-instance variable fields
    for k in per_inst_keys:
        items = var_lists[k]
        stacked, is_unif = _stack_or_object([np.asarray(x) for x in items])
        out[k] = stacked
        if k in ("kl_clean_patched", "gen_patched") and not is_unif:
            uniform_global = False

    # Resolve position grids. Must match numeric layout: if any per-instance
    # numeric field is per-instance (object), grids must also be per-instance,
    # otherwise downstream indexing breaks.
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
        # Build per-instance object array of length N_total
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

    # Final cross-shard audit: every per-instance field has exactly N_total entries.
    expected_N = sum(n for _, n in shard_counts)
    audit_fields = list(per_inst_keys) + list(concat_keys)
    for k in audit_fields:
        if k not in out:
            continue
        n_have = int(np.atleast_1d(out[k]).shape[0])
        assert n_have == expected_N, (
            f"Post-merge audit failed: '{k}' has {n_have} entries; "
            f"expected {expected_N} (sum of per-shard n_inst). "
            f"Per-shard counts: {shard_counts}"
        )
    for k in grid_keys:
        if k not in out:
            continue
        arr = out[k]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            n_have = int(arr.shape[0])
            assert n_have == expected_N, (
                f"Post-merge audit failed: grid '{k}' has {n_have} "
                f"per-instance entries; expected {expected_N}."
            )

    merged_path = in_dir / f"trace_{task}_{model}_symbol_swap_{granularity}.npz"
    np.savez_compressed(merged_path, **out)

    print(f"  [merge audit] {merged_path.name}: "
          f"N={expected_N} from {len(paths)} shard(s) "
          f"({', '.join(f'{name}:{n}' for name, n in shard_counts)})  "
          f"uniform={uniform_global}", flush=True)

    if delete_shards:
        for p in paths:
            p.unlink()

    return merged_path


# ═════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════

def _trace_configs_for_args(args):
    """Mirror _trace_worker's config-set construction so main() knows
    which granularities to merge.
    """
    grans = []
    if args.granularity in ("single", "both"):
        grans.append("single")
    if args.granularity in ("window", "both"):
        grans.append("window")
    return grans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model", choices=["pause", "coconut", "coconut_u", "codi"],
                        required=True)
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Determines checkpoint paths, dtype, "
             "transformer-block layout, and KV-cache type.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Cap on test instances (for debugging).")
    parser.add_argument("--granularity", choices=["single", "window", "both"],
                        default="single")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--prompt_coverage", choices=["all", "last_n"], default="last_n")
    parser.add_argument("--last_n", type=int, default=15)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(OUTPUTS / "causal_trace"))
    parser.add_argument("--debug", action="store_true",
                        help="Print a verbose per-instance trace for the FIRST "
                             "instance of each granularity (positions, sub-pass "
                             "schedule, gen tokens, KL summary). Recommended with "
                             "--max_instances 1 --n_gpus 1.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of sites to batch into one patched forward. "
                             "1 = unbatched (reference behavior). For GPT-2 small "
                             "on seq len ~100, batch sizes up to 64-128 are usually "
                             "fine; tune to fit GPU memory.")
    parser.add_argument("--verify_batched", action="store_true",
                        help="When --batch_size > 1, re-run the first 8 sites of "
                             "the first instance unbatched and assert KL match. "
                             "Use once after changing the batched code path.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.model_family / f"{args.task}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs -> {out_dir}")

    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.n_gpus <= 1:
        _trace_worker(0, 1, args, out_dir)
    else:
        mp.spawn(_trace_worker, args=(args.n_gpus, args, out_dir),
                 nprocs=args.n_gpus, join=True)

    # Merge rank shards. Offline scoring happens in score_trace.py.
    print("\n=== Merging rank shards ===", flush=True)
    for gran in _trace_configs_for_args(args):
        try:
            merged = merge_rank_shards(
                out_dir, args.task, args.model, gran,
                delete_shards=True,
            )
            print(f"  merged -> {merged.name}", flush=True)
        except FileNotFoundError as e:
            print(f"  skip {gran}: {e}", flush=True)
            continue

    print("\nDone. Run score_trace.py to compute IE and produce plots.",
          flush=True)


if __name__ == "__main__":
    main()