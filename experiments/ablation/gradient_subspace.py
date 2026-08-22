"""
Gradient-weighted causal subspace extraction.

This script extracts the per-timestep gradient subspace B_t for a given
(task, model) pair and saves it to disk. Two consumers downstream:

    1. gradient_subspace_interventions.py
       -- ablates / amplifies thoughts along B_t, used in Section 5.
    2. gradient_subspace_diagnosis.py
       -- runs Q1 (alignment of mu_t-mu with B_t) and Q3 (principal-angle
          stability of B_t across t), used in Section 6.

Both consumers read bases.npz from this script's output directory.

Motivation for the gradient subspace
------------------------------------
INLP-based amnesic probing assumes the model's causal structure lives
in a linear subspace recoverable by linear classifiers. When this is
wrong (e.g. nonlinear features, distributed representations, or
context-dependent reads), INLP either under-detects causal directions
or inflates apparent ones. Aggregate evidence from Section 5 -- INLP
and Rand steering matching on PaT/CODI; near-zero ablation drops
despite clear performance reliance -- suggests this assumption is too
strong for the continuous thought regime.

Gradients side-step the linear-feature assumption. The gradient
    g_{i,t} = dL_i / dh_{i,t}     (with downstream thoughts h_{i,>t} held fixed)
is the locally steepest direction in which a small perturbation to
the thought vector h_{i,t} alone would most change the model's loss
on the answer. By construction, this direction is causal (it's the
first-order Taylor coefficient of the model's output around h_{i,t}).
It captures whatever the downstream layers actually read from h_{i,t},
regardless of how the features are encoded.

Note on partial vs. total gradient
----------------------------------
This script computes PARTIAL gradients: each h_t is detached and
re-introduced as a fresh leaf before being fed into the next forward
pass, so the gradient at h_t is computed with all other h_{t'} held
at their actual values. The total gradient (which would also include
the cascade dh_{t+1}/dh_t * dL/dh_{t+1} + ...) is not tractable in
current HF transformers because past_key_values are detached between
forward calls. The partial gradient is the standard quantity used by
Integrated Gradients and most local-attribution methods in mech
interp.

Aggregating gradients across instances at each timestep:
    # G_t = stack g_{i,t} for i = 1..N        shape (N, D)
    # G_t = U S V^T                            (SVD)
    # B_t = top-k right singular vectors V[:, :k]  shape (D, k)

B_t is an orthonormal basis of the subspace that locally drives the
model's output across the test set at timestep t. The rank k is
chosen by an explained-variance threshold (default 95%) so the
subspace captures the bulk of the gradient mass.

Loss target
-----------
L_i is the negative log-likelihood of the gold answer tokens:
    # L_i = - sum_{j=0..M-1} log p(y_{i,j} | prompt, thoughts, y_{i,<j})
This is the cleanest "what would change the right answer" target.
For wrong-prediction instances we still use the gold answer -- the
gradient then points in the direction that would correct the model.

Output layout
-------------
All artefacts go to:
    BASE_DIR / outputs / gradient_geometry / <task> / <model> /
        bases.npz      -- {f"B_t{t}": B_t}, the per-timestep bases
        gradients.npz  -- raw G (N, T, D) gradients, H (N, T, D) thought
                          vectors, losses, ranks
        extraction.json -- extraction stats (no diagnostics)

Usage
-----
    python -u -m experiments.ablation.gradient_subspace --task prosqa --model pause
    python -u -m experiments.ablation.gradient_subspace --task prosqa --model coconut
    python -u -m experiments.ablation.gradient_subspace --task prosqa --model coconut_u
    python -u -m experiments.ablation.gradient_subspace --task prosqa --model codi
    python -u -m experiments.ablation.gradient_subspace --task gsm    --model pause
    python -u -m experiments.ablation.gradient_subspace --task gsm    --model coconut
    python -u -m experiments.ablation.gradient_subspace --task gsm    --model coconut_u
    python -u -m experiments.ablation.gradient_subspace --task gsm    --model codi

    # Llama family (add --model_family llama):
    python -u -m experiments.ablation.gradient_subspace --task gsm \
        --model coconut_u --model_family llama
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, set_seed
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    is_pause_model,
)

# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def get_gold_answer_text(sample, task, model):
    raw = sample.get("answer", "").strip()
    if task == "prosqa":
        return ("The answer is: " if model == "codi" else "### ") + raw
    # gsm
    if model == "codi":
        num = raw.split("####")[-1].strip() if "####" in raw else raw
        return "The answer is: " + num
    # coconut / pause: full CoT (raw already contains '#### N')
    return "### " + raw


# ═══════════════════════════════════════════════════════════════════
# Coconut: gradient extraction with manual recurrence
# ═══════════════════════════════════════════════════════════════════
#
# For each instance we want g_t = dL/dh_t for t = 0..K. We snapshot
# every h_t as a fresh leaf, feed each leaf into the next forward
# pass, and let autograd give us PARTIAL gradients. See module
# docstring for the partial vs. total discussion.
#
#   1. NO torch.no_grad() in the recurrence.
#   2. .detach()+.clone() between recurrence steps -- this is what
#      makes the gradients PARTIAL.
#   3. We register `h_t.retain_grad()` implicitly via the leaf trick
#      so .grad is available after backward(), without hooks.
#

def extract_gradient_coconut(
    base_model, tokenizer, sample, n_thoughts, device,
    start_id, end_id, gold_text,
):
    """
    Run coconut recurrence with autograd, then backprop the gold-answer
    NLL to obtain dL/dh_t at every recurrence step.

    Math:
        # h_t_leaf = h_t.detach().clone().requires_grad_(True)
        # next forward gets h_t_leaf (not h_t)
        # nll = ... (built using all h_t_leaf in the forward chain)
        # g_t = d(nll) / d(h_t_leaf), computed independently per t
    """
    # Defensive: ensure model params have requires_grad=True so the
    # forward pass produces grad-active activations. We never optimize
    # these params; this just keeps autograd alive.
    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    # ── Encode prompt + start_latent ───────────────────────────────
    from src.utils import tokenize_question_for_recurrence
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    input_ids = torch.tensor(
        [question_tokens + [start_id]], device=device,
    )

    # First forward: prompt -> h_0 (state at start_latent position)
    outputs = base_model(
        input_ids=input_ids, output_hidden_states=True, use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]    # (D,)

    # Snapshot h_0 as a leaf.
    h_leaf = h.detach().clone().requires_grad_(True)
    h_leaves = [h_leaf]
    past_kv = outputs.past_key_values
    cur = h_leaf

    # ── Recurrence: each step uses the previous step's leaf ────────
    for t in range(1, n_thoughts + 1):
        ct = cur.unsqueeze(0).unsqueeze(0)           # (1, 1, D)
        outputs = base_model(
            inputs_embeds=ct, past_key_values=past_kv,
            output_hidden_states=True, use_cache=True,
        )
        h_next = outputs.hidden_states[-1][0, 0, :]
        h_next_leaf = h_next.detach().clone().requires_grad_(True)
        h_leaves.append(h_next_leaf)
        cur = h_next_leaf
        past_kv = outputs.past_key_values

    # ── Feed end_latent, decode answer ─────────────────────────────
    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(
        input_ids=end_input, past_key_values=past_kv, use_cache=True,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]

    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    if len(gold_ids) == 0:
        D = h.shape[0]
        return (np.zeros((n_thoughts + 1, D), dtype=np.float32),
                np.zeros((n_thoughts + 1, D), dtype=np.float32), 0.0)

    # NLL on the gold answer:
    #   # nll = - sum_j log p(y_j | prompt, thoughts, y_{<j})
    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    nll = -log_probs_first[gold_ids[0]]

    for j in range(1, len(gold_ids)):
        gold_token_input = torch.tensor([[gold_ids[j - 1]]], device=device)
        out = base_model(
            input_ids=gold_token_input, past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        nll = nll - next_logp[gold_ids[j]]

    # ── Backprop ───────────────────────────────────────────────────
    # allow_unused=True: defensive only. We expect every leaf to reach
    # nll through its forward pass. If it doesn't, we pad with zeros
    # rather than crashing.
    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )

    grads_arr = []
    n_unused = 0
    for t_idx, g in enumerate(grads_t):
        if g is None:
            n_unused += 1
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    if n_unused > 0:
        print(f"  [grad] note: {n_unused}/{len(h_leaves)} leaves "
              f"were unused (zero-padded).")
    grads = np.stack(grads_arr, axis=0)
    hidden = np.stack(
        [hl.detach().to(torch.float32).cpu().numpy() for hl in h_leaves], axis=0,
    )

    return grads, hidden, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# Pause: gradient extraction with single forward pass
# ═══════════════════════════════════════════════════════════════════
#
# For pause models, all thoughts are processed in a single forward
# pass with the learned `pause_embedding` substituted in at thought
# positions. We obtain dL/dh_t by:
#   1. Build inputs_embeds.
#   2. Substitute pause_embedding at thought positions, but do so
#      in a way that preserves a per-position gradient handle.
#   3. Forward, compute NLL on gold answer, backprop.
#

def extract_gradient_pause(
    coconut_model, base_model, tokenizer, sample, n_thoughts, device,
    start_id, latent_id, end_id, gold_text,
):
    """
    Same gradient target as the coconut extractor; single forward pass.

    Strategy:
      For each thought position t we need a tensor h_t whose gradient
      we can read. We do this by initializing h_t = pause_embedding
      and requesting grad on it via .detach().clone().requires_grad_(True),
      then placing each h_t into inputs_embeds with a copy that keeps
      the graph.
    """
    from src.utils import tokenize_question_for_recurrence
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    input_ids_list = (
        question_tokens + [start_id] + [latent_id] * n_thoughts + [end_id]
    )
    input_ids = torch.tensor([input_ids_list], device=device)

    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    embedding = coconut_model.embedding
    inputs_embeds_base = embedding(input_ids)            # (1, L, D)
    pause_emb = coconut_model.pause_embedding             # (D,)

    # Positions:
    #   t=0 corresponds to the <start_latent> position
    #   t=1..K correspond to the K <latent> positions (substituted)
    L = input_ids.shape[1]
    start_latent_pos = len(question_tokens)
    start_of_latent = start_latent_pos + 1
    thought_positions = [start_latent_pos] + [
        start_of_latent + i for i in range(n_thoughts)
    ]

    # Build T leaf tensors with requires_grad=True.
    h_leaves = []
    for t_idx in range(n_thoughts + 1):
        if t_idx == 0:
            # h_0 is the prompt's hidden state at <start_latent>. We
            # can't get a leaf gradient on a hidden state directly,
            # so for the pause model the analogue is gradient through
            # the *embedding at start_latent*. This measures
            # dL / d(input embedding at start_latent), not dL / d(hidden state).
            init = inputs_embeds_base[0, start_latent_pos, :].detach().clone()
        else:
            # All other thought positions: pause_embedding (the same
            # value the model receives at training/eval).
            init = pause_emb.detach().clone()
        leaf = init.requires_grad_(True)
        h_leaves.append(leaf)

    # Splice leaves into inputs_embeds via a fresh tensor that keeps
    # the graph. We can't write to inputs_embeds_base in-place because
    # that breaks autograd.
    embeds_per_pos = []
    for j in range(L):
        if j in thought_positions:
            t_idx = thought_positions.index(j)
            embeds_per_pos.append(h_leaves[t_idx])
        else:
            embeds_per_pos.append(inputs_embeds_base[0, j, :].detach())
    inputs_embeds = torch.stack(embeds_per_pos, dim=0).unsqueeze(0)  # (1, L, D)

    attention_mask = torch.ones((1, L), dtype=torch.long, device=device)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    outputs = base_model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]

    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    if len(gold_ids) == 0:
        D = pause_emb.shape[0]
        return (np.zeros((n_thoughts + 1, D), dtype=np.float32),
                np.zeros((n_thoughts + 1, D), dtype=np.float32), 0.0)

    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    nll = -log_probs_first[gold_ids[0]]

    for j in range(1, len(gold_ids)):
        gold_token_input = torch.tensor([[gold_ids[j - 1]]], device=device)
        out = base_model(
            input_ids=gold_token_input, past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        nll = nll - next_logp[gold_ids[j]]

    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )
    grads_arr = []
    for t_idx, g in enumerate(grads_t):
        if g is None:
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    grads = np.stack(grads_arr, axis=0)
    hidden = np.stack(
        [hl.detach().to(torch.float32).cpu().numpy() for hl in h_leaves], axis=0,
    )

    return grads, hidden, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# CODI: gradient extraction
# ═══════════════════════════════════════════════════════════════════
#
# CODI is structurally similar to coconut but with an additional
# projection prj() between recurrence steps. We grad through prj(h_t)
# the same way as for coconut.
#

def extract_gradient_codi(codi_dict, sample, n_thoughts, device, gold_text, task):
    """
    Same gradient computation as coconut, with CODI's prompt /
    projection conventions (see codi_dict for fields).
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']

    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
    if prj is not None:
        for p in prj.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)

    # ── Build prompt: [question] [eos]? [bot] ──────────────────────
    question_text = sample["question"].strip().replace('  ', ' ')
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L_prompt = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L_prompt, device=device).unsqueeze(0)

    outputs = base_model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True,
        output_hidden_states=True,
    )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]   # h at [bot] position, t=0

    # Snapshot leaf for t=0; preserve original dtype (CODI is bfloat16).
    orig_dtype = h.dtype
    h_leaf = h.detach().clone().to(orig_dtype).requires_grad_(True)
    h_leaves = [h_leaf]
    cur = h_leaf

    latent = cur.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    running_mask = attention_mask
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        pos_t = torch.tensor([[L_prompt + t - 1]], device=device)
        outputs = base_model(
            inputs_embeds=latent, attention_mask=running_mask,
            position_ids=pos_t, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        h_next = outputs.hidden_states[-1][0, -1, :]
        h_next_leaf = h_next.detach().clone().to(orig_dtype).requires_grad_(True)
        h_leaves.append(h_next_leaf)
        cur = h_next_leaf
        latent = cur.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    # ── eot + ([eos]?) ────────────────────────────────────────────
    if remove_eos:
        eot_ids = [eot_id]
    else:
        eot_ids = [eot_id, tokenizer.eos_token_id]
    eot_input = torch.tensor([eot_ids], device=device)
    eot_emb = embedding_fn(eot_input)
    eot_len = eot_emb.size(1)
    eot_pos = torch.arange(L_prompt + n_thoughts,
                           L_prompt + n_thoughts + eot_len,
                           device=device).unsqueeze(0)
    running_mask = torch.cat(
        [running_mask, torch.ones((1, eot_len), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )
    outputs = base_model(
        inputs_embeds=eot_emb, attention_mask=running_mask,
        position_ids=eot_pos, use_cache=True, past_key_values=past_kv,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]
    current_pos = L_prompt + n_thoughts + eot_len

    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    if len(gold_ids) == 0:
        D = h.shape[0]
        return (np.zeros((n_thoughts + 1, D), dtype=np.float32),
                np.zeros((n_thoughts + 1, D), dtype=np.float32), 0.0)

    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    nll = -log_probs_first[gold_ids[0]]

    for j in range(1, len(gold_ids)):
        gold_token_input = torch.tensor([[gold_ids[j - 1]]], device=device)
        gold_emb = embedding_fn(gold_token_input)
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        decode_pos = torch.tensor([[current_pos]], device=device)
        out = base_model(
            inputs_embeds=gold_emb, past_key_values=past_kv,
            use_cache=True, attention_mask=running_mask,
            position_ids=decode_pos,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        nll = nll - next_logp[gold_ids[j]]
        current_pos += 1

    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )
    grads_arr = []
    for t_idx, g in enumerate(grads_t):
        if g is None:
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    grads = np.stack(grads_arr, axis=0)
    hidden = np.stack(
        [hl.detach().to(torch.float32).cpu().numpy() for hl in h_leaves], axis=0,
    )

    return grads, hidden, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# Subspace from a gradient matrix
# ═══════════════════════════════════════════════════════════════════
#
# Given G_t of shape (N, D), compute SVD G_t = U S V^T. The right
# singular vectors V[:, :k] are an orthonormal basis of the subspace
# carrying most of the gradient mass across instances.
#
# Math:
#   # G = U @ diag(S) @ V^T,   U (N, r), S (r,), V (D, r)
#   # energy_i  = S_i^2
#   # cum_var_k = sum_{i<=k} S_i^2 / sum_i S_i^2
#   # k = min { r : cum_var_r >= rho }, rho = explained_variance
#

def subspace_from_gradients(G, explained_variance=0.95, max_rank=None):
    """
    Args:
        G: (N, D) gradient matrix.
        explained_variance: rho in (0, 1], cumulative-energy threshold.
        max_rank: optional cap on rank (useful when D is large).

    Returns:
        B:        (D, k) orthonormal basis (right singular vectors).
        k:        int rank.
        sv:       (min(N,D),) singular values.
        cum_var:  (min(N,D),) cumulative fraction of energy explained.
    """
    G = np.asarray(G, dtype=np.float64)
    # full_matrices=False -> U (N, r), V^T (r, D), where r = min(N, D).
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    energy = S ** 2
    total = energy.sum()
    if total <= 0:
        # Degenerate: zero gradient matrix. Return empty basis.
        D = G.shape[1]
        return np.zeros((D, 0)), 0, S, np.zeros_like(S)
    cum_var = np.cumsum(energy) / total

    # Smallest k such that cum_var[k-1] >= explained_variance
    k = int(np.searchsorted(cum_var, explained_variance) + 1)
    k = min(k, len(S))
    if max_rank is not None:
        k = min(k, max_rank)

    B = Vt[:k, :].T   # (D, k), orthonormal columns
    return B, k, S, cum_var


# ═══════════════════════════════════════════════════════════════════
# JSON-serialisable conversion
# ═══════════════════════════════════════════════════════════════════

def deep_convert(obj):
    if isinstance(obj, dict):
        return {str(k): deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════
# Phase 1: extract gradients across the test set, build bases per t
# ═══════════════════════════════════════════════════════════════════

def extract_subspace(args, data, codi_dict, coconut_model, base_model,
                     tokenizer, latent_id, start_id, end_id, D, pause):
    """
    Returns:
        bases:             dict[t -> (D, k_t) np.ndarray]
        G:                 (N, T, D) raw gradients
        H:                 (N, T, D) raw hidden states (thought vectors h_t)
        losses:            (N,) per-instance NLLs
        ranks:             list[int] of subspace ranks per t
        sv_per_t:          dict[t -> list[float]]  singular values
        cum_var_per_t:     dict[t -> list[float]]  cumulative variance
        n_skipped:         int
    """
    N = len(data)
    K = args.n_thoughts
    T = K + 1
    print(f"[grad] instances={N}  T={T}  D={D}")

    G = np.zeros((N, T, D), dtype=np.float32)
    H = np.zeros((N, T, D), dtype=np.float32)
    losses = np.zeros(N, dtype=np.float32)
    n_skipped = 0

    for idx, sample in enumerate(data):
        if idx % 25 == 0 and idx > 0:
            print(f"  [grad] {idx}/{N}  mean NLL so far: "
                  f"{losses[:idx].mean():.4f}")
        gold_text = get_gold_answer_text(sample, args.task, args.model)
        if not gold_text:
            n_skipped += 1
            continue
        try:
            if codi_dict is not None:
                grads, hidden, loss = extract_gradient_codi(
                    codi_dict, sample, K, args.device, gold_text, args.task,
                )
            elif pause:
                grads, hidden, loss = extract_gradient_pause(
                    coconut_model, base_model, tokenizer, sample, K,
                    args.device, start_id, latent_id, end_id, gold_text,
                )
            else:
                grads, hidden, loss = extract_gradient_coconut(
                    base_model, tokenizer, sample, K, args.device,
                    start_id, end_id, gold_text,
                )
        except Exception as e:
            print(f"  [grad] WARN instance {idx} failed: "
                  f"{type(e).__name__}: {e}")
            n_skipped += 1
            continue

        G[idx] = grads
        H[idx] = hidden
        losses[idx] = loss

    print(f"\n[grad] Extracted gradients for {N - n_skipped}/{N} instances "
          f"(skipped {n_skipped})")
    if (losses > 0).any():
        print(f"[grad] mean NLL: {losses[losses > 0].mean():.4f}")

    # ── SVD per timestep -> bases ──────────────────────────────────
    bases = {}
    ranks = []
    sv_per_t = {}
    cum_var_per_t = {}

    print()
    print(f"[svd] explained_variance threshold rho = {args.explained_variance}")
    print(f"[svd] {'t':>3}  {'rank':>6}  {'sv_max':>10}  {'sv_min_kept':>12}  "
          f"{'cum_var_kept':>13}")
    for t in range(T):
        G_t = G[:, t, :]      # (N, D)
        # Drop rows from failed instances so they don't bias the SVD.
        row_norms = np.linalg.norm(G_t, axis=1)
        mask = row_norms > 0
        G_t_eff = G_t[mask]
        if G_t_eff.shape[0] < 2:
            print(f"  [svd] WARN t={t}: <2 nonzero rows; skipping")
            bases[t] = np.zeros((D, 0))
            ranks.append(0)
            sv_per_t[t] = []
            cum_var_per_t[t] = []
            continue
        B, k, sv, cum_var = subspace_from_gradients(
            G_t_eff,
            explained_variance=args.explained_variance,
            max_rank=args.max_rank,
        )
        bases[t] = B
        ranks.append(k)
        sv_per_t[t] = sv.tolist()
        cum_var_per_t[t] = cum_var.tolist()
        sv_min_kept = sv[k - 1] if k > 0 else 0.0
        cum_var_kept = cum_var[k - 1] if k > 0 else 0.0
        print(f"  [svd] {t:>3}  {k:>6}  {sv[0]:>10.4e}  "
              f"{sv_min_kept:>12.4e}  {cum_var_kept:>13.4f}")

    return bases, G, H, losses, ranks, sv_per_t, cum_var_per_t, n_skipped


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-weighted causal subspace extraction. "
                    "Saves bases.npz and gradients.npz for downstream "
                    "consumers (interventions, diagnosis)."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model",
        choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut_u",
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Determines checkpoint paths and dtype.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Gradient extraction is slower than inference; "
                             "200 is usually enough for a stable subspace.")
    parser.add_argument("--explained_variance", type=float, default=0.95,
                        help="rho in (0,1]: keep top singular vectors that "
                             "together explain at least rho of the gradient "
                             "energy at that timestep.")
    parser.add_argument("--max_rank", type=int, default=None,
                        help="Optional hard cap on subspace rank per t.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = (BASE_DIR / "outputs" / "gradient_geometry"
                  / args.model_family / args.task / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] task={args.task}  model={args.model}  family={args.model_family}")
    print(f"[main] output_dir: {output_dir}")

    is_codi = (args.model == "codi")

    # ── Load model ─────────────────────────────────────────────────
    if is_codi:
        codi_dict = setup_codi_model(args.task, args.device, family=args.model_family)
        D = codi_dict['hidden_size']
        coconut_model = base_model = tokenizer = None
        latent_id = start_id = end_id = None
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device,
                                      family=args.model_family)
        # GPT-2 exposes `n_embd`; Llama exposes `hidden_size`.
        D = getattr(base_model.config, "n_embd",
                    getattr(base_model.config, "hidden_size", None))
        if D is None:
            raise RuntimeError("Could not determine hidden size from model config")

    # We must be outside any no_grad context. setup_*_model puts models
    # in eval() (which disables dropout/batchnorm-train-mode but does
    # NOT disable autograd). Be explicit just in case any helper changed
    # the global grad mode.
    torch.set_grad_enabled(True)

    # ── Load test data ─────────────────────────────────────────────
    data = load_data(args.task, args.max_instances)
    pause = (not is_codi) and is_pause_model(coconut_model)

    # ── Extract gradients + per-timestep SVD ───────────────────────
    bases, G, H, losses, ranks, sv_per_t, cum_var_per_t, n_skipped = \
        extract_subspace(
            args, data, codi_dict, coconut_model, base_model,
            tokenizer, latent_id, start_id, end_id, D, pause,
        )
    N = len(data)
    T = args.n_thoughts + 1

    # ── Save artefacts ─────────────────────────────────────────────
    bases_path = output_dir / "bases.npz"
    np.savez(bases_path, **{f"B_t{t}": bases[t] for t in range(T)})
    print(f"\n[main] Saved bases -> {bases_path}")

    grads_path = output_dir / "gradients.npz"
    np.savez(
        grads_path,
        G=G,                                # (N, T, D) gradients dL/dh_t
        H=H,                                # (N, T, D) thought vectors h_t
        losses=losses,                      # (N,)
        ranks=np.array(ranks),
    )
    print(f"[main] Saved raw gradients + hidden states -> {grads_path}")

    summary = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "n_instances": int(N),
        "n_skipped": int(n_skipped),
        "T": int(T),
        "D": int(D),
        "explained_variance": args.explained_variance,
        "max_rank": args.max_rank,
        "subspace_ranks_per_t": ranks,
        "singular_values_per_t": sv_per_t,
        "cumulative_variance_per_t": cum_var_per_t,
        "mean_nll": (
            float(losses[losses > 0].mean()) if (losses > 0).any() else 0.0
        ),
    }
    summary_path = output_dir / "extraction.json"
    with open(summary_path, "w") as f:
        json.dump(deep_convert(summary), f, indent=2)
    print(f"[main] Saved extraction summary -> {summary_path}")
    print(f"\n[main] Extraction complete. "
          f"Run gradient_subspace_diagnosis.py for Q1/Q3 geometry, or "
          f"gradient_subspace_interventions.py for steering.")


if __name__ == "__main__":
    main()