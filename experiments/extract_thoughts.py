"""
Extract continuous thought vectors from a trained Coconut or CODI model.

For each ProsQA/GSM instance, runs K steps of continuous-thought recurrence
and saves the final hidden state h_t at each step t = 0, ..., K.

Output: a single .pt file containing:
    - "thoughts": Tensor of shape (N, K+1, D)
        N = number of instances, K = recurrence steps,
        D = hidden dim (768 for GPT-2, 2048 for Llama-3.2-1B)
    - "instance_indices": list of ints, mapping row i to its index
    - "n_thoughts": int, the K used

Usage:
    python extract_thoughts.py --model coconut --n_thoughts 6
    python extract_thoughts.py --model codi --task gsm --n_thoughts 6
    python extract_thoughts.py --model coconut_u --model_family llama --n_thoughts 6
"""

import json
import queue
import torch
import argparse
import tempfile
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoTokenizer
from src.utils import (
    is_pause_model, setup_model_and_tokenizer, setup_codi_model,
    _shard_indices,
)
from src.config import (
    PROSQA_TRAIN, PROSQA_TEST, GSM_TRAIN, GSM_TEST, THOUGHTS
)
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )

# ═══════════════════════════════════════════════════════════════════
# Data Formatting
# ═══════════════════════════════════════════════════════════════════

def _data_path(task, split):
    if task == "prosqa":
        return PROSQA_TRAIN if split == "train" else PROSQA_TEST
    return GSM_TRAIN if split == "train" else GSM_TEST


def load_data(task, split="test", max_instances=None):
    path = _data_path(task, split)
    with open(path, "r") as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    print(f"[INFO] Loaded {len(data)} {task} ({split}) instances from {path}")
    return data

# ═══════════════════════════════════════════════════════════════════
# Thought extraction: pause
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_thoughts_single_instance(
    coconut_model, base_model, tokenizer, sample, n_thoughts, device,
    start_id, latent_id, end_id,
):
    """
    Extract thought vectors for a single ProsQA instance.

    For coconut models: runs manual hidden-state recurrence (K steps).
    For pause models: runs a single forward pass and extracts hidden
        states at the thought token positions.

    Returns: Tensor of shape (n_thoughts + 1, D)
        For coconut: [h_0, h_1, ..., h_K] from recurrence
        For pause: hidden_states[-1] at [thought_pos_0, ..., thought_pos_K]
            where thought_pos_0 is the position of <start_latent> (the last
            position before thoughts begin), and thought_pos_1..K are the
            K thought positions.

    Note on indexing alignment:
        For coconut, h_0 = hidden state after processing the prompt
        (at the <start_latent> position), and h_1..K are from recurrence.
        For pause, we extract the same positions: the hidden state at
        <start_latent> and at each <latent> token position.
        This ensures the thought vectors are semantically aligned across
        model types for probing and analysis.
    """
    pause = is_pause_model(coconut_model)
    # GPT-2 exposes `n_embd`; Llama exposes `hidden_size`.
    hidden_dim = getattr(base_model.config, "n_embd",
                         getattr(base_model.config, "hidden_size", None))
    if hidden_dim is None:
        raise RuntimeError("Could not determine hidden size from model config")

    if pause:
        return _extract_thoughts_pause(
            coconut_model, base_model, tokenizer, sample,
            n_thoughts, device, start_id, latent_id, end_id, hidden_dim,
        )
    else:
        return _extract_thoughts_coconut(
            base_model, tokenizer, sample, n_thoughts, 
            device, hidden_dim, start_id
        )


def _extract_thoughts_pause(
    coconut_model, base_model, tokenizer, sample,
    n_thoughts, device, start_id, latent_id, end_id, hidden_dim,
):
    """
    Pause model: single forward pass, extract hidden states at thought positions.

    Builds: [question_tokens] <start_latent> <latent>*K <end_latent>
    Replaces <latent> embeddings with pause_embedding.
    Runs single forward pass.
    Returns hidden_states[-1] at the relevant positions.
    """
    thoughts = torch.zeros(n_thoughts + 1, hidden_dim)

    # Build input sequence with thought tokens.
    # Prompt tokenization is family-aware (Llama uses the chat template,
    # GPT-2 uses raw "{q}\n"); tokenize_question_for_recurrence mirrors
    # dataset.py:_build_question_tokens so positions align with training.
    from src.utils import tokenize_question_for_recurrence
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])

    input_ids_list = (
        question_tokens
        + [start_id]
        + [latent_id] * n_thoughts
        + [end_id]
    )
    input_ids = torch.tensor([input_ids_list], device=device)

    # Get embeddings and replace latent positions with pause_embedding
    embedding = coconut_model.embedding
    inputs_embeds = embedding(input_ids)

    pause_emb = coconut_model.pause_embedding
    start_of_latent = len(question_tokens) + 1  # after question + <start_latent>

    inputs_embeds = inputs_embeds.clone()
    for i in range(n_thoughts):
        pos = start_of_latent + i
        inputs_embeds[0, pos, :] = pause_emb

    # Single forward pass
    attention_mask = torch.ones_like(input_ids, device=device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    outputs = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )

    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, D)

    # Extract hidden states at thought-relevant positions
    # thoughts[0] = hidden state at <start_latent> position
    #   (analogous to h_0 in coconut: the state after processing the prompt)
    start_latent_pos = len(question_tokens)  # position of <start_latent>
    thoughts[0] = last_hidden[0, start_latent_pos, :].cpu()

    # thoughts[1..K] = hidden states at <latent> positions
    for i in range(n_thoughts):
        pos = start_of_latent + i
        thoughts[i + 1] = last_hidden[0, pos, :].cpu()

    return thoughts


def _extract_thoughts_coconut(
    base_model, tokenizer, sample, n_thoughts, device, hidden_dim, start_id,
):
    """
    Coconut model: manual hidden-state recurrence.
    """
    thoughts = torch.zeros(n_thoughts + 1, hidden_dim)

    # Tokenize question (family-aware) and append <start-latent> token.
    # Mirrors gradient_subspace.py / superposition.py recurrence prompts.
    from src.utils import tokenize_question_for_recurrence
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    input_ids = torch.tensor([question_tokens + [start_id]], device=device)

    # Pass explicit attention_mask + position_ids at every step, matching the
    # canonical unbatched recurrence in superposition.py. Batch=1, no padding:
    #   # Step 0: positions [0, 1, ..., L-1],  L = len(question_tokens)+1
    #   # Step t (t=1..K): new-token position_id = L + t - 1
    #   # attention_mask grows by 1 each step (all real tokens, no pad)
    # GPT-2 (absolute PE) auto-computes the same ids from the cache, so this
    # is a no-op there; for Llama (RoPE) it makes the recurrence positions
    # explicit and audit-equivalent to the sibling scripts.
    L = input_ids.size(1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]
    thoughts[0] = h.cpu()
    past_kv = outputs.past_key_values

    # Recurrence: feed h back as input embedding for K steps
    ct = h.unsqueeze(0).unsqueeze(0)
    running_mask = attention_mask                          # (1, L)
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        pos_t = torch.tensor([[L + t - 1]], device=device)

        outputs = base_model(
            inputs_embeds=ct,
            attention_mask=running_mask,
            position_ids=pos_t,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :]
        thoughts[t] = h.cpu()
        ct = h.unsqueeze(0).unsqueeze(0)
        past_kv = outputs.past_key_values

    return thoughts

# ═══════════════════════════════════════════════════════════════════
# Shared CODI helpers
# ═══════════════════════════════════════════════════════════════════

def _codi_normalize_question(question_text: str) -> str:
    """
    Normalize a question string the way CODI test.py does.

    Mirrors test.py's batched tokenization preprocessing:
        questions = [s["question"].strip().replace('  ', ' ') for s in batch]

    Both the batched and unbatched CODI extractors must use this so that
    tokenization produces identical input_ids across call paths. Without
    this, an instance with dirty whitespace would tokenize differently in
    the two paths and silently produce different hidden states.
    """
    return question_text.strip().replace('  ', ' ')


@torch.no_grad()
def extract_thoughts_codi(
    codi_dict: dict,
    question_text: str,
    n_thoughts: int,
    device: str,
) -> torch.Tensor:
    """
    Run CODI recurrence for one instance and collect thought vectors.
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    use_prj = codi_dict['use_prj']
    hidden_size = codi_dict['hidden_size']

    thoughts = torch.zeros(n_thoughts + 1, hidden_size)

    # Tokenize: question + [eos, bot] — whitespace normalized to match test.py
    question_text = _codi_normalize_question(question_text)
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    if codi_dict['remove_eos']:
        input_ids_list = question_tokens + [bot_id]
    else:
        input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([input_ids_list], device=device)

    # Step 0: encode question + <bot>
    outputs = base_model(
        input_ids=input_ids, use_cache=True,
        output_hidden_states=True
    )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]
    thoughts[0] = h.cpu()

    # Project for feeding back
    latent = h.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    # Steps 1..K
    for t in range(n_thoughts):
        outputs = base_model(
            inputs_embeds=latent, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][0, -1, :]
        thoughts[t + 1] = h.cpu()

        latent = h.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    return thoughts


# ═══════════════════════════════════════════════════════════════════
# Thought extraction: CODI, batched
# ═══════════════════════════════════════════════════════════════════
#
# Mirrors the batched inference path in remove_thoughts.check_codi_inference
# (which in turn mirrors CODI test.py --greedy True with left-padding).
# Collects hidden states only; does not feed the [eot] delimiter or decode
# the answer, because downstream probing / scaffolding only needs the K+1
# pre-projection thought vectors.
#
# Left-padding layout for a batch of B rows with padded length L:
#
#   absolute index:  0 .. p_b-1    p_b .. L-1    L .. L+K-1
#   token (row b):   [PAD, ...]    [q_b, bot]    [thoughts]
#   position_id:     all 0         0 .. n_b-1    n_b .. n_b+K-1
#
#   where n_b = L - p_b = attention_mask.sum(row=b) = # of real tokens.
#
# CRITICAL: HuggingFace GPT-2 does NOT auto-correct position_ids from
# attention_mask. By default it uses torch.arange(0, L), which assigns
# position_ids [p_b, ..., L-1] to the real-token region — different from
# the unbatched case where they would be [0, ..., n_b - 1]. Since GPT-2
# uses absolute positional embeddings, this silently changes every
# hidden state.
#
# The fix (verified empirically vs the unbatched extractor, diff = 0):
#   1. Pass position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
#      at step 0. Pads get position_id 0 (clamped, harmless because
#      they're attention-masked out); real tokens get 0..n_b-1 per row.
#   2. At recurrence step t = 1..K, pass the full accumulated
#      attention_mask (so pads remain masked from the new token's
#      attention) and explicit position_ids = n_b + t - 1 per row.
#
# Hidden states saved (pre-projection, matching the unbatched extractor):
#   thoughts[b, 0]  = h at the [bot] position after step 0
#   thoughts[b, t]  = h after recurrence step t, for t = 1..K
#
# on_batch_start_fn receives max(n_b) across the batch — the smallest
# position_id at which every row's recurrence region has started. The
# WPEWrapper cutoff in pe_ablation uses this to decide which positions
# to ablate.


@torch.no_grad()
def extract_thoughts_codi_batch(
    codi_dict: dict,
    samples: list,
    n_thoughts: int,
    device: str,
    batch_size: int = 32,
    on_batch_start_fn=None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Batched CODI thought extraction.

    Args:
        codi_dict: as returned by setup_codi_model(task, device).
        samples: list of dicts with a "question" key.
        n_thoughts: K, the number of recurrence steps.
        device: e.g. "cuda" or "cuda:0".
        batch_size: number of instances per forward pass (left-padded).
        on_batch_start_fn: optional callable(cutoff: int) invoked once per
            batch before any forward pass. cutoff = max(real_len) over the
            batch, i.e. the smallest position_id that is guaranteed to be
            a recurrence-region position for every row. Used by pe_ablation
            to set the WPEWrapper cutoff for the batch.
        verbose: print progress every 100 instances.

    Returns:
        torch.Tensor of shape (N, K+1, D) on CPU.

    Implementation notes:
        - Row ordering in the returned tensor matches the input `samples`
          order; no length-sorting.
        - Tokenization mirrors remove_thoughts.check_codi_inference exactly:
          question.strip().replace('  ', ' '), padding='longest', left-pad
          (which is set in setup_codi_model via padding_side='left').
        - Hidden states are taken from outputs.hidden_states[-1][:, -1, :],
          i.e. last layer, last position, pre-projection — same as the
          unbatched extractor.
    """
    import math

    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']
    hidden_size = codi_dict['hidden_size']

    N = len(samples)
    K = n_thoughts
    all_thoughts = torch.zeros(N, K + 1, hidden_size)

    n_batches = math.ceil(N / batch_size)
    suffix_len = 1 if remove_eos else 2  # [bot] or [eos, bot]

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, N)
        batch = samples[batch_start:batch_end]
        B = len(batch)

        if verbose and batch_start % 100 == 0:
            print(f"  [extract_thoughts_codi_batch] {batch_start}/{N}")

        # ── Tokenize with left-padding (mirrors test.py + remove_thoughts) ──
        questions = [_codi_normalize_question(s["question"]) for s in batch]
        enc = tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = enc["input_ids"].to(device)              # (B, L_q)
        attention_mask = enc["attention_mask"].to(device)    # (B, L_q)

        # Append [bot] (+ optional [eos]) — every row gets suffix_len real tokens
        if remove_eos:
            suffix = torch.tensor(
                [[bot_id]] * B, dtype=torch.long, device=device,
            )
        else:
            suffix = torch.tensor(
                [[tokenizer.eos_token_id, bot_id]] * B,
                dtype=torch.long, device=device,
            )
        input_ids = torch.cat([input_ids, suffix], dim=1)        # (B, L_q + suffix_len)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(suffix)], dim=1,
        )

        # Padded prefix length L = total length of (padded q + suffix).
        # Thoughts will live at position_ids L, L+1, ..., L+K-1 — NO, that's
        # wrong under left-padding. See comment below.
        L = input_ids.shape[1]

        # ── Position IDs: MUST be cumsum-shifted under left-padding ──
        # HF GPT-2 does NOT auto-correct position_ids from attention_mask.
        # By default it uses torch.arange(0, L), which gives the real-token
        # region position_ids [p_b, p_b+1, ..., L-1] in row b with p_b pads
        # — different from the unbatched case where real tokens occupy
        # [0, 1, ..., n_b - 1]. That breaks both PE lookup and the logits.
        # Correct mapping: pos_ids = cumsum(mask) - 1, clamped at 0.
        # After step 0 the real-token region of row b has position_ids
        # [0, 1, ..., n_b - 1] where n_b = L - p_b = attention_mask.sum(b).
        #
        # # pos_ids[b, j] = (sum_{k<=j} mask[b, k]) - 1,   clamped >= 0
        # # real_len[b]   = sum_k mask[b, k]   (= n_b, the # real tokens)
        # # The last real token of row b gets position_id real_len[b] - 1.
        # # Therefore, at recurrence step t (t = 1..K), the new token's
        # # position_id for row b is:  real_len[b] + t - 1.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        real_len = attention_mask.sum(dim=1)              # (B,), n_b per row

        # The WPEWrapper cutoff callback expects the absolute position_id
        # at which "thought region" starts. Under the corrected position
        # semantics, the first thought position is real_len - 1 + 1 =
        # real_len[b] for row b — but real_len differs per row. If every
        # row has the same real_len (e.g. because max_instances == 1 or
        # all questions tokenize to the same length), a scalar cutoff works.
        # Otherwise, per-row cutoffs would be needed. pe_ablation.py sets
        # the cutoff to L = padded prefix length below; with the corrected
        # position_ids that means ablating positions >= L, which may NOT
        # correspond to the thought region for every row. We therefore
        # pass max(real_len) as the "safe" scalar cutoff: positions >=
        # max(real_len) are guaranteed to be thought positions for every
        # row (since every row has at most max(real_len) real prefix
        # tokens). Rows with fewer real tokens would also have their
        # suffix-tail position_ids >= max(real_len) - 1 + 1 = max(real_len),
        # but the suffix has already been consumed by step 0, so this
        # applies only to recurrence positions, which is exactly what
        # we want to ablate.
        if on_batch_start_fn is not None:
            on_batch_start_fn(int(real_len.max().item()))

        # ── Step 0: encode padded question + suffix ──
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][:, -1, :]              # (B, D)

        # thoughts[:, 0] = h at [bot] position (pre-projection)
        all_thoughts[batch_start:batch_end, 0, :] = h.float().cpu()

        # Project for feeding back as inputs_embeds
        latent = h.unsqueeze(1)                              # (B, 1, D)
        if use_prj and prj is not None:
            latent = prj(latent)

        # ── Steps 1..K: latent recurrence ──
        # Accumulate attention_mask across steps (pads stay masked; new
        # recurrence tokens are all real). Pass explicit per-row
        # position_ids matching the unbatched case: row b's new token at
        # recurrence step t lives at position real_len[b] + t - 1.
        running_mask = attention_mask                         # (B, L)
        for t in range(1, K + 1):
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=running_mask.dtype,
                                          device=device)],
                dim=1,
            )
            pos_t = (real_len + (t - 1)).unsqueeze(1)        # (B, 1)

            outputs = base_model(
                inputs_embeds=latent,
                attention_mask=running_mask,
                position_ids=pos_t,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][:, -1, :]          # (B, D)

            all_thoughts[batch_start:batch_end, t, :] = h.float().cpu()

            latent = h.unsqueeze(1)
            if use_prj and prj is not None:
                latent = prj(latent)

    return all_thoughts


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU: instance-sharded extraction
# ═══════════════════════════════════════════════════════════════════
#
# Mirrors the mp.Process + Queue pattern in
# experiments/ablation/gradient_subspace_interventions.py: each rank
# loads its own model copy on cuda:{rank} ONCE, extracts thoughts for
# its shard of instances, and returns the shard (+ its original
# indices) via a Queue. The parent scatters shards back into a single
# (N, K+1, D) tensor ordered like the input data.
#
# `splits_data` is a list of (split_name, data) pairs so that, with
# --split both, one worker set is spawned and the model loaded once per
# GPU, then reused for both the test-split and train-split shards —
# instead of respawning processes / reloading the model a second time
# (each split has its own shard sizes, since test/train differ in
# length, so sharding is still done per split inside the worker).
#
# Shards are exchanged via disk, NOT through the mp.Queue directly. A
# train shard can be multi-GB (e.g. ~24k llama-family instances x 7 x
# 2048 floats); torch.multiprocessing's fd-based tensor IPC (used when a
# large CPU tensor is put() into a Queue) has been observed to raise a
# spurious EOFError / ConnectionResetError under load, which is NOT a
# queue.Empty and so was propagating out of the polling loop below and
# killing the parent before any shard could be collected/saved — even
# though the workers themselves had finished (or nearly finished)
# extraction. Sending only small metadata (paths + indices) through the
# Queue and torch.save()/torch.load()-ing the actual tensors sidesteps
# that IPC race entirely.

def _extract_thoughts_worker(
    rank, world_size, task, model_name, model_family, n_thoughts,
    splits_data, batch_size, tmp_dir, return_queue,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    K = n_thoughts

    is_codi = (model_name == "codi")
    if is_codi:
        codi_dict = setup_codi_model(task, device, family=model_family)
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device, family=model_family)
        D = getattr(base_model.config, "n_embd",
                    getattr(base_model.config, "hidden_size", None))
        if D is None:
            raise RuntimeError("Could not determine hidden size from model config")

    meta = {}
    for split_name, data in splits_data:
        indices = _shard_indices(len(data), world_size, rank)
        shard_data = [data[i] for i in indices]
        print(f"[rank {rank}] split={split_name}: processing "
              f"{len(shard_data)} instances on {device}", flush=True)

        if is_codi:
            thoughts = extract_thoughts_codi_batch(
                codi_dict, shard_data, K, device, batch_size=batch_size,
                verbose=(rank == 0),
            )
        else:
            thoughts = torch.zeros(len(shard_data), K + 1, D)
            for local_idx, sample in enumerate(shard_data):
                if local_idx % 50 == 0:
                    print(f"[rank {rank}] split={split_name} "
                          f"{local_idx}/{len(shard_data)}", flush=True)
                thoughts[local_idx] = extract_thoughts_single_instance(
                    coconut_model, base_model, tokenizer, sample, K, device,
                    start_id, latent_id, end_id,
                )

        shard_path = tmp_dir / f"shard_r{rank}_{split_name}.pt"
        torch.save(thoughts, shard_path)
        meta[split_name] = {"indices": indices, "path": str(shard_path)}
        print(f"[rank {rank}] split={split_name}: saved shard -> {shard_path}",
              flush=True)

    return_queue.put({"rank": rank, "meta": meta})


def extract_thoughts_multigpu(
    task, model_name, model_family, n_thoughts, splits_data, n_gpus,
    batch_size=32,
):
    """
    Spawn n_gpus workers ONCE; each loads its own model copy on cuda:{rank}
    and extracts thoughts for its shard of every split in `splits_data`
    (a list of (split_name, data) pairs).

    Returns: dict {split_name: (N_split, K+1, D) tensor ordered like the
    corresponding `data`}.
    """
    # Shard files land under THOUGHTS (project storage), not the default
    # system temp dir: a compute node's /tmp is sometimes small or tmpfs
    # (RAM-backed), and a llama train shard can be multi-GB — writing
    # several of those concurrently risks filling /tmp outright, which
    # would be a second, dumber way to lose a run.
    THOUGHTS.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="extract_thoughts_shards_",
                                    dir=str(THOUGHTS)))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_extract_thoughts_worker,
            args=(rank, n_gpus, task, model_name, model_family, n_thoughts,
                  splits_data, batch_size, tmp_dir, q),
        )
        p.start()
        procs.append(p)

    def _kill_all_workers():
        # On any fatal error below, stop every worker instead of leaving
        # survivors to keep burning GPU-hours for a result nobody will
        # ever collect — exactly what happened before this fix existed.
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)

    shards = []
    try:
        for _ in range(n_gpus):
            while True:
                try:
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
                except (EOFError, ConnectionResetError, BrokenPipeError) as e:
                    # Transient IPC hiccup on the (now tiny) metadata
                    # message. Only a small dict travels through the queue
                    # now, so retrying is safe and cheap; treat a dead
                    # worker the same way as the queue.Empty branch above.
                    print(f"[WARN] transient queue receive error ({e!r}); "
                          "retrying...", flush=True)
                    for p in procs:
                        if not p.is_alive() and p.exitcode != 0:
                            raise RuntimeError(
                                f"Worker process {p.pid} crashed with exit code {p.exitcode}. "
                                "Check console for CUDA Out-Of-Memory or other runtime errors."
                            )
    except BaseException:
        _kill_all_workers()
        raise

    for p in procs:
        p.join()

    K = n_thoughts
    all_thoughts = {}
    for split_name, data in splits_data:
        split_thoughts = None
        for shard in shards:
            r = shard["meta"][split_name]
            shard_tensor = torch.load(r["path"], map_location="cpu",
                                      weights_only=False)
            if split_thoughts is None:
                D = shard_tensor.shape[-1]
                split_thoughts = torch.zeros(len(data), K + 1, D)
            for local_idx, idx in enumerate(r["indices"]):
                split_thoughts[idx] = shard_tensor[local_idx]
            Path(r["path"]).unlink(missing_ok=True)
        all_thoughts[split_name] = split_thoughts

    tmp_dir.rmdir()
    return all_thoughts


def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous thought vectors from Coconut/CODI model."
    )
    parser.add_argument(
        "--task", type=str, choices=["prosqa", "gsm"],
        default="prosqa",
    )
    parser.add_argument(
        "--model", type=str,
        choices=["coconut", "coconut_u", "pause", "codi", "base", "cot"],
        default="coconut",
        help="Recursion-trained: coconut, coconut_u, pause, codi. "
             "Forced-recursion controls (no recursion training): base, cot. "
             "base/cot use the same Coconut wrapper as coconut, but with "
             "untrained / token-level-CoT-trained weights respectively — "
             "the model is forced to recurse at inference time even though "
             "it was never optimized for that regime.",
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Determines checkpoint paths, dtype, and "
             "prompt formatting (Llama uses the chat template).",
    )
    parser.add_argument(
        "--split", type=str, choices=["test", "train", "both"], default="test",
        help="Which data split(s) to extract thoughts from: 'test' (default; "
             "matches evaluation instances), 'train', or 'both' (extracts "
             "both in one run, sharing a single loaded model / worker set). "
             "Train-split thoughts are saved with a '_train' filename "
             "suffix, matching the layout expected by mean_ablation.py / "
             "markovianity_test.py.",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Only valid with --split test or --split train (a single "
             "output file). With --split both, output paths are always "
             "the default per-split paths, since two files are produced.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for CODI extraction (ignored for coconut/pause, "
             "which use per-instance manual recurrence). Matches the default "
             "in CODI test.py.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=1,
        help="If >1, shard instances across GPUs 0..n_gpus-1, each loading "
             "its own model copy (mirrors gradient_subspace_interventions.py).",
    )
    args = parser.parse_args()

    if args.split == "both" and args.output_path is not None:
        parser.error("--output_path cannot be used with --split both "
                     "(two files are produced; use the default paths).")

    splits = ["test", "train"] if args.split == "both" else [args.split]

    def _output_path(split):
        if args.output_path is not None:
            return args.output_path
        suffix = "_train" if split == "train" else ""
        # Namespace by family so gpt2 and llama runs don't clobber each other.
        return str(
            THOUGHTS / f"{args.model_family}/{args.task}/"
            f"thoughts_{args.model}{suffix}.pt"
        )

    print(f"[INFO] Model: {args.model}  Family: {args.model_family}")
    print(f"[INFO] Task: {args.task}  Split(s): {splits}")
    print(f"[INFO] Recurrence steps K={args.n_thoughts}")
    print(f"[INFO] GPUs: {args.n_gpus}")

    K = args.n_thoughts
    is_codi = (args.model == "codi")
    data_by_split = {s: load_data(args.task, s, args.max_instances) for s in splits}

    if args.n_gpus > 1:
        total_n = sum(len(d) for d in data_by_split.values())
        print(f"[INFO] Multi-GPU: sharding {total_n} instances across "
              f"{args.n_gpus} GPUs (model loaded once per GPU, shared "
              f"across split(s) {splits})")
        thoughts_by_split = extract_thoughts_multigpu(
            args.task, args.model, args.model_family, K,
            list(data_by_split.items()), args.n_gpus,
            batch_size=args.batch_size,
        )
    else:
        print(f"[INFO] Device: {args.device}")
        # Initialize the model ONCE, reused across every split.
        if is_codi:
            codi_dict = setup_codi_model(args.task, args.device, family=args.model_family)
            D = codi_dict['hidden_size']
        else:
            coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = setup_model_and_tokenizer(
                args.task, args.model, args.device, family=args.model_family
            )
            # GPT-2 exposes `n_embd`; Llama exposes `hidden_size`.
            D = getattr(base_model.config, "n_embd",
                        getattr(base_model.config, "hidden_size", None))
            if D is None:
                raise RuntimeError("Could not determine hidden size from model config")

        thoughts_by_split = {}
        for split, data in data_by_split.items():
            N = len(data)
            if is_codi:
                # Batched: one call does the whole dataset, internally chunked.
                print(f"[INFO] split={split}: CODI batched extraction "
                      f"(batch_size={args.batch_size})")
                thoughts_by_split[split] = extract_thoughts_codi_batch(
                    codi_dict, data, K, args.device,
                    batch_size=args.batch_size,
                )
            else:
                all_thoughts = torch.zeros(N, K + 1, D)
                for idx, sample in enumerate(data):
                    if idx % 50 == 0:
                        print(f"[INFO] split={split}: Processing instance {idx}/{N}")
                    thoughts = extract_thoughts_single_instance(
                        coconut_model, base_model, tokenizer, sample, K, args.device,
                        start_id, latent_id, end_id,
                    )
                    all_thoughts[idx] = thoughts
                thoughts_by_split[split] = all_thoughts

    for split, data in data_by_split.items():
        all_thoughts = thoughts_by_split[split]
        N, Kp1, D = all_thoughts.shape
        output_path = _output_path(split)
        save_dict = {
            "thoughts": all_thoughts,
            "instance_indices": list(range(N)),
            "n_thoughts": K,
            "model": args.model,
            "model_family": args.model_family,
            "data_path": str(_data_path(args.task, split)),
            "split": split,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, output_path)
        print(f"[INFO] split={split}: Saved {N} x {Kp1} x {D} thought "
              f"vectors to {output_path}")

if __name__ == "__main__":
    main()