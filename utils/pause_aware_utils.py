"""
Pause-aware inference utilities for experiment scripts.

The pause model (M3) was trained with feedback_mode="pause_curriculum":
  - Thought positions receive a fixed learned nn.Parameter embedding
  - All thought tokens are processed in a SINGLE forward pass
  - No hidden-state recurrence ever occurs during training

Therefore, experiment scripts must NOT run manual recurrence on pause models.
Instead, they should:
  1. Build input: [question_tokens] <start_latent> <latent>*K <end_latent>
  2. Replace <latent> embeddings with the learned pause_embedding
  3. Run a single forward pass
  4. Extract hidden states at thought positions from the single pass

This module provides functions that transparently handle both coconut
(recurrence) and pause (single-pass) models.
"""

import torch
import torch.nn.functional as F
from contThought.coconut import Coconut
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_pause_model(coconut_model_or_mode):
    """Check whether a model or mode string indicates a pause model."""
    if isinstance(coconut_model_or_mode, str):
        return coconut_model_or_mode == "pause"
    if isinstance(coconut_model_or_mode, Coconut):
        return getattr(coconut_model_or_mode, 'feedback_mode', 'continuous') == 'pause_curriculum'
    return False


# ═══════════════════════════════════════════════════════════════════
# Thought extraction: pause-aware
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
    hidden_dim = base_model.config.n_embd

    if pause:
        return _extract_thoughts_pause(
            coconut_model, base_model, tokenizer, sample,
            n_thoughts, device, start_id, latent_id, end_id, hidden_dim,
        )
    else:
        return _extract_thoughts_coconut(
            base_model, tokenizer, sample,
            n_thoughts, device, hidden_dim,
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

    # Build input sequence with thought tokens
    question_text = sample["question"]
    question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)

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

    for i in range(n_thoughts):
        pos = start_of_latent + i
        inputs_embeds = inputs_embeds.clone()
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
        if i + 1 <= n_thoughts:
            thoughts[i + 1] = last_hidden[0, pos, :].cpu()

    return thoughts


def _extract_thoughts_coconut(
    base_model, tokenizer, sample, n_thoughts, device, hidden_dim,
):
    """
    Coconut model: manual hidden-state recurrence.
    Original logic from extract_thoughts.py.
    """
    thoughts = torch.zeros(n_thoughts + 1, hidden_dim)

    input_ids = tokenizer.encode(
        sample["question"] + " <|start-latent|>", return_tensors="pt"
    ).to(device)

    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]
    thoughts[0] = h.cpu()
    past_kv = outputs.past_key_values

    ct = h.unsqueeze(0).unsqueeze(0)
    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=ct,
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
# Batch thought extraction with labels: pause-aware
# ═══════════════════════════════════════════════════════════════════

def _get_candidate_info(sample):
    """
    Extract correct concept and all concepts from a ProsQA instance.
    Duplicated from steering_experiment.py to keep this module self-contained.
    """
    answer_text = sample["answer"].strip().rstrip(".")
    correct_concept = answer_text.split(" is a ")[-1].strip()

    all_concepts = set()
    all_concepts.add(correct_concept)
    for step in sample.get("steps", []):
        parts = step.split(" is a ")
        if len(parts) == 2:
            concept_a = parts[0].replace("Every ", "").strip()
            concept_b = parts[1].strip().rstrip(".")
            all_concepts.add(concept_a)
            all_concepts.add(concept_b)

    return correct_concept, sorted(all_concepts)


@torch.no_grad()
def extract_thoughts_with_labels(
    coconut_model, base_model, tokenizer, data, n_thoughts, device,
    start_id, latent_id, end_id,
):
    """
    Extract thought vectors AND labels/concepts for all instances.

    Drop-in replacement for steering_experiment.py's
    extract_thoughts_with_labels() that correctly handles both
    coconut (recurrence) and pause (single-pass) models.

    Returns:
        thoughts: (N, T, D) tensor — T = n_thoughts + 1
        labels: list of N correct concept strings
        all_concepts: sorted list of all unique concepts across dataset
    """
    N = len(data)
    D = base_model.config.n_embd
    T = n_thoughts + 1
    thoughts = torch.zeros(N, T, D)
    labels = []
    all_concepts = set()

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"  [Extract] {idx}/{N}")

        # ── Label extraction (data-side, model-independent) ──
        correct_concept, concepts = _get_candidate_info(sample)
        labels.append(correct_concept)
        all_concepts.update(concepts)

        # ── Thought extraction (model-aware: pause vs coconut) ──
        thoughts_single = extract_thoughts_single_instance(
            coconut_model, base_model, tokenizer, sample,
            n_thoughts, device, start_id, latent_id, end_id,
        )
        thoughts[idx] = thoughts_single

    return thoughts, labels, sorted(all_concepts)


# ═══════════════════════════════════════════════════════════════════
# Intervened inference: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_intervened_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn,
    start_id=None, latent_id=None,
):
    """
    Run inference with interventions, handling pause and coconut models.

    For coconut: intervenes at each recurrence step (original behavior).
    For pause: runs single forward pass, applies intervention to hidden
        states at thought positions via a hook, then decodes.

    intervention_fn: callable (h, t) -> h_modified
        h: (D,) tensor on device
        t: step index (0 = start_latent position, 1..K = thought positions)
    """
    pause = is_pause_model(coconut_model)

    if pause:
        return _intervened_inference_pause(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
            start_id, latent_id,
        )
    else:
        return _intervened_inference_coconut(
            base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
        )


def _intervened_inference_pause(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn,
    start_id, latent_id,
):
    """
    Pause model: single forward pass with hook-based interventions.

    Strategy:
        1. Build input with pause embeddings
        2. Register a forward hook on the last transformer layer
           that modifies hidden states at thought positions
        3. Run single forward pass (hook fires automatically)
        4. Decode answer from the output logits
    """
    question_text = sample["question"]
    question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)

    input_ids_list = (
        question_tokens
        + [start_id]
        + [latent_id] * n_thoughts
        + [end_id]
    )
    input_ids = torch.tensor([input_ids_list], device=device)

    # Build embeddings with pause tokens
    embedding = coconut_model.embedding
    inputs_embeds = embedding(input_ids)
    pause_emb = coconut_model.pause_embedding

    start_of_latent = len(question_tokens) + 1
    for i in range(n_thoughts):
        pos = start_of_latent + i
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, pos, :] = pause_emb

    # Determine thought positions for intervention
    start_latent_pos = len(question_tokens)  # <start_latent>
    thought_positions = [start_latent_pos]  # t=0
    for i in range(n_thoughts):
        thought_positions.append(start_of_latent + i)  # t=1..K

    # Register hook on the last transformer layer to apply interventions
    # GPT-2's last layer is base_model.transformer.h[-1]
    last_layer = base_model.transformer.h[-1]
    hook_handle = None

    def intervention_hook(module, input, output):
        # output is a tuple: (hidden_states, *optional)
        # hidden_states shape: (batch, seq_len, D)
        hidden_states = output[0]
        modified = hidden_states.clone()

        for t_idx, pos in enumerate(thought_positions):
            if pos < modified.shape[1]:
                h = modified[0, pos, :]
                h_modified = intervention_fn(h, t_idx)
                modified[0, pos, :] = h_modified

        # Return modified output (preserve tuple structure)
        return (modified,) + output[1:]

    hook_handle = last_layer.register_forward_hook(intervention_hook)

    try:
        # Single forward pass with hook active
        attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # Decode answer (greedy, from the last position onward)
    generated = []
    next_logits = outputs.logits[0, -1, :]

    for _ in range(128):
        next_token = next_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        out = base_model(
            input_ids=torch.tensor([[next_token]], device=device),
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = out.logits[0, -1, :]
        past_kv = out.past_key_values

    text = tokenizer.decode(generated, skip_special_tokens=True)
    answer = text.split("#")[-1].replace(",", "").strip()
    correct_answer = sample.get("answer", "").replace(",", "").strip()

    return {
        "predicted": answer,
        "correct": correct_answer,
        "is_correct": answer == correct_answer,
        'text': text
    }


def _intervened_inference_coconut(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn,
):
    """
    Coconut model: original recurrence-based intervened inference.
    Copied from steering_experiment.py's run_intervened_inference.
    """
    input_ids = tokenizer.encode(
        sample["question"] + " <|start-latent|>", return_tensors="pt"
    ).to(device)

    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]
    past_kv = outputs.past_key_values

    h = intervention_fn(h, 0)
    ct = h.unsqueeze(0).unsqueeze(0)

    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=ct,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :]
        h = intervention_fn(h, t)
        ct = h.unsqueeze(0).unsqueeze(0)
        past_kv = outputs.past_key_values

    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(
        input_ids=end_input, past_key_values=past_kv, use_cache=True,
    )
    past_kv = outputs.past_key_values

    generated = []
    next_logits = outputs.logits[0, -1, :]
    for _ in range(128):
        next_token = next_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        out = base_model(
            input_ids=torch.tensor([[next_token]], device=device),
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = out.logits[0, -1, :]
        past_kv = out.past_key_values

    text = tokenizer.decode(generated, skip_special_tokens=True)
    answer = text.split("#")[-1].replace(",", "").strip()
    correct_answer = sample.get("answer", "").replace(",", "").strip()

    return {
        "predicted": answer,
        "correct": correct_answer,
        "is_correct": answer == correct_answer,
        'text': text
    }


# ═══════════════════════════════════════════════════════════════════
# Alpha sweep: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_alpha_sweep_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, steering_vectors, alphas,
    start_id=None, latent_id=None,
):
    """
    Run inference for multiple alphas.
    For coconut: batched recurrence (original).
    For pause: batched single forward pass with hooks.
    """
    pause = is_pause_model(coconut_model)

    if pause:
        # For pause, run each alpha separately with hook-based intervention
        # (batching hooks is complex and error-prone)
        results = {}
        for alpha in alphas:
            v_tensors = {t: torch.tensor(v, dtype=torch.float32, device=device)
                         for t, v in steering_vectors.items()}

            def steer_fn(h, t, _alpha=alpha, _v=v_tensors):
                if t in _v:
                    return h + _alpha * _v[t]
                return h

            r = run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                n_thoughts, device, steer_fn,
                start_id=start_id, latent_id=latent_id,
            )
            results[alpha] = r["predicted"].split("#")[-1].replace(",", "").strip()

        return results
    else:
        # Coconut: use original batched recurrence
        return _alpha_sweep_coconut(
            base_model, tokenizer, end_id, sample,
            n_thoughts, device, steering_vectors, alphas,
        )


def _alpha_sweep_coconut(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, steering_vectors, alphas,
):
    """Batched alpha sweep for coconut models."""
    n_alphas = len(alphas)
    
    # 1. Expand the prompt to batch size = n_alphas
    input_ids = tokenizer.encode(sample["question"] + " <|start-latent|>", return_tensors="pt").to(device)
    input_ids = input_ids.repeat(n_alphas, 1)
    
    alpha_tensor = torch.tensor(alphas, dtype=torch.float32, device=device).view(n_alphas, 1)
    v_tensors = {t: torch.tensor(v, dtype=torch.float32, device=device) for t, v in steering_vectors.items()}
                 
    def sweep_intervention_fn(h, t):
        if t in v_tensors:
            return h + (alpha_tensor * v_tensors[t])
        return h

    outputs = base_model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
    h = outputs.hidden_states[-1][:, -1, :] 
    past_kv = outputs.past_key_values

    h = sweep_intervention_fn(h, 0)
    ct = h.unsqueeze(1) 

    for t in range(1, n_thoughts + 1):
        outputs = base_model(inputs_embeds=ct, past_key_values=past_kv, output_hidden_states=True, use_cache=True)
        h = outputs.hidden_states[-1][:, 0, :]
        h = sweep_intervention_fn(h, t)
        ct = h.unsqueeze(1)
        past_kv = outputs.past_key_values

    end_input = torch.tensor([[end_id]] * n_alphas, device=device)
    outputs = base_model(input_ids=end_input, past_key_values=past_kv, use_cache=True)
    past_kv = outputs.past_key_values

    next_logits = outputs.logits[:, -1, :]
    generated_ids = [[] for _ in range(n_alphas)]
    finished = [False] * n_alphas

    for _ in range(128):
        next_tokens = next_logits.argmax(dim=-1)
        for b_idx in range(n_alphas):
            if not finished[b_idx]:
                if next_tokens[b_idx].item() == tokenizer.eos_token_id:
                    finished[b_idx] = True
                else:
                    generated_ids[b_idx].append(next_tokens[b_idx].item())
        if all(finished):
            break
        out = base_model(input_ids=next_tokens.unsqueeze(1), past_key_values=past_kv, use_cache=True)
        next_logits = out.logits[:, -1, :]
        past_kv = out.past_key_values

    results = {}
    for b_idx, alpha in enumerate(alphas):
        text = tokenizer.decode(generated_ids[b_idx], skip_special_tokens=True)
        answer = text.split("#")[-1].replace(",", "").strip()
        results[alpha] = answer

    return results


# ═══════════════════════════════════════════════════════════════════
# Superposition probing: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def probe_concept_probs_pauseaware(
    coconut_model, tokenizer, input_ids, candidates, device, k,
    start_id=None, latent_id=None, end_id_tok=None, sample=None,
):
    """
    Probe concept probabilities after k thoughts.

    For coconut: uses Coconut.forward() (recurrence), then probes.
    For pause: builds input with k pause embeddings, runs single
        forward pass, then probes from the last thought position.

    The probing methodology is identical: feed " Every" prefix after
    the thought positions, then compute p(concept) autoregressively.
    """
    import math

    pause = is_pause_model(coconut_model)
    base_model = coconut_model.base_causallm
    embedding = coconut_model.embedding

    if pause:
        # Build input with pause embeddings for k thoughts
        question_text = sample["question"]
        question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)

        input_ids_list = (
            question_tokens
            + [start_id]
            + [latent_id] * k
            + [end_id_tok]
        )
        input_ids_t = torch.tensor([input_ids_list], device=device)

        inputs_embeds = embedding(input_ids_t)
        pause_emb = coconut_model.pause_embedding
        start_of_latent = len(question_tokens) + 1

        for i in range(k):
            pos = start_of_latent + i
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, pos, :] = pause_emb

        # Single forward pass
        full_outputs = base_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=True,
        )
        kv_cache = full_outputs.past_key_values
        current_pos = inputs_embeds.shape[1]

    else:
        # Coconut: use forward() for recurrence, then get KV cache
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)

        outputs = coconut_model.forward(
            input_ids, attention_mask, labels, position_ids
        )
        inputs_embeds_out = outputs.inputs_embeds

        full_outputs = base_model(inputs_embeds=inputs_embeds_out, use_cache=True)
        kv_cache = full_outputs.past_key_values
        current_pos = inputs_embeds_out.shape[1]

    # From here, probing is identical for both models
    prefix_tokens = tokenizer.encode(" Every", add_special_tokens=False)
    prefix_out = None
    for pt in prefix_tokens:
        pt_embed = embedding(torch.tensor([[pt]], device=device))
        pos_id = torch.tensor([[current_pos]], device=device)
        prefix_out = base_model(
            inputs_embeds=pt_embed,
            past_key_values=kv_cache,
            position_ids=pos_id,
            use_cache=True,
        )
        kv_cache = prefix_out.past_key_values
        current_pos += 1

    if prefix_out is None:
        raise RuntimeError('Prefix tokenization for " Every" returned no tokens.')

    logits_at_concept = prefix_out.logits[0, 0, :]

    concept_log_probs = {}
    for concept_name, _ in candidates:
        c_tokens = tokenizer.encode(" " + concept_name, add_special_tokens=False)
        if len(c_tokens) == 0:
            concept_log_probs[concept_name] = float("-inf")
            continue

        p_first = F.softmax(logits_at_concept, dim=-1)[c_tokens[0]].item()
        log_p = math.log(p_first + 1e-30)

        if len(c_tokens) > 1:
            concept_cache = tuple((k_.clone(), v_.clone()) for k_, v_ in kv_cache)
            concept_pos = current_pos

            for i in range(len(c_tokens) - 1):
                t_embed = embedding(torch.tensor([[c_tokens[i]]], device=device))
                cp_id = torch.tensor([[concept_pos]], device=device)
                c_out = base_model(
                    inputs_embeds=t_embed,
                    past_key_values=concept_cache,
                    position_ids=cp_id,
                    use_cache=True,
                )
                concept_cache = c_out.past_key_values
                concept_pos += 1

                p_next = F.softmax(c_out.logits[0, 0, :], dim=-1)
                log_p += math.log(p_next[c_tokens[i + 1]].item() + 1e-30)

        concept_log_probs[concept_name] = log_p

    concept_probs = {c: math.exp(lp) for c, lp in concept_log_probs.items()}
    return concept_probs, concept_log_probs


# ═══════════════════════════════════════════════════════════════════
# Normal inference (no intervention): pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_normal_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device,
    start_id=None, latent_id=None,
):
    """
    Normal inference without intervention.
    For pause: single forward pass + decode.
    For coconut: recurrence + decode.
    """
    identity_fn = lambda h, t: h
    return run_intervened_inference_pauseaware(
        coconut_model, base_model, tokenizer, end_id, sample,
        n_thoughts, device, identity_fn,
        start_id=start_id, latent_id=latent_id,
    )