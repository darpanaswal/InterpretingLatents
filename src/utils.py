import re
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contThought.coconut import Coconut
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import BASE_GPT2, PROSQA_MODELS, GSM_MODELS, PROSQA_TEST, GSM_TEST

# CODI generates CoT reasoning before the answer on GSM8k, which can exceed
# 128 tokens.  256 matches remove_thoughts.py / the original CODI test.py.
MAX_DECODE_TOKENS = 256

_peft_tp_shard_patched = False


def _patch_peft_tp_shard_noop():
    """
    Some peft versions unconditionally import
    transformers.integrations.tensor_parallel inside their internal
    _maybe_shard_state_dict_for_tp() -- even when tensor-parallel loading
    isn't in use -- which doesn't exist in transformers==4.48.3 (pinned
    here; newer transformers breaks eager attention for this project) and
    crashes every LoRA checkpoint load with:
        ModuleNotFoundError: No module named 'transformers.integrations.tensor_parallel'

    We never use transformers' own tensor-parallel sharding of a single
    model across GPUs -- this project's multi-GPU usage is data-parallel
    (one full model replica per rank, sharding instances not weights), so
    the function's real job (shard the state dict across a TP mesh) is a
    no-op for us regardless; the bug is that even the "not using TP" path
    unconditionally does the import before it would find that out. Patching
    it to just return the state dict unchanged is what the correct
    behavior would already be for our case.

    Idempotent (checked via a module-level flag) and defensive: if a given
    peft version doesn't have this function at all, this is a silent no-op.
    The existing non-zero-adapter-norm sanity check right after
    PeftModel.from_pretrained() would already catch a genuinely broken
    load, so this is self-verifying at each call site that uses it.
    """
    global _peft_tp_shard_patched
    if _peft_tp_shard_patched:
        return
    try:
        import peft.utils.save_and_load as _peft_save_and_load
        if hasattr(_peft_save_and_load, "_maybe_shard_state_dict_for_tp"):
            def _noop_shard_for_tp(model, state_dict, adapter_name):
                return state_dict
            _peft_save_and_load._maybe_shard_state_dict_for_tp = _noop_shard_for_tp
    except ImportError:
        pass
    _peft_tp_shard_patched = True


def extract_answer_number(text: str, task: str = "gsm"):
    """
    Extract the answer based on the task.
    Handles both CODI format ('The answer is:') and Coconut/Pause format ('###').
    """
    text = text.replace(',', '')

    if task == "prosqa":
        # 1. Check for CODI format
        if "The answer is:" in text:
            ans = text.split("The answer is:")[-1].strip()
            ans = ans.split("\n")[0].strip()
            # CODI sometimes misses the period, so we enforce it
            if not ans.endswith("."):
                ans = ans + "."
            return ans
            
        # 2. Check for Pause/Coconut format
        elif "#" in text:
            # Splits "### Bob is a shumpus." -> isolating the string after the last hash
            ans = text.split("#")[-1].strip()
            return ans
            
        # 3. Fallback
        else:
            return text.strip()

    # --- GSM Logic (Numeric Extraction) ---
    pred = re.findall(r'-?\d+\.?\d*', text)
    if not pred:
        return float('inf')
    
    return float(pred[-1])


def _compare_answers(predicted_text, sample, task):
    """Compare predicted answer to gold, returning (predicted_str, correct_str, is_correct)."""
    
    # 1. Clean and extract the model's prediction using our unified function
    pred_ans = extract_answer_number(predicted_text, task=task)
    
    # 2. Extract and format the gold ground truth
    gold_text = sample.get("answer", "").replace(",", "").strip()
    
    if task == "gsm":
        # GSM gold answers have explanations. Isolate the number after "####" 
        # so we don't accidentally extract a number from the explanation text.
        if "####" in gold_text:
            gold_text = gold_text.split("####")[-1].strip()
        gold_ans = extract_answer_number(gold_text, task="gsm")
    else:
        # ProsQA gold answers are already the raw target string (e.g., "Sally is a sterpus.")
        gold_ans = gold_text
        
    # 3. Compare and return! 
    # (Converting to strings to maintain the original return signature)
    return str(pred_ans), str(gold_ans), pred_ans == gold_ans

def clean_state_dict_keys(state_dict):
    """
    Strip FSDP / DDP wrapper prefixes from checkpoint keys.

    run.py saves checkpoints via parallel_model.state_dict() where
    parallel_model is FSDP(Coconut(...)) or DDP(Coconut(...)).

    For GPT-2, GPT2Block is commented out of the FSDP auto_wrap_policy
    (run.py line 175), so FSDP treats the whole model as a single flat
    module — effectively DDP. The resulting keys may have prefixes:

        FSDP:  _fsdp_wrapped_module.base_causallm.transformer.h.0...
        DDP:   module.base_causallm.transformer.h.0...
        Clean: base_causallm.transformer.h.0...

    The Coconut class expects keys starting with "base_causallm.".
    We strip any wrapper prefixes to normalize.
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_fsdp_wrapped_module."):
            new_k = new_k[len("_fsdp_wrapped_module."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned[new_k] = v
    return cleaned

# ═══════════════════════════════════════════════════════════════════
# Pause-aware utility functions
# ═══════════════════════════════════════════════════════════════════

"""
The pause model was trained with feedback_mode="pause_curriculum":
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

def is_pause_model(coconut_model_or_mode):
    """Check whether a model or mode string indicates a pause model."""
    if isinstance(coconut_model_or_mode, str):
        return coconut_model_or_mode == "pause"
    if isinstance(coconut_model_or_mode, Coconut):
        return getattr(coconut_model_or_mode, 'feedback_mode', 'continuous') == 'pause_curriculum'
    return False


def tokenize_question_for_recurrence(tokenizer, question_text):
    """
    Tokenize the question prefix that precedes <start_latent> for
    pause / coconut / coconut_u inference. Mirrors the training-time format:

      - Llama instruct (chat_template present): apply_chat_template with
        add_generation_prompt=True, then encode with add_special_tokens=False
        (chat template already includes BOS).
      - Otherwise (GPT-2, non-instruct Llama): raw "{q}\n" with
        add_special_tokens=True.

    Returns a python list of token ids (no batch dim). Use this everywhere
    a recurrence-style prompt is built manually (i.e. anywhere the code
    constructs [question_tokens] + [start_latent] + ...).

    Why centralized: every llama pause/coconut script previously used raw
    encode, which is format-OOD for instruct-tuned Llama. At K=K_max the
    pause embeddings masked the mismatch (~99.8% on ProsQA), but at K=0
    the mismatch was exposed (66.4% on ProsQA), producing a spurious
    "thoughts help by 33%" signal that was actually a format artefact.
    """
    use_chat_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )

    if use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(prompt, add_special_tokens=False)
    else:
        return tokenizer.encode(question_text + "\n", add_special_tokens=True)

# ═══════════════════════════════════════════════════════════════════
# Intervened inference: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_intervened_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn,
    start_id=None, latent_id=None, task="prosqa",
):
    """
    Run inference with interventions, handling pause and coconut models.

    For coconut: intervenes at each recurrence step (original behavior).
    For pause: runs single forward pass, applies intervention to hidden
        states at thought positions via a hook, then decodes.

    intervention_fn: callable (h, t) -> h_modified
        h: (D,) tensor on device
        t: step index (0 = start_latent position, 1..K = thought positions)
    task: "prosqa" or "gsm" — controls answer extraction and comparison.
    """
    pause = is_pause_model(coconut_model)

    if pause:
        return _intervened_inference_pause(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
            start_id, latent_id, task=task,
        )
    else:
        return _intervened_inference_coconut(
            base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn, task=task,
        )


def _intervened_inference_pause(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn,
    start_id, latent_id, task="prosqa",
):
    """
    Pause model: single forward pass with embedding-level interventions.
 
    Strategy:
        1. Build input with pause embeddings
        2. Apply intervention_fn to the embeddings at each thought position
           BEFORE the forward pass, so corrupted values propagate through
           all transformer layers and affect downstream positions including
           <end-latent>
        3. Run single forward pass on the modified embeddings
        4. Decode answer from the output logits
 
    Why embedding-level, not hook-based:
        A hook on the last transformer layer modifies the output hidden
        states at thought positions, but nothing reads those positions
        afterward — decoding starts from <end-latent> (the last position)
        using the KV cache that was already computed. The corrupted values
        never propagate. By intervening at the embedding level, the
        modified embeddings flow through all 12 layers and affect the
        KV cache entries that downstream positions attend to.
    """
    question_text = sample["question"]
    question_tokens = tokenize_question_for_recurrence(tokenizer, question_text)
 
    input_ids_list = (
        question_tokens
        + [start_id]
        + [latent_id] * n_thoughts
        + [end_id]
    )
    input_ids = torch.tensor([input_ids_list], device=device)
 
    # Build embeddings with pause tokens
    embedding = coconut_model.embedding
    inputs_embeds = embedding(input_ids).clone()
    pause_emb = coconut_model.pause_embedding
 
    start_of_latent = len(question_tokens) + 1
    for i in range(n_thoughts):
        pos = start_of_latent + i
        inputs_embeds[0, pos, :] = pause_emb
 
    # Apply interventions at the embedding level BEFORE the forward pass.
    #
    # thought_positions maps step index t to sequence position:
    #   t=0: <start_latent> position (last prompt position before thoughts)
    #   t=1..K: the K thought (pause) token positions
    #
    # intervention_fn(embedding_vec, t) -> modified_embedding_vec
    # The modified embedding propagates through all transformer layers.
    start_latent_pos = len(question_tokens)  # position of <start_latent>
    thought_positions = [start_latent_pos]   # t=0
    for i in range(n_thoughts):
        thought_positions.append(start_of_latent + i)  # t=1..K
 
    for t_idx, pos in enumerate(thought_positions):
        h = inputs_embeds[0, pos, :]
        h_modified = intervention_fn(h, t_idx)
        inputs_embeds[0, pos, :] = h_modified
 
    # Single forward pass on modified embeddings
    attention_mask = torch.ones_like(input_ids, device=device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
 
    outputs = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
 
    # Decode answer (greedy, from the last position onward)
    generated = []
    next_logits = outputs.logits[0, -1, :]
 
    for _ in range(MAX_DECODE_TOKENS):
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
    answer, correct_answer, is_correct = _compare_answers(text, sample, task)
 
    return {
        "predicted": answer,
        "correct": correct_answer,
        "is_correct": is_correct,
        'text': text
    }


def _intervened_inference_coconut(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, intervention_fn, task="prosqa",
):
    """
    Coconut model: original recurrence-based intervened inference.
    Copied from steering_experiment.py's run_intervened_inference.
    """
    # Use the same training-format-aware tokenization as the pause path.
    # For llama (instruct), this applies the chat template. For gpt2, this
    # is raw "{q}\n". Then append <start_latent> id manually.
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    start_id_local = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    input_ids = torch.tensor(
        [question_tokens + [start_id_local]], device=device,
    )

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
    for _ in range(MAX_DECODE_TOKENS):
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
    answer, correct_answer, is_correct = _compare_answers(text, sample, task)

    return {
        "predicted": answer,
        "correct": correct_answer,
        "is_correct": is_correct,
        'text': text
    }


# ═══════════════════════════════════════════════════════════════════
# Alpha sweep: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_alpha_sweep_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device, steering_vectors, alphas,
    start_id=None, latent_id=None, task="prosqa",
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
                start_id=start_id, latent_id=latent_id, task=task,
            )
            results[alpha] = r["predicted"]

        return results
    else:
        # Coconut: use original batched recurrence
        return _alpha_sweep_coconut(
            base_model, tokenizer, end_id, sample,
            n_thoughts, device, steering_vectors, alphas, task=task,
        )


def _alpha_sweep_coconut(
    base_model, tokenizer, end_id, sample,
    n_thoughts, device, steering_vectors, alphas, task="prosqa",
):
    """Batched alpha sweep for coconut models."""
    n_alphas = len(alphas)
    
    # 1. Expand the prompt to batch size = n_alphas
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    start_id_local = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    input_ids = torch.tensor(
        [question_tokens + [start_id_local]], device=device,
    )
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

    for _ in range(MAX_DECODE_TOKENS):
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
        # Extract answer string using the same logic as single-instance inference
        answer, _, _ = _compare_answers(text, sample, task)
        results[alpha] = answer

    return results

# ═══════════════════════════════════════════════════════════════════
# Normal inference (no intervention): pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_normal_inference_pauseaware(
    coconut_model, base_model, tokenizer, end_id, sample,
    n_thoughts, device,
    start_id=None, latent_id=None, task="prosqa",
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
        start_id=start_id, latent_id=latent_id, task=task,
    )






# ═══════════════════════════════════════════════════════════════════
# Model-loading
# ═══════════════════════════════════════════════════════════════════

def get_checkpoint_path(task, mode, family="gpt2"):
    model_path = PROSQA_MODELS if task == "prosqa" else GSM_MODELS
    family_path = model_path / family
    
    if mode == "base":
        return None
        
    if family == "llama":
        if mode == "coconut_u":
            return str(family_path / "coconut-u0.3")
        else:
            return str(family_path / mode)

    if mode == "cot":
        if task == "prosqa" and family == "gpt2":
            return str(family_path / "cot/best_checkpoint.pt")
        else:
            return str(family_path / "cot/checkpoint_best")
    if mode == "pause":
        return str(family_path / "pause/checkpoint_best")
    if mode == "coconut":
        return str(family_path / "coconut/checkpoint_best")
    if mode == "coconut_u":
        return str(family_path / "coconut-u0.3/checkpoint_best")
    raise ValueError(f"Unsupported mode: {mode}")

def setup_model_and_tokenizer(task, mode, device, family="gpt2"):
    """
    Load GPT-2 or Llama, add Coconut special tokens, wrap in Coconut class.
    Handles PeftModel for Llama and standard state_dicts for GPT-2.
    """
    import os
    from src.config import BASE_GPT2, BASE_LLAMA
    checkpoint_path = get_checkpoint_path(task, mode, family)

    # Full-FT Llama: checkpoint dir is a complete HF repo (resized vocab + trained
    # latent-token embeddings baked in). Detected by presence of config.json and
    # ABSENCE of adapter_config.json (which marks a LoRA adapter dir). Loaded directly
    # as the model; base-model load, LoRA wrap, and resize/copy are all skipped.
    llama_full_ft = False
    if family == "llama" and mode != "base":
        llama_full_ft = (
            os.path.isdir(checkpoint_path)
            and os.path.exists(os.path.join(checkpoint_path, "config.json"))
            and not os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
        )

    if family == "gpt2":
        base_model_path = BASE_GPT2
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
    elif family == "llama":
        if llama_full_ft:
            # tokenizer also from checkpoint: it carries the latent special tokens.
            base_model_path = checkpoint_path
            print(f"Loading Llama full-FT checkpoint: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        else:
            base_model_path = BASE_LLAMA
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- CoT: load checkpoint BEFORE adding special tokens (GPT-2 only) ---
    if mode == "cot" and family == "gpt2":
        print(f"Loading CoT checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys (first 5): {unexpected[:5]}")

    # --- Add special tokens (all modes) ---
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize embeddings + copy "<<" into the new <|latent|>/<|start-latent|>/<|end-latent|>
    # slots. Llama cot is wrapped via LoRA whose adapter shapes target the *original*
    # (un-resized) base vocab — so for cot we must resize AFTER the LoRA load. Other
    # modes (gpt2 all, llama base, llama pause/coconut/coconut_u) can resize now.
    def _resize_and_copy_target(m):
        m.resize_token_embeddings(len(tokenizer))
        embeddings = m.get_input_embeddings()
        if family == "llama":
            target_id = tokenizer.encode("<<", add_special_tokens=False)[0]
        else:
            target_id = tokenizer.convert_tokens_to_ids("<<")
        lm_head = m.get_output_embeddings()
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id].clone()

    # Full-FT pause/coconut/coconut_u checkpoints already carry resized embeddings +
    # trained latent vectors: resizing or overwriting with the "<<" copy would clobber
    # trained weights, so skip both. Full-FT cot checkpoints are the exception — cot
    # training never introduces latent tokens, so even a full-FT cot checkpoint still
    # needs the embedding table resized and the "<<" fallback copied into the new slots.
    resize_after_lora = (family == "llama" and mode == "cot" and not llama_full_ft)
    needs_resize_now = not llama_full_ft or (mode == "cot" and llama_full_ft)
    if not resize_after_lora and needs_resize_now:
        _resize_and_copy_target(model)

    # --- Llama LoRA Loading (skipped for full-FT) ---
    if family == "llama" and mode != "base" and not llama_full_ft:
        from peft import PeftModel
        _patch_peft_tp_shard_noop()
        print(f"Loading Llama LoRA checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        # Sanity: confirm an adapter is active and has non-zero norm.
        # Silent no-op load (e.g. due to PEFT version drift) is the suspected
        # cause of CoT predictions looking identical to base Llama.
        try:
            active = getattr(model, "active_adapter", None)
            lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
            total_norm = sum(p.detach().float().norm().item() ** 2 for _, p in lora_params) ** 0.5
            print(f"  [LoRA] active_adapter={active}, "
                  f"#lora_params={len(lora_params)}, total_norm={total_norm:.3f}")
            if len(lora_params) == 0 or total_norm < 1e-6:
                print("  [LoRA] WARNING: no LoRA params loaded or norm is ~0 — "
                      "adapter likely not active. Predictions will mirror base.")
        except Exception as e:
            print(f"  [LoRA] sanity check failed: {e}")
        if resize_after_lora:
            _resize_and_copy_target(model)

    # --- Wrap in Coconut (all modes, both families) ---
    # Wrapping llama base/cot in Coconut mirrors the gpt2 base/cot setup used as the
    # logit-lens control: the underlying HF model is still callable directly via
    # `base_causallm`, and Coconut-specific code paths (pause_embedding, latent loop)
    # are only invoked by experiments that explicitly need them.
    feedback_mode = "pause_curriculum" if mode == "pause" else "continuous"
    coconut_model = Coconut(model, latent_id, start_id, end_id,
                            tokenizer.eos_token_id,
                            feedback_mode=feedback_mode)

    # --- Load checkpoint ---
    if family == "gpt2" and mode in ("pause", "coconut", "coconut_u"):
        print(f"Loading Coconut checkpoint: {checkpoint_path}")
        raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = clean_state_dict_keys(raw_state_dict)

        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith("base_causallm"):
            print(f"  WARNING: first key is '{sample_key}'")
            print(f"  Expected keys starting with 'base_causallm.*'.")

        missing, unexpected = coconut_model.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(unexpected)
        print(f"  Loaded {n_loaded}/{len(state_dict)} keys")
        if missing:
            print(f"  Missing (first 5): {missing[:5]}")
        if unexpected:
            print(f"  Unexpected (first 5): {unexpected[:5]}")

    elif family == "llama" and mode in ("pause", "coconut", "coconut_u"):
        extra_state_path = os.path.join(checkpoint_path, "coconut_extras.pt")
        if os.path.exists(extra_state_path):
            print(f"Loading Coconut extras: {extra_state_path}")
            extra_state = torch.load(extra_state_path, map_location="cpu")
            coconut_model.load_state_dict(extra_state, strict=False)

    coconut_model = coconut_model.to(device)
    if family == "llama":
        coconut_model = coconut_model.to(torch.bfloat16)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return coconut_model, base_model, tokenizer, latent_id, start_id, end_id, checkpoint_path

# ═══════════════════════════════════════════════════════════════════
# Model loading: CODI
# ═══════════════════════════════════════════════════════════════════

def setup_codi_model(task, device, use_prj=True, prj_dim=None, remove_eos=True, family="gpt2"):
    # prj_dim defaults: GPT-2 CODI ckpts use 768; Llama CODI ckpts use 2048
    # (matches codiModel.TrainingArguments.prj_dim default of 2048 and the
    # Llama-3.2-1B hidden_size). Pass explicitly to override.
    if prj_dim is None:
        prj_dim = 768 if family == "gpt2" else 2048
    """Load CODI model matching test.py configuration.
    
    Args:
        remove_eos: If True (default, matches the training config), the input
            format is [question] [bot] and the eot delimiter is [eot] only.
            If False, input is [question] [eos] [bot] and delimiter is [eot] [eos].
    """
    from src.config import BASE_GPT2, BASE_LLAMA
    
    if task == "prosqa":
        model_path = PROSQA_MODELS
    else:
        model_path = GSM_MODELS
    
    codi_dir = model_path / family / "codi"

    if family == "gpt2":
        base_model_path = str(BASE_GPT2)
        tokenizer_path = "gpt2"
    elif family == "llama":
        base_model_path = str(BASE_LLAMA)
        tokenizer_path = str(BASE_LLAMA)
    else:
        raise ValueError(f"Unsupported model family for CODI: {family}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        use_fast=False,         # Match testing script
        padding_side="left",    # Match testing script
        model_max_length=1024   # Match testing script
    )
    ori_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(ori_vocab_size + 3)
    bot_id = ori_vocab_size + 1
    eot_id = ori_vocab_size + 2
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = ori_vocab_size 
    
    hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))

    from peft import LoraConfig, TaskType, get_peft_model
    
    if family == "gpt2":
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=True,
        r=128, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=target_modules, 
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)

    prj = None
    if use_prj:
        prj = nn.Sequential(
            nn.Dropout(0.0), nn.Linear(hidden_size, prj_dim),
            nn.GELU(), nn.Linear(prj_dim, hidden_size),
        )
        prj.add_module("ln", nn.LayerNorm(hidden_size))

    ckpt_sf = codi_dir / "model.safetensors"
    ckpt_bin = codi_dir / "pytorch_model.bin"
    if ckpt_sf.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_sf))
        print(f"[CODI] Loaded from {ckpt_sf}")
    elif ckpt_bin.exists():
        state_dict = torch.load(str(ckpt_bin), map_location="cpu", weights_only=False)
        print(f"[CODI] Loaded from {ckpt_bin}")
    else:
        raise FileNotFoundError(f"No CODI checkpoint in {codi_dir}")

    prj_sd, codi_sd = {}, {}
    for k, v in state_dict.items():
        if k.startswith('prj.'):
            prj_sd[k] = v
        else:
            clean_k = k[len("codi."):] if k.startswith("codi.") else k
            codi_sd[clean_k] = v

    missing, unexpected = model.load_state_dict(codi_sd, strict=False)
    print(f"[CODI] Base: {len(codi_sd)-len(unexpected)} loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    if prj is not None and prj_sd:
        prj_local = {k[len("prj."):]: v for k, v in prj_sd.items()}
        prj_missing, prj_unexpected = prj.load_state_dict(prj_local, strict=False)
        print(f"[CODI] Projection: {len(prj_local)-len(prj_unexpected)} loaded, "
              f"{len(prj_missing)} missing, {len(prj_unexpected)} unexpected")

    model.tie_weights()
    # CODI test.py: model.to(torch.bfloat16) — must match training precision
    model = model.to(device).to(torch.bfloat16).eval()
    if prj is not None:
        prj = prj.to(device).to(torch.bfloat16).eval()

    if family == "gpt2":
        embedding_fn = model.get_base_model().transformer.wte
    else:
        embedding_fn = model.get_base_model().model.embed_tokens
        
    lm_head_fn = model.get_base_model().lm_head

    return {
        'model': model, 'prj': prj, 'tokenizer': tokenizer,
        'embedding_fn': embedding_fn, 'lm_head': lm_head_fn,
        'bot_id': bot_id, 'eot_id': eot_id,
        'hidden_size': hidden_size, 'use_prj': use_prj,
        'remove_eos': remove_eos, 'ori_vocab_size': ori_vocab_size,
    }


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


# ═══════════════════════════════════════════════════════════════════
# Serialization helpers
# ═══════════════════════════════════════════════════════════════════

def deep_convert(obj):
    """Recursively convert numpy/torch types to native Python for JSON."""
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


def normalize_text_for_flip(text):
    """
    Canonicalize raw decoded text for flip comparison.

    Strip whitespace and lowercase so that trivial formatting differences
    don't register as flips. We intentionally do NOT parse numbers or
    extract labels here — two different garbled outputs should register
    as two different strings, not collapse into one 'inf' bucket.
    """
    if text is None:
        return ""
    return text.strip().lower()


# ═══════════════════════════════════════════════════════════════════
# Intervention closures
# ═══════════════════════════════════════════════════════════════════

def make_projection_intervention(projections, device):
    """
    Dtype-safe nullspace/subspace projection intervention.

    Builds P_tensors in float32, upcasts h for the matmul, casts back:
        # h_proj = (P_t @ h.float()).to(h.dtype)
    """
    P_tensors = {
        t: torch.tensor(P, dtype=torch.float32, device=device)
        for t, P in projections.items()
    }

    def intervention_fn(h, t):
        if t not in P_tensors:
            return h
        orig_dtype = h.dtype
        return (P_tensors[t] @ h.float()).to(orig_dtype)

    return intervention_fn


# ═══════════════════════════════════════════════════════════════════
# Eval loops (non-batched, per-instance)
# ═══════════════════════════════════════════════════════════════════

def run_eval_with_intervention(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, intervention_fn, label="",
    start_id=None, latent_id=None, task="prosqa",
):
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")
        r = run_intervened_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
            start_id=start_id, latent_id=latent_id, task=task,
        )
        if r["is_correct"]:
            n_correct += 1
    accuracy = n_correct / len(data)
    print(f"    [{label}] Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return accuracy


# ═══════════════════════════════════════════════════════════════════
# CODI: non-batched eval wrappers
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_codi_single_alpha(
    codi_dict, sample, n_thoughts, device, intervention_fn,
    task="gsm",
):
    """
    Unbatched CODI inference with a per-step intervention_fn(h, t) -> h'.

    Used for:
      - baseline (intervention_fn = identity)
      - ablation (intervention_fn = make_projection_intervention)
      - sanity-check reference (one call per alpha, compared against
        batched paths).

    Mirrors random_corruption.run_codi_corruption_eval but factored so
    any (h, t) -> h' callable can be plugged in.

    Args:
        task: "prosqa" or "gsm" — controls answer extraction logic.

    Returns: {"predicted": str, "correct": str, "is_correct": bool,
              "text": str}
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']
    bot_id = codi_dict['bot_id']

    # ── Step 0: encode prompt [question] ([eos]?) [bot] ──
    question_tokens = tokenizer.encode(
        sample["question"].strip().replace('  ', ' '),
        add_special_tokens=True,
    )
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    L = input_ids.size(1)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    outputs = base_model(
        input_ids=input_ids, use_cache=True, output_hidden_states=True,
        attention_mask=attention_mask, position_ids=position_ids,
    )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]
    h = intervention_fn(h, 0)

    latent = h.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    # ── Steps 1..K ──
    running_mask = attention_mask
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        pos_t = torch.tensor([[L + t - 1]], device=device)

        outputs = base_model(
            inputs_embeds=latent, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
            attention_mask=running_mask, position_ids=pos_t,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][0, -1, :]
        h = intervention_fn(h, t)

        latent = h.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    # ── eot + ([eos]?) ──
    if remove_eos:
        eot_row = [eot_id]
    else:
        eot_row = [eot_id, tokenizer.eos_token_id]
    eot_ids = torch.tensor([eot_row], device=device)
    eot_emb = embedding_fn(eot_ids)
    eot_len = eot_emb.size(1)

    eot_pos = torch.arange(L + n_thoughts, L + n_thoughts + eot_len,
                           device=device).unsqueeze(0)
    running_mask = torch.cat(
        [running_mask, torch.ones((1, eot_len), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )
    outputs = base_model(
        inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
        attention_mask=running_mask, position_ids=eot_pos,
    )
    past_kv = outputs.past_key_values
    vocab_size = base_model.config.vocab_size
    next_logits = outputs.logits[0, -1, :vocab_size - 1]

    current_pos = L + n_thoughts + eot_len

    # ── Greedy decode (embedding feed, eot excluded by clip) ──
    generated = []
    for _ in range(MAX_DECODE_TOKENS):
        next_token = next_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        next_emb = embedding_fn(
            torch.tensor([next_token], device=device)
        ).unsqueeze(0)
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        decode_pos = torch.tensor([[current_pos]], device=device)
        out = base_model(
            inputs_embeds=next_emb,
            past_key_values=past_kv,
            use_cache=True,
            attention_mask=running_mask,
            position_ids=decode_pos,
        )
        next_logits = out.logits[0, -1, :vocab_size - 1]
        past_kv = out.past_key_values
        current_pos += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    answer, correct_answer, is_correct = _compare_answers(text, sample, task)
    return {"predicted": answer, "correct": correct_answer,
            "is_correct": is_correct, "text": text}


def run_codi_eval_with_intervention(
    codi_dict, data, n_thoughts, device, intervention_fn, label="",
    task="gsm",
):
    """CODI equivalent of run_eval_with_intervention (for ablation)."""
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")
        r = run_codi_single_alpha(
            codi_dict, sample, n_thoughts, device, intervention_fn,
            task=task,
        )
        if r["is_correct"]:
            n_correct += 1
    accuracy = n_correct / len(data)
    print(f"    [{label}] Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return accuracy


def run_codi_baseline(codi_dict, data, n_thoughts, device, task="gsm"):
    """Run CODI baseline (identity intervention), returning
    (baseline_acc, baseline_texts)."""
    identity_fn = lambda h, t: h
    baseline_texts = []
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [Baseline] {idx}/{len(data)}")
        r = run_codi_single_alpha(
            codi_dict, sample, n_thoughts, device, identity_fn,
            task=task,
        )
        baseline_texts.append(r["text"])
        if r["is_correct"]:
            n_correct += 1
    return n_correct / len(data), baseline_texts


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU sharding utilities
# ═══════════════════════════════════════════════════════════════════

def _shard_indices(n_items, world_size, rank):
    """Contiguous shard assignment: rank r handles items [r*chunk, (r+1)*chunk)."""
    # chunk = ceil(n_items / world_size); last rank may be shorter
    chunk = (n_items + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_items)
    return list(range(start, end))


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


# ═══════════════════════════════════════════════════════════════════
# Steering regime diagnostics
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
#       (or a random subspace) breaks the forward pass.
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

    median_per_t = h_norms.median(dim=0).values
    p10_per_t = h_norms.quantile(0.1, dim=0)
    p90_per_t = h_norms.quantile(0.9, dim=0)
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

    print(f"  {'alpha':>10}  {'r (pooled)':>12}  {'regime (pooled)':>18}  "
          f"regime by t [t=0..T-1]")
    print(f"  {'-'*10}  {'-'*12}  {'-'*18}  {'-'*40}")
    for r in regime_info["regimes_per_alpha"]:
        per_t_str = " ".join(lab[0] for lab in r["regime_per_t"])
        print(f"  {r['alpha']:>10g}  {r['ratio_pooled']:>12.3f}  "
              f"{r['regime_pooled']:>18}  {per_t_str}")
    print(f"  (per-t legend: G=GENUINE, T=TRANSITION, M=MAGNITUDE)")
    print()