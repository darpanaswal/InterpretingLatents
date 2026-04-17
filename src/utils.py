import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from contThought.coconut import Coconut
from src.config import BASE_GPT2, PROSQA_MODELS, GSM_MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer_number(text):
    """Extract the last number from text as a float.
    Matches CODI test.py's extract_answer_number exactly.
    Returns float('inf') if no number is found."""
    text = text.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', text)
    if not pred:
        return float('inf')
    return float(pred[-1])


def _compare_answers(predicted_text, sample, task):
    """Compare predicted answer to gold, returning (predicted_str, correct_str, is_correct).

    For ProsQA: string comparison on the concept after '#### ' or '#'.
    For GSM: numeric comparison via extract_answer_number.
    """
    if task == "gsm":
        pred_num = extract_answer_number(predicted_text)
        gold_text = sample.get("answer", "").replace(",", "").strip()
        # Gold answer may contain '####'; extract the part after it
        if "####" in gold_text:
            gold_text = gold_text.split("####")[-1].strip()
        gold_num = extract_answer_number(gold_text)
        return str(pred_num), str(gold_num), pred_num == gold_num
    else:
        # ProsQA: original string comparison
        answer = predicted_text.split("#")[-1].replace(",", "").strip()
        correct_answer = sample.get("answer", "").replace(",", "").strip()
        return answer, correct_answer, answer == correct_answer

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

def get_checkpoint_path(task, mode):
    model_path = PROSQA_MODELS if task == "prosqa" else GSM_MODELS
    if mode == "base":
        return None
    if mode == "cot":
        if model_path == PROSQA_MODELS:
            return str(model_path / "cot/best_checkpoint.pt")
        else:
            return str(model_path / "cot/checkpoint_best")
    if mode == "pause":
        return str(model_path / "pause/checkpoint_best")
    if mode == "coconut":
        return str(model_path / "coconut/checkpoint_best")
    if mode == "coconut_u":
        return str(model_path / "coconut-u0.3/checkpoint_best")
    raise ValueError(f"Unsupported mode: {mode}")

def setup_model_and_tokenizer(task, mode, device):
    """
    Load GPT-2, add Coconut special tokens, wrap in Coconut class.

    Loading order matters and differs by mode:

    --mode base:
        1. Load pretrained GPT-2.
        2. Add special tokens, resize embeddings, init with "<<".
        3. Wrap in Coconut.

    --mode cot:
        1. Load pretrained GPT-2.
        2. Load CoT checkpoint (plain GPT-2 state dict, original vocab).
        3. Add special tokens, resize embeddings, init with "<<".
        4. Wrap in Coconut.

    --mode coconut / coconut_u / pause:
        1. Load pretrained GPT-2.
        2. Add special tokens, resize embeddings, init with "<<".
        3. Wrap in Coconut.
        4. Load checkpoint (keys: base_causallm.*).
    """
    checkpoint_path = get_checkpoint_path(task, mode)

    model = AutoModelForCausalLM.from_pretrained(BASE_GPT2)
    tokenizer = AutoTokenizer.from_pretrained(BASE_GPT2)
    tokenizer.pad_token = tokenizer.eos_token

    # --- CoT: load checkpoint BEFORE adding special tokens ---
    if mode == "cot":
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

    # Resize embeddings — run.py line 147
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings with "<<" — run.py lines 149-157
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    # --- Wrap in Coconut (all modes) ---
    feedback_mode = "pause_curriculum" if mode == "pause" else "continuous"
    coconut_model = Coconut(model, latent_id, start_id, end_id,
                            tokenizer.eos_token_id,
                            feedback_mode=feedback_mode)


    # --- Coconut/Pause: load checkpoint AFTER wrapping ---
    if mode in ("pause", "coconut", "coconut_u"):
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

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return coconut_model, base_model, tokenizer, latent_id, start_id, end_id, checkpoint_path

# ═══════════════════════════════════════════════════════════════════
# Model loading: CODI
# ═══════════════════════════════════════════════════════════════════

def setup_codi_model(device, use_prj=True, prj_dim=768, remove_eos=True):
    """Load CODI model matching test.py configuration.
    
    Args:
        remove_eos: If True (default, matches the training config), the input
            format is [question] [bot] and the eot delimiter is [eot] only.
            If False, input is [question] [eos] [bot] and delimiter is [eot] [eos].
    """
    codi_dir = GSM_MODELS / "codi"

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_GPT2), torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_GPT2))
    ori_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(ori_vocab_size + 3)
    bot_id = ori_vocab_size + 1
    eot_id = ori_vocab_size + 2
    tokenizer.pad_token = tokenizer.eos_token
    hidden_size = model.config.n_embd

    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True,
        r=128, lora_alpha=32, lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True,
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
        prj.load_state_dict(prj_local, strict=False)
        print(f"[CODI] Projection: {len(prj_local)} keys loaded")

    model.tie_weights()
    # CODI test.py: model.to(torch.bfloat16) — must match training precision
    model = model.to(device).to(torch.bfloat16).eval()
    if prj is not None:
        prj = prj.to(device).to(torch.bfloat16).eval()

    embedding_fn = model.get_base_model().transformer.wte
    lm_head_fn = model.get_base_model().lm_head

    return {
        'model': model, 'prj': prj, 'tokenizer': tokenizer,
        'embedding_fn': embedding_fn, 'lm_head': lm_head_fn,
        'bot_id': bot_id, 'eot_id': eot_id,
        'hidden_size': hidden_size, 'use_prj': use_prj,
        'remove_eos': remove_eos, 'ori_vocab_size': ori_vocab_size,
    }