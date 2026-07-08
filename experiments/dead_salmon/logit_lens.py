"""
Logit Lens: Attention + Logit Lens across datasets and model types.

For each thought position, two analyses are performed:
  1. DECODED TOKENS (Logit Lens): project hidden state through LM head,
     report top-k tokens the model is "thinking" at this step.
  2. ATTENDED TOKENS: which input tokens does this thought attend to most
     strongly? (Last-layer attention, averaged over heads.)

Additionally for GSM8k:
  3. INTERMEDIATE RESULT TRACKING: do decoded tokens match CoT intermediates?
  4. SUPERPOSITION: do >=2 different intermediates appear simultaneously?

And for all tasks:
  5. ATTENTION MASS: fraction of attention on prompt vs latent vs self.

Usage:
    python -m experiments.dead_salmon.logit_lens --task prosqa --model pause --k 6
    python -m experiments.dead_salmon.logit_lens --task gsm --model coconut --k 6
    python -m experiments.dead_salmon.logit_lens --task gsm --model codi --k 6
"""

import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST
from src.utils import setup_model_and_tokenizer as setup_coconut_model, setup_codi_model
from src.bootstrap_stats import report_mean_with_ci


# ═══════════════════════════════════════════════════════════════════
# GSM8k CoT parsing
# ═══════════════════════════════════════════════════════════════════

def maybe_force_fp32(args, coconut_model, base_model, lm_head):
    """Optionally upcast a Llama model to fp32 for the logit lens.

    setup_coconut_model casts Llama to bf16 (utils.py). bf16 round-trip of
    fed-back hidden states (|h| ~ 100) loses ~2-3 significant digits and can
    accumulate across recurrence steps. --fp32 isolates dtype effects from the
    extraction/position-faithfulness fix. No-op for GPT-2 (already fp32).
    """
    if not getattr(args, "fp32", False) or args.model_family != "llama":
        return coconut_model, base_model, lm_head
    coconut_model = coconut_model.float()
    base_model = coconut_model.base_causallm
    lm_head = base_model.get_output_embeddings()
    print("  [fp32] Llama upcast to float32 for logit lens.")
    return coconut_model, base_model, lm_head


def parse_gsm_steps(steps_list):
    """
    Parse "steps" field: ["<<16-3-4=9>>", "<<9*2=18>>"]
    Returns: [{'expression': '16-3-4', 'result': '9', 'full': '16-3-4=9'}, ...]
    """
    parsed = []
    for step_str in steps_list:
        inner = step_str.strip().lstrip('<').rstrip('>')
        parts = inner.split('=')
        if len(parts) == 2:
            parsed.append({
                'expression': parts[0].strip(),
                'result': parts[1].strip(),
                'full': inner,
            })
    return parsed


def extract_all_intermediate_numbers(cot_steps):
    return {step['result'] for step in cot_steps}


def check_token_matches_intermediate(decoded_token, intermediate_numbers):
    cleaned = decoded_token.strip()
    if cleaned in intermediate_numbers:
        return True, cleaned
    return False, None

# ═══════════════════════════════════════════════════════════════════
# Attention utilities
# ═══════════════════════════════════════════════════════════════════

def get_top_attended_tokens(attn_over_positions, input_token_ids, tokenizer, top_k=10):
    """
    Given attention weights over all KV positions and the corresponding
    input token IDs, return the top-k most-attended input tokens.

    Args:
        attn_over_positions: 1D numpy array of attention weights, length = KV length
        input_token_ids: list of token IDs for the input sequence (prompt only,
                         NOT including latent/thought positions which have no token ID)
        tokenizer: for decoding token IDs
        top_k: how many to return

    Returns: list of (decoded_token_str, attention_weight) for the top-k
             attended positions that fall within the input token range.
    """
    n_input = len(input_token_ids)
    # Only consider attention over input positions (not thought/latent positions)
    input_attn = attn_over_positions[:n_input]

    top_indices = np.argsort(input_attn)[::-1][:top_k]
    results = []
    for idx in top_indices:
        tid = input_token_ids[idx]
        try:
            tok_str = tokenizer.decode([tid])
        except (TypeError, KeyError):
            # CODI special tokens (bot, eot, pad) are added by index
            # without registering in the tokenizer vocabulary.
            tok_str = f"<special:{tid}>"
        results.append((tok_str, float(input_attn[idx])))
    return results


def compute_attention_mass(attn_over_positions, n_input_tokens, n_thought_positions):
    """
    Break down attention into: prompt, latent/thought, self.

    For a sequence laid out as:
      [input_tokens (n_input)] [thought_positions (n_thought)] [current_position]

    attn_over_positions covers all prior positions (everything the current
    token can attend to). For the last thought or <end> token, this includes
    all input + all prior thoughts + itself.

    Args:
        attn_over_positions: 1D array of attention weights
        n_input_tokens: number of input (prompt) positions
        n_thought_positions: number of latent/thought positions before current

    Returns: dict with attn_prompt, attn_latent, attn_self
    """
    total = len(attn_over_positions)
    attn_prompt = float(attn_over_positions[:n_input_tokens].sum())
    attn_latent = float(attn_over_positions[n_input_tokens:total-1].sum()) if total > n_input_tokens + 1 else 0.0
    attn_self = float(attn_over_positions[-1])
    return {'attn_prompt': attn_prompt, 'attn_latent': attn_latent, 'attn_self': attn_self}


# ═══════════════════════════════════════════════════════════════════
# Hidden state + attention extraction: Coconut-family
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_coconut_full(
    coconut_model, base_model, tokenizer, question_text,
    k, device, start_id, latent_id, end_id, attn_top_k=10,
):
    """
    Extract hidden states, per-thought attended tokens, and end-token
    attention mass for Coconut-family models.

    Returns:
        hidden_states: list of k+1 tensors of shape (D,)
        per_thought_attended: list of k+1 lists of (token_str, weight)
        end_attn_mass: dict with attn_prompt, attn_latent, attn_self
        input_token_ids: list of prompt token IDs (for reference)
    """
    is_pause = (coconut_model.feedback_mode == "pause_curriculum")
    hidden_states = []
    per_thought_attended = []

    if is_pause:
        from src.utils import tokenize_question_for_recurrence
        question_tokens = tokenize_question_for_recurrence(tokenizer, question_text)
        input_ids_list = question_tokens + [start_id] + [latent_id] * k + [end_id]
        input_ids = torch.tensor([input_ids_list], device=device)

        inputs_embeds = coconut_model.embedding(input_ids)
        start_of_latent = len(question_tokens) + 1
        for i in range(k):
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, start_of_latent + i, :] = coconut_model.pause_embedding

        outputs = base_model(
            inputs_embeds=inputs_embeds,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]
        # Last-layer attention: (1, n_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1][0]  # (n_heads, seq_len, seq_len)

        # h_0 at <start_latent> position
        start_latent_pos = len(question_tokens)
        hidden_states.append(last_hidden[0, start_latent_pos, :])
        # Attended tokens for h_0: attention of position start_latent_pos over all prior
        attn_h0 = last_attn[:, start_latent_pos, :].mean(dim=0).float().cpu().numpy()
        per_thought_attended.append(
            get_top_attended_tokens(attn_h0, question_tokens, tokenizer, attn_top_k)
        )

        # h_1..h_k at latent positions
        for i in range(k):
            pos = start_of_latent + i
            hidden_states.append(last_hidden[0, pos, :])
            attn_hi = last_attn[:, pos, :].mean(dim=0).float().cpu().numpy()
            per_thought_attended.append(
                get_top_attended_tokens(attn_hi, question_tokens, tokenizer, attn_top_k)
            )

        # End-token attention mass — averaged across ALL layers and heads.
        # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer.
        # Stack the end-token row from each layer → (n_layers, n_heads, seq_len),
        # then mean over layers and heads → (seq_len,).
        all_layer_end_attn = torch.stack(
            [a[0, :, -1, :] for a in outputs.attentions], dim=0
        )  # (n_layers, n_heads, seq_len)
        end_attn = all_layer_end_attn.float().mean(dim=(0, 1)).cpu().numpy()
        n_input = len(question_tokens) + 1  # prompt + <start_latent>
        end_mass = compute_attention_mass(end_attn, n_input, k)

        input_token_ids = question_tokens

    else:
        # ── Coconut continuous recurrence (M2) ──
        #
        # FAITHFUL to training (coconut.py Coconut.forward continuous path +
        # dataset.py:209 layout). The previous implementation used single-token
        # forwards through a kv-cache with NO latent tokens in the sequence and
        # implicit position_ids; under RoPE (Llama) that injects wrong positions
        # at every layer, pushing fed-back states off the trained manifold
        # (decode → '<<' / code-token garbage). GPT-2 (learned absolute pos)
        # tolerated it, which is why only Llama collapsed.
        #
        # Training layout (dataset.py get_question_latent_dataset):
        #     tokens       = question + [start] + [latent]*k + [end]
        #     position_ids = list(range(len(tokens)))          # contiguous
        # Continuous loop (coconut.py): latent slot p is overwritten with the
        # LAST-LAYER hidden state at position p-1:
        #     tensor_list[b][p] = hidden_states[b, p - 1 - offset, :]
        # i.e. the thought injected at slot p is h[p-1], NOT h[p]. We replicate
        # this exactly: full-sequence forward, fill slots left-to-right, one
        # forward per fill so each thought sees previously-filled thoughts.
        from src.utils import tokenize_question_for_recurrence
        question_tokens = tokenize_question_for_recurrence(tokenizer, question_text)

        input_ids_list = question_tokens + [start_id] + [latent_id] * k + [end_id]
        input_ids = torch.tensor([input_ids_list], device=device)
        L = input_ids.shape[1]

        # Contiguous positions + full mask, exactly as dataset.py builds them.
        position_ids = torch.arange(L, device=device).unsqueeze(0)
        attn_mask = torch.ones((1, L), device=device, dtype=torch.long)

        start_latent_pos = len(question_tokens)          # <start-latent> position
        latent_positions = [start_latent_pos + 1 + i for i in range(k)]  # k latent slots

        inputs_embeds = coconut_model.embedding(input_ids)

        # h_0: hidden state at <start-latent>, read from the first (all-slots-empty)
        # pass — analogous to the pause branch's h_0 and the original t=0 point.
        first_outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask, position_ids=position_ids,
            output_hidden_states=True, output_attentions=True, use_cache=False,
        )
        last_hidden = first_outputs.hidden_states[-1]
        last_attn = first_outputs.attentions[-1][0]      # (n_heads, L, L)

        hidden_states.append(last_hidden[0, start_latent_pos, :])
        attn_h0 = last_attn[:, start_latent_pos, :].mean(dim=0).float().cpu().numpy()
        per_thought_attended.append(
            get_top_attended_tokens(attn_h0, question_tokens, tokenizer, attn_top_k)
        )

        # Sequential refill: slot p ← h[p-1] from the current pass, then re-run.
        # Matches coconut.py's max_n_latents passes (one fill per pass).
        for i, p in enumerate(latent_positions):
            outputs = base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask, position_ids=position_ids,
                output_hidden_states=True, output_attentions=True, use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]
            last_attn = outputs.attentions[-1][0]        # (n_heads, L, L)

            h_prev = last_hidden[0, p - 1, :]            # thought injected at slot p
            hidden_states.append(h_prev)

            # Attention of slot p over all positions (this thought's attention).
            attn_hi = last_attn[:, p, :].mean(dim=0).float().cpu().numpy()
            per_thought_attended.append(
                get_top_attended_tokens(attn_hi, question_tokens, tokenizer, attn_top_k)
            )

            # Fill slot p for subsequent passes.
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, p, :] = h_prev

        # End-token attention mass: read the <end-latent> row from the final pass
        # (all latent slots filled), averaged across ALL layers and heads.
        final_outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask, position_ids=position_ids,
            output_hidden_states=True, output_attentions=True, use_cache=False,
        )
        end_pos = L - 1                                  # <end-latent> position
        all_layer_end_attn = torch.stack(
            [a[0, :, end_pos, :] for a in final_outputs.attentions], dim=0
        )  # (n_layers, n_heads, L)
        end_attn = all_layer_end_attn.float().mean(dim=(0, 1)).cpu().numpy()
        n_input = len(question_tokens) + 1               # prompt + <start-latent>
        end_mass = compute_attention_mass(end_attn, n_input, k)

        input_token_ids = question_tokens

    return hidden_states, per_thought_attended, end_mass, input_token_ids


# ═══════════════════════════════════════════════════════════════════
# Hidden state + attention extraction: CODI
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_codi_full(codi_dict, question_text, k, device, attn_top_k=10):
    """
    Extract hidden states + per-thought attended tokens for CODI.

    Matches CODI's test.py recurrence with output_attentions=True.

    Returns:
        hidden_states: list of k+1 tensors (BEFORE projection)
        per_thought_attended: list of k+1 lists of (token_str, weight)
        end_attn_mass: dict with attn_prompt, attn_latent, attn_self
            (measured at the [eot] token, analogous to <end-latent> in Coconut)
        input_token_ids: list of prompt token IDs
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    use_prj = codi_dict['use_prj']

    # Tokenize: question + [eos, bot]
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    if codi_dict.get('remove_eos', True):
        input_ids_list = question_tokens + [bot_id]
    else:
        input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([input_ids_list], device=device)
    attention_mask = torch.ones_like(input_ids)           # (1, L)

    hidden_states = []
    per_thought_attended = []

    # Position IDs: no padding (B=1), so just arange.
    # pos_ids[0, j] = j,  j = 0, ..., L-1
    position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    # real_len = L (scalar, used to compute recurrence positions)
    real_len = input_ids.size(1)

    # Step 0: encode question + <bot>
    outputs = base_model(
        input_ids=input_ids, use_cache=True,
        output_hidden_states=True, output_attentions=True,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    past_kv = outputs.past_key_values

    h = outputs.hidden_states[-1][0, -1, :]
    hidden_states.append(h)

    # Attention of <bot> position over the input
    attn_h0 = outputs.attentions[-1][0, :, -1, :].mean(dim=0).float().cpu().numpy()
    per_thought_attended.append(
        get_top_attended_tokens(attn_h0, input_ids_list, tokenizer, attn_top_k)
    )

    # Project for feeding back
    latent = h.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    # Steps 1..K
    # running_mask grows by 1 each step; position at step t = real_len + t - 1
    running_mask = attention_mask                         # (1, L)
    for t in range(1, k + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        # pos_t = real_len + t - 1
        pos_t = torch.tensor([[real_len + t - 1]], device=device)

        outputs = base_model(
            inputs_embeds=latent, use_cache=True,
            output_hidden_states=True, output_attentions=True,
            past_key_values=past_kv,
            attention_mask=running_mask,
            position_ids=pos_t,
        )
        past_kv = outputs.past_key_values

        h = outputs.hidden_states[-1][0, -1, :]
        hidden_states.append(h)

        # Attention: (1, n_heads, 1, kv_len) — new position over all KV
        attn_ht = outputs.attentions[-1][0, :, 0, :].mean(dim=0).float().cpu().numpy()
        per_thought_attended.append(
            get_top_attended_tokens(attn_ht, input_ids_list, tokenizer, attn_top_k)
        )

        latent = h.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    # ── [eot] step: capture attention mass (analogous to <end-latent> in Coconut) ──
    # The [eot] token is the decoder entry point for CODI. Its attention
    # distribution over [prompt | thoughts | self] answers the same question
    # as <end-latent> does for Coconut: does the model read from thoughts?
    #
    # Sequence layout at this point:
    #   positions 0..L-1           = prompt (question + [bot])
    #   positions L..L+k-1         = k thought vectors
    #   position  L+k              = [eot] (current token)
    #
    # n_input = L (prompt length), n_thought = k
    eot_ids = torch.tensor([[eot_id]], device=device)
    eot_pos = torch.tensor([[real_len + k]], device=device)
    running_mask = torch.cat(
        [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )
    eot_out = base_model(
        input_ids=eot_ids, use_cache=True,
        output_attentions=True,
        past_key_values=past_kv,
        attention_mask=running_mask,
        position_ids=eot_pos,
    )
    # eot attention over all KV positions, averaged across ALL layers and heads.
    #
    # outputs.attentions[-1] gives the last layer only, but information from
    # thought vectors may be consumed by earlier layers and written into the
    # residual stream. The last layer then reads processed features at prompt
    # positions, making it appear that thoughts are ignored. Averaging across
    # all layers gives a fairer picture of where information flows.
    #
    # eot_out.attentions is a tuple of (1, n_heads, 1, kv_len) per layer.
    # Stack → (n_layers, n_heads, kv_len), mean over layers and heads → (kv_len,)
    all_layer_attn = torch.stack(
        [a[0, :, 0, :] for a in eot_out.attentions], dim=0
    )  # (n_layers, n_heads, kv_len)
    eot_attn = all_layer_attn.float().mean(dim=(0, 1)).cpu().numpy()  # (kv_len,)
    end_mass = compute_attention_mass(eot_attn, len(input_ids_list), k)

    return hidden_states, per_thought_attended, end_mass, input_ids_list


# ═══════════════════════════════════════════════════════════════════
# Logit lens
# ═══════════════════════════════════════════════════════════════════

def logit_lens_topk(hidden_vec, lm_head, tokenizer, top_k=8):
    logits = lm_head(hidden_vec)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    return [
        (tokenizer.decode([top_indices[i].item()]), top_probs[i].item())
        for i in range(top_k)
    ]


# ═══════════════════════════════════════════════════════════════════
# GSM8k per-instance analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_gsm_instance(hidden_states, per_thought_attended, lm_head,
                          tokenizer, cot_steps, top_k=8):
    """
    Per-instance analysis combining logit lens + attended tokens +
    intermediate result tracking.
    """
    all_intermediates = extract_all_intermediate_numbers(cot_steps)

    per_step = []
    for t, h in enumerate(hidden_states):
        decoded = logit_lens_topk(h, lm_head, tokenizer, top_k)

        matched_numbers = set()
        for tok_str, prob in decoded:
            is_match, num = check_token_matches_intermediate(tok_str, all_intermediates)
            if is_match:
                matched_numbers.add(num)

        step_aligned = False
        if t < len(cot_steps):
            step_aligned = cot_steps[t]['result'] in matched_numbers

        attended = per_thought_attended[t] if t < len(per_thought_attended) else []

        per_step.append({
            'step': t,
            'decoded_tokens': [(tok, prob) for tok, prob in decoded],
            'attended_tokens': [(tok, w) for tok, w in attended],
            'n_intermediate_hits': len(matched_numbers),
            'has_any_hit': len(matched_numbers) > 0,
            'has_superposition': len(matched_numbers) >= 2,
            'matched_numbers': sorted(matched_numbers),
            'step_aligned': step_aligned,
        })

    return {
        'n_cot_steps': len(cot_steps),
        'all_intermediate_numbers': sorted(all_intermediates),
        'per_step': per_step,
    }


# ═══════════════════════════════════════════════════════════════════
# ProsQA per-instance analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_prosqa_instance(hidden_states, per_thought_attended, end_mass,
                             lm_head, tokenizer, top_k=5):
    """Per-instance ProsQA analysis: logit lens + attended tokens."""
    per_step = []
    for t, h in enumerate(hidden_states):
        decoded = logit_lens_topk(h, lm_head, tokenizer, top_k)
        attended = per_thought_attended[t] if t < len(per_thought_attended) else []
        per_step.append({
            'step': t,
            'decoded_tokens': decoded,
            'attended_tokens': attended,
        })

    top1_tokens = [s['decoded_tokens'][0][0].strip() for s in per_step]
    is_degenerate = len(set(top1_tokens)) == 1

    return {
        'per_step': per_step,
        'top1_tokens': top1_tokens,
        'is_degenerate': is_degenerate,
        'end_attn_mass': end_mass,
    }


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path, "r") as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    print(f"[INFO] Loaded {len(data)} {task} instances from {path}")
    return data


# ═══════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════

def run_prosqa_experiment(args):
    device = torch.device(args.device)

    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model("prosqa", device, family=args.model_family)
        tokenizer = codi_dict['tokenizer']
        lm_head = codi_dict['lm_head']
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_coconut_model("prosqa", args.model, device, family=args.model_family)
        lm_head = base_model.get_output_embeddings()
        coconut_model, base_model, lm_head = maybe_force_fp32(
            args, coconut_model, base_model, lm_head)

    data = load_data("prosqa", args.max_instances)
    k = args.k

    results = []
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"  [{args.model}] {idx}/{len(data)}")

        if is_codi:
            hidden_states, per_attended, end_mass, _ = extract_codi_full(
                codi_dict, sample["question"], k, device,
            )
        else:
            hidden_states, per_attended, end_mass, _ = extract_coconut_full(
                coconut_model, base_model, tokenizer,
                sample["question"], k, device, start_id, latent_id, end_id,
            )
        analysis = analyze_prosqa_instance(
            hidden_states, per_attended, end_mass, lm_head, tokenizer,
        )
        analysis['answer'] = sample.get('answer', '')
        results.append(analysis)

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"PROSQA: {args.model} (K={k}, N={len(data)})")
    print(f"{'='*70}")

    out_dir = BASE_DIR / "outputs" / "logit_lens" / args.model_family
    out_dir.mkdir(parents=True, exist_ok=True)
    cis_jsonl = str(out_dir / f"prosqa_{args.model}_k{k}_cis.jsonl")
    vectors_dir = out_dir / f"prosqa_{args.model}_k{k}_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    base_ctx = {'task': 'prosqa', 'model': args.model, 'family': args.model_family}

    # Attention mass (Coconut/Pause only — CODI has no <end-latent>)
    attn_results = [r for r in results if r['end_attn_mass'] is not None]
    if attn_results:
        prompt_vec = [r['end_attn_mass']['attn_prompt'] for r in attn_results]
        latent_vec = [r['end_attn_mass']['attn_latent'] for r in attn_results]
        self_vec   = [r['end_attn_mass']['attn_self']   for r in attn_results]

        ci_prompt = report_mean_with_ci(
            prompt_vec, metric="attn_prompt", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_prompt.npz"))
        ci_latent = report_mean_with_ci(
            latent_vec, metric="attn_latent", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_latent.npz"))
        ci_self = report_mean_with_ci(
            self_vec, metric="attn_self", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_self.npz"))

        mean_prompt, mean_latent, mean_self = ci_prompt.point, ci_latent.point, ci_self.point
        uses = mean_latent >= 0.05

        print(f"\n  ATTENTION MASS (mean over {len(attn_results)} instances):")
        print(f"    Prompt:  {mean_prompt:.2%}")
        print(f"    Latent:  {mean_latent:.2%}")
        print(f"    Self:    {mean_self:.2%}")
        print(f"    -> {'USES' if uses else 'IGNORES'} latent tokens")
    else:
        mean_prompt = mean_latent = mean_self = None
        uses = None
        print(f"\n  ATTENTION MASS: N/A (CODI has no <end-latent> token)")

    # is_degenerate: binary per-instance metric
    degen_vec = [int(r['is_degenerate']) for r in results]
    ci_degen = report_mean_with_ci(
        degen_vec, metric="frac_degenerate", context={**base_ctx, 'condition': 'logit_lens'},
        cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "frac_degenerate.npz"))

    n_degen = sum(degen_vec)
    frac = ci_degen.point
    label = "Degenerate" if frac > 0.8 else ("Evolving" if frac < 0.2 else "Mixed")
    print(f"\n  LOGIT LENS:")
    print(f"    Degenerate: {n_degen}/{len(data)} ({frac:.1%})")
    print(f"    Classification: {label}")

    print(f"\n  EXAMPLES (first 3):")
    for i in range(min(3, len(data))):
        r = results[i]
        print(f"    [{i}] Answer: {r['answer']}")
        print(f"         Decoded top-1: {r['top1_tokens']}")
        print(f"         -> {'DEGEN' if r['is_degenerate'] else 'EVOLV'}")
        # Show attended tokens for step 0
        if r['per_step'][0]['attended_tokens']:
            att_str = [f"{tok}({w:.3f})" for tok, w in r['per_step'][0]['attended_tokens'][:5]]
            print(f"         Attended (step 0): {att_str}")

    # ── Save ──
    save = {
        'task': 'prosqa', 'model': args.model, 'model_family': args.model_family,
        'k': k, 'n': len(data),
        'attention': {'mean_prompt': float(mean_prompt), 'mean_latent': float(mean_latent),
                      'mean_self': float(mean_self), 'uses_latent': bool(uses)} if mean_prompt is not None else None,
        'logit_lens': {'n_degenerate': n_degen, 'frac_degenerate': frac, 'label': label},
        'per_instance': [
            {'idx': i, 'answer': results[i]['answer'],
             'top1_tokens': results[i]['top1_tokens'],
             'degenerate': results[i]['is_degenerate'],
             'end_attn': results[i]['end_attn_mass'],
             'per_step': [
                 {'decoded': [(t, f"{p:.4f}") for t, p in s['decoded_tokens'][:5]],
                  'attended': [(t, f"{w:.4f}") for t, w in s['attended_tokens'][:5]]}
                 for s in results[i]['per_step']
             ]}
            for i in range(len(data))
        ],
    }
    path = out_dir / f"prosqa_{args.model}_k{k}.json"
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved to {path}")
    print(f"  CIs saved to {cis_jsonl}")


def run_gsm_experiment(args):
    device = torch.device(args.device)

    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model("gsm", device, family=args.model_family)
        tokenizer = codi_dict['tokenizer']
        lm_head = codi_dict['lm_head']
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_coconut_model("gsm", args.model, device, family=args.model_family)
        lm_head = base_model.get_output_embeddings()
        coconut_model, base_model, lm_head = maybe_force_fp32(
            args, coconut_model, base_model, lm_head)

    data = load_data("gsm", args.max_instances)
    k = args.k
    top_k = args.top_k

    step_hits = defaultdict(list)
    step_superposition = defaultdict(list)
    step_alignment = defaultdict(list)
    attn_mass_results = []
    instance_results = []
    n_skipped = 0

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"  [{args.model}] {idx}/{len(data)}")

        steps_raw = sample.get("steps", [])
        cot_steps = parse_gsm_steps(steps_raw)
        if len(cot_steps) == 0:
            n_skipped += 1
            continue

        # Extract hidden states + attention
        if is_codi:
            hidden_states, per_attended, end_mass, input_tids = extract_codi_full(
                codi_dict, sample["question"], k, device,
            )
        else:
            hidden_states, per_attended, end_mass, input_tids = extract_coconut_full(
                coconut_model, base_model, tokenizer, sample["question"],
                k, device, start_id, latent_id, end_id,
            )

        if end_mass is not None:
            attn_mass_results.append(end_mass)

        analysis = analyze_gsm_instance(
            hidden_states, per_attended, lm_head, tokenizer, cot_steps, top_k,
        )

        for sd in analysis['per_step']:
            t = sd['step']
            step_hits[t].append(sd['has_any_hit'])
            step_superposition[t].append(sd['has_superposition'])
            step_alignment[t].append(sd['step_aligned'])

        instance_results.append({
            'idx': idx,
            'n_cot_steps': analysis['n_cot_steps'],
            'intermediates': analysis['all_intermediate_numbers'],
            'per_step': [
                {'step': s['step'], 'has_hit': s['has_any_hit'],
                 'has_superposition': s['has_superposition'],
                 'matched': s['matched_numbers'], 'step_aligned': s['step_aligned'],
                 'decoded': [(t, f"{p:.4f}") for t, p in s['decoded_tokens'][:5]],
                 'attended': [(t, f"{w:.4f}") for t, w in s['attended_tokens'][:5]]}
                for s in analysis['per_step']
            ],
        })

    # ── Report ──
    n_valid = len(data) - n_skipped
    print(f"\n{'='*70}")
    print(f"GSM8K: {args.model} (K={k}, N={n_valid}, skipped={n_skipped}, top_k={top_k})")
    print(f"{'='*70}")

    out_dir = BASE_DIR / "outputs" / "logit_lens" / args.model_family
    out_dir.mkdir(parents=True, exist_ok=True)
    cis_jsonl = str(out_dir / f"gsm_{args.model}_k{k}_cis.jsonl")
    vectors_dir = out_dir / f"gsm_{args.model}_k{k}_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    base_ctx = {'task': 'gsm', 'model': args.model, 'family': args.model_family}

    # Attention mass (Coconut/Pause only — CODI has no <end-latent>)
    if attn_mass_results:
        prompt_vec = [r['attn_prompt'] for r in attn_mass_results]
        latent_vec = [r['attn_latent'] for r in attn_mass_results]
        self_vec   = [r['attn_self']   for r in attn_mass_results]

        ci_mp = report_mean_with_ci(
            prompt_vec, metric="attn_prompt", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_prompt.npz"))
        ci_ml = report_mean_with_ci(
            latent_vec, metric="attn_latent", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_latent.npz"))
        ci_ms = report_mean_with_ci(
            self_vec, metric="attn_self", context={**base_ctx, 'condition': 'end_token'},
            cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "attn_self.npz"))

        mp, ml, ms = ci_mp.point, ci_ml.point, ci_ms.point
        print(f"\n  ATTENTION MASS (<end-latent>, mean):")
        print(f"    Prompt:  {mp:.2%}")
        print(f"    Latent:  {ml:.2%}")
        print(f"    Self:    {ms:.2%}")
        print(f"    -> {'USES' if ml >= 0.05 else 'IGNORES'} latent tokens")

    print(f"\n  PER-STEP ANALYSIS:")
    print(f"  {'Step':>6}  {'Hit Rate':>10}  {'Superpos.':>10}  {'Aligned':>10}  {'N':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    summary = {}
    for t in sorted(step_hits.keys()):
        n = len(step_hits[t])
        # Bootstrap CI per timestep — one record per t
        ctx_t = {**base_ctx, 'condition': 'per_step', 't': t}
        ci_hr = report_mean_with_ci(
            step_hits[t], metric="hit_rate", context=ctx_t, cis_jsonl=cis_jsonl)
        ci_sr = report_mean_with_ci(
            step_superposition[t], metric="superposition_rate", context=ctx_t, cis_jsonl=cis_jsonl)
        ci_ar = report_mean_with_ci(
            step_alignment[t], metric="alignment_rate", context=ctx_t, cis_jsonl=cis_jsonl)

        hr, sr, ar = ci_hr.point, ci_sr.point, ci_ar.point
        print(f"  {t:>6}  {hr:>10.1%}  {sr:>10.1%}  {ar:>10.1%}  {n:>6}")
        summary[str(t)] = {'hit_rate': float(hr), 'superposition_rate': float(sr),
                           'alignment_rate': float(ar), 'n': n}

    # Overall (pooled across timesteps) — save per-instance vectors for headline
    all_hits = [v for vals in step_hits.values() for v in vals]
    all_super = [v for vals in step_superposition.values() for v in vals]
    ci_oh = report_mean_with_ci(
        all_hits, metric="overall_hit_rate", context={**base_ctx, 'condition': 'pooled'},
        cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "overall_hit_rate.npz"))
    ci_os = report_mean_with_ci(
        all_super, metric="overall_superposition_rate", context={**base_ctx, 'condition': 'pooled'},
        cis_jsonl=cis_jsonl, vector_npz=str(vectors_dir / "overall_superposition_rate.npz"))
    print(f"\n  OVERALL:")
    print(f"    Any intermediate hit:         {ci_oh.point:.1%}")
    print(f"    Superposition (>=2 different): {ci_os.point:.1%}")

    # Examples with both decoded and attended
    print(f"\n  EXAMPLES (first 3 with hits):")
    shown = 0
    for r in instance_results:
        if shown >= 3:
            break
        if not any(s['has_hit'] for s in r['per_step']):
            continue
        print(f"    [{r['idx']}] Intermediates: {r['intermediates']}")
        for s in r['per_step']:
            hit = "Y" if s['has_hit'] else " "
            sup = "S" if s['has_superposition'] else " "
            dec3 = ", ".join(f"{t}({p})" for t, p in s['decoded'][:3])
            att3 = ", ".join(f"{t}({w})" for t, w in s['attended'][:3])
            print(f"      t={s['step']}: [{hit}{sup}] matched={s['matched']}")
            print(f"        Decoded:  {dec3}")
            print(f"        Attended: {att3}")
        shown += 1

    # ── Save ──
    save = {
        'task': 'gsm', 'model': args.model, 'model_family': args.model_family,
        'k': k, 'top_k': top_k,
        'n_instances': n_valid, 'n_skipped': n_skipped,
        'summary': summary,
        'overall': {'hit_rate': float(ci_oh.point),
                    'superposition_rate': float(ci_os.point)},
        'attention_mass': {
            'mean_prompt': float(np.mean([r['attn_prompt'] for r in attn_mass_results])),
            'mean_latent': float(np.mean([r['attn_latent'] for r in attn_mass_results])),
            'mean_self': float(np.mean([r['attn_self'] for r in attn_mass_results])),
        } if attn_mass_results else None,
        'per_instance': instance_results[:50],
    }
    path = out_dir / f"gsm_{args.model}_k{k}.json"
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved to {path}")
    print(f"  CIs saved to {cis_jsonl}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="attention + logit lens + attended tokens + intermediate result tracking."
    )
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument("--model", type=str,
                        choices=["base", "cot", "pause", "coconut", "coconut_u", "codi"],
                        default="coconut")
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
        help="Base model family. Determines checkpoint paths and dtype.",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--fp32", action="store_true",
                        help="Upcast Llama to float32 for the logit lens "
                             "(isolates bf16 round-trip error; no-op for GPT-2).")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8,
                        help="Top-k tokens for logit lens (GSM only).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.task == "prosqa":
        run_prosqa_experiment(args)
    elif args.task == "gsm":
        run_gsm_experiment(args)


if __name__ == "__main__":
    main()