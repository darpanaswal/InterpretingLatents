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
    python -m experiments.probe_thoughts.logit_lens --task prosqa --model pause --k 6
    python -m experiments.probe_thoughts.logit_lens --task gsm --model coconut --k 3
    python -m experiments.probe_thoughts.logit_lens --task gsm --model codi --k 6
"""

import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST
from src.utils import setup_model_and_tokenizer as setup_coconut_model, setup_codi_model


# ═══════════════════════════════════════════════════════════════════
# GSM8k CoT parsing
# ═══════════════════════════════════════════════════════════════════

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
        tok_str = tokenizer.decode([input_token_ids[idx]])
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
        question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)
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

        # End-token attention mass
        end_attn = last_attn[:, -1, :].mean(dim=0).float().cpu().numpy()
        n_input = len(question_tokens) + 1  # prompt + <start_latent>
        end_mass = compute_attention_mass(end_attn, n_input, k)

        input_token_ids = question_tokens

    else:
        # Coconut recurrence
        input_ids = tokenizer.encode(
            question_text + "\n<|start-latent|>", return_tensors="pt"
        ).to(device)
        input_token_ids_full = input_ids[0].tolist()

        # Step 0: process prompt
        outputs = base_model(
            input_ids=input_ids,
            output_hidden_states=True, use_cache=True, output_attentions=True,
        )
        h = outputs.hidden_states[-1][0, -1, :]
        past_kv = outputs.past_key_values
        hidden_states.append(h)

        # Attention of the last position (start_latent) over prompt
        # outputs.attentions[-1]: (1, n_heads, seq_len, seq_len)
        # Last position's attention over all prior positions
        attn_h0 = outputs.attentions[-1][0, :, -1, :].mean(dim=0).float().cpu().numpy()
        per_thought_attended.append(
            get_top_attended_tokens(attn_h0, input_token_ids_full, tokenizer, attn_top_k)
        )

        # Steps 1..K: recurrence
        for t in range(k):
            outputs = base_model(
                inputs_embeds=h.unsqueeze(0).unsqueeze(0),
                past_key_values=past_kv,
                output_hidden_states=True, use_cache=True, output_attentions=True,
            )
            h = outputs.hidden_states[-1][0, 0, :]
            past_kv = outputs.past_key_values
            hidden_states.append(h)

            # This step's attention: (1, n_heads, 1, kv_len)
            # The single new position attends over all KV cache positions
            attn_ht = outputs.attentions[-1][0, :, 0, :].mean(dim=0).float().cpu().numpy()
            per_thought_attended.append(
                get_top_attended_tokens(attn_ht, input_token_ids_full, tokenizer, attn_top_k)
            )

        # <end-latent> step for attention mass
        end_input = torch.tensor([[end_id]], device=device)
        outputs = base_model(
            input_ids=end_input, past_key_values=past_kv,
            output_hidden_states=True, use_cache=True, output_attentions=True,
        )
        end_attn = outputs.attentions[-1][0, :, 0, :].mean(dim=0).float().cpu().numpy()
        n_input = len(input_token_ids_full)
        end_mass = compute_attention_mass(end_attn, n_input, k)

        input_token_ids = input_token_ids_full

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
        end_attn_mass: None (CODI has no <end-latent> token)
        input_token_ids: list of prompt token IDs
    """
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    use_prj = codi_dict['use_prj']

    # Tokenize: question + [eos, bot]
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    if codi_dict.get('remove_eos', True):
        input_ids_list = question_tokens + [bot_id]
    else:
        input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([input_ids_list], device=device)

    hidden_states = []
    per_thought_attended = []

    # Step 0: encode question + <bot>
    outputs = base_model(
        input_ids=input_ids, use_cache=True,
        output_hidden_states=True, output_attentions=True,
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
    for t in range(k):
        outputs = base_model(
            inputs_embeds=latent, use_cache=True,
            output_hidden_states=True, output_attentions=True,
            past_key_values=past_kv,
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

    return hidden_states, per_thought_attended, None, input_ids_list


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
    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
        setup_coconut_model("prosqa", args.model, device)
    lm_head = base_model.get_output_embeddings()
    data = load_data("prosqa", args.max_instances)
    k = args.k

    results = []
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"  [{args.model}] {idx}/{len(data)}")

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

    mean_prompt = np.mean([r['end_attn_mass']['attn_prompt'] for r in results])
    mean_latent = np.mean([r['end_attn_mass']['attn_latent'] for r in results])
    mean_self = np.mean([r['end_attn_mass']['attn_self'] for r in results])
    uses = mean_latent >= 0.05

    print(f"\n  ATTENTION MASS (mean over {len(data)} instances):")
    print(f"    Prompt:  {mean_prompt:.2%}")
    print(f"    Latent:  {mean_latent:.2%}")
    print(f"    Self:    {mean_self:.2%}")
    print(f"    -> {'USES' if uses else 'IGNORES'} latent tokens")

    n_degen = sum(1 for r in results if r['is_degenerate'])
    frac = n_degen / len(data)
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
    out_dir = BASE_DIR / "outputs" / "logit_lens"
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {
        'task': 'prosqa', 'model': args.model, 'k': k, 'n': len(data),
        'attention': {'mean_prompt': float(mean_prompt), 'mean_latent': float(mean_latent),
                      'mean_self': float(mean_self), 'uses_latent': bool(uses)}, # <-- Cast to bool() here
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


def run_gsm_experiment(args):
    device = torch.device(args.device)

    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(device)
        tokenizer = codi_dict['tokenizer']
        lm_head = codi_dict['lm_head']
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_coconut_model("gsm", args.model, device)
        lm_head = base_model.get_output_embeddings()

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

    # Attention mass (Coconut/Pause only — CODI has no <end-latent>)
    if attn_mass_results:
        mp = np.mean([r['attn_prompt'] for r in attn_mass_results])
        ml = np.mean([r['attn_latent'] for r in attn_mass_results])
        ms = np.mean([r['attn_self'] for r in attn_mass_results])
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
        hr = np.mean(step_hits[t])
        sr = np.mean(step_superposition[t])
        ar = np.mean(step_alignment[t])
        print(f"  {t:>6}  {hr:>10.1%}  {sr:>10.1%}  {ar:>10.1%}  {n:>6}")
        summary[str(t)] = {'hit_rate': float(hr), 'superposition_rate': float(sr),
                           'alignment_rate': float(ar), 'n': n}

    all_hits = [v for vals in step_hits.values() for v in vals]
    all_super = [v for vals in step_superposition.values() for v in vals]
    print(f"\n  OVERALL:")
    print(f"    Any intermediate hit:         {np.mean(all_hits):.1%}")
    print(f"    Superposition (>=2 different): {np.mean(all_super):.1%}")

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
    out_dir = BASE_DIR / "outputs" / "logit_lens"
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {
        'task': 'gsm', 'model': args.model, 'k': k, 'top_k': top_k,
        'n_instances': n_valid, 'n_skipped': n_skipped,
        'summary': summary,
        'overall': {'hit_rate': float(np.mean(all_hits)),
                    'superposition_rate': float(np.mean(all_super))},
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
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8,
                        help="Top-k tokens for logit lens (GSM only).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.model == "codi" and args.task != "gsm":
        parser.error("CODI is only available for --task gsm")

    if args.task == "prosqa":
        run_prosqa_experiment(args)
    elif args.task == "gsm":
        run_gsm_experiment(args)


if __name__ == "__main__":
    main()