"""
Remove Thoughts

Sanity checks for continuous thought experiments across ProsQA and GSM8k.

Two checks:

1. IDENTITY CODEBOOK: Run run_quantized_inference with a "codebook" that is
   just the set of all thought vectors for that instance (so quantize(h) = h
   exactly). If accuracy differs from normal (unquantized) inference, our
   decoding logic is wrong.

2. POSITION ID AUDIT: Check what position IDs GPT-2 assigns during manual
   recurrence with inputs_embeds + past_key_values. GPT-2 uses absolute
   position embeddings. When past_key_values is provided, HuggingFace
   auto-computes position_ids = past_seq_len .. past_seq_len + new_seq_len - 1.
   We verify this is what actually happens, and compare to what the Coconut
   wrapper does.

Usage:
    # ProsQA
    python -m experiments.dead_salmon.remove_thoughts --task prosqa --models coconut coconut_u pause

    # GSM8k (including CODI)
    python -m experiments.dead_salmon.remove_thoughts --task gsm --models coconut coconut_u pause codi

    # Single model
    python -m experiments.dead_salmon.remove_thoughts --task gsm --models codi --max_instances 100
"""

import json
import torch
import argparse
from src.config import PROSQA_TEST, GSM_TEST, THOUGHT_ABLATION
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    extract_answer_number,
)
from src.bootstrap_stats import (
    paired_bootstrap_diff,
    mcnemar_test,
    report_mean_with_ci,
    save_record,
)

def load_data(path, max_instances=None):
    """Load JSON data from a path."""
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def format_prompt(sample, tokenizer):
    """Format prompt for coconut/pause models (not used by CODI)."""
    return tokenizer.encode(sample["question"] + " <|start-latent|>", return_tensors="pt")


# ════════════════════════════════════════════════════════════════════
# GSM answer comparison
# ════════════════════════════════════════════════════════════════════

def _gsm_gold_number(sample):
    """Extract the numeric gold answer from a GSM8k sample.

    Gold answer field may be a plain number ("18") or contain
    '####' delimiters ("... #### 18"). We extract the part after
    '####' if present, then parse the last number.
    """
    gold_text = sample.get("answer", "").replace(",", "").strip()
    if "####" in gold_text:
        gold_text = gold_text.split("####")[-1].strip()
    return extract_answer_number(gold_text, task="gsm")


# ════════════════════════════════════════════════════════════════════
# CHECK 1: Identity codebook — does our decoding logic match normal
#          inference when quantization is a no-op?
# ════════════════════════════════════════════════════════════════════

def check_identity_codebook(
    coconut_model, base_model, tokenizer, start_id, end_id, latent_id,
    data, n_thoughts, device, task="prosqa",
    title="CHECK 1: Identity codebook (normal inference accuracy)",
):
    """
    Compare normal inference to the Coconut wrapper's generate().
    If they match, our manual recurrence + decoding is correct.

    For ProsQA: answer is "X is a <concept>." — string comparison.
    For GSM8k: answer is numeric — extract last number, compare as float.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    n_correct = 0
    mismatches = []
    per_instance = []  # 1/0 per instance for bootstrap

    for idx, sample in enumerate(data):
        from src.utils import run_normal_inference_pauseaware
        result = run_normal_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device,
            start_id=start_id, latent_id=latent_id, task=task,
        )
        answer = result["predicted"]
        is_correct = result["is_correct"]
        correct_answer = result["correct"]
        full_text = result["text"]

        per_instance.append(int(is_correct))
        if is_correct:
            n_correct += 1
        else:
            mismatches.append({
                "idx": idx,
                "predicted": answer,
                "correct": correct_answer,
                "full_output": full_text[:200],
            })

        if idx < 5:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  [{idx}] {status}: predicted='{answer}' gold='{correct_answer}'")

    accuracy = n_correct / len(data)
    print(f"\n  Normal inference accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    print(f"  (This should match the checkpoint's known test accuracy.)")
    print(f"  (If it doesn't, our manual recurrence or decoding is wrong.)")

    if mismatches and len(mismatches) <= 20:
        print(f"\n  Mismatched instances:")
        for m in mismatches:
            print(f"    [{m['idx']}] predicted='{m['predicted']}' gold='{m['correct']}'")
            print(f"         output: '{m['full_output']}'")

    return accuracy, mismatches, per_instance


# ════════════════════════════════════════════════════════════════════
# CHECK 1 (CODI variant): Normal inference for CODI
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def check_codi_inference(codi_dict, data, n_thoughts, device, task="gsm",
                         batch_size=1,
                         title="CHECK 1: CODI normal inference"):
    """
    Run CODI inference in batches and check accuracy.

    Matches CODI test.py inference exactly for --greedy True.
    With batch_size > 1, uses left-padding (--padding_side=left).
    """
    import math

    print("\n" + "=" * 60)
    print(title + f" (batch_size={batch_size})")
    print("=" * 60)

    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']
    vocab_size = base_model.config.vocab_size

    n_correct = 0
    mismatches = []
    per_instance = []  # 1/0 per instance for bootstrap
    max_new_tokens = 256

    n_batches = math.ceil(len(data) / batch_size)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]
        B = len(batch)

        # ── Tokenize the batch with left-padding ──
        # Mirrors test.py: tokenizer(questions, return_tensors="pt", padding="longest")
        # Then append [bot] (or [eos, bot] if remove_eos=False) to every row.
        questions = [s["question"].strip().replace('  ', ' ') for s in batch]
        enc = tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = enc["input_ids"].to(device)          # (B, L)
        attention_mask = enc["attention_mask"].to(device)  # (B, L)

        # Append [bot] (+ optional [eos])
        if remove_eos:
            suffix = torch.tensor(
                [[bot_id]] * B, dtype=torch.long, device=device,
            )
        else:
            suffix = torch.tensor(
                [[tokenizer.eos_token_id, bot_id]] * B,
                dtype=torch.long, device=device,
            )
        input_ids = torch.cat([input_ids, suffix], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(suffix)], dim=1,
        )

        # ── POS-FIX: build position_ids from left-padded attention_mask ──
        # pos_ids[b, j] = cumsum(mask[b, :j+1]) - 1,  clamped >= 0
        # real_len[b]   = sum(mask[b, :])  = number of real tokens in row b
        # The last real token (bot_id) in row b gets position real_len[b] - 1.
        # Recurrence step t (1-indexed) gets position real_len[b] + t - 1.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        real_len = attention_mask.sum(dim=1)              # (B,)

        # ── Step 0: encode question + <bot> ──
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        # Last-position hidden state for every row in batch
        h = outputs.hidden_states[-1][:, -1, :]          # (B, D)
        latent = h.unsqueeze(1)                          # (B, 1, D)
        if use_prj and prj is not None:
            latent = prj(latent)

        # ── Steps 1..K: latent recurrence ──
        # running_mask grows by 1 each step; position_ids for row b at
        # recurrence step t (1-indexed) = real_len[b] + t - 1.
        running_mask = attention_mask                     # (B, L)
        for t in range(1, n_thoughts + 1):
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=running_mask.dtype,
                                          device=device)],
                dim=1,
            )
            # pos_t[b] = real_len[b] + t - 1
            pos_t = (real_len + (t - 1)).unsqueeze(1)    # (B, 1)

            outputs = base_model(
                inputs_embeds=latent,
                attention_mask=running_mask,
                position_ids=pos_t,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][:, -1, :]      # (B, D)
            latent = h.unsqueeze(1)
            if use_prj and prj is not None:
                latent = prj(latent)

        # ── Feed eot delimiter before decoding ──
        if remove_eos:
            eot_ids = torch.tensor([[eot_id]] * B, device=device)
        else:
            eot_ids = torch.tensor(
                [[eot_id, tokenizer.eos_token_id]] * B, device=device,
            )
        eot_emb = embedding_fn(eot_ids)                  # (B, eot_len, D)
        eot_len = eot_emb.size(1)                        # 1 or 2

        # Position IDs for eot token(s): continue from where recurrence ended.
        # After K recurrence steps, last position used = real_len + K - 1.
        # eot occupies positions real_len + K, ..., real_len + K + eot_len - 1.
        # eot_pos[b, j] = real_len[b] + K + j,  j = 0, ..., eot_len - 1
        eot_pos = (real_len + n_thoughts).unsqueeze(1) + torch.arange(
            eot_len, device=device,
        ).unsqueeze(0)                                   # (B, eot_len)
        running_mask = torch.cat(
            [running_mask, torch.ones((B, eot_len), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )

        outputs = base_model(
            inputs_embeds=eot_emb,
            attention_mask=running_mask,
            position_ids=eot_pos,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :vocab_size - 1]  # (B, V-1)

        # Track the next decode position per row:
        # current_pos[b] = real_len[b] + K + eot_len
        current_pos = real_len + n_thoughts + eot_len    # (B,)

        # ── Greedy decode, per-sequence early stopping ──
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        pred_tokens = [[] for _ in range(B)]

        for step in range(max_new_tokens):
            next_token_ids = next_logits.argmax(dim=-1)  # (B,)

            # Append to per-row token lists for not-yet-finished sequences
            for b in range(B):
                if not finished[b]:
                    tok = next_token_ids[b].item()
                    pred_tokens[b].append(tok)
                    if tok == tokenizer.eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            # Feed next-token embeddings for the whole batch (finished rows
            # compute wastefully but their outputs are ignored)
            next_emb = embedding_fn(next_token_ids).unsqueeze(1)  # (B, 1, D)
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=running_mask.dtype,
                                          device=device)],
                dim=1,
            )
            decode_pos = current_pos.unsqueeze(1)        # (B, 1)
            out = base_model(
                inputs_embeds=next_emb,
                attention_mask=running_mask,
                position_ids=decode_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            next_logits = out.logits[:, -1, :vocab_size - 1]
            past_kv = out.past_key_values
            current_pos = current_pos + 1

        # ── Decode and score each row in the batch ──
        for b, sample in enumerate(batch):
            idx = batch_start + b
            text = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)

            if task == "gsm":
                pred_ans = extract_answer_number(text, task)
                gold_ans = _gsm_gold_number(sample)
                is_correct = (pred_ans == gold_ans)
            elif task == "prosqa":
                pred_ans = extract_answer_number(text, task)
                gold_ans = sample.get("answer", "").strip()
                is_correct = (gold_ans in pred_ans) or (pred_ans == gold_ans)

            per_instance.append(int(is_correct))
            if is_correct:
                n_correct += 1
            else:
                mismatches.append({
                    "idx": idx,
                    "predicted": str(pred_ans),
                    "correct": str(gold_ans),
                    "full_output": text[:200],
                })

            if idx < 5:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"  [{idx}] {status}: predicted='{pred_ans}' gold='{gold_ans}'")

    accuracy = n_correct / len(data)
    print(f"\n  Normal inference accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")

    if mismatches and len(mismatches) <= 20:
        print(f"\n  Mismatched instances:")
        for m in mismatches:
            print(f"    [{m['idx']}] predicted='{m['predicted']}' gold='{m['correct']}'")

    return accuracy, mismatches, per_instance


# ════════════════════════════════════════════════════════════════════
# CHECK 2: Position ID audit
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def check_position_ids(base_model, tokenizer, sample, n_thoughts, device):
    """
    Trace what position IDs GPT-2 uses during manual recurrence.

    HuggingFace GPT-2 auto-computes position_ids when not explicitly provided:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]  # KV cache seq dim
            position_ids = torch.arange(past_length, past_length + seq_len)
        else:
            position_ids = torch.arange(0, seq_len)

    We verify this by hooking into the model and recording actual position IDs.
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Position ID audit")
    print("=" * 60)

    input_ids = format_prompt(sample, tokenizer).to(device)
    prompt_len = input_ids.shape[1]

    recorded_positions = []
    recorded_kv_lengths = []

    # ── Step 0: Prompt ──
    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    past_kv = outputs.past_key_values

    # KV cache shape: past_kv[layer][0] is keys, shape (batch, n_heads, seq_len, head_dim)
    kv_len = past_kv[0][0].shape[2]
    recorded_kv_lengths.append(kv_len)
    recorded_positions.append(list(range(prompt_len)))
    print(f"  Step 0 (prompt): {prompt_len} tokens, KV cache length={kv_len}")
    print(f"    Position IDs: [0, ..., {prompt_len - 1}]")

    h = outputs.hidden_states[-1][0, -1, :]
    continuous_thought = h.unsqueeze(0).unsqueeze(0)

    # ── Steps 1..K: Recurrence ──
    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=continuous_thought,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        kv_len = past_kv[0][0].shape[2]
        recorded_kv_lengths.append(kv_len)

        # GPT-2 auto-computes: position_id = past_length for a single new token
        # past_length = kv_len_before_this_step = kv_len - 1
        inferred_pos = kv_len - 1
        recorded_positions.append([inferred_pos])

        print(f"  Step {t} (thought): KV cache length={kv_len}, "
              f"inferred position_id={inferred_pos}")

        h = outputs.hidden_states[-1][0, 0, :]
        continuous_thought = h.unsqueeze(0).unsqueeze(0)

    # ── After recurrence: <|end-latent|> ──
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(
        input_ids=end_input,
        past_key_values=past_kv,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
    kv_len = past_kv[0][0].shape[2]
    inferred_pos = kv_len - 1
    print(f"  <end-latent>: KV cache length={kv_len}, inferred position_id={inferred_pos}")

    # ── Summary ──
    total_positions = prompt_len + n_thoughts + 1  # prompt + K thoughts + end_latent
    print(f"\n  Total sequence positions used: {total_positions}")
    print(f"  Final KV cache length: {kv_len}")
    print(f"  Position IDs are sequential: {kv_len == total_positions}")

    if kv_len != total_positions:
        print(f"  WARNING: KV cache length ({kv_len}) != expected ({total_positions})")
        print(f"  This suggests position ID mismatch!")

    print(f"\n  Note: The Coconut training script (coconut_forward in run.py) also")
    print(f"  does NOT pass position_ids explicitly during recurrence steps.")
    print(f"  It relies on GPT-2's auto-inference from past_key_values length,")
    print(f"  which is exactly what our manual loop does.")

    return recorded_positions, recorded_kv_lengths


# ════════════════════════════════════════════════════════════════════
# CHECK 3: Hidden state comparison — does quantize(h)=h give the
#          exact same final hidden state as no quantization?
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def check_hidden_state_identity(base_model, tokenizer, sample, n_thoughts, device):
    """
    Run recurrence twice — once normal, once with explicit quantize(h)=h —
    and verify the hidden states are bitwise identical at every step.

    This catches subtle bugs like the codebook lookup introducing float
    precision differences.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Hidden state identity (h == quantize_identity(h))")
    print("=" * 60)

    input_ids = format_prompt(sample, tokenizer).to(device)

    # ── Run 1: Normal ──
    outputs = base_model(
        input_ids=input_ids, output_hidden_states=True, use_cache=True,
    )
    h_normal = [outputs.hidden_states[-1][0, -1, :].clone()]
    past_kv = outputs.past_key_values
    ct = h_normal[0].unsqueeze(0).unsqueeze(0)

    for t in range(n_thoughts):
        outputs = base_model(
            inputs_embeds=ct, past_key_values=past_kv,
            output_hidden_states=True, use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :].clone()
        h_normal.append(h)
        ct = h.unsqueeze(0).unsqueeze(0)
        past_kv = outputs.past_key_values

    # ── Run 2: Identity quantization (lookup through a codebook of size 1) ──
    outputs = base_model(
        input_ids=input_ids, output_hidden_states=True, use_cache=True,
    )
    h_quant = [outputs.hidden_states[-1][0, -1, :].clone()]
    past_kv = outputs.past_key_values
    ct = h_quant[0].unsqueeze(0).unsqueeze(0)

    for t in range(n_thoughts):
        # "Quantize" by doing a no-op: index into a 1-element codebook
        codebook = ct.squeeze(0).squeeze(0).unsqueeze(0)  # (1, D)
        # dist to single entry is 0, argmin is 0, result is ct itself
        dists = torch.cdist(ct.squeeze(0), codebook).squeeze(0)
        code_idx = dists.argmin().item()
        h_q = codebook[code_idx]

        outputs = base_model(
            inputs_embeds=h_q.unsqueeze(0).unsqueeze(0),
            past_key_values=past_kv,
            output_hidden_states=True, use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :].clone()
        h_quant.append(h)
        ct = h.unsqueeze(0).unsqueeze(0)
        past_kv = outputs.past_key_values

    # ── Compare ──
    all_match = True
    for t in range(len(h_normal)):
        diff = (h_normal[t] - h_quant[t]).abs().max().item()
        match = "EXACT" if diff == 0.0 else f"DIFF={diff:.2e}"
        if diff > 0.0:
            all_match = False
        print(f"  Step {t}: {match}")

    if all_match:
        print(f"\n  All hidden states are bitwise identical.")
    else:
        print(f"\n  WARNING: Hidden states differ! The codebook lookup path")
        print(f"  introduces numerical differences.")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for thought vector experiments.")
    parser.add_argument(
        "--task", type=str, choices=["prosqa", "gsm"],
        default="prosqa",
    )
    parser.add_argument("--batch_size", type=int, default=32,
                    help="Inference batch size. Matches test.py's batch_size (default 32).")
    parser.add_argument(
        "--models", type=str, nargs="+",
        choices=["coconut", "coconut_u", "pause", "codi"],
        default=None,
        help="Which models to check. Defaults: prosqa=[coconut, coconut_u, pause], gsm=[coconut, coconut_u, pause, codi]",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set defaults based on task
    if args.models is None:
        args.models = ["coconut", "coconut_u", "pause", "codi"]

    # Set data path
    data_path = args.data_path or str(PROSQA_TEST if args.task == "prosqa" else GSM_TEST)
    output_base = THOUGHT_ABLATION / args.task
    output_base.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path, args.max_instances)
    print(f"[INFO] Task: {args.task}, data: {data_path}, instances: {len(data)}")

    for model_name in args.models:
        print(f"\n{'#'*60}")
        print(f"# Task: {args.task}, Model: {model_name}")
        print(f"{'#'*60}")

        is_codi = (model_name == "codi")

        if is_codi:
            # CODI: special handling
            codi_dict = setup_codi_model(args.task, args.device)

            # Check 1: Normal inference
            accuracy, mismatches, correct_kn = check_codi_inference(
                codi_dict, data, args.n_thoughts, args.device,
                task=args.task,
                batch_size=args.batch_size,
                title=f"CHECK 1: CODI normal inference (K={args.n_thoughts})",
            )

            accuracy_k0, _, correct_k0 = check_codi_inference(
                codi_dict, data, 0, args.device,
                task=args.task,
                batch_size=args.batch_size,
                title="CHECK 4: CODI baseline (K=0, no recurrence)",
            )

            # Skip position ID and hidden state checks for CODI
            print("  Skipping position ID and hidden state checks (CODI uses LoRA + projection)")

        else:
            # Coconut/Pause models
            coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
                setup_model_and_tokenizer(args.task, model_name, args.device)

            # Check 1: Normal inference accuracy (K=n_thoughts)
            accuracy, mismatches, correct_kn = check_identity_codebook(
                coconut_model, base_model, tokenizer, start_id, end_id, latent_id,
                data, args.n_thoughts, args.device, task=args.task,
                title=f"CHECK 1: Identity codebook (normal inference accuracy, K={args.n_thoughts})",
            )

            # Checks 2 & 3: Only for coconut models (not pause, not CODI)
            if not coconut_model.feedback_mode == "pause_curriculum":
                check_position_ids(base_model, tokenizer, data[0], args.n_thoughts, args.device)
                check_hidden_state_identity(base_model, tokenizer, data[0], args.n_thoughts, args.device)
            else:
                print("  Skipping position ID and hidden state checks (not applicable to pause)")

            # Check 4: K=0 baseline — no recurrence at all
            accuracy_k0, _, correct_k0 = check_identity_codebook(
                coconut_model, base_model, tokenizer, start_id, end_id, latent_id,
                data, 0, args.device, task=args.task,
                title="CHECK 4: Baseline (K=0, no recurrence)",
            )

            # Free model memory
            del coconut_model, base_model

        # ── Bootstrap CIs ──
        ctx = {"task": args.task, "model": model_name, "n_thoughts": args.n_thoughts}
        cis_path = str(output_base / f"{model_name}/removeThoughts_cis.jsonl")

        # Pattern A: accuracy CI for K=n_thoughts
        ci_kn = report_mean_with_ci(
            correct_kn, metric=f"accuracy_K{args.n_thoughts}",
            context=ctx, cis_jsonl=cis_path, log=True,
        )

        # Pattern A: accuracy CI for K=0
        ctx_k0 = {**ctx, "n_thoughts": 0}
        ci_k0 = report_mean_with_ci(
            correct_k0, metric="accuracy_K0",
            context=ctx_k0, cis_jsonl=cis_path, log=True,
        )

        # Pattern C: paired difference (K=n vs K=0) on identical instances
        # diff_i = correct_kn_i - correct_k0_i; positive means recurrence helps
        diff_res = paired_bootstrap_diff(
            correct_kn, correct_k0, metric="acc_diff_Kn_minus_K0",
        )
        save_record(cis_path, diff_res, context=ctx)
        print(f"  [CI] paired diff (K{args.n_thoughts}−K0): {diff_res.to_short_str()}")

        # McNemar exact test on same paired vectors
        mc = mcnemar_test(correct_kn, correct_k0, metric="mcnemar_Kn_vs_K0")
        save_record(cis_path, mc, context=ctx)
        print(f"  [CI] McNemar p={mc['p_value']:.4g}  "
              f"(b={mc['b_a_only']}, c={mc['c_b_only']}, "
              f"n_disc={mc['n_discordant']})")

        # ── Summary ──
        print("\n" + "=" * 60)
        print(f"SUMMARY ({args.task}/{model_name})")
        print("=" * 60)
        print(f"  Normal inference (K={args.n_thoughts}): {ci_kn.to_short_str()}")
        print(f"  K=0 (no recurrence):                    {ci_k0.to_short_str()}")
        if abs(accuracy_k0 - accuracy) < 0.02:
            print(f"  WARNING: K=0 matches K={args.n_thoughts} — recurrence may be unnecessary")
        else:
            print(f"  K=0 drops by {accuracy - accuracy_k0:.1%} — recurrence provides signal")
        print(f"  Paired diff:  {diff_res.to_short_str()}")
        print(f"  McNemar p={mc['p_value']:.4g}")

        # ── Save baselines JSON ──
        baselines = {
            "task": args.task,
            "model": model_name,
            "n_thoughts": args.n_thoughts,
            "n_instances": len(data),
            "unquantized_accuracy": accuracy,
            "k0_accuracy": accuracy_k0,
            "n_correct_unquantized": int(accuracy * len(data)),
            "n_correct_k0": int(accuracy_k0 * len(data)),
            "bootstrap": {
                f"accuracy_K{args.n_thoughts}": ci_kn.to_short_str(),
                "accuracy_K0": ci_k0.to_short_str(),
                "paired_diff": diff_res.to_short_str(),
                "mcnemar_p": mc["p_value"],
            },
            "mismatched_instances_unquantized": [
                {"idx": m["idx"], "predicted": m["predicted"], "correct": m["correct"]}
                for m in mismatches
            ],
        }
        baselines_path = output_base / f"{model_name}/removeThoughts.json"
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"  Baselines saved to {baselines_path}")
        print(f"  CIs saved to {cis_path}")

        # Free GPU memory before loading next model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()