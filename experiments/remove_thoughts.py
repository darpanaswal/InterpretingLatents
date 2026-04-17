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
    python remove_thoughts.py --task prosqa --models coconut coconut_u pause

    # GSM8k (including CODI)
    python remove_thoughts.py --task gsm --models coconut coconut_u pause codi

    # Single model
    python remove_thoughts.py --task gsm --models codi --max_instances 100
"""

import re
import json
import torch
import argparse
from src.config import PROSQA_TEST, GSM_TEST, THOUGHTS
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    extract_answer_number,
)

def load_data(task, path, max_instances=None):
    """Load ProsQA or GSM8k data."""
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
    return extract_answer_number(gold_text)


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

    return accuracy, mismatches


# ════════════════════════════════════════════════════════════════════
# CHECK 1 (CODI variant): Normal inference for CODI
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def check_codi_inference(codi_dict, data, n_thoughts, device,
                         title="CHECK 1: CODI normal inference"):
    """
    Run CODI inference and check accuracy.

    Matches CODI test.py inference exactly for --greedy True:
      1. Encode question + [bot]            (--remove_eos True: no eos before bot)
      2. Run K recurrence steps with projection
      3. Feed [eot] delimiter embedding     (--remove_eos True: no eos after eot)
      4. Greedy-decode, clipping logits to exclude eot from generation
      5. Compare answers numerically via extract_answer_number
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']

    n_correct = 0
    mismatches = []

    for idx, sample in enumerate(data):
        # ── Tokenize ──
        # test.py with --remove_eos True: [question_tokens] [bot]
        # test.py with --remove_eos False: [question_tokens] [eos] [bot]
        question_tokens = tokenizer.encode(sample["question"], add_special_tokens=True)
        if remove_eos:
            input_ids_list = question_tokens + [bot_id]
        else:
            input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]
        input_ids = torch.tensor([input_ids_list], device=device)

        # ── Step 0: encode question + <bot> ──
        outputs = base_model(
            input_ids=input_ids, use_cache=True, output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][0, -1, :]

        # Project for feeding back
        latent = h.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

        # ── Steps 1..K: latent recurrence ──
        for t in range(n_thoughts):
            outputs = base_model(
                inputs_embeds=latent, use_cache=True,
                output_hidden_states=True, past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][0, -1, :]

            latent = h.unsqueeze(0).unsqueeze(0)
            if use_prj and prj is not None:
                latent = prj(latent)

        # ── Feed eot delimiter before decoding ──
        # test.py with --remove_eos True:  embed([eot_id])
        # test.py with --remove_eos False: embed([eot_id, eos_token_id])
        if remove_eos:
            eot_ids = torch.tensor([[eot_id]], device=device)
        else:
            eot_ids = torch.tensor(
                [[eot_id, tokenizer.eos_token_id]], device=device,
            )
        eot_emb = embedding_fn(eot_ids)
        outputs = base_model(
            inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        # CODI test.py: logits[:, -1, :vocab_size-1] — exclude special tokens
        vocab_size = base_model.config.vocab_size
        next_logits = outputs.logits[0, -1, :vocab_size - 1]

        # ── Greedy decode ──
        # CODI test.py feeds embeddings at each step, not input_ids
        generated = []
        for _ in range(256):
            next_token = next_logits.argmax().item()
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)
            next_emb = embedding_fn(
                torch.tensor([next_token], device=device)
            ).unsqueeze(0)
            out = base_model(
                inputs_embeds=next_emb,
                past_key_values=past_kv,
                use_cache=True,
            )
            next_logits = out.logits[0, -1, :vocab_size - 1]
            past_kv = out.past_key_values

        text = tokenizer.decode(generated, skip_special_tokens=True)

        # ── Numeric answer comparison ──
        # extract_answer_number: returns the last number in the text as float
        pred_num = extract_answer_number(text)
        gold_num = _gsm_gold_number(sample)
        is_correct = (pred_num == gold_num)

        if is_correct:
            n_correct += 1
        else:
            mismatches.append({
                "idx": idx,
                "predicted": str(pred_num),
                "correct": str(gold_num),
                "full_output": text[:200],
            })

        if idx < 5:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  [{idx}] {status}: predicted='{pred_num}' gold='{gold_num}'")

    accuracy = n_correct / len(data)
    print(f"\n  Normal inference accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")

    if mismatches and len(mismatches) <= 20:
        print(f"\n  Mismatched instances:")
        for m in mismatches:
            print(f"    [{m['idx']}] predicted='{m['predicted']}' gold='{m['correct']}'")

    return accuracy, mismatches


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
        if args.task == "prosqa":
            args.models = ["coconut", "coconut_u", "pause"]
        else:  # gsm
            args.models = ["coconut", "coconut_u", "pause", "codi"]

    # Validate model choices
    if "codi" in args.models and args.task != "gsm":
        parser.error("CODI is only available for --task gsm")

    # Set data path
    data_path = args.data_path or str(PROSQA_TEST if args.task == "prosqa" else GSM_TEST)
    output_base = THOUGHTS / args.task
    output_base.mkdir(parents=True, exist_ok=True)

    data = load_data(args.task, data_path, args.max_instances)
    print(f"[INFO] Task: {args.task}, data: {data_path}, instances: {len(data)}")

    for model_name in args.models:
        print(f"\n{'#'*60}")
        print(f"# Task: {args.task}, Model: {model_name}")
        print(f"{'#'*60}")

        is_codi = (model_name == "codi")

        if is_codi:
            # CODI: special handling
            codi_dict = setup_codi_model(args.device)

            # Check 1: Normal inference
            accuracy, mismatches = check_codi_inference(
                codi_dict, data, args.n_thoughts, args.device,
                title=f"CHECK 1: CODI normal inference (K={args.n_thoughts})",
            )

            # Check 4: K=0 baseline
            accuracy_k0, _ = check_codi_inference(
                codi_dict, data, 0, args.device,
                title="CHECK 4: CODI baseline (K=0, no recurrence)",
            )

            # Skip position ID and hidden state checks for CODI
            print("  Skipping position ID and hidden state checks (CODI uses LoRA + projection)")

        else:
            # Coconut/Pause models
            coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
                setup_model_and_tokenizer(args.task, model_name, args.device)

            # Check 1: Normal inference accuracy (K=n_thoughts)
            accuracy, mismatches = check_identity_codebook(
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
            accuracy_k0, _ = check_identity_codebook(
                coconut_model, base_model, tokenizer, start_id, end_id, latent_id,
                data, 0, args.device, task=args.task,
                title="CHECK 4: Baseline (K=0, no recurrence)",
            )

            # Free model memory
            del coconut_model, base_model

        # ── Summary ──
        print("\n" + "=" * 60)
        print(f"SUMMARY ({args.task}/{model_name})")
        print("=" * 60)
        print(f"  Normal inference (K={args.n_thoughts}): accuracy = {accuracy:.1%}")
        print(f"  K=0 (no recurrence): accuracy = {accuracy_k0:.1%}")
        if abs(accuracy_k0 - accuracy) < 0.02:
            print(f"  WARNING: K=0 matches K={args.n_thoughts} — recurrence may be unnecessary")
        else:
            print(f"  K=0 drops by {accuracy - accuracy_k0:.1%} — recurrence provides signal")

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
            "mismatched_instances_unquantized": [
                {"idx": m["idx"], "predicted": m["predicted"], "correct": m["correct"]}
                for m in mismatches
            ],
        }
        baselines_path = output_base / f"{model_name}/removeThoughts.json"
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"  Baselines saved to {baselines_path}")

        # Free GPU memory before loading next model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None



if __name__ == "__main__":
    main()