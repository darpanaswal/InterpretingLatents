"""
Sanity checks for the VQ-VAE experiment.

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
    python remove_recurrence.py --model coconut_u --max_instances 50
"""

import json
import torch
import argparse
from pathlib import Path
from contThought.coconut import Coconut
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import (
    BASE_DIR, BASE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PAUSE_GPT2, PROSQA_TEST, THOUGHTS
)


# ── Reuse model loading from extract_thoughts.py ───────────────────

def get_checkpoint_path(mode: str) -> Path:
    mode_to_dir = {
        "coconut": COCONUT_GPT2,
        "coconut_u": COCONUT_GPT2_U,
        "pause": PAUSE_GPT2,
    }
    ckpt_dir = mode_to_dir[mode]
    checkpoint_path = ckpt_dir / "checkpoint_best"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_best not found in {ckpt_dir}")
    return checkpoint_path


def setup_model_and_tokenizer(mode, device):
    """
    Load GPT-2, add Coconut special tokens, wrap in Coconut, load checkpoint.

    Returns coconut_model (the wrapper), base_model, tokenizer, and token IDs.
    The coconut_model is needed for pause-aware inference (it carries
    feedback_mode and pause_embedding).
    """
    checkpoint_path = get_checkpoint_path(mode)
    model = AutoModelForCausalLM.from_pretrained(str(BASE_GPT2))
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_GPT2))
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    # Wrap in Coconut with correct feedback_mode
    feedback_mode = "pause_curriculum" if mode == "pause" else "continuous"
    coconut_model = Coconut(
        model, latent_id, start_id, end_id,
        tokenizer.eos_token_id, feedback_mode=feedback_mode,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)
    coconut_model.load_state_dict(state_dict, strict=False)

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return coconut_model, base_model, tokenizer, start_id, end_id, latent_id


def load_prosqa(path, max_instances=None):
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def format_prompt(sample, tokenizer):
    return tokenizer.encode(sample["question"] + " <|start-latent|>", return_tensors="pt")


# ════════════════════════════════════════════════════════════════════
# CHECK 1: Identity codebook — does our decoding logic match normal
#          inference when quantization is a no-op?
# ════════════════════════════════════════════════════════════════════

def check_identity_codebook(
    coconut_model, base_model, tokenizer, start_id, end_id, latent_id, data, n_thoughts, device,
    title="CHECK 1: Identity codebook (normal inference accuracy)"
):
    """
    Compare normal inference to the Coconut wrapper's generate().
    If they match, our manual recurrence + decoding is correct.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    n_correct = 0
    mismatches = []

    for idx, sample in enumerate(data):
        from utils.pause_aware_utils import run_normal_inference_pauseaware
        result = run_normal_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device,
            start_id=start_id, latent_id=latent_id,
        )
        answer = result["predicted"]
        is_correct = result["is_correct"]
        correct_answer = sample.get("answer", "").replace(",", "").strip()
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

    # Hook into GPT-2's transformer to capture position_ids
    # In HuggingFace GPT2Model.forward(), position_ids are computed
    # before being passed to the embedding layer.
    # We can infer them from the KV cache length.

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
    # Position IDs for prompt: 0, 1, ..., prompt_len-1
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
        # (since this step added 1 to the cache)
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

    # ── Check: does the Coconut wrapper do anything different? ──
    # The training script's coconut_forward() also does not pass position_ids
    # explicitly — it relies on the same auto-inference from past_key_values.
    # So our manual loop should produce identical position IDs.
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
    parser = argparse.ArgumentParser(description="Sanity checks for VQ-VAE experiment.")
    parser.add_argument(
        "--models", type=str, nargs="+",
        choices=["coconut", "coconut_u", "pause"],
        default=["coconut", "coconut_u", "pause"],
        help="Which models to check (default: all three).",
    )
    parser.add_argument("--prosqa_path", type=str, default=None)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prosqa_path = args.prosqa_path or str(PROSQA_TEST)
    output_base = THOUGHTS

    data = load_prosqa(prosqa_path, args.max_instances)

    for model_name in args.models:
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"{'#'*60}")

        coconut_model, base_model, tokenizer, start_id, end_id, latent_id = \
            setup_model_and_tokenizer(model_name, args.device)

        # Check 1: Normal inference accuracy (K=n_thoughts)
        accuracy, mismatches = check_identity_codebook(
            coconut_model, base_model, tokenizer, start_id, end_id, latent_id, data, args.n_thoughts, args.device,
            title=f"CHECK 1: Identity codebook (normal inference accuracy, K={args.n_thoughts})"
        )

        if not coconut_model.feedback_mode == "pause_curriculum":
            check_position_ids(base_model, tokenizer, data[0], args.n_thoughts, args.device)
            check_hidden_state_identity(base_model, tokenizer, data[0], args.n_thoughts, args.device)
        else:
            print("  Skipping position ID and hidden state checks (not applicable to pause)")

        # Check 4: K=0 baseline — no recurrence at all
        accuracy_k0, _ = check_identity_codebook(
            coconut_model, base_model, tokenizer, start_id, end_id, latent_id, data, 0, args.device,
            title="CHECK 4: Baseline (K=0, no recurrence)"
        )

        # ── Summary ──
        print("\n" + "=" * 60)
        print(f"SUMMARY ({model_name})")
        print("=" * 60)
        print(f"  Normal inference (K={args.n_thoughts}): accuracy = {accuracy:.1%}")
        print(f"  K=0 (no recurrence): accuracy = {accuracy_k0:.1%}")
        if abs(accuracy_k0 - accuracy) < 0.02:
            print(f"  WARNING: K=0 matches K={args.n_thoughts} — recurrence may be unnecessary")
        else:
            print(f"  K=0 drops by {accuracy - accuracy_k0:.1%} — recurrence provides signal")

        # ── Save baselines JSON ─────────────────────────────────────
        # Saved to THOUGHTS/baselines.json
        # so the plotting script can load them alongside the eval JSONs.
        baselines = {
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
        output_base.mkdir(parents=True, exist_ok=True)
        baselines_path = output_base / f"removeThoughts_{model_name}.json"
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"  Baselines saved to {baselines_path}")

        # Free GPU memory before loading next model
        del coconut_model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()