"""
Extract continuous thought vectors from a trained Coconut model.

For each ProsQA instance, runs K steps of continuous-thought recurrence
and saves the final hidden state h_t at each step t = 0, ..., K.

Output: a single .pt file containing:
    - "thoughts": Tensor of shape (N, K+1, D)
        N = number of instances, K = recurrence steps, D = hidden dim (768 for GPT-2)
    - "instance_indices": list of ints, mapping row i to its ProsQA index
    - "n_thoughts": int, the K used

Usage:
    python extract_thoughts.py --model coconut --n_thoughts 6
    python extract_thoughts.py --model coconut_u --n_thoughts 6 --max_instances 100
"""

import json
import torch
import argparse
from pathlib import Path
from contThought.coconut import Coconut
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import (
    BASE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PAUSE_GPT2, PROSQA_TEST, THOUGHTS
)


# ── Checkpoint resolution ───────────────────────────────────────────
# Coconut checkpoints are saved as:
#   torch.save(parallel_model.state_dict(), save_dir / f"checkpoint_{epoch}")
# So they're single files (no extension) containing FSDP/DDP state dicts
# with keys like "base_causallm.transformer.h.0.attn.c_attn.weight".

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


def setup_model_and_tokenizer(mode: str, device: str):
    """
    Load GPT-2, add Coconut special tokens, wrap in Coconut, load checkpoint.

    Loading order for Coconut modes:
        1. Load pretrained GPT-2 from BASE_GPT2
        2. Add special tokens, resize embeddings, init new embeddings with "<<"
        3. Wrap in Coconut
        4. Load Coconut checkpoint (keys: base_causallm.*)
    """
    checkpoint_path = get_checkpoint_path(mode)

    # Step 1: Load base GPT-2
    model = AutoModelForCausalLM.from_pretrained(str(BASE_GPT2))
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_GPT2))
    tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Add special tokens and initialize embeddings
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

    # Step 3: Wrap in Coconut
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Step 4: Load checkpoint
    print(f"Loading Coconut checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)

    sample_key = next(iter(state_dict.keys()))
    if not sample_key.startswith("base_causallm"):
        print(f"  WARNING: first key is '{sample_key}', expected 'base_causallm.*'")

    missing, unexpected = coconut_model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(unexpected)
    print(f"  Loaded {n_loaded}/{len(state_dict)} keys")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    coconut_model = coconut_model.to(device)
    coconut_model.eval()

    # Extract the base GPT-2 from the Coconut wrapper for direct forward passes.
    # We run recurrence manually (matching the training script's pattern),
    # so we need the raw AutoModelForCausalLM, not the Coconut wrapper.
    base_model = coconut_model.base_causallm

    return coconut_model, base_model, tokenizer, start_id, end_id, latent_id


def load_prosqa(path: str, max_instances: int = None) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    if max_instances is not None:
        data = data[:max_instances]
    print(f"[INFO] Loaded {len(data)} ProsQA instances from {path}")
    return data


def format_prompt(sample: dict, tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Tokenize a ProsQA question and append <|start-latent|>.
    Returns input_ids of shape (1, seq_len).
    """
    question_text = sample["question"]
    prompt = question_text + " <|start-latent|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    return input_ids


@torch.no_grad()
def extract_thoughts_single(
    base_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    n_thoughts: int,
    device: str,
) -> torch.Tensor:
    """
    Run Coconut recurrence for one instance and collect thought vectors.

    Recurrence (matching training script coconut_forward):
        1. Forward pass on prompt (up to <|start-latent|>)
        2. h_0 = hidden_states[-1][0, -1, :] — last-layer, last-position
        3. For t = 1..K:
             Feed h_{t-1} as inputs_embeds (1, 1, D), bypassing embedding layer
             h_t = hidden_states[-1][0, 0, :]
        4. Collect [h_0, h_1, ..., h_K]

    Returns: Tensor of shape (n_thoughts + 1, D)
    """
    hidden_dim = base_model.config.n_embd  # 768 for GPT-2
    thoughts = torch.zeros(n_thoughts + 1, hidden_dim)

    # ── Step 0: Process the prompt ──────────────────────────────────
    input_ids = input_ids.to(device)
    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )

    # h_0 = final-layer hidden state at the last token position
    # outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, D)
    # Index [-1] = output of last transformer layer
    h = outputs.hidden_states[-1][0, -1, :]  # (D,)
    thoughts[0] = h.cpu()
    past_key_values = outputs.past_key_values

    # ── Steps 1..K: Continuous thought recurrence ───────────────────
    # Feed h_{t-1} as inputs_embeds — shape (1, 1, D)
    continuous_thought = h.unsqueeze(0).unsqueeze(0)

    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=continuous_thought,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :]  # (D,)
        thoughts[t] = h.cpu()

        continuous_thought = h.unsqueeze(0).unsqueeze(0)
        past_key_values = outputs.past_key_values

    return thoughts


def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous thought vectors from Coconut model."
    )
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument("--prosqa_path", type=str, default=None)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    prosqa_path = args.prosqa_path or str(PROSQA_TEST)
    output_path = args.output_path or str(
        THOUGHTS / f"thoughts_{args.model}.pt"
    )

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] ProsQA data: {prosqa_path}")
    print(f"[INFO] Recurrence steps K={args.n_thoughts}")
    print(f"[INFO] Device: {args.device}")

    coconut_model, base_model, tokenizer, start_id, end_id, latent_id = setup_model_and_tokenizer(
        args.model, args.device
    )
    data = load_prosqa(prosqa_path, args.max_instances)

    N = len(data)
    D = base_model.config.n_embd
    K = args.n_thoughts

    all_thoughts = torch.zeros(N, K + 1, D)

    for idx, sample in enumerate(data):
        if idx % 50 == 0:
            print(f"[INFO] Processing instance {idx}/{N}")

        input_ids = format_prompt(sample, tokenizer)
        from utils.pause_aware_utils import extract_thoughts_single_instance
        thoughts = extract_thoughts_single_instance(
            coconut_model, base_model, tokenizer, sample, K, args.device,
            start_id, end_id, latent_id,
        )
        all_thoughts[idx] = thoughts

    save_dict = {
        "thoughts": all_thoughts,
        "instance_indices": list(range(N)),
        "n_thoughts": K,
        "model": args.model,
        "prosqa_path": prosqa_path,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"[INFO] Saved {N} x {K+1} x {D} thought vectors to {output_path}")


if __name__ == "__main__":
    main()