"""
Extract continuous thought vectors from a trained Coconut or CODI model.

For each ProsQA/GSM instance, runs K steps of continuous-thought recurrence
and saves the final hidden state h_t at each step t = 0, ..., K.

Output: a single .pt file containing:
    - "thoughts": Tensor of shape (N, K+1, D)
        N = number of instances, K = recurrence steps, D = hidden dim (768 for GPT-2)
    - "instance_indices": list of ints, mapping row i to its index
    - "n_thoughts": int, the K used

Usage:
    python extract_thoughts.py --model coconut --n_thoughts 6
    python extract_thoughts.py --model codi --task gsm --n_thoughts 6
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from src.utils import is_pause_model, setup_model_and_tokenizer, setup_codi_model
from src.config import (
    PROSQA_TEST, GSM_TEST, THOUGHTS
)

# ═══════════════════════════════════════════════════════════════════
# Data Formatting
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
    hidden_dim = base_model.config.n_embd

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

    # Tokenize question and append <start-latent> token
    question_tokens = tokenizer.encode(sample["question"], add_special_tokens=True)
    input_ids = torch.tensor([question_tokens + [start_id]], device=device)

    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]
    thoughts[0] = h.cpu()
    past_kv = outputs.past_key_values

    # Recurrence: feed h back as input embedding for K steps
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

    # Tokenize: question + [eos, bot]
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous thought vectors from Coconut/CODI model."
    )
    parser.add_argument(
        "--task", type=str, choices=["prosqa", "gsm"],
        default="prosqa",
    )
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.model == "codi" and args.task != "gsm":
        parser.error("CODI is only available for --task gsm")

    prosqa_path = str(PROSQA_TEST)
    gsm_path = str(GSM_TEST)
    output_path = args.output_path or str(
        THOUGHTS / f"{args.task}/thoughts_{args.model}.pt"
    )

    print(f"[INFO] Model: {args.model}")
    if args.task == "prosqa":
        print(f"[INFO] ProsQA data: {prosqa_path}")
    else:
        print(f"[INFO] GSM8k data: {gsm_path}")
    print(f"[INFO] Recurrence steps K={args.n_thoughts}")
    print(f"[INFO] Device: {args.device}")

    # Initialize models based on model type
    is_codi = (args.model == "codi")
    if is_codi:
        codi_dict = setup_codi_model(args.device)
        D = codi_dict['hidden_size']
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = setup_model_and_tokenizer(
            args.task, args.model, args.device
        )
        D = base_model.config.n_embd

    data = load_data(args.task, args.max_instances)
    N = len(data)
    K = args.n_thoughts

    all_thoughts = torch.zeros(N, K + 1, D)

    for idx, sample in enumerate(data):
        if idx % 50 == 0:
            print(f"[INFO] Processing instance {idx}/{N}")

        if is_codi:
            thoughts = extract_thoughts_codi(
                codi_dict, sample["question"], K, args.device
            )
        else:
            thoughts = extract_thoughts_single_instance(
                coconut_model, base_model, tokenizer, sample, K, args.device,
                start_id, latent_id, end_id,
            )
            
        all_thoughts[idx] = thoughts

    save_dict = {
        "thoughts": all_thoughts,
        "instance_indices": list(range(N)),
        "n_thoughts": K,
        "model": args.model,
        "data_path": prosqa_path if args.task == "prosqa" else gsm_path,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"[INFO] Saved {N} x {K+1} x {D} thought vectors to {output_path}")

if __name__ == "__main__":
    main()