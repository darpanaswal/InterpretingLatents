"""
Inspect Mechanics: Attention + Logit Lens for Continuous Thoughts vs Pause Tokens.

This script runs the specified model on a sample question and performs two checks:
1. ATTENTION: Analyzes where the <|end-latent|> token is looking before generating the answer.
2. LOGIT LENS: Projects the hidden states into the vocabulary to see what words 
               the model is "thinking" at each step or pause position.

Usage:
    python -m experiments.probe_thoughts.inspect_thought_mechanics --model pause --k 6
    python -m experiments.probe_thoughts.inspect_thought_mechanics --model coconut --k 6
    python -m experiments.probe_thoughts.inspect_thought_mechanics --model coconut_u --k 6
"""

import json
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

from contThought.coconut import Coconut
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import BASE_GPT2, PAUSE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PROSQA_TEST, BASE_DIR

def setup_model(model_name, device):
    """Load GPT-2, add special tokens, and load the appropriate checkpoint."""
    if model_name == "pause":
        checkpoint_path = PAUSE_GPT2 / "checkpoint_best"
        feedback_mode = "pause_curriculum"
    elif model_name == "coconut_u":
        checkpoint_path = COCONUT_GPT2_U / "checkpoint_best"
        feedback_mode = "continuous"
    else:
        checkpoint_path = COCONUT_GPT2 / "checkpoint_best"
        feedback_mode = "continuous"
        
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

    coconut_model = Coconut(
        model, latent_id, start_id, end_id, 
        tokenizer.eos_token_id, feedback_mode=feedback_mode
    )

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)
    coconut_model.load_state_dict(state_dict, strict=False)

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    
    return coconut_model, coconut_model.base_causallm, tokenizer, start_id, latent_id, end_id


def inspect_mechanics(coconut_model, base_model, tokenizer, sample, k, device, start_id, latent_id, end_id, model_name):
    question_text = sample["question"]
    question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)
    
    is_pause = (coconut_model.feedback_mode == "pause_curriculum")
    
    # Data structures to unify the evaluation
    hidden_states_to_probe = []
    end_token_attn = None
    seq_len = len(question_tokens) + 1 + k + 1  # Prompt + Start + K + End
    
    with torch.no_grad():
        if is_pause:
            # -------------------------------------------------------------
            # PAUSE MODEL: Single Pass Spatial Execution
            # -------------------------------------------------------------
            input_ids_list = question_tokens + [start_id] + [latent_id] * k + [end_id]
            input_ids = torch.tensor([input_ids_list], device=device)
            
            inputs_embeds = coconut_model.embedding(input_ids)
            start_of_latent = len(question_tokens) + 1
            for i in range(k):
                inputs_embeds[0, start_of_latent + i, :] = coconut_model.pause_embedding

            outputs = base_model(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False
            )
            
            last_hidden = outputs.hidden_states[-1]
            # Grab h_0
            hidden_states_to_probe.append(last_hidden[0, len(question_tokens), :])
            # Grab h_1 to h_k
            for i in range(k):
                hidden_states_to_probe.append(last_hidden[0, start_of_latent + i, :])
                
            end_token_attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0).cpu().numpy()

        else:
            # -------------------------------------------------------------
            # COCONUT MODEL: Temporal Recurrence Execution
            # -------------------------------------------------------------
            input_ids = tokenizer.encode(question_text + "\n<|start-latent|>", return_tensors="pt").to(device)
            
            # Step 0
            outputs = base_model(input_ids=input_ids, output_hidden_states=True, use_cache=True, output_attentions=True)
            h = outputs.hidden_states[-1][0, -1, :]
            past_kv = outputs.past_key_values
            hidden_states_to_probe.append(h)
            
            # Steps 1..K
            for _ in range(k):
                outputs = base_model(
                    inputs_embeds=h.unsqueeze(0).unsqueeze(0), 
                    past_key_values=past_kv, 
                    output_hidden_states=True, 
                    use_cache=True, 
                    output_attentions=True
                )
                h = outputs.hidden_states[-1][0, 0, :]
                past_kv = outputs.past_key_values
                hidden_states_to_probe.append(h)
                
            # <end-latent> Step to get Attention Matrix
            end_input = torch.tensor([[end_id]], device=device)
            outputs = base_model(
                input_ids=end_input, 
                past_key_values=past_kv, 
                output_hidden_states=True, 
                use_cache=True, 
                output_attentions=True
            )
            end_token_attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0).cpu().numpy()

    # ====================================================================
    # PART A: ATTENTION ANALYSIS
    # ====================================================================
    idx_prompt_end = len(question_tokens)
    idx_start_latent = idx_prompt_end
    idx_latent_start = idx_start_latent + 1
    idx_latent_end = idx_latent_start + k

    attn_prompt = end_token_attn[:idx_prompt_end].sum()
    attn_start = end_token_attn[idx_start_latent]
    attn_latent = end_token_attn[idx_latent_start:idx_latent_end].sum()
    attn_end = end_token_attn[-1]

    latent_label = "Pause tokens" if is_pause else "Continuous thoughts"

    print(f"\n{'='*70}")
    print(f"PART A: ATTENTION MASS FOR <|end-latent|> TOKEN (K={k})")
    print(f"{'='*70}")
    print(f"  Prompt tokens (0 to {idx_prompt_end-1}):      {attn_prompt:.2%}")
    print(f"  <|start-latent|> token ({idx_start_latent}):         {attn_start:.2%}")
    print(f"  {latent_label} ({idx_latent_start} to {idx_latent_end-1}):      {attn_latent:.2%}")
    print(f"  Self-attention ({seq_len-1}):                 {attn_end:.2%}")
    print(f"{'-'*70}")
    
    if attn_latent < 0.05:
        print(f"  CONCLUSION: The model is almost entirely ignoring the {latent_label.lower()}.")
    else:
        print(f"  CONCLUSION: The model is heavily utilizing the {latent_label.lower()}.")

    # Generate X-axis labels seamlessly for both models
    tokens_str = [tokenizer.decode([tok]).replace('Ġ', ' ') for tok in question_tokens]
    tokens_str.append("<start>")
    for i in range(k):
        tokens_str.append(f"Pause {i+1}" if is_pause else f"Thought {i+1}")
    tokens_str.append("<end>")
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(seq_len), end_token_attn, color='royalblue')
    plt.axvspan(idx_latent_start - 0.5, idx_latent_end - 0.5, color='red', alpha=0.1, label=latent_label)
    plt.title(f"[{model_name}] Where does the <|end-latent|> token look?", fontsize=14)
    plt.xlabel("Sequence Position")
    plt.ylabel("Average Attention Weight")
    
    x_ticks = list(range(0, idx_prompt_end, max(1, idx_prompt_end // 10))) + list(range(idx_prompt_end, seq_len))
    plt.xticks(x_ticks, [tokens_str[i] for i in x_ticks], rotation=45, ha='right')
    
    plt.legend()
    plt.tight_layout()
    
    out_dir = BASE_DIR / "outputs" / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_attention.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[INFO] Attention plot saved to {out_path}")

    # ====================================================================
    # PART B: LOGIT LENS (VOCABULARY PROJECTION)
    # ====================================================================
    lm_head = base_model.get_output_embeddings()

    print(f"\n{'='*70}")
    print(f"PART B: LOGIT LENS - WHAT IS THE MODEL THINKING AT EACH STEP?")
    print(f"{'='*70}")
    print(f"Target Answer for this instance: {sample['answer'].strip()}")
    print("-" * 70)

    labels = ["start-latent"] + [f"Step {i+1}" for i in range(k)]

    for step_idx, hidden_vec in enumerate(hidden_states_to_probe):
        logits = lm_head(hidden_vec)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print(f"Position: [ {labels[step_idx]} ]")
        for i in range(5):
            token_id = top_indices[i].item()
            token_str = tokenizer.decode([token_id])
            clean_str = repr(token_str) 
            prob = top_probs[i].item()
            print(f"   {i+1}. {prob:6.2%} -> {clean_str}")
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["coconut", "coconut_u", "pause"], default="coconut")
    parser.add_argument("--k", type=int, default=6, help="Number of latent tokens/thoughts")
    parser.add_argument("--instance_idx", type=int, default=0, help="Which ProsQA instance to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    coconut_model, base_model, tokenizer, start_id, latent_id, end_id = setup_model(args.model, device)
    
    with open(PROSQA_TEST, "r") as f:
        data = json.load(f)
        
    sample = data[args.instance_idx]
    
    inspect_mechanics(
        coconut_model, base_model, tokenizer, sample, args.k, 
        device, start_id, latent_id, end_id, args.model
    )

if __name__ == "__main__":
    main()