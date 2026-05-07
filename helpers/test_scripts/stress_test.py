"""
stress_test.py

A standalone test suite to verify the consistency of batched CODI thought extraction
against unbatched execution and end-to-end evaluation accuracy.

Usage:
    python stress_test.py --task gsm --n_thoughts 6 --batch_size 32
    python stress_test.py --task prosqa --n_thoughts 6 --batch_size 32
"""

import math
import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
import warnings
from typing import List, Dict

import test
from test import evaluation, extract_answer_number, compute_accuracy, ModelArguments, DataArguments, TrainingArguments
from experiments.extract_thoughts import (
    extract_thoughts_codi,
    extract_thoughts_codi_batch,
    _codi_normalize_question,
    load_data
)
from src.utils import setup_codi_model
from contThought.codiModel import build_position_ids_from_mask # Imported the POS-FIX helper

warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")

# ═══════════════════════════════════════════════════════════════════
# Metrics & Tolerances
# ═══════════════════════════════════════════════════════════════════

COS_TOLERANCE = 0.999
MEAN_DIFF_TOLERANCE = 2e-2
ACC_TOLERANCE = 0.2

def calc_metrics(base: torch.Tensor, test_tensor: torch.Tensor) -> tuple:
    b_flat = base.flatten(1).float()
    t_flat = test_tensor.flatten(1).float()
    
    cos_sim = F.cosine_similarity(b_flat, t_flat, dim=1).mean().item()
    max_diff = torch.max(torch.abs(base.float() - test_tensor.float())).item()
    mean_diff = torch.mean(torch.abs(base.float() - test_tensor.float())).item()
    return cos_sim, max_diff, mean_diff

# ═══════════════════════════════════════════════════════════════════
# Test Definitions
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_test_a(codi_dict: dict, samples: List[Dict], device: str, n_thoughts: int, gen_kwargs: dict, expected_baseline: float, task_name: str):
    print("\n--- Running Test A: Accuracy Equivalence (Patched test.py match) ---")
    
    batch_size = 32
    base_model = codi_dict['model']
    tokenizer = codi_dict['tokenizer']
    remove_eos = codi_dict['remove_eos']
    prj = codi_dict['prj']
    use_prj = codi_dict['use_prj']
    
    ans_pred_list = []
    gold_answers = []
    
    # Task-specific ground truth parsing
    for s in samples:
        if "prosqa" in task_name or "prontoqa" in task_name:
            ans = str(s.get("answer", "")).replace(",", "").strip()
            gold_answers.append(ans)
        else:
            ans = str(s.get("answer", "0"))
            if "####" in ans:
                ans = ans.split("####")[-1]
            ans = ans.replace(",", "").strip()
            try:
                ans = float(ans)
            except ValueError:
                ans = float('inf')
            gold_answers.append(ans)
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        curr_bs = len(batch_samples)
        
        questions = [_codi_normalize_question(s["question"]) for s in batch_samples]
        enc = tokenizer(questions, return_tensors="pt", padding="longest")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        if remove_eos:
            suffix = torch.tensor([[codi_dict['bot_id']]] * curr_bs, dtype=torch.long, device=device)
        else:
            suffix = torch.tensor([[tokenizer.eos_token_id, codi_dict['bot_id']]] * curr_bs, dtype=torch.long, device=device)
            
        input_ids = torch.cat([input_ids, suffix], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(suffix)], dim=1)
        
        # PATCH 1: Explicit position_ids from mask
        encoder_position_ids = build_position_ids_from_mask(attention_mask)
        
        outputs = base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=encoder_position_ids,
            use_cache=True, 
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        
        if use_prj:
            latent_embd = prj(latent_embd)
            
        # Initialize trackers for latent and decoding loops
        running_mask = attention_mask.clone()
        current_position = encoder_position_ids[:, -1]
            
        # PATCH 2: Feed attention_mask and position_ids during latent recurrence
        for t in range(n_thoughts):
            current_position = current_position + 1
            running_mask = torch.cat(
                [running_mask, torch.ones((curr_bs, 1), dtype=running_mask.dtype, device=device)], dim=1
            )
            latent_position_ids = current_position.unsqueeze(1)

            outputs = base_model(
                inputs_embeds=latent_embd, 
                attention_mask=running_mask,
                position_ids=latent_position_ids,
                use_cache=True, 
                output_hidden_states=True, 
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if use_prj: latent_embd = prj(latent_embd)
            
        if remove_eos:
            eot_emb = base_model.get_input_embeddings()(torch.tensor([codi_dict['eot_id']], dtype=torch.long, device=device)).unsqueeze(0)
        else:
            eot_emb = base_model.get_input_embeddings()(torch.tensor([codi_dict['eot_id'], tokenizer.eos_token_id], dtype=torch.long, device=device)).unsqueeze(0)
        
        eot_emb = eot_emb.expand(curr_bs, -1, -1)
        output = eot_emb
        
        # PATCH 3: Setup EOT position tracking and feed mask/position during decoding
        eot_len = eot_emb.size(1)
        current_step_position_ids = current_position.unsqueeze(1) + torch.arange(1, eot_len + 1, device=device).unsqueeze(0)
        running_mask = torch.cat(
            [running_mask, torch.ones((curr_bs, eot_len), dtype=running_mask.dtype, device=device)], dim=1
        )
        current_position = current_position + eot_len

        finished = torch.zeros(curr_bs, dtype=torch.bool, device=device)
        pred_tokens = [[] for _ in range(curr_bs)]
        
        for _ in range(gen_kwargs["max_new_tokens"]):
            out = base_model(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=running_mask,
                position_ids=current_step_position_ids,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :base_model.config.vocab_size-1]
            
            next_token_ids = torch.argmax(logits, dim=-1)
            
            for b in range(curr_bs):
                if not finished[b]:
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True
            
            if finished.all():
                break
                
            output = base_model.get_input_embeddings()(next_token_ids).unsqueeze(1).to(device)
            
            # Advance mask and position for next token
            current_position = current_position + 1
            current_step_position_ids = current_position.unsqueeze(1)
            running_mask = torch.cat(
                [running_mask, torch.ones((curr_bs, 1), dtype=running_mask.dtype, device=device)], dim=1
            )
            
        for mini_step, pred_token in enumerate(pred_tokens):
            decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
            ans_pred_list.append(extract_answer_number(decoded_pred))
        
    acc = compute_accuracy(gold_answers, ans_pred_list) * 100
    baseline_acc = expected_baseline 
    diff = abs(baseline_acc - acc)
    
    passed = diff <= ACC_TOLERANCE
    return {"baseline_acc": baseline_acc, "test_acc": acc, "diff": diff, "passed": passed}

@torch.no_grad()
def run_test_b(codi_dict: dict, samples: List[Dict], device: str, n_thoughts: int) -> list:
    print("\n--- Running Test B: Batched vs Unbatched Equivalence ---")
    test_samples = samples[:32]
    
    unbatched_thoughts = []
    for s in test_samples:
        t = extract_thoughts_codi(codi_dict, s["question"], n_thoughts, device)
        unbatched_thoughts.append(t)
    unbatched_thoughts = torch.stack(unbatched_thoughts)
    
    results = []
    for bs in [1, 4, 16, 32]:
        batched_thoughts = extract_thoughts_codi_batch(codi_dict, test_samples, n_thoughts, device, batch_size=bs, verbose=False)
        cos_sim, max_diff, mean_diff = calc_metrics(unbatched_thoughts, batched_thoughts)
        
        passed = cos_sim >= COS_TOLERANCE and mean_diff <= MEAN_DIFF_TOLERANCE
            
        results.append({
            "bs": bs, "cos_sim": cos_sim, "max_diff": max_diff, 
            "mean_diff": mean_diff, "passed": passed
        })
        
    return results

@torch.no_grad()
def run_test_c(codi_dict: dict, samples: List[Dict], device: str, n_thoughts: int) -> dict:
    print("\n--- Running Test C: Heterogeneous-length Stress ---")
    tokenizer = codi_dict['tokenizer']
    
    lengths = [len(tokenizer.encode(_codi_normalize_question(s["question"]))) for s in samples]
    min_idx, max_idx = np.argmin(lengths), np.argmax(lengths)
    hetero_samples = [samples[min_idx], samples[max_idx]]
    
    unbatched_thoughts = torch.stack([
        extract_thoughts_codi(codi_dict, s["question"], n_thoughts, device) 
        for s in hetero_samples
    ])
    
    batched_thoughts = extract_thoughts_codi_batch(codi_dict, hetero_samples, n_thoughts, device, batch_size=2, verbose=False)
    cos_sim, max_diff, mean_diff = calc_metrics(unbatched_thoughts, batched_thoughts)
    passed = cos_sim >= COS_TOLERANCE and mean_diff <= MEAN_DIFF_TOLERANCE
    
    return {"cos_sim": cos_sim, "max_diff": max_diff, "passed": passed}

@torch.no_grad()
def run_test_d(codi_dict: dict, samples: List[Dict], device: str, n_thoughts: int) -> dict:
    print("\n--- Running Test D: Whitespace Normalization Determinism ---")
    base_q = samples[0]["question"]
    
    variants = [
        {"question": base_q},
        {"question": "   " + base_q},
        {"question": base_q + "   "},
        {"question": base_q.replace(" ", "  ")}
    ]
    
    thoughts = extract_thoughts_codi_batch(codi_dict, variants, n_thoughts, device, batch_size=4, verbose=False)
    
    passed = True
    max_diffs = []
    
    for i in range(1, 4):
        max_diff = torch.max(torch.abs(thoughts[0] - thoughts[i])).item()
        max_diffs.append(max_diff)
        if max_diff > 0:
            passed = False
            
    return {"max_diffs": max_diffs, "passed": passed}

# ═══════════════════════════════════════════════════════════════════
# Main Runner & Reporter
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stress Test for CODI extractors")
    parser.add_argument("--task", type=str, default="gsm")
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_instances", type=int, default=None) 
    parser.add_argument("--expected_baseline", type=float, default=None, help="Known test.py accuracy to verify against")
    args = parser.parse_args()

    # Dynamic baseline setting
    if args.expected_baseline is None:
        target_baseline = 79.0 if args.task == "prosqa" else 43.21
    else:
        target_baseline = args.expected_baseline

    class MockDataArgs:
        def __init__(self, data_name):
            self.data_name = data_name
            
    test.data_args = MockDataArgs(args.task)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Stress Test on {device}...")
    
    codi_dict = setup_codi_model(args.task, device)
    samples = load_data(args.task, args.max_instances)
    
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": False, 
    }

    res_a = run_test_a(codi_dict, samples, device, args.n_thoughts, gen_kwargs, target_baseline, args.task)
    res_b = run_test_b(codi_dict, samples, device, args.n_thoughts)
    res_c = run_test_c(codi_dict, samples, device, args.n_thoughts)
    res_d = run_test_d(codi_dict, samples, device, args.n_thoughts)
    
    print("\n" + "="*60)
    print(" STRESS TEST VERDICT TABLE")
    print("="*60)
    
    status_a = "PASS" if res_a["passed"] else "FAIL"
    print(f"[Test A] End-to-End Accuracy Match: {status_a}")
    print(f"         Baseline Acc: {res_a['baseline_acc']:.2f}% | Test Acc: {res_a['test_acc']:.2f}% | Δ: {res_a['diff']:.4f}")
    
    print("\n[Test B] Batched vs Unbatched Equivalence:")
    for rb in res_b:
        status_b = "PASS" if rb["passed"] else "FAIL"
        print(f"         BS={rb['bs']:<2} -> {status_b} | Cos: {rb['cos_sim']:.5f} | Max|Δ|: {rb['max_diff']:.2e} | Mean|Δ|: {rb['mean_diff']:.2e}")
        
    status_c = "PASS" if res_c["passed"] else "FAIL"
    print(f"\n[Test C] Heterogeneous-Length Padding: {status_c}")
    print(f"         Cos: {res_c['cos_sim']:.5f} | Max|Δ|: {res_c['max_diff']:.2e}")
    
    status_d = "PASS" if res_d["passed"] else "FAIL"
    print(f"\n[Test D] Whitespace Determinism: {status_d}")
    print(f"         Max|Δ| vs Baseline: V1={res_d['max_diffs'][0]:.2e}, V2={res_d['max_diffs'][1]:.2e}, V3={res_d['max_diffs'][2]:.2e}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()