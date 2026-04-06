"""
CONTROL EXPERIMENT: Superposition across model conditions
==========================================================

Research question:
    Is the superposition phenomenon an artifact of Coconut's recursion
    training specifically, or does any model that understands ProsQA's
    graph structure exhibit it when forced through continuous-thought
    recurrence?

Conditions:
    --mode base      →  pretrained GPT-2, no finetuning
    --mode cot       →  GPT-2 finetuned on ProsQA with standard CoT
    --mode pause     →  GPT-2 finetuned with pause tokens (no thoughts)
    --mode coconut   →  Coconut-trained checkpoint (u=0.0)
    --mode coconut_u →  Coconut-trained checkpoint (u=0.3)

Probing methodology:
    For each k in {0, 1, ..., K}:
        1. Build input: [Question] <bot> [k latent tokens] <eot>
        2. Coconut.forward() runs the hidden-state recurrence.
        3. Probe: feed " Every" prefix, compute p(concept) for each
           candidate = prod_i p(token_i | context, "Every", token_{<i}).
        4. Record RAW joint probabilities (no normalization across
           candidates).

    Candidates at each k are the DEPTH FRONTIER: nodes exactly at
    distance max(k, 1) from root.

Usage:
    python superposition_control.py --mode base
    python superposition_control.py --mode coconut
    python superposition_control.py --mode pause
"""

import math
import json
import torch
import argparse
from collections import defaultdict
from contThought.coconut import Coconut
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from contThought.dataset import get_dataset, get_question_latent_dataset
from utils.config import (
    BASE_GPT2,
    PROSQA_MODELS,
    GSM_MODELS,
    PROSQA_TEST,
    CONTROL_EXPT,
    Config
)

# ============================================================================
# GRAPH UTILITIES
# ============================================================================

def build_children_map(edges):
    """Parent → list of children, from ProsQA edge list."""
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    return children


def bfs_reachable(edges, start):
    """Set of all node indices reachable from `start` via directed edges."""
    visited = set()
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for u, v in edges:
            if u == node and v not in visited:
                queue.append(v)
    return visited


def get_candidates_at_depth_k(instance, k):
    """
    Returns candidates for probing after k continuous thoughts.

    Paper methodology (Section 4.3, Figure 5):
        Candidates are the DEPTH FRONTIER — nodes at exactly
        depth max(k, 1) from the root.

        k=0: depth-1 nodes (children of root; model hasn't taken
             a latent step yet, but we probe what it "sees")
        k=1: depth-1 nodes (Figure 5, left)
        k=2: depth-2 nodes (Figure 5, right — grandchildren)
        k=n: depth-n nodes

    Each candidate is (concept_name, is_on_correct_path_to_target).
    """
    root_idx = instance["root"]
    target_idx = instance["target"]
    symbols = instance["idx_to_symbol"]
    edges = instance["edges"]

    children_map = build_children_map(edges)

    target_depth = max(k, 1)

    # BFS with depth tracking
    depth_of = {root_idx: 0}
    queue = [root_idx]

    while queue:
        node = queue.pop(0)
        d = depth_of[node]
        if d >= target_depth:
            continue
        for child in children_map[node]:
            if child not in depth_of:
                depth_of[child] = d + 1
                queue.append(child)

    # Collect only nodes at exactly target_depth
    frontier_indices = [
        idx for idx, d in depth_of.items()
        if d == target_depth
    ]

    # Build candidate list with correctness labels
    candidates = []
    for c_idx in frontier_indices:
        reachable_from_c = bfs_reachable(edges, c_idx)
        is_correct = target_idx in reachable_from_c
        candidates.append((symbols[c_idx], is_correct))

    root_name = symbols[root_idx]
    target_name = symbols[target_idx]

    return root_name, candidates, target_name


# ============================================================================
# PATH HELPERS
# ============================================================================

def get_checkpoint_path(task, mode):
    model_path = PROSQA_MODELS if task == "prosqa" else GSM_MODELS
    if mode == "base":
        return None
    if mode == "cot":
        return str(model_path / "cot/best_checkpoint.pt")
    if mode == "pause":
        return str(model_path / "pause/checkpoint_best")
    if mode == "coconut":
        return str(model_path / "coconut/checkpoint_best")
    if mode == "coconut_u":
        return str(model_path / "coconut_u/checkpoint_best")
    raise ValueError(f"Unsupported mode: {mode}")


def get_output_path(task, mode):
    output_dir = CONTROL_EXPT / task
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "base": "results_base.json",
        "cot": "results_cot.json",
        "pause": "results_pause.json",
        "coconut": "results_coconut.json",
        "coconut_u": "results_coconut_u.json",
    }
    return str(output_dir / filename_map[mode])

# ============================================================================
# MODEL + TOKENIZER SETUP
# ============================================================================

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

    return coconut_model, tokenizer, latent_id, start_id, end_id, checkpoint_path


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(prob_dict, candidates):
    """
    Compute metrics matching the paper's analysis (Section 4.3, Figures 5-7).

    prob_dict:   {concept_name: raw_joint_probability}
    candidates:  [(concept_name, is_correct)]

    Returns metrics aligned with the paper's framing:
        - value_correct:   sum of p(concept) for correct candidates
        - value_incorrect: sum of p(concept) for incorrect candidates
        - top1_prob:       max p(concept) across all candidates
        - top1_is_correct: whether the highest-probability candidate is correct
        - candidate_mass:  total probability mass on candidates (< 1.0)
        - num_candidates:  number of candidates at this depth
    """
    correct_names = {c for c, ok in candidates if ok}

    vals = sorted(prob_dict.values(), reverse=True)
    n = len(vals)

    # Value function: sum of raw probs for correct vs incorrect
    value_correct = sum(p for c, p in prob_dict.items() if c in correct_names)
    value_incorrect = sum(p for c, p in prob_dict.items() if c not in correct_names)

    # Top-1 analysis
    top1_name = max(prob_dict, key=prob_dict.get) if prob_dict else None
    top1_prob = prob_dict[top1_name] if top1_name else 0.0
    top1_is_correct = top1_name in correct_names if top1_name else False

    # Candidate mass (how much total probability the model assigns
    # to graph-relevant candidates vs. the full vocabulary)
    candidate_mass = sum(prob_dict.values())

    # Entropy over the raw distribution (for parallelism analysis, Figure 6)
    # We normalize locally just for entropy computation
    if candidate_mass > 1e-30:
        normed = {c: p / candidate_mass for c, p in prob_dict.items()}
        entropy = -sum(p * math.log2(p) for p in normed.values() if p > 1e-30)
    else:
        entropy = 0.0
    max_entropy = math.log2(n) if n > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Cumulative values for parallelism analysis (Figure 6)
    top2_cumul = sum(vals[:2])
    top3_cumul = sum(vals[:3])

    return {
        "value_correct": value_correct,
        "value_incorrect": value_incorrect,
        "top1_prob": top1_prob,
        "top1_is_correct": top1_is_correct,
        "candidate_mass": candidate_mass,
        "num_candidates": n,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": norm_entropy,
        "top1_cumul": vals[0] if vals else 0.0,
        "top2_cumul": top2_cumul,
        "top3_cumul": top3_cumul,
    }


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset_for_k(base_dataset, k, start_id, latent_id, end_id):
    """
    Build inference dataset with exactly k continuous thoughts.
    Uses the authors' get_question_latent_dataset():
        [question_tokens] <start-latent> <latent>*k <end-latent>
    """
    override_configs = Config({
        "pad_latent_to_max": False,
        "max_latent_stage": k,
        "c_thought": 1,
    })

    return get_question_latent_dataset(
        scheduled_stage=k,
        base_dataset_valid=base_dataset,
        configs=override_configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coconut_model, tokenizer, latent_id, start_id, end_id, checkpoint_path = \
        setup_model_and_tokenizer(args.task, args.mode, device)

    print(f"Loading data: {PROSQA_TEST}")
    with open(PROSQA_TEST) as f:
        raw_data = json.load(f)

    base_dataset = get_dataset(PROSQA_TEST, tokenizer, max_size=args.num_samples)

    num_samples = min(args.num_samples, len(raw_data))
    max_k = args.max_thoughts
    output_path = get_output_path(args.task, args.mode)

    print(f"Samples: {num_samples}, max thoughts: {max_k}")
    print(f"Output path: {output_path}\n")

    all_results = defaultdict(dict)

    for k in range(0, max_k + 1):
        print(f"=== k = {k} continuous thoughts ===")

        dataset_k = prepare_dataset_for_k(
            base_dataset, k, start_id, latent_id, end_id
        )

        for si in range(min(num_samples, len(dataset_k))):
            instance = raw_data[si]
            root_name, candidates, target_name = get_candidates_at_depth_k(instance, k)

            if len(candidates) < 2:
                continue

            input_ids = torch.tensor([dataset_k[si]["input_ids"]], device=device)

            from utils.pause_aware_utils import probe_concept_probs_pauseaware
            concept_probs, concept_log_probs = probe_concept_probs_pauseaware(
                coconut_model, tokenizer, input_ids, candidates, device, k,
                start_id=start_id, latent_id=latent_id,
                end_id_tok=end_id, sample=instance,
            )

            metrics = compute_metrics(concept_probs, candidates)

            all_results[si][k] = {
                "root": root_name,
                "target": target_name,
                "candidates": [c for c, _ in candidates],
                "correct_candidates": [c for c, ok in candidates if ok],
                "concept_probs": concept_probs,
                "concept_log_probs": concept_log_probs,
                "metrics": metrics,
            }

            if si < 3:
                sorted_probs = sorted(concept_probs.items(), key=lambda x: -x[1])
                top3_str = ", ".join(f"{n}:{p:.4f}" for n, p in sorted_probs[:3])
                print(
                    f"  [{si}] H_norm={metrics['normalized_entropy']:.3f} "
                    f"top1={metrics['top1_prob']:.3f} "
                    f"correct={metrics['top1_is_correct']} | {top3_str}"
                )

    # --- Aggregate ---
    MODE_LABELS = {
        "base": "Base GPT-2 (no training)",
        "cot": "CoT-finetuned GPT-2 (no recurrence training)",
        "pause": "GPT-2 finetuned with pause tokens",
        "coconut": "Coconut-trained GPT-2 (u=0.0)",
        "coconut_u": "Coconut-trained GPT-2 (u=0.3)",
    }
    print("\n" + "=" * 70)
    print(f"AGGREGATE — {MODE_LABELS[args.mode]}")
    print("=" * 70)

    summary = {}
    for k in range(0, max_k + 1):
        metrics_lists = defaultdict(list)
        for si in all_results:
            if k in all_results[si]:
                m = all_results[si][k]["metrics"]
                for key in m:
                    if isinstance(m[key], (int, float)):
                        metrics_lists[key].append(m[key])
                    elif isinstance(m[key], bool):
                        metrics_lists[key].append(float(m[key]))

        if not metrics_lists.get("entropy"):
            continue

        def mean_std(xs):
            mu = sum(xs) / len(xs)
            std = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
            return mu, std

        n = len(metrics_lists["entropy"])
        me, se = mean_std(metrics_lists["entropy"])
        mne, _ = mean_std(metrics_lists["normalized_entropy"])
        vc, _ = mean_std(metrics_lists["value_correct"])
        vi, _ = mean_std(metrics_lists["value_incorrect"])
        mt1, _ = mean_std(metrics_lists["top1_prob"])
        t1_correct = sum(metrics_lists["top1_is_correct"]) / n
        cm, _ = mean_std(metrics_lists["candidate_mass"])
        mt2, _ = mean_std(metrics_lists["top2_cumul"])
        mt3, _ = mean_std(metrics_lists["top3_cumul"])

        summary[k] = {
            "n": n,
            "mean_normalized_entropy": mne,
            "mean_entropy": me,
            "std_entropy": se,
            "mean_value_correct": vc,
            "mean_value_incorrect": vi,
            "mean_top1_prob": mt1,
            "top1_correct_frac": t1_correct,
            "mean_candidate_mass": cm,
            "mean_top2_cumul": mt2,
            "mean_top3_cumul": mt3,
        }
        print(
            f"  k={k}: H/Hmax={mne:.3f}  "
            f"P(correct)={vc:.3f}  P(incorrect)={vi:.3f}  "
            f"top1_correct={t1_correct:.1%}  "
            f"mass={cm:.3f}  (n={n})"
        )

    # --- Save ---
    output = {
        "experiment": "superposition_control",
        "mode": args.mode,
        "checkpoint": checkpoint_path,
        "data_path": str(PROSQA_TEST),
        "num_samples": num_samples,
        "max_thoughts": max_k,
        "fixes_applied": [
            "Fix 1: Raw joint probabilities (no normalization across candidates)",
            "Fix 2: Depth frontier only (candidates at exactly depth max(k,1))",
            "Fix 3: Tokenization verified correct (no change needed)",
        ],
        "summary": {str(k): v for k, v in summary.items()},
        "per_instance": {
            str(si): {
                str(k): {
                    "concept_probs": all_results[si][k]["concept_probs"],
                    "concept_log_probs": all_results[si][k]["concept_log_probs"],
                    "metrics": all_results[si][k]["metrics"],
                    "root": all_results[si][k]["root"],
                    "target": all_results[si][k]["target"],
                    "candidates": all_results[si][k]["candidates"],
                    "correct_candidates": all_results[si][k]["correct_candidates"],
                }
                for k in all_results[si]
            }
            for si in all_results
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument("--mode", choices=["base", "cot", "pause", "coconut", "coconut_u"], default="base")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_thoughts", type=int, default=6)
    args = parser.parse_args()

    run_experiment(args)