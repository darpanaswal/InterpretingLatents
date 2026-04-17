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
from src.config import PROSQA_TEST, CONTROL_EXPT, Config
from src.utils import setup_model_and_tokenizer, is_pause_model
from contThought.dataset import get_dataset, get_question_latent_dataset

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


# ═══════════════════════════════════════════════════════════════════
# Superposition probing: pause-aware
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def probe_concept_probs_pauseaware(
    coconut_model, tokenizer, input_ids, candidates, device, k,
    start_id=None, latent_id=None, end_id_tok=None, sample=None,
):
    """
    Probe concept probabilities after k thoughts.

    For coconut: uses Coconut.forward() (recurrence), then probes.
    For pause: builds input with k pause embeddings, runs single
        forward pass, then probes from the last thought position.

    The probing methodology is identical: feed " Every" prefix after
    the thought positions, then compute p(concept) autoregressively.
    """
    import math

    pause = is_pause_model(coconut_model)
    base_model = coconut_model.base_causallm
    embedding = coconut_model.embedding

    if pause:
        # Build input with pause embeddings for k thoughts
        question_text = sample["question"]
        question_tokens = tokenizer.encode(question_text + "\n", add_special_tokens=True)

        input_ids_list = (
            question_tokens
            + [start_id]
            + [latent_id] * k
            + [end_id_tok]
        )
        input_ids_t = torch.tensor([input_ids_list], device=device)

        inputs_embeds = embedding(input_ids_t)
        pause_emb = coconut_model.pause_embedding
        start_of_latent = len(question_tokens) + 1

        for i in range(k):
            pos = start_of_latent + i
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, pos, :] = pause_emb

        # Single forward pass
        full_outputs = base_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=True,
        )
        kv_cache = full_outputs.past_key_values
        current_pos = inputs_embeds.shape[1]

    else:
        # Coconut: use forward() for recurrence, then get KV cache
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)

        outputs = coconut_model.forward(
            input_ids, attention_mask, labels, position_ids
        )
        inputs_embeds_out = outputs.inputs_embeds

        full_outputs = base_model(inputs_embeds=inputs_embeds_out, use_cache=True)
        kv_cache = full_outputs.past_key_values
        current_pos = inputs_embeds_out.shape[1]

    # From here, probing is identical for both models
    prefix_tokens = tokenizer.encode(" Every", add_special_tokens=False)
    prefix_out = None
    for pt in prefix_tokens:
        pt_embed = embedding(torch.tensor([[pt]], device=device))
        pos_id = torch.tensor([[current_pos]], device=device)
        prefix_out = base_model(
            inputs_embeds=pt_embed,
            past_key_values=kv_cache,
            position_ids=pos_id,
            use_cache=True,
        )
        kv_cache = prefix_out.past_key_values
        current_pos += 1

    if prefix_out is None:
        raise RuntimeError('Prefix tokenization for " Every" returned no tokens.')

    logits_at_concept = prefix_out.logits[0, 0, :]

    concept_log_probs = {}
    for concept_name, _ in candidates:
        c_tokens = tokenizer.encode(" " + concept_name, add_special_tokens=False)
        if len(c_tokens) == 0:
            concept_log_probs[concept_name] = float("-inf")
            continue

        p_first = F.softmax(logits_at_concept, dim=-1)[c_tokens[0]].item()
        log_p = math.log(p_first + 1e-30)

        if len(c_tokens) > 1:
            concept_cache = tuple((k_.clone(), v_.clone()) for k_, v_ in kv_cache)
            concept_pos = current_pos

            for i in range(len(c_tokens) - 1):
                t_embed = embedding(torch.tensor([[c_tokens[i]]], device=device))
                cp_id = torch.tensor([[concept_pos]], device=device)
                c_out = base_model(
                    inputs_embeds=t_embed,
                    past_key_values=concept_cache,
                    position_ids=cp_id,
                    use_cache=True,
                )
                concept_cache = c_out.past_key_values
                concept_pos += 1

                p_next = F.softmax(c_out.logits[0, 0, :], dim=-1)
                log_p += math.log(p_next[c_tokens[i + 1]].item() + 1e-30)

        concept_log_probs[concept_name] = log_p

    concept_probs = {c: math.exp(lp) for c, lp in concept_log_probs.items()}
    return concept_probs, concept_log_probs

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coconut_model, _, tokenizer, latent_id, start_id, end_id, checkpoint_path = \
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