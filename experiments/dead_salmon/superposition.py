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
    --mode codi      →  CODI (latent distillation)

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
    python -m experiments.dead_salmon.superposition --mode base
    python -m experiments.dead_salmon.superposition --mode cot
    python -m experiments.dead_salmon.superposition --mode pause
    python -m experiments.dead_salmon.superposition --mode coconut
    python -m experiments.dead_salmon.superposition --mode coconut_u
    python -m experiments.dead_salmon.superposition --mode codi
"""

import math
import json
import torch
import argparse
import torch.nn.functional as F
from collections import defaultdict
from src.config import PROSQA_TEST, CONTROL_EXPT, Config
from contThought.dataset import get_dataset, get_question_latent_dataset
from src.utils import setup_model_and_tokenizer, setup_codi_model, is_pause_model
from src.bootstrap_stats import report_mean_with_ci

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

def get_output_path(mode):
    output_dir = CONTROL_EXPT
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir / f"results_{mode}.json")

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(prob_dict, candidates):
    """
    Compute metrics matching the paper's analysis.

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
    codi_dict=None,
):
    """
    Probe concept probabilities after k thoughts.

    For coconut: uses Coconut.forward() (recurrence), then probes.
    For pause: builds input with k pause embeddings, runs single
        forward pass, then probes from the last thought position.
    For CODI: runs encoder + K recurrence steps + eot with proper
        attention_mask/position_ids, then probes.

    The probing methodology is identical: feed " Every" prefix after
    the thought positions, then compute p(concept) autoregressively.
    """
    import math

    is_codi = (codi_dict is not None)

    if is_codi:
        # ── CODI path ──
        base_model = codi_dict['model']
        prj = codi_dict['prj']
        embedding = codi_dict['embedding_fn']
        bot_id = codi_dict['bot_id']
        eot_id = codi_dict['eot_id']
        use_prj = codi_dict['use_prj']
        remove_eos = codi_dict['remove_eos']

        question_text = sample["question"]
        question_tokens = tokenizer.encode(question_text, add_special_tokens=True)

        if remove_eos:
            input_ids_list = question_tokens + [bot_id]
        else:
            input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]

        input_ids_t = torch.tensor([input_ids_list], device=device)
        attention_mask = torch.ones_like(input_ids_t)     # (1, L), no padding
        # pos_ids[0, j] = j,  real_len = L
        L = input_ids_t.size(1)
        position_ids = torch.arange(L, device=device).unsqueeze(0)

        # Step 0: encode question + [bot]
        outputs = base_model(
            input_ids=input_ids_t, use_cache=True, output_hidden_states=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        kv_cache = outputs.past_key_values
        h = outputs.hidden_states[-1][0, -1, :]           # (D,)
        latent = h.unsqueeze(0).unsqueeze(0)               # (1, 1, D)
        if use_prj and prj is not None:
            latent = prj(latent)

        # Steps 1..K: latent recurrence
        # running_mask grows by 1 each step; position at step t = L + t - 1
        running_mask = attention_mask                       # (1, L)
        for t in range(1, k + 1):
            running_mask = torch.cat(
                [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                          device=device)],
                dim=1,
            )
            pos_t = torch.tensor([[L + t - 1]], device=device)

            outputs = base_model(
                inputs_embeds=latent, use_cache=True, output_hidden_states=True,
                past_key_values=kv_cache,
                attention_mask=running_mask,
                position_ids=pos_t,
            )
            kv_cache = outputs.past_key_values
            h = outputs.hidden_states[-1][0, -1, :]
            latent = h.unsqueeze(0).unsqueeze(0)
            if use_prj and prj is not None:
                latent = prj(latent)

        # Feed [eot] delimiter
        eot_input = torch.tensor([[eot_id]], device=device)
        eot_pos = torch.tensor([[L + k]], device=device)
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        outputs = base_model(
            input_ids=eot_input, use_cache=True,
            past_key_values=kv_cache,
            attention_mask=running_mask,
            position_ids=eot_pos,
        )
        kv_cache = outputs.past_key_values
        # current_pos = L + k + 1 (next position after eot)
        current_pos = L + k + 1

    else:
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

    is_codi = (args.mode == "codi")

    if is_codi:
        codi_dict = setup_codi_model(args.task, device)
        tokenizer = codi_dict['tokenizer']
        checkpoint_path = str(codi_dict.get('checkpoint_path', 'codi'))
        coconut_model = None
        start_id = latent_id = end_id = None
    else:
        codi_dict = None
        coconut_model, _, tokenizer, latent_id, start_id, end_id, checkpoint_path = \
            setup_model_and_tokenizer(args.task, args.mode, device)

    print(f"Loading data: {PROSQA_TEST}")
    with open(PROSQA_TEST) as f:
        raw_data = json.load(f)

    if not is_codi:
        base_dataset = get_dataset(PROSQA_TEST, tokenizer, max_size=args.num_samples)

    num_samples = min(args.num_samples, len(raw_data))
    max_k = args.max_thoughts
    output_path = get_output_path(args.mode)

    print(f"Samples: {num_samples}, max thoughts: {max_k}")
    print(f"Output path: {output_path}\n")

    all_results = defaultdict(dict)

    for k in range(0, max_k + 1):
        print(f"=== k = {k} continuous thoughts ===")

        if not is_codi:
            dataset_k = prepare_dataset_for_k(
                base_dataset, k, start_id, latent_id, end_id
            )
            n_iter = min(num_samples, len(dataset_k))
        else:
            n_iter = num_samples

        for si in range(n_iter):
            instance = raw_data[si]
            root_name, candidates, target_name = get_candidates_at_depth_k(instance, k)

            if len(candidates) < 2:
                continue

            if is_codi:
                input_ids = None
            else:
                input_ids = torch.tensor([dataset_k[si]["input_ids"]], device=device)

            concept_probs, concept_log_probs = probe_concept_probs_pauseaware(
                coconut_model, tokenizer, input_ids, candidates, device, k,
                start_id=start_id, latent_id=latent_id,
                end_id_tok=end_id, sample=instance,
                codi_dict=codi_dict,
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

    MODE_LABELS = {
        "base": "Base GPT-2 (no training)",
        "cot": "CoT-finetuned GPT-2 (no recurrence training)",
        "pause": "GPT-2 finetuned with pause tokens",
        "coconut": "Coconut-trained GPT-2 (u=0.0)",
        "coconut_u": "Coconut-trained GPT-2 (u=0.3)",
        "codi": "CODI (latent distillation)",
    }
    print("\n" + "=" * 70)
    print(f"AGGREGATE — {MODE_LABELS[args.mode]}")
    print("=" * 70)

    # ── Bootstrap CI output paths ──
    ci_dir = CONTROL_EXPT / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    cis_jsonl = str(ci_dir / f"superposition_{args.mode}.jsonl")
    # Clear stale records from previous runs
    import os as _os
    if _os.path.exists(cis_jsonl):
        _os.remove(cis_jsonl)

    # Metrics for which we emit CI records (all per-instance scalars from
    # compute_metrics).  Listed explicitly so the reader can verify coverage.
    CI_METRICS = [
        "normalized_entropy", "entropy", "value_correct", "value_incorrect",
        "top1_prob", "top1_is_correct", "candidate_mass",
        "top1_cumul", "top2_cumul", "top3_cumul",
    ]

    summary = {}
    for k in range(0, max_k + 1):
        # ── Collect per-instance vectors for this timestep ──
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

        n = len(metrics_lists["entropy"])
        base_ctx = {"task": args.task, "model": args.mode, "t": k}

        # ── CI record per metric, per timestep ──
        ci_results = {}
        for mname in CI_METRICS:
            if mname not in metrics_lists:
                continue
            vec = metrics_lists[mname]

            # save_per_instance_vector for the headline metric only
            # headline = top1_is_correct (binary accuracy of superposition probe)
            vec_path = None
            if mname == "top1_is_correct":
                vec_path = str(ci_dir / f"vec_{args.mode}_k{k}_{mname}.npz")

            res = report_mean_with_ci(
                values=vec,
                metric=mname,
                context=base_ctx,
                cis_jsonl=cis_jsonl,
                vector_npz=vec_path,
                log=False,            # we print our own summary line below
            )
            ci_results[mname] = res

        # ── Build summary dict (point estimates, compatible with JSON output) ──
        summary[k] = {
            "n": n,
            "mean_normalized_entropy": ci_results["normalized_entropy"].point,
            "mean_entropy":            ci_results["entropy"].point,
            "mean_value_correct":      ci_results["value_correct"].point,
            "mean_value_incorrect":    ci_results["value_incorrect"].point,
            "mean_top1_prob":          ci_results["top1_prob"].point,
            "top1_correct_frac":       ci_results["top1_is_correct"].point,
            "mean_candidate_mass":     ci_results["candidate_mass"].point,
            "mean_top2_cumul":         ci_results["top2_cumul"].point,
            "mean_top3_cumul":         ci_results["top3_cumul"].point,
        }

        # ── Console summary (now with CI on headline metric) ──
        h = ci_results["top1_is_correct"]
        print(
            f"  k={k}: H/Hmax={ci_results['normalized_entropy'].point:.3f}  "
            f"P(correct)={ci_results['value_correct'].point:.3f}  "
            f"P(incorrect)={ci_results['value_incorrect'].point:.3f}  "
            f"top1_correct={h.point:.1%} "
            f"[{h.ci_low:.1%}, {h.ci_high:.1%}]  "
            f"mass={ci_results['candidate_mass'].point:.3f}  (n={n})"
        )

    print(f"\nCI records: {cis_jsonl}")

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
    parser.add_argument("--mode", choices=["base", "cot", "pause", "coconut", "coconut_u", "codi"], default="base")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_thoughts", type=int, default=6)
    args = parser.parse_args()

    run_experiment(args)