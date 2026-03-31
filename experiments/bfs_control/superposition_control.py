"""
CONTROL EXPERIMENT: Superposition across three model conditions
===============================================================

Research question:
    Is the superposition phenomenon an artifact of Coconut's recursion
    training specifically, or does any model that understands ProsQA's
    graph structure exhibit it when forced through continuous-thought
    recurrence?

Three conditions:
    --mode base     →  pretrained GPT-2, no finetuning
    --mode cot      →  GPT-2 finetuned on ProsQA with standard CoT
                       (understands the task, but never trained with recurrence)
    --mode coconut  →  Coconut-trained checkpoint

The probing methodology is identical across all three conditions
(Section 4.3 / Figure 5 of the Coconut paper):
    For each k in {0, 1, ..., K}:
        1. Build input: [Question] <bot> [k latent tokens] <eot>
        2. Coconut.forward() runs the hidden-state recurrence.
        3. Probe: feed " Every" prefix, compute p(concept) for each
           candidate = prod_i p(token_i | context, "Every", token_{<i}).
        4. Normalize across candidates to get a distribution.

Usage (run from the coconut/ repo directory):
    python superposition_control.py --mode base
    python superposition_control.py --mode cot
    python superposition_control.py --mode coconut
"""

import math
import json
import torch
import argparse
import torch.nn.functional as F
from collections import defaultdict
from contThought.coconut import Coconut
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from contThought.dataset import get_dataset, get_question_latent_dataset
from utils.config import (
    BASE_GPT2,
    COT_GPT2,
    PAUSE_GPT2,
    COCONUT_GPT2,
    COCONUT_GPT2_U,
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


def get_candidate_info(instance):
    """
    Returns:
        root_name:  str, e.g. "Alex"
        candidates: list of (concept_name, is_on_correct_path)
        target_name: str
    """
    root_idx = instance["root"]
    target_idx = instance["target"]
    symbols = instance["idx_to_symbol"]
    edges = instance["edges"]

    child_indices = build_children_map(edges)[root_idx]

    candidates = []
    for c_idx in child_indices:
        reachable = bfs_reachable(edges, c_idx)
        is_correct = target_idx in reachable
        candidates.append((symbols[c_idx], is_correct))

    return symbols[root_idx], candidates, symbols[target_idx]

def get_candidates_at_depth_k(instance, k):
    """
    Returns candidates for probing after k continuous thoughts.

    Paper methodology (Section 4.3, Figure 5):
        k=0 or k=1:  children of root (depth 1)
        k>=2:        all nodes reachable from root within k hops,
                     excluding the root itself

    Each candidate is (concept_name, is_on_correct_path_to_target).

    BFS from root, collecting nodes at depths 1..k.
    For k<2 we clamp to depth 1 (the model hasn't taken a latent
    step yet at k=0; at k=1 it has taken one step and the paper's
    Figure 5 left shows depth-1 candidates).
    """
    root_idx = instance["root"]
    target_idx = instance["target"]
    symbols = instance["idx_to_symbol"]
    edges = instance["edges"]

    children_map = build_children_map(edges)

    # BFS with depth tracking
    # depth_of[node] = shortest distance from root
    max_depth = max(k, 1)  # clamp: at k=0 we still probe depth 1

    depth_of = {root_idx: 0}
    queue = [root_idx]
    candidate_indices = []

    while queue:
        node = queue.pop(0)
        d = depth_of[node]
        if d >= max_depth:
            continue
        for child in children_map[node]:
            if child not in depth_of:
                depth_of[child] = d + 1
                queue.append(child)
                candidate_indices.append(child)
            # If already visited at a shorter depth, skip
            # (BFS guarantees first visit is shortest)

    # Deduplicate (BFS already handles this via depth_of check)
    # Build candidate list with correctness labels
    # A candidate is "correct" if the target is reachable from it
    candidates = []
    for c_idx in candidate_indices:
        reachable_from_c = bfs_reachable(edges, c_idx)
        is_correct = target_idx in reachable_from_c
        candidates.append((symbols[c_idx], is_correct))

    root_name = symbols[root_idx]
    target_name = symbols[target_idx]

    return root_name, candidates, target_name


# ============================================================================
# PATH HELPERS
# ============================================================================

def get_checkpoint_path(mode):
    """
    Fixed checkpoint selection:
        - base     → no checkpoint
        - cot      → COT_GPT2 / best_checkpoint.pt
        - coconut  → COCONUT_GPT2 / checkpoint_50
                     (run.py line 397: torch.save(states, ".../checkpoint_{epoch+1}")
                      — no .pt extension)
    """
    if mode == "base":
        return None
    if mode == "cot":
        return str(COT_GPT2 / "best_checkpoint.pt")
    if mode == "pause":
        return str(PAUSE_GPT2 / "checkpoint_best")
    if mode == "cot":
        return str(COT_GPT2 / "best_checkpoint.pt")
    if mode == "coconut":
        return str(COCONUT_GPT2 / "checkpoint_best")
    if mode == "coconut_u":
        return str(COCONUT_GPT2_U / "checkpoint_best")
    raise ValueError(f"Unsupported mode: {mode}")


def get_output_path(mode):
    """
    Fixed output location:
        CONTROL_EXPT / results_<mode>.json
    """
    output_dir = CONTROL_EXPT
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

def setup_model_and_tokenizer(mode, device):
    """
    Load GPT-2, add Coconut special tokens, wrap in Coconut class.

    Loading order matters and differs by mode:

    --mode base:
        1. Load pretrained GPT-2.
        2. Add special tokens, resize embeddings, init with "<<".
        3. Wrap in Coconut.

    --mode cot:
        1. Load pretrained GPT-2.
        2. Load CoT checkpoint (plain GPT-2 state dict, original vocab size).
        3. Add special tokens, resize embeddings, init with "<<".
        4. Wrap in Coconut.
        (Matches run.py lines 120-126: loading a base model checkpoint
         when keys don't start with "base_causallm".)

    --mode coconut:
        1. Load pretrained GPT-2.
        2. Add special tokens, resize embeddings, init with "<<".
        3. Wrap in Coconut.
        4. Load Coconut checkpoint (keys: base_causallm.*, possibly with
           FSDP/DDP prefixes stripped by clean_state_dict_keys).
        (Matches run.py lines 133-138 and 166: loading Coconut state dict
         into the Coconut wrapper after construction.)
    """
    checkpoint_path = get_checkpoint_path(mode)

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
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # --- Coconut: load checkpoint AFTER wrapping ---
    if mode in ("pause", "coconut", "coconut_u"):
        print(f"Loading Coconut checkpoint: {checkpoint_path}")
        raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = clean_state_dict_keys(raw_state_dict)

        # Sanity check: keys should start with "base_causallm"
        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith("base_causallm"):
            print(f"  WARNING: first key is '{sample_key}'")
            print(f"  Expected keys starting with 'base_causallm.*'.")
            print(f"  This may not be a Coconut checkpoint.")

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
# PROBING: concept probability after k thoughts
# ============================================================================

def probe_concept_probs(coconut_model, tokenizer, input_ids, candidates, device):
    """
    After the Coconut forward pass processes the latent tokens, probe for
    concept probabilities by measuring token-level generation probs.

    Replicates Figure 5 methodology:
        p(concept) = prod_{i} p(token_i | context, "Every", token_{<i})

    Procedure:
        1. Coconut.forward() processes [Question] <bot> [latent]*k <eot>,
           running the hidden-state recurrence over the latent positions.
        2. Feed " Every" prefix autoregressively to reach concept-name position.
        3. Compute p(concept) = p(tok_0 | ctx+"Every") * p(tok_1 | ...) * ...

    The Coconut.forward() call uses the authors' own code for the
    continuous-thought recurrence. The only variable between conditions
    is the model weights.
    """
    base_model = coconut_model.base_causallm
    embedding = coconut_model.embedding

    with torch.no_grad():
        # Step 1: Coconut forward pass
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)

        outputs = coconut_model.forward(
            input_ids, attention_mask, labels, position_ids
        )
        inputs_embeds = outputs.inputs_embeds

        # Step 2: Run base model over full inputs_embeds, then feed " Every"
        full_outputs = base_model(inputs_embeds=inputs_embeds, use_cache=True)
        kv_cache = full_outputs.past_key_values
        current_pos = inputs_embeds.shape[1]

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

        # logits at concept-name start position
        logits_at_concept = prefix_out.logits[0, 0, :]

        # Step 3: Compute p(concept) for each candidate
        concept_log_probs = {}
        for concept_name, _ in candidates:
            c_tokens = tokenizer.encode(" " + concept_name, add_special_tokens=False)
            if len(c_tokens) == 0:
                concept_log_probs[concept_name] = float("-inf")
                continue

            # p(first concept token | context + " Every")
            p_first = F.softmax(logits_at_concept, dim=-1)[c_tokens[0]].item()
            log_p = math.log(p_first + 1e-30)

            # p(remaining tokens) — autoregressive, on a cloned KV cache
            if len(c_tokens) > 1:
                concept_cache = tuple((k.clone(), v.clone()) for k, v in kv_cache)
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

    # Normalize via softmax over log-probs
    max_lp = max(concept_log_probs.values())
    exp_probs = {c: math.exp(lp - max_lp) for c, lp in concept_log_probs.items()}
    Z = sum(exp_probs.values())
    concept_probs = {c: ep / Z for c, ep in exp_probs.items()}

    return concept_probs, concept_log_probs


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(prob_dict):
    """
    entropy:            H = -sum p log2(p)
    max_entropy:        log2(N) for N candidates
    normalized_entropy: H / max_entropy, in [0, 1]
    top1:               max(p)
    top2_cumul:         sum of top-2 probabilities
    top3_cumul:         sum of top-3 probabilities
    """
    vals = sorted(prob_dict.values(), reverse=True)
    n = len(vals)

    entropy = -sum(p * math.log2(p) for p in vals if p > 1e-30)
    max_entropy = math.log2(n) if n > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": norm_entropy,
        "top1": vals[0] if vals else 0.0,
        "top2_cumul": sum(vals[:2]),
        "top3_cumul": sum(vals[:3]),
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
        setup_model_and_tokenizer(args.mode, device)

    print(f"Loading data: {PROSQA_TEST}")
    with open(PROSQA_TEST) as f:
        raw_data = json.load(f)

    base_dataset = get_dataset(PROSQA_TEST, tokenizer, max_size=args.num_samples)

    num_samples = min(args.num_samples, len(raw_data))
    max_k = args.max_thoughts
    output_path = get_output_path(args.mode)

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

            concept_probs, concept_log_probs = probe_concept_probs(
                coconut_model, tokenizer, input_ids, candidates, device
            )

            metrics = compute_metrics(concept_probs)

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
                    f"  [{si}] H={metrics['entropy']:.3f} "
                    f"top1={metrics['top1']:.3f} | {top3_str}"
                )

    # --- Aggregate ---
    MODE_LABELS = {
        "base": "Base GPT-2 (no training)",
        "cot": "CoT-finetuned GPT-2 (no recurrence training)",
        "pause": "GPT-2 finetuned with pause tokens (no thought tokens)",
        "coconut": "Coconut-trained GPT-2",
        "coconut_u": "Coconut-trained GPT-2 with uniform probability = 0.3",
    }
    print("\n" + "=" * 70)
    print(f"AGGREGATE — {MODE_LABELS[args.mode]}")
    print("=" * 70)

    summary = {}
    for k in range(0, max_k + 1):
        entropies, top1s, top2s, top3s = [], [], [], []
        for si in all_results:
            if k in all_results[si]:
                m = all_results[si][k]["metrics"]
                entropies.append(m["entropy"])
                top1s.append(m["top1"])
                top2s.append(m["top2_cumul"])
                top3s.append(m["top3_cumul"])

        if not entropies:
            continue

        def ms(xs):
            mu = sum(xs) / len(xs)
            return mu, (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

        me, se = ms(entropies)
        mt1, st1 = ms(top1s)
        mt2, _ = ms(top2s)
        mt3, _ = ms(top3s)

        summary[k] = {
            "mean_entropy": me,
            "std_entropy": se,
            "mean_top1": mt1,
            "std_top1": st1,
            "mean_top2_cumul": mt2,
            "mean_top3_cumul": mt3,
            "n": len(entropies),
        }
        print(
            f"  k={k}: H={me:.3f}+/-{se:.3f}  "
            f"top1={mt1:.3f}+/-{st1:.3f}  "
            f"top2={mt2:.3f}  top3={mt3:.3f}  (n={len(entropies)})"
        )

    # --- Save ---
    output = {
        "experiment": "superposition_control",
        "mode": args.mode,
        "checkpoint": checkpoint_path,
        "data_path": str(PROSQA_TEST),
        "num_samples": num_samples,
        "max_thoughts": max_k,
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
    parser.add_argument("--mode", choices=["base", "cot", "pause", "coconut", 'coconut_u'], default="base")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_thoughts", type=int, default=6)
    args = parser.parse_args()

    run_experiment(args)