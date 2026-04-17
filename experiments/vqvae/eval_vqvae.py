"""
Evaluate VQ-VAE quantization on Coconut continuous thoughts.

Two evaluation modes:

1. INTERVENTION: Run Coconut inference but replace each continuous thought
   h_t with its nearest codebook entry z_q before feeding it back.
   Measure task accuracy. This answers: can the reasoning survive
   discretization at codebook size K?

2. TRAJECTORY ANALYSIS: For each instance, extract the codebook assignment
   sequence (j_0, j_1, ..., j_K) and stratify by BFS behavioral category
   from the control experiment. This answers: do BFS and anti-BFS instances
   traverse different codebook trajectories?

Usage:
    # All codebooks, both modes, single model load
    python eval_vqvae.py --codebook_paths results/codebook_K4.pt results/codebook_K8.pt \\
        results/codebook_K16.pt --mode both --bfs_categories_path categories.json

    # Trajectory analysis only (no GPU needed)
    python eval_vqvae.py --codebook_paths results/codebook_K*.pt --mode trajectory \\
        --bfs_categories_path categories.json

    # Intervention only
    python eval_vqvae.py --codebook_paths results/codebook_K16.pt --mode intervention
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from contThought.coconut import Coconut
from src.utils import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import BASE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PAUSE_GPT2, PROSQA_TEST


# ── Model loading (identical to extract_thoughts.py) ────────────────

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
    Load GPT-2, add special tokens, wrap in Coconut, load checkpoint.
    Returns (base_model, tokenizer, start_id, end_id, latent_id).
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

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    print(f"Loading Coconut checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)
    missing, unexpected = coconut_model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(unexpected)
    print(f"  Loaded {n_loaded}/{len(state_dict)} keys")

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return base_model, tokenizer, start_id, end_id, latent_id


def load_prosqa(path: str, max_instances: int = None) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    if max_instances is not None:
        data = data[:max_instances]
    return data


def format_prompt(sample: dict, tokenizer: AutoTokenizer) -> torch.Tensor:
    question_text = sample["question"]
    prompt = question_text + " <|start-latent|>"
    return tokenizer.encode(prompt, return_tensors="pt")


# ════════════════════════════════════════════════════════════════════
# MODE 1: Intervention — quantized recurrence
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_quantized_inference(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    end_id: int,
    codebook: torch.Tensor,
    sample: dict,
    n_thoughts: int,
    device: str,
) -> dict:
    """
    Run Coconut recurrence with quantization intervention.

    Normal:   h_0 -> model -> h_1 -> model -> h_2 -> ... -> answer
    Quantized: h_0 -> quantize -> model -> h_1 -> quantize -> ... -> answer

    quantize(h_t) = c_j  where  j = argmin_i || h_t - c_i ||_2

    After K quantized steps, feed <|end-latent|> and greedy-decode the answer.
    Answer extraction follows the training script: split on "#", take last part.
    """
    codebook_dev = codebook.to(device)
    input_ids = format_prompt(sample, tokenizer).to(device)

    # ── Step 0: Process prompt ──────────────────────────────────────
    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]  # (D,)
    past_kv = outputs.past_key_values
    trajectory = []

    # ── Steps 1..K: Quantized recurrence ────────────────────────────
    for t in range(n_thoughts):
        # Quantize: snap h to nearest codebook entry
        # || h - c_i ||^2 for all codebook entries
        dists = torch.cdist(h.unsqueeze(0), codebook_dev).squeeze(0)  # (num_codes,)
        code_idx = dists.argmin().item()
        h_q = codebook_dev[code_idx]  # (D,)
        trajectory.append(code_idx)

        # Feed quantized vector as next input
        outputs = base_model(
            inputs_embeds=h_q.unsqueeze(0).unsqueeze(0),  # (1, 1, D)
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :]
        past_kv = outputs.past_key_values

    # ── Decode answer ───────────────────────────────────────────────
    # Feed <|end-latent|> to transition from latent mode to answer generation
    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(
        input_ids=end_input,
        past_key_values=past_kv,
        use_cache=True,
    )
    past_kv = outputs.past_key_values

    # Greedy decode up to 128 tokens (matching training script's max_new_tokens)
    generated = []
    next_logits = outputs.logits[0, -1, :]
    for _ in range(128):
        next_token = next_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        out = base_model(
            input_ids=torch.tensor([[next_token]], device=device),
            past_key_values=past_kv,
            use_cache=True,
        )
        next_logits = out.logits[0, -1, :]
        past_kv = out.past_key_values

    # Answer extraction matching training script:
    # text_output.split("#")[-1].replace(",", "").strip()
    text_output = tokenizer.decode(generated, skip_special_tokens=True)
    answer_output = text_output.split("#")[-1].replace(",", "").strip()
    correct_answer = sample.get("answer", "").replace(",", "").strip()

    return {
        "predicted": answer_output,
        "correct": correct_answer,
        "is_correct": answer_output == correct_answer,
        "trajectory": trajectory,
        "full_output": text_output,
    }


def run_intervention_eval(
    base_model, tokenizer, end_id, codebook, data, n_thoughts, device
) -> dict:
    results = []
    n_correct = 0

    for idx, sample in enumerate(data):
        if idx % 25 == 0:
            print(f"  [Eval] Instance {idx}/{len(data)}")

        r = run_quantized_inference(
            base_model, tokenizer, end_id, codebook,
            sample, n_thoughts, device,
        )
        r["instance_idx"] = idx
        results.append(r)
        if r["is_correct"]:
            n_correct += 1

    accuracy = n_correct / len(data)
    print(f"  Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return {"accuracy": accuracy, "n_correct": n_correct, "results": results}


# ════════════════════════════════════════════════════════════════════
# MODE 2: Trajectory analysis stratified by BFS category
# ════════════════════════════════════════════════════════════════════

def load_bfs_categories(path: str) -> dict:
    """
    Load per-instance BFS categories from control experiment.

    Expected: JSON mapping instance_idx -> category string.
    Categories: "no_superposition", "transient", "sustained_no_convergence",
                "bfs", "anti_bfs"
    """
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {d["instance_idx"]: d["category"] for d in data}
    elif isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    else:
        raise ValueError(f"Unexpected format in {path}")


def analyze_trajectories(
    assignments: torch.Tensor,
    bfs_categories: dict = None,
    num_codes: int = None,
) -> dict:
    """
    Analyze codebook assignment trajectories.

    assignments: (N, T) — codebook indices per instance per step.
    bfs_categories: optional dict instance_idx -> category.

    Computes:
        - Trajectory diversity: unique trajectories / N
        - Per-category: transition matrices, self-transition rates, codes used
    """
    N, T = assignments.shape
    K = num_codes or int(assignments.max().item()) + 1

    trajectories = [tuple(assignments[i].tolist()) for i in range(N)]
    traj_counts = Counter(trajectories)
    n_unique = len(traj_counts)

    result = {
        "n_instances": N,
        "n_steps": T,
        "n_unique_trajectories": n_unique,
        "trajectory_diversity": n_unique / N,
        "top_trajectories": traj_counts.most_common(10),
    }

    if bfs_categories is None:
        return result

    # ── Stratified analysis ─────────────────────────────────────────
    category_stats = {}

    for cat in sorted(set(bfs_categories.values())):
        cat_indices = [i for i in range(N) if bfs_categories.get(i) == cat]
        if not cat_indices:
            continue

        cat_trajectories = [trajectories[i] for i in cat_indices]
        cat_counts = Counter(cat_trajectories)

        # Transition matrix T_cat[i, j] = count of code i -> code j transitions
        transition_matrix = np.zeros((K, K), dtype=int)
        for traj in cat_trajectories:
            for step in range(len(traj) - 1):
                transition_matrix[traj[step], traj[step + 1]] += 1

        # Self-transition rate = trace(T) / sum(T)
        total_transitions = transition_matrix.sum()
        self_transitions = np.trace(transition_matrix)
        self_transition_rate = (
            self_transitions / total_transitions if total_transitions > 0 else 0.0
        )

        codes_used = set()
        for traj in cat_trajectories:
            codes_used.update(traj)

        category_stats[cat] = {
            "n_instances": len(cat_indices),
            "n_unique_trajectories": len(cat_counts),
            "trajectory_diversity": len(cat_counts) / len(cat_indices),
            "self_transition_rate": float(self_transition_rate),
            "n_codes_used": len(codes_used),
            "codes_used": sorted(codes_used),
            "top_trajectories": cat_counts.most_common(5),
            "transition_matrix": transition_matrix.tolist(),
        }

    result["per_category"] = category_stats
    return result


# ════════════════════════════════════════════════════════════════════

def deep_convert(obj):
    """Make everything JSON-serializable."""
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, tuple):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def eval_single_codebook(
    codebook_path: str,
    mode: str,
    bfs_categories: dict,
    base_model,
    tokenizer,
    end_id: int,
    data: list,
    n_thoughts: int,
    device: str,
) -> dict:
    """
    Run trajectory analysis and/or intervention eval for one codebook.

    base_model, tokenizer, end_id, data can be None if mode == "trajectory".
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Loading codebook from {codebook_path}")
    cb_data = torch.load(codebook_path, map_location="cpu", weights_only=False)
    codebook = cb_data["codebook"]       # (num_codes, D)
    assignments = cb_data["assignments"] # (N, T)
    num_codes = cb_data["num_codes"]
    print(f"[INFO] Codebook: K={num_codes}, D={codebook.shape[1]}")
    print(f"[INFO] Training metrics: {cb_data['metrics']}")

    results = {"codebook_path": codebook_path, "num_codes": num_codes}

    # ── Trajectory analysis ─────────────────────────────────────────
    if mode in ("trajectory", "both"):
        print("\n[Trajectory Analysis]")
        traj_results = analyze_trajectories(assignments, bfs_categories, num_codes)
        results["trajectory"] = traj_results

        print(f"  Unique trajectories: {traj_results['n_unique_trajectories']}/{traj_results['n_instances']}")
        print(f"  Trajectory diversity: {traj_results['trajectory_diversity']:.2%}")

        if "per_category" in traj_results:
            print(f"\n  Per-category breakdown:")
            for cat, stats in traj_results["per_category"].items():
                print(
                    f"    {cat:>28s}: "
                    f"{stats['n_instances']:>3d} instances, "
                    f"{stats['n_unique_trajectories']:>3d} unique traj, "
                    f"self-transition={stats['self_transition_rate']:.1%}, "
                    f"codes used={stats['n_codes_used']}"
                )

    # ── Intervention evaluation ─────────────────────────────────────
    if mode in ("intervention", "both"):
        print("\n[Intervention Evaluation]")
        eval_results = run_intervention_eval(
            base_model, tokenizer, end_id, codebook,
            data, n_thoughts, device,
        )
        results["intervention"] = {
            "accuracy": eval_results["accuracy"],
            "n_correct": eval_results["n_correct"],
            "n_total": len(data),
            "trajectories": [r["trajectory"] for r in eval_results["results"]],
        }

    # ── Save per-codebook results ───────────────────────────────────
    output_path = str(Path(codebook_path).with_suffix(".eval.json"))
    serializable = deep_convert(results)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[INFO] Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQ-VAE codebooks on Coconut thoughts."
    )
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument(
        "--codebook_paths", type=str, nargs="+", required=True,
        help="One or more .pt codebook files from train_vqvae.py.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["intervention", "trajectory", "both"],
        default="both",
    )
    parser.add_argument("--prosqa_path", type=str, default=None)
    parser.add_argument("--bfs_categories_path", type=str, default=None)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ── Load BFS categories (shared across all codebooks) ───────────
    bfs_categories = None
    if args.bfs_categories_path:
        bfs_categories = load_bfs_categories(args.bfs_categories_path)
        print(f"[INFO] Loaded BFS categories for {len(bfs_categories)} instances")

    # ── Load model and data ONCE (shared across all codebooks) ──────
    base_model, tokenizer, end_id, data = None, None, None, None
    if args.mode in ("intervention", "both"):
        print("[INFO] Loading model (once for all codebooks)...")
        base_model, tokenizer, start_id, end_id, latent_id = setup_model_and_tokenizer(
            args.model, args.device
        )
        prosqa_path = args.prosqa_path or str(PROSQA_TEST)
        data = load_prosqa(prosqa_path, args.max_instances)
        print(f"[INFO] Model and data loaded. Will evaluate {len(args.codebook_paths)} codebooks.\n")

    # ── Loop over codebooks ─────────────────────────────────────────
    summary = {}
    for cb_path in args.codebook_paths:
        results = eval_single_codebook(
            codebook_path=cb_path,
            mode=args.mode,
            bfs_categories=bfs_categories,
            base_model=base_model,
            tokenizer=tokenizer,
            end_id=end_id,
            data=data,
            n_thoughts=args.n_thoughts,
            device=args.device,
        )
        K = results["num_codes"]
        summary[K] = {
            "codebook_path": cb_path,
        }
        if "intervention" in results:
            summary[K]["accuracy"] = results["intervention"]["accuracy"]
        if "trajectory" in results:
            summary[K]["trajectory_diversity"] = results["trajectory"]["trajectory_diversity"]
            summary[K]["n_unique_trajectories"] = results["trajectory"]["n_unique_trajectories"]

    # ── Print sweep summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    header = f"{'K':>6}"
    if args.mode in ("intervention", "both"):
        header += f"  {'Accuracy':>10}"
    if args.mode in ("trajectory", "both"):
        header += f"  {'Unique Traj':>12}  {'Diversity':>10}"
    print(header)
    print("-" * len(header))

    for K in sorted(summary.keys()):
        s = summary[K]
        line = f"{K:>6}"
        if "accuracy" in s:
            line += f"  {s['accuracy']:>10.1%}"
        if "trajectory_diversity" in s:
            line += f"  {s['n_unique_trajectories']:>12}  {s['trajectory_diversity']:>10.2%}"
        print(line)


if __name__ == "__main__":
    main()