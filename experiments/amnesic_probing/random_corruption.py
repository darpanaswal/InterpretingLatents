"""
Stage 1: Progressive Corruption of Latent Thoughts via Random Noise.

For a given noise scale λ, the corrupted thought vector at step t:
    h'_t = h_t + λ * σ_{h_t} * ε,    ε ~ N(0, I)
where σ_{h_t} is the standard deviation of h_t across feature dimensions.

Supports ProsQA and GSM8k across all model types including CODI.
Runs multiple seeds per noise level for error bounds.

Usage:
    python -m experiments.amnesic_probing.random_corruption --task prosqa --model pause
    python -m experiments.amnesic_probing.random_corruption --task prosqa --model coconut_u
    python -m experiments.amnesic_probing.random_corruption --task gsm --model codi
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    run_intervened_inference_pauseaware,
    _compare_answers,
)


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def deep_convert(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════
# Intervention
# ═══════════════════════════════════════════════════════════════════

def make_noise_intervention(noise_scale, device):
    """
    Corrupted vector: h' = h + λ * σ_h * ε
    where ε ~ N(0, I) and σ_h = std(h) across feature dimensions.
    """
    def intervention_fn(h, t):
        std_h = torch.std(h, dim=-1, keepdim=True)
        noise = torch.randn_like(h, device=device)
        return h + (noise * std_h * noise_scale)
    return intervention_fn


# ═══════════════════════════════════════════════════════════════════
# Evaluation wrappers (batched over n_runs per instance)
# ═══════════════════════════════════════════════════════════════════

def run_corruption_eval(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, noise_scale, n_runs, label="",
    start_id=None, latent_id=None, task="prosqa",
):
    """Run n_runs corrupted inferences per instance in a single pass.

    For each instance, generates n_runs independent noise interventions
    (with different seeds) and runs inference for each. Returns the
    mean accuracy across runs.
    """
    # n_correct[r] = number correct for run r
    n_correct = [0] * n_runs

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")

        for r in range(n_runs):
            # Per-instance, per-run seed for reproducibility
            torch.manual_seed(42 + r * 1000 + idx * 7 + int(noise_scale * 100))
            noise_fn = make_noise_intervention(noise_scale, device)

            result = run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                n_thoughts, device, noise_fn,
                start_id=start_id, latent_id=latent_id, task=task,
            )
            if result["is_correct"]:
                n_correct[r] += 1

    accs = [c / len(data) for c in n_correct]
    mean_acc = np.mean(accs)
    print(f"    [{label}] Mean Accuracy: {mean_acc:.1%} (over {n_runs} runs)")
    return accs


@torch.no_grad()
def run_codi_corruption_eval(
    codi_dict, data, n_thoughts, device, noise_scale, n_runs, label="",
):
    """CODI version: n_runs corrupted inferences per instance in one pass."""
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']

    n_correct = [0] * n_runs

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")

        question_text = sample["question"]
        question_tokens = tokenizer.encode(question_text, add_special_tokens=True)

        if remove_eos:
            input_ids_list = question_tokens + [bot_id]
        else:
            input_ids_list = question_tokens + [tokenizer.eos_token_id, bot_id]
        input_ids = torch.tensor([input_ids_list], device=device)

        for r in range(n_runs):
            torch.manual_seed(42 + r * 1000 + idx * 7 + int(noise_scale * 100))
            noise_fn = make_noise_intervention(noise_scale, device)

            # Step 0
            outputs = base_model(
                input_ids=input_ids, use_cache=True, output_hidden_states=True,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][0, -1, :]
            h = noise_fn(h, 0)

            latent = h.unsqueeze(0).unsqueeze(0)
            if use_prj and prj is not None:
                latent = prj(latent)

            # Steps 1..K
            for t in range(1, n_thoughts + 1):
                outputs = base_model(
                    inputs_embeds=latent, use_cache=True,
                    output_hidden_states=True, past_key_values=past_kv,
                )
                past_kv = outputs.past_key_values
                h = outputs.hidden_states[-1][0, -1, :]
                h = noise_fn(h, t)

                latent = h.unsqueeze(0).unsqueeze(0)
                if use_prj and prj is not None:
                    latent = prj(latent)

            # Decode
            eot_input = torch.tensor([[eot_id]], device=device)
            outputs = base_model(
                input_ids=eot_input, past_key_values=past_kv, use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[0, -1, :]

            generated = []
            for _ in range(128):
                next_token = next_logits.argmax().item()
                if next_token == tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                out = base_model(
                    input_ids=torch.tensor([[next_token]], device=device),
                    past_key_values=past_kv, use_cache=True,
                )
                next_logits = out.logits[0, -1, :]
                past_kv = out.past_key_values

            text = tokenizer.decode(generated, skip_special_tokens=True)
            _, _, is_correct = _compare_answers(text, sample, "gsm")
            if is_correct:
                n_correct[r] += 1

    accs = [c / len(data) for c in n_correct]
    mean_acc = np.mean(accs)
    print(f"    [{label}] Mean Accuracy: {mean_acc:.1%} (over {n_runs} runs)")
    return accs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Progressive random corruption of latent thought vectors."
    )
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--noise_sweep", type=str, default="0.1,0.5,1.0,5.0,10.0,25.0,50.0",
    )
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.model == "codi" and args.task != "gsm":
        parser.error("CODI is only available for --task gsm")

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "steering" / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and data ─────────────────────────────────────────
    is_codi = (args.model == "codi")
    codi_dict = None
    coconut_model = base_model = tokenizer = None
    end_id = start_id = latent_id = None

    if is_codi:
        codi_dict = setup_codi_model(args.device)
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device)

    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Task: {args.task}, Model: {args.model}, "
          f"instances: {len(data)}, device: {args.device}")

    # ── Baseline ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    identity_fn = lambda h, t: h
    if is_codi:
        # noise_scale=0 → h + 0*noise = h, equivalent to identity
        baseline_accs = run_codi_corruption_eval(
            codi_dict, data, args.n_thoughts, args.device,
            noise_scale=0.0, n_runs=1, label="Baseline",
        )
        baseline_acc = baseline_accs[0]
    else:
        n_correct = 0
        for idx, sample in enumerate(data):
            if idx % 100 == 0:
                print(f"    [Baseline] {idx}/{len(data)}")
            r = run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                args.n_thoughts, args.device, identity_fn,
                start_id=start_id, latent_id=latent_id, task=args.task,
            )
            if r["is_correct"]:
                n_correct += 1
        baseline_acc = n_correct / len(data)
    print(f"  → Baseline Accuracy: {baseline_acc:.1%}")

    # ── Progressive corruption ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PROGRESSIVE RANDOM CORRUPTION ({args.task.upper()})")
    print("=" * 60)

    noise_scales = [float(n) for n in args.noise_sweep.split(",")]
    n_runs = args.n_runs
    corruption_results = {"0.0": baseline_acc}

    for scale in noise_scales:
        print(f"\n  Noise Scale: {scale} ({n_runs} runs batched per instance)")

        if is_codi:
            accs = run_codi_corruption_eval(
                codi_dict, data, args.n_thoughts, args.device,
                scale, n_runs, label=f"Noise {scale}",
            )
        else:
            accs = run_corruption_eval(
                coconut_model, base_model, tokenizer, end_id, data,
                args.n_thoughts, args.device, scale, n_runs,
                label=f"Noise {scale}",
                start_id=start_id, latent_id=latent_id, task=args.task,
            )

        mean_acc = np.mean(accs)
        corruption_results[str(scale)] = mean_acc
        print(f"  → Noise {scale}: {mean_acc:.1%}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  CORRUPTION SUMMARY ({args.task.upper()})")
    print(f"  {'='*60}")
    print(f"  {'Noise Scale':>12}  {'Accuracy':>10}  {'Drop':>10}")
    for scale in sorted([float(k) for k in corruption_results.keys()]):
        acc = corruption_results[str(scale)]
        drop = baseline_acc - acc
        print(f"  {scale:>12.2f}  {acc:>10.1%}  {drop:>10.1%}")

    # ── Save ────────────────────────────────────────────────────────
    save = {
        "task": args.task,
        "model": args.model,
        "n_runs": n_runs,
        "baseline_accuracy": baseline_acc,
        "per_scale": {
            str(scale): float(acc)
            for scale, acc in [
                (float(k), v) for k, v in corruption_results.items()
            ]
        },
    }
    path = output_dir / "random_corruption_results.json"
    with open(path, "w") as f:
        json.dump(deep_convert(save), f, indent=2)
    print(f"  Saved to {path}")


if __name__ == "__main__":
    main()