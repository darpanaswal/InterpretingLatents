"""
Interventions: INLP Ablation + Direction-Agnostic Steering.

Loads precomputed inlp_results.pt (from inlp.py) and applies two types
of causal interventions on thought vectors during inference:

1. ABLATION: Apply INLP nullspace projection (remove label-encoding
   directions) and random control projection, measure accuracy drop.

2. STEERING: For each example, extract its concept-encoding component
   from the thought vector and push along/against it:

       # Concept component at timestep t:
       #   c_t = (I - P_t) @ h_t
       # where P_t is the INLP nullspace projection (removes concept info).
       # (I - P_t) projects ONTO the concept-encoding subspace.
       #
       # Steering direction (unit vector):
       #   d_t = c_t / ||c_t||
       #
       # Steered vector (push along concept direction):
       #   h'_t = h_t + alpha * d_t
       #
       # Flip criterion (direction-agnostic, parser-agnostic):
       #   raw_text(h') != raw_text(h)
       # We don't care WHERE the output moves, only WHETHER it moves.
       # Using raw decoded text (not parsed labels) avoids collapsing
       # different garbled outputs into a single 'inf' bucket on GSM.

   Sweep alpha in {50, 100, 250, 500}. Report flip rate.

   Control: same procedure but using (I - P_rand_t) directions instead
   of INLP-identified concept directions.

Usage:
    python -m experiments.amnesic_probing.interventions --task prosqa --model coconut
    python -m experiments.amnesic_probing.interventions --task prosqa --model coconut_u
    python -m experiments.amnesic_probing.interventions --task prosqa --model pause

    python -m experiments.amnesic_probing.interventions --task gsm --model pause
    python -m experiments.amnesic_probing.interventions --task gsm --model coconut
    python -m experiments.amnesic_probing.interventions --task gsm --model coconut_u
    python -m experiments.amnesic_probing.interventions --task gsm --model codi
"""

import json
import torch
import argparse
import subprocess
import numpy as np
from pathlib import Path
from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, THOUGHTS
from src.utils import (
    setup_model_and_tokenizer,
    run_intervened_inference_pauseaware,
    extract_answer_number,
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


def extract_predicted_label(predicted_text, task):
    """Extract label from model output (used for accuracy only, NOT for flip)."""
    if task == "prosqa":
        text = predicted_text.strip().rstrip(".")
        if " is a " in text:
            return text.split(" is a ")[-1].strip()
        return text
    else:
        return str(extract_answer_number(predicted_text))


def normalize_text_for_flip(text):
    """
    Canonicalize raw decoded text for flip comparison.

    Strip whitespace and lowercase so that trivial formatting differences
    don't register as flips. We intentionally do NOT parse numbers or
    extract labels here — two different garbled outputs should register
    as two different strings, not collapse into one 'inf' bucket.
    """
    if text is None:
        return ""
    return text.strip().lower()


def deep_convert(obj):
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
# Intervention factories
# ═══════════════════════════════════════════════════════════════════

def make_projection_intervention(projections, device):
    """Projected vector: h'_t = P_t @ h_t."""
    P_tensors = {t: torch.tensor(P, dtype=torch.float32, device=device)
                 for t, P in projections.items()}

    def intervention_fn(h, t):
        if t in P_tensors:
            return P_tensors[t] @ h
        return h
    return intervention_fn


def make_concept_steering_intervention(concept_projectors, alpha, device):
    """
    Per-instance concept steering.

    For each thought vector h_t at timestep t:
        # concept component: c_t = C_t @ h_t   where C_t = I - P_t
        # direction: d_t = c_t / ||c_t||
        # steered: h'_t = h_t + alpha * d_t

    This pushes h_t further along its own concept direction.
    No target label required.
    """
    C_tensors = {t: torch.tensor(C, dtype=torch.float32, device=device)
                 for t, C in concept_projectors.items()}

    def intervention_fn(h, t):
        if t not in C_tensors:
            return h
        # c_t = C_t @ h_t  (project onto concept subspace)
        c = C_tensors[t] @ h
        norm = c.norm()
        if norm < 1e-8:
            return h
        # d_t = c_t / ||c_t||
        d = c / norm
        # h'_t = h_t + alpha * d_t
        return h + alpha * d
    return intervention_fn


# ═══════════════════════════════════════════════════════════════════
# Eval wrapper
# ═══════════════════════════════════════════════════════════════════

def run_eval_with_intervention(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, intervention_fn, label="",
    start_id=None, latent_id=None, task="prosqa",
):
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")
        r = run_intervened_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
            start_id=start_id, latent_id=latent_id, task=task,
        )
        if r["is_correct"]:
            n_correct += 1
    accuracy = n_correct / len(data)
    print(f"    [{label}] Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return accuracy


def run_steering_sweep(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, concept_projectors, alphas,
    baseline_texts, label_name,
    start_id=None, latent_id=None, task="prosqa",
):
    """
    For each example, steer along its concept direction at each alpha.
    Flip criterion is direction-agnostic AND parser-agnostic:

        # flip if normalize(raw_text(h')) != normalize(raw_text(h))

    Using raw text instead of parsed labels is critical on GSM: when
    steering produces malformed output, extract_answer_number collapses
    everything to 'inf', making INLP and Rand perturbations look
    identical even when their raw outputs differ completely.

    concept_projectors: dict {t: C_t} where C_t = I - P_t
        C_t projects onto concept-encoding subspace (INLP or random control).

    Returns: {alpha: {"n_flipped": int, "n_total": int, "flip_rate": float}}
    """
    n_total = len(data)
    n_flipped = {alpha: 0 for alpha in alphas}

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label_name}] {idx}/{n_total}")

        base_text = baseline_texts[idx]

        for alpha in alphas:
            steer_fn = make_concept_steering_intervention(
                concept_projectors, alpha, device,
            )
            r = run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                n_thoughts, device, steer_fn,
                start_id=start_id, latent_id=latent_id, task=task,
            )
            steered_text = r.get("text", r.get("predicted", ""))

            # Direction-agnostic + parser-agnostic: did the raw text change?
            if normalize_text_for_flip(steered_text) != normalize_text_for_flip(base_text):
                n_flipped[alpha] += 1

    results = {}
    for alpha in alphas:
        flip_rate = n_flipped[alpha] / max(n_total, 1)
        results[alpha] = {
            "n_flipped": n_flipped[alpha],
            "n_total": n_total,
            "flip_rate": flip_rate,
        }
        print(f"    [{label_name}] Alpha {alpha}: "
              f"{n_flipped[alpha]}/{n_total} flipped ({flip_rate:.1%})")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interventions: INLP ablation + direction-agnostic steering."
    )
    parser.add_argument("--task", type=str, choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model", type=str, choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--alpha_sweep", type=str, default="0.1,0.5,1,5,10,25,50,100,250,500")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "inlp" / args.task / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and data ─────────────────────────────────────────
    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
        setup_model_and_tokenizer(args.task, args.model, args.device)
    data = load_data(args.task, args.max_instances)
    print(f"[INFO] Task: {args.task}, Model: {args.model}, "
          f"instances: {len(data)}, device: {args.device}")

    # ── Load thoughts + INLP ───────────────────────────────────────
    thoughts_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not thoughts_path.exists():
        print(f"  [INFO] Triggering thought extraction...")
        cmd = ["python", "-u", "-m", "experiments.probe_thoughts.extract_thoughts",
               "--task", args.task, "--model", args.model,
               "--n_thoughts", str(args.n_thoughts)]
        if args.max_instances:
            cmd.extend(["--max_instances", str(args.max_instances)])
        subprocess.run(cmd, check=True)

    print(f"  Loading thoughts from {thoughts_path}")
    thoughts = torch.load(thoughts_path, map_location="cpu", weights_only=False)["thoughts"]

    inlp_path = BASE_DIR / f"outputs/inlp/{args.task}/{args.model}/inlp_results.pt"
    if not inlp_path.exists():
        raise FileNotFoundError(f"Please run inlp.py to extract inlp_results for {args.model}--{args.task}")

    print(f"[INFO] Loading INLP from {inlp_path}")
    inlp_data = torch.load(inlp_path, map_location="cpu", weights_only=False)
    projections = {int(k) if isinstance(k, str) else k: v
                   for k, v in inlp_data["projections"].items()}
    rand_projections = {int(k) if isinstance(k, str) else k: v
                        for k, v in inlp_data["rand_projections"].items()}

    # ── Concept projectors: C_t = I - P_t ──────────────────────────
    # P_t projects onto the nullspace (removes concept info).
    # I - P_t projects ONTO the concept-encoding subspace.
    D = thoughts.shape[2]
    I = np.eye(D)
    concept_projectors_inlp = {t: I - P for t, P in projections.items()}
    concept_projectors_rand = {t: I - P for t, P in rand_projections.items()}

    # ── Baseline (per-instance) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    identity_fn = lambda h, t: h
    baseline_results = []
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [Baseline] {idx}/{len(data)}")
        baseline_results.append(run_intervened_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            args.n_thoughts, args.device, identity_fn,
            start_id=start_id, latent_id=latent_id, task=args.task,
        ))
    baseline_acc = sum(1 for r in baseline_results if r["is_correct"]) / len(data)

    # Cache raw baseline text for flip comparison (not parsed labels —
    # see run_steering_sweep docstring for why).
    baseline_texts = [
        r.get("text", r.get("predicted", ""))
        for r in baseline_results
    ]
    print(f"  Baseline Accuracy: {baseline_acc:.1%}")

    # ════════════════════════════════════════════════════════════════
    # PART 1: INLP ABLATION
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("INLP ABLATION")
    print("=" * 60)

    print("  Running INLP ablation...")
    inlp_fn = make_projection_intervention(projections, args.device)
    inlp_acc = run_eval_with_intervention(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, inlp_fn, label="INLP",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    print("  Running Rand control...")
    rand_fn = make_projection_intervention(rand_projections, args.device)
    rand_acc = run_eval_with_intervention(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, rand_fn, label="Rand",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    print(f"\n  {'='*50}")
    print(f"  ABLATION SUMMARY")
    print(f"  {'='*50}")
    print(f"  Baseline:     {baseline_acc:.1%}")
    print(f"  INLP:         {inlp_acc:.1%}  (drop: {baseline_acc - inlp_acc:.1%})")
    print(f"  Rand control: {rand_acc:.1%}  (drop: {baseline_acc - rand_acc:.1%})")

    ablation_results = {
        "task": args.task, "model": args.model,
        "baseline_accuracy": baseline_acc,
        "inlp_accuracy": inlp_acc, "rand_accuracy": rand_acc,
        "accuracy_drop_inlp": baseline_acc - inlp_acc,
        "accuracy_drop_rand": baseline_acc - rand_acc,
    }
    abl_path = output_dir / "ablation_results.json"
    with open(abl_path, "w") as f:
        json.dump(deep_convert(ablation_results), f, indent=2)
    print(f"  Saved to {abl_path}")

    # ════════════════════════════════════════════════════════════════
    # PART 2: DIRECTION-AGNOSTIC STEERING
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIRECTION-AGNOSTIC STEERING")
    print("=" * 60)

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # --- INLP concept directions ---
    print("\n  Steering along INLP concept directions:")
    inlp_steering = run_steering_sweep(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, concept_projectors_inlp, alphas,
        baseline_texts, label_name="INLP",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    # --- Random control directions ---
    print("\n  Steering along random control directions:")
    rand_steering = run_steering_sweep(
        coconut_model, base_model, tokenizer, end_id, data,
        args.n_thoughts, args.device, concept_projectors_rand, alphas,
        baseline_texts, label_name="Rand",
        start_id=start_id, latent_id=latent_id, task=args.task,
    )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  STEERING SUMMARY (direction-agnostic flip rate)")
    print(f"  {'='*60}")
    print(f"  {'Alpha':>8}  {'INLP Flip':>12}  {'Rand Flip':>12}  {'Interpretation'}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*30}")
    for alpha in alphas:
        inlp_fr = inlp_steering[alpha]["flip_rate"]
        rand_fr = rand_steering[alpha]["flip_rate"]
        if inlp_fr < 0.02 and rand_fr < 0.02:
            interp = "Robust (causal inertness)"
        elif inlp_fr > 0.05 and rand_fr < 0.02:
            interp = "Concept dirs are causal"
        elif inlp_fr > 0.05 and rand_fr > 0.05:
            interp = "Generally fragile"
        else:
            interp = "Ambiguous"
        print(f"  {alpha:>8g}  {inlp_fr:>11.1%}  {rand_fr:>11.1%}  {interp}")

    steering_results = {
        "task": args.task, "model": args.model,
        "alphas": alphas,
        "inlp_steering": deep_convert(inlp_steering),
        "rand_steering": deep_convert(rand_steering),
    }
    steer_path = output_dir / "steering_results.json"
    with open(steer_path, "w") as f:
        json.dump(deep_convert(steering_results), f, indent=2)
    print(f"\n  Saved to {steer_path}")


if __name__ == "__main__":
    main()