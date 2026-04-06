"""
Amnesic Probing via INLP on Continuous Thought Vectors.

Implements the methodology from Elazar et al. (2021) "Amnesic Probing:
Behavioral Explanation with Amnesic Counterfactuals" adapted for
Coconut's continuous thought vectors on ProsQA.

Three stages:

Stage 1 (RANDOM CORRUPTION): Progressively corrupt thought vectors with
    calibrated Gaussian noise to establish a dose-response baseline.

Stage 2 (INLP ABLATION): At each recurrence step t, iteratively train
    linear classifiers to predict the correct concept from h_t, project
    onto their nullspace, and repeat until no classifier beats majority.
    Re-run inference with the projection applied. Controls: random
    direction removal.

Stage 3 (STEERING): Add a contrastive direction to steer the model
    toward a specific wrong answer. Sweep over alpha values.

Pause-aware: Correctly handles both coconut (recurrence) and pause
    (single-pass) models. The pause model processes all thought tokens
    in a single forward pass with a fixed learned embedding — no
    recurrence. Interventions on pause models are applied via forward
    hooks on the last transformer layer.

Usage:
    python -m experiments.amnesic_probing --stage all --model pause
    python -m experiments.amnesic_probing --stage all --model coconut
    python -m experiments.amnesic_probing --stage all --model coconut_u
"""

import json
import torch
import argparse
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.svm import LinearSVC
from contThought.coconut import Coconut
from sklearn.metrics import accuracy_score
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.pause_aware_utils import (
    is_pause_model,
    run_intervened_inference_pauseaware,
    run_alpha_sweep_inference_pauseaware,
)
from utils.config import (
    BASE_DIR, BASE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PAUSE_GPT2, PROSQA_TEST
)


# ═══════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════

def extract_labels_and_concepts(data):
    """Fast CPU extraction of concepts directly from the dataset."""
    labels = []
    all_concepts = set()
    for sample in data:
        answer_text = sample["answer"].strip().rstrip(".")
        correct_concept = answer_text.split(" is a ")[-1].strip()
        labels.append(correct_concept)

        all_concepts.add(correct_concept)
        for step in sample.get("steps", []):
            parts = step.split(" is a ")
            if len(parts) == 2:
                concept_a = parts[0].replace("Every ", "").strip()
                concept_b = parts[1].strip().rstrip(".")
                all_concepts.add(concept_a)
                all_concepts.add(concept_b)

    return labels, sorted(all_concepts)


def _extract_concept(answer_text):
    """Extract concept from an answer like 'X is a <concept>.'"""
    answer_text = answer_text.strip().rstrip(".")
    if " is a " in answer_text:
        return answer_text.split(" is a ")[-1].strip()
    return answer_text


def load_prosqa(path, max_instances=None):
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

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


def setup_model_and_tokenizer(mode, device):
    """
    Load GPT-2, add Coconut special tokens, wrap in Coconut, load checkpoint.

    Returns coconut_model (the wrapper), base_model, tokenizer, and token IDs.
    The coconut_model is needed for pause-aware inference (it carries
    feedback_mode and pause_embedding).
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

    # Wrap in Coconut with correct feedback_mode
    feedback_mode = "pause_curriculum" if mode == "pause" else "continuous"
    coconut_model = Coconut(
        model, latent_id, start_id, end_id,
        tokenizer.eos_token_id, feedback_mode=feedback_mode,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)
    coconut_model.load_state_dict(state_dict, strict=False)

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return coconut_model, base_model, tokenizer, start_id, end_id, latent_id


# ═══════════════════════════════════════════════════════════════════
# Intervention factories
# ═══════════════════════════════════════════════════════════════════

def make_noise_intervention(noise_scale=0.1, device="cuda"):
    """
    Create an intervention that adds scaled Gaussian noise to the thought vector.

    noise_scale: fraction of the vector's standard deviation to add as noise.
    """
    def intervention_fn(h, t):
        std_h = torch.std(h, dim=-1, keepdim=True)
        noise = torch.randn_like(h, device=device)
        return h + (noise * std_h * noise_scale)

    return intervention_fn


def make_projection_intervention(projections, device):
    """Create an intervention function that applies INLP projection at each step."""
    P_tensors = {t: torch.tensor(P, dtype=torch.float32, device=device)
                 for t, P in projections.items()}

    def intervention_fn(h, t):
        if t in P_tensors:
            return P_tensors[t] @ h
        return h

    return intervention_fn


def make_identity_intervention():
    """No intervention (baseline)."""
    return lambda h, t: h


# ═══════════════════════════════════════════════════════════════════
# Evaluation wrapper (pause-aware)
# ═══════════════════════════════════════════════════════════════════

def run_eval_with_intervention(
    coconut_model, base_model, tokenizer, end_id, data,
    n_thoughts, device, intervention_fn, label="",
    start_id=None, latent_id=None,
):
    """Run inference on all instances with a given intervention, return accuracy."""
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")
        r = run_intervened_inference_pauseaware(
            coconut_model, base_model, tokenizer, end_id, sample,
            n_thoughts, device, intervention_fn,
            start_id=start_id, latent_id=latent_id,
        )
        if r["is_correct"]:
            n_correct += 1

    accuracy = n_correct / len(data)
    print(f"    [{label}] Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return accuracy


# ═══════════════════════════════════════════════════════════════════
# INLP (Iterative Nullspace Projection)
# ═══════════════════════════════════════════════════════════════════

def compute_nullspace_projection(W):
    """
    Compute the projection matrix onto the nullspace of W.

    Given classifier weight matrix W of shape (n_classes, D),
    P = I - V^T V where V = top singular vectors of W.
    """
    if W.ndim == 1:
        W = W.reshape(1, -1)

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = np.sum(S > 1e-10)
    basis = Vt[:rank, :]
    P = np.eye(W.shape[1]) - basis.T @ basis
    return P


def run_inlp(X, y, max_iter=300, convergence_threshold=1.0):
    """
    Iterative Nullspace Projection (Ravfogel et al., 2020).

    Iteratively trains linear classifiers and projects onto their nullspace
    until accuracy drops to majority baseline.

    Returns:
        P_total: (D, D) accumulated nullspace projection
        n_iterations: number of INLP iterations
        n_directions_removed: total directions removed
        classifiers: list of trained classifiers
    """
    N, D = X.shape
    majority_acc = np.max(np.bincount(y)) / N * 100

    P_total = np.eye(D)
    X_projected = X.copy()
    classifiers = []
    n_directions_removed = 0

    for iteration in range(max_iter):
        clf = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf.fit(X_projected, y)
        y_pred = clf.predict(X_projected)
        acc = accuracy_score(y, y_pred) * 100

        if acc <= majority_acc + convergence_threshold:
            break

        classifiers.append(clf)

        W = clf.coef_
        P_i = compute_nullspace_projection(W)
        P_total = P_i @ P_total
        X_projected = X @ P_total.T
        n_directions_removed += W.shape[0]

        if (iteration + 1) % 10 == 0:
            print(f"      Iter {iteration+1}: acc={acc:.1f}% "
                  f"(majority={majority_acc:.1f}%), "
                  f"removed {n_directions_removed} directions")

    final_acc = acc if iteration > 0 else majority_acc
    print(f"      INLP converged after {iteration+1} iterations, "
          f"final acc={final_acc:.1f}%, "
          f"removed {n_directions_removed} directions")

    return P_total, iteration + 1, n_directions_removed, classifiers


def compute_random_projection(D, n_directions_removed, seed=42):
    """Rand control: remove the same number of random directions."""
    rng = np.random.RandomState(seed)
    random_dirs = rng.randn(n_directions_removed, D)
    Q, _ = np.linalg.qr(random_dirs.T)
    n_actual = min(n_directions_removed, D)
    basis = Q[:, :n_actual].T
    P_rand = np.eye(D) - basis.T @ basis
    return P_rand


def run_inlp_all_steps(thoughts, labels, all_concepts):
    """
    Run INLP at each recurrence step to find concept-encoding directions.

    Returns:
        projections: dict t -> (D, D) nullspace projection
        rand_projections: dict t -> (D, D) random control projection
        inlp_stats: dict t -> stats
        concept_to_idx: dict concept -> int
    """
    concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
    y = np.array([concept_to_idx[l] for l in labels])

    N, T, D = thoughts.shape
    projections = {}
    rand_projections = {}
    inlp_stats = {}

    print(f"\n  Running INLP for {T} steps, {len(all_concepts)} concepts, {N} instances")

    for t in range(T):
        print(f"\n    Step {t}:")
        X = thoughts[:, t, :].numpy()

        P, n_iter, n_dirs, classifiers = run_inlp(X, y)
        projections[t] = P

        P_rand = compute_random_projection(D, n_dirs, seed=42 + t)
        rand_projections[t] = P_rand

        # Verify: probe accuracy after INLP
        X_clean = X @ P.T
        clf_verify = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf_verify.fit(X_clean, y)
        verify_acc = accuracy_score(y, clf_verify.predict(X_clean)) * 100

        clf_orig = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf_orig.fit(X, y)
        orig_acc = accuracy_score(y, clf_orig.predict(X)) * 100

        inlp_stats[t] = {
            "original_probe_acc": orig_acc,
            "post_inlp_probe_acc": verify_acc,
            "n_iterations": n_iter,
            "n_directions_removed": n_dirs,
        }

        print(f"      Original probe acc: {orig_acc:.1f}%")
        print(f"      Post-INLP probe acc: {verify_acc:.1f}%")

    return projections, rand_projections, inlp_stats, concept_to_idx


# ═══════════════════════════════════════════════════════════════════
# Steering vectors
# ═══════════════════════════════════════════════════════════════════

def compute_steering_vectors(thoughts, labels, target_concept, all_concepts):
    """
    Compute contrastive direction for steering toward target_concept.

    v_t = mean(h_t | correct = target) - mean(h_t | correct != target)

    Returns: dict t -> (D,) numpy unit vector, or None if target not in data.
    """
    N, T, D = thoughts.shape

    target_mask = np.array([l == target_concept for l in labels])
    n_target = target_mask.sum()

    if n_target == 0 or n_target == N:
        return None

    vectors = {}
    for t in range(T):
        X = thoughts[:, t, :].numpy()
        mean_target = X[target_mask].mean(axis=0)
        mean_other = X[~target_mask].mean(axis=0)
        v = mean_target - mean_other
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm
        vectors[t] = v

    return vectors


# ═══════════════════════════════════════════════════════════════════
# Verification checks
# ═══════════════════════════════════════════════════════════════════

def verify_temporal_structure(thoughts, projections, rand_projections):
    """
    Measure how much temporal structure survives after INLP and Rand projections.
    """
    N, T, D = thoughts.shape

    def variance_decomposition(X):
        mu = X.mean(dim=(0, 1))
        mu_t = X.mean(dim=0)
        mu_i = X.mean(dim=1)

        var_total = ((X - mu) ** 2).sum(dim=2).mean().item()
        var_timestep = ((mu_t - mu) ** 2).sum(dim=1).mean().item()
        var_instance = ((mu_i - mu) ** 2).sum(dim=1).mean().item()

        return var_total, var_timestep, var_instance

    vt_orig, vs_orig, vi_orig = variance_decomposition(thoughts)

    thoughts_inlp = torch.zeros_like(thoughts)
    for t in range(T):
        P = projections.get(t)
        if P is not None:
            P_tensor = torch.tensor(P, dtype=torch.float32)
            thoughts_inlp[:, t, :] = (thoughts[:, t, :] @ P_tensor.T)
        else:
            thoughts_inlp[:, t, :] = thoughts[:, t, :]
    vt_inlp, vs_inlp, vi_inlp = variance_decomposition(thoughts_inlp)

    thoughts_rand = torch.zeros_like(thoughts)
    for t in range(T):
        P = rand_projections.get(t)
        if P is not None:
            P_tensor = torch.tensor(P, dtype=torch.float32)
            thoughts_rand[:, t, :] = (thoughts[:, t, :] @ P_tensor.T)
        else:
            thoughts_rand[:, t, :] = thoughts[:, t, :]
    vt_rand, vs_rand, vi_rand = variance_decomposition(thoughts_rand)

    print(f"\n  {'='*60}")
    print(f"  TEMPORAL STRUCTURE VERIFICATION")
    print(f"  {'='*60}")
    print(f"  {'':>20}  {'Total Var':>12}  {'Timestep %':>12}  {'Instance %':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")

    ts_pct_orig = vs_orig / max(vt_orig, 1e-8) * 100
    ti_pct_orig = vi_orig / max(vt_orig, 1e-8) * 100
    print(f"  {'Original':>20}  {vt_orig:>12.1f}  {ts_pct_orig:>11.1f}%  {ti_pct_orig:>11.1f}%")

    ts_pct_inlp = vs_inlp / max(vt_inlp, 1e-8) * 100
    ti_pct_inlp = vi_inlp / max(vt_inlp, 1e-8) * 100
    print(f"  {'After INLP':>20}  {vt_inlp:>12.1f}  {ts_pct_inlp:>11.1f}%  {ti_pct_inlp:>11.1f}%")

    ts_pct_rand = vs_rand / max(vt_rand, 1e-8) * 100
    ti_pct_rand = vi_rand / max(vt_rand, 1e-8) * 100
    print(f"  {'After Rand':>20}  {vt_rand:>12.1f}  {ts_pct_rand:>11.1f}%  {ti_pct_rand:>11.1f}%")

    ts_retained_inlp = vs_inlp / max(vs_orig, 1e-8) * 100
    ts_retained_rand = vs_rand / max(vs_orig, 1e-8) * 100
    print(f"\n  Temporal variance retained:")
    print(f"    INLP: {ts_retained_inlp:.1f}% of original")
    print(f"    Rand: {ts_retained_rand:.1f}% of original")

    return {
        "original": {"total": vt_orig, "timestep_pct": ts_pct_orig, "instance_pct": ti_pct_orig},
        "inlp": {"total": vt_inlp, "timestep_pct": ts_pct_inlp, "instance_pct": ti_pct_inlp,
                  "temporal_retained_pct": ts_retained_inlp},
        "rand": {"total": vt_rand, "timestep_pct": ts_pct_rand, "instance_pct": ti_pct_rand,
                  "temporal_retained_pct": ts_retained_rand},
    }


def verify_steering_moves_representation(
    thoughts, labels, all_targets, steering_vectors_per_target,
    alpha, all_concepts, concept_to_idx, actual_flip_rates,
):
    """
    Check whether steering actually changes what the probe predicts,
    across ALL target concepts.
    """
    N, T, D = thoughts.shape
    y = np.array([concept_to_idx[l] for l in labels])

    probes_verify = {}
    for t in range(T):
        clf = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf.fit(thoughts[:, t, :].numpy(), y)
        probes_verify[t] = clf

    print(f"\n  {'='*60}")
    print(f"  STEERING VERIFICATION (does steering move the probe's prediction?)")
    print(f"  Alpha = {alpha}")
    print(f"  {'='*60}")

    total_probe_changed = 0
    total_probe_possible = 0

    for target in all_targets:
        vectors = steering_vectors_per_target.get(target)
        if vectors is None:
            continue

        target_changed = 0
        target_possible = 0

        for t in range(T):
            if t not in vectors:
                continue

            X_orig = thoughts[:, t, :].numpy()
            v = vectors[t]

            X_steered = X_orig + alpha * v.reshape(1, -1)

            pred_orig = probes_verify[t].predict(X_orig)
            pred_steered = probes_verify[t].predict(X_steered)

            n_changed = int((pred_orig != pred_steered).sum())
            target_changed += n_changed
            target_possible += len(pred_orig)

        total_probe_changed += target_changed
        total_probe_possible += target_possible

        probe_change_rate = target_changed / max(target_possible, 1)
        model_flip_rate = actual_flip_rates.get(target, 0.0)
        print(f"    Target '{target}': probe changed {probe_change_rate:.1%}, "
              f"model flipped {model_flip_rate:.1%}")

    overall_probe_rate = total_probe_changed / max(total_probe_possible, 1)
    overall_model_flip = np.mean([actual_flip_rates.get(t, 0.0) for t in all_targets
                                  if t in actual_flip_rates])

    print(f"\n  Overall: probe predictions changed {overall_probe_rate:.1%}, "
          f"model output flipped {overall_model_flip:.1%}")

    if overall_probe_rate > 0.1 and overall_model_flip < 0.02:
        print(f"  → Steering moves representations but NOT model output")
        print(f"  → INTERPRETATION A: model CANNOT use concept info from thought vectors")
    elif overall_probe_rate > 0.1 and overall_model_flip >= 0.02:
        print(f"  → Steering moves BOTH representations AND model output")
        print(f"  → Concept info IS causally connected to the output")
    else:
        print(f"  → Steering does NOT move representations sufficiently")
        print(f"  → Experiment is INCONCLUSIVE for this alpha")

    return overall_probe_rate, overall_model_flip


# ═══════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════

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
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Amnesic probing, random corruption, and steering on continuous thoughts."
    )
    parser.add_argument(
        "--stage", type=str,
        choices=["random_corruption", "ablation", "steering", "all"],
        default="all",
    )
    parser.add_argument(
        "--model", type=str,
        choices=["coconut", "coconut_u", "pause"],
        default="coconut",
    )
    parser.add_argument("--prosqa_path", type=str, default=None)
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--inlp_path", type=str, default=None,
        help="Path to saved INLP results to skip recomputing them in Stage 2.",
    )
    parser.add_argument(
        "--noise_sweep", type=str, default="0.1,0.5,1.0,5.0,10.0,25.0,50.0",
        help="Comma-separated noise scales for progressive corruption.",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--alpha_sweep", type=str, default=None,
        help="Comma-separated alpha values to sweep (e.g., '1,5,10,20,50').",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    prosqa_path = args.prosqa_path or str(PROSQA_TEST)
    output_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "outputs" / "steering" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and data ─────────────────────────────────────────
    coconut_model, base_model, tokenizer, start_id, end_id, latent_id = \
        setup_model_and_tokenizer(args.model, args.device)
    data = load_prosqa(prosqa_path, args.max_instances)
    print(f"[INFO] Model: {args.model}, instances: {len(data)}, device: {args.device}")

    run_random_corruption = args.stage in ("random_corruption", "all")
    run_ablation = args.stage in ("ablation", "all")
    run_steering = args.stage in ("steering", "all")

    if run_random_corruption or run_ablation or run_steering:
        print("\n" + "=" * 60)
        print("STAGE 0: THOUGHT LOADING & BASELINE")
        print("=" * 60)

        # 1. Resolve path to thoughts
        from utils.config import THOUGHTS
        thoughts_path = Path(THOUGHTS) / f"thoughts_{args.model}.pt"

        # 2. Call external script if thoughts don't exist
        if not thoughts_path.exists():
            print(f"  [INFO] Thoughts not found at {thoughts_path}. Triggering extraction script...")

            cmd = [
                "python", "-u", "-m", "experiments.probe_thoughts.extract_thoughts",
                "--model", args.model,
                "--n_thoughts", str(args.n_thoughts)
            ]
            if args.prosqa_path:
                cmd.extend(["--prosqa_path", args.prosqa_path])
            if args.max_instances:
                cmd.extend(["--max_instances", str(args.max_instances)])

            subprocess.run(cmd, check=True)
            print("  [INFO] Extraction complete.")

        # 3. Load the thoughts tensor
        print(f"  Loading thoughts from {thoughts_path}")
        extracted_data = torch.load(thoughts_path, map_location="cpu", weights_only=False)
        thoughts = extracted_data["thoughts"]

        # 4. Extract labels and concepts from the JSON (no model needed)
        labels, all_concepts = extract_labels_and_concepts(data)

        # 5. Compute shared baseline (no intervention)
        print("\n  Computing shared baseline (no intervention)...")
        baseline_results = []
        for idx, sample in enumerate(data):
            if idx % 100 == 0:
                print(f"    [Baseline] {idx}/{len(data)}")
            baseline_results.append(run_intervened_inference_pauseaware(
                coconut_model, base_model, tokenizer, end_id, sample,
                args.n_thoughts, args.device, make_identity_intervention(),
                start_id=start_id, latent_id=latent_id,
            ))

        n_correct = sum(1 for r in baseline_results if r["is_correct"])
        baseline_acc = n_correct / len(data)
        print(f"  → Shared Baseline Accuracy: {baseline_acc:.1%}")

    # ════════════════════════════════════════════════════════════════
    # STAGE 1: PROGRESSIVE RANDOM CORRUPTION
    # ════════════════════════════════════════════════════════════════
    if run_random_corruption:
        print("\n" + "=" * 60)
        print("STAGE 1: PROGRESSIVE RANDOM CORRUPTION")
        print("=" * 60)

        noise_scales = [float(n) for n in args.noise_sweep.split(",")]
        corruption_results = {"0.0": baseline_acc}

        for scale in noise_scales:
            print(f"\n  Testing Noise Scale: {scale}")
            noise_fn = make_noise_intervention(noise_scale=scale, device=args.device)

            acc = run_eval_with_intervention(
                coconut_model, base_model, tokenizer, end_id, data,
                args.n_thoughts, args.device, noise_fn,
                label=f"Noise {scale}",
                start_id=start_id, latent_id=latent_id,
            )

            corruption_results[str(scale)] = acc
            print(f"  → Accuracy at {scale} noise: {acc:.1%}")

        print(f"\n  {'='*50}")
        print(f"  CORRUPTION SUMMARY")
        print(f"  {'='*50}")
        print(f"  {'Noise Scale':>12}  {'Accuracy':>12}  {'Drop':>10}")
        for scale in sorted([float(k) for k in corruption_results.keys()]):
            acc = corruption_results[str(scale)]
            drop = baseline_acc - acc
            print(f"  {scale:>12.2f}  {acc:>12.1%}  {drop:>10.1%}")

        corruption_path = output_dir / "random_corruption_results.json"
        with open(corruption_path, "w") as f:
            json.dump(deep_convert(corruption_results), f, indent=2)
        print(f"  Saved to {corruption_path}")

    # ════════════════════════════════════════════════════════════════
    # STAGE 2: TARGETED CORRUPTIONS (INLP ABLATION)
    # ════════════════════════════════════════════════════════════════
    if run_ablation:
        print("\n" + "=" * 60)
        print("STAGE 2: TARGETED CORRUPTIONS (INLP ABLATION)")
        print("=" * 60)

        projections = None
        rand_projections = None
        concept_to_idx = None

        if args.inlp_path and Path(args.inlp_path).exists():
            print(f"  Loading precomputed INLP from {args.inlp_path}")
            inlp_data = torch.load(args.inlp_path, map_location="cpu", weights_only=False)
            projections = {int(k) if isinstance(k, str) else k: v for k, v in inlp_data["projections"].items()}
            rand_projections = {int(k) if isinstance(k, str) else k: v for k, v in inlp_data["rand_projections"].items()}
            concept_to_idx = inlp_data["concept_to_idx"]
        else:
            print("  Computing INLP nullspace projections...")
            projections, rand_projections, inlp_stats, concept_to_idx = \
                run_inlp_all_steps(thoughts, labels, all_concepts)

            inlp_save = {
                "projections": {t: P for t, P in projections.items()},
                "rand_projections": {t: P for t, P in rand_projections.items()},
                "inlp_stats": inlp_stats,
                "concept_to_idx": concept_to_idx,
                "all_concepts": all_concepts,
            }
            torch.save(inlp_save, output_dir / "inlp_results.pt")

        print("\n  Running INLP ablation (remove concept info)...")
        inlp_fn = make_projection_intervention(projections, args.device)
        inlp_acc = run_eval_with_intervention(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, inlp_fn, label="INLP",
            start_id=start_id, latent_id=latent_id,
        )

        print("  Running Rand control (random direction removal)...")
        rand_fn = make_projection_intervention(rand_projections, args.device)
        rand_acc = run_eval_with_intervention(
            coconut_model, base_model, tokenizer, end_id, data,
            args.n_thoughts, args.device, rand_fn, label="Rand",
            start_id=start_id, latent_id=latent_id,
        )

        ablation_results = {
            "baseline_accuracy": baseline_acc,
            "inlp_accuracy": inlp_acc,
            "rand_accuracy": rand_acc,
            "accuracy_drop_inlp": baseline_acc - inlp_acc,
            "accuracy_drop_rand": baseline_acc - rand_acc,
        }

        print(f"\n  {'='*50}")
        print(f"  ABLATION SUMMARY")
        print(f"  {'='*50}")
        print(f"  Baseline:     {baseline_acc:.1%}")
        print(f"  INLP:         {inlp_acc:.1%}  (drop: {baseline_acc - inlp_acc:.1%})")
        print(f"  Rand control: {rand_acc:.1%}  (drop: {baseline_acc - rand_acc:.1%})")

        ablation_results["temporal_verification"] = verify_temporal_structure(
            thoughts, projections, rand_projections
        )

        ablation_path = output_dir / "ablation_results.json"
        with open(ablation_path, "w") as f:
            json.dump(deep_convert(ablation_results), f, indent=2)
        print(f"  Saved to {ablation_path}")

    # ════════════════════════════════════════════════════════════════
    # STAGE 3: TARGETED INJECTIONS (STEERING)
    # ════════════════════════════════════════════════════════════════
    if run_steering:
        print("\n" + "=" * 60)
        print("STAGE 3: TARGETED INJECTIONS (STEERING)")
        print("=" * 60)

        alphas = [args.alpha]
        if args.alpha_sweep:
            alphas = [float(a) for a in args.alpha_sweep.split(",")]

        concept_counts = Counter(labels)
        top_concepts = [c for c, _ in concept_counts.most_common(5)]

        all_steering_results = {alpha: {} for alpha in alphas}

        for target in top_concepts:
            vectors = compute_steering_vectors(thoughts, labels, target, all_concepts)
            if vectors is None:
                continue

            target_data = [s for s, l in zip(data, labels) if l != target]
            if not target_data:
                continue

            print(f"    Running steered inference for target '{target}' across all alphas...")

            n_flipped_per_alpha = {alpha: 0 for alpha in alphas}

            for idx, sample in enumerate(target_data):
                if idx % 100 == 0:
                    print(f"      [Steering] {idx}/{len(target_data)}")

                steered_answers = run_alpha_sweep_inference_pauseaware(
                    coconut_model, base_model, tokenizer, end_id, sample,
                    args.n_thoughts, args.device, vectors, alphas,
                    start_id=start_id, latent_id=latent_id,
                )

                original_idx = data.index(sample)
                base_concept = _extract_concept(baseline_results[original_idx]["predicted"])

                for alpha in alphas:
                    steered_concept = _extract_concept(steered_answers[alpha])
                    if steered_concept == target and base_concept != target:
                        n_flipped_per_alpha[alpha] += 1

            for alpha in alphas:
                flip_rate = n_flipped_per_alpha[alpha] / max(len(target_data), 1)
                all_steering_results[alpha][target] = {
                    "n_flipped": n_flipped_per_alpha[alpha],
                    "n_total": len(target_data),
                    "flip_rate": flip_rate,
                }
                print(f"      Alpha {alpha}: {n_flipped_per_alpha[alpha]}/{len(target_data)} flipped ({flip_rate:.1%})")

        print(f"\n  {'='*50}")
        print(f"  STEERING SUMMARY")
        print(f"  {'='*50}")
        print(f"  {'Alpha':>8}  {'Target':>20}  {'Flip Rate':>12}")
        for alpha in alphas:
            for target, r in all_steering_results.get(alpha, {}).items():
                print(f"  {alpha:>8.1f}  {target:>20}  {r['flip_rate']:>12.1%}")

        steering_path = output_dir / "steering_results.json"
        with open(steering_path, "w") as f:
            json.dump(deep_convert(all_steering_results), f, indent=2)

        # Verification: does steering move the probe's prediction?
        if 'concept_to_idx' not in locals() or concept_to_idx is None:
            if args.inlp_path and Path(args.inlp_path).exists():
                print(f"  Loading concept mapping from {args.inlp_path} for verification...")
                inlp_data = torch.load(args.inlp_path, map_location="cpu", weights_only=False)
                concept_to_idx = inlp_data["concept_to_idx"]
            else:
                print("  Computing concept mapping for verification...")
                _, _, _, concept_to_idx = run_inlp_all_steps(thoughts, labels, all_concepts)

        max_alpha = max(alphas)
        steering_vectors_per_target = {
            t: compute_steering_vectors(thoughts, labels, t, all_concepts)
            for t in top_concepts
        }
        verify_steering_moves_representation(
            thoughts, labels, top_concepts, steering_vectors_per_target,
            max_alpha, all_concepts, concept_to_idx,
            {t: r["flip_rate"] for t, r in all_steering_results.get(max_alpha, {}).items()}
        )


if __name__ == "__main__":
    main()