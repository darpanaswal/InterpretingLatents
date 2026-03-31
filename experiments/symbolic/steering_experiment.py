"""
Amnesic Probing via INLP on Continuous Thought Vectors.

Implements the methodology from Elazar et al. (2021) "Amnesic Probing:
Behavioral Explanation with Amnesic Counterfactuals" adapted for
Coconut's continuous thought vectors on ProsQA.

Three stages:

Stage 1 (INLP): At each recurrence step t, iteratively train linear
    classifiers to predict the correct concept from h_t, project onto
    their nullspace, and repeat until no classifier beats majority.
    This produces a projection matrix P_t that removes ALL linearly
    decodable concept information from step-t thought vectors.

Stage 2 (ABLATION): Re-run inference with P_t applied to each thought
    vector before feeding it back. Measure accuracy drop.
    Controls:
        - Rand: project out the same number of random directions
        - Selectivity: concatenate concept label back after INLP and
          check if performance recovers

Stage 3 (STEERING): Add a contrastive direction (mean difference
    between thought vectors for target concept vs others) to steer
    the model toward a specific wrong answer.

Usage:
    python steering_experiment.py --stage all --model coconut
    python steering_experiment.py --stage all --model coconut_u
    python steering_experiment.py --stage all --model pause
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from contThought.coconut import Coconut
from sklearn.metrics import accuracy_score
from utils.utilities import clean_state_dict_keys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import (
    BASE_DIR, BASE_GPT2, COCONUT_GPT2, COCONUT_GPT2_U, PAUSE_GPT2,
    PROSQA_TEST, CONTROL_EXPT
)


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

    print(f"Loading checkpoint: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(raw_state_dict)
    coconut_model.load_state_dict(state_dict, strict=False)

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    base_model = coconut_model.base_causallm

    return base_model, tokenizer, start_id, end_id, latent_id


def load_prosqa(path, max_instances=None):
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


def format_prompt(sample, tokenizer):
    return tokenizer.encode(sample["question"] + " <|start-latent|>", return_tensors="pt")


# ═══════════════════════════════════════════════════════════════════
# ProsQA utilities
# ═══════════════════════════════════════════════════════════════════

def get_candidate_info(sample):
    """Extract correct concept and all concepts from a ProsQA instance."""
    answer_text = sample["answer"].strip().rstrip(".")
    correct_concept = answer_text.split(" is a ")[-1].strip()

    all_concepts = set()
    all_concepts.add(correct_concept)
    for step in sample.get("steps", []):
        parts = step.split(" is a ")
        if len(parts) == 2:
            concept_a = parts[0].replace("Every ", "").strip()
            concept_b = parts[1].strip().rstrip(".")
            all_concepts.add(concept_a)
            all_concepts.add(concept_b)

    return correct_concept, sorted(all_concepts)


def _extract_concept(answer_text):
    """Extract concept from an answer like 'X is a <concept>.'"""
    answer_text = answer_text.strip().rstrip(".")
    if " is a " in answer_text:
        return answer_text.split(" is a ")[-1].strip()
    return answer_text


# ═══════════════════════════════════════════════════════════════════
# Thought extraction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_thoughts_with_labels(base_model, tokenizer, data, n_thoughts, device):
    """
    Extract thought vectors and correct concept labels.

    Returns:
        thoughts: (N, T, D) tensor
        labels: list of N correct concept strings
        all_concepts: sorted list of all unique concepts
    """
    N = len(data)
    D = base_model.config.n_embd
    T = n_thoughts + 1
    thoughts = torch.zeros(N, T, D)
    labels = []
    all_concepts = set()

    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"  [Extract] {idx}/{N}")

        correct_concept, concepts = get_candidate_info(sample)
        labels.append(correct_concept)
        all_concepts.update(concepts)

        input_ids = format_prompt(sample, tokenizer).to(device)

        outputs = base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, -1, :]
        thoughts[idx, 0] = h.cpu()
        past_kv = outputs.past_key_values
        ct = h.unsqueeze(0).unsqueeze(0)

        for t in range(1, T):
            outputs = base_model(
                inputs_embeds=ct,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
            )
            h = outputs.hidden_states[-1][0, 0, :]
            thoughts[idx, t] = h.cpu()
            ct = h.unsqueeze(0).unsqueeze(0)
            past_kv = outputs.past_key_values

    return thoughts, labels, sorted(all_concepts)


# ═══════════════════════════════════════════════════════════════════
# INLP (Iterative Nullspace Projection)
# ═══════════════════════════════════════════════════════════════════

def compute_nullspace_projection(W):
    """
    Compute the projection matrix onto the nullspace of W.

    Given classifier weight matrix W of shape (n_classes, D),
    the rowspace of W spans the directions used for classification.
    The nullspace projection removes these directions:

        P = I - W^T (W W^T)^{-1} W

    This guarantees W @ P @ h = 0 for any h, i.e., the classifier
    can no longer extract information from the projected representation.
    """
    # W: (n_classes, D) or (1, D) for binary
    if W.ndim == 1:
        W = W.reshape(1, -1)

    # P = I - W^T (W W^T)^{-1} W
    # Using SVD for numerical stability
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # The rows of Vt corresponding to nonzero singular values span the rowspace
    # Project onto rowspace, then subtract from identity
    # Rowspace basis = Vt[:rank, :]
    rank = np.sum(S > 1e-10)
    basis = Vt[:rank, :]  # (rank, D)
    # P_rowspace = basis^T @ basis
    # P_nullspace = I - P_rowspace
    P = np.eye(W.shape[1]) - basis.T @ basis
    return P


def run_inlp(X, y, max_iter=300, convergence_threshold=1.0):
    """
    Iterative Nullspace Projection (Ravfogel et al., 2020).

    Iteratively:
        1. Train a linear classifier to predict y from X
        2. If accuracy <= majority + convergence_threshold: stop
        3. Project X onto the nullspace of the classifier's weight matrix
        4. Accumulate the projection into a single matrix P_total

    X: (N, D) numpy array of representations
    y: (N,) numpy array of integer labels

    Returns:
        P_total: (D, D) numpy array, the accumulated nullspace projection
        n_iterations: number of INLP iterations performed
        n_directions_removed: total number of directions removed
        classifiers: list of trained classifiers (for analysis)
    """
    N, D = X.shape
    n_classes = len(np.unique(y))
    majority_acc = np.max(np.bincount(y)) / N * 100

    # Start with identity projection
    P_total = np.eye(D)
    X_projected = X.copy()
    classifiers = []
    n_directions_removed = 0

    for iteration in range(max_iter):
        # Train linear SVM (following Ravfogel et al.)
        clf = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf.fit(X_projected, y)
        y_pred = clf.predict(X_projected)
        acc = accuracy_score(y, y_pred) * 100

        # Check convergence: stop when accuracy is within threshold of majority
        if acc <= majority_acc + convergence_threshold:
            break

        classifiers.append(clf)

        # Compute nullspace projection for this classifier's weights
        # clf.coef_: (n_classes, D) for multiclass, (1, D) for binary
        W = clf.coef_  # (n_classes_or_1, D)
        P_i = compute_nullspace_projection(W)

        # Accumulate: P_total = P_i @ P_total
        P_total = P_i @ P_total

        # Project data for next iteration
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
    """
    Rand control: remove the same number of random directions.

    Generate random orthonormal directions and project them out.
    This controls for rank reduction damaging the representation
    generically vs. removing concept-specific information.
    """
    rng = np.random.RandomState(seed)
    # Generate random directions
    random_dirs = rng.randn(n_directions_removed, D)
    # Orthogonalize via QR decomposition
    Q, _ = np.linalg.qr(random_dirs.T)
    # Q: (D, n_directions_removed), columns are orthonormal
    # Take only as many as we removed
    n_actual = min(n_directions_removed, D)
    basis = Q[:, :n_actual].T  # (n_actual, D)
    P_rand = np.eye(D) - basis.T @ basis
    return P_rand


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Run INLP at each timestep
# ═══════════════════════════════════════════════════════════════════

def run_inlp_all_steps(thoughts, labels, all_concepts):
    """
    Run INLP at each recurrence step to find concept-encoding directions.

    Returns:
        projections: dict t -> (D, D) numpy projection matrix
        rand_projections: dict t -> (D, D) random control projection
        inlp_stats: dict t -> {n_iterations, n_directions, final_probe_acc, ...}
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
        X = thoughts[:, t, :].numpy()  # (N, D)

        # Run INLP
        P, n_iter, n_dirs, classifiers = run_inlp(X, y)
        projections[t] = P

        # Rand control: same number of random directions removed
        P_rand = compute_random_projection(D, n_dirs, seed=42 + t)
        rand_projections[t] = P_rand

        # Verify: probe accuracy after INLP should be near majority
        X_clean = X @ P.T
        clf_verify = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf_verify.fit(X_clean, y)
        verify_acc = accuracy_score(y, clf_verify.predict(X_clean)) * 100

        # Also check original probe accuracy (before INLP)
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
# Stage 2 & 3: Intervened inference
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_intervened_inference(
    base_model, tokenizer, end_id, sample, n_thoughts, device,
    intervention_fn,
):
    """
    Run inference with an intervention applied at every recurrence step.

    intervention_fn: callable (h, t) -> h_modified
        Takes the thought vector h (D,) on device and step index t,
        returns the modified vector.
    """
    input_ids = format_prompt(sample, tokenizer).to(device)

    outputs = base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]
    past_kv = outputs.past_key_values

    # Intervene at step 0
    h = intervention_fn(h, 0)
    ct = h.unsqueeze(0).unsqueeze(0)

    for t in range(1, n_thoughts + 1):
        outputs = base_model(
            inputs_embeds=ct,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = outputs.hidden_states[-1][0, 0, :]

        # Intervene at step t
        h = intervention_fn(h, t)
        ct = h.unsqueeze(0).unsqueeze(0)
        past_kv = outputs.past_key_values

    # Decode answer
    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(input_ids=end_input, past_key_values=past_kv, use_cache=True)
    past_kv = outputs.past_key_values

    generated = []
    next_logits = outputs.logits[0, -1, :]
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
    answer = text.split("#")[-1].replace(",", "").strip()
    correct_answer = sample.get("answer", "").replace(",", "").strip()

    return {
        "predicted": answer,
        "correct": correct_answer,
        "is_correct": answer == correct_answer,
    }


def make_projection_intervention(projections, device):
    """Create an intervention function that applies INLP projection at each step."""
    # Pre-convert to tensors
    P_tensors = {t: torch.tensor(P, dtype=torch.float32, device=device)
                 for t, P in projections.items()}

    def intervention_fn(h, t):
        if t in P_tensors:
            # h_clean = P @ h
            return P_tensors[t] @ h
        return h

    return intervention_fn


def make_identity_intervention():
    """No intervention (baseline)."""
    return lambda h, t: h


def make_steering_intervention(steering_vectors, alpha, device):
    """
    Create an intervention that adds a contrastive steering vector at each step.

    steering_vectors: dict t -> (D,) numpy array
    """
    v_tensors = {t: torch.tensor(v, dtype=torch.float32, device=device)
                 for t, v in steering_vectors.items()}

    def intervention_fn(h, t):
        if t in v_tensors:
            # h_steered = h + alpha * v_hat
            return h + alpha * v_tensors[t]
        return h

    return intervention_fn


def run_eval_with_intervention(
    base_model, tokenizer, end_id, data, n_thoughts, device,
    intervention_fn, label="",
):
    """Run inference on all instances with a given intervention, return accuracy."""
    n_correct = 0
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"    [{label}] {idx}/{len(data)}")
        r = run_intervened_inference(
            base_model, tokenizer, end_id, sample, n_thoughts, device,
            intervention_fn,
        )
        if r["is_correct"]:
            n_correct += 1

    accuracy = n_correct / len(data)
    print(f"    [{label}] Accuracy: {n_correct}/{len(data)} = {accuracy:.1%}")
    return accuracy


# ═══════════════════════════════════════════════════════════════════
# Verification checks
# ═══════════════════════════════════════════════════════════════════

def verify_temporal_structure(thoughts, projections, rand_projections):
    """
    Measure how much temporal structure survives after INLP and Rand projections.

    Computes the variance decomposition (timestep vs instance) on:
        - Original thought vectors
        - INLP-projected vectors
        - Rand-projected vectors

    If INLP preserves temporal structure (high timestep variance fraction)
    while removing concept info, the 0.4% drop is genuinely from concept
    removal. If INLP destroys temporal structure, the drop is from
    temporal damage.
    """
    N, T, D = thoughts.shape

    def variance_decomposition(X):
        """
        X: (N, T, D) tensor

        Var_total = E[||h_{i,t} - mu||^2]
        Var_timestep = E_t[||mu_t - mu||^2]   (between-step variance)
        Var_instance = E_i[||mu_i - mu||^2]    (between-instance variance)
        """
        mu = X.mean(dim=(0, 1))
        mu_t = X.mean(dim=0)       # (T, D)
        mu_i = X.mean(dim=1)       # (N, D)

        var_total = ((X - mu) ** 2).sum(dim=2).mean().item()
        var_timestep = ((mu_t - mu) ** 2).sum(dim=1).mean().item()
        var_instance = ((mu_i - mu) ** 2).sum(dim=1).mean().item()

        return var_total, var_timestep, var_instance

    # Original
    vt_orig, vs_orig, vi_orig = variance_decomposition(thoughts)

    # INLP-projected
    thoughts_inlp = torch.zeros_like(thoughts)
    for t in range(T):
        P = projections.get(t)
        if P is not None:
            P_tensor = torch.tensor(P, dtype=torch.float32)
            # h_projected = P @ h  for each instance
            thoughts_inlp[:, t, :] = (thoughts[:, t, :] @ P_tensor.T)
        else:
            thoughts_inlp[:, t, :] = thoughts[:, t, :]
    vt_inlp, vs_inlp, vi_inlp = variance_decomposition(thoughts_inlp)

    # Rand-projected
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

    # How much temporal variance was destroyed?
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
    thoughts, labels, steering_vectors, alpha, all_concepts, concept_to_idx,
):
    """
    Check whether steering actually changes what the probe predicts.

    For each instance where correct != target:
        1. Take original h_t
        2. Apply steering: h_steered = h_t + alpha * v_t
        3. Run the probe on both h_t and h_steered
        4. Check if the probe's prediction changes to the target concept

    If the probe predicts the target after steering but the model's output
    doesn't change, that confirms interpretation A: the model CAN'T use
    the concept info from thought vectors even when it's injected.
    """
    N, T, D = thoughts.shape
    y = np.array([concept_to_idx[l] for l in labels])

    # Train fresh probes on original thoughts (for verification only)
    from sklearn.svm import LinearSVC
    probes_verify = {}
    for t in range(T):
        clf = LinearSVC(max_iter=5000, dual="auto", C=0.1)
        clf.fit(thoughts[:, t, :].numpy(), y)
        probes_verify[t] = clf

    # For each target concept in the steering vectors dict
    target_concept = list(steering_vectors.keys())[0] if isinstance(
        list(steering_vectors.keys())[0], str) else None

    # steering_vectors is dict t -> (D,) numpy
    # Apply steering to all instances and check probe predictions
    results_per_step = {}
    for t in range(T):
        if t not in steering_vectors:
            continue

        X_orig = thoughts[:, t, :].numpy()
        v = steering_vectors[t]

        # Apply steering
        # h_steered = h + alpha * v
        X_steered = X_orig + alpha * v.reshape(1, -1)

        # Probe predictions before and after
        pred_orig = probes_verify[t].predict(X_orig)
        pred_steered = probes_verify[t].predict(X_steered)

        # How many predictions changed?
        n_changed = (pred_orig != pred_steered).sum()
        n_total = len(pred_orig)

        results_per_step[t] = {
            "n_changed": int(n_changed),
            "n_total": n_total,
            "change_rate": n_changed / n_total,
        }

    print(f"\n  {'='*60}")
    print(f"  STEERING VERIFICATION (does steering move the probe's prediction?)")
    print(f"  Alpha = {alpha}")
    print(f"  {'='*60}")
    print(f"  {'Step':>6}  {'Probe predictions changed':>28}")
    for t in sorted(results_per_step.keys()):
        r = results_per_step[t]
        print(f"  {t:>6}  {r['n_changed']:>6}/{r['n_total']} ({r['change_rate']:.1%})")

    total_changed = sum(r["n_changed"] for r in results_per_step.values())
    total_possible = sum(r["n_total"] for r in results_per_step.values())
    overall_rate = total_changed / max(total_possible, 1)

    if overall_rate > 0.1:
        print(f"\n  Steering DOES move representations in probe-space ({overall_rate:.1%} changed)")
        print(f"  → But model output didn't change (0% flip rate)")
        print(f"  → INTERPRETATION A: model CANNOT use concept info from thought vectors")
    else:
        print(f"\n  Steering does NOT move representations in probe-space ({overall_rate:.1%} changed)")
        print(f"  → Steering vector may be too weak or in wrong direction")
        print(f"  → Experiment is INCONCLUSIVE for this alpha")

    return results_per_step, overall_rate

def compute_steering_vectors(thoughts, labels, target_concept, all_concepts):
    """
    Compute contrastive direction for steering toward target_concept.

    At each step t, the steering vector is:
        v_t = mean(h_t | correct = target) - mean(h_t | correct != target)

    This is the direction in representation space that distinguishes
    instances where target_concept is correct from all others.

    Returns: dict t -> (D,) numpy unit vector
    """
    N, T, D = thoughts.shape

    target_mask = np.array([l == target_concept for l in labels])
    n_target = target_mask.sum()

    if n_target == 0 or n_target == N:
        return None

    vectors = {}
    for t in range(T):
        X = thoughts[:, t, :].numpy()
        # mean(h | correct = target) - mean(h | correct != target)
        mean_target = X[target_mask].mean(axis=0)
        mean_other = X[~target_mask].mean(axis=0)
        v = mean_target - mean_other
        # Normalize to unit vector
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm
        vectors[t] = v

    return vectors


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
        description="Amnesic probing (INLP) and steering on continuous thought vectors."
    )
    parser.add_argument(
        "--stage", type=str,
        choices=["inlp", "ablation", "steering", "all"],
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
        help="Path to saved INLP results (for ablation/steering without re-running INLP).",
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Steering strength (default: 10.0).",
    )
    parser.add_argument(
        "--alpha_sweep", type=str, default=None,
        help="Comma-separated alpha values to sweep (e.g., '1,5,10,20,50').",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
    )
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
    base_model, tokenizer, start_id, end_id, latent_id = \
        setup_model_and_tokenizer(args.model, args.device)
    data = load_prosqa(prosqa_path, args.max_instances)
    print(f"[INFO] Model: {args.model}, instances: {len(data)}, device: {args.device}")

    run_inlp_stage = args.stage in ("inlp", "all")
    run_ablation = args.stage in ("ablation", "all")
    run_steering = args.stage in ("steering", "all")

    projections = None
    rand_projections = None
    thoughts = None
    labels = None
    all_concepts = None
    concept_to_idx = None

    # ── Stage 1: INLP ──────────────────────────────────────────────
    if run_inlp_stage or run_ablation or run_steering:
        # Always need thoughts for INLP or steering vectors
        print("\n" + "=" * 60)
        print("EXTRACTING THOUGHT VECTORS")
        print("=" * 60)
        thoughts, labels, all_concepts = extract_thoughts_with_labels(
            base_model, tokenizer, data, args.n_thoughts, args.device
        )

    if run_inlp_stage:
        print("\n" + "=" * 60)
        print("STAGE 1: INLP (Iterative Nullspace Projection)")
        print("=" * 60)

        projections, rand_projections, inlp_stats, concept_to_idx = \
            run_inlp_all_steps(thoughts, labels, all_concepts)

        # Save INLP results
        inlp_save = {
            "projections": {t: P for t, P in projections.items()},
            "rand_projections": {t: P for t, P in rand_projections.items()},
            "inlp_stats": inlp_stats,
            "concept_to_idx": concept_to_idx,
            "all_concepts": all_concepts,
            "n_thoughts": args.n_thoughts,
            "model": args.model,
        }
        inlp_path = output_dir / "inlp_results.pt"
        torch.save(inlp_save, inlp_path)
        print(f"\n  INLP results saved to {inlp_path}")

        # Print summary
        print(f"\n  INLP Summary:")
        print(f"  {'Step':>6}  {'Orig Probe':>12}  {'Post-INLP':>12}  {'Dirs Removed':>14}")
        for t in sorted(inlp_stats.keys()):
            s = inlp_stats[t]
            print(f"  {t:>6}  {s['original_probe_acc']:>11.1f}%  "
                  f"{s['post_inlp_probe_acc']:>11.1f}%  "
                  f"{s['n_directions_removed']:>14}")

    # ── Load INLP if not computed ───────────────────────────────────
    if (run_ablation or run_steering) and projections is None:
        inlp_path = args.inlp_path or str(output_dir / "inlp_results.pt")
        print(f"\n  Loading INLP from {inlp_path}")
        inlp_data = torch.load(inlp_path, map_location="cpu", weights_only=False)
        projections = {int(k) if isinstance(k, str) else k: v
                       for k, v in inlp_data["projections"].items()}
        rand_projections = {int(k) if isinstance(k, str) else k: v
                            for k, v in inlp_data["rand_projections"].items()}
        concept_to_idx = inlp_data["concept_to_idx"]
        all_concepts = inlp_data["all_concepts"]

    # ── Stage 2: Ablation ───────────────────────────────────────────
    if run_ablation:
        print("\n" + "=" * 60)
        print("STAGE 2: ABLATION (Amnesic Probing)")
        print("=" * 60)

        # Baseline (no intervention)
        print("\n  Running baseline (no intervention)...")
        baseline_acc = run_eval_with_intervention(
            base_model, tokenizer, end_id, data, args.n_thoughts, args.device,
            make_identity_intervention(), label="Baseline",
        )

        # INLP ablation (remove concept information)
        print("\n  Running INLP ablation (remove concept info)...")
        inlp_fn = make_projection_intervention(projections, args.device)
        inlp_acc = run_eval_with_intervention(
            base_model, tokenizer, end_id, data, args.n_thoughts, args.device,
            inlp_fn, label="INLP",
        )

        # Rand control (remove same number of random directions)
        print("\n  Running Rand control (random direction removal)...")
        rand_fn = make_projection_intervention(rand_projections, args.device)
        rand_acc = run_eval_with_intervention(
            base_model, tokenizer, end_id, data, args.n_thoughts, args.device,
            rand_fn, label="Rand",
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

        if baseline_acc - inlp_acc > baseline_acc - rand_acc + 0.02:
            print(f"  → INLP drop > Rand drop: concept info IS causally used")
        else:
            print(f"  → INLP drop ≤ Rand drop: NO evidence concept info is causally used")

        ablation_path = output_dir / "ablation_results.json"
        with open(ablation_path, "w") as f:
            json.dump(deep_convert(ablation_results), f, indent=2)
        print(f"  Saved to {ablation_path}")

        # Verification: how much temporal structure survives each projection?
        temporal_check = verify_temporal_structure(
            thoughts, projections, rand_projections
        )
        ablation_results["temporal_verification"] = temporal_check
        # Re-save with verification data
        with open(ablation_path, "w") as f:
            json.dump(deep_convert(ablation_results), f, indent=2)

    # ── Stage 3: Steering ───────────────────────────────────────────
    if run_steering:
        print("\n" + "=" * 60)
        print("STAGE 3: STEERING (contrastive direction injection)")
        print("=" * 60)

        alphas = [args.alpha]
        if args.alpha_sweep:
            alphas = [float(a) for a in args.alpha_sweep.split(",")]

        # Pick target concepts for steering (use the most common wrong concepts)
        from collections import Counter
        concept_counts = Counter(labels)
        # Pick top-3 most common concepts as steering targets
        top_concepts = [c for c, _ in concept_counts.most_common(5)]

        all_steering_results = {}

        for alpha in alphas:
            print(f"\n  Alpha = {alpha}")
            alpha_results = {}

            for target in top_concepts:
                # Compute contrastive steering vector for this target
                vectors = compute_steering_vectors(
                    thoughts, labels, target, all_concepts
                )
                if vectors is None:
                    continue

                steer_fn = make_steering_intervention(vectors, alpha, args.device)

                # Run on instances where target is NOT the correct answer
                target_data = [s for s, l in zip(data, labels) if l != target]
                if not target_data:
                    continue

                n_flipped = 0
                n_total = len(target_data)

                for idx, sample in enumerate(target_data):
                    # Baseline
                    r_base = run_intervened_inference(
                        base_model, tokenizer, end_id, sample,
                        args.n_thoughts, args.device,
                        make_identity_intervention(),
                    )
                    # Steered
                    r_steer = run_intervened_inference(
                        base_model, tokenizer, end_id, sample,
                        args.n_thoughts, args.device,
                        steer_fn,
                    )

                    steered_concept = _extract_concept(r_steer["predicted"])
                    base_concept = _extract_concept(r_base["predicted"])

                    if steered_concept == target and base_concept != target:
                        n_flipped += 1

                flip_rate = n_flipped / max(n_total, 1)
                print(f"    Target '{target}': {n_flipped}/{n_total} flipped ({flip_rate:.1%})")

                alpha_results[target] = {
                    "n_flipped": n_flipped,
                    "n_total": n_total,
                    "flip_rate": flip_rate,
                }

            all_steering_results[alpha] = alpha_results

        # Print summary
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
        print(f"  Saved to {steering_path}")

        # Verification: does steering actually move the probe's predictions?
        # Use the first target and largest alpha for maximum signal
        first_target = top_concepts[0]
        max_alpha = max(alphas)
        vectors = compute_steering_vectors(thoughts, labels, first_target, all_concepts)
        if vectors is not None:
            verify_steering_moves_representation(
                thoughts, labels, vectors, max_alpha,
                all_concepts, concept_to_idx,
            )


if __name__ == "__main__":
    main()