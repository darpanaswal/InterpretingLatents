"""
Test 1: Positional-embedding ablation.

Falsifies: "the LDA temporal subspace is just PE leakage through the transformer."

Procedure:
    1. Load trained model (coconut / coconut_u / pause / codi / base / cot).
    2. Patch wpe (positional embedding table) so that at every thought
       position, the model sees one of four substitutions:
         - zero            : wpe(pos) = 0
         - constant        : wpe(pos) = wpe(first_thought_position)   (all thoughts share one PE)
         - random_gaussian : wpe(pos) = R[pos - cutoff], R drawn once ~ N(0, sigma^2 I)
                             with sigma^2 matched to the learned PE table's per-entry
                             variance. Breaks the learned identity of PE while keeping
                             per-position distinctness and magnitude.
         - random_shuffle  : wpe(pos) = wpe(cutoff + pi(pos - cutoff)), pi a fixed random
                             permutation. Every thought position gets a real learned PE
                             vector, but the wrong one. Breaks the position -> PE mapping
                             while preserving the marginal distribution of PE values.
    3. Re-extract thoughts on the same instances.
    4. Compare original vs ablated along:
         - variance decomposition     (does % timestep variance collapse?)
         - LDA cluster separation     (Fisher ratio: high = clean scaffold)
         - principal angles Q_orig vs Q_ablated  (did the subspace move?)
         - held-out timestep accuracy (does the signal survive cross-instance?)

Hypothesis controls (base / cot):
    base / cot are NOT recursion-trained. We force them to recurse at inference
    time by wrapping in Coconut and feeding last-layer hidden states back as
    input embeddings, exactly like coconut/coconut_u. This separates two
    sources of any temporal scaffold:

        - "recursion training" : scaffolding learned because the model was
          optimized end-to-end with hidden-state feedback in the loop.
        - "process of recursion": scaffolding that emerges merely from the
          act of feeding hidden states back through a transformer with
          per-position PE — no training on this regime needed.

    Predictions:
        - If scaffold appears in base + forced recursion, it is mechanistic
          (PE-driven feedback geometry), not training-acquired.
        - If scaffold appears only in coconut/coconut_u/pause/codi but not
          in base/cot, it is training-acquired.
        - cot vs base disambiguates further: token-level CoT training without
          continuous recurrence training. A scaffold in cot but not base
          implicates language-modelling-on-reasoning as the source.

Interpretation:
    Each mode tests a different property of PE:
        zero            : is PE needed at all?
        constant        : sanity check — shared additive shift should not move scaffold.
        random_gaussian : does the scaffold need the LEARNED PE, or just any
                          distinct-per-position vector of the right magnitude?
        random_shuffle  : does the scaffold need the specific p -> wpe(p) mapping,
                          or does any assignment of learned PE values to positions do?

    Under H0 (PE leakage):  scaffold collapses after ablation.
        - var_timestep % drops sharply; LDA separation drops; Q rotates;
          held-out accuracy drops to near 1/T.
    Under H1 (emergent scaffold): scaffold survives ablation.
        - metrics move only mildly; Q_orig ~ Q_ablated in principal angles.

Usage (single cell):
    python -m experiments.probe_thoughts.pe_ablation \
        --task gsm --model codi --mode zero
    python -m experiments.probe_thoughts.pe_ablation \
        --task gsm --model codi --mode random_gaussian --seed 0
    python -m experiments.probe_thoughts.pe_ablation \
        --task gsm --model codi --mode random_shuffle --seed 0
    python -m experiments.probe_thoughts.pe_ablation \
        --task prosqa --model codi --mode zero

    # Forced-recursion controls:
    python -m experiments.probe_thoughts.pe_ablation \
        --task prosqa --model base --mode zero
    python -m experiments.probe_thoughts.pe_ablation \
        --task prosqa --model cot --mode random_shuffle --seed 0

Random modes write reports named report_<mode>_seed<seed>.json and thought
tensors thoughts_ablated_<mode>_seed<seed>.pt, so multiple seeds can coexist
in the same directory.  Deterministic modes (zero, constant) ignore seed and
keep their historical filenames (report_<mode>.json, thoughts_ablated_<mode>.pt).

Run all cells by looping over {task, model, mode, seed}.
"""

import sys
import json
import types
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import THOUGHTS, PROSQA_TEST, GSM_TEST
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    is_pause_model,
)
from src.bootstrap_stats import (
       report_mean_with_ci,
       paired_bootstrap_diff, mcnemar_test,
       bootstrap_r2, bootstrap_variance_decomposition,
       save_record, save_per_instance_vector,
   )

# Reuse the exact extraction logic from extract_thoughts.py, no duplication
from experiments.extract_thoughts import (
    load_data,
    extract_thoughts_single_instance,
    extract_thoughts_codi_batch,
)
from deprecated_experiments.temporal_scaffold.scaffolding_metrics import (
    variance_decomposition,
    fit_lda_subspace,
    principal_angles,
    held_out_timestep_accuracy,
    lda_cluster_separation,
    split_instances,
)


# ═══════════════════════════════════════════════════════════════════
# WPE patcher
# ═══════════════════════════════════════════════════════════════════

# The WPE ablation has two challenges:
#
#   (a) Thought positions differ per instance: position index depends on the
#       question length. We cannot simply zero rows of wpe.weight because
#       rows are shared with non-thought positions of other instances.
#
#   (b) For coconut/CODI recurrence, position_ids are computed internally by
#       HuggingFace GPT-2 based on past_length. We cannot easily override
#       them externally.
#
# Solution: wrap wpe.forward. The wrapped module has a per-instance cutoff;
# for any position_id >= cutoff, return the mode-specific substitution.
# Non-thought positions are unaffected, and the cutoff is set fresh for every
# instance.
#
# For the random_* modes, the random table / permutation is drawn ONCE at
# wrapper construction time (seed-controlled) and reused across every
# instance. This matches the deterministic style of zero/constant: one fixed
# counterfactual PE behaviour, applied identically to every instance.


# Maximum number of thought positions we need to cover per instance. The
# wrapper pre-allocates random tables of this size, indexed by
# (position_id - cutoff). Must be >= (n_thoughts + downstream_decoded_tokens).
# n_thoughts <= 6 in practice; pause's end_latent adds 1; CODI's eot adds 1;
# a buffer of 16 is comfortable and costs nothing.
_MAX_THOUGHT_SPAN = 16


class WPEWrapper(torch.nn.Module):
    """
    Wraps GPT-2's wpe (nn.Embedding) so that position ids >= cutoff are
    replaced with a mode-specific substitution.

    cutoff is the absolute position of the FIRST thought token in the
    current instance. Must be set with set_cutoff(cutoff) before each
    forward pass and cleared afterwards.

    # For position p and cutoff c, define:
    #   zero            : wpe'(p) = 0                              if p >= c
    #   constant        : wpe'(p) = wpe(c)                         if p >= c
    #   random_gaussian : wpe'(p) = R[p - c]                       if p >= c
    #                     R[k] ~ N(0, sigma^2 I),  i.i.d. across k, drawn once
    #                     sigma^2 = Var( { wpe(p)_d : all p, d } )  (per-entry variance
    #                                                                of the learned table)
    #   random_shuffle  : wpe'(p) = wpe(c + pi(p - c))             if p >= c
    #                     pi in Sym({0, ..., M-1}),  drawn once,  M = _MAX_THOUGHT_SPAN
    # For p < c: wpe'(p) = wpe(p) in all modes.
    """

    def __init__(self, original_wpe, mode, seed=0):
        super().__init__()
        assert mode in (
            "zero", "constant", "random_gaussian", "random_shuffle", "off"
        )
        self.wpe = original_wpe
        self.mode = mode
        self.cutoff = None     # set per-instance

        # Random state is drawn once, here, with a CPU-side generator seeded
        # explicitly. We avoid touching torch.manual_seed / numpy global state.
        gen = torch.Generator(device="cpu").manual_seed(seed)

        if mode == "random_gaussian":
            # # sigma = sqrt( mean_{p,d}( (wpe(p)_d - mean(wpe))^2 ) )
            wpe_w = original_wpe.weight.detach().cpu()             # (P_max, D)
            sigma = wpe_w.std().item()                             # scalar, per-entry std
            D = wpe_w.shape[1]
            R = torch.randn(_MAX_THOUGHT_SPAN, D, generator=gen) * sigma
            # Buffer so .to(device) moves it alongside the module.
            self.register_buffer(
                "random_table", R.to(original_wpe.weight.dtype),
            )
            self.sigma = sigma
            self.shuffle_perm = None

        elif mode == "random_shuffle":
            # # pi : {0, ..., M-1} -> {0, ..., M-1}, uniform over the symmetric group
            perm = torch.randperm(_MAX_THOUGHT_SPAN, generator=gen)
            self.register_buffer("shuffle_perm", perm)
            self.random_table = None
            self.sigma = None

        else:
            self.random_table = None
            self.shuffle_perm = None
            self.sigma = None

        self.seed = seed

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff

    def clear(self):
        self.cutoff = None

    def forward(self, position_ids):
        # Unmodified output first; apply substitutions only at masked positions.
        out = self.wpe(position_ids)
        if self.cutoff is None or self.mode == "off":
            return out

        mask = (position_ids >= self.cutoff)
        if not mask.any():
            return out

        if self.mode == "zero":
            out = out.clone()
            out[mask] = 0.0
            return out

        if self.mode == "constant":
            ref = self.wpe.weight[self.cutoff].to(out.dtype)
            out = out.clone()
            out[mask] = ref
            return out

        # Both random modes need per-position offsets in [0, _MAX_THOUGHT_SPAN).
        # Clamp defensively: downstream decoded tokens can extend a little past
        # thought region, and we'd rather reuse the last random slot than crash.
        offsets = (position_ids - self.cutoff).clamp(
            min=0, max=_MAX_THOUGHT_SPAN - 1,
        )

        if self.mode == "random_gaussian":
            # # wpe'(p) = R[p - c]
            out = out.clone()
            out[mask] = self.random_table[offsets[mask]].to(out.dtype)
            return out

        if self.mode == "random_shuffle":
            # # wpe'(p) = wpe(c + pi(p - c))
            shuffled_positions = self.cutoff + self.shuffle_perm[offsets[mask]]
            out = out.clone()
            out[mask] = self.wpe.weight[shuffled_positions].to(out.dtype)
            return out

        # Unreachable given the assert in __init__.
        return out


def install_wpe_wrapper(base_model, mode, seed=0):
    """
    Replace base_model.transformer.wpe with a WPEWrapper.
    Returns the wrapper for per-instance cutoff setting.

    For GPT-2 (AutoModelForCausalLM), the path is:
        base_model.transformer.wpe
    For PEFT-wrapped CODI, unwrap first:
        base_model.get_base_model().transformer.wpe

    seed only affects the random_* modes; ignored by zero/constant.
    """
    transformer = _get_transformer(base_model)
    original_wpe = transformer.wpe
    wrapper = WPEWrapper(original_wpe, mode, seed=seed).to(
        next(base_model.parameters()).device
    )
    wrapper = wrapper.to(original_wpe.weight.dtype)
    transformer.wpe = wrapper
    return wrapper


def restore_wpe(base_model, wrapper):
    """Put back the original wpe module."""
    transformer = _get_transformer(base_model)
    transformer.wpe = wrapper.wpe


def _get_transformer(base_model):
    """Unwrap PEFT/LoRA wrappers to reach the GPT-2 transformer."""
    m = base_model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m.transformer


# ═══════════════════════════════════════════════════════════════════
# Per-instance thought positions
# ═══════════════════════════════════════════════════════════════════

def get_thought_cutoff(sample, tokenizer, mode, codi_dict=None):
    """
    Absolute position of the first thought token in a given instance.

    All thought positions (and downstream decoding positions) have
    position_id >= cutoff. The WPEWrapper uses this cutoff to decide
    which positions to ablate.

    Layouts (mirrored from extract_thoughts.py):

      Coconut / Coconut-u / base / cot:
          [question_tokens, start_latent, thought_1, ..., thought_K]
          cutoff = len(question_tokens) + 1   (first thought_i)

      Pause:
          [question_tokens + '\\n', start_latent, latent*K, end_latent]
          cutoff = len(question_tokens_with_nl) + 1

      CODI:
          [question_tokens, (eos?), bot, thought_1, ..., thought_K]
          cutoff = len(question_tokens) + (1 if remove_eos else 2) + 1
                 = first thought position

    The cutoff is the first thought position, so thought tokens AND any
    downstream decoding (end_latent / eot / answer) will all have
    ablated PE. That is intentional — the claim is specifically about
    what the model sees at and after the thought region.

    Note: base and cot use the same Coconut layout — they are wrapped in
    the Coconut class with feedback_mode='continuous' so that they can be
    forced to recurse at inference time, even though they were never
    trained on this regime.
    """
    q = sample["question"]
    if mode == "pause":
        q_tokens = tokenizer.encode(q + "\n", add_special_tokens=True)
        return len(q_tokens) + 1
    if mode in ("coconut", "coconut_u", "base", "cot"):
        q_tokens = tokenizer.encode(q, add_special_tokens=True)
        return len(q_tokens) + 1
    if mode == "codi":
        q_tokens = tokenizer.encode(q, add_special_tokens=True)
        n_prefix = len(q_tokens) + (1 if codi_dict["remove_eos"] else 2)
        return n_prefix + 1
    raise ValueError(mode)


# ═══════════════════════════════════════════════════════════════════
# Ablated-thought extraction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_ablated_thoughts(task, model_name, mode, n_thoughts,
                             max_instances, device, seed=0):
    """
    Load model, install WPEWrapper, iterate instances, collect thoughts
    with PE ablated at every thought position.

    mode in {"zero", "constant", "random_gaussian", "random_shuffle"}.
    seed controls the one-time random draw for random_* modes; ignored by
    zero/constant.

    Returns torch.Tensor (N, K+1, D).
    """
    is_codi = (model_name == "codi")

    if is_codi:
        codi_dict = setup_codi_model(task, device)
        base_model = codi_dict["model"]
        tokenizer = codi_dict["tokenizer"]
        hidden_dim = codi_dict["hidden_size"]
        start_id = latent_id = end_id = None
        coconut_model = None
    else:
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(task, model_name, device)
        codi_dict = None
        hidden_dim = base_model.config.n_embd

    wrapper = install_wpe_wrapper(base_model, mode=mode, seed=seed)

    try:
        data = load_data(task, max_instances)
        N = len(data)
        K = n_thoughts

        if is_codi:
            # Use batched extractor with batch_size=1 so that:
            #   (a) position_ids and attention_mask are correctly computed, and
            #   (b) on_batch_start_fn sets the per-instance WPE cutoff.
            # With batch_size=1 there is no left-padding, so
            # max(real_len) == L == the correct per-instance cutoff.
            all_thoughts = extract_thoughts_codi_batch(
                codi_dict, data, K, device,
                batch_size=1,
                on_batch_start_fn=lambda cutoff: wrapper.set_cutoff(cutoff),
                verbose=True,
            )
        else:
            all_thoughts = torch.zeros(N, K + 1, hidden_dim)

            for idx, sample in enumerate(data):
                if idx % 100 == 0:
                    print(f"  [ablation={mode} seed={seed}] {idx}/{N}")

                cutoff = get_thought_cutoff(
                    sample, tokenizer, model_name, codi_dict=codi_dict,
                )
                wrapper.set_cutoff(cutoff)

                thoughts = extract_thoughts_single_instance(
                    coconut_model, base_model, tokenizer, sample,
                    K, device, start_id, latent_id, end_id,
                )
                all_thoughts[idx] = thoughts

                wrapper.clear()
    finally:
        restore_wpe(base_model, wrapper)

    return all_thoughts


# ═══════════════════════════════════════════════════════════════════
# Metrics comparison
# ═══════════════════════════════════════════════════════════════════

def compare_scaffolds(thoughts_orig, thoughts_ablated, seed=0):
    """
    Run the full metric battery comparing original vs ablated thoughts.

    Hypothesis summary:
        PE-leakage null predicts: scaffold disappears under ablation.
        Emergent-scaffold predicts: scaffold survives.

    Metrics:
        1. Variance decomposition (original & ablated)
        2. LDA cluster separation  (original & ablated)
        3. Principal angles between Q_orig and Q_ablated (1.0 = same subspace)
        4. Held-out timestep classification accuracy using each subspace
    """
    result = {}

    # 1. Variance decomposition
    result["variance_original"] = variance_decomposition(thoughts_orig)
    result["variance_ablated"] = variance_decomposition(thoughts_ablated)

    # 2. LDA cluster separation
    result["lda_separation_original"] = lda_cluster_separation(thoughts_orig)
    result["lda_separation_ablated"] = lda_cluster_separation(thoughts_ablated)

    # 3. Principal angles: fit LDA on each in full, compare subspaces
    Q_orig, _ = fit_lda_subspace(thoughts_orig)
    Q_abl, _ = fit_lda_subspace(thoughts_ablated)
    cos_angles, rad_angles = principal_angles(Q_orig, Q_abl)
    result["principal_angles_cos"] = cos_angles.tolist()
    result["principal_angles_rad"] = rad_angles.tolist()
    result["principal_angles_mean_cos"] = float(cos_angles.mean())

    # 4. Held-out timestep accuracy (split on instances)
    tr_o, te_o, _, _ = split_instances(thoughts_orig, test_frac=0.3, seed=seed)
    tr_a, te_a, _, _ = split_instances(thoughts_ablated, test_frac=0.3, seed=seed)
    Q_o_tr, _ = fit_lda_subspace(tr_o)
    Q_a_tr, _ = fit_lda_subspace(tr_a)
    result["heldout_acc_original"] = held_out_timestep_accuracy(tr_o, te_o, Q_o_tr)
    result["heldout_acc_ablated"] = held_out_timestep_accuracy(tr_a, te_a, Q_a_tr)

    return result


def _fmt_var(v):
    return (f"timestep={v['pct_timestep']:.2f}%  "
            f"instance={v['pct_instance']:.2f}%  "
            f"residual={v['pct_residual']:.2f}%  "
            f"total={v['var_total']:.2f}")


def print_report(result, model_name, task, mode):
    line = "=" * 72
    print(f"\n{line}")
    print(f"PE ABLATION REPORT  —  task={task}  model={model_name}  mode={mode}")
    print(line)

    print(f"\n  Variance decomposition:")
    print(f"    original:  {_fmt_var(result['variance_original'])}")
    print(f"    ablated :  {_fmt_var(result['variance_ablated'])}")

    print(f"\n  LDA Fisher separation (higher = cleaner timestep clusters):")
    print(f"    original:  {result['lda_separation_original']:.3f}")
    print(f"    ablated :  {result['lda_separation_ablated']:.3f}")
    ratio = result['lda_separation_ablated'] / max(
        result['lda_separation_original'], 1e-12
    )
    print(f"    ablated / original = {ratio:.3f}")

    print(f"\n  Held-out timestep classification accuracy (chance = 1/T):")
    print(f"    original:  {result['heldout_acc_original']:.3f}")
    print(f"    ablated :  {result['heldout_acc_ablated']:.3f}")

    cos = np.asarray(result["principal_angles_cos"])
    print(f"\n  Principal angles between Q_orig and Q_ablated:")
    print(f"    mean cos(theta):  {cos.mean():.3f}")
    print(f"    min  cos(theta):  {cos.min():.3f}   (worst-aligned direction)")
    print(f"    all  cos(theta):  [{', '.join(f'{x:.3f}' for x in cos)}]")


# ═══════════════════════════════════════════════════════════════════
# Output naming
# ═══════════════════════════════════════════════════════════════════

def _is_random_mode(mode):
    return mode in ("random_gaussian", "random_shuffle")


def _thoughts_filename(mode, seed):
    """
    Deterministic modes keep their historical, seed-free filenames so existing
    downstream scripts (diagnose_thoughts.py) continue to find them. Random
    modes include the seed so multiple draws can coexist.
    """
    if _is_random_mode(mode):
        return f"thoughts_ablated_{mode}_seed{seed}.pt"
    return f"thoughts_ablated_{mode}.pt"


def _report_filename(mode, seed):
    if _is_random_mode(mode):
        return f"report_{mode}_seed{seed}.json"
    return f"report_{mode}.json"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], required=True)
    parser.add_argument("--model",
                        choices=["coconut", "coconut_u", "pause", "codi",
                                 "base", "cot"],
                        required=True,
                        help="Recursion-trained models: coconut, coconut_u, "
                             "pause, codi. Forced-recursion controls (no "
                             "recursion training): base, cot.")
    parser.add_argument(
        "--mode",
        choices=["zero", "constant", "random_gaussian", "random_shuffle"],
        required=True,
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the one-time random draw in random_* modes. "
             "Also seeds the held-out instance split. Ignored for "
             "zero/constant modes as far as ablation behaviour goes.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        THOUGHTS / args.task / args.model / "pe_ablation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load original thoughts (the unablated baseline) --------
    orig_path = THOUGHTS / args.task / f"thoughts_{args.model}.pt"
    if not orig_path.exists():
        raise FileNotFoundError(
            f"Original thoughts not found at {orig_path}. "
            f"Run extract_thoughts.py first."
        )
    orig_data = torch.load(orig_path, map_location="cpu", weights_only=False)
    thoughts_orig = orig_data["thoughts"]

    if args.max_instances is not None:
        thoughts_orig = thoughts_orig[:args.max_instances]
    print(f"[INFO] Loaded original thoughts: {tuple(thoughts_orig.shape)}")

    # -------- Re-extract thoughts under PE ablation --------
    print(f"[INFO] Extracting thoughts with PE ablation mode='{args.mode}' "
          f"seed={args.seed}")
    thoughts_ablated = extract_ablated_thoughts(
        task=args.task, model_name=args.model, mode=args.mode,
        n_thoughts=args.n_thoughts, max_instances=args.max_instances,
        device=args.device, seed=args.seed,
    )
    print(f"[INFO] Ablated thoughts: {tuple(thoughts_ablated.shape)}")

    # Save ablated thoughts (for downstream diagnose_thoughts.py if desired)
    ablated_path = out_dir / _thoughts_filename(args.mode, args.seed)
    torch.save({
        "thoughts": thoughts_ablated,
        "mode": args.mode,
        "seed": args.seed,
        "model": args.model,
        "task": args.task,
    }, ablated_path)
    print(f"[INFO] Saved ablated thoughts to {ablated_path}")

    # -------- Compare --------
    result = compare_scaffolds(
        thoughts_orig, thoughts_ablated, seed=args.seed,
    )
    print_report(result, args.model, args.task, args.mode)

    # Save JSON report
    report_path = out_dir / _report_filename(args.mode, args.seed)
    report = {
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "n_instances": int(thoughts_orig.shape[0]),
        "n_thoughts": int(thoughts_orig.shape[1]) - 1,
        "metrics": result,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to {report_path}")


if __name__ == "__main__":
    main()