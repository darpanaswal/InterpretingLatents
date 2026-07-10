"""
Predicted-token gradient-subspace extraction + gold/pred subspace comparison.

WHY THIS SCRIPT EXISTS
----------------------
gradient_subspace.py builds B_t from the gradient of the GOLD-answer NLL:
    g_{i,t} = d/dh_{i,t} [ -sum_j log p(y*_{i,j} | ...) ]   (y* = gold)
A reviewer objection: because B_t is defined by the gold label, "amplifying
B_t flips the output" could be a restatement of how B_t was built (it points
toward the answer we inserted), not a fact about the model's own computation.

This script recomputes the subspace from the model's OWN predicted tokens:
    g_{i,t} = d/dh_{i,t} [ -sum_j log p(yhat_{i,j} | ...) ]  (yhat = argmax)
i.e. the target is the greedy-decoded answer the model actually produces, with
NO reference to the gold label. Everything else -- the recurrence, the leaf
trick, the partial-gradient convention, the per-t SVD, the rank rule -- is
identical to gradient_subspace.py.

It then loads the existing gold bases.npz and reports, per timestep t, the
principal-angle overlap between span(B_t^pred) and span(B_t^gold):

    # M = (B_t^pred)^T B_t^gold          (k_pred, k_gold)
    # sigma = singular values of M       in [0, 1]
    # mean cos^2 = mean(sigma^2)         subspace-alignment scalar in [0,1]

mean cos^2 -> 1 : the two subspaces coincide  => B_t is NOT a gold-label
                  artifact; the loss-sensitive directions are intrinsic to
                  the model's own computation.
mean cos^2 -> 0 : the subspaces are orthogonal => the gold-defined B_t is
                  label-specific and the objection stands.

This reuses the SAME principal-angle quantity as the Figure-5 stability metric
mean(sigma(B_t^T B_{t+1})^2), just across the (pred, gold) basis pair at fixed
t instead of the (t, t+1) pair.

OUTPUT LAYOUT
-------------
    Predicted-token artefacts go in their OWN tree, separate from the gold
    gradient_geometry/ outputs:

    BASE_DIR / outputs / gradient_geometry_predtoken / <family>/<task>/<model>/
        bases.npz              -- {f"B_t{t}": B_t^pred}
        gradients.npz
        predtoken_comparison.json  -- per-t rank_pred, rank_gold, mean cos^2,
                                      and the full principal-angle spectrum

    The gold bases it compares against are READ from the original tree,
    BASE_DIR / outputs / gradient_geometry / <family>/<task>/<model>/bases.npz
    (override with --gold_bases).

USAGE
-----
    python -u -m experiments.ablation.gradient_subspace_predtoken \
        --task prosqa --model pause
    python -u -m experiments.ablation.gradient_subspace_predtoken \
        --task prosqa --model coconut
    python -u -m experiments.ablation.gradient_subspace_predtoken \
        --task gsm --model codi
    # add --model_family llama for the Llama checkpoints.

    # For a quick look, 150-200 instances gives a stable subspace:
    python -u -m experiments.ablation.gradient_subspace_predtoken \
        --task prosqa --model coconut --max_instances 200
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path

from src.config import BASE_DIR, PROSQA_TEST, GSM_TEST, set_seed
from src.utils import (
    setup_model_and_tokenizer,
    setup_codi_model,
    is_pause_model,
    tokenize_question_for_recurrence,
)
# Reuse the exact SVD/rank machinery and JSON helper from the gold script so
# the two subspaces are built with identical numerics.
from experiments.ablation.gradient_subspace import (
    subspace_from_gradients,
    deep_convert,
)


# ═══════════════════════════════════════════════════════════════════
# Data loading (mirrors gradient_subspace.load_data)
# ═══════════════════════════════════════════════════════════════════

def load_data(task, max_instances=None):
    path = PROSQA_TEST if task == "prosqa" else GSM_TEST
    with open(path) as f:
        data = json.load(f)
    if max_instances:
        data = data[:max_instances]
    return data


# ═══════════════════════════════════════════════════════════════════
# Greedy predicted-token target
# ═══════════════════════════════════════════════════════════════════
#
# We need the model's own answer tokens yhat_{i,0..M-1}. We decode them
# greedily at the answer boundary and use them AS the NLL target. The NLL
# of an argmax-decoded sequence is well-defined and differentiable w.r.t.
# the thought leaves exactly as the gold NLL is:
#     # nll = - sum_j log p(yhat_j | prompt, thoughts, yhat_{<j})
# Since yhat_j = argmax p(. | ...), each -log p(yhat_j|...) is the minimum
# achievable NLL at that step; the gradient still flows through the thought
# leaves because the logits at every step depend on them.
#
# max_new: number of answer tokens to decode. We stop early at eos.
# ═══════════════════════════════════════════════════════════════════


def _principal_angle_cos2(B_pred, B_gold):
    """
    mean cos^2 principal-angle overlap between two orthonormal bases.

    Math:
        # M = B_pred^T B_gold           (k_pred, k_gold)
        # sigma = svd(M).S              singular values, each in [0, 1]
        # returns mean(sigma^2), and the sigma spectrum

    Both bases have orthonormal columns, so sigma_i = cos(theta_i) are the
    cosines of the principal angles between the two subspaces.
    """
    if B_pred.size == 0 or B_gold.size == 0:
        return float("nan"), []
    M = B_pred.T @ B_gold                     # (k_pred, k_gold)
    sv = np.linalg.svd(M, compute_uv=False)   # min(k_pred, k_gold) values
    cos2 = float(np.mean(sv ** 2))
    return cos2, sv.tolist()


# ═══════════════════════════════════════════════════════════════════
# Coconut extractor (predicted-token target)
# ═══════════════════════════════════════════════════════════════════
#
# Identical to gradient_subspace.extract_gradient_coconut EXCEPT the target
# token ids come from greedy decoding instead of tokenizer.encode(gold_text).
# ═══════════════════════════════════════════════════════════════════

def extract_gradient_coconut_pred(
    base_model, tokenizer, sample, n_thoughts, device,
    start_id, end_id, max_new,
):
    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    input_ids = torch.tensor([question_tokens + [start_id]], device=device)

    outputs = base_model(
        input_ids=input_ids, output_hidden_states=True, use_cache=True,
    )
    h = outputs.hidden_states[-1][0, -1, :]

    h_leaf = h.detach().clone().requires_grad_(True)
    h_leaves = [h_leaf]
    past_kv = outputs.past_key_values
    cur = h_leaf

    for t in range(1, n_thoughts + 1):
        ct = cur.unsqueeze(0).unsqueeze(0)
        outputs = base_model(
            inputs_embeds=ct, past_key_values=past_kv,
            output_hidden_states=True, use_cache=True,
        )
        h_next = outputs.hidden_states[-1][0, 0, :]
        h_next_leaf = h_next.detach().clone().requires_grad_(True)
        h_leaves.append(h_next_leaf)
        cur = h_next_leaf
        past_kv = outputs.past_key_values

    end_input = torch.tensor([[end_id]], device=device)
    outputs = base_model(
        input_ids=end_input, past_key_values=past_kv, use_cache=True,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]

    # ── Greedy target: yhat_0 = argmax, then feed it back ──────────
    eos_id = tokenizer.eos_token_id
    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    yhat0 = int(torch.argmax(answer_logits_first).item())
    if yhat0 == eos_id:
        # Degenerate: model predicts eos immediately. No usable target.
        D = h.shape[0]
        return np.zeros((n_thoughts + 1, D), dtype=np.float32), 0.0
    nll = -log_probs_first[yhat0]
    pred_ids = [yhat0]

    for _ in range(1, max_new):
        prev = torch.tensor([[pred_ids[-1]]], device=device)
        out = base_model(
            input_ids=prev, past_key_values=past_kv, use_cache=True,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        yhat = int(torch.argmax(next_logits).item())
        if yhat == eos_id:
            break
        nll = nll - next_logp[yhat]
        pred_ids.append(yhat)

    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )
    grads_arr = []
    for t_idx, g in enumerate(grads_t):
        if g is None:
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    grads = np.stack(grads_arr, axis=0)
    return grads, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# Pause extractor (predicted-token target)
# ═══════════════════════════════════════════════════════════════════
#
# Identical to gradient_subspace.extract_gradient_pause EXCEPT the target.
# ═══════════════════════════════════════════════════════════════════

def extract_gradient_pause_pred(
    coconut_model, base_model, tokenizer, sample, n_thoughts, device,
    start_id, latent_id, end_id, max_new,
):
    question_tokens = tokenize_question_for_recurrence(tokenizer, sample["question"])
    input_ids_list = (
        question_tokens + [start_id] + [latent_id] * n_thoughts + [end_id]
    )
    input_ids = torch.tensor([input_ids_list], device=device)

    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    embedding = coconut_model.embedding
    inputs_embeds_base = embedding(input_ids)
    pause_emb = coconut_model.pause_embedding

    L = input_ids.shape[1]
    start_latent_pos = len(question_tokens)
    start_of_latent = start_latent_pos + 1
    thought_positions = [start_latent_pos] + [
        start_of_latent + i for i in range(n_thoughts)
    ]

    h_leaves = []
    for t_idx in range(n_thoughts + 1):
        if t_idx == 0:
            init = inputs_embeds_base[0, start_latent_pos, :].detach().clone()
        else:
            init = pause_emb.detach().clone()
        h_leaves.append(init.requires_grad_(True))

    embeds_per_pos = []
    for j in range(L):
        if j in thought_positions:
            t_idx = thought_positions.index(j)
            embeds_per_pos.append(h_leaves[t_idx])
        else:
            embeds_per_pos.append(inputs_embeds_base[0, j, :].detach())
    inputs_embeds = torch.stack(embeds_per_pos, dim=0).unsqueeze(0)

    attention_mask = torch.ones((1, L), dtype=torch.long, device=device)
    position_ids = torch.arange(L, device=device).unsqueeze(0)

    outputs = base_model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]

    eos_id = tokenizer.eos_token_id
    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    yhat0 = int(torch.argmax(answer_logits_first).item())
    if yhat0 == eos_id:
        return np.zeros((n_thoughts + 1, pause_emb.shape[0]), dtype=np.float32), 0.0
    nll = -log_probs_first[yhat0]
    pred_ids = [yhat0]

    for _ in range(1, max_new):
        prev = torch.tensor([[pred_ids[-1]]], device=device)
        out = base_model(
            input_ids=prev, past_key_values=past_kv, use_cache=True,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        yhat = int(torch.argmax(next_logits).item())
        if yhat == eos_id:
            break
        nll = nll - next_logp[yhat]
        pred_ids.append(yhat)

    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )
    grads_arr = []
    for t_idx, g in enumerate(grads_t):
        if g is None:
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    grads = np.stack(grads_arr, axis=0)
    return grads, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# CODI extractor (predicted-token target)
# ═══════════════════════════════════════════════════════════════════
#
# Identical to gradient_subspace.extract_gradient_codi EXCEPT the target.
# ═══════════════════════════════════════════════════════════════════

def extract_gradient_codi_pred(codi_dict, sample, n_thoughts, device, max_new):
    base_model = codi_dict['model']
    prj = codi_dict['prj']
    tokenizer = codi_dict['tokenizer']
    bot_id = codi_dict['bot_id']
    eot_id = codi_dict['eot_id']
    embedding_fn = codi_dict['embedding_fn']
    use_prj = codi_dict['use_prj']
    remove_eos = codi_dict['remove_eos']

    for p in base_model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
    if prj is not None:
        for p in prj.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)

    question_text = sample["question"].strip().replace('  ', ' ')
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L_prompt = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(L_prompt, device=device).unsqueeze(0)

    outputs = base_model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True, output_hidden_states=True,
    )
    past_kv = outputs.past_key_values
    h = outputs.hidden_states[-1][0, -1, :]

    orig_dtype = h.dtype
    h_leaf = h.detach().clone().to(orig_dtype).requires_grad_(True)
    h_leaves = [h_leaf]
    cur = h_leaf

    latent = cur.unsqueeze(0).unsqueeze(0)
    if use_prj and prj is not None:
        latent = prj(latent)

    running_mask = attention_mask
    for t in range(1, n_thoughts + 1):
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        pos_t = torch.tensor([[L_prompt + t - 1]], device=device)
        outputs = base_model(
            inputs_embeds=latent, attention_mask=running_mask,
            position_ids=pos_t, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        h_next = outputs.hidden_states[-1][0, -1, :]
        h_next_leaf = h_next.detach().clone().to(orig_dtype).requires_grad_(True)
        h_leaves.append(h_next_leaf)
        cur = h_next_leaf
        latent = cur.unsqueeze(0).unsqueeze(0)
        if use_prj and prj is not None:
            latent = prj(latent)

    if remove_eos:
        eot_ids = [eot_id]
    else:
        eot_ids = [eot_id, tokenizer.eos_token_id]
    eot_input = torch.tensor([eot_ids], device=device)
    eot_emb = embedding_fn(eot_input)
    eot_len = eot_emb.size(1)
    eot_pos = torch.arange(L_prompt + n_thoughts,
                           L_prompt + n_thoughts + eot_len,
                           device=device).unsqueeze(0)
    running_mask = torch.cat(
        [running_mask, torch.ones((1, eot_len), dtype=running_mask.dtype,
                                  device=device)],
        dim=1,
    )
    outputs = base_model(
        inputs_embeds=eot_emb, attention_mask=running_mask,
        position_ids=eot_pos, use_cache=True, past_key_values=past_kv,
    )
    past_kv = outputs.past_key_values
    answer_logits_first = outputs.logits[0, -1, :]
    current_pos = L_prompt + n_thoughts + eot_len

    eos_id = tokenizer.eos_token_id
    log_probs_first = torch.log_softmax(answer_logits_first, dim=-1)
    yhat0 = int(torch.argmax(answer_logits_first).item())
    if yhat0 == eos_id:
        D = h.shape[0]
        return np.zeros((n_thoughts + 1, D), dtype=np.float32), 0.0
    nll = -log_probs_first[yhat0]
    pred_ids = [yhat0]

    for _ in range(1, max_new):
        prev = torch.tensor([[pred_ids[-1]]], device=device)
        prev_emb = embedding_fn(prev)
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype,
                                      device=device)],
            dim=1,
        )
        decode_pos = torch.tensor([[current_pos]], device=device)
        out = base_model(
            inputs_embeds=prev_emb, past_key_values=past_kv,
            use_cache=True, attention_mask=running_mask,
            position_ids=decode_pos,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_logp = torch.log_softmax(next_logits, dim=-1)
        yhat = int(torch.argmax(next_logits).item())
        if yhat == eos_id:
            break
        nll = nll - next_logp[yhat]
        pred_ids.append(yhat)
        current_pos += 1

    grads_t = torch.autograd.grad(
        outputs=nll, inputs=h_leaves,
        retain_graph=False, allow_unused=True,
    )
    grads_arr = []
    for t_idx, g in enumerate(grads_t):
        if g is None:
            grads_arr.append(np.zeros(h_leaves[t_idx].shape[0], dtype=np.float32))
        else:
            grads_arr.append(g.detach().to(torch.float32).cpu().numpy())
    grads = np.stack(grads_arr, axis=0)
    return grads, float(nll.detach().cpu().item())


# ═══════════════════════════════════════════════════════════════════
# Extraction driver
# ═══════════════════════════════════════════════════════════════════

def extract_subspace_pred(args, data, codi_dict, coconut_model, base_model,
                          tokenizer, latent_id, start_id, end_id, D, pause):
    N = len(data)
    K = args.n_thoughts
    T = K + 1
    print(f"[pred] instances={N}  T={T}  D={D}  max_new={args.max_new}")

    G = np.zeros((N, T, D), dtype=np.float32)
    losses = np.zeros(N, dtype=np.float32)
    n_skipped = 0

    for idx, sample in enumerate(data):
        if idx % 25 == 0 and idx > 0:
            print(f"  [pred] {idx}/{N}  mean NLL so far: "
                  f"{losses[:idx].mean():.4f}")
        try:
            if codi_dict is not None:
                grads, loss = extract_gradient_codi_pred(
                    codi_dict, sample, K, args.device, args.max_new,
                )
            elif pause:
                grads, loss = extract_gradient_pause_pred(
                    coconut_model, base_model, tokenizer, sample, K,
                    args.device, start_id, latent_id, end_id, args.max_new,
                )
            else:
                grads, loss = extract_gradient_coconut_pred(
                    base_model, tokenizer, sample, K, args.device,
                    start_id, end_id, args.max_new,
                )
        except Exception as e:
            print(f"  [pred] WARN instance {idx} failed: "
                  f"{type(e).__name__}: {e}")
            n_skipped += 1
            continue
        G[idx] = grads
        losses[idx] = loss

    print(f"\n[pred] Extracted gradients for {N - n_skipped}/{N} instances "
          f"(skipped {n_skipped})")
    if (losses > 0).any():
        print(f"[pred] mean predicted-token NLL: {losses[losses > 0].mean():.4f}")

    # ── SVD per timestep -> bases (same machinery as gold script) ──
    bases = {}
    ranks = []
    print()
    print(f"[svd] rho = {args.explained_variance}")
    print(f"[svd] {'t':>3}  {'rank':>6}")
    for t in range(T):
        G_t = G[:, t, :]
        row_norms = np.linalg.norm(G_t, axis=1)
        G_t_eff = G_t[row_norms > 0]
        if G_t_eff.shape[0] < 2:
            print(f"  [svd] WARN t={t}: <2 nonzero rows; empty basis")
            bases[t] = np.zeros((D, 0))
            ranks.append(0)
            continue
        B, k, _sv, _cv = subspace_from_gradients(
            G_t_eff, explained_variance=args.explained_variance,
            max_rank=args.max_rank,
        )
        bases[t] = B
        ranks.append(k)
        print(f"  [svd] {t:>3}  {k:>6}")

    return bases, G, losses, ranks, n_skipped


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Predicted-token gradient subspace + gold/pred "
                    "principal-angle comparison."
    )
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    parser.add_argument(
        "--model", choices=["coconut", "coconut_u", "pause", "codi"],
        default="coconut",
    )
    parser.add_argument(
        "--model_family", type=str, choices=["gpt2", "llama"], default="gpt2",
    )
    parser.add_argument("--n_thoughts", type=int, default=6)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--max_new", type=int, default=8,
                        help="Number of answer tokens to greedily decode as "
                             "the NLL target. Matches typical answer length; "
                             "stops early at eos.")
    parser.add_argument("--explained_variance", type=float, default=0.95)
    parser.add_argument("--max_rank", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write predicted-token artefacts. Default: "
             "BASE_DIR/outputs/gradient_geometry_predtoken/"
             "<family>/<task>/<model>. Kept in a SEPARATE tree from the "
             "gold gradient_geometry/ outputs so nothing collides.",
    )
    parser.add_argument(
        "--gold_bases", type=str, default=None,
        help="Path to the gold bases.npz to compare against. Defaults to "
             "the standard gradient_geometry/<family>/<task>/<model>/bases.npz "
             "(the ORIGINAL gold tree, not the predtoken output tree).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Predicted-token artefacts go in their own tree, fully separate from
    # the gold gradient_geometry/ outputs.
    output_dir = (Path(args.output_dir) if args.output_dir else
                  BASE_DIR / "outputs" / "gradient_geometry_predtoken"
                  / args.model_family / args.task / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] task={args.task}  model={args.model}  family={args.model_family}")
    print(f"[main] output_dir: {output_dir}")

    is_codi = (args.model == "codi")

    if is_codi:
        codi_dict = setup_codi_model(args.task, args.device, family=args.model_family)
        D = codi_dict['hidden_size']
        coconut_model = base_model = tokenizer = None
        latent_id = start_id = end_id = None
    else:
        codi_dict = None
        coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
            setup_model_and_tokenizer(args.task, args.model, args.device,
                                      family=args.model_family)
        D = getattr(base_model.config, "n_embd",
                    getattr(base_model.config, "hidden_size", None))
        if D is None:
            raise RuntimeError("Could not determine hidden size from model config")

    torch.set_grad_enabled(True)

    data = load_data(args.task, args.max_instances)
    pause = (not is_codi) and is_pause_model(coconut_model)

    bases_pred, G, losses, ranks, n_skipped = extract_subspace_pred(
        args, data, codi_dict, coconut_model, base_model,
        tokenizer, latent_id, start_id, end_id, D, pause,
    )
    T = args.n_thoughts + 1

    # ── Save predicted-token artefacts ─────────────────────────────
    # Folder is already the dedicated predtoken tree, so filenames need no
    # suffix -- they mirror the gold script's names within their own tree.
    bases_path = output_dir / "bases.npz"
    np.savez(bases_path, **{f"B_t{t}": bases_pred[t] for t in range(T)})
    print(f"\n[main] Saved predicted-token bases -> {bases_path}")

    grads_path = output_dir / "gradients.npz"
    np.savez(grads_path, G=G, losses=losses, ranks=np.array(ranks))
    print(f"[main] Saved predicted-token gradients -> {grads_path}")

    # ── Load gold bases and compare via principal angles ───────────
    # Gold lives in the ORIGINAL gradient_geometry/ tree, not our output_dir.
    gold_path = (Path(args.gold_bases) if args.gold_bases
                 else (BASE_DIR / "outputs" / "gradient_geometry"
                       / args.model_family / args.task / args.model
                       / "bases.npz"))
    if not gold_path.exists():
        print(f"\n[cmp] gold bases not found at {gold_path}; run "
              f"gradient_subspace.py first. Skipping comparison.")
        return

    gold_blob = np.load(gold_path)
    gold_bases = {int(k[len('B_t'):]): gold_blob[k]
                  for k in gold_blob.files if k.startswith('B_t')}

    print(f"\n[cmp] Principal-angle overlap  span(B_t^pred) vs span(B_t^gold)")
    print(f"[cmp] {'t':>3}  {'k_pred':>6}  {'k_gold':>6}  {'mean cos^2':>10}")
    per_t = {}
    cos2_vals = []
    for t in range(T):
        B_pred = bases_pred[t]
        B_gold = gold_bases.get(t, np.zeros((D, 0)))
        cos2, spectrum = _principal_angle_cos2(B_pred, B_gold)
        per_t[t] = {
            "k_pred": int(B_pred.shape[1]),
            "k_gold": int(B_gold.shape[1]),
            "mean_cos2": cos2,
            "principal_cosines": spectrum,
        }
        if not np.isnan(cos2):
            cos2_vals.append(cos2)
        print(f"[cmp] {t:>3}  {B_pred.shape[1]:>6}  {B_gold.shape[1]:>6}  "
              f"{cos2:>10.4f}")

    mean_over_t = float(np.mean(cos2_vals)) if cos2_vals else float("nan")
    print(f"[cmp] {'-'*30}")
    print(f"[cmp] mean cos^2 over t (excl. empty): {mean_over_t:.4f}")
    print(f"[cmp] interpretation: ->1 means gold subspace is NOT a label "
          f"artifact (directions are the model's own).")

    comparison = {
        "task": args.task,
        "model": args.model,
        "model_family": args.model_family,
        "n_instances": int(len(data)),
        "n_skipped": int(n_skipped),
        "max_new": args.max_new,
        "explained_variance": args.explained_variance,
        "ranks_pred_per_t": ranks,
        "gold_bases_path": str(gold_path),
        "per_t": per_t,
        "mean_cos2_over_t": mean_over_t,
        "mean_pred_nll": (
            float(losses[losses > 0].mean()) if (losses > 0).any() else 0.0
        ),
    }
    cmp_path = output_dir / "predtoken_comparison.json"
    with open(cmp_path, "w") as f:
        json.dump(deep_convert(comparison), f, indent=2)
    print(f"\n[main] Saved comparison -> {cmp_path}")


if __name__ == "__main__":
    main()