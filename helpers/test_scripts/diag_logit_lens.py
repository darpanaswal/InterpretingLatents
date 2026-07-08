"""
Diagnostic: why do recurrence-path models (cot, coconut, coconut_u, pause)
collapse on Llama but alternate on GPT-2, while CODI survives on both?

Hypotheses checked:
  A. The "<<"-copy into <|start-latent|>/<|end-latent|>/<|latent|> slots did
     not land on the embedding/head actually used at forward time (PeftModel
     modules_to_save / wrong tensor copy).
  B. get_output_embeddings() returns a stale (pre-resize) or untied head, so
     the logit lens decodes h into the wrong vocab space -> intermediates
     never match -> hit-rate floor.
  C. Fed-back last-layer h leaves the embedding manifold under recurrence
     (decode degenerates step over step).

Run from repo root (so `import src...` resolves), e.g.:
    python -m experiments.dead_salmon.diag_logit_lens --family llama --model cot --task gsm
or drop this file anywhere on the path and:
    python diag_logit_lens.py --family llama --model cot --task gsm

CODI is handled via a separate branch (setup_codi_model), since it builds its
own lm_head and feeds latents through `prj` rather than raw hidden states.
"""

import argparse
import torch

from src.utils import (
    setup_model_and_tokenizer as setup_coconut_model,
    setup_codi_model,
    tokenize_question_for_recurrence,
)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_emb_head(m):
    """Resolve input embeddings + output head through whatever wrapper m is
    (PeftModel / Coconut.base_causallm / plain HF model)."""
    emb = m.get_input_embeddings()
    head = m.get_output_embeddings()
    return emb, head


def resolve_target_id(tokenizer, family):
    # Mirror utils._resize_and_copy_target exactly.
    if family == "llama":
        return tokenizer.encode("<<", add_special_tokens=False)[0]
    return tokenizer.convert_tokens_to_ids("<<")


def check_slot_copy(emb, head, tokenizer, family, latent_id, start_id, end_id):
    """HYPOTHESIS A + B (slot copy landed; head is right size/tied)."""
    banner("[A/B] Latent-slot copy + head wiring")
    tgt = resolve_target_id(tokenizer, family)
    print(f"  vocab(tokenizer)        = {len(tokenizer)}")
    print(f"  emb.weight.shape        = {tuple(emb.weight.shape)}")
    print(f"  head.weight.shape       = {tuple(head.weight.shape)}")
    print(f"  emb rows == vocab?      = {emb.weight.shape[0] == len(tokenizer)}")
    print(f"  head rows == vocab?     = {head.weight.shape[0] == len(tokenizer)}")
    # Tied? Same storage means head is the transpose-tied input emb (common in HF).
    tied = emb.weight.data_ptr() == head.weight.data_ptr()
    print(f"  emb/head tied (same ptr)= {tied}")
    print(f"  '<<' target_id          = {tgt}")
    print(f"  slot ids: latent={latent_id} start={start_id} end={end_id}")

    print("\n  Per-slot: does row == '<<' row?  (expect True if copy landed)")
    ok = True
    for name, tid in [("latent", latent_id), ("start", start_id), ("end", end_id)]:
        if tid >= emb.weight.shape[0] or tid >= head.weight.shape[0] or tgt >= emb.weight.shape[0]:
            print(f"    {name:6s} id={tid}: OUT OF RANGE for emb/head -> slot never allocated")
            ok = False
            continue
        emb_eq = torch.equal(emb.weight.data[tid], emb.weight.data[tgt])
        head_eq = torch.equal(head.weight.data[tid], head.weight.data[tgt])
        emb_norm = emb.weight.data[tid].float().norm().item()
        head_norm = head.weight.data[tid].float().norm().item()
        print(f"    {name:6s} id={tid}: emb_eq={emb_eq!s:5s} head_eq={head_eq!s:5s} "
              f"|emb|={emb_norm:8.4f} |head|={head_norm:8.4f}")
        ok = ok and emb_eq and head_eq

    print(f"\n  => slot copy fully landed on BOTH emb and head: {ok}")
    if not ok:
        print("     If emb_eq True but head_eq False -> head is stale/untied (Hyp B).")
        print("     If both False -> copy hit a different tensor copy (Hyp A, PeftModel).")
    return ok


def check_lora(m):
    """Confirm an adapter is actually active (silent no-op load mirrors base)."""
    banner("[LoRA] adapter sanity")
    lora_params = [(n, p) for n, p in m.named_parameters() if "lora_" in n]
    if not lora_params:
        print("  no lora_ params found (either not a PeftModel here, or adapter not loaded)")
        return
    total_norm = sum(p.detach().float().norm().item() ** 2 for _, p in lora_params) ** 0.5
    active = getattr(m, "active_adapter", None)
    print(f"  active_adapter={active}  #lora_params={len(lora_params)}  total_norm={total_norm:.4f}")
    # Does any adapter / saved module touch embeddings or the head?
    touch = [n for n, _ in m.named_parameters()
             if ("embed" in n.lower() or "lm_head" in n.lower())
             and ("lora" in n.lower() or "modules_to_save" in n.lower())]
    if touch:
        print("  WARNING: adapter/modules_to_save touches emb/head — resize-through-wrapper")
        print("           may write the wrong copy. Affected params (first 5):")
        for n in touch[:5]:
            print(f"             {n}")
    else:
        print("  adapter does NOT wrap emb/head (resize should reach the real tensors)")


@torch.no_grad()
def check_recurrence_decode(coconut_model, base_model, tokenizer, lm_head,
                            question, k, device, family,
                            latent_id, start_id, end_id, top_k=8):
    """HYPOTHESIS C: does decode degenerate step-over-step under recurrence?
    Replicates extract_coconut_full's recurrence else-branch and logit-lenses
    each h_t through the SAME lm_head logit_lens.py uses."""
    banner("[C] Recurrence decode trajectory (top-1 per thought step)")
    is_pause = (getattr(coconut_model, "feedback_mode", "continuous") == "pause_curriculum")
    print(f"  feedback_mode={getattr(coconut_model,'feedback_mode',None)}  is_pause={is_pause}")

    question_tokens = tokenize_question_for_recurrence(tokenizer, question)
    print(f"  prompt token count (post chat-template if Llama): {len(question_tokens)}")
    print(f"  first 12 prompt ids: {question_tokens[:12]}")

    hidden_states = []

    if is_pause:
        emb_fn = coconut_model.embedding
        input_ids_list = question_tokens + [start_id] + [latent_id] * k + [end_id]
        input_ids = torch.tensor([input_ids_list], device=device)
        inputs_embeds = emb_fn(input_ids)
        start_of_latent = len(question_tokens) + 1
        for i in range(k):
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, start_of_latent + i, :] = coconut_model.pause_embedding
        out = base_model(inputs_embeds=inputs_embeds,
                         output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]
        hidden_states.append(last_hidden[0, len(question_tokens), :])  # h_0 at <start>
        for i in range(k):
            hidden_states.append(last_hidden[0, start_of_latent + i, :])
    else:
        start_id_local = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        input_ids = torch.tensor([question_tokens + [start_id_local]], device=device)
        out = base_model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        h = out.hidden_states[-1][0, -1, :]
        past_kv = out.past_key_values
        hidden_states.append(h)
        for _ in range(k):
            out = base_model(inputs_embeds=h.unsqueeze(0).unsqueeze(0),
                             past_key_values=past_kv,
                             output_hidden_states=True, use_cache=True)
            h = out.hidden_states[-1][0, 0, :]
            past_kv = out.past_key_values
            hidden_states.append(h)

    print(f"\n  {'t':>3}  {'|h|':>9}  top-{top_k} decoded tokens (lm_head)")
    print(f"  {'-'*3}  {'-'*9}  {'-'*40}")
    prev_top1 = None
    repeats = 0
    for t, h in enumerate(hidden_states):
        h = h.to(lm_head.weight.dtype)
        logits = lm_head(h)
        probs = torch.softmax(logits.float(), dim=-1)
        tp, ti = torch.topk(probs, top_k)
        toks = []
        for p, i in zip(tp.tolist(), ti.tolist()):
            try:
                s = tokenizer.decode([i]).replace("\n", "\\n")
            except Exception:
                s = f"<id:{i}>"
            toks.append(f"{s!r}({p:.2f})")
        top1 = ti[0].item()
        if top1 == prev_top1:
            repeats += 1
        prev_top1 = top1
        print(f"  {t:>3}  {h.float().norm().item():>9.3f}  {', '.join(toks)}")

    n = len(hidden_states)
    print(f"\n  top-1 repeated-from-prev-step: {repeats}/{n-1}")
    if repeats >= n - 2:
        print("  => DEGENERATE: decode is frozen (Hyp C, or dead/untrained head).")
    else:
        print("  => decode evolves across steps (no freeze).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["gpt2", "llama"], required=True)
    ap.add_argument("--model",
                    choices=["base", "cot", "pause", "coconut", "coconut_u", "codi"],
                    required=True)
    ap.add_argument("--task", choices=["prosqa", "gsm"], default="gsm")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--question", type=str, default=None,
                    help="Override probe question. Default is a tiny GSM-style sum.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.question is None:
        if args.task == "gsm":
            args.question = ("Natalia sold 48 clips in April and half as many in May. "
                             "How many clips did she sell altogether?")
        else:
            args.question = "Is alex a wumpus? alex is a rompus. every rompus is a wumpus."

    banner(f"DIAGNOSTIC  family={args.family}  model={args.model}  task={args.task}  k={args.k}")

    if args.model == "codi":
        codi = setup_codi_model(args.task, device, family=args.family)
        tokenizer = codi["tokenizer"]
        lm_head = codi["lm_head"]
        base_model = codi["model"]
        print("  CODI: builds its own lm_head and feeds latents via prj (use_prj="
              f"{codi.get('use_prj')}).")
        print(f"  lm_head.weight.shape = {tuple(lm_head.weight.shape)}  vocab={len(tokenizer)}")
        print(f"  bot_id={codi.get('bot_id')} eot_id={codi.get('eot_id')} "
              f"prj_present={codi.get('prj') is not None}")
        # CODI has no <|latent|> slots / recurrence-else-branch to check; the
        # alternation it shows comes from prj-fed latents, not raw h feedback.
        emb = base_model.get_input_embeddings()
        print(f"  emb.weight.shape = {tuple(emb.weight.shape)}")
        print("\n  (CODI is the control that survives; nothing further to diagnose here.)")
        return

    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, ckpt = \
        setup_coconut_model(args.task, args.model, device, family=args.family)
    lm_head = base_model.get_output_embeddings()  # exactly what logit_lens.py uses
    print(f"  checkpoint: {ckpt}")

    emb, head = get_emb_head(base_model)
    check_slot_copy(emb, head, tokenizer, args.family, latent_id, start_id, end_id)

    if args.family == "llama" and args.model != "base":
        check_lora(base_model)

    check_recurrence_decode(coconut_model, base_model, tokenizer, lm_head,
                            args.question, args.k, device, args.family,
                            latent_id, start_id, end_id, top_k=args.top_k)

    banner("READING THE OUTPUT")
    print("""  - [A/B] head_eq False (esp. with emb_eq True)  -> stale/untied head (Hyp B)
  - [A/B] both emb_eq & head_eq False            -> copy hit wrong tensor (Hyp A)
  - [A/B] head rows < vocab                       -> decoding through pre-resize head
  - [LoRA] total_norm ~0 or wraps emb/head        -> adapter no-op / resize misroute
  - [C] top-1 frozen across steps                 -> decode degenerate / dead head
  Compare gpt2 vs llama for the SAME --model to localize the family-specific break.""")


if __name__ == "__main__":
    main()