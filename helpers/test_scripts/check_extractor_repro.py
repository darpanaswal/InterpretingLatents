"""
Reproduction check for the logit_lens recurrence-extractor fix.

Runs TWO extractors on the same model and compares their per-thought-step
top-1 decoded tokens:

  OLD  = single-token forwards through a kv-cache, no latent tokens in the
         sequence, implicit position_ids  (the buggy path under RoPE).
  NEW  = training-faithful full-sequence forward with [latent]*k present,
         contiguous position_ids, slot p filled with h[p-1]
         (the patched extract_coconut_full else-branch).

Decision rule:
  - GPT-2 coconut: NEW should REPRODUCE the OLD alternating figure
    (GPT-2's absolute-position path happened to coincide with training).
    High top-1 agreement => the fix is safe and you can rerun Llama.
    Low agreement => the OLD GPT-2 figure itself was not training-faithful;
    re-examine the whole panel before trusting either.
  - Llama coconut: NEW should DIVERGE from OLD (OLD was broken on Llama).
    Inspect NEW alone: it should decode plausible intermediates instead of
    '<<' / code-token garbage.

Run from repo root (so `import src...` resolves). Put the PATCHED logit_lens.py
on the path as src/experiments/.../logit_lens.py, or point --logit_lens_path
at it; this script imports extract_coconut_full from it.

    python check_extractor_repro.py --family gpt2  --model coconut --task gsm
    python check_extractor_repro.py --family llama --model coconut --task gsm --fp32
"""

import argparse
import importlib.util
import torch

from src.utils import (
    setup_model_and_tokenizer as setup_coconut_model,
    tokenize_question_for_recurrence,
)


# ── OLD extractor (verbatim buggy kv-cache path, decode only) ──
@torch.no_grad()
def old_recurrence_hidden_states(coconut_model, base_model, tokenizer,
                                 question_text, k, device):
    question_tokens = tokenize_question_for_recurrence(tokenizer, question_text)
    start_id_local = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    input_ids = torch.tensor([question_tokens + [start_id_local]], device=device)
    hidden_states = []

    outputs = base_model(input_ids=input_ids,
                         output_hidden_states=True, use_cache=True)
    h = outputs.hidden_states[-1][0, -1, :]
    past_kv = outputs.past_key_values
    hidden_states.append(h)

    for _ in range(k):
        outputs = base_model(inputs_embeds=h.unsqueeze(0).unsqueeze(0),
                             past_key_values=past_kv,
                             output_hidden_states=True, use_cache=True)
        h = outputs.hidden_states[-1][0, 0, :]
        past_kv = outputs.past_key_values
        hidden_states.append(h)
    return hidden_states


def load_new_extractor(logit_lens_path):
    """Import extract_coconut_full from the patched logit_lens.py."""
    if logit_lens_path is None:
        # assume it's importable on the path
        from importlib import import_module
        for modname in (
            "experiments.dead_salmon.logit_lens",
            "logit_lens",
        ):
            try:
                m = import_module(modname)
                return m.extract_coconut_full
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Could not import logit_lens; pass --logit_lens_path /path/to/logit_lens.py")
    spec = importlib.util.spec_from_file_location("patched_logit_lens", logit_lens_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_coconut_full


def top1(h, lm_head, tokenizer):
    h = h.to(lm_head.weight.dtype)
    logits = lm_head(h)
    tid = int(torch.argmax(logits.float()).item())
    try:
        s = tokenizer.decode([tid]).replace("\n", "\\n")
    except Exception:
        s = f"<id:{tid}>"
    return tid, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["gpt2", "llama"], required=True)
    ap.add_argument("--model",
                    choices=["coconut", "coconut_u", "cot", "base"],
                    default="coconut",
                    help="Recurrence-path models only (not pause/codi).")
    ap.add_argument("--task", choices=["prosqa", "gsm"], default="gsm")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=20, help="number of questions to average over")
    ap.add_argument("--fp32", action="store_true", help="upcast Llama to fp32")
    ap.add_argument("--logit_lens_path", type=str, default=None,
                    help="path to the PATCHED logit_lens.py")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    extract_new = load_new_extractor(args.logit_lens_path)

    coconut_model, base_model, tokenizer, latent_id, start_id, end_id, _ = \
        setup_coconut_model(args.task, args.model, device, family=args.family)
    if args.fp32 and args.family == "llama":
        coconut_model = coconut_model.float()
        base_model = coconut_model.base_causallm
        print("[fp32] Llama upcast to float32.")
    lm_head = base_model.get_output_embeddings()

    # Load a few questions.
    from src.config import PROSQA_TEST, GSM_TEST
    import json
    path = GSM_TEST if args.task == "gsm" else PROSQA_TEST
    with open(path) as f:
        data = [json.loads(l) for l in f] if str(path).endswith(".jsonl") else json.load(f)
    data = data[:args.n]

    print(f"\n{'='*78}")
    print(f"REPRO CHECK  family={args.family} model={args.model} task={args.task} "
          f"k={args.k} n={len(data)} fp32={args.fp32}")
    print(f"{'='*78}")
    print("Per-step top-1 agreement between OLD (kv-cache) and NEW (full-seq) extractors.")
    print("GPT-2 coconut: expect HIGH agreement (fix reproduces old figure).")
    print("Llama coconut: expect LOW agreement (old was broken); inspect NEW tokens.\n")

    n_steps = args.k + 1
    agree = [0] * n_steps
    counted = 0
    show = 0

    for sample in data:
        q = sample["question"]
        old_h = old_recurrence_hidden_states(
            coconut_model, base_model, tokenizer, q, args.k, device)
        new_h, _, _, _ = extract_new(
            coconut_model, base_model, tokenizer, q,
            args.k, device, start_id, latent_id, end_id)

        if len(old_h) != n_steps or len(new_h) != n_steps:
            print(f"  [skip] step count mismatch old={len(old_h)} new={len(new_h)}")
            continue
        counted += 1

        old_t1 = [top1(h, lm_head, tokenizer) for h in old_h]
        new_t1 = [top1(h, lm_head, tokenizer) for h in new_h]
        for t in range(n_steps):
            if old_t1[t][0] == new_t1[t][0]:
                agree[t] += 1

        if show < 3:
            print(f"  Q: {q[:70]}...")
            print(f"    {'t':>3}  {'OLD top-1':>16}  {'NEW top-1':>16}  match")
            for t in range(n_steps):
                m = "Y" if old_t1[t][0] == new_t1[t][0] else " "
                print(f"    {t:>3}  {old_t1[t][1]!r:>16}  {new_t1[t][1]!r:>16}   {m}")
            print()
            show += 1

    print(f"{'-'*78}")
    print(f"Per-step top-1 agreement over {counted} questions:")
    print(f"  {'t':>3}  {'agree':>7}  {'rate':>7}")
    for t in range(n_steps):
        r = agree[t] / counted if counted else 0.0
        print(f"  {t:>3}  {agree[t]:>3}/{counted:<3}  {r:>6.1%}")
    overall = sum(agree) / (counted * n_steps) if counted else 0.0
    print(f"\n  Overall top-1 agreement: {overall:.1%}")
    if args.family == "gpt2":
        print("  GPT-2: >~80% => fix reproduces old figure, safe to rerun Llama.")
        print("         low   => old GPT-2 path was not training-faithful either.")
    else:
        print("  Llama: low agreement is EXPECTED (old broken). Inspect NEW top-1")
        print("         above — should be plausible intermediates, not '<<'/code junk.")


if __name__ == "__main__":
    main()