"""
Verify bmarti44's M2 (COCONUT) and M3 (Pause-curriculum) checkpoints
against claimed training configurations.

Checks:
1. State dict key structure — M2 should NOT have pause_embedding, M3 SHOULD
2. Vocab size — both should be 50,260 (GPT-2 50,257 + 3 special tokens)
3. Weight divergence from base GPT-2 — both should be fine-tuned, not random
4. M2 vs M3 weight divergence — they should differ (different training dynamics)
5. Embedding rows for special tokens — should be trained (not zero/random init)
6. Parameter count consistency
7. Pause embedding properties (M3 only)
"""

import torch
import sys
import os
import numpy as np
from collections import OrderedDict

def load_checkpoint(path):
    """Load a checkpoint, handling both direct state_dicts and wrapped formats."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path), {}
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Clean DDP/FSDP prefixes (module.*)
    cleaned = OrderedDict()
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("_orig_mod.", "")
        cleaned[k] = v
    return cleaned, {}

def strip_prefix(sd, prefix="base_causallm."):
    """Strip a common prefix from state dict keys."""
    new = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            new[k[len(prefix):]] = v
        else:
            new[k] = v
    return new

def check_structure(name, sd):
    """Check 1: Key structure and presence/absence of pause_embedding."""
    print(f"\n{'='*60}")
    print(f"  {name}: State Dict Structure")
    print(f"{'='*60}")
    
    has_pause = any("pause" in k.lower() for k in sd.keys())
    prefixed_keys = [k for k in sd.keys() if k.startswith("base_causallm.")]
    unprefixed_keys = [k for k in sd.keys() if not k.startswith("base_causallm.")]
    
    print(f"  Total keys: {len(sd)}")
    print(f"  Keys with 'base_causallm.' prefix: {len(prefixed_keys)}")
    print(f"  Other keys: {unprefixed_keys if unprefixed_keys else 'none'}")
    print(f"  Has pause_embedding: {has_pause}")
    
    return has_pause

def check_vocab_size(name, sd):
    """Check 2: Verify vocab dimension = 50,260."""
    print(f"\n{'='*60}")
    print(f"  {name}: Vocab Size Check")
    print(f"{'='*60}")
    
    # Find the token embedding weight
    emb_key = None
    for k in sd.keys():
        if "wte.weight" in k or "embed_tokens" in k:
            emb_key = k
            break
    
    if emb_key is None:
        print("  WARNING: Could not find token embedding weight!")
        return None
    
    vocab_size, emb_dim = sd[emb_key].shape
    print(f"  Embedding key: {emb_key}")
    print(f"  Vocab size: {vocab_size} (expected 50260)")
    print(f"  Embedding dim: {emb_dim} (expected 768 for GPT-2)")
    
    if vocab_size != 50260:
        print(f"  *** MISMATCH: expected 50260, got {vocab_size}")
    if emb_dim != 768:
        print(f"  *** MISMATCH: expected 768, got {emb_dim}")
    
    return vocab_size, emb_dim

def check_special_token_embeddings(name, sd):
    """Check 5: The last 3 embedding rows should be trained, not zero/default."""
    print(f"\n{'='*60}")
    print(f"  {name}: Special Token Embeddings (last 3 rows)")
    print(f"{'='*60}")
    
    emb_key = None
    for k in sd.keys():
        if "wte.weight" in k:
            emb_key = k
            break
    if emb_key is None:
        print("  Could not find embedding weight")
        return
    
    emb = sd[emb_key]
    special = emb[-3:]  # last 3 rows: <|start-latent|>, <|end-latent|>, <|latent|>
    normal_sample = emb[1000:1003]  # some normal tokens for comparison
    
    for i, tok_name in enumerate(["<|start-latent|>", "<|end-latent|>", "<|latent|>"]):
        row = special[i]
        print(f"  {tok_name}:")
        print(f"    norm={row.norm().item():.4f}, mean={row.mean().item():.6f}, "
              f"std={row.std().item():.6f}, min={row.min().item():.4f}, max={row.max().item():.4f}")
        is_zero = row.abs().max().item() < 1e-8
        if is_zero:
            print(f"    *** WARNING: This embedding is all zeros (untrained!)")
    
    print(f"  Normal token sample (idx 1000-1002) for comparison:")
    for i in range(3):
        row = normal_sample[i]
        print(f"    norm={row.norm().item():.4f}, mean={row.mean().item():.6f}, std={row.std().item():.6f}")

def check_divergence_from_gpt2(name, sd, base_gpt2_dir):
    """Check 3: Compare against base GPT-2 to verify fine-tuning happened."""
    print(f"\n{'='*60}")
    print(f"  {name}: Divergence from Base GPT-2")
    print(f"{'='*60}")
    
    try:
        base_path = os.path.join(base_gpt2_dir, "model.safetensors")
        if os.path.exists(base_path):
            from safetensors.torch import load_file
            base_sd = load_file(base_path)
        else:
            base_path = find_checkpoint(base_gpt2_dir)
            if base_path and base_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                base_sd = load_file(base_path)
            elif base_path:
                base_sd = torch.load(base_path, map_location="cpu", weights_only=False)
            else:
                print(f"  Could not find base GPT-2 weights in {base_gpt2_dir}")
                return None
    except Exception as e:
        print(f"  Could not load base GPT-2: {e}")
        return None
    
    clean_sd = strip_prefix(sd)
    
    # Try to match keys - base safetensors may or may not have prefixes
    # Print samples to help debug
    base_keys_sample = list(base_sd.keys())[:3]
    clean_keys_sample = list(clean_sd.keys())[:3]
    print(f"  Base GPT-2 key samples: {base_keys_sample}")
    print(f"  Checkpoint key samples: {clean_keys_sample}")
    
    # Try stripping common prefixes from base_sd too
    base_clean = strip_prefix(base_sd, "model.")
    
    diffs = {}
    matched = 0
    for k in base_clean:
        if k in clean_sd:
            if base_clean[k].shape == clean_sd[k].shape:
                diff = (base_clean[k].float() - clean_sd[k].float()).norm().item()
                rel_diff = diff / (base_clean[k].float().norm().item() + 1e-12)
                diffs[k] = (diff, rel_diff)
                matched += 1
    
    if not diffs:
        print("  No matching keys found — prefix mismatch?")
        return None
    
    abs_diffs = [v[0] for v in diffs.values()]
    rel_diffs = [v[1] for v in diffs.values()]
    
    print(f"  Matched {matched}/{len(base_clean)} base GPT-2 keys")
    print(f"  Absolute L2 diff: mean={np.mean(abs_diffs):.4f}, max={np.max(abs_diffs):.4f}")
    print(f"  Relative diff:    mean={np.mean(rel_diffs):.4f}, max={np.max(rel_diffs):.4f}")
    
    zero_diff = sum(1 for d in abs_diffs if d < 1e-8)
    if zero_diff == len(abs_diffs):
        print("  *** WARNING: Model is IDENTICAL to base GPT-2 (not fine-tuned!)")
    elif zero_diff > len(abs_diffs) * 0.5:
        print(f"  *** WARNING: {zero_diff}/{len(abs_diffs)} layers are unchanged from base")
    else:
        print(f"  {zero_diff}/{len(abs_diffs)} layers unchanged (expected: few or none)")
    
    # Show top-5 most changed layers
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1][1], reverse=True)
    print(f"\n  Top 5 most changed layers:")
    for k, (abs_d, rel_d) in sorted_diffs[:5]:
        print(f"    {k}: abs={abs_d:.4f}, rel={rel_d:.4f}")
    
    return diffs

def check_m2_vs_m3(sd_m2, sd_m3):
    """Check 4: M2 and M3 should have different weights (different training)."""
    print(f"\n{'='*60}")
    print(f"  M2 vs M3: Cross-Model Divergence")
    print(f"{'='*60}")
    
    clean_m2 = strip_prefix(sd_m2)
    clean_m3 = strip_prefix(sd_m3)
    
    common_keys = set(clean_m2.keys()) & set(clean_m3.keys())
    # Exclude pause_embedding
    common_keys = {k for k in common_keys if "pause" not in k.lower()}
    
    diffs = {}
    for k in common_keys:
        if clean_m2[k].shape == clean_m3[k].shape:
            diff = (clean_m2[k].float() - clean_m3[k].float()).norm().item()
            rel = diff / (clean_m2[k].float().norm().item() + 1e-12)
            diffs[k] = (diff, rel)
    
    abs_diffs = [v[0] for v in diffs.values()]
    rel_diffs = [v[1] for v in diffs.values()]
    
    print(f"  Compared {len(diffs)} shared parameter tensors")
    print(f"  Absolute L2 diff: mean={np.mean(abs_diffs):.4f}, max={np.max(abs_diffs):.4f}")
    print(f"  Relative diff:    mean={np.mean(rel_diffs):.4f}, max={np.max(rel_diffs):.4f}")
    
    identical = sum(1 for d in abs_diffs if d < 1e-8)
    if identical == len(abs_diffs):
        print("  *** CRITICAL: M2 and M3 have IDENTICAL weights!")
        print("      This means they are the SAME model or one was copied from the other.")
    elif identical > len(abs_diffs) * 0.8:
        print(f"  *** SUSPICIOUS: {identical}/{len(abs_diffs)} tensors are identical")
    else:
        print(f"  {identical}/{len(abs_diffs)} tensors identical (expected: few or none)")
    
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1][1], reverse=True)
    print(f"\n  Top 5 most divergent layers:")
    for k, (abs_d, rel_d) in sorted_diffs[:5]:
        print(f"    {k}: abs={abs_d:.4f}, rel={rel_d:.4f}")
    print(f"\n  Bottom 5 (least divergent):")
    for k, (abs_d, rel_d) in sorted_diffs[-5:]:
        print(f"    {k}: abs={abs_d:.4f}, rel={rel_d:.4f}")

def check_pause_embedding(name, sd):
    """Check 7: Analyze the pause embedding if present."""
    print(f"\n{'='*60}")
    print(f"  {name}: Pause Embedding Analysis")
    print(f"{'='*60}")
    
    pause_key = None
    for k in sd:
        if "pause" in k.lower():
            pause_key = k
            break
    
    if pause_key is None:
        print("  No pause embedding found (expected for M2/COCONUT)")
        return
    
    pe = sd[pause_key]
    print(f"  Key: {pause_key}")
    print(f"  Shape: {pe.shape}")
    print(f"  norm={pe.norm().item():.4f}, mean={pe.mean().item():.6f}, "
          f"std={pe.std().item():.6f}")
    
    # Compare to the <|latent|> token embedding (idx 50259)
    emb_key = [k for k in sd if "wte.weight" in k]
    if emb_key:
        latent_emb = sd[emb_key[0]][-1]  # last token = <|latent|>
        cos_sim = torch.nn.functional.cosine_similarity(
            pe.float().flatten().unsqueeze(0),
            latent_emb.float().flatten().unsqueeze(0)
        ).item()
        l2 = (pe.float().flatten() - latent_emb.float().flatten()).norm().item()
        print(f"  vs <|latent|> embedding: cosine_sim={cos_sim:.4f}, L2={l2:.4f}")

def find_checkpoint(directory):
    """Find the model file in a checkpoint directory."""
    for fname in ["pytorch_model.bin", "model.bin", "model.safetensors",
                   "consolidated.pth", "checkpoint.pt"]:
        p = os.path.join(directory, fname)
        if os.path.exists(p):
            return p
    # Maybe the directory itself is a file
    if os.path.isfile(directory):
        return directory
    # List what's there
    if os.path.isdir(directory):
        files = os.listdir(directory)
        # Pick any .bin or .pt file
        for f in files:
            if f.endswith((".bin", ".pt", ".pth", ".safetensors")):
                return os.path.join(directory, f)
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpt2", 
                        default="/home/getalp/aswald/contReason/model/gpt2")
    parser.add_argument("--m2-dir",
                        default="/home/getalp/aswald/contReason/model/prosqa/coconut/checkpoint_best")
    parser.add_argument("--m3-dir",
                        default="/home/getalp/aswald/contReason/model/prosqa/pause/checkpoint_best")
    args = parser.parse_args()
    
    # checkpoint_best is the file itself (raw state_dict via torch.save)
    m2_path = args.m2_dir
    m3_path = args.m3_dir
    
    for name, path in [("M2", m2_path), ("M3", m3_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)
    
    print(f"Loading M2 from: {m2_path}")
    sd_m2, meta_m2 = load_checkpoint(m2_path)
    print(f"Loading M3 from: {m3_path}")
    sd_m3, meta_m3 = load_checkpoint(m3_path)
    
    # Run all checks
    # 1. Structure
    m2_has_pause = check_structure("M2 (COCONUT)", sd_m2)
    m3_has_pause = check_structure("M3 (Pause)", sd_m3)
    
    print(f"\n  VERDICT: M2 has pause_embedding = {m2_has_pause} (expected: False)")
    print(f"  VERDICT: M3 has pause_embedding = {m3_has_pause} (expected: True)")
    if m2_has_pause:
        print("  *** PROBLEM: M2 should NOT have a pause embedding!")
    if not m3_has_pause:
        print("  *** PROBLEM: M3 SHOULD have a pause embedding!")
    
    # 2. Vocab size
    check_vocab_size("M2", sd_m2)
    check_vocab_size("M3", sd_m3)
    
    # 5. Special token embeddings
    check_special_token_embeddings("M2", sd_m2)
    check_special_token_embeddings("M3", sd_m3)
    
    # 7. Pause embedding
    check_pause_embedding("M2", sd_m2)
    check_pause_embedding("M3", sd_m3)
    
    # 3. Divergence from base GPT-2
    print("\n\n" + "#"*60)
    print("  Checking divergence from base GPT-2...")
    print("  (requires downloading openai-community/gpt2)")
    print("#"*60)
    check_divergence_from_gpt2("M2", sd_m2, args.base_gpt2)
    check_divergence_from_gpt2("M3", sd_m3, args.base_gpt2)
    
    # 4. M2 vs M3
    check_m2_vs_m3(sd_m2, sd_m3)
    
    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print("""
  Key things to look for in the output above:
  
  1. M2 should NOT have pause_embedding; M3 SHOULD
  2. Both should have vocab_size=50260, emb_dim=768
  3. Special token embeddings should be non-zero (trained)
  4. Both should diverge substantially from base GPT-2
  5. M2 and M3 should differ from each other (different training)
  6. If M2 ≈ M3 (very low divergence), the checkpoints may be
     copies or trained with the same feedback_mode (suspicious)
  7. M3's pause_embedding should look like a trained embedding
     (non-trivial norm, not identical to <|latent|> token)
""")

if __name__ == "__main__":
    main()