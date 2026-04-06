"""
Verify Checkpoint Identities (The True Fingerprint).

Checks if unverified HuggingFace checkpoints for COCONUT and PAUSE are actually
distinct models by inspecting the existence and drift of the custom 'pause_embedding'
parameter, which is unique to the Pause architecture.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from utils.config import BASE_GPT2, PAUSE_GPT2, COCONUT_GPT2
from utils.utilities import clean_state_dict_keys

def get_clean_state_dict(checkpoint_dir):
    checkpoint_path = checkpoint_dir / "checkpoint_best"
    print(f"Loading {checkpoint_path}...")
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return clean_state_dict_keys(raw_state_dict)

def find_tensor_by_suffix(state_dict, suffix):
    """Dynamically search for a tensor key ending in the given suffix."""
    for key, tensor in state_dict.items():
        if key.endswith(suffix):
            return tensor
    return None

def verify_pause_fingerprint(state_dict, tokenizer, model_label):
    """
    Look for the 'pause_embedding' key and measure how far it drifted 
    from the initial '<<' vocabulary embedding.
    """
    pause_emb = find_tensor_by_suffix(state_dict, "pause_embedding")
    wte_weight = find_tensor_by_suffix(state_dict, "wte.weight")
    
    if pause_emb is None:
        print(f"[{model_label}] ❌ 'pause_embedding' key NOT FOUND in state dict.")
        return False, 0.0
        
    print(f"[{model_label}] ✅ 'pause_embedding' key FOUND!")
    
    if wte_weight is not None:
        target_id = tokenizer.convert_tokens_to_ids("<<")
        init_vec = wte_weight[target_id]
        
        # Calculate L2 distance from initialization
        drift = torch.norm(pause_emb - init_vec).item()
        print(f"[{model_label}] Drift from '<<' initialization: {drift:.6f}")
        return True, drift
    else:
        print(f"[{model_label}] Could not find 'wte.weight' to calculate drift.")
        return True, -1.0

def compare_global_weights(dict1, dict2, layer_suffix="h.6.mlp.c_fc.weight"):
    """Compare a deep transformer layer to see if the models share a recent trajectory."""
    w1 = find_tensor_by_suffix(dict1, layer_suffix)
    w2 = find_tensor_by_suffix(dict2, layer_suffix)
    
    if w1 is None or w2 is None:
        print(f"  [Warning] Could not find layer '{layer_suffix}' for comparison.")
        return None, None
        
    cos_sim = torch.nn.functional.cosine_similarity(w1.flatten().unsqueeze(0), w2.flatten().unsqueeze(0)).item()
    l2_dist = torch.norm(w1 - w2).item()
    
    return cos_sim, l2_dist

def main():
    print("="*65)
    print("THE TRUE CHECKPOINT VERIFICATION AUDIT")
    print("="*65)
    
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_GPT2))
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    
    coconut_dict = get_clean_state_dict(COCONUT_GPT2)
    pause_dict = get_clean_state_dict(PAUSE_GPT2)
    
    # -------------------------------------------------------------------
    # TEST 1: The 'pause_embedding' Fingerprint
    # -------------------------------------------------------------------
    print("\n[TEST 1] Custom Pause Embedding Parameter Check")
    print("-" * 65)
    coconut_has_pause, coconut_drift = verify_pause_fingerprint(coconut_dict, tokenizer, "COCONUT")
    print("")
    pause_has_pause, pause_drift = verify_pause_fingerprint(pause_dict, tokenizer, "PAUSE")
    
    print("\nDIAGNOSIS:")
    if not coconut_has_pause and pause_has_pause and pause_drift > 0.1:
        print("✅ SUCCESS: The models are correctly labeled!")
        print("   - COCONUT doesn't have the pause_embedding (as expected).")
        print("   - PAUSE has the pause_embedding and it was actively trained.")
    elif not pause_has_pause:
        print("❌ CRITICAL ERROR: The PAUSE checkpoint is missing the 'pause_embedding'!")
        print("   This proves the repo owner uploaded a standard COCONUT model by mistake.")
    elif pause_has_pause and pause_drift < 1e-5:
        print("❌ CRITICAL ERROR: The PAUSE model has the parameter, but it was never trained!")
    
    # -------------------------------------------------------------------
    # TEST 2: Global Trajectory
    # -------------------------------------------------------------------
    print("\n[TEST 2] Global Weight Divergence (Middle Transformer Layer)")
    print("-" * 65)
    sim, dist = compare_global_weights(coconut_dict, pause_dict)
    
    if sim is not None:
        print(f"Cosine Similarity: {sim:.4f}")
        print(f"L2 Distance:       {dist:.4f}")

if __name__ == "__main__":
    main()