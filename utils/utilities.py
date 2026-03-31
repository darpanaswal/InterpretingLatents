def clean_state_dict_keys(state_dict):
    """
    Strip FSDP / DDP wrapper prefixes from checkpoint keys.

    run.py saves checkpoints via parallel_model.state_dict() where
    parallel_model is FSDP(Coconut(...)) or DDP(Coconut(...)).

    For GPT-2, GPT2Block is commented out of the FSDP auto_wrap_policy
    (run.py line 175), so FSDP treats the whole model as a single flat
    module — effectively DDP. The resulting keys may have prefixes:

        FSDP:  _fsdp_wrapped_module.base_causallm.transformer.h.0...
        DDP:   module.base_causallm.transformer.h.0...
        Clean: base_causallm.transformer.h.0...

    The Coconut class expects keys starting with "base_causallm.".
    We strip any wrapper prefixes to normalize.
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_fsdp_wrapped_module."):
            new_k = new_k[len("_fsdp_wrapped_module."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned[new_k] = v
    return cleaned