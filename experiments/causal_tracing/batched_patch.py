# ═════════════════════════════════════════════════════════════════════
# Batched activation patching
# ═════════════════════════════════════════════════════════════════════
#
# Motivation
# ----------
# Per-instance trace runs ~n_layers * n_components * n_positions ≈ 1000
# patched forward passes. Each is a single-instance GPT-2 small forward,
# which is memory-bound and badly underutilizes a modern GPU. Sites
# differ only in WHERE the patch lands, not in the input tokens or
# sub-pass schedule, so we can stack B sites into one batched forward.
#
# Correctness invariant
# ---------------------
# For row b in a B-batched forward, the sequence of tensor ops executed
# is bit-identical to a B=1 forward on the same instance, except that at
# the hooked sites, row b's activations are overwritten with value_b.
# The transformer forward (GPT-2 or Llama) is pointwise across the batch
# dim everywhere except attention, which is itself pointwise across the
# batch dim under an identical mask and identical inputs. Thus
# broadcasting B identical copies of inputs_embeds and scattering per-row
# patches reproduces, for each row, exactly the unbatched patched forward
# for that site.
#
# The Coconut recurrence (h_{i+1} = h_i_last_hidden) is also per-row in
# the batched form: each row's recurrent_h is read from row b of the
# hidden states, so rows that diverged because of an earlier patch
# remain correctly diverged through the recurrence.
#
# What this module provides
# -------------------------
#   BatchedActivationPatcher: one hook per (layer, component), scatters
#     per-row patches at per-row local positions.
#
#   _batched_coconut_patched_forward / _batched_run_patched_forward_codi /
#   _batched_run_patched_forward: B-aware versions of the corresponding
#   single-row drivers in the parent module.
#
#   _batched_greedy_decode: vectorized greedy decode over batch dim.
#   _per_position_kl_batched: KL per (row, gen_position).
#
#   build_batched_site_specs: flatten (layer, component, position) sites
#   from trace_instance into per-row patch specs ready to feed into
#   BatchedActivationPatcher.
#
# Sites in this module mean: one row's patch target. A row's "spec" is:
#   {
#     "layers":     list[int]   target layer set (window or single)
#     "component":  str
#     "abs_positions":      list[int]
#     "values_per_abs_pos": dict[int -> Tensor(D,)]   keyed at the
#         resolved per-layer dimension; see _row_values_lookup below.
#   }
#
# Because different rows can target different (layer, component), we
# pre-compute per-row "active masks" inside BatchedActivationPatcher and
# only scatter the rows whose target lands on the current hook's
# (layer, component). Other rows pass through unchanged — same behavior
# the model would have given on a clean pass.
# ═════════════════════════════════════════════════════════════════════

import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────────────
# Architecture + KV-cache helpers (GPT-2 + Llama)
# ─────────────────────────────────────────────────────────────────────
#
# Kept local (not imported from causal_trace) to preserve this module's
# standalone-ness and avoid a circular import. Semantics mirror the
# parent module's helpers exactly:
#
#   GPT-2 block submodules : .attn  / .mlp
#   Llama block submodules : .self_attn / .mlp
#
# KV cache: GPT-2 returns a legacy tuple of per-layer (k, v) tensors
# shaped (batch, n_heads, seq_len, head_dim). Newer transformers returns
# a DynamicCache for Llama, which is neither subscriptable as [L][kv] nor
# rebuildable via `for k, v in cache`. The shim below normalizes both.
# ─────────────────────────────────────────────────────────────────────

def _resolve_submodule_names(block):
    """Return (attn_name, mlp_name) for one transformer block."""
    if hasattr(block, "attn"):
        attn_name = "attn"            # GPT-2
    elif hasattr(block, "self_attn"):
        attn_name = "self_attn"       # Llama
    else:
        raise AttributeError(
            f"Block {type(block).__name__} has neither .attn nor .self_attn"
        )
    return attn_name, "mlp"


def _kv_to_legacy(past_kv):
    """List of (key, value) tensor pairs regardless of cache type."""
    if past_kv is None:
        return None
    if hasattr(past_kv, "to_legacy_cache"):
        return [(k, v) for (k, v) in past_kv.to_legacy_cache()]
    return [(k, v) for (k, v) in past_kv]


def _kv_seq_len(past_kv):
    """Current cached sequence length (axis-2 of any layer's key tensor)."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    return int(past_kv[0][0].shape[2])


def _kv_trim(past_kv, n):
    """Trim every layer's K/V to the first `n` positions, returning a
    cache of the SAME type the model produced (Cache -> DynamicCache,
    legacy -> list of trimmed (k, v) pairs)."""
    legacy = _kv_to_legacy(past_kv)
    trimmed = tuple((k[:, :, :n, :], v[:, :, :n, :]) for (k, v) in legacy)
    if hasattr(past_kv, "to_legacy_cache"):
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(trimmed)
    return list(trimmed)


# ─────────────────────────────────────────────────────────────────────
# BatchedActivationPatcher
# ─────────────────────────────────────────────────────────────────────
#
# The patcher is constructed with:
#   blocks            : list of GPT-2 transformer blocks
#   batch_size B      : number of patch sites packed into one forward
#   row_specs         : list[B] of dicts with keys
#                       "layers" (list[int]), "component" (str),
#                       "abs_positions" (list[int]),
#                       "values_per_layer_abs"
#                         (dict[(layer, abs_pos) -> Tensor(D,)])
#
# Storage rationale
# -----------------
# values_per_layer_abs is keyed by (layer, abs_pos). For single sites
# (one position), it has |layers| entries. For joint sites, it has
# |layers| * |abs_positions| entries. The recorder cache provides the
# clean activation at each (layer, component, abs_pos); the row-spec
# builder pulls them out once and stores per-(layer, abs_pos).
#
# Sub-pass tracking
# -----------------
# Caller invokes set_pass_offset(offset, length) at the start of each
# sub-pass; the hook then writes any (row, abs_pos) whose abs_pos falls
# inside [offset, offset+length).
# ─────────────────────────────────────────────────────────────────────


class BatchedActivationPatcher:
    def __init__(self, blocks, batch_size, row_specs):
        self.blocks = blocks
        self.B = batch_size
        self.row_specs = row_specs
        assert len(row_specs) == batch_size, "row_specs length must match batch_size"

        self.handles = []
        self.current_offset = 0
        self.current_length = 0

        # Pre-compute, for each (layer, component), the list of
        #   (row_idx, abs_pos, value_tensor)
        # entries. This avoids re-iterating row_specs on every hook call.
        # Layout: self._index[(layer, component)] -> list of (b, abs_pos, value)
        self._index = {}
        for b, spec in enumerate(row_specs):
            comp = spec["component"]
            for L in spec["layers"]:
                key = (L, comp)
                bucket = self._index.setdefault(key, [])
                for ap in spec["abs_positions"]:
                    v = spec["values_per_layer_abs"][(L, ap)]
                    bucket.append((b, ap, v))

    def set_pass_offset(self, offset, length):
        self.current_offset = offset
        self.current_length = length

    # Build the per-call write plan for the active sub-pass:
    #   returns list of (row_idx, local_pos, value_tensor) for entries
    #   whose abs_pos lands inside [offset, offset+length).
    def _writes_for(self, layer, component):
        bucket = self._index.get((layer, component), None)
        if not bucket:
            return None
        off, ln = self.current_offset, self.current_length
        out = []
        for (b, ap, v) in bucket:
            lp = ap - off
            if 0 <= lp < ln:
                out.append((b, lp, v))
        return out if out else None

    # ─────────────────────────────────────────────────────────────────
    # Scatter helper
    # ─────────────────────────────────────────────────────────────────
    #
    # # Math: for each write entry (b, lp, v):
    # #   new[b, lp, :] = v
    # # Other (b, lp) entries are untouched. We must clone() to avoid
    # # mutating the autograd graph state of the original tensor — same
    # # safety reason as the unbatched ActivationPatcher.
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _scatter(tensor, writes):
        new = tensor.clone()
        # Cheap path: collect indices and values and do a single
        # advanced-indexing assignment. We stage them into tensors built
        # on the same device/dtype as `new`.
        rows = torch.tensor([w[0] for w in writes], device=new.device, dtype=torch.long)
        cols = torch.tensor([w[1] for w in writes], device=new.device, dtype=torch.long)
        vals = torch.stack([w[2].to(dtype=new.dtype, device=new.device) for w in writes], dim=0)
        new[rows, cols, :] = vals
        return new

    def _make_block_pre_hook(self, layer):
        def hook(module, inputs):
            writes = self._writes_for(layer, "resid_pre")
            if writes is None:
                return None
            h = inputs[0]
            new_h = self._scatter(h, writes)
            return (new_h,) + inputs[1:]
        return hook

    def _make_attn_hook(self, layer):
        def hook(module, inputs, output):
            writes = self._writes_for(layer, "attn_out")
            if writes is None:
                return output
            if isinstance(output, tuple):
                return (self._scatter(output[0], writes),) + output[1:]
            return self._scatter(output, writes)
        return hook

    def _make_mlp_hook(self, layer):
        def hook(module, inputs, output):
            writes = self._writes_for(layer, "mlp_out")
            if writes is None:
                return output
            if isinstance(output, tuple):
                return (self._scatter(output[0], writes),) + output[1:]
            return self._scatter(output, writes)
        return hook

    def attach(self):
        for L, block in enumerate(self.blocks):
            attn_name, mlp_name = _resolve_submodule_names(block)
            self.handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(L)))
            self.handles.append(getattr(block, attn_name).register_forward_hook(self._make_attn_hook(L)))
            self.handles.append(getattr(block, mlp_name).register_forward_hook(self._make_mlp_hook(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @contextmanager
    def patching(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()


# ─────────────────────────────────────────────────────────────────────
# Row-spec builder
# ─────────────────────────────────────────────────────────────────────
#
# Translate a flat list of trace sites
#   (L_center, comp, pi, abs_pos, kind, slab_or_none)
# into per-row patch specs that BatchedActivationPatcher consumes.
#
# Args
# ----
# sites              : list of site descriptors (one per row in the batch)
# layers_for_site_fn : L_center -> list[int]   (single or window)
# lookup_clean_value : (layer, component, abs_pos) -> Tensor(D,) or None
# joint_positions    : dict[label -> list[abs_pos]]  (from registry)
#
# Returns
# -------
# (specs, skipped_indices)
#   specs           : list of row-spec dicts; length == len(sites) - len(skipped)
#   skipped_indices : list[int] indices into `sites` that could not be
#                     materialized because lookup_clean_value returned
#                     None for some (layer, abs_pos). Caller must mark
#                     these as NaN in the output array, exactly as the
#                     unbatched path skips them.
# ─────────────────────────────────────────────────────────────────────

def build_batched_row_specs(sites, layers_for_site_fn, lookup_clean_value,
                            joint_positions):
    specs = []
    kept_site_indices = []
    skipped_indices = []
    for si, (L_center, comp, pi, abs_pos, kind, label) in enumerate(sites):
        target_layers = layers_for_site_fn(L_center)
        is_joint = kind in ("joint_prompt", "joint_thought")
        if is_joint:
            slab = joint_positions[label]
            if not slab:
                skipped_indices.append(si)
                continue
            abs_positions = list(slab)
        else:
            abs_positions = [abs_pos]

        values = {}
        ok = True
        for tL in target_layers:
            for ap in abs_positions:
                v = lookup_clean_value(tL, comp, ap)
                if v is None:
                    ok = False
                    break
                values[(tL, ap)] = v
            if not ok:
                break
        if not ok:
            skipped_indices.append(si)
            continue

        specs.append({
            "layers": list(target_layers),
            "component": comp,
            "abs_positions": abs_positions,
            "values_per_layer_abs": values,
        })
        kept_site_indices.append(si)
    return specs, kept_site_indices, skipped_indices


# ─────────────────────────────────────────────────────────────────────
# Batched greedy decode
# ─────────────────────────────────────────────────────────────────────
#
# # Math: per row b, per step j (mirror of unbatched greedy_decode):
# #   dist[b, j, :] = softmax(logits[b, :V_eff])
# #   token[b, j]   = argmax(logits[b, :V_eff])
# # KV cache advances by one token per step; same as unbatched, but
# # batch dim B everywhere.
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batched_greedy_decode(base_model, first_logits, past_kv, n_steps, device,
                           vocab_limit=None):
    """
    Inputs
    ------
    first_logits : (B, V) at A_b
    past_kv      : KV cache built with batch dim B
    n_steps      : int

    Returns
    -------
    generated_ids : LongTensor (B, n_steps)
    distributions : FloatTensor (B, n_steps, V_eff)   fp32
    """
    def _slice(logits):
        if vocab_limit is None:
            return logits
        return logits[..., :vocab_limit - 1]

    cur_logits = _slice(first_logits)               # (B, V_eff)
    B, V_eff = cur_logits.shape

    distributions = torch.empty((B, n_steps, V_eff), dtype=torch.float32, device=device)
    generated_ids = torch.empty((B, n_steps), dtype=torch.long, device=device)

    for j in range(n_steps):
        dist_j = F.softmax(cur_logits.float(), dim=-1)
        distributions[:, j, :] = dist_j
        tok_j = cur_logits.argmax(dim=-1)           # (B,)
        generated_ids[:, j] = tok_j

        if j == n_steps - 1:
            break

        past_len = _kv_seq_len(past_kv)
        attn_mask = torch.ones((B, past_len + 1), dtype=torch.long, device=device)
        position_ids = torch.full((B, 1), past_len, dtype=torch.long, device=device)
        input_ids = tok_j.unsqueeze(1)              # (B, 1)

        outputs = base_model(
            input_ids=input_ids,
            past_key_values=past_kv,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        cur_logits = _slice(outputs.logits[:, -1, :])

    return generated_ids, distributions


# ─────────────────────────────────────────────────────────────────────
# Batched per-position KL
# ─────────────────────────────────────────────────────────────────────
#
# # Math (same as the unbatched _per_position_kl, vectorized over B):
# #   KL(P_clean^(j) || P_patched^(b, j)) =
# #       sum_v exp(log_p_clean[j,v]) * (log_p_clean[j,v] - log_q[b,j,v])
# # Implemented via F.kl_div with log_target=True and target=log_p_clean
# # broadcast across B.
# ─────────────────────────────────────────────────────────────────────

def _per_position_kl_batched(p_clean_dists, p_other_dists):
    """
    Inputs (fp32):
        p_clean_dists : (N, V)
        p_other_dists : (B, N, V)
    Returns numpy (B, N) fp32.
    """
    log_p = torch.log(p_clean_dists.clamp_min(1e-30))      # (N, V)
    log_q = torch.log(p_other_dists.clamp_min(1e-30))      # (B, N, V)
    log_p_b = log_p.unsqueeze(0).expand_as(log_q)          # (B, N, V)
    kl = F.kl_div(log_q, log_p_b, log_target=True, reduction='none').sum(dim=-1)
    return kl.detach().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Batched Coconut/PaT patched forward
# ─────────────────────────────────────────────────────────────────────
#
# Mirror of _coconut_patched_forward_inline but with batch dim B. The
# clean grid (`input_ids`, partner-substituted `embeds`) is identical
# across rows; only the patcher writes differ per row.
#
# Important: when we tile embeds to (B, seq, D), we must clone to ensure
# the Coconut recurrence's in-place write at next_latent_pos remains
# per-row (the recurrent_h read from outputs.hidden_states is per-row,
# so we want the assignment to also stay per-row, which it does since
# embeds has batch dim B).
# ─────────────────────────────────────────────────────────────────────

def _tile_embeds(embeds, B):
    # embeds : (1, seq, D) -> (B, seq, D), independent storage.
    return embeds.expand(B, -1, -1).contiguous()


def _is_pause_model_safe(coconut_model):
    # Defensive — match the parent module's helper without importing it
    # (to avoid circular import if this file is split out).
    return hasattr(coconut_model, "pause_embedding")


@torch.no_grad()
def _batched_coconut_patched_forward(
    coconut_model, base_model, input_ids, embeds, n_thoughts, device,
    batched_patcher,
):
    """
    Batched analogue of _coconut_patched_forward_inline.

    Inputs (single-row in shape; we tile internally):
        input_ids : (1, seq)         on device
        embeds    : (1, seq, D)      on device (already partner-substituted)
        batched_patcher : BatchedActivationPatcher with .B = B

    Returns:
        ans_logits : (B, V)
        past_kv    : KV cache with batch dim B
    """
    B = batched_patcher.B
    seq_len = input_ids.shape[1]

    # Tile to (B, seq, D); independent storage so the recurrence write
    # at next_latent_pos doesn't alias across rows.
    embeds_b = _tile_embeds(embeds, B)

    is_pause = _is_pause_model_safe(coconut_model)

    if is_pause:
        batched_patcher.set_pass_offset(0, seq_len)
        pause_emb = coconut_model.pause_embedding
        latent_positions = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()
        for pos in latent_positions:
            embeds_b[:, pos, :] = pause_emb
        with batched_patcher.patching():
            outputs = base_model(
                inputs_embeds=embeds_b,
                attention_mask=torch.ones((B, seq_len), device=device, dtype=torch.long),
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1),
                use_cache=True,
            )
        return outputs.logits[:, -1, :], outputs.past_key_values

    # Continuous Coconut
    latent_indices = (input_ids[0] == coconut_model.latent_token_id).nonzero().squeeze(-1).tolist()

    if len(latent_indices) == 0:
        batched_patcher.set_pass_offset(0, seq_len)
        with batched_patcher.patching():
            outputs = base_model(
                inputs_embeds=embeds_b,
                attention_mask=torch.ones((B, seq_len), device=device, dtype=torch.long),
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1),
                use_cache=True,
            )
        return outputs.logits[:, -1, :], outputs.past_key_values

    next_compute_range = (0, latent_indices[0])
    kv_cache = None
    last_step_logits = None
    max_n_latents = len(latent_indices)
    pass_idx = 0

    with batched_patcher.patching():
        while True:
            cur_start, cur_end = next_compute_range
            cur_len = cur_end - cur_start
            batched_patcher.set_pass_offset(cur_start, cur_len)

            attn_mask = torch.ones((B, cur_end), device=device, dtype=torch.long)
            pos_ids  = torch.arange(cur_start, cur_end, device=device).unsqueeze(0).expand(B, -1)

            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=embeds_b[:, cur_start:cur_end, :],
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    use_cache=True, output_hidden_states=True,
                )
            else:
                # Trim KV to cur_start (matches single-row driver).
                past_kv_trim = _kv_trim(kv_cache, cur_start)
                outputs = base_model(
                    inputs_embeds=embeds_b[:, cur_start:cur_end, :],
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=past_kv_trim, use_cache=True,
                    output_hidden_states=True,
                )
            kv_cache = outputs.past_key_values
            last_step_logits = outputs.logits[:, -1, :]    # (B, V)

            if pass_idx + 1 >= max_n_latents:
                final_start = cur_end
                final_end = seq_len
                if final_end > final_start:
                    batched_patcher.set_pass_offset(final_start, final_end - final_start)
                    past_kv_trim = _kv_trim(kv_cache, final_start)
                    attn_mask = torch.ones((B, final_end), device=device, dtype=torch.long)
                    pos_ids  = torch.arange(final_start, final_end, device=device).unsqueeze(0).expand(B, -1)
                    outputs = base_model(
                        inputs_embeds=embeds_b[:, final_start:final_end, :],
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        past_key_values=past_kv_trim, use_cache=True,
                        output_hidden_states=True,
                    )
                    kv_cache = outputs.past_key_values
                    last_step_logits = outputs.logits[:, -1, :]
                break

            # Recurrence — per-row hidden state at the local position.
            next_latent_pos = latent_indices[pass_idx]
            hidden_states = outputs.hidden_states[-1]      # (B, cur_len, D)
            local_pos = next_latent_pos - 1 - cur_start
            recurrent_h = hidden_states[:, local_pos, :]   # (B, D)
            embeds_b[:, next_latent_pos, :] = recurrent_h
            next_compute_range = (next_latent_pos, next_latent_pos + 1)
            pass_idx += 1

    return last_step_logits, kv_cache


# ─────────────────────────────────────────────────────────────────────
# Batched CODI patched forward
# ─────────────────────────────────────────────────────────────────────
#
# Mirror of _run_patched_forward_codi with batch dim B.
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batched_run_patched_forward_codi(codi_dict, sample, partner_sample, n_thoughts, device,
                                      *, batched_patcher):
    model = codi_dict["model"]
    prj = codi_dict["prj"]
    tokenizer = codi_dict["tokenizer"]
    bot_id = codi_dict["bot_id"]
    eot_id = codi_dict["eot_id"]
    embedding_fn = codi_dict["embedding_fn"]
    use_prj = codi_dict["use_prj"]
    remove_eos = codi_dict["remove_eos"]
    B = batched_patcher.B

    # Clean prompt grid (B=1) then tile.
    question_tokens = tokenizer.encode(
        sample["question"].strip().replace("  ", " "), add_special_tokens=True,
    )
    if remove_eos:
        ids = question_tokens + [bot_id]
    else:
        ids = question_tokens + [tokenizer.eos_token_id, bot_id]
    input_ids = torch.tensor([ids], device=device)
    L = input_ids.size(1)
    embeds = embedding_fn(input_ids).clone()

    # Partner substitution (same logic as single-row).
    partner_q_tokens = tokenizer.encode(
        partner_sample["question"].strip().replace("  ", " "), add_special_tokens=True,
    )
    partner_ids = torch.tensor([partner_q_tokens], device=device)
    partner_embeds = embedding_fn(partner_ids).clone()
    q_len = L - 1
    if partner_embeds.shape[1] >= q_len:
        embeds[0, :q_len, :] = partner_embeds[0, :q_len, :]
    else:
        mean_e = partner_embeds[0].mean(dim=0)
        embeds[0, :partner_embeds.shape[1], :] = partner_embeds[0]
        for p in range(partner_embeds.shape[1], q_len):
            embeds[0, p, :] = mean_e

    embeds_b = _tile_embeds(embeds, B)

    attention_mask = torch.ones((B, L), device=device, dtype=torch.long)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    with batched_patcher.patching():
        # Sub-pass 0
        batched_patcher.set_pass_offset(0, L)
        outputs = model(
            inputs_embeds=embeds_b, use_cache=True, output_hidden_states=True,
            attention_mask=attention_mask, position_ids=position_ids,
        )
        past_kv = outputs.past_key_values
        h = outputs.hidden_states[-1][:, -1, :]            # (B, D)
        latent = h.unsqueeze(1)                            # (B, 1, D)
        if use_prj and prj is not None:
            latent = prj(latent)
        running_mask = attention_mask

        for t in range(1, n_thoughts + 1):
            running_mask = torch.cat(
                [running_mask,
                 torch.ones((B, 1), dtype=running_mask.dtype, device=device)],
                dim=1,
            )
            pos_t = torch.full((B, 1), L + t - 1, dtype=torch.long, device=device)
            batched_patcher.set_pass_offset(L + t - 1, 1)
            outputs = model(
                inputs_embeds=latent, use_cache=True, output_hidden_states=True,
                past_key_values=past_kv, attention_mask=running_mask,
                position_ids=pos_t,
            )
            past_kv = outputs.past_key_values
            h = outputs.hidden_states[-1][:, -1, :]
            latent = h.unsqueeze(1)
            if use_prj and prj is not None:
                latent = prj(latent)

        # eot pass
        if remove_eos:
            eot_row = [eot_id]
        else:
            eot_row = [eot_id, tokenizer.eos_token_id]
        eot_ids = torch.tensor([eot_row], device=device)
        eot_emb = embedding_fn(eot_ids).expand(B, -1, -1).contiguous()
        eot_len = eot_emb.size(1)
        eot_pos = torch.arange(L + n_thoughts, L + n_thoughts + eot_len, device=device).unsqueeze(0).expand(B, -1)
        running_mask = torch.cat(
            [running_mask, torch.ones((B, eot_len), dtype=running_mask.dtype, device=device)], dim=1,
        )
        batched_patcher.set_pass_offset(L + n_thoughts, eot_len)
        outputs = model(
            inputs_embeds=eot_emb, use_cache=True, past_key_values=past_kv,
            attention_mask=running_mask, position_ids=eot_pos,
            output_hidden_states=True,
        )

    return outputs.logits[:, -1, :], outputs.past_key_values


# ─────────────────────────────────────────────────────────────────────
# Dispatcher for batched patched forward
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batched_run_patched_forward(
    is_codi, coconut_model, base_model, tokenizer, codi_dict,
    sample, partner_sample, n_thoughts, device,
    start_id, latent_id, end_id,
    *, batched_patcher,
    build_coconut_inputs_fn,
):
    """
    B-aware mirror of _run_patched_forward. Uses the parent module's
    build_coconut_inputs (passed in) so we don't duplicate prompt-building
    logic here.
    """
    if is_codi:
        return _batched_run_patched_forward_codi(
            codi_dict, sample, partner_sample, n_thoughts, device,
            batched_patcher=batched_patcher,
        )

    # Coconut/PaT
    input_ids, q_len = build_coconut_inputs_fn(
        coconut_model, tokenizer, sample, n_thoughts, device,
        start_id, latent_id, end_id,
    )
    embedding = coconut_model.embedding
    embeds = embedding(input_ids).clone()

    # Partner substitution. Must use the SAME tokenization as the clean
    # grid (build_coconut_inputs -> tokenize_question_for_recurrence:
    # chat template for instruct Llama, raw "{q}\n" for GPT-2). Raw encode
    # here would be format-OOD for instruct Llama and misalign prompt
    # lengths against the clean grid. Imported from src.utils (no circular
    # import: utils does not import this module).
    from src.utils import tokenize_question_for_recurrence
    partner_q_tokens = tokenize_question_for_recurrence(
        tokenizer, partner_sample["question"]
    )
    partner_ids = torch.tensor([partner_q_tokens], device=device)
    partner_embeds = embedding(partner_ids).clone()
    if partner_embeds.shape[1] >= q_len:
        embeds[0, :q_len, :] = partner_embeds[0, :q_len, :]
    else:
        mean_e = partner_embeds[0].mean(dim=0)
        embeds[0, :partner_embeds.shape[1], :] = partner_embeds[0]
        for p in range(partner_embeds.shape[1], q_len):
            embeds[0, p, :] = mean_e

    return _batched_coconut_patched_forward(
        coconut_model, base_model, input_ids, embeds, n_thoughts, device,
        batched_patcher,
    )