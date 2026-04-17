"""
D1 named-object geometry slot module.

Implements the minimal slot encoder + slot-attention + slot→decoder interface
used by R2 (and R3 ablation) in the staged hybrid V3 plan.

Inputs per sample (already matched upstream by pilot_d1_slots.py):
- visual_features: (K, d_vis) box-pooled features from the Qwen3-VL vision tower
- box_coords:      (K, 5)     [x1, y1, x2, y2, area], all in [0,1]
- scalar_depth:    (K, 1)     mean Depth Anything V2 depth inside the box
- K = number of matched named objects (≤ K_MAX = 6)

Outputs:
- slot_embeds:     (K, d_model) projected to the Qwen3-VL LLM hidden size, ready
                   for prepend into the inputs_embeds sequence
- pair_logits:     (K*(K-1),)   pair-order BCE targets; used only by the
                                auxiliary head, not returned from forward()

Design notes:
- Slot attention is intentionally tiny (1 layer, 4 heads, d_slot=256) so that
  the total trainable footprint stays under ~4M params on top of the V2 LoRA.
- The gate is NOT implemented here; pilot_d1_slots.py decides per-sample
  gate_on/off and either calls this module or skips it entirely.
- This module does NOT know about Qwen3-VL internals; pilot_d1_slots.py owns
  the forward-pass splice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX = 6
D_SLOT = 256
N_HEADS = 4
BOX_FEAT_DIM = 5  # [x1, y1, x2, y2, area]
DEPTH_FEAT_DIM = 1


class SlotEncoder(nn.Module):
    """2-layer MLP: (visual, box, depth) → d_slot."""

    def __init__(self, d_vis: int, d_slot: int = D_SLOT):
        super().__init__()
        self.d_vis = d_vis
        self.d_slot = d_slot
        self.fc1 = nn.Linear(d_vis + BOX_FEAT_DIM + DEPTH_FEAT_DIM, d_slot * 2)
        self.fc2 = nn.Linear(d_slot * 2, d_slot)

    def forward(
        self,
        visual: torch.Tensor,  # (B, K, d_vis)
        box_coords: torch.Tensor,  # (B, K, 5)
        depths: torch.Tensor,  # (B, K, 1)
    ) -> torch.Tensor:
        x = torch.cat([visual, box_coords, depths], dim=-1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class SlotAttention(nn.Module):
    """One layer of MultiheadAttention over slots, with residual + LN."""

    def __init__(self, d_slot: int = D_SLOT, n_heads: int = N_HEADS, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_slot,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_slot)

    def forward(
        self,
        slots: torch.Tensor,  # (B, K, d_slot)
        slot_mask: torch.Tensor,  # (B, K) bool, True = valid slot
    ) -> torch.Tensor:
        # key_padding_mask: True means *ignore*, so invert slot_mask
        key_padding_mask = ~slot_mask
        out, _ = self.attn(slots, slots, slots, key_padding_mask=key_padding_mask)
        return self.ln(slots + out)


class SlotToDecoder(nn.Module):
    """Project slots up to Qwen3-VL hidden size before splicing into inputs_embeds."""

    def __init__(self, d_slot: int, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_slot, d_model)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.fc(slots)


class PairOrderHead(nn.Module):
    """BCE head on ordered pair embeddings: predict 1[d_i < d_j] from (slot_i, slot_j)."""

    def __init__(self, d_slot: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d_slot * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # slots: (B, K, d_slot)
        B, K, D = slots.shape
        if K < 2:
            return slots.new_zeros((B, 0))
        # Build all ordered pairs (i, j) with i != j
        i_idx, j_idx = torch.meshgrid(
            torch.arange(K, device=slots.device),
            torch.arange(K, device=slots.device),
            indexing="ij",
        )
        mask = i_idx != j_idx
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        pair_feat = torch.cat([slots[:, i_flat], slots[:, j_flat]], dim=-1)  # (B, P, 2D)
        x = F.gelu(self.fc1(pair_feat))
        return self.fc2(x).squeeze(-1)  # (B, P)


class D1SlotModule(nn.Module):
    """Full D1 slot path: encoder + slot-attention + projector + pair-aux head.

    Total trainable footprint target: ~3-4M parameters depending on d_vis and
    d_model (Qwen3-VL-8B hidden size is 3584).
    """

    def __init__(self, d_vis: int, d_model: int, d_slot: int = D_SLOT, n_heads: int = N_HEADS):
        super().__init__()
        self.d_vis = d_vis
        self.d_model = d_model
        self.d_slot = d_slot
        self.encoder = SlotEncoder(d_vis, d_slot)
        self.attn = SlotAttention(d_slot, n_heads)
        self.pair_head = PairOrderHead(d_slot)
        # SlotToDecoder is NOT used in Strategy B (adapter handles d_slot→d_model).
        # Omitted entirely so it doesn't inflate the trainable param count.

    def forward(
        self,
        visual: torch.Tensor,       # (B, K, d_vis)
        box_coords: torch.Tensor,   # (B, K, 5)
        depths: torch.Tensor,       # (B, K, 1)
        slot_mask: torch.Tensor,    # (B, K) bool
    ):
        slots = self.encoder(visual, box_coords, depths)   # (B, K, d_slot)
        slots = self.attn(slots, slot_mask)                  # (B, K, d_slot)
        pair_logits = self.pair_head(slots)                  # (B, P)
        # Return slots at d_slot (for the cross-attn adapter) — NOT projected.
        # SlotToDecoder.projector is no longer used in Strategy B; the adapter
        # handles the d_slot → d_model interface internally via q_down/out_up.
        return slots, pair_logits

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DepthTokenModule(nn.Module):
    """R4 ablation: minimal depth-token path (no vision features, no slot attention).

    Input per object: box_coords (5) + depth (1) = 6 dims.
    2-layer MLP → d_slot. No slot-attention layer. Same pair-order head.
    This tests whether the slot structure (encoder + attention over vision
    features) matters, or whether raw depth/box tokens suffice.
    """

    def __init__(self, d_slot: int = D_SLOT):
        super().__init__()
        self.d_slot = d_slot
        input_dim = BOX_FEAT_DIM + DEPTH_FEAT_DIM  # 5 + 1 = 6
        self.fc1 = nn.Linear(input_dim, d_slot)
        self.fc2 = nn.Linear(d_slot, d_slot)
        self.pair_head = PairOrderHead(d_slot)

    def forward(
        self,
        box_coords: torch.Tensor,   # (B, K, 5)
        depths: torch.Tensor,       # (B, K, 1)
        slot_mask: torch.Tensor,    # (B, K) bool
    ):
        x = torch.cat([box_coords, depths], dim=-1)  # (B, K, 6)
        x = F.gelu(self.fc1(x))
        tokens = self.fc2(x)                          # (B, K, d_slot)
        pair_logits = self.pair_head(tokens)
        return tokens, pair_logits

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Strategy B: lightweight cross-attention adapter for forward-hook injection
# ---------------------------------------------------------------------------

class SlotCrossAttentionAdapter(nn.Module):
    """Lightweight cross-attention adapter injected via a forward hook on
    the model's final LayerNorm (before lm_head).

    Design goals:
    - Queries come from decoder hidden states (projected down to d_slot).
    - Keys/Values come from slot_embeds (already at d_slot from the slot path).
    - Small projections keep the param budget under ~0.5M.
    - Output is projected back up to d_model and added as a residual.

    On ungated samples, this module is never called — the hook checks a
    per-forward flag and returns the hidden state unchanged if the flag is off.
    This preserves **exact R1 behavior on ungated samples**.

    Param budget at d_model=3584, d_slot=256, n_heads=4:
      q_proj:   3584 * 256 = ~918K
      out_proj: 256 * 3584 = ~918K
      (k/v projections are identity from d_slot, so free)
      gate_scalar: 1
      Total: ~1.84M — well within budget.

    Actually we can shrink further by projecting queries through a bottleneck:
      q_down:  3584 * 256  = 918K
      out_up:  256 * 3584  = 918K
      Total:   ~1.84M
    This is the same as above. Acceptable.
    """

    def __init__(self, d_model: int, d_slot: int = D_SLOT, n_heads: int = N_HEADS):
        super().__init__()
        self.d_model = d_model
        self.d_slot = d_slot
        self.n_heads = n_heads

        self.q_down = nn.Linear(d_model, d_slot, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_slot,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.out_up = nn.Linear(d_slot, d_model, bias=False)
        self.gate_scalar = nn.Parameter(torch.zeros(1))
        self.ln = nn.LayerNorm(d_slot)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, L, d_model)
        slot_embeds: torch.Tensor,     # (B, K, d_slot) — from SlotToDecoder is d_model actually
        slot_mask: torch.Tensor,       # (B, K) bool, True=valid
    ) -> torch.Tensor:
        """Returns the residual to add to hidden_states (same shape)."""
        q = self.q_down(hidden_states)              # (B, L, d_slot)
        kv = slot_embeds                            # (B, K, d_slot)
        key_padding_mask = ~slot_mask               # True = ignore
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        attn_out = self.ln(attn_out)                # (B, L, d_slot)
        residual = self.out_up(attn_out)            # (B, L, d_model)
        return torch.sigmoid(self.gate_scalar) * residual

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SlotHookManager:
    """Manages the forward hook on model.language_model.model.norm that
    injects the cross-attention residual for gated samples.

    Usage:
        hook_mgr = SlotHookManager(model, adapter)
        hook_mgr.register()
        # For each forward pass:
        hook_mgr.set_slot_context(slot_embeds, slot_mask)  # or None for ungated
        outputs = model(**inputs, labels=labels)
        hook_mgr.clear_slot_context()

    On ungated samples, call set_slot_context(None, None) or simply don't call
    it — the hook checks for None and returns the hidden state unchanged,
    preserving exact R1 behavior.
    """

    def __init__(self, model: nn.Module, adapter: SlotCrossAttentionAdapter):
        self.model = model
        self.adapter = adapter
        self._hook_handle = None
        self._slot_embeds: torch.Tensor | None = None
        self._slot_mask: torch.Tensor | None = None

    def _find_final_norm(self) -> nn.Module:
        """Locate the final LayerNorm before lm_head.
        Tries common paths for raw Qwen3-VL and PEFT-wrapped variants."""
        candidates = [
            "base_model.model.model.language_model.norm",
            "base_model.model.model.norm",
            "base_model.model.language_model.model.norm",
            "model.model.language_model.norm",
            "model.model.norm",
            "model.language_model.model.norm",
            "language_model.model.norm",
            "model.norm",
        ]
        def _is_norm(m):
            return isinstance(m, (nn.LayerNorm,)) or "RMSNorm" in type(m).__name__

        for path in candidates:
            parts = path.split(".")
            mod = self.model
            try:
                for p in parts:
                    mod = getattr(mod, p)
                if _is_norm(mod):
                    return mod
            except (AttributeError, TypeError):
                continue
        norm_modules = [
            (n, type(m).__name__) for n, m in self.model.named_modules()
            if "norm" in n.lower() and _is_norm(m)
        ]
        raise RuntimeError(
            f"Cannot locate final norm. Found these norm modules: {norm_modules}"
        )

    def register(self):
        norm = self._find_final_norm()
        self._hook_handle = norm.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def set_slot_context(
        self,
        slot_embeds: torch.Tensor | None,
        slot_mask: torch.Tensor | None,
    ):
        self._slot_embeds = slot_embeds
        self._slot_mask = slot_mask

    def clear_slot_context(self):
        self._slot_embeds = None
        self._slot_mask = None

    def _hook_fn(self, module, input, output):
        if self._slot_embeds is None:
            return output
        residual = self.adapter(output, self._slot_embeds, self._slot_mask)
        return output + residual
