"""Static-cache draft step for LayerSkip drafting, captured as a CUDA Graph on CUDA.

PROTOTYPE. ``forward_early`` grows the KV cache by one position per draft
step, so its tensor shapes change every call and cannot be replayed as a
CUDA Graph. ``GraphDrafter`` instead keeps pre-allocated key/value buffers
of fixed length ``max_len`` for the draft layers. A one-token draft step then
has identical shapes every time: it writes its key/value at a position held
in a device tensor and masks attention beyond that position.

On CUDA the step is captured once and replayed, which removes the per-layer
kernel-launch overhead. Elsewhere (or with ``use_graph=False``) the same
static-cache code runs eagerly, which lets the maths be tested on CPU.

Only the draft layers use the buffers. The engine's legacy tuple cache stays
the source of truth: ``draft`` copies in positions the buffers do not hold
yet and returns a tuple cache extended with the drafted positions, exactly
like repeated ``forward_early`` calls.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# Warm-up runs on a side stream before capture, as required by torch.cuda.graph.
_WARMUP_STEPS = 3


class GraphDrafter:
    """Draft tokens through layers ``0..depth-1`` with a fixed-shape KV cache."""

    def __init__(self, model: torch.nn.Module, depth: int, max_len: int,
                 use_graph: Optional[bool] = None) -> None:
        layers = model.model.layers
        if not 1 <= depth < len(layers):
            raise ValueError("Draft depth must be in [1, number of layers).")
        self.model, self.depth, self.max_len = model, depth, max_len
        self.use_graph = (model.device.type == "cuda") if use_graph is None else use_graph
        if self.use_graph and model.device.type != "cuda":
            raise ValueError("CUDA Graph capture requires a CUDA model.")

        attention = layers[0].self_attn
        config = model.config
        kv_heads = config.num_key_value_heads
        dtype, device = next(model.parameters()).dtype, model.device
        shape = (1, kv_heads, max_len, attention.head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(depth)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(depth)]
        self.slots = torch.arange(max_len, device=device)

        # Static inputs/outputs: the captured graph always reads and writes these.
        self.token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.position = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.out_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.out_hidden = torch.zeros((1, 1, config.hidden_size), dtype=dtype, device=device)

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.valid_len = 0     # buffer positions known to equal the engine's cache
        self.drafted_until = 0

    # ── Engine-facing API ────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget buffer contents; call at the start of every request."""
        self.valid_len = self.drafted_until = 0

    def commit(self, cache_len: int) -> None:
        """Record the engine's cache length after a speculative cycle's crop.

        Drafted positions below ``cache_len`` were accepted, so the buffers
        still match the engine cache up to there.
        """
        self.valid_len = min(self.drafted_until, cache_len)

    @torch.inference_mode()
    def draft(self, cache: tuple, pending: torch.Tensor, steps: int,
              eos_t: Optional[torch.Tensor]) -> tuple[list[torch.Tensor], tuple, torch.Tensor]:
        """Run up to ``steps`` draft steps starting from ``pending``.

        Returns (proposed token tensors, tuple cache extended with the drafted
        positions for the draft layers, accumulated exit-query hidden states),
        matching the engine's ``forward_early`` loop.
        """
        start = cache[0][0].shape[2]
        if start + steps > self.max_len:
            raise ValueError(f"GraphDrafter max_len={self.max_len} is too small for "
                             f"{start + steps} positions.")
        self._sync(cache, start)

        proposed, hidden_states = [], []
        token = pending
        for step in range(steps):
            self.token.copy_(token)
            self.position.fill_(start + step)
            self._run_step()
            token = self.out_token.clone()
            proposed.append(token)
            hidden_states.append(self.out_hidden.clone())
            if eos_t is not None and torch.isin(token, eos_t).item():
                break

        drafted = len(proposed)
        self.drafted_until = start + drafted
        end = start + drafted
        extended = tuple(
            (torch.cat([k, self.keys[i][:, :, start:end]], dim=2),
             torch.cat([v, self.values[i][:, :, start:end]], dim=2))
            if i < self.depth else (k, v)
            for i, (k, v) in enumerate(cache)
        )
        return proposed, extended, torch.cat(hidden_states, dim=1)

    # ── Internals ────────────────────────────────────────────────────────

    def _sync(self, cache: tuple, length: int) -> None:
        """Copy cache positions the buffers do not hold yet (prefill, ordinary steps)."""
        first = min(self.valid_len, length)
        if first < length:
            for i in range(self.depth):
                self.keys[i][:, :, first:length].copy_(cache[i][0][:, :, first:length])
                self.values[i][:, :, first:length].copy_(cache[i][1][:, :, first:length])
        self.valid_len = length

    def _run_step(self) -> None:
        if not self.use_graph:
            self._step()
            return
        if self.graph is None:
            self._capture()
        self.graph.replay()

    def _capture(self) -> None:
        # Warm-up writes land at the current position, which _step overwrites
        # again on replay, so the buffers stay consistent.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_STEPS):
                self._step()
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def _step(self) -> None:
        """One draft token through the draft layers; fixed shapes, graph-safe."""
        model = self.model
        hidden = model.model.embed_tokens(self.token)
        cos, sin = model.model.rotary_emb(hidden, self.position)
        # Attend to every filled slot up to and including the current position.
        visible = (self.slots <= self.position.view(-1)).view(1, 1, 1, -1)
        write_at = self.position.view(-1)

        for i, layer in enumerate(model.model.layers[:self.depth]):
            attention = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)
            query = attention.q_proj(normed).view(1, 1, -1, attention.head_dim).transpose(1, 2)
            key = attention.k_proj(normed).view(1, 1, -1, attention.head_dim).transpose(1, 2)
            value = attention.v_proj(normed).view(1, 1, -1, attention.head_dim).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            self.keys[i].index_copy_(2, write_at, key)
            self.values[i].index_copy_(2, write_at, value)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                repeat_kv(self.keys[i], attention.num_key_value_groups),
                repeat_kv(self.values[i], attention.num_key_value_groups),
                attn_mask=visible, scale=attention.scaling,
            )
            attended = attended.transpose(1, 2).reshape(1, 1, -1)
            hidden = residual + attention.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        self.out_hidden.copy_(hidden)
        logits = model.lm_head(model.model.norm(hidden))
        self.out_token.copy_(logits[:, -1].argmax(-1, keepdim=True))
