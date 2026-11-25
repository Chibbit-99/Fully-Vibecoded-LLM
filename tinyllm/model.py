"""From-scratch GPT (decoder-only transformer).

Implemented without any HuggingFace code: multi-head causal self-attention,
pre-LayerNorm blocks, GELU-tanh MLP, weight-tied output head.

The forward pass is kept deliberately plain (no flash-attention, no fused
kernels) so that ``web/engine.js`` and ``tinyllm/numpy_engine.py`` can mirror it
operation for operation and produce numerically close logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        d = config.n_embd
        self.c_attn = nn.Linear(d, 3 * d, bias=config.bias)  # qkv fused
        self.c_proj = nn.Linear(d, d, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, past: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past is not None:  # incremental decoding: append cached k/v
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        new_past = (k, v)

        Tq, Tk = q.size(2), k.size(2)
        # causal mask only needed when we feed more than one query token
        mask = None
        if Tq > 1:
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(
                diagonal=Tk - Tq
            )
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_past


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh approximation of GELU -- identical formula used in numpy / js
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.ln_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.ln_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past: torch.Tensor | None = None):
        attn_out, new_past = self.attn(self.ln_1(x), past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_past


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.ln_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
            n -= self.wte.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list | None = None,
    ):
        B, T = idx.shape
        device = idx.device
        past = None if past_key_values is None else past_key_values[0]
        if past is not None and past[0].size(2) + T > self.config.block_size:
            # context window is full: drop the oldest cached positions
            keep = self.config.block_size - T
            past_key_values = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in past_key_values]
            past = past_key_values[0]
        pos_offset = 0 if past is None else past[0].size(2)
        if pos_offset + T > self.config.block_size:
            raise ValueError("sequence longer than block_size")
        pos = torch.arange(pos_offset, pos_offset + T, device=device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        new_past = []
        for i, block in enumerate(self.blocks):
            block_past = None if past_key_values is None else past_key_values[i]
            x, kv = block(x, block_past)
            new_past.append(kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_past

    # -- checkpoint helpers -------------------------------------------------
    def state_dict_for_export(self) -> dict:
        """Flat, engine-friendly state dict (numpy-able values)."""
        sd = {}
        sd["wte"] = self.wte.weight.detach()
        sd["wpe"] = self.wpe.weight.detach()
        for i, block in enumerate(self.blocks):
            p = f"blocks.{i}."
            sd[p + "ln_1.weight"] = block.ln_1.weight.detach()
            sd[p + "ln_1.bias"] = block.ln_1.bias.detach()
            sd[p + "attn.c_attn.weight"] = block.attn.c_attn.weight.detach()
            if block.attn.c_attn.bias is not None:
                sd[p + "attn.c_attn.bias"] = block.attn.c_attn.bias.detach()
            sd[p + "attn.c_proj.weight"] = block.attn.c_proj.weight.detach()
            if block.attn.c_proj.bias is not None:
                sd[p + "attn.c_proj.bias"] = block.attn.c_proj.bias.detach()
            sd[p + "ln_2.weight"] = block.ln_2.weight.detach()
            sd[p + "ln_2.bias"] = block.ln_2.bias.detach()
            sd[p + "mlp.c_fc.weight"] = block.mlp.c_fc.weight.detach()
            if block.mlp.c_fc.bias is not None:
                sd[p + "mlp.c_fc.bias"] = block.mlp.c_fc.bias.detach()
            sd[p + "mlp.c_proj.weight"] = block.mlp.c_proj.weight.detach()
            if block.mlp.c_proj.bias is not None:
                sd[p + "mlp.c_proj.bias"] = block.mlp.c_proj.bias.detach()
        sd["ln_f.weight"] = self.ln_f.weight.detach()
        sd["ln_f.bias"] = self.ln_f.bias.detach()
        if not self.config.tie_weights:
            sd["lm_head.weight"] = self.lm_head.weight.detach()
        return sd

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop_tokens: set[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with an incremental KV cache."""
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        past = None
        out = idx
        for _ in range(max_new_tokens):
            ctx = out if past is None else out[:, -1:]
            if ctx.size(1) > self.config.block_size:
                ctx = ctx[:, -self.config.block_size :]
            logits, _, past = self.forward(ctx, past_key_values=past)
            logits = logits[:, -1, :].float()
            # Sample first, *then* test the new token for a stop marker.  The
            # order matters: prompts end with <|assistant|>, which is itself in
            # the stop set, so checking out[0, -1] before sampling looked at the
            # prompt's own last token and broke out on the very first iteration.
            # Every completion under this trainer came back empty -- which is
            # exactly what samples.txt contained, for every run, unnoticed.
            # NumpyGPT.generate and the browser worker both already sample
            # first; this now matches them.
            next_id = _sample(logits, temperature, top_k, top_p, gen)
            if stop_tokens and int(next_id) in stop_tokens:
                # a control token ends the turn; it is not part of the reply,
                # so it is not appended (mirrors NumpyGPT.generate)
                break
            out = torch.cat((out, next_id), dim=1)
        return out


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    gen: torch.Generator,
) -> torch.Tensor:
    """Temperature / top-k / top-p sampling for a (B, V) logit tensor."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # keep the smallest set whose cumulative probability >= top_p
        remove = cumulative - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=gen)


def build_model(config: GPTConfig) -> GPT:
    return GPT(config)
