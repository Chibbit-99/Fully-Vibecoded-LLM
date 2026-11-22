"""Configuration objects for the TinyGPT model.

Everything that defines a checkpoint lives here so that the training script,
the exporter, the NumPy engine and the JavaScript engine all agree on a single
source of truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class GPTConfig:
    """Hyper-parameters of the decoder-only transformer.

    The defaults describe the model that ships with this repository
    (~6.4M parameters, trained on Tiny Shakespeare).
    """

    name: str = "tiny-gpt"
    vocab_size: int = 1280  # 256 raw bytes + 1024 BPE merges
    block_size: int = 256  # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 288
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    gelu_tanh: bool = True  # tanh approximation -> identical in numpy/js
    ln_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GPTConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def num_params(self) -> int:
        """Analytic parameter count (handy before instantiating torch)."""
        d, L, b = self.n_embd, self.n_layer, int(self.bias)
        emb = self.vocab_size * d + self.block_size * d
        attn = 3 * d * d + 3 * b * d + d * d + b * d  # c_attn + c_proj
        mlp = 4 * d * d + 4 * b * d + 4 * d * d + b * d  # c_fc + c_proj
        ln = 4 * d  # two LayerNorms (weight + bias each)
        head = 0 if self.tie_weights else self.vocab_size * d
        return emb + L * (attn + mlp + ln) + 2 * d + head  # + final LayerNorm
