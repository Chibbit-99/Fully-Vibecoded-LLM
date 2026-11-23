"""NumPy-only inference engine + checkpoint exporter.

This module is deliberately free of PyTorch so the trained model can be served
anywhere numpy runs.  It mirrors ``tinyllm/model.py`` operation for operation and
supports an incremental KV cache, so generation is fast on CPU.

Checkpoint format (``weights.json``)
------------------------------------
A single JSON file shared by the numpy engine and the browser engine:

    {
      "format": "tinyllm-weights-v1",
      "dtype": "float16",
      "config": {...GPTConfig...},
      "tokenizer": {...ByteBPE...},
      "weights": { "<tensor name>": {"shape": [..], "data": [..floats..] } }
    }

Tensors are stored row-major, matching PyTorch's linear layer layout
(``y = x @ W.T + b``).  float16 keeps the file ~13 MB for the 6.4M-param model
while costing no measurable quality in generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import GPTConfig
from .tokenizer import ByteBPE

FORMAT = "tinyllm-weights-v1"


# ---------------------------------------------------------------------------
# exporter (imports torch lazily so the inference path never needs it)
# ---------------------------------------------------------------------------
def export_checkpoint(
    model,
    config: GPTConfig,
    tokenizer: ByteBPE,
    path: str | Path,
    dtype: str = "float16",
    encoding: str = "raw",
) -> Path:
    """Serialise the model for the NumPy / JS engines.

    ``encoding="raw"`` (default) stores each tensor as a float16 buffer encoded
    in base64 -- ~2 bytes/parameter, so the 3.5M-parameter model is ~7 MB and
    loads in the browser in one fetch.  ``encoding="json"`` writes plain
    decimal arrays, useful for debugging and small test models.
    """
    import base64

    path = Path(path)
    state = model.state_dict_for_export()
    np_dtype = np.float16 if dtype == "float16" else np.float32
    weights: Dict[str, Any] = {}
    for name, tensor in state.items():
        arr = tensor.detach().to("cpu").float().numpy().astype(np_dtype)
        entry: Dict[str, Any] = {"shape": list(arr.shape)}
        if encoding == "raw":
            entry["encoding"] = "base64"
            entry["data"] = base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")
        else:
            entry["data"] = arr.reshape(-1).tolist()
        weights[name] = entry

    payload = {
        "format": FORMAT,
        "dtype": dtype,
        "encoding": encoding,
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "weights": weights,
    }
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def _decode_tensor(entry: Dict[str, Any]) -> np.ndarray:
    """Turn one weight entry into a float32 array (either encoding)."""
    shape = tuple(entry["shape"])
    if entry.get("encoding") == "base64":
        import base64

        raw = base64.b64decode(entry["data"])
        arr = np.frombuffer(raw, dtype=np.float16)
        return arr.astype(np.float32).reshape(shape)
    return np.asarray(entry["data"], dtype=np.float32).reshape(shape)


def load_checkpoint(path: str | Path) -> tuple[Dict[str, np.ndarray], GPTConfig, ByteBPE]:
    path = Path(path)
    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=False)
        weights = {k: raw[k].astype(np.float32) for k in raw.files if k != "__meta__"}
        meta = json.loads(str(raw["__meta__"]))
        return weights, GPTConfig.from_dict(meta["config"]), ByteBPE.from_dict(meta["tokenizer"])

    with open(path) as f:
        payload = json.load(f)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    weights = {name: _decode_tensor(t) for name, t in payload["weights"].items()}
    return weights, GPTConfig.from_dict(payload["config"]), ByteBPE.from_dict(payload["tokenizer"])


def export_npz(weights: Dict[str, np.ndarray], config: GPTConfig, tokenizer: ByteBPE, path: str | Path) -> Path:
    """Compact companion file for the CLI (fast to load, not browser-friendly)."""
    path = Path(path)
    meta = json.dumps({"config": config.to_dict(), "tokenizer": tokenizer.to_dict()})
    np.savez(str(path), __meta__=np.array(meta), **weights)
    return path


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _gelu(x: np.ndarray) -> np.ndarray:
    """tanh approximation of GELU (matches torch's approximate='tanh')."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class NumpyGPT:
    """Decoder-only transformer inference in pure NumPy."""

    def __init__(self, weights: Dict[str, np.ndarray], config: GPTConfig, tokenizer: ByteBPE) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.w = weights
        if "lm_head.weight" not in self.w:
            self.w["lm_head.weight"] = self.w["wte"]  # weight tying

    # -- factory -----------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NumpyGPT":
        weights, config, tokenizer = load_checkpoint(path)
        return cls(weights, config, tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    # -- primitives --------------------------------------------------------
    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        y = x @ self.w[name + ".weight"].T
        b = self.w.get(name + ".bias")
        return y + b if b is not None else y

    def _layernorm(self, x: np.ndarray, name: str) -> np.ndarray:
        eps = self.config.ln_eps
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return self.w[name + ".weight"] * (x - mean) / np.sqrt(var + eps) + self.w[name + ".bias"]

    def _block(self, x: np.ndarray, i: int, past: Optional[tuple]) -> tuple[np.ndarray, tuple]:
        c = self.config
        p = f"blocks.{i}."
        h = self._layernorm(x, p + "ln_1")
        T = h.shape[1]
        qkv = self._linear(h, p + "attn.c_attn")
        q, k, v = np.split(qkv, 3, axis=-1)
        hd, nh = c.head_dim, c.n_head

        def heads(t):  # (1, T, C) -> (nh, T, hd)
            return t.reshape(1, T, nh, hd).transpose(0, 2, 1, 3)[0]

        q, k, v = heads(q), heads(k), heads(v)
        if past is not None:
            k = np.concatenate([past[0], k], axis=1)
            v = np.concatenate([past[1], v], axis=1)
        Tk = k.shape[1]
        att = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)
        if T > 1:
            mask = np.tril(np.ones((T, Tk), dtype=bool), k=Tk - T)
            att = np.where(mask, att, -np.inf)
        att = _softmax(att)
        y = (att @ v).transpose(1, 0, 2).reshape(1, T, c.n_embd)
        x = x + self._linear(y, p + "attn.c_proj")
        h2 = self._layernorm(x, p + "ln_2")
        x = x + self._linear(_gelu(self._linear(h2, p + "mlp.c_fc")), p + "mlp.c_proj")
        return x, (k, v)

    # -- forward -----------------------------------------------------------
    def forward(self, idx: List[int] | np.ndarray, past: Optional[list] = None):
        """Return (logits (1,T,V), new kv cache)."""
        c = self.config
        idx_arr = np.asarray(idx, dtype=np.int64).reshape(1, -1)
        T = idx_arr.shape[1]
        if past is not None and past[0][0].shape[1] + T > c.block_size:
            keep = c.block_size - T
            past = [(k[:, -keep:], v[:, -keep:]) for k, v in past]
        offset = 0 if past is None else past[0][0].shape[1]
        if offset + T > c.block_size:
            raise ValueError(f"context length {offset + T} exceeds block_size {c.block_size}")
        x = self.w["wte"][idx_arr] + self.w["wpe"][offset : offset + T][None, :, :]
        new_past = []
        for i in range(c.n_layer):
            x, pkv = self._block(x, i, None if past is None else past[i])
            new_past.append(pkv)
        x = self._layernorm(x, "ln_f")
        logits = self._linear(x, "lm_head")
        return logits, new_past

    # -- sampling ----------------------------------------------------------
    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Sample a token id from the last row of ``logits``."""
        logits = np.asarray(logits, dtype=np.float64)[-1].copy()
        if temperature <= 0:
            return int(np.argmax(logits))
        logits /= temperature
        if top_k is not None and 0 < top_k < logits.size:
            kth = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth] = -np.inf
        if top_p is not None and 0 < top_p < 1.0:
            order = np.argsort(-logits)
            probs = _softmax(logits[order])
            cumulative = np.cumsum(probs)
            cutoff = np.searchsorted(cumulative, top_p) + 1
            keep = order[: max(1, cutoff)]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked
        probs = _softmax(logits)
        rng = rng or np.random.default_rng()
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    def generate(
        self,
        prompt: str | List[int],
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        seed: Optional[int] = None,
        stop_strings: Optional[List[str]] = None,
        stop_tokens: Optional[set] = None,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            ids = [0]
        ids = ids[-self.config.block_size :]
        past = None
        # prime the cache with the prompt
        ctx = ids
        produced: List[int] = []
        text = ""
        for _ in range(max_new_tokens):
            logits, past = self.forward(ctx, past)
            nxt = self.sample(logits[0], temperature, top_k, top_p, rng)
            produced.append(nxt)
            ctx = [nxt]
            if stop_tokens and nxt in stop_tokens:
                # a control token ended the turn; do not echo it
                produced.pop()
                text = self.tokenizer.decode(produced)
                break
            new_text = self.tokenizer.decode(produced)
            if stop_strings and any(s and s in new_text for s in stop_strings):
                # trim to the first stop string
                cut = min(new_text.index(s) for s in stop_strings if s and s in new_text)
                text = new_text[:cut]
                break
            text = new_text
            if past[0][0].shape[1] >= self.config.block_size:
                break
        return {
            "prompt": prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt),
            "text": text,
            "tokens_generated": len(produced),
            "finish_reason": "stop" if len(produced) < max_new_tokens else "length",
        }
