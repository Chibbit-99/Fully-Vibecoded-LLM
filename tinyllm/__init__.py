"""TinyLLM -- a GPT-style language model written from scratch.

Modules
-------
config     : hyper-parameters (single source of truth for all engines)
tokenizer  : byte-level BPE trained from scratch
model      : PyTorch training / inference model
dataset    : Tiny Shakespeare loading + batching
numpy_engine : dependency-light inference engine (numpy only)
server     : FastAPI inference server (optional)
"""

from .config import GPTConfig
from .tokenizer import ByteBPE

__all__ = ["GPTConfig", "ByteBPE"]
__version__ = "1.0.0"
