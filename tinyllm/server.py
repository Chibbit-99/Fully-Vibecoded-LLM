"""FastAPI inference server for the trained TinyGPT model.

Two engines are available:

* ``numpy`` (default) -- pure NumPy, no PyTorch needed. Uses the exported
  ``model/weights.json`` (float16) and a KV cache.
* ``torch`` -- the original checkpoint, if PyTorch is installed.

Run it:

    uvicorn tinyllm.server:app --host 0.0.0.0 --port 8000
    # or
    python -m tinyllm.server --checkpoint model/weights.json --port 8000

Then:

    curl -s localhost:8000/api/health
    curl -s localhost:8000/api/generate \
         -H 'content-type: application/json' \
         -d '{"prompt": "ROMEO:", "max_tokens": 120, "temperature": 0.8}'

A static demo of the browser engine is served at ``/`` when ``web/`` exists.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:  # FastAPI is optional: the engine itself has no web dependency
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False
    BaseModel = object  # type: ignore

from .config import GPTConfig
from .numpy_engine import NumpyGPT, load_checkpoint
from .tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "model" / "weights.json"

DEFAULT_SYSTEM_PROMPT = "You are TinyGPT, a small language model trained on Shakespeare."


# ---------------------------------------------------------------------------
# engine wrapper (importable without FastAPI installed)
# ---------------------------------------------------------------------------
class TinyEngine:
    """Small convenience wrapper around NumpyGPT with timing + metadata."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, meta_path: str | Path | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.model = NumpyGPT.from_checkpoint(self.checkpoint)
        self.config: GPTConfig = self.model.config
        self.tokenizer: ByteBPE = self.model.tokenizer
        self.meta: Dict[str, Any] = {}
        meta_candidate = Path(meta_path) if meta_path else self.checkpoint.parent / "meta.json"
        if meta_candidate.exists():
            self.meta = json.loads(meta_candidate.read_text())
        self.loaded_at = time.time()

    # -- introspection -----------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "name": "tinyllm",
            "checkpoint": str(self.checkpoint),
            "config": self.config.to_dict(),
            "vocab_size": self.tokenizer.vocab_size,
            "parameters": self.meta.get("params"),
            "val_loss": self.meta.get("val_loss"),
            "bits_per_byte": self.meta.get("bits_per_byte"),
            "trained_steps": self.meta.get("steps"),
            "trained_tokens": self.meta.get("tokens_seen"),
            "training_dataset": "Tiny Shakespeare",
        }

    # -- inference ---------------------------------------------------------
    def generate(
        self,
        prompt: str = "",
        system_prompt: str = "",
        max_tokens: int = 120,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if system_prompt:
            prompt = f"{system_prompt}\n{prompt}"
        result = self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop_strings=stop,
        )
        elapsed = time.perf_counter() - t0
        n = result["tokens_generated"]
        return {
            "prompt": prompt,
            "completion": result["text"],
            "text": prompt + result["text"],
            "tokens_generated": n,
            "finish_reason": result["finish_reason"],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": n,
            },
            "timings": {
                "total_ms": round(elapsed * 1000, 1),
                "tokens_per_second": round(n / elapsed, 2) if elapsed > 0 else None,
            },
            "engine": "numpy",
        }

    def tokenize(self, text: str) -> Dict[str, Any]:
        ids = self.tokenizer.encode(text)
        return {
            "tokens": ids,
            "count": len(ids),
            "pieces": [self.tokenizer.decode([i]) for i in ids],
            "roundtrip_ok": self.tokenizer.decode(ids) == text,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
if HAVE_FASTAPI:

    class GenerateRequest(BaseModel):
        prompt: str = Field("", description="Text to complete")
        system_prompt: str = Field("", description="Optional text prepended to the prompt")
        max_tokens: int = Field(120, ge=1, le=1024)
        temperature: float = Field(0.8, ge=0.0, le=5.0, description="0 = greedy")
        top_k: int | None = Field(40, ge=0, description="0 or null disables top-k")
        top_p: float | None = Field(0.95, gt=0, le=1.0)
        seed: int | None = Field(None, description="Fix for reproducible sampling")
        stop: List[str] | None = Field(None, description="Stop strings")

    class TokenizeRequest(BaseModel):
        text: str = ""

    def create_app(engine: TinyEngine | None = None) -> "FastAPI":
        app = FastAPI(
            title="TinyLLM",
            description="A GPT written from scratch. Trained by GitHub Actions, served by NumPy.",
            version="1.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        state: Dict[str, Any] = {"engine": engine}

        def get_engine() -> TinyEngine:
            if state["engine"] is None:
                checkpoint = Path(os.environ.get("TINYLLM_CHECKPOINT", DEFAULT_CHECKPOINT))
                if not checkpoint.exists():
                    raise HTTPException(status_code=503, detail=f"No checkpoint at {checkpoint}")
                state["engine"] = TinyEngine(checkpoint)
            return state["engine"]

        @app.get("/api/health")
        def health() -> Dict[str, Any]:
            eng = get_engine()
            return {
                "status": "ok",
                "model": eng.info(),
                "uptime_seconds": round(time.time() - eng.loaded_at, 1),
            }

        @app.get("/api/model")
        def model_info() -> Dict[str, Any]:
            return get_engine().info()

        @app.post("/api/generate")
        def generate(req: GenerateRequest) -> Dict[str, Any]:
            eng = get_engine()
            return eng.generate(
                prompt=req.prompt,
                system_prompt=req.system_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k or None,
                top_p=req.top_p,
                seed=req.seed,
                stop=req.stop,
            )

        @app.get("/api/generate")
        def generate_get(
            prompt: str = "ROMEO:",
            max_tokens: int = 80,
            temperature: float = 0.8,
            top_k: int = 40,
            top_p: float = 0.95,
            seed: int | None = None,
        ) -> Dict[str, Any]:
            """GET convenience endpoint so the API can be tried from a browser bar."""
            return get_engine().generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k or None,
                top_p=top_p,
                seed=seed,
            )

        @app.post("/api/tokenize")
        def tokenize(req: TokenizeRequest) -> Dict[str, Any]:
            return get_engine().tokenize(req.text)

        web_dir = ROOT / "web"
        model_dir = ROOT / "model"
        if model_dir.exists():
            # /model/weights.json etc. so the browser demo can be served here too
            app.mount("/model", StaticFiles(directory=model_dir), name="model")
        if web_dir.exists():
            app.mount("/static", StaticFiles(directory=web_dir), name="static")

            @app.get("/")
            def index() -> FileResponse:
                return FileResponse(web_dir / "index.html")

            @app.get("/{path:path}")
            def static_or_index(path: str):
                candidate = web_dir / path
                if candidate.is_file():
                    return FileResponse(candidate)
                raise HTTPException(status_code=404, detail="not found")

        return app

    app = create_app()

else:  # pragma: no cover
    app = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Serve TinyLLM (FastAPI + NumPy engine)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("TINYLLM_CHECKPOINT", DEFAULT_CHECKPOINT)),
        help="weights.json (or .npz) produced by scripts/train.py",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    if not HAVE_FASTAPI:
        raise SystemExit("FastAPI is not installed. `pip install -r requirements.txt`")
    import uvicorn

    engine = TinyEngine(args.checkpoint)
    print(f"[tinyllm] loaded {args.checkpoint} | {engine.info()['parameters']} params")
    uvicorn.run(create_app(engine), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
