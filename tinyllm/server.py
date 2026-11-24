"""FastAPI inference server for the trained TinyGPT model.

Two engines are available:

* ``numpy`` (default) -- pure NumPy, no PyTorch needed. Uses the exported
  ``model/spore/<name>/weights.json`` (float16) and a KV cache.
* ``torch`` -- the original checkpoint, if PyTorch is installed.

Run it:

    uvicorn tinyllm.server:app --host 0.0.0.0 --port 8000
    # or
    python -m tinyllm.server --port 8000                    # auto-picks a
    python -m tinyllm.server --checkpoint spore-1-12M --port 8000

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
from .retrieval import Factbook, context_for
from .sft import ASSISTANT, END, render_history, render_prompt
from .tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parent.parent
SPORE_DIR = ROOT / "model" / "spore"
FACTBOOK = ROOT / "data" / "spore" / "factbook.json"


def _default_checkpoint() -> Path:
    """Prefer a published Spore model; fall back to a legacy model/weights.json.

    Spore is the only model family in this repo, but the fallback keeps older
    checkouts (and any user checkpoint passed via --checkpoint) working.
    """
    published = sorted(SPORE_DIR.glob("*/weights.json"))
    # smallest first: the 12M loads fastest and is the sensible default
    if published:
        return min(published, key=lambda p: p.stat().st_size)
    return ROOT / "model" / "weights.json"


DEFAULT_CHECKPOINT = _default_checkpoint()

DEFAULT_SYSTEM_PROMPT = "You are Spore, a small language model trained from scratch."


def _count_params(model) -> int:
    """Unique parameter count, without double-counting tied weights.

    `lm_head.weight` is an alias of `wte` when weights are tied, so summing
    every array in the dict overcounts by one embedding matrix.
    """
    seen, total = set(), 0
    for arr in model.w.values():
        ident = arr.__array_interface__["data"][0]
        if ident in seen:
            continue
        seen.add(ident)
        total += int(arr.size)
    return total


def _strip_turn(text: str) -> str:
    """Cut a completion at the first control token and tidy whitespace."""
    for marker in (END, ASSISTANT):
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# engine wrapper (importable without FastAPI installed)
# ---------------------------------------------------------------------------
class TinyEngine:
    """Small convenience wrapper around NumpyGPT with timing + metadata."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        meta_path: str | Path | None = None,
        factbook: str | Path | None = None,
        use_retrieval: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model = NumpyGPT.from_checkpoint(self.checkpoint)
        self.config: GPTConfig = self.model.config
        self.tokenizer: ByteBPE = self.model.tokenizer
        self.meta: Dict[str, Any] = {}
        meta_candidate = Path(meta_path) if meta_path else self.checkpoint.parent / "meta.json"
        if meta_candidate.exists():
            self.meta = json.loads(meta_candidate.read_text())
        # retrieval grounding is what lets a 12M model answer factual questions
        self.factbook: Factbook | None = None
        if use_retrieval:
            fb = Path(factbook) if factbook else FACTBOOK
            try:
                self.factbook = Factbook.load(fb)
            except FileNotFoundError:
                self.factbook = None
        self.is_chat = bool(self.tokenizer.special_tokens)
        self.loaded_at = time.time()

    # -- introspection -----------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.meta.get("model") or "tinyllm",
            "checkpoint": str(self.checkpoint),
            "config": self.config.to_dict(),
            "vocab_size": self.tokenizer.vocab_size,
            # Count from the weights when meta.json is absent (a chunk that
            # exported but has not been published yet has no meta.json).
            "parameters": self.meta.get("params") or _count_params(self.model),
            "val_loss": self.meta.get("best_val_loss") or self.meta.get("val_loss"),
            "trained_steps": self.meta.get("steps"),
            "trained_tokens": self.meta.get("tokens_seen"),
            # report the corpus the checkpoint was actually trained on rather
            # than a constant: the Shakespeare model and Spore differ
            "training_dataset": (
                "WikiText-2 + TinyStories (prose + chat)"
                if self.is_chat
                else "Tiny Shakespeare"
            ),
            "chat": self.is_chat,
            "retrieval": self.factbook is not None,
            "retrieval_passages": len(self.factbook.passages) if self.factbook else 0,
            "samples": self.meta.get("samples"),
        }

    def build_prompt(self, prompt: str, history: Optional[List[Dict[str, str]]] = None,
                     grounding: bool = True) -> tuple[str, List[Dict[str, str]]]:
        """Render a chat prompt, optionally grounding it with retrieved text."""
        if not self.is_chat:
            return prompt, []
        context = context_for(prompt, self.factbook) if grounding else ""
        turns = [(m.get("role", "user"), m.get("content", "")) for m in (history or [])]
        if context:
            # the grounding block goes only with the *newest* question
            turns.append(("user", prompt))
            rendered = render_history(turns[:-1]) + render_prompt(prompt, context)
        elif turns:
            turns.append(("user", prompt))
            rendered = render_history(turns)
        else:
            rendered = render_prompt(prompt)
        hits = self.factbook.search(prompt, k=1) if (self.factbook and grounding) else []
        return rendered, hits

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
        history: Optional[List[Dict[str, str]]] = None,
        grounding: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.is_chat:
            rendered, hits = self.build_prompt(prompt, history, grounding)
            stop_tokens = {self.tokenizer.encode_special(END)}
            # the model must never continue as the user
            stop_tokens.add(self.tokenizer.encode_special(ASSISTANT))
            result = self.model.generate(
                rendered,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                stop_strings=stop,
                stop_tokens=stop_tokens,
            )
            completion = _strip_turn(result["text"])
            grounding_used = str(hits[0]["text"])[:700] if hits else ""
        else:
            if system_prompt:
                prompt = f"{system_prompt}\n{prompt}"
            rendered = prompt
            hits = []
            grounding_used = ""
            result = self.model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                stop_strings=stop,
            )
            completion = result["text"]
        elapsed = time.perf_counter() - t0
        n = result["tokens_generated"]
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,
            "grounding": grounding_used,
            "retrieved": [{"title": h.get("title"), "score": h.get("score")} for h in hits],
            "tokens_generated": n,
            "finish_reason": result["finish_reason"],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(rendered)),
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

    class Message(BaseModel):
        role: str = Field("user", description="'user' or 'assistant'")
        content: str = ""

    class GenerateRequest(BaseModel):
        prompt: str = Field("", description="Text to complete, or the user's question")
        system_prompt: str = Field("", description="Optional text prepended to the prompt")
        max_tokens: int = Field(120, ge=1, le=1024)
        temperature: float = Field(0.8, ge=0.0, le=5.0, description="0 = greedy")
        top_k: int | None = Field(40, ge=0, description="0 or null disables top-k")
        top_p: float | None = Field(0.95, gt=0, le=1.0)
        seed: int | None = Field(None, description="Fix for reproducible sampling")
        stop: List[str] | None = Field(None, description="Stop strings")
        history: List[Message] | None = Field(None, description="Previous turns, oldest first")
        grounding: bool = Field(True, description="Retrieve a passage to ground the answer")

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

        @app.get("/api/models")
        def list_models() -> Dict[str, Any]:
            """Spore checkpoints found under model/spore/ (the demo uses this)."""
            found = []
            for d in sorted(SPORE_DIR.glob("*/weights.json")):
                found.append({"name": d.parent.name, "checkpoint": str(d)})
            return {"current": str(get_engine().checkpoint), "models": found}

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
                history=[m.model_dump() for m in (req.history or [])],
                grounding=req.grounding,
            )

        @app.get("/api/generate")
        def generate_get(
            prompt: str = "What is water?",
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
        data_dir = ROOT / "data"
        if model_dir.exists():
            # /model/spore/<name>/weights.json so the browser demo can load them
            app.mount("/model", StaticFiles(directory=model_dir), name="model")
        if data_dir.exists():
            # /data/spore/factbook.json -> the demo's in-browser retrieval
            app.mount("/data", StaticFiles(directory=data_dir), name="data")
        if web_dir.exists():
            app.mount("/static", StaticFiles(directory=web_dir), name="static")

            @app.get("/")
            def index() -> FileResponse:
                # web/ *is* the Spore playground: index.html plus the JS that
                # loads model/spore/<name>/ directly.
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
        default=os.environ.get("TINYLLM_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        help="weights.json / .npz, or a Spore model name such as spore-1-12M",
    )
    p.add_argument("--no-retrieval", action="store_true", help="disable factbook grounding")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    if not HAVE_FASTAPI:
        raise SystemExit("FastAPI is not installed. `pip install -r requirements.txt`")
    import uvicorn

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists() and (SPORE_DIR / args.checkpoint / "weights.json").exists():
        checkpoint = SPORE_DIR / args.checkpoint / "weights.json"
    engine = TinyEngine(checkpoint, use_retrieval=not args.no_retrieval)
    info = engine.info()
    print(f"[tinyllm] loaded {checkpoint} | {info['parameters']} params | "
          f"chat={info['chat']} retrieval={info['retrieval']}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
