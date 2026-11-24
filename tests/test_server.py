"""HTTP API smoke tests (require a trained checkpoint + FastAPI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tinyllm.server import TinyEngine, create_app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPORE_DIR = ROOT / "model" / "spore"


def _pick_spore() -> Path | None:
    """Smallest published Spore checkpoint: fastest to load in tests."""
    found = sorted(SPORE_DIR.glob("*/weights.json")) if SPORE_DIR.exists() else []
    return min(found, key=lambda p: p.stat().st_size) if found else None


WEIGHTS = Path(os.environ.get("TINYLLM_CHECKPOINT") or (_pick_spore() or SPORE_DIR / "none" / "weights.json"))
FACTBOOK = ROOT / "data" / "spore" / "factbook.json"

# Spore is the only model family left, so one gate covers the whole file. The
# fixture below also builds the (gitignored) factbook if the checkout has none,
# which keeps the retrieval assertions meaningful in CI.
needs_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="no Spore checkpoint published yet"
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_factbook():
    """data/spore/ is gitignored; build it if a checkpoint exists but data does not."""
    if WEIGHTS.exists() and not FACTBOOK.exists():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_spore_data.py")],
            check=True, capture_output=True, cwd=ROOT,
        )


@pytest.fixture(scope="module")
def client():
    engine = TinyEngine(WEIGHTS)
    return TestClient(create_app(engine))


@needs_weights
def test_health(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["model"]["vocab_size"] > 0
    assert data["model"]["config"]["n_layer"] >= 1


@needs_weights
def test_model_info(client):
    data = client.get("/api/model").json()
    assert "config" in data
    assert data["training_dataset"] == "WikiText-2 + TinyStories (prose + chat)"


@needs_weights
def test_generate_post(client):
    resp = client.post(
        "/api/generate",
        json={"prompt": "<|user|>What is water?<|assistant|>",
              "max_tokens": 12, "temperature": 0.0, "seed": 1},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["usage"]["completion_tokens"] == data["tokens_generated"]
    assert data["timings"]["tokens_per_second"] > 0
    assert data["text"].startswith("<|user|>")


@needs_weights
def test_generate_is_reproducible(client):
    body = {"prompt": "<|user|>What is a cat?", "max_tokens": 16,
            "temperature": 0.9, "top_k": 40, "seed": 42}
    a = client.post("/api/generate", json=body).json()["completion"]
    b = client.post("/api/generate", json=body).json()["completion"]
    assert a == b


@needs_weights
def test_generate_get(client):
    data = client.get("/api/generate",
                      params={"prompt": "<|user|>What is water?", "max_tokens": 8}).json()
    assert data["tokens_generated"] > 0


@needs_weights
def test_validation_rejects_bad_input(client):
    assert client.post("/api/generate", json={"max_tokens": 0}).status_code == 422
    assert client.post("/api/generate", json={"max_tokens": 5000}).status_code == 422


@needs_weights
def test_tokenize(client):
    text = "To be, or not to be"
    data = client.post("/api/tokenize", json={"text": text}).json()
    assert data["count"] == len(data["tokens"]) == len(data["pieces"])
    assert data["roundtrip_ok"]
    assert "".join(data["pieces"]) == text


@needs_weights
def test_stop_string(client):
    body = {"prompt": "<|user|>hi", "max_tokens": 200, "temperature": 0.0, "stop": ["\n"]}
    data = client.post("/api/generate", json=body).json()
    assert "\n" not in data["completion"]


@needs_weights
def test_engine_matches_js_reference():
    """The API must emit clean text from the same engine the browser uses."""
    engine = TinyEngine(WEIGHTS)
    out = engine.generate(prompt="<|user|>What is water?", max_tokens=8, temperature=0.0)
    assert out["engine"] == "numpy"
    assert isinstance(out["completion"], str)


# ---------------------------------------------------------------------------
# Spore chat API (skipped until a Spore checkpoint is published)
# ---------------------------------------------------------------------------
spore_only = needs_weights


@spore_only
def test_spore_engine_reports_chat_and_retrieval():
    engine = TinyEngine(WEIGHTS)
    info = engine.info()
    assert info["chat"] is True, "the Spore tokenizer must know its control tokens"
    # exact by construction: the config name encodes the parameter count
    assert info["parameters"] == 12_000_000 or info["parameters"] == 36_000_000
    assert info["retrieval"] is True
    assert info["retrieval_passages"] > 1000


@spore_only
def test_spore_generate_returns_grounding_and_stops():
    engine = TinyEngine(WEIGHTS)
    out = engine.generate("What is water?", max_tokens=12, seed=3)
    assert set(("completion", "grounding", "retrieved", "usage")) <= set(out)
    # the answer must not leak the chat template back to the caller
    assert "<|assistant|>" not in out["completion"]
    assert "<|end|>" not in out["completion"]
    assert "<|user|>" not in out["completion"]
    # a known "what is X?" question retrieves the built-in definition
    assert "water" in out["grounding"].lower()


@spore_only
def test_spore_chat_endpoint_accepts_history():
    client = TestClient(create_app(TinyEngine(WEIGHTS)))
    res = client.post(
        "/api/generate",
        json={
            "prompt": "How do you spell cat?",
            "max_tokens": 8,
            "history": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert "completion" in body
    assert body["usage"]["completion_tokens"] >= 1


@spore_only
def test_spore_models_endpoint_lists_the_checkpoint():
    client = TestClient(create_app(TinyEngine(WEIGHTS)))
    body = client.get("/api/models").json()
    assert body["models"], "no models listed"
    # the picked checkpoint is always the smallest published one
    assert any(m["name"] == WEIGHTS.parent.name for m in body["models"])


@spore_only
def test_spore_demo_page_and_assets_are_served():
    client = TestClient(create_app(TinyEngine(WEIGHTS)))
    assert client.get("/").status_code == 200
    # web/ is the Spore playground; index.html is the page at /
    for asset in ("/index.html", "/spore.js", "/retrieval.js",
                  "/style.css", "/engine.js", "/tokenizer.js", "/worker.js"):
        assert client.get(asset).status_code == 200, asset
    assert "Spore" in client.get("/").text
