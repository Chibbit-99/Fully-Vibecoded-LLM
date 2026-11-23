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
WEIGHTS = Path(os.environ.get("TINYLLM_CHECKPOINT", ROOT / "model" / "weights.json"))

# The legacy TinyGPT tests need model/weights.json; the Spore tests need their
# own checkpoint.  Skipping at module level would hide the Spore tests whenever
# the Shakespeare weights are absent (and vice versa), so each group is gated
# separately.
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights available")


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
    assert data["training_dataset"] == "Tiny Shakespeare"


@needs_weights
def test_generate_post(client):
    resp = client.post(
        "/api/generate",
        json={"prompt": "ROMEO:", "max_tokens": 12, "temperature": 0.0, "seed": 1},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["completion"]
    assert data["tokens_generated"] > 0
    assert data["usage"]["completion_tokens"] == data["tokens_generated"]
    assert data["timings"]["tokens_per_second"] > 0
    assert data["text"].startswith("ROMEO:")


@needs_weights
def test_generate_is_reproducible(client):
    body = {"prompt": "The king", "max_tokens": 16, "temperature": 0.9, "top_k": 40, "seed": 42}
    a = client.post("/api/generate", json=body).json()["completion"]
    b = client.post("/api/generate", json=body).json()["completion"]
    assert a == b


@needs_weights
def test_generate_get(client):
    data = client.get("/api/generate", params={"prompt": "JULIET:", "max_tokens": 8}).json()
    assert data["tokens_generated"] > 0


@needs_weights
def test_validation_rejects_bad_input(client):
    assert client.post("/api/generate", json={"max_tokens": 0}).status_code == 422
    assert client.post("/api/generate", json={"max_tokens": 5000}).status_code == 422


@needs_weights
def test_tokenize(client):
    data = client.post("/api/tokenize", json={"text": "To be, or not to be"}).json()
    assert data["count"] == len(data["tokens"]) == len(data["pieces"])
    assert data["roundtrip_ok"]
    assert "".join(data["pieces"]) == "To be, or not to be"


@needs_weights
def test_stop_string(client):
    body = {"prompt": "", "max_tokens": 200, "temperature": 0.0, "stop": ["\n"]}
    data = client.post("/api/generate", json=body).json()
    assert "\n" not in data["completion"]


@needs_weights
def test_engine_matches_js_reference():
    """The API must emit the same text the browser engine does for greedy decoding."""
    engine = TinyEngine(WEIGHTS)
    out = engine.generate(prompt="ROMEO:", max_tokens=16, temperature=0.0)
    assert out["engine"] == "numpy"
    assert len(out["completion"]) > 0


# ---------------------------------------------------------------------------
# Spore chat API (skipped until a Spore checkpoint is published)
# ---------------------------------------------------------------------------
SPORE_WEIGHTS = ROOT / "model" / "spore" / "spore-1-12M" / "weights.json"
FACTBOOK = ROOT / "data" / "spore" / "factbook.json"


@pytest.fixture(scope="module", autouse=True)
def _ensure_factbook():
    """data/spore/ is gitignored; build it if the checkout does not have it.

    Only runs when a Spore checkpoint is published, and takes ~2 s.
    """
    if SPORE_WEIGHTS.exists() and not FACTBOOK.exists():
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_spore_data.py")],
            check=True, capture_output=True, cwd=ROOT,
        )


spore_only = pytest.mark.skipif(
    not SPORE_WEIGHTS.exists(), reason="no Spore checkpoint published yet"
)


@spore_only
def test_spore_engine_reports_chat_and_retrieval():
    engine = TinyEngine(SPORE_WEIGHTS)
    info = engine.info()
    assert info["chat"] is True, "the Spore tokenizer must know its control tokens"
    assert info["parameters"] == 12_000_000
    assert info["retrieval"] is True
    assert info["retrieval_passages"] > 1000


@spore_only
def test_spore_generate_returns_grounding_and_stops():
    engine = TinyEngine(SPORE_WEIGHTS)
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
    client = TestClient(create_app(TinyEngine(SPORE_WEIGHTS)))
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
    client = TestClient(create_app(TinyEngine(SPORE_WEIGHTS)))
    body = client.get("/api/models").json()
    assert any(m["name"] == "spore-1-12M" for m in body["models"])


@spore_only
def test_spore_demo_page_and_assets_are_served():
    client = TestClient(create_app(TinyEngine(SPORE_WEIGHTS)))
    assert client.get("/").status_code == 200
    for asset in ("/spore.js", "/retrieval.js", "/spore.css", "/spore.html"):
        assert client.get(asset).status_code == 200, asset
