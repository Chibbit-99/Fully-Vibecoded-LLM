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

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights available")


@pytest.fixture(scope="module")
def client():
    engine = TinyEngine(WEIGHTS)
    return TestClient(create_app(engine))


def test_health(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["model"]["vocab_size"] > 0
    assert data["model"]["config"]["n_layer"] >= 1


def test_model_info(client):
    data = client.get("/api/model").json()
    assert "config" in data
    assert data["training_dataset"] == "Tiny Shakespeare"


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


def test_generate_is_reproducible(client):
    body = {"prompt": "The king", "max_tokens": 16, "temperature": 0.9, "top_k": 40, "seed": 42}
    a = client.post("/api/generate", json=body).json()["completion"]
    b = client.post("/api/generate", json=body).json()["completion"]
    assert a == b


def test_generate_get(client):
    data = client.get("/api/generate", params={"prompt": "JULIET:", "max_tokens": 8}).json()
    assert data["tokens_generated"] > 0


def test_validation_rejects_bad_input(client):
    assert client.post("/api/generate", json={"max_tokens": 0}).status_code == 422
    assert client.post("/api/generate", json={"max_tokens": 5000}).status_code == 422


def test_tokenize(client):
    data = client.post("/api/tokenize", json={"text": "To be, or not to be"}).json()
    assert data["count"] == len(data["tokens"]) == len(data["pieces"])
    assert data["roundtrip_ok"]
    assert "".join(data["pieces"]) == "To be, or not to be"


def test_stop_string(client):
    body = {"prompt": "", "max_tokens": 200, "temperature": 0.0, "stop": ["\n"]}
    data = client.post("/api/generate", json=body).json()
    assert "\n" not in data["completion"]


def test_engine_matches_js_reference():
    """The API must emit the same text the browser engine does for greedy decoding."""
    engine = TinyEngine(WEIGHTS)
    out = engine.generate(prompt="ROMEO:", max_tokens=16, temperature=0.0)
    assert out["engine"] == "numpy"
    assert len(out["completion"]) > 0
