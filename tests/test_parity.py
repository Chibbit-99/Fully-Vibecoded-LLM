"""Cross-engine parity tests: PyTorch <-> NumPy <-> JavaScript.

These tests guarantee the browser demo, the Python server and the training
code all run the *same* model.  They need a trained checkpoint; set
``TINYLLM_CHECKPOINT`` to point at a ``weights.json`` (defaults to
``runs/local/weights.json``) and ``TINYLLM_CKPT_PT`` for the PyTorch
``model.pt`` used for the logits comparison.

    pytest tests -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("TINYLLM_CHECKPOINT", ROOT / "runs/local/weights.json"))
CKPT_PT = Path(os.environ.get("TINYLLM_CKPT_PT", ROOT / "runs/local/model.pt"))

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained checkpoint available")

PROMPT_CASES = [
    "ROMEO:",
    "To be, or not to be",
    "The king",
    "  spaced\ntext\t",
    "héllo wörld ✨ 1234",
    "ALL CAPS and lowercase 99.5%",
]


@needs_weights
def test_tokenizer_matches_python(tmp_path):
    from tinyllm.numpy_engine import load_checkpoint

    _, _, tok = load_checkpoint(WEIGHTS)
    cases = {c: tok.encode(c) for c in PROMPT_CASES}
    assert all(tok.decode(v) == k for k, v in cases.items()), "python round-trip failed"
    (tmp_path / "tok_cases.json").write_text(json.dumps(cases))
    # JS side is exercised through the full harness below


@needs_weights
def test_js_engine_parity(tmp_path):
    """Run scripts/js_parity.mjs and require a clean exit."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")

    from tinyllm.numpy_engine import load_checkpoint

    weights, config, tok = load_checkpoint(WEIGHTS)
    engine_weights = {k: v for k, v in weights.items()}

    from tinyllm.numpy_engine import NumpyGPT

    engine = NumpyGPT(engine_weights, config, tok)
    expected = {
        "tokenizer": {c: tok.encode(c) for c in PROMPT_CASES},
        "logits": [],
        "greedy": {"prompt": "ROMEO:", "tokens": 24, "text": ""},
    }

    for text in PROMPT_CASES:
        ids = tok.encode(text)[: config.block_size]
        logits, _ = engine.forward(ids)
        expected["logits"].append({"tokens": ids, "logits": logits[0, -1].tolist()})

    # greedy reference
    engine2 = NumpyGPT(dict(weights), config, tok)
    prompt_ids = tok.encode(expected["greedy"]["prompt"])
    produced = []
    past = None
    ctx = prompt_ids
    for _ in range(expected["greedy"]["tokens"]):
        logits, past = engine2.forward(ctx, past)
        nxt = int(np.argmax(logits[0, -1]))
        produced.append(nxt)
        ctx = [nxt]
    expected["greedy"]["text"] = tok.decode(produced)

    exp_path = tmp_path / "expected.json"
    exp_path.write_text(json.dumps(expected))
    result = subprocess.run(
        ["node", str(ROOT / "scripts" / "js_parity.mjs"), str(WEIGHTS), str(exp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout, result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_weights
@pytest.mark.skipif(not CKPT_PT.exists(), reason="no torch checkpoint")
def test_numpy_matches_torch():
    import torch

    from tinyllm.config import GPTConfig
    from tinyllm.model import GPT
    from tinyllm.numpy_engine import NumpyGPT, load_checkpoint

    ckpt = torch.load(CKPT_PT, map_location="cpu", weights_only=False)
    config = GPTConfig.from_dict(ckpt["config"])
    weights, _, tok = load_checkpoint(WEIGHTS)
    engine = NumpyGPT(weights, config, tok)
    assert abs(config.num_params() - GPT(config).num_parameters()) < 2000

    torch_model = GPT(config)
    torch_model.load_state_dict(ckpt["model"])
    torch_model.eval()

    ids = tok.encode("ROMEO: thou art")
    with torch.no_grad():
        t_logits, _, _ = torch_model(torch.tensor([ids]))
    n_logits, _ = engine.forward(ids)
    diff = float(np.abs(t_logits[0].numpy() - n_logits[0]).max())
    # float16 storage: generous but still far below a wrong-kernel error
    assert diff < 0.05, f"numpy vs torch logit diff {diff}"
    assert np.argmax(t_logits[0, -1].numpy()) == np.argmax(n_logits[0, -1])
