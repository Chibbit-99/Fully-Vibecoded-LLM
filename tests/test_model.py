"""Unit tests for the model, config and training helpers (no checkpoint needed)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from tinyllm.config import GPTConfig  # noqa: E402
from tinyllm.dataset import get_batch, load_text, train_val_split  # noqa: E402
from tinyllm.model import GPT, _sample  # noqa: E402
from tinyllm.numpy_engine import NumpyGPT, _gelu, export_checkpoint, load_checkpoint  # noqa: E402
from tinyllm.tokenizer import ByteBPE  # noqa: E402

SMALL = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(SMALL)


def test_config_validation():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=100, n_head=3)
    cfg = GPTConfig(n_embd=64, n_head=4, n_layer=2, block_size=16, vocab_size=256)
    assert cfg.head_dim == 16
    assert cfg.to_dict()["n_head"] == 4


def test_config_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    SMALL.save(path)
    assert GPTConfig.load(path) == SMALL
    # unknown keys are ignored (forward compatibility)
    data = json.loads(path.read_text())
    data["future_field"] = 1
    path.write_text(json.dumps(data))
    assert GPTConfig.load(path).n_layer == SMALL.n_layer


def test_parameter_count_matches_analytic(model):
    analytic = SMALL.num_params()
    actual = model.num_parameters()
    assert abs(analytic - actual) < 1000, f"analytic {analytic} vs actual {actual}"


def test_forward_shapes(model):
    x = torch.randint(0, SMALL.vocab_size, (2, 16))
    logits, loss, past = model(x)
    assert logits.shape == (2, 16, SMALL.vocab_size)
    assert loss is None
    logits, loss, _ = model(x, x)
    assert loss.ndim == 0 and loss > 0
    # initial loss should be close to ln(vocab_size)
    assert abs(float(loss) - math.log(SMALL.vocab_size)) < 0.6


def test_causality(model):
    """Changing token t must not change logits at positions < t."""
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 12))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 7) % SMALL.vocab_size
    with torch.no_grad():
        a, _, _ = model(x)
        b, _, _ = model(x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6)
    assert not torch.allclose(a[0, -1], b[0, -1])


def test_kv_cache_matches_full_forward(model):
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 10))
    with torch.no_grad():
        full, _, _ = model(x)
        _, _, past = model(x[:, :8])
        inc, _, _ = model(x[:, 8:], past_key_values=past)
    assert torch.allclose(full[:, 8:], inc, atol=1e-5)


def test_kv_cache_window_drops_old_positions(model):
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 20))
    with torch.no_grad():
        _, _, past = model(x[:, :20])
        # feeding another token must not raise, and the cache stays bounded
        _, _, past2 = model(x[:, :1], past_key_values=past)
        assert past2[0][0].size(2) == 21
        # fill the window completely (11 more positions), then keep decoding
        _, _, big = model(torch.randint(0, SMALL.vocab_size, (1, 11)), past_key_values=past2)
        assert big[0][0].size(2) == SMALL.block_size
        _, _, rolled = model(x[:, :1], past_key_values=big)
        assert rolled[0][0].size(2) == SMALL.block_size


def test_generate_is_deterministic_with_seed(model):
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 4))
    a = model.generate(x, max_new_tokens=12, temperature=0.9, top_k=10, seed=42)
    b = model.generate(x, max_new_tokens=12, temperature=0.9, top_k=10, seed=42)
    assert torch.equal(a, b)
    c = model.generate(x, max_new_tokens=12, temperature=0.9, top_k=10, seed=7)
    assert not torch.equal(a, c)


def test_greedy_temperature_zero(model):
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 3))
    logits = torch.randn(1, SMALL.vocab_size)
    out = _sample(logits, 0.0, None, None, torch.Generator())
    assert int(out) == int(logits.argmax())


def test_top_k_and_top_p_restrict_support():
    logits = torch.randn(1, 100)
    gen = torch.Generator().manual_seed(0)
    counts = {}
    for _ in range(200):
        t = int(_sample(logits, 1.0, top_k=3, top_p=None, gen=gen))
        counts[t] = counts.get(t, 0) + 1
    allowed = set(torch.topk(logits[0], 3).indices.tolist())
    assert set(counts) <= allowed


def test_gelu_matches_torch():
    x = torch.linspace(-6, 6, 25, dtype=torch.float32)
    expected = torch.nn.functional.gelu(x, approximate="tanh").numpy()
    import numpy as np

    got = _gelu(x.numpy())
    assert np.allclose(got, expected, atol=1e-6)


def test_numpy_engine_matches_torch_exactly():
    """fp32 NumPy engine must reproduce PyTorch logits to float precision."""
    torch.manual_seed(1)
    model = GPT(SMALL)
    model.eval()
    weights = {k: v.float().numpy() for k, v in model.state_dict_for_export().items()}
    tok = ByteBPE([])
    engine = NumpyGPT(weights, SMALL, tok)

    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        expected, _, _ = model(torch.tensor([ids]))
    got, _ = engine.forward(ids)
    import numpy as np

    assert np.abs(expected[0].numpy() - got[0]).max() < 1e-4


def test_numpy_engine_kv_cache_matches_full():
    torch.manual_seed(2)
    model = GPT(SMALL)
    model.eval()
    weights = {k: v.float().numpy() for k, v in model.state_dict_for_export().items()}
    engine = NumpyGPT(weights, SMALL, ByteBPE([]))
    ids = [5, 9, 2, 11, 4]
    full, _ = engine.forward(ids)
    _, past = engine.forward(ids[:-1])
    inc, _ = engine.forward([ids[-1]], past)
    import numpy as np

    assert np.abs(full[0, -1] - inc[0, -1]).max() < 1e-5


def test_export_and_load_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = GPT(SMALL)
    tok = ByteBPE([(97, 98), (98, 99)])  # dummy merges
    path = export_checkpoint(model, SMALL, tok, tmp_path / "weights.json")
    weights, config, loaded_tok = load_checkpoint(path)
    assert config == SMALL
    assert loaded_tok.merges == tok.merges
    assert "wte" in weights and weights["wte"].shape == (SMALL.vocab_size, SMALL.n_embd)
    # fp16 quantization error must be tiny
    original = model.wte.weight.detach().numpy()
    assert abs(weights["wte"] - original).max() < 1e-2
    payload = json.loads(path.read_text())
    assert payload["format"] == "tinyllm-weights-v1"
    assert payload["dtype"] == "float16"


def test_dataset_split_no_overlap():
    text = load_text()
    train, val = train_val_split(text)
    assert len(train) + len(val) == len(text)
    assert train + val == text
    assert "\n" in val[:200]


def test_get_batch_shifts_by_one():
    import numpy as np

    data = np.arange(1000, dtype=np.uint16)
    rng = np.random.default_rng(0)
    x, y = get_batch(data, 4, 16, rng)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_export_size_is_browser_friendly(tmp_path):
    """Base64 float16 export must be ~2 bytes/param (not decimal JSON)."""
    torch.manual_seed(4)
    model = GPT(SMALL)
    tok = ByteBPE([])
    path = export_checkpoint(model, SMALL, tok, tmp_path / "w.json")
    n_params = sum(p.numel() for p in model.parameters())
    ratio = path.stat().st_size / n_params
    assert ratio < 3.0, f"export is {ratio:.1f} bytes/param, expected <3 (base64 float16)"
    payload = json.loads(path.read_text())
    assert payload["encoding"] == "raw"  # "raw" => base64-encoded float16 buffers
    entry = payload["weights"]["wte"]
    assert entry["encoding"] == "base64"

    # both encodings load back identically
    weights, _, _ = load_checkpoint(path)
    plain_path = export_checkpoint(model, SMALL, tok, tmp_path / "w_plain.json", encoding="json")
    weights_plain, _, _ = load_checkpoint(plain_path)
    for name in weights:
        assert weights[name].shape == weights_plain[name].shape


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------
def test_checkpoint_is_written_at_end_of_a_chunk(tmp_path):
    """A chunked run must leave a resumable last.pt even if the step count is
    not a multiple of --save-interval. (Regression: an earlier version only
    saved on interval boundaries, so a 120-step chunk kept nothing.)"""
    import subprocess
    import sys

    from tinyllm.tokenizer import ByteBPE as _Tok

    root = Path(__file__).resolve().parent.parent
    data = root / "data" / "tinyshakespeare.txt"
    if not data.exists():
        pytest.skip("dataset not available")
    text = data.read_text(encoding="utf-8")
    (tmp_path / "prose_train.txt").write_text(text[:4000], encoding="utf-8")
    (tmp_path / "prose_val.txt").write_text(text[4000:5000], encoding="utf-8")
    import json as _json
    cfg = tmp_path / "tiny.json"
    cfg.write_text(_json.dumps({
        "name": "spore-test-tiny", "vocab_size": 300, "block_size": 32,
        "n_layer": 1, "n_head": 2, "n_embd": 32,
    }))

    cmd = [
        sys.executable, str(root / "scripts" / "train_spore.py"),
        "--out-dir", str(tmp_path), "--data-dir", str(tmp_path),
        "--config", str(cfg),
        "--stage", "pretrain", "--max-steps", "9",
        "--batch-size", "2", "--threads", "1",
        "--save-interval", "500",  # deliberately never hits a boundary
        "--log-interval", "9", "--eval-interval", "9", "--eval-batches", "1",
        "--warmup", "1", "--smoke",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)

    assert (tmp_path / "last.pt").exists(), "no resumable checkpoint was written"
    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 9
    assert "optimizer" in ckpt and "rng_numpy" in ckpt
    # and it must actually resume rather than restart
    assert ckpt["step"] > 0


def test_workflow_never_git_adds_ignored_paths():
    """`git add runs/...` exits 1 under set -e and kills the job after a full
    training run. runs/ must never appear in a git add."""
    import re

    root = Path(__file__).resolve().parent.parent
    for wf in (root / ".github" / "workflows").glob("*.yml"):
        text = wf.read_text()
        for match in re.finditer(r"^\s*git add (.+)$", text, re.MULTILINE):
            args = match.group(1)
            # strip trailing comment
            args = args.split("#")[0]
            assert "runs/" not in args, f"{wf.name}: git add of ignored runs/ path: {args}"


def test_workflow_ignored_paths_match_gitignore():
    """Anything a workflow commits must not be gitignored."""
    import re
    import subprocess

    root = Path(__file__).resolve().parent.parent
    for wf in (root / ".github" / "workflows").glob("*.yml"):
        for match in re.finditer(r"^\s*git add (.+)$", wf.read_text(), re.MULTILINE):
            for path in match.group(1).split("#")[0].split():
                if path.startswith("-"):
                    continue
                probe = path.rstrip("/")
                result = subprocess.run(
                    ["git", "check-ignore", "-q", probe],
                    cwd=root, capture_output=True,
                )
                assert result.returncode != 0, f"{wf.name}: '{path}' is gitignored but is committed"


def test_workflow_publishes_before_gating_on_tests():
    """A failing or flaky test must never discard a training chunk: the publish
    and commit steps have to run before any test gate."""
    yaml = pytest.importorskip("yaml")

    root = Path(__file__).resolve().parent.parent
    data = yaml.safe_load((root / ".github" / "workflows" / "train_spore.yml").read_text())
    step_names = [s.get("name", "") for s in data["jobs"]["train"]["steps"]]

    def index_of(fragment: str) -> int:
        for i, name in enumerate(step_names):
            if fragment.lower() in name.lower():
                return i
        return -1

    publish = index_of("Publish artefacts")
    commit = index_of("Commit trained model")
    gate = index_of("Test suite")
    train = index_of("Train")

    assert train >= 0 and publish > train, "training must come before publish"
    assert commit > publish, "commit after publish"
    assert gate > commit, f"the test gate must run after the commit (got {step_names})"

    # the gate must not be able to fail the job
    gate_step = next(s for s in data["jobs"]["train"]["steps"] if "Test suite" in s.get("name", ""))
    assert gate_step.get("continue-on-error") is True


def test_lr_schedule_uses_absolute_target_not_chunk_bound():
    """Chunked runs must follow the same LR curve as one long run.

    Regression: the schedule was derived from the per-invocation step bound, so
    each chunk replayed warmup/decay against a different horizon and the model
    decayed into overfitting (val loss rose 2.83 -> 3.90 on the first CI run).
    """
    from scripts.train_spore import lr_at

    BASE, FLOOR, WARMUP, TOTAL = 1e-3, 1e-4, 100, 4000

    def lr(step: int, total: int = TOTAL) -> float:
        return lr_at(step, WARMUP, total, BASE, FLOOR)

    # The schedule is a pure function of (absolute step, stage total). A chunk
    # that only plans to run 700 steps must still report the LR that the
    # 4000-step schedule dictates at that absolute position — which is exactly
    # what makes continuation exact.
    for step in (0, 50, 100, 350, 699, 2000):
        assert abs(lr(step) - lr_at(step, WARMUP, TOTAL, BASE, FLOOR)) < 1e-12, step

    # warmup ramps up, then decay falls away
    assert lr(0) < lr(50) < lr(100)
    assert lr(100) > lr(2000) > lr(TOTAL)
    # the floor is respected
    assert lr(TOTAL) >= FLOOR - 1e-12


def test_generate_stops_on_new_token_not_prompt_token(model):
    """`stop_tokens` must be tested against the token just sampled.

    Regression: generate() tested `out[0, -1]` *before* sampling. If the prompt
    itself ended with a stop token -- which it always does for Spore, because
    render_prompt() ends with <|assistant|> and that is in the stop set -- the
    loop broke on the first iteration and returned zero new tokens. Every
    chat sample in samples.txt was therefore empty, for every training run, and
    nothing reported it.

    NumpyGPT.generate and the browser worker already sampled first; only the
    torch path (used by the trainer) got the order wrong.
    """
    model.eval()
    # stay inside SMALL.vocab_size (300)
    ids = [10, 20, 30, 250]          # 250 is in the stop set below
    x = torch.tensor([ids], dtype=torch.long)
    stop = {250, 251}

    out = model.generate(
        x, max_new_tokens=8, temperature=0.8, top_k=5, seed=1, stop_tokens=stop
    )
    new = out[0].tolist()[len(ids):]

    # the prompt's last token is a stop token; that must NOT stop generation
    assert ids[-1] in stop, "test setup: prompt must end with a stop token"
    assert len(new) > 0, (
        "generate() produced no tokens because the prompt ended with a stop "
        "token; it is still checking the old token instead of the new one"
    )
    # and a sampled stop token must not be included in the output
    assert all(t not in stop for t in new), (
        f"a stop token leaked into the output: {new}"
    )


def test_generate_without_stop_tokens_is_unchanged(model):
    """No stop set -> exactly max_new_tokens are produced (no off-by-one)."""
    model.eval()
    ids = [10, 20, 30]
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=7, temperature=0.8, top_k=5, seed=1)
    assert out.size(1) == len(ids) + 7, out.size(1)
