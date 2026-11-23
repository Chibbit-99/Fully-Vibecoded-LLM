# TinyLLM — convenience targets.
#
#   make install     create .venv and install everything
#   make train       train from scratch (CPU, ~90 min at the shipped size)
#   make export      re-export model/weights.json from the latest checkpoint
#   make test        full test suite
#   make demo        serve the browser demo (http://localhost:8080/web/)
#   make serve       serve the HTTP API   (http://localhost:8000)
#   make sample      generate some text from the CLI
#   make clean       remove runs/ and caches

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PORT ?= 8080
API_PORT ?= 8000
CONFIG ?= configs/small.json
STEPS ?= 2400

.PHONY: help install tokenizer train resume export test test-fast demo serve sample lint build info clean

help:
	@echo "TinyLLM targets:"
	@echo "  install     venv + dependencies"
	@echo "  tokenizer   train the BPE vocabulary only"
	@echo "  train       full training run -> runs/main/"
	@echo "  resume      continue an interrupted run"
	@echo "  export      publish runs/main -> model/"
	@echo "  test        all tests (unit + parity + API)"
	@echo "  demo        serve the browser demo"
	@echo "  serve       serve the HTTP inference API"
	@echo "  sample      print sample generations"
	@echo "  info        model metadata"
	@echo "  lint        validate the GitHub workflows"
	@echo "  clean       remove runs/ and __pycache__"

.venv/bin/python:
	python3 -m venv .venv

install: .venv/bin/python
	$(PIP) install --upgrade pip -q
	$(PIP) install numpy pytest fastapi uvicorn httpx pyyaml -q
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || $(PIP) install torch
	@echo "done — run 'make train'"

tokenizer:
	$(PY) -c "from tinyllm.tokenizer import ByteBPE; from tinyllm.dataset import load_text; t=ByteBPE.train(load_text(), 1280); t.save('model/tokenizer.json'); print('vocab', t.vocab_size)"

train:
	@mkdir -p runs/main
	@[ -f model/tokenizer.json ] && cp model/tokenizer.json runs/main/ || true
	$(PY) -u scripts/train.py --out-dir runs/main --config $(CONFIG) \
	  --tokenizer runs/main/tokenizer.json --max-steps $(STEPS)

resume:
	$(PY) -u scripts/train.py --out-dir runs/main --config $(CONFIG) \
	  --tokenizer runs/main/tokenizer.json --max-steps $(STEPS) --resume runs/main/last.pt

# copy the training artefacts into model/ exactly like the CI workflow does
export:
	$(PY) -c "import json,shutil,pathlib; \
runs=pathlib.Path('runs/main'); model=pathlib.Path('model'); model.mkdir(exist_ok=True); \
shutil.copy(runs/'weights.json', model/'weights.json'); \
shutil.copy(runs/'config.json', model/'config.json'); \
shutil.copy(runs/'tokenizer.json', model/'tokenizer.json'); \
log=json.load(open(runs/'train_log.json')); meta=dict(log['meta']); meta['history']=log['history']; \
json.dump(meta, open(model/'meta.json','w'), indent=2); \
[shutil.copy(runs/f, model/f) for f in ('loss_curve.svg','samples.txt') if (runs/f).exists()]; \
print('published model/')"

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests/test_tokenizer.py tests/test_model.py -q

demo:
	$(PY) scripts/serve_web.py --port $(PORT)

serve:
	$(PY) -m tinyllm.server --port $(API_PORT)

sample:
	$(PY) -m tinyllm generate "ROMEO:" -n 160 -t 0.85

info:
	$(PY) -m tinyllm info

lint:
	$(PY) scripts/validate_workflows.py

clean:
	rm -rf runs prepare .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "cleaned"
