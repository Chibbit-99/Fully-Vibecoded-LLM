# TinyLLM / Spore — convenience targets.
#
#   make install     create .venv and install everything
#   make data        build the Spore corpus + retrieval factbook (~2 s)
#   make train       stage 1 (pretrain) then stage 2 (chat tuning)
#   make resume      continue an interrupted stage
#   make test        full test suite
#   make demo        serve the Spore playground (http://localhost:8080/)
#   make serve       serve the HTTP API     (http://localhost:8000)
#   make smoke       headless end-to-end check of the browser stack
#   make info        model metadata
#   make lint        validate the GitHub workflows
#   make clean       remove runs/ and caches

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PORT ?= 8080
API_PORT ?= 8000
MODEL ?= spore-1-12M
CONFIG ?= configs/spore_12m.json
OUT ?= runs/$(MODEL)
PRETRAIN_TOKENS ?= 40000000
SFT_TOKENS ?= 12000000

.PHONY: help install data train pretrain sft resume test test-fast demo serve smoke info lint clean

help:
	@echo "Spore targets:"
	@echo "  install     venv + dependencies"
	@echo "  data        build the corpus and factbook"
	@echo "  train       pretrain then chat-tune (both stages)"
	@echo "  pretrain    stage 1 only -> $(OUT)/"
	@echo "  sft         stage 2 only, from stage 1's weights"
	@echo "  resume      continue an interrupted stage"
	@echo "  test        all tests"
	@echo "  demo        serve the browser playground"
	@echo "  serve       serve the HTTP inference API"
	@echo "  smoke       run the browser stack end-to-end in Node"
	@echo "  info        model metadata"
	@echo "  lint        validate the GitHub workflows"
	@echo "  clean       remove runs/ and __pycache__"

.venv/bin/python:
	python3 -m venv .venv

install: .venv/bin/python
	$(PIP) install --upgrade pip -q
	$(PIP) install numpy pytest fastapi uvicorn httpx pyyaml -q
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || $(PIP) install torch
	@echo "done — run 'make data' then 'make train'"

data:
	$(PY) scripts/build_spore_data.py

pretrain:
	$(PY) -u scripts/train_spore.py --config $(CONFIG) --out-dir $(OUT) \
	  --stage pretrain --token-budget $(PRETRAIN_TOKENS) --export

sft:
	$(PY) -u scripts/train_spore.py --config $(CONFIG) --out-dir $(OUT) \
	  --stage sft --token-budget $(SFT_TOKENS) --export

train: pretrain sft

resume:
	$(PY) -u scripts/train_spore.py --config $(CONFIG) --out-dir $(OUT) \
	  --stage pretrain --token-budget $(PRETRAIN_TOKENS) --export \
	  --resume $(OUT)/last.pt

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests/test_tokenizer.py tests/test_model.py tests/test_spore.py -q

demo:
	$(PY) scripts/serve_web.py --port $(PORT)

serve:
	$(PY) -m tinyllm.server --port $(API_PORT)

smoke:
	node scripts/spore_smoke.mjs

info:
	$(PY) -m tinyllm info

lint:
	$(PY) scripts/validate_workflows.py

clean:
	rm -rf runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "cleaned"
