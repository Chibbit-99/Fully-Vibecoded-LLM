/**
 * Demo app: loads the exported checkpoint, hands it to a Web Worker and
 * streams generated text into the page.
 *
 * The weights file lives at `model/weights.json` when served from the repo
 * root (`python scripts/serve_web.py`), which is also how GitHub Pages sees it.
 */
import { ByteBPE } from "./tokenizer.js";

const WEIGHTS_URL = new URL("../model/weights.json", import.meta.url).href;
const META_URL = new URL("../model/meta.json", import.meta.url).href;

const $ = (id) => document.getElementById(id);
const els = {
  prompt: $("prompt"),
  generate: $("generate"),
  stop: $("stop"),
  output: $("output"),
  stats: $("stats"),
  progress: $("progress"),
  badges: {
    params: $("badge-params"),
    config: $("badge-config"),
    val: $("badge-val"),
  },
  about: {
    params: $("about-params"),
    vocab: $("about-vocab"),
    ctx: $("about-ctx"),
    val: $("about-val"),
  },
  loadStatus: $("load-status"),
};

const PROMPTS = [
  "ROMEO:",
  "HAMLET:\nTo be, or not to be",
  "KING HENRY:\nOnce more unto the breach",
  "JULIET:\nO Romeo, Romeo",
  "MERCUTIO:\nA plague o' both your houses",
  "SECOND CITIZEN:\nThe people",
  "MACBETH:\nOut, damned spot",
  "PORTIA:\nThe quality of mercy",
  "First Musician:\n",
  "DUKE:\nIf music be the food of love",
];

let worker = null;
let ready = false;
let generating = false;
let promptTokens = 0;

function setOutput(prompt, text, streaming) {
  const escape = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  els.output.innerHTML =
    `<span class="prompt-part">${escape(prompt)}</span>` +
    `<span class="new-part">${escape(text)}</span>` +
    (streaming ? '<span class="caret"></span>' : "");
  els.output.scrollTop = els.output.scrollHeight;
}

function setBusy(busy) {
  generating = busy;
  els.generate.disabled = !ready || busy;
  els.stop.disabled = !busy;
  els.generate.textContent = busy ? "Generating…" : "Generate";
}

async function init() {
  els.loadStatus.textContent = "Loading weights…";
  try {
    const metaResp = await fetch(META_URL);
    if (metaResp.ok) {
      const meta = await metaResp.json();
      if (meta.val_loss !== undefined) {
        els.badges.val.textContent = `val loss ${meta.val_loss}`;
        els.badges.val.classList.add("ok");
        els.about.val.textContent = `${meta.val_loss} (${meta.bits_per_byte} bits/byte)`;
      }
    }
  } catch {
    /* meta.json is optional */
  }

  try {
    const resp = await fetch(WEIGHTS_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${WEIGHTS_URL}`);
    const payload = await resp.json();

    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => {
      els.loadStatus.innerHTML = `<span class="error">worker error: ${e.message}</span>`;
    };
    worker.postMessage({ type: "init", payload });
    els.loadStatus.textContent = "Compiling model…";
  } catch (err) {
    els.loadStatus.innerHTML =
      `<span class="error">Could not load weights: ${err.message}. ` +
      `Serve the repo root (python scripts/serve_web.py) and make sure model/weights.json exists.</span>`;
  }
}

function onWorkerMessage(event) {
  const msg = event.data;
  if (msg.type === "ready") {
    ready = true;
    const p = msg.params;
    els.badges.params.textContent = `${(p / 1e6).toFixed(2)}M params`;
    els.badges.params.classList.add("ok");
    els.badges.config.textContent = `${msg.config.nLayer}L · ${msg.config.nHead}H · d${msg.config.nEmbd} · ctx ${msg.config.blockSize}`;
    els.about.params.textContent = p.toLocaleString();
    els.about.vocab.textContent = msg.vocabSize.toLocaleString();
    els.about.ctx.textContent = msg.config.blockSize;
    els.loadStatus.textContent = `Model ready in ${msg.ms} ms — running locally`;
    setBusy(false);
    els.stats.textContent = "";
    return;
  }
  if (msg.type === "token") {
    const prompt = els.prompt.value;
    promptTokens = msg.promptTokens;
    setOutput(prompt, msg.text, true);
    const max = +$("maxTokens").value;
    els.progress.style.width = `${Math.min(100, (msg.produced / max) * 100)}%`;
    return;
  }
  if (msg.type === "done") {
    setBusy(false);
    els.progress.style.width = "100%";
    setOutput(els.prompt.value, msg.text, false);
    els.stats.textContent =
      `${msg.produced} tokens · ${msg.tokPerSec} tok/s · ${msg.ms} ms` +
      (msg.promptTokens ? ` · ${msg.promptTokens} prompt tokens` : "");
    return;
  }
  if (msg.type === "error") {
    setBusy(false);
    els.stats.innerHTML = `<span class="error">${msg.message}</span>`;
  }
}

function generate() {
  if (!ready || generating) return;
  const prompt = els.prompt.value;
  const maxTokens = +$("maxTokens").value;
  els.progress.style.width = "0%";
  setBusy(true);
  setOutput(prompt, "", true);
  els.stats.textContent = "…";
  worker.postMessage({
    type: "generate",
    prompt,
    maxTokens,
    temperature: +$("temperature").value,
    topK: +$("topK").value,
    topP: +$("topP").value,
    seed: document.getElementById("seedLock").checked ? 1337 : Math.floor(Math.random() * 1e9),
  });
}

function stop() {
  if (!worker || !generating) return;
  worker.terminate();
  worker = null;
  ready = false;
  setBusy(false);
  els.stats.textContent = "stopped";
  init(); // respawn the worker so the button works again
}

// ---- wire up UI -----------------------------------------------------------
for (const [id, outId] of [
  ["maxTokens", "maxTokensOut"],
  ["temperature", "temperatureOut"],
  ["topK", "topKOut"],
  ["topP", "topPOut"],
]) {
  const input = $(id);
  const out = $(outId);
  const sync = () => (out.textContent = input.value);
  input.addEventListener("input", sync);
  sync();
}

els.generate.addEventListener("click", generate);
els.stop.addEventListener("click", stop);
els.prompt.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    generate();
  }
});
$("shuffle").addEventListener("click", () => {
  els.prompt.value = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  els.prompt.focus();
});

init();
