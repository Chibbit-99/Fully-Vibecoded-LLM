/**
 * Spore chat demo — runs the whole model in the browser.
 *
 * Flow: load weights.json (float16, ~24 MB for 12M) -> hand it to a worker ->
 * on each message, retrieve a grounding passage, render the chat prompt, and
 * stream the reply until the model emits <|end|>.
 *
 * The tokenizer and retrieval code are the same files the Python side uses, so
 * the browser produces the same ids and the same retrieval as the server.
 */
import { ByteBPE } from "./tokenizer.js";
import { Factbook, builtinFact, contextFor, titleFromQuestion } from "./retrieval.js";

// Deployment layouts (repo root vs GitHub Pages) — same trick as app.js.
const MODEL_ROOT = new URL("../model/spore/", import.meta.url);
const FALLBACK_MODEL_ROOT = new URL("./model/spore/", import.meta.url);
const FACTBOOK_CANDIDATES = [
  new URL("../data/spore/factbook.json", import.meta.url).href,
  new URL("./data/spore/factbook.json", import.meta.url).href,
  new URL("./factbook.json", import.meta.url).href,
];

const USER = "<|user|>";
const ASSISTANT = "<|assistant|>";
const END = "<|end|>";
const CONTEXT = "<|context|>";

const $ = (id) => document.getElementById(id);
const els = {
  badges: { model: $("badge-model"), params: $("badge-params"), val: $("badge-val"), ctx: $("badge-context") },
  transcript: $("transcript"),
  empty: $("empty-state"),
  suggestions: $("suggestions"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  stop: $("stop"),
  temperature: $("temperature"),
  temperatureOut: $("temperatureOut"),
  maxTokens: $("maxTokens"),
  maxTokensOut: $("maxTokensOut"),
  grounding: $("grounding"),
  reset: $("reset"),
  status: $("status"),
  picker: $("model-picker"),
  select: $("modelSelect"),
  pickerHint: $("picker-hint"),
  about: {
    params: $("about-params"), vocab: $("about-vocab"), ctx: $("about-ctx"),
    val: $("about-val"), retrieval: $("about-retrieval"),
  },
};

const SUGGESTIONS = [
  "What is water?",
  "Who is Michael Jordan?",
  "What is a cat?",
  "What is the Boat Race?",
  "How do you spell cat?",
  "What is 7 plus 5?",
  "Tell me about Oldham",
  "Are you a human?",
];

let worker = null;
let tokenizer = null;
let factbook = null;
let meta = null;
let ready = false;
let busy = false;
let history = []; // [{role, content}]
let models = [];
let blockSize = 512; // overwritten with the loaded config's block_size

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function setStatus(text) {
  els.status.textContent = text || "";
}

function badge(el, text) {
  if (el) el.textContent = text;
}

function fmt(n) {
  return typeof n === "number" ? n.toLocaleString() : String(n ?? "–");
}

async function fetchFirst(urls) {
  let lastErr = null;
  for (const url of urls) {
    try {
      const res = await fetch(url, { cache: "no-cache" });
      if (res.ok) return { res, url };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("not found: " + urls.join(", "));
}

// ---------------------------------------------------------------------------
// prompt rendering (mirrors tinyllm/sft.py)
// ---------------------------------------------------------------------------
function renderTurn({ user, context }) {
  if (context) return `${USER}${context}${CONTEXT}${user}${ASSISTANT}`;
  return `${USER}${user}${ASSISTANT}`;
}

function renderHistory(turns, context) {
  let out = "";
  let first = true;
  for (const t of turns) {
    if (t.role === "assistant") {
      out += `${ASSISTANT}${t.content}${END}`;
    } else {
      if (first && context) out += `${USER}${context}${CONTEXT}${t.content}`;
      else out += `${USER}${t.content}`;
      first = false;
    }
  }
  return out + ASSISTANT;
}

function buildPrompt(question, context) {
  const priorTurns = history.slice(0, -1); // the newest user turn is added by hand
  if (!priorTurns.length) {
    return context ? `${USER}${context}${CONTEXT}${question}${ASSISTANT}` : renderTurn({ user: question });
  }
  return renderHistory(priorTurns, context) + renderTurn({ user: question, context });
}

// Trim a rendered prompt from the left at a turn boundary so it fits the
// context window without cutting a token in half mid-word.
function fitPrompt(prompt, maxTokens) {
  let ids = tokenizer.encode(prompt);
  if (ids.length <= maxTokens) return { ids, truncated: false };
  // drop the oldest whole turns
  const userIds = tokenizer.encode(USER);
  let start = 0;
  while (ids.length > maxTokens && start < ids.length - 8) {
    const next = ids.indexOf(userIds[0], start + 1);
    if (next < 0) break;
    start = next;
  }
  ids = ids.slice(start);
  if (ids.length > maxTokens) ids = ids.slice(-maxTokens);
  return { ids, truncated: true };
}

// ---------------------------------------------------------------------------
// transcript UI
// ---------------------------------------------------------------------------
function addMessage(role, text) {
  if (els.empty) {
    els.empty.hidden = true;
    els.empty.remove();
    els.empty = null;
  }
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🌱";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = text;
  bubble.appendChild(body);
  wrap.append(avatar, bubble);
  els.transcript.appendChild(wrap);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return { wrap, bubble, body };
}

function addGrounding(bubble, passage, hits) {
  const detail = document.createElement("details");
  detail.className = "grounded";
  const summary = document.createElement("summary");
  summary.textContent = hits && hits.length
    ? `grounded on “${hits[0].title}” (score ${hits[0].score})`
    : "grounded on a built-in fact";
  const quote = document.createElement("blockquote");
  quote.textContent = passage;
  detail.append(summary, quote);
  bubble.appendChild(detail);
}

function scroll() {
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
async function listModels() {
  // Prefer the server's /api/models (it knows the filesystem); fall back to a
  // hard-coded list so the static GitHub Pages build still shows both sizes.
  const fallback = [{ name: "spore-1-12M" }, { name: "spore-1-36M" }];
  try {
    const res = await fetch(new URL("../api/models", import.meta.url), { cache: "no-cache" });
    if (res.ok) {
      const data = await res.json();
      const found = (data.models || []).map((m) => ({ name: m.name }));
      if (found.length) return found;
    }
  } catch {
    /* static hosting: expected */
  }
  // keep only the ones that actually fetch, probing both deployment layouts
  const out = [];
  for (const m of fallback) {
    for (const root of [MODEL_ROOT, FALLBACK_MODEL_ROOT]) {
      try {
        const head = await fetch(new URL(`${m.name}/weights.json`, root).href, {
          method: "HEAD",
        });
        if (head.ok) {
          out.push(m);
          break;
        }
      } catch {
        /* try the next layout */
      }
    }
  }
  return out.length ? out : fallback;
}

async function loadModel(name) {
  ready = false;
  setStatus(`loading ${name}…`);
  const candidates = [
    new URL(`${name}/weights.json`, MODEL_ROOT).href,
    new URL(`${name}/weights.json`, FALLBACK_MODEL_ROOT).href,
  ];
  const { res, url } = await fetchFirst(candidates);
  const bytes = Number(res.headers.get("content-length") || 0);
  const mb = bytes ? ` (${(bytes / 1e6).toFixed(1)} MB)` : "";
  setStatus(`fetching weights${mb}…`);
  const payload = await res.json();

  tokenizer = ByteBPE.fromJSON(payload.tokenizer);
  if (!tokenizer.specialTokens.length) {
    throw new Error("this checkpoint is not a chat model (no control tokens)");
  }
  meta = null;
  try {
    const mres = await fetch(url.replace("weights.json", "meta.json"), { cache: "no-cache" });
    if (mres.ok) meta = await mres.json();
  } catch {
    /* optional */
  }

  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.postMessage({ type: "init", payload });

  const cfg = payload.config;
  blockSize = cfg.block_size;
  badge(els.badges.model, name);
  badge(els.badges.params, `${fmt(meta?.params ?? null)} params`);
  badge(els.badges.val, meta?.best_val_loss != null ? `val ${meta.best_val_loss}` : "untrained demo");
  badge(els.badges.ctx, `ctx ${cfg.block_size}`);
  els.about.params.textContent = meta?.params ? fmt(meta.params) : "–";
  els.about.vocab.textContent = fmt(payload.config.vocab_size);
  els.about.ctx.textContent = `${cfg.block_size} tokens`;
  els.about.val.textContent = meta?.best_val_loss != null ? meta.best_val_loss : "–";
  els.about.retrieval.textContent = factbook ? `${fmt(factbook.passages.length)} passages (BM25)` : "off";
}

async function loadFactbook() {
  try {
    const { res } = await fetchFirst(FACTBOOK_CANDIDATES);
    factbook = Factbook.fromJSON(await res.json());
  } catch {
    factbook = null; // demo still works, just ungrounded
  }
  if (!factbook) {
    els.grounding.checked = false;
    els.grounding.disabled = true;
    els.grounding.parentElement.title = "factbook not found — run scripts/build_spore_data.py";
  }
}

// ---------------------------------------------------------------------------
// worker protocol
// ---------------------------------------------------------------------------
function onWorkerMessage(event) {
  const msg = event.data;
  if (msg.type === "ready") {
    ready = true;
    busy = false;
    setStatus(`ready — ${fmt(msg.params)} parameters, ${msg.vocabSize} tokens`);
    els.send.disabled = false;
    els.input.focus();
    return;
  }
  if (msg.type === "token") {
    if (!current) return;
    current.body.textContent = msg.text;
    scroll();
    return;
  }
  if (msg.type === "done") {
    if (!current) return;
    const text = (msg.text || "").trim();
    current.body.textContent = text || "(the model produced nothing)";
    current.bubble.querySelector(".caret")?.remove();
    if (pending && text) {
      history.push({ role: "assistant", content: text });
    }
    setStatus(
      msg.stopped
        ? `stopped after ${msg.produced} tokens (${msg.tokPerSec} tok/s)`
        : `${msg.produced} tokens in ${(msg.ms / 1000).toFixed(1)}s (${msg.tokPerSec} tok/s)`
    );
    finishGeneration();
    return;
  }
  if (msg.type === "error") {
    setStatus(`error: ${msg.message}`);
    if (current) current.body.textContent = "(generation failed)";
    finishGeneration();
  }
}

let current = null;
let pending = null;

function finishGeneration() {
  busy = false;
  current = null;
  pending = null;
  els.send.disabled = false;
  els.stop.hidden = true;
  els.input.focus();
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------
async function send(question) {
  if (!ready || busy) return;
  const text = String(question).trim();
  if (!text) return;

  els.suggestions.hidden = true;
  addMessage("user", text);
  history.push({ role: "user", content: text });
  pending = text;

  // ---- retrieve --------------------------------------------------------
  let passage = "";
  let hits = [];
  const useGrounding = els.grounding.checked && els.grounding.disabled === false;
  if (useGrounding && factbook) {
    passage = contextFor(text, factbook);
    if (passage) {
      const title = titleFromQuestion(text);
      const builtin = title ? builtinFact(title) : null;
      hits = builtin ? [{ title: "built-in fact", score: "1" }] : factbook.search(text, 1);
    }
  }

  // ---- render ----------------------------------------------------------
  const prompt = buildPrompt(text, passage);
  const maxNew = Number(els.maxTokens.value);
  const room = blockSize - maxNew - 8;
  const { ids, truncated } = fitPrompt(prompt, Math.max(64, room));

  if (truncated) {
    addMessage("assistant", "(trimmed the oldest turns to fit the context window)");
  }

  const shown = addMessage("assistant", "");
  const caret = document.createElement("span");
  caret.className = "caret";
  shown.body.appendChild(caret);
  current = shown;
  if (passage) addGrounding(shown.bubble, passage, hits);

  busy = true;
  els.send.disabled = true;
  els.stop.hidden = false;
  setStatus("thinking…");

  worker.postMessage({
    type: "generate",
    prompt: tokenizer.decode(ids), // send the exact trimmed text
    maxTokens: maxNew,
    temperature: Number(els.temperature.value),
    topK: 40,
    topP: 0.92,
    seed: (Date.now() & 0xffff) | 1,
    stopTokens: [END, USER, ASSISTANT],
    deterministic: Number(els.temperature.value) <= 0.05,
  });
}


// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value;
  els.input.value = "";
  send(q);
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.stop.addEventListener("click", () => {
  if (worker) worker.postMessage({ type: "cancel" });
});

els.temperature.addEventListener("input", () => {
  els.temperatureOut.textContent = els.temperature.value;
});
els.maxTokens.addEventListener("input", () => {
  els.maxTokensOut.textContent = els.maxTokens.value;
});

els.reset.addEventListener("click", () => {
  history = [];
  els.transcript.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.id = "empty-state";
  empty.innerHTML =
    '<p class="big">Ask Spore something.</p><p class="muted">It is a very small model. ' +
    "It answers best when a passage from its training data is retrieved and shown to it first.</p>";
  els.transcript.appendChild(empty);
  els.empty = empty;
  els.suggestions.hidden = false;
  setStatus("ready");
});

for (const q of SUGGESTIONS) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = q;
  b.addEventListener("click", () => send(q));
  els.suggestions.appendChild(b);
}

els.send.disabled = true;

(async () => {
  try {
    await loadFactbook();
    models = await listModels();
    if (models.length > 1) {
      els.picker.hidden = false;
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.name;
        els.select.appendChild(opt);
      }
      els.select.value = models[0].name;
      els.pickerHint.textContent = "the larger model is slower but knows more";
    }
    // prefer the small model: it loads fast and the demo stays usable on mobile
    const preferred =
      models.find((m) => /12M/i.test(m.name)) || models[0] || { name: "spore-1-12M" };
    if (els.select) els.select.value = preferred.name;
    await loadModel(preferred.name);
    els.select?.addEventListener("change", async () => {
      if (busy) return;
      history = [];
      els.reset.click();
      await loadModel(els.select.value);
    });
  } catch (err) {
    setStatus(`could not load a model: ${err.message}`);
    const warn = document.createElement("p");
    warn.className = "warn";
    warn.innerHTML =
      "No Spore checkpoint found under <code>model/spore/</code>. Train one with the " +
      "<strong>Train Spore</strong> GitHub Actions workflow, or read " +
      "<code>docs/SPORE.md</code> for the exact steps.";
    els.transcript.appendChild(warn);
  }
})();
