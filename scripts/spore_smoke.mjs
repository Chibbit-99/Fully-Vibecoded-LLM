/**
 * Headless smoke test for the Spore browser stack.
 *
 * Loads a published checkpoint exactly the way web/index.html does — tokenizer,
 * factbook, numpy engine — then runs retrieval + a short generation and asserts
 * the output is sane. Run with:
 *
 *     node scripts/spore_smoke.mjs [model-dir]
 *
 * Defaults to the smallest published model under model/spore/. Exits 0 on
 * success, 1 on failure, and prints a short report either way.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function fail(msg) {
  console.error(`[smoke] FAIL: ${msg}`);
  process.exit(1);
}

function pickModel() {
  const base = path.join(ROOT, "model", "spore");
  if (!fs.existsSync(base)) fail(`no ${base}; train a Spore model first`);
  const dirs = fs
    .readdirSync(base, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => path.join(base, d.name))
    .filter((p) => fs.existsSync(path.join(p, "weights.json")));
  if (!dirs.length) fail(`no model/spore/*/weights.json published yet`);
  // smallest file == fewest params == fastest to run on a 2-core runner
  return dirs.sort(
    (a, b) =>
      fs.statSync(path.join(a, "weights.json")).size -
      fs.statSync(path.join(b, "weights.json")).size,
  )[0];
}

const modelDir = process.argv[2] ? path.resolve(process.argv[2]) : pickModel();
const payloadPath = path.join(modelDir, "weights.json");
console.log(`[smoke] model: ${path.relative(ROOT, payloadPath)}`);

const { ByteBPE } = await import(path.join(ROOT, "web", "tokenizer.js"));
const { Factbook, contextFor } = await import(path.join(ROOT, "web", "retrieval.js"));
const { TinyGPT } = await import(path.join(ROOT, "web", "engine.js"));

const t0 = Date.now();
const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
const tok = ByteBPE.fromJSON(payload.tokenizer);
const model = new TinyGPT(payload);
const loadMs = Date.now() - t0;
console.log(
  `[smoke] loaded ${model.parameterCount.toLocaleString()} params in ${loadMs} ms`,
);

// ---- retrieval -----------------------------------------------------------
const factbookPath = path.join(ROOT, "data", "spore", "factbook.json");
if (!fs.existsSync(factbookPath)) {
  fail(`missing ${path.relative(ROOT, factbookPath)}; run scripts/build_spore_data.py`);
}
const book = Factbook.fromJSON(JSON.parse(fs.readFileSync(factbookPath, "utf8")));
const ctx = contextFor("What is water?", book);
if (!ctx || !/water/i.test(ctx)) fail(`retrieval returned no water passage: ${ctx?.slice(0, 80)}`);
console.log(`[smoke] retrieval grounded on ${ctx.length} chars`);

// ---- prompt + generation -------------------------------------------------
const question = "What is water?";
// Control tokens are plain text markers, exactly as tinyllm/sft.py renders them.
const prompt = `<|user|>${ctx}<|context|>${question}<|assistant|>`;
const ids = tok.encode(prompt);
if (!ids.length) fail("tokenizer produced zero ids");

// Control tokens the model must be able to emit to terminate a turn.
const stops = new Set(["<|end|>", "<|user|>", "<|assistant|>"].map((t) => tok.idOf(t)));
if ([...stops].some((id) => typeof id !== "number" || id < 0)) {
  fail("control tokens are missing from the vocabulary");
}

model.reset();
let logits = null;
for (const id of ids) logits = model.forward([id]);

const tGen = Date.now();
let produced = 0;
let stopped = false;
let seed = 12345;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const out = [];
while (produced < 12) {
  const next = TinyGPT.sample(logits, { temperature: 0.7, topK: 40, topP: 0.92, rng });
  if (stops.has(next)) {
    stopped = true;
    break;
  }
  out.push(next);
  logits = model.forward([next]);
  produced += 1;
}
const genMs = Date.now() - tGen;
const text = tok.decode(out);

console.log(
  `[smoke] ${ids.length} prompt tokens -> ${produced} new tokens in ${genMs} ms ` +
    `(${((produced / genMs) * 1000).toFixed(1)} tok/s), stopped=${stopped}`,
);
console.log(`[smoke] sample: ${JSON.stringify(text.slice(0, 80))}`);

if (produced === 0) fail("model produced no tokens");
if (!Number.isFinite(genMs) || genMs <= 0) fail("generation timing is invalid");
if (typeof text !== "string") fail("decode did not return text");

console.log("[smoke] OK");
