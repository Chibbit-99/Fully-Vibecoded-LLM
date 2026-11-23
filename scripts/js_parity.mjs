/**
 * Cross-engine parity harness (run with Node).
 *
 *   node scripts/js_parity.mjs <weights.json> <expected.json>
 *
 * Verifies the browser engine against the Python engines:
 *   1. tokenizer ids match Python for a set of tricky strings
 *   2. full-sequence logits match the NumPy engine (printed for comparison)
 *   3. fixed-seed greedy generation reproduces the Python reference text
 *
 * The Python side writes `expected.json` (see tests/test_parity.py).
 */
import fs from "node:fs";
import { ByteBPE } from "../web/tokenizer.js";
import { TinyGPT } from "../web/engine.js";

const [weightsPath, expectedPath] = process.argv.slice(2);
if (!weightsPath || !expectedPath) {
  console.error("usage: node scripts/js_parity.mjs <weights.json> <expected.json>");
  process.exit(2);
}

const t0 = Date.now();
const payload = JSON.parse(fs.readFileSync(weightsPath, "utf8"));
const expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
const model = new TinyGPT(payload);
const tokenizer = ByteBPE.fromJSON(payload.tokenizer);
const loadMs = Date.now() - t0;

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};

console.log(
  `loaded ${(weightsPath.split("/").pop())} in ${loadMs} ms | ` +
    `params=${model.parameterCount.toLocaleString()} vocab=${tokenizer.vocabSize} ` +
    `layers=${model.cfg.nLayer} d=${model.cfg.nEmbd} ctx=${model.cfg.blockSize}`
);

// ---- 1. tokenizer ---------------------------------------------------------
let tokOk = true;
for (const [text, ids] of Object.entries(expected.tokenizer)) {
  const got = tokenizer.encode(text);
  const same = got.length === ids.length && got.every((v, i) => v === ids[i]);
  if (!same) {
    tokOk = false;
    console.log(`  mismatch for ${JSON.stringify(text)}:\n   js=${got.slice(0, 30)}\n   py=${ids.slice(0, 30)}`);
  }
}
check(`tokenizer parity (${Object.keys(expected.tokenizer).length} strings)`, tokOk);

// ---- 2. logits ------------------------------------------------------------
let maxDiff = 0;
let bestOf = true;
for (const { tokens, logits } of expected.logits) {
  model.reset();
  const got = model.forward(tokens);
  for (let i = 0; i < got.length; i++) maxDiff = Math.max(maxDiff, Math.abs(got[i] - logits[i]));
  const argmax = (a) => a.indexOf(Math.max(...a));
  if (argmax(logits) !== argmax(Array.from(got))) bestOf = false;
}
check(
  `logits parity (${expected.logits.length} sequences, vocab=${expected.logits[0].logits.length})`,
  maxDiff < 2e-2,
  `max|diff|=${maxDiff.toExponential(3)} (float16 tolerance)`
);
check("argmax parity on last logits", bestOf);

// ---- 3. greedy generation -------------------------------------------------
if (expected.greedy) {
  tokenizer.encode(expected.greedy.prompt);
  model.reset();
  const ids = tokenizer.encode(expected.greedy.prompt);
  const produced = [];
  const tGen = Date.now();
  for (let i = 0; i < ids.length - 1; i++) model.forward([ids[i]]); // prime cache
  let next = model.forward([ids[ids.length - 1]]);
  for (let n = 0; n < expected.greedy.tokens; n++) {
    let best = 0;
    for (let i = 1; i < next.length; i++) if (next[i] > next[best]) best = i;
    produced.push(best);
    next = model.forward([best]);
  }
  const genMs = Date.now() - tGen;
  const text = tokenizer.decode(produced);
  check(
    `greedy generation (${expected.greedy.tokens} tokens, ${(genMs / expected.greedy.tokens).toFixed(0)} ms/token)`,
    text === expected.greedy.text,
    text === expected.greedy.text ? "" : `\n   js=${JSON.stringify(text.slice(0, 120))}\n   py=${JSON.stringify(expected.greedy.text.slice(0, 120))}`
  );
}

console.log(failures === 0 ? "\nALL PARITY CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
