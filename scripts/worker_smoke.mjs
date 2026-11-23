/**
 * Browser-protocol smoke test, run under Node.
 *
 * Imports web/worker.js with a tiny `self` shim so the exact code the browser
 * executes (worker -> engine -> tokenizer) is exercised without a headless
 * browser.  Fails if the worker does not report ready, errors, or produces no
 * text.
 *
 *   node scripts/worker_smoke.mjs model/weights.json
 */
import fs from "node:fs";

const weightsPath = process.argv[2] || "model/weights.json";
if (!fs.existsSync(weightsPath)) {
  console.error(`missing ${weightsPath} — train a model first`);
  process.exit(2);
}

const payload = JSON.parse(fs.readFileSync(weightsPath, "utf8"));
const messages = [];
let onmessage = null;
globalThis.self = {
  postMessage: (m) => messages.push(m),
  set onmessage(fn) {
    onmessage = fn;
  },
};

const workerUrl = new URL("../web/worker.js", import.meta.url).href;
await import(workerUrl + "?t=" + Date.now());

onmessage({ data: { type: "init", payload } });
const ready = messages.find((m) => m.type === "ready");
if (!ready) {
  console.error("worker never became ready:", messages.slice(0, 3));
  process.exit(1);
}
console.log(
  `ready in ${ready.ms} ms | ${ready.params.toLocaleString()} params | ` +
    `${ready.config.nLayer}L d${ready.config.nEmbd} ctx${ready.config.blockSize}`
);

onmessage({
  data: {
    type: "generate",
    prompt: "ROMEO:",
    maxTokens: 24,
    temperature: 0.8,
    topK: 40,
    topP: 0.95,
    seed: 1337,
  },
});

const error = messages.find((m) => m.type === "error");
if (error) {
  console.error("worker error:", error.message);
  process.exit(1);
}
const done = messages.find((m) => m.type === "done");
const streamed = messages.filter((m) => m.type === "token").length;
if (!done || done.produced !== 24) {
  console.error("generation incomplete:", { done, streamed });
  process.exit(1);
}
if (typeof done.text !== "string" || done.text.length === 0) {
  console.error("empty generation");
  process.exit(1);
}
console.log(`streamed ${streamed} updates | ${done.produced} tokens @ ${done.tokPerSec} tok/s`);
console.log(`output: ${JSON.stringify(done.text)}`);
console.log("\nWORKER SMOKE TEST PASSED");
