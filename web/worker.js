/**
 * Web Worker: runs the transformer off the main thread so the UI stays smooth.
 * Receives weights from the main thread (transferred, zero-copy) and streams
 * generated tokens back as they are produced.
 */
import { ByteBPE } from "./tokenizer.js";
import { TinyGPT } from "./engine.js";

let model = null;
let tokenizer = null;
let busy = false;
let cancelled = false;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

self.onmessage = (event) => {
  const msg = event.data;

  if (msg.type === "init") {
    const t0 = performance.now();
    model = new TinyGPT(msg.payload);
    tokenizer = ByteBPE.fromJSON(msg.payload.tokenizer);
    self.postMessage({
      type: "ready",
      ms: Math.round(performance.now() - t0),
      params: model.parameterCount,
      config: model.cfg,
      vocabSize: tokenizer.vocabSize,
    });
    return;
  }

  if (msg.type === "cancel") {
    cancelled = true;
    return;
  }

  if (msg.type === "generate") {
    if (!model) {
      self.postMessage({ type: "error", message: "model not loaded" });
      return;
    }
    if (busy) return;
    busy = true;
    cancelled = false;
    try {
      const { prompt, maxTokens, temperature, topK, topP, seed } = msg;
      // ids of control tokens that end a turn (a chat model is trained to emit
      // <|end|>; <|assistant|>/<|user|> appearing means it went off script and
      // we must stop before it writes the user's next line)
      const stopIds = new Set(
        (msg.stopTokens || []).map((t) =>
          tokenizer.specialIds && tokenizer.specialIds.has(t) ? tokenizer.specialIds.get(t) : -1
        ).filter((i) => i >= 0)
      );
      let stopped = false;
      const rng = msg.deterministic ? () => 1e-9 : mulberry32(seed ?? 1234);
      const ids = tokenizer.encode(prompt);
      // keep the prompt inside the context window (leave room to generate)
      const ctx = model.cfg.blockSize;
      const start = Math.max(0, ids.length - Math.floor(ctx * 0.75));
      let priming = ids.slice(start);
      model.reset();
      let produced = 0;
      let text = "";
      const pieces = [];
      // prime the KV cache with the prompt, one token at a time
      let logits = null;
      for (const id of priming) logits = model.forward([id]);
      const windowLeft = Math.max(0, ctx - priming.length);

      const t0 = performance.now();
      let lastPost = t0;
      while (produced < maxTokens) {
        const next = TinyGPT.sample(logits, {
          temperature,
          topK: topK || null,
          topP: topP || null,
          rng,
        });
        if (stopIds.has(next)) {
          stopped = true;
          produced++;  // counts toward the token budget but is not shown
          break;
        }
        produced++;
        const piece = tokenizer.decode([next]);
        pieces.push({ id: next, text: piece });
        text += piece;
        if (produced % 3 === 0 || produced === maxTokens) {
          self.postMessage({
            type: "token",
            text,
            produced,
            promptTokens: ids.length,
            pieces: pieces.slice(),
          });
        }
        if (cancelled) { stopped = true; break; }
        if (produced >= windowLeft) break;
        logits = model.forward([next]);
        // keep the UI responsive on long generations
        if (produced % 16 === 0) {
          const now = performance.now();
          if (now - lastPost > 100) lastPost = now;
        }
      }
      const ms = performance.now() - t0;
      self.postMessage({
        type: "done",
        text,
        produced,
        pieces,
        stopped,
        promptTokens: ids.length,
        ms: Math.round(ms),
        tokPerSec: +(produced / (ms / 1000)).toFixed(2),
      });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
    } finally {
      busy = false;
    }
  }
};
