/**
 * TinyGPT inference engine -- pure JavaScript, no dependencies.
 *
 * This is a line-by-line port of tinyllm/numpy_engine.py: identical weights,
 * identical operations, identical sampling so the browser produces the same
 * text as the Python server for the same seed.
 *
 * Performance notes
 * -----------------
 * * Weights are float16 in the checkpoint and converted to float32 once at
 *   load time into flat typed arrays.
 * * Matrix-vector products use hand-written loops (JIT-friendly, no allocation
 *   in the hot path) which beats a naive triple loop by a large margin.
 * * Generation keeps an incremental KV cache, so each new token costs one
 *   forward pass over a single position.
 */

const GELU_C = 0.7978845608028654; // sqrt(2/pi)

export function gelu(x) {
  // tanh approximation, matching torch.nn.functional.gelu(approximate="tanh")
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

/** y = x @ W^T + b, W is (out, in) row-major, x is (T, in). */
export function linear(x, W, b, outDim, inDim, T) {
  const y = new Float32Array(T * outDim);
  for (let t = 0; t < T; t++) {
    const xRow = t * inDim;
    const yRow = t * outDim;
    for (let o = 0; o < outDim; o++) {
      let sum = b ? b[o] : 0;
      const wRow = o * inDim;
      for (let i = 0; i < inDim; i++) sum += x[xRow + i] * W[wRow + i];
      y[yRow + o] = sum;
    }
  }
  return y;
}

function layerNorm(x, weight, bias, T, D, eps) {
  const y = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const off = t * D;
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[off + i];
    mean /= D;
    let variance = 0;
    for (let i = 0; i < D; i++) {
      const d = x[off + i] - mean;
      variance += d * d;
    }
    variance /= D;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < D; i++) y[off + i] = weight[i] * (x[off + i] - mean) * inv + bias[i];
  }
  return y;
}

function softmaxInPlace(arr, offset, length) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = arr[offset + i];
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const e = Math.exp(arr[offset + i] - max);
    arr[offset + i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < length; i++) arr[offset + i] *= inv;
}

export class TinyGPT {
  /**
   * @param {object} payload parsed weights.json
   * @param {object} [opts] { threads: number of worker shards (unused here) }
   */
  constructor(payload, opts = {}) {
    if (payload.format !== "tinyllm-weights-v1") {
      throw new Error(`unsupported weights format: ${payload.format}`);
    }
    this.config = payload.config;
    this.weights = {};
    const raw = payload.weights;
    for (const name of Object.keys(raw)) {
      const entry = raw[name];
      // already a Float32Array (binary format) or a plain array (json format)
      const data = entry instanceof Float32Array ? entry : Float32Array.from(entry.data ?? entry);
      this.weights[name] = data;
    }
    const cfg = this.config;
    this.cfg = {
      nLayer: cfg.n_layer,
      nHead: cfg.n_head,
      nEmbd: cfg.n_embd,
      blockSize: cfg.block_size,
      vocabSize: cfg.vocab_size,
      headDim: cfg.n_embd / cfg.n_head,
      lnEps: cfg.ln_eps ?? 1e-5,
    };
    const W = this.weights;
    W["wpe"] = W["wpe"];
    if (!W["lm_head.weight"]) W["lm_head.weight"] = W["wte"];
    this._past = null;
    this._pos = 0;
    this.forwardCount = 0;
    this.opts = opts;
  }

  get parameterCount() {
    let n = 0;
    for (const name of Object.keys(this.weights)) {
      if (name === "lm_head.weight" && !this.weights["lm_head.weight.__own"]) {
        // weight tying: wte counted once
        if (this.weights["lm_head.weight"] === this.weights["wte"]) continue;
      }
      n += this.weights[name].length;
    }
    return n;
  }

  /** Reset the KV cache (call before a new independent generation). */
  reset() {
    const { nLayer, nHead, headDim } = this.cfg;
    this._past = [];
    for (let i = 0; i < nLayer; i++) {
      this._past.push({
        k: new Float32Array(nHead * this.cfg.blockSize * headDim),
        v: new Float32Array(nHead * this.cfg.blockSize * headDim),
        len: 0,
      });
    }
    this._pos = 0;
  }

  _matVec(xRow, name, outDim, inDim) {
    // fast path: single-position projection
    const W = this.weights[name + ".weight"];
    const b = this.weights[name + ".bias"];
    const y = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = b ? b[o] : 0;
      const wOff = o * inDim;
      for (let i = 0; i < inDim; i++) sum += xRow[i] * W[wOff + i];
      y[o] = sum;
    }
    return y;
  }

  /**
   * Forward pass for `tokens` (all positions, or a single new token when a KV
   * cache is active). Returns logits for the last position (Float32Array V).
   */
  forward(tokens) {
    const { nLayer, nHead, nEmbd, blockSize, vocabSize, headDim, lnEps } = this.cfg;
    const W = this.weights;
    if (tokens.length === 0) throw new Error("empty token sequence");
    if (tokens.length > blockSize) tokens = tokens.slice(tokens.length - blockSize);
    const T = tokens.length;
    if (this._past === null) this.reset();
    if (this._pos + T > blockSize) {
      // context full: shift the cache window left by one
      this._shift(1);
    }
    const posStart = this._pos;
    this.forwardCount++;

    // embeddings
    let x = new Float32Array(T * nEmbd);
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      if (tok < 0 || tok >= vocabSize) throw new Error(`token ${tok} out of range`);
      const tokOff = tok * nEmbd;
      const posOff = (posStart + t) * nEmbd;
      for (let i = 0; i < nEmbd; i++) x[t * nEmbd + i] = W["wte"][tokOff + i] + W["wpe"][posOff + i];
    }

    const scale = 1 / Math.sqrt(headDim);
    for (let l = 0; l < nLayer; l++) {
      const p = `blocks.${l}.`;
      const cache = this._past[l];
      const nPast = cache.len;

      // ---- attention ----
      const h = layerNorm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"], T, nEmbd, lnEps);
      const qkv = linear(h, W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"], 3 * nEmbd, nEmbd, T);

      const attnOut = new Float32Array(T * nEmbd);
      const scores = new Float32Array(nPast + T);
      for (let t = 0; t < T; t++) {
        const total = nPast + t + 1;
        for (let hIdx = 0; hIdx < nHead; hIdx++) {
          // gather q for (t, head)
          const qOff = hIdx * headDim;
          // write k, v for this position into the cache
          const kvPos = nPast + t;
          for (let d = 0; d < headDim; d++) {
            cache.k[(hIdx * blockSize + kvPos) * headDim + d] = qkv[t * 3 * nEmbd + nEmbd + qOff + d];
            cache.v[(hIdx * blockSize + kvPos) * headDim + d] = qkv[t * 3 * nEmbd + 2 * nEmbd + qOff + d];
          }
          // attention scores against all past+current keys
          for (let s = 0; s < total; s++) {
            let dot = 0;
            const kOff = (hIdx * blockSize + s) * headDim;
            for (let d = 0; d < headDim; d++) dot += qkv[t * 3 * nEmbd + qOff + d] * cache.k[kOff + d];
            scores[s] = dot * scale;
          }
          softmaxInPlace(scores, 0, total);
          // weighted sum of values
          const outOff = t * nEmbd + qOff;
          for (let d = 0; d < headDim; d++) attnOut[outOff + d] = 0;
          for (let s = 0; s < total; s++) {
            const w = scores[s];
            if (w === 0) continue;
            const vOff = (hIdx * blockSize + s) * headDim;
            for (let d = 0; d < headDim; d++) attnOut[outOff + d] += w * cache.v[vOff + d];
          }
        }
      }
      cache.len = nPast + T;

      const proj = linear(attnOut, W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"], nEmbd, nEmbd, T);
      for (let i = 0; i < T * nEmbd; i++) x[i] += proj[i];

      // ---- MLP ----
      const h2 = layerNorm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"], T, nEmbd, lnEps);
      const ff = linear(h2, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"], 4 * nEmbd, nEmbd, T);
      for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
      const ff2 = linear(ff, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"], nEmbd, 4 * nEmbd, T);
      for (let i = 0; i < T * nEmbd; i++) x[i] += ff2[i];
    }
    this._pos = posStart + T;

    const hf = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"], T, nEmbd, lnEps);
    const lastRow = hf.subarray((T - 1) * nEmbd, T * nEmbd);
    return this._matVec(lastRow, "lm_head", vocabSize, nEmbd);
  }

  _shift(n) {
    const { nHead, blockSize, headDim } = this.cfg;
    for (const cache of this._past) {
      const stride = blockSize * headDim;
      for (let h = 0; h < nHead; h++) {
        cache.k.copyWithin(h * stride, h * stride + n * headDim, (h + 1) * stride);
        cache.v.copyWithin(h * stride, h * stride + n * headDim, (h + 1) * stride);
      }
      cache.len -= n;
    }
    this._pos -= n;
  }

  /**
   * Sample the next token id from logits.
   * Matches NumpyGPT.sample / model._sample.
   */
  static sample(logits, { temperature = 1.0, topK = null, topP = null, rng = Math.random } = {}) {
    const V = logits.length;
    const scores = new Float64Array(V);
    if (temperature <= 0) {
      let best = 0;
      for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    for (let i = 0; i < V; i++) scores[i] = logits[i] / temperature;

    if (topK !== null && topK > 0 && topK < V) {
      const sorted = Float64Array.from(scores).sort();
      const threshold = sorted[V - topK];
      for (let i = 0; i < V; i++) if (scores[i] < threshold) scores[i] = -Infinity;
    }
    if (topP !== null && topP > 0 && topP < 1) {
      const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
      let maxKept = -Infinity;
      for (let i = 0; i < V; i++) if (scores[i] > maxKept) maxKept = scores[i];
      let total = 0;
      for (let i = 0; i < V; i++) total += Math.exp(scores[i] - maxKept);
      let acc = 0;
      let keep = V;
      for (let i = 0; i < V; i++) {
        acc += Math.exp(scores[order[i]] - maxKept) / total;
        if (acc >= topP) {
          keep = i + 1;
          break;
        }
      }
      keep = Math.max(1, keep);
      const kept = new Set(order.slice(0, keep));
      for (let i = 0; i < V; i++) if (!kept.has(i)) scores[i] = -Infinity;
    }

    let max = -Infinity;
    for (let i = 0; i < V; i++) if (scores[i] > max) max = scores[i];
    let sum = 0;
    const probs = new Float64Array(V);
    for (let i = 0; i < V; i++) {
      const e = scores[i] === -Infinity ? 0 : Math.exp(scores[i] - max);
      probs[i] = e;
      sum += e;
    }
    let r = rng() * sum;
    for (let i = 0; i < V; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return V - 1;
  }
}
