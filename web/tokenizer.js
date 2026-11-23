/**
 * Byte-level BPE tokenizer -- JavaScript port of tinyllm/tokenizer.py.
 *
 * The algorithm is identical to the Python implementation so that the browser
 * tokenises text exactly like the training code (verified by tests/parity).
 *
 *   id < 256         -> raw byte
 *   id >= 256        -> BPE merge (id - 256)
 */

const LETTER = 1;
const DIGIT = 2;
const SPACE = 3;
const OTHER = 4;

function byteClass(b) {
  if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) return LETTER;
  if (b >= 48 && b <= 57) return DIGIT;
  if (b === 32 || b === 9 || b === 10 || b === 13) return SPACE;
  return OTHER;
}

/** Split encoded bytes into pre-tokenisation chunks (mirrors split_chunks). */
export function splitChunks(bytes) {
  const chunks = [];
  if (bytes.length === 0) return chunks;
  let start = 0;
  let prev = byteClass(bytes[0]);
  for (let i = 1; i < bytes.length; i++) {
    const cur = byteClass(bytes[i]);
    if (cur !== prev || cur === OTHER) {
      chunks.push(bytes.subarray(start, i));
      start = i;
    }
    prev = cur;
  }
  chunks.push(bytes.subarray(start));
  return chunks;
}

function applyMerges(chunk, ranks) {
  let parts = Array.from(chunk);
  while (parts.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < parts.length - 1; i++) {
      const r = ranks.get(parts[i] * 65536 + parts[i + 1]);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestIndex = i;
      }
    }
    if (bestIndex < 0) break;
    parts.splice(bestIndex, 2, 256 + bestRank);
  }
  return parts;
}

export class ByteBPE {
  /** @param {Array<[number, number]>} merges */
  constructor(merges = []) {
    this.merges = merges.map((m) => [m[0], m[1]]);
    this.ranks = new Map();
    this.merges.forEach(([a, b], i) => this.ranks.set(a * 65536 + b, i));
    this.vocabSize = 256 + this.merges.length;
    this._cache = new Map(); // chunk string -> ids (speed for repeated words)
  }

  static fromJSON(data) {
    if (!data || data.type !== "byte-bpe") throw new Error("invalid tokenizer format");
    return new ByteBPE(data.merges);
  }

  /** @param {string} text @returns {number[]} */
  encode(text) {
    const out = [];
    const bytes = new TextEncoder().encode(text);
    for (const chunk of splitChunks(bytes)) {
      const key = chunk.length <= 32 ? String.fromCharCode.apply(null, chunk) : null;
      if (key !== null && this._cache.has(key)) {
        for (const id of this._cache.get(key)) out.push(id);
        continue;
      }
      const ids = applyMerges(chunk, this.ranks);
      if (key !== null) this._cache.set(key, ids);
      for (const id of ids) out.push(id);
    }
    return out;
  }

  _expand(token, out) {
    if (token < 256) {
      out.push(token);
      return;
    }
    const [a, b] = this.merges[token - 256];
    this._expand(a, out);
    this._expand(b, out);
  }

  /** @param {Iterable<number>} ids @returns {string} */
  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (id < 0 || id >= this.vocabSize) throw new Error(`token id ${id} outside vocabulary`);
      this._expand(id, bytes);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }
}
