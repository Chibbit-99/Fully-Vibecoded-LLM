/**
 * Byte-level BPE tokenizer -- JavaScript port of tinyllm/tokenizer.py.
 *
 * The algorithm is identical to the Python implementation so that the browser
 * tokenises text exactly like the training code (verified by tests/parity).
 *
 *   id < 256         -> raw byte (NUL stands for a collapsed word-boundary space)
 *   id >= 256        -> BPE merge (id - 256)
 *   id >= 256+nMerges-> control token (<|user|>, <|assistant|>, ...)
 */

// Marker byte for a collapsed word-boundary space.  Must match
// tinyllm/tokenizer.py: _SPLIT.
const SPLIT = 0;

// High-frequency words that are split from the surrounding whitespace.  Keep
// this list in the same order as _WORD_SPLITS in tinyllm/tokenizer.py.
const WORD_SPLITS = [
  "the","and","of","to","a","in","is","was","that","it","he","she","they",
  "for","on","with","as","at","be","this","from","have","not","but","by","or",
  "his","her","which","one","all","were","when","there","can","an","their",
  "we","more","has","had","you","are","will","if",
];
const SPLIT_WORDS = new Set(WORD_SPLITS);

function isLetter(b) {
  return (b >= 65 && b <= 90) || (b >= 97 && b <= 122);
}

/**
 * Encode to UTF-8 and mark high-frequency word boundaries with NUL,
 * mirroring presplit() in tinyllm/tokenizer.py.
 */
export function presplit(text) {
  const data = new TextEncoder().encode(text);
  const out = new Uint8Array(data.length);
  let o = 0;
  let i = 0;
  const n = data.length;
  while (i < n) {
    const c = data[i];
    if (c === 32) {
      let j = i + 1;
      while (j < n && isLetter(data[j])) j += 1;
      const word = String.fromCharCode.apply(null, data.subarray(i + 1, j));
      if (SPLIT_WORDS.has(word)) {
        if (o > 0 && out[o - 1] === 32) out[o - 1] = SPLIT;
        else out[o++] = SPLIT;
        i += 1;
        continue;
      }
      out[o++] = c;
      i += 1;
      continue;
    }
    out[o++] = c;
    i += 1;
  }
  return out.subarray(0, o);
}

/** Split pre-split bytes: NUL becomes its own single-byte chunk. */
export function chunksFromPresplit(data) {
  const chunks = [];
  let start = 0;
  for (let i = 0; i <= data.length; i++) {
    if (i === data.length || data[i] === SPLIT) {
      if (i > start) chunks.push(...splitChunks(data.subarray(start, i)));
      if (i < data.length) chunks.push(data.subarray(i, i + 1));
      start = i + 1;
    }
  }
  return chunks;
}

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

/** Split so each special token is its own element (longest token first). */
export function splitOnSpecials(text, specials) {
  const ordered = specials.slice().sort((a, b) => b.length - a.length);
  const out = [];
  let buf = "";
  let i = 0;
  while (i < text.length) {
    let hit = null;
    for (const tok of ordered) {
      if (text.startsWith(tok, i)) { hit = tok; break; }
    }
    if (hit !== null) {
      if (buf) { out.push(buf); buf = ""; }
      out.push(hit);
      i += hit.length;
    } else {
      buf += text[i];
      i += 1;
    }
  }
  if (buf) out.push(buf);
  return out;
}

export class ByteBPE {
  /**
   * @param {Array<[number, number]>} merges
   * @param {string[]} specialTokens control strings appended after the merges
   */
  constructor(merges = [], specialTokens = []) {
    this.merges = merges.map((m) => [m[0], m[1]]);
    this.ranks = new Map();
    this.merges.forEach(([a, b], i) => this.ranks.set(a * 65536 + b, i));
    this.specialTokens = specialTokens.slice();
    this.specialIds = new Map();
    this.specialTokens.forEach((t, i) => this.specialIds.set(t, 256 + this.merges.length + i));
    this.vocabSize = 256 + this.merges.length + this.specialTokens.length;
    this._cache = new Map(); // chunk string -> ids (speed for repeated words)
  }

  static fromJSON(data) {
    if (!data || data.type !== "byte-bpe") throw new Error("invalid tokenizer format");
    return new ByteBPE(data.merges, data.special_tokens || []);
  }

  /** Id of a control token, e.g. idOf("<|assistant|>"). */
  idOf(token) {
    if (!this.specialIds.has(token)) throw new Error(`unknown special token ${token}`);
    return this.specialIds.get(token);
  }

  /** @param {string} text @returns {number[]} */
  encode(text) {
    if (this.specialTokens.length) {
      const pieces = splitOnSpecials(text, this.specialTokens);
      if (pieces.length > 1 || (pieces.length && this.specialIds.has(pieces[0]))) {
        const out = [];
        for (const piece of pieces) {
          const sid = this.specialIds.get(piece);
          if (sid !== undefined) out.push(sid);
          else out.push(...this._encodePlain(piece));
        }
        return out;
      }
    }
    return this._encodePlain(text);
  }

  _encodePlain(text) {
    const out = [];
    // escape literal NULs as a NUL pair (see tinyllm/tokenizer.py)
    text = text.split("\u0000").join("\u0000\u0000");
    for (const chunk of chunksFromPresplit(presplit(text))) {
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
    const nMerges = this.merges.length;
    const enc = new TextEncoder();
    for (const id of ids) {
      if (id < 0 || id >= this.vocabSize) throw new Error(`token id ${id} outside vocabulary`);
      if (id >= 256 + nMerges) {
        const tok = this.specialTokens[id - 256 - nMerges];
        for (const b of enc.encode(tok)) bytes.push(b);
      } else {
        this._expand(id, bytes);
      }
    }
    // A NUL pair is an escaped literal NUL; a lone NUL is a word boundary.
    const raw = new Uint8Array(bytes.length);
    let j = 0;
    let i = 0;
    while (i < bytes.length) {
      if (bytes[i] === SPLIT) {
        if (i + 1 < bytes.length && bytes[i + 1] === SPLIT) { raw[j++] = 0; i += 2; }
        else { raw[j++] = 32; i += 1; }
      } else {
        raw[j++] = bytes[i++];
      }
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(raw.subarray(0, j));
  }
}
