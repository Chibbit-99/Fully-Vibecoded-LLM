/**
 * Browser port of tinyllm/retrieval.py — BM25 over the factbook plus the
 * hand-written fallback facts.  Small enough to run on the main thread; the
 * factbook is a few hundred KB of JSON.
 *
 * Mirrors the Python ranking exactly (same idf, same length normalisation, same
 * source prior) so the demo retrieves the same passage the server would.
 */

const STOPWORDS = new Set(
  ("a an the and or but if then than that this these those of in on at to for " +
    "from by with without as is are was were be been being do does did doing have " +
    "has had having i you he she it we they me him her them my your his its our " +
    "their what which who whom whose when where why how not no nor so such only " +
    "own same too very can will just should now about into over after before " +
    "again further once here there all any both each few more most other some").split(" ")
);

const SOURCE_PRIOR = { wikipedia: 1.25, tinystories: 0.85 };

const BUILTIN_FACTS = {
  water: "Water is a clear liquid made of hydrogen and oxygen. All animals and plants need water to live.",
  fire: "Fire is the heat and light given off when something burns.",
  sun: "The Sun is the star at the centre of the solar system. It gives the Earth light and heat.",
  moon: "The Moon is the natural satellite that orbits the Earth. It is about a quarter of the Earth's width.",
  rain: "Rain is water that falls from clouds in the sky.",
  snow: "Snow is frozen water that falls from clouds as small white ice crystals.",
  wind: "Wind is air moving across the surface of the Earth.",
  earth: "The Earth is the third planet from the Sun and the only planet known to have life.",
  computer: "A computer is a machine that follows instructions to work with information.",
  book: "A book is a set of printed pages that tells a story or gives information.",
  language: "A language is a system of words that people use to speak and write to each other.",
  model: "A model is a small program that has learned patterns from examples.",
  cat: "A cat is a small furry animal that people often keep as a pet.",
  dog: "A dog is a friendly animal that people often keep as a pet and as a helper.",
  tree: "A tree is a tall plant with a wooden trunk and branches that grow leaves.",
  river: "A river is a long stream of water that flows across the land towards the sea.",
  mountain: "A mountain is a very high part of the land, much higher than the hills around it.",
  school: "A school is a place where children go to learn.",
  story: "A story is a description of people and events, real or imagined, told to entertain or inform.",
};

const TOKEN_RE = /[a-z0-9][a-z0-9'\-]*/g;

export function tokenize(text) {
  const out = [];
  const lower = String(text).toLowerCase();
  const m = lower.match(TOKEN_RE) || [];
  for (const t of m) {
    if (!STOPWORDS.has(t) && t.length > 1) out.push(t);
  }
  return out;
}

/** Crude suffix stripping — must match _stem() in tinyllm/retrieval.py. */
export function stem(word) {
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (word.length > suf.length + 3 && word.endsWith(suf)) {
      return word.slice(0, -suf.length);
    }
  }
  return word;
}

export class Factbook {
  constructor(passages, k1 = 1.5, b = 0.75) {
    this.passages = passages || [];
    this.k1 = k1;
    this.b = b;
    this.postings = new Map(); // term -> [[doc, tf], ...]
    this.docLen = [];
    this.priors = [];
    for (let i = 0; i < this.passages.length; i++) {
      const p = this.passages[i];
      const title = String(p.title || "");
      let toks = tokenize(title + " " + String(p.text || "")).map(stem);
      // repeat title tokens (weighted) so a title hit outranks a body hit
      toks = toks.concat(tokenize(title).map(stem), tokenize(title).map(stem));
      this.docLen.push(toks.length);
      this.priors.push(SOURCE_PRIOR[p.source] ?? 1.0);
      const counts = new Map();
      for (const t of toks) counts.set(t, (counts.get(t) || 0) + 1);
      for (const [term, tf] of counts) {
        if (!this.postings.has(term)) this.postings.set(term, []);
        this.postings.get(term).push([i, tf]);
      }
    }
    const total = this.docLen.reduce((a, b2) => a + b2, 0);
    this.avgLen = this.docLen.length ? total / this.docLen.length : 0;
    this.n = this.passages.length;
  }

  static fromJSON(payload) {
    const passages = Array.isArray(payload) ? payload : payload.passages || [];
    return new Factbook(passages);
  }

  find(title) {
    const t = String(title).toLowerCase().trim().replace(/[.?!]+$/, "");
    if (!t) return null;
    for (const p of this.passages) {
      if (String(p.title || "").toLowerCase() === t) return p;
    }
    for (const p of this.passages) {
      if (String(p.title || "").toLowerCase().startsWith(t)) return p;
    }
    return null;
  }

  search(query, k = 2, minScore = 0.6) {
    const terms = tokenize(query).map(stem);
    if (!terms.length || !this.n) return [];
    const scores = new Map();
    for (const term of new Set(terms)) {
      const post = this.postings.get(term);
      if (!post) continue;
      const idf = Math.log(1 + (this.n - post.length + 0.5) / (post.length + 0.5));
      for (const [doc, tf] of post) {
        const dl = this.docLen[doc] || 1;
        const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / Math.max(1e-9, this.avgLen));
        scores.set(doc, (scores.get(doc) || 0) + (idf * tf * (this.k1 + 1)) / denom);
      }
    }
    const ranked = [...scores.entries()]
      .map(([doc, s]) => [doc, s * this.priors[doc]])
      .sort((a, b2) => b2[1] - a[1])
      .slice(0, Math.max(1, k));
    const out = [];
    for (const [doc, score] of ranked) {
      if (score < minScore) continue;
      out.push({ ...this.passages[doc], score: Math.round(score * 1000) / 1000 });
    }
    return out;
  }
}

const TITLE_Q = /^(?:what|who|where)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.+?)[?.]?$/i;
const TELL_Q = /^tell me about\s+(.+?)[?.]?$/i;
const NON_LOOKUP = /\b(your|you|my|me|i|we|us|our|yourself)\b/i;

export function titleFromQuestion(question) {
  const q = String(question).trim();
  for (const re of [TITLE_Q, TELL_Q]) {
    const m = q.match(re);
    if (m) return m[1].trim();
  }
  return null;
}

export function builtinFact(title) {
  if (!title) return null;
  let key = String(title).toLowerCase().trim().replace(/[.?!]+$/, "");
  key = key.replace(/^(the|a|an)\s+/, "").replace(/s$/, "");
  return BUILTIN_FACTS[key] || null;
}

/**
 * Grounding passage for a question, or "" when nothing matches.
 * Same policy as context_for() in Python: never ground a question about the
 * assistant, and prefer no context over a weak match.
 */
export function contextFor(question, book, maxChars = 700, minScore = 6.0) {
  if (!book) return "";
  if (NON_LOOKUP.test(question)) return "";
  const title = titleFromQuestion(question);
  if (title) {
    const fact = builtinFact(title);
    if (fact) return fact.slice(0, maxChars);
    const hit = book.find(title);
    if (hit) return String(hit.text || "").slice(0, maxChars).trim();
  }
  const hits = book.search(question, 1);
  if (!hits.length || (hits[0].score || 0) < minScore) return "";
  return String(hits[0].text || "").slice(0, maxChars).trim();
}
