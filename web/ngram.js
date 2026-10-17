// The deployed char_balanced model in JavaScript: scikit-learn's char_wb TF-IDF
// (2-4 grams, sublinear tf, l2 norm) and multinomial logistic regression, plus
// the word marking from src/inkuru/predict.py. It reads the exported weights in
// web/model/ and gives the same answers as the Python desk.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Ngram = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Python's whitespace (str.split, str.strip, re's \s). JavaScript's \s also
  // matches U+FEFF and misses U+001C-U+001F and U+0085, so it is spelled out.
  const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const SPLIT = new RegExp(`[${WS}]+`, "u");
  const STRIP = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");
  // Python's [^\W\d_]: letters plus non-decimal numbers, as in predict._WORD.
  const WORD_CHAR = "\\p{L}\\p{Nl}\\p{No}";
  const WORD = new RegExp(`[${WORD_CHAR}]+(?:['’][${WORD_CHAR}]+)*`, "gu");
  const LETTER = /\p{L}/u;

  const pySplit = (text) => text.split(SPLIT).filter((part) => part.length > 0);
  const pyStrip = (text) => text.replace(STRIP, "");
  // Python's str.lower() works one code point at a time (no final-sigma rule).
  const pyLower = (text) => Array.from(text, (ch) => ch.toLowerCase()).join("");
  const letters = (text) => {
    let count = 0;
    for (const ch of text) if (LETTER.test(ch)) count += 1;
    return count;
  };
  // Python compares strings by code point.
  const byCodePoint = (a, b) => {
    const x = Array.from(a);
    const y = Array.from(b);
    for (let i = 0; i < Math.min(x.length, y.length); i += 1) {
      const d = x[i].codePointAt(0) - y[i].codePointAt(0);
      if (d !== 0) return d;
    }
    return x.length - y.length;
  };

  // sklearn's _char_wb_ngrams on lowercased text.
  function charWbNgrams(text, minN, maxN) {
    const out = [];
    for (const word of pySplit(text)) {
      const w = [" ", ...word, " "];
      for (let n = minN; n <= maxN; n += 1) {
        let offset = 0;
        out.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) {
          offset += 1;
          out.push(w.slice(offset, offset + n).join(""));
        }
        if (offset === 0) break; // a word shorter than n is counted once
      }
    }
    return out;
  }

  function fromParts(meta, names, idfBuffer, coefBuffer) {
    const labels = meta.labels;
    const k = labels.length;
    const n = names.length;
    const idf = new Float64Array(idfBuffer);
    const coef = new Float64Array(coefBuffer); // k rows of n, in labels order
    const intercept = meta.intercept;
    if (idf.length !== n || coef.length !== k * n) throw new Error("model files do not match");
    const vocab = new Map(names.map((name, index) => [name, index]));
    const [minN, maxN] = meta.ngram_range;

    // TfidfVectorizer.transform for one document: sorted indices and values.
    function vectorize(text) {
      const counts = new Map();
      for (const gram of charWbNgrams(pyLower(text), minN, maxN)) {
        const j = vocab.get(gram);
        if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
      }
      const indices = Array.from(counts.keys()).sort((a, b) => a - b);
      const values = indices.map((j) => (Math.log(counts.get(j)) + 1) * idf[j]);
      let squares = 0;
      for (const v of values) squares += v * v;
      if (squares > 0) {
        const norm = Math.sqrt(squares);
        for (let i = 0; i < values.length; i += 1) values[i] /= norm;
      }
      return { indices, values };
    }

    // decision_function then softmax, as LogisticRegression.predict_proba.
    function probabilities(row) {
      const z = new Array(k).fill(0);
      for (let i = 0; i < row.indices.length; i += 1) {
        const j = row.indices[i];
        const v = row.values[i];
        for (let c = 0; c < k; c += 1) z[c] += v * coef[c * n + j];
      }
      for (let c = 0; c < k; c += 1) z[c] += intercept[c];
      const top = Math.max(...z);
      const e = z.map((x) => Math.exp(x - top));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map((x) => x / sum);
    }

    // predict._cores: the strongest fragments, reduced to their letters.
    function cores(row, weight, limit, skip) {
      const ranked = [];
      for (let i = 0; i < row.indices.length; i += 1) {
        const j = row.indices[i];
        const core = pyLower(names[j].replaceAll(" ", ""));
        if (skip.has(core) || letters(core) < 3) continue;
        const pull = weight(j) * row.values[i];
        if (pull > 0) ranked.push([pull, core]);
      }
      ranked.sort((a, b) => (b[0] - a[0]) || byCodePoint(a[1], b[1]));
      const chosen = [];
      const seen = new Set();
      for (const [pull, core] of ranked) {
        if (seen.has(core)) continue;
        seen.add(core);
        chosen.push([pull, core]);
        if (chosen.length >= limit) break;
      }
      return chosen;
    }

    // predict._marked_passage: mark whole words, at most 6 for the section and 4 for the rival.
    function markedPassage(text, support, rival) {
      const matches = Array.from(text.matchAll(WORD));
      const bestSupport = new Map();
      const bestRival = new Map();
      for (const match of matches) {
        const word = pyLower(match[0]);
        if (letters(word) < 3) continue;
        let s = 0;
        let r = 0;
        for (const [pull, core] of support) if (word.includes(core)) s += pull;
        for (const [pull, core] of rival) if (word.includes(core)) r += pull;
        if (s > (bestSupport.get(word) || 0)) bestSupport.set(word, s);
        if (r > (bestRival.get(word) || 0)) bestRival.set(word, r);
      }
      const maxS = Math.max(0, ...bestSupport.values());
      const maxR = Math.max(0, ...bestRival.values());
      const rank = (best) => (a, b) => (best.get(b) - best.get(a)) || byCodePoint(a, b);
      const supportKeys = new Set(
        Array.from(bestSupport.keys())
          .filter((key) => maxS > 0 && bestSupport.get(key) >= 0.45 * maxS)
          .sort(rank(bestSupport))
          .slice(0, 6),
      );
      const rivalKeys = new Set(
        Array.from(bestRival.keys())
          .filter((key) => !supportKeys.has(key) && maxR > 0 && bestRival.get(key) >= 0.45 * maxR)
          .sort(rank(bestRival))
          .slice(0, 4),
      );
      const parts = [];
      let cursor = 0;
      for (const match of matches) {
        if (match.index > cursor) parts.push({ text: text.slice(cursor, match.index), role: "plain" });
        const key = pyLower(match[0]);
        const role = supportKeys.has(key) ? "support" : rivalKeys.has(key) ? "rival" : "plain";
        parts.push({ text: match[0], role });
        cursor = match.index + match[0].length;
      }
      if (cursor < text.length) parts.push({ text: text.slice(cursor), role: "plain" });
      if (parts.length === 0) parts.push({ text, role: "plain" });
      return parts;
    }

    // predict.Router._linear_reason
    function reason(text, row, winner, runner) {
      const support = cores(row, (j) => coef[winner * n + j], 25, new Set());
      const rivalWeight = (j) => coef[runner * n + j] - coef[winner * n + j];
      const rival = cores(row, rivalWeight, 15, new Set(support.map(([, core]) => core)));
      const parts = markedPassage(text, support, rival);
      const marked = parts.some((part) => part.role !== "plain");
      const label = labels[winner];
      const kin = meta.kin[label];
      const summary = marked
        ? `Words marked in the paste raised ${kin} (${label}).`
        : `No word in this paste carried a positive weight for ${kin} (${label}). ` +
          "The score is the class weight, not a phrase in the text.";
      const rivalKin = parts.some((part) => part.role === "rival") ? meta.kin[labels[runner]] : null;
      return { summary, parts, marked, rival_kin: rivalKin };
    }

    // predict.Router.route
    function route(text) {
      if (pySplit(text).length === 0) throw new Error("Paste a Kinyarwanda headline or article first.");
      const stripped = pyStrip(text);
      const row = vectorize(stripped);
      const scores = probabilities(row);
      const order = scores.map((_, i) => i).sort((a, b) => (scores[b] - scores[a]) || (a - b));
      const top = order.slice(0, 3).map((i) => ({
        label: labels[i],
        kin: meta.kin[labels[i]],
        blurb: meta.blurb[labels[i]],
        p: scores[i],
      }));
      const best = top[0];
      return {
        system: meta.system,
        label: best.label,
        kin: best.kin,
        blurb: best.blurb,
        confidence: best.p,
        filed: best.p >= meta.threshold,
        threshold: meta.threshold,
        top,
        scores,
        reason: reason(stripped, row, order[0], order[1]),
      };
    }

    return { meta, route };
  }

  async function load(base) {
    const get = async (name) => {
      const response = await fetch(base + name);
      if (!response.ok) throw new Error(`could not load ${name}`);
      return response;
    };
    const meta = await (await get("meta.json")).json();
    const [names, idf, coef] = await Promise.all([
      get("vocab.json").then((r) => r.json()),
      get("idf.f64").then((r) => r.arrayBuffer()),
      get("coef.f64").then((r) => r.arrayBuffer()),
    ]);
    return fromParts(meta, names, idf, coef);
  }

  return { load, fromParts, charWbNgrams, pyLower, pySplit, pyStrip };
});
