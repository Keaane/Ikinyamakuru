// Answers for scripts/web_parity.py: runs web/ngram.js on every input in the JSON file named by the
// first argument and prints one result per input, in the shape the Python side compares.
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const Ngram = require(path.join(root, "web", "ngram.js"));
const file = (name) => fs.readFileSync(path.join(root, "web", "model", name));
const buffer = (name) => {
  const bytes = file(name);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const model = Ngram.fromParts(
  JSON.parse(file("meta.json")), JSON.parse(file("vocab.json")), buffer("idf.f64"), buffer("coef.f64"),
);

const inputs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const answers = inputs.map((text) => {
  try {
    const result = model.route(text);
    return {
      label: result.label,
      filed: result.filed,
      top: result.top.map((item) => item.label),
      scores: result.scores,
      marked: result.reason.parts.filter((part) => part.role !== "plain").map((part) => [part.role, part.text]),
    };
  } catch (error) {
    return { error: error.message };
  }
});
process.stdout.write(JSON.stringify(answers));
