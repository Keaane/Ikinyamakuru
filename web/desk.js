// The desk page in the browser. Loads web/model/, files the paste with
// Ngram.route, and draws the same answer block as app/templates/index.html.
(function () {
  "use strict";

  const MAX_CHARS = 6000; // app/app.py keeps the first 6,000 characters
  const form = document.querySelector("form.ask");
  const box = document.getElementById("text");
  const button = form.querySelector("button[type=submit]");
  const ready = Ngram.load("model/");
  ready.catch(() => {}); // reported when the editor presses File

  // ?text= fills the box without filing, as on the Flask desk.
  const prefill = new URLSearchParams(window.location.search).get("text");
  if (prefill) box.value = Array.from(prefill).slice(0, MAX_CHARS).join("");

  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&#34;", "'": "&#39;" };
  const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => ESCAPES[ch]);
  const pct = (p) => (p * 100).toFixed(1);

  function answerHtml(result) {
    const head = result.filed
      ? `<p class="kicker">Filed</p>
        <h2 lang="rw">${esc(result.kin)}</h2>
        <p class="en">${esc(result.label)}</p>
        <p class="blurb">${esc(result.blurb)}</p>`
      : `<p class="kicker">Left for an editor</p>
        <h2>Not filed</h2>
        <p class="blurb">The top guess is <span lang="rw">${esc(result.kin)}</span> (${esc(result.label)}) at ${pct(result.confidence)}%, under the ${pct(result.threshold)}% line.</p>`;
    const tops = result.top.map((item) => `
          <li>
            <span class="kin" lang="rw">${esc(item.kin)}</span>
            <span class="pct">${pct(item.p)}%</span>
            <span class="label">${esc(item.label)}</span>
            <span class="meter" style="--p: ${pct(item.p)}"><span></span></span>
          </li>`).join("");
    const reason = result.reason;
    const key = reason.marked
      ? `<p class="key">
            <span><mark class="support">${esc(result.kin)}</mark> raised this section</span>
            ${reason.rival_kin ? `<span><mark class="rival">${esc(reason.rival_kin)}</mark> raised the next guess</span>` : ""}
          </p>`
      : `<p>${esc(reason.summary)}</p>`;
    const passage = reason.parts.map((part) =>
      part.role === "plain" ? esc(part.text) : `<mark class="${part.role}">${esc(part.text)}</mark>`).join("");
    return `
        <div class="decision">
          ${head}
          <p class="score">${pct(result.confidence)}% on the top section.</p>
          <ol class="tops">${tops}
          </ol>
        </div>
        <div class="why">
          <h3>Why this section</h3>
          ${key}
          <p class="passage" lang="rw">${passage}</p>
        </div>
      `;
  }

  function show(className, html) {
    const old = document.querySelector(".answer");
    if (old) old.remove();
    const section = document.createElement("section");
    section.className = `answer ${className}`;
    section.innerHTML = html;
    form.closest(".sheet").after(section); // the answer is wider than the form's column
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.requestAnimationFrame(() => {
      section.scrollIntoView({ block: "start", behavior: still ? "auto" : "smooth" });
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = Array.from(box.value).slice(0, MAX_CHARS).join("");
    button.disabled = true;
    try {
      const model = await ready;
      const result = model.route(text);
      show(result.filed ? "filed" : "held", answerHtml(result));
    } catch (error) {
      const message = error && error.message && error.message.startsWith("Paste")
        ? error.message
        : "The model did not load. Check the connection and reload the page.";
      show("error", `<p>${esc(message)}</p>`);
    } finally {
      button.disabled = false;
    }
  });
})();
