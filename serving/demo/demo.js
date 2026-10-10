"use strict";
/* Live demo page: pick a document, ask the served model, render the verifier's audit trail.
   Rendering helpers come from the dashboard's audit.js. */

const GROUP_TITLE = { test: "FinQA test questions", counterfactual: "Counterfactual documents (edited figures)" };

async function getJson(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function renderResult(host, ex, res) {
  host.replaceChildren();
  host.append(el("div", { class: "stats" }, [
    el("span", {}, [document.createTextNode("Latency "), el("b", { text: `${res.latency_s.toFixed(2)} s` })]),
    el("span", {}, [document.createTextNode("Output tokens "), el("b", { text: String(res.completion_tokens ?? "n/a") })]),
    res.gold !== null ? el("span", {}, [document.createTextNode("Gold "), el("b", { text: fmtNum(res.gold) })]) : el("span", { text: "Custom question: no gold answer" }),
  ]));
  const cols = el("div", { class: "answer-cols" }, [auditBlock("LedgerMind GRPO, live", res.audit, res.correct)]);
  if (res.correct === null) cols.querySelectorAll(".badge").forEach((b) => { if (b.textContent === "Wrong") b.remove(); });
  host.append(...renderDocument(ex.document, res.audit), cols);
  host.append(el("details", { style: "margin-top:10px" }, [el("summary", { class: "note", text: "Raw model output" }), el("div", { class: "raw", text: res.raw })]));
}

async function main() {
  const [info, examples] = await Promise.all([getJson("/api/info"), getJson("/api/examples")]);
  const status = document.getElementById("status");
  status.append(el("span", {}, [el("span", { class: "live-dot" }), el("b", { text: info.model_label })]), el("span", { text: `on ${info.gpu}, vLLM` }));
  const list = document.getElementById("q-list"), detail = document.getElementById("q-detail");
  const buttons = [];
  let last = null;
  examples.forEach((ex, i) => {
    if (ex.source !== last) { list.append(el("div", { class: "q-group", text: GROUP_TITLE[ex.source] })); last = ex.source; }
    const b = el("button", { class: "q-item", type: "button", role: "option" }, [el("span", { class: "qid", text: ex.id }), document.createTextNode(ex.question)]);
    b.addEventListener("click", () => select(i));
    buttons.push(b); list.append(b);
  });
  function select(i) {
    buttons.forEach((b) => b.removeAttribute("aria-current"));
    buttons[i].setAttribute("aria-current", "true");
    const ex = examples[i];
    const input = el("input", { type: "text", value: ex.question, "aria-label": "Question" });
    const run = el("button", { class: "run", type: "button", text: "Run on GPU" });
    const out = el("div");
    detail.replaceChildren(...[
      el("div", { class: "q-group", style: "margin:0", text: `${GROUP_TITLE[ex.source]}  ·  ${ex.id}` }),
      ex.original_gold !== null ? el("div", { class: "kv" }, [el("span", {}, [document.createTextNode("Answer in the real filing "), el("b", { text: fmtNum(ex.original_gold) })]), el("span", {}, [document.createTextNode("Answer for this edited document "), el("b", { text: fmtNum(ex.gold) })])]) : null,
      el("div", { class: "ask" }, [input, run]),
      out,
    ].filter(Boolean));
    out.append(...renderDocument(ex.document, null));
    run.addEventListener("click", async () => {
      run.disabled = true; run.textContent = "Running...";
      try {
        const res = await getJson("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: ex.key, question: input.value }) });
        renderResult(out, ex, res);
      } catch (err) {
        out.replaceChildren(el("p", { class: "empty", text: `Request failed: ${err.message}` }));
      } finally { run.disabled = false; run.textContent = "Run on GPU"; }
    });
  }
  if (examples.length) select(0);
}
main().catch((err) => document.querySelector("main").prepend(el("p", { class: "empty", text: `Demo failed to load: ${err.message}` })));
