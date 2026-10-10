"use strict";
/* Shared rendering of one answer's audit trail, used by the results dashboard and the
   live demo. All text from data goes in through textContent. */

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.append(c);
  return node;
}
const fmtNum = (v) => (typeof v === "boolean" ? (v ? "yes" : "no") : v === null || v === undefined ? "none" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 5 }));

function verdictBadge(v) {
  const map = { accepted: ["good", "Accepted"], rejected: ["bad", "Rejected"], invalid: ["bad", "Invalid output"], abstained: ["neutral", "Abstained"] };
  const [cls, label] = map[v.status] || ["neutral", v.status];
  return el("span", { class: `badge ${cls}`, text: label });
}
function correctBadge(ok) { return el("span", { class: `badge ${ok ? "good" : "bad"}`, text: ok ? "Correct" : "Wrong" }); }

function renderDocument(doc, audit) {
  const cites = new Map(), sentCites = new Map();
  for (const e of (audit && audit.evidence) || []) {
    const key = e.source.type === "table" ? `${e.source.row},${e.source.col}` : `s${e.source.sent}`;
    const map = e.source.type === "table" ? cites : sentCites;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(e);
  }
  const t = el("table", { class: "doc-table" });
  doc.table.forEach((row, r) => {
    const tr = el("tr");
    row.forEach((cell, c) => {
      const hits = cites.get(`${r},${c}`);
      const td = el("td", { class: [r === 0 ? "first-row" : "", hits ? `cited${hits.some((e) => e.check !== "ok") ? " fail" : ""}` : ""].join(" ").trim() });
      td.append(document.createTextNode(cell));
      if (hits) for (const e of hits) td.append(el("span", { class: "cite-tag", text: e.id }));
      tr.append(td);
    });
    t.append(tr);
  });
  const sentences = el("div", { class: "sentences" });
  doc.sentences.forEach((text, i) => {
    const hits = sentCites.get(`s${i}`);
    if (!hits && doc.sentences.length > 12 && sentCites.size === 0 && i > 5) return;
    const p = el("p", { class: hits ? "cited" : "" }, [el("span", { class: "sn", text: `s${i}` }), document.createTextNode(text)]);
    if (hits) for (const e of hits) p.append(el("span", { class: "cite-tag", text: e.id }));
    sentences.append(p);
  });
  return [el("div", { class: "table-wrap" }, t), sentences];
}

function auditBlock(title, audit, correct) {
  const box = el("div", { class: "panel" }, [el("div", { class: "panel-head" }, [el("h3", { text: title }), el("div", {}, [verdictBadge(audit), document.createTextNode(" "), correctBadge(correct)])])]);
  if (audit.plan) box.append(el("div", { class: "plan", text: `plan: ${audit.plan}` }));
  box.append(el("div", { class: "kv" }, [el("span", {}, [document.createTextNode("Executed value "), el("b", { text: fmtNum(audit.value) })]), audit.answer_unit ? el("span", {}, [document.createTextNode("Unit "), el("b", { text: audit.answer_unit })]) : null]));
  if (audit.evidence && audit.evidence.length) {
    const ul = el("ul", { class: "checks" });
    for (const e of audit.evidence) {
      const where = e.source.type === "table" ? `table row ${e.source.row}, col ${e.source.col}` : `sentence ${e.source.sent}`;
      ul.append(el("li", {}, [el("span", { class: "cid", text: e.id }), el("span", { class: `badge ${e.check === "ok" ? "good" : "bad"}`, text: e.check === "ok" ? "Printed there" : e.check }),
        el("span", { text: `${fmtNum(e.value)}  ${e.label}` }), el("span", { class: "mono", style: "color:var(--text-muted)", text: `${where}: "${e.cited_text || ""}"${e.used ? "" : "  (unused)"}` })]));
    }
    box.append(ul);
  }
  for (const r of [...(audit.reasons || []), ...(audit.warnings || [])]) box.append(el("p", { class: "note", text: `${r.code}: ${r.detail}` }));
  return box;
}

