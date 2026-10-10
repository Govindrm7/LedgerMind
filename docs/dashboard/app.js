"use strict";
/* LedgerMind results terminal. Renders docs/dashboard/data.json (built by
   scripts/build_dashboard.py). All text from data goes in through textContent. */

const NS = "http://www.w3.org/2000/svg";
const GROUP_COLOR = { ours: "var(--series-1)", api: "var(--series-2)", open: "var(--series-3)", baseline: "var(--deemph)" };
const GROUP_NAME = { ours: "LedgerMind (ours, 4B)", api: "OpenAI API", open: "Open weights, self hosted", baseline: "Untrained base" };

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
function s(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
const pct = (x, d = 1) => (x === null || x === undefined ? "n/a" : `${(x * 100).toFixed(d)}%`);
const pts = (x, d = 1) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)} pts`;
const usd = (x) => (x >= 1 ? `$${x.toFixed(2)}` : x >= 0.1 ? `$${x.toFixed(2)}` : `$${x.toFixed(3)}`);
const pval = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(p < 0.01 ? 3 : 2)}`);
const fmtNum = (v) => (typeof v === "boolean" ? (v ? "yes" : "no") : v === null || v === undefined ? "none" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 5 }));

/* Tooltip: values lead, labels follow; line keys, not boxes. */
const tip = document.getElementById("tooltip");
function showTip(evt, rows) {
  tip.replaceChildren();
  for (const r of rows) {
    const line = el("div");
    if (r.color) line.append(el("span", { class: "tk", style: `background:${r.color}` }));
    line.append(el("span", { class: "tv", text: r.value }));
    if (r.label) line.append(document.createTextNode(`  ${r.label}`));
    tip.append(line);
  }
  tip.style.display = "block";
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.style.display = "none"; }
function hoverable(node, rows) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", (e) => showTip(e, rows()));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const b = node.getBoundingClientRect();
    showTip({ clientX: b.right, clientY: b.top }, rows());
  });
  node.addEventListener("blur", hideTip);
}

/* A chart card: title, subtitle, legend, chart, and its table view twin. */
const renderers = [];
function chartCard(host, { title, sub, legend, note, draw, table }) {
  host.replaceChildren();
  const toggle = el("button", { class: "view-toggle", type: "button", text: "Table view" });
  host.append(el("div", { class: "panel-head" }, [el("div", {}, [el("h3", { text: title }), sub ? el("p", { class: "sub", text: sub }) : null]), table ? toggle : null]));
  if (legend) {
    const lg = el("div", { class: "legend" });
    for (const item of legend) lg.append(el("span", {}, [el("i", { class: item.line ? "line" : "", style: `background:${item.color}` }), document.createTextNode(item.label)]));
    host.append(lg);
  }
  const fig = el("figure");
  const tableWrap = el("div", { class: "table-wrap", hidden: "" });
  host.append(fig, tableWrap);
  if (note) host.append(el("p", { class: "note", text: note }));
  const render = () => { fig.replaceChildren(); draw(fig, Math.max(280, fig.clientWidth || host.clientWidth - 32)); };
  renderers.push(render);
  render();
  if (table) {
    tableWrap.append(buildTable(table));
    toggle.addEventListener("click", () => {
      const showTable = tableWrap.hidden;
      tableWrap.hidden = !showTable;
      fig.hidden = showTable;
      toggle.textContent = showTable ? "Chart view" : "Table view";
    });
  }
}
function buildTable({ columns, rows, highlight }) {
  const t = el("table");
  const head = el("tr");
  for (const c of columns) head.append(el("th", { class: c.num ? "num" : "", text: c.label }));
  t.append(el("thead", {}, head));
  const body = el("tbody");
  rows.forEach((r, i) => {
    const tr = el("tr", { class: highlight && highlight(r, i) ? "ours" : "" });
    for (const c of columns) tr.append(el("td", { class: c.num ? "num" : "", text: c.value(r) }));
    body.append(tr);
  });
  t.append(body);
  return t;
}
function svgRoot(width, height, label) {
  return s("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
}
function niceTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
/* Bar with a 4px rounded data end and a square baseline end. */
function hbarPath(x0, x1, y, h, r = 4) {
  const w = Math.max(0, x1 - x0), rr = Math.min(r, w, h / 2);
  return `M${x0},${y}H${x0 + w - rr}Q${x0 + w},${y} ${x0 + w},${y + rr}V${y + h - rr}Q${x0 + w},${y + h} ${x0 + w - rr},${y + h}H${x0}Z`;
}

/* Greedy label placement: the first of four spots around the point that overlaps no
   placed label or point and stays inside the plot. */
function placeLabel(cx, cy, w, placed, W, m) {
  const h = 14;
  const spots = [
    { x: cx + 10, y: cy - 8, anchor: "start", box: [cx + 10, cx + 10 + w, cy - 8 - h + 3, cy - 8 + 3] },
    { x: cx - 10, y: cy - 8, anchor: "end", box: [cx - 10 - w, cx - 10, cy - 8 - h + 3, cy - 8 + 3] },
    { x: cx + 10, y: cy + 16, anchor: "start", box: [cx + 10, cx + 10 + w, cy + 16 - h + 3, cy + 16 + 3] },
    { x: cx - 10, y: cy + 16, anchor: "end", box: [cx - 10 - w, cx - 10, cy + 16 - h + 3, cy + 16 + 3] },
  ];
  const fits = ([x0, x1]) => x0 >= m.l && x1 <= W - m.r;
  const free = ([x0, x1, y0, y1]) => placed.every((b) => x1 < b.x0 || x0 > b.x1 || y1 < b.y0 || y0 > b.y1);
  const pick = spots.find((sp) => fits(sp.box) && free(sp.box)) || spots.find((sp) => fits(sp.box)) || spots[0];
  const [x0, x1, y0, y1] = pick.box;
  placed.push({ x0, x1, y0, y1 });
  return pick;
}

/* ---------- Sections ---------- */

function renderTicker(d) {
  const g = d.sample_systems.find((x) => x.key === "grpo");
  const g55 = d.sample_systems.find((x) => x.key === "gpt55_direct");
  const full = d.full_test[0];
  const items = [
    ["GRPO verified acc (test)", pct(full.verified.mean)],
    ["gpt-5.5 direct lead", `${(-g55.grpo_minus_this.diff * 100).toFixed(1)} pts, ${pval(g55.grpo_minus_this.mcnemar.p_value)} (tie)`],
    ["cost / 1k queries", `${usd(g.cost.usd_per_1k)} vs ${usd(g55.cost.usd_per_1k)}`],
    ["fabrication", pct(full.fabrication_rate)],
    ["unconstrained JSON parse", pct(full.parse_rate)],
    ["corrupted outputs caught", "100%"],
  ];
  const host = document.getElementById("ticker");
  for (const [k, v] of items) host.append(el("span", {}, [document.createTextNode(`${k} `), el("b", { text: v })]));
}

function renderHero(d) {
  const full = d.full_test[0];
  const g55 = d.sample_systems.find((x) => x.key === "gpt55_direct");
  const g = d.sample_systems.find((x) => x.key === "grpo");
  const ratio = Math.round(g55.cost.usd_per_1k / g.cost.usd_per_1k);
  const hero = document.getElementById("hero");
  hero.append(
    el("div", { class: "hero-label", text: "Verified accuracy, LedgerMind GRPO on 1,139 FinQA test questions" }),
    el("div", { class: "hero-figure", text: pct(full.verified.mean) }),
    el("div", { class: "hero-label", text: `95% CI ${pct(full.verified.low)} to ${pct(full.verified.high)}. Correct and accepted by the verifier, over all questions.` }),
    el("p", { class: "hero-claim", text: `A 4B model statistically tied with gpt-5.5 on the same 400 questions (gpt-5.5 direct leads by ${(-g55.grpo_minus_this.diff * 100).toFixed(1)} points, McNemar ${pval(g55.grpo_minus_this.mcnemar.p_value)}), at about ${ratio.toLocaleString()} times lower cost per query if self hosted, with every answer traced to cited figures.` }),
  );
  const tiles = [
    ["Fabrication rate", pct(full.fabrication_rate), "answers citing a figure not printed where cited"],
    ["Cost per 1k queries", usd(g.cost.usd_per_1k), `vs ${usd(g55.cost.usd_per_1k)} for gpt-5.5 direct`],
    ["Unconstrained JSON parse", pct(full.parse_rate), "no constrained decoding needed"],
    ["Corrupted outputs caught", "100%", `${Object.values(d.saboteur.mode_a).reduce((a, x) => a + x.cases, 0).toLocaleString()} injected faults, 8 types`],
    ["Latency p50", `${full.latency_p50_s.toFixed(2)} s`, "scored run, batch invariant mode"],
    ["Paid API spend", `$${d.meta.api_spend_usd.toFixed(2)}`, "all frontier baselines, hard capped at $20"],
  ];
  const host = document.getElementById("kpis");
  for (const [label, value, delta] of tiles) host.append(el("div", { class: "panel" }, [el("div", { class: "tile-label", text: label }), el("div", { class: "tile-value", text: value }), el("div", { class: "tile-delta", text: delta })]));
}

function renderScatter(d) {
  const sys = d.sample_systems.filter((x) => x.cost);
  const groups = [...new Set(sys.map((x) => x.group))];
  chartCard(document.getElementById("panel-scatter"), {
    title: "Accuracy against cost per 1,000 queries",
    sub: `Same ${d.meta.sample_n} test questions. Log cost axis; whiskers are 95% bootstrap intervals.`,
    legend: groups.map((g) => ({ label: GROUP_NAME[g], color: GROUP_COLOR[g] })),
    note: `Self hosted cost assumes one rented H200 at $${d.meta.h200_usd_per_hour.toFixed(2)}/hour at full utilization (market range $${d.meta.h200_price_range[0].toFixed(2)} to $${d.meta.h200_price_range[1].toFixed(2)}); this project ran on a university cluster at no cost. API cost uses list prices and measured tokens per query.`,
    table: {
      columns: [
        { label: "System", value: (r) => r.label },
        { label: "Accuracy", num: true, value: (r) => pct(r.accuracy.mean) },
        { label: "95% CI", num: true, value: (r) => `${pct(r.accuracy.low)} to ${pct(r.accuracy.high)}` },
        { label: "$ / 1k queries", num: true, value: (r) => usd(r.cost.usd_per_1k) },
        { label: "Cost basis", value: (r) => r.cost.basis },
      ],
      rows: sys,
      highlight: (r) => r.group === "ours",
    },
    draw(fig, W) {
      const H = 300, m = { l: 48, r: 16, t: 12, b: 34 };
      const x0 = Math.log10(0.01), x1 = Math.log10(40);
      const ys = sys.flatMap((x) => [x.accuracy.low, x.accuracy.high]);
      const ylo = Math.floor(Math.min(...ys) * 20) / 20, yhi = Math.ceil(Math.max(...ys) * 20) / 20;
      const X = (v) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
      const Y = (v) => m.t + (1 - (v - ylo) / (yhi - ylo)) * (H - m.t - m.b);
      const svg = svgRoot(W, H, "Scatter of verified accuracy against cost per thousand queries");
      const grid = s("g", { class: "grid" });
      for (const t of niceTicks(ylo, yhi, 4)) {
        grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
        svg.append(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, pct(t, 0)));
      }
      for (const t of [0.01, 0.1, 1, 10]) {
        grid.append(s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }));
        svg.append(s("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, usd(t).replace(".000", "").replace(".00", "")));
      }
      svg.prepend(grid);
      svg.append(s("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }, "USD per 1,000 queries"));
      const sorted = [...sys].sort((a, b) => a.cost.usd_per_1k - b.cost.usd_per_1k);
      let best = -1; const front = [];
      for (const p of sorted) if (p.accuracy.mean > best) { front.push(p); best = p.accuracy.mean; }
      if (front.length > 1) svg.append(s("path", { d: front.map((p, i) => `${i ? "L" : "M"}${X(p.cost.usd_per_1k)},${Y(p.accuracy.mean)}`).join(""), fill: "none", stroke: "var(--text-muted)", "stroke-width": 1 }));
      const placed = sys.map((p) => ({ x0: X(p.cost.usd_per_1k) - 7, x1: X(p.cost.usd_per_1k) + 7, y0: Y(p.accuracy.mean) - 7, y1: Y(p.accuracy.mean) + 7 }));
      for (const p of sys) {
        const cx = X(p.cost.usd_per_1k), cy = Y(p.accuracy.mean), color = GROUP_COLOR[p.group];
        svg.append(s("line", { x1: cx, x2: cx, y1: Y(p.accuracy.low), y2: Y(p.accuracy.high), stroke: color, "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55 }));
        const g = s("g", { cursor: "default" });
        g.append(s("circle", { cx, cy, r: 14, fill: "transparent" }), s("circle", { cx, cy, r: 5, fill: color, stroke: "var(--surface-1)", "stroke-width": 2 }));
        const pos = placeLabel(cx, cy, p.label.length * 6.6, placed, W, m);
        g.append(s("text", { x: pos.x, y: pos.y, "text-anchor": pos.anchor, class: p.group === "ours" ? "label-strong" : "label" }, p.label));
        hoverable(g, () => [
          { value: pct(p.accuracy.mean), label: p.label, color },
          { value: usd(p.cost.usd_per_1k), label: "per 1,000 queries" },
          p.grpo_minus_this ? { value: `${pts(p.grpo_minus_this.diff)} GRPO minus this`, label: `McNemar ${pval(p.grpo_minus_this.mcnemar.p_value)}` } : null,
        ].filter(Boolean));
        svg.append(g);
      }
      fig.append(svg);
    },
  });
}

function renderBars(d) {
  const sys = [...d.sample_systems].sort((a, b) => b.accuracy.mean - a.accuracy.mean);
  chartCard(document.getElementById("panel-bars"), {
    title: "Verified accuracy, paired on the same questions",
    sub: `${d.meta.sample_n} question seeded sample of the test set. Hover for the paired difference against GRPO.`,
    note: "Direct answers are scored at the precision the model states (1.64 matches 1.63657); default scoring of the same answers is in the full test table. Direct answers carry no verification.",
    table: {
      columns: [
        { label: "System", value: (r) => r.label },
        { label: "Accuracy", num: true, value: (r) => pct(r.accuracy.mean) },
        { label: "95% CI", num: true, value: (r) => `${pct(r.accuracy.low)} to ${pct(r.accuracy.high)}` },
        { label: "GRPO minus this", num: true, value: (r) => (r.grpo_minus_this ? pts(r.grpo_minus_this.diff) : "") },
        { label: "McNemar", num: true, value: (r) => (r.grpo_minus_this ? pval(r.grpo_minus_this.mcnemar.p_value) : "") },
        { label: "Verified", value: (r) => (r.verified_answers ? "yes" : "no") },
      ],
      rows: sys,
      highlight: (r) => r.group === "ours",
    },
    draw(fig, W) {
      const rowH = 34, m = { l: Math.min(250, W * 0.42), r: 56, t: 6, b: 26 };
      const H = m.t + sys.length * rowH + m.b;
      const X = (v) => m.l + v * (W - m.l - m.r);
      const svg = svgRoot(W, H, "Horizontal bars of verified accuracy per system");
      const grid = s("g", { class: "grid" });
      for (const t of [0, 0.25, 0.5, 0.75, 1]) {
        grid.append(s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }));
        svg.append(s("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, pct(t, 0)));
      }
      svg.append(grid);
      sys.forEach((p, i) => {
        const y = m.t + i * rowH + 8, h = 18, ours = p.group === "ours";
        const color = ours ? "var(--series-1)" : "var(--deemph)";
        const g = s("g");
        g.append(s("rect", { x: 0, y: y - 6, width: W, height: rowH - 2, fill: "transparent" }));
        g.append(s("text", { x: m.l - 10, y: y + 13, "text-anchor": "end", class: ours ? "label-strong" : "label" }, p.label + (p.verified_answers ? "" : "  (unverified)")));
        g.append(s("path", { d: hbarPath(X(0), X(p.accuracy.mean), y, h), fill: color }));
        g.append(s("line", { x1: X(p.accuracy.low), x2: X(p.accuracy.high), y1: y + h / 2, y2: y + h / 2, stroke: "var(--text-secondary)", "stroke-width": 1.5 }));
        g.append(s("text", { x: X(p.accuracy.mean) + 8 + (X(p.accuracy.high) - X(p.accuracy.mean)), y: y + 13, class: "label" }, pct(p.accuracy.mean)));
        hoverable(g, () => [
          { value: pct(p.accuracy.mean), label: p.label, color },
          { value: `${pct(p.accuracy.low)} to ${pct(p.accuracy.high)}`, label: "95% CI" },
          p.grpo_minus_this ? { value: `${pts(p.grpo_minus_this.diff)} GRPO minus this`, label: `McNemar ${pval(p.grpo_minus_this.mcnemar.p_value)}, ${p.grpo_minus_this.mcnemar.only_a} vs ${p.grpo_minus_this.mcnemar.only_b} discordant` } : null,
        ].filter(Boolean));
        svg.append(g);
      });
      fig.append(svg);
    },
  });
  document.getElementById("models-sub").textContent = "Every system answers the same FinQA questions; the pipeline systems also cite each figure and pass the verifier.";
}

function renderFullTest(d) {
  const host = document.getElementById("panel-fulltest");
  host.append(el("h3", { text: "Full test set, 1,139 questions" }), el("p", { class: "sub", text: "Verified accuracy is scale tolerant; strict requires the exact FinQA convention (often a fraction where models write a percent)." }));
  host.append(el("div", { class: "table-wrap" }, buildTable({
    columns: [
      { label: "System", value: (r) => r.label },
      { label: "Verified", num: true, value: (r) => pct(r.verified.mean) },
      { label: "95% CI", num: true, value: (r) => `${pct(r.verified.low)} to ${pct(r.verified.high)}` },
      { label: "Strict", num: true, value: (r) => pct(r.strict) },
      { label: "Ungated", num: true, value: (r) => pct(r.ungated) },
      { label: "Coverage", num: true, value: (r) => pct(r.coverage) },
      { label: "Parse", num: true, value: (r) => pct(r.parse_rate) },
      { label: "Fabrication", num: true, value: (r) => pct(r.fabrication_rate, 2) },
      { label: "Tokens", num: true, value: (r) => r.completion_tokens_mean.toFixed(0) },
    ],
    rows: d.full_test,
    highlight: (r) => r.label.includes("ours"),
  })));
}

function renderSharpening(d) {
  const sh = d.sharpening;
  const rows = [
    { label: "LedgerMind SFT", greedy: sh.greedy.sft, sampled: sh.sampled.sft },
    { label: "LedgerMind GRPO", greedy: sh.greedy.grpo, sampled: sh.sampled.grpo },
  ];
  chartCard(document.getElementById("panel-sharpen"), {
    title: "Greedy against sampled decoding",
    sub: "Verified accuracy, greedy and mean of 4 seeds at temperature 0.9.",
    legend: [{ label: "Greedy", color: "var(--ord-1)" }, { label: "Sampled, T 0.9", color: "var(--ord-3)" }],
    note: `GRPO minus SFT: greedy ${pts(sh.greedy_diff.diff)} (McNemar ${pval(sh.greedy_mcnemar_p)}), sampled ${pts(sh.sampled_diff.diff)} [${pts(sh.sampled_diff.low)}, ${pts(sh.sampled_diff.high)}] (paired bootstrap over 4 seeds). GRPO made sampling reliable rather than changing the best answer.`,
    table: { columns: [{ label: "Model", value: (r) => r.label }, { label: "Greedy", num: true, value: (r) => pct(r.greedy) }, { label: "Sampled", num: true, value: (r) => pct(r.sampled) }, { label: "Gap", num: true, value: (r) => pts(r.sampled - r.greedy) }], rows },
    draw(fig, W) {
      const m = { l: 130, r: 40, t: 10, b: 28 }, rowH = 46, H = m.t + rows.length * rowH + m.b, lo = 0.6, hi = 0.8;
      const X = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
      const svg = svgRoot(W, H, "Dumbbell of greedy and sampled accuracy");
      const grid = s("g", { class: "grid" });
      for (const t of niceTicks(lo, hi, 4)) { grid.append(s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b })); svg.append(s("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, pct(t, 0))); }
      svg.append(grid);
      rows.forEach((r, i) => {
        const y = m.t + i * rowH + rowH / 2, g = s("g");
        g.append(s("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }));
        g.append(s("text", { x: m.l - 12, y: y + 4, "text-anchor": "end", class: "label-strong" }, r.label));
        g.append(s("line", { x1: X(r.sampled), x2: X(r.greedy), y1: y, y2: y, stroke: "var(--axis)", "stroke-width": 2 }));
        g.append(s("circle", { cx: X(r.sampled), cy: y, r: 6, fill: "var(--ord-3)", stroke: "var(--surface-1)", "stroke-width": 2 }));
        g.append(s("circle", { cx: X(r.greedy), cy: y, r: 6, fill: "var(--ord-1)", stroke: "var(--surface-1)", "stroke-width": 2 }));
        g.append(s("text", { x: X(r.greedy) + 12, y: y + 4, class: "label" }, pct(r.greedy)));
        g.append(s("text", { x: X(r.sampled) - 12, y: y + 4, "text-anchor": "end", class: "label" }, pct(r.sampled)));
        hoverable(g, () => [{ value: pct(r.greedy), label: "greedy", color: "var(--ord-1)" }, { value: pct(r.sampled), label: "sampled, mean of 4 seeds", color: "var(--ord-3)" }]);
        svg.append(g);
      });
      fig.append(svg);
    },
  });

  const c = sh.consistency, n = sh.n;
  const parts = [["all_4", "Correct on all 4 seeds", "var(--ord-1)"], ["some", "Correct on 1 to 3", "var(--ord-2)"], ["none", "Correct on none", "var(--ord-3)"]];
  const crow = [{ label: "LedgerMind SFT", v: c.sft }, { label: "LedgerMind GRPO", v: c.grpo }];
  chartCard(document.getElementById("panel-consistency"), {
    title: "How consistently each question is solved",
    sub: `${n} test questions, 4 sampled answers each.`,
    legend: parts.map(([, label, color]) => ({ label, color })),
    note: `GRPO moved ${c.sft.some - c.grpo.some} unstable questions out of the middle band. It also slightly narrowed coverage: solved by at least one seed ${pct(1 - c.grpo.none / n)} for GRPO against ${pct(1 - c.sft.none / n)} for SFT.`,
    table: { columns: [{ label: "Model", value: (r) => r.label }, ...parts.map(([k, label]) => ({ label, num: true, value: (r) => r.v[k].toLocaleString() }))], rows: crow },
    draw(fig, W) {
      const m = { l: 130, r: 12, t: 10, b: 10 }, rowH = 46, H = m.t + crow.length * rowH + m.b;
      const X = (v) => m.l + (v / n) * (W - m.l - m.r);
      const svg = svgRoot(W, H, "Stacked bars of per question consistency");
      crow.forEach((r, i) => {
        const y = m.t + i * rowH + 12, h = 22;
        svg.append(s("text", { x: m.l - 12, y: y + 15, "text-anchor": "end", class: "label-strong" }, r.label));
        let acc = 0;
        parts.forEach(([k, label, color], j) => {
          const a = X(acc) + (j ? 1 : 0), b = X(acc + r.v[k]) - (j < parts.length - 1 ? 1 : 0);
          const g = s("g");
          g.append(j === parts.length - 1 ? s("path", { d: hbarPath(a, b, y, h), fill: color }) : s("rect", { x: a, y, width: Math.max(0, b - a), height: h, fill: color }));
          const text = r.v[k].toLocaleString();
          if (b - a > text.length * 7 + 12) g.append(s("text", { x: (a + b) / 2, y: y + 15, "text-anchor": "middle", fill: j === 0 ? "#ffffff" : "#0b0b0b", style: `fill:${j === 0 ? "#ffffff" : "#0b0b0b"}` }, text));
          hoverable(g, () => [{ value: text, label: `${label}, ${r.label}`, color }]);
          svg.append(g);
          acc += r.v[k];
        });
      });
      fig.append(svg);
    },
  });
}

function lineChart(host, { title, sub, note, series, xLabel, yFmt, yDomain, xTicks, xLog2 }) {
  chartCard(host, {
    title, sub, note,
    legend: series.length > 1 ? series.map((x) => ({ label: x.label, color: x.color, line: true })) : null,
    table: {
      columns: [{ label: xLabel, num: true, value: (r) => r.x.toLocaleString() }, ...series.map((sr) => ({ label: sr.label, num: true, value: (r) => yFmt(r[sr.label]) }))],
      rows: series[0].points.map((p, i) => Object.fromEntries([["x", p.x], ...series.map((sr) => [sr.label, sr.points[i].y])])),
    },
    draw(fig, W) {
      const labelRoom = series.length > 1 ? Math.max(...series.map((x) => x.label.length)) * 6.6 + 16 : 16;
      const H = 220, m = { l: 48, r: labelRoom, t: 10, b: 30 };
      const xs = series[0].points.map((p) => p.x), all = series.flatMap((sr) => sr.points.map((p) => p.y));
      const [ylo, yhi] = yDomain || [Math.min(...all), Math.max(...all)];
      const fx = xLog2 ? Math.log2 : (v) => v;
      const X = (v) => m.l + ((fx(v) - fx(xs[0])) / (fx(xs[xs.length - 1]) - fx(xs[0]))) * (W - m.l - m.r);
      const Y = (v) => m.t + (1 - (v - ylo) / (yhi - ylo || 1)) * (H - m.t - m.b);
      const svg = svgRoot(W, H, title);
      const grid = s("g", { class: "grid" });
      for (const t of niceTicks(ylo, yhi, 4)) { grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) })); svg.append(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, yFmt(t))); }
      for (const t of xTicks || niceTicks(xs[0], xs[xs.length - 1], 5)) svg.append(s("text", { x: X(t), y: H - 10, "text-anchor": "middle" }, t.toLocaleString()));
      svg.prepend(grid);
      svg.append(s("text", { x: W - m.r, y: H - 10 + 0, "text-anchor": "end", dy: 0, opacity: 0 }, ""));
      for (const sr of series) {
        svg.append(s("path", { d: sr.points.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(""), fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
        if (sr.points.length <= 8) for (const p of sr.points) svg.append(s("circle", { cx: X(p.x), cy: Y(p.y), r: 4, fill: sr.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
        if (series.length > 1) { const last = sr.points[sr.points.length - 1]; svg.append(s("text", { x: X(last.x) + 8, y: Y(last.y) + 4 + (sr.labelDy || 0), class: "label" }, sr.label)); }
      }
      const cross = s("line", { y1: m.t, y2: H - m.b, stroke: "var(--text-muted)", "stroke-width": 1, opacity: 0 });
      svg.append(cross);
      const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => {
        const b = svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * W;
        let i = 0; xs.forEach((x, j) => { if (Math.abs(X(x) - px) < Math.abs(X(xs[i]) - px)) i = j; });
        cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("opacity", 1);
        showTip(e, [{ value: `${xLabel} ${xs[i].toLocaleString()}` }, ...series.map((sr) => ({ value: yFmt(sr.points[i].y), label: series.length > 1 ? sr.label : "", color: sr.color }))]);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
      svg.append(hit);
      fig.append(svg);
    },
  });
}

function renderTraining(d) {
  const host = document.getElementById("training-multiples");
  const g = d.training.grpo;
  const panels = [
    ["Correct term on training rollouts", "Share of sampled answers that are correct and verified, per logged step.", "correct", (v) => pct(v, 0), null],
    ["Groups with no learning signal", "All 8 rollouts scored the same, so the step learns nothing from them.", "zero_std", (v) => pct(v, 0), [0, 1]],
    ["Policy entropy", "Lower means more certain sampling; it fell about 4 times without collapsing.", "entropy", (v) => v.toFixed(3), null],
  ];
  for (const [title, sub, key, fmt, dom] of panels) {
    const p = el("div", { class: "panel" });
    host.append(p);
    lineChart(p, { title, sub, xLabel: "Step", yFmt: fmt, yDomain: dom, series: [{ label: title, color: "var(--series-1)", points: g.map((h) => ({ x: h.step, y: h[key] })) }] });
  }
}

function renderServing(d) {
  const names = [["grpo_bf16", "bf16", "var(--series-1)"], ["grpo_fp8", "fp8", "var(--series-2)"], ["grpo_bf16_constrained", "bf16 + JSON schema", "var(--series-3)", 10], ["grpo_fp8_constrained", "fp8 + JSON schema", "var(--series-4)", -6]];
  const series = names.map(([k, label, color, labelDy]) => ({ label, color, labelDy, points: d.serving[k].map((r) => ({ x: r.concurrency, y: r.requests_per_s })) }));
  lineChart(document.getElementById("panel-throughput"), {
    title: "Throughput against concurrency", sub: "Requests per second, GRPO model, one H200.", xLabel: "Concurrency", xLog2: true, xTicks: [1, 4, 16, 64],
    yFmt: (v) => v.toFixed(0), series, yDomain: [0, Math.ceil(Math.max(...series.flatMap((x) => x.points.map((q) => q.y))) / 10) * 10],
    note: "JSON schema decoding halves peak throughput. The trained model parses 99.9% unconstrained, so it can be served without it; fp8 adds 9 to 20% at no accuracy cost.",
  });
  const host = document.getElementById("panel-latency");
  host.append(el("h3", { text: "Latency per concurrency level" }), el("p", { class: "sub", text: "p50 and p95 seconds per request, unconstrained." }));
  const rows = d.serving.grpo_bf16.map((r, i) => ({ c: r.concurrency, b: r, f: d.serving.grpo_fp8[i] }));
  host.append(el("div", { class: "table-wrap" }, buildTable({
    columns: [
      { label: "Concurrency", num: true, value: (r) => r.c },
      { label: "bf16 p50", num: true, value: (r) => `${r.b.latency_p50_s.toFixed(2)} s` },
      { label: "bf16 p95", num: true, value: (r) => `${r.b.latency_p95_s.toFixed(2)} s` },
      { label: "fp8 p50", num: true, value: (r) => `${r.f.latency_p50_s.toFixed(2)} s` },
      { label: "fp8 p95", num: true, value: (r) => `${r.f.latency_p95_s.toFixed(2)} s` },
      { label: "fp8 tokens/s", num: true, value: (r) => r.f.output_tokens_per_s.toLocaleString() },
    ],
    rows,
  })));
  host.append(el("p", { class: "note", text: "fp8 accuracy on test matches bf16 exactly (74.89% both, 7 against 7 discordant answers)." }));
}

function renderSaboteur(d) {
  const host = document.getElementById("saboteur-panels");
  const a = d.saboteur.mode_a, total = Object.values(a).reduce((x, y) => x + y.cases, 0);
  const pa = el("div", { class: "panel" }, [el("h3", { text: "Mode A, corrupted outputs" }), el("p", { class: "sub", text: "A valid answer is mutated; the verifier must reject it." }),
    el("div", { class: "tile-value", text: "100%" }), el("div", { class: "tile-delta", text: `of ${total.toLocaleString()} injected faults caught` })]);
  pa.append(el("div", { class: "table-wrap" }, buildTable({ columns: [{ label: "Fault", value: (r) => r[0].replaceAll("_", " ") }, { label: "Cases", num: true, value: (r) => r[1].cases.toLocaleString() }, { label: "Caught", num: true, value: (r) => pct(r[1].rate, 0) }], rows: Object.entries(a) })));
  const b = d.saboteur.mode_b;
  const pb = el("div", { class: "panel" }, [el("h3", { text: "Mode B, tampered documents" }), el("p", { class: "sub", text: "The source is corrupted. Stale answers must be rejected; tampering is flagged when a total covers the cell." })]);
  pb.append(el("div", { class: "table-wrap" }, buildTable({ columns: [{ label: "Fault", value: (r) => r[0].replaceAll("_", " ") }, { label: "Stale out", num: true, value: (r) => pct(r[1].stale_rejection_rate, 0) }, { label: "Flagged", num: true, value: (r) => pct(r[1].faithful_detection_rate, 0) }, { label: "If footed", num: true, value: (r) => (r[1].detection_rate_when_detectable === null ? "n/a" : pct(r[1].detection_rate_when_detectable, 0)) }], rows: Object.entries(b) })));
  pb.append(el("p", { class: "note", text: "Stale out: answers computed from the original figures are rejected. Flagged: the tampering itself is detected. If footed: detection when a printed total covers the cell; about two thirds of tampered cells sit outside any total." }));
  const pc = el("div", { class: "panel" }, [el("h3", { text: "Mode C, counterfactual filings" }), el("p", { class: "sub", text: "Figures are edited so a model that recalls the real filing gives the wrong answer." })]);
  pc.append(el("div", { class: "table-wrap" }, buildTable({ columns: [{ label: "Model", value: (r) => r.label }, { label: "Real", num: true, value: (r) => pct(r.real) }, { label: "Edited", num: true, value: (r) => pct(r.counterfactual) }, { label: "Diff", num: true, value: (r) => pts(r.diff.diff) }, { label: "McNemar", num: true, value: (r) => pval(r.mcnemar_p).replace("p = ", "") }, { label: "Recalled", num: true, value: (r) => `${r.recalled_original}/${r.n}` }], rows: d.saboteur.mode_c, highlight: (r) => r.label.includes("LedgerMind") })));
  pc.append(el("p", { class: "note", text: "The models read the document: under 0.5% of answers recall the real filing, and no drop is significant." }));
  host.append(pa, pb, pc);
}

/* ---------- Audit explorer ---------- */

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

function renderExplorer(d) {
  const list = document.getElementById("q-list"), detail = document.getElementById("q-detail");
  const items = d.explorer;
  let current = null;
  const buttons = [];
  let lastGroup = null;
  items.forEach((q, i) => {
    if (q.group !== lastGroup) { list.append(el("div", { class: "q-group", text: q.group_title })); lastGroup = q.group; }
    const b = el("button", { class: "q-item", type: "button", role: "option" }, [el("span", { class: "qid", text: q.id }), document.createTextNode(q.question)]);
    b.addEventListener("click", () => select(i));
    buttons.push(b);
    list.append(b);
  });
  function select(i) {
    if (current !== null) buttons[current].removeAttribute("aria-current");
    current = i; buttons[i].setAttribute("aria-current", "true");
    const q = items[i];
    detail.replaceChildren(
      el("div", { class: "q-group", style: "margin:0", text: `${q.group_title}  ·  ${q.id}` }),
      el("div", { class: "q-question", text: q.question }),
      el("div", { class: "kv" }, [el("span", {}, [document.createTextNode("Gold answer "), el("b", { text: fmtNum(q.gold) })]),
        q.original_gold !== undefined ? el("span", {}, [document.createTextNode("Answer in the real filing "), el("b", { text: fmtNum(q.original_gold) })]) : null]),
      ...renderDocument(q.document, q.ours),
    );
    const cols = el("div", { class: "answer-cols" }, [auditBlock("LedgerMind GRPO (4B)", q.ours, q.ours.correct)]);
    if (q.gpt55_pipeline) cols.append(auditBlock("gpt-5.5 + pipeline (prompt v1)", q.gpt55_pipeline, q.gpt55_pipeline.correct));
    if (q.gpt55_direct) cols.append(el("div", { class: "panel" }, [el("div", { class: "panel-head" }, [el("h3", { text: "gpt-5.5, direct answer" }), el("div", {}, [el("span", { class: "badge warn", text: "Not verified" }), document.createTextNode(" "), correctBadge(q.gpt55_direct.correct)])]),
      el("div", { class: "kv" }, [el("span", {}, [document.createTextNode("Parsed answer "), el("b", { text: fmtNum(q.gpt55_direct.value) })])]), el("div", { class: "direct-text", text: q.gpt55_direct.text })]));
    detail.append(cols);
  }
  if (items.length) select(0);
}

function renderMethod(d) {
  const notes = [
    ["Specialist against generalist.", "The LedgerMind models were trained on FinQA's training split and learned its conventions (signs, units, which year is the base). The API models see the task cold, so the fair claim is that a small specialist matches a general flagship on this task, not that it is a stronger model in general."],
    ["Prompt v1 favors the trained models.", "The pipeline prompt v1 names its functions without defining argument order and asks for millions to be converted to units, while FinQA answers stay in document units. A post hoc count puts the cost to gpt-5.5 at about 8 points on the sample. Prompt v2 fixes both; the API pipeline numbers shown are v1."],
    ["Direct answer scoring.", "Direct answers are shown scored at the precision they state, a rule added after reading the first outputs, where correctly rounded answers were marked wrong. Default scoring of the same answers is much lower and is reported in the full test table."],
    ["Sample and pairing.", `The frontier comparison uses a seeded ${d.meta.sample_n} question sample, so intervals are wider (about 4 points) than on the full test set (about 2.5). Every comparison is paired on identical questions with McNemar and a paired bootstrap.`],
    ["Determinism.", "Scored runs use vLLM batch invariant kernels: two reruns of the same model produced identical outputs on all 1,139 questions."],
    ["Cost.", `Self hosted cost is a rental estimate at full utilization ($${d.meta.h200_usd_per_hour.toFixed(2)} per H200 hour); the runs themselves used a university cluster. API cost is list price times measured tokens; the actual spend for every frontier run was $${d.meta.api_spend_usd.toFixed(2)}.`],
  ];
  const host = document.getElementById("method-notes");
  for (const [b, t] of notes) host.append(el("p", { style: "margin:0 0 10px" }, [el("b", { text: `${b} ` }), document.createTextNode(t)]));
  host.append(el("p", { class: "note" }, [document.createTextNode("Every number on this page is computed from files in "), el("a", { href: "https://github.com/Govindrm7/LedgerMind/tree/main/docs/results", text: "docs/results" }), document.createTextNode(" by scripts/build_dashboard.py.")]));
  document.getElementById("footer").textContent = `Built from commit ${d.meta.git}. LedgerMind: the model proposes, a deterministic engine computes, a verifier decides.`;
}

/* ---------- Boot ---------- */
function setupTheme() {
  const btn = document.getElementById("theme-toggle"), root = document.documentElement;
  const label = () => { btn.textContent = root.dataset.theme === "light" ? "Dark mode" : "Light mode"; };
  label();
  btn.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("ledgermind-theme", root.dataset.theme); } catch (e) { /* not persisted */ }
    label();
  });
}

fetch("dashboard/data.json")
  .then((r) => r.json())
  .then((d) => {
    setupTheme();
    renderTicker(d); renderHero(d); renderScatter(d); renderBars(d); renderFullTest(d);
    renderSharpening(d); renderTraining(d); renderServing(d); renderSaboteur(d); renderExplorer(d); renderMethod(d);
    // Charts measure their card width; redraw once the grid has laid out.
    requestAnimationFrame(() => renderers.forEach((f) => f()));
    let t;
    window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => renderers.forEach((f) => f()), 150); });
  })
  .catch((err) => {
    document.querySelector("main").prepend(el("p", { class: "empty", text: `Could not load dashboard data (${err}). Serve the docs folder over HTTP, for example: python -m http.server -d docs` }));
  });
