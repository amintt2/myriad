"use strict";
// Agents view: a task split into sub-agents, run on the network and shown live as a tree (columns =
// dependency depth), a timeline and a summary. Talks only to its own origin; POSTs carry the page token.
(function () {
  const I = window.I18N;
  I.extend("fr", {
    "nav.agents": "Agents",
    "ag.title": "Sous-agents",
    "ag.lead": "Une tâche découpée en sous-tâches confiées en parallèle à des pairs différents, chacune avec son petit contexte, puis vérifiées et assemblées.",
    "ag.demo": "Plan de démonstration", "ag.task": "Tâche", "ag.task_ph": "Décrivez la tâche à découper en sous-agents…",
    "ag.mode_auto": "plan automatique", "ag.mode_plan": "plan JSON", "ag.merge": "Fusion finale", "ag.run": "Lancer",
    "ag.plan_hint": "Liste de sous-tâches : id, prompt, role, skill (ou family, model), depends_on, k, verify (commande autorisée), context.",
    "ag.tree": "L'arbre des sous-agents", "ag.summary": "Résumé", "ag.timeline": "Chronologie", "ag.result": "Résultat",
    "ag.st.pending": "en attente", "ag.st.running": "en cours", "ag.st.verifying": "vérification", "ag.st.escalating": "escalade",
    "ag.st.ok": "terminé", "ag.st.failed": "échec", "ag.st.skipped": "sauté", "ag.st.cancelled": "annulé",
    "ag.orchestrator": "orchestrateur", "ag.merge_node": "fusion", "ag.waiting_peer": "en attente d'un pair…",
    "ag.any_peer": "n'importe quel pair", "ag.swarm": "essaim", "ag.fallback": "repli",
    "ag.verify_ok": "{c} réussie", "ag.verify_ko": "{c} échouée", "ag.escalated": "escaladé",
    "ag.cands": "{n} candidats", "ag.tokens": "{n} jetons", "ag.output": "Voir la sortie", "ag.candidates": "Candidats",
    "ag.wall": "Temps réel", "ag.sum": "Somme des sous-agents", "ag.speedup": "Parallélisme", "ag.tok": "Jetons",
    "ag.done_n": "Sous-tâches réussies", "ag.status.ok": "réussi", "ag.status.partial": "partiel",
    "ag.status.failed": "échec", "ag.status.timeout": "délai dépassé", "ag.status.running": "en cours",
    "ag.planning": "L'orchestrateur écrit le plan…", "ag.plan_auto": "plan automatique", "ag.plan_explicit": "plan donné",
    "ag.bad_json": "Plan JSON invalide : {m}", "ag.need_task": "Décrivez d'abord la tâche (ou chargez le plan de démonstration).",
    "ag.result_wait": "Les résultats s'assembleront ici.", "ag.merged": "fusionné", "ag.concat": "concaténé",
    "ag.deps": "après {d}", "ag.interrupted": "le flux s'est interrompu avant la fin de l'exécution",
    "ag.e1": "Découper", "ag.e1d": "L'orchestrateur écrit un plan de sous-tâches, ou vous le donnez en JSON.",
    "ag.e2": "Confier", "ag.e2d": "Chaque sous-tâche part vers un pair différent, avec son petit contexte.",
    "ag.e3": "Vérifier", "ag.e3d": "Une commande autorisée contrôle la sortie\u00a0; un échec fait escalader.",
    "ag.e4": "Assembler", "ag.e4d": "Les résultats sont mis bout à bout, ou fusionnés par un dernier pair.",
    "ag.empty_hint": "Pour voir l'arbre s'exécuter, lancez le plan de démonstration.",
  });
  I.extend("en", {
    "nav.agents": "Agents",
    "ag.title": "Sub-agents",
    "ag.lead": "One task split into sub-tasks handed in parallel to different peers, each with its own small context, then verified and assembled.",
    "ag.demo": "Demo plan", "ag.task": "Task", "ag.task_ph": "Describe the task to split into sub-agents…",
    "ag.mode_auto": "automatic plan", "ag.mode_plan": "JSON plan", "ag.merge": "Final merge", "ag.run": "Run",
    "ag.plan_hint": "A list of sub-tasks: id, prompt, role, skill (or family, model), depends_on, k, verify (an allowed command), context.",
    "ag.tree": "The sub-agent tree", "ag.summary": "Summary", "ag.timeline": "Timeline", "ag.result": "Result",
    "ag.st.pending": "waiting", "ag.st.running": "running", "ag.st.verifying": "verifying", "ag.st.escalating": "escalating",
    "ag.st.ok": "done", "ag.st.failed": "failed", "ag.st.skipped": "skipped", "ag.st.cancelled": "cancelled",
    "ag.orchestrator": "orchestrator", "ag.merge_node": "merge", "ag.waiting_peer": "waiting for a peer…",
    "ag.any_peer": "any peer", "ag.swarm": "swarm", "ag.fallback": "fallback",
    "ag.verify_ok": "{c} passed", "ag.verify_ko": "{c} failed", "ag.escalated": "escalated",
    "ag.cands": "{n} candidates", "ag.tokens": "{n} tokens", "ag.output": "Show output", "ag.candidates": "Candidates",
    "ag.wall": "Wall time", "ag.sum": "Sum of sub-agents", "ag.speedup": "Parallelism", "ag.tok": "Tokens",
    "ag.done_n": "Sub-tasks done", "ag.status.ok": "done", "ag.status.partial": "partial",
    "ag.status.failed": "failed", "ag.status.timeout": "deadline exceeded", "ag.status.running": "running",
    "ag.planning": "The orchestrator is writing the plan…", "ag.plan_auto": "automatic plan", "ag.plan_explicit": "given plan",
    "ag.bad_json": "Invalid JSON plan: {m}", "ag.need_task": "Describe the task first (or load the demo plan).",
    "ag.result_wait": "The results will be assembled here.", "ag.merged": "merged", "ag.concat": "concatenated",
    "ag.deps": "after {d}", "ag.interrupted": "the stream stopped before the run ended",
    "ag.e1": "Split", "ag.e1d": "The orchestrator writes a plan of sub-tasks, or you give it as JSON.",
    "ag.e2": "Delegate", "ag.e2d": "Each sub-task goes to a different peer, with its own small context.",
    "ag.e3": "Verify", "ag.e3d": "An allowed command checks the output; a failure escalates.",
    "ag.e4": "Assemble", "ag.e4d": "The results are put end to end, or merged by one last peer.",
    "ag.empty_hint": "To watch the tree run, start the demo plan.",
  });
  const t = I.t;
  const $ = (id) => document.getElementById(id);
  const TOKEN = document.querySelector('meta[name="myriad-token"]').content;
  const fam = (f) => (window.NetViz && window.NetViz.familyColor ? window.NetViz.familyColor(f) : "var(--accent)");
  const loc = () => I.locale();
  const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toLocaleString(loc(), { maximumFractionDigits: d }));
  const secs = (ms) => (ms === null || ms === undefined ? "–" : `${fmt(ms / 1000, ms < 10000 ? 1 : 0)} s`);
  const short = (m) => (m || "?").split("/").pop().replace(/-GGUF$/i, "");
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "style") for (const [p, val] of Object.entries(v)) e.style.setProperty(p, val);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids) if (c !== null && c !== undefined && c !== false) e.append(c);
    return e;
  }

  // ------------------------------------------------------------ state
  const R = { busy: false, mode: "auto", subs: new Map(), order: [], planner: null, merge: null, result: null,
              t0: 0, budget: null, source: null, combine: null, error: null, open: new Set(), timer: null };
  function sub(id) {
    if (id === "plan") return R.planner || (R.planner = { id: "plan", status: "running", jobs: [] });
    if (id === "merge") return R.merge || (R.merge = { id: "merge", status: "pending", jobs: [] });
    if (!R.subs.has(id)) { R.subs.set(id, { id, status: "pending", jobs: [], verifs: [] }); R.order.push(id); }
    return R.subs.get(id);
  }
  function absorb(s, d) {  // a server snapshot of a sub-task (keeps live-only fields)
    if (!d) return s;
    Object.assign(s, d, { jobs: s.jobs || [], verifs: s.verifs || [] });
    return s;
  }
  const elapsed = () => (R.result ? R.result.timing.wall_ms : performance.now() - R.t0);

  function handle(ev) {
    switch (ev.type) {
      case "start": break;
      case "planning": absorb(sub("plan"), ev.planner); break;
      case "plan":
        R.source = ev.source; R.combine = ev.combine; R.budget = ev.budget;
        if (ev.planner) absorb(sub("plan"), ev.planner);
        for (const d of ev.subtasks) absorb(sub(d.id), d);
        if (ev.combine === "merge") sub("merge");
        break;
      case "subtask": absorb(sub(ev.subtask.id), ev.subtask); break;
      case "job": {
        const s = sub(ev.subtask);
        s.jobs.push({ job_id: ev.job_id, node_id: ev.node_id, model: ev.model, family: ev.family, attempt: ev.attempt,
                      tag_match: ev.tag_match, state: "asked" });
        if (s.status === "pending") s.status = "running";
        break;
      }
      case "answered": case "job_failed": {
        const s = sub(ev.subtask), j = s.jobs.find((x) => x.job_id === ev.job_id);
        if (j) Object.assign(j, { state: ev.type === "answered" ? "ok" : "failed", tokens: ev.tokens, error: ev.error });
        break;
      }
      case "verifying": sub(ev.subtask).status = "verifying"; break;
      case "verified": sub(ev.subtask).verifs.push(ev); break;
      case "escalate": Object.assign(sub(ev.subtask), { status: "escalating", escalated: true }); break;
      case "done": {
        const r = ev.result; R.result = r;
        if (r.planner) absorb(sub("plan"), r.planner);
        for (const d of r.subtasks) absorb(sub(d.id), d);
        if (r.merge) absorb(sub("merge"), r.merge);
        else if (R.merge && R.merge.status === "pending") R.merge.status = "skipped";
        R.error = r.error;
        break;
      }
      case "error": R.error = ev.message; break;
      default: break;
    }
    schedule();
  }
  let raf = 0;
  function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); }); }

  // ------------------------------------------------------------ rendering
  function statusTag(st) {
    const cls = { ok: "ok", failed: "bad", skipped: "", cancelled: "warn", running: "info", verifying: "info", escalating: "warn" }[st] ?? "";
    return el("span", { class: `tag ${cls}`, text: t(`ag.st.${st}`) });
  }
  function routeChips(s) {
    const chips = [];
    if (s.id === "plan") chips.push(el("span", { class: "chip", text: s.skill ? `#${s.skill}` : t("ag.orchestrator") }));
    else if (s.skill) chips.push(el("span", { class: "chip", text: `#${s.skill}` }));
    else if (s.family) chips.push(el("span", { class: "chip" }, el("i", { style: { background: fam(s.family) } }), s.family));
    else if (s.model) chips.push(el("span", { class: "chip", text: short(s.model) }));
    else if (s.id !== "merge") chips.push(el("span", { class: "chip", text: t("ag.any_peer") }));
    if ((s.k || 1) > 1) chips.push(el("span", { class: "chip", text: `k=${s.k}` }));
    if (s.verify_command) chips.push(el("span", { class: "chip mono", text: `✓ ${s.verify_command}` }));
    return chips;
  }
  function livePeer(s) {
    if (s.peer && s.peer.node_id && ["ok", "failed"].includes(s.status)) return s.peer;
    const js = (s.jobs || []).filter((j) => j.state === "asked");
    return js.length ? js[js.length - 1] : (s.peer && s.peer.node_id ? s.peer : null);
  }
  function card(s, kind) {
    const peer = livePeer(s);
    const live = ["running", "verifying", "escalating"].includes(s.status);
    const name = kind === "plan" ? t("ag.orchestrator") : kind === "merge" ? t("ag.merge_node") : s.id;
    const c = el("article", { class: `ag-card st-${s.status}${kind !== "sub" ? " ag-special" : ""}`, "data-id": s.id,
                               style: { "--fam": peer ? fam(peer.family) : "var(--line-2)" }, tabindex: "0" });
    c.append(el("div", { class: "ag-card-top" }, el("span", { class: "ag-id", text: name }), statusTag(s.status)));
    if (s.role && kind === "sub") c.append(el("div", { class: "ag-role", text: s.role }));
    const chips = routeChips(s);
    if (chips.length) c.append(el("div", { class: "ag-chips" }, ...chips));
    const pl = el("div", { class: "ag-peer" });
    if (peer) {
      pl.append(el("i", { style: { background: fam(peer.family) } }), el("b", { text: short(peer.model) }),
        el("span", { class: "hint", text: ` ${peer.family || "?"} · ${(peer.node_id || "").slice(0, 6)}` }));
      if (peer.tag_match === false) pl.append(el("span", { class: "tag warn", text: t("ag.fallback") }));
      if (live && s.status !== "verifying") pl.append(el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")));
    } else if (live) {
      pl.append(el("span", { class: "hint", text: t("ag.waiting_peer") }), el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")));
    } else if (s.depends_on && s.depends_on.length && s.status === "pending") {
      pl.append(el("span", { class: "hint", text: t("ag.deps", { d: s.depends_on.join(", ") }) }));
    }
    c.append(pl);
    const flags = el("div", { class: "ag-flags" });
    const verifs = s.verifs || [];
    const vres = s.verify || (verifs.length ? verifs[verifs.length - 1] : null);
    const cmd = s.verify_command || (verifs[0] && verifs[0].command) || "";
    if (s.status === "verifying") flags.append(el("span", { class: "tag info", text: `⧗ ${cmd}` }));
    else if (vres) flags.append(el("span", { class: `tag ${vres.passed ? "ok" : "bad"}`, text: `${vres.passed ? "✓" : "✗"} ${t(vres.passed ? "ag.verify_ok" : "ag.verify_ko", { c: cmd })}` }));
    if (s.escalated) flags.append(el("span", { class: "tag warn", text: `↑ ${t("ag.escalated")}` }));
    const ms = s.ms ?? (s.started_ms !== undefined && s.started_ms !== null && live ? elapsed() - s.started_ms : null);
    const meta = [];
    if (s.tokens) meta.push(t("ag.tokens", { n: fmt(s.tokens, 0) }));
    if (ms) meta.push(secs(ms));
    if (s.candidates && s.candidates.length > 1) meta.push(t("ag.cands", { n: s.candidates.length }));
    if (meta.length) flags.append(el("span", { class: "ag-meta", text: meta.join(" · ") }));
    if (flags.childNodes.length) c.append(flags);
    if (live) c.append(el("div", { class: "ag-prog" }, el("span")));
    if (s.error && !["ok"].includes(s.status)) c.append(el("div", { class: "ag-err", text: s.error }));
    if ((s.text && kind !== "plan") || (s.candidates && s.candidates.length)) {
      const det = el("details", { class: "ag-det" });
      det.open = R.open.has(s.id);
      det.addEventListener("toggle", () => { if (det.open) R.open.add(s.id); else R.open.delete(s.id); });
      det.append(el("summary", { text: t("ag.output") }));
      if (s.text) det.append(el("pre", { text: s.text }));
      if (s.candidates && s.candidates.length > 1 || (s.candidates || []).some((x) => x.verify)) {
        const ul = el("ul", { class: "ag-cands" });
        for (const x of s.candidates) {
          ul.append(el("li", { class: x.chosen ? "chosen" : "" }, el("i", { style: { background: fam(x.family) } }),
            el("span", { text: `${short(x.model)} · ${x.family || "?"}` }),
            x.verify ? el("span", { class: `tag ${x.verify.passed ? "ok" : "bad"}`, text: x.verify.passed ? "✓" : `✗ ${x.verify.exit_code ?? ""}` }) : null,
            x.chosen ? el("span", { class: "tag ok", text: "★" }) : null));
        }
        det.append(el("div", { class: "ag-sub", text: t("ag.candidates") }), ul);
        const bad = (s.candidates || []).find((x) => x.verify && !x.verify.passed && x.verify.output);
        if (bad) det.append(el("pre", { class: "ag-vout", text: bad.verify.output.slice(-600) }));
      }
      c.append(det);
    }
    return c;
  }
  function columns() {
    const cols = new Map();
    for (const id of R.order) {
      const s = R.subs.get(id), lv = s.level ?? 0;
      if (!cols.has(lv)) cols.set(lv, []);
      cols.get(lv).push(s);
    }
    return [...cols.entries()].sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  }
  function renderGraph() {
    const box = $("ag-cols"); box.replaceChildren();
    if (R.planner) box.append(el("div", { class: "ag-col" }, card(R.planner, "plan")));
    else if (!R.order.length) return;
    for (const col of columns()) box.append(el("div", { class: "ag-col" }, ...col.map((s) => card(s, "sub"))));
    if (R.merge) box.append(el("div", { class: "ag-col" }, card(R.merge, "merge")));
    requestAnimationFrame(drawEdges);
  }
  function drawEdges() {
    const svg = $("ag-edges"), g = $("ag-graph");
    const gb = g.getBoundingClientRect();
    const cols = $("ag-cols");  // the drawing covers the cards only (the svg itself must not widen the box)
    svg.setAttribute("width", cols.offsetWidth); svg.setAttribute("height", cols.offsetHeight);
    svg.replaceChildren();
    const pos = (id) => {
      const n = g.querySelector(`.ag-card[data-id="${CSS.escape(id)}"]`);
      if (!n) return null;
      const b = n.getBoundingClientRect();
      return { l: b.left - gb.left + g.scrollLeft, r: b.right - gb.left + g.scrollLeft, y: b.top - gb.top + g.scrollTop + Math.min(30, b.height / 2) };
    };
    const ns = "http://www.w3.org/2000/svg";
    const edge = (from, to, st) => {
      const a = pos(from), b = pos(to); if (!a || !b) return;
      const dx = Math.max(24, (b.l - a.r) / 2);
      const p = document.createElementNS(ns, "path");
      p.setAttribute("d", `M${a.r},${a.y} C${a.r + dx},${a.y} ${b.l - dx},${b.y} ${b.l},${b.y}`);
      p.setAttribute("class", `ag-edge e-${st}`);
      svg.append(p);
    };
    for (const id of R.order) {
      const s = R.subs.get(id);
      if (R.planner && !(s.depends_on || []).length) edge("plan", id, s.status);
      for (const d of s.depends_on || []) edge(d, id, s.status);
    }
    if (R.merge) {
      // drawn from the leaves only (the merge reads every result, but the tree stays readable)
      const leaves = R.order.filter((id) => !R.order.some((o) => (R.subs.get(o).depends_on || []).includes(id)));
      for (const d of leaves) edge(d, "merge", R.merge.status);
    }
  }
  function renderStats() {
    const box = $("ag-stats"); box.replaceChildren();
    const r = R.result, subs = [...R.subs.values()];
    const wall = r ? r.timing.wall_ms : R.busy ? elapsed() : null;
    const sum = r ? r.timing.subtasks_sum_ms : subs.reduce((a, s) => a + (s.ms ?? (s.started_ms !== null && s.started_ms !== undefined && R.busy && ["running", "verifying", "escalating"].includes(s.status) ? elapsed() - s.started_ms : 0)), 0);
    const tokens = r ? r.usage.completion_tokens : subs.reduce((a, s) => a + (s.tokens || 0), 0) + ((R.planner && R.planner.tokens) || 0);
    const ok = subs.filter((s) => s.status === "ok").length;
    const tile = (k, v, sub, hl) => el("div", { class: `ag-stat${hl ? " hl" : ""}` }, el("div", { class: "stat-k", text: k }), el("div", { class: "ag-v", text: v }), sub ? el("div", { class: "stat-s", text: sub }) : null);
    const speed = wall && sum ? sum / wall : null;
    box.append(tile(t("ag.wall"), secs(wall), r ? t(`ag.status.${r.status}`) : t("ag.status.running")),
      tile(t("ag.sum"), secs(sum), ""),
      tile(t("ag.speedup"), speed ? `×${fmt(speed, 1)}` : "–", "", true),
      tile(t("ag.tok"), fmt(tokens, 0), R.budget ? `/ ${fmt(R.budget.max_tokens, 0)}` : ""),
      tile(t("ag.done_n"), `${ok}/${subs.length || "–"}`, R.source ? t(`ag.plan_${R.source}`) : ""));
  }
  function renderGantt() {
    const box = $("ag-gantt"); box.replaceChildren();
    const wall = Math.max(1, elapsed());
    const rows = [...(R.planner ? [R.planner] : []), ...R.order.map((id) => R.subs.get(id)), ...(R.merge ? [R.merge] : [])];
    for (const s of rows) {
      const st = s.started_ms, en = s.ended_ms ?? (st !== null && st !== undefined ? elapsed() : null);
      const bar = el("div", { class: "ag-track" });
      if (st !== null && st !== undefined) {
        const peer = livePeer(s);
        bar.append(el("span", { class: `ag-bar st-${s.status}`, style: { left: `${(st / wall) * 100}%`, width: `${Math.max(0.6, ((en - st) / wall) * 100)}%`, "--fam": peer ? fam(peer.family) : "var(--accent)" } }));
      }
      const label = s.id === "plan" ? t("ag.orchestrator") : s.id === "merge" ? t("ag.merge_node") : s.id;
      box.append(el("div", { class: "ag-row" }, el("span", { class: "ag-lbl", text: label }), bar));
    }
  }
  function renderBadges() {
    const b = $("ag-badges"); b.replaceChildren();
    if (R.source) b.append(el("span", { class: "tag info", text: t(`ag.plan_${R.source}`) }));
    if (R.result) {
      const st = R.result.status;
      b.append(el("span", { class: `tag ${st === "ok" ? "ok" : st === "partial" ? "warn" : "bad"}`, text: t(`ag.status.${st}`) }));
      b.append(el("span", { class: "tag", text: t(R.result.merged ? "ag.merged" : "ag.concat") }));
    }
  }
  function renderResult() {
    const box = $("ag-result"), err = $("ag-error");
    err.hidden = !R.error; err.textContent = R.error ? t("err.prefix", { m: R.error }) : "";
    if (R.result) { box.classList.remove("wait"); box.textContent = R.result.result || ""; }
    else if (R.planner && R.planner.status === "running" && !R.order.length) { box.classList.add("wait"); box.textContent = t("ag.planning"); }
    else { box.classList.add("wait"); box.textContent = t("ag.result_wait"); }
  }
  function render() {
    if ($("ag-out").hidden) return;
    renderGraph(); renderStats(); renderGantt(); renderBadges(); renderResult();
  }

  // ------------------------------------------------------------ actions
  function setMode(m, showPlan = true) {
    R.mode = m;
    document.querySelectorAll("#ag-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === m)));
    $("ag-plan-box").hidden = m !== "plan" || !showPlan;
  }
  document.querySelectorAll("#ag-mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  function body() {
    const task = $("ag-task").value.trim();
    const merge = $("ag-merge").checked;
    if (R.mode === "auto") {
      if (!task) throw new Error(t("ag.need_task"));
      return { task, plan: "auto", combine: merge ? "merge" : "concat" };  // the user's choice wins over the planner's
    }
    let p;
    try { p = JSON.parse($("ag-plan").value); } catch (e) { throw new Error(t("ag.bad_json", { m: e.message })); }
    const b = Array.isArray(p) ? { plan: p } : { ...p };
    if (task) b.task = task;
    b.combine = merge ? "merge" : "concat";
    return b;
  }
  async function run(b) {
    if (R.busy) return;
    Object.assign(R, { busy: true, subs: new Map(), order: [], planner: null, merge: null, result: null, error: null,
                       t0: performance.now(), budget: b.budget || null, source: b.plan === "auto" ? "auto" : "explicit", combine: null });
    R.open.clear();
    $("ag-out").hidden = false; $("ag-run").disabled = true; $("ag-demo").disabled = true;
    if (b.plan === "auto") sub("plan");
    render();
    clearInterval(R.timer); R.timer = setInterval(() => { if (R.busy) { renderStats(); renderGantt(); } }, 250);
    try {
      const r = await fetch("/api/agents/run", { method: "POST", body: JSON.stringify(b),
        headers: { "Content-Type": "application/json", "X-Myriad-Token": TOKEN } });
      if (!r.ok || !r.body) { const j = await r.json().catch(() => ({})); throw new Error(j.error || `HTTP ${r.status}`); }
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
          for (const line of chunk.split("\n")) {
            if (line.startsWith("data: {")) { try { handle(JSON.parse(line.slice(6))); } catch (e) { /* ignore a bad event */ } }
          }
        }
      }
      if (!R.result && !R.error) R.error = t("ag.interrupted");  // the stream ended without its result
    } catch (e) {
      R.error = e.message;
    } finally {
      if (!R.result) {  // nothing runs any more: no agent may look active
        for (const s of [R.planner, R.merge, ...R.subs.values()]) {
          if (s && ["running", "verifying", "escalating", "pending"].includes(s.status)) s.status = s.status === "pending" ? "skipped" : "cancelled";
        }
      }
      R.busy = false; clearInterval(R.timer); R.timer = null;
      $("ag-run").disabled = false; $("ag-demo").disabled = false;
      render();
    }
  }
  $("ag-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    let b;
    try { b = body(); } catch (e) { R.error = e.message; $("ag-out").hidden = false; renderResult(); return; }
    run(b);
  });
  for (const id of ["ag-task", "ag-plan"]) {
    $(id).addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); $("ag-form").requestSubmit(); }
    });
  }
  $("ag-demo").addEventListener("click", async () => {
    try {
      const r = await fetch(`/api/agents/demo?lang=${I.lang}`);
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      $("ag-task").value = d.task;
      const { task, ...rest } = d;
      $("ag-plan").value = JSON.stringify(rest, null, 2);
      $("ag-merge").checked = d.combine === "merge";
      setMode("plan", false);  // the plan JSON stays one click away ("plan JSON")
      run(d);
    } catch (e) { R.error = e.message; $("ag-out").hidden = false; renderResult(); }
  });
  window.addEventListener("resize", () => { if (!$("ag-out").hidden) requestAnimationFrame(drawEdges); });
  document.addEventListener("langchange", () => { render(); });
  // the view may become visible after a run started in it: redraw the edges at the right size
  // The edges are measured on the laid-out cards: redraw whenever the graph's box changes, e.g. when
  // the view becomes visible again after a run that finished while another view was shown.
  if (window.ResizeObserver) new ResizeObserver(() => { if (!$("ag-out").hidden) drawEdges(); }).observe($("ag-graph"));
  document.getElementById("tab-agents").addEventListener("click", () => schedule());
  window.MyriadAgents = { run, state: R };
})();
