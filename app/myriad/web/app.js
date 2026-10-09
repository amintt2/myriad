"use strict";
// Myriad local UI. Talks only to its own origin (127.0.0.1); mutating calls carry the page token.
(function () {
  const TOKEN = document.querySelector('meta[name="myriad-token"]').content;
  const { t } = window.I18N;
  const $ = (id) => document.getElementById(id);
  const famColor = window.NetViz.familyColor;
  // a key sent by the network (model, family): own properties only ("constructor" is just a name)
  const own = (o, k) => (o && Object.prototype.hasOwnProperty.call(o, k) ? o[k] : undefined);
  const S = { status: null, network: null, setup: null, wizardOpen: false, wizardDismissed: false, seenJobs: new Map(),
              limitsDirty: false, prevCounters: {}, chatBusy: false, hint: "" };

  // ---------------------------------------------------------------- helpers
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
    for (const c of kids) if (c !== null && c !== undefined) e.append(c);
    return e;
  }
  async function api(path, body) {
    const opts = body === undefined ? {} : { method: "POST", body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", "X-Myriad-Token": TOKEN } };
    const r = await fetch(path, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { const err = new Error((j.error && (j.error.message || j.error)) || `HTTP ${r.status}`); err.status = r.status; throw err; }
    return j;
  }
  const loc = () => window.I18N.locale();
  const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toLocaleString(loc(), { maximumFractionDigits: d }));
  function fmtBytes(b) {
    if (!b && b !== 0) return "–";
    const u = ["o", "Ko", "Mo", "Go", "To"], ue = ["B", "KB", "MB", "GB", "TB"];
    let i = 0; while (b >= 1000 && i < 4) { b /= 1000; i++; }
    return `${fmt(b, i >= 3 ? 2 : 1)} ${(window.I18N.lang === "fr" ? u : ue)[i]}`;
  }
  function fmtDur(s) {
    if (s === null || s === undefined) return "";
    if (s < 60) return `${Math.round(s)} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")}`;
    return `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")}`;
  }
  function toast(msg) {
    const tEl = $("toast"); tEl.textContent = msg; tEl.hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => { tEl.hidden = true; }, 3500);
  }
  function countTo(node, value, digits) {
    const prev = S.prevCounters[node.id];
    S.prevCounters[node.id] = value;
    if (value === null || value === undefined) { node.textContent = "–"; return; }
    if (prev === undefined || prev === null || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      node.textContent = fmt(value, digits); return;
    }
    if (prev === value) return;
    const t0 = performance.now(), dur = 700;
    const tick = (now) => {
      const u = Math.min(1, (now - t0) / dur), e = 1 - (1 - u) ** 3;
      node.textContent = fmt(prev + (value - prev) * e, digits);
      if (u < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    const card = node.closest(".fig"); if (card) { card.classList.remove("bump"); void card.offsetWidth; card.classList.add("bump"); }
  }

  // ---------------------------------------------------------------- theme & language
  function applyTheme(th) {
    document.documentElement.dataset.theme = th;
    try { localStorage.setItem("myriad.theme", th); } catch (e) { /* ignore */ }
    if (viz) { viz.readTheme(); renderAll(); }  // family colours have a light and a dark variant
  }
  let theme = "dark";
  try { theme = localStorage.getItem("myriad.theme") || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"); } catch (e) { /* ignore */ }
  document.documentElement.dataset.theme = theme;
  $("theme-btn").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
    window.I18N.setLang(b.dataset.lang);
    api("/api/prefs", { lang: b.dataset.lang }).catch(() => {});
  }));
  document.addEventListener("langchange", () => { renderAll(); if (S.wizardOpen) renderWizard(); renderExamples(); });

  // ---------------------------------------------------------------- navigation
  const tabs = [...document.querySelectorAll(".nav-item")];
  function show(view, focus) {
    for (const b of tabs) {
      const on = b.dataset.view === view;
      b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1;
      $(`view-${b.dataset.view}`).hidden = !on;
    }
    if (focus) $("main").focus();
    if (location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  }
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => show(b.dataset.view));
    b.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        const n = tabs[(i + (ev.key === "ArrowDown" ? 1 : tabs.length - 1)) % tabs.length];
        n.focus(); show(n.dataset.view);
      }
    });
  });
  const viewFromHash = () => { const v = (location.hash || "").slice(1).replace(/[^a-z]/g, ""); return tabs.some((b) => b.dataset.view === v) ? v : "dashboard"; };
  show(viewFromHash());
  window.addEventListener("hashchange", () => show(viewFromHash()));

  // ---------------------------------------------------------------- network map
  const tip = $("viz-tip");
  const viz = window.NetViz.create($("netviz"), {
    youLabel: () => t("viz.you"),
    label: (n) => t("viz.label", { n }),
    onHover(p, x, y) {
      if (!p) { tip.hidden = true; return; }
      tip.replaceChildren();
      if (p === "me") {
        const n = S.status && S.status.node;
        tip.append(el("b", { text: t("viz.you") }), el("div", { text: (n && n.model) || t("node.client_only") }));
      } else {
        const rel = S.network && own(S.network.reliability, p.model);
        tip.append(el("b", { text: p.model }),
          el("div", { text: `${p.family || "?"} · ${p.node_id.slice(0, 8)}` }),
          el("div", { text: `${t("peers.load")} ${p.busy}/${p.max_parallel} · ${p.accepting ? t("peers.available") : t("peers.paused")}` }),
          rel ? el("div", { text: `${t("peers.rel")} ${fmt(rel.p * 100, 1)} %` }) : null);
      }
      tip.hidden = false;
      const card = $("netviz").getBoundingClientRect();
      tip.style.left = `${Math.min(x + 14, card.width - 200)}px`;
      tip.style.top = `${Math.min(y + 14, card.height - 90)}px`;
    },
  });

  // ---------------------------------------------------------------- dashboard
  function engineText(n) {
    if (!n.engine) return "–";
    return `${n.engine.engine} · ${t(`engine.${n.engine.state}`)}`;
  }
  function renderConn() {
    const s = S.status, c = $("conn"), txt = $("conn-text");
    if (!s) { c.className = "conn bad"; txt.textContent = t("conn.ui_down"); return; }
    if (!s.node) {
      const starting = s.runtime && s.runtime.state === "démarrage";
      c.className = "conn warn"; txt.textContent = starting ? t("conn.starting") : t("conn.setup"); return;
    }
    const st = s.node.state;
    c.className = "conn " + (st === "connecté" ? "ok" : st === "connexion" ? "warn" : "bad");
    txt.textContent = st === "connecté" ? t("conn.connected") : st === "connexion" ? t("conn.connecting") : t("conn.offline");
  }
  function renderNode() {
    const s = S.status; if (!s) return;
    const n = s.node;
    const badge = $("node-badge"), stEl = $("node-state");
    if (!n) {
      badge.className = "node-badge warn"; stEl.textContent = t("conn.setup");
      $("node-model").textContent = "…";
      const rerr = s.runtime && s.runtime.error, e = $("node-error");
      e.hidden = !rerr; e.textContent = rerr || "";
      return;
    }
    const paused = !n.accepting;
    const ok = n.state === "connecté";
    badge.className = "node-badge " + (!ok ? "bad" : paused || !n.serving ? "warn" : "ok");
    stEl.textContent = !ok ? t(`state.${n.state}`) : paused ? t("state.paused") : n.serving ? t("state.serving") : t("state.connecté");
    const m = $("node-model"); m.replaceChildren();
    if (n.model) m.append(el("i", { style: { background: famColor(n.family) } }), n.model.split("/").pop().replace(/-GGUF$/i, ""));
    else m.append(t("node.client_only"));
    $("node-sub").textContent = n.model ? t("node.params", { p: fmt(n.params_b), f: n.family || "?" }) : t("node.no_model");
    const hw = (S.setup && S.setup.hardware) || S.hardware;
    $("node-gpu").textContent = hw ? (hw.best_gpu ? `${hw.best_gpu.name}${hw.best_gpu.vram_gb ? ` · ${fmt(hw.best_gpu.vram_gb)}\u00a0${t("unit.gb")}` : ""}` : hw.cpu) : "–";
    $("node-engine").textContent = engineText(n);
    $("node-jobs").textContent = t("node.jobs_v", { r: n.running, m: n.max_parallel });
    $("node-served").textContent = t("node.served_v", { s: fmt(n.stats.served, 0), f: fmt(n.stats.failed, 0), t: fmt(n.stats.tokens, 0) });
    $("node-id").textContent = n.node_id;
    $("node-api").textContent = s.gateway || "–";
    const err = $("node-error"); err.hidden = !n.last_error; err.textContent = n.last_error || "";
    const pb = $("pause-btn");
    pb.setAttribute("aria-pressed", String(paused));
    $("pause-label").textContent = paused ? t("resume") : t("pause");
    pb.hidden = !n.serving;
    if (!S.limitsDirty) {
      $("accepting").checked = n.accepting;
      $("max-parallel").value = n.max_parallel; $("mp-out").value = n.max_parallel;
      const h = n.active_hours;
      $("sched-on").checked = !!h; $("hours").hidden = !h;
      if (h) { const [a, b] = h.split("-"); $("h-from").value = a; $("h-to").value = b; }
    }
    $("limits").hidden = !n.serving;
    $("change-model").hidden = !S.setup;
    viz.setMe({ model: n.model, family: n.family, serving: n.serving, accepting: n.accepting_now, busy: n.running });
  }
  function renderActivity() {
    const n = S.status && S.status.node, ul = $("recent");
    ul.replaceChildren();
    if (!n || !n.recent.length) { ul.append(el("li", { class: "empty", text: t("activity.empty") })); return; }
    for (const j of n.recent) {
      const cls = { "servi": "ok", "échec": "bad", "annulé": "warn", "en cours": "info" }[j.status] || "";
      const when = new Date(j.ts * 1000).toLocaleTimeString(loc());
      const extra = j.tokens !== undefined ? t("activity.tokens", { n: j.tokens }) : j.error || "";
      ul.append(el("li", {},
        el("time", { text: when }), el("span", { class: `tag ${cls}`, text: t(`status.${j.status}`) }),
        el("span", { text: t("activity.job", { id: j.job_id.slice(0, 8), who: j.requester }) }),
        el("span", { class: "hint", text: [extra, j.ms !== undefined ? `${fmt(j.ms, 0)} ms` : ""].filter(Boolean).join(" · ") })));
    }
  }
  function trackJobs() {
    // A new job served by this node: a comet from a peer to us; when it completes, one back.
    const n = S.status && S.status.node; if (!n) return;
    for (const j of n.recent) {
      const prev = S.seenJobs.get(j.job_id);
      if (prev === undefined) {
        const from = viz.randomPeer();
        if (from && Date.now() / 1000 - j.ts < 15) viz.pulse(from, "me", famColor(n.family));
        S.seenJobs.set(j.job_id, { status: j.status, from });
      } else if (prev.status === "en cours" && j.status !== "en cours") {
        if (prev.from && j.status === "servi") viz.pulse("me", prev.from, famColor(n.family));
        prev.status = j.status;
      }
    }
    if (S.seenJobs.size > 400) S.seenJobs = new Map([...S.seenJobs].slice(-200));
  }
  function renderCounters() {
    const s = S.status, nw = S.network, st = nw && nw.stats, n = s && s.node;
    const peers = nw ? nw.peers.filter((p) => p.node_id !== nw.me) : [];
    countTo($("s-peers"), st && st.nodes_online !== undefined ? Math.max(0, st.nodes_online - 1) : nw ? peers.length : null, 0);
    $("s-peers-sub").textContent = st && st.nodes_serving !== undefined ? t("stat.serving", { n: st.nodes_serving }) : "";
    const fams = st && st.families ? Object.keys(st.families) : [];
    countTo($("s-fam"), nw ? fams.length : null, 0);
    $("s-fam-sub").textContent = fams.slice(0, 4).join(" · ");
    $("s-fam-sub").title = fams.join(" · ");
    countTo($("s-ntps"), st && st.tokens_per_s !== null && st.tokens_per_s !== undefined ? st.tokens_per_s : null, 1);
    $("s-ntps-sub").textContent = st && st.partial ? t("stat.partial") : st && st.jobs_per_min !== undefined ? t("stat.jobs_min", { n: fmt(st.jobs_per_min, 1) }) : "";
    countTo($("s-mtps"), n ? n.tokens_per_s : null, 1);
    $("s-mtps-sub").textContent = !n ? "" : !n.accepting ? t("stat.paused") : n.running ? t("stat.running", { n: n.running }) : t("stat.idle");
    const acct = st && st.account;
    countTo($("s-earned"), acct ? acct.earned : null, 1);
    countTo($("s-spent"), acct ? acct.spent : null, 1);
    $("s-balance").textContent = s && s.balance !== null && s.balance !== undefined ? t("stat.balance", { n: fmt(s.balance, 1) }) : "";
    $("s-spent-sub").textContent = acct ? t("stat.bought", { n: fmt(acct.jobs_bought, 0) }) : "";
    if (acct) $("s-earned").title = t("stat.served_jobs", { n: acct.jobs_served });
  }
  function renderLegend() {
    const nw = S.network, lg = $("legend"); lg.replaceChildren();
    if (!nw) return;
    const fams = new Map();  // a Map: a family named "constructor" or "__proto__" is just a name
    for (const p of nw.peers) if (p.node_id !== nw.me) fams.set(p.family || "?", (fams.get(p.family || "?") || 0) + 1);
    for (const [f, c] of [...fams].sort((a, b) => b[1] - a[1])) {
      lg.append(el("span", { class: "chip" }, el("i", { style: { background: famColor(f) } }), f, el("b", { text: String(c) })));
    }
    $("viz-empty").hidden = fams.size > 0;
  }
  function renderPeersTable() {
    const nw = S.network, tb = $("peers"); tb.replaceChildren();
    if (!nw && !S.netTried) {  // first load: placeholder rows rather than "no peer"
      for (let i = 0; i < 4; i++) tb.append(el("tr", { class: "skel" }, ...Array.from({ length: 7 }, () => el("td", {}, el("span", { class: "skeleton" })))));
      return;
    }
    if (!nw || !nw.peers.length) { tb.append(el("tr", {}, el("td", { colspan: "7", class: "empty", text: t("peers.none") }))); return; }
    const rows = [...nw.peers].sort((a, b) => (a.family || "").localeCompare(b.family || "") || a.model.localeCompare(b.model));
    for (const p of rows) {
      const rel = own(nw.reliability, p.model);
      const load = p.max_parallel ? Math.min(1, p.busy / p.max_parallel) : 0;
      tb.append(el("tr", { class: p.node_id === nw.me ? "me" : "", "data-node": p.node_id },
        el("td", { class: "mono", text: p.node_id.slice(0, 10) + (p.node_id === nw.me ? ` (${t("peers.you")})` : "") }),
        el("td", { text: p.model }),
        el("td", {}, el("span", { class: "chip" }, el("i", { style: { background: famColor(p.family) } }), p.family || "?")),
        el("td", { class: "num", text: rel ? `${fmt(rel.p * 100, 1)} %` : "–" }),
        el("td", { class: "num", text: `${fmt((p.reputation ?? 1) * 100, 0)} %` }),
        el("td", { class: "num" }, el("span", { class: "meter" }, el("span", { style: { width: `${load * 100}%` } })), `${p.busy}/${p.max_parallel}`),
        el("td", {}, el("span", { class: `tag ${p.accepting ? "ok" : "warn"}`, text: p.accepting ? t("peers.available") : t("peers.paused") }))));
    }
  }
  function renderAbout() {
    const s = S.status;
    if (s && s.research) $("research-link").href = s.research;
    const base = (s && s.gateway) || "http://127.0.0.1:8400/v1";
    $("api-snippet").textContent =
      `from openai import OpenAI\nclient = OpenAI(base_url="${base}", api_key="-")\n` +
      `r = client.chat.completions.create(model="myriad", messages=[{"role": "user", "content": "…"}])\nprint(r.choices[0].message.content)`;
  }
  function renderTrackerBanner() {
    const n = S.status && S.status.node, ban = $("tracker-banner");
    const down = n && n.state !== "connecté";
    if (!down) { S.downSince = null; ban.hidden = true; return; }
    S.downSince = S.downSince || Date.now();
    if (Date.now() - S.downSince < 4000) return;  // a short reconnection is not worth a banner
    ban.hidden = false;
    const why = n.last_error ? ` (${n.last_error.length > 120 ? n.last_error.slice(0, 120) + "…" : n.last_error})` : "";
    $("tracker-down-text").textContent = t("tracker.down", { url: n.tracker, why });
    $("tracker-change-btn").hidden = !S.setup;
    if ($("tracker-form").hidden) $("tracker-url").value = n.tracker;
  }
  $("tracker-change-btn").addEventListener("click", () => {
    $("tracker-form").hidden = false; $("tracker-url").focus(); $("tracker-url").select();
  });
  $("tracker-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try {
      await api("/api/tracker", { url: $("tracker-url").value.trim() });
      $("tracker-form").hidden = true; S.downSince = null; toast(t("tracker.saved")); refresh();
    } catch (e) { toast(t("err.prefix", { m: e.message })); }
  });
  // ---------------------------------------------------------------- app updates
  function dismissedUpdate() { try { return localStorage.getItem("myriad.update.later"); } catch (e) { return null; } }
  function renderUpdate() {
    const u = S.status && S.status.update, ban = $("update-banner"), card = $("upd-card");
    card.hidden = !u;
    if (!u) { ban.hidden = true; return; }
    // About card: version, state, settings
    $("upd-cur").textContent = t("upd.current", { v: u.current });
    const st = $("upd-state");
    st.textContent = t(`upd.s.${u.state}`, { v: u.latest || "" });
    st.className = "upd-pill " + ({ up_to_date: "ok", ready: "ok", available: "info", downloading: "info", error: "bad" }[u.state] || "");
    if (!S.updDirty) { $("upd-auto").checked = !!u.auto_update; $("upd-quit").checked = !!u.install_on_quit; }
    $("upd-quit").closest(".switch-row").hidden = !u.can_apply && u.kind === "source";
    const rs = $("upd-reason"); rs.hidden = !(u.available && !u.can_apply);
    if (!rs.hidden) rs.textContent = t("upd.manual", { why: t(`upd.why.${u.kind}`) });
    // Banner
    const show = u.available && (u.state !== "available" || dismissedUpdate() !== u.latest || S.updShown === u.latest);
    if (!show || u.state === "up_to_date") { ban.hidden = true; return; }
    if (ban.hidden) S.updShown = u.latest;
    ban.hidden = false;
    ban.classList.toggle("is-ready", u.state === "ready");
    ban.classList.toggle("is-error", u.state === "error");
    const p = u.progress, v = u.latest;
    let title = t("upd.available", { v }), text = "";
    if (S.updRestarting || u.state === "installing") { title = t("upd.installing"); text = t("upd.restarting"); }
    else if (u.state === "downloading") {
      title = p && p.state === "vérification" ? t("upd.verifying") : t("upd.downloading", { v });
      if (p && p.total) text = t("upd.progress", { done: fmtBytes(p.done), total: fmtBytes(p.total) }) + (p.eta_s ? ` · ${t("dl.eta", { s: fmtDur(p.eta_s) })}` : "");
    } else if (u.state === "ready") {
      title = t("upd.ready", { v }); text = u.pending === "quit" ? t("upd.ready_quit") : t("upd.ready_sub");
    } else if (u.state === "error") { text = t("upd.error", { e: u.error || "?" }); }
    else if (!u.can_apply) { text = t("upd.manual", { why: t(`upd.why.${u.kind}`) }); }
    $("upd-title").textContent = title; $("upd-text").textContent = text;
    $("upd-notes").href = u.notes_url || "#"; $("upd-notes").hidden = !u.notes_url;
    $("upd-open").href = u.notes_url || "#";
    const bar = $("upd-bar"), pct = p && p.total ? Math.min(100, (100 * p.done) / p.total) : 0;
    bar.hidden = u.state !== "downloading";
    $("upd-fill").style.width = `${pct.toFixed(1)}%`;
    $("upd-apply").hidden = !(u.state === "ready" && u.can_apply) || S.updRestarting;
    $("upd-download").hidden = !(u.can_apply && (u.state === "available" || u.state === "error"));
    $("upd-open").hidden = u.can_apply;
    $("upd-dismiss").hidden = u.state !== "available";
  }
  $("upd-dismiss").addEventListener("click", () => {
    const u = S.status && S.status.update; if (!u) return;
    try { localStorage.setItem("myriad.update.later", u.latest); } catch (e) { /* ignore */ }
    S.updShown = null; $("update-banner").hidden = true;
  });
  async function updateCall(path, body) {
    const r = await api(path, body === undefined ? {} : body);
    if (S.status) S.status.update = r.update || r;
    renderUpdate();
    return r;
  }
  $("upd-apply").addEventListener("click", async () => {
    $("upd-apply").disabled = true;
    try { await updateCall("/api/update/apply"); S.updRestarting = true; renderUpdate(); toast(t("upd.restarting")); }
    catch (e) { toast(t("err.prefix", { m: e.message })); }
    finally { $("upd-apply").disabled = false; }
  });
  $("upd-download").addEventListener("click", async () => {
    try { await updateCall("/api/update/download"); } catch (e) { toast(t("err.prefix", { m: e.message })); }
  });
  $("upd-check").addEventListener("click", async () => {
    const b = $("upd-check"), msg = $("upd-msg"); b.disabled = true; msg.textContent = t("upd.s.checking");
    try {
      const r = await updateCall("/api/update/check");
      msg.textContent = r.state === "error" ? t("upd.error", { e: r.error }) : r.available ? t("upd.available", { v: r.latest }) : t("upd.checked");
      if (r.available) { S.updShown = r.latest; renderUpdate(); }
    } catch (e) { msg.textContent = t("err.prefix", { m: e.message }); }
    finally { b.disabled = false; }
  });
  for (const [id, key] of [["upd-auto", "auto_update"], ["upd-quit", "install_on_quit"]]) {
    $(id).addEventListener("change", async (ev) => {
      S.updDirty = true;
      try { await updateCall("/api/update/settings", { [key]: ev.target.checked }); $("upd-msg").textContent = t("saved"); }
      catch (e) { $("upd-msg").textContent = t("err.prefix", { m: e.message }); }
      finally { S.updDirty = false; }
    });
  }

  function renderAll() {
    renderUpdate();
    renderTrackerBanner();
    renderConn(); renderNode(); renderActivity(); renderCounters(); renderLegend(); renderPeersTable(); renderAbout();
  }

  // hours selects
  function fillHours(sel) {
    for (let h = 0; h <= 24; h++) sel.append(el("option", { value: String(h), text: `${String(h).padStart(2, "0")}:00` }));
  }
  for (const id of ["h-from", "h-to", "wz-from", "wz-to"]) fillHours($(id));
  $("h-from").value = "8"; $("h-to").value = "23"; $("wz-from").value = "8"; $("wz-to").value = "23";
  for (const id of ["accepting", "max-parallel", "sched-on", "h-from", "h-to"]) {
    $(id).addEventListener("input", () => { S.limitsDirty = true; });
  }
  $("max-parallel").addEventListener("input", (e) => { $("mp-out").value = e.target.value; });
  $("sched-on").addEventListener("change", (e) => { $("hours").hidden = !e.target.checked; });
  $("limits").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const msg = $("limits-msg");
    try {
      await api("/api/limits", { accepting: $("accepting").checked, max_parallel: Number($("max-parallel").value),
        active_hours: $("sched-on").checked ? `${$("h-from").value}-${$("h-to").value}` : null });
      S.limitsDirty = false; msg.textContent = t("saved"); refresh();
    } catch (e) { msg.textContent = t("err.prefix", { m: e.message }); }
  });
  $("pause-btn").addEventListener("click", async () => {
    const n = S.status && S.status.node; if (!n) return;
    try { await api("/api/limits", { accepting: !n.accepting }); S.limitsDirty = false; refresh(); }
    catch (e) { toast(t("err.prefix", { m: e.message })); }
  });

  // ---------------------------------------------------------------- polling
  async function refresh() {
    try { S.status = await api("/api/status"); } catch (e) { S.status = null; renderConn(); return; }
    if (!S.status.node) S.netTried = true;  // no node: no network to wait for
    const s = S.status;
    if (!s.node && s.runtime && !s.runtime.configured && !S.wizardOpen && !S.wizardDismissed) openWizard(0);
    if (s.node) {
      try { S.network = await api("/api/network"); } catch (e) { /* tracker unreachable: state shows it */ }
      S.netTried = true;
      if (S.network) {
        viz.setPeers(S.network.peers, S.network.me);
        const st = S.network.stats;
        viz.ambient(st && st.jobs_per_min ? st.jobs_per_min / 60 : 0);
      }
      trackJobs();
    }
    renderAll();
  }
  async function loadSetupInfo() {
    try { S.hardware = await api("/api/hardware"); } catch (e) { /* ignore */ }
    try { S.setup = await api("/api/setup"); } catch (e) { /* no wizard here (command line node) */ }
  }

  // ---------------------------------------------------------------- chat
  // A conversation: each question becomes a turn (question, answer, then a collapsible "how the swarm
  // decided" panel with the vote, the certificate and the peers asked). Only the last turn is live.
  const EX = ["chat.ex1", "chat.ex2", "chat.ex3"];
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function renderExamples() {
    const box = $("examples"); box.replaceChildren();
    for (const k of EX) {
      box.append(el("button", { type: "button", text: t(k), onclick: () => { $("question").value = t(k); autosize(); $("question").focus(); } }));
    }
  }
  renderExamples();
  function autosize() {
    const q = $("question");
    q.style.height = "auto";
    q.style.height = `${Math.min(220, q.scrollHeight)}px`;
  }
  $("question").addEventListener("input", autosize);
  $("k").addEventListener("input", (e) => { $("k-out").value = e.target.value; });
  document.querySelectorAll("#hint-seg button").forEach((b) => b.addEventListener("click", () => {
    S.hint = b.dataset.hint;
    document.querySelectorAll("#hint-seg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  }));
  $("question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); $("chat").requestSubmit(); }
  });
  $("chat-new").addEventListener("click", () => {
    if (S.chatBusy) return;
    $("thread").querySelectorAll(".turn").forEach((x) => x.remove());
    $("chat-empty").hidden = false; $("chat-new").hidden = true;
    $("question").focus();
  });
  function scrollToEnd() {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: reduced() ? "auto" : "smooth" });
  }
  // a label that follows the language switch (I18N.apply re-translates every [data-i18n])
  const L = (tag, key, attrs) => el(tag, { ...(attrs || {}), "data-i18n": key, text: t(key) });
  const chevron = () => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("class", "chev"); s.setAttribute("aria-hidden", "true");
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path"); p.setAttribute("d", "M6 9.5l6 6 6-6");
    s.append(p); return s;
  };

  const C = { peers: new Map(), hint: null, final: null, T: null };
  function newTurn(q) {
    $("chat-empty").hidden = true; $("chat-new").hidden = false;
    const T = {
      meta: el("span", { class: "a-meta" }),
      big: el("div", { class: "big-answer", hidden: true }),
      text: el("div", { class: "a-text wait", text: t("chat.thinking") }),
      err: el("p", { class: "error", hidden: true }),
      dots: el("span", { class: "swarm-dots", "aria-hidden": "true" }),
      sum: el("span", { class: "dec-sum" }),
      vote: el("div", { class: "vote" }),
      cert: el("div", { class: "cert" }),
      swarm: el("ol", { class: "swarm" }),
      count: el("span", { class: "count" }),
    };
    T.details = el("details", { class: "decision" },
      el("summary", {}, T.dots, L("span", "chat.decided", { class: "dec-title" }), T.sum, chevron()),
      el("div", { class: "dec-body" },
        el("section", { class: "dec-vote" }, L("h3", "chat.vote"), T.vote, T.cert),
        el("section", { class: "dec-peers" }, el("h3", {}, L("span", "chat.peers"), " ", T.count), T.swarm)));
    T.root = el("article", { class: "turn" },
      el("div", { class: "turn-q" }, el("p", { class: "q-bubble", text: q })),
      el("div", { class: "turn-a" },
        el("div", { class: "a-head" }, el("img", { src: "/static/icon.svg", alt: "", width: "20", height: "20" }),
          L("span", "chat.swarm", { class: "a-who" }), T.meta),
        T.big, T.text, T.err, T.details));
    $("thread").append(T.root);
    return T;
  }
  function renderDecisionSummary() {
    const T = C.T; if (!T) return;
    const peers = [...C.peers.values()];
    const ok = peers.filter((p) => p.status === "ok").length;
    const f = C.final;
    T.count.textContent = `${ok}/${C.peers.size}`;
    T.sum.replaceChildren();
    if (!f) { T.sum.textContent = peers.length ? t("chat.live", { a: ok, n: peers.length }) : ""; return; }
    const parts = [t(`rule.${f.decision}`), t("chat.agreed", { a: f.peers_answered, n: f.peers_asked })];
    if (f.certificate) parts.push(t("badge.cert"));
    if (f.early_stop) parts.push(t("badge.early"));
    T.sum.textContent = parts.join(" · ");
  }
  function peerDot(p) {
    if (!p.dot) { p.dot = el("span", { class: "sd" }); C.T.dots.append(p.dot); }
    p.dot.style.setProperty("--fam", famColor(p.family));
    p.dot.className = "sd " + ({ asked: "thinking", thinking: "thinking", ok: "ok", erreur: "err" }[p.status] || "off") + (p.chosen ? " chosen" : "");
    p.dot.title = (p.model || "").split("/").pop();
  }
  function peerCard(jid) {
    const p = C.peers.get(jid);
    if (!p.el) { p.el = el("li", { class: "peer" }); C.T.swarm.append(p.el); }
    peerDot(p);
    const e = p.el;
    e.replaceChildren();
    if (p.node_id) e.dataset.node = p.node_id;  // security.js adds « Bloquer ce nœud »
    e.style.setProperty("--fam", famColor(p.family));
    e.classList.toggle("chosen", !!p.chosen);
    e.classList.toggle("cancelled", p.status === "annulé" || p.status === "sans réponse");
    const name = (p.model || p.node_id || "?").split("/").pop().replace(/-GGUF$/i, "");
    const ans = el("div", { class: "p-ans" });
    if (p.status === "asked" || p.status === "thinking") ans.append(el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")));
    else if (p.answer !== undefined && p.answer !== null) {
      const agree = C.final && C.final.answer !== null && C.final.answer !== undefined ? (p.answer === C.final.answer ? " agree" : " disagree") : "";
      ans.append(el("span", { class: `peer-ans${agree}`, text: p.answer }));
    } else if (p.error) ans.append(el("span", { class: "hint", text: p.error }));
    const time = el("span", { class: "peer-time", text: p.ms ? `${fmt(p.ms, 0)} ms${p.tokens ? ` · ${p.tokens} tok` : ""}` : "" });
    const status = p.chosen ? el("span", { class: "tag ok", text: t("peer.chosen") })
      : el("span", { class: `tag${p.status === "ok" ? " info" : p.status === "erreur" ? " bad" : ""}`, text: t(`peer.${p.status}`) });
    const weight = el("span", { class: "p-weight" });
    if (p.weight !== undefined) {
      const wmax = Math.max(...[...C.peers.values()].map((x) => x.weight || 0), 0.001);
      const bar = el("span", { class: "wbar" }, el("span"));
      weight.title = t("peer.weight", { w: fmt(p.weight, 2) });
      weight.append(bar, t("peer.weight", { w: fmt(p.weight, 2) }));
      requestAnimationFrame(() => { bar.firstChild.style.width = `${(p.weight / wmax) * 100}%`; });
    }
    e.append(el("span", { class: "p-dot" }),
      el("div", { class: "p-who" }, el("span", { class: "p-name", text: name }),
        el("span", { class: "p-meta", text: `${p.family || "?"} · ${(p.node_id || "").slice(0, 8)}` })),
      ans, time, weight, el("span", { class: "p-status" }, status));
    if (p.text) e.append(el("details", {}, el("summary", { text: t("peer.show") }), el("pre", { text: p.text })));
    renderDecisionSummary();
  }
  function renderVote() {
    const T = C.T; if (!T) return;
    const box = T.vote, cert = T.cert;
    box.replaceChildren(); cert.replaceChildren(); cert.className = "cert";
    const f = C.final;
    const peers = [...C.peers.values()];
    if (f && f.decision !== "vote") { box.append(el("p", { class: "hint", text: t("vote.none") })); return; }
    // tallies: weights when known (final), else one voice per answer
    const groups = new Map();
    for (const p of peers) {
      if (p.status !== "ok" || p.answer === null || p.answer === undefined) continue;
      if (!groups.has(p.answer)) groups.set(p.answer, []);
      groups.get(p.answer).push(p);
    }
    const w = (p) => (f ? p.weight || 0 : 1);
    const pendingW = peers.filter((p) => !["ok", "erreur"].includes(p.status)).reduce((a, p) => a + (f ? p.weight || 0 : 1), 0);
    const total = Math.max(0.001, peers.reduce((a, p) => a + (f ? p.weight || 0 : 1), 0));
    const sorted = [...groups.entries()].sort((a, b) => b[1].reduce((s, p) => s + w(p), 0) - a[1].reduce((s, p) => s + w(p), 0));
    if (!sorted.length) box.append(el("p", { class: "hint", text: t("vote.waiting") }));
    sorted.forEach(([ans, ps]) => {
      const sum = ps.reduce((s, p) => s + w(p), 0);
      const bar = el("div", { class: "bar" });
      for (const p of ps) {
        const seg = el("span", { style: { background: famColor(p.family), width: "0%" }, title: (p.model || "").split("/").pop() });
        bar.append(seg);
        requestAnimationFrame(() => { seg.style.width = `${(w(p) / total) * 100}%`; });
      }
      box.append(el("div", { class: `vote-row${f && f.answer === ans ? " win" : ""}` },
        el("span", { class: "lbl", text: ans, title: ans }), bar, el("span", { class: "val", text: f ? fmt(sum, 2) : `×${ps.length}` })));
    });
    if (pendingW > 0) {
      box.append(el("div", { class: "vote-row pending" }, el("span", { class: "lbl", text: "…" }), el("div", { class: "bar" }),
        el("span", { class: "val", text: f ? fmt(pendingW, 2) : `×${pendingW}` })));
    }
    if (!f) return;
    const lead = sorted.length ? sorted[0][1].reduce((s, p) => s + w(p), 0) : 0;
    const second = sorted.length > 1 ? sorted[1][1].reduce((s, p) => s + w(p), 0) : 0;
    const rest = second + pendingW, mx = Math.max(lead, rest, 0.001);
    cert.classList.toggle("ok", !!f.certificate);
    cert.append(el("div", { class: "verdict", text: f.early_stop ? t("cert.early") : f.certificate ? t("cert.ok") : t("cert.no") }));
    const a = el("span", { class: "a" }), b = el("span", { class: "b" });
    cert.append(el("div", { class: "scale" }, a, b),
      el("div", { class: "legend" }, el("span", { text: `${t("cert.lead")} ${fmt(lead, 2)}` }), el("span", { text: `${t("cert.rest")} ${fmt(rest, 2)}` })));
    requestAnimationFrame(() => { a.style.width = `${(lead / mx) * 100}%`; b.style.width = `${(rest / mx) * 100}%`; });
  }
  function typewrite(node, text) {
    node.classList.remove("wait"); node.replaceChildren();
    if (reduced()) { node.textContent = text; return; }
    const span = document.createTextNode(""), caret = el("span", { class: "caret" });
    node.append(span, caret);
    let i = 0; const step = Math.max(2, Math.ceil(text.length / 160));
    const tick = () => {
      i = Math.min(text.length, i + step); span.textContent = text.slice(0, i);
      if (i < text.length) setTimeout(tick, 14); else caret.remove();
    };
    tick();
  }
  function handle(ev) {
    const T = C.T;
    if (ev.type === "start") { C.hint = ev.task_hint; return; }
    if (ev.type === "asked" || ev.type === "assigned") {
      const p = C.peers.get(ev.job_id) || {};
      Object.assign(p, { node_id: ev.node_id || p.node_id, model: ev.model || p.model, family: ev.family || p.family, status: "thinking" });
      C.peers.set(ev.job_id, p); peerCard(ev.job_id); renderVote();
      if (ev.node_id && viz.has(ev.node_id)) viz.pulse("me", ev.node_id);
      return;
    }
    if (ev.type === "answered") {
      const p = C.peers.get(ev.job_id) || {};
      Object.assign(p, { status: "ok", answer: ev.answer, text: ev.text, ms: ev.ms, tokens: ev.tokens, model: ev.model || p.model, node_id: ev.node_id || p.node_id });
      C.peers.set(ev.job_id, p); peerCard(ev.job_id); renderVote();
      if (p.node_id && viz.has(p.node_id)) viz.pulse(p.node_id, "me");
      return;
    }
    if (ev.type === "failed") {
      const p = C.peers.get(ev.job_id); if (!p) return;
      Object.assign(p, { status: "erreur", error: ev.error, ms: ev.ms }); peerCard(ev.job_id); renderVote(); return;
    }
    if (ev.type === "final") {
      // The gateway's validated outcome wins over what the stream showed: a result it rejected (bad
      // signature…) loses its answer and its text.
      const m = ev.body.myriad; C.final = m;
      const texts = ev.texts || {};
      for (const fp of m.peers) {
        let jid = [...C.peers.entries()].find(([, p]) => p.node_id === fp.node_id && !p.matched);
        if (!jid) { jid = [`x-${fp.node_id}`, {}]; C.peers.set(jid[0], jid[1]); }
        const p = jid[1], ok = fp.status === "ok";
        const text = Object.prototype.hasOwnProperty.call(texts, jid[0]) ? texts[jid[0]] : ok ? p.text : undefined;
        Object.assign(p, { matched: true, node_id: fp.node_id, model: fp.model, family: fp.family, weight: fp.weight, chosen: fp.chosen,
                           answer: ok ? fp.answer : null, text, error: fp.error, ms: p.ms || fp.latency_ms, tokens: fp.completion_tokens || p.tokens,
                           status: ok ? "ok" : fp.status === "en attente" ? "sans réponse" : fp.status });
      }
      for (const p of C.peers.values()) {  // asked, but not in the gateway's list: nothing to show
        if (!p.matched) Object.assign(p, { answer: null, text: undefined, status: ["asked", "thinking", "ok"].includes(p.status) ? "sans réponse" : p.status });
      }
      for (const k of C.peers.keys()) peerCard(k);
      renderVote();
      const text = ev.body.choices[0].message.content;
      if (m.answer) { T.big.hidden = false; T.big.replaceChildren(el("b", { text: m.answer }), L("small", "chat.answer")); }
      typewrite(T.text, text);
      // one polite announcement of the whole answer (the typewriter itself is not a live region: too chatty)
      $("chat-announce").textContent = `${t("chat.swarm")}. ${m.answer ? `${m.answer}. ` : ""}${text}`;
      T.meta.textContent = t("chat.meta", { rule: t(`rule.${m.decision}`), a: m.peers_answered, n: m.peers_asked,
        ms: fmt(m.latency_ms, 0), hint: t(`hint.${m.task_hint}`) });
      renderDecisionSummary();
      return;
    }
    if (ev.type === "error") {
      T.err.hidden = false; T.err.textContent = t("err.prefix", { m: ev.message });
      $("chat-announce").textContent = T.err.textContent;
      T.text.textContent = ""; T.text.classList.remove("wait"); T.text.hidden = true;
      // nothing runs any more: no peer may look busy
      for (const [jid, p] of C.peers) if (p.status === "asked" || p.status === "thinking") { p.status = "sans réponse"; peerCard(jid); }
      renderVote();
      if (!C.peers.size) T.details.hidden = true;
    }
  }
  $("chat").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = $("question").value.trim();
    if (!q || S.chatBusy) return;
    S.chatBusy = true;
    const btn = $("send"); btn.disabled = true; $("chat-new").disabled = true;
    C.peers = new Map(); C.final = null;
    C.T = newTurn(q);
    $("chat-announce").textContent = "";
    $("question").value = ""; autosize();
    scrollToEnd();
    try {
      const r = await fetch("/api/chat/stream", { method: "POST", headers: { "Content-Type": "application/json", "X-Myriad-Token": TOKEN },
        body: JSON.stringify({ message: q, k: Number($("k").value) || null, task_hint: S.hint || null,
                               ...(window.MyriadSecurity ? window.MyriadSecurity.take() : {}) }) });
      if (!r.ok || !r.body) { const j = await r.json().catch(() => ({})); throw new Error(j.error || `HTTP ${r.status}`); }
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "", ended = false;  // ended: a final or error event arrived
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
          for (const line of chunk.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            let ev; try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
            if (ev.type === "final" || ev.type === "error") ended = true;
            try { handle(ev); } catch (e) { /* ignore a bad event */ }
          }
        }
      }
      if (!ended) throw new Error(t("chat.interrupted"));  // the stream closed before its result
    } catch (e) {
      handle({ type: "error", message: e.message });
    } finally {
      S.chatBusy = false; btn.disabled = false; $("chat-new").disabled = false; refresh();
      scrollToEnd();
    }
  });

  // ---------------------------------------------------------------- wizard
  const W = { step: 0, model: null, quant: "Q4_K_M", clientOnly: false, polling: null, installing: false, mode: "first" };
  const steps = [...document.querySelectorAll("#wz-steps li")];
  function openWizard(step, mode) {
    S.wizardOpen = true; W.mode = mode || "first"; W.step = step || 0;
    $("wizard").hidden = false;
    $("wz-close").hidden = W.mode === "first";
    loadSetup();
    renderWizard();
    setTimeout(() => $("wz-next").focus(), 50);
    heroStart();
  }
  function closeWizard() {
    S.wizardOpen = false; S.wizardDismissed = true; $("wizard").hidden = true; heroStop();
    clearInterval(W.polling); W.polling = null; refresh();
  }
  async function loadSetup(refresh) {
    try {
      S.setup = await api(`/api/setup${refresh ? "?refresh=true" : ""}`);
      const d = S.setup.defaults;
      if (!W.model) { W.model = d.model || S.setup.recommendation.model; W.quant = d.quant || S.setup.recommendation.quant; }
      $("wz-tracker").value = d.tracker_url;
      $("wz-mp").value = d.max_parallel; $("wz-mp-out").value = d.max_parallel;
      $("wz-accepting").checked = d.accepting;
      // always in step with the server, "no hours" included (else an old range would be sent again)
      $("wz-sched").checked = !!d.active_hours; $("wz-hours").hidden = !d.active_hours;
      if (d.active_hours) { const [a, b] = d.active_hours.split("-"); $("wz-from").value = a; $("wz-to").value = b; }
      if (S.setup.job && S.setup.job.state === "en cours") { W.step = 4; startPolling(); }
      renderWizard();
    } catch (e) { $("wz-error").hidden = false; $("wz-error").textContent = t("err.prefix", { m: e.message }); }
  }
  function renderWizard() {
    steps.forEach((li, i) => { li.classList.toggle("current", i === W.step); li.classList.toggle("done", i < W.step); });
    document.querySelectorAll(".wz-page").forEach((p) => { p.hidden = Number(p.dataset.page) !== W.step; });
    $("wz-back").hidden = W.step === 0 || W.installing || (W.mode !== "first" && W.step === 2);
    const next = $("wz-next");
    next.textContent = W.step === 3 ? t("wz.install") : W.step === 4 ? (jobState() === "terminé" ? t("wz.finish") : jobState() === "erreur" || jobState() === "annulé" ? t("wz.retry") : t("wz.cancel")) : t("next");
    next.classList.toggle("primary", !(W.step === 4 && jobState() === "en cours"));
    next.disabled = W.step >= 1 && W.step <= 3 && (!S.setup || (W.step >= 2 && !W.clientOnly && !W.model));
    renderHardware(); renderModels(); renderBackends(); renderSummary();
  }
  function jobState() { return S.setup && S.setup.job ? S.setup.job.state : null; }
  function renderHardware() {
    const box = $("hw-grid"), hw = S.setup && S.setup.hardware;
    if (!hw) return;
    box.replaceChildren();
    const tile = (k, v, s, hl) => el("div", { class: `hw${hl ? " hl" : ""}` }, el("div", { class: "k", text: k }), el("div", { class: "v", text: v }), s ? el("div", { class: "s", text: s }) : null);
    const g = hw.best_gpu;
    box.append(
      tile(t("wz.gpu"), g ? g.name : "–", g ? (g.unified ? `${fmt(g.vram_gb)} ${t("unit.gb")} · ${t("wz.unified")}` : g.vram_gb ? `${fmt(g.vram_gb)} ${t("unit.gb")} VRAM` : "") : t("wz.no_gpu"), !!g),
      tile(t("wz.cpu"), hw.cpu, t("wz.cores", { n: hw.cores })),
      tile(t("wz.ram"), hw.ram_gb ? `${fmt(hw.ram_gb)} ${t("unit.gb")}` : "–", `${hw.os} · ${hw.arch}`),
      tile(t("wz.accel"), hw.accelerator.toUpperCase(), `${t("wz.bandwidth")} ≈ ${fmt(hw.bandwidth_gbs, 0)} ${t("unit.gb")}/s`, true),
      tile(t("wz.budget"), `${fmt(hw.model_budget_gb)} ${t("unit.gb")}`, ""));
  }
  function renderModels() {
    const box = $("models"), st = S.setup; if (!st) return;
    box.replaceChildren();
    box.classList.toggle("disabled", W.clientOnly);
    document.querySelectorAll("#quant-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.quant === W.quant)));
    const maxTps = Math.max(...st.catalog.map((m) => m.quants[W.quant] ? m.quants[W.quant].tps : 0), 1);
    for (const m of st.catalog) {
      const q = m.quants[W.quant]; if (!q) continue;
      const rec = st.recommendation.model === m.id;
      const card = el("button", { type: "button", role: "radio", class: `model${q.fits ? "" : " nofit"}`, "aria-checked": String(W.model === m.id),
        tabindex: W.model === m.id ? "0" : "-1", style: { "--fam": famColor(m.family) },
        onclick: () => { W.model = m.id; renderWizard(); } });
      const speed = el("div", { class: "speed" }, el("span", { style: { width: `${Math.min(100, (q.tps / maxTps) * 100)}%` } }));
      card.append(el("div", { class: "name", text: m.name }),
        el("div", { class: "meta", text: `${m.family} · ${fmt(m.params_b)} G · ${m.licence.toUpperCase()} · ${fmtBytes(q.size)}` }),
        el("div", { class: "blurb", text: window.I18N.lang === "fr" ? m.blurb_fr : m.blurb_en }),
        el("div", { class: "row" }, el("span", { class: "tps", text: t("wz.tps", { n: fmt(q.tps, 0) }) }),
          el("span", { class: "hint", text: m.peers ? t("wz.on_net", { n: m.peers }) : "" })), speed,
        el("div", { class: "flags" }, rec ? el("span", { class: "tag ok", text: t("wz.recommended") }) : null,
          !q.fits ? el("span", { class: "tag bad", text: t("wz.too_big") }) : null,
          !m.peers && Object.keys(st.network_families || {}).length ? el("span", { class: "tag info", text: t("wz.rare") }) : null));
      card.addEventListener("keydown", (ev) => {
        const cards = [...box.querySelectorAll(".model")], i = cards.indexOf(card);
        if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(ev.key)) {
          ev.preventDefault();
          const n = cards[(i + (ev.key === "ArrowRight" || ev.key === "ArrowDown" ? 1 : cards.length - 1)) % cards.length];
          n.click(); box.querySelector('[aria-checked="true"]').focus();
        }
      });
      box.append(card);
    }
  }
  function renderBackends() {
    const sel = $("wz-backend"), st = S.setup; if (!st) return;
    const cur = sel.value;
    sel.replaceChildren(el("option", { value: "", text: st.engine.existing ? t("wz.backend_existing", { p: st.engine.existing }) : `${t("wz.backend_auto")} (${st.engine.default_backend})` }));
    for (const b of st.engine.backends) sel.append(el("option", { value: b.id, text: `${b.id.toUpperCase()} · ${fmtBytes(b.size)}${b.installed ? " ✓" : ""}` }));
    sel.value = cur;
  }
  function renderSummary() {
    const st = S.setup, sm = $("wz-summary");
    if (!st || W.step < 2) { sm.textContent = ""; return; }
    if (W.clientOnly) { sm.textContent = t("wz.summary_client"); return; }
    const m = st.catalog.find((x) => x.id === W.model);
    sm.textContent = m ? t("wz.summary", { model: m.name, quant: W.quant, size: fmtBytes(m.quants[W.quant].size) }) : "";
  }
  document.querySelectorAll("#quant-seg button").forEach((b) => b.addEventListener("click", () => { W.quant = b.dataset.quant; renderModels(); renderSummary(); }));
  $("client-only").addEventListener("change", (e) => { W.clientOnly = e.target.checked; renderWizard(); });
  $("wz-mp").addEventListener("input", (e) => { $("wz-mp-out").value = e.target.value; });
  $("wz-sched").addEventListener("change", (e) => { $("wz-hours").hidden = !e.target.checked; });
  $("hw-refresh").addEventListener("click", () => loadSetup(true));
  $("wz-back").addEventListener("click", () => { if (W.step > 0) { W.step -= 1; renderWizard(); } });
  $("wz-close").addEventListener("click", closeWizard);
  $("change-model").addEventListener("click", () => { S.wizardDismissed = false; openWizard(2, "change"); });
  $("wizard").addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && W.mode !== "first" && !W.installing) closeWizard();
    if (ev.key === "Tab") { // keep the focus inside the dialog
      const f = [...$("wizard").querySelectorAll("button, input, select, [tabindex]")].filter((x) => !x.disabled && x.offsetParent !== null && x.tabIndex >= 0);
      if (!f.length) return;
      if (ev.shiftKey && document.activeElement === f[0]) { ev.preventDefault(); f[f.length - 1].focus(); }
      else if (!ev.shiftKey && document.activeElement === f[f.length - 1]) { ev.preventDefault(); f[0].focus(); }
    }
  });
  $("wz-next").addEventListener("click", async () => {
    if (W.step < 3) { W.step += 1; renderWizard(); return; }
    if (W.step === 3 || (W.step === 4 && ["erreur", "annulé"].includes(jobState()))) return install();
    if (W.step === 4 && jobState() === "terminé") { closeWizard(); show("dashboard"); return; }
    if (W.step === 4 && jobState() === "en cours") { try { await api("/api/setup/cancel", {}); } catch (e) { /* ignore */ } pollOnce(); }
  });
  async function install() {
    $("wz-error").hidden = true;
    if (!S.setup || (!W.clientOnly && !W.model)) {  // setup data not loaded yet: never install by default
      $("wz-error").hidden = false; $("wz-error").textContent = t("wz.not_ready"); return;
    }
    const body = { model: W.clientOnly ? null : W.model, quant: W.quant, max_parallel: Number($("wz-mp").value),
      accepting: $("wz-accepting").checked, active_hours: $("wz-sched").checked ? `${$("wz-from").value}-${$("wz-to").value}` : null,
      tracker_url: $("wz-tracker").value.trim() || null, backend: $("wz-backend").value || null };
    try {
      const r = await api("/api/setup/install", body);
      if (S.setup) S.setup.job = r.job; W.step = 4; W.installing = true; renderWizard(); renderDownloads(); startPolling();
    } catch (e) { W.step = 4; renderWizard(); $("wz-error").hidden = false; $("wz-error").textContent = t("err.prefix", { m: e.message }); }
  }
  function startPolling() { clearInterval(W.polling); W.polling = setInterval(pollOnce, 700); pollOnce(); }
  async function pollOnce() {
    try {
      const r = await api("/api/setup/progress");
      S.setup.job = r.job; renderDownloads();
      const st = jobState();
      if (st !== "en cours") {
        clearInterval(W.polling); W.polling = null; W.installing = false;
        if (st === "terminé") { $("wz-done").hidden = false; refresh(); }
        if (st === "erreur") { $("wz-error").hidden = false; $("wz-error").textContent = t("err.prefix", { m: r.job.error }); }
      }
      renderWizard();
    } catch (e) { /* transient */ }
  }
  function renderDownloads() {
    const box = $("dl-list"), job = S.setup && S.setup.job;
    if (!job) return;
    box.replaceChildren();
    for (const s of job.steps) {
      const pct = s.total ? Math.min(100, (s.done / s.total) * 100) : 0;
      const label = s.id === "model" ? t("dl.model") : t("dl.engine");
      const state = s.state === "en cours"
        ? `${fmtBytes(s.done)} / ${fmtBytes(s.total)} · ${fmtBytes(s.speed_bps)}/s${s.eta_s ? ` · ${t("dl.eta", { s: fmtDur(s.eta_s) })}` : ""}`
        : `${t(`dl.${s.state}`)}${s.total ? ` · ${fmtBytes(s.total)}` : ""}`;
      const bar = el("span"); bar.style.width = `${pct}%`;
      box.append(el("div", { class: `dl${s.state === "terminé" ? " done" : ""}` },
        el("div", { class: "dl-top" }, el("span", { class: "dl-name", text: `${label} · ${s.label}` }), el("span", { class: "dl-state", text: state })),
        el("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(pct)), "aria-label": label }, bar)));
    }
    $("wz-done").hidden = job.state !== "terminé";
  }

  // welcome hero: a myriad of dots gathering into one disc (the logo)
  let heroRaf = null;
  function heroStart() {
    const c = $("wz-hero"), ctx = c.getContext("2d");
    const N = 140, GA = Math.PI * (3 - Math.sqrt(5));
    const pts = Array.from({ length: N }, (_, i) => ({ i, x: Math.random(), y: Math.random(), f: Math.sqrt((i + 0.5) / N) }));
    const stops = [[0, [255, 214, 110]], [0.35, [255, 106, 160]], [0.7, [124, 108, 255]], [1, [58, 214, 255]]];
    const col = (f) => { for (let k = 1; k < stops.length; k++) if (f <= stops[k][0]) { const [a0, c0] = stops[k - 1], [a1, c1] = stops[k]; const u = (f - a0) / (a1 - a0); return `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * u)).join(",")})`; } return "rgb(58,214,255)"; };
    const t0 = performance.now();
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function frame(now) {
      const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      if (c.width !== Math.round(r.width * dpr)) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
      const u = still ? 1 : Math.min(1, (now - t0) / 2200), e = 1 - (1 - u) ** 3;
      const cx = r.width / 2, cy = r.height / 2, R = Math.min(r.height * 0.46, r.width * 0.3), rot = still ? 0 : (now - t0) / 9000;
      for (const p of pts) {
        const a = p.i * GA + rot, tx = cx + Math.cos(a) * R * p.f, ty = cy + Math.sin(a) * R * p.f;
        const sx = p.x * r.width, sy = p.y * r.height;
        const x = sx + (tx - sx) * e, y = sy + (ty - sy) * e;
        ctx.fillStyle = col(p.f); ctx.globalAlpha = 0.35 + 0.65 * e;
        ctx.beginPath(); ctx.arc(x, y, (1.6 + 3.2 * (1 - p.f)) * (0.6 + 0.4 * e), 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      heroRaf = requestAnimationFrame(frame);
    }
    cancelAnimationFrame(heroRaf); heroRaf = requestAnimationFrame(frame);
  }
  function heroStop() { cancelAnimationFrame(heroRaf); heroRaf = null; }

  // ---------------------------------------------------------------- start
  window.I18N.apply();
  loadSetupInfo().then(renderNode);
  refresh();
  setInterval(refresh, 2500);
})();
