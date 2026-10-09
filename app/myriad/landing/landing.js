// Landing page behaviour: language and theme (remembered), download links resolved from the latest
// GitHub release, live figures of the network (GET /v1/stats and /v1/health of this tracker), the
// screenshot viewer and the reveal-on-scroll effects. No dependency, no inline code (strict CSP).
(function () {
  "use strict";
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const I18N = window.MyriadI18n || { en: {}, dyn: { fr: {}, en: {} } };
  const REPO = "amintt2/myriad";
  const RELEASES = `https://github.com/${REPO}/releases`;
  const LATEST = `${RELEASES}/latest`;
  const API = `https://api.github.com/repos/${REPO}/releases/latest`;
  const DOWNLOAD_PREFIX = `https://github.com/${REPO}/releases/download/`;
  const POLL_MS = 10000;
  const RELEASE_TTL_MS = 30 * 60 * 1000;
  const FAMILY_COLORS = { // same as the app's live map (myriad/web/netviz.js)
    qwen: "#8b7bff", gemma: "#38c8ff", granite: "#35d49a", smollm: "#ffb547", mistral: "#ff7f50",
    phi: "#ff5fa2", llama: "#a3e05a", olmo: "#f2d14b", deepseek: "#5b8cff", falcon: "#c792ea", exaone: "#4dd0e1",
  };

  // ---------- storage (may be unavailable: private mode, blocked site data) ----------
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* ignored */ } },
  };

  // ---------- language ----------
  let lang = document.documentElement.getAttribute("data-lang") === "en" ? "en" : "fr";
  const original = new Map(); // element -> French markup / attribute, captured once
  const fmt = (s, vars) => String(s).replace(/\{(\w+)\}/g, (_, k) => (vars && k in vars ? vars[k] : ""));
  const t = (key, vars) => fmt((I18N.dyn[lang] || {})[key] || I18N.dyn.fr[key] || key, vars);
  const nf = (v, digits) => new Intl.NumberFormat(lang === "fr" ? "fr-FR" : "en-GB",
    { maximumFractionDigits: digits || 0, minimumFractionDigits: digits || 0 }).format(v);

  function applyLang() {
    const en = lang === "en";
    for (const el of $$("[data-i18n]")) {
      if (!original.has(el)) original.set(el, el.innerHTML);
      const v = en ? I18N.en[el.dataset.i18n] : original.get(el);
      if (v != null) el.innerHTML = v;
    }
    for (const [attr, data] of [["aria-label", "i18nAria"], ["alt", "i18nAlt"]]) {
      for (const el of $$(`[data-${data === "i18nAria" ? "i18n-aria" : "i18n-alt"}]`)) {
        const key = `${attr}:fr`;
        if (!el[key]) el[key] = el.getAttribute(attr) || "";
        const v = en ? I18N.en[el.dataset[data]] : el[key];
        if (v != null) el.setAttribute(attr, v);
      }
    }
    for (const el of $$("[data-n]")) { // research figures, in the visitor's number format
      const raw = el.dataset.n, v = parseFloat(raw);
      if (!isFinite(v)) continue;
      const digits = raw.includes(".") ? raw.split(".")[1].length : 0;
      let s = nf(v, digits);
      if (el.hasAttribute("data-sign") && v > 0) s = `+${s}`;
      if (el.dataset.unit === "%") s = lang === "fr" ? `${s} %` : `${s}%`;
      el.textContent = s;
    }
    document.documentElement.lang = lang;
    document.documentElement.setAttribute("data-lang", lang);
    document.title = t("title");
    const desc = $('meta[name="description"]');
    if (desc) desc.setAttribute("content", t("description"));
    for (const b of $$("[data-lang]", $(".seg"))) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
    document.documentElement.classList.add("i18n-done");
    applyTheme(false);
    renderDownloads();
    renderStats();
  }

  // ---------- theme ----------
  function theme() { return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark"; }
  function applyTheme(save) {
    const th = theme();
    const btn = $("#theme");
    if (btn) btn.setAttribute("aria-label", th === "dark" ? t("toLight") : t("toDark"));
    const meta = $('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", th === "dark" ? "#07080d" : "#f5f6fb");
    if (save) store.set("myriad.site.theme", th);
  }

  // ---------- downloads ----------
  const PLATFORM_NAMES = { win: "Windows", mac: "macOS", linux: "Linux" };
  const os = { name: null, arch: null, mobile: false };
  function detectOs() {
    const ua = navigator.userAgent || "";
    const plat = ((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "").toLowerCase();
    os.mobile = /android|iphone|ipad|ipod|mobile/i.test(ua) || (navigator.userAgentData && navigator.userAgentData.mobile) === true
      || (/macintosh/i.test(ua) && navigator.maxTouchPoints > 1); // iPadOS says "Macintosh"
    if (os.mobile) return;
    if (plat.startsWith("win") || /windows/i.test(ua)) os.name = "win";
    else if (plat.startsWith("mac") || /mac os x|macintosh/i.test(ua)) os.name = "mac";
    else if (plat.includes("linux") || /linux|x11/i.test(ua)) os.name = "linux";
    if (os.name === "mac" && navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
      navigator.userAgentData.getHighEntropyValues(["architecture"]).then((v) => {
        if (v && v.architecture) { os.arch = v.architecture === "arm" ? "arm" : "x86"; renderDownloads(); }
      }).catch(() => {});
    }
  }

  // Asset kind -> matcher on the file name (see .github/workflows/release.yml for the names).
  const MATCHERS = {
    "win-exe": (n) => /^Myriad-Setup-.*\.exe$/i.test(n),
    "win-zip": (n) => /windows.*\.zip$/i.test(n),
    "mac-arm": (n) => /arm64\.dmg$/i.test(n),
    "mac-x64": (n) => /(x86_64|x64|intel)\.dmg$/i.test(n),
    "linux-appimage": (n) => /\.AppImage$/i.test(n),
    "linux-deb": (n) => /\.deb$/i.test(n),
    "linux-tgz": (n) => /linux.*\.tar\.gz$/i.test(n),
    "sums": (n) => /^SHA256SUMS(\.txt)?$/i.test(n),
  };
  let release = null; // {tag, date, assets: {kind: url}} or {none: true}

  function parseRelease(j) {
    if (!j || typeof j.tag_name !== "string" || !Array.isArray(j.assets)) return null;
    const assets = {};
    for (const a of j.assets) {
      const name = a && typeof a.name === "string" ? a.name : "";
      const url = a && typeof a.browser_download_url === "string" ? a.browser_download_url : "";
      if (!url.startsWith(DOWNLOAD_PREFIX)) continue; // only files of this repository's releases
      for (const kind in MATCHERS) if (!assets[kind] && MATCHERS[kind](name)) assets[kind] = url;
    }
    return { tag: j.tag_name.slice(0, 40), date: typeof j.published_at === "string" ? j.published_at : null,
             url: typeof j.html_url === "string" && j.html_url.startsWith(RELEASES) ? j.html_url : LATEST, assets };
  }

  async function loadRelease() {
    try {
      const cached = JSON.parse(store.get("myriad.site.release") || "null");
      if (cached && Date.now() - cached.at < RELEASE_TTL_MS && cached.release) { release = cached.release; renderDownloads(); return; }
    } catch (e) { /* ignored */ }
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 8000);
      const r = await fetch(API, { headers: { Accept: "application/vnd.github+json" }, signal: ctl.signal,
                                   credentials: "omit", referrerPolicy: "no-referrer", cache: "no-cache" });
      clearTimeout(timer);
      if (r.status === 404) release = { none: true }; // no release published yet
      else if (r.ok) release = parseRelease(await r.json());
      // 403/429 (rate limit) and other errors: keep the links to the releases page.
      if (release) store.set("myriad.site.release", JSON.stringify({ at: Date.now(), release }));
    } catch (e) { /* offline, blocked, rate-limited: fallback links stay */ }
    renderDownloads();
  }

  function assetUrl(kind) {
    return (release && release.assets && release.assets[kind]) || (release && release.url) || LATEST;
  }

  function primaryChoice() {
    const a = (release && release.assets) || {};
    if (os.mobile || !os.name) return null;
    if (os.name === "win") return a["win-exe"] || !a["win-zip"] ? { kind: "win-exe", sub: "dlWinSub" } : { kind: "win-zip", sub: "dlWinZipSub" };
    if (os.name === "mac") {
      if (os.arch === "x86") return { kind: "mac-x64", sub: "dlMacX64Sub" };
      if (os.arch === "arm") return { kind: "mac-arm", sub: "dlMacArmSub" };
      // Safari and Firefox do not tell Apple Silicon from Intel: let the visitor choose.
      return { kind: null, sub: "dlMacAnySub", choose: true };
    }
    return { kind: "linux-appimage", sub: "dlLinuxSub" };
  }

  function renderDownloads() {
    const main = $("#dl-main"), label = $("#dl-main-label"), sub = $("#dl-main-sub");
    if (!main) return;
    const choice = primaryChoice();
    delete main.dataset.choose;
    if (os.mobile) {
      label.textContent = t("dlMobile"); sub.textContent = t("dlMobileSub"); main.href = release && release.url || LATEST;
    } else if (choice) {
      label.textContent = t("dlFor", { os: PLATFORM_NAMES[os.name] });
      sub.textContent = t(choice.sub);
      main.href = choice.kind ? assetUrl(choice.kind) : (release && release.url) || LATEST;
      if (choice.choose) main.dataset.choose = "mac"; else delete main.dataset.choose;
    } else {
      label.textContent = lang === "en" ? I18N.en["dl.generic"] : original.get(label) || label.textContent;
      sub.textContent = t("dlMobileSub"); main.href = release && release.url || LATEST;
    }
    for (const a of $$("[data-asset]")) {
      a.href = assetUrl(a.dataset.asset);
      const known = release && release.assets && release.assets[a.dataset.asset];
      if (release && !release.none && !known) a.setAttribute("data-missing", ""); else a.removeAttribute("data-missing");
    }
    const rel = $("#dl-release");
    if (rel) {
      if (release && release.none) rel.textContent = t("noRelease");
      else if (release && release.tag) {
        let date = "";
        try { if (release.date) date = new Date(release.date).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-GB", { day: "numeric", month: "short", year: "numeric" }); } catch (e) { /* ignored */ }
        rel.textContent = date ? t("release", { tag: release.tag, date }) : release.tag;
      } else rel.textContent = "";
    }
  }

  // ---------- live network ----------
  let stats = null, health = null, lastOk = 0, lastTry = 0, failed = false, pollTimer = 0, inflight = false;
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : 0);

  async function getJson(path) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 6000);
    try {
      const r = await fetch(path, { cache: "no-store", credentials: "same-origin", signal: ctl.signal });
      if (!r.ok) throw new Error(String(r.status));
      return await r.json();
    } finally { clearTimeout(timer); }
  }

  async function poll() {
    clearTimeout(pollTimer);
    if (inflight) return;
    inflight = true;
    lastTry = Date.now();
    try {
      const [s, h] = await Promise.all([getJson("/v1/stats"), health ? Promise.resolve(health) : getJson("/v1/health").catch(() => null)]);
      if (!s || typeof s !== "object") throw new Error("bad stats");
      stats = s; health = h; failed = false; lastOk = Date.now();
    } catch (e) {
      failed = true;
    }
    inflight = false;
    renderStats();
    if (!document.hidden) pollTimer = setTimeout(poll, POLL_MS);
  }
  document.addEventListener("visibilitychange", () => {
    clearTimeout(pollTimer);
    if (document.hidden || inflight) return; // a running poll re-arms itself when it ends
    const since = Date.now() - lastTry;
    if (since > POLL_MS / 2) poll();
    else pollTimer = setTimeout(poll, POLL_MS - since);
  });

  function setStat(id, text) {
    const el = document.getElementById(id);
    if (!el || el.textContent === text) return;
    el.textContent = text;
    const v = el.closest(".stat-v") || el;
    v.classList.remove("bump"); void v.offsetWidth; v.classList.add("bump");
  }

  function renderStats() {
    const box = $("#net");
    if (!box) return;
    const timeStr = (ms) => new Date(ms).toLocaleTimeString(lang === "fr" ? "fr-FR" : "en-GB");
    const pill = $("#live-pill"), pillText = $("#live-pill-text");
    if (failed && !stats) {
      box.dataset.state = "error";
      $("#net-error").hidden = false; $("#net-empty").hidden = true;
      $("#net-status-text").textContent = t("statusErr", { time: timeStr(lastTry), s: POLL_MS / 1000 });
      if (pill) pill.hidden = true;
      return;
    }
    if (!stats) return; // still loading
    const serving = num(stats.nodes_serving), online = num(stats.nodes_online);
    const fams = stats.families && typeof stats.families === "object" ? Object.entries(stats.families) : [];
    const empty = serving === 0;
    box.dataset.state = failed ? "error" : empty ? "empty" : "ok";
    $("#net-error").hidden = !failed;
    $("#net-empty").hidden = !empty;
    setStat("s-peers", nf(serving));
    $("#s-peers-sub").textContent = online === 1 ? t("peersSub1") : t("peersSub", { n: nf(online) });
    setStat("s-families", nf(fams.length));
    $("#s-families-sub").textContent = fams.length ? fams.slice(0, 4).map(([f]) => f).join(" · ") : t("none");
    setStat("s-tps", nf(num(stats.tokens_per_s), num(stats.tokens_per_s) < 100 ? 1 : 0));
    setStat("s-jobs", nf(num(stats.jobs_per_min), num(stats.jobs_per_min) < 100 ? 1 : 0));
    $("#s-jobs-sub").textContent = t("jobsSub", { n: nf(num(stats.requesters_per_min), 1) });
    const chips = $("#s-chips");
    const sig = JSON.stringify(fams);
    if (chips.dataset.sig !== sig) {
      chips.dataset.sig = sig;
      chips.textContent = "";
      for (const [fam, count] of fams.slice(0, 16)) {
        const c = document.createElement("span");
        c.className = "chip";
        const dot = document.createElement("i");
        dot.style.background = FAMILY_COLORS[fam] || "#9aa3b8"; // CSSOM, allowed by the CSP
        c.append(dot, document.createTextNode(`${String(fam).slice(0, 40)} · ${nf(num(count))}`));
        chips.append(c);
      }
    }
    const proto = health && typeof health.protocol === "string"
      ? `${health.protocol}${health.protocol_version ? ` (${health.protocol_version})` : ""}` : "essaim/1";
    $("#net-status-text").textContent = failed
      ? t("statusErr", { time: timeStr(lastTry), s: POLL_MS / 1000 })
      : t("statusOk", { proto, time: timeStr(lastOk) });
    if (pill) {
      pill.hidden = false;
      pill.classList.toggle("quiet", empty);
      pillText.textContent = empty ? t("pill0") : serving === 1 ? t("pill1") : t("pill", { n: nf(serving) });
    }
  }

  // ---------- screenshots ----------
  function initGallery() {
    const dlg = $("#lightbox"), img = $("#lightbox-img");
    if (!dlg || typeof dlg.showModal !== "function") return; // the images stay plain images
    let opener = null;
    for (const b of $$(".shot-btn")) {
      b.addEventListener("click", () => {
        const src = b.dataset.full;
        if (!src || !src.startsWith("/static/landing/img/")) return;
        const thumb = $("img", b);
        img.src = src; img.alt = thumb ? thumb.alt : "";
        opener = b;
        dlg.showModal();
      });
    }
    $("#lightbox-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    dlg.addEventListener("close", () => { if (opener) opener.focus(); });
  }

  // ---------- scroll effects ----------
  function initReveal() {
    const header = $(".top");
    const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const fill = (root) => {
      for (const f of $$(".bar-f", root)) f.style.width = `${Math.max(0, Math.min(100, parseFloat(f.dataset.w) || 0))}%`;
      const minis = $$(".mini-f", root);
      const max = Math.max(1, ...minis.map((m) => parseFloat(m.dataset.h) || 0));
      for (const m of minis) m.style.height = `${(100 * (parseFloat(m.dataset.h) || 0) / max) * 0.6}%`;
    };
    const targets = $$(".sec-head, .card, .shot, .faq details");
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      fill(document);
      for (const el of targets) el.classList.add("in");
      return;
    }
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("in");
        fill(e.target);
        io.unobserve(e.target);
      }
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    for (const el of targets) {
      const r = el.getBoundingClientRect();
      if (r.top > window.innerHeight) el.classList.add("reveal"); // only what is below the fold animates
      io.observe(el);
    }
  }

  // ---------- start ----------
  function init() {
    for (const b of $$("[data-lang]", $(".seg"))) {
      b.addEventListener("click", () => {
        if (lang === b.dataset.lang) return;
        lang = b.dataset.lang;
        store.set("myriad.site.lang", lang);
        applyLang();
      });
    }
    $("#theme").addEventListener("click", () => {
      document.documentElement.setAttribute("data-theme", theme() === "dark" ? "light" : "dark");
      applyTheme(true);
    });
    $("#dl-main").addEventListener("click", (e) => {
      if ($("#dl-main").dataset.choose !== "mac") return;
      e.preventDefault(); // Mac of unknown architecture: open the list on the two .dmg files
      $("#platforms").open = true;
      const first = $('[data-asset="mac-arm"]');
      if (first) first.focus();
    });
    detectOs();
    applyLang();
    initGallery();
    initReveal();
    if (window.MyriadSwarm) window.MyriadSwarm.start($("#swarm"));
    poll();
    loadRelease();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
