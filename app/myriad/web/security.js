"use strict";
// Protection settings and pre-send safeguards. Self-contained on purpose: every decision is made by the
// local JSON API (GET /api/security, POST /api/security/*), this module only shows it, with the classes
// the rest of the interface already uses, so that a restyling of the interface keeps working.
//  - a « Sécurité et confidentialité » card in the About view (filled into #sec-card);
//  - « Bloquer ce nœud » / « Signaler » on peer cards (chat) and peer rows (Peers view): elements that
//    carry data-node="<node id>";
//  - before a question leaves: a sensitivity indicator under the composer, and a confirmation dialog when
//    the guard asks for one (send as is, mask, keep local); app.js adds MyriadSecurity.take() to its body.
(function () {
  const I = window.I18N;
  const t = I.t;
  const TOKEN = document.querySelector('meta[name="myriad-token"]').content;
  I.extend("fr", {
    "sec.title": "Sécurité et confidentialité",
    "sec.lead": "Le traqueur ne voit que des métadonnées (qui paie, quand, quelles tailles). Le pair qui calcule lit forcément la question : rien ne l'empêche de la copier. Masquez ce qui est sensible ou gardez-le en local.",
    "sec.e2e": "Exiger le chiffrement de bout en bout", "sec.local": "Mode local : ne rien envoyer au réseau",
    "sec.trusted_only": "Seulement mes nœuds de confiance", "sec.quarantine": "Quarantaine automatique des pairs douteux",
    "sec.rotate": "Changer de pairs à chaque question", "sec.min_rel": "Fiabilité minimale des modèles (%)",
    "sec.history": "Tours d'historique envoyés (0 : tous)", "sec.privacy": "Données sensibles détectées avant l'envoi",
    "sec.mode.mask": "masquer", "sec.mode.warn": "prévenir", "sec.mode.local": "toujours local", "sec.mode.off": "ignorer",
    "sec.type.private_key": "Clés privées", "sec.type.api_key": "Clés d'API, jetons, mots de passe", "sec.type.card": "Cartes bancaires",
    "sec.type.iban": "IBAN", "sec.type.email": "Adresses e-mail", "sec.type.phone": "Téléphones", "sec.type.path": "Chemins et noms d'utilisateur",
    "sec.type.name": "Noms de personnes", "sec.type.custom": "Mes termes", "sec.type.other": "Autres",
    "sec.terms": "Mes termes sensibles (un par ligne)", "sec.blocked": "Nœuds bloqués", "sec.families": "Familles bloquées",
    "sec.trusted": "Nœuds de confiance", "sec.quarantined": "En quarantaine", "sec.none": "aucun",
    "sec.unblock": "Débloquer", "sec.untrust": "Retirer", "sec.lift": "Lever", "sec.add": "Ajouter",
    "sec.invite": "Mon code d'invitation", "sec.invite_ph": "Code d'invitation ou identifiant de nœud",
    "sec.family_ph": "Famille (qwen, gemma…)", "sec.swarm": "Essaim privé", "sec.swarm_on": "actif ({g}…)", "sec.swarm_off": "inactif",
    "sec.swarm_ph": "Secret partagé (12 caractères au moins)", "sec.swarm_set": "Rejoindre", "sec.swarm_clear": "Quitter",
    "sec.swarm_gen": "Générer un secret", "sec.serve": "Limites quand je sers", "sec.rate": "Jobs par demandeur et par minute",
    "sec.conc": "Jobs simultanés par demandeur", "sec.prompt": "Taille maximale d'une question (jetons)",
    "sec.save": "Enregistrer", "sec.saved": "Enregistré.", "sec.block": "Bloquer ce nœud", "sec.report": "Signaler",
    "sec.blocked_ok": "Nœud bloqué : il ne recevra plus vos questions et vos jobs ne lui seront plus servis.",
    "sec.reported_ok": "Signalement envoyé au traqueur.", "sec.reason": "Raison (facultative)",
    "sec.tracker_no_e2e": "Ce traqueur ne prend pas en charge le chiffrement de bout en bout.",
    "sec.ind.none": "Rien de sensible détecté.", "sec.ind.mask": "Sera masqué avant l'envoi : {m}.",
    "sec.ind.ask": "Contenu sensible détecté : {d}. Une confirmation sera demandée.", "sec.ind.local": "Restera sur cette machine ({d}).",
    "sec.ind.readers": "Lu par : {r}.", "sec.read.e2e": "le pair qui calcule (chiffré de bout en bout ; le traqueur ne lit pas)",
    "sec.read.plain": "le pair qui calcule et, sans chiffrement, le traqueur", "sec.read.local": "personne d'autre que cette machine",
    "sec.dlg.title": "Cette question contient des données sensibles", "sec.dlg.body": "Détecté : {d}. Le pair qui calcule la lira en clair.",
    "sec.dlg.send": "Envoyer tel quel", "sec.dlg.always": "Ne plus demander pour ces types", "sec.dlg.local": "Garder en local",
    "sec.dlg.cancel": "Annuler",
  });
  I.extend("en", {
    "sec.title": "Security and privacy",
    "sec.lead": "The tracker only sees metadata (who pays, when, which sizes). The peer that computes necessarily reads the question: nothing prevents it from copying it. Mask what is sensitive or keep it local.",
    "sec.e2e": "Require end-to-end encryption", "sec.local": "Local mode: send nothing to the network",
    "sec.trusted_only": "Only my trusted nodes", "sec.quarantine": "Automatic quarantine of dubious peers",
    "sec.rotate": "Change peers at every question", "sec.min_rel": "Minimum model reliability (%)",
    "sec.history": "History turns sent (0: all)", "sec.privacy": "Sensitive data detected before sending",
    "sec.mode.mask": "mask", "sec.mode.warn": "warn", "sec.mode.local": "always local", "sec.mode.off": "ignore",
    "sec.type.private_key": "Private keys", "sec.type.api_key": "API keys, tokens, passwords", "sec.type.card": "Card numbers",
    "sec.type.iban": "IBAN", "sec.type.email": "E-mail addresses", "sec.type.phone": "Phone numbers", "sec.type.path": "Paths and user names",
    "sec.type.name": "People's names", "sec.type.custom": "My terms", "sec.type.other": "Other",
    "sec.terms": "My sensitive terms (one per line)", "sec.blocked": "Blocked nodes", "sec.families": "Blocked families",
    "sec.trusted": "Trusted nodes", "sec.quarantined": "Quarantined", "sec.none": "none",
    "sec.unblock": "Unblock", "sec.untrust": "Remove", "sec.lift": "Lift", "sec.add": "Add",
    "sec.invite": "My invite code", "sec.invite_ph": "Invite code or node id",
    "sec.family_ph": "Family (qwen, gemma…)", "sec.swarm": "Private swarm", "sec.swarm_on": "on ({g}…)", "sec.swarm_off": "off",
    "sec.swarm_ph": "Shared secret (12 characters at least)", "sec.swarm_set": "Join", "sec.swarm_clear": "Leave",
    "sec.swarm_gen": "Generate a secret", "sec.serve": "Limits when I serve", "sec.rate": "Jobs per requester and minute",
    "sec.conc": "Concurrent jobs per requester", "sec.prompt": "Largest question (tokens)",
    "sec.save": "Save", "sec.saved": "Saved.", "sec.block": "Block this node", "sec.report": "Report",
    "sec.blocked_ok": "Node blocked: it will no longer get your questions nor be served by you.",
    "sec.reported_ok": "Report sent to the tracker.", "sec.reason": "Reason (optional)",
    "sec.tracker_no_e2e": "This tracker does not support end-to-end encryption.",
    "sec.ind.none": "Nothing sensitive detected.", "sec.ind.mask": "Masked before sending: {m}.",
    "sec.ind.ask": "Sensitive content detected: {d}. You will be asked to confirm.", "sec.ind.local": "Will stay on this machine ({d}).",
    "sec.ind.readers": "Read by: {r}.", "sec.read.e2e": "the peer that computes (end-to-end encrypted; the tracker cannot read)",
    "sec.read.plain": "the peer that computes and, without encryption, the tracker", "sec.read.local": "nobody but this machine",
    "sec.dlg.title": "This question contains sensitive data", "sec.dlg.body": "Detected: {d}. The peer that computes will read it in clear.",
    "sec.dlg.send": "Send as is", "sec.dlg.always": "Do not ask again for these types", "sec.dlg.local": "Keep it local",
    "sec.dlg.cancel": "Cancel",
  });

  const $ = (id) => document.getElementById(id);
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
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
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    return j;
  }
  function toast(msg) {
    const box = $("toast");
    if (!box) return;
    box.textContent = msg; box.hidden = false;
    clearTimeout(toast.h); toast.h = setTimeout(() => { box.hidden = true; }, 3500);
  }
  const TYPES = ["private_key", "api_key", "card", "iban", "email", "phone", "path", "name", "custom"];
  const typeList = (o) => Object.entries(o || {}).map(([k, n]) => `${t(`sec.type.${k}`).toLowerCase()} (${n})`).join(", ");

  // ------------------------------------------------------------------ settings card
  let S = null;
  async function load(quiet) {
    const card = $("sec-card");
    if (quiet && card && card.contains(document.activeElement)) return;  // never under the user's fingers
    try { S = await api("/api/security"); render(); } catch (e) { /* the node starts: try again later */ }
  }
  function sw(key, label) {
    const box = el("input", { type: "checkbox", class: "switch", role: "switch" });
    box.checked = !!S[key];
    box.addEventListener("change", () => save({ [key]: box.checked }));
    return el("label", { class: "switch-row" }, el("span", { text: t(label) }), box);
  }
  function num(key, label, min, max, scale) {
    const inp = el("input", { type: "number", min: String(min), max: String(max), value: String(Math.round((S[key] || 0) * (scale || 1))) });
    inp.addEventListener("change", () => save({ [key]: Number(inp.value) / (scale || 1) }));
    return el("label", { class: "field" }, el("span", { text: t(label) }), inp);
  }
  function list(title, items, label, action, key) {
    const ul = el("ul", { class: "activity" });
    if (!items.length) ul.append(el("li", { class: "empty", text: t("sec.none") }));
    for (const x of items) {
      ul.append(el("li", {}, el("span", { class: "mono", text: (x.node_id || x.family || "").slice(0, 16) }),
        el("span", { class: "hint", text: ` ${x.label || x.reason || ""}` }),
        el("button", { type: "button", class: "btn ghost small", text: t(label),
          onclick: () => post(action, { [key]: x.node_id || x.family }) })));
    }
    return el("div", {}, el("h3", { class: "sec-title", text: t(title) }), ul);
  }
  function adder(ph, action, key) {
    const inp = el("input", { type: "text", placeholder: t(ph), spellcheck: "false" });
    return el("div", { class: "form-foot" }, inp, el("button", { type: "button", class: "btn small", text: t("sec.add"),
      onclick: () => { if (inp.value.trim()) post(action, { [key]: inp.value.trim() }); } }));
  }
  async function save(body) {
    try { S = (await api("/api/security/settings", body)).security; render(); toast(t("sec.saved")); }
    catch (e) { toast(t("err.prefix", { m: e.message })); }
  }
  async function post(action, body) {
    try { const r = await api(`/api/security/${action}`, body); S = r.security || S; render(); return r; }
    catch (e) { toast(t("err.prefix", { m: e.message })); return null; }
  }
  function render() {
    const card = $("sec-card");
    if (!card || !S) return;
    const modes = (S.privacy && S.privacy.modes) || {};
    const privacy = el("div", {}, el("h3", { class: "sec-title", text: t("sec.privacy") }));
    for (const ty of TYPES) {
      const sel = el("select", {});
      for (const m of ["mask", "warn", "local", "off"]) {
        const o = el("option", { value: m, text: t(`sec.mode.${m}`) });
        if (modes[ty] === m) o.selected = true;
        sel.append(o);
      }
      sel.addEventListener("change", () => save({ privacy: { modes: { [ty]: sel.value } } }));
      privacy.append(el("label", { class: "field" }, el("span", { text: t(`sec.type.${ty}`) }), sel));
    }
    const terms = el("textarea", { rows: "2", spellcheck: "false" });
    terms.value = ((S.privacy && S.privacy.terms) || []).join("\n");
    terms.addEventListener("change", () => save({ privacy: { terms: terms.value.split("\n").map((x) => x.trim()).filter(Boolean) } }));
    privacy.append(el("label", { class: "field" }, el("span", { text: t("sec.terms") }), terms));
    const secret = el("input", { type: "password", placeholder: t("sec.swarm_ph"), autocomplete: "off" });
    const swarm = el("div", {}, el("h3", { class: "sec-title", text: t("sec.swarm") }),
      el("p", { class: "hint", text: S.swarm.enabled ? t("sec.swarm_on", { g: S.swarm.group }) : t("sec.swarm_off") }),
      el("div", { class: "form-foot" }, secret,
        el("button", { type: "button", class: "btn small", text: t("sec.swarm_gen"), onclick: () => {
          const b = new Uint8Array(24); crypto.getRandomValues(b);
          secret.type = "text"; secret.value = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join(""); } }),
        el("button", { type: "button", class: "btn small", text: t("sec.swarm_set"), onclick: () => post("swarm", { secret: secret.value }) }),
        S.swarm.enabled ? el("button", { type: "button", class: "btn ghost small", text: t("sec.swarm_clear"), onclick: () => post("swarm", { secret: null }) }) : null));
    card.replaceChildren(...[
      el("h2", { class: "sec-title", id: "sec-heading", text: t("sec.title") }),
      el("p", { class: "hint", text: t("sec.lead") }),
      S.tracker_features && S.tracker_features.length && !S.tracker_e2e ? el("p", { class: "error", text: t("sec.tracker_no_e2e") }) : null,
      sw("require_e2e", "sec.e2e"), sw("local_only", "sec.local"), sw("trusted_only", "sec.trusted_only"),
      sw("quarantine", "sec.quarantine"), sw("rotate_peers", "sec.rotate"),
      num("min_reliability", "sec.min_rel", 0, 100, 100), num("max_history_turns", "sec.history", 0, 1000),
      privacy,
      list("sec.blocked", S.blocked || [], "sec.unblock", "unblock", "node_id"),
      list("sec.families", S.blocked_families || [], "sec.unblock", "unblock_family", "family"),
      adder("sec.family_ph", "block_family", "family"),
      list("sec.trusted", S.trusted || [], "sec.untrust", "untrust", "node_id"),
      adder("sec.invite_ph", "trust", "node_id"),
      el("p", { class: "hint" }, `${t("sec.invite")} : `, el("code", { class: "mono", text: S.invite || "–" })),
      list("sec.quarantined", S.quarantined || [], "sec.lift", "quarantine/lift", "node_id"),
      swarm,
      el("h3", { class: "sec-title", text: t("sec.serve") }),
      num("rate_per_min", "sec.rate", 0, 10000), num("max_concurrent", "sec.conc", 0, 64), num("max_prompt_tokens", "sec.prompt", 1, 1000000),
    ].filter(Boolean));  // (replaceChildren would show a null as the text "null")
  }

  // ------------------------------------------------------------------ « Bloquer ce nœud » on peer cards
  function decorate(root) {
    root.querySelectorAll("[data-node]").forEach((e) => {
      const nid = e.dataset.node;
      if (!nid || e.querySelector(":scope .sec-actions")) return;
      if (S && S.node_id === nid) return;  // not on our own node
      const where = e.tagName === "TR" ? e.lastElementChild : e;
      if (!where) return;
      where.append(el("span", { class: "sec-actions" },
        el("button", { type: "button", class: "btn ghost small", text: t("sec.block"), onclick: async (ev) => {
          ev.stopPropagation();
          const reason = window.prompt(t("sec.reason"), "") ?? null;
          if (reason === null) return;
          if (await post("block", { node_id: nid, reason })) toast(t("sec.blocked_ok"));
        } }),
        el("button", { type: "button", class: "btn ghost small", text: t("sec.report"), onclick: async (ev) => {
          ev.stopPropagation();
          const note = window.prompt(t("sec.reason"), "");
          if (note === null) return;
          if (await post("report", { node_id: nid, reason: "other", note })) toast(t("sec.reported_ok"));
        } })));
    });
  }
  new MutationObserver(() => decorate(document)).observe(document.body, { childList: true, subtree: true });

  // ------------------------------------------------------------------ before a question leaves
  let pending = {};
  let bypass = false;
  const form = $("chat"), question = $("question");
  const indicator = el("p", { class: "hint", id: "sec-indicator", role: "status", "aria-live": "polite" });
  if (form) form.after(indicator);
  async function check(text) {
    return api("/api/security/check", { text });
  }
  function describe(c) {
    if (c.decision === "local") return t("sec.ind.local", { d: typeList(c.detected) });
    if (!Object.keys(c.detected || {}).length) return t("sec.ind.none");
    const parts = [];
    if (Object.keys(c.masked || {}).length) parts.push(t("sec.ind.mask", { m: typeList(c.masked) }));
    if (c.decision === "ask") parts.push(t("sec.ind.ask", { d: typeList(c.unmasked) }));
    const r = c.readers || {};
    parts.push(t("sec.ind.readers", { r: r.local_only ? t("sec.read.local") : r.e2e_required ? t("sec.read.e2e") : t("sec.read.plain") }));
    return parts.join(" ");
  }
  let timer = null;
  if (question) question.addEventListener("input", () => {
    clearTimeout(timer);
    const text = question.value;
    if (!text.trim()) { indicator.textContent = ""; return; }
    timer = setTimeout(async () => { try { indicator.textContent = describe(await check(text)); } catch (e) { indicator.textContent = ""; } }, 500);
  });
  function ask(c) {
    return new Promise((resolve) => {
      const always = el("input", { type: "checkbox", class: "switch", role: "switch" });
      const dlg = el("dialog", { class: "about" });
      const done = (v) => { dlg.close(); dlg.remove(); resolve(v); };
      dlg.append(el("h2", { class: "sec-title", text: t("sec.dlg.title") }),
        el("p", { text: t("sec.dlg.body", { d: typeList(c.unmasked) }) }),
        el("p", { class: "hint", text: Object.keys(c.masked || {}).length ? t("sec.ind.mask", { m: typeList(c.masked) }) : "" }),
        el("label", { class: "switch-row" }, el("span", { text: t("sec.dlg.always") }), always),
        el("div", { class: "form-foot" },
          el("button", { type: "button", class: "btn ghost", text: t("sec.dlg.cancel"), onclick: () => done(null) }),
          el("button", { type: "button", class: "btn", text: t("sec.dlg.local"), onclick: () => done({ local: true }) }),
          el("button", { type: "button", class: "btn primary", text: t("sec.dlg.send"), onclick: async () => {
            if (always.checked) await post("confirm", { types: c.confirm_types || [] });
            done({ confirm: true });
          } })));
      dlg.addEventListener("cancel", () => done(null));
      document.body.append(dlg);
      dlg.showModal();
    });
  }
  if (form) form.addEventListener("submit", async (ev) => {
    if (bypass) { bypass = false; return; }
    const text = question ? question.value.trim() : "";
    if (!text) return;
    ev.preventDefault(); ev.stopImmediatePropagation();
    let c = null;
    try { c = await check(text); } catch (e) { c = null; }
    pending = {};
    if (c && c.decision === "ask") {
      const choice = await ask(c);
      if (!choice) return;
      pending = choice;
    }
    if (question && question.value.trim() !== text) {
      // edited meanwhile: a consent given for the checked text must not cover another one (Codex review)
      pending = {};
      form.requestSubmit();
      return;
    }
    bypass = true;
    indicator.textContent = "";
    form.requestSubmit();
  }, true);

  window.MyriadSecurity = {
    take() { const p = pending; pending = {}; return p; },
    reload: load,
  };
  document.addEventListener("langchange", render);
  load();
  setInterval(() => load(true), 15000);
})();
