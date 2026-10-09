"use strict";
// A minimal fake DOM to run the interface scripts (myriad/web/*.js) under Node in the tests: enough
// for them to load, render into plain objects, and react to events. No layout, no real canvas.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const WEB = path.join(__dirname, "..", "..", "myriad", "web");

class ClassList {
  constructor(el) { this.el = el; }
  _set() { return new Set((this.el.className || "").split(/\s+/).filter(Boolean)); }
  _save(s) { this.el.className = [...s].join(" "); }
  add(...c) { const s = this._set(); c.forEach((x) => s.add(x)); this._save(s); }
  remove(...c) { const s = this._set(); c.forEach((x) => s.delete(x)); this._save(s); }
  toggle(c, force) {
    const s = this._set(); const on = force === undefined ? !s.has(c) : !!force;
    if (on) s.add(c); else s.delete(c);
    this._save(s); return on;
  }
  contains(c) { return this._set().has(c); }
}

// A 2D context (and gradient) where every method is a no-op.
const ctx2d = new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : () => ctx2d),
  set: (t, k, v) => { t[k] = v; return true; },
});

class El {
  constructor(tag, id) {
    this.tagName = String(tag).toUpperCase(); this.id = id || "";
    this.children = []; this.attrs = {}; this.dataset = {}; this.listeners = {};
    this.className = ""; this.hidden = false; this._text = ""; this.value = ""; this.checked = false;
    this.disabled = false; this.open = false; this.parentNode = null; this.content = "tok";
    this.offsetWidth = 100; this.offsetHeight = 100; this.scrollHeight = 10; this.scrollLeft = 0; this.scrollTop = 0;
    this.width = 0; this.height = 0; this.offsetParent = {}; this.tabIndex = 0;
    const style = {}; style.setProperty = (k, v) => { style[k] = v; }; style.removeProperty = (k) => { delete style[k]; };
    this.style = style;
    this.classList = new ClassList(this);
  }
  get textContent() { return this._text + this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get firstChild() { return this.children[0] || null; }
  append(...kids) {
    for (const c of kids) {
      if (c === null || c === undefined) continue;
      if (typeof c === "object") { c.parentNode = this; this.children.push(c); } else this.children.push(String(c));
    }
  }
  replaceChildren(...kids) { this.children = []; this._text = ""; this.append(...kids); }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((c) => c !== this); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener() {}
  fire(type, ev) { return (this.listeners[type] || []).map((f) => f(ev || { preventDefault() {}, target: this })); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "hidden") this.hidden = true; }  // reflected, like the DOM
  getAttribute(k) { return this.attrs[k]; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  getContext() { return ctx2d; }
  getBoundingClientRect() { return { left: 0, top: 0, right: 400, bottom: 300, width: 400, height: 300 }; }
  focus() {} select() {} blur() {}
  requestSubmit() { this.fire("submit"); }
  // every descendant element (depth first)
  all() { const out = []; for (const c of this.children) if (typeof c === "object") { out.push(c, ...c.all()); } return out; }
}

function load(scripts, opts) {
  const o = opts || {};
  const ids = new Map();
  const rafs = [];
  const documentElement = new El("html");
  const docListeners = {};
  const document = {
    documentElement,
    getElementById(id) { if (!ids.has(id)) ids.set(id, new El("div", id)); return ids.get(id); },
    querySelector() { return new El("meta"); },
    querySelectorAll() { return []; },
    createElement: (tag) => new El(tag),
    createElementNS: (ns, tag) => new El(tag),
    createTextNode: (text) => { const e = new El("#text"); e.textContent = text; return e; },
    addEventListener(type, fn) { (docListeners[type] = docListeners[type] || []).push(fn); },
    dispatchEvent(ev) { (docListeners[ev.type] || []).forEach((f) => f(ev)); },
    get hidden() { return false; },
  };
  const store = {};
  const window = {
    document,
    console,
    performance,
    TextDecoder,
    TextEncoder,
    setTimeout: (f, ms) => { const h = setTimeout(f, ms); if (h.unref) h.unref(); return h; },
    clearTimeout,
    setInterval: () => 0,
    clearInterval: () => {},
    requestAnimationFrame: (f) => { rafs.push(f); return rafs.length; },
    cancelAnimationFrame: () => {},
    matchMedia: () => ({ matches: false }),
    getComputedStyle: () => ({ getPropertyValue: () => "" }),
    localStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } },
    history: { replaceState() {} },
    location: { hash: "" },
    CSS: { escape: (s) => s },
    CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } },
    ResizeObserver: class { observe() {} },
    MutationObserver: class { observe() {} },
    addEventListener() {},
    scrollTo() {},
    devicePixelRatio: 1,
    fetch: o.fetch || (async () => ({ ok: false, status: 503, json: async () => ({}) })),
  };
  window.window = window;
  vm.createContext(window);
  for (const s of scripts) vm.runInContext(fs.readFileSync(path.join(WEB, s), "utf8"), window, { filename: s });
  return {
    window, document, $: (id) => document.getElementById(id),
    // run the animation frames queued so far (once each)
    flushFrames() { const fs_ = rafs.splice(0); for (const f of fs_) f(performance.now() + 16); },
    tick: (ms) => new Promise((r) => setTimeout(r, ms || 0)),
  };
}

function json(body, status) {
  return { ok: (status || 200) < 400, status: status || 200, json: async () => body, body: null };
}

// A fetch Response whose body streams the given server-sent events, then ends.
function sseResponse(events) {
  const enc = new TextEncoder();
  const chunks = events.map((e) => enc.encode(`data: ${JSON.stringify(e)}\n\n`));
  return { ok: true, status: 200, json: async () => ({}),
           body: { getReader() { return { read: async () => (chunks.length ? { value: chunks.shift(), done: false } : { value: undefined, done: true }) }; } } };
}

module.exports = { load, json, sseResponse, El };
