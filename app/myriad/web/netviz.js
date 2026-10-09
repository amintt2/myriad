"use strict";
// Live network map on a <canvas>: your node at the centre (drawn as the logo, a small spiral of dots), peers
// around it grouped by model family (one colour per family), sparks travelling along the links when jobs
// flow. Calm by design: thin lines, crisp dots, slow motion, none at all with prefers-reduced-motion.
// No dependency.
(function () {
  // One colour per model family, a darker variant for the light theme (dots need ~3:1 on paper).
  const FAMILY_COLORS = {
    qwen: "#9a8cff", gemma: "#4cc0ee", granite: "#40c994", smollm: "#f0b44f", mistral: "#f08659",
    phi: "#ee6fa6", llama: "#a6d65e", olmo: "#e8cd55", deepseek: "#6f95ff", falcon: "#c79be8", exaone: "#55cbd9",
  };
  const FAMILY_COLORS_LIGHT = {
    qwen: "#6b58e0", gemma: "#1786b8", granite: "#16865a", smollm: "#b0740d", mistral: "#c85524",
    phi: "#c93f7a", llama: "#5d8c1c", olmo: "#987f0c", deepseek: "#3461d6", falcon: "#8a55b8", exaone: "#13879a",
  };
  // the logo's gradient, core to rim
  const LOGO = [[0, [255, 191, 121]], [0.3, [255, 120, 152]], [0.65, [140, 108, 244]], [1, [70, 190, 255]]];
  function hash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967295;
  }
  const isLight = () => document.documentElement.dataset.theme === "light";
  function familyColor(f) {
    const light = isLight();
    if (!f) return light ? "#7a776f" : "#9a978f";
    const m = light ? FAMILY_COLORS_LIGHT : FAMILY_COLORS;
    // own properties only: a peer may announce "constructor", "toString" or "__proto__" as its family
    if (Object.prototype.hasOwnProperty.call(m, f)) return m[f];
    return `hsl(${Math.round(hash(String(f)) * 360)}, ${light ? "55%, 42%" : "65%, 66%"})`;
  }
  function logoColor(u) {
    for (let k = 1; k < LOGO.length; k++) {
      if (u <= LOGO[k][0]) {
        const [a0, c0] = LOGO[k - 1], [a1, c1] = LOGO[k], v = (u - a0) / (a1 - a0);
        return c0.map((x, j) => Math.round(x + (c1[j] - x) * v));
      }
    }
    return LOGO[LOGO.length - 1][1];
  }
  const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

  function create(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const o = opts || {};
    let W = 0, H = 0, dpr = 1;
    let nodes = new Map(); // id -> {peer, ang, rad, glow, born}
    const me = { model: null, family: null, serving: false, busy: 0, accepting: true, glow: 0 };
    let pulses = [];
    let ripples = [];
    let ambientRate = 0, ambientAcc = 0;
    let theme = {};
    let rot = 0, last = performance.now(), hoverId = null, running = true;
    // the "you" mark: a phyllotaxis spiral like the logo
    const GA = Math.PI * (3 - Math.sqrt(5)), ME_N = 34;
    const meDots = Array.from({ length: ME_N }, (_, i) => {
      const f = Math.sqrt((i + 0.5) / ME_N), c = logoColor(f);
      return { a: i * GA, f, c: `rgb(${c.join(",")})` };
    });

    function readTheme() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
      theme = { ring: v("--viz-ring", "rgba(237,235,230,.06)"), text: v("--text-2", "#a3a09a"), bg: v("--bg", "#101113"),
                ink: v("--text", "#edebe6"), accent: v("--accent", "#ffb27a"), link: parseFloat(v("--viz-link", ".16")) || 0.16,
                light: isLight() };
    }
    function resize() {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.max(10, r.width); H = Math.max(10, r.height);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      layout();
    }
    const radius = () => Math.min(W, H) * 0.42;
    const stretch = () => Math.min(1.35, Math.max(1, W / H * 0.82));
    function layout() {
      const list = [...nodes.values()].sort((a, b) => (a.peer.family || "").localeCompare(b.peer.family || "")
        || a.id.localeCompare(b.id));
      const n = list.length;
      const fams = [...new Set(list.map((x) => x.peer.family || "?"))];
      const gap = n > 3 ? 1.2 : 0;
      const total = n + gap * fams.length || 1;
      const R = radius();
      const rings = n <= 14 ? 1 : n <= 40 ? 2 : 3;
      let slot = 0, prevFam = null;
      list.forEach((nd, i) => {
        const f = nd.peer.family || "?";
        if (f !== prevFam) { slot += gap; prevFam = f; }
        const ang = (slot / total) * Math.PI * 2 - Math.PI / 2;
        slot += 1;
        const ring = rings === 1 ? 0 : i % rings;
        const jitter = (hash(nd.id) - 0.5) * 0.1 * R;
        nd.ang = ang;
        nd.rad = R * (rings === 1 ? 0.9 : 0.58 + 0.38 * ring / (rings - 1)) + jitter;
      });
    }
    function pos(nd) {
      const a = nd.ang + rot;
      return [W / 2 + Math.cos(a) * nd.rad * stretch(), H / 2 + Math.sin(a) * nd.rad];
    }

    function setPeers(peers, meId) {
      const seen = new Set();
      for (const p of peers) {
        if (p.node_id === meId) continue;
        seen.add(p.node_id);
        const nd = nodes.get(p.node_id);
        if (nd) {
          if ((p.busy || 0) > (nd.peer.busy || 0)) pulse("me", p.node_id, null, true);
          nd.peer = p;
        } else {
          nodes.set(p.node_id, { id: p.node_id, peer: p, born: performance.now(), glow: 0 });
        }
      }
      for (const id of [...nodes.keys()]) if (!seen.has(id)) nodes.delete(id);
      layout();
      canvas.setAttribute("aria-label", (o.label || ((n) => `${n}`))(nodes.size));
    }
    function setMe(m) { Object.assign(me, m); }
    function pulse(from, to, color, ambient) {
      if (pulses.length > 60) return;
      const id = (k) => (k === "me" ? "me" : nodes.has(k) ? k : null);
      const a = id(from), b = id(to);
      if (!a || !b || a === b) return;
      // colour = the family of the peer at the far end (a job takes the colour of who serves it)
      const fam = (k) => (k === "me" ? me.family : nodes.get(k).peer.family);
      pulses.push({ from: a, to: b, t: 0, dur: (ambient ? 2.2 : 1.5) * (0.85 + Math.random() * 0.3),
                    fam: color ? null : fam(b === "me" ? a : b), color, bend: (Math.random() - 0.5) * 0.35 });
    }
    function randomPeer() {
      const ids = [...nodes.keys()];
      return ids.length ? ids[Math.floor(Math.random() * ids.length)] : null;
    }

    function step(now) {
      if (!running) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const still = reduceMotion();
      if (!still) rot += dt * 0.006;
      // ambient traffic between peers (jobs of the rest of the network), capped to stay calm
      ambientAcc += dt * Math.min(ambientRate, 2.5);
      while (ambientAcc >= 1) {
        ambientAcc -= 1;
        const a = randomPeer(), b = randomPeer();
        if (a && b && a !== b && !still) pulse(a, b, null, true);
      }
      draw(now, dt, still);
      requestAnimationFrame(step);
    }

    function draw(now, dt, still) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2, cy = H / 2, R = radius(), sx = stretch();
      // a faint warm floor under "you"
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.9);
      g.addColorStop(0, hexA(theme.accent, theme.light ? 0.10 : 0.07));
      g.addColorStop(1, hexA(theme.accent, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.ellipse(cx, cy, R * 0.9 * sx, R * 0.9, 0, 0, Math.PI * 2); ctx.fill();
      // orbit guides: two hairlines
      ctx.strokeStyle = theme.ring;
      ctx.lineWidth = 1;
      for (const k of [0.9, 0.45]) {
        ctx.beginPath(); ctx.ellipse(cx, cy, R * k * sx, R * k, 0, 0, Math.PI * 2); ctx.stroke();
      }
      // links to every peer, fading in from the centre
      const P = new Map();
      for (const nd of nodes.values()) P.set(nd.id, pos(nd));
      ctx.lineWidth = 1;
      for (const nd of nodes.values()) {
        const [x, y] = P.get(nd.id);
        const col = familyColor(nd.peer.family);
        const grad = ctx.createLinearGradient(cx, cy, x, y);
        grad.addColorStop(0.15, hexA(col, 0));
        grad.addColorStop(1, hexA(col, theme.link * (nd.peer.accepting ? 1 : 0.4) + (nd.glow || 0) * 0.25));
        ctx.strokeStyle = grad;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      }
      // sparks: a short fading tail and a bright head, eased along a gentle curve
      const arrived = [];
      for (const p of pulses) {
        p.t += dt / p.dur * (still ? 3 : 1);
        if (p.t >= 1) { arrived.push(p); continue; }
        const a = p.from === "me" ? [cx, cy] : P.get(p.from), b = p.to === "me" ? [cx, cy] : P.get(p.to);
        if (!a || !b) { p.t = 1; continue; }
        const col = p.color || familyColor(p.fam);
        const mx = (a[0] + b[0]) / 2 + (b[1] - a[1]) * p.bend, my = (a[1] + b[1]) / 2 - (b[0] - a[0]) * p.bend;
        const at = (t) => [(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * mx + t * t * b[0],
                           (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * my + t * t * b[1]];
        const e = ease(p.t), fade = Math.min(1, p.t * 6, (1 - p.t) * 6);
        const N = 10, len = 0.16;
        let prev = at(Math.max(0, e - len));
        ctx.lineCap = "round";
        for (let k = 1; k <= N; k++) {
          const pt = at(Math.max(0, e - len + (len * k) / N));
          ctx.strokeStyle = hexA(col, (k / N) * 0.75 * fade);
          ctx.lineWidth = 0.6 + (k / N) * 1.4;
          ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke();
          prev = pt;
        }
        const [hx, hy] = at(e);
        ctx.fillStyle = hexA(col, 0.22 * fade);
        ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = theme.light ? hexA(col, fade) : hexA("#ffffff", 0.95 * fade);
        ctx.beginPath(); ctx.arc(hx, hy, 1.8, 0, Math.PI * 2); ctx.fill();
      }
      if (arrived.length) {
        pulses = pulses.filter((p) => p.t < 1);
        for (const p of arrived) {
          ripples.push({ id: p.to, t: 0, color: p.color || familyColor(p.fam) });
          if (p.to === "me") me.glow = 1;
          else if (nodes.has(p.to)) nodes.get(p.to).glow = 1;
        }
      }
      // ripples: one thin ring
      ripples = ripples.filter((r) => (r.t += dt * 1.1) < 1);
      for (const r of ripples) {
        const c = r.id === "me" ? [cx, cy] : P.get(r.id);
        if (!c) continue;
        const u = ease(r.t);
        ctx.strokeStyle = hexA(r.color, 0.45 * (1 - u));
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(c[0], c[1], (r.id === "me" ? 26 : 7) + u * 16, 0, Math.PI * 2); ctx.stroke();
      }
      // peers: crisp dots, separated from their link by a ring of background
      for (const nd of nodes.values()) {
        const [x, y] = P.get(nd.id);
        const col = familyColor(nd.peer.family);
        const born = ease(Math.min(1, (now - nd.born) / 700));
        const size = (3.6 + Math.min(4, (nd.peer.max_parallel || 1)) * 0.55) * (0.3 + 0.7 * born);
        nd.glow = Math.max(0, (nd.glow || 0) - dt * 0.9);
        const busy = (nd.peer.busy || 0) > 0;
        if (busy) {  // a slow breathing halo while it works
          const b = still ? 0.5 : 0.5 + 0.5 * Math.sin(now / 700 + hash(nd.id) * 6);
          ctx.strokeStyle = hexA(col, 0.18 + 0.22 * b);
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x, y, size + 4 + b * 2, 0, Math.PI * 2); ctx.stroke();
        }
        if (nd.glow > 0) {
          ctx.fillStyle = hexA(col, 0.18 * nd.glow);
          ctx.beginPath(); ctx.arc(x, y, size + 6, 0, Math.PI * 2); ctx.fill();
        }
        ctx.fillStyle = theme.bg;
        ctx.beginPath(); ctx.arc(x, y, size + 2, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
        if (nd.peer.accepting) { ctx.fillStyle = col; ctx.fill(); }
        else { ctx.strokeStyle = hexA(col, 0.8); ctx.lineWidth = 1.25; ctx.beginPath(); ctx.arc(x, y, size - 0.6, 0, Math.PI * 2); ctx.stroke(); }
        if (hoverId === nd.id) {
          ctx.strokeStyle = theme.ink; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x, y, size + 5, 0, Math.PI * 2); ctx.stroke();
        }
      }
      // you: the logo, gently breathing; a thin accent arc turns while you serve the network
      me.glow = Math.max(0, me.glow - dt * 0.8);
      const breathe = still ? 1 : 1 + Math.sin(now / 1400) * 0.025;
      const rMe = 19 * breathe;
      ctx.fillStyle = theme.bg;
      ctx.beginPath(); ctx.arc(cx, cy, rMe + 6, 0, Math.PI * 2); ctx.fill();
      if (me.glow > 0) {
        ctx.fillStyle = hexA(theme.accent, 0.16 * me.glow);
        ctx.beginPath(); ctx.arc(cx, cy, rMe + 10, 0, Math.PI * 2); ctx.fill();
      }
      const spin = still ? 0 : now / 26000;
      for (const d of meDots) {
        const a = d.a + spin;
        ctx.fillStyle = d.c;
        ctx.beginPath(); ctx.arc(cx + Math.cos(a) * rMe * d.f, cy + Math.sin(a) * rMe * d.f, 0.7 + 2.1 * (1 - d.f), 0, Math.PI * 2); ctx.fill();
      }
      if (me.serving && me.accepting) {
        const a0 = still ? -Math.PI / 2 : now / 2400;
        const span = Math.PI * (me.busy > 0 ? 1.2 : 0.45);
        ctx.strokeStyle = hexA(theme.accent, 0.9); ctx.lineWidth = 1.5; ctx.lineCap = "round";
        ctx.beginPath(); ctx.arc(cx, cy, rMe + 8, a0, a0 + span); ctx.stroke();
      }
      ctx.fillStyle = theme.text;
      ctx.font = "500 12px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(o.youLabel ? o.youLabel() : "You", cx, cy + rMe + 28);
    }

    function hexA(c, a) {
      if (c.startsWith("#")) {
        const n = parseInt(c.slice(1), 16);
        return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
      }
      if (c.startsWith("hsl(")) return c.replace("hsl(", "hsla(").replace(")", `, ${a})`);
      if (c.startsWith("rgb(")) return c.replace("rgb(", "rgba(").replace(")", `, ${a})`);
      return c;
    }

    canvas.addEventListener("mousemove", (ev) => {
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      let best = null, bd = 16 * 16;
      for (const nd of nodes.values()) {
        const [px, py] = pos(nd);
        const d = (px - x) ** 2 + (py - y) ** 2;
        if (d < bd) { bd = d; best = nd; }
      }
      const meHit = (W / 2 - x) ** 2 + (H / 2 - y) ** 2 < 26 * 26;
      hoverId = best ? best.id : null;
      canvas.style.cursor = best || meHit ? "default" : "";
      if (o.onHover) o.onHover(best ? best.peer : meHit ? "me" : null, x, y);
    });
    canvas.addEventListener("mouseleave", () => { hoverId = null; if (o.onHover) o.onHover(null); });
    new ResizeObserver(resize).observe(canvas);
    document.addEventListener("visibilitychange", () => {
      running = !document.hidden;
      if (running) { last = performance.now(); requestAnimationFrame(step); }
    });
    // the theme switch changes the colours read from CSS
    new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    readTheme();
    resize();
    requestAnimationFrame(step);
    return {
      setPeers, setMe, pulse, readTheme,
      ambient(r) { ambientRate = Math.max(0, Math.min(r || 0, 6)); },
      randomPeer,
      peerIds: () => [...nodes.keys()],
      has: (id) => nodes.has(id),
    };
  }
  window.NetViz = { create, familyColor };
})();
