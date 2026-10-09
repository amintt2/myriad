"use strict";
// Live network map on a <canvas>: your node at the centre, peers around it grouped by model family
// (one colour per family), comets travelling along the links when jobs flow. No dependency.
(function () {
  const FAMILY_COLORS = {
    qwen: "#8b7bff", gemma: "#38c8ff", granite: "#35d49a", smollm: "#ffb547", mistral: "#ff7f50",
    phi: "#ff5fa2", llama: "#a3e05a", olmo: "#f2d14b", deepseek: "#5b8cff", falcon: "#c792ea", exaone: "#4dd0e1",
  };
  function hash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967295;
  }
  function familyColor(f) {
    if (!f) return "#9aa3b8";
    if (FAMILY_COLORS[f]) return FAMILY_COLORS[f];
    return `hsl(${Math.round(hash(f) * 360)}, 70%, 64%)`;
  }
  const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function create(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const o = opts || {};
    let W = 0, H = 0, dpr = 1;
    let nodes = new Map(); // id -> {peer, x, y, tx, ty, a, glow, ripple}
    let me = { model: null, family: null, serving: false, busy: 0, accepting: true, x: 0, y: 0, glow: 0 };
    let pulses = [];
    let ripples = [];
    let ambientRate = 0, ambientAcc = 0;
    let theme = {};
    let rot = 0, last = performance.now(), hoverId = null, running = true;

    function readTheme() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
      theme = { edge: v("--viz-edge", "rgba(140,150,190,.16)"), text: v("--text-2", "#aab"), bg: v("--surface", "#11131c"),
                me1: v("--accent-warm", "#ffd36e"), me2: v("--accent", "#7c6cff") };
    }
    function resize() {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.max(10, r.width); H = Math.max(10, r.height);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      layout();
    }
    function layout() {
      const list = [...nodes.values()].sort((a, b) => (a.peer.family || "").localeCompare(b.peer.family || "")
        || a.id.localeCompare(b.id));
      const n = list.length;
      const fams = [...new Set(list.map((x) => x.peer.family || "?"))];
      const gap = n > 3 ? 1.2 : 0;
      const total = n + gap * fams.length || 1;
      const R = Math.min(W, H) * 0.40;
      const rings = n <= 14 ? 1 : n <= 40 ? 2 : 3;
      let slot = 0, prevFam = null;
      list.forEach((nd, i) => {
        const f = nd.peer.family || "?";
        if (f !== prevFam) { slot += gap; prevFam = f; }
        const ang = (slot / total) * Math.PI * 2 - Math.PI / 2;
        slot += 1;
        const ring = rings === 1 ? 0 : i % rings;
        const jitter = (hash(nd.id) - 0.5) * 0.08 * R;
        const rad = R * (rings === 1 ? 1 : 0.62 + 0.38 * ring / (rings - 1)) + jitter;
        nd.ang = ang; nd.rad = rad;
        if (nd.x === undefined) { nd.x = W / 2; nd.y = H / 2; }
      });
    }
    function pos(nd) {
      const a = nd.ang + rot;
      return [W / 2 + Math.cos(a) * nd.rad * (W > H * 1.25 ? 1.25 : 1), H / 2 + Math.sin(a) * nd.rad];
    }

    function setPeers(peers, meId) {
      const seen = new Set();
      for (const p of peers) {
        if (p.node_id === meId) continue;
        seen.add(p.node_id);
        const nd = nodes.get(p.node_id);
        if (nd) {
          if ((p.busy || 0) > (nd.peer.busy || 0)) pulse("me", p.node_id, familyColor(p.family), true);
          nd.peer = p;
        } else {
          nodes.set(p.node_id, { id: p.node_id, peer: p, born: performance.now() });
        }
      }
      for (const id of [...nodes.keys()]) if (!seen.has(id)) nodes.delete(id);
      layout();
      canvas.setAttribute("aria-label", (o.label || ((n) => `${n}`))(nodes.size));
    }
    function setMe(m) { Object.assign(me, m); }
    function pulse(from, to, color, ambient) {
      if (pulses.length > 120) return;
      const id = (k) => (k === "me" ? "me" : nodes.has(k) ? k : null);
      const a = id(from), b = id(to);
      if (!a || !b || a === b) return;
      const fam = (k) => (k === "me" ? me.family : nodes.get(k).peer.family);
      pulses.push({ from: a, to: b, t: 0, speed: (ambient ? 0.55 : 0.8) * (0.85 + Math.random() * 0.3),
                    color: color || familyColor(fam(b === "me" ? a : b)), bend: (Math.random() - 0.5) * 0.5 });
    }
    function randomPeer() {
      const ids = [...nodes.keys()];
      return ids.length ? ids[Math.floor(Math.random() * ids.length)] : null;
    }
    function pt(id) { return id === "me" ? [W / 2, H / 2] : pos(nodes.get(id)); }

    function step(now) {
      if (!running) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const still = reduceMotion();
      if (!still) rot += dt * 0.012;
      // ambient traffic between peers (jobs of the rest of the network)
      ambientAcc += dt * ambientRate;
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
      const cx = W / 2, cy = H / 2;
      // soft radial backdrop
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.6);
      g.addColorStop(0, "rgba(124,108,255,0.10)");
      g.addColorStop(0.5, "rgba(56,200,255,0.03)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
      // orbit guides
      ctx.strokeStyle = theme.edge;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 6]);
      const R = Math.min(W, H) * 0.40;
      ctx.beginPath();
      ctx.ellipse(cx, cy, R * (W > H * 1.25 ? 1.25 : 1), R, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // links to every peer
      const P = new Map();
      for (const nd of nodes.values()) P.set(nd.id, pos(nd));
      for (const nd of nodes.values()) {
        const [x, y] = P.get(nd.id);
        const grad = ctx.createLinearGradient(cx, cy, x, y);
        grad.addColorStop(0, "rgba(255,255,255,0.02)");
        grad.addColorStop(1, hexA(familyColor(nd.peer.family), nd.peer.accepting ? 0.28 : 0.10));
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      }
      // pulses (comets)
      const arrived = [];
      for (const p of pulses) {
        p.t += dt * p.speed * (still ? 3 : 1);
        if (p.t >= 1) { arrived.push(p); continue; }
        const a = p.from === "me" ? [cx, cy] : P.get(p.from), b = p.to === "me" ? [cx, cy] : P.get(p.to);
        if (!a || !b) { p.t = 1; continue; }
        const mx = (a[0] + b[0]) / 2 + (b[1] - a[1]) * p.bend, my = (a[1] + b[1]) / 2 - (b[0] - a[0]) * p.bend;
        const at = (t) => [(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * mx + t * t * b[0],
                           (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * my + t * t * b[1]];
        for (let k = 8; k >= 0; k--) {
          const tt = Math.max(0, p.t - k * 0.018);
          const [x, y] = at(tt);
          ctx.fillStyle = hexA(p.color, (1 - k / 9) * 0.55);
          ctx.beginPath(); ctx.arc(x, y, 2.6 * (1 - k / 12), 0, Math.PI * 2); ctx.fill();
        }
        const [hx, hy] = at(p.t);
        ctx.shadowColor = p.color; ctx.shadowBlur = 14;
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(hx, hy, 2.4, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      }
      if (arrived.length) {
        pulses = pulses.filter((p) => p.t < 1);
        for (const p of arrived) {
          ripples.push({ id: p.to, t: 0, color: p.color });
          if (p.to === "me") me.glow = 1;
          else if (nodes.has(p.to)) nodes.get(p.to).glow = 1;
        }
      }
      // ripples
      ripples = ripples.filter((r) => (r.t += dt * 1.6) < 1);
      for (const r of ripples) {
        const c = r.id === "me" ? [cx, cy] : P.get(r.id);
        if (!c) continue;
        ctx.strokeStyle = hexA(r.color, 0.6 * (1 - r.t));
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(c[0], c[1], 8 + r.t * 26, 0, Math.PI * 2); ctx.stroke();
      }
      // peers
      for (const nd of nodes.values()) {
        const [x, y] = P.get(nd.id);
        const col = familyColor(nd.peer.family);
        const born = Math.min(1, (now - nd.born) / 600);
        const size = (4.5 + Math.min(4, (nd.peer.max_parallel || 1)) * 0.9) * (0.4 + 0.6 * born);
        nd.glow = Math.max(0, (nd.glow || 0) - dt * 1.2);
        const busy = (nd.peer.busy || 0) > 0;
        if (busy || nd.glow > 0) {
          const pulseA = busy ? 0.35 + 0.25 * Math.sin(now / 260 + hash(nd.id) * 6) : 0;
          const gr = ctx.createRadialGradient(x, y, 0, x, y, size * 4);
          gr.addColorStop(0, hexA(col, Math.max(pulseA, nd.glow * 0.7)));
          gr.addColorStop(1, hexA(col, 0));
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(x, y, size * 4, 0, Math.PI * 2); ctx.fill();
        }
        ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
        if (nd.peer.accepting) { ctx.fillStyle = col; ctx.fill(); }
        else { ctx.strokeStyle = hexA(col, 0.7); ctx.lineWidth = 1.5; ctx.stroke(); }
        if (hoverId === nd.id) {
          ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, size + 4, 0, Math.PI * 2); ctx.stroke();
        }
      }
      // me
      me.glow = Math.max(0, me.glow - dt * 0.9);
      const breathe = still ? 0 : Math.sin(now / 900) * 1.5;
      const rMe = 17 + breathe;
      const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, rMe * (3.2 + me.glow * 1.5));
      halo.addColorStop(0, hexA(theme.me1, 0.35 + me.glow * 0.3));
      halo.addColorStop(0.4, hexA(theme.me2, 0.16));
      halo.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = halo;
      ctx.beginPath(); ctx.arc(cx, cy, rMe * (3.2 + me.glow * 1.5), 0, Math.PI * 2); ctx.fill();
      const core = ctx.createRadialGradient(cx - 5, cy - 6, 2, cx, cy, rMe);
      core.addColorStop(0, "#fff4d6"); core.addColorStop(0.45, theme.me1); core.addColorStop(1, "#ff6aa0");
      ctx.fillStyle = core;
      ctx.beginPath(); ctx.arc(cx, cy, rMe, 0, Math.PI * 2); ctx.fill();
      if (me.serving && me.accepting) {  // rotating ring: serving the network
        ctx.strokeStyle = hexA(theme.me2, 0.9); ctx.lineWidth = 2.2;
        const a0 = still ? 0 : now / 700;
        ctx.beginPath(); ctx.arc(cx, cy, rMe + 7, a0, a0 + Math.PI * (me.busy > 0 ? 1.4 : 0.6)); ctx.stroke();
      }
      ctx.fillStyle = theme.text;
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(o.youLabel ? o.youLabel() : "You", cx, cy + rMe + 22);
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
      const meHit = (W / 2 - x) ** 2 + (H / 2 - y) ** 2 < 22 * 22;
      hoverId = best ? best.id : null;
      if (o.onHover) o.onHover(best ? best.peer : meHit ? "me" : null, x, y);
    });
    canvas.addEventListener("mouseleave", () => { hoverId = null; if (o.onHover) o.onHover(null); });
    new ResizeObserver(resize).observe(canvas);
    document.addEventListener("visibilitychange", () => {
      running = !document.hidden;
      if (running) { last = performance.now(); requestAnimationFrame(step); }
    });
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
