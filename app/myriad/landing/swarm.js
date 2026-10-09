// Hero animation: a murmuration of small dots (the peers) that swirls across the hero, then converges
// into one disc, the Myriad logo (one answer), holds, and scatters again. A plain 2D canvas, O(N) per
// frame, no dependency. Paused when the hero is off-screen or the tab hidden; a still logo when the
// visitor prefers reduced motion.
(function () {
  "use strict";
  const GOLDEN = Math.PI * (3 - Math.sqrt(5));
  // The icon's gradient, from the centre of the disc to its edge (myriad/web/icon.svg).
  const STOPS = [[0, [255, 191, 121]], [0.12, [251, 106, 163]], [0.48, [124, 108, 255]], [1, [59, 213, 255]]];
  // Timeline of one cycle (seconds): flock, gather, hold, release.
  const FLOCK = 6.5, GATHER = 2.6, HOLD = 3.6, RELEASE = 1.4;
  const CYCLE = FLOCK + GATHER + HOLD + RELEASE;
  const START = FLOCK - 1.6; // the first gathering comes soon after the page opens

  function colorAt(f) {
    for (let i = 1; i < STOPS.length; i++) {
      const [f1, c1] = STOPS[i];
      if (f <= f1) {
        const [f0, c0] = STOPS[i - 1];
        const k = (f - f0) / (f1 - f0 || 1);
        return c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
      }
    }
    return STOPS[STOPS.length - 1][1];
  }
  const smooth = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

  function start(canvas) {
    if (!canvas || !canvas.getContext) return null;
    const ctx = canvas.getContext("2d");
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dpr = 1, dots = [], cx = 0, cy = 0, R = 0, narrow = false;
    let raf = 0, last = 0, clock = START, visible = true, running = false;
    const pointer = { x: -1e4, y: -1e4 };

    function layout() {
      const rect = canvas.getBoundingClientRect();
      W = Math.max(1, rect.width); H = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      narrow = W < 980;
      // Narrow screens: a smaller disc in the top-right corner, half out of view, faded by the CSS.
      if (narrow) { R = Math.min(W * 0.38, 190); cx = W - R * 0.5; cy = R * 0.8 + 30; }
      else { cx = W * 0.76; cy = H * 0.5; R = Math.min(H * 0.34, W * 0.2); }
      const n = Math.round(Math.max(180, Math.min(narrow ? 320 : 620, (W * H) / 2100)));
      const spacing = R * Math.sqrt(Math.PI / n);
      const old = dots;
      dots = [];
      for (let i = 0; i < n; i++) {
        const f = (i + 0.5) / n;
        const r = R * Math.sqrt(f), a = i * GOLDEN;
        const [cr, cg, cb] = colorAt(f);
        const prev = old[i];
        dots.push({
          x: prev ? prev.x : Math.random() * W, y: prev ? prev.y : Math.random() * H,
          vx: prev ? prev.vx : (Math.random() - 0.5) * 40, vy: prev ? prev.vy : (Math.random() - 0.5) * 40,
          tx: cx + r * Math.cos(a), ty: cy + r * Math.sin(a),
          big: spacing * (0.42 - 0.2 * f), small: 0.9 + Math.random() * 1.1,
          color: `rgb(${cr},${cg},${cb})`, seed: Math.random() * 1000,
        });
      }
    }

    // Phase of the cycle: 0 = free flock, 1 = in the disc.
    function shapeWeight(t) {
      const u = t % CYCLE;
      if (u < FLOCK) return 0;
      if (u < FLOCK + GATHER) return smooth((u - FLOCK) / GATHER);
      if (u < FLOCK + GATHER + HOLD) return 1;
      return 1 - smooth((u - FLOCK - GATHER - HOLD) / RELEASE);
    }

    function step(dt, t) {
      const w = shapeWeight(t);
      const releasing = t % CYCLE > FLOCK + GATHER + HOLD;
      // The flock's leader wanders along a Lissajous curve over the whole hero.
      const ax = W * (0.5 + 0.36 * Math.sin(t * 0.21) * Math.cos(t * 0.07));
      const ay = H * (0.5 + 0.3 * Math.sin(t * 0.33 + 1.3));
      const speed = narrow ? 70 : 110;
      const breathe = 1 + 0.025 * Math.sin(t * 1.6);
      const rot = t * 0.05;
      const cr = Math.cos(rot), sr = Math.sin(rot);
      for (const d of dots) {
        // Free flight: a smooth, time-varying flow field (streams that bend and fold) plus a pull to the leader.
        const ang = Math.sin(d.x * 0.0042 + t * 0.35) * 1.7 + Math.cos(d.y * 0.0051 - t * 0.27) * 1.7 + Math.sin(d.seed + t * 0.5) * 0.35;
        let fx = Math.cos(ang) * speed + (ax - d.x) * 0.28;
        let fy = Math.sin(ang) * speed + (ay - d.y) * 0.28;
        if (releasing) { fx += (d.x - cx) * 0.9; fy += (d.y - cy) * 0.9; }
        // The disc slot, slowly turning and breathing.
        const ox = (d.tx - cx) * breathe, oy = (d.ty - cy) * breathe;
        const sx = cx + ox * cr - oy * sr, sy = cy + ox * sr + oy * cr;
        // Pointer: dots step aside.
        const px = d.x - pointer.x, py = d.y - pointer.y, pd = px * px + py * py;
        if (pd < 8100) { const k = (1 - pd / 8100) * 900 / (Math.sqrt(pd) + 1); fx += px * k * 0.02; fy += py * k * 0.02; }
        // Blend: steer to the flow (velocity target) and spring to the slot.
        const k1 = 1 - w;
        d.vx += ((fx - d.vx) * 1.6 * k1 + ((sx - d.x) * 26 - d.vx * 9) * w) * dt;
        d.vy += ((fy - d.vy) * 1.6 * k1 + ((sy - d.y) * 26 - d.vy * 9) * w) * dt;
        d.x += d.vx * dt; d.y += d.vy * dt;
        d.size = d.small + (d.big - d.small) * w;
      }
    }

    function draw(trail) {
      if (trail) {
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "rgba(0,0,0,0.32)";
        ctx.fillRect(0, 0, W, H);
        ctx.globalCompositeOperation = "source-over";
      } else {
        ctx.clearRect(0, 0, W, H);
      }
      for (const d of dots) {
        ctx.fillStyle = d.color;
        ctx.beginPath();
        ctx.arc(d.x, d.y, Math.max(0.6, d.size), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function still() {
      for (const d of dots) { d.x = d.tx; d.y = d.ty; d.size = d.big; }
      draw(false);
    }

    function frame(now) {
      raf = 0;
      if (!running) return;
      const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016);
      last = now;
      clock += dt;
      step(dt, clock);
      draw(true);
      raf = requestAnimationFrame(frame);
    }

    function update() {
      const want = visible && !document.hidden && !reduce;
      if (want && !running) { running = true; last = 0; raf = requestAnimationFrame(frame); }
      else if (!want && running) { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
    }

    layout();
    if (reduce) still();
    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { layout(); if (reduce) still(); }, 120);
    });
    document.addEventListener("visibilitychange", update);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; update(); }).observe(canvas);
    }
    const hero = canvas.parentElement;
    hero.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      const r = canvas.getBoundingClientRect();
      pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top;
    });
    hero.addEventListener("pointerleave", () => { pointer.x = pointer.y = -1e4; });
    update();
    return { layout };
  }

  window.MyriadSwarm = { start };
})();
