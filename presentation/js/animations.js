/* Slide animations: energy landscape (A), Oja 2D toy (B), 25x25 -> 25x50 transition (C). */
(function (global) {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ACCENT = "#e8590c";
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

  function tween(duration, onFrame, onDone) {
    if (reduceMotion || duration <= 0) {
      onFrame(1);
      if (onDone) onDone();
      return () => {};
    }
    let raf;
    const t0 = performance.now();
    const frame = (t) => {
      const p = Math.min(1, (t - t0) / duration);
      onFrame(ease(p));
      if (p < 1) raf = requestAnimationFrame(frame);
      else if (onDone) onDone();
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }

  // ------------------------------------------------------------------ A: energy landscape
  class EnergyLandscape {
    constructor(svg) {
      this.svg = svg;
      this.W = 760;
      this.H = 300;
      this.wells = [
        { c: 0.17, d: 1.0, w: 0.07 },
        { c: 0.45, d: 0.65, w: 0.06 },
        { c: 0.78, d: 1.15, w: 0.09 },
      ];
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const x = i / 200;
        pts.push(`${this.px(x)},${this.py(this.E(x))}`);
      }
      const ns = "http://www.w3.org/2000/svg";
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", "M" + pts.join(" L"));
      path.setAttribute("class", "landscape");
      svg.append(path);
      this.wells.forEach((w, k) => {
        const t = document.createElementNS(ns, "text");
        t.setAttribute("x", this.px(w.c));
        t.setAttribute("y", this.py(this.E(w.c)) + 28);
        t.setAttribute("class", "well-label");
        t.textContent = ["wzorzec 1", "stan fałszywy", "wzorzec 2"][k];
        svg.append(t);
      });
      this.trail = document.createElementNS(ns, "path");
      this.trail.setAttribute("class", "trail");
      svg.append(this.trail);
      this.ball = document.createElementNS(ns, "circle");
      this.ball.setAttribute("r", "11");
      this.ball.setAttribute("class", "ball");
      svg.append(this.ball);
      this.label = document.createElementNS(ns, "text");
      this.label.setAttribute("x", "16");
      this.label.setAttribute("y", "26");
      this.label.setAttribute("class", "energy-readout");
      svg.append(this.label);
      this.place(0.64);
    }

    E(x) {
      let e = 0.03 * Math.sin(9 * x);
      for (const w of this.wells) e -= w.d * Math.exp(-((x - w.c) ** 2) / (2 * w.w * w.w));
      return e;
    }

    dE(x) {
      const h = 1e-4;
      return (this.E(x + h) - this.E(x - h)) / (2 * h);
    }

    px(x) {
      return 30 + x * (this.W - 60);
    }

    py(e) {
      const emax = 0.1;
      const emin = -1.25;
      return 50 + ((emax - e) / (emax - emin)) * (this.H - 90);
    }

    place(x, y = this.py(this.E(x)) - 11) {
      this.ball.setAttribute("cx", this.px(x));
      this.ball.setAttribute("cy", y);
    }

    stop() {
      if (this.cancel) this.cancel();
      clearInterval(this.timer);
    }

    async() {
      this.stop();
      let x = 0.64;
      const path = [];
      this.label.textContent = "tryb asynchroniczny: E nie rośnie";
      const step = () => {
        const dx = Math.max(-0.0015, Math.min(0.0015, 0.0002 * this.dE(x)));
        x -= dx;
        path.push(`${this.px(x)},${this.py(this.E(x)) - 11}`);
        this.trail.setAttribute("d", "M" + path.join(" L"));
        this.place(x);
        return Math.abs(dx) < 2e-5;
      };
      if (reduceMotion) {
        while (!step());
        return;
      }
      let raf;
      const frame = () => {
        let done = false;
        for (let k = 0; k < 4 && !done; k++) done = step();
        if (!done) raf = requestAnimationFrame(frame);
        else this.label.textContent = "tryb asynchroniczny: minimum lokalne = wzorzec 2";
      };
      raf = requestAnimationFrame(frame);
      this.cancel = () => cancelAnimationFrame(raf);
    }

    sync() {
      this.stop();
      this.trail.setAttribute("d", "");
      const a = 0.31;
      const b = 0.59;
      const ya = this.py(this.E(a)) - 11;
      const yb = this.py(this.E(b)) - 11;
      this.label.textContent = "tryb synchroniczny: s → s′ → s (cykl długości 2)";
      let left = true;
      const jump = () => {
        const [x0, x1] = left ? [a, b] : [b, a];
        const [y0, y1] = left ? [ya, yb] : [yb, ya];
        left = !left;
        this.cancel = tween(700, (t) => {
          const y = lerp(y0, y1, t) - Math.sin(Math.PI * t) * 70;
          this.place(lerp(x0, x1, t), y);
        });
      };
      this.place(a, ya);
      jump();
      if (!reduceMotion) this.timer = setInterval(jump, 900);
    }

    reset() {
      this.stop();
      this.trail.setAttribute("d", "");
      this.label.textContent = "";
      this.place(0.64);
    }
  }

  // ------------------------------------------------------------------ B: Oja 2D toy
  class OjaToy {
    constructor(canvas, info) {
      this.c = canvas;
      this.info = info;
      this.rng = HF.mulberry32(2137);
      const g = () => {
        const u = Math.max(1e-9, this.rng());
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.rng());
      };
      const angle = 0.55;
      this.v1 = [Math.cos(angle), Math.sin(angle)];
      this.pts = Array.from({ length: 260 }, () => {
        const a = 1.6 * g();
        const b = 0.45 * g();
        return [a * this.v1[0] - b * this.v1[1], a * this.v1[1] + b * this.v1[0]];
      });
      this.reset();
    }

    reset() {
      this.stop();
      this.t = 0;
      this.ws = Array.from({ length: 6 }, (_, k) => {
        const a = (k / 6) * 2 * Math.PI + 0.4;
        const r = 0.35 + 1.1 * this.rng();
        return [r * Math.cos(a), r * Math.sin(a)];
      });
      this.draw();
    }

    stop() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
    }

    run() {
      this.stop();
      const eta = 0.01;
      const frame = () => {
        for (let k = 0; k < 6; k++) {
          const x = this.pts[Math.floor(this.rng() * this.pts.length)];
          for (const w of this.ws) {
            const y = w[0] * x[0] + w[1] * x[1];
            w[0] += eta * y * (x[0] - y * w[0]);
            w[1] += eta * y * (x[1] - y * w[1]);
          }
          this.t += 1;
        }
        this.draw();
        if (this.t < 2400) this.raf = requestAnimationFrame(frame);
      };
      if (reduceMotion) {
        while (this.t < 2400) frame();
        return;
      }
      this.raf = requestAnimationFrame(frame);
    }

    draw() {
      const { c } = this;
      const ctx = c.getContext("2d");
      const S = c.width / 7;
      const cx = c.width / 2;
      const cy = c.height / 2;
      const P = (x, y) => [cx + x * S, cy - y * S];
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = "rgba(51, 65, 85, 0.35)";
      for (const [x, y] of this.pts) {
        const [u, v] = P(x, y);
        ctx.beginPath();
        ctx.arc(u, v, 2.6, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.setLineDash([7, 6]);
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(...P(-3.4 * this.v1[0], -3.4 * this.v1[1]));
      ctx.lineTo(...P(3.4 * this.v1[0], 3.4 * this.v1[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#2563eb";
      ctx.font = "18px sans-serif";
      ctx.fillText("v₁ (PC1)", ...P(2.2 * this.v1[0] + 0.1, 2.2 * this.v1[1] + 0.45));
      let align = 0;
      for (const w of this.ws) {
        const [x, y] = P(w[0], w[1]);
        const n = Math.hypot(w[0], w[1]);
        align += Math.abs((w[0] * this.v1[0] + w[1] * this.v1[1]) / n);
        ctx.strokeStyle = ACCENT;
        ctx.fillStyle = ACCENT;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
        const a = Math.atan2(y - cy, x - cx);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - 12 * Math.cos(a - 0.4), y - 12 * Math.sin(a - 0.4));
        ctx.lineTo(x - 12 * Math.cos(a + 0.4), y - 12 * Math.sin(a + 0.4));
        ctx.fill();
      }
      const norms = this.ws.map((w) => Math.hypot(w[0], w[1]));
      this.info.innerHTML =
        `krok ${this.t} · średnie |cos(w<sub>i</sub>, v<sub>1</sub>)| = ${(align / this.ws.length).toFixed(2)}` +
        ` · ‖w<sub>i</sub>‖ ∈ [${Math.min(...norms).toFixed(2)}, ${Math.max(...norms).toFixed(2)}]`;
    }
  }

  // ------------------------------------------------------------------ C: 25x25 -> 25x50
  function rdbu(v) {
    const t = Math.max(-1, Math.min(1, v));
    const white = [247, 247, 247];
    const end = t >= 0 ? [178, 24, 43] : [33, 102, 172];
    const a = Math.abs(t);
    return `rgb(${white.map((w, i) => Math.round(lerp(w, end[i], a))).join(",")})`;
  }

  class WideTransition {
    constructor(bitmap, heatmap, mText, hebbText) {
      this.bitmap = bitmap;
      this.heatmap = heatmap;
      this.mText = mText;
      this.hebbText = hebbText;
      const d = DATA.wide;
      this.small = HF.parseBits(d.pattern_small);
      this.wide = HF.parseBits(d.pattern_wide);
      this.mats = { small: d.overlap_small, wide: d.overlap_wide, control: d.overlap_control };
      const cmp = Object.fromEntries(DATA.compare_25x50.map((r) => [r.dataset, r]));
      this.stats = {
        small: cmp["large-25x25"],
        wide: cmp["large-25x50"],
        control: cmp["large-25x25 + losowa połowa"],
      };
      this.cur = "small";
      this.slide = 0;
      this.draw(0, this.mats.small);
      this.texts(this.stats.small);
    }

    meanOffDiag(M) {
      let s = 0;
      let n = 0;
      M.forEach((row, i) =>
        row.forEach((v, j) => {
          if (i !== j) {
            s += Math.abs(v);
            n += 1;
          }
        })
      );
      return s / n;
    }

    texts(st, m) {
      const P = this.mats.small.length;
      this.mText.textContent = (m ?? st.mean_abs_overlap).toFixed(2).replace(".", ",");
      this.hebbText.textContent = `${st.stable_hebb}/${P}`;
    }

    draw(slide, M) {
      const b = this.bitmap;
      const ctx = b.getContext("2d");
      const cell = Math.floor(b.width / 25);
      ctx.clearRect(0, 0, b.width, b.height);
      for (let r = 0; r < 25; r++)
        for (let c = 0; c < 25; c++) {
          ctx.fillStyle = this.small[r * 25 + c] > 0 ? "#fff" : "#1f2937";
          ctx.fillRect(c * cell, r * cell, cell, cell);
        }
      ctx.strokeStyle = "#94a3b8";
      ctx.setLineDash([6, 5]);
      ctx.strokeRect(0.5, 25 * cell + 0.5, 25 * cell - 1, 25 * cell - 1);
      ctx.setLineDash([]);
      if (slide > 0) {
        const offset = (1 - slide) * 25 * cell;
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 25 * cell, 25 * cell, 25 * cell);
        ctx.clip();
        for (let r = 25; r < 50; r++)
          for (let c = 0; c < 25; c++) {
            ctx.fillStyle = this.wide[r * 25 + c] > 0 ? "#fff" : "#1f2937";
            ctx.fillRect(c * cell, r * cell + offset, cell, cell);
          }
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 3;
        ctx.strokeRect(1.5, 25 * cell + offset + 1.5, 25 * cell - 3, 25 * cell - 3);
        ctx.restore();
      }
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, 25 * cell - 1, 25 * cell - 1);

      const h = this.heatmap;
      const hc = h.getContext("2d");
      const P = M.length;
      const s = Math.floor(h.width / P);
      hc.clearRect(0, 0, h.width, h.height);
      hc.font = `${Math.floor(s / 3.6)}px sans-serif`;
      hc.textAlign = "center";
      hc.textBaseline = "middle";
      for (let i = 0; i < P; i++)
        for (let j = 0; j < P; j++) {
          hc.fillStyle = rdbu(M[i][j]);
          hc.fillRect(j * s, i * s, s - 2, s - 2);
          hc.fillStyle = Math.abs(M[i][j]) > 0.6 ? "#fff" : "#111";
          hc.fillText(M[i][j].toFixed(2).replace(".", ","), j * s + s / 2, i * s + s / 2);
        }
    }

    go(target) {
      if (this.cancel) this.cancel();
      const from = this.mats[this.cur];
      const to = this.mats[target];
      const s0 = this.slide;
      const s1 = target === "wide" ? 1 : 0;
      const m0 = this.meanOffDiag(from);
      const m1 = this.meanOffDiag(to);
      const stats = this.stats[target];
      this.cur = target;
      this.cancel = tween(1600, (t) => {
        this.slide = lerp(s0, s1, t);
        const M = from.map((row, i) => row.map((v, j) => lerp(v, to[i][j], t)));
        this.draw(this.slide, M);
        this.mText.textContent = lerp(m0, m1, t).toFixed(2).replace(".", ",");
      }, () => this.texts(stats));
    }
  }

  global.SlideAnimations = { EnergyLandscape, OjaToy, WideTransition, reduceMotion };
})(window);
