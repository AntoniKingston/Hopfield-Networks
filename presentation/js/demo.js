/* Live Hopfield demo: pixel grid, sync/async animation, energy plot and overlaps. */
(function (global) {
  "use strict";

  const COLORS = { on: "#ffffff", off: "#1f2937", flip: "232, 89, 12", grid: "#cbd5e1" };
  const RULE_LABEL = { hebb: "Hebb", oja: "Oja", projection: "projekcyjna" };
  const MAX_STEPS = 40;

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) node.append(c);
    return node;
  }

  function drawBitmap(canvas, s, shape, flips) {
    const [H, W] = shape;
    const ctx = canvas.getContext("2d");
    const cell = Math.floor(Math.min(canvas.width / W, canvas.height / H));
    const ox = Math.floor((canvas.width - cell * W) / 2);
    const oy = Math.floor((canvas.height - cell * H) / 2);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        ctx.fillStyle = s[i] > 0 ? COLORS.on : COLORS.off;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
        if (flips && flips[i] > 0) {
          const lw = Math.min(4, Math.max(2, cell * 0.15));
          ctx.strokeStyle = `rgba(${COLORS.flip}, ${flips[i]})`;
          ctx.lineWidth = lw;
          ctx.strokeRect(ox + c * cell + lw / 2, oy + r * cell + lw / 2, cell - lw, cell - lw);
        }
      }
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(ox - 0.5, oy - 0.5, cell * W + 1, cell * H + 1);
  }

  class HopfieldDemo {
    /**
     * sets: { name: {shape, patterns: [bits], oja?: b64} }
     * options: { rules, presetInput (bits), noise (bool), seed, mode ("async" | "sync") }
     */
    constructor(root, sets, options = {}) {
      this.root = root;
      this.sets = sets;
      this.opts = { rules: ["hebb", "oja", "projection"], noise: true, seed: 2137, mode: "async", ...options };
      this.rng = HF.mulberry32(this.opts.seed);
      this.cache = {};
      this.running = false;
      this.build();
      this.load(Object.keys(sets)[0]);
    }

    build() {
      const o = this.opts;
      this.ui = {};
      const u = this.ui;
      u.dataset = el("select", {}, Object.keys(this.sets).map((n) => el("option", { value: n, text: n })));
      u.rule = el("select", {}, o.rules.map((r) => el("option", { value: r, text: RULE_LABEL[r] })));
      u.mode = el("div", { class: "seg" }, [
        el("button", { "data-mode": "async", class: "on", text: "asynchr." }),
        el("button", { "data-mode": "sync", text: "synchr." }),
      ]);
      u.noise = el("input", { type: "range", min: "0", max: "50", step: "5", value: "30" });
      u.noiseVal = el("span", { class: "val", text: "30%" });
      u.corrupt = el("button", { text: o.noise ? "Zaszum" : "Wejście" });
      u.random = el("button", { text: "Losowy stan" });
      u.step = el("button", { text: "Krok" });
      u.play = el("button", { class: "primary", text: "▶ Odtwarzaj" });

      const controls = el("div", { class: "demo-controls" }, [
        el("label", {}, ["Zbiór ", u.dataset]),
        el("label", {}, ["Reguła ", u.rule]),
        u.mode,
      ]);
      const controls2 = el("div", { class: "demo-controls" }, []);
      if (o.noise) controls2.append(el("label", {}, ["Szum ", u.noise, u.noiseVal]));
      controls2.append(u.corrupt);
      if (o.noise) controls2.append(u.random);
      controls2.append(u.step, u.play);
      if (Object.keys(this.sets).length === 1) controls.firstChild.style.display = "none";
      if (o.rules.length === 1) controls.children[1].style.display = "none";

      u.state = el("canvas", { class: "demo-state", width: "420", height: "420" });
      u.energy = el("canvas", { class: "demo-energy", width: "420", height: "150" });
      u.thumbs = el("div", { class: "demo-thumbs" });
      u.status = el("div", { class: "demo-status" });
      u.counter = el("div", { class: "demo-counter" });

      this.root.append(
        controls,
        controls2,
        el("div", { class: "demo-main" }, [
          el("div", { class: "demo-left" }, [u.state, u.counter]),
          el("div", { class: "demo-right" }, [
            el("div", { class: "demo-caption", text: "Energia E(s)" }),
            u.energy,
            el("div", { class: "demo-caption", text: "Nakładanie m z wzorcami (kliknij, aby wybrać cel)" }),
            u.thumbs,
          ]),
        ]),
        u.status
      );

      u.dataset.addEventListener("change", () => this.load(u.dataset.value));
      u.rule.addEventListener("change", () => this.setRule(u.rule.value));
      u.mode.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        for (const x of u.mode.children) x.classList.toggle("on", x === b);
        this.mode = b.dataset.mode;
        this.resetRun();
      });
      u.noise.addEventListener("input", () => (u.noiseVal.textContent = `${u.noise.value}%`));
      u.corrupt.addEventListener("click", () => this.start());
      u.random.addEventListener("click", () => this.start(true));
      u.step.addEventListener("click", () => this.stepOnce());
      u.play.addEventListener("click", () => (this.running ? this.stop() : this.play()));
      this.mode = this.opts.mode;
      for (const x of u.mode.children) x.classList.toggle("on", x.dataset.mode === this.mode);
    }

    load(name) {
      this.stop();
      this.name = name;
      this.ui.dataset.value = name;
      const d = this.sets[name];
      this.shape = d.shape;
      this.X = d.patterns.map(HF.parseBits);
      this.target = 0;
      this.setRule(this.ui.rule.value, true);
      this.buildThumbs();
      this.start();
    }

    network(rule) {
      const key = `${this.name}/${rule}`;
      if (!this.cache[key]) {
        const d = this.sets[this.name];
        const W = rule === "oja" ? HF.decodeFloat32(d.oja) : null;
        this.cache[key] = new HF.Network(this.X, rule, W);
      }
      return this.cache[key];
    }

    setRule(rule, silent) {
      this.net = this.network(rule);
      this.stableFlags = this.X.map((x) => this.net.isStable(x));
      if (!silent) {
        this.buildThumbs();
        this.resetRun();
      }
    }

    buildThumbs() {
      const u = this.ui;
      u.thumbs.replaceChildren();
      this.bars = this.X.map((x, k) => {
        const c = el("canvas", { width: "60", height: "60" });
        drawBitmap(c, x, this.shape);
        const bar = el("div", { class: "bar" }, [el("div", { class: "fill" })]);
        const val = el("div", { class: "mval" });
        const box = el("div", { class: "thumb" + (k === this.target ? " target" : "") }, [c, bar, val]);
        if (this.stableFlags) box.classList.add(this.stableFlags[k] ? "stable" : "unstable");
        box.title = this.stableFlags && this.stableFlags[k] ? "wzorzec stabilny" : "wzorzec niestabilny";
        box.addEventListener("click", () => {
          this.target = k;
          for (const t of u.thumbs.children) t.classList.remove("target");
          box.classList.add("target");
          this.start();
        });
        u.thumbs.append(box);
        return { fill: bar.firstChild, val };
      });
    }

    start(random = false) {
      this.stop();
      const x = this.X[this.target];
      if (this.opts.presetInput) this.initial = HF.parseBits(this.opts.presetInput);
      else if (random) this.initial = HF.randomState(x.length, this.rng);
      else this.initial = HF.corrupt(x, Number(this.ui.noise.value) / 100, this.rng);
      this.resetRun();
    }

    resetRun() {
      this.stop();
      this.s = Float64Array.from(this.initial);
      this.history = [Float64Array.from(this.s)];
      this.energies = [[0, this.net.energy(this.s)]];
      this.steps = 0;
      this.done = null;
      this.flips = new Float64Array(this.s.length);
      this.order = null;
      this.render();
    }

    /** Advance by `chunk` neurons (async) or one step (sync). Returns true when finished. */
    advance(chunk) {
      if (this.done) return true;
      const N = this.s.length;
      if (this.mode === "sync") {
        const next = this.net.syncStep(this.s);
        for (let i = 0; i < N; i++) if (next[i] !== this.s[i]) this.flips[i] = 1;
        const prev = this.history.length >= 2 ? this.history[this.history.length - 2] : null;
        const fixed = HF.equal(next, this.s);
        this.s = next;
        if (!fixed) {
          this.steps += 1;
          this.history.push(Float64Array.from(next));
          this.energies.push([this.steps, this.net.energy(next)]);
        }
        if (fixed) this.done = "fixed";
        else if (prev && HF.equal(next, prev)) this.done = "cycle";
        else if (this.steps >= MAX_STEPS) this.done = "limit";
        return !!this.done;
      }
      if (!this.order) {
        this.order = HF.shuffle(Array.from({ length: N }, (_, i) => i), this.rng);
        this.pos = 0;
        this.changed = false;
      }
      const end = Math.min(N, this.pos + chunk);
      for (; this.pos < end; this.pos++) {
        const i = this.order[this.pos];
        if (this.net.updateNeuron(this.s, i)) {
          this.flips[i] = 1;
          this.changed = true;
        }
      }
      this.energies.push([this.steps + this.pos / N, this.net.energy(this.s)]);
      if (this.pos >= N) {
        this.order = null;
        if (!this.changed) {
          this.energies.pop();
          this.done = "fixed";
        } else {
          this.steps += 1;
          if (this.steps >= MAX_STEPS) this.done = "limit";
        }
      }
      return !!this.done;
    }

    stepOnce() {
      this.stop();
      if (this.mode === "sync") this.advance();
      else {
        do this.advance(this.s.length);
        while (this.order);
      }
      this.render();
    }

    play() {
      if (this.done) this.resetRun();
      this.running = true;
      this.ui.play.textContent = "❚❚ Pauza";
      const N = this.s.length;
      const chunk = Math.max(1, Math.ceil(N / 45));
      let last = 0;
      const tick = (t) => {
        if (!this.running) return;
        const interval = this.mode === "sync" ? 650 : 16;
        if (t - last >= interval) {
          last = t;
          const finished = this.advance(chunk);
          this.render();
          if (finished) return this.stop();
        } else {
          this.fade();
          drawBitmap(this.ui.state, this.s, this.shape, this.flips);
        }
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }

    stop() {
      this.running = false;
      if (this.raf) cancelAnimationFrame(this.raf);
      if (this.ui) this.ui.play.textContent = "▶ Odtwarzaj";
    }

    fade() {
      const f = this.mode === "sync" ? 0.97 : 0.9;
      for (let i = 0; i < this.flips.length; i++) if (this.flips[i] > 0.02) this.flips[i] *= f;
      else this.flips[i] = 0;
    }

    classify() {
      for (let k = 0; k < this.X.length; k++) {
        if (HF.equal(this.s, this.X[k])) return k === this.target ? "oryginał ✓" : `inny wzorzec (#${k + 1})`;
        if (this.X[k].every((v, i) => v === -this.s[i])) return `negatyw wzorca #${k + 1}`;
      }
      return "stan fałszywy";
    }

    render() {
      this.fade();
      drawBitmap(this.ui.state, this.s, this.shape, this.flips);
      const m = this.net.overlaps(this.s);
      this.bars.forEach((b, k) => {
        const v = m[k];
        b.fill.style.width = `${Math.abs(v) * 50}%`;
        b.fill.style.left = v >= 0 ? "50%" : `${50 - Math.abs(v) * 50}%`;
        b.fill.classList.toggle("neg", v < 0);
        b.val.textContent = v.toFixed(2);
      });
      const unit = this.mode === "sync" ? "krok" : "przebieg";
      this.ui.counter.textContent = `N = ${this.s.length}, ${unit}: ${this.steps}`;
      const E = this.energies[this.energies.length - 1][1];
      let status = `E = ${E.toFixed(2)}`;
      if (this.done === "fixed") status += ` · punkt stały → ${this.classify()}`;
      else if (this.done === "cycle") status += " · cykl długości 2 (s → s′ → s)";
      else if (this.done === "limit") status += " · przerwano po limicie kroków";
      this.ui.status.textContent = status;
      this.ui.status.className = "demo-status" + (this.done ? ` done-${this.done}` : "");
      this.drawEnergy();
    }

    drawEnergy() {
      const c = this.ui.energy;
      const ctx = c.getContext("2d");
      const pts = this.energies;
      ctx.clearRect(0, 0, c.width, c.height);
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const xmax = Math.max(4, ...xs);
      let ymin = Math.min(...ys);
      let ymax = Math.max(...ys);
      if (ymax - ymin < 1e-6) {
        ymin -= 1;
        ymax += 1;
      }
      const pad = 24;
      const X = (x) => pad + (x / xmax) * (c.width - pad - 8);
      const Y = (y) => 8 + (1 - (y - ymin) / (ymax - ymin)) * (c.height - pad - 8);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, 8);
      ctx.lineTo(pad, c.height - pad);
      ctx.lineTo(c.width - 8, c.height - pad);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.font = "12px sans-serif";
      ctx.fillText(ymax.toFixed(1), 2, 16);
      ctx.fillText(ymin.toFixed(1), 2, c.height - pad);
      ctx.fillText(this.mode === "sync" ? "krok" : "przebieg", c.width - 60, c.height - 6);
      ctx.strokeStyle = "#e8590c";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.fillStyle = "#e8590c";
      for (const [x, y] of pts) if (Number.isInteger(x)) ctx.fillRect(X(x) - 2.5, Y(y) - 2.5, 5, 5);
    }
  }

  global.HopfieldDemo = HopfieldDemo;
})(window);
