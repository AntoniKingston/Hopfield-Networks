/* Initialise reveal.js, fill data-driven slides and hook animations to slide events. */
(function () {
  "use strict";

  const D = window.DATA;
  const pct = (v) => `${Math.round(v * 100)}%`;
  const fmt = (v) => v.toFixed(2).replace(".", ",");

  // ---------------------------------------------------------------- data-driven content
  function stabilityTable() {
    const t = document.getElementById("stability-table");
    const cls = (k, P) => (k === P ? "cell-ok" : k === 0 ? "cell-bad" : "");
    const rows = D.stability
      .map(
        (r) =>
          `<tr${r.dataset === "large-25x25" ? ' class="hl"' : ""}><td>${r.dataset}</td><td>${r.P}</td>` +
          `<td>${r.N}</td><td>${fmt(r.mean_m)}</td><td class="${cls(r.hebb, r.P)}">${r.hebb}</td>` +
          `<td class="${cls(r.oja, r.P)}">${r.oja}</td><td class="${cls(r.projection, r.P)}">${r.projection}</td></tr>`
      )
      .join("");
    t.innerHTML =
      "<thead><tr><th>zbiór</th><th>P</th><th>N</th><th>śr. |m|</th><th>Hebb</th><th>Oja</th>" +
      `<th>projekcyjna</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function noiseStats() {
    const by = Object.fromEntries(D.noise.map((r) => [r["zbiór"], r]));
    const items = [
      [pct(by["large-25x25"]["H: dokł. 10\\%"]), "Hebb, large-25x25,<br>dokładnie przy 10% szumu"],
      [pct(by["small-7x7"]["H: dokł. 10\\%"]), "Hebb, small-7x7,<br>dokładnie przy 10% szumu"],
      [pct(by["large-25x25"]["P: $p_{90}$"]), "projekcyjna, large-25x25:<br>p<sub>90</sub> (≥90% odtworzeń)"],
      [pct(by["letters-14x20"]["P: dokł. 30\\%"]), "projekcyjna, 26 liter,<br>dokładnie przy 30% szumu"],
    ];
    document.getElementById("noise-stats").innerHTML = items
      .map(([v, l]) => `<div class="fragment"><div class="big-number">${v}</div>${l}</div>`)
      .join("");
  }

  function stable5x5() {
    const s = D.stable_5x5[0];
    document.getElementById("n5x5").textContent = s.P;
    document.getElementById("n5x5-cost").innerHTML =
      `ceną są małe baseny: minimalny margines \\(2/N\\), po odwróceniu 1&nbsp;piksela powrót do wzorca ` +
      `tylko w&nbsp;${pct(s.random_attraction_async)} przypadków; maks. \\(|m|=${fmt(s.max_abs_overlap).replace(",", "{,}")}\\)`;
  }

  // ---------------------------------------------------------------- title bitmap
  function titlePixels() {
    const root = document.getElementById("title-pixels");
    const d = D.datasets["animals-14x9"];
    const x = HF.parseBits(d.patterns[0]);
    const net = new HF.Network(d.patterns.map(HF.parseBits), "projection");
    const rng = HF.mulberry32(D.seed);
    const cells = Array.from(x, () => root.appendChild(document.createElement("span")));
    let timer;
    const paint = (s) => cells.forEach((c, i) => c.classList.toggle("on", s[i] > 0));
    return {
      run() {
        clearInterval(timer);
        const s = HF.corrupt(x, 0.35, rng);
        const order = HF.shuffle(Array.from(x, (_, i) => i), rng);
        let k = 0;
        paint(s);
        if (SlideAnimations.reduceMotion) return paint(x);
        timer = setInterval(() => {
          for (let j = 0; j < 4 && k < order.length; j++, k++) net.updateNeuron(s, order[k]);
          paint(s);
          if (k >= order.length) clearInterval(timer);
        }, 60);
      },
      stop: () => clearInterval(timer),
    };
  }

  stabilityTable();
  noiseStats();
  stable5x5();

  // ---------------------------------------------------------------- reveal
  Reveal.initialize({
    width: 1280,
    height: 720,
    margin: 0.05,
    hash: true,
    center: false,
    transition: "fade",
    transitionSpeed: "fast",
    slideNumber: "c/t",
    katex: { local: "vendor/katex" },
    plugins: [RevealNotes, RevealMath.KaTeX],
  }).then(() => {
    const A = SlideAnimations;
    const title = titlePixels();

    const land = new A.EnergyLandscape(document.getElementById("landscape"));
    document.getElementById("landscape-async").onclick = () => land.async();
    document.getElementById("landscape-sync").onclick = () => land.sync();
    document.getElementById("landscape-reset").onclick = () => land.reset();

    const oja = new A.OjaToy(document.getElementById("oja-canvas"), document.getElementById("oja-info"));
    document.getElementById("oja-run").onclick = () => oja.run();
    document.getElementById("oja-reset").onclick = () => oja.reset();

    const wide = new A.WideTransition(
      document.getElementById("wide-bitmap"),
      document.getElementById("wide-heatmap"),
      document.getElementById("wide-m"),
      document.getElementById("wide-hebb")
    );

    const demos = {};
    const ensureDemo = (kind) => {
      if (demos[kind]) return;
      if (kind === "main") {
        const sets = {};
        for (const [name, d] of Object.entries(D.datasets)) sets[name] = d;
        const order = ["large-25x25", "animals-14x9", "small-7x7", "OCRA-12x30-cut"];
        const sorted = Object.fromEntries(order.map((n) => [n, sets[n]]));
        demos.main = new HopfieldDemo(document.getElementById("demo-main"), sorted, { seed: D.seed });
      } else {
        const o = D.oscillation;
        demos.osc = new HopfieldDemo(
          document.getElementById("demo-osc"),
          { "X (6×6)": { shape: o.shape, patterns: [o.pattern] } },
          { rules: ["hebb"], noise: false, presetInput: o.input, seed: D.seed, mode: "sync" }
        );
      }
    };

    const onEnter = (slide) => {
      title.stop();
      land.stop();
      oja.stop();
      Object.values(demos).forEach((d) => d.stop());
      const idx = Reveal.getIndices(slide).h;
      if (idx === 0) title.run();
      if (slide.querySelector("#landscape")) land.reset();
      if (slide.querySelector("#oja-canvas")) {
        oja.reset();
        oja.run();
      }
      if (slide.dataset.demo) ensureDemo(slide.dataset.demo);
      if (slide.querySelector("#wide-bitmap")) {
        const shown = (a) => slide.querySelector(`[data-anim="${a}"]`).classList.contains("visible");
        wide.go(shown("control") ? "control" : shown("wide") ? "wide" : "small");
      }
    };
    Reveal.on("slidechanged", (e) => onEnter(e.currentSlide));
    onEnter(Reveal.getCurrentSlide());

    Reveal.on("fragmentshown", (e) => {
      const a = e.fragment.dataset.anim;
      if (a === "wide") wide.go("wide");
      if (a === "control") wide.go("control");
    });
    Reveal.on("fragmenthidden", (e) => {
      const a = e.fragment.dataset.anim;
      if (a === "wide") wide.go("small");
      if (a === "control") wide.go("wide");
    });
  });
})();
