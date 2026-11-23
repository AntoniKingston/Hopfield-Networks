/* Minimal Hopfield network for the live demo (mirrors src/hopfield.py). */
(function (global) {
  "use strict";

  const TIE_TOL = 1e-9;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  function parseBits(str) {
    const v = new Float64Array(str.length);
    for (let i = 0; i < str.length; i++) v[i] = str[i] === "1" ? 1 : -1;
    return v;
  }

  function decodeFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float64Array(new Float32Array(bytes.buffer));
  }

  function invert(M, n) {
    const A = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) A[i * 2 * n + j] = M[i * n + j];
      A[i * 2 * n + n + i] = 1;
    }
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) {
        if (Math.abs(A[r * 2 * n + c]) > Math.abs(A[p * 2 * n + c])) p = r;
      }
      if (p !== c) {
        for (let k = 0; k < 2 * n; k++) {
          const t = A[c * 2 * n + k];
          A[c * 2 * n + k] = A[p * 2 * n + k];
          A[p * 2 * n + k] = t;
        }
      }
      const piv = A[c * 2 * n + c];
      for (let k = 0; k < 2 * n; k++) A[c * 2 * n + k] /= piv;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = A[r * 2 * n + c];
        if (f === 0) continue;
        for (let k = 0; k < 2 * n; k++) A[r * 2 * n + k] -= f * A[c * 2 * n + k];
      }
    }
    const inv = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) inv[i * n + j] = A[i * 2 * n + n + j];
    return inv;
  }

  class Network {
    /** patterns: array of Float64Array (+-1), rule: "hebb" | "projection" | "oja". */
    constructor(patterns, rule, ojaWeights) {
      this.patterns = patterns;
      this.P = patterns.length;
      this.N = patterns[0].length;
      this.rule = rule;
      const N = this.N;
      if (rule === "oja") {
        this.W = ojaWeights;
      } else if (rule === "projection") {
        this.W = this.projection();
      } else {
        this.W = new Float64Array(N * N);
        for (const x of patterns)
          for (let i = 0; i < N; i++) {
            const xi = x[i] / N;
            for (let j = 0; j < N; j++) this.W[i * N + j] += xi * x[j];
          }
      }
      for (let i = 0; i < N; i++) this.W[i * N + i] = 0;
    }

    projection() {
      const { P, N, patterns: X } = this;
      const G = new Float64Array(P * P);
      for (let a = 0; a < P; a++)
        for (let b = 0; b < P; b++) {
          let s = 0;
          for (let k = 0; k < N; k++) s += X[a][k] * X[b][k];
          G[a * P + b] = s;
        }
      const Ginv = invert(G, P);
      const A = Array.from({ length: P }, () => new Float64Array(N));
      for (let a = 0; a < P; a++)
        for (let b = 0; b < P; b++) {
          const g = Ginv[a * P + b];
          for (let k = 0; k < N; k++) A[a][k] += g * X[b][k];
        }
      const W = new Float64Array(N * N);
      for (let a = 0; a < P; a++)
        for (let i = 0; i < N; i++) {
          const xi = X[a][i];
          for (let j = 0; j < N; j++) W[i * N + j] += xi * A[a][j];
        }
      return W;
    }

    field(s, i) {
      const N = this.N;
      let h = 0;
      for (let j = 0; j < N; j++) h += this.W[i * N + j] * s[j];
      return h;
    }

    activate(h, prev) {
      if (Math.abs(h) <= TIE_TOL) return prev;
      return h > 0 ? 1 : -1;
    }

    /** Synchronous step: returns a new state. */
    syncStep(s) {
      const out = new Float64Array(this.N);
      for (let i = 0; i < this.N; i++) out[i] = this.activate(this.field(s, i), s[i]);
      return out;
    }

    /** Asynchronous update of neuron i in place; returns true if it flipped. */
    updateNeuron(s, i) {
      const v = this.activate(this.field(s, i), s[i]);
      const changed = v !== s[i];
      s[i] = v;
      return changed;
    }

    energy(s) {
      const N = this.N;
      let e = 0;
      for (let i = 0; i < N; i++) e += s[i] * this.field(s, i);
      return -0.5 * e;
    }

    overlaps(s) {
      return this.patterns.map((x) => {
        let m = 0;
        for (let i = 0; i < this.N; i++) m += x[i] * s[i];
        return m / this.N;
      });
    }

    isStable(x) {
      for (let i = 0; i < this.N; i++) if (this.activate(this.field(x, i), x[i]) !== x[i]) return false;
      return true;
    }
  }

  function corrupt(x, fraction, rng) {
    const s = Float64Array.from(x);
    const idx = shuffle(Array.from(x, (_, i) => i), rng);
    const k = Math.round(fraction * x.length);
    for (let i = 0; i < k; i++) s[idx[i]] *= -1;
    return s;
  }

  function randomState(n, rng) {
    return Float64Array.from({ length: n }, () => (rng() < 0.5 ? -1 : 1));
  }

  function equal(a, b) {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  global.HF = { Network, mulberry32, shuffle, parseBits, decodeFloat32, corrupt, randomState, equal };
})(window);
