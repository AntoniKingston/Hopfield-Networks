from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

Mode = Literal["sync", "async"]


@dataclass
class RecallResult:
    state: np.ndarray
    history: list[np.ndarray] = field(default_factory=list)
    converged: bool = False
    cycle_length: int | None = None

    @property
    def steps(self) -> int:
        return len(self.history) - 1


class HopfieldBase(ABC):
    """Dynamics, analysis and visualization shared by the dense and low-rank networks.

    Subclasses define the weights through `local_field(state)` (h = W s) and an
    asynchronous sweep that updates neurons one at a time.
    """

    def __init__(self, shape: tuple[int, int], patterns: np.ndarray, dtype=np.float64):
        """`shape` is (rows, cols); a file named `name-WxH.csv` has shape (H, W)."""
        self.shape = tuple(shape)
        self.size = self.shape[0] * self.shape[1]
        self.dtype = dtype
        # Hebbian fields are multiples of 1/N, so exact ties (h = 0) are common; rounding errors
        # below this tolerance must not decide the sign.
        self.tie_tol = 1e-9 if np.dtype(dtype).itemsize >= 8 else 1e-5

        X = np.atleast_2d(np.asarray(patterns, dtype=dtype))
        if X.shape[1] != self.size:
            raise ValueError(f"patterns have {X.shape[1]} values, expected {self.size}")
        if not np.isin(X, (-1, 1)).all():
            raise ValueError("patterns must contain only -1 and 1")
        self.patterns = X

    @classmethod
    def from_csv(cls, path: str | Path, shape: tuple[int, int], **kwargs):
        return cls(shape, np.loadtxt(path, delimiter=","), **kwargs)

    @abstractmethod
    def local_field(self, state: np.ndarray) -> np.ndarray:
        """h = W s for the whole state."""

    @abstractmethod
    def update_neurons(self, state: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Asynchronous updates of the neurons in `order`, one at a time (returns a new state)."""

    def recall_async(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One asynchronous sweep: every neuron is updated once, in random order."""
        return self.update_neurons(state, rng.permutation(self.size))

    # ---------- dynamics ----------

    def _activate(self, h: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Sign function that keeps the previous value when the local field is (almost) 0."""
        tol = self.tie_tol
        return np.where(h > tol, 1, np.where(h < -tol, -1, previous)).astype(previous.dtype)

    def _as_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=self.dtype)

    def recall(self, state: np.ndarray) -> np.ndarray:
        """One synchronous step: all neurons are updated at once."""
        state = self._as_state(state)
        return self._activate(self.local_field(state), state)

    def run(
        self,
        state: np.ndarray,
        mode: Mode = "sync",
        max_iter: int = 100,
        seed: int | np.random.Generator | None = None,
        frames_per_sweep: int = 1,
    ) -> RecallResult:
        """Iterate until a fixed point, a cycle, or `max_iter` steps (sync) / sweeps (async).

        History has one entry per synchronous step; an asynchronous sweep is recorded as
        `frames_per_sweep` snapshots, so the gradual updates can be watched.
        """
        rng = np.random.default_rng(seed)
        state = self._as_state(state)
        history = [state]
        for _ in range(max_iter):
            if mode == "sync":
                state = self.recall(state)
                if np.array_equal(state, history[-1]):
                    return RecallResult(state, history, converged=True)
                history.append(state)
                if len(history) >= 3 and np.array_equal(state, history[-3]):
                    return RecallResult(state, history, cycle_length=2)
            else:
                start, frames = state, []
                for chunk in np.array_split(rng.permutation(self.size), frames_per_sweep):
                    state = self.update_neurons(state, chunk)
                    frames.append(state)
                if np.array_equal(state, start):
                    return RecallResult(state, history, converged=True)
                history.extend(frames)
        return RecallResult(state, history)

    def recall_until_convergence(self, state: np.ndarray, **kwargs) -> np.ndarray:
        return self.run(state, **kwargs).state

    # ---------- analysis ----------

    def energy(self, state: np.ndarray) -> float:
        state = self._as_state(state)
        return float(-0.5 * state @ self.local_field(state))

    def stability_margins(self) -> np.ndarray:
        """x_i * h_i for every training pattern and neuron (P x N); > 0 means the bit is stable."""
        H = np.stack([self.local_field(x) for x in self.patterns])
        return self.patterns * H

    def stable_patterns(self, strict: bool = False) -> np.ndarray:
        """Boolean mask of training patterns that are fixed points.

        With `strict=False` a zero local field counts as stable (the neuron keeps its state).
        """
        margins = self.stability_margins()
        if strict:
            return (margins > self.tie_tol).all(axis=1)
        return (margins >= -self.tie_tol).all(axis=1)

    def is_stable(self, state: np.ndarray) -> bool:
        state = self._as_state(state)
        return bool(np.array_equal(self.recall(state), state))

    def match(self, state: np.ndarray) -> int | None:
        """Index of the training pattern equal to `state` or to its inverse, else None."""
        for i, x in enumerate(self.patterns):
            if np.array_equal(state, x) or np.array_equal(state, -x):
                return i
        return None

    def overlaps(self, state: np.ndarray) -> np.ndarray:
        """m_mu = x_mu . s / N for every training pattern."""
        return self.patterns @ self._as_state(state) / self.size

    @staticmethod
    def corrupt(pattern: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
        """Flip `fraction` of the pixels chosen uniformly at random."""
        pattern = np.array(pattern)
        flip = rng.choice(pattern.size, size=round(fraction * pattern.size), replace=False)
        pattern[flip] *= -1
        return pattern

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(np.array([-1, 1], dtype=self.dtype), size=self.size)

    # ---------- visualization ----------

    def plot_pattern(self, pattern: np.ndarray, ax=None, title: str | None = None):
        show = ax is None
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(np.asarray(pattern).reshape(self.shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        if show:
            plt.show()

    def plot_history(self, history: list[np.ndarray], cols: int = 6, show: bool = True):
        rows = -(-len(history) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k < len(history):
                self.plot_pattern(history[k], ax=ax, title=f"krok {k}")
            else:
                ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def show_history(self, history: list[np.ndarray]):
        """Interactive slider over the convergence steps (for Jupyter)."""
        from ipywidgets import IntSlider, interact

        def draw(step: int):
            self.plot_pattern(
                history[step], title=f"krok {step}, E = {self.energy(history[step]):.3f}"
            )

        interact(draw, step=IntSlider(min=0, max=len(history) - 1, value=0))

    def save_history_gif(self, history: list[np.ndarray], path: str | Path, fps: int = 2):
        """Animated GIF of the convergence steps (one frame per step/sweep)."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots()
        image = ax.imshow(history[0].reshape(self.shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

        def update(k: int):
            image.set_data(history[k].reshape(self.shape))
            ax.set_title(f"krok {k}, E = {self.energy(history[k]):.3f}")
            return (image,)

        FuncAnimation(fig, update, frames=len(history)).save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)


class Hopfield(HopfieldBase):
    """Hopfield network with an explicit N x N weight matrix."""

    def __init__(self, shape: tuple[int, int], patterns: np.ndarray, dtype=np.float64):
        super().__init__(shape, patterns, dtype)
        self.weights = np.zeros((self.size, self.size), dtype=dtype)

    def local_field(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ self._as_state(state)

    def update_neurons(self, state: np.ndarray, order: np.ndarray) -> np.ndarray:
        state = np.array(state, dtype=self.dtype)
        for i in order:
            if self.weights[i] @ state * state[i] < -self.tie_tol:
                state[i] = -state[i]
        return state

    # ---------- training ----------

    def hebbian_train(self) -> np.ndarray:
        X = self.patterns
        self.weights = (X.T @ X / self.size).astype(self.dtype)
        np.fill_diagonal(self.weights, 0)
        return self.weights

    def projection_train(self) -> np.ndarray:
        """Pseudo-inverse rule W = X^T (X X^T)^+ X (Personnaz et al.), with zero diagonal."""
        X = self.patterns.astype(np.float64)
        self.weights = (X.T @ np.linalg.pinv(X @ X.T) @ X).astype(self.dtype)
        np.fill_diagonal(self.weights, 0)
        return self.weights

    def normal_initialization(self, variance: float, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, np.sqrt(variance), (self.size, self.size)).astype(self.dtype)
        np.fill_diagonal(self.weights, 0)
        return self.weights

    def _init_weights(self, init: str, seed: int):
        if init == "hebb":
            self.hebbian_train()
        elif init == "random":
            self.normal_initialization(1 / self.size, seed)
        elif init == "zeros":
            self.weights = np.zeros((self.size, self.size), dtype=self.dtype)
        elif init != "current":
            raise ValueError(f"unknown init: {init}")

    def _iterative_train(self, update, epochs, init, symmetric, tol, shuffle, seed, info_interval):
        self._init_weights(init, seed)
        rng = np.random.default_rng(seed)
        W = self.weights
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(self.patterns)) if shuffle else range(len(self.patterns))
            max_change = 0.0
            for x in self.patterns[order]:
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = update(W, x)
                np.fill_diagonal(delta, 0)
                if symmetric:
                    delta = (delta + delta.T) / 2
                W += delta
                max_change = max(max_change, float(np.abs(delta).max()))

            if not np.isfinite(W).all():
                raise FloatingPointError(f"weights diverged in epoch {epoch}; lower learning_rate")
            if info_interval and epoch % info_interval == 0:
                print(f"Epoch {epoch}: max weight change {max_change:.2e}")
            if max_change < tol:
                break
        return W

    def oja_train(
        self,
        learning_rate: float | None = None,
        epochs: int = 100,
        init: Literal["hebb", "random", "zeros", "current"] = "hebb",
        symmetric: bool = True,
        tol: float = 1e-6,
        shuffle: bool = True,
        seed: int = 42,
        info_interval: int | None = None,
    ) -> np.ndarray:
        """Oja's rule applied to every neuron: dw_ij = lr * y_i * (x_j - y_i * w_ij), y = W x.

        Stops early when the largest single weight change in an epoch drops below `tol`.
        The default rate is 0.1 / N: one update changes y_i by about lr * |x|^2 * y_i =
        lr * N * y_i, so the rule diverges once lr * N is of order 1.
        """
        lr = 0.1 / self.size if learning_rate is None else learning_rate

        def update(W, x):
            y = W @ x
            return lr * (np.outer(y, x) - (y**2)[:, None] * W)

        return self._iterative_train(
            update, epochs, init, symmetric, tol, shuffle, seed, info_interval
        )

    def delta_train(
        self,
        learning_rate: float = 0.5,
        epochs: int = 100,
        init: Literal["hebb", "random", "zeros", "current"] = "zeros",
        symmetric: bool = True,
        tol: float = 1e-6,
        shuffle: bool = True,
        seed: int = 42,
        info_interval: int | None = None,
    ) -> np.ndarray:
        """Delta (Widrow-Hoff) rule dW = (lr / N) * (x - W x) x^T.

        The step is divided by N because the update is stable only for lr * |x|^2 / N < 2.
        Started from zeros, it converges to the projection rule.
        """
        step = learning_rate / self.size

        def update(W, x):
            return step * np.outer(x - W @ x, x)

        return self._iterative_train(
            update, epochs, init, symmetric, tol, shuffle, seed, info_interval
        )

    def train(self, rule: str, **kwargs) -> np.ndarray:
        rules = {
            "hebb": self.hebbian_train,
            "oja": self.oja_train,
            "delta": self.delta_train,
            "projection": self.projection_train,
        }
        if rule not in rules:
            raise ValueError(f"unknown rule: {rule}")
        return rules[rule](**kwargs)

    # ---------- diagnostics ----------

    def principal_component(self) -> np.ndarray:
        """First eigenvector of the (uncentered) pattern correlation matrix X^T X / P."""
        _, _, Vt = np.linalg.svd(self.patterns.astype(np.float64), full_matrices=False)
        return Vt[0]

    def pc_alignment(self) -> np.ndarray:
        """|cos| between every weight row and the first principal component of the patterns."""
        v = self.principal_component()
        norms = np.linalg.norm(self.weights, axis=1)
        return np.abs(self.weights @ v) / np.where(norms > 0, norms, 1)

    def weight_rank(self, rtol: float = 1e-3) -> int:
        s = np.linalg.svd(self.weights.astype(np.float64), compute_uv=False)
        return int((s > rtol * s[0]).sum())


class LowRankHopfield(HopfieldBase):
    """Hopfield network that never builds W, for bitmaps too large for an N x N matrix.

    Both supported rules have the form W = A^T X - diag(d), so h = A^T (X s) - d * s is computed
    through the P overlaps m = X s:
      hebb:        A = X / N,             d_i = P / N
      projection:  A = (X X^T)^+ X,       d_i = x_i^T (X X^T)^+ x_i   (x_i = column i of X)
    Memory is O(P N) instead of O(N^2).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        patterns: np.ndarray,
        rule: Literal["hebb", "projection"] = "hebb",
        dtype=np.float64,
    ):
        super().__init__(shape, patterns, dtype)
        self.tie_tol = 1e-9
        self.train(rule)

    def train(self, rule: Literal["hebb", "projection"]):
        X = self.patterns.astype(np.float64)
        if rule == "hebb":
            A = X / self.size
        elif rule == "projection":
            A = np.linalg.pinv(X @ X.T) @ X
        else:
            raise ValueError(f"rule {rule!r} is not supported without an explicit weight matrix")
        self.rule = rule
        # O(P N) arrays, so float64 is cheap and keeps rounding far below `tie_tol`.
        self._X = X
        self._AT = np.ascontiguousarray(A.T)
        self._XT = np.ascontiguousarray(X.T)
        self._d = (X * A).sum(axis=0)

    def local_field(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        return self._AT @ (self._X @ state) - self._d * state

    def update_neurons(self, state: np.ndarray, order: np.ndarray) -> np.ndarray:
        state = np.array(state, dtype=np.float64)
        m = self._X @ state
        for i in order:
            h = self._AT[i] @ m - self._d[i] * state[i]
            if h * state[i] < -self.tie_tol:
                state[i] = -state[i]
                m += (2 * state[i]) * self._XT[i]
        return state.astype(self.dtype)

    def weight_row(self, i: int) -> np.ndarray:
        row = self._AT[i] @ self._X
        row[i] = 0
        return row.astype(self.dtype)
