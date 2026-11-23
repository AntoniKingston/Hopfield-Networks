"""One function per question of Project 2. Every function is deterministic for a given seed."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from hopfield import Hopfield, HopfieldBase, LowRankHopfield, RecallResult
from hopfield_datasets import (
    DATASETS,
    Dataset,
    downsample,
    load_dataset,
    pattern_stats,
)

SEED = 2026
RULES = ["hebb", "oja", "projection"]
MODES = ["sync", "async"]
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
OJA_PARAMS = {"epochs": 100, "init": "hebb"}  # learning rate 0.1 / N

_cache: dict[tuple, Hopfield] = {}


def trained_network(dataset: Dataset, rule: str, seed: int = SEED, **params) -> Hopfield:
    """Dense network trained with `rule`; Oja uses OJA_PARAMS unless overridden."""
    key = (dataset.name, rule, seed, tuple(sorted(params.items())))
    if key not in _cache:
        net = Hopfield(dataset.shape, dataset.patterns)
        if rule == "oja":
            net.oja_train(seed=seed, **{**OJA_PARAMS, **params})
        elif rule == "delta":
            net.delta_train(seed=seed, **params)
        else:
            net.train(rule, **params)
        _cache[key] = net
    return _cache[key]


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, *keys])


def classify(net: HopfieldBase, result: RecallResult, target: int | None = None) -> str:
    """exact / inverse (of `target`), other (another stored pattern or its inverse), spurious,
    cycle, unconverged. Without a target, stored patterns are `stored` or `inverse`."""
    if result.cycle_length:
        return "cycle"
    if not result.converged:
        return "unconverged"
    idx = net.match(result.state)
    if idx is None:
        return "spurious"
    positive = np.array_equal(result.state, net.patterns[idx])
    if target is None:
        return "stored" if positive else "inverse"
    if idx == target:
        return "exact" if positive else "inverse"
    return "other"


# ---------- Q1/Q2: stability of training patterns ----------


def dataset_table(names: Iterable[str] = DATASETS) -> pd.DataFrame:
    rows = []
    for name in names:
        d = load_dataset(name)
        rows.append(
            {"dataset": name, "shape": f"{d.shape[1]}x{d.shape[0]}"} | pattern_stats(d.patterns)
        )
    return pd.DataFrame(rows)


def stability_table(
    names: Iterable[str] = DATASETS,
    rules: Iterable[str] = ("hebb", "oja", "delta", "projection"),
    seed: int = SEED,
) -> pd.DataFrame:
    rows = []
    for name in names:
        d = load_dataset(name)
        for rule in rules:
            net = trained_network(d, rule, seed)
            margins = net.stability_margins()
            rows.append(
                {
                    "dataset": name,
                    "rule": rule,
                    "P": d.n_patterns,
                    "N": d.n_neurons,
                    "stable": int(net.stable_patterns().sum()),
                    "stable_strict": int(net.stable_patterns(strict=True).sum()),
                    "unstable_bits": float((margins < -net.tie_tol).mean()),
                    "pc_alignment": float(net.pc_alignment().mean()),
                }
            )
    return pd.DataFrame(rows)


def oja_trajectory(
    names: Iterable[str] = ("small-7x7", "animals-14x9", "large-25x25"),
    relative_rates: Iterable[float] = (0.01, 0.1, 1.0, 3.0),
    inits: Iterable[str] = ("hebb", "random"),
    epochs: int = 100,
    seed: int = SEED,
) -> pd.DataFrame:
    """Stable patterns and alignment with the first principal component after every Oja epoch.

    The learning rate is `relative_rate / N`; diverging runs end with a row with `diverged`.
    """
    rows = []
    for name in names:
        d = load_dataset(name)
        for init in inits:
            for c in relative_rates:
                net = Hopfield(d.shape, d.patterns)
                net._init_weights(init, seed)
                for epoch in range(epochs + 1):
                    if epoch > 0:
                        try:
                            net.oja_train(
                                learning_rate=c / d.n_neurons,
                                epochs=1,
                                init="current",
                                seed=seed + epoch,
                            )
                        except FloatingPointError:
                            rows.append(
                                {
                                    "dataset": name,
                                    "init": init,
                                    "relative_rate": c,
                                    "epoch": epoch,
                                    "diverged": True,
                                }
                            )
                            break
                    s = np.linalg.svd(net.weights, compute_uv=False)
                    rows.append(
                        {
                            "dataset": name,
                            "init": init,
                            "relative_rate": c,
                            "diverged": False,
                            "epoch": epoch,
                            "stable": int(net.stable_patterns().sum()),
                            "pc_alignment": float(net.pc_alignment().mean()),
                            "top_singular_share": float(s[0] ** 2 / (s**2).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def random_capacity(
    sizes: Iterable[int] = (100, 400),
    loads: Iterable[float] = (0.02, 0.05, 0.08, 0.1, 0.12, 0.138, 0.16, 0.2, 0.3),
    trials: int = 10,
    seed: int = SEED,
) -> pd.DataFrame:
    """Hebb rule on random unbiased patterns: fraction of unstable bits and of stable patterns
    as a function of the load P/N (reference curve for the theory)."""
    rows = []
    for n in sizes:
        rng = _rng(seed, 10, n)
        for load in loads:
            p = max(1, round(load * n))
            bits, patterns, attracted = [], [], []
            for _ in range(trials):
                X = rng.choice([-1.0, 1.0], (p, n))
                margins = hebb_margins(X)
                bits.append((margins < 0).mean())
                patterns.append((margins >= 0).all(axis=1).mean())
                net = Hopfield((1, n), X)
                net.hebbian_train()
                final = net.run(X[0], mode="async", seed=rng).state
                attracted.append(X[0] @ final / n)
            rows.append(
                {
                    "N": n,
                    "load": p / n,
                    "unstable_bits": float(np.mean(bits)),
                    "stable_patterns": float(np.mean(patterns)),
                    "final_overlap": float(np.mean(attracted)),
                }
            )
    return pd.DataFrame(rows)


# ---------- Q3: 25x25 vs 25x50 ----------


def variants_25x50(seed: int = SEED) -> list[Dataset]:
    """25x25, 25x50 (= 25x25 plus a constant +1 block), and a control where the extra block
    holds independent random pixels instead of the shared constant."""
    small = load_dataset("large-25x25")
    large = load_dataset("large-25x50")
    rng = _rng(seed, 3)
    random_half = np.hstack([small.patterns, rng.choice([-1.0, 1.0], small.patterns.shape)])
    return [
        small,
        large,
        Dataset("large-25x25 + losowa połowa", random_half, large.shape),
    ]


def compare_25x25_25x50(seed: int = SEED) -> pd.DataFrame:
    small, large = load_dataset("large-25x25"), load_dataset("large-25x50")
    top = large.patterns[:, : small.n_neurons]
    bottom = large.patterns[:, small.n_neurons :]
    print(
        "25x50 top half equals 25x25:",
        np.array_equal(top, small.patterns),
        "| bottom half constant +1:",
        bool((bottom == 1).all()),
    )
    rows = []
    for d in variants_25x50(seed):
        row = {"dataset": d.name} | pattern_stats(d.patterns)
        for rule in RULES:
            row[f"stable_{rule}"] = int(trained_network(d, rule, seed).stable_patterns().sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Q4/Q5: recovery from damaged input ----------


def noise_recovery(
    datasets: Iterable[Dataset],
    rules: Iterable[str] = RULES,
    modes: Iterable[str] = MODES,
    noise_levels: Iterable[float] = NOISE_LEVELS,
    trials: int = 10,
    max_iter: int = 50,
    seed: int = SEED,
    networks: dict[tuple[str, str], HopfieldBase] | None = None,
) -> pd.DataFrame:
    """Corrupt every pattern by flipping a fraction of pixels and record where the net ends up.

    The same corrupted inputs are used for every rule and mode.
    """
    rows = []
    for d_idx, d in enumerate(datasets):
        rng = _rng(seed, 4, d_idx)
        inputs = {
            (p, mu, t): HopfieldBase.corrupt(d.patterns[mu], p, rng)
            for p in noise_levels
            for mu in range(d.n_patterns)
            for t in range(trials)
        }
        for rule in rules:
            net = networks[(d.name, rule)] if networks else trained_network(d, rule, seed)
            for mode in modes:
                run_rng = _rng(seed, 5, d_idx)
                for (p, mu, t), x in inputs.items():
                    result = net.run(x, mode=mode, max_iter=max_iter, seed=run_rng)
                    rows.append(
                        {
                            "dataset": d.name,
                            "rule": rule,
                            "mode": mode,
                            "noise": p,
                            "pattern": mu,
                            "trial": t,
                            "outcome": classify(net, result, mu),
                            "overlap": float(d.patterns[mu] @ result.state / d.n_neurons),
                            "steps": result.steps,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of exact recalls and mean final overlap per dataset/rule/mode/noise."""
    keys = ["dataset", "rule", "mode", "noise"]
    out = df.groupby(keys).agg(
        exact=("outcome", lambda s: (s == "exact").mean()),
        exact_or_inverse=("outcome", lambda s: s.isin(["exact", "inverse"]).mean()),
        overlap=("overlap", "mean"),
        steps=("steps", "mean"),
    )
    return out.reset_index()


# ---------- Q6: random input ----------


def random_input_outcomes(
    names: Iterable[str] = DATASETS,
    rules: Iterable[str] = RULES,
    modes: Iterable[str] = MODES,
    n_inputs: int = 200,
    max_iter: int = 100,
    seed: int = SEED,
) -> pd.DataFrame:
    rows = []
    for d_idx, name in enumerate(names):
        d = load_dataset(name)
        rng = _rng(seed, 6, d_idx)
        inputs = [rng.choice([-1.0, 1.0], d.n_neurons) for _ in range(n_inputs)]
        for rule in rules:
            net = trained_network(d, rule, seed)
            for mode in modes:
                run_rng = _rng(seed, 7, d_idx)
                for k, x in enumerate(inputs):
                    result = net.run(x, mode=mode, max_iter=max_iter, seed=run_rng)
                    overlaps = net.overlaps(result.state)
                    rows.append(
                        {
                            "dataset": name,
                            "rule": rule,
                            "mode": mode,
                            "input": k,
                            "outcome": classify(net, result),
                            "steps": result.steps,
                            "energy": net.energy(result.state),
                            "max_abs_overlap": float(np.abs(overlaps).max()),
                            "closest": int(np.abs(overlaps).argmax()),
                            "final": result.state.astype(np.int8).tobytes(),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_random(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "rule", "mode"]
    counts = pd.crosstab([df[k] for k in keys], df["outcome"], normalize="index")
    distinct = df.groupby(keys)["final"].nunique().rename("distinct_final_states")
    steps = df.groupby(keys)["steps"].mean().rename("mean_steps")
    return counts.join(distinct).join(steps).reset_index()


# ---------- Q7: as many stable 5x5 patterns as possible ----------


def hebb_margins(X: np.ndarray) -> np.ndarray:
    W = X.T @ X / X.shape[1]
    np.fill_diagonal(W, 0)
    return X * (X @ W)


def _distinct(X: np.ndarray) -> bool:
    return len({tuple(r) for r in np.vstack([X, -X])}) == 2 * len(X)


def _anneal(X: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[np.ndarray, bool]:
    """Flip single bits to drive the number of non-strictly-stable bits to 0."""
    X = X.copy()
    violations = int((hebb_margins(X) <= 1e-9).sum())
    for t in range(iters):
        if violations == 0:
            break
        temperature = 1.0 - t / iters + 1e-3
        mu, i = rng.integers(len(X)), rng.integers(X.shape[1])
        X[mu, i] *= -1
        new = int((hebb_margins(X) <= 1e-9).sum())
        if new <= violations or rng.random() < np.exp((violations - new) / temperature):
            violations = new
        else:
            X[mu, i] *= -1
    return X, violations == 0 and _distinct(X)


def search_stable_5x5(
    n: int = 25, iters: int = 100_000, attempts: int = 10, seed: int = SEED
) -> tuple[np.ndarray, pd.DataFrame]:
    """Grow a set one random vector at a time, repairing it by simulated annealing, until no
    set of the next size with all patterns strictly stable under Hebb is found."""
    rng = _rng(seed, 7)
    X = rng.choice([-1.0, 1.0], (1, n))
    best, log = X, []
    while len(X) <= n:
        used = 0
        for _ in range(attempts):
            used += 1
            Y, ok = _anneal(X, rng, iters)
            if ok:
                break
        log.append({"P": len(X), "found": ok, "attempts": used})
        if not ok:
            break
        best = Y
        X = np.vstack([Y, rng.choice([-1.0, 1.0], (1, n))])
    return best, pd.DataFrame(log)


def projection_random_sets(
    n: int = 25, sizes: Iterable[int] = range(1, 26), trials: int = 20, seed: int = SEED
) -> pd.DataFrame:
    """Fraction of random sets of each size that are fully stable under the projection rule."""
    rng = _rng(seed, 8)
    rows = []
    for size in sizes:
        ok_proj = ok_hebb = 0
        for _ in range(trials):
            X = rng.choice([-1.0, 1.0], (size, n))
            net = Hopfield((5, n // 5), X)
            net.projection_train()
            ok_proj += net.stable_patterns(strict=True).all()
            ok_hebb += (hebb_margins(X) > 1e-9).all()
        rows.append({"P": size, "projection": ok_proj / trials, "hebb": ok_hebb / trials})
    return pd.DataFrame(rows)


# ---------- Q8: oscillation ----------


def oscillation_example(seed: int = SEED) -> dict:
    """A 6x6 net storing one pattern; an input orthogonal to it (x . s = 0) gives h = -s / N,
    so a synchronous step flips every neuron and the state alternates between s and -s."""
    shape = (6, 6)
    x = -np.ones(shape)
    for k in range(6):
        x[k, k] = x[k, 5 - k] = 1
    x = x.ravel()
    s = np.ones(36)
    s[18:] = -1  # upper half white, lower half black
    # make x . s = 0 by flipping pixels of s where it agrees with x
    agree = np.flatnonzero(s == x)
    s[agree[: (int(x @ s) // 2)]] *= -1
    assert x @ s == 0
    net = Hopfield(shape, x[None])
    net.hebbian_train()
    sync = net.run(s, mode="sync", max_iter=10)
    asyn = net.run(s, mode="async", max_iter=10, seed=seed)
    return {
        "net": net,
        "pattern": x,
        "input": s,
        "sync": sync,
        "async": asyn,
        "sync_energy": [net.energy(h) for h in sync.history],
        "async_energy": [net.energy(h) for h in asyn.history],
    }


# ---------- Q9: large bitmaps ----------


def large_bitmap_networks(cats: Dataset) -> dict[tuple[str, str], HopfieldBase]:
    return {
        (cats.name, rule): LowRankHopfield(cats.shape, cats.patterns, rule=rule)
        for rule in ("hebb", "projection")
    }


def occlude(pattern: np.ndarray, shape: tuple[int, int], fraction: float, rng) -> np.ndarray:
    """Replace the bottom `fraction` of the rows with random pixels."""
    img = np.array(pattern).reshape(shape)
    rows = round(fraction * shape[0])
    if rows:
        img[-rows:] = rng.choice([-1.0, 1.0], (rows, shape[1]))
    return img.ravel()


def large_bitmap_experiment(
    cats: Dataset,
    noise_levels: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.45),
    occlusions: Iterable[float] = (0.25, 0.5, 0.75),
    oja_factor: int = 4,
    seed: int = SEED,
) -> dict:
    networks = large_bitmap_networks(cats)
    stable = {rule: int(net.stable_patterns().sum()) for (_, rule), net in networks.items()}
    noise = noise_recovery(
        [cats], ("hebb", "projection"), MODES, noise_levels, trials=2, seed=seed, networks=networks
    )

    rng = _rng(seed, 9)
    rows = []
    for frac in occlusions:
        for mu in range(cats.n_patterns):
            x = occlude(cats.patterns[mu], cats.shape, frac, rng)
            for (_, rule), net in networks.items():
                result = net.run(x, mode="sync", max_iter=30)
                rows.append(
                    {
                        "rule": rule,
                        "occlusion": frac,
                        "pattern": mu,
                        "outcome": classify(net, result, mu),
                        "overlap": float(cats.patterns[mu] @ result.state / cats.n_neurons),
                    }
                )
    occlusion = pd.DataFrame(rows)

    small = downsample(cats, oja_factor)
    small_rows = []
    for rule in RULES:
        net = trained_network(small, rule, seed)
        small_rows.append(
            {
                "dataset": small.name,
                "rule": rule,
                "N": small.n_neurons,
                "stable": int(net.stable_patterns().sum()),
                "pc_alignment": float(net.pc_alignment().mean()),
            }
        )
    return {
        "stable": stable,
        "noise": noise,
        "occlusion": occlusion,
        "downsampled": pd.DataFrame(small_rows),
        "small": small,
    }
