import numpy as np
import pytest

from hopfield import Hopfield, LowRankHopfield
from hopfield_datasets import DATASETS, load_dataset, shape_from_name


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def walsh(n: int) -> np.ndarray:
    """Rows of a Sylvester Hadamard matrix: mutually orthogonal +-1 vectors."""
    H = np.array([[1.0]])
    while len(H) < n:
        H = np.block([[H, H], [H, -H]])
    return H


def test_hebb_stores_single_pattern(rng):
    x = rng.choice([-1.0, 1.0], 30)
    net = Hopfield((5, 6), x)
    net.hebbian_train()
    assert net.is_stable(x)
    assert net.is_stable(-x)
    noisy = net.corrupt(x, 0.2, rng)
    assert np.array_equal(net.run(noisy).state, x)


def test_hebb_stores_orthogonal_patterns():
    X = walsh(16)[1:8]
    net = Hopfield((4, 4), X)
    net.hebbian_train()
    assert net.stable_patterns(strict=True).all()


def test_hebb_is_not_cumulative(rng):
    net = Hopfield((3, 3), rng.choice([-1.0, 1.0], (2, 9)))
    first = net.hebbian_train().copy()
    assert np.array_equal(first, net.hebbian_train())
    assert np.allclose(first, first.T)
    assert np.all(np.diag(first) == 0)


def test_zero_field_keeps_state():
    net = Hopfield((1, 2), [[1, -1]])
    net.weights = np.zeros((2, 2))
    s = np.array([1.0, -1.0])
    assert np.array_equal(net.recall(s), s)
    assert np.array_equal(net.recall_async(s, np.random.default_rng(0)), s)


@pytest.mark.parametrize("rule", ["hebb", "projection", "oja", "delta"])
def test_async_energy_never_increases(rule, rng):
    d = load_dataset("animals-14x9")
    net = Hopfield(d.shape, d.patterns)
    net.train(rule)
    for _ in range(10):
        result = net.run(net.random_state(rng), mode="async", seed=rng)
        energies = [net.energy(s) for s in result.history]
        assert np.all(np.diff(energies) <= 1e-9)
        assert result.converged


def test_sync_two_cycle_detected():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    s = np.array([1.0, 1.0, 1.0, 1.0])  # x . s = 0, so h = -s / N
    net = Hopfield((2, 2), x)
    net.hebbian_train()
    result = net.run(s, mode="sync")
    assert result.cycle_length == 2
    assert np.array_equal(result.history[1], -s)
    assert net.run(s, mode="async", seed=0).converged


def test_projection_stores_correlated_patterns():
    d = load_dataset("letters-14x20")
    net = Hopfield(d.shape, d.patterns)
    net.projection_train()
    assert net.stable_patterns().all()
    net.delta_train(epochs=50)
    assert net.stable_patterns().all()


def test_oja_moves_towards_principal_component():
    d = load_dataset("small-7x7")
    net = Hopfield(d.shape, d.patterns)

    def top_share(W):
        s = np.linalg.svd(W, compute_uv=False)
        return s[0] ** 2 / (s**2).sum()

    net.hebbian_train()
    hebb_alignment, hebb_share = net.pc_alignment().mean(), top_share(net.weights)
    net.oja_train(epochs=300)
    assert np.isfinite(net.weights).all()
    assert net.pc_alignment().mean() > hebb_alignment + 0.1
    assert top_share(net.weights) > hebb_share


@pytest.mark.parametrize("rule", ["hebb", "projection"])
def test_low_rank_matches_dense(rule, rng):
    d = load_dataset("animals-14x9")
    dense = Hopfield(d.shape, d.patterns)
    dense.train(rule)
    low = LowRankHopfield(d.shape, d.patterns, rule=rule)
    for k in range(10):
        s = dense.random_state(rng)
        assert np.allclose(dense.local_field(s), low.local_field(s))
        assert np.isclose(dense.energy(s), low.energy(s))
        assert np.array_equal(dense.run(s).state, low.run(s).state)
        a = dense.run(s, mode="async", seed=k).state
        b = low.run(s, mode="async", seed=k).state
        assert np.array_equal(a, b)
    assert np.array_equal(dense.stable_patterns(), low.stable_patterns())
    assert np.allclose(low.weight_row(3), dense.weights[3])


def test_shape_from_name():
    assert shape_from_name("animals-14x9") == (9, 14)
    assert shape_from_name("large-25x25.plus") == (25, 25)
    assert shape_from_name("OCRA-12x30-cut") == (30, 12)


@pytest.mark.parametrize("name", DATASETS)
def test_datasets_load(name):
    d = load_dataset(name)
    assert d.patterns.shape[1] == d.shape[0] * d.shape[1]
    assert np.isin(d.patterns, (-1, 1)).all()


def test_rejects_bad_patterns():
    with pytest.raises(ValueError):
        Hopfield((2, 2), [[1, -1, 1]])
    with pytest.raises(ValueError):
        Hopfield((1, 3), [[1, 0, 1]])
