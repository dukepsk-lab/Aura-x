"""L6b allocation (Roadmap v3) — Dirichlet-policy PPO allocator."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from aurax.config import load_params
from aurax.l6_risk import (
    AllocationConfig,
    DirichletAllocator,
    covariance_risk,
    portfolio_log_return,
    portfolio_turnover,
    reward,
)

_NO_TORCH = importlib.util.find_spec("torch") is None
_deep = pytest.mark.skipif(_NO_TORCH, reason="torch (the `deep` extra) not installed")


# --- pure reward components (dependency-free) ---------------------------------
def test_reward_increases_with_log_return_decreases_with_costs():
    base = reward(0.01, 0.0, 0.0)
    assert reward(0.02, 0.0, 0.0) > base
    assert reward(0.01, 1.0, 0.0) < base
    assert reward(0.01, 0.0, 1.0) < base
    assert base == pytest.approx(0.01)


def test_covariance_risk_rewards_diversification():
    w = np.array([0.5, 0.5])
    correlated = np.array([[1e-4, 0.9e-4], [0.9e-4, 1e-4]])
    uncorrelated = np.array([[1e-4, 0.0], [0.0, 1e-4]])
    assert covariance_risk(w, correlated) > covariance_risk(w, uncorrelated)
    # degenerate (non-PSD) input never produces a negative sqrt argument
    assert covariance_risk(np.array([1.0, -1.0]), np.array([[1.0, 5.0], [5.0, 1.0]])) >= 0.0


def test_portfolio_turnover_and_log_return():
    assert portfolio_turnover(np.array([0.6, 0.4]), np.array([0.5, 0.5])) == pytest.approx(0.2)
    assert portfolio_turnover(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert portfolio_log_return(np.array([1.0, 0.0]), np.array([0.01, 0.05])) == pytest.approx(np.log1p(0.01))


def test_allocation_config_from_params_reads_default_yaml():
    cfg = AllocationConfig.from_params(load_params())
    assert cfg.turnover_penalty == pytest.approx(0.001)
    assert cfg.risk_penalty == pytest.approx(0.5)
    assert cfg.hidden == 32
    # missing section → dataclass defaults
    assert AllocationConfig.from_params({}) == AllocationConfig()


def test_allocate_before_fit_raises():
    with pytest.raises(RuntimeError):
        DirichletAllocator().allocate(np.array([0.01, -0.01]), np.eye(2) * 1e-4)


# --- PPO training (torch-gated) -----------------------------------------------
def _two_instrument_task(t=60, seed=0):
    """A is a clear, low-noise edge (mean +1%); B is a clear loser (mean -0.5%);
    zero cross-correlation. The optimal stationary policy overweights A."""
    rng = np.random.default_rng(seed)
    expected = np.tile(np.array([0.010, -0.005]), (t, 1))
    covariance = np.array([[1e-4, 0.0], [0.0, 1e-4]])
    realized = expected + rng.normal(0.0, 0.001, size=(t, 2))
    return expected, covariance, realized


@_deep
def test_fit_produces_valid_simplex_weights():
    expected, covariance, realized = _two_instrument_task()
    alloc = DirichletAllocator(AllocationConfig(hidden=16, epochs_per_update=4)).fit(
        expected, covariance, realized, n_updates=20
    )
    w = alloc.allocate(expected[0], covariance)
    assert w.shape == (2,)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-5)


@_deep
def test_allocator_learns_to_prefer_the_better_instrument():
    expected, covariance, realized = _two_instrument_task(t=60)
    cfg = AllocationConfig(hidden=16, epochs_per_update=4, seed=0)
    alloc = DirichletAllocator(cfg).fit(expected, covariance, realized, n_updates=50)

    # mean step reward improved over training (the policy actually learned).
    assert alloc.reward_history_[-1] > alloc.reward_history_[0]

    w = alloc.allocate(expected[0], covariance)
    assert w[0] > w[1]  # overweights the higher-expected-return, equal-risk instrument


@_deep
def test_allocate_tracks_turnover_reference_and_reset():
    expected, covariance, realized = _two_instrument_task()
    alloc = DirichletAllocator(AllocationConfig(hidden=16, epochs_per_update=4)).fit(
        expected, covariance, realized, n_updates=15
    )
    alloc.allocate(expected[0], covariance)
    assert not np.allclose(alloc._prev_weights, 0.5)  # moved off the equal-weight start
    alloc.reset()
    np.testing.assert_allclose(alloc._prev_weights, [0.5, 0.5])


@_deep
def test_allocate_wrong_instrument_count_raises():
    expected, covariance, realized = _two_instrument_task()
    alloc = DirichletAllocator(AllocationConfig(hidden=16)).fit(expected, covariance, realized, n_updates=5)
    with pytest.raises(ValueError):
        alloc.allocate(np.array([0.01, 0.02, 0.03]), np.eye(3))


@_deep
def test_fit_rejects_mismatched_shapes():
    expected, covariance, realized = _two_instrument_task()
    t = len(expected)
    with pytest.raises(ValueError):
        DirichletAllocator().fit(expected, covariance, realized[:-1], n_updates=1)
    with pytest.raises(ValueError):  # static (K,K) covariance with the wrong K
        DirichletAllocator().fit(expected, np.eye(3), realized, n_updates=1)
    with pytest.raises(ValueError):  # per-step (T,K,K) covariance with the wrong T
        DirichletAllocator().fit(expected, np.tile(np.eye(2), (t - 1, 1, 1)), realized, n_updates=1)


@_deep
def test_allocator_is_deterministic_given_seed():
    expected, covariance, realized = _two_instrument_task()
    cfg = AllocationConfig(hidden=16, epochs_per_update=4, seed=7)
    w1 = DirichletAllocator(cfg).fit(expected, covariance, realized, n_updates=15).allocate(expected[0], covariance)
    w2 = DirichletAllocator(cfg).fit(expected, covariance, realized, n_updates=15).allocate(expected[0], covariance)
    np.testing.assert_allclose(w1, w2, atol=1e-5)
