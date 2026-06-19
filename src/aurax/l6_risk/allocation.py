"""Layer 6b — Allocation (Roadmap v3): Dirichlet-policy PPO allocator.

For 2 instruments, full RL portfolio optimization is overkill — the §6 rules
(``RiskManager``) capture most of the value, and remain the default path. Once
the universe grows, this layer learns to allocate *across* instruments under
the same reward-engineering philosophy that motivates the rest of the system::

    maximize  log_return − λ₁·turnover − λ₂·risk(covariance)

:class:`DirichletAllocator` is a Dirichlet-policy PPO agent: its actor maps
(expected return, covariance, previous weights) to Dirichlet concentration
parameters, whose *sample* is a portfolio weight vector that natively
satisfies the simplex constraint (weights ≥ 0, sum to 1) — no projection step
needed. :func:`reward` is the pure objective both training and any offline
evaluation share.

Like the deep L3 ensemble members, this is a genuinely "heavy" model: PyTorch
is a lazy, optional dependency (the ``deep`` extra). The rest of the system
(L0–L8, including L6's ``RiskManager``) works fully without it — L6b is only
worthwhile once allocation across 3+ instruments is itself a real decision.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AllocationConfig:
    """Reward shaping (§6b) + PPO hyperparameters."""

    # reward: maximize log_return − turnover_penalty·turnover − risk_penalty·risk
    turnover_penalty: float = 0.001   # λ₁
    risk_penalty: float = 0.5         # λ₂
    # policy/value network + PPO
    hidden: int = 32
    lr: float = 1e-3
    gamma: float = 0.95
    gae_lambda: float = 0.90
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    epochs_per_update: int = 8
    max_grad_norm: float = 0.5
    alpha_floor: float = 1e-2          # keeps Dirichlet away from degenerate corners
    seed: int = 0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> AllocationConfig:
        a = params.get("allocation", {})
        return cls(
            turnover_penalty=a.get("turnover_penalty", 0.001),
            risk_penalty=a.get("risk_penalty", 0.5),
            hidden=a.get("hidden", 32),
            lr=a.get("lr", 1e-3),
            gamma=a.get("gamma", 0.95),
            gae_lambda=a.get("gae_lambda", 0.90),
            clip_eps=a.get("clip_eps", 0.2),
            entropy_coef=a.get("entropy_coef", 1e-3),
            value_coef=a.get("value_coef", 0.5),
            epochs_per_update=a.get("epochs_per_update", 8),
            max_grad_norm=a.get("max_grad_norm", 0.5),
            alpha_floor=a.get("alpha_floor", 1e-2),
            seed=a.get("seed", 0),
        )


def reward(
    log_return: float,
    turnover: float,
    covariance_risk: float,
    config: AllocationConfig | None = None,
) -> float:
    """The cost/risk-penalized allocation objective (fully specified, pure)."""
    cfg = config or AllocationConfig()
    return log_return - cfg.turnover_penalty * turnover - cfg.risk_penalty * covariance_risk


# --- pure formulas (the environment's reward components) ---------------------
def covariance_risk(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Portfolio covariance risk ``√(wᵀΣw)`` — same form as L6's correlation cap."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    return float(np.sqrt(max(0.0, w @ cov @ w)))


def portfolio_turnover(weights: np.ndarray, prev_weights: np.ndarray) -> float:
    """``Σ|Δweight|`` between consecutive allocations — the cost proxy λ₁ penalizes."""
    return float(np.abs(np.asarray(weights, dtype=float) - np.asarray(prev_weights, dtype=float)).sum())


def portfolio_log_return(weights: np.ndarray, realized: np.ndarray) -> float:
    """Log-return of the weighted portfolio for one step's realized returns."""
    r = float(np.dot(np.asarray(weights, dtype=float), np.asarray(realized, dtype=float)))
    return float(np.log1p(max(r, -0.999)))


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
        raise RuntimeError(
            "DirichletAllocator requires PyTorch. Install with:  pip install -e '.[deep]'"
        ) from exc
    return torch, nn, F


class DirichletAllocator:
    """Dirichlet-policy PPO allocator over the instrument simplex (Roadmap v3).

    ``fit`` trains on a historical ``(expected, covariance, realized)`` series
    via PPO (clipped surrogate objective, GAE advantages, an entropy bonus
    against premature collapse to a corner of the simplex). ``allocate``
    returns the Dirichlet *mean* (``α/Σα``), not a sample — live position
    sizing has no business being stochastic.
    """

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()
        self.n_instruments: int | None = None
        self.reward_history_: list[float] = []
        self._actor = None
        self._critic = None
        self._torch = None
        self._F = None
        self._prev_weights: np.ndarray | None = None

    # --- state representation -------------------------------------------------
    @staticmethod
    def _state(expected: np.ndarray, covariance: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(expected, dtype=np.float32).reshape(-1),
                np.asarray(covariance, dtype=np.float32).reshape(-1),
                np.asarray(prev_weights, dtype=np.float32).reshape(-1),
            ]
        )

    @staticmethod
    def _state_dim(k: int) -> int:
        return k + k * k + k  # expected (k) + flattened covariance (k²) + prev weights (k)

    def _alpha(self, raw):  # noqa: ANN001
        return self._F.softplus(raw) + self.config.alpha_floor

    def _build_networks(self, torch, nn, state_dim: int, k: int):  # noqa: ANN001
        hidden = self.config.hidden
        actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, k),
        )
        critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        return actor, critic

    @staticmethod
    def _broadcast_covariance(covariance: np.ndarray, t: int, k: int) -> np.ndarray:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim == 2:
            cov = np.broadcast_to(cov, (t, k, k))
        if cov.shape != (t, k, k):
            raise ValueError(f"covariance must be shaped ({t},{k},{k}) or ({k},{k}), got {cov.shape}")
        return cov

    # --- training ---------------------------------------------------------------
    def fit(
        self,
        expected: np.ndarray,
        covariance: np.ndarray,
        realized: np.ndarray,
        *,
        n_updates: int = 100,
    ) -> DirichletAllocator:
        """Train the Dirichlet PPO policy on a historical allocation sequence.

        ``expected``/``realized`` are ``(T, K)`` — forecasted vs. realized
        per-instrument returns at each step. ``covariance`` is ``(T, K, K)``
        (point-in-time estimates) or a single static ``(K, K)``, broadcast
        across steps. Each of the ``T`` steps is one allocation decision and
        turnover links consecutive steps, so every rollout walks the series in
        order (never shuffled); only the PPO *update* (over the fixed,
        already-collected rollout) reuses each step independently of its
        neighbours, which is what the clipped importance-sampling ratio is for.
        """
        torch, nn, F = _import_torch()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        cfg = self.config

        expected = np.asarray(expected, dtype=float)
        realized = np.asarray(realized, dtype=float)
        t, k = expected.shape
        if realized.shape != (t, k):
            raise ValueError("expected and realized must share shape (T, K)")
        covariance = self._broadcast_covariance(covariance, t, k)
        self.n_instruments = k

        actor, critic = self._build_networks(torch, nn, self._state_dim(k), k)
        opt = torch.optim.Adam(itertools.chain(actor.parameters(), critic.parameters()), lr=cfg.lr)
        self._torch, self._F = torch, F  # _alpha() needs these during the rollout below

        init_weights = np.full(k, 1.0 / k)
        self.reward_history_ = []
        for _ in range(n_updates):
            states, actions, logps, values, rewards = [], [], [], [], []
            pw = init_weights
            for step in range(t):
                state = self._state(expected[step], covariance[step], pw)
                state_t = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    alpha = self._alpha(actor(state_t))
                    dist = torch.distributions.Dirichlet(alpha)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    value = critic(state_t).squeeze(-1)
                w = action.numpy()
                r = reward(
                    portfolio_log_return(w, realized[step]),
                    portfolio_turnover(w, pw),
                    covariance_risk(w, covariance[step]),
                    cfg,
                )
                states.append(state)
                actions.append(w)
                logps.append(float(logp))
                values.append(float(value))
                rewards.append(r)
                pw = w
            self.reward_history_.append(float(np.mean(rewards)))

            # GAE advantages (finite-horizon rollout → bootstrap terminal value 0).
            values_arr = np.asarray([*values, 0.0])
            adv = np.zeros(t)
            last = 0.0
            for i in reversed(range(t)):
                delta = rewards[i] + cfg.gamma * values_arr[i + 1] - values_arr[i]
                last = delta + cfg.gamma * cfg.gae_lambda * last
                adv[i] = last
            returns = adv + values_arr[:t]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
            old_logp_t = torch.as_tensor(np.asarray(logps), dtype=torch.float32)
            adv_t = torch.as_tensor(adv, dtype=torch.float32)
            returns_t = torch.as_tensor(returns, dtype=torch.float32)

            for _ in range(cfg.epochs_per_update):
                alpha = self._alpha(actor(states_t))
                dist = torch.distributions.Dirichlet(alpha)
                new_logp = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
                ratio = torch.exp(torch.clamp(new_logp - old_logp_t, -20.0, 20.0))
                surr = torch.min(ratio * adv_t, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t)
                policy_loss = -surr.mean()
                value_loss = F.mse_loss(critic(states_t).squeeze(-1), returns_t)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    itertools.chain(actor.parameters(), critic.parameters()), cfg.max_grad_norm
                )
                opt.step()

        self._actor, self._critic = actor, critic
        self._prev_weights = init_weights
        return self

    # --- inference ---------------------------------------------------------------
    def allocate(self, expected: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Simplex weights (≥0, sum to 1) for one step's expected return + covariance."""
        if self._actor is None:
            raise RuntimeError("DirichletAllocator.fit(...) must be called before allocate()")
        expected = np.asarray(expected, dtype=float)
        k = expected.shape[0]
        if k != self.n_instruments:
            raise ValueError(f"expected {self.n_instruments} instruments, got {k}")
        prev = self._prev_weights if self._prev_weights is not None else np.full(k, 1.0 / k)
        state = self._state(expected, np.asarray(covariance, dtype=float), prev)
        torch = self._torch
        with torch.no_grad():
            alpha = self._alpha(self._actor(torch.as_tensor(state, dtype=torch.float32)))
        weights = (alpha / alpha.sum()).numpy()
        self._prev_weights = weights
        return weights

    def reset(self) -> None:
        """Forget the turnover reference point (e.g. at the start of a new live session)."""
        if self.n_instruments is not None:
            self._prev_weights = np.full(self.n_instruments, 1.0 / self.n_instruments)
