"""A dependency-free Gaussian HMM (diagonal covariance) for regime detection.

Implements Baum-Welch (EM) fitting, forward-backward posteriors and Viterbi
decoding in log-space (via ``scipy.special.logsumexp``) — numerically stable and
deterministic, with no third-party ML dependency, so Layer 2 runs and tests
anywhere. ``hmmlearn`` would be a drop-in alternative when ``[models]`` is present.

States are unlabeled here; mapping them to ``trend`` / ``range`` / ``shock`` by
their learned characteristics is the detector's job
(:mod:`aurax.l2_regime.detector`).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_LOG_FLOOR = 1e-300


class GaussianHMM:
    """Gaussian-emission HMM with diagonal covariances."""

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        min_covar: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.seed = seed

    # --- standardisation (fit-time stats stored) -----------------------------
    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-9
        return (x - self.mean_) / self.std_

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    # --- init ----------------------------------------------------------------
    def _init_params(self, z: np.ndarray) -> None:
        k = self.n_states
        # Quantile init on the first feature → low/mid/high seed states (sensible
        # for a volatility-led regime model) and reproducible.
        order = np.argsort(z[:, 0])
        bins = np.array_split(order, k)
        self.means_ = np.array([z[b].mean(axis=0) for b in bins])
        self.covars_ = np.array([z[b].var(axis=0) + self.min_covar for b in bins])
        self.startprob_ = np.full(k, 1.0 / k)
        trans = np.full((k, k), 0.1 / max(k - 1, 1))
        np.fill_diagonal(trans, 0.9)
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)

    # --- emissions -----------------------------------------------------------
    def _log_emission(self, z: np.ndarray) -> np.ndarray:
        out = np.empty((len(z), self.n_states))
        for k in range(self.n_states):
            var = self.covars_[k]
            out[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (z - self.means_[k]) ** 2 / var, axis=1)
        return out

    # --- forward / backward --------------------------------------------------
    def _forward(self, log_em: np.ndarray) -> tuple[np.ndarray, float]:
        log_trans = np.log(self.transmat_ + _LOG_FLOOR)
        log_alpha = np.empty_like(log_em)
        log_alpha[0] = np.log(self.startprob_ + _LOG_FLOOR) + log_em[0]
        for t in range(1, len(log_em)):
            log_alpha[t] = log_em[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def _backward(self, log_em: np.ndarray) -> np.ndarray:
        log_trans = np.log(self.transmat_ + _LOG_FLOOR)
        log_beta = np.zeros_like(log_em)
        for t in range(len(log_em) - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + (log_em[t + 1] + log_beta[t + 1])[None, :], axis=1
            )
        return log_beta

    # --- fit (Baum-Welch) ----------------------------------------------------
    def fit(self, x: np.ndarray) -> GaussianHMM:
        z = self._standardize_fit(np.asarray(x, dtype=float))
        self._init_params(z)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_em = self._log_emission(z)
            log_alpha, ll = self._forward(log_em)
            log_beta = self._backward(log_em)

            gamma = np.exp(log_alpha + log_beta - ll)  # (T, K)
            log_trans = np.log(self.transmat_ + _LOG_FLOOR)
            # xi summed over time, vectorised: (T-1, K, K)
            log_xi = (
                log_alpha[:-1, :, None]
                + log_trans[None, :, :]
                + (log_em[1:] + log_beta[1:])[:, None, :]
                - ll
            )
            xi_sum = np.exp(log_xi).sum(axis=0)

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()
            row = xi_sum.sum(axis=1, keepdims=True)
            self.transmat_ = np.where(row > 0, xi_sum / np.where(row == 0, 1, row), self.transmat_)
            gk = gamma.sum(axis=0)
            for k in range(self.n_states):
                if gk[k] <= 0:
                    continue
                self.means_[k] = (gamma[:, k, None] * z).sum(axis=0) / gk[k]
                self.covars_[k] = (
                    gamma[:, k, None] * (z - self.means_[k]) ** 2
                ).sum(axis=0) / gk[k] + self.min_covar

            if ll - prev_ll < self.tol:
                break
            prev_ll = ll
        self.n_features_ = z.shape[1]
        self.converged_loglik_ = ll
        return self

    # --- inference -----------------------------------------------------------
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Posterior state probabilities ``P(state | all observations)`` (T, K)."""
        z = self._standardize(np.asarray(x, dtype=float))
        log_em = self._log_emission(z)
        log_alpha, ll = self._forward(log_em)
        log_beta = self._backward(log_em)
        return np.exp(log_alpha + log_beta - ll)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """Viterbi most-likely state path (smoother than posterior argmax)."""
        z = self._standardize(np.asarray(x, dtype=float))
        log_em = self._log_emission(z)
        log_trans = np.log(self.transmat_ + _LOG_FLOOR)
        t_len = len(z)
        delta = np.empty((t_len, self.n_states))
        psi = np.zeros((t_len, self.n_states), dtype=int)
        delta[0] = np.log(self.startprob_ + _LOG_FLOOR) + log_em[0]
        for t in range(1, t_len):
            m = delta[t - 1][:, None] + log_trans
            psi[t] = m.argmax(axis=0)
            delta[t] = log_em[t] + m.max(axis=0)
        states = np.empty(t_len, dtype=int)
        states[-1] = int(delta[-1].argmax())
        for t in range(t_len - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most-likely state per observation (posterior argmax)."""
        return self.predict_proba(x).argmax(axis=1)
