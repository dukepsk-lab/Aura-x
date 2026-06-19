"""Layer 2 — Regime Detection (HMM + Hurst/KER gating).

* :class:`RegimeDetector` / :class:`RegimeConfig`  the regime router
* :class:`GaussianHMM`  dependency-free Gaussian HMM (Baum-Welch / Viterbi)

Output regimes (``trend`` / ``range`` / ``shock``) condition Layer 3 member
weights and Layer 6 sizing; ``shock`` lets the system stand down.
"""

from __future__ import annotations

from .detector import RegimeConfig, RegimeDetector
from .hmm import GaussianHMM

__all__ = ["RegimeDetector", "RegimeConfig", "GaussianHMM"]
