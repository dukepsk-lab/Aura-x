"""Lifecycle — model training, persistence and decay-triggered retraining.

* :class:`TrainingPipeline` / :class:`TrainedModel`  fit primary+meta, save/load
* :class:`RetrainOrchestrator` / :class:`RetrainPolicy` / :class:`RetrainResult`
  decay → retrain → validation-gate → promote (the automated v2 loop)
"""

from __future__ import annotations

from .retrain import RetrainOrchestrator, RetrainPolicy, RetrainResult
from .training import TrainedModel, TrainingPipeline

__all__ = [
    "TrainingPipeline",
    "TrainedModel",
    "RetrainOrchestrator",
    "RetrainPolicy",
    "RetrainResult",
]
