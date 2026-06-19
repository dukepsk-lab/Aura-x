"""Decay-triggered retraining orchestrator (Roadmap v2 — lifecycle).

Closes the loop the architecture asks for: L8 detects decay → retrain on fresh
data → **the validation gate decides promotion** (nothing reaches live without
clearing §5) → alert. A retrained model that fails the gate is rejected, not
shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..l3_primary import PrimarySignalModel
from ..l5_meta import MetaGatedPrimary
from ..l8_monitoring import AlertKind, DecayConfig, DecayMonitor, DecayReport, LogNotifier, Notifier
from ..validation import ValidationConfig, ValidationReport, run_validation
from .training import TrainedModel, TrainingPipeline


@dataclass
class RetrainPolicy:
    decay: DecayConfig = field(default_factory=DecayConfig)
    require_gate_pass: bool = True   # promote only if the retrained stack clears the gate


@dataclass
class RetrainResult:
    triggered: bool
    decay: DecayReport
    promoted: bool = False
    model: TrainedModel | None = None
    report: ValidationReport | None = None


class RetrainOrchestrator:
    """Run decay detection → retrain → validate → promote, with alerts."""

    def __init__(
        self,
        pipeline: TrainingPipeline,
        policy: RetrainPolicy | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.policy = policy or RetrainPolicy()
        self.notifier = notifier or LogNotifier()
        self.decay = DecayMonitor(self.policy.decay)

    def maybe_retrain(
        self,
        *,
        monitor_returns: pd.Series,
        monitor_prob: np.ndarray,
        monitor_outcome: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series,
        ret: pd.Series,
        labels: pd.Series,
        close: pd.Series | None = None,
        regimes: pd.Series | None = None,
        context: pd.DataFrame | None = None,
        context_cols: list[str] | None = None,
        sample_weight: pd.Series | None = None,
        validation_config: ValidationConfig | None = None,
    ) -> RetrainResult:
        """Retrain only if recent performance has decayed; promote only on a GO."""
        report = self.decay.evaluate(monitor_returns, monitor_prob, monitor_outcome)
        if not report.retrain_due:
            return RetrainResult(triggered=False, decay=report)

        self.notifier.send(AlertKind.DECAY, "retrain triggered: " + "; ".join(report.reasons()))
        model = self.pipeline.train(
            X, y, t1=t1, ret=ret, regimes=regimes, context=context, sample_weight=sample_weight
        )

        gate_report = None
        promoted = True
        if self.policy.require_gate_pass:
            primary_config = self.pipeline.primary_config
            meta_config = self.pipeline.meta_config

            def make_estimator():
                return MetaGatedPrimary(
                    lambda: PrimarySignalModel(primary_config),
                    t1=t1, regimes=regimes, context_cols=context_cols or [], meta_config=meta_config,
                )

            gate_report = run_validation(
                make_estimator, X, y, t1=t1, ret=ret, labels=labels, close=close,
                sample_weight=sample_weight, config=validation_config or ValidationConfig(),
            )
            promoted = gate_report.passed

        self.notifier.send(
            AlertKind.DECAY,
            f"retrained model {'PROMOTED' if promoted else 'REJECTED by the gate'}",
        )
        return RetrainResult(
            triggered=True, decay=report, promoted=promoted,
            model=model if promoted else None, report=gate_report,
        )
