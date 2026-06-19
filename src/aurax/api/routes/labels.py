"""Layer 4 read surface — generate triple-barrier labels from stored bars."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...config import load_params
from ...enums import Timeframe

router = APIRouter(prefix="/labels", tags=["labels"])


@router.get("/{symbol}")
def make_labels(
    symbol: str,
    timeframe: Timeframe = Timeframe.H4,
    limit: int = Query(200, le=2000),
) -> dict[str, object]:
    """Triple-barrier label stored bars and return the latest labels + summary."""
    try:
        from ...db import BarRepository
    except ModuleNotFoundError as exc:
        raise HTTPException(503, "DB extra not installed (pip install -e '.[db]')") from exc

    from ...l4_labeling import LabelConfig, Labeler

    bars = BarRepository().load_bars(symbol, timeframe)
    if bars.empty:
        raise HTTPException(404, f"no bars stored for {symbol} {timeframe.value}")

    cfg = LabelConfig.from_params(load_params())
    labels = Labeler(cfg).make_labels(bars)
    tail = labels.tail(limit).reset_index()
    tail = tail.astype(object).where(tail.notna(), None)
    dist = labels["label"].value_counts().sort_index().to_dict()
    return {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "count": int(len(labels)),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "labels": tail.to_dict(orient="records"),
    }
