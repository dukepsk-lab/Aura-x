"""Layer 0 read surface — query stored OHLCV bars."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from ...enums import Timeframe

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/bars")
def get_bars(
    symbol: str = Query(..., examples=["EURUSD"]),
    timeframe: Timeframe = Timeframe.H4,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(500, le=5000),
) -> dict[str, object]:
    """Return stored bars (most recent ``limit`` within the optional window)."""
    try:
        from ...db import BarRepository
    except ModuleNotFoundError as exc:
        raise HTTPException(503, "DB extra not installed (pip install -e '.[db]')") from exc

    df = BarRepository().load_bars(symbol, timeframe, start, end).tail(limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "count": int(len(df)),
        "bars": df.reset_index().to_dict(orient="records"),
    }
