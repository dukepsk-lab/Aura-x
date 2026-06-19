"""Layer 1 read surface — build features on demand from stored bars."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...config import load_params
from ...enums import Timeframe

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/{symbol}")
def build_features(
    symbol: str,
    timeframe: Timeframe = Timeframe.H4,
    limit: int = Query(200, le=2000),
) -> dict[str, object]:
    """Build the v1 feature matrix from stored bars and return the latest rows."""
    try:
        from ...db import BarRepository
    except ModuleNotFoundError as exc:
        raise HTTPException(503, "DB extra not installed (pip install -e '.[db]')") from exc

    from ...l1_features import build_feature_matrix

    params = load_params()
    repo = BarRepository()
    bars = repo.load_bars(symbol, timeframe)
    if bars.empty:
        raise HTTPException(404, f"no bars stored for {symbol} {timeframe.value}")

    partner = _partner_close(params, symbol, timeframe, repo)
    matrix = build_feature_matrix(bars, params, partner_close=partner).tail(limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "feature_count": int(matrix.shape[1]),
        "columns": list(matrix.columns),
        "rows": matrix.reset_index().to_dict(orient="records"),
    }


def _partner_close(params, symbol, timeframe, repo):
    """Resolve the configured cross-pair partner's close series, if available."""
    xp = params.get("cross_pair", {})
    leg_a, leg_b = xp.get("leg_a"), xp.get("leg_b")
    partner_symbol = leg_b if symbol == leg_a else leg_a if symbol == leg_b else None
    if not partner_symbol:
        return None
    partner_bars = repo.load_bars(partner_symbol, timeframe)
    return None if partner_bars.empty else partner_bars["close"]
