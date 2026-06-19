"""Layer 8 read surface — dashboard metrics from the trade journal (VENUS-X)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("/dashboard")
def dashboard(limit: int = Query(500, le=5000)) -> dict[str, object]:
    """Rolling cost-adjusted Sharpe, hit-rate and meta-calibration from the journal."""
    try:
        from sqlalchemy import text

        from ...db import get_engine
    except ModuleNotFoundError as exc:
        raise HTTPException(503, "DB extra not installed (pip install -e '.[db]')") from exc

    import pandas as pd

    from ...l8_monitoring import journal_metrics

    sql = text(
        "SELECT pnl_net, meta_prob, side, regime FROM market.trades "
        "WHERE status = 'closed' ORDER BY close_ts DESC LIMIT :limit"
    )
    trades = pd.read_sql(sql, get_engine(), params={"limit": limit})
    metrics = journal_metrics(trades)
    return {"window_trades": int(len(trades)), "metrics": metrics}
