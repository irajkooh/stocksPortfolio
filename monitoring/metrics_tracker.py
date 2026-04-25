"""
Track RL model performance metrics over time and persist to SQLite.
Metrics recorded after each optimisation run are used by drift_detector.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config import DATA_DIR

logger = logging.getLogger(__name__)

_DB_URL = f"sqlite:///{DATA_DIR / 'mlops_metrics.db'}"
_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)
Base = declarative_base()


# ── ORM ───────────────────────────────────────────────────────────────────────

class MetricRecord(Base):
    __tablename__ = "metric_records"
    id              = Column(Integer, primary_key=True)
    run_id          = Column(String, nullable=False)
    tickers         = Column(String, nullable=False)   # JSON list
    period          = Column(String)
    timesteps       = Column(Integer)
    budget          = Column(Float)
    sharpe_rl       = Column(Float)
    sharpe_eq       = Column(Float)
    annual_return   = Column(Float)
    annual_vol      = Column(Float)
    weights_json    = Column(Text)                     # JSON dict
    created_at      = Column(DateTime, default=datetime.utcnow)


def init_metrics_db():
    Base.metadata.create_all(bind=_engine)


def record_run(result: dict, run_id: str | None = None) -> str:
    """Persist optimisation result metrics. Returns the run_id."""
    init_metrics_db()
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    m = result.get("metrics", {})
    db = _Session()
    try:
        rec = MetricRecord(
            run_id        = run_id,
            tickers       = json.dumps(result.get("tickers", [])),
            period        = result.get("period", ""),
            timesteps     = result.get("timesteps"),
            budget        = result.get("budget"),
            sharpe_rl     = m.get("rl_sharpe"),
            sharpe_eq     = m.get("eq_sharpe"),
            annual_return = m.get("rl_annual_return"),
            annual_vol    = m.get("rl_annual_vol"),
            weights_json  = json.dumps(result.get("weights", {})),
        )
        db.add(rec); db.commit()
        logger.info("Recorded run %s: Sharpe=%.3f", run_id, m.get("rl_sharpe", 0))
        return run_id
    finally:
        db.close()


def load_history(tickers: list[str] | None = None, limit: int = 200) -> pd.DataFrame:
    """Return metric history as a DataFrame, optionally filtered by tickers."""
    init_metrics_db()
    db = _Session()
    try:
        rows = db.query(MetricRecord).order_by(MetricRecord.created_at.desc()).limit(limit).all()
    finally:
        db.close()

    records = []
    for r in rows:
        t = json.loads(r.tickers)
        if tickers and not set(tickers).issubset(set(t)):
            continue
        records.append({
            "run_id":       r.run_id,
            "tickers":      t,
            "sharpe_rl":    r.sharpe_rl,
            "sharpe_eq":    r.sharpe_eq,
            "annual_return":r.annual_return,
            "annual_vol":   r.annual_vol,
            "created_at":   r.created_at,
        })
    return pd.DataFrame(records)
