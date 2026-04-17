"""
Drift Detection for the RL Portfolio Model
==========================================

Three drift signals are monitored:

1. Sharpe Drift    — model's Sharpe ratio degrades > SHARPE_THRESHOLD vs baseline
2. Return Drift    — expected annual return drops > RETURN_THRESHOLD vs baseline
3. Data Drift      — statistical shift in asset returns (PSI / KS test)

A drift report is written to  data/drift_report.json  and the exit code of
scripts/check_drift.py reflects whether retraining is needed (used by CI/CD).
"""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

from core.config import DATA_DIR
from monitoring.metrics_tracker import load_history

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = DATA_DIR / "drift_report.json"

# ── Thresholds (overridable via env) ─────────────────────────────────────────
SHARPE_THRESHOLD  = float(os.environ.get("DRIFT_SHARPE_THRESHOLD",  "0.15"))
RETURN_THRESHOLD  = float(os.environ.get("DRIFT_RETURN_THRESHOLD",  "0.10"))
KS_P_VALUE_CUTOFF = float(os.environ.get("DRIFT_KS_P_CUTOFF",       "0.05"))
MONITORING_WINDOW = int(os.environ.get("MONITORING_WINDOW_DAYS",    "30"))


class DriftResult(NamedTuple):
    drift_detected:    bool
    reasons:           list[str]
    sharpe_change_pct: float
    return_change_pct: float
    data_drift_tickers: list[str]
    baseline_sharpe:   float
    current_sharpe:    float
    report:            dict


# ── Performance drift ─────────────────────────────────────────────────────────

def _performance_drift(history: pd.DataFrame) -> tuple[bool, list[str], float, float]:
    if len(history) < 2:
        return False, [], 0.0, 0.0

    history = history.sort_values("created_at")
    baseline = history.iloc[0]["sharpe_rl"]
    recent   = history.iloc[-1]["sharpe_rl"]

    if baseline in (None, 0.0) or pd.isna(baseline):
        return False, [], 0.0, 0.0

    sharpe_change = (baseline - recent) / abs(baseline)

    b_ret = history.iloc[0]["annual_return"]
    r_ret = history.iloc[-1]["annual_return"]
    ret_change = ((b_ret - r_ret) / abs(b_ret)) if b_ret not in (None, 0.0) else 0.0

    reasons = []
    if sharpe_change > SHARPE_THRESHOLD:
        reasons.append(
            f"Sharpe dropped {sharpe_change*100:.1f}% "
            f"(baseline {baseline:.3f} → current {recent:.3f})"
        )
    if ret_change > RETURN_THRESHOLD:
        reasons.append(
            f"Expected return dropped {ret_change*100:.1f}% "
            f"(baseline {b_ret:.2f}% → current {r_ret:.2f}%)"
        )

    return bool(reasons), reasons, float(sharpe_change), float(ret_change)


# ── Data / distribution drift (KS test) ──────────────────────────────────────

def _data_drift(tickers: list[str], window_days: int = MONITORING_WINDOW) -> tuple[bool, list[str]]:
    """
    Compare distribution of recent returns vs historical baseline using KS test.
    Returns (drift_detected, list_of_drifted_tickers).
    """
    try:
        from services.stock_service import get_historical
    except ImportError:
        return False, []

    drifted = []
    cutoff  = datetime.utcnow() - timedelta(days=window_days * 2)

    for ticker in tickers:
        try:
            hist = get_historical(ticker, period="2y")
            if hist.empty or len(hist) < window_days * 2:
                continue
            returns = hist["Close"].pct_change().dropna().values
            baseline_r = returns[: -window_days]
            recent_r   = returns[-window_days:]

            _, p_value = stats.ks_2samp(baseline_r, recent_r)
            if p_value < KS_P_VALUE_CUTOFF:
                drifted.append(ticker)
                logger.info("Data drift detected for %s (KS p=%.4f)", ticker, p_value)
        except Exception as exc:
            logger.warning("KS test failed for %s: %s", ticker, exc)

    return bool(drifted), drifted


# ── Main entry point ──────────────────────────────────────────────────────────

def check_drift(tickers: list[str] | None = None) -> DriftResult:
    """
    Run all drift checks and write a JSON report.
    Returns DriftResult (drift_detected=True → retraining needed).
    """
    history = load_history(tickers=tickers, limit=50)

    perf_drift, reasons, sh_chg, ret_chg = _performance_drift(history)
    baseline_sh = float(history.iloc[0]["sharpe_rl"]) if not history.empty else 0.0
    current_sh  = float(history.iloc[-1]["sharpe_rl"]) if not history.empty else 0.0

    if tickers:
        data_drift, drifted_tickers = _data_drift(tickers)
        if data_drift:
            reasons.append(
                f"Return distribution drift detected for: {', '.join(drifted_tickers)}"
            )
    else:
        data_drift, drifted_tickers = False, []

    drift_detected = perf_drift or data_drift

    report = {
        "timestamp":           datetime.utcnow().isoformat(),
        "drift_detected":      drift_detected,
        "reasons":             reasons,
        "sharpe_change_pct":   round(sh_chg * 100, 2),
        "return_change_pct":   round(ret_chg * 100, 2),
        "data_drift_tickers":  drifted_tickers,
        "baseline_sharpe":     round(baseline_sh, 4),
        "current_sharpe":      round(current_sh, 4),
        "thresholds": {
            "sharpe":  SHARPE_THRESHOLD,
            "return":  RETURN_THRESHOLD,
            "ks_pval": KS_P_VALUE_CUTOFF,
        },
    }

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Drift report written to %s  (drift=%s)", DRIFT_REPORT_PATH, drift_detected)

    return DriftResult(
        drift_detected     = drift_detected,
        reasons            = reasons,
        sharpe_change_pct  = sh_chg,
        return_change_pct  = ret_chg,
        data_drift_tickers = drifted_tickers,
        baseline_sharpe    = baseline_sh,
        current_sharpe     = current_sh,
        report             = report,
    )
