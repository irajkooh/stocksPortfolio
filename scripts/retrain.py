"""
Retraining script — triggered by CI/CD when drift is detected.

Usage:
    python scripts/retrain.py --tickers AAPL MSFT GOOGL --budget 100000
                               --period 2y --timesteps 20000

Steps
-----
1. Fetch latest market data
2. Run max-Sharpe optimisation
3. Evaluate: compare new Sharpe vs baseline
4. If improved → record metrics (becomes new baseline)
5. Exit 0 (success) or 1 (model did not improve)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from core.database import init_db  # noqa: E402
from services.optimizer import optimize_portfolio  # noqa: E402
from monitoring.metrics_tracker import record_run, load_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",   nargs="+", required=True)
    parser.add_argument("--budget",    type=float, default=100_000.0)
    parser.add_argument("--period",    default="2y")
    parser.add_argument("--timesteps", type=int, default=20_000,
                        help="Stored for record-keeping only (MVO does not use RL steps)")
    parser.add_argument("--force",     action="store_true",
                        help="Record even if new model is worse")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    logger.info("Optimising %s  budget=$%,.0f  period=%s",
                tickers, args.budget, args.period)

    init_db()

    result = optimize_portfolio(
        tickers=tickers,
        budget=args.budget,
        target_vol=0.15,
        lookback=args.period,
    )
    if "error" in result:
        logger.error("Optimisation failed: %s", result["error"])
        sys.exit(1)

    m = result["metrics"]
    new_sharpe = m["sharpe"]
    logger.info("New Sharpe: %.4f  Return: %.2f%%  Vol: %.2f%%",
                new_sharpe, m["expected_return"] * 100, m["expected_vol"] * 100)

    # ── Compare with baseline ─────────────────────────────────────────────────
    history = load_history(tickers=tickers, limit=5)
    if not history.empty:
        baseline_sharpe = float(history.sort_values("created_at").iloc[-1]["sharpe_rl"])
        logger.info("Baseline Sharpe: %.4f", baseline_sharpe)
        if new_sharpe < baseline_sharpe and not args.force:
            logger.warning(
                "New Sharpe (%.4f) is worse than baseline (%.4f). "
                "Not recording. Use --force to override.",
                new_sharpe, baseline_sharpe,
            )
            sys.exit(1)

    # ── Record ────────────────────────────────────────────────────────────────
    record_payload = {
        "tickers":   tickers,
        "period":    args.period,
        "timesteps": args.timesteps,
        "budget":    args.budget,
        "metrics": {
            "rl_sharpe":        new_sharpe,
            "eq_sharpe":        None,
            "rl_annual_return": m["expected_return"],
            "rl_annual_vol":    m["expected_vol"],
        },
        "weights": {t: v.get("weight", 0.0) for t, v in result.get("allocations", {}).items()},
    }
    run_id = record_run(record_payload)
    logger.info("✅  Retrain complete. Run ID: %s", run_id)
    sys.exit(0)


if __name__ == "__main__":
    main()
