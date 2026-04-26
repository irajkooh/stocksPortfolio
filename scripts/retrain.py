"""
Retraining script — triggered by CI/CD when drift is detected.

Usage:
    python scripts/retrain.py --tickers AAPL MSFT GOOGL --budget 100000
                               --period 2y --timesteps 20000

Steps
-----
1. Fetch latest market data
2. Train PPO model
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
from services.rl_optimizer import optimize_portfolio  # noqa: E402
from monitoring.metrics_tracker import record_run, load_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",   nargs="+", required=True)
    parser.add_argument("--budget",    type=float, default=100_000.0)
    parser.add_argument("--period",    default="2y")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--force",     action="store_true",
                        help="Record even if new model is worse")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    logger.info("Retraining on %s  budget=$%,.0f  period=%s  steps=%d",
                tickers, args.budget, args.period, args.timesteps)

    init_db()

    # ── Train ─────────────────────────────────────────────────────────────────
    result = optimize_portfolio(
        tickers   = tickers,
        budget    = args.budget,
        period    = args.period,
        timesteps = args.timesteps,
    )
    if "error" in result:
        logger.error("Optimisation failed: %s", result["error"])
        sys.exit(1)

    new_sharpe = result["metrics"]["rl_sharpe"]
    logger.info("New model Sharpe: %.4f", new_sharpe)

    # ── Compare with baseline ─────────────────────────────────────────────────
    history = load_history(tickers=tickers, limit=5)
    if not history.empty:
        baseline_sharpe = float(history.sort_values("created_at").iloc[-1]["sharpe_rl"])
        logger.info("Baseline Sharpe: %.4f", baseline_sharpe)
        if new_sharpe < baseline_sharpe and not args.force:
            logger.warning(
                "New model (%.4f) is worse than baseline (%.4f). "
                "Not recording. Use --force to override.",
                new_sharpe, baseline_sharpe,
            )
            sys.exit(1)

    # ── Record ────────────────────────────────────────────────────────────────
    result["tickers"]   = tickers
    result["period"]    = args.period
    result["timesteps"] = args.timesteps
    run_id = record_run(result)
    logger.info("✅  Retrain complete. Run ID: %s", run_id)
    sys.exit(0)


if __name__ == "__main__":
    main()
