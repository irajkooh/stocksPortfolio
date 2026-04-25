"""
CI/CD hook: exit 1 if drift detected (triggers retraining workflow).

Usage (GitHub Actions):
    python scripts/check_drift.py --tickers AAPL MSFT GOOGL
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from monitoring.drift_detector import check_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Ticker symbols to check; omit to check all history")
    args = parser.parse_args()

    result = check_drift(tickers=args.tickers)

    print(json.dumps(result.report, indent=2))

    if result.drift_detected:
        print("\n⚠️  DRIFT DETECTED — retraining required.", file=sys.stderr)
        for r in result.reasons:
            print(f"   • {r}", file=sys.stderr)
        sys.exit(1)          # CI picks this up as failure → triggers retrain job
    else:
        print("\n✅  No drift detected — model is healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
