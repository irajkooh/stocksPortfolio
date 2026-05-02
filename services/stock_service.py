import time
import yfinance as yf
import pandas as pd

_cache: dict = {}
_TTL = 300  # 5-minute cache


def _cached(key: str, fn, ttl: int = _TTL):
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < ttl:
        return _cache[key]["data"]
    data = fn()
    _cache[key] = {"ts": now, "data": data}
    return data


def get_stock_info(ticker: str) -> dict:
    def fetch():
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info          # light endpoint — never times out on shared infra
            price = float(fi.last_price or 0.0)
            # .info is heavier; fetch separately, fall back gracefully
            try:
                info = t.info
            except Exception:
                info = {}
            return {
                "ticker":     ticker,
                "name":       info.get("longName") or info.get("shortName", ticker),
                "price":      price or float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0),
                "change_pct": info.get("regularMarketChangePercent", 0.0),
                "sector":     info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio":   info.get("trailingPE"),
                "week52_high": fi.year_high or info.get("fiftyTwoWeekHigh"),
                "week52_low":  fi.year_low  or info.get("fiftyTwoWeekLow"),
                "currency":   fi.currency  or info.get("currency", "USD"),
            }
        except Exception as e:
            return {"ticker": ticker, "name": ticker, "price": 0.0, "error": str(e)}

    return _cached(f"info_{ticker}", fetch)


def get_historical(ticker: str, period: str = "1y") -> pd.DataFrame:
    def fetch():
        try:
            return yf.Ticker(ticker).history(period=period)
        except Exception:
            return pd.DataFrame()

    return _cached(f"hist_{ticker}_{period}", fetch)


def get_period_changes(ticker: str) -> dict[str, float]:
    """Return 1d/1mo/3mo/1y price change % using real-time price as current."""
    def fetch():
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            cur = float(fi.last_price or 0.0)
            # previous_close from fast_info is the most reliable 1d reference
            prev_close = float(
                fi.regular_market_previous_close
                or fi.previous_close
                or 0.0
            )

            hist = t.history(period="1y")["Close"].dropna()
            if not cur:
                cur = float(hist.iloc[-1]) if not hist.empty else 0.0
            if not cur:
                return {}

            def _pct(ref_price: float) -> float:
                return round((cur - ref_price) / ref_price * 100, 2) if ref_price else 0.0

            # 1d: prefer fast_info prev-close; fall back to hist[-2]
            if prev_close:
                change_1d = _pct(prev_close)
            elif len(hist) >= 2:
                change_1d = _pct(float(hist.iloc[-2]))
            else:
                change_1d = 0.0

            return {
                "change_1d_pct":  change_1d,
                "change_1mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 22)])) if not hist.empty else 0.0,
                "change_3mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 64)])) if not hist.empty else 0.0,
                "change_1y_pct":  _pct(float(hist.iloc[0]))                       if not hist.empty else 0.0,
            }
        except Exception:
            return {}

    return _cached(f"periods_{ticker}", fetch)


def get_batch_prices(tickers: list[str]) -> dict[str, float]:
    return {t: get_stock_info(t).get("price", 0.0) for t in tickers}


def validate_ticker(ticker: str) -> bool:
    try:
        fi = yf.Ticker(ticker).fast_info
        return bool(fi.last_price and fi.last_price > 0)
    except Exception:
        return False
