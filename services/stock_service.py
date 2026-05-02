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
            # history() uses /v8/finance/chart — most reliable on shared infra
            hist2 = t.history(period="5d")
            price = float(hist2["Close"].dropna().iloc[-1]) if not hist2.empty else 0.0
            # .info is best-effort only (heavy endpoint, may time out)
            try:
                info = t.info
            except Exception:
                info = {}
            # fast_info as last-resort fallback
            try:
                fi = t.fast_info
                fi_price = float(fi.last_price or 0.0)
                fi_yh    = fi.year_high
                fi_yl    = fi.year_low
                fi_cur   = fi.currency
            except Exception:
                fi_price = fi_yh = fi_yl = fi_cur = None
            return {
                "ticker":     ticker,
                "name":       info.get("longName") or info.get("shortName", ticker),
                "price":      price or fi_price or float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0),
                "change_pct": info.get("regularMarketChangePercent", 0.0),
                "sector":     info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio":   info.get("trailingPE"),
                "week52_high": fi_yh or info.get("fiftyTwoWeekHigh"),
                "week52_low":  fi_yl or info.get("fiftyTwoWeekLow"),
                "currency":   fi_cur or info.get("currency", "USD"),
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
    """Return 1d/1mo/3mo/1y price change % derived entirely from history()."""
    def fetch():
        try:
            # history() is the most reliable endpoint on all infra
            hist = yf.Ticker(ticker).history(period="1y")["Close"].dropna()
            if hist.empty:
                return {}
            cur = float(hist.iloc[-1])
            if not cur:
                return {}

            def _pct(ref_price: float) -> float:
                return round((cur - ref_price) / ref_price * 100, 2) if ref_price else 0.0

            return {
                "change_1d_pct":  _pct(float(hist.iloc[-2]))                      if len(hist) >= 2     else 0.0,
                "change_1mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 22)]))  if not hist.empty    else 0.0,
                "change_3mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 64)]))  if not hist.empty    else 0.0,
                "change_1y_pct":  _pct(float(hist.iloc[0]))                        if not hist.empty    else 0.0,
            }
        except Exception:
            return {}

    return _cached(f"periods_{ticker}", fetch)


def get_batch_prices(tickers: list[str]) -> dict[str, float]:
    return {t: get_stock_info(t).get("price", 0.0) for t in tickers}


def validate_ticker(ticker: str) -> bool:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return not hist.empty and float(hist["Close"].dropna().iloc[-1]) > 0
    except Exception:
        return False
