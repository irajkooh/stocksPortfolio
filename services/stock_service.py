import time
import requests
import yfinance as yf
import pandas as pd

_cache: dict = {}
_TTL = 300  # 5-minute cache

# Browser-like session so Yahoo Finance doesn't block cloud/shared IPs
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


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
            info = yf.Ticker(ticker, session=_session).info
            return {
                "ticker":     ticker,
                "name":       info.get("longName") or info.get("shortName", ticker),
                "price":      info.get("currentPrice") or info.get("regularMarketPrice", 0.0),
                "change_pct": info.get("regularMarketChangePercent", 0.0),
                "sector":     info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio":   info.get("trailingPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low":  info.get("fiftyTwoWeekLow"),
                "currency":   info.get("currency", "USD"),
            }
        except Exception as e:
            return {"ticker": ticker, "name": ticker, "price": 0.0, "error": str(e)}

    return _cached(f"info_{ticker}", fetch)


def get_historical(ticker: str, period: str = "1y") -> pd.DataFrame:
    def fetch():
        try:
            return yf.Ticker(ticker, session=_session).history(period=period)
        except Exception:
            return pd.DataFrame()

    return _cached(f"hist_{ticker}_{period}", fetch)


def get_period_changes(ticker: str) -> dict[str, float]:
    """Return 1d/1mo/3mo/1y price change % using real-time price as current."""
    def fetch():
        try:
            hist = yf.Ticker(ticker, session=_session).history(period="1y")["Close"].dropna()
            if hist.empty:
                return {}
            stock = get_stock_info(ticker)
            cur = float(stock.get("price") or 0) or float(hist.iloc[-1])
            def _pct(ref_price: float) -> float:
                return round((cur - ref_price) / ref_price * 100, 2) if ref_price else 0.0
            return {
                "change_1d_pct":  _pct(float(hist.iloc[-1])),
                "change_1mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 22)])),
                "change_3mo_pct": _pct(float(hist.iloc[max(0, len(hist) - 64)])),
                "change_1y_pct":  _pct(float(hist.iloc[0])),
            }
        except Exception:
            return {}

    return _cached(f"periods_{ticker}", fetch)


def get_batch_prices(tickers: list[str]) -> dict[str, float]:
    return {t: get_stock_info(t).get("price", 0.0) for t in tickers}


def validate_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker, session=_session).info
        return bool(info.get("symbol") or info.get("shortName"))
    except Exception:
        return False
