from fastapi import APIRouter
from services.stock_service import get_stock_info, get_historical

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{ticker}")
def stock_info(ticker: str):
    return get_stock_info(ticker.upper())


@router.get("/{ticker}/history")
def stock_history(ticker: str, period: str = "1y"):
    df = get_historical(ticker.upper(), period=period)
    if df.empty:
        return {"ticker": ticker, "data": []}
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return {"ticker": ticker, "data": df[cols].to_dict(orient="records")}
