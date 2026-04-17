from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from core.database import get_db
from core.models import (
    HoldingDB, HoldingCreate, HoldingUpdate, HoldingOut,
    Portfolio, PortfolioCreate, PortfolioOut,
)
from services.stock_service import get_batch_prices, get_stock_info, validate_ticker

router = APIRouter(tags=["portfolio"])


# ── Portfolio management ──────────────────────────────────────────────────────

@router.get("/portfolios", response_model=list[PortfolioOut])
def list_portfolios(db: Session = Depends(get_db)):
    return db.query(Portfolio).order_by(Portfolio.name).all()


@router.post("/portfolios", response_model=PortfolioOut, status_code=201)
def create_portfolio(body: PortfolioCreate, db: Session = Depends(get_db)):
    if db.query(Portfolio).filter(Portfolio.name == body.name).first():
        raise HTTPException(400, f"Portfolio '{body.name}' already exists")
    p = Portfolio(name=body.name)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/portfolios/{portfolio_id}")
def delete_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    p = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not p:
        raise HTTPException(404, f"Portfolio {portfolio_id} not found")
    db.delete(p)
    db.commit()
    return {"message": f"Portfolio '{p.name}' deleted"}


@router.put("/portfolios/{portfolio_id}", response_model=PortfolioOut)
def rename_portfolio(portfolio_id: int, body: PortfolioCreate, db: Session = Depends(get_db)):
    p = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not p:
        raise HTTPException(404, f"Portfolio {portfolio_id} not found")
    p.name = body.name
    db.commit()
    db.refresh(p)
    return p


# ── Holdings (scoped to a portfolio) ─────────────────────────────────────────

def _get_holding_or_404(ticker: str, portfolio_id: int, db: Session) -> HoldingDB:
    h = (
        db.query(HoldingDB)
        .filter(HoldingDB.portfolio_id == portfolio_id,
                HoldingDB.ticker == ticker.upper())
        .first()
    )
    if not h:
        raise HTTPException(404, f"{ticker} not found in portfolio {portfolio_id}")
    return h


@router.get("/portfolio", response_model=list[HoldingOut])
def list_holdings(
    portfolio_id: int = Query(1, description="Portfolio ID"),
    db: Session = Depends(get_db),
):
    return (
        db.query(HoldingDB)
        .filter(HoldingDB.portfolio_id == portfolio_id)
        .all()
    )


@router.post("/portfolio", response_model=HoldingOut, status_code=201)
def add_holding(
    body: HoldingCreate,
    portfolio_id: int = Query(1, description="Portfolio ID"),
    db: Session = Depends(get_db),
):
    ticker = body.ticker.upper()
    if (
        db.query(HoldingDB)
        .filter(HoldingDB.portfolio_id == portfolio_id, HoldingDB.ticker == ticker)
        .first()
    ):
        raise HTTPException(400, f"{ticker} already in portfolio {portfolio_id}")
    if not validate_ticker(ticker):
        raise HTTPException(400, f"Invalid ticker: {ticker}")
    h = HoldingDB(
        ticker=ticker, shares=body.shares,
        purchase_price=body.purchase_price, portfolio_id=portfolio_id,
    )
    db.add(h)
    db.commit()
    db.refresh(h)
    return h


@router.put("/portfolio/{ticker}", response_model=HoldingOut)
def update_holding(
    ticker: str,
    body: HoldingUpdate,
    portfolio_id: int = Query(1),
    db: Session = Depends(get_db),
):
    h = _get_holding_or_404(ticker, portfolio_id, db)
    if body.shares         is not None:
        h.shares = body.shares
    if body.purchase_price is not None:
        h.purchase_price = body.purchase_price
    db.commit()
    db.refresh(h)
    return h


@router.delete("/portfolio/{ticker}")
def delete_holding(
    ticker: str,
    portfolio_id: int = Query(1),
    db: Session = Depends(get_db),
):
    h = _get_holding_or_404(ticker, portfolio_id, db)
    db.delete(h)
    db.commit()
    return {"message": f"{ticker} removed from portfolio {portfolio_id}"}


@router.get("/portfolio/summary")
def portfolio_summary(
    portfolio_id: int = Query(1),
    db: Session = Depends(get_db),
):
    holdings = (
        db.query(HoldingDB)
        .filter(HoldingDB.portfolio_id == portfolio_id)
        .all()
    )
    if not holdings:
        return {"total_value": 0, "total_cost": 0,
                "total_pnl": 0, "total_pnl_pct": 0, "holdings": []}

    prices = get_batch_prices([h.ticker for h in holdings])
    rows, total_value, total_cost = [], 0.0, 0.0

    for h in holdings:
        price  = prices.get(h.ticker, 0.0)
        value  = price * h.shares
        cost   = h.purchase_price * h.shares
        pnl    = value - cost
        total_value += value
        total_cost  += cost
        info = get_stock_info(h.ticker)
        rows.append({
            "ticker":         h.ticker,
            "name":           info.get("name", h.ticker),
            "shares":         h.shares,
            "purchase_price": h.purchase_price,
            "current_price":  price,
            "value":          round(value, 2),
            "cost":           round(cost, 2),
            "pnl":            round(pnl, 2),
            "pnl_pct":        round(pnl / cost * 100, 2) if cost else 0.0,
            "sector":         info.get("sector", "Unknown"),
            "change_pct":     info.get("change_pct", 0.0),
        })

    pnl_total = total_value - total_cost
    return {
        "total_value":   round(total_value, 2),
        "total_cost":    round(total_cost, 2),
        "total_pnl":     round(pnl_total, 2),
        "total_pnl_pct": round(pnl_total / total_cost * 100, 2) if total_cost else 0.0,
        "holdings":      rows,
    }
