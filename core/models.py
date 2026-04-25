"""ORM models + Pydantic schemas for the watchlist-based portfolio."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Float, Text
)
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel, ConfigDict, field_validator

Base = declarative_base()


class PortfolioDB(Base):
    __tablename__ = "portfolios"
    id          = Column(Integer, primary_key=True)
    name        = Column(String(80), nullable=False, unique=True)
    description = Column(String(200), default="")
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    holdings    = relationship(
        "HoldingDB", back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    allocation  = relationship(
        "PortfolioAllocationDB", back_populates="portfolio",
        cascade="all, delete-orphan", uselist=False,
    )


class HoldingDB(Base):
    __tablename__ = "holdings"
    id             = Column(Integer, primary_key=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"),
                            nullable=False)
    ticker         = Column(String(10), nullable=False)
    shares         = Column(Float, nullable=True)
    purchase_price = Column(Float, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio    = relationship("PortfolioDB", back_populates="holdings")

    __table_args__ = (UniqueConstraint("portfolio_id", "ticker",
                                       name="uq_holding_portfolio_ticker"),)


class PortfolioAllocationDB(Base):
    __tablename__ = "portfolio_allocations"
    portfolio_id     = Column(Integer,
                              ForeignKey("portfolios.id", ondelete="CASCADE"),
                              primary_key=True)
    budget           = Column(Float, nullable=False)
    target_vol       = Column(Float, nullable=False)
    lookback         = Column(String(4), nullable=False)
    expected_return  = Column(Float, nullable=False)
    expected_vol     = Column(Float, nullable=False)
    sharpe           = Column(Float, nullable=False)
    risk_free_rate   = Column(Float, nullable=False)
    cash_dollars     = Column(Float, nullable=False)
    allocations_json = Column(Text, nullable=False)
    frontier_json    = Column(Text, default="[]")
    commentary       = Column(Text, default="")
    created_at       = Column(DateTime, default=datetime.utcnow)

    portfolio        = relationship("PortfolioDB", back_populates="allocation")


# ── Pydantic schemas ──────────────────────────────────────────────
class PortfolioCreate(BaseModel):
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name required")
        return v


class PortfolioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class HoldingCreate(BaseModel):
    ticker: str

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker required")
        if v == "CASH":
            raise ValueError("'CASH' is reserved and auto-injected")
        return v


class HoldingUpdate(BaseModel):
    shares: float | None = None
    purchase_price: float | None = None


class HoldingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    portfolio_id: int
    ticker: str
    created_at: datetime


class AllocationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    portfolio_id: int
    budget: float
    target_vol: float
    lookback: str
    expected_return: float
    expected_vol: float
    sharpe: float
    risk_free_rate: float
    cash_dollars: float
    allocations_json: str
    commentary: str
    created_at: datetime


class PortfolioSummary(BaseModel):
    portfolio: PortfolioOut
    watchlist: list[HoldingOut]
    allocation: Optional[AllocationOut] = None
