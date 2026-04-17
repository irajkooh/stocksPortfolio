from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ── ORM ───────────────────────────────────────────────────────────────────────

class Portfolio(Base):
    __tablename__ = "portfolios"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    holdings   = relationship(
        "HoldingDB", back_populates="portfolio", cascade="all, delete-orphan"
    )


class HoldingDB(Base):
    __tablename__ = "holdings"
    id             = Column(Integer, primary_key=True, index=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    ticker         = Column(String, nullable=False, index=True)
    shares         = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    portfolio      = relationship("Portfolio", back_populates="holdings")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_ticker"),
    )


# ── Pydantic — Portfolio ──────────────────────────────────────────────────────

class PortfolioCreate(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        return v


class PortfolioOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    model_config = {"from_attributes": True}


# ── Pydantic — Holding ────────────────────────────────────────────────────────

class HoldingCreate(BaseModel):
    ticker:         str
    shares:         float
    purchase_price: float

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("shares", "purchase_price")
    @classmethod
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class HoldingUpdate(BaseModel):
    shares:         Optional[float] = None
    purchase_price: Optional[float] = None


class HoldingOut(BaseModel):
    id:             int
    portfolio_id:   int
    ticker:         str
    shares:         float
    purchase_price: float
    created_at:     datetime

    model_config = {"from_attributes": True}


class PortfolioSummary(BaseModel):
    total_value:   float
    total_cost:    float
    total_pnl:     float
    total_pnl_pct: float
    holdings:      list[dict]
