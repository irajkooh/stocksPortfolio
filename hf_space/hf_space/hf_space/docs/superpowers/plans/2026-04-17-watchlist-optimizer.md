# Budget-Optimized Watchlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the app around a symbol-only watchlist and a Markowitz optimizer with always-on CASH, user-configurable risk-free rate (default 4%, entered via slider or `"4.56%"` textbox), a runtime banner showing environment/device/LLM, and budget-based allocation.

**Architecture:** Replace cost-basis holdings with watchlist entries. Replace PPO/stable-baselines3 optimizer with a scipy SLSQP Markowitz solver that injects a risk-free CASH asset using a user-supplied rf. Persist a single latest allocation per portfolio. Rename `rl_optimizer` graph node to `optimizer` end-to-end. Add a yellow runtime banner across all tabs and a `core/runtime.DEVICE` single source of truth for CUDA/MPS/CPU.

**Tech Stack:** Python 3.13, Gradio 5, SQLAlchemy 2, Pydantic v2, scipy.optimize (SLSQP), LangGraph, langchain-groq, Plotly, yfinance, pytest, torch (for device detection only).

**Spec:** [docs/superpowers/specs/2026-04-17-watchlist-optimizer-design.md](../specs/2026-04-17-watchlist-optimizer-design.md)

---

## Pre-flight

Run once before starting:

```bash
cd /Users/ik/UVcodes/stocksPortfolio
uv sync
```

Create the tests directory skeleton:

```bash
mkdir -p tests && touch tests/__init__.py
```

---

## Task 1: `core/runtime.py` — device detection

**Files:**
- Create: `core/runtime.py`
- Create: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_runtime.py`:

```python
from unittest.mock import patch
from core import runtime


def test_detect_device_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runtime.detect_device() == "cuda"


def test_detect_device_mps(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    assert runtime.detect_device() == "mps"


def test_detect_device_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    assert runtime.detect_device() == "cpu"


def test_module_level_device_constant_exists():
    assert runtime.DEVICE in {"cuda", "mps", "cpu"}
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.runtime'`.

- [ ] **Step 3: Implement `core/runtime.py`**

```python
"""Runtime environment helpers: device (cuda/mps/cpu) detection."""
from __future__ import annotations


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


DEVICE: str = detect_device()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add core/runtime.py tests/__init__.py tests/test_runtime.py
git commit -m "feat(runtime): add detect_device helper and DEVICE constant"
```

---

## Task 2: `services/parsing.py` — `parse_rf` helper

**Files:**
- Create: `services/parsing.py`
- Create: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_parsing.py`:

```python
import pytest
from services.parsing import parse_rf


@pytest.mark.parametrize("text, expected", [
    ("4.56%", 0.0456),
    ("4.56", 0.0456),
    ("0.0456", 0.0456),
    ("   5 % ", 0.05),
    ("0%", 0.0),
    ("20%", 0.20),
])
def test_parse_rf_valid(text, expected):
    assert parse_rf(text) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("text, expected", [
    ("25%", 0.20),
    ("-1%", 0.0),
    ("-0.05", 0.0),
    ("100", 0.20),
])
def test_parse_rf_clamped(text, expected):
    assert parse_rf(text) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("text", ["", "abc", "%", "4.5.6"])
def test_parse_rf_invalid(text):
    with pytest.raises(ValueError):
        parse_rf(text)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.parsing'`.

- [ ] **Step 3: Implement `services/parsing.py`**

```python
"""Small parsers for user-supplied UI values."""
from __future__ import annotations

_RF_MIN = 0.0
_RF_MAX = 0.20


def parse_rf(text: str) -> float:
    """
    Parse a risk-free-rate string into a decimal in [0.0, 0.20].

    Accepts:
      "4.56%"   -> 0.0456
      "4.56"    -> 0.0456   (bare number >= 1 is treated as a percent)
      "0.0456"  -> 0.0456   (bare number <  1 is treated as a decimal)
      "   5 % " -> 0.05
    Clamps to [0.0, 0.20]. Raises ValueError on non-numeric input.
    """
    if not isinstance(text, str):
        raise ValueError("parse_rf: expected str")
    s = text.strip()
    if not s:
        raise ValueError("parse_rf: empty")
    had_percent = s.endswith("%")
    if had_percent:
        s = s[:-1].strip()
    try:
        value = float(s)
    except ValueError as e:
        raise ValueError(f"parse_rf: not a number: {text!r}") from e
    if had_percent or abs(value) >= 1.0:
        value = value / 100.0
    return max(_RF_MIN, min(_RF_MAX, value))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_parsing.py -v`
Expected: all parametrized cases pass.

- [ ] **Step 5: Commit**

```bash
git add services/parsing.py tests/test_parsing.py
git commit -m "feat(parsing): add parse_rf accepting '4.56%' / '4.56' / '0.0456'"
```

---

## Task 3: Data model — drop cost-basis cols, add `portfolio_allocations`

**Files:**
- Modify: `core/models.py:24-102`
- Modify: `core/database.py:14-82`

- [ ] **Step 1: Rewrite `core/models.py`**

Replace the full contents of `core/models.py` with:

```python
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
    id           = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"),
                          nullable=False)
    ticker       = Column(String(10), nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
```

- [ ] **Step 2: Update `core/database.py` migration**

Open `core/database.py`, and replace the `_maybe_migrate()` body with a minimal drop-and-recreate that handles schema changes for this release. Keep the `init_db()`/`ensure_default_portfolio()` structure untouched. Use the following replacement (edit only `_maybe_migrate` — leave other functions alone):

```python
def _maybe_migrate() -> None:
    """
    Schema-change migration for the watchlist release.

    On old DBs, the 'holdings' table has columns (shares, purchase_price)
    which no longer exist on the model, and there is no
    'portfolio_allocations' table. We detect either condition with a single
    PRAGMA, then drop holdings + portfolio_allocations + re-create.
    (Portfolios table is unchanged so we preserve it.)
    """
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "holdings" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("holdings")}
    needs_migration = (
        "shares" in cols
        or "purchase_price" in cols
        or "portfolio_allocations" not in insp.get_table_names()
    )
    if not needs_migration:
        return
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS portfolio_allocations"))
        conn.execute(text("DROP TABLE IF EXISTS holdings"))
```

- [ ] **Step 3: Write migration test**

Create `tests/test_database_migration.py`:

```python
import os
import tempfile
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def test_migration_drops_stale_holdings(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    url = f"sqlite:///{tmp.name}"
    try:
        eng = create_engine(url)
        with eng.begin() as c:
            c.execute(text(
                "CREATE TABLE holdings ("
                "id INTEGER PRIMARY KEY, portfolio_id INTEGER, ticker TEXT, "
                "shares REAL, purchase_price REAL)"
            ))
            c.execute(text(
                "CREATE TABLE portfolios ("
                "id INTEGER PRIMARY KEY, name TEXT, description TEXT, "
                "created_at DATETIME, updated_at DATETIME)"
            ))

        from core import database as db_mod
        monkeypatch.setattr(db_mod, "engine", eng)
        monkeypatch.setattr(
            db_mod, "SessionLocal",
            sessionmaker(bind=eng, autocommit=False, autoflush=False),
        )
        db_mod._maybe_migrate()

        from sqlalchemy import inspect
        names = set(inspect(eng).get_table_names())
        assert "holdings" not in names
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_database_migration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/models.py core/database.py tests/test_database_migration.py
git commit -m "refactor(data): watchlist holdings + portfolio_allocations table"
```

---

## Task 4: `services/optimizer.py` — Markowitz engine

**Files:**
- Create: `services/optimizer.py`
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_optimizer.py`:

```python
import numpy as np
import pandas as pd
import pytest

import services.optimizer as opt_mod
from services.optimizer import optimize_portfolio, build_plots


@pytest.fixture
def synthetic_prices(monkeypatch):
    """3 tickers, 300 trading days, deterministic."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2023-01-01", periods=300)
    mu_daily = {"AAA": 0.0008, "BBB": 0.0003, "CCC": 0.0005}
    sig_daily = {"AAA": 0.012,  "BBB": 0.018,  "CCC": 0.009}
    frames = {}
    for t in ("AAA", "BBB", "CCC"):
        rets = rng.normal(mu_daily[t], sig_daily[t], size=300)
        close = 100 * np.exp(np.cumsum(rets))
        frames[t] = pd.DataFrame({"Close": close}, index=idx)

    def fake_get_historical(ticker, period="2y"):
        return frames[ticker].copy()

    def fake_get_stock_info(ticker):
        return {"price": float(frames[ticker]["Close"].iloc[-1])}

    monkeypatch.setattr(opt_mod, "get_historical",  fake_get_historical)
    monkeypatch.setattr(opt_mod, "get_stock_info",  fake_get_stock_info)
    return frames


def test_weights_sum_to_one(synthetic_prices):
    res = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                             target_vol=0.15, frontier_samples=200)
    total = sum(v["weight"] for v in res["allocations"].values()) \
          + res["cash_dollars"] / 100_000
    assert total == pytest.approx(1.0, abs=1e-4)


def test_no_ticker_exceeds_cap(synthetic_prices):
    res = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                             target_vol=0.30, max_weight=0.40,
                             frontier_samples=200)
    for v in res["allocations"].values():
        assert v["weight"] <= 0.40 + 1e-4


def test_too_few_tickers_raises(synthetic_prices):
    with pytest.raises(ValueError, match="at least 3"):
        optimize_portfolio(["AAA", "BBB"], budget=100_000,
                           target_vol=0.15, frontier_samples=100)


def test_sharpe_uses_supplied_rf(synthetic_prices):
    r1 = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                            target_vol=0.20, risk_free_rate=0.0,
                            frontier_samples=100)
    r2 = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                            target_vol=0.20, risk_free_rate=0.10,
                            frontier_samples=100)
    assert r1["metrics"]["risk_free_rate"] == 0.0
    assert r2["metrics"]["risk_free_rate"] == 0.10
    assert r1["metrics"]["sharpe"] > r2["metrics"]["sharpe"]


def test_infeasible_target_vol_warns(synthetic_prices):
    res = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                             target_vol=0.001, frontier_samples=100)
    assert any("infeasible" in w.lower() or "min-var" in w.lower()
               for w in res["warnings"])


def test_build_plots_returns_three_figures(synthetic_prices):
    res = optimize_portfolio(["AAA", "BBB", "CCC"], budget=100_000,
                             target_vol=0.15, frontier_samples=100)
    figs = build_plots(res)
    assert len(figs) == 3
    for f in figs:
        assert hasattr(f, "to_dict")
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.optimizer'`.

- [ ] **Step 3: Implement `services/optimizer.py`**

```python
"""Markowitz mean-variance optimizer with always-on risk-free CASH asset."""
from __future__ import annotations
import json
import logging
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from services.stock_service import get_historical, get_stock_info

log = logging.getLogger(__name__)

TRADING_DAYS = 252
MIN_TRADING_DAYS = 60
CASH = "CASH"


def _collect_returns(tickers: list[str], lookback: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        df = get_historical(t, period=lookback)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames[t] = df["Close"]
    if len(frames) < 3:
        raise ValueError(f"need at least 3 tickers with price history, got {len(frames)}")
    prices = pd.concat(frames.values(), axis=1, keys=frames.keys()).dropna()
    if len(prices) < MIN_TRADING_DAYS:
        raise ValueError(
            f"not enough history: {len(prices)} rows < {MIN_TRADING_DAYS}"
        )
    return prices.pct_change().dropna()


def _solve_max_return(mu: np.ndarray, cov: np.ndarray, target_var: float,
                      max_w_risky: float, n_risky: int) -> tuple[np.ndarray, bool]:
    """Solve: maximize wᵀμ  s.t. wᵀΣw ≤ target_var, Σw = 1, bounds."""
    n = len(mu)  # n_risky + 1 (cash is last)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w_risky)] * n_risky + [(0.0, 1.0)]  # cash unbounded up to 1
    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: target_var - w @ cov @ w},
    ]
    res = minimize(lambda w: -float(w @ mu), x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 300})
    return res.x, res.success


def _solve_min_var(mu: np.ndarray, cov: np.ndarray,
                   max_w_risky: float, n_risky: int) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w_risky)] * n_risky + [(0.0, 1.0)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(lambda w: float(w @ cov @ w), x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 300})
    return res.x


def optimize_portfolio(
    tickers: list[str],
    budget: float,
    target_vol: float,
    lookback: str = "2y",
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
    frontier_samples: int = 5_000,
) -> dict[str, Any]:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if len(set(tickers)) < 3:
        raise ValueError("need at least 3 unique tickers")

    returns_df = _collect_returns(tickers, lookback)
    used = list(returns_df.columns)

    mu_annual  = returns_df.mean().values * TRADING_DAYS
    cov_annual = returns_df.cov().values * TRADING_DAYS
    n_risky = len(used)

    mu_aug  = np.concatenate([mu_annual, [risk_free_rate]])
    cov_aug = np.zeros((n_risky + 1, n_risky + 1))
    cov_aug[:n_risky, :n_risky] = cov_annual

    warnings: list[str] = []
    target_var = target_vol ** 2

    w_star, ok = _solve_max_return(mu_aug, cov_aug, target_var,
                                   max_weight, n_risky)
    realized_vol = float(np.sqrt(w_star @ cov_aug @ w_star))
    if (not ok) or realized_vol > target_vol + 1e-3:
        w_star = _solve_min_var(mu_aug, cov_aug, max_weight, n_risky)
        warnings.append(
            f"target_vol={target_vol:.3f} infeasible; fell back to min-var portfolio"
        )

    w_star = np.clip(w_star, 0.0, None)
    w_star = w_star / w_star.sum()

    exp_ret = float(w_star @ mu_aug)
    exp_vol = float(np.sqrt(w_star @ cov_aug @ w_star))
    sharpe  = (exp_ret - risk_free_rate) / exp_vol if exp_vol > 1e-9 else 0.0

    allocations: dict[str, dict[str, float]] = {}
    for i, t in enumerate(used):
        w = float(w_star[i])
        if w < 1e-6:
            continue
        try:
            price = float(get_stock_info(t).get("price") or 0.0)
        except Exception as e:
            log.warning("price fetch failed for %s: %s", t, e)
            price = 0.0
        dollars = w * budget
        shares = (dollars / price) if price > 0 else 0.0
        allocations[t] = {"weight": w, "dollars": dollars,
                          "shares": shares, "price": price}
    cash_dollars = float(w_star[-1]) * budget

    frontier_points = _build_frontier(mu_aug, cov_aug, max_weight, n_risky,
                                      frontier_samples)

    return {
        "allocations":     allocations,
        "cash_dollars":    cash_dollars,
        "metrics": {
            "expected_return": exp_ret,
            "expected_vol":    exp_vol,
            "sharpe":          sharpe,
            "target_vol":      target_vol,
            "risk_free_rate":  risk_free_rate,
        },
        "frontier_points": frontier_points,
        "returns_df":      returns_df,
        "warnings":        warnings,
    }


def _build_frontier(mu: np.ndarray, cov: np.ndarray, max_w_risky: float,
                    n_risky: int, samples: int) -> list[dict[str, float]]:
    w_min = _solve_min_var(mu, cov, max_w_risky, n_risky)
    v_min = float(np.sqrt(w_min @ cov @ w_min))
    v_max = float(np.sqrt(np.max(np.diag(cov))))
    if v_max <= v_min:
        return [{"vol": v_min, "return": float(w_min @ mu)}]
    grid = np.linspace(v_min, v_max, max(2, int(samples)))
    pts: list[dict[str, float]] = []
    for v in grid:
        w, ok = _solve_max_return(mu, cov, v * v, max_w_risky, n_risky)
        if not ok:
            continue
        pts.append({"vol": float(np.sqrt(w @ cov @ w)),
                    "return": float(w @ mu)})
    return pts


def build_plots(result: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    allocs = result["allocations"]
    cash   = result["cash_dollars"]
    budget = cash + sum(v["dollars"] for v in allocs.values())

    labels  = list(allocs.keys()) + [CASH]
    dollars = [v["dollars"] for v in allocs.values()] + [cash]
    weights = [d / budget if budget else 0.0 for d in dollars]

    fig_pie = go.Figure(go.Pie(labels=labels, values=weights, hole=0.35))
    fig_pie.update_layout(title="Allocation (weights)", template="plotly_dark")

    fig_bar = go.Figure(go.Bar(x=labels, y=dollars))
    fig_bar.update_layout(title="Dollar allocation",
                          yaxis_title="USD", template="plotly_dark")

    pts = result["frontier_points"]
    vols  = [p["vol"]    for p in pts]
    rets  = [p["return"] for p in pts]
    fig_f = go.Figure(go.Scatter(x=vols, y=rets, mode="lines",
                                 name="Efficient frontier"))
    m = result["metrics"]
    fig_f.add_trace(go.Scatter(x=[m["expected_vol"]], y=[m["expected_return"]],
                               mode="markers", marker=dict(size=12, color="#00D4FF"),
                               name="Chosen"))
    fig_f.update_layout(title="Efficient frontier",
                        xaxis_title="Volatility (σ)",
                        yaxis_title="Expected return",
                        template="plotly_dark")

    return fig_pie, fig_bar, fig_f


def save_allocation(portfolio_id: int, result: dict[str, Any],
                    budget: float, target_vol: float, lookback: str,
                    commentary: str = "") -> None:
    """Persist (overwrite) the single allocation row for this portfolio."""
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    m = result["metrics"]
    payload = json.dumps(result["allocations"])
    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, portfolio_id)
        if row is None:
            row = PortfolioAllocationDB(portfolio_id=portfolio_id)
            s.add(row)
        row.budget           = float(budget)
        row.target_vol       = float(target_vol)
        row.lookback         = lookback
        row.expected_return  = float(m["expected_return"])
        row.expected_vol     = float(m["expected_vol"])
        row.sharpe           = float(m["sharpe"])
        row.risk_free_rate   = float(m["risk_free_rate"])
        row.cash_dollars     = float(result["cash_dollars"])
        row.allocations_json = payload
        row.commentary       = commentary
        s.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add services/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): Markowitz SLSQP with cash asset + frontier"
```

---

## Task 5: Delete old `services/rl_optimizer.py`

**Files:**
- Delete: `services/rl_optimizer.py`

- [ ] **Step 1: Delete the file**

```bash
git rm services/rl_optimizer.py
```

- [ ] **Step 2: Verify nothing imports it**

Run: `uv run python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p)]"`
Expected: exits cleanly (no import errors yet — they come in Task 6).

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove services/rl_optimizer.py (PPO/stable-baselines3)"
```

---

## Task 6: Rename `rl_optimizer` → `optimizer` across agent graph

**Files:**
- Modify: `agents/state.py:19,37`
- Rename/rewrite: `agents/rl_optimizer_agent.py` → `agents/optimizer_agent.py`
- Modify: `agents/graph.py:42,56,64-65`
- Modify: `agents/supervisor.py:32,51-53,60`
- Modify: `ui/components/chatbot.py:14-20`

- [ ] **Step 1: Update `agents/state.py`**

Edit line 19: rename `rl_result: dict` → `optimizer_result: dict`.
Edit line 37: in `empty_state`, rename `rl_result={}` → `optimizer_result={}`.

- [ ] **Step 2: Write the new `agents/optimizer_agent.py`**

```python
"""LangGraph node: Markowitz optimizer."""
from __future__ import annotations
import logging
import re
from services.optimizer import optimize_portfolio, build_plots, save_allocation
from services.database import get_holdings  # NOTE: may live in core.database; see step 3

log = logging.getLogger(__name__)

_BUDGET_RE  = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:k|K|thousand)?")
_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


def _extract_budget(msg: str) -> float | None:
    m = _BUDGET_RE.search(msg or "")
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    if re.search(r"\d\s*(k|K|thousand)", msg):
        val *= 1000
    return val if val >= 1000 else None


def _extract_target_vol(msg: str) -> float | None:
    for m in _PERCENT_RE.finditer(msg or ""):
        v = float(m.group(1)) / 100
        if 0.02 <= v <= 0.60:
            return v
    return None


def optimizer_node(state: dict) -> dict:
    if "optimizer" not in state.get("active_agents", []):
        return state
    portfolio_id = state.get("portfolio_id", 1)
    msg = state.get("user_message", "")

    budget = _extract_budget(msg)
    target_vol = _extract_target_vol(msg)
    if budget is None or target_vol is None:
        state["optimizer_result"] = {
            "error": "need budget and target risk (e.g. '$50k at 18%')"
        }
        state.setdefault("agent_status", []).append(
            "🤖 Optimizer: missing budget/target_vol in message"
        )
        return state

    from core.database import SessionLocal
    from core.models import HoldingDB
    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]

    try:
        result = optimize_portfolio(
            tickers=tickers, budget=budget, target_vol=target_vol,
            lookback="2y", risk_free_rate=0.04,
        )
    except Exception as e:
        log.exception("optimizer failed")
        state["optimizer_result"] = {"error": str(e)}
        state.setdefault("agent_status", []).append(f"🤖 Optimizer failed: {e}")
        return state

    fig_pie, fig_bar, fig_frontier = build_plots(result)
    save_allocation(portfolio_id, result, budget=budget,
                    target_vol=target_vol, lookback="2y")

    state["optimizer_result"] = result
    state.setdefault("charts", []).extend([fig_pie, fig_bar, fig_frontier])
    state.setdefault("active_agents", []).append("optimizer")
    state.setdefault("agent_status", []).append(
        f"🤖 Optimizer: {len(result['allocations'])} tickers, "
        f"cash ${result['cash_dollars']:.0f}"
    )
    return state
```

- [ ] **Step 3: Remove obsolete agent file**

```bash
git rm agents/rl_optimizer_agent.py
```

- [ ] **Step 4: Update `agents/graph.py`**

Edit line 42: `from agents.rl_optimizer_agent import rl_optimizer_node` → `from agents.optimizer_agent import optimizer_node`.
Edit line 56: `g.add_node("rl_optimizer", rl_optimizer_node)` → `g.add_node("optimizer", optimizer_node)`.
Edit lines 64–65: replace every `"rl_optimizer"` with `"optimizer"` in edges.

- [ ] **Step 5: Update `agents/supervisor.py`**

Edit line 32: `_KEYWORD_MAP["rl_optimizer"]` → `_KEYWORD_MAP["optimizer"]`.
Edit lines 51–53: replace `"rl_optimizer"` with `"optimizer"` in the system prompt, and change the phrase "RL-based weight optimisation" to "Markowitz mean-variance optimisation with cash reserve".
Edit line 60: `"Include rl_optimizer whenever..."` → `"Include optimizer whenever..."`.

- [ ] **Step 6: Update `ui/components/chatbot.py`**

Edit lines 14–20: change the `_AGENT_LABELS` dict entry `"rl_optimizer": "🤖 RL Optimizer"` → `"optimizer": "🤖 Optimizer"`.

- [ ] **Step 7: Run agent smoke test**

```bash
uv run python -c "from agents.graph import get_graph; g = get_graph(); print('OK', list(g.get_graph().nodes))"
```

Expected: prints `OK [...'optimizer'...]` with no exception.

- [ ] **Step 8: Commit**

```bash
git add agents/state.py agents/optimizer_agent.py agents/graph.py \
        agents/supervisor.py ui/components/chatbot.py
git commit -m "refactor(agents): rename rl_optimizer -> optimizer end-to-end"
```

---

## Task 7: Runtime banner + theme CSS

**Files:**
- Modify: `ui/theme.py` (append two CSS rules)
- Modify: `ui/frontend.py` (inject banner above tabs)
- Modify: `agents/knowledge_base_agent.py` (propagate `DEVICE` to HF embeddings; only if the file instantiates `HuggingFaceEmbeddings` — skip if not)

- [ ] **Step 1: Append CSS rules to `ui/theme.py`**

At the end of the `CUSTOM_CSS` string (just before its closing `"""`), add:

```css
/* ── Runtime banner (top of UI) ───────────────────────── */
.runtime-banner {
    color: #FFD700 !important;
    background: #0d1118 !important;
    border: 1px solid #2a2a1a !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    margin: 0 0 10px !important;
    font-family: ui-monospace, "JetBrains Mono", monospace !important;
    font-size: .82rem !important;
    letter-spacing: .02em;
}

/* ── Watchlist dataframe: black cell text ─────────────── */
.watchlist-df table td,
.watchlist-df .table-wrap td,
.watchlist-df tbody td {
    color: #000 !important;
}
```

- [ ] **Step 2: Write banner helper in `ui/frontend.py`**

Near the top of the file, right below the existing imports block, add:

```python
from core import runtime
from core import config as _cfg


def _llm_label() -> str:
    if _cfg.GROQ_API_KEY:
        return _cfg.GROQ_MODEL
    if getattr(_cfg, "OLLAMA_MODEL", ""):
        return f"ollama:{_cfg.OLLAMA_MODEL}"
    return "<no LLM key>"


def _env_label() -> str:
    if _cfg.IS_HF_SPACE:
        import os
        return f"HF Space: {os.environ.get('SPACE_ID', '<unknown>')}"
    return "locally"


def _runtime_banner_html() -> str:
    text = (
        f"Running: {_env_label()} | "
        f"Device: {runtime.DEVICE} | "
        f"LLM: {_llm_label()}"
    )
    return f'<div class="runtime-banner">{text}</div>'
```

- [ ] **Step 3: Mount the banner at the top of the Blocks layout**

Inside the function that builds the Gradio Blocks (search for `with gr.Blocks(` in `ui/frontend.py`), add a single `gr.HTML(_runtime_banner_html())` call as the first child inside the `with gr.Blocks(...)` block, before the existing header/title.

Example context:

```python
with gr.Blocks(theme=get_theme(), css=CUSTOM_CSS, title="…") as demo:
    gr.HTML(_runtime_banner_html())           # ← NEW — must be first child
    gr.HTML(_MERMAID_HTML)                     # existing — stays below banner
    # … rest of the existing layout
```

- [ ] **Step 4: Propagate device to embeddings (only if applicable)**

Run:

```bash
grep -rn "HuggingFaceEmbeddings" agents/ services/
```

For every `HuggingFaceEmbeddings(...)` instantiation found, edit it to pass `model_kwargs={"device": runtime.DEVICE}` (importing `from core.runtime import DEVICE` at the top of that file). Example:

```python
from core.runtime import DEVICE
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
)
```

If no hits, skip this step.

- [ ] **Step 5: Manual smoke**

```bash
uv run python -c "from ui.frontend import _runtime_banner_html; print(_runtime_banner_html())"
```

Expected: prints HTML containing `Running: locally | Device: <cuda|mps|cpu> | LLM: <model>`.

- [ ] **Step 6: Commit**

```bash
git add ui/theme.py ui/frontend.py agents/knowledge_base_agent.py
git commit -m "feat(ui): yellow runtime banner (env/device/LLM) + device propagation"
```

---

## Task 8: Portfolio tab — ticker-only watchlist with pinned CASH

**Files:**
- Modify: `ui/frontend.py` (Portfolio tab section + `_load_summary`, `_holdings_df`, `add_holding`, remove `update_holding`, `remove_holding`)

- [ ] **Step 1: Rewrite the watchlist loader helpers**

In `ui/frontend.py`, replace the existing `_load_summary` (around lines 163–206) and `_holdings_df` (around 209–227) with:

```python
CASH_ROW = ("CASH", 1.00, 0.00)   # (ticker, price, daily Δ%)


def _watchlist_df(portfolio_id: int) -> list[list]:
    """Return rows for the watchlist dataframe; CASH pinned at top."""
    from core.database import SessionLocal
    from core.models import HoldingDB
    from services.stock_service import get_stock_info

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]

    rows: list[list] = [[CASH_ROW[0], f"${CASH_ROW[1]:.2f}",
                         f"{CASH_ROW[2]:+.2f}%"]]
    for t in sorted(set(tickers)):
        try:
            info = get_stock_info(t) or {}
            price  = float(info.get("price") or 0.0)
            change = float(info.get("change_percent") or 0.0)
            rows.append([t, f"${price:.2f}", f"{change:+.2f}%"])
        except Exception:
            rows.append([t, "—", "—"])
    return rows
```

- [ ] **Step 2: Replace `add_holding` and delete update/remove variants for shares/price**

In `ui/frontend.py`, replace the `add_holding(ticker, shares, price)` function (around lines 265–290) with:

```python
def add_ticker(ticker: str, portfolio_id: int):
    from core.database import SessionLocal
    from core.models import HoldingDB, HoldingCreate
    try:
        payload = HoldingCreate(ticker=ticker)
    except Exception as e:
        return gr.update(value=""), f"❌ {e}", gr.update()
    with SessionLocal() as s:
        exists = (s.query(HoldingDB)
                   .filter_by(portfolio_id=portfolio_id, ticker=payload.ticker)
                   .first())
        if exists:
            return gr.update(value=""), f"ℹ️ {payload.ticker} already on watchlist", \
                   gr.update()
        s.add(HoldingDB(portfolio_id=portfolio_id, ticker=payload.ticker))
        s.commit()
    rows = _watchlist_df(portfolio_id)
    tickers_for_dropdown = [r[0] for r in rows if r[0] != "CASH"]
    return (gr.update(value=""),
            f"✅ Added {payload.ticker}",
            gr.update(choices=tickers_for_dropdown, value=None),
            rows)


def remove_ticker(ticker: str, portfolio_id: int):
    from core.database import SessionLocal
    from core.models import HoldingDB
    if not ticker or ticker == "CASH":
        rows = _watchlist_df(portfolio_id)
        return "⚠️ pick a ticker (CASH is not removable)", rows, gr.update()
    with SessionLocal() as s:
        row = (s.query(HoldingDB)
                .filter_by(portfolio_id=portfolio_id, ticker=ticker).first())
        if row:
            s.delete(row); s.commit()
    rows = _watchlist_df(portfolio_id)
    tickers_for_dropdown = [r[0] for r in rows if r[0] != "CASH"]
    return f"🗑️ Removed {ticker}", rows, \
           gr.update(choices=tickers_for_dropdown, value=None)
```

Delete `update_holding` (around lines 316–339) entirely.

- [ ] **Step 3: Replace the Portfolio tab UI block**

Locate the Portfolio tab (around lines 490–523) and replace with:

```python
with gr.Tab("📋 Portfolio"):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Add ticker")
            ticker_in  = gr.Textbox(label="Ticker (e.g. AAPL)", max_lines=1)
            add_btn    = gr.Button("Add", variant="primary")
            add_status = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Remove ticker")
            remove_dd     = gr.Dropdown(label="Select to remove", choices=[])
            remove_btn    = gr.Button("Remove", variant="secondary")
            remove_status = gr.Markdown()

    watchlist_df = gr.Dataframe(
        headers=["Ticker", "Price", "Daily Δ%"],
        datatype=["str", "str", "str"],
        interactive=False,
        elem_classes=["watchlist-df"],
        label="Watchlist (CASH pinned)",
    )

    add_btn.click(
        add_ticker,
        inputs=[ticker_in, portfolio_id_state],
        outputs=[ticker_in, add_status, remove_dd, watchlist_df],
    )
    remove_btn.click(
        remove_ticker,
        inputs=[remove_dd, portfolio_id_state],
        outputs=[remove_status, watchlist_df, remove_dd],
    )
```

Make sure `portfolio_id_state` is the existing `gr.State` holding the active portfolio_id. On load, wire the demo's `.load(...)` handler to also populate `watchlist_df` and `remove_dd`.

- [ ] **Step 4: Smoke-launch Gradio**

```bash
uv run python -c "from ui.frontend import build_ui; build_ui()"
```

Expected: no ImportError, no missing-attribute errors. (Do not actually `.launch()` in the plan script — just build.)

- [ ] **Step 5: Commit**

```bash
git add ui/frontend.py
git commit -m "feat(ui): watchlist-only portfolio tab with pinned CASH row"
```

---

## Task 9: Optimizer tab with paired risk-free rate control

**Files:**
- Modify: `ui/components/optimizer_ui.py` (or the Optimizer tab section inside `ui/frontend.py` — follow whichever is authoritative)
- Uses: `services.parsing.parse_rf` from Task 2, `services.optimizer.optimize_portfolio` from Task 4

- [ ] **Step 1: Locate the Optimizer tab block**

Search: `grep -n "opt_period\|opt_steps\|RL Optimiser\|RL Optimizer" ui/frontend.py ui/components/optimizer_ui.py`
The tab currently lives around `ui/frontend.py:528-567`. Replace that block with the new layout below.

- [ ] **Step 2: Replace the Optimizer tab UI**

```python
with gr.Tab("🤖 Optimizer"):
    with gr.Row():
        with gr.Column(scale=1):
            opt_budget = gr.Number(
                label="Budget ($)",
                value=100_000, minimum=1_000, step=1_000,
            )
            opt_target_vol = gr.Slider(
                label="Target risk (annual vol, %)",
                minimum=5, maximum=40, value=15, step=0.5,
            )
            with gr.Row():
                opt_rf_slider = gr.Slider(
                    label="Risk-free rate (%)",
                    minimum=0, maximum=20, value=4.00, step=0.25,
                )
                opt_rf_text = gr.Textbox(
                    label="…or type (e.g. 4.56%)",
                    value="4.00%", max_lines=1,
                )
            opt_lookback = gr.Dropdown(
                label="Lookback window",
                choices=["1y", "2y", "3y", "5y"], value="2y",
            )
            opt_frontier = gr.Slider(
                label="Frontier samples (higher = smoother, slower)",
                minimum=2_000, maximum=999_999, value=5_000, step=1_000,
            )
            opt_btn = gr.Button("Optimize", variant="primary")
        with gr.Column(scale=2):
            opt_status    = gr.Markdown()
            with gr.Row():
                m_ret  = gr.Textbox(label="Expected return", interactive=False)
                m_vol  = gr.Textbox(label="Expected vol",    interactive=False)
                m_shrp = gr.Textbox(label="Sharpe",          interactive=False)
                m_cash = gr.Textbox(label="Cash reserve ($)",interactive=False)
            fig_pie      = gr.Plot(label="Allocation")
            fig_bar      = gr.Plot(label="Dollar allocation")
            fig_frontier = gr.Plot(label="Efficient frontier")
            opt_commentary = gr.Markdown()


# ── Two-way binding: slider <-> textbox ──────────────────────────
def _sync_slider_to_text(pct: float):
    return f"{float(pct):.2f}%"


def _sync_text_to_slider(text: str, current_pct: float):
    try:
        dec = parse_rf(text)            # services.parsing.parse_rf
    except Exception:
        gr.Warning(f"Invalid rate: {text!r}")
        return gr.update(), gr.update(value=f"{current_pct:.2f}%")
    return gr.update(value=dec * 100), gr.update(value=f"{dec * 100:.2f}%")


opt_rf_slider.change(_sync_slider_to_text,
                     inputs=[opt_rf_slider], outputs=[opt_rf_text])
opt_rf_text.submit(_sync_text_to_slider,
                   inputs=[opt_rf_text, opt_rf_slider],
                   outputs=[opt_rf_slider, opt_rf_text])
```

- [ ] **Step 3: Wire the Optimize button**

```python
from services.optimizer import optimize_portfolio, build_plots, save_allocation
from services.parsing  import parse_rf


def run_optimize(budget, target_vol_pct, rf_text, lookback, frontier_samples,
                 portfolio_id):
    from core.database import SessionLocal
    from core.models import HoldingDB
    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
    try:
        rf = parse_rf(rf_text)
    except Exception as e:
        return (f"❌ {e}", "", "", "", "", None, None, None, "")
    try:
        result = optimize_portfolio(
            tickers=tickers,
            budget=float(budget),
            target_vol=float(target_vol_pct) / 100.0,
            lookback=lookback,
            risk_free_rate=rf,
            frontier_samples=int(frontier_samples),
        )
    except Exception as e:
        return (f"❌ {e}", "", "", "", "", None, None, None, "")
    fig_p, fig_b, fig_f = build_plots(result)
    save_allocation(portfolio_id, result, budget=float(budget),
                    target_vol=float(target_vol_pct) / 100.0,
                    lookback=lookback)
    m = result["metrics"]
    commentary = "\n\n".join([f"- {w}" for w in result["warnings"]]) or \
                 "Optimization complete."
    return (
        "✅ Optimized",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        f"{m['sharpe']:.3f}",
        f"${result['cash_dollars']:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
    )


opt_btn.click(
    run_optimize,
    inputs=[opt_budget, opt_target_vol, opt_rf_text, opt_lookback,
            opt_frontier, portfolio_id_state],
    outputs=[opt_status, m_ret, m_vol, m_shrp, m_cash,
             fig_pie, fig_bar, fig_frontier, opt_commentary],
)
```

- [ ] **Step 4: Manual smoke**

Run: `uv run python main.py` (or however the app launches), open the UI, add 3 tickers, click Optimize with default values. Confirm:
- Banner shows at top with yellow text.
- Typing `4.56%` in the rf textbox and pressing Enter moves the slider to 4.56.
- Dragging the slider updates the textbox to `X.XX%`.
- Results populate, charts render, Sharpe is finite.

- [ ] **Step 5: Commit**

```bash
git add ui/frontend.py ui/components/optimizer_ui.py
git commit -m "feat(ui): optimizer tab (budget/vol/rf pair/lookback/frontier)"
```

---

## Task 10: Dashboard tab (hybrid — live watchlist + last plan)

**Files:**
- Modify: `ui/components/dashboard.py`
- Modify: `ui/frontend.py` (Dashboard tab section)

- [ ] **Step 1: Rewrite `ui/components/dashboard.py` chart helpers**

Replace the entire file with:

```python
"""Dashboard helpers: live-watchlist rows + last-plan rendering."""
from __future__ import annotations
import json
import plotly.graph_objects as go


def live_watchlist_rows(portfolio_id: int) -> list[list]:
    """CASH pinned; columns: Ticker | Price | 1d % | 1mo % | 3mo % | 1y %."""
    from core.database import SessionLocal
    from core.models import HoldingDB
    from services.stock_service import get_stock_info, get_historical

    rows: list[list] = [["CASH", "$1.00", "0.00%", "0.00%", "0.00%", "0.00%"]]
    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
    for t in sorted(set(tickers)):
        info = get_stock_info(t) or {}
        price = float(info.get("price") or 0.0)
        hist  = get_historical(t, period="1y")
        pcts = []
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            last = closes.iloc[-1]
            for days in (1, 21, 63, 252):
                if len(closes) > days:
                    pcts.append((last / closes.iloc[-days-1] - 1) * 100)
                else:
                    pcts.append(0.0)
        else:
            pcts = [0.0, 0.0, 0.0, 0.0]
        rows.append([t, f"${price:.2f}",
                     f"{pcts[0]:+.2f}%", f"{pcts[1]:+.2f}%",
                     f"{pcts[2]:+.2f}%", f"{pcts[3]:+.2f}%"])
    return rows


def last_plan_rows(portfolio_id: int) -> tuple[list[list], dict | None]:
    """Returns (dollar_rows, metrics) — rows include CASH; metrics None if no plan."""
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, portfolio_id)
        if row is None:
            return [], None
        allocs = json.loads(row.allocations_json)
        rows = []
        for ticker, v in allocs.items():
            rows.append([ticker,
                         f"{v['weight']*100:.2f}%",
                         f"${v['dollars']:,.0f}",
                         f"{v['shares']:.2f}",
                         f"${v['price']:.2f}"])
        rows.insert(0, ["CASH", f"{(row.cash_dollars/row.budget)*100:.2f}%",
                        f"${row.cash_dollars:,.0f}", "—", "$1.00"])
        metrics = {
            "budget":          row.budget,
            "expected_return": row.expected_return,
            "expected_vol":    row.expected_vol,
            "sharpe":          row.sharpe,
            "cash_dollars":    row.cash_dollars,
            "created_at":      row.created_at,
        }
        return rows, metrics


def last_plan_pie(portfolio_id: int) -> go.Figure | None:
    rows, metrics = last_plan_rows(portfolio_id)
    if not rows:
        return None
    labels = [r[0] for r in rows]
    values = [float(r[1].rstrip("%")) for r in rows]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.35))
    fig.update_layout(title="Last optimized allocation", template="plotly_dark")
    return fig
```

- [ ] **Step 2: Rewrite the Dashboard tab block in `ui/frontend.py`**

Replace the existing Dashboard tab (search for `with gr.Tab("📊` or similar) with:

```python
with gr.Tab("📊 Dashboard"):
    gr.Markdown("### Live watchlist")
    dash_watch = gr.Dataframe(
        headers=["Ticker", "Price", "1d %", "1mo %", "3mo %", "1y %"],
        datatype=["str", "str", "str", "str", "str", "str"],
        interactive=False,
        elem_classes=["watchlist-df"],
    )
    gr.Markdown("### Last optimized plan")
    with gr.Row():
        d_budget = gr.Textbox(label="Budget",         interactive=False)
        d_ret    = gr.Textbox(label="Expected return",interactive=False)
        d_vol    = gr.Textbox(label="Expected vol",   interactive=False)
        d_shrp   = gr.Textbox(label="Sharpe",         interactive=False)
        d_cash   = gr.Textbox(label="Cash",           interactive=False)
    dash_pie = gr.Plot(label="Allocation")
    dash_table = gr.Dataframe(
        headers=["Ticker", "Weight", "Dollars", "Shares", "Price"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=False,
        elem_classes=["watchlist-df"],
    )
    d_stamp = gr.Markdown()

    refresh_btn = gr.Button("Refresh dashboard")


def refresh_dashboard(portfolio_id: int):
    from ui.components.dashboard import (
        live_watchlist_rows, last_plan_rows, last_plan_pie,
    )
    watch = live_watchlist_rows(portfolio_id)
    rows, m = last_plan_rows(portfolio_id)
    if m is None:
        return (watch, "—", "—", "—", "—", "—", None, [],
                "_Run the Optimizer to see your plan._")
    return (
        watch,
        f"${m['budget']:,.0f}",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        f"{m['sharpe']:.3f}",
        f"${m['cash_dollars']:,.0f}",
        last_plan_pie(portfolio_id),
        rows,
        f"_Last optimized: {m['created_at'].strftime('%Y-%m-%d %H:%M:%S')}_",
    )


refresh_btn.click(
    refresh_dashboard,
    inputs=[portfolio_id_state],
    outputs=[dash_watch, d_budget, d_ret, d_vol, d_shrp, d_cash,
             dash_pie, dash_table, d_stamp],
)
```

- [ ] **Step 3: Hook into `demo.load(...)`**

Find the existing `demo.load(...)` call and add `dash_watch, d_budget, d_ret, d_vol, d_shrp, d_cash, dash_pie, dash_table, d_stamp` to the call that refreshes the dashboard on app start (either by chaining a second `.load()` or by extending the existing handler). Pattern:

```python
demo.load(refresh_dashboard, inputs=[portfolio_id_state],
          outputs=[dash_watch, d_budget, d_ret, d_vol, d_shrp, d_cash,
                   dash_pie, dash_table, d_stamp])
```

- [ ] **Step 4: Smoke-launch**

```bash
uv run python -c "from ui.frontend import build_ui; build_ui()"
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add ui/components/dashboard.py ui/frontend.py
git commit -m "feat(ui): dashboard shows live watchlist + last optimized plan"
```

---

## Task 11: Update Mermaid diagram (remove PPO/SB3 references)

**Files:**
- Modify: `ui/frontend.py:8-43` (the `_MERMAID_HTML` constant)

- [ ] **Step 1: Edit the Mermaid block**

Read [ui/frontend.py:8-43](ui/frontend.py#L8-L43). Find any node text containing `PPO`, `stable-baselines3`, or `RL` and change:
- Node label "RL Optimizer (PPO)" → "Optimizer (Markowitz)"
- Any description line mentioning "PPO"/"stable-baselines3" → "scipy SLSQP mean-variance"
- Node IDs like `rl_optimizer` → `optimizer`

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "from ui.frontend import _MERMAID_HTML; assert 'PPO' not in _MERMAID_HTML and 'stable-baselines3' not in _MERMAID_HTML and 'rl_optimizer' not in _MERMAID_HTML; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add ui/frontend.py
git commit -m "docs(ui): mermaid diagram reflects Markowitz optimizer"
```

---

## Task 12: Drop unused deps from `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:36-38`

- [ ] **Step 1: Remove RL dependencies**

Edit `pyproject.toml` and delete the three lines:

```toml
    "stable-baselines3>=2.3.0",
    "gymnasium>=0.29.0",
    "shimmy>=1.0.0",
```

Also remove the `# RL` comment header above them.

- [ ] **Step 2: Regenerate lock**

```bash
uv lock
```

- [ ] **Step 3: Smoke install**

```bash
uv sync
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): drop stable-baselines3, gymnasium, shimmy"
```

---

## Task 13: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `_secrets/_Instructions.md`
- Modify: `_secrets/_Plan.md`
- Modify: `_secrets/_HowItWorks.md`

- [ ] **Step 1: Update `README.md`**

Replace all references to RL / PPO / stable-baselines3 with Markowitz / scipy / cash-reserve. Add a "Runtime banner" note (explains the yellow bar). Make sure the tab list reflects: Portfolio (watchlist), Optimizer, Dashboard.

- [ ] **Step 2: Update `_secrets/_Instructions.md`**

Append a top-of-file note:

```markdown
> ⚠️ On upgrade from an older version: delete `data/portfolio.db` before launching.
> The holdings schema no longer has `shares` / `purchase_price` and the app will
> otherwise fail to start.
```

Remove any setup steps mentioning CUDA/GPU install requirements specific to stable-baselines3.

- [ ] **Step 3: Update `_secrets/_Plan.md`**

Rewrite the architecture section: watchlist model, Markowitz optimizer, CASH injection, runtime banner, paired risk-free-rate input, device detection single source of truth.

- [ ] **Step 4: Update `_secrets/_HowItWorks.md`**

Rewrite the "Optimization math" section:

- Markowitz mean-variance
- SLSQP solver (scipy)
- Target volatility constraint wᵀΣw ≤ σ²
- Risk-free CASH asset with user-supplied rf (default 4%, editable via slider or `"4.56%"` textbox)
- Efficient frontier as `np.linspace(min_vol, max_vol, frontier_samples)` grid sweep
- Sharpe = (expected_return − rf) / expected_vol using the *same* rf the user entered

Remove PPO / gymnasium / stable-baselines3 prose.

- [ ] **Step 5: Commit**

```bash
git add README.md _secrets/_Instructions.md _secrets/_Plan.md _secrets/_HowItWorks.md
git commit -m "docs: watchlist + Markowitz optimizer across all docs"
```

---

## Task 14: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end: fresh DB, add tickers, run optimizer, read allocation back."""
import pytest
from unittest.mock import patch


def test_end_to_end_optimizer_persists(tmp_path, monkeypatch):
    # Point DB to a temp file
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Re-import config & database with the new env var
    import importlib
    from core import config, database, models
    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(models)

    database.init_db()

    with database.SessionLocal() as s:
        p = models.PortfolioDB(name="test")
        s.add(p); s.commit(); pid = p.id
        for t in ("AAPL", "MSFT", "GOOGL"):
            s.add(models.HoldingDB(portfolio_id=pid, ticker=t))
        s.commit()

    # Stub out yfinance calls
    import numpy as np, pandas as pd
    from services import optimizer as opt_mod
    idx = pd.bdate_range("2023-01-01", periods=300)
    def fake_hist(t, period="2y"):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        rets = rng.normal(0.0005, 0.012, size=300)
        return pd.DataFrame({"Close": 100*np.exp(np.cumsum(rets))}, index=idx)
    def fake_info(t):
        return {"price": 100.0}
    monkeypatch.setattr(opt_mod, "get_historical", fake_hist)
    monkeypatch.setattr(opt_mod, "get_stock_info",  fake_info)

    result = opt_mod.optimize_portfolio(
        ["AAPL", "MSFT", "GOOGL"], budget=50_000, target_vol=0.15,
        frontier_samples=200, risk_free_rate=0.045,
    )
    opt_mod.save_allocation(pid, result, budget=50_000,
                            target_vol=0.15, lookback="2y")

    with database.SessionLocal() as s:
        row = s.get(models.PortfolioAllocationDB, pid)
        assert row is not None
        assert row.budget == 50_000
        assert row.risk_free_rate == pytest.approx(0.045)
        assert row.allocations_json.startswith("{")
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(integration): end-to-end optimizer + persistence"
```

---

## Self-Review Checklist

Before handing off execution, re-verify:

- [ ] Every spec section (§3.1–§10) maps to a task.
- [ ] §5.0 (runtime banner) → Task 7.
- [ ] §6 paired rf control → Task 2 (parser) + Task 9 (UI binding).
- [ ] `optimizer_result` (not `rl_result`) used everywhere in Task 6.
- [ ] `optimizer` (not `rl_optimizer`) in graph, state, supervisor, chatbot, agent file, optimizer tab label.
- [ ] `build_plots` is a 3-tuple (pie, bar, frontier) in Task 4 and Task 9.
- [ ] No task references `update_holding`, `shares`, `purchase_price`, `PPO`, or `stable-baselines3` (except removal steps).
- [ ] CSS classes `.runtime-banner` and `.watchlist-df` both defined in Task 7 CSS block.
- [ ] DB migration documented in user-facing instructions (Task 13).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-17-watchlist-optimizer.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
