# Budget-Optimized Watchlist — Design Spec

**Date:** 2026-04-17
**Status:** Approved for implementation planning
**Scope:** Portfolio tab, Optimizer tab, Dashboard tab, agent graph, data model, docs

---

## 1. Motivation

Today the Portfolio tab requires the user to enter **shares** and **purchase_price** for each ticker, which only makes sense for existing holdings and ties every computation to a cost basis. The user's actual need is a decision-support tool: *"given these tickers and this budget, tell me how to allocate — and how much to leave in cash — to maximize return at a chosen risk level."*

This spec reshapes the product around that workflow. The Portfolio tab becomes a symbol-only watchlist; the Optimizer becomes the center of gravity; the Dashboard reports the last optimized plan alongside live market data for the watchlist.

## 2. User-facing flow

1. User creates / selects a portfolio.
2. On the **Portfolio** tab, user adds tickers (symbol only — no shares, no price).
3. On the **Optimizer** tab, user sets budget, target annual volatility, and lookback window; clicks Optimize.
4. Optimizer returns per-ticker dollar allocations, share counts (derived from live yfinance price), a cash reserve, and metrics (expected return, expected volatility, Sharpe). The full result persists to the database (one row per portfolio, overwritten each run).
5. **Dashboard** shows: live watchlist prices and returns (top) + the persisted last-optimized plan (bottom).
6. **Chat** can also trigger optimization if the user supplies budget and target risk in the message. Results persist the same way.

## 3. Data model changes

### 3.1 `holdings` table — drop cost-basis columns
- Drop: `shares`, `purchase_price`.
- Keep: `id`, `portfolio_id`, `ticker`, `created_at`, `updated_at`, `UniqueConstraint(portfolio_id, ticker)`.
- Table name stays `holdings` (avoids a table rename migration). Conceptually it is now a **watchlist entry** table.
- **CASH is not stored in `holdings`.** It is injected at optimize time and rendered in UI tables as a synthetic, non-removable row. This keeps the table clean (cash is not a market security) while guaranteeing cash always appears alongside user tickers in every view.

### 3.2 `portfolio_allocations` table — new
Single row per portfolio (`portfolio_id` is primary key). Overwritten on each optimizer run.

| column | type | notes |
|---|---|---|
| `portfolio_id` | Integer PK, FK → portfolios.id, ON DELETE CASCADE | |
| `budget` | Float | USD |
| `target_vol` | Float | annualized, decimal (e.g., 0.15 = 15%) |
| `lookback` | String | one of `"1y"`, `"2y"`, `"3y"`, `"5y"` |
| `expected_return` | Float | annualized, decimal |
| `expected_vol` | Float | annualized, decimal |
| `sharpe` | Float | (μ − rf) / σ, rf from user input |
| `risk_free_rate` | Float | annualized, decimal (e.g., 0.04 = 4%) |
| `cash_dollars` | Float | USD held as cash |
| `allocations_json` | Text (JSON) | `{ticker: {weight, dollars, shares, price}}` |
| `commentary` | Text | LLM-generated interpretation |
| `created_at` | DateTime | |

### 3.3 Pydantic models
- `HoldingCreate` → `{ticker: str}` (validator: strip + upper).
- `HoldingUpdate` removed (nothing to update on a watchlist entry).
- `HoldingOut` → `{id, portfolio_id, ticker, created_at}`.
- New `AllocationOut` mirroring `portfolio_allocations` row.

### 3.4 Migration
Fresh SQLite schema drop + recreate is acceptable (project is pre-production; existing data in `data/portfolio.db` is demo-only). Document in `_secrets/_Instructions.md` that users must delete `portfolio.db` on upgrade.

## 4. Optimizer engine

### 4.1 New module: `services/optimizer.py`
Replaces `services/rl_optimizer.py`. Old file is deleted.

### 4.2 Algorithm — Markowitz mean-variance
- Fetch daily close prices via `get_historical(ticker, period=<lookback>)`.
- Compute daily return series, mean vector μ, covariance Σ. Annualize: μ × 252, Σ × 252.
- Augment with a risk-free "cash" asset: μ_cash = `risk_free_rate` (user-supplied, default 0.04), σ_cash = 0, cov(cash, any) = 0.
- Solve via `scipy.optimize.minimize` (SLSQP):

  ```
  maximize   wᵀμ  (equivalently minimize −wᵀμ)
  subject to wᵀΣw ≤ target_vol²
             Σw = 1
             0 ≤ wᵢ ≤ 0.40   for each risky ticker
             0 ≤ w_cash ≤ 1
  ```

- If infeasible (e.g., target_vol below the min-variance portfolio's vol), fall back to the minimum-variance portfolio and surface a warning in `commentary`.

### 4.3 Public API
```python
def optimize_portfolio(
    tickers: list[str],
    budget: float,
    target_vol: float,          # decimal, e.g. 0.15
    lookback: str = "2y",       # "1y" | "2y" | "3y" | "5y"
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
    frontier_samples: int = 5_000,   # grid resolution for efficient-frontier chart
) -> dict: ...
```

Returns:
```python
{
  "allocations": {ticker: {"weight": float, "dollars": float,
                           "shares": float, "price": float}},
  "cash_dollars": float,
  "metrics": {"expected_return": float, "expected_vol": float,
              "sharpe": float, "target_vol": float, "risk_free_rate": float},
  "frontier_points": [{"vol": float, "return": float}],  # for the chart
  "returns_df": pd.DataFrame,
  "warnings": list[str],        # e.g. ["target vol infeasible, used min-var"]
}
```

### 4.4 Constraints / guardrails
- **Min 3 tickers** (40% cap requires ≥ 3 to sum to 100%). Note: CASH is injected as an always-available asset, so the user must still add ≥ 3 *risky* tickers.
- **Min history**: require ≥ 60 trading days after dropping NaNs.
- **No shorts**: enforced by wᵢ ≥ 0.
- Typical solve time: < 500 ms for ≤ 20 tickers.

### 4.5 Efficient-frontier sample count (the "iterations" knob)
Markowitz itself is a closed-form solve and has no training loop, but the **efficient frontier chart** is rendered by sweeping `target_vol` across a grid and re-solving at each grid point. The grid resolution is user-tunable via the Optimizer tab's **Frontier samples** slider (see §6).

- Parameter: `frontier_samples: int` (added to `optimize_portfolio` signature).
- Range: 2,000 – 999,999 (per user request for "maximum training").
- Default: 5,000.
- Implementation: `np.linspace(min_vol, max_vol, frontier_samples)`, solve SLSQP at each point.
- Only affects frontier chart smoothness; the chosen-point solve is a single SLSQP call and runs in < 500 ms regardless.
- At the high end of the range (999,999 samples), frontier generation is deliberately expensive (~30–60 s); the UI must show a progress indicator and run the solve in a background thread so Gradio stays responsive.

### 4.6 Plots — `build_plots(result) → (fig_alloc_pie, fig_dollar_bar, fig_frontier)`
- **Pie**: ticker weights + cash slice (cash colored neutral gray).
- **Bar**: dollar amount per ticker, cash as its own bar.
- **Efficient frontier**: sweep `target_vol` from min-var to max-return, plot the frontier curve; mark the chosen point.

Cumulative-return vs. equal-weight backtest (in the old `build_plots`) is removed — it conflated in-sample fit with performance and was misleading.

## 5.0 UI — Global runtime banner

A single-line yellow banner rendered at the very top of `ui/frontend.py` (above the existing header/tabs), persistent across all tabs.

**Format** (exact string, left-aligned):

```text
Running: <env> | Device: <device> | LLM: <model>
```

**Field resolution:**

- `<env>` — `"HF Space: {SPACE_ID}"` if `core/config.IS_HF_SPACE` is true (reads the `SPACE_ID` env var), otherwise `"locally"`.
- `<device>` — resolved by a new helper `core/runtime.py::detect_device() -> str`:
  - `"cuda"` if `torch.cuda.is_available()`
  - else `"mps"` if `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()`
  - else `"cpu"`
  - Helper also exports `DEVICE` (module-level string, computed once at import time) so other modules (embeddings, future model loaders) can import a single source of truth.
- `<model>` — `core/config.GROQ_MODEL` when `GROQ_API_KEY` is set, else `core/config.OLLAMA_MODEL` if present, else `"<no LLM key>"`.

**Styling** — CSS class `.runtime-banner` in `ui/theme.py`: `color: #FFD700` (amber/yellow, already in palette), background `#0d1118`, border `1px solid #2a2a1a`, font-family monospace, padding `6px 14px`, border-radius `8px`, margin-bottom `10px`, font-size `.82rem`.

**Device propagation**: `core/runtime.DEVICE` is consumed by `HuggingFaceEmbeddings(model_kwargs={"device": DEVICE})` in whichever module currently instantiates the embedder (likely `agents/knowledge_base_agent.py` or a rag helper). Default `"cpu"` behavior unchanged.

## 5. UI — Portfolio tab

Simplified to watchlist management:

- **Add ticker**: one `gr.Textbox(label="Ticker")` + Add button. Validation: non-empty, uppercase, basic regex `^[A-Z0-9.\-]{1,10}$`. The literal string `CASH` is rejected (reserved).
- **Remove ticker**: `gr.Dropdown(label="Select to remove")` populated from current watchlist + Remove button. `CASH` is excluded from this dropdown (not removable).
- **Watchlist table** (`gr.Dataframe`): columns = Ticker, Price, Daily Δ%. Price and Δ% come from `get_stock_info`.
  - **CASH row**: always rendered as the first row with Price = `$1.00`, Daily Δ% = `0.00%`. Not editable, not removable, appears regardless of whether the user has added any tickers.
  - **Cell foreground color**: `#000` (black) on all data cells, so prices are readable against the light dataframe background. Applied via Gradio's `elem_classes` + a CSS rule in `ui/theme.py` (e.g. `.watchlist-df td { color: #000; }`).
- No Update section, no Shares column, no Buy Price column.

## 6. UI — Optimizer tab

Rename from "🤖 RL Optimizer" to "🤖 Optimizer".

**Inputs panel**:
- Budget `gr.Number` — default 100,000, min 1,000, step 1,000.
- Target Risk `gr.Slider` — range 5–40 (shown as %), default 15, step 0.5; displayed value e.g. "15.0%".
- Risk-free Rate — **paired control**: a `gr.Slider` (range 0–20, shown as %, default 4.00, step 0.25) bound two-way with a `gr.Textbox` (default `"4.00%"`). Either control can set the value:
  - **Slider → Textbox**: on slider `.change`, write the formatted string (e.g. `"4.25%"`) into the textbox.
  - **Textbox → Slider**: on textbox `.submit`, parse via helper `parse_rf(text: str) -> float` which accepts `"4.56%"`, `"4.56"`, `"0.0456"`, whitespace-tolerant; returns a decimal in `[0.0, 0.20]`, clamped, and updates the slider. Invalid input leaves the prior value and flashes a Gradio warning.
  - Both controls feed into the optimizer as a `decimal` (e.g. `0.0456`).
  - Label on the pair: "Risk-free rate (annual)". The textbox is the source of truth for "precise" values like `4.56%`; the slider is for quick drag.
  - Used both as cash-asset return and as the Sharpe denominator's rf.
  - `parse_rf` lives in `services/optimizer.py` (or a small `core/parsing.py` if that grows) and is unit-tested.
- Lookback `gr.Dropdown` — options `["1y", "2y", "3y", "5y"]`, default `"2y"`.
- Frontier samples `gr.Slider` — range 2,000–999,999, default 5,000, step 1,000. Label: "Frontier samples (higher = smoother frontier chart, slower solve)". Wired to `optimize_portfolio(..., frontier_samples=...)`. Tooltip warns that values > 100k take > 10 s.
- "Optimize" button.

**Outputs panel**:
- Metrics row: Expected Return, Expected Vol, Sharpe Ratio, Cash Reserve ($).
- Allocation pie chart.
- Dollar allocation bar chart.
- Efficient frontier chart.
- Commentary (`gr.Markdown`): LLM-generated interpretation citing chosen ticker tilts, why cash was or wasn't held, Sharpe vs. target-vol tradeoff. Also surfaces any `warnings` from the engine.

## 7. UI — Dashboard tab (hybrid)

- **Top half — Live Watchlist** (`gr.Dataframe`): Ticker, Price, 1d %, 1mo %, 3mo %, 1y %.
  - CASH row pinned to the top: Price `$1.00`, all return columns `0.00%`.
  - Cell foreground color: `#000` (black), via the same `.watchlist-df td { color: #000; }` CSS rule.
- **Bottom half — Last Optimized Plan**:
  - Metrics cards: Budget, Expected Return, Expected Vol, Sharpe, Cash.
  - Allocation pie (mirrors Optimizer tab).
  - Dollar allocation table: Ticker, Weight %, Dollars, Shares, Price. CASH row always included (Shares column shown as em-dash `—`, Price `$1.00`). Same black-text CSS applied.
  - Timestamp: "Last optimized: <created_at>".
  - Empty state (no row in `portfolio_allocations` for this portfolio): single line — "Run the Optimizer to see your plan."

## 8. Agent graph integration

- **Rename graph node** `rl_optimizer` → `optimizer`. Update:
  - `agents/graph.py` (node registration + edges).
  - `agents/` node implementation file (rename + internals swap).
  - `agents/supervisor` prompt so it routes to the new name.
  - `ui/components/chatbot.py` — `_AGENT_LABELS`: `"optimizer": "🤖 Optimizer"`.
- **Inputs from chat**: optimizer agent extracts budget and target_vol from the user message (regex/LLM extraction). If missing, returns a clarifying response ("What budget and risk level?") rather than guessing.
- **Persistence**: both the chat path and the button path call the same `optimize_portfolio` + `save_allocation(portfolio_id, result)` functions. Dashboard always reflects the latest run.

## 9. Documentation updates

All four files below get updated in the same PR:
- `README.md` — overview, screenshots / descriptions reflect watchlist + Markowitz.
- `_secrets/_Instructions.md` — setup flow; add note about deleting `portfolio.db` on upgrade.
- `_secrets/_Plan.md` — architecture snapshot.
- `_secrets/_HowItWorks.md` — optimization math section rewritten (Markowitz, target-vol, cash reserve, user-supplied risk-free rate with 4% default).

Remove all references to RL / PPO / gymnasium / stable-baselines3 from docs. (stable-baselines3 and gymnasium can come out of `pyproject.toml` as well — they're no longer used.)

## 10. Out of scope (explicit non-goals)

- Transaction costs / slippage modeling.
- Multi-period / rebalancing schedule.
- Short selling.
- Multi-currency (USD only).
- Run history (only latest persisted).
- User-configurable max-weight cap (hardcoded 40%).
- P&L tracking / cost-basis tracking (removed entirely).

## 11. Testing notes

- Unit: `optimize_portfolio` on a synthetic 3-asset fixture — verify weights sum to 1, cap respected, target_vol respected (or warning emitted), Sharpe sign matches direction of μ.
- Integration: end-to-end — add 3 tickers, run optimizer, read back from `portfolio_allocations`, confirm Dashboard renders.
- Manual: UI smoke test — each tab renders; Optimizer button produces charts; chat "optimize my portfolio with $50k at 18% risk" routes correctly.
- Unit: `parse_rf` — verifies `"4.56%"` → `0.0456`, `"4.56"` → `0.0456`, `"0.0456"` → `0.0456`, `"   5 % "` → `0.05`, `"25%"` → clamped `0.20`, `"-1%"` → clamped `0.0`, `"abc"` → raises `ValueError`.
- Unit: `detect_device` — monkeypatch `torch.cuda.is_available` / `torch.backends.mps.is_available` to assert each branch returns `"cuda"` / `"mps"` / `"cpu"`.
- Manual: banner renders on app start; shows `locally` when `SPACE_ID` is unset, `HF Space: <id>` when set; device matches the running machine; LLM reflects `core/config.py`.
