# AI Portfolio Manager — Complete Project Plan

> Single-user (multi-portfolio) AI-driven portfolio manager.
> Local: Gradio + FastAPI + Ollama. Cloud: HF Space + Groq.
> MLOps via GitHub Actions. Package manager: **uv**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Gradio UI  (port 7860)                                     │
│  ┌──────────┬──────────┬──────────┬────────────────────┐   │
│  │Dashboard │Portfolio │RL Optim. │  AI Assistant      │   │
│  └──────────┴──────────┴──────────┴────────────────────┘   │
│  Portfolio Selector Bar (create / rename / delete)          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  LangGraph Graph      │
         │  Supervisor           │
         │  ├─ Market Intel      │  yfinance
         │  ├─ Portfolio Analyst │  SQLite (per portfolio)
         │  ├─ Risk Manager      │  numpy/scipy
         │  ├─ RL Optimizer      │  stable-baselines3 PPO
         │  ├─ Knowledge Base    │  ChromaDB + Wikipedia
         │  └─ Synthesizer       │  Groq / Ollama LLM
         └───────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  FastAPI  (port 8000) │  /portfolios, /portfolio, /stocks
         └───────────────────────┘
                     │
              SQLite  +  ChromaDB  (data/ folder)
```

---

## File Structure

```
portfolio-app/
├── main.py                    ← entry point; kills ports, starts API + Gradio
├── pyproject.toml             ← uv project config + all dependencies
├── requirements.txt           ← kept for HF Space pip fallback
├── README.md                  ← HF Space frontmatter (app_file: main.py)
│
├── core/
│   ├── config.py              ← env vars, paths, ports
│   ├── database.py            ← SQLAlchemy engine, migration, session
│   └── models.py              ← Portfolio + HoldingDB ORM; Pydantic schemas
│
├── agents/
│   ├── state.py               ← PortfolioAgentState TypedDict + empty_state()
│   ├── graph.py               ← LangGraph StateGraph assembly
│   ├── supervisor.py          ← LLM routing + keyword fallback
│   ├── market_intel.py        ← yfinance live data
│   ├── portfolio_analyst.py   ← P&L, allocation (portfolio-scoped)
│   ├── risk_manager.py        ← Sharpe, VaR, drawdown, Sortino
│   ├── rl_optimizer_agent.py  ← PPO auto-trigger (portfolio-scoped)
│   ├── knowledge_base_agent.py← RAG via ChromaDB
│   └── synthesizer.py         ← merges agent outputs → final response
│
├── services/
│   ├── stock_service.py       ← yfinance wrappers
│   ├── llm_service.py         ← Groq / Ollama LLM factory
│   ├── knowledge_base.py      ← ChromaDB population + query
│   └── rl_optimizer.py        ← PPO env, training, build_plots()
│
├── ui/
│   ├── theme.py               ← Gradio dark theme + CUSTOM_CSS
│   ├── gradio_interface.py    ← Blocks layout; portfolio selector; all handlers
│   └── components/
│       ├── chatbot.py         ← run_agents(), tts_speak(), agent_badges_html()
│       ├── dashboard.py       ← plotly chart builders
│       └── optimizer_ui.py    ← standalone RL optimizer UI runner
│
├── api/
│   ├── server.py              ← FastAPI app factory
│   └── routes/
│       ├── portfolio.py       ← /portfolios CRUD + /portfolio holdings (per portfolio_id)
│       ├── stocks.py          ← /stocks price lookup
│       ├── optimizer.py       ← /optimize REST endpoint
│       └── chat.py            ← /chat REST endpoint
│
├── mcp/
│   └── server.py              ← FastMCP stdio server for Claude Desktop
│
├── scripts/
│   ├── populate_kb.py         ← Wikipedia → ChromaDB (run once)
│   ├── check_drift.py         ← Sharpe/Return/KS drift detection
│   └── retrain.py             ← PPO retrain + baseline update
│
├── monitoring/
│   ├── metrics_tracker.py     ← record baseline metrics
│   └── drift_detector.py      ← compare current vs baseline
│
├── tests/                     ← pytest test suite
│
├── .github/workflows/
│   ├── ci_cd.yml              ← Lint → Test → Deploy to HF Space
│   └── scheduled_monitoring.yml ← Weekly drift check + retrain
│
└── _secrets/                  ← git-excluded
    ├── .env                   ← GROQ_API_KEY, HF_TOKEN, ports, thresholds
    ├── Instructions.md        ← this file's sister — local setup guide
    ├── HowItWorks.md          ← technical deep-dive
    └── plan.md                ← this file
```

---

## Step-by-Step Build Plan

### Phase 1 — Local Setup & Skeleton

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create project: `mkdir portfolio-app && cd portfolio-app && uv init`
3. Copy `pyproject.toml` dependencies
4. Run `uv sync` → creates `.venv` automatically
5. Copy `.env` from `_secrets/.env`, fill `GROQ_API_KEY`
6. Verify: `uv run python -c "import gradio, langgraph; print('OK')`

### Phase 2 — Core / Database

7. `core/config.py` — load env vars, define paths + ports
8. `core/models.py` — `Portfolio` ORM + `HoldingDB` (portfolio_id FK) + Pydantic schemas
9. `core/database.py` — SQLAlchemy engine, `_maybe_migrate()`, `init_db()`,
   `_ensure_default_portfolio()`
10. Verify: `uv run python -c "from core.database import init_db; init_db(); print('DB OK')"`

### Phase 3 — Services

11. `services/stock_service.py` — yfinance wrappers (prices, info, historical)
12. `services/llm_service.py` — Groq/Ollama LLM factory with fallback logic
13. `services/knowledge_base.py` — ChromaDB client, `populate()`, `query()`
14. `services/rl_optimizer.py` — gymnasium Env, PPO training, `build_plots()`

### Phase 4 — LangGraph Agents

15. `agents/state.py` — `PortfolioAgentState` TypedDict with `active_portfolio_id`
16. `agents/supervisor.py` — LLM routing + keyword fallback map
17. `agents/market_intel.py` — fetch live prices for tickers in message
18. `agents/portfolio_analyst.py` — P&L for `active_portfolio_id`
19. `agents/risk_manager.py` — Sharpe/VaR/drawdown from portfolio_data
20. `agents/rl_optimizer_agent.py` — extract budget + tickers (portfolio-scoped), run PPO
21. `agents/knowledge_base_agent.py` — RAG query, strict "I don't know" fallback
22. `agents/synthesizer.py` — merge all outputs into final_response
23. `agents/graph.py` — assemble `StateGraph`, compile, `get_graph()` with lru_cache

### Phase 5 — FastAPI Backend

24. `api/server.py` — FastAPI app factory, include all routers
25. `api/routes/portfolio.py` — `/portfolios` CRUD + `/portfolio` holdings (portfolio_id param)
26. `api/routes/stocks.py` — price lookup endpoints
27. `api/routes/optimizer.py` — synchronous optimize endpoint
28. `api/routes/chat.py` — chat endpoint calling the agent graph
29. Verify: `uv run python main.py` → visit http://localhost:8000/api/docs

### Phase 6 — Gradio UI

30. `ui/theme.py` — dark Gradio theme + CUSTOM_CSS (portfolio-bar, metric-card, agent-badge…)
31. `ui/components/dashboard.py` — plotly chart builders (pie, bar, performance, sector)
32. `ui/components/optimizer_ui.py` — `run_and_render()` for standalone RL tab
33. `ui/components/chatbot.py` — `run_agents(msg, history, portfolio_id)`, TTS, badges
34. `ui/gradio_interface.py`:
    - Portfolio Selector Bar (dropdown, create, rename, delete)
    - Tab 1: Dashboard (metric cards + 4 charts + holdings table)
    - Tab 2: Portfolio (add/update/remove holdings, scoped to active portfolio)
    - Tab 3: RL Optimizer (auto-populate tickers, run PPO, show 4 charts)
    - Tab 4: AI Assistant (LangGraph chat, Mermaid diagram toggle, TTS)
    - Wire portfolio switch → refresh all tabs

### Phase 7 — MCP Server

35. `mcp/server.py` — FastMCP stdio tools: `get_portfolio`, `add_holding`,
    `get_prices`, `optimize`, `ask_kb`

### Phase 8 — MLOps Scripts

36. `scripts/populate_kb.py` — Wikipedia → ChromaDB (idempotent)
37. `scripts/check_drift.py` — Sharpe + Return + KS drift, exit 0/1
38. `scripts/retrain.py` — PPO retrain, update baseline, save model

### Phase 9 — CI/CD

39. `.github/workflows/ci_cd.yml`:
    - `astral-sh/setup-uv@v3` → `uv sync --group dev`
    - `uv run ruff check .` → `uv run pytest tests/`
    - Deploy to HF Space via rsync + git push

40. `.github/workflows/scheduled_monitoring.yml`:
    - Every Monday 08:00 UTC: `uv run python scripts/check_drift.py`
    - If drift: `uv run python scripts/retrain.py`
    - Create GitHub Issue if drift persists

### Phase 10 — Deploy

41. Create GitHub repo: `github.com/new` → name: **stocksPortfolio** (public, no README)
42. Push:
    ```bash
    git init && git branch -M main
    git remote add origin https://github.com/irajkooh/stocksPortfolio.git
    git add -A && git commit -m "Initial portfolio manager"
    git push -u origin main
    ```
43. Add GitHub secrets at `github.com/irajkooh/stocksPortfolio/settings/secrets/actions`:
    `GROQ_API_KEY`, `HF_TOKEN`, `HF_USERNAME=irajkoohi`
44. Add `GROQ_API_KEY` secret to HF Space settings
45. Verify HF Space at `huggingface.co/spaces/irajkoohi/stocksPortfolio`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `uv` as package manager | Fast, lock-file-based, native `pyproject.toml` support |
| Multiple named portfolios | Flexible for different strategies per user |
| Portfolio ID in agent state | Agents always query the correct portfolio |
| Port kill on startup | Prevents "address already in use" errors in dev |
| `fuser -k port/tcp` | Works on Linux/HF Space; netstat+taskkill on Windows |
| `_maybe_migrate()` in database.py | Handles schema upgrade without breaking existing data |
| `app_file: main.py` in README | Avoids HF Space's `app.py` requirement |
| Groq on HF Space, Ollama local | Free unlimited cloud LLM; fast local fallback |
| ChromaDB persistent | Vector store survives restarts; Wikipedia populated once |
| `lru_cache` on `get_graph()` | LangGraph compiled once per process |

---

## Steps to Continue (from any point)

### If you just cloned / unzipped the project
```bash
cd portfolio-app
uv sync                          # install all dependencies
cp _secrets/.env .env            # fill GROQ_API_KEY
uv run python main.py            # start the app
```

### If you want to add a new agent
1. Create `agents/new_agent.py` with a `new_agent_node(state)` function
2. Add it to `agents/graph.py` → `g.add_node("new_agent", new_agent_node)`
3. Add an edge in the graph sequence
4. Add routing keywords in `agents/supervisor.py` keyword map
5. Update `agents/synthesizer.py` to consume the new agent's output
6. Add badge label in `ui/components/chatbot.py` `_AGENT_LABELS`

### If you want to add a new portfolio field (e.g., target allocation)
1. Add column to `Portfolio` in `core/models.py`
2. Add a Pydantic schema field
3. Update `core/database.py` migration logic
4. Add UI control in `ui/gradio_interface.py`

### If you want to add a new API endpoint
1. Add route in the appropriate `api/routes/*.py` file
2. Include the router in `api/server.py` if it's a new file

### If you want to improve RL
1. Edit `services/rl_optimizer.py` — modify `PortfolioEnv.step()` reward function
2. Increase `timesteps` in the Slider (RL Optimizer tab) or default in retrain.py
3. Push to GitHub → CI/CD triggers scheduled retrain on next drift check

### If you need to reset the database
```bash
rm data/portfolio.db
uv run python main.py   # init_db() recreates it with Default portfolio
```

### If you want to add tests
```bash
# tests/ folder, use pytest
uv run pytest tests/ -v
uv add --group dev pytest-mock   # add test dependencies as needed
```

### Useful commands
```bash
uv run python main.py                     # start app
uv run python scripts/populate_kb.py     # rebuild knowledge base
uv run python scripts/check_drift.py --tickers AAPL MSFT GOOGL
uv run python scripts/retrain.py --tickers AAPL MSFT GOOGL --budget 100000
uv run ruff check . --fix                 # lint and auto-fix
uv run pytest tests/ -v                  # run all tests
```
