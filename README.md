---
title: StocksPortfolio
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: true
---

# AI Portfolio Manager

Markowitz mean-variance portfolio optimisation with a RAG-powered AI assistant.

## Features

- **Live watchlist** — 8-column table per portfolio: Ticker · Price · 1d% · 1mo% · 3mo% · 1y% · Sharpe · Sortino; includes CASH row (RF-rate-scaled returns) and an optimised-weighted portfolio summary row; mobile-scrollable
- **Markowitz SLSQP optimiser** — budget allocation with target-vol constraint, always-on CASH anchor, efficient-frontier chart (Monte Carlo scatter + red dashed frontier curve), allocation pie, dollar bar chart; stores results per portfolio in SQLite
- **Dashboard** — metric cards (budget · expected return · volatility · Sharpe · Sortino · 95% VaR · cash); Portfolio vs S&P 500 split-line chart from the last optimisation date; last-plan allocation table and pie
- **Multi-portfolio** — create, rename, and delete named portfolios; each tab scopes to the selected portfolio
- **Holdings manager** — add / update / remove positions (ticker + shares + purchase price)
- **LangGraph multi-agent AI** — Supervisor → Market Intel · Portfolio Analyst · Risk Manager · Optimizer · Knowledge Base → Synthesizer; Groq LLM on HF Space, Ollama locally
- **RAG knowledge base** — ~45 Wikipedia finance articles in ChromaDB; strict "I don't know" fallback
- **Text-to-speech** — toggle per-message TTS for chat answers
- **FastAPI REST backend** — `/portfolios`, `/portfolio`, `/stocks`, `/optimize`, `/chat`
- **FastMCP server** — Claude Desktop integration via stdio MCP tools

## HF Space Secrets Required

| Secret | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (free at console.groq.com) |

## Local Setup

```bash
cp .env.example .env   # add GROQ_API_KEY or leave blank for Ollama fallback
uv sync
uv run python app.py
# Gradio UI : http://localhost:7860
# FastAPI   : http://localhost:8000/api/docs
```
