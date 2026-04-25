---
title: StocksPortfolio
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
---

# AI Portfolio Manager

Markowitz-optimised portfolio management with RAG-powered assistant.

## Features
- Real-time stock prices via yfinance
- Watchlist builder — track any tickers, auto-includes CASH
- Markowitz mean-variance optimisation (scipy SLSQP) with budget allocation
- Dashboard: live watchlist prices + last optimised plan breakdown
- RAG chatbot backed by financial knowledge base (ChromaDB + Wikipedia)
- Text-to-speech toggle for chat answers
- FastAPI REST backend + FastMCP server

## HF Space Secrets Required
| Secret | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (free at console.groq.com) |

## Local Setup
```bash
cp .env.example .env   # add GROQ_API_KEY or leave for Ollama
pip install -r requirements.txt
python main.py
```
