---
title: StocksPortfolio
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: main.py
pinned: true
---

# AI Portfolio Manager

RL-optimised portfolio management with RAG-powered assistant.

## Features
- Real-time stock prices via yfinance
- PPO reinforcement learning weight optimisation with budget allocation
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
