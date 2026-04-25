"""
Populate ChromaDB knowledge base from free Wikipedia articles.
Run once at startup (skipped automatically if already populated).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Topics covering: portfolio theory, RL/optimisation, stock market, investing
TOPICS = [
    # Portfolio / risk theory
    "Modern portfolio theory",
    "Efficient frontier",
    "Capital asset pricing model",
    "Sharpe ratio",
    "Treynor ratio",
    "Jensen's alpha",
    "Alpha (finance)",
    "Beta (finance)",
    "Portfolio diversification",
    "Risk-adjusted return on capital",
    "Value at risk",
    "Expected shortfall",
    "Markowitz model",
    # Optimisation / RL
    "Reinforcement learning",
    "Q-learning",
    "Proximal policy optimization",
    "Portfolio optimization",
    "Mean-variance analysis",
    "Kelly criterion",
    "Stochastic control",
    # Stock market fundamentals
    "Stock market",
    "Stock valuation",
    "Price–earnings ratio",
    "Dividend yield",
    "Market capitalization",
    "Earnings per share",
    "Bull market",
    "Bear market",
    "Market volatility",
    "Stock market index",
    # Investment strategies
    "Investment strategy",
    "Dollar cost averaging",
    "Index fund",
    "Exchange-traded fund",
    "Asset allocation",
    "Rebalancing investments",
    "Long-term investing",
    "Passive management",
    "Active management",
    # Fixed income / macro
    "Bond (finance)",
    "Yield curve",
    "Inflation",
    "Interest rate",
]


def populate(force: bool = False) -> None:
    from services.knowledge_base import add_documents, kb_size

    if not force and kb_size() > 0:
        logger.info("Knowledge base already has %d chunks — skipping.", kb_size())
        return

    try:
        import wikipedia
        wikipedia.set_lang("en")
    except ImportError:
        logger.error("wikipedia package not installed — skipping KB population.")
        return

    texts, metas = [], []
    for topic in TOPICS:
        try:
            page = wikipedia.page(topic, auto_suggest=False, preload=False)
            # Use first 4 000 chars to stay within chunk budget
            texts.append(page.content[:4_000])
            metas.append({"source": "wikipedia", "topic": topic, "url": page.url})
            logger.info("✓ %s", topic)
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                texts.append(page.content[:4_000])
                metas.append({"source": "wikipedia", "topic": topic})
                logger.info("✓ %s (via disambiguation → %s)", topic, e.options[0])
            except Exception:
                logger.warning("✗ Skipped (disambiguation): %s", topic)
        except Exception as exc:
            logger.warning("✗ Skipped %s: %s", topic, exc)

    if texts:
        add_documents(texts, metas)
        logger.info("Knowledge base populated: %d articles → %d chunks total.",
                    len(texts), kb_size())
    else:
        logger.warning("No articles were fetched — knowledge base is empty.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    populate(force="--force" in sys.argv)
