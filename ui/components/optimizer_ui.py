import plotly.graph_objects as go
from services.rl_optimizer import optimize_portfolio, build_plots

_EMPTY = go.Figure()
_EMPTY.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                     font=dict(color="white"))


def run_and_render(
    tickers_raw: str,
    budget:      float,
    period:      str,
    timesteps:   int,
) -> tuple[str, go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Called by Gradio button. Returns (summary_md, fig_weights, fig_perf,
    fig_frontier, fig_budget).
    """
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if len(tickers) < 2:
        return "**Please enter at least 2 ticker symbols.**", _EMPTY, _EMPTY, _EMPTY, _EMPTY

    result = optimize_portfolio(tickers, budget=budget, period=period,
                                timesteps=int(timesteps))
    if "error" in result:
        return f"**Error:** {result['error']}", _EMPTY, _EMPTY, _EMPTY, _EMPTY

    fig_w, fig_p, fig_f, fig_b = build_plots(result)
    summary = _format_summary(result)
    return summary, fig_w, fig_p, fig_f, fig_b


def _format_summary(r: dict) -> str:
    tickers = r["tickers"]
    allocs  = r["allocations"]
    m       = r["metrics"]
    budget  = r["budget"]

    lines = [
        f"## RL Optimisation Results  —  Budget: **${budget:,.0f}**\n",
        "### Optimal Weights & Dollar Allocations\n",
        "| Ticker | Weight | $ Allocation | Shares to Buy | Price |",
        "|--------|--------|-------------|---------------|-------|",
    ]
    for t in sorted(tickers, key=lambda x: -allocs[x]["weight"]):
        a = allocs[t]
        bar = "█" * max(1, int(a["weight"] * 24))
        lines.append(
            f"| **{t}** | {a['weight']*100:.1f}% {bar} "
            f"| ${a['dollars']:,.2f} | {a['shares']:.4f} | ${a['price']:.2f} |"
        )

    imp = m["rl_sharpe"] - m["eq_sharpe"]
    sign = "+" if imp >= 0 else ""
    lines += [
        "\n### Performance Metrics\n",
        f"| Metric | RL Optimised | Equal Weight |",
        f"|--------|-------------|--------------|",
        f"| Sharpe Ratio    | **{m['rl_sharpe']:.3f}** | {m['eq_sharpe']:.3f} |",
        f"| Annual Return   | **{m['rl_annual_return']:.2f}%** | {m['eq_annual_return']:.2f}% |",
        f"| Annual Volatility | {m['rl_annual_vol']:.2f}% | — |",
        "\n### Recommendation\n",
        (f"RL optimisation **improves Sharpe by {sign}{imp:.3f}** vs equal weighting. "
         f"Following these weights is expected to yield better risk-adjusted returns."
         if imp > 0 else
         f"Equal weighting is competitive (Δ Sharpe = {imp:.3f}). "
         f"Consider adding more uncorrelated assets to improve diversification."),
    ]
    return "\n".join(lines)
