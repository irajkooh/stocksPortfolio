import plotly.graph_objects as go
import pandas as pd
from services.stock_service import get_historical

COLORS = ["#00D4FF","#7B2FBE","#00FF94","#FF6B35","#FFD700",
          "#FF69B4","#4ECDC4","#A78BFA","#F87171","#34D399"]

_LAYOUT = dict(
    paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
    font=dict(color="white", size=12),
    margin=dict(t=50, b=40, l=50, r=20),
)


def allocation_pie(holdings: list[dict]) -> go.Figure:
    if not holdings:
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Allocation",
            annotations=[dict(text="No holdings yet", showarrow=False,
                              font=dict(color="#9CA3AF", size=14))],
            **_LAYOUT,
        )
        return fig

    labels = [h["ticker"] for h in holdings]
    values = [h["value"]  for h in holdings]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.48,
        marker=dict(colors=COLORS[:len(labels)]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Portfolio Allocation",
        legend=dict(font=dict(color="white")),
        **_LAYOUT,
    )
    return fig


def pnl_bar(holdings: list[dict]) -> go.Figure:
    if not holdings:
        fig = go.Figure()
        fig.update_layout(title="P&L by Stock", **_LAYOUT)
        return fig

    tickers = [h["ticker"] for h in holdings]
    pnls    = [h["pnl"]    for h in holdings]
    colors  = ["#00FF94" if p >= 0 else "#FF4757" for p in pnls]

    fig = go.Figure(go.Bar(
        x=tickers, y=pnls,
        marker_color=colors,
        text=[f"${p:+,.2f}" for p in pnls],
        textposition="outside", textfont=dict(color="white"),
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:+,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Unrealised P&L per Holding",
        yaxis=dict(tickprefix="$", zeroline=True, zerolinecolor="#374151"),
        **_LAYOUT,
    )
    return fig


def performance_chart(tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, ticker in enumerate(tickers[:8]):
        hist = get_historical(ticker, period="1y")
        if hist.empty:
            continue
        norm = hist["Close"] / hist["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=hist.index, y=norm, name=ticker,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{ticker}</b>  %{{x|%b %d}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title="1-Year Price Performance (Normalised to 100)",
        xaxis_title="", yaxis_title="Index (100 = start)",
        legend=dict(font=dict(color="white")),
        hovermode="x unified",
        **_LAYOUT,
    )
    return fig


def sector_bar(holdings: list[dict]) -> go.Figure:
    if not holdings:
        return go.Figure()
    sector_map: dict[str, float] = {}
    for h in holdings:
        s = h.get("sector", "Unknown")
        sector_map[s] = sector_map.get(s, 0) + h["value"]
    sectors = list(sector_map.keys())
    values  = list(sector_map.values())
    fig = go.Figure(go.Bar(
        x=sectors, y=values,
        marker_color=COLORS[:len(sectors)],
        text=[f"${v:,.0f}" for v in values],
        textposition="outside", textfont=dict(color="white"),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Exposure by Sector",
        yaxis=dict(tickprefix="$"),
        **_LAYOUT,
    )
    return fig
