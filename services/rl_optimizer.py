"""
Portfolio RL Optimizer
======================
State   : [rolling_returns (lookback×N), rolling_volatility (N),
           pairwise_correlations (N×N), current_weights (N)]
Action  : weight vector ∈ [0,1]^N  (normalised to sum=1, no shorting)
Reward  : rolling Sharpe ratio  (annualised)
Algorithm: PPO via stable-baselines3
"""
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import plotly.graph_objects as go

from services.stock_service import get_historical

logger = logging.getLogger(__name__)
LOOKBACK = 20
COLORS   = ["#00D4FF", "#7B2FBE", "#00FF94", "#FF6B35", "#FFD700",
            "#FF69B4", "#4ECDC4", "#A78BFA", "#F87171", "#34D399"]


# ── Environment ──────────────────────────────────────────────────────────────

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, returns: pd.DataFrame, lookback: int = LOOKBACK):
        super().__init__()
        self.ret      = returns.values.astype(np.float32)   # (T, N)
        self.tickers  = returns.columns.tolist()
        self.N        = len(self.tickers)
        self.lookback = lookback
        self.T        = len(self.ret)
        self.t        = lookback

        # action: N weights (raw, will be softmax-normalised)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
        )
        # state dim: returns(lookback×N) + vol(N) + corr(N×N) + weights(N)
        obs_dim = self.N * lookback + self.N + self.N * self.N + self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _normalise(self, action: np.ndarray) -> np.ndarray:
        w = np.abs(action).astype(np.float64)
        s = w.sum()
        return (w / s).astype(np.float32) if s > 1e-8 else self.weights.copy()

    def _obs(self) -> np.ndarray:
        window = self.ret[self.t - self.lookback: self.t]           # (L, N)
        vol    = window.std(axis=0) * np.sqrt(252)                  # (N,)
        corr   = np.corrcoef(window.T).flatten()                    # (N²,)
        flat_r = window.flatten()                                    # (L×N,)
        return np.concatenate([flat_r, vol, corr, self.weights]).astype(np.float32)

    def _sharpe(self, weights: np.ndarray) -> float:
        window = self.ret[max(0, self.t - self.lookback): self.t]
        pr = window @ weights
        if len(pr) < 2 or pr.std() < 1e-8:
            return 0.0
        return float((pr.mean() / pr.std()) * np.sqrt(252))

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t       = self.lookback
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        return self._obs(), {}

    def step(self, action):
        self.weights = self._normalise(action)
        daily_ret    = float(self.ret[self.t] @ self.weights)
        sharpe       = self._sharpe(self.weights)
        reward       = sharpe + daily_ret * 10.0

        self.t      += 1
        terminated   = self.t >= self.T - 1
        return self._obs(), reward, terminated, False, {}


# ── Main optimiser ────────────────────────────────────────────────────────────

def optimize_portfolio(
    tickers:   list[str],
    budget:    float = 100_000.0,
    period:    str   = "2y",
    timesteps: int   = 10_000,
) -> dict:
    """
    Run PPO optimisation and return weights + dollar allocations.

    Returns
    -------
    dict with keys:
        weights         – {ticker: weight}
        allocations     – {ticker: {"dollars": ..., "shares": ..., "price": ...}}
        metrics         – Sharpe, return, vol (RL vs equal-weight)
        returns_df      – pd.DataFrame for plotting
        prices_df       – pd.DataFrame for plotting
        final_weights   – np.ndarray
        tickers         – list[str]
        budget          – float
    """
    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    price_map = {}
    for t in tickers:
        df = get_historical(t, period=period)
        if not df.empty:
            price_map[t] = df["Close"]
    valid = list(price_map.keys())

    if len(valid) < 2:
        return {"error": "Need at least 2 tickers with valid price history."}

    prices  = pd.DataFrame(price_map).dropna()
    returns = prices.pct_change().dropna()

    if len(returns) < LOOKBACK + 10:
        return {"error": "Not enough historical data (need > 30 trading days)."}

    # ── 2. Train ──────────────────────────────────────────────────────────────
    logger.info("Training PPO on %d assets, %d days, %d steps", len(valid), len(returns), timesteps)
    env   = DummyVecEnv([lambda: PortfolioEnv(returns)])
    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=min(256, len(returns) - LOOKBACK - 1),
        batch_size=64,
        n_epochs=10,
    )
    model.learn(total_timesteps=timesteps)

    # ── 3. Extract weights via deterministic rollout ──────────────────────────
    test_env = PortfolioEnv(returns)
    obs, _   = test_env.reset()
    w_history: list[np.ndarray] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        w = np.abs(action.astype(np.float64))
        w = (w / w.sum()) if w.sum() > 1e-8 else np.full(len(valid), 1.0 / len(valid))
        w_history.append(w.astype(np.float32))
        obs, _, done, _, _ = test_env.step(action)

    final_w = np.mean(w_history, axis=0).astype(np.float64)
    final_w = final_w / final_w.sum()

    # ── 4. Current prices → share counts ─────────────────────────────────────
    from services.stock_service import get_stock_info
    allocations: dict[str, dict] = {}
    for ticker, weight in zip(valid, final_w):
        dollars = round(budget * weight, 2)
        info    = get_stock_info(ticker)
        price   = info.get("price", 0.0)
        shares  = (dollars / price) if price > 0 else 0.0
        allocations[ticker] = {
            "weight":  round(float(weight), 6),
            "dollars": dollars,
            "shares":  round(shares, 4),
            "price":   round(price, 4),
        }

    # ── 5. Performance metrics ────────────────────────────────────────────────
    eq_w = np.full(len(valid), 1.0 / len(valid))
    rl_r = returns.values @ final_w
    eq_r = returns.values @ eq_w

    def _sharpe(r): return (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    def _ann_ret(r): return r.mean() * 252 * 100
    def _ann_vol(r): return r.std() * np.sqrt(252) * 100

    return {
        "weights":       {t: round(float(w), 4) for t, w in zip(valid, final_w)},
        "allocations":   allocations,
        "metrics": {
            "rl_sharpe":        round(_sharpe(rl_r), 4),
            "eq_sharpe":        round(_sharpe(eq_r), 4),
            "rl_annual_return": round(_ann_ret(rl_r), 2),
            "rl_annual_vol":    round(_ann_vol(rl_r), 2),
            "eq_annual_return": round(_ann_ret(eq_r), 2),
        },
        "returns_df":    returns,
        "prices_df":     prices,
        "final_weights": final_w,
        "tickers":       valid,
        "budget":        budget,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#111827",
    font=dict(color="white", size=12),
    margin=dict(t=50, b=40, l=50, r=20),
)


def build_plots(result: dict) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Return (fig_weights, fig_perf, fig_frontier, fig_budget)."""
    tickers  = result["tickers"]
    fw       = result["final_weights"]
    returns  = result["returns_df"]
    budget   = result["budget"]
    allocs   = result["allocations"]
    eq_w     = np.full(len(tickers), 1.0 / len(tickers))

    # ── Pie: allocation ───────────────────────────────────────────────────────
    fig_w = go.Figure(go.Pie(
        labels=tickers,
        values=[round(float(w) * 100, 2) for w in fw],
        hole=0.45,
        marker=dict(colors=COLORS[:len(tickers)]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig_w.update_layout(title="Optimal Allocation", **_LAYOUT)

    # ── Cumulative returns: RL vs Equal-weight ────────────────────────────────
    rl_cum = (1 + returns.values @ fw).cumprod()
    eq_cum = (1 + returns.values @ eq_w).cumprod()
    fig_p  = go.Figure()
    fig_p.add_trace(go.Scatter(y=rl_cum, name="RL Optimised",
                               line=dict(color="#00D4FF", width=2)))
    fig_p.add_trace(go.Scatter(y=eq_cum, name="Equal Weight",
                               line=dict(color="#7B2FBE", width=2, dash="dash")))
    fig_p.update_layout(
        title="Cumulative Returns: RL vs Equal Weight",
        xaxis_title="Trading Days", yaxis_title="Growth of $1",
        legend=dict(font=dict(color="white")), **_LAYOUT,
    )

    # ── Efficient frontier (Monte Carlo) ─────────────────────────────────────
    N, rng     = len(tickers), np.random.default_rng(42)
    sim_ret, sim_vol, sim_sh = [], [], []
    for _ in range(1500):
        w  = rng.dirichlet(np.ones(N))
        r  = returns.values @ w
        mu = r.mean() * 252 * 100
        sg = r.std()  * np.sqrt(252) * 100
        sim_ret.append(mu); sim_vol.append(sg)
        sim_sh.append(mu / sg if sg > 0 else 0)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=sim_vol, y=sim_ret, mode="markers",
        marker=dict(color=sim_sh, colorscale="Viridis", size=4, showscale=True,
                    colorbar=dict(title="Sharpe", tickfont=dict(color="white"))),
        name="Random Portfolios",
    ))
    rl_v = result["metrics"]["rl_annual_vol"]
    rl_r = result["metrics"]["rl_annual_return"]
    fig_f.add_trace(go.Scatter(
        x=[rl_v], y=[rl_r], mode="markers",
        marker=dict(color="#00D4FF", size=16, symbol="star",
                    line=dict(color="white", width=1)),
        name="RL Optimal ★",
    ))
    fig_f.update_layout(
        title="Efficient Frontier (Monte Carlo 1 500 portfolios)",
        xaxis_title="Volatility %", yaxis_title="Expected Return %",
        legend=dict(font=dict(color="white")), **_LAYOUT,
    )

    # ── Budget bar chart ──────────────────────────────────────────────────────
    dollars = [allocs[t]["dollars"] for t in tickers]
    fig_b   = go.Figure(go.Bar(
        x=tickers, y=dollars,
        marker_color=COLORS[:len(tickers)],
        text=[f"${d:,.0f}" for d in dollars],
        textposition="outside",
        textfont=dict(color="white"),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig_b.update_layout(
        title=f"Dollar Allocation  (Budget: ${budget:,.0f})",
        yaxis_title="USD ($)", xaxis_title="",
        yaxis=dict(tickprefix="$"), **_LAYOUT,
    )

    return fig_w, fig_p, fig_f, fig_b
