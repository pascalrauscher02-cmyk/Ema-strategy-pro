"""
╔══════════════════════════════════════════════════════════════╗
║         EMA STRATEGY PRO  ·  v1.0                           ║
║         Multi-Timeframe Crypto Trading Dashboard             ║
╠══════════════════════════════════════════════════════════════╣
║  Install:  pip install streamlit ccxt pandas-ta plotly      ║
║  Run:      streamlit run ema_strategy_pro.py                ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import ccxt
from datetime import datetime, timezone

# ── yfinance optional import (fallback) ──────────────────────
try:
    import yfinance as yf
    _YFINANCE_OK = True
except ImportError:
    _YFINANCE_OK = False

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EMA Strategy Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════
COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT",
    "MATIC/USDT", "LINK/USDT", "OP/USDT", "ARB/USDT",
]

TIMEFRAMES  = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
HIGHER_TF   = {
    "1m": "5m", "5m": "15m", "15m": "1h",
    "30m": "2h", "1h": "4h", "4h": "1d", "1d": "1w",
}
ATR_MULT    = {
    "1m": 1.5, "5m": 1.5, "15m": 2.0,
    "30m": 2.0, "1h": 2.5, "4h": 3.0, "1d": 4.0,
}
REFRESH_MAP = {"Off": 0, "30s": 30, "1 min": 60, "2 min": 120, "5 min": 300}
N_BARS      = 450   # candles to fetch

# ════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Fira+Code:wght@300;400;500;600&display=swap');

:root {
  --bg:      #070b14;
  --surface: #0c1526;
  --card:    #0f1e36;
  --border:  #1a3056;
  --accent:  #00e5ff;
  --green:   #00ff9f;
  --red:     #ff3d5a;
  --yellow:  #ffd166;
  --purple:  #bd93f9;
  --text:    #cdd9f0;
  --muted:   #4a6080;
}

html, body, [class*="css"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Fira Code', monospace !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: 'Fira Code', monospace !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
}
[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: 10px !important;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 17px !important;
  color: var(--accent) !important;
}

/* ── Signal cards ── */
.sig-long {
  background: linear-gradient(135deg, #003d1f 0%, #001a0d 100%);
  border: 1px solid var(--green);
  border-left: 5px solid var(--green);
  box-shadow: 0 0 40px rgba(0,255,159,0.12);
  border-radius: 10px;
  padding: 22px 24px;
}
.sig-short {
  background: linear-gradient(135deg, #3d0010 0%, #1a0008 100%);
  border: 1px solid var(--red);
  border-left: 5px solid var(--red);
  box-shadow: 0 0 40px rgba(255,61,90,0.12);
  border-radius: 10px;
  padding: 22px 24px;
}
.sig-neutral {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 5px solid var(--muted);
  border-radius: 10px;
  padding: 22px 24px;
}
.sig-label {
  font-family: 'Orbitron', monospace;
  font-size: 30px;
  font-weight: 900;
  letter-spacing: 5px;
  margin: 0 0 4px 0;
}
.sig-sub {
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin: 0;
}
.sig-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 14px;
  font-size: 12px;
}
.sig-cell-label { color: var(--muted); font-size: 10px; letter-spacing: 1px; }
.sig-cell-val   { color: var(--text);  font-weight: 600; margin-top: 2px; }

/* ── Section titles ── */
.sec-title {
  font-family: 'Orbitron', monospace;
  font-size: 11px;
  color: var(--accent);
  letter-spacing: 3px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 7px;
  margin: 18px 0 10px 0;
}

/* ── Stat rows ── */
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid #121e30;
  font-size: 12px;
}
.s-label { color: var(--muted); }
.s-val   { color: var(--text);  }
.s-pos   { color: var(--green); font-weight: 600; }
.s-neg   { color: var(--red);   font-weight: 600; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  font-family: 'Fira Code', monospace !important;
  font-size: 12px !important;
  color: var(--muted) !important;
  padding: 12px 20px !important;
  letter-spacing: 1px;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--card) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: 'Fira Code', monospace !important;
  font-size: 12px !important;
  letter-spacing: 1px;
  border-radius: 5px !important;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: var(--bg) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.4);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

hr { border-color: var(--border) !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
</style>
"""

# ════════════════════════════════════════════════════════════════
#  DATA FETCHING  ·  Multi-source fallback chain
#  Priority: Binance → Bybit → OKX → Gate.io → yFinance → Demo
# ════════════════════════════════════════════════════════════════

# Map our "BTC/USDT" symbols to yfinance tickers
_YF_MAP = {
    "BTC/USDT": "BTC-USD",  "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",  "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",  "DOGE/USDT":"DOGE-USD",
    "ADA/USDT": "ADA-USD",  "AVAX/USDT":"AVAX-USD",
    "MATIC/USDT":"MATIC-USD","LINK/USDT":"LINK-USD",
    "OP/USDT":  "OP-USD",   "ARB/USDT": "ARB-USD",
}

# Map our timeframe strings to yfinance interval/period combos
_YF_TF = {
    "1m":  ("1m",  "7d"),
    "5m":  ("5m",  "60d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h":  ("1h",  "730d"),
    "4h":  ("4h",  "730d"),   # yf uses "4h" since 0.2.x, falls back
    "1d":  ("1d",  "max"),
}

# ccxt exchanges to try in order
_CCXT_EXCHANGES = [
    ("Binance",  lambda: ccxt.binance({"enableRateLimit": True})),
    ("Bybit",    lambda: ccxt.bybit({"enableRateLimit": True})),
    ("OKX",      lambda: ccxt.okx({"enableRateLimit": True})),
    ("Gate.io",  lambda: ccxt.gateio({"enableRateLimit": True})),
    ("KuCoin",   lambda: ccxt.kucoin({"enableRateLimit": True})),
]


def _ccxt_fetch(symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame | None, str]:
    """Try each ccxt exchange until one works. Returns (df, source_name)."""
    for name, factory in _CCXT_EXCHANGES:
        try:
            ex  = factory()
            # Some exchanges use slightly different symbol formats
            sym = symbol  # e.g. "BTC/USDT"
            raw = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            if raw and len(raw) >= 50:
                df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)
                return df.astype(float), name
        except Exception:
            continue
    return None, ""


def _yfinance_fetch(symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame | None, str]:
    """Fallback: yfinance (uses USD pairs, slightly different volume)."""
    if not _YFINANCE_OK:
        return None, ""
    ticker = _YF_MAP.get(symbol)
    if not ticker:
        return None, ""
    tf_str, period = _YF_TF.get(timeframe, ("1h", "730d"))
    try:
        raw = yf.download(
            ticker, period=period, interval=tf_str,
            progress=False, auto_adjust=True, prepost=False,
        )
        if raw is None or raw.empty:
            return None, ""
        raw = raw.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low",  "Close": "close", "Volume": "volume",
        })
        raw = raw[["open","high","low","close","volume"]].dropna()
        raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None)
        raw.index.name = "ts"
        # Trim to limit rows
        if len(raw) > limit:
            raw = raw.iloc[-limit:]
        return raw.astype(float), "yFinance"
    except Exception:
        return None, ""


def _demo_fetch(symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, str]:
    """
    Last-resort: generate synthetic OHLCV via geometric Brownian motion.
    Gives a working app for demo/offline use.
    """
    rng    = np.random.default_rng(42)
    prices = [30_000.0 if "BTC" in symbol else
              2_000.0  if "ETH" in symbol else 150.0]
    for _ in range(limit - 1):
        prices.append(prices[-1] * np.exp(rng.normal(0, 0.012)))

    prices = np.array(prices)
    noise  = prices * 0.005

    freqs = {"1m": "1min", "5m": "5min", "15m": "15min",
              "30m": "30min", "1h": "h", "4h": "4h", "1d": "D"}
    freq = freqs.get(timeframe, "h")

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq=freq)
    df  = pd.DataFrame({
        "open":   prices - noise * rng.uniform(0, 1, limit),
        "high":   prices + noise * rng.uniform(0.5, 2, limit),
        "low":    prices - noise * rng.uniform(0.5, 2, limit),
        "close":  prices,
        "volume": rng.uniform(500, 5000, limit) * prices / 10_000,
    }, index=idx)
    df.index.name = "ts"
    return df.astype(float), "DEMO (offline)"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(
    symbol: str, timeframe: str, limit: int = N_BARS
) -> tuple[pd.DataFrame | None, str]:
    """
    Returns (df, source_name).
    Tries: Binance → Bybit → OKX → Gate.io → KuCoin → yFinance → Demo.
    """
    df, src = _ccxt_fetch(symbol, timeframe, limit)
    if df is not None:
        return df, src

    df, src = _yfinance_fetch(symbol, timeframe, limit)
    if df is not None:
        return df, src

    df, src = _demo_fetch(symbol, timeframe, limit)
    return df, src


# ════════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["ema_f"]  = ta.ema(df["close"], length=fast)
    df["ema_s"]  = ta.ema(df["close"], length=slow)
    df["ema200"] = ta.ema(df["close"], length=200)
    df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["vol_ma"] = ta.sma(df["volume"], length=20)

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        df["adx"] = adx_df[adx_cols[0]] if adx_cols else np.nan
    else:
        df["adx"] = np.nan

    return df


# ════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════
def make_signals(
    df: pd.DataFrame,
    htf_ema200: float | None,
    use_rsi: bool,
    use_vol: bool,
    tf: str,
) -> pd.DataFrame:
    df = df.copy()
    df["prev_f"] = df["ema_f"].shift(1)
    df["prev_s"] = df["ema_s"].shift(1)

    cross_up   = (df["ema_f"] > df["ema_s"]) & (df["prev_f"] <= df["prev_s"])
    cross_down = (df["ema_f"] < df["ema_s"]) & (df["prev_f"] >= df["prev_s"])

    # HTF bias filter
    if htf_ema200 is not None:
        htf_bull = df["close"] > htf_ema200
        htf_bear = df["close"] < htf_ema200
    else:
        htf_bull = pd.Series(True,  index=df.index)
        htf_bear = pd.Series(True,  index=df.index)

    rsi_long_ok  = (df["rsi"] < 70) if use_rsi else pd.Series(True, index=df.index)
    rsi_short_ok = (df["rsi"] > 30) if use_rsi else pd.Series(True, index=df.index)
    vol_ok       = (df["volume"] > df["vol_ma"]) if use_vol else pd.Series(True, index=df.index)

    df["signal"] = 0
    df.loc[cross_up   & htf_bull & rsi_long_ok  & vol_ok, "signal"] =  1
    df.loc[cross_down & htf_bear & rsi_short_ok & vol_ok, "signal"] = -1

    # ATR trailing stop lines (for display)
    mult = ATR_MULT.get(tf, 2.0)
    df["trail_long"]  = df["close"] - mult * df["atr"]
    df["trail_short"] = df["close"] + mult * df["atr"]

    return df


# ════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════
def run_backtest(
    df_raw: pd.DataFrame,
    htf_ema200: float | None,
    tf: str,
    capital: float,
    risk_pct: float,
    use_rsi: bool,
    use_vol: bool,
    fast: int,
    slow: int,
) -> tuple[list, list, pd.DataFrame]:

    df = add_indicators(df_raw, fast, slow)
    df = make_signals(df, htf_ema200, use_rsi, use_vol, tf)
    df = df.dropna(subset=["ema_f", "ema_s", "atr", "rsi"]).copy()
    df = df.reset_index()           # ts becomes a column

    mult   = ATR_MULT.get(tf, 2.0)
    trades = []
    equity = [capital]
    cap    = capital
    peak   = capital
    pos    = 0        # 1=long, -1=short, 0=flat
    ep     = 0.0      # entry price
    sz     = 0.0      # position size (units)
    ts_sl  = 0.0      # trailing stop price
    eidx   = 0        # entry bar index

    def _ts_col():
        return "ts" if "ts" in df.columns else df.columns[0]

    def close_trade(i: int, xp: float, reason: str) -> None:
        nonlocal cap, pos
        pnl = ((xp - ep) if pos == 1 else (ep - xp)) * sz
        cap += pnl
        tc  = _ts_col()
        trades.append({
            "Entry Time":  df.at[eidx, tc],
            "Exit Time":   df.at[i,    tc],
            "Direction":   "🟢 Long" if pos == 1 else "🔴 Short",
            "Entry Price": round(ep, 4),
            "Exit Price":  round(xp, 4),
            "P&L $":       round(pnl, 2),
            "Exit Reason": reason,
        })
        pos = 0

    for i in range(1, len(df)):
        row   = df.iloc[i]
        atr_v = row["atr"] if not np.isnan(row["atr"]) else row["close"] * 0.01
        risk_amt = cap * (risk_pct / 100)

        # ── Hard drawdown guard (-5 %)
        if cap < peak * 0.95 and pos != 0:
            close_trade(i, row["close"], "Max DD -5%")
            equity.append(cap)
            continue

        peak = max(peak, cap)

        # ── Update trailing stop & check hit
        if pos == 1:
            new_ts = row["close"] - mult * atr_v
            ts_sl  = max(ts_sl, new_ts)
            if row["low"] <= ts_sl:
                close_trade(i, max(float(row["open"]), ts_sl), "Trail Stop")
        elif pos == -1:
            new_ts = row["close"] + mult * atr_v
            ts_sl  = min(ts_sl, new_ts)
            if row["high"] >= ts_sl:
                close_trade(i, min(float(row["open"]), ts_sl), "Trail Stop")

        # ── New signal → reverse if needed
        sig = int(row["signal"])
        if sig != 0 and sig != pos:
            if pos != 0:
                close_trade(i, float(row["close"]), "Reversal")
            # Open new position
            pos   = sig
            ep    = float(row["close"])
            sz    = risk_amt / (mult * atr_v) if atr_v > 0 else 0
            ts_sl = ep - mult * atr_v if pos == 1 else ep + mult * atr_v
            eidx  = i

        equity.append(cap)

    return trades, equity, df


# ════════════════════════════════════════════════════════════════
#  STATISTICS
# ════════════════════════════════════════════════════════════════
def calc_stats(trades: list, equity: list, capital: float) -> dict:
    if not trades:
        return {}

    tdf  = pd.DataFrame(trades)
    wins = tdf[tdf["P&L $"] > 0]
    loss = tdf[tdf["P&L $"] <= 0]

    pf_den = abs(loss["P&L $"].sum())
    pf     = wins["P&L $"].sum() / pf_den if pf_den > 0 else float("inf")

    eq   = np.array(equity)
    rmax = np.maximum.accumulate(eq)
    dd   = (eq - rmax) / rmax * 100
    max_dd = float(dd.min())

    rets   = pd.Series(equity).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252 * 24)) if rets.std() > 0 else 0.0

    return {
        "Total Trades":  len(tdf),
        "Win Rate":      f"{len(wins)/len(tdf)*100:.1f}%",
        "Total P&L":     f"${tdf['P&L $'].sum():,.2f}",
        "Total Return":  f"{(equity[-1]/capital-1)*100:.2f}%",
        "Profit Factor": f"{pf:.2f}",
        "Max Drawdown":  f"{max_dd:.2f}%",
        "Avg Win":       f"${wins['P&L $'].mean():.2f}" if len(wins) else "$0.00",
        "Avg Loss":      f"${loss['P&L $'].mean():.2f}" if len(loss) else "$0.00",
        "Sharpe Ratio":  f"{sharpe:.2f}",
        "Final Capital": f"${equity[-1]:,.2f}",
    }


# ════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════════════
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#070b14",
    plot_bgcolor="#0c1526",
    font=dict(family="Fira Code, monospace", color="#4a6080", size=11),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis_rangeslider_visible=False,
    legend=dict(
        orientation="h", y=1.03, x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
)


def build_price_chart(df: pd.DataFrame, trades: list, fast: int, slow: int) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        shared_xaxes=True,
        vertical_spacing=0.012,
        subplot_titles=("", "Volume", "RSI (14)", "ADX (14)"),
    )

    # ── Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#00ff9f", increasing_fillcolor="#002e16",
        decreasing_line_color="#ff3d5a", decreasing_fillcolor="#2e000d",
        line_width=1,
    ), row=1, col=1)

    # ── EMAs
    ema_cfg = [
        ("ema_f",  f"EMA {fast}", "#00e5ff", "solid"),
        ("ema_s",  f"EMA {slow}", "#ffd166", "solid"),
        ("ema200", "EMA 200",     "#bd93f9", "dot"),
    ]
    for col, name, color, dash in ema_cfg:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.5, dash=dash), opacity=0.9,
            ), row=1, col=1)

    # ── Trailing stop (long side) – dashed red
    if "trail_long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["trail_long"], name="Trail Stop",
            line=dict(color="#ff3d5a", width=0.9, dash="dash"), opacity=0.45,
        ), row=1, col=1)

    # ── Trade entry markers
    for entries, sym, col, label in [
        ([t for t in trades if "Long"  in t["Direction"]], "triangle-up",   "#00ff9f", "Long  Entry"),
        ([t for t in trades if "Short" in t["Direction"]], "triangle-down", "#ff3d5a", "Short Entry"),
    ]:
        if entries:
            fig.add_trace(go.Scatter(
                x=[t["Entry Time"] for t in entries],
                y=[t["Entry Price"] for t in entries],
                mode="markers", name=label,
                marker=dict(symbol=sym, size=11, color=col,
                            line=dict(width=1, color="#ffffff")),
            ), row=1, col=1)

    # ── Volume bars
    v_colors = [
        "#002e16" if c >= o else "#2e000d"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume", marker_color=v_colors, opacity=0.8,
    ), row=2, col=1)
    if "vol_ma" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vol_ma"],
            line=dict(color="#ffd166", width=1.2), name="Vol MA",
        ), row=2, col=1)

    # ── RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            line=dict(color="#00e5ff", width=1.4), name="RSI",
        ), row=3, col=1)
        for lvl, col in [(70, "#ff3d5a"), (50, "#4a6080"), (30, "#00ff9f")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col,
                          line_width=0.8, opacity=0.5, row=3, col=1)

    # ── ADX
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["adx"],
            line=dict(color="#bd93f9", width=1.4), name="ADX",
        ), row=4, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="#ffd166",
                      line_width=0.8, opacity=0.6, row=4, col=1)

    fig.update_layout(**_LAYOUT, height=740)
    fig.update_yaxes(gridcolor="#111e33", zerolinecolor="#111e33")
    fig.update_xaxes(gridcolor="#111e33")
    return fig


def build_equity_chart(equity: list, capital: float) -> go.Figure:
    eq   = np.array(equity)
    bars = list(range(len(eq)))
    rmax = np.maximum.accumulate(eq)
    dd   = (eq - rmax) / rmax * 100

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity Curve ($)", "Drawdown (%)"),
    )

    fig.add_trace(go.Scatter(
        x=bars, y=eq, name="Equity",
        line=dict(color="#00ff9f", width=2),
        fill="tozeroy", fillcolor="rgba(0,255,159,0.07)",
    ), row=1, col=1)
    fig.add_hline(y=capital, line_dash="dash",
                  line_color="#4a6080", line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bars, y=dd, name="Drawdown",
        line=dict(color="#ff3d5a", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,61,90,0.10)",
    ), row=2, col=1)
    fig.add_hline(y=-5, line_dash="dash",
                  line_color="#ff3d5a", line_width=0.8, opacity=0.5, row=2, col=1)

    fig.update_layout(**_LAYOUT, height=440)
    fig.update_yaxes(gridcolor="#111e33", zerolinecolor="#111e33")
    return fig


# ════════════════════════════════════════════════════════════════
#  SIGNAL BOX HTML
# ════════════════════════════════════════════════════════════════
def signal_box(
    sig: int, coin: str, tf: str, price: float,
    rsi: float, adx: float, htf_bias: str,
    trail_long: float, trail_short: float,
) -> str:
    if sig == 1:
        cls   = "sig-long"
        label = "▲  LONG"
        color = "#00ff9f"
        trail_disp = f"${trail_long:,.4f}"
    elif sig == -1:
        cls   = "sig-short"
        label = "▼  SHORT"
        color = "#ff3d5a"
        trail_disp = f"${trail_short:,.4f}"
    else:
        cls   = "sig-neutral"
        label = "◆  FLAT / WAIT"
        color = "#4a6080"
        trail_disp = "—"

    bias_color = "#00ff9f" if "BULL" in htf_bias.upper() else "#ff3d5a"

    return f"""
<div class="{cls}">
  <p class="sig-label" style="color:{color}">{label}</p>
  <p class="sig-sub">{coin} · {tf.upper()} · {datetime.utcnow().strftime("%H:%M:%S UTC")}</p>

  <div class="sig-grid">
    <div>
      <div class="sig-cell-label">PRICE</div>
      <div class="sig-cell-val">${price:,.4f}</div>
    </div>
    <div>
      <div class="sig-cell-label">TRAIL STOP</div>
      <div class="sig-cell-val">{trail_disp}</div>
    </div>
    <div>
      <div class="sig-cell-label">RSI (14)</div>
      <div class="sig-cell-val">{rsi:.1f}</div>
    </div>
    <div>
      <div class="sig-cell-label">ADX (14)</div>
      <div class="sig-cell-val">{adx:.1f}</div>
    </div>
    <div style="grid-column:1/-1">
      <div class="sig-cell-label">HTF BIAS</div>
      <div class="sig-cell-val" style="color:{bias_color}">{htf_bias}</div>
    </div>
  </div>
</div>
"""


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar() -> tuple:
    sb = st.sidebar

    sb.markdown(
        '<p style="font-family:Orbitron,monospace;font-size:15px;color:#00e5ff;'
        'letter-spacing:3px;font-weight:900;margin-bottom:2px">⚡ EMA STRATEGY PRO</p>'
        '<p style="font-size:10px;color:#4a6080;letter-spacing:2px;margin:0">'
        'v1.0 · Multi-Timeframe · Crypto</p>',
        unsafe_allow_html=True,
    )
    sb.divider()

    sb.markdown('<div class="sec-title">MARKET</div>', unsafe_allow_html=True)
    coin = sb.selectbox("Symbol", COINS, index=0)
    tf   = sb.selectbox("Timeframe", TIMEFRAMES, index=4)   # default 1h

    htf_label = HIGHER_TF.get(tf, "—")
    sb.caption(f"📐 Trend filter: EMA 200 on **{htf_label}**")

    sb.markdown('<div class="sec-title">EMA PARAMETERS</div>', unsafe_allow_html=True)
    fast = sb.slider("EMA Fast", 5,  50, 20, 1)
    slow = sb.slider("EMA Slow", 20, 200, 50, 1)
    if fast >= slow:
        sb.warning("⚠ Fast EMA must be < Slow EMA")

    sb.markdown('<div class="sec-title">RISK MANAGEMENT</div>', unsafe_allow_html=True)
    capital  = sb.number_input("Start Capital ($)", 500, 1_000_000, 10_000, 500)
    risk_pct = sb.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    sb.caption("⚙ Position size = (capital × risk%) ÷ (ATR × multiplier)")

    sb.markdown('<div class="sec-title">SIGNAL FILTERS</div>', unsafe_allow_html=True)
    use_rsi = sb.checkbox("RSI filter  (Long < 70 · Short > 30)", value=True)
    use_vol = sb.checkbox("Volume filter  (cross > Vol MA 20)",    value=True)

    sb.markdown('<div class="sec-title">AUTO-REFRESH</div>', unsafe_allow_html=True)
    ref_label = sb.selectbox("Interval", list(REFRESH_MAP.keys()), index=2)
    ref_secs  = REFRESH_MAP[ref_label]

    sb.divider()
    sb.markdown(
        '<p style="font-size:10px;color:#4a6080;text-align:center;line-height:1.6">'
        '⚠ For educational & research purposes only.<br>'
        'Not financial advice. Crypto trading involves<br>significant risk of loss.</p>',
        unsafe_allow_html=True,
    )
    return coin, tf, fast, slow, capital, risk_pct, use_rsi, use_vol, ref_secs


# ════════════════════════════════════════════════════════════════
#  AUTO-REFRESH  (session-state based, no hard sleep blocking)
# ════════════════════════════════════════════════════════════════
def handle_auto_refresh(ref_secs: int) -> None:
    if ref_secs == 0:
        return

    now = time.time()
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = now

    elapsed   = now - st.session_state.last_refresh
    remaining = max(0, int(ref_secs - elapsed))

    st.sidebar.caption(f"🔄 Next refresh in **{remaining}s**")

    if elapsed >= ref_secs:
        st.session_state.last_refresh = now
        st.cache_data.clear()

    time.sleep(1)
    st.rerun()


# ════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    coin, tf, fast, slow, capital, risk_pct, use_rsi, use_vol, ref_secs = render_sidebar()

    # ── Header bar ──────────────────────────────────────────────
    hcol1, hcol2, hcol3 = st.columns([3, 1.2, 1])
    _header_ph = hcol1.empty()   # filled after fetch so we know src_main
    with hcol2:
        st.metric("UTC Time", datetime.utcnow().strftime("%H:%M:%S"))
    with hcol3:
        refresh_btn = st.button("⟳  Refresh Now", use_container_width=True)
        if refresh_btn:
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # ── Fetch data ───────────────────────────────────────────────
    status = st.empty()
    status.info("⏳ Fetching market data… trying Binance → Bybit → OKX → Gate.io → yFinance")

    df_main, src_main = fetch_ohlcv(coin, tf)
    htf               = HIGHER_TF.get(tf, "4h")
    df_htf,  src_htf  = fetch_ohlcv(coin, htf, limit=250)

    status.empty()

    # Now fill in the header with the actual source
    _header_ph.markdown(
        f'<h1 style="font-family:Orbitron,monospace;font-size:20px;color:#00e5ff;'
        f'letter-spacing:4px;margin:0;padding:0">⚡ EMA STRATEGY PRO</h1>'
        f'<p style="font-size:11px;color:#4a6080;letter-spacing:2px;margin:4px 0 0 0">'
        f'{coin} · {tf.upper()} · HTF: {HIGHER_TF.get(tf,"?").upper()}'
        f' &nbsp;·&nbsp; SOURCE: {src_main}</p>',
        unsafe_allow_html=True,
    )

    status.empty()

    # ── Source badge ─────────────────────────────────────────────
    is_demo = "DEMO" in src_main
    badge_color = "#ffd166" if is_demo else "#00ff9f"
    badge_icon  = "⚠ DEMO"  if is_demo else "✓ LIVE"
    st.sidebar.markdown(
        f'<div style="background:#0f1e36;border:1px solid {badge_color};border-radius:6px;'
        f'padding:8px 12px;margin-bottom:8px;font-size:11px">'
        f'<span style="color:{badge_color};font-weight:700">{badge_icon}</span>'
        f' &nbsp;<span style="color:#4a6080">Data:</span>'
        f' <span style="color:#cdd9f0">{src_main}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if is_demo:
        st.warning(
            "⚠ **All exchanges unreachable & yFinance unavailable** — showing synthetic "
            "DEMO data generated via Brownian motion. Strategy logic is fully functional "
            "but prices are not real. Check your internet connection or install yfinance:\n"
            "```\npip install yfinance\n```"
        )

    if df_main is None or df_main.empty:
        st.error("⚠ No data available even from fallback sources.")
        st.stop()

    # ── HTF EMA200 ────────────────────────────────────────────────
    htf_ema200_val = None
    htf_bias_str   = "UNKNOWN (no HTF data)"

    if df_htf is not None and not df_htf.empty:
        df_htf_ind = add_indicators(df_htf, fast, slow)
        if "ema200" in df_htf_ind.columns and df_htf_ind["ema200"].notna().any():
            htf_ema200_val = float(df_htf_ind["ema200"].iloc[-1])
            htf_last_close = float(df_htf_ind["close"].iloc[-1])
            if htf_last_close > htf_ema200_val:
                htf_bias_str = f"BULLISH  (Price > {htf} EMA200 @ ${htf_ema200_val:,.2f})"
            else:
                htf_bias_str = f"BEARISH  (Price < {htf} EMA200 @ ${htf_ema200_val:,.2f})"

    # ── Indicators + signals on main df ──────────────────────────
    df = add_indicators(df_main, fast, slow)
    df = make_signals(df, htf_ema200_val, use_rsi, use_vol, tf)

    # ── Current state ─────────────────────────────────────────────
    last = df.iloc[-1]

    # Find most recent non-zero signal (= current active direction)
    recent_signals = df[df["signal"] != 0]
    curr_sig = int(recent_signals["signal"].iloc[-1]) if not recent_signals.empty else 0

    curr_price    = float(last["close"])
    curr_rsi      = float(last["rsi"])   if not np.isnan(last["rsi"])   else 0.0
    curr_adx      = float(last["adx"])   if "adx" in last and not np.isnan(last["adx"]) else 0.0
    curr_tl       = float(last["trail_long"])  if not np.isnan(last["trail_long"])  else curr_price
    curr_ts       = float(last["trail_short"]) if not np.isnan(last["trail_short"]) else curr_price

    # ── Backtest ──────────────────────────────────────────────────
    trades, equity, df_bt = run_backtest(
        df_main, htf_ema200_val, tf, capital, risk_pct,
        use_rsi, use_vol, fast, slow,
    )
    stats = calc_stats(trades, equity, capital)

    # ════════════════════════════════════════════════════════════
    #  TABS
    # ════════════════════════════════════════════════════════════
    tab_live, tab_bt, tab_log = st.tabs([
        "📡   LIVE SIGNAL",
        "📊   BACKTEST",
        "📋   TRADE LOG",
    ])

    # ──────────────────────────────────────────────────────────
    #  Tab 1  ·  Live Signal
    # ──────────────────────────────────────────────────────────
    with tab_live:
        sig_col, chart_col = st.columns([1, 3.2])

        with sig_col:
            # Signal box
            st.markdown(
                signal_box(curr_sig, coin, tf, curr_price,
                           curr_rsi, curr_adx, htf_bias_str, curr_tl, curr_ts),
                unsafe_allow_html=True,
            )

            # Indicator table
            st.markdown('<div class="sec-title" style="margin-top:18px">INDICATOR VALUES</div>',
                        unsafe_allow_html=True)

            ind_items = {
                f"EMA {fast}":  f"${last['ema_f']:,.2f}"   if not np.isnan(last['ema_f'])  else "N/A",
                f"EMA {slow}":  f"${last['ema_s']:,.2f}"   if not np.isnan(last['ema_s'])  else "N/A",
                "EMA 200":      f"${last['ema200']:,.2f}"   if not np.isnan(last['ema200']) else "N/A",
                "ATR (14)":     f"${last['atr']:,.4f}"      if not np.isnan(last['atr'])    else "N/A",
                "RSI (14)":     f"{curr_rsi:.2f}",
                "ADX (14)":     f"{curr_adx:.2f}",
                "Volume":       f"{last['volume']:,.0f}",
                "Vol MA (20)":  f"{last['vol_ma']:,.0f}"    if not np.isnan(last['vol_ma']) else "N/A",
                "HTF EMA200":   f"${htf_ema200_val:,.2f}"   if htf_ema200_val else "N/A",
                "Trail (Long)": f"${curr_tl:,.4f}",
                "Trail (Short)":f"${curr_ts:,.4f}",
                "ATR Mult.":    f"×{ATR_MULT.get(tf, 2.0)}",
            }
            for k, v in ind_items.items():
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span class="s-label">{k}</span>'
                    f'<span class="s-val">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with chart_col:
            st.plotly_chart(
                build_price_chart(df, trades, fast, slow),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )

    # ──────────────────────────────────────────────────────────
    #  Tab 2  ·  Backtest
    # ──────────────────────────────────────────────────────────
    with tab_bt:
        if stats:
            st.markdown('<div class="sec-title">PERFORMANCE SUMMARY</div>',
                        unsafe_allow_html=True)

            cols = st.columns(5)
            for idx, (label, val) in enumerate(stats.items()):
                with cols[idx % 5]:
                    st.metric(label, val)

            st.markdown("")
            st.plotly_chart(
                build_equity_chart(equity, capital),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Win / Loss distribution mini chart
            if trades:
                st.markdown('<div class="sec-title">P&L DISTRIBUTION</div>',
                            unsafe_allow_html=True)
                tdf = pd.DataFrame(trades)
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=tdf["P&L $"],
                    nbinsx=40,
                    marker_color=np.where(tdf["P&L $"] >= 0, "#00ff9f", "#ff3d5a"),
                    opacity=0.8,
                    name="P&L Distribution",
                ))
                fig_dist.update_layout(**_LAYOUT, height=260,
                                       xaxis_title="P&L ($)", yaxis_title="Count")
                fig_dist.update_yaxes(gridcolor="#111e33")
                st.plotly_chart(fig_dist, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            st.info(
                "No trades were generated with the current settings.\n\n"
                "Try: **disabling RSI/Volume filters**, or choose a longer timeframe "
                "that has more data / more crossovers."
            )

    # ──────────────────────────────────────────────────────────
    #  Tab 3  ·  Trade Log
    # ──────────────────────────────────────────────────────────
    with tab_log:
        if trades:
            tdf = pd.DataFrame(trades)

            total_pnl = tdf["P&L $"].sum()
            wins      = (tdf["P&L $"] > 0).sum()
            color_pnl = "#00ff9f" if total_pnl >= 0 else "#ff3d5a"

            st.markdown(
                f'<p style="font-size:13px;margin-bottom:8px">'
                f'<b>{len(tdf)}</b> trades &nbsp;·&nbsp; '
                f'<b>{wins}</b> wins &nbsp;·&nbsp; '
                f'Total P&L: <b style="color:{color_pnl}">${total_pnl:,.2f}</b></p>',
                unsafe_allow_html=True,
            )

            def _color_pnl(val):
                if isinstance(val, (int, float)):
                    return "color:#00ff9f" if val > 0 else "color:#ff3d5a"
                return ""

            try:
                styled = tdf.style.map(_color_pnl, subset=["P&L $"])
            except AttributeError:
                styled = tdf.style.applymap(_color_pnl, subset=["P&L $"])

            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No trades to display. Run a backtest first.")

    # ── Auto-refresh ──────────────────────────────────────────────
    handle_auto_refresh(ref_secs)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
