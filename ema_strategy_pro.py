"""
╔══════════════════════════════════════════════════════════════╗
║         EMA STRATEGY PRO  ·  v2.0                           ║
║         Multi-Timeframe Crypto · Regime-Aware Engine        ║
╠══════════════════════════════════════════════════════════════╣
║  Install:  pip install -r requirements.txt                  ║
║  Run:      streamlit run ema_strategy_pro.py                ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import ccxt
from datetime import datetime

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
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
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT",
    "XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT",
    "MATIC/USDT","LINK/USDT","OP/USDT","ARB/USDT",
]
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d"]
HIGHER_TF  = {
    "1m":"5m","5m":"15m","15m":"1h",
    "30m":"2h","1h":"4h","4h":"1d","1d":"1w",
}
ATR_MULT = {
    "1m":1.5,"5m":1.5,"15m":2.0,
    "30m":2.0,"1h":2.5,"4h":3.0,"1d":4.0,
}
REFRESH_MAP = {"Off":0,"30s":30,"1 min":60,"2 min":120,"5 min":300}

R_STRONG_BULL, R_WEAK_BULL, R_NEUTRAL, R_WEAK_BEAR, R_STRONG_BEAR = 2,1,0,-1,-2

REGIME_LABEL = {
    2:"STRONG BULL", 1:"WEAK BULL", 0:"NEUTRAL", -1:"WEAK BEAR", -2:"STRONG BEAR",
}
REGIME_COLOR = {
    2:"#00ff9f", 1:"#64ffda", 0:"#ffd166", -1:"#ff9a6c", -2:"#ff3d5a",
}

N_BARS = 1000

# ════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Fira+Code:wght@300;400;500;600&display=swap');
:root{
  --bg:#070b14;--surface:#0c1526;--card:#0f1e36;--border:#1a3056;
  --accent:#00e5ff;--green:#00ff9f;--red:#ff3d5a;--yellow:#ffd166;
  --purple:#bd93f9;--text:#cdd9f0;--muted:#4a6080;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;
  font-family:'Fira Code',monospace!important}
section[data-testid="stSidebar"]{background:var(--surface)!important;
  border-right:1px solid var(--border)!important}
section[data-testid="stSidebar"] *{font-family:'Fira Code',monospace!important}
[data-testid="metric-container"]{background:var(--card)!important;
  border:1px solid var(--border)!important;border-radius:8px!important;padding:14px 16px!important}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:10px!important;
  letter-spacing:1.5px;text-transform:uppercase}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;
  font-size:17px!important;color:var(--accent)!important}
.sig-long{background:linear-gradient(135deg,#003d1f,#001a0d);border:1px solid var(--green);
  border-left:5px solid var(--green);box-shadow:0 0 40px rgba(0,255,159,.12);border-radius:10px;padding:22px 24px}
.sig-short{background:linear-gradient(135deg,#3d0010,#1a0008);border:1px solid var(--red);
  border-left:5px solid var(--red);box-shadow:0 0 40px rgba(255,61,90,.12);border-radius:10px;padding:22px 24px}
.sig-neutral{background:var(--card);border:1px solid var(--border);border-left:5px solid var(--muted);
  border-radius:10px;padding:22px 24px}
.sig-label{font-family:'Orbitron',monospace;font-size:26px;font-weight:900;letter-spacing:4px;margin:0 0 4px 0}
.sig-sub{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin:0}
.sig-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;font-size:12px}
.sig-cell-label{color:var(--muted);font-size:10px;letter-spacing:1px}
.sig-cell-val{color:var(--text);font-weight:600;margin-top:2px}
.sec-title{font-family:'Orbitron',monospace;font-size:10px;color:var(--accent);letter-spacing:3px;
  text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:7px;margin:16px 0 8px 0}
.stat-row{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid #121e30;font-size:12px}
.s-label{color:var(--muted)}.s-val{color:var(--text)}.s-pos{color:var(--green);font-weight:600}
.s-neg{color:var(--red);font-weight:600}
[data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:1px solid var(--border)!important;gap:0!important}
[data-baseweb="tab"]{font-family:'Fira Code',monospace!important;font-size:12px!important;
  color:var(--muted)!important;padding:12px 20px!important;letter-spacing:1px}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important}
.stButton>button{background:var(--card)!important;border:1px solid var(--accent)!important;
  color:var(--accent)!important;font-family:'Fira Code',monospace!important;font-size:12px!important;
  letter-spacing:1px;border-radius:5px!important;transition:all .2s}
.stButton>button:hover{background:var(--accent)!important;color:var(--bg)!important;
  box-shadow:0 0 18px rgba(0,229,255,.4)}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
hr{border-color:var(--border)!important}
h1,h2,h3{font-family:'Orbitron',monospace!important}
.regime-bar{border-radius:8px;padding:12px 18px;margin-bottom:10px;display:flex;align-items:center;gap:14px}
.regime-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.regime-name{font-family:'Orbitron',monospace;font-size:13px;font-weight:700;letter-spacing:2px}
.regime-sub{font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:2px}
.score-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:6px 0}
.score-label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px}
.score-track{background:#111e33;border-radius:4px;height:8px;overflow:hidden}
.score-fill{height:100%;border-radius:4px}
.score-val{font-family:'Orbitron',monospace;font-size:14px;font-weight:700;margin-top:5px}
.src-badge{background:#0f1e36;border-radius:6px;padding:7px 12px;font-size:11px;margin-bottom:8px}
</style>
"""

# ════════════════════════════════════════════════════════════════
#  DATA SOURCES
# ════════════════════════════════════════════════════════════════
_CCXT_EXCHANGES = [
    ("Binance", lambda: ccxt.binance({"enableRateLimit":True})),
    ("Bybit",   lambda: ccxt.bybit({"enableRateLimit":True})),
    ("OKX",     lambda: ccxt.okx({"enableRateLimit":True})),
    ("Gate.io", lambda: ccxt.gateio({"enableRateLimit":True})),
    ("KuCoin",  lambda: ccxt.kucoin({"enableRateLimit":True})),
]
_YF_MAP = {
    "BTC/USDT":"BTC-USD","ETH/USDT":"ETH-USD","SOL/USDT":"SOL-USD",
    "BNB/USDT":"BNB-USD","XRP/USDT":"XRP-USD","DOGE/USDT":"DOGE-USD",
    "ADA/USDT":"ADA-USD","AVAX/USDT":"AVAX-USD","MATIC/USDT":"MATIC-USD",
    "LINK/USDT":"LINK-USD","OP/USDT":"OP-USD","ARB/USDT":"ARB-USD",
}
_YF_TF = {
    "1m":("1m","7d"),"5m":("5m","60d"),"15m":("15m","60d"),
    "30m":("30m","60d"),"1h":("1h","730d"),"4h":("4h","730d"),"1d":("1d","max"),
}


def _from_ccxt(symbol, timeframe, limit):
    for name, factory in _CCXT_EXCHANGES:
        try:
            raw = factory().fetch_ohlcv(symbol, timeframe, limit=limit)
            if raw and len(raw) >= 50:
                df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)
                return df.astype(float), name
        except Exception:
            continue
    return None, ""


def _from_yf(symbol, timeframe, limit):
    if not _YF_OK:
        return None, ""
    ticker = _YF_MAP.get(symbol)
    if not ticker:
        return None, ""
    tf_str, period = _YF_TF.get(timeframe, ("1h","730d"))
    try:
        raw = yf.download(ticker, period=period, interval=tf_str,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None, ""
        raw = raw.rename(columns={"Open":"open","High":"high","Low":"low",
                                   "Close":"close","Volume":"volume"})
        raw = raw[["open","high","low","close","volume"]].dropna()
        raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None)
        raw.index.name = "ts"
        if len(raw) > limit:
            raw = raw.iloc[-limit:]
        return raw.astype(float), "yFinance"
    except Exception:
        return None, ""


def _demo(symbol, timeframe, limit):
    rng  = np.random.default_rng(42)
    seed = 30_000. if "BTC" in symbol else 2_000. if "ETH" in symbol else 150.
    p    = [seed]
    for _ in range(limit - 1):
        p.append(p[-1] * np.exp(rng.normal(0.0002, 0.013)))
    p = np.array(p)
    n = p * 0.006
    freqs = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min",
              "1h":"h","4h":"4h","1d":"D"}
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit,
                        freq=freqs.get(timeframe,"h"))
    df = pd.DataFrame({
        "open":  p - n*rng.uniform(0,1,limit),
        "high":  p + n*rng.uniform(0.5,2,limit),
        "low":   p - n*rng.uniform(0.5,2,limit),
        "close": p,
        "volume":rng.uniform(500,5000,limit)*p/10_000,
    }, index=idx)
    df.index.name = "ts"
    return df.astype(float), "DEMO (offline)"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(symbol, timeframe, limit=N_BARS):
    df, src = _from_ccxt(symbol, timeframe, limit)
    if df is not None:
        return df, src
    df, src = _from_yf(symbol, timeframe, limit)
    if df is not None:
        return df, src
    return _demo(symbol, timeframe, limit)


# ════════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════════
def add_indicators(df, fast=20, slow=50):
    df = df.copy()
    df["ema_f"]   = ta.ema(df["close"], length=fast)
    df["ema_s"]   = ta.ema(df["close"], length=slow)
    df["ema200"]  = ta.ema(df["close"], length=200)
    df["atr"]     = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi"]     = ta.rsi(df["close"], length=14)
    df["vol_ma"]  = ta.sma(df["volume"], length=20)
    df["slope_f"] = df["ema_f"].pct_change(5) * 100
    df["slope200"]= df["ema200"].pct_change(5) * 100

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None and not adx_df.empty:
        acol = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        df["adx"] = adx_df[acol[0]] if acol else np.nan
    else:
        df["adx"] = np.nan
    return df


# ════════════════════════════════════════════════════════════════
#  REGIME ENGINE
#  Scores: EMA200 distance, EMA slope, RSI bias, ADX amplifier
#  → smoothed 5-state regime per bar
# ════════════════════════════════════════════════════════════════
def calc_regime(df):
    score = pd.Series(0.0, index=df.index)

    # Distance from EMA200 in ATR units (capped at ±3)
    atr_safe = df["atr"].replace(0, np.nan)
    dist_atr = (df["close"] - df["ema200"]) / atr_safe
    score += dist_atr.clip(-3, 3)

    # Fast EMA slope contribution
    score += df["slope_f"].clip(-2, 2) * 0.4

    # RSI bias (-1 to +1 range)
    rsi_bias = (df["rsi"] - 50) / 25
    score += rsi_bias.clip(-1, 1)

    # ADX amplifies in strong trends
    adx_amp = (df["adx"].fillna(20) / 25).clip(0.5, 2.0)
    score *= adx_amp

    # Discretise to 5 regimes
    regime = pd.cut(
        score,
        bins=[-np.inf, -2.5, -0.8, 0.8, 2.5, np.inf],
        labels=[R_STRONG_BEAR, R_WEAK_BEAR, R_NEUTRAL, R_WEAK_BULL, R_STRONG_BULL],
    ).astype(float).fillna(0).astype(int)

    # Smooth: require 3 consecutive matching bars before changing
    smooth = regime.copy()
    for i in range(2, len(regime)):
        if regime.iloc[i] == regime.iloc[i-1] == regime.iloc[i-2]:
            smooth.iloc[i] = regime.iloc[i]
        else:
            smooth.iloc[i] = smooth.iloc[i-1]
    return smooth


# ════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE v2
#
#  Entry types:
#   Cross  – EMA fast crosses EMA slow (primary signal)
#   Bounce – Price returns to EMA fast band in trending market
#             (re-entry = more trades without lowering quality)
#
#  Regime gate:
#   Strong Bull  → Longs ONLY (blocks all shorts)
#   Weak Bull    → Longs OK, Shorts only if ADX > 20
#   Neutral      → Both, but ADX > 20 required for either side
#   Weak Bear    → Shorts OK, Longs only if ADX > 20
#   Strong Bear  → Shorts ONLY (blocks all longs)
# ════════════════════════════════════════════════════════════════
def make_signals(df, use_rsi, use_vol, tf):
    df    = df.copy()
    mult  = ATR_MULT.get(tf, 2.0)
    pf, ps = df["ema_f"].shift(1), df["ema_s"].shift(1)

    cross_up   = (df["ema_f"] > df["ema_s"]) & (pf <= ps)
    cross_down = (df["ema_f"] < df["ema_s"]) & (pf >= ps)

    # Bounce: price within 0.3 ATR of EMA fast while trend intact
    band = 0.3 * df["atr"]
    near_ema = (df["close"] >= df["ema_f"] - band) & (df["close"] <= df["ema_f"] + band)

    bounce_bull = near_ema & (df["ema_f"] > df["ema_s"]) & (df["slope_f"] > 0)
    bounce_bear = near_ema & (df["ema_f"] < df["ema_s"]) & (df["slope_f"] < 0)

    # Filters
    rsi_l_ok = (df["rsi"] < 70) if use_rsi else pd.Series(True, index=df.index)
    rsi_s_ok = (df["rsi"] > 30) if use_rsi else pd.Series(True, index=df.index)
    vol_ok   = (df["volume"] > df["vol_ma"]) if use_vol else pd.Series(True, index=df.index)
    adx_ok   = df["adx"].fillna(0) > 20

    raw_long  = (cross_up   | bounce_bull) & rsi_l_ok & vol_ok
    raw_short = (cross_down | bounce_bear) & rsi_s_ok & vol_ok

    # Regime gates
    regime        = df["regime"]
    long_allowed  = (regime > R_STRONG_BEAR)  # not in strong bear
    short_allowed = (regime < R_STRONG_BULL)  # not in strong bull

    neutral = regime == R_NEUTRAL
    long_allowed  = long_allowed  & (~neutral | adx_ok)
    short_allowed = short_allowed & (~neutral | adx_ok)

    # In weak bear/bull: counter-trend needs ADX confirmation
    weak_bull = regime == R_WEAK_BULL
    weak_bear = regime == R_WEAK_BEAR
    short_allowed = short_allowed & (~weak_bull | adx_ok)
    long_allowed  = long_allowed  & (~weak_bear | adx_ok)

    df["signal"] = 0
    df.loc[raw_long  & long_allowed,  "signal"] =  1
    df.loc[raw_short & short_allowed, "signal"] = -1

    # Trailing stop display lines
    df["trail_long"]  = df["close"] - mult * df["atr"]
    df["trail_short"] = df["close"] + mult * df["atr"]

    # Annotate signal type
    df["sig_type"] = "—"
    df.loc[cross_up   & (df["signal"] ==  1), "sig_type"] = "Cross ↑"
    df.loc[bounce_bull& (df["signal"] ==  1), "sig_type"] = "Bounce ↑"
    df.loc[cross_down & (df["signal"] == -1), "sig_type"] = "Cross ↓"
    df.loc[bounce_bear& (df["signal"] == -1), "sig_type"] = "Bounce ↓"

    return df


# ════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════
def run_backtest(df_raw, tf, capital, risk_pct, use_rsi, use_vol, fast, slow):
    df = add_indicators(df_raw, fast, slow)
    df["regime"] = calc_regime(df)
    df = make_signals(df, use_rsi, use_vol, tf)
    df = df.dropna(subset=["ema_f","ema_s","atr","rsi"]).copy().reset_index()

    mult   = ATR_MULT.get(tf, 2.0)
    trades = []
    equity = [capital]
    cap    = capital
    peak   = capital
    pos    = 0
    ep     = 0.0
    sz     = 0.0
    ts_sl  = 0.0
    eidx   = 0
    sig_tp = "—"
    tsc    = "ts" if "ts" in df.columns else df.columns[0]

    def close_trade(i, xp, reason):
        nonlocal cap, pos
        pnl = ((xp - ep) if pos == 1 else (ep - xp)) * sz
        cap += pnl
        trades.append({
            "Entry Time":  df.at[eidx, tsc],
            "Exit Time":   df.at[i,    tsc],
            "Direction":   "🟢 Long" if pos == 1 else "🔴 Short",
            "Type":        sig_tp,
            "Entry Price": round(ep, 4),
            "Exit Price":  round(xp, 4),
            "Size":        round(sz, 6),
            "P&L $":       round(pnl, 2),
            "Exit Reason": reason,
            "Regime":      REGIME_LABEL.get(int(df.at[eidx, "regime"]), "?"),
        })
        pos = 0

    for i in range(1, len(df)):
        row   = df.iloc[i]
        atr_v = float(row["atr"]) if not np.isnan(row["atr"]) else float(row["close"])*0.01
        risk_amt = cap * (risk_pct / 100)

        if cap < peak * 0.95 and pos != 0:
            close_trade(i, float(row["close"]), "Max DD")
            equity.append(cap)
            continue
        peak = max(peak, cap)

        if pos == 1:
            new_ts = float(row["close"]) - mult * atr_v
            ts_sl  = max(ts_sl, new_ts)
            if float(row["low"]) <= ts_sl:
                close_trade(i, max(float(row["open"]), ts_sl), "Trail Stop")
        elif pos == -1:
            new_ts = float(row["close"]) + mult * atr_v
            ts_sl  = min(ts_sl, new_ts)
            if float(row["high"]) >= ts_sl:
                close_trade(i, min(float(row["open"]), ts_sl), "Trail Stop")

        sig = int(row["signal"])
        if sig != 0 and sig != pos:
            if pos != 0:
                close_trade(i, float(row["close"]), "Reversal")
            pos   = sig
            ep    = float(row["close"])
            sz    = risk_amt / (mult * atr_v) if atr_v > 0 else 0
            ts_sl = ep - mult*atr_v if pos == 1 else ep + mult*atr_v
            eidx  = i
            sig_tp = str(row["sig_type"])

        equity.append(cap)

    return trades, equity, df


# ════════════════════════════════════════════════════════════════
#  STATISTICS
# ════════════════════════════════════════════════════════════════
def calc_stats(trades, equity, capital):
    if not trades:
        return {}
    tdf  = pd.DataFrame(trades)
    wins = tdf[tdf["P&L $"] > 0]
    loss = tdf[tdf["P&L $"] <= 0]
    pfd  = abs(loss["P&L $"].sum())
    pf   = wins["P&L $"].sum() / pfd if pfd > 0 else float("inf")
    eq   = np.array(equity)
    rmax = np.maximum.accumulate(eq)
    dd   = (eq - rmax) / rmax * 100
    rets = pd.Series(equity).pct_change().dropna()
    sh   = (rets.mean() / rets.std() * np.sqrt(252*24)) if rets.std() > 0 else 0.0
    return {
        "Total Trades":  len(tdf),
        "Win Rate":      f"{len(wins)/len(tdf)*100:.1f}%",
        "Total P&L":     f"${tdf['P&L $'].sum():,.2f}",
        "Total Return":  f"{(equity[-1]/capital-1)*100:.2f}%",
        "Profit Factor": f"{pf:.2f}",
        "Max Drawdown":  f"{dd.min():.2f}%",
        "Sharpe Ratio":  f"{sh:.2f}",
        "Avg Win":       f"${wins['P&L $'].mean():.2f}" if len(wins) else "$0",
        "Avg Loss":      f"${loss['P&L $'].mean():.2f}" if len(loss) else "$0",
        "Final Capital": f"${equity[-1]:,.2f}",
        "Best Trade":    f"${tdf['P&L $'].max():.2f}",
        "Worst Trade":   f"${tdf['P&L $'].min():.2f}",
        "Avg Trade":     f"${tdf['P&L $'].mean():.2f}",
        "Long Trades":   int((tdf["Direction"].str.contains("Long")).sum()),
        "Short Trades":  int((tdf["Direction"].str.contains("Short")).sum()),
    }


# ════════════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════════════
_LAY = dict(
    template="plotly_dark",
    paper_bgcolor="#070b14",
    plot_bgcolor="#0c1526",
    font=dict(family="Fira Code, monospace", color="#4a6080", size=11),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.04, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

_REGIME_BG = {
    2:"rgba(0,255,159,0.04)",1:"rgba(100,255,218,0.03)",0:"rgba(0,0,0,0)",
    -1:"rgba(255,154,108,0.03)",-2:"rgba(255,61,90,0.04)",
}


def build_price_chart(df, trades, fast, slow):
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.52,0.16,0.16,0.16],
        shared_xaxes=True, vertical_spacing=0.012,
        subplot_titles=("","Volume","RSI (14)","ADX (14)"),
    )

    # Regime background bands on price panel
    if "regime" in df.columns:
        rg = df["regime"]
        prev_r, start = rg.iloc[0], df.index[0]
        for idx_v, r in rg.items():
            if r != prev_r:
                fig.add_vrect(x0=start, x1=idx_v,
                              fillcolor=_REGIME_BG.get(int(prev_r),"rgba(0,0,0,0)"),
                              layer="below", line_width=0, row=1, col=1)
                start, prev_r = idx_v, r
        fig.add_vrect(x0=start, x1=df.index[-1],
                      fillcolor=_REGIME_BG.get(int(prev_r),"rgba(0,0,0,0)"),
                      layer="below", line_width=0, row=1, col=1)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#00ff9f", increasing_fillcolor="#002e16",
        decreasing_line_color="#ff3d5a", decreasing_fillcolor="#2e000d",
        line_width=1,
    ), row=1, col=1)

    # EMAs
    for col, name, color, dash in [
        ("ema_f",  f"EMA {fast}", "#00e5ff","solid"),
        ("ema_s",  f"EMA {slow}", "#ffd166","solid"),
        ("ema200", "EMA 200",     "#bd93f9","dot"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.5, dash=dash), opacity=0.9,
            ), row=1, col=1)

    # Trailing stop line
    if "trail_long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["trail_long"], name="Trail Stop",
            line=dict(color="#ff3d5a", width=0.8, dash="dash"), opacity=0.4,
        ), row=1, col=1)

    # Entry markers
    for fn, sym, col, label in [
        (lambda t:"Long"  in t["Direction"], "triangle-up",   "#00ff9f","Long Entry"),
        (lambda t:"Short" in t["Direction"], "triangle-down", "#ff3d5a","Short Entry"),
    ]:
        sub = [t for t in trades if fn(t)]
        if sub:
            fig.add_trace(go.Scatter(
                x=[t["Entry Time"] for t in sub],
                y=[t["Entry Price"] for t in sub],
                mode="markers", name=label,
                marker=dict(symbol=sym, size=10, color=col,
                            line=dict(width=1, color="#fff")),
            ), row=1, col=1)

    # Volume
    vc = ["#002e16" if c >= o else "#2e000d"
          for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"],
                         name="Vol", marker_color=vc, opacity=0.8), row=2, col=1)
    if "vol_ma" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vol_ma"],
                                  line=dict(color="#ffd166",width=1.2), name="Vol MA"),
                      row=2, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
                                  line=dict(color="#00e5ff",width=1.4), name="RSI"),
                      row=3, col=1)
        for lvl, c in [(70,"#ff3d5a"),(50,"#4a6080"),(30,"#00ff9f")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=c,
                          line_width=0.8, opacity=0.5, row=3, col=1)

    # ADX
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["adx"],
                                  line=dict(color="#bd93f9",width=1.4), name="ADX"),
                      row=4, col=1)
        for lvl in [25, 20]:
            fig.add_hline(y=lvl, line_dash="dot", line_color="#ffd166",
                          line_width=0.7, opacity=0.5, row=4, col=1)

    fig.update_layout(**_LAY, height=760)
    fig.update_yaxes(gridcolor="#111e33", zerolinecolor="#111e33")
    fig.update_xaxes(gridcolor="#111e33")
    return fig


def build_equity_chart(equity, capital):
    eq   = np.array(equity)
    bars = list(range(len(eq)))
    rmax = np.maximum.accumulate(eq)
    dd   = (eq - rmax) / rmax * 100
    fig  = make_subplots(rows=2, cols=1, row_heights=[0.65,0.35],
                         shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Equity Curve ($)","Drawdown (%)"))
    fig.add_trace(go.Scatter(x=bars, y=eq, name="Equity",
                              line=dict(color="#00ff9f",width=2),
                              fill="tozeroy", fillcolor="rgba(0,255,159,0.07)"),
                  row=1, col=1)
    fig.add_hline(y=capital, line_dash="dash", line_color="#4a6080",
                  line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=bars, y=dd, name="Drawdown",
                              line=dict(color="#ff3d5a",width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,61,90,0.10)"),
                  row=2, col=1)
    fig.add_hline(y=-5, line_dash="dash", line_color="#ff3d5a",
                  line_width=0.8, opacity=0.5, row=2, col=1)
    fig.update_layout(**_LAY, height=440)
    fig.update_yaxes(gridcolor="#111e33", zerolinecolor="#111e33")
    return fig


def build_regime_chart(df):
    if "regime" not in df.columns:
        return None

    fig = go.Figure()

    # Coloured area fill per regime level
    for rv, rc in REGIME_COLOR.items():
        # semi-transparent horizontal band: fill between rv-0.5 and rv+0.5
        r_hex = rc.lstrip("#")
        r_int = tuple(int(r_hex[i:i+2], 16) for i in (0, 2, 4))
        fill_c = f"rgba({r_int[0]},{r_int[1]},{r_int[2]},0.08)"
        fig.add_hrect(
            y0=rv - 0.48, y1=rv + 0.48,
            fillcolor=fill_c, line_width=0,
            layer="below",
        )

    # Main regime line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["regime"].astype(float),
        mode="lines",
        line=dict(color="#cdd9f0", width=2),
        name="Regime",
    ))

    # Horizontal reference lines (no annotation kwargs — causes Plotly version conflicts)
    shapes = []
    for rv, rc in REGIME_COLOR.items():
        shapes.append(dict(
            type="line", xref="paper", yref="y",
            x0=0, x1=1, y0=rv, y1=rv,
            line=dict(color=rc, width=0.8, dash="dot"),
            opacity=0.7,
        ))

    # Label annotations added separately (avoids add_hline annotation bug)
    annotations = []
    for rv, label in REGIME_LABEL.items():
        annotations.append(dict(
            xref="paper", yref="y",
            x=1.01, y=rv,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=9, color=REGIME_COLOR[rv],
                      family="Fira Code, monospace"),
            xanchor="left",
        ))

    lay = {k: v for k, v in _LAY.items()
           if k not in ("xaxis_rangeslider_visible", "legend")}
    fig.update_layout(
        **lay,
        height=230,
        shapes=shapes,
        annotations=annotations,
        yaxis=dict(
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["S.Bear", "W.Bear", "Neutral", "W.Bull", "S.Bull"],
            gridcolor="#111e33",
            range=[-2.6, 2.6],
        ),
        showlegend=False,
        margin=dict(l=0, r=110, t=20, b=0),
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  HTML COMPONENTS
# ════════════════════════════════════════════════════════════════
def signal_box_html(sig, coin, tf, price, rsi, adx, regime_int,
                    trail_l, trail_s, sig_type=""):
    if sig == 1:
        cls, label, color = "sig-long",  "▲  LONG",       "#00ff9f"
        trail_disp = f"${trail_l:,.4f}"
    elif sig == -1:
        cls, label, color = "sig-short", "▼  SHORT",      "#ff3d5a"
        trail_disp = f"${trail_s:,.4f}"
    else:
        cls, label, color = "sig-neutral","◆  FLAT / WAIT","#4a6080"
        trail_disp = "—"
    rc = REGIME_COLOR.get(regime_int, "#ffd166")
    rl = REGIME_LABEL.get(regime_int, "?")
    st_tag = f"· {sig_type}" if sig_type and sig_type != "—" else ""
    return f"""
<div class="{cls}">
  <p class="sig-label" style="color:{color}">{label}</p>
  <p class="sig-sub">{coin} · {tf.upper()} · {datetime.utcnow().strftime("%H:%M:%S UTC")} {st_tag}</p>
  <div class="sig-grid">
    <div><div class="sig-cell-label">PRICE</div>
         <div class="sig-cell-val">${price:,.4f}</div></div>
    <div><div class="sig-cell-label">TRAIL STOP</div>
         <div class="sig-cell-val">{trail_disp}</div></div>
    <div><div class="sig-cell-label">RSI (14)</div>
         <div class="sig-cell-val">{rsi:.1f}</div></div>
    <div><div class="sig-cell-label">ADX (14)</div>
         <div class="sig-cell-val">{adx:.1f}</div></div>
    <div style="grid-column:1/-1">
      <div class="sig-cell-label">REGIME</div>
      <div class="sig-cell-val" style="color:{rc}">{rl}</div>
    </div>
  </div>
</div>"""


def regime_bar_html(regime_int):
    rc  = REGIME_COLOR.get(regime_int, "#ffd166")
    rl  = REGIME_LABEL.get(regime_int, "?")
    sub = {
        2: "Strong uptrend · Longs only · Shorts blocked",
        1: "Upside bias · Longs preferred · Shorts need ADX>20",
        0: "No clear direction · Both sides · ADX>20 required",
       -1: "Downside bias · Shorts preferred · Longs need ADX>20",
       -2: "Strong downtrend · Shorts only · Longs blocked",
    }.get(regime_int, "")
    return f"""
<div class="regime-bar" style="background:rgba(0,0,0,.3);border:1px solid {rc}44;border-left:4px solid {rc}">
  <div class="regime-dot" style="background:{rc};box-shadow:0 0 8px {rc}88"></div>
  <div>
    <div class="regime-name" style="color:{rc}">{rl}</div>
    <div class="regime-sub">{sub}</div>
  </div>
</div>"""


def score_bar_html(label, value_pct, color):
    return f"""
<div class="score-wrap">
  <div class="score-label">{label}</div>
  <div class="score-track">
    <div class="score-fill" style="width:{min(value_pct,100):.0f}%;background:{color}"></div>
  </div>
  <div class="score-val" style="color:{color}">{value_pct:.0f}<span style="font-size:10px;color:#4a6080">%</span></div>
</div>"""


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    sb = st.sidebar
    sb.markdown(
        '<p style="font-family:Orbitron,monospace;font-size:15px;color:#00e5ff;'
        'letter-spacing:3px;font-weight:900;margin-bottom:2px">⚡ EMA STRATEGY PRO</p>'
        '<p style="font-size:10px;color:#4a6080;letter-spacing:2px;margin:0">'
        'v2.0 · Regime-Aware · Multi-Source</p>',
        unsafe_allow_html=True,
    )
    sb.divider()
    sb.markdown('<div class="sec-title">MARKET</div>', unsafe_allow_html=True)
    coin = sb.selectbox("Symbol", COINS, index=0)
    tf   = sb.selectbox("Timeframe", TIMEFRAMES, index=4)
    sb.caption(f"📐 HTF filter → {HIGHER_TF.get(tf,'?').upper()}")

    sb.markdown('<div class="sec-title">EMA PARAMETERS</div>', unsafe_allow_html=True)
    fast = sb.slider("EMA Fast", 5, 50, 20, 1)
    slow = sb.slider("EMA Slow", 20, 200, 50, 1)
    if fast >= slow:
        sb.warning("⚠ Fast must be < Slow")

    sb.markdown('<div class="sec-title">RISK MANAGEMENT</div>', unsafe_allow_html=True)
    capital  = sb.number_input("Start Capital ($)", 500, 1_000_000, 10_000, 500)
    risk_pct = sb.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    sb.caption(f"Pos. size = capital×{risk_pct}% ÷ (ATR×{ATR_MULT.get(tf,2.0)})")

    sb.markdown('<div class="sec-title">SIGNAL FILTERS</div>', unsafe_allow_html=True)
    use_rsi = sb.checkbox("RSI filter  (L<70 · S>30)", value=True)
    use_vol = sb.checkbox("Volume filter  (cross > VolMA)", value=True)

    sb.markdown('<div class="sec-title">AUTO-REFRESH</div>', unsafe_allow_html=True)
    ref_label = sb.selectbox("Interval", list(REFRESH_MAP.keys()), index=2)
    ref_secs  = REFRESH_MAP[ref_label]

    sb.divider()
    sb.markdown(
        '<p style="font-size:10px;color:#4a6080;text-align:center;line-height:1.6">'
        '⚠ Educational & research only.<br>Not financial advice.</p>',
        unsafe_allow_html=True,
    )
    return coin, tf, fast, slow, capital, risk_pct, use_rsi, use_vol, ref_secs


# ════════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ════════════════════════════════════════════════════════════════
def handle_auto_refresh(ref_secs):
    if ref_secs == 0:
        return
    now = time.time()
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = now
    elapsed = now - st.session_state.last_refresh
    remaining = max(0, int(ref_secs - elapsed))
    st.sidebar.caption(f"🔄 Next refresh in **{remaining}s**")
    if elapsed >= ref_secs:
        st.session_state.last_refresh = now
        st.cache_data.clear()
    time.sleep(1)
    st.rerun()


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    coin, tf, fast, slow, capital, risk_pct, use_rsi, use_vol, ref_secs = render_sidebar()

    hcol1, hcol2, hcol3 = st.columns([3, 1.2, 1])
    _hph = hcol1.empty()
    with hcol2:
        st.metric("UTC Time", datetime.utcnow().strftime("%H:%M:%S"))
    with hcol3:
        if st.button("⟳  Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()
    st.divider()

    # Fetch
    status = st.empty()
    status.info("⏳ Fetching … Binance → Bybit → OKX → Gate.io → yFinance")
    df_main, src_main = fetch_ohlcv(coin, tf)
    htf               = HIGHER_TF.get(tf, "4h")
    df_htf,  _        = fetch_ohlcv(coin, htf, limit=500)
    status.empty()

    is_demo     = "DEMO" in src_main
    badge_color = "#ffd166" if is_demo else "#00ff9f"
    badge_icon  = "⚠ DEMO"  if is_demo else "✓ LIVE"
    st.sidebar.markdown(
        f'<div class="src-badge" style="border:1px solid {badge_color}44">'
        f'<span style="color:{badge_color};font-weight:700">{badge_icon}</span>'
        f' <span style="color:#4a6080">via</span>'
        f' <span style="color:#cdd9f0">{src_main}</span></div>',
        unsafe_allow_html=True,
    )
    if is_demo:
        st.warning(
            "⚠ **All live sources unreachable** — synthetic DEMO data. "
            "Strategy logic is functional but prices are not real.\n\n"
            "`pip install yfinance` for a live fallback."
        )

    if df_main is None or df_main.empty:
        st.error("⚠ No data available."); st.stop()

    # HTF EMA200
    htf_ema200_val = None
    if df_htf is not None and not df_htf.empty:
        d = add_indicators(df_htf, fast, slow)
        if d["ema200"].notna().any():
            htf_ema200_val = float(d["ema200"].iloc[-1])

    # Process main
    df = add_indicators(df_main, fast, slow)
    df["regime"] = calc_regime(df)
    df = make_signals(df, use_rsi, use_vol, tf)

    last        = df.iloc[-1]
    curr_regime = int(last["regime"]) if not np.isnan(last.get("regime", 0)) else 0
    curr_price  = float(last["close"])
    curr_rsi    = float(last["rsi"])  if not np.isnan(last["rsi"])   else 50.0
    curr_adx    = float(last["adx"])  if not np.isnan(last.get("adx",np.nan)) else 0.0
    curr_tl     = float(last["trail_long"])  if not np.isnan(last["trail_long"])  else curr_price
    curr_ts_    = float(last["trail_short"]) if not np.isnan(last["trail_short"]) else curr_price

    rs          = df[df["signal"] != 0]
    curr_sig    = int(rs["signal"].iloc[-1])   if not rs.empty else 0
    curr_sigtype= str(rs["sig_type"].iloc[-1]) if not rs.empty else "—"

    bull_pct = (curr_regime + 2) / 4 * 100
    rsi_pct  = float(curr_rsi)
    adx_pct  = min(float(curr_adx) / 50 * 100, 100)
    rc_col   = REGIME_COLOR.get(curr_regime, "#ffd166")

    if htf_ema200_val:
        htf_bias = (
            f"BULLISH (Price > {htf} EMA200 @ ${htf_ema200_val:,.2f})"
            if curr_price > htf_ema200_val
            else f"BEARISH (Price < {htf} EMA200 @ ${htf_ema200_val:,.2f})"
        )
    else:
        htf_bias = "HTF EMA200 unavailable"

    # Backtest
    trades, equity, df_bt = run_backtest(
        df_main, tf, capital, risk_pct, use_rsi, use_vol, fast, slow
    )
    stats = calc_stats(trades, equity, capital)

    _hph.markdown(
        f'<h1 style="font-family:Orbitron,monospace;font-size:20px;color:#00e5ff;'
        f'letter-spacing:4px;margin:0;padding:0">⚡ EMA STRATEGY PRO</h1>'
        f'<p style="font-size:11px;color:#4a6080;letter-spacing:2px;margin:4px 0 0 0">'
        f'{coin} · {tf.upper()} · HTF: {htf.upper()} · SOURCE: {src_main}</p>',
        unsafe_allow_html=True,
    )

    # ── TABS
    tab_live, tab_regime, tab_bt_tab, tab_log = st.tabs([
        "📡  LIVE SIGNAL", "🧭  REGIME", "📊  BACKTEST", "📋  TRADE LOG",
    ])

    # ── Tab 1: Live Signal
    with tab_live:
        sc, cc = st.columns([1, 3.2])
        with sc:
            st.markdown(
                signal_box_html(curr_sig, coin, tf, curr_price, curr_rsi,
                                curr_adx, curr_regime, curr_tl, curr_ts_,
                                curr_sigtype),
                unsafe_allow_html=True,
            )
            st.markdown(regime_bar_html(curr_regime), unsafe_allow_html=True)
            st.markdown(score_bar_html("BULL / BEAR SCORE", bull_pct, rc_col), unsafe_allow_html=True)
            st.markdown(score_bar_html("RSI (14)", rsi_pct, "#00e5ff"), unsafe_allow_html=True)
            st.markdown(score_bar_html("ADX STRENGTH", adx_pct, "#bd93f9"), unsafe_allow_html=True)

            st.markdown('<div class="sec-title">INDICATOR VALUES</div>', unsafe_allow_html=True)
            for k, v in {
                f"EMA {fast}":  f"${last['ema_f']:,.2f}"  if not np.isnan(last['ema_f'])  else "N/A",
                f"EMA {slow}":  f"${last['ema_s']:,.2f}"  if not np.isnan(last['ema_s'])  else "N/A",
                "EMA 200":      f"${last['ema200']:,.2f}"  if not np.isnan(last['ema200']) else "N/A",
                "HTF EMA200":   f"${htf_ema200_val:,.2f}"  if htf_ema200_val else "N/A",
                "ATR (14)":     f"${last['atr']:,.4f}"     if not np.isnan(last['atr'])    else "N/A",
                "EMA Slope":    f"{last['slope_f']:.3f}%"  if not np.isnan(last['slope_f']) else "N/A",
                "Volume":       f"{last['volume']:,.0f}",
                "Vol MA (20)":  f"{last['vol_ma']:,.0f}"   if not np.isnan(last['vol_ma']) else "N/A",
                "Trail Long":   f"${curr_tl:,.4f}",
                "Trail Short":  f"${curr_ts_:,.4f}",
                "ATR Mult.":    f"×{ATR_MULT.get(tf,2.0)}",
                "HTF Bias":     htf_bias,
                "Total Trades": f"{stats.get('Total Trades','—')}",
                "Win Rate":     f"{stats.get('Win Rate','—')}",
            }.items():
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span class="s-label">{k}</span>'
                    f'<span class="s-val">{v}</span></div>',
                    unsafe_allow_html=True,
                )
        with cc:
            st.plotly_chart(
                build_price_chart(df, trades, fast, slow),
                use_container_width=True,
                config={"displayModeBar":True,"scrollZoom":True},
            )

    # ── Tab 2: Regime
    with tab_regime:
        rc1, rc2 = st.columns([1.5, 2.5])
        with rc1:
            st.markdown('<div class="sec-title">CURRENT REGIME</div>', unsafe_allow_html=True)
            st.markdown(regime_bar_html(curr_regime), unsafe_allow_html=True)

            st.markdown('<div class="sec-title">REGIME RULES</div>', unsafe_allow_html=True)
            for name, col, rule in [
                ("Strong Bull","#00ff9f","Longs only · Cross + Bounce entries"),
                ("Weak Bull",  "#64ffda","Longs preferred · Shorts need ADX>20"),
                ("Neutral",    "#ffd166","Both sides · ADX>20 + Vol required"),
                ("Weak Bear",  "#ff9a6c","Shorts preferred · Longs need ADX>20"),
                ("Strong Bear","#ff3d5a","Shorts only · Cross + Bounce entries"),
            ]:
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span style="color:{col};font-weight:600;font-size:11px">{name}</span>'
                    f'<span class="s-label" style="font-size:10px">{rule}</span></div>',
                    unsafe_allow_html=True,
                )

            if trades:
                st.markdown('<div class="sec-title">P&L BY REGIME</div>', unsafe_allow_html=True)
                tdf = pd.DataFrame(trades)
                rg_g = tdf.groupby("Regime")["P&L $"].agg(["count","sum"]).reset_index()
                for _, row_ in rg_g.iterrows():
                    pc = "#00ff9f" if row_["sum"] >= 0 else "#ff3d5a"
                    st.markdown(
                        f'<div class="stat-row">'
                        f'<span class="s-label">{row_["Regime"]}</span>'
                        f'<span class="s-val">{int(row_["count"])} trades · '
                        f'<span style="color:{pc}">${row_["sum"]:,.2f}</span></span></div>',
                        unsafe_allow_html=True,
                    )

        with rc2:
            st.markdown('<div class="sec-title">REGIME TIMELINE</div>', unsafe_allow_html=True)
            rf = build_regime_chart(df)
            if rf:
                st.plotly_chart(rf, use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="sec-title">TIME IN EACH REGIME</div>', unsafe_allow_html=True)
            if "regime" in df.columns:
                rp = df["regime"].value_counts(normalize=True).sort_index() * 100
                fig_rg = go.Figure(go.Bar(
                    y=[REGIME_LABEL.get(int(k),"?") for k in rp.index],
                    x=rp.values, orientation="h",
                    marker_color=[REGIME_COLOR.get(int(k),"#fff") for k in rp.index],
                    text=[f"{v:.1f}%" for v in rp.values], textposition="outside",
                ))
                fig_rg.update_layout(**_LAY, height=220,
                                     xaxis_title="% of bars",
                                     margin=dict(l=0,r=50,t=10,b=0))
                st.plotly_chart(fig_rg, use_container_width=True, config={"displayModeBar":False})

    # ── Tab 3: Backtest
    with tab_bt_tab:
        if stats:
            st.markdown('<div class="sec-title">PERFORMANCE SUMMARY</div>', unsafe_allow_html=True)
            keys = ["Total Trades","Win Rate","Total P&L","Total Return","Profit Factor",
                    "Max Drawdown","Sharpe Ratio","Avg Win","Avg Loss","Final Capital"]
            cols = st.columns(5)
            for i, k in enumerate(keys):
                with cols[i % 5]:
                    st.metric(k, stats.get(k,"—"))

            st.markdown('<div class="sec-title">LONG / SHORT / BEST / WORST</div>', unsafe_allow_html=True)
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Long Trades",  stats.get("Long Trades","—"))
            b2.metric("Short Trades", stats.get("Short Trades","—"))
            b3.metric("Best Trade",   stats.get("Best Trade","—"))
            b4.metric("Worst Trade",  stats.get("Worst Trade","—"))

            st.plotly_chart(build_equity_chart(equity, capital),
                            use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="sec-title">P&L DISTRIBUTION</div>', unsafe_allow_html=True)
            tdf = pd.DataFrame(trades)
            fd = go.Figure(go.Histogram(
                x=tdf["P&L $"], nbinsx=50,
                marker_color=np.where(tdf["P&L $"] >= 0,"#00ff9f","#ff3d5a"),
                opacity=0.8,
            ))
            fd.update_layout(**_LAY, height=240,
                             xaxis_title="P&L ($)", yaxis_title="Count",
                             yaxis_gridcolor="#111e33")
            st.plotly_chart(fd, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info(
                "No trades generated with current settings.\n\n"
                "Tips: disable RSI/Volume filters, or use a lower timeframe "
                "(5m/15m) to see more bounce + cross signals."
            )

    # ── Tab 4: Trade Log
    with tab_log:
        if trades:
            tdf     = pd.DataFrame(trades)
            tot_pnl = tdf["P&L $"].sum()
            wins    = (tdf["P&L $"] > 0).sum()
            cpnl    = "#00ff9f" if tot_pnl >= 0 else "#ff3d5a"
            st.markdown(
                f'<p style="font-size:13px;margin-bottom:8px">'
                f'<b>{len(tdf)}</b> trades · <b>{wins}</b> wins · '
                f'Total P&L: <b style="color:{cpnl}">${tot_pnl:,.2f}</b></p>',
                unsafe_allow_html=True,
            )
            fc1, fc2, fc3 = st.columns(3)
            dir_f = fc1.selectbox("Direction", ["All","Long","Short"])
            reg_f = fc2.selectbox("Regime",    ["All"] + list(REGIME_LABEL.values()))
            typ_f = fc3.selectbox("Type",      ["All","Cross ↑","Cross ↓","Bounce ↑","Bounce ↓"])

            view = tdf.copy()
            if dir_f != "All": view = view[view["Direction"].str.contains(dir_f)]
            if reg_f != "All": view = view[view["Regime"] == reg_f]
            if typ_f != "All": view = view[view["Type"] == typ_f]

            def _c(val):
                if isinstance(val, (int, float)):
                    return "color:#00ff9f" if val > 0 else "color:#ff3d5a"
                return ""
            try:
                styled = view.style.map(_c, subset=["P&L $"])
            except AttributeError:
                styled = view.style.applymap(_c, subset=["P&L $"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No trades to display.")

    handle_auto_refresh(ref_secs)


if __name__ == "__main__":
    main()
