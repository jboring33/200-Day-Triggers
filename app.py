"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term Trend Screening.
Features: Interactive Preset Selection, Custom Ticker Overrides, Top Run Button, 
          Dynamic URL Sync for Bookmarks, and Status vs Signal Methodology Documentation.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="200 Day Triggers", page_icon="📈", layout="wide")

REPO_NAME = "200 Day Triggers"

PRESET_CONFIGS = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.75,
        "rationale": (
            "Broad indices exhibit moderate volatility and high liquidity. A 2.0% buffer "
            "filters benchmark noise, RVOL of 0.80 allows steady index tracking, "
            "and 0.75% ATR screens out stagnant conditions."
        ),
    },
    "Mag7": {
        "tickers": "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA",
        "buffer": 2.5,
        "rvol": 1.00,
        "min_atr": 1.00,
        "rationale": (
            "High-beta mega-caps require wider noise filters for intraday swings (2.5%). "
            "An RVOL of 1.00 ensures volume-backed breakouts, while a 1.00% ATR "
            "floor filters low-range consolidation."
        ),
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50,
        "rationale": (
            "Sectors vary in beta (e.g., XLE vs XLU). A 2.0% buffer balances ranges, "
            "an RVOL of 0.85 confirms sector rotation, and a 0.50% ATR floor "
            "accounts for defensive ETFs."
        ),
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00,
        "rationale": (
            "Fixed-income ETFs move in tight price bands. A tight 1.0% buffer prevents "
            "false regime changes, an RVOL of 0.10 prevents false flags, and ATR "
            "of 0.00% retains low-volatility income assets."
        ),
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
        "rationale": (
            "Option-overlay ETFs experience dampened upside volatility. A 1.5% buffer "
            "captures trend shifts without over-filtering capped moves, paired "
            "with lower RVOL (0.75) and ATR (0.25%)."
        ),
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50,
        "rationale": (
            "High-dividend funds exhibit lower beta than growth assets. A 2.0% buffer "
            "absorbs equity fluctuation, while 0.75 RVOL and 0.50% ATR capture "
            "steady dividend accumulation."
        ),
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00,
        "rationale": (
            "Crypto and commodities experience intense volatility spikes. A 4.0% buffer "
            "prevents premature triggers, RVOL of 1.25 demands heavy volume conviction, "
            "and 2.00% ATR filters out dead-zone chop."
        ),
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": (
            "Neutral baseline settings for custom watchlists. Standard 2.0% buffer, "
            "1.00 RVOL threshold, and no minimum ATR restriction."
        ),
    },
}

# --- Indicator Calculations ---

def calculate_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss)))


def calculate_adx(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(window=window).mean()


def calculate_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(window=k_window).min()
    high_max = df["High"].rolling(window=k_window).max()
    k_pct = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    d_pct = k_pct.rolling(window=d_window).mean()
    return k_pct, d_pct


def get_bollinger_status(close, p_curr):
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
    lower_bb = float((sma20 - (std20 * 2)).iloc[-1])

    bw_series = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
    bw_curr = float(bw_series.iloc[-1]) * 100
    bw_avg = float(bw_series.rolling(50).mean().iloc[-1]) * 100
    bw_min = float(bw_series.rolling(50).min().iloc[-1]) * 100

    is_squeeze = bw_curr <= (bw_min * 1.15)
    is_expanding = bw_curr > (bw_avg * 1.20)

    if is_squeeze:
        return f"SQUEEZE ({bw_curr:.1f}%) - Compression before expansion", is_squeeze
    if is_expanding:
        return f"EXPANDING VOLATILITY ({bw_curr:.1f}% vs avg {bw_avg:.1f}%)", False
    if p_curr >= upper_bb:
        return f"UPPER BAND STRETCH (${upper_bb:.2f}) - Extended", False
    if p_curr <= lower_bb:
        return f"LOWER BAND STRETCH (${lower_bb:.2f}) - Oversold", False
    return f"NORMAL RANGE ({bw_curr:.1f}% width) - Mid: ${sma20.iloc[-1]:.2f}", False


def evaluate_signal_and_status(
    p_curr, upper_th, lower_th, slope, adx, rsi, stoch, 
    obv_slope, is_squeeze, p_ema20, p_sma50, rvol_confirmed, rvol, eff_buf
):
    is_above = p_curr > upper_th
    is_below = p_curr < lower_th

    if is_above and slope in ["UP", "FLAT"]:
        if adx < 20:
            return "⚪ Consolidating", "NO TREND", f"ADX ({adx:.1f}) < 20 indicates non-trending state."
        if rsi > 70 or stoch > 80:
            return "🟡 Overbought", "HOLD", f"RSI ({rsi:.1f}) / Stoch ({stoch:.1f}) indicate extension."
        if obv_slope < 0 and rsi < 50:
            return "🔴 Divergence", "WARNING", "Negative OBV volume outflow signals distribution."
        if is_squeeze:
            return "🟡 Squeeze", "WATCH", "Bollinger Band Squeeze active."
        if p_curr > p_ema20 and p_curr > p_sma50 and rvol_confirmed:
            return "🟢 Bullish", "ACCUMULATE", f"Confirmed breakout above buffer (RVOL: {rvol:.2f}x)."
        if p_curr <= p_ema20 or p_curr <= p_sma50:
            return "🟡 Pullback", "WAIT", "Macro trend is UP, but price pulling back below short MAs."
        return "🟡 Warning", "LOW VOLUME", f"Price above buffer, but RVOL ({rvol:.2f}x) lacks confirmation."

    if is_below and slope == "DOWN":
        if adx < 20:
            return "⚪ Consolidating", "WEAK BEAR", f"Price below 200 EMA, but weak ADX ({adx:.1f})."
        if rvol_confirmed:
            return "🔴 Bearish", "SELL", f"Selling volume confirmed below buffer (RVOL: {rvol:.2f}x)."
        return "🔴 Bearish", "SELL LOW VOL", f"Price below buffer with DOWN slope (RVOL: {rvol:.2f}x)."

    if lower_th <= p_curr <= upper_th:
        return "⚪ Neutral", "BUFFER ZONE", f"Price within +/-{eff_buf*100:.1f}% noise buffer of 200 EMA."

    return "🟡 Warning", "CAUTION", f"Price crossed EMA but slope ({slope}) does not confirm."


def analyze_enhanced_sma_strategy(tickers, default_buffer_pct=0.020, rvol_threshold=0.80):
    results = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 205:
                continue

            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            open_p = df["Open"].squeeze()
            volume = df["Volume"].squeeze()

            df_clean = pd.DataFrame({"High": high, "Low": low, "Close": close, "Open": open_p})
            p_curr = float(close.iloc[-1])
            p_open = float(open_p.iloc[-1])
            v_curr = float(volume.iloc[-1])

            vol_20_sma = float(volume.rolling(20).mean().iloc[-1])
            rvol = v_curr / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold

            is_green = p_curr >= p_open
            if rvol >= 1.25 and is_green:
                vol_status = "BULLISH (HEAVY BUYING)"
            elif rvol >= 1.25 and not is_green:
                vol_status = "BEARISH (HEAVY SELLING)"
            else:
                vol_status = "NORMAL / MODERATE"

            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(50).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()

            p_ema20 = float(ema20.iloc[-1])
            p_sma50 = float(sma50.iloc[-1])
            p_ema200 = float(ema200.iloc[-1])
            p_prev_ema200 = float(ema200.iloc[-6])

            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / p_curr) * 100
            eff_buf = max(default_buffer_pct, (atr / p_curr) * 1.5)

            upper_th = p_ema200 * (1.0 + eff_buf)
            lower_th = p_ema200 * (1.0 - eff_buf)

            rsi = float(calculate_rsi(close).iloc[-1])
            adx = float(calculate_adx(df_clean).
