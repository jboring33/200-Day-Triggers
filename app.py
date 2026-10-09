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
            "Broad indices exhibit moderate volatility and high liquidity. A 2.0% buffer filters out benchmark noise, "
            "an RVOL of 0.80 allows for steady index accumulation tracking, and a 0.75% ATR minimum screens out stagnant conditions."
        ),
    },
    "Mag7": {
        "tickers": "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA",
        "buffer": 2.5,
        "rvol": 1.00,
        "min_atr": 1.00,
        "rationale": (
            "High-beta mega-cap tech stocks require wider noise filters to accommodate intraday price swings (2.5%). "
            "An RVOL threshold of 1.00 ensures breakouts are backed by institutional volume, while a 1.00% ATR floor filters low-range consolidation."
        ),
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50,
        "rationale": (
            "Sectors vary in beta (e.g., XLE vs XLU). A 2.0% buffer balances defensive and cyclical ranges, "
            "an RVOL of 0.85 confirms sector rotation, and a 0.50% ATR floor accounts for lower-volatility defensive sector ETFs."
        ),
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00,
        "rationale": (
            "Fixed-income and ultra-short ETFs move in tight price bands with lower volume variance. A tight 1.0% buffer prevents "
            "false macro regime changes, an RVOL of 0.10 prevents false non-volume flags, and an ATR of 0.00% ensures low-volatility income assets aren't filtered out."
        ),
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
        "rationale": (
            "Option-overlay ETFs capped by call sales experience dampened upside volatility. A 1.5% buffer captures trend shifts without "
            "over-filtering option-capped moves, paired with lower RVOL (0.75) and ATR (0.25%) requirements suited for yield strategies."
        ),
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50,
        "rationale": (
            "Value and high-dividend funds exhibit lower beta than growth assets. A 2.0% buffer absorbs standard equity fluctuation, "
            "while 0.75 RVOL and 0.50% ATR capture steady dividend accumulation without demanding high-growth momentum spikes."
        ),
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00,
        "rationale": (
            "Crypto ETFs and precious metals experience intense volatility spikes and false breakouts. A wide 4.0% buffer prevents premature triggers, "
            "a high RVOL threshold (1.25) demands heavy volume conviction, and a 2.00% ATR floor filters out dead-zone chop."
        ),
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": (
            "Neutral baseline settings tailored for user customization. Standard 2.0% buffer, average 1.00 RVOL threshold, and no minimum ATR restriction."
        ),
    },
}


def calculate_atr(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss)))


def calculate_adx(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    tr_smooth = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(window=window).mean()


def calculate_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(window=k_window).min()
    high_max = df["High"].rolling(window=k_window).max()
    k_percent = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent


def calculate_obv_slope(close, volume, window=10):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.diff(window)


def analyze_enhanced_sma_strategy(
    tickers,
    interval="1d",
    slope_window=5,
    default_buffer_pct=0.020,
    rvol_threshold=0.80,
):
    results = []
    lookback = "10y" if interval == "1wk" else "2y"

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=lookback,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df.empty or len(df) < 205:
                continue

            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            open_p = df["Open"].squeeze()
            volume = df["Volume"].squeeze()

            df_clean = pd.DataFrame(
                {"High": high, "Low": low, "Close": close, "Open": open_p}
            )
            curr_price = float(close.iloc[-1])
            curr_open = float(open_p.iloc[-1])
            curr_volume = float(volume.iloc[-1])

            vol_20_sma = float(volume.rolling(20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold

            is_green_day = curr_price >= curr_open
            if rvol >= 1.25 and is_green_day:
                volume_status = "BULLISH (HEAVY BUYING)"
            elif rvol >= 1.25 and not is_green_day:
                volume_status = "BEARISH (HEAVY SELLING)"
            else:
                volume_status = "NORMAL / MODERATE"

            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(50).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()

            curr_ema20 = float(ema20.iloc[-1])
            curr_sma50 = float(sma50.iloc[-1])
            curr_ema200 = float(ema200.iloc[-1])
            prev_ema200 = float(ema200.iloc[-(slope_window + 1)])

            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / curr_price) * 100
            effective_buffer = max(default_buffer_pct, (atr / curr_price) * 1.5)

            upper_th = curr_ema200 * (1.0 + effective_buffer)
            lower_th = curr_ema200 * (1.0 - effective_buffer)

            rsi = float(calculate_rsi(close).iloc[-1])
            adx = float(calculate_adx(df_clean).iloc[-1])
            k_stoch, d_stoch = calculate_stochastic(df_clean)
            curr_stoch = float(k_stoch.iloc[-1])

            obv_slope = float(calculate_obv_slope(close, volume).iloc[-1])
            obv_label = "BULLISH (INFLOW)" if obv_slope > 0 else "BEARISH (OUTFLOW)"

            # --- Bollinger Band Volatility Evaluation ---
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
            lower_bb = float((sma20 - (std20 * 2)).iloc[-1])

            bb_width_series = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
            curr_bb_width = float(bb_width_series.iloc[-1]) * 100
            avg_50_bb_width = float(bb_width_series.rolling(50).mean().iloc[-1]) * 100
            min_50_bb_width = float(bb_width_series.rolling(50).min().iloc
