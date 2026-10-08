"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term Trend & Volatility Screening.
             Enhanced with ADX Trend Strength, RSI Momentum Filters, OBV Volume Flow Slopes,
             Bollinger Band Compression, Dynamic ATR Buffers, Preset Instance Routing, and URL Persistence.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="200 Day Triggers",
    page_icon="📈",
    layout="wide"
)

REPO_NAME = "200 Day Triggers"

# --- Preset Strategy Configurations (Optimized for Weeks/Months Holding Horizons) ---
PRESET_CONFIGS = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.75
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00
    }
}

# --- Technical Indicator Helper Functions ---

def calculate_atr(df, window=14):
    """Calculates Average True Range (ATR)."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_rsi(close, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, window=14):
    """Calculates Average Directional Index (ADX) to measure trend strength."""
    high, low, close = df['High'], df['Low'], df['Close']
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_smooth = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(window=window).mean()

def calculate_obv_slope(close, volume, window=10):
    """Calculates 10-period On-Balance Volume (OBV) trend slope to track institutional volume flow."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.diff(window)

def analyze_enhanced_sma_strategy(tickers, interval="1d", slope_window=5, default_buffer_pct=0.020, rvol_threshold=0.80):
    """Evaluates tickers across Macro Trends, ADX strength, RSI momentum, OBV volume flow, and BB compression."""
    results = []
    
    lookback_period = "10y" if interval == "1wk" else "2y"
    timeframe_label = "Weekly" if interval == "1wk" else "Daily"
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=lookback_period, interval=interval, auto_adjust=True, progress=False)
            
            if df.empty or len(df) < 200 + slope_window:
                st.warning(f"Skipping {ticker}: Insufficient historical data for {timeframe_label} timeframe (requires >205 bars).")
                continue
            
            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
            high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
            low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
            
            curr_price = float(close.iloc[-1])
            curr_volume = float(volume.iloc[-1])
            
            # 1. RVOL Calculation
            vol_20_sma = float(volume.rolling(window=20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold
            
            # 2. Moving Average Calculations
            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(window=50).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            sma200 = close.rolling(window=200).mean()
            
            curr_ema20 = float(ema20.iloc[-1])
            curr_sma50 = float(sma50.iloc[-1])
            curr_ema200 = float(ema200.iloc[-1])
            curr_sma200 = float(sma200.iloc[-1])
            prev_ema200 = float(ema200.iloc[-(slope_window + 1)])
            
            # 3. Volatility Metrics (ATR & Dynamic Buffer)
            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / curr_price) * 100
            effective_buffer_pct = max(default_buffer_pct, (atr / curr_price) * 1.5)
            
            upper_threshold = curr_ema200 * (1 + effective_buffer_pct)
            lower_threshold = curr_ema200 * (1 - effective_buffer_pct)
            
            # 4. Multi-Indicator Technical Calculations
            rsi = float(calculate_rsi(close, 14).iloc[-1])
            adx = float(calculate_adx(df_clean, 14).iloc[-1])
            obv_slope = float(calculate_obv_slope(close, volume, 10).iloc[-1])
            
            # Bollinger Bands & Squeeze Metrics
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
            lower_bb = float((sma20 - (std20 * 2)).iloc[-1])
            bb_width = ((upper_bb - lower_bb) / float(sma20.iloc[-1])) * 100
            
            # 20-period historical minimum band width to detect Squeeze
            hist_bb_width = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
            min_bb_width = float(hist_bb_width.rolling(window=50).min().iloc[-1]) * 100
            is_squeeze = bb_width <= (min_bb_width * 1.15)
            
            # Slope Calculation
            ema_diff_pct = ((curr_ema200 - prev_ema200) / prev_ema200
