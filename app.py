"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Trend & Volatility Screening with ATR Dynamic Buffers, ATR% Filters, RVOL Volume Confirmation,
             Directional Volume Labels, Flyover Tooltips, Parameter Tuning Guidance, Dynamic Preset Profiles,
             and Full URL Parameter Persistence.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="200 Day Triggers",
    page_icon="📈",
    layout="wide"
)

REPO_NAME = "200 Day Triggers"

# --- Strategy Preset Profiles (Derived from Parameter Tuning Guide) ---
PRESET_PROFILES = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DIA, EMXC, VEA",
        "buffer": 1.5,
        "rvol": 1.00,
        "min_atr": 0.75
    },
    "Bond / Fixed Income ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,  # Disable/lower RVOL requirement for credit roll liquidity
        "min_atr": 0.00   # Disable ATR gate for ultra-low volatility
    },
    "Covered Call ETFs": {
        "tickers": "JEPQ, JEPI, GPIQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.80,
        "min_atr": 0.50
    },
    "Dividend & Value ETFs": {
        "tickers": "SPHD, USMV, SCHD, VIG, VYM",
        "buffer": 1.5,
        "rvol": 1.25,
        "min_atr": 0.50
    },
    "Speculative & High Volatility": {
        "tickers": "FBTC, IAUM, NVDA, TSLA, TQQQ",
        "buffer": 3.5,
        "rvol": 1.50,
        "min_atr": 2.00
    },
    "Custom / Manual Override": None
}

def calculate_atr(df, window=14):
    """Calculates the Average True Range (ATR) to measure asset volatility."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def analyze_enhanced_sma_strategy(tickers, lookback_period="2y", slope_window=5, default_buffer_pct=0.015, rvol_threshold=1.00):
    """Evaluates tickers across Macro (200 EMA), Short-Term (20 EMA / 50 SMA) trends, ATR% metrics, and RVOL volume logic."""
    results = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=lookback_period, auto_adjust=True, progress=False)
            
            if df.empty or len(df) < 200 + slope_window:
                st.warning(f"Skipping {ticker}: Insufficient historical data (requires >205 trading days).")
                continue
            
            # Extract single-column Series from yfinance data
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
                
            high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
            low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            
            # Clean OHLC DataFrame for ATR
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
