"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term Trend & Volatility Screening.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="200 Day Triggers", page_icon="📈", layout="wide")

REPO_NAME = "200 Day Triggers"

# --- Preset Strategy Configurations ---
PRESET_CONFIGS = {
    "Broad Market": {"tickers": "SPY, QQQ, DJIA, EMXC, VEA", "buffer": 2.0, "rvol": 0.80, "min_atr": 0.75},
    "Sector ETFs": {"tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT", "buffer": 2.0, "rvol": 0.85, "min_atr": 0.50},
    "Bond ETFs": {"tickers": "SGOV, JPST, JAAA, JBBB, SCYB", "buffer": 1.0, "rvol": 0.10, "min_atr": 0.00},
    "Covered Call ETFs": {"tickers": "JEPI, JEPQ, XYLD, QYLD", "buffer": 1.5, "rvol": 0.75, "min_atr": 0.25},
    "Dividend ETFs": {"tickers": "USMV, SPHD, SCHD, VYM", "buffer": 2.0, "rvol": 0.75, "min_atr": 0.50},
    "Speculative & Commodities": {"tickers": "FBTC, IBIT, IAUM, GLD, SLV", "buffer": 4.0, "rvol": 1.25, "min_atr": 2.00},
    "Custom": {"tickers": "SPY, QQQ, NVDA, EMXC, VGT", "buffer": 2.0, "rvol": 1.00, "min_atr": 0.00}
}

# --- Technical Indicator Helpers ---

def calculate_atr(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_adx(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close']
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(window=window).mean()

def calculate_obv_slope(close, volume, window=10):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.diff(window)

def analyze_enhanced_sma_strategy(tickers, interval="1d", slope_window=5, default_buffer_pct=0.020, rvol_threshold=0.80):
    results = []
    lookback = "10y" if interval == "1wk" else "2y"
    timeframe_label = "Weekly" if interval == "1wk" else "Daily"
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=lookback, interval=interval, auto_adjust=True, progress=False)
            if df.empty or len(df) < 205:
                continue
            
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
            
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
            curr_price = float(close.iloc[-1])
            curr_volume = float(volume.iloc[-1])
            
            vol_20_sma = float(volume.rolling(20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold
            
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
            
            upper_th = curr_ema200 * (1 + effective_buffer)
            lower_th = curr_ema200 * (1 - effective_buffer)
            
            rsi = float(calculate_rsi(close).iloc[-1])
            adx = float(calculate_adx(df_clean).iloc[-1])
            obv_slope = float(calculate_obv_slope(close, volume).iloc[-1])
            
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
            lower_bb = float((sma20 - (std20 * 2)).iloc[-1])
            bb_width = ((upper_bb - lower_bb) / float(sma20.iloc[-1])) * 100
            
            hist_bb = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
            min_bb = float(hist_bb.rolling(50).min().iloc[-1]) * 100
            is_squeeze = bb_width <= (min_bb * 1.15)
            
            ema_diff = ((curr_ema200 - prev_ema200) / prev_ema200) * 100
            slope = "UP" if ema_diff > 0.05 else ("DOWN" if ema_diff < -0.05 else "FLAT")
            
            # Single-line decision blocks to prevent syntax breaks
            if curr_price > upper_th and slope in ["UP", "FLAT"]:
                if adx < 20:
                    status_short, full_signal, reason = "⚪ Consolidating", "CONSOLIDATION / NO TREND", f"Price >200 EMA, but weak ADX ({adx:.1f}) indicates sideways movement."
                elif rsi > 70:
                    status_short, full_signal, reason = "🟡 Overbought", "HOLD / DO NOT ADD", f"Overbought condition with RSI ({rsi:.1f}) > 70 and BB Width at {bb_width:.1f}%."
                elif obv_slope < 0 and rsi < 50:
                    status_short, full_signal, reason = "🔴 Divergence", "WARNING / NEAR-TERM DISTRIBUTION", f"Negative OBV volume flow and weak RSI ({rsi:.1f}) signal distribution."
                elif is_squeeze:
                    status_short, full_signal, reason = "🟡 Squeeze", "WATCH / VOLATILITY SQUEEZE", f"Bollinger Band squeeze active (BB Width: {bb_width:.1f}%)."
                elif curr_price > curr_ema20 and curr_price > curr_sma50 and rvol_confirmed:
                    status_short, full_signal, reason = "🟢 Bullish", "ACCUMULATE / BREAKOUT", f"Clear breakout above buffer. ADX={adx:.1f}, RSI={rsi:.1f}, RVOL={rvol:.2f}x."
                elif curr_price <= curr_ema20 or curr_price <= curr_sma50:
                    status_short, full_signal, reason = "🟡 Pullback", "MACRO BULL / WAIT FOR ENTRY", "Macro trend is UP (>200 EMA), but price is pulling back below short MAs."
                else:
                    status_short, full_signal, reason = "🟡 Warning", "BULLISH / LOW VOLUME", f"Price > 200 EMA buffer, but RVOL ({rvol:.2f}x) is below target."
            elif curr_price < lower_th and slope == "DOWN":
                if adx < 20:
                    status_short, full_signal, reason = "⚪ Consolidating", "SIDEWAYS / WEAK BEAR TREND", f"Price < 200 EMA, but weak trend strength with ADX ({adx:.1f})."
                elif rvol_confirmed:
                    status_short, full_signal, reason = "🔴 Bearish", "SELL / CASH OUT", f"Price < buffer with slope DOWN, confirmed by selling volume (RVOL: {rvol:.2f}x)."
                else:
                    status_short, full_signal, reason = "🔴 Bearish", "SELL / LOW VOL DOWN", f"Price < buffer & slope DOWN (RVOL: {rvol:.2f}x)."
            elif lower_th <= curr_price <= upper_th:
                status_short, full_signal, reason = "⚪ Neutral", "NOISE BUFFER ZONE", f"Price within +/-{effective_buffer*100:.1f}% noise buffer of 200 EMA."
            else:
                status_short, full_signal, reason = "🟡 Warning", "WATCH / CAUTION", f"Price crossed EMA but slope ({slope}) or buffer does not confirm direction."

            pct_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            obv_label = "INFLOW" if obv_slope > 0 else "OUTFLOW"
            
            flyover = f"[{full_signal}] {reason} | Timeframe: {timeframe_label} | Price: ${curr_price:.2f} \vert{} ADX: {adx:.1f} \vert{} RSI: {rsi:.1f} \vert{} OBV: {obv_label} \vert{} BB Width: {bb_width:.1f}\% \vert{} RVOL: {rvol:.2f}x \vert{} ATR\%: {atr_pct:.2f}\% \vert{} 200 EMA:${curr_ema200:.2f} ({pct_ema200:+.2f}%)"

            results.append({
                "Status & Signal": status_short,
                "Ticker": ticker,
                "Trigger Details
