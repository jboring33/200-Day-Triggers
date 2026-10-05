"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Trend & Volatility Screening with ATR Dynamic Buffers, ATR% Filters, RVOL Volume Confirmation,
             Preset Instance Routing, and Dynamic URL Parameter Persistence.
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

# --- Preset Strategy Configurations ---
PRESET_CONFIGS = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 1.5,
        "rvol": 1.00,
        "min_atr": 0.75
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU",
        "buffer": 1.5,
        "rvol": 1.00,
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
        "buffer": 1.5,
        "rvol": 1.00,
        "min_atr": 0.50
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 3.5,
        "rvol": 1.50,
        "min_atr": 2.00
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA",
        "buffer": 2.0,
        "rvol": 1.25,
        "min_atr": 0.00
    }
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
            
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
                
            high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
            low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
            
            curr_price = float(close.iloc[-1])
            curr_volume = float(volume.iloc[-1])
            
            vol_20_sma = float(volume.rolling(window=20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold
            
            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(window=50).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            sma200 = close.rolling(window=200).mean()
            
            curr_ema20 = float(ema20.iloc[-1])
            curr_sma50 = float(sma50.iloc[-1])
            curr_ema200 = float(ema200.iloc[-1])
            curr_sma200 = float(sma200.iloc[-1])
            prev_ema200 = float(ema200.iloc[-(slope_window + 1)])
            
            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / curr_price) * 100
            effective_buffer_pct = max(default_buffer_pct, (atr / curr_price) * 1.5)
            
            upper_threshold = curr_ema200 * (1 + effective_buffer_pct)
            lower_threshold = curr_ema200 * (1 - effective_buffer_pct)
            
            ema_diff_pct = ((curr_ema200 - prev_ema200) / prev_ema200) * 100
            if ema_diff_pct > 0.05:
                slope = "UP"
            elif ema_diff_pct < -0.05:
                slope = "DOWN"
            else:
                slope = "FLAT"
                
            ma_cross = "GOLDEN CROSS (Bullish)" if curr_sma50 > curr_sma200 else "DEATH CROSS (Bearish)"
                
            above_20_ema = curr_price > curr_ema20
            above_50_sma = curr_price > curr_sma50
            
            if above_20_ema and above_50_sma:
                short_term_momentum = "BULLISH (Above 20 EMA & 50 SMA)"
            elif not above_20_ema and not above_50_sma:
                short_term_momentum = "BEARISH (Below 20 EMA & 50 SMA)"
            elif above_20_ema and not above_50_sma:
                short_term_momentum = "MIXED (Above 20 EMA / Below 50 SMA)"
            else:
                short_term_momentum = "PULLBACK (Below 20 EMA / Above 50 SMA)"
                
            if curr_price > upper_threshold and above_20_ema and above_50_sma:
                vol_label = "STRONG BUYING VOLUME" if rvol_confirmed else "LOW VOLUME"
            elif curr_price < lower_threshold and slope == "DOWN":
                vol_label = "STRONG SELLING VOLUME" if rvol_confirmed else "LOW VOLUME"
            else:
                vol_label = "HIGH VOLUME" if rvol_confirmed else "LOW VOLUME"

            vol_str = f"RVOL: {rvol:.2f}x ({vol_label})"
            pct_from_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            
            if curr_price > upper_threshold and slope in ["UP", "FLAT"]:
                if above_20_ema and above_50_sma:
                    if rvol_confirmed:
                        status_short = "🟢 Bullish"
                        full_signal = "BUY / BULLISH HOLD"
                        reason = f"Price > {effective_buffer_pct*100:.1f}% buffer above 200 EMA with full short-term momentum & {vol_label.lower()} ({rvol:.2f}x)."
                    else:
                        status_short = "🟡 Warning"
                        full_signal = "BULLISH / LOW VOLUME"
                        reason = f"Price > 200 EMA buffer & short MAs aligned, but volume status is {vol_label} (RVOL: {rvol:.2f}x < {rvol_threshold:.2f}x)."
                else:
                    status_short = "🟡 Pullback"
                    full_signal = "MACRO BULL / WAIT FOR ENTRY"
                    reason = "Macro trend is UP (>200 EMA), but price is below 20 EMA/50 SMA. Wait for momentum reclaim."
            elif curr_price < lower_threshold and slope == "DOWN":
                if rvol_confirmed:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / CASH OUT"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA on {vol_label.lower()} ({rvol:.2f}x) & slope is DOWN."
                else:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / LOW VOL DOWN"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA & slope DOWN (RVOL: {rvol:.2f}x, {vol_label})."
            elif lower_threshold <= curr_price <= upper_threshold:
                status_short = "⚪ Neutral"
                full_signal = "NOISE BUFFER ZONE"
                reason = f"Price within +/-{effective_buffer_pct*100:.1f}% noise buffer of 200 EMA; avoid whipsaw."
            else:
                status_short = "🟡 Warning"
                full_signal = "WATCH / CAUTION"
                reason = f"Price crossed EMA but slope ({slope}) or buffer does not confirm exit/entry."

            flyover_summary = (
                f"[{full_signal}] {reason} | "
                f"Price: ${curr_price:.2f} | "
                f"{vol_str} | "
                f"ATR%: {atr_pct:.2f}% | "
                f"20 EMA: ${curr_ema20:.2f} | "
                f"50 SMA: ${curr_sma50:.2f} | "
                f"200 EMA: ${curr_ema200:.2f} | "
                f"Dist 200EMA: {pct_from_ema200:+.2f}% | "
                f"Buffer: +/-{effective_buffer_pct*100:.1f}% | "
                f"Short Trend: {short_term_momentum} | "
                f"Cross: {ma_cross}"
            )

            results.append({
                "Status & Signal": status_short,
                "Ticker": ticker,
                "Trigger Details": flyover_summary,
                "Full Signal": full_signal,
                "Price": round(curr_price, 2),
                "RVOL": round(rvol, 2),
                "ATR (%)": round(atr_pct, 2),
                "Volume Label": vol_label,
                "20 EMA": round(curr_ema20, 2),
                "50 SMA": round(curr_sma50, 2),
                "200 EMA": round(curr_ema200, 2),
                "Dist 200EMA (%)": f"{pct_from_ema200:+.2f}%",
                "Dynamic Buffer": f"+/-{effective_buffer_pct*100:.1f}%",
                "Reason": reason
            })
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
            
    return pd.DataFrame(results)

# --- Streamlit Dashboard UI ---

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# --- URL Parameter & State Management ---
query_params = st.query_params
url_preset = query_params.get("preset", "Broad Market")

if url_preset not in PRESET_CONFIGS:
    url_preset = "Broad Market"

# Session State Initialization for Selected Preset
if "current_preset" not in st.session_state:
    st.session_state["current_preset"] = url_preset

st.sidebar.header("Screener Configuration")

# --- Primary Action Button (Top Positioned) ---
run_screener = st.sidebar.button("🚀 Run Screener", type="primary", use_container_width=True)

st.sidebar.divider()

# Preset Dropdown Selector
selected_preset = st.sidebar.selectbox(
    "Target Instance Preset:",
    options=list(PRESET_CONFIGS.keys()),
    index=list(PRESET_CONFIGS.keys()).index(st.session_state["current_preset"]),
    help="Select a tuned instance preset to automatically populate parameters based on asset class volatility profiles."
)

# Detect if the preset changed in the UI and load standard preset values
preset_changed = selected_preset != st.session_state["current_preset"]
if preset_changed:
    st.session_state["current_preset"] = selected_preset

active_defaults = PRESET_CONFIGS[selected_preset]

# Resolve dynamic values (URL parameters take priority unless preset selector was toggled)
default_tickers = active_defaults["tickers"] if preset_changed else query_params.get("tickers", active_defaults["tickers"])
try:
    default_buffer = active_defaults["buffer"] if preset_changed else float(query_params.get("buffer", active_defaults["buffer"]))
except ValueError:
    default_buffer = active_defaults["buffer"]

try:
    default_rvol = active_defaults["rvol"] if preset_changed else float(query_params.get("rvol", active_defaults["rvol"]))
except ValueError:
    default_rvol = active_defaults["rvol"]

try:
    default_min_atr = active_defaults["min_atr"] if preset_changed else float(query_params.get("min_atr", active_defaults["min_atr"]))
except ValueError:
    default_min_atr = active_defaults["min_atr"]

# Sidebar Parameter Inputs
ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (comma-separated):", 
    value=default_tickers,
    placeholder="e.g. SPY, QQQ, NVDA, BTC-USD",
    help="Type tickers here. The URL bar updates dynamically so you can bookmark your exact custom run."
)

buffer_setting = st.sidebar.slider(
    "Base Buffer Noise Filter (%)", 
    min_value=1.0, 
    max_value=5.0, 
    value=default_buffer, 
    step=0.5,
    help="Sets the percentage dead zone around the 200 EMA to avoid false breakout whipsaws. Compared automatically against 1.5x ATR%."
)

rvol_setting = st.sidebar.slider(
    "Min RVOL Breakout Confirmation (x)", 
    min_value=0.1, 
    max_value=2.5, 
    value=default_rvol, 
    step=0.05,
    help="Minimum Relative Volume required to confirm trend breakout strength (Current Volume / 20-Day SMA Volume)."
)

min_atr_setting = st.sidebar.slider(
    "Min Daily Volatility / ATR (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=default_min_atr, 
    step=0.25,
    help="Filters out assets whose 14-day Average True Range is below this percentage of price."
)

# Synchronize URL Query Parameters live
clean_ticker_str = ", ".join([t.strip().upper() for t in ticker_input.split(",") if t.strip()])
if clean_ticker_str:
    st.query_params["tickers"] = clean_ticker_str

st.query_params["preset"] = selected_preset
st.query_params["buffer"] = f"{buffer_setting:.1f}"
st.query_params["rvol"] = f"{rvol_setting:.2f}"
st.query_params["min_atr"] = f"{min_atr_setting:.2f}"

# Execution Logic
if run_screener or ("ran_once" in st.session_state and clean_ticker_str):
    st.session_state["ran_once"] = True
    
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if tickers_list:
        with st.spinner(f"Fetching market data and calculating indicators for [{selected_preset}] preset..."):
            df_results = analyze_enhanced_sma_strategy(
                tickers_list, 
                default_buffer_pct=buffer_setting / 100.0,
                rvol_threshold=rvol_setting
            )
            
        if not df_results.empty:
            if min_atr_setting > 0:
                df_results = df_results[df_results["ATR (%)"] >= min_atr_setting].reset_index(drop=True)
            
        if not df_results.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tickers", len(df_results))
            col2.metric("Full Bullish Signals", len(df_results[df_results["Status & Signal"].str.contains("Bullish")]))
            col3.metric("Macro Bull (Pullback)", len(df_results[df_results["Status & Signal"].str.contains("Pullback")]))
            col4.metric("Bearish Signals", len(df_results[df_results["Status & Signal"].str.contains("Bearish")]))
            
            st.divider()
            
            st.dataframe(
                df_results[["Status & Signal", "Ticker", "Trigger Details"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status & Signal": st.column_config.TextColumn(
                        "Status & Signal", 
                        width="small",
                        help="Short trend status."
                    ),
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Trigger Details": st.column_config.TextColumn(
                        "Trigger Details", 
                        width="large",
                        help="Hover over any cell to view full price, RVOL, ATR%, 20 EMA, 50 SMA, 200 EMA, and volatility buffer metrics"
                    ),
                }
            )
            
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download CSV Report ({selected_preset})",
                data=csv,
                file_name=f"200_day_triggers_{selected_preset.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No tickers matched the current ATR% volatility filter criteria.")
    else:
        st.warning("Please enter at least one ticker in the sidebar to run the screener.")
else:
    st.info("👈 Select an Instance Preset in the sidebar and click 'Run Screener' to analyze.")
