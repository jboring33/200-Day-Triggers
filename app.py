"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Weekly Trend & Volatility Screening with Dynamic ATR Buffers, ATR% Filters, 
             Scaled Weekly RVOL Volume Confirmation, Steady Accumulation Logic, 
             Preset Instance Routing, Parameter Guidance, and Dynamic URL Persistence.
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

# --- Weekly Target Instance Configurations (Tuned for Weeks-to-Months Horizons) ---
PRESET_CONFIGS = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 1.25
    },
    "Mag 7 Tech": {
        "tickers": "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA",
        "buffer": 3.0,
        "rvol": 1.00,
        "min_atr": 2.50
    },
    "Market Traction & RS Leaders": {
        "tickers": "LLY, WMT, COST, BRK-B, GE, ETN, PWR, VRT, RTX, IJH, EWJ",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 1.50
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 1.00
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
        "rvol": 0.70,
        "min_atr": 0.50
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.75
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.10,
        "min_atr": 3.00
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA",
        "buffer": 2.0,
        "rvol": 0.85,
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

def analyze_enhanced_sma_strategy(tickers, slope_window=5, default_buffer_pct=0.020, rvol_threshold=0.85):
    """Evaluates tickers across Macro (200 EMA), Short-Term (20 EMA / 50 SMA) trends on Weekly timeframe,
    applying dynamic ATR% buffers, scaled weekly RVOL thresholds, and steady accumulation logic."""
    results = []
    
    interval = "1wk"
    lookback_period = "10y"
    timeframe_label = "Weekly"
    min_required_bars = 52  # Requires at least 1 full year (~52 weekly bars)
    
    # Tweak 1: Auto-scale RVOL threshold specifically for weekly bar aggregation
    effective_rvol_threshold = rvol_threshold * 0.85
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=lookback_period, interval=interval, auto_adjust=True, progress=False)
            
            if df.empty or len(df) < min_required_bars:
                st.warning(f"Skipping {ticker}: Insufficient historical data for {timeframe_label} timeframe (requires >{min_required_bars} weekly bars / ~1 year).")
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
            
            # RVOL Calculation (Current Weekly Volume / 20-Bar SMA Volume)
            vol_20_sma = float(volume.rolling(window=20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= effective_rvol_threshold
            
            # Moving Average Calculations
            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(window=min(50, len(close)), min_periods=10).mean()
            ema200 = close.ewm(span=min(200, len(close)), adjust=False).mean()
            sma200 = close.rolling(window=min(200, len(close)), min_periods=20).mean()
            
            curr_ema20 = float(ema20.iloc[-1])
            curr_sma50 = float(sma50.iloc[-1])
            curr_ema200 = float(ema200.iloc[-1])
            curr_sma200 = float(sma200.iloc[-1])
            
            # Slope Calculation
            actual_slope_offset = min(slope_window + 1, len(ema200) - 1)
            prev_ema200 = float(ema200.iloc[-actual_slope_offset])
            
            # Volatility Calculations (ATR & ATR%)
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
                vol_label = "STRONG BUYING VOLUME" if rvol_confirmed else "STEADY / MODERATE VOLUME"
            elif curr_price < lower_threshold and slope == "DOWN":
                vol_label = "STRONG SELLING VOLUME" if rvol_confirmed else "LOW VOLUME"
            else:
                vol_label = "HIGH VOLUME" if rvol_confirmed else "LOW VOLUME"

            vol_str = f"RVOL: {rvol:.2f}x ({vol_label})"
            pct_from_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            
            # Tweak 2: Refined Signal Logic with "Steady Accumulation" Zone
            if curr_price > upper_threshold and slope in ["UP", "FLAT"]:
                if above_20_ema and above_50_sma:
                    if rvol_confirmed:
                        status_short = "🟢 Bullish"
                        full_signal = "BUY / BULLISH HOLD"
                        reason = f"Price > {effective_buffer_pct*100:.1f}% buffer above 200 EMA with full short momentum & strong buying volume ({rvol:.2f}x >= {effective_rvol_threshold:.2f}x)."
                    elif rvol >= (effective_rvol_threshold * 0.80):
                        # Steady accumulation status (80%-100% of threshold)
                        status_short = "🟢 Bullish (Steady Vol)"
                        full_signal = "BULLISH HOLD / STEADY ACCUMULATION"
                        reason = f"Price > 200 EMA buffer with MAs aligned. Volume ({rvol:.2f}x) represents steady institutional accumulation."
                    else:
                        status_short = "🟡 Warning"
                        full_signal = "BULLISH / LOW VOLUME"
                        reason = f"Price > 200 EMA buffer & short MAs aligned, but weekly volume ({rvol:.2f}x) is significantly below threshold ({effective_rvol_threshold:.2f}x)."
                else:
                    status_short = "🟡 Pullback"
                    full_signal = "MACRO BULL / WAIT FOR ENTRY"
                    reason = "Macro trend is UP (>200 EMA), but price is pulling back below 20 EMA/50 SMA. Wait for momentum reclaim."
            elif curr_price < lower_threshold and slope == "DOWN":
                if rvol_confirmed:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / CASH OUT"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA on selling volume ({rvol:.2f}x) & slope DOWN."
                else:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / LOW VOL DOWN"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA & slope DOWN (RVOL: {rvol:.2f}x)."
            elif lower_threshold <= curr_price <= upper_threshold:
                status_short = "⚪ Neutral"
                full_signal = "NOISE BUFFER ZONE"
                reason = f"Price within +/-{effective_buffer_pct*100:.1f}% noise buffer of 200 EMA; avoid whipsaw."
            else:
                status_short = "🟡 Warning"
                full_signal = "WATCH / CAUTION"
                reason = f"Price crossed 200 EMA but slope ({slope}) or buffer does not confirm exit/entry."

            flyover_summary = (
                f"[{full_signal}] {reason} | "
                f"Timeframe: {timeframe_label} | "
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
                "Timeframe": timeframe_label,
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
st.caption("Weekly Position Screener | Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# --- URL Parameter & State Management ---
query_params = st.query_params
url_preset = query_params.get("preset", "Broad Market")

if url_preset not in PRESET_CONFIGS:
    url_preset = "Broad Market"

# Session State Initialization
if "current_preset" not in st.session_state:
    st.session_state["current_preset"] = url_preset

st.sidebar.header("Screener Configuration")

# --- Primary Action Button ---
run_screener = st.sidebar.button("🚀 Run Screener", type="primary", use_container_width=True)

st.sidebar.divider()

# Locked Timeframe Indicator
st.sidebar.info("📅 **Timeframe Locked:** Weekly (`1wk`)\nOptimized for multi-week & multi-month position trading horizons.")

# Preset Dropdown Selector
selected_preset = st.sidebar.selectbox(
    "Target Instance Preset:",
    options=list(PRESET_CONFIGS.keys()),
    index=list(PRESET_CONFIGS.keys()).index(st.session_state["current_preset"]),
    help="Select a tuned preset to automatically load tailored weekly parameters."
)

preset_changed = selected_preset != st.session_state["current_preset"]
if preset_changed:
    st.session_state["current_preset"] = selected_preset

active_defaults = PRESET_CONFIGS[selected_preset]

# Resolve Parameter Defaults
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

# Sidebar Inputs
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
    help="Percentage dead zone around the 200 EMA to avoid false breakout whipsaws. Compared automatically against 1.5x ATR%."
)

rvol_setting = st.sidebar.slider(
    "Min Weekly RVOL Confirmation (x)", 
    min_value=0.1, 
    max_value=2.5, 
    value=default_rvol, 
    step=0.05,
    help="Minimum Weekly Relative Volume required to confirm trend strength (Current Weekly Vol / 20-Week Vol SMA)."
)

min_atr_setting = st.sidebar.slider(
    "Min Weekly Volatility / ATR (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=default_min_atr, 
    step=0.25,
    help="Filters out assets whose 14-week Average True Range is below this percentage of price."
)

# Synchronize URL Query Parameters live
clean_ticker_str = ", ".join([t.strip().upper() for t in ticker_input.split(",") if t.strip()])
if clean_ticker_str:
    st.query_params["tickers"] = clean_ticker_str

st.query_params["preset"] = selected_preset
st.query_params["interval"] = "1wk"
st.query_params["buffer"] = f"{buffer_setting:.1f}"
st.query_params["rvol"] = f"{rvol_setting:.2f}"
st.query_params["min_atr"] = f"{min_atr_setting:.2f}"

# Execution Logic
if run_screener or ("ran_once" in st.session_state and clean_ticker_str):
    st.session_state["ran_once"] = True
    
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if tickers_list:
        with st.spinner(f"Fetching weekly market data and calculating indicators for [{selected_preset}] preset..."):
            df_results = analyze_enhanced_sma_strategy(
                tickers_list, 
                default_buffer_pct=buffer_setting / 100.0,
                rvol_threshold=rvol_setting
            )
            
        if not df_results.empty:
            if min_atr_setting > 0:
                df_results = df_results[df_results["ATR (%)"] >= min_atr_setting].reset_index(drop=True)
            
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
                        help="Weekly trend status. See reference guide below."
                    ),
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Trigger Details": st.column_config.TextColumn(
                        "Trigger Details", 
                        width="large",
                        help="Hover over any cell to view full price, weekly RVOL, ATR%, 20 EMA, 50 SMA, 200 EMA, and volatility buffer metrics"
                    ),
                }
            )
            
            # --- Guidance Section ---
            with st.expander("📖 Signal Reference Guide & Weekly Parameter Tuning", expanded=False):
                guide_template = """
### Signal Definitions (Weekly Focus)
- **Bullish (BUY / BULLISH HOLD):** Price is above the weekly 200 EMA (plus dynamic buffer), short-term MAs are aligned, and RVOL confirms **STRONG BUYING VOLUME** (>= {rvol:.2f}x).
- **Bullish (Steady Vol):** Price is aligned above moving averages on steady accumulation volume (80%-100% of target RVOL).
- **Pullback (MACRO BULL / WAIT FOR ENTRY):** Weekly macro trend remains long-term bullish (>200 EMA), but price is pulling back toward short-term MAs. Optimal buy zone for position traders.
- **Warning (BULLISH / LOW VOLUME or CAUTION):** Price is above targets, but volume is significantly low, signaling lack of institutional accumulation.
- **Neutral (NOISE BUFFER ZONE):** Price is consolidating within the noise buffer zone around the 200 EMA. Avoid buying or selling to prevent whipsaws.
- **Bearish (SELL / CASH OUT):** Price is below the weekly 200 EMA (minus dynamic buffer) with downward slope and selling volume.

---

### Parameter Tuning Guide (Weekly Horizon)

#### 1. Weekly Base Buffer Noise Filter (%) [Current: {buffer:.1f}%]
- **WHAT IT IS:** Mandatory percentage dead zone around the weekly 200 EMA (compared against 1.5x Weekly ATR%).
- **RECOMMENDED:** 2.0% for broad indices, 3.0% for high-beta tech (Mag 7), and 4.0% for crypto/commodities.

#### 2. Min Weekly RVOL Confirmation (x) [Current: {rvol:.2f}x]
- **WHAT IT IS:** Measures current weekly volume against its 20-week SMA volume.
- **AUTOMATIC TWEAK:** Scaled down by 0.85x internally to account for multi-day volume accumulation dynamics on weekly charts.

#### 3. Min Weekly ATR (%) [Current: {min_atr:.2f}%]
- **WHAT IT IS:** The 14-week Average True Range expressed as a percentage of current price.
- **RECOMMENDED:** 1.25%-1.50% for standard equities; 2.50%+ for high-growth tech.
"""
                st.markdown(
                    guide_template.format(
                        rvol=rvol_setting,
                        buffer=buffer_setting,
                        min_atr=min_atr_setting
                    )
                )
        else:
            st.info("No tickers met the specified criteria or dataset returned empty.")
    else:
        st.warning("Please enter at least one valid ticker symbol in the sidebar.")
