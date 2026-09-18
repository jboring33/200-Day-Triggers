"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Trend & Volatility Screening with ATR Dynamic Buffers and URL Parameter Persistence.
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

def analyze_enhanced_sma_strategy(tickers, lookback_period="2y", slope_window=5, default_buffer_pct=0.02):
    """Evaluates tickers across Macro (200 EMA) and Short-Term (20 EMA / 50 SMA) trends."""
    results = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=lookback_period, auto_adjust=True, progress=False)
            
            if df.empty or len(df) < 200 + slope_window:
                st.warning(f"Skipping {ticker}: Insufficient historical data (requires >205 trading days).")
                continue
            
            # --- Ensure single-column Series extraction for yfinance data ---
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
                
            high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
            low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
            
            # Clean OHLC DataFrame for ATR
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
            
            # Extract current scalar price
            curr_price = float(close.iloc[-1])
            
            # Moving Average Calculations
            ema20 = close.ewm(span=20, adjust=False).mean()
            sma50 = close.rolling(window=50).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            sma200 = close.rolling(window=200).mean()
            
            curr_ema20 = float(ema20.iloc[-1])
            curr_sma50 = float(sma50.iloc[-1])
            curr_ema200 = float(ema200.iloc[-1])
            curr_sma200 = float(sma200.iloc[-1])
            prev_ema200 = float(ema200.iloc[-(slope_window + 1)])
            
            # Dynamic Volatility Buffer
            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / curr_price)
            effective_buffer_pct = max(default_buffer_pct, atr_pct * 1.5)
            
            upper_threshold = curr_ema200 * (1 + effective_buffer_pct)
            lower_threshold = curr_ema200 * (1 - effective_buffer_pct)
            
            # Slope Calculation
            ema_diff_pct = ((curr_ema200 - prev_ema200) / prev_ema200) * 100
            if ema_diff_pct > 0.05:
                slope = "UP"
            elif ema_diff_pct < -0.05:
                slope = "DOWN"
            else:
                slope = "FLAT"
                
            # Golden/Death Cross Evaluation
            ma_cross = "GOLDEN CROSS (Bullish)" if curr_sma50 > curr_sma200 else "DEATH CROSS (Bearish)"
                
            # Short-Term Momentum Alignment Check
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
                
            # Signal Logic
            pct_from_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            
            if curr_price > upper_threshold and slope in ["UP", "FLAT"]:
                if above_20_ema and above_50_sma:
                    signal = "BUY / BULLISH HOLD"
                    status = "🟢 BULLISH"
                    reason = f"Price > {effective_buffer_pct*100:.1f}% buffer above 200 EMA with full short-term momentum."
                else:
                    signal = "MACRO BULL / WAIT FOR ENTRY"
                    status = "🟡 PULLBACK"
                    reason = f"Macro trend is UP (>200 EMA), but price is below 20 EMA/50 SMA. Wait for momentum reclaim."
            elif curr_price < lower_threshold and slope == "DOWN":
                signal = "SELL / CASH OUT"
                status = "🔴 BEARISH"
                reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA & slope is DOWN."
            elif lower_threshold <= curr_price <= upper_threshold:
                signal = "NOISE BUFFER ZONE"
                status = "⚪ NEUTRAL"
                reason = f"Price within ±{effective_buffer_pct*100:.1f}% noise buffer of 200 EMA; avoid whipsaw."
            else:
                signal = "WATCH / CAUTION"
                status = "🟡 WARNING"
                reason = f"Price crossed EMA but slope ({slope}) or buffer does not confirm exit/entry."

            results.append({
                "Status": status,
                "Status & Signal": f"{status} - {signal}",
                "Ticker": ticker,
                "Price": round(curr_price, 2),
                "20 EMA": round(curr_ema20, 2),
                "50 SMA": round(curr_sma50, 2),
                "200 EMA": round(curr_ema200, 2),
                "Dist 200EMA (%)": f"{pct_from_ema200:+.2f}%",
                "Dynamic Buffer": f"±{effective_buffer_pct*100:.1f}%",
                "Short-Term Trend": short_term_momentum,
                "50/200 Trend": ma_cross,
                "Reason": reason
            })
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
            
    return pd.DataFrame(results)

# --- Streamlit Dashboard UI ---

st.title(f"📈 {REPO_NAME}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- URL Parameter Management ---
DEFAULT_TICKER_STRING = "SPY, QQQ, GLD, BTC-USD, NVDA, AAPL, TLT, IWM"

# Read tickers parameter from URL bar if present
query_params = st.query_params
initial_tickers = query_params.get("tickers", DEFAULT_TICKER_STRING)

st.sidebar.header("Screener Configuration")

# Input field bound to initial_tickers value
ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (comma-separated):", 
    value=initial_tickers,
    help="Changes to this list automatically update your browser URL so you can bookmark/favorite your custom watchlist."
)

buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100

# Update URL query parameters based on current user input
clean_ticker_str = ", ".join([t.strip().upper() for t in ticker_input.split(",") if t.strip()])
st.query_params["tickers"] = clean_ticker_str

if st.sidebar.button("Run Screener", type="primary") or "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True
    
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    with st.spinner("Fetching market data and calculating indicators..."):
        df_results = analyze_enhanced_sma_strategy(tickers_list, default_buffer_pct=buffer_setting)
        
    if not df_results.empty:
        # Key Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickers", len(df_results))
        col2.metric("Full Bullish Signals", len(df_results[df_results["Status"].str.contains("BULLISH")]))
        col3.metric("Macro Bull (Pullback)", len(df_results[df_results["Status"].str.contains("PULLBACK")]))
        col4.metric("Bearish Signals", len(df_results[df_results["Status"].str.contains("BEARISH")]))
        
        st.divider()
        
        # Display Streamlit Table with custom Column Layout
        st.dataframe(
            df_results[["Status", "Status & Signal", "Ticker", "Price", "20 EMA", "50 SMA", "200 EMA", "Dynamic Buffer", "Reason"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="🟢 Bullish | 🟡 Pullback/Warning | 🔴 Bearish | ⚪ Neutral", width="small"),
                "Status & Signal": st.column_config.TextColumn("Combined Signal & Status", width="medium"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Price": st.column_config.NumberColumn(format="$%.2f", help="Current Close Price"),
                "20 EMA": st.column_config.NumberColumn(format="$%.2f"),
                "50 SMA": st.column_config.NumberColumn(format="$%.2f"),
                "200 EMA": st.column_config.NumberColumn(format="$%.2f"),
                "Dynamic Buffer": st.column_config.TextColumn("Buffer", help="Effective volatility-adjusted buffer percentage"),
                "Reason": st.column_config.TextColumn("Trigger Details / Flyover", help="Hover to view full execution reason and metrics"),
            }
        )
        
        # Download CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name="200_day_triggers_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No results to display.")
