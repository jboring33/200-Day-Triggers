"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Trend & Volatility Screening with ATR Dynamic Buffers, RVOL Volume Confirmation,
             Flyover Tooltips, and URL Parameter Persistence.
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

def analyze_enhanced_sma_strategy(tickers, lookback_period="2y", slope_window=5, default_buffer_pct=0.02, rvol_threshold=1.25):
    """Evaluates tickers across Macro (200 EMA), Short-Term (20 EMA / 50 SMA) trends, and RVOL volume logic."""
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
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            
            # Clean OHLC DataFrame for ATR
            df_clean = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
            
            # Extract current scalar price & volume
            curr_price = float(close.iloc[-1])
            curr_volume = float(volume.iloc[-1])
            
            # RVOL Calculation (Current Volume / 20-Day SMA Volume)
            vol_20_sma = float(volume.rolling(window=20).mean().iloc[-1])
            rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
            rvol_confirmed = rvol >= rvol_threshold
            
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
                
            # Signal Logic with RVOL Confirmation
            pct_from_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            vol_str = f"RVOL: {rvol:.2f}x ({'HIGH VOLUME CONFIRMED' if rvol_confirmed else 'LOW VOLUME'})"
            
            if curr_price > upper_threshold and slope in ["UP", "FLAT"]:
                if above_20_ema and above_50_sma:
                    if rvol_confirmed:
                        status_short = "🟢 Bullish"
                        full_signal = "BUY / BULLISH HOLD"
                        reason = f"Price > {effective_buffer_pct*100:.1f}% buffer above 200 EMA with full short-term momentum & strong volume ({rvol:.2f}x)."
                    else:
                        status_short = "🟡 Warning"
                        full_signal = "BULLISH / LOW VOLUME"
                        reason = f"Price > 200 EMA buffer & short MAs aligned, but RVOL ({rvol:.2f}x) is below {rvol_threshold}x threshold."
                else:
                    status_short = "🟡 Pullback"
                    full_signal = "MACRO BULL / WAIT FOR ENTRY"
                    reason = f"Macro trend is UP (>200 EMA), but price is below 20 EMA/50 SMA. Wait for momentum reclaim."
            elif curr_price < lower_threshold and slope == "DOWN":
                if rvol_confirmed:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / CASH OUT"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA on high volume ({rvol:.2f}x) & slope is DOWN."
                else:
                    status_short = "🔴 Bearish"
                    full_signal = "SELL / LOW VOL DOWN"
                    reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA & slope DOWN (RVOL: {rvol:.2f}x)."
            elif lower_threshold <= curr_price <= upper_threshold:
                status_short = "⚪ Neutral"
                full_signal = "NOISE BUFFER ZONE"
                reason = f"Price within ±{effective_buffer_pct*100:.1f}% noise buffer of 200 EMA; avoid whipsaw."
            else:
                status_short = "🟡 Warning"
                full_signal = "WATCH / CAUTION"
                reason = f"Price crossed EMA but slope ({slope}) or buffer does not confirm exit/entry."

            # Construct comprehensive Flyover detail string
            flyover_summary = (
                f"[{full_signal}] {reason} | Price: ${curr_price:.2f} | {vol_str} | "
                f"20 EMA: ${curr_ema20:.2f} | 50 SMA: ${curr_sma50:.2f} | 200 EMA: ${curr_ema200:.2f} | "
                f"Dist 200EMA: {pct_from_ema200:+.2f}% | Buffer: ±{effective_buffer_pct*100:.1f}% | "
                f"Short Trend: {short_term_momentum} | Cross: {ma_cross}"
            )

            results.append({
                "Status & Signal": status_short,
                "Ticker": ticker,
                "Trigger Details": flyover_summary,
                # Retain raw metrics for CSV export downloading
                "Full Signal": full_signal,
                "Price": round(curr_price, 2),
                "RVOL": round(rvol, 2),
                "20 EMA": round(curr_ema20, 2),
                "50 SMA": round(curr_sma50, 2),
                "200 EMA": round(curr_ema200, 2),
                "Dist 200EMA (%)": f"{pct_from_ema200:+.2f}%",
                "Dynamic Buffer": f"±{effective_buffer_pct*100:.1f}%",
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

query_params = st.query_params
initial_tickers = query_params.get("tickers", DEFAULT_TICKER_STRING)

st.sidebar.header("Screener Configuration")

ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (comma-separated):", 
    value=initial_tickers,
    help="Changes to this list automatically update your browser URL so you can bookmark/favorite your custom watchlist."
)

buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100
rvol_setting = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", min_value=1.0, max_value=2.5, value=1.25, step=0.05)

clean_ticker_str = ", ".join([t.strip().upper() for t in ticker_input.split(",") if t.strip()])
st.query_params["tickers"] = clean_ticker_str

if st.sidebar.button("Run Screener", type="primary") or "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True
    
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    with st.spinner("Fetching market data and calculating indicators..."):
        df_results = analyze_enhanced_sma_strategy(
            tickers_list, 
            default_buffer_pct=buffer_setting,
            rvol_threshold=rvol_setting
        )
        
    if not df_results.empty:
        # Key Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickers", len(df_results))
        col2.metric("Full Bullish Signals", len(df_results[df_results["Status & Signal"].str.contains("Bullish")]))
        col3.metric("Macro Bull (Pullback)", len(df_results[df_results["Status & Signal"].str.contains("Pullback")]))
        col4.metric("Bearish Signals", len(df_results[df_results["Status & Signal"].str.contains("Bearish")]))
        
        st.divider()
        
        # Display Streamlit Table
        st.dataframe(
            df_results[["Status & Signal", "Ticker", "Trigger Details"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status & Signal": st.column_config.TextColumn(
                    "Status & Signal", 
                    width="small",
                    help="Short trend status. See reference legend below for full definitions."
                ),
                "Ticker": st.column_config.TextColumn(
                    "Ticker", 
                    width="small"
                ),
                "Trigger Details": st.column_config.TextColumn(
                    "Trigger Details", 
                    width="large",
                    help="Hover over any cell to view full price, RVOL, 20 EMA, 50 SMA, 200 EMA, and volatility buffer metrics"
                ),
            }
        )
        
        # Reference Legend
        with st.expander("📖 Signal Reference Guide", expanded=False):
            st.markdown(f"""
            * **🟢 Bullish (BUY / BULLISH HOLD):** Price is above the 200 EMA (plus dynamic buffer), short-term momentum MAs are aligned, and RVOL confirms institutional participation ($\ge {rvol_setting:.2f}\text{{x}}$).
            * **🟡 Pullback (MACRO BULL / WAIT FOR ENTRY):** Macro trend remains long-term bullish (>200 EMA), but price is pulling back below short-term MAs. Wait for momentum reclaim before entering.
            * **🟡 Warning (BULLISH / LOW VOLUME or CAUTION):** Price is above targets, but RVOL is below the ${rvol_setting:.2f}\text{{x}}$ threshold, signaling low-volume breakout risk; or slope/buffer conditions are incomplete.
            * **⚪ Neutral (NOISE BUFFER ZONE):** Price is consolidating within the ± dynamic buffer zone around the 200 EMA. Avoid buying or selling to prevent whipsaws.
            * **🔴 Bearish (SELL / CASH OUT):** Price is below the 200 EMA (minus dynamic buffer) with a downward slope.
            """)
        
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
          
