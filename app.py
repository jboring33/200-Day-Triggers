"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term (20 EMA / 50 SMA)
             Trend & Volatility Screening with ATR Dynamic Buffers, ATR% Filters, RVOL Volume Confirmation,
             Directional Volume Labels, Flyover Tooltips, Parameter Tuning Guidance, and URL Parameter Persistence.
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
    """Evaluates tickers across Macro (200 EMA), Short-Term (20 EMA / 50 SMA) trends, ATR% metrics, and RVOL volume logic."""
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
            
            # Volatility Calculations (ATR & ATR%)
            atr = float(calculate_atr(df_clean).iloc[-1])
            atr_pct = (atr / curr_price) * 100  # ATR as a percentage of price
            effective_buffer_pct = max(default_buffer_pct, (atr / curr_price) * 1.5)
            
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
                
            # Directional Volume Labeling
            if curr_price > upper_threshold and above_20_ema and above_50_sma:
                vol_label = "STRONG BUYING VOLUME" if rvol_confirmed else "LOW VOLUME"
            elif curr_price < lower_threshold and slope == "DOWN":
                vol_label = "STRONG SELLING VOLUME" if rvol_confirmed else "LOW VOLUME"
            else:
                vol_label = "HIGH VOLUME" if rvol_confirmed else "LOW VOLUME"

            # Dynamically format vol_str using vol_label
            vol_str = f"RVOL: {rvol:.2f}x ({vol_label})"
            pct_from_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100
            
            # Signal Logic with RVOL Confirmation
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
                    reason = f"Macro trend is UP (>200 EMA), but price is below 20 EMA/50 SMA. Wait for momentum reclaim."
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

            # Construct summary string with standard text dividers
            flyover_summary = (
                f"[{full_signal}] {reason} | Price: ${curr_price:.2f} | {vol_str} | "
                f"ATR%: {atr_pct:.2f}% | 20 EMA: ${curr_ema20:.2f} \vert{} 50 SMA:${curr_sma50:.2f} | "
                f"200 EMA: ${curr_ema200:.2f} | Dist 200EMA: {pct_from_ema200:+.2f}% | "
                f"Buffer: +/-{effective_buffer_pct*100:.1f}% | Short Trend: {short_term_momentum} | Cross: {ma_cross}"
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
            st.error(f"Error processing {ticker}: {e}")
            
    return pd.DataFrame(results)

# --- Streamlit Dashboard UI ---

st.title(f"📈 {REPO_NAME}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- URL Parameter Management & Session Binding ---
query_params = st.query_params
url_tickers = query_params.get("tickers", "")

if "watchlist_input" not in st.session_state:
    st.session_state["watchlist_input"] = url_tickers

st.sidebar.header("Screener Configuration")

ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (comma-separated):", 
    key="watchlist_input",
    placeholder="e.g. SPY, QQQ, NVDA, BTC-USD",
    help="Type tickers here and run. The URL bar will update so you can bookmark this custom run in your browser."
)

buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100
rvol_setting = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", min_value=0.1, max_value=2.5, value=1.25, step=0.05)
min_atr_setting = st.sidebar.slider(
    "Min Daily Volatility / ATR (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.0, 
    step=0.25,
    help="Filters out assets whose 14-day Average True Range is below this percentage of price."
)

clean_ticker_str = ", ".join([t.strip().upper() for t in ticker_input.split(",") if t.strip()])
if clean_ticker_str:
    st.query_params["tickers"] = clean_ticker_str

run_screener = st.sidebar.button("Run Screener", type="primary")

if run_screener or ("ran_once" in st.session_state and clean_ticker_str):
    st.session_state["ran_once"] = True
    
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if tickers_list:
        with st.spinner("Fetching market data and calculating indicators..."):
            df_results = analyze_enhanced_sma_strategy(
                tickers_list, 
                default_buffer_pct=buffer_setting,
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
                        help="Short trend status. See reference legend below for full definitions."
                    ),
                    "Ticker": st.column_config.TextColumn(
                        "Ticker", 
                        width="small"
                    ),
                    "Trigger Details": st.column_config.TextColumn(
                        "Trigger Details", 
                        width="large",
                        help="Hover over any cell to view full price, RVOL, volume classification, ATR%, 20 EMA, 50 SMA, 200 EMA, and volatility buffer metrics"
                    ),
                }
            )
            
            with st.expander("📖 Signal Reference Guide & Parameter Tuning", expanded=False):
                st.markdown(
                    "### Signal Definitions\n"
                    f"* **Bullish (BUY / BULLISH HOLD):** Price is above the 200 EMA (plus dynamic buffer), short-term MAs are aligned, and RVOL confirms **STRONG BUYING VOLUME** (>= {rvol_setting:.2f}x).\n"
                    "* **Pullback (MACRO BULL / WAIT FOR ENTRY):** Macro trend remains long-term bullish (>200 EMA), but price is pulling back below short-term MAs. Wait for momentum reclaim before entering.\n"
                    f"* **Warning (BULLISH / LOW VOLUME or CAUTION):** Price is above targets, but volume status is **LOW VOLUME** (below the {rvol_setting:.2f}x threshold), signaling low-volume breakout risk; or slope/buffer conditions are incomplete.\n"
                    "* **Neutral (NOISE BUFFER ZONE):** Price is consolidating within the noise buffer zone around the 200 EMA. Avoid buying or selling to prevent whipsaws.\n"
                    "* **Bearish (SELL / CASH OUT):** Price is below the 200 EMA (minus dynamic buffer) with a downward slope and **STRONG SELLING VOLUME**.\n\n"
                    "---\n\n"
                    "### Parameter Tuning Guide: What, When & Why\n\n"
                    f"#### 1. Base Buffer Noise Filter (%) [Current: {buffer_setting*100:.1f}%]\n"
                    "* **WHAT IT IS:** Sets a mandatory percentage 'dead zone' around the 200 EMA. The app automatically compares this base value against `1.5 * ATR%` and uses whichever is larger to build a dynamic ceiling and floor around the long-term trendline.\n"
                    "* **WHY USE IT:** Long-term trendlines suffer from 'whipsaws'—where daily noise temporarily pushes price a fraction of a percent above or below the line without a true trend change. The buffer forces price to prove a structural breakout before generating a signal.\n"
                    "* **WHEN TO LOWER (1.0% - 1.5%):**\n"
                    "  * **Low-Beta Assets & Broad Indexes:** Large-cap index ETFs (SPY, VTI, SCHD) move methodically. A 2% buffer may cause significantly delayed signals.\n"
                    "  * **Low-Volatility Regimes:** When the market VIX is under 15 and daily candle ranges are tight.\n"
                    "* **WHEN TO RAISE (3.0% - 5.0%):**\n"
                    "  * **High-Beta & Crypto Assets:** High-volatility growth stocks (NVDA, TSLA) or crypto (BTC, ETH) routinely swing 2%-4% in a single day. A tight buffer generates constant false buy/sell signals.\n"
                    "  * **Market Volatility Spikes:** During macro panics or high VIX regimes (>25), expansion of noise requires wider margins.\n\n"

                    f"#### 2. Min RVOL Breakout Confirmation (x) [Current: {rvol_setting:.2f}x]\n"
                    "* **WHAT IT IS:** Relative Volume (RVOL) compares current trading volume against the asset's 20-day average volume. A value of 1.25x means today's volume is 25% higher than normal.\n"
                    "* **WHY USE IT:** Institutional funds move markets; retail traders do not. Price moving above a trendline on light volume is often a bull trap. Requiring elevated RVOL ensures institutional backing on breakout signals.\n"
                    "* **WHEN TO LOWER (0.1x - 1.0x):**\n"
                    "  * **Off-Hours / Early Session Screening:** When scanning market data early in the trading session before full daily volume has accumulated.\n"
                    "  * **Broad Market ETFs & Ultra-Short Fixed Income:** Liquidity-rich ETFs (SPY, QQQ, SGOV) move with index rebalancing rather than retail volume spikes, meaning RVOL rarely spikes as aggressively as in individual equities.\n"
                    "* **WHEN TO RAISE (1.5x - 2.5x):**\n"
                    "  * **Earning Plays & High-Conviction Breakouts:** When filtering exclusively for explosive, high-confidence momentum movers where major institutional accumulation is required.\n\n"

                    f"#### 3. Min Daily Volatility / ATR (%) [Current: {min_atr_setting:.2f}%]\n"
                    "* **WHAT IT IS:** Uses the 14-day Average True Range expressed as a percentage of share price to measure daily percentage movement range.\n"
                    "* **WHY USE IT:** It acts as an activity gate. It filters out sluggish, range-bound assets that take months to move, allowing focus on assets with active daily ranges.\n"
                    "* **WHEN TO SET TO 0.0% (DISABLED):**\n"
                    "  * **Core Conservative Holdings & Cash Alternatives:** When evaluating capital preservation funds (SGOV, JPST), dividend ETFs (SPHD), or ultra-short fixed-income where daily price ranges are near zero.\n"
                    "* **WHEN TO RAISE (1.5% - 2.0%):**\n"
                    "  * **Tactical Equity & Swing Screening:** When seeking liquid growth stocks or sector ETFs that move enough daily to justify tactical swing positioning.\n"
                    "* **WHEN TO RAISE (3.0%+):**\n"
                    "  * **High-Beta / Momentum Trading:** When filtering strictly for rapid movers, leveraged ETFs (TQQQ, SOXL), or crypto assets."
                )
            
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="200_day_triggers_report.csv",
                mime="text/csv"
            )
        else:
            st.info("No tickers matched the current ATR% volatility filter criteria.")
    else:
        st.warning("Please enter at least one ticker in the sidebar to run the screener.")
else:
    st.info("👈 Enter your tickers in the sidebar and click 'Run Screener' to analyze.")
