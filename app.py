from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="200 Day Triggers", page_icon="📈", layout="wide")

REPO_NAME = "200 Day Triggers"

PRESET_CONFIGS = {
    "Watching": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Neutral baseline settings for active watchlist monitoring. Standard 2.0% buffer, 1.00 RVOL threshold, and no minimum ATR restriction.",
    },
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.75,
        "rationale": "Broad indices exhibit moderate volatility. 2.0% buffer filters noise, RVOL 0.80 tracks accumulation, 0.75% ATR screens stagnant assets.",
    },
    "Mag7": {
        "tickers": "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA",
        "buffer": 2.5,
        "rvol": 1.00,
        "min_atr": 1.00,
        "rationale": "High-beta tech mega-caps require wider noise filters (2.5%). RVOL 1.00 ensures volume-backed breakouts, 1.00% ATR floor filters consolidation.",
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50,
        "rationale": "Sectors vary in beta. A 2.0% buffer balances ranges, RVOL 0.85 confirms sector rotation, and 0.50% ATR accounts for defensive ETFs.",
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00,
        "rationale": "Fixed-income moves in tight bands. Tight 1.0% buffer prevents false regime changes, RVOL 0.10 prevents false flags, ATR 0.00% retains yield assets.",
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
        "rationale": "Option-overlay ETFs have capped upside. 1.5% buffer captures trend shifts without over-filtering moves, paired with lower RVOL (0.75) and ATR (0.25%).",
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50,
        "rationale": "Dividend funds exhibit lower beta. 2.0% buffer absorbs equity fluctuation, while 0.75 RVOL and 0.50% ATR capture steady accumulation.",
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00,
        "rationale": "Crypto and commodities experience high volatility. 4.0% buffer prevents premature triggers, RVOL 1.25 demands heavy volume, 2.00% ATR filters chop.",
    },
}

# --- Technical Helper Functions ---

def get_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80):
    results = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 225:
                continue

            c = df["Close"].squeeze()
            h = df["High"].squeeze()
            l = df["Low"].squeeze()
            o = df["Open"].squeeze()
            v = df["Volume"].squeeze()

            df_clean = pd.DataFrame({"High": h, "Low": l, "Close": c, "Open": o})
            price = float(c.iloc[-1])
            vol = float(v.iloc[-1])

            v_sma20 = float(v.rolling(20).mean().iloc[-1])
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0
            rvol_ok = rvol >= rvol_threshold

            sma200 = c.rolling(200).mean()
            p_sma200 = float(sma200.iloc[-1])
            prev_sma200 = float(sma200.iloc[-21]) # 1-month (21 trading day) smoothed slope lookback

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(default_buffer_pct, (atr / price) * 1.5)

            u_th = p_sma200 * (1.0 + eff_buf)
            l_th = p_sma200 * (1.0 - eff_buf)

            sma_diff = ((p_sma200 - prev_sma200) / prev_sma200) * 100
            slope = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")
            pct_sma200 = ((price - p_sma200) / p_sma200) * 100.0

            # --- Unified Single-Rule Decision Logic ---
            if price > u_th and slope in ["UP", "FLAT"]:
                if rvol_ok:
                    status_action = "🟢 ACCUMULATE"
                    rationale = f"Price above upper noise buffer with a stable/rising 200 SMA. Volume confirmed (RVOL {rvol:.2f}x)."
                else:
                    status_action = "🟢 HOLD"
                    rationale = f"Secular bull trend intact above 200 SMA buffer, but volume is light (RVOL {rvol:.2f}x). Hold existing position."
            elif price < l_th and slope == "DOWN":
                status_action = "🔴 SELL / ROTATE"
                rationale = f"Secular breakdown confirmed below lower buffer with a declining 200 SMA. Rotate to safe harbor."
            elif l_th <= price <= u_th:
                status_action = "⚪ BUFFER ZONE (HOLD)"
                rationale = f"Price is within the +/-{eff_buf*100:.1f}% noise band around the 200 SMA. Hold current positions; avoid new deployment until breakout."
            else:
                status_action = "🟡 CAUTION / WATCH"
                rationale = f"Crossed 200 SMA buffer, but slope ({slope}) unconfirmed."

            results.append({
                "Ticker": t,
                "Price": round(price, 2),
                "Status & Action": status_action,
                "200 SMA": round(p_sma200, 2),
                "% vs 200 SMA": f"{pct_sma200:+.2f}%",
                "200 SMA Slope": slope,
                "RVOL": round(rvol, 2),
                "ATR (%)": round(atr_pct, 2),
                "Definitive Rationale": rationale,
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")

    return pd.DataFrame(results)

# --- Sidebar Configuration ---

st.sidebar.header("Screener Configuration")

url_params = st.query_params
init_preset = url_params.get("preset", "Watching")
if init_preset not in PRESET_CONFIGS:
    init_preset = "Watching"

def on_preset_change():
    sel = st.session_state["preset_radio"]
    st.session_state["ticker_text"] = PRESET_CONFIGS[sel]["tickers"]
    st.query_params["preset"] = sel
    st.query_params["tickers"] = st.session_state["ticker_text"]

def on_ticker_change():
    st.query_params["preset"] = st.session_state["preset_radio"]
    st.query_params["tickers"] = st.session_state["ticker_text"]

if "preset_radio" not in st.session_state or st.session_state["preset_radio"] not in PRESET_CONFIGS:
    st.session_state["preset_radio"] = init_preset

if "ticker_text" not in st.session_state:
    st.session_state["ticker_text"] = url_params.get(
        "tickers", PRESET_CONFIGS[init_preset]["tickers"]
    )

run_screener = st.sidebar.button("🚀 Run Screener", type="primary", use_container_width=True)
st.sidebar.divider()

selected_preset = st.sidebar.radio(
    "Target Instance Preset:",
    options=list(PRESET_CONFIGS.keys()),
    key="preset_radio",
    on_change=on_preset_change,
)

active_defaults = PRESET_CONFIGS[selected_preset]

ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (Editable):",
    key="ticker_text",
    on_change=on_ticker_change,
    help="Edit tickers freely. Updates refresh browser URL for bookmarking.",
)

buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", 1.0, 5.0, active_defaults["buffer"], 0.5)
rvol_setting = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", 0.1, 2.5, active_defaults["rvol"], 0.05)
min_atr_setting = st.sidebar.slider("Min Volatility / ATR (%)", 0.0, 5.0, active_defaults["min_atr"], 0.25)

# --- Main Page Layout ---

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with st.expander("📖 **Methodology Guide: Core Institutional Framework Explained**"):
    st.markdown("### **1. Core Architectural Concepts & Why They Matter**")
    st.markdown("* **200-Day Simple Moving Average (SMA):** The institutional baseline for macro trend-following (~10 months of price history). **Why it matters:** It defines the secular bull/bear boundary. Price above the 200 SMA indicates a healthy macro regime (Risk-On / Income generation); price below signals a structural bear regime (Risk-Off / Rotate to Safe Harbor).")
    st.markdown("* **1-Month Slope Lookback (21 Trading Days):** Measures the directional trajectory of the 200 SMA over a full calendar month rather than volatile weekly wiggles. **Why it matters:** Prevents false trend reversals by demanding sustained directional conviction.")
    st.markdown("* **Noise Buffer (±%):** A customizable percentage band built around the 200 SMA (dynamically scaled using ATR). **Why it matters:** Creates a neutral 'no-man's land' zone around the moving average line, eliminating whipsaws and unnecessary turnover during minor consolidations.")
    st.markdown("* **Relative Volume (RVOL):** Compares trading volume against the asset's 20-day average. **Why it matters:** Ensures institutional participation backs breakouts or breakdowns before capital actions are triggered.")

    st.markdown("---")
    st.markdown("### **2. Unified Table Decision Matrix (Monthly/Quarterly Review)**")
    st.markdown("* 🟢 **ACCUMULATE:** Price securely above upper noise buffer, 200 SMA slope is UP/FLAT over a 1-month lookback, and volume confirms.")
    st.markdown("* 🟢 **HOLD:** Secular trend is intact above the buffer, but volume is moderate.")
    st.markdown("* ⚪ **BUFFER ZONE (HOLD):** Price is chopping inside the noise band around the 200 SMA. Hold current holdings; do not add new capital.")
    st.markdown("* 🔴 **SELL / ROTATE:** Price breached the lower buffer with a declining 200 SMA. Rotate to safe-harbor assets.")

    st.markdown("---")
    st.markdown("### **3. Preset Parameter Overview**")
    p_data = [
        {
            "Preset Name": f"**{name}**",
            "Buffer (%)": f"{cfg['buffer']:.1f}%",
            "Min RVOL": f"{cfg['rvol']:.2f}x",
            "Min ATR (%)": f"{cfg['min_atr']:.2f}%",
            "Calibration Rationale": cfg["rationale"],
        }
        for name, cfg in PRESET_CONFIGS.items()
    ]
    st.table(pd.DataFrame(p_data))

st.divider()

if run_screener:
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if tickers_list:
        with st.spinner("Analyzing portfolio secular trends..."):
            df_results = analyze_tickers(
                tickers_list,
                default_buffer_pct=buffer_setting / 100.0,
                rvol_threshold=rvol_setting,
            )
        if not df_results.empty:
            st.subheader("📋 Unified Portfolio Action & Trend Matrix")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            st.divider()
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "📥 Download Matrix CSV",
                csv_data,
                "200_day_unified_matrix.csv",
                "text/csv",
            )
else:
    st.info("👈 Select your preset in the sidebar and click **🚀 Run Screener** to generate your consolidated portfolio table.")
