from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="200 Day Triggers", page_icon="📈", layout="wide")

REPO_NAME = "200 Day Triggers"
STATE_FILE = "watchlist_state.json"

ASSET_DESCRIPTIONS = {
    # Core & Existing Database Tickers
    "SPY": "S&P 500 Large-Cap ETF",
    "QQQ": "Nasdaq-100 Growth ETF",
    "DJIA": "Dow Jones Industrial Average ETF",
    "EMXC": "Emerging Markets ex-China ETF",
    "VEA": "Vanguard Developed Markets ETF",
    "NVDA": "NVIDIA Corporation (AI/Semiconductors)",
    "AAPL": "Apple Inc. (Consumer Tech)",
    "MSFT": "Microsoft Corporation (Cloud/Software)",
    "AMZN": "Amazon.com Inc. (E-Commerce/Cloud)",
    "GOOGL": "Alphabet Inc. (Search/AI)",
    "META": "Meta Platforms Inc. (Social Media/AI)",
    "TSLA": "Tesla Inc. (EV/Clean Energy)",
    "XLC": "Communication Services Sector ETF",
    "XLY": "Consumer Discretionary Sector ETF",
    "XLP": "Consumer Staples Sector ETF",
    "XLE": "Energy Sector ETF",
    "XLF": "Financial Sector ETF",
    "XLV": "Health Care Sector ETF",
    "XLI": "Industrial Sector ETF",
    "XLB": "Materials Sector ETF",
    "XLRE": "Real Estate Sector ETF",
    "XLK": "Technology Sector ETF",
    "XLU": "Utilities Sector ETF",
    "VGT": "Vanguard Information Technology ETF",
    "SGOV": "iShares 0-3 Month Treasury Bond ETF",
    "JPST": "JPMorgan Ultra-Short Income ETF",
    "JAAA": "JPMorgan Collateralized Loan Obligation ETF",
    "JBBB": "JPMorgan High Yield CLO ETF",
    "SCYB": "BNY Mellon High Yield Beta ETF",
    "JEPI": "JPMorgan Equity Premium Income ETF",
    "JEPQ": "JPMorgan Nasdaq Equity Premium Income ETF",
    "XYLD": "Global X S&P 500 Covered Call ETF",
    "QYLD": "Global X Nasdaq 100 Covered Call ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "SPHD": "Invesco S&P 500 High Dividend Low Vol ETF",
    "SCHD": "Schwab U.S. Dividend Equity ETF",
    "VYM": "Vanguard High Dividend Yield ETF",
    "FBTC": "Fidelity Wise Origin Bitcoin Fund",
    "IBIT": "iShares Bitcoin Trust",
    "IAUM": "iShares Gold Trust Micro",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    # Recommendation & Discovery Universe Descriptions
    "IWM": "iShares Russell 2000 ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "IWF": "iShares Russell 1000 Growth ETF",
    "IWD": "iShares Russell 1000 Value ETF",
    "DGRO": "iShares Core Dividend Growth ETF",
    "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "SMH": "VanEck Semiconductor ETF",
    "XSD": "SPDR S&P Semiconductor ETF",
    "IGV": "iShares Expanded Tech-Software Sector ETF",
    "ITA": "iShares U.S. Aerospace & Defense ETF",
    "XHB": "SPDR S&P Homebuilders ETF",
    "XME": "SPDR S&P Metals & Mining ETF",
    "XOP": "SPDR S&P Oil & Gas Exploration & Production ETF",
    "KRE": "SPDR S&P Regional Banking ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "SOXX": "iShares Semiconductor ETF",
    "XBI": "SPDR S&P Biotech ETF",
}

PRESET_CONFIGS = {
    "Watching": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Your personal active watchlist for monitoring core holdings and custom setups.",
    },
    "Recommendations": {
        "tickers": "SMH, IWF, MTUM, QUAL, IGV, ITA, XHB, XOP, IWM, DGRO",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Curated pool of top institutional buy candidates and factor/sector leaders, scannable for fresh deployment.",
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

DISCOVERY_UNIVERSE = [
    "IWM", "VWO", "EFA", "IWF", "IWD", "DGRO", "NOBL", "QUAL", "MTUM", 
    "SMH", "XSD", "IGV", "ITA", "XHB", "XME", "XOP", "KRE", "HYG", 
    "LQD", "TLT", "SOXX", "XBI"
]

# --- Persistence Helper Functions ---

def load_persistent_state():
    data = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass
    
    # Ensure all current presets from PRESET_CONFIGS exist
    for name, cfg in PRESET_CONFIGS.items():
        if name not in data or not data[name].strip():
            data[name] = cfg["tickers"]
            
    return data

def save_persistent_state(state_dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_dict, f, indent=4)
    except Exception:
        pass

# --- Technical & Discovery Helper Functions ---

def get_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])

def discover_top_candidates(existing_tickers, top_n=10):
    candidates = [t for t in DISCOVERY_UNIVERSE if t not in existing_tickers]
    scored = []
    for t in candidates:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 225:
                continue
            c = df["Close"].squeeze()
            price = float(c.iloc[-1])
            sma200 = c.rolling(200).mean()
            p_sma200 = float(sma200.iloc[-1])
            prev_sma200 = float(sma200.iloc[-21])
            
            sma_diff = ((p_sma200 - prev_sma200) / prev_sma200) * 100
            slope = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")
            pct_above = ((price - p_sma200) / p_sma200) * 100.0
            
            if price > p_sma200 and slope == "UP":
                scored.append((t, pct_above))
        except Exception:
            continue
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored[:top_n]]

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80, min_atr_threshold=0.50):
    results = []

    for t in tickers:
        try:
            description = ASSET_DESCRIPTIONS.get(t, f"Asset ({t})")

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
            prev_sma200 = float(sma200.iloc[-21])

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(default_buffer_pct, (atr / price) * 1.5)

            u_th = p_sma200 * (1.0 + eff_buf)
            l_th = p_sma200 * (1.0 - eff_buf)

            sma_diff = ((p_sma200 - prev_sma200) / prev_sma200) * 100
            
            if sma_diff > 0.05:
                slope_label = "🟢 UP"
                slope_val = "UP"
            elif sma_diff < -0.05:
                slope_label = "🔴 DOWN"
                slope_val = "DOWN"
            else:
                slope_label = "⚪ FLAT"
                slope_val = "FLAT"

            pct_sma200 = ((price - p_sma200) / p_sma200) * 100.0

            rvol_label = f"🟢 {rvol:.2f}x" if rvol_ok else f"🟡 {rvol:.2f}x"
            atr_label = f"🟢 {atr_pct:.2f}%" if atr_pct >= min_atr_threshold else f"🔴 {atr_pct:.2f}%"

            if price > u_th and slope_val in ["UP", "FLAT"]:
                if rvol_ok:
                    status_action = "🟢 ACCUMULATE"
                else:
                    status_action = "🟢 HOLD"
            elif price < l_th and slope_val == "DOWN":
                status_action = "🔴 SELL / ROTATE"
            elif l_th <= price <= u_th:
                status_action = "⚪ BUFFER ZONE"
            else:
                status_action = "🟡 CAUTION"

            results.append({
                "Ticker": t,
                "Description": description,
                "Price": round(price, 2),
                "Status & Action": status_action,
                "200 SMA": round(p_sma200, 2),
                "% vs 200 SMA": f"{pct_sma200:+.2f}%",
                "200 SMA Slope": slope_label,
                "RVOL": rvol_label,
                "ATR (%)": atr_label,
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

if "preset_tickers" not in st.session_state:
    st.session_state["preset_tickers"] = load_persistent_state()
else:
    # Safely merge any newly introduced presets if session state already existed
    for name, cfg in PRESET_CONFIGS.items():
        if name not in st.session_state["preset_tickers"] or not st.session_state["preset_tickers"][name].strip():
            st.session_state["preset_tickers"][name] = cfg["tickers"]

def on_preset_change():
    sel = st.session_state["preset_radio"]
    st.session_state["ticker_text"] = st.session_state["preset_tickers"][sel]
    st.query_params["preset"] = sel
    st.query_params["tickers"] = st.session_state["ticker_text"]

def on_ticker_change():
    sel = st.session_state["preset_radio"]
    val = st.session_state["ticker_text"]
    st.session_state["preset_tickers"][sel] = val
    save_persistent_state(st.session_state["preset_tickers"])
    st.query_params["preset"] = sel
    st.query_params["tickers"] = val

if "preset_radio" not in st.session_state or st.session_state["preset_radio"] not in PRESET_CONFIGS:
    st.session_state["preset_radio"] = init_preset

if "ticker_text" not in st.session_state:
    st.session_state["ticker_text"] = st.session_state["preset_tickers"][init_preset]

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
    help="Edit tickers freely. Edits are saved to server disk and survive app sleep cycles.",
)

if selected_preset == "Recommendations":
    if st.sidebar.button("🔍 Scan & Refresh Recommendations"):
        with st.spinner("Scanning global ETF universe for top bull-market candidates..."):
            all_current = set()
            for t_str in st.session_state["preset_tickers"].values():
                all_current.update([t.strip().upper() for t in t_str.split(",") if t.strip()])
            
            new_picks = discover_top_candidates(all_current, top_n=10)
            if new_picks:
                st.session_state["preset_tickers"]["Recommendations"] = ", ".join(new_picks)
                st.session_state["ticker_text"] = st.session_state["preset_tickers"]["Recommendations"]
                save_persistent_state(st.session_state["preset_tickers"])
                st.sidebar.success("Recommendations updated with top 10 current candidates!")
                st.rerun()
            else:
                st.sidebar.warning("No new qualifying candidates found.")

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
    st.markdown("* ⚪ **BUFFER ZONE:** Price is chopping inside the noise band around the 200 SMA. Hold current holdings; do not add new capital.")
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
                min_atr_threshold=min_atr_setting,
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
