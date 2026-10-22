from datetime import datetime
import json
import os
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="200 Day Triggers",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

REPO_NAME = "200 Day Triggers"
STATE_FILE = "watchlist_state.json"

ASSET_DESCRIPTIONS = {
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
    "BRK-B": "Berkshire Hathaway Inc. Class B",
    "BRK.B": "Berkshire Hathaway Inc. Class B",
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
    "QQQI": "NEOS Nasdaq-100 High Income ETF",
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
        "rationale": "Your active watchlist for monitoring custom setups.",
    },
    "Recommendations": {
        "tickers": "",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.50,
        "rationale": "Autonomous buy engine scanning strictly for verified ACCUMULATE breakout candidates.",
    },
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.75,
        "rationale": "Broad indices exhibiting moderate volatility.",
    },
    "Mag7": {
        "tickers": "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA",
        "buffer": 2.5,
        "rvol": 1.00,
        "min_atr": 1.00,
        "rationale": "High-beta tech mega-caps requiring wider noise filters.",
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50,
        "rationale": "Broad sector and industry tracking pool.",
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00,
        "rationale": "Fixed-income assets in tight bands.",
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, QQQI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
        "rationale": "Option-overlay funds with income focus.",
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50,
        "rationale": "Dividend funds with lower beta.",
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00,
        "rationale": "Crypto and commodities with high volatility.",
    },
}

def load_persistent_state():
    data = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass
    
    for name, cfg in PRESET_CONFIGS.items():
        if name not in data:
            data[name] = cfg["tickers"]
            
    if "Covered Call ETFs" in data and "QQQI" not in data["Covered Call ETFs"]:
        data["Covered Call ETFs"] = PRESET_CONFIGS["Covered Call ETFs"]["tickers"]
        
    return data

def save_persistent_state(state_dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_dict, f, indent=4)
    except Exception:
        pass

def clean_df_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])

def get_macd(close):
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(hist.iloc[-1])

def get_asset_description(symbol):
    if symbol in ASSET_DESCRIPTIONS:
        return ASSET_DESCRIPTIONS[symbol]
    try:
        t = yf.Ticker(symbol)
        info = t.info
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return f"US Listed Asset ({symbol})"

@st.cache_data(ttl=3600)
def fetch_open_world_universe():
    return [
        "SPY", "QQQ", "IWM", "SMH", "XLF", "XLE", "XLK", "XLV", "XLI", 
        "XLP", "XLY", "XLC", "XLB", "XLU", "XLRE", "IGV", "ITA", "XHB", 
        "XME", "XOP", "KRE", "SOXX", "XBI", "GLD", "SLV", "MTUM", "QUAL", 
        "IWF", "IWD", "DGRO", "VWO", "EFA", "NVDA", "AAPL", "MSFT", "AMZN", 
        "GOOGL", "META", "TSLA", "SCHD", "JEPI", "QQQI", "JEPQ", "DJIA", "VEA",
        "TLT", "HYG", "LQD", "IBIT", "FBTC", "IAUM"
    ]

def discover_open_world_candidates(top_n=10):
    live_universe = fetch_open_world_universe()
    scored = []
    
    for t in live_universe:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            df = clean_df_columns(df)
            if df.empty or len(df) < 225:
                continue
            
            c = df["Close"].squeeze()
            v = df["Volume"].squeeze()
            h = df["High"].squeeze()
            l = df["Low"].squeeze()
            o = df["Open"].squeeze()

            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
            if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
            if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]
            if isinstance(o, pd.DataFrame): o = o.iloc[:, 0]

            df_clean = pd.DataFrame({"High": h, "Low": l, "Close": c, "Open": o})
            price = float(c.iloc[-1])
            vol = float(v.iloc[-1])
            v_sma20 = float(v.rolling(20).mean().iloc[-1])
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0

            sma200 = c.rolling(200).mean()
            p_sma200 = float(sma200.iloc[-1])
            prev_sma200 = float(sma200.iloc[-21])
            sma_diff = ((p_sma200 - prev_sma200) / prev_sma200) * 100
            slope_val = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(0.02, (atr / price) * 1.5)
            u_th = p_sma200 * (1.0 + eff_buf)

            ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            macd_hist = get_macd(c)

            # Strict Accumulation Check for Discovery
            is_uptrend = (slope_val == "UP")
            above_buffer = (price > u_th)
            short_term_bullish = (price > ema20) and (ema20 > sma50) and (macd_hist > 0)
            valid_volume_volatility = (rvol >= 0.80) and (atr_pct >= 0.50)

            if is_uptrend and above_buffer and short_term_bullish and valid_volume_volatility:
                score = ((price - p_sma200) / p_sma200) * rvol
                scored.append((t, score))
        except Exception:
            continue
            
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored[:top_n]]

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80, min_atr_threshold=0.50):
    results = []
    for t in tickers:
        try:
            description = get_asset_description(t)
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            df = clean_df_columns(df)
            if df.empty or len(df) < 225:
                continue

            c = df["Close"].squeeze()
            h = df["High"].squeeze()
            l = df["Low"].squeeze()
            o = df["Open"].squeeze()
            v = df["Volume"].squeeze()

            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
            if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]
            if isinstance(o, pd.DataFrame): o = o.iloc[:, 0]
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]

            df_clean = pd.DataFrame({"High": h, "Low": l, "Close": c, "Open": o})
            price = float(c.iloc[-1])
            vol = float(v.iloc[-1])

            v_sma20 = float(v.rolling(20).mean().iloc[-1])
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0

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
                slope_label, slope_val = "🟢 UP", "UP"
            elif sma_diff < -0.05:
                slope_label, slope_val = "🔴 DOWN", "DOWN"
            else:
                slope_label, slope_val = "⚪ FLAT", "FLAT"

            pct_sma200 = ((price - p_sma200) / p_sma200) * 100.0
            rvol_ok = rvol >= rvol_threshold
            atr_ok = atr_pct >= min_atr_threshold

            rvol_label = f"🟢 {rvol:.2f}x" if rvol_ok else f"🟡 {rvol:.2f}x"
            atr_label = f"🟢 {atr_pct:.2f}%" if atr_ok else f"🔴 {atr_pct:.2f}%"
            
            buf_pct_val = eff_buf * 100.0
            target_pct_val = default_buffer_pct * 100.0
            if buf_pct_val <= (target_pct_val * 1.25):
                buffer_label = f"🟢 {buf_pct_val:.2f}%"
            elif buf_pct_val <= (target_pct_val * 1.75):
                buffer_label = f"🟡 {buf_pct_val:.2f}%"
            else:
                buffer_label = f"🔴 {buf_pct_val:.2f}%"

            # Short-term momentum & moving average evaluation
            ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            macd_hist = get_macd(c)

            short_term_bullish = (price > ema20) and (ema20 > sma50) and (macd_hist > 0)

            # Strict ACCUMULATE definition:
            # 1. Price above upper noise buffer threshold
            # 2. 200 SMA slope is strictly UP
            # 3. RVOL and ATR criteria met
            # 4. Short-term MAs (20 EMA / 50 SMA) and MACD histogram confirming upward slope
            if price > u_th and slope_val == "UP" and rvol_ok and atr_ok and short_term_bullish:
                status_action = "🟢 ACCUMULATE"
            elif price > u_th and slope_val == "UP":
                status_action = "🟢 HOLD"
            elif l_th <= price <= u_th:
                status_action = "⚪ BUFFER ZONE"
            elif price < l_th and slope_val == "DOWN":
                status_action = "🔴 SELL / ROTATE"
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
                "Buffer (%)": buffer_label,
                "RVOL": rvol_label,
                "ATR (%)": atr_label,
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")

    return pd.DataFrame(results)

# --- Session State & URL Params Initialization ---

url_params = st.query_params
init_preset = url_params.get("preset", "Watching")
if init_preset not in PRESET_CONFIGS:
    init_preset = "Watching"

if "preset_tickers" not in st.session_state:
    st.session_state["preset_tickers"] = load_persistent_state()
else:
    for name, cfg in PRESET_CONFIGS.items():
        if name not in st.session_state["preset_tickers"] or (name != "Recommendations" and not st.session_state["preset_tickers"][name].strip()):
            st.session_state["preset_tickers"][name] = cfg["tickers"]

if "preset_radio" not in st.session_state or st.session_state["preset_radio"] not in PRESET_CONFIGS:
    st.session_state["preset_radio"] = init_preset

def on_preset_change():
    sel = st.session_state["preset_radio"]
    st.query_params["preset"] = sel
    st.query_params["tickers"] = st.session_state["preset_tickers"][sel]

# --- Sidebar Configuration (Read Me & Screener Parameters) ---

with st.sidebar:
    if st.button("🔄 Reset All Watchlists to Defaults", use_container_width=True, type="secondary"):
        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
            except Exception:
                pass
        for name, cfg in PRESET_CONFIGS.items():
            st.session_state["preset_tickers"][name] = cfg["tickers"]
        st.success("Watchlists reset to defaults!")
        st.rerun()

    st.divider()

    with st.expander("📖 **Read Me**", expanded=False):
        st.markdown("### **1. Quick Start Guide**")
        st.markdown("1. **Select a Watchlist Preset**: Use the horizontal preset buttons at the top of the main panel to choose a curated asset pool.")
        st.markdown("2. **Customize Tickers**: The text box below the preset buttons displays active tickers. Add or remove comma-separated stock or ETF symbols freely. All changes are saved automatically.")
        st.markdown("3. **Run the Screener**: Click the **🚀 Run Screener** button directly below the ticker input box to analyze secular trends.")
        st.markdown("4. **Fine-Tune Parameters (Optional)**: Adjust core thresholds like the base buffer noise filter, minimum RVOL, or minimum ATR below.")

        st.markdown("---")
        st.markdown("### **2. Special Mode: 'Recommendations'**")
        st.markdown("If you select the **Recommendations** preset, the app acts as an autonomous open-world discovery engine:")
        st.markdown("1. Click the **🔍 Scan & Refresh** button that appears on the left side of the ticker input box.")
        st.markdown("2. The engine scans liquid U.S. ETFs and strictly filters for assets exhibiting a rising 200 SMA slope, upper buffer breakout, confirmed RVOL/ATR, and positive short-term momentum (20 EMA > 50 SMA & MACD positive).")
        st.markdown("3. Once populated, click **🚀 Run Screener** to evaluate them.")

        st.markdown("---")
        st.markdown("### **3. Core Architectural Concepts**")
        st.markdown("* **200-Day Simple Moving Average (SMA):** The institutional baseline for macro trend-following (~10 months of price history). Defines the secular bull/bear boundary.")
        st.markdown("* **1-Month Slope Lookback (21 Trading Days):** Measures the directional trajectory of the 200 SMA over a full calendar month to prevent false trend reversals.")
        st.markdown("* **Noise Buffer (±%):** A customizable percentage band built around the 200 SMA (dynamically scaled using ATR) to eliminate whipsaws during minor consolidations.")
        st.markdown("* **Relative Volume (RVOL):** Compares trading volume against the asset's 20-day average to ensure institutional participation.")

        st.markdown("---")
        st.markdown("### **4. Status & Action Classification Matrix**")
        st.markdown("Primary status is determined strictly by absolute price position relative to the 200 SMA upper and lower noise buffer thresholds:")
        st.markdown("* 🟢 **ACCUMULATE:** Price is above the Upper Threshold, 200 SMA Slope is **UP**, RVOL and ATR satisfy minimum parameters, and short-term MAs (20/50) + MACD confirm upward momentum.")
        st.markdown("* 🟢 **HOLD:** Price is above the Upper Threshold and 200 SMA Slope is **UP**, but short-term volume or momentum indicators are unconfirmed.")
        st.markdown("* ⚪ **BUFFER ZONE:** Price is between the Lower and Upper Thresholds. Capital actions are held neutral to prevent whipsaws.")
        st.markdown("* 🟡 **CAUTION:** Price is outside the noise band, but diverges from the moving average slope trajectory.")
        st.markdown("* 🔴 **SELL / ROTATE:** Price is below the Lower Threshold and 200 SMA Slope is **DOWN**. Triggers risk mitigation and rotation into safe-harbor assets.")

        st.markdown("---")
        st.markdown("### **5. Indicator Visual Badging Logic**")
        st.markdown("* **200 SMA Slope:** 🟢 **UP** (> +0.05%), ⚪ **FLAT** (-0.05% to +0.05%), 🔴 **DOWN** (< -0.05%).")
        st.markdown("* **Buffer (%):** 🟢 **Green** if effective buffer <= 1.25x target baseline; 🟡 **Yellow** if <= 1.75x; 🔴 **Red** if wider than 1.75x.")
        st.markdown("* **RVOL:** 🟢 **Green** if >= minimum threshold; 🟡 **Yellow** if below.")
        st.markdown("* **ATR (%):** 🟢 **Green** if >= minimum threshold; 🔴 **Red** if below.")

    st.header("Screener Parameters")
    st.divider()

selected_preset = st.session_state["preset_radio"]
active_defaults = PRESET_CONFIGS[selected_preset]

if selected_preset == "Recommendations":
    st.sidebar.info("💡 **Autonomous Mode**: Recommendations evaluate each ticker individually based on live market discovery.")
    buffer_setting = 2.0
    rvol_setting = 0.80
    min_atr_setting = 0.50
else:
    buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", 1.0, 5.0, active_defaults["buffer"], 0.5)
    rvol_setting = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", 0.1, 2.5, active_defaults["rvol"], 0.05)
    min_atr_setting = st.sidebar.slider("Min Volatility / ATR (%)", 0.0, 5.0, active_defaults["min_atr"], 0.25)

# --- Main Layout ---

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# --- Top Header Preset Selector Bar ---
selected_preset = st.radio(
    "Preset Selection",
    options=list(PRESET_CONFIGS.keys()),
    key="preset_radio",
    on_change=on_preset_change,
    horizontal=True,
    label_visibility="collapsed"
)

# --- Ticker Input Row with Unique Preset Keys to Prevent Blank Screens ---
current_tickers_val = st.session_state["preset_tickers"].get(selected_preset, "")

if selected_preset == "Recommendations":
    col_scan, col_input = st.columns([1, 3])
    with col_scan:
        if st.button("🔍 Scan & Refresh", use_container_width=True, type="secondary"):
            with st.spinner("Executing Strict Accumulation Discovery..."):
                new_picks = discover_open_world_candidates(top_n=10)
                if new_picks:
                    st.session_state["preset_tickers"]["Recommendations"] = ", ".join(new_picks)
                    save_persistent_state(st.session_state["preset_tickers"])
                    st.success("Recommendations populated with top ACCUMULATE candidates!")
                    st.rerun()
                else:
                    st.warning("No qualifying accumulation candidates found in current market scan.")
    with col_input:
        ticker_input = st.text_input(
            "Watchlist Tickers (Editable):",
            value=current_tickers_val,
            key=f"ticker_box_{selected_preset}",
            help="Edit tickers freely. Changes are saved instantly.",
        )
else:
    ticker_input = st.text_input(
        "Watchlist Tickers (Editable):",
        value=current_tickers_val,
        key=f"ticker_box_{selected_preset}",
        help="Edit tickers freely. Changes are saved instantly.",
    )

# Instantly sync and persist changes
if ticker_input != current_tickers_val:
    st.session_state["preset_tickers"][selected_preset] = ticker_input
    save_persistent_state(st.session_state["preset_tickers"])
    st.query_params["preset"] = selected_preset
    st.query_params["tickers"] = ticker_input

# --- Action Button directly below Watchlist Ticker entry box ---
run_screener = st.button("🚀 Run Screener", type="primary", use_container_width=True)

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
            
            st.markdown(
                """
                <style>
                dataframe, table, th, td {
                    white-space: normal !important;
                    word-wrap: break-word !important;
                }
                td:nth-child(2), th:nth-child(2) {
                    min-width: 250px;
                    max-width: 350px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

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
    if selected_preset == "Recommendations":
        st.info("💡 **Workflow Notice**: Click **🔍 Scan & Refresh** first to dynamically scan and load active **ACCUMULATE** breakout candidates, then click **🚀 Run Screener** to evaluate them.")
    else:
        st.info("💡 Enter your tickers above and click **🚀 Run Screener** to generate your consolidated portfolio table.")
