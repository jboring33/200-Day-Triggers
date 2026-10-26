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
CONFIG_FILE = "config.json"

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg_data = json.load(f)
            ASSET_DESCRIPTIONS = cfg_data.get("asset_descriptions", {})
            PRESET_CONFIGS = cfg_data.get("preset_configs", {})
    except Exception:
        ASSET_DESCRIPTIONS = {}
        PRESET_CONFIGS = {"Watching": {"tickers": "", "rationale": "Default watchlist"}}
else:
    ASSET_DESCRIPTIONS = {}
    PRESET_CONFIGS = {"Watching": {"tickers": "", "rationale": "Default watchlist"}}

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
        "GOOGL", "META", "TSLA", "SCHD", "JEPI", "QQQI", "SPYI", "JEPQ", "DJIA", "VEA",
        "TLT", "HYG", "LQD", "IBIT", "FBTC", "IAUM"
    ]

def discover_open_world_candidates(top_n=10, buffer_pct=0.02, rvol_th=0.80, atr_th=0.50):
    live_universe = fetch_open_world_universe()
    scored = []
    
    for t in live_universe:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            df = clean_df_columns(df)
            if df.empty or len(df) < 60:
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
            v_sma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0

            baseline_series = c.rolling(200).mean() if len(c) >= 200 else c.rolling(min(50, len(c)-10)).mean()
            p_baseline = float(baseline_series.iloc[-1])
            prev_baseline = float(baseline_series.iloc[-21]) if len(baseline_series) >= 21 else p_baseline
            sma_diff = ((p_baseline - prev_baseline) / prev_baseline) * 100
            slope_val = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(buffer_pct, (atr / price) * 1.5)
            u_th = p_baseline * (1.0 + eff_buf)

            ema20_s = c.ewm(span=20, adjust=False).mean()
            sma50_s = c.rolling(50).mean()
            ema20 = float(ema20_s.iloc[-1]) if len(c) >= 20 else price
            sma50 = float(sma50_s.iloc[-1]) if len(c) >= 50 else p_baseline
            macd_hist = get_macd(c) if len(c) >= 26 else 0.0

            rvol_ok = rvol >= rvol_th
            atr_ok = atr_pct >= atr_th
            short_term_bullish = (price >= ema20) and (macd_hist >= -0.01)

            if price > u_th and slope_val in ["UP", "FLAT"] and rvol_ok and atr_ok and short_term_bullish:
                score = ((price - p_baseline) / p_baseline) * rvol
                scored.append((t, score))
        except Exception:
            continue
            
    scored.sort(key=lambda x: x[1], reverse=True)
    picks = [item[0] for item in scored[:top_n]]
    return picks

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80, min_atr_threshold=0.50):
    results = []
    for t in tickers:
        try:
            description = get_asset_description(t)
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            df = clean_df_columns(df)
            if df.empty or len(df) < 60:
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

            v_sma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0

            baseline_series = c.rolling(200).mean() if len(c) >= 200 else c.rolling(min(50, len(c)-10)).mean()
            p_baseline = float(baseline_series.iloc[-1])
            prev_baseline_21 = float(baseline_series.iloc[-21]) if len(baseline_series) >= 21 else p_baseline
            prev_baseline_daily = float(baseline_series.iloc[-2]) if len(baseline_series) >= 2 else p_baseline

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(default_buffer_pct, (atr / price) * 1.5)

            u_th = p_baseline * (1.0 + eff_buf)
            l_th = p_baseline * (1.0 - eff_buf)

            sma_diff = ((p_baseline - prev_baseline_21) / prev_baseline_21) * 100
            slope_val = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")
            baseline_arrow = "🟢 ↗" if p_baseline >= prev_baseline_daily else "🔴 ↘"

            pct_baseline = ((price - p_baseline) / p_baseline) * 100.0
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

            ema20_series = c.ewm(span=20, adjust=False).mean()
            ema20_curr = float(ema20_series.iloc[-1]) if len(c) >= 20 else price
            ema20_prev = float(ema20_series.iloc[-2]) if len(c) >= 20 else ema20_curr
            ema20_arrow = "🟢 ↗" if ema20_curr >= ema20_prev else "🔴 ↘"

            sma50_series = c.rolling(50).mean()
            sma50_curr = float(sma50_series.iloc[-1]) if len(c) >= 50 else p_baseline
            sma50_prev = float(sma50_series.iloc[-2]) if len(c) >= 50 else sma50_curr
            sma50_arrow = "🟢 ↗" if sma50_curr >= sma50_prev else "🔴 ↘"

            macd_hist = get_macd(c) if len(c) >= 26 else 0.0
            macd_label = "🟢 Bullish" if macd_hist > 0 else "🔴 Bearish"

            short_term_bullish = (price >= ema20_curr) and (macd_hist >= -0.01)

            if price > u_th and slope_val in ["UP", "FLAT"] and rvol_ok and atr_ok and short_term_bullish:
                status_action = "🟢 ACCUMULATE"
            elif price > u_th and slope_val in ["UP", "FLAT"]:
                status_action = "🟢 HOLD"
            elif l_th <= price <= u_th:
                status_action = "⚪ BUFFER ZONE"
            elif price < l_th and slope_val == "DOWN":
                status_action = "🔴 SELL / ROTATE"
            else:
                status_action = "🟡 CAUTION"

            results.append({
                "Status & Action": status_action,
                "Ticker": t,
                "Description": description,
                "Price": round(price, 2),
                "200 SMA": f"{round(p_baseline, 2)} {baseline_arrow}",
                "% vs Baseline": f"{pct_baseline:+.2f}%",
                "50 SMA": f"{sma50_curr:.2f} {sma50_arrow}",
                "20 EMA": f"{ema20_curr:.2f} {ema20_arrow}",
                "MACD Trend": macd_label,
                "Buffer (%)": buffer_label,
                "Inst. Vol (RVOL)": rvol_label,
                "Volatility (ATR %)": atr_label,
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")

    return pd.DataFrame(results)

try:
    url_params = st.query_params
    init_preset = url_params.get("preset", "Watching")
    if not isinstance(init_preset, str):
        init_preset = init_preset[0] if len(init_preset) > 0 else "Watching"
except Exception:
    init_preset = "Watching"

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
    try:
        st.query_params["preset"] = sel
        st.query_params["tickers"] = st.session_state["preset_tickers"][sel]
    except Exception:
        pass

def on_text_input_change():
    sel = st.session_state["preset_radio"]
    key = f"ticker_box_{sel}"
    if key in st.session_state:
        val = st.session_state[key]
        st.session_state["preset_tickers"][sel] = val
        save_persistent_state(st.session_state["preset_tickers"])
        try:
            st.query_params["preset"] = sel
            st.query_params["tickers"] = val
        except Exception:
            pass

with st.sidebar:
    if st.button("🔄 Reset All Watchlists to Defaults", use_container_width=True, type="secondary"):
        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
            except Exception:
                pass
        for name, cfg in PRESET_CONFIGS.items():
            st.session_state["preset_tickers"][name] = cfg["tickers"]
            if f"ticker_box_{name}" in st.session_state:
                st.session_state[f"ticker_box_{name}"] = cfg["tickers"]
        st.success("Watchlists reset to defaults!")
        st.rerun()

    st.divider()

    with st.expander("📖 **Read Me**", expanded=False):
        st.markdown("### **Preset Rationales & Guide**")
        for p_name, p_cfg in PRESET_CONFIGS.items():
            st.markdown(f"* **{p_name}**: {p_cfg['rationale']}")

    st.header("Screener Parameters")
    st.divider()
    
    global_buffer = st.sidebar.slider("Base Buffer Noise Filter (%)", 1.0, 5.0, 2.0, 0.5)
    global_rvol = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", 0.1, 2.5, 0.80, 0.05)
    global_atr = st.sidebar.slider("Min Volatility / ATR (%)", 0.0, 5.0, 0.50, 0.25)

selected_preset = st.session_state["preset_radio"]

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

selected_preset = st.radio(
    "Preset Selection",
    options=list(PRESET_CONFIGS.keys()),
    key="preset_radio",
    on_change=on_preset_change,
    horizontal=True,
    label_visibility="collapsed"
)

current_tickers_val = st.session_state["preset_tickers"].get(selected_preset, "")

if selected_preset == "Recommendations":
    col_scan, col_input = st.columns([1, 3])
    with col_scan:
        if st.button("🔍 Scan & Refresh", use_container_width=True, type="secondary"):
            with st.spinner("Scanning Market for Genuine Breakouts..."):
                new_picks = discover_open_world_candidates(
                    top_n=10,
                    buffer_pct=global_buffer / 100.0,
                    rvol_th=global_rvol,
                    atr_th=global_atr
                )
                if new_picks:
                    picks_str = ", ".join(new_picks)
                    st.session_state["preset_tickers"]["Recommendations"] = picks_str
                    st.session_state["ticker_box_Recommendations"] = picks_str
                    save_persistent_state(st.session_state["preset_tickers"])
                    st.session_state["scan_feedback"] = ("success", f"Found {len(new_picks)} verified ACCUMULATE candidates!")
                else:
                    st.session_state["preset_tickers"]["Recommendations"] = ""
                    st.session_state["ticker_box_Recommendations"] = ""
                    save_persistent_state(st.session_state["preset_tickers"])
                    st.session_state["scan_feedback"] = ("warning", "⚠️ No qualifying ACCUMULATE breakout candidates found matching current parameters. No buy recommendations at this time.")
                st.rerun()
    with col_input:
        ticker_input = st.text_input(
            "Watchlist Tickers (Editable):",
            value=current_tickers_val,
            key=f"ticker_box_{selected_preset}",
            on_change=on_text_input_change,
        )
    
    # Persistent scan message block so it doesn't flash and disappear
    if "scan_feedback" in st.session_state:
        f_type, f_msg = st.session_state["scan_feedback"]
        if f_type == "success":
            st.success(f_msg)
        else:
            st.warning(f_msg)
else:
    ticker_input = st.text_input(
        "Watchlist Tickers (Editable):",
        value=current_tickers_val,
        key=f"ticker_box_{selected_preset}",
        on_change=on_text_input_change,
    )

if ticker_input != current_tickers_val:
    st.session_state["preset_tickers"][selected_preset] = ticker_input
    st.session_state[f"ticker_box_{selected_preset}"] = ticker_input
    save_persistent_state(st.session_state["preset_tickers"])
    try:
        st.query_params["preset"] = selected_preset
        st.query_params["tickers"] = ticker_input
    except Exception:
        pass

run_screener = st.button("🚀 Run Screener", type="primary", use_container_width=True)

st.divider()

if run_screener:
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if tickers_list:
        with st.spinner("Analyzing portfolio secular trends..."):
            df_results = analyze_tickers(
                tickers_list,
                default_buffer_pct=global_buffer / 100.0,
                rvol_threshold=global_rvol,
                min_atr_threshold=global_atr,
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
            st.warning("⚠️ No valid data could be retrieved for the specified symbols.")
    else:
        st.warning("⚠️ Please enter at least one ticker symbol above before running the screener.")
else:
    if selected_preset == "Recommendations":
        st.info("💡 **Workflow Notice**: Click **🔍 Scan & Refresh** to check for active buy candidates. If none meet strict breakout criteria, the warning banner will stay on screen.")
    else:
        st.info("💡 Enter your tickers above and click **🚀 Run Screener** to generate your consolidated portfolio table.")
