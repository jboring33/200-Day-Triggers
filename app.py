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
        "tickers": "",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Your blank personal active watchlist for monitoring custom setups.",
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
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU",
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
    fallback_scored = []
    
    for t in live_universe:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            df = clean_df_columns(df)
            if df.empty or len(df) < 65:
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

            has_200 = len(c) >= 200
            if has_200:
                baseline_series = c.rolling(200).mean()
                p_baseline = float(baseline_series.iloc[-1])
                prev_baseline = float(baseline_series.iloc[-21])
                sma_diff = ((p_baseline - prev_baseline) / prev_baseline) * 100
                slope_val = "UP" if sma_diff > 0.05 else ("DOWN" if sma_diff < -0.05 else "FLAT")
            else:
                w = min(50, len(c) - 10)
                baseline_series = c.rolling(w).mean()
                p_baseline = float(baseline_series.iloc[-1])
                prev_baseline = float(baseline_series.iloc[-max(5, len(c)//4)])
                slope_val = "UP" if p_baseline >= prev_baseline else "FLAT"

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(0.02, (atr / price) * 1.5)
            u_th = p_baseline * (1.0 + eff_buf)

            ema20_s = c.ewm(span=20, adjust=False).mean()
            sma50_s = c.rolling(50).mean()
            ema20 = float(ema20_s.iloc[-1]) if len(c) >= 20 else price
            sma50 = float(sma50_s.iloc[-1]) if len(c) >= 50 else p_baseline
            macd_hist = get_macd(c) if len(c) >= 26 else 0.0

            score = ((price - p_baseline) / p_baseline) * max(0.5, rvol)

            if price >= p_baseline and slope_val != "DOWN":
                fallback_scored.append((t, score))

            is_uptrend = (slope_val in ["UP", "FLAT"])
            above_buffer = (price >= p_baseline)
            short_term_bullish = (price >= ema20) and (ema20 >= sma50) and (macd_hist >= -0.01)
            valid_volume_volatility = (rvol >= 0.70) and (atr_pct >= 0.30)

            if is_uptrend and above_buffer and short_term_bullish and valid_volume_volatility:
                scored.append((t, score))
        except Exception:
            continue
            
    scored.sort(key=lambda x: x[1], reverse=True)
    picks = [item[0] for item in scored[:top_n]]
    
    if not picks and fallback_scored:
        fallback_scored.sort(key=lambda x: x[1], reverse=True)
        picks = [item[0] for item in fallback_scored[:top_n]]

    return picks if picks else ["SPY", "QQQ", "NVDA", "SMH", "XLK", "AMZN", "MSFT", "IBIT"]

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80, min_atr_threshold=0.50, is_recommendations=False):
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

            has_200 = len(c) >= 200
            if has_200:
                baseline_series = c.rolling(200).mean()
                p_baseline = float(baseline_series.iloc[-1])
                prev_baseline_21 = float(baseline_series.iloc[-21])
                prev_baseline_daily = float(baseline_series.iloc[-2])
            else:
                w = min(50, len(c) - 10)
                baseline_series = c.rolling(w).mean()
                p_baseline = float(baseline_series.iloc[-1])
                prev_baseline_21 = float(baseline_series.iloc[-max(5, len(c)//4)])
                prev_baseline_daily = float(baseline_series.iloc[-2]) if len(baseline_series) >= 2 else p_baseline

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(default_buffer_pct, (atr / price) * 1.5)

            u_th = p_baseline * (1.0 + eff_buf)
            l_th = p_baseline * (1.0 - eff_buf)

            sma_diff = ((p_baseline - prev_baseline_21) / prev_baseline_21) * 100
            if sma_diff > 0.05:
                slope_val = "UP"
            elif sma_diff < -0.05:
                slope_val = "DOWN"
            else:
                slope_val = "FLAT"

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

            short_term_bullish = (price > ema20_curr) and (ema20_curr >= sma50_curr) and (macd_hist > 0)
            short_term_bearish = (price < ema20_curr) and (ema20_curr <= sma50_curr) and (macd_hist < 0)

            if is_recommendations:
                if price >= p_baseline and slope_val in ["UP", "FLAT"]:
                    status_action = "🟢 ACCUMULATE"
                else:
                    status_action = "🟢 HOLD"
            else:
                if price > u_th and slope_val == "UP" and rvol_ok and atr_ok and short_term_bullish:
                    status_action = "🟢 ACCUMULATE"
                elif price > u_th and slope_val == "UP":
                    status_action = "🟢 HOLD"
                elif l_th <= price <= u_th:
                    status_action = "⚪ BUFFER ZONE"
                elif price < l_th and short_term_bearish:
                    status_action = "🟡 EARLY WARNING (TRIM)"
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

def on_text_input_change():
    sel = st.session_state["preset_radio"]
    key = f"ticker_box_{sel}"
    if key in st.session_state:
        val = st.session_state[key]
        st.session
