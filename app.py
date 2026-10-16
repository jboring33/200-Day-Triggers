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
        "tickers": "",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Your blank personal active watchlist for monitoring custom setups.",
    },
    "Recommendations": {
        "tickers": "SMH, IWF, MTUM, QUAL, IGV, ITA, XHB, XOP, IWM, DGRO",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
        "rationale": "Autonomous discovery pool where each ticker is evaluated individually against its own adaptive ATR and trend profile.",
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

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80, min_atr_threshold=0.50, is_recommendations=False):
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

            sma200 = c.rolling(200).mean()
            p_sma200 = float(sma200.iloc[-1])
            prev_sma200 = float(sma200.iloc[-21])

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100

            if is_recommendations:
                eff_buf = max(0.015, (atr / price) * 1.25)
                rvol_threshold_val = 0.80
                min_atr_val = 0.00
            else:
                eff_buf = max(default_buffer_pct, (atr / price) * 1.5)
                rvol_threshold_val = rvol_threshold
                min_atr_val = min_atr_threshold

            u_th = p_sma200 * (1.0 + eff_buf)
            l_th =
            
