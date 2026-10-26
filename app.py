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
    return
