"""
Repository: 200 Day Triggers
Description: Streamlit Web Application for Macro (200 EMA) and Short-Term Trend Screening.
Features: Interactive Preset Selection, Custom Ticker Overrides, and Dynamic URL Sync for Bookmarks.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="200 Day Triggers", page_icon="📈", layout="wide")

REPO_NAME = "200 Day Triggers"

PRESET_CONFIGS = {
    "Broad Market": {
        "tickers": "SPY, QQQ, DJIA, EMXC, VEA",
        "buffer": 2.0,
        "rvol": 0.80,
        "min_atr": 0.75,
    },
    "Sector ETFs": {
        "tickers": "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, VGT",
        "buffer": 2.0,
        "rvol": 0.85,
        "min_atr": 0.50,
    },
    "Bond ETFs": {
        "tickers": "SGOV, JPST, JAAA, JBBB, SCYB",
        "buffer": 1.0,
        "rvol": 0.10,
        "min_atr": 0.00,
    },
    "Covered Call ETFs": {
        "tickers": "JEPI, JEPQ, XYLD, QYLD",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
    },
    "Dividend ETFs": {
        "tickers": "USMV, SPHD, SCHD, VYM",
        "buffer": 2.0,
        "rvol": 0.75,
        "min_atr": 0.50,
    },
    "Speculative & Commodities": {
        "tickers": "FBTC, IBIT, IAUM, GLD, SLV",
        "buffer": 4.0,
        "rvol": 1.25,
        "min_atr": 2.00,
    },
    "Custom": {
        "tickers": "SPY, QQQ, NVDA, EMXC, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.00,
    },
}


def calculate_atr(df, window=14):
  high, low, close = df["High"], df["Low"], df["Close"]
  tr = pd.concat(
      [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
      axis=1,
  ).max(axis=1)
  return tr.rolling(window=window).mean()


def calculate_rsi(close, window=14):
  delta = close.diff()
  gain = delta.where(delta > 0, 0).rolling(window=window).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
  return 100 - (100 / (1 + (gain / loss)))


def calculate_adx(df, window=14):
  high, low, close = df["High"], df["Low"], df["Close"]
  up_move = high - high.shift(1)
  down_move = low.shift(1) - low
  plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
  minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
  tr = pd.concat(
      [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
      axis=1,
  ).max(axis=1)
  tr_smooth = tr.rolling(window=window).mean()
  plus_di = (
      100
      * (
          pd.Series(plus_dm, index=df.index).rolling(window=window).mean()
          / tr_smooth
      )
  )
  minus_di = (
      100
      * (
          pd.Series(minus_dm, index=df.index).rolling(window=window).mean()
          / tr_smooth
      )
  )
  dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
  return dx.rolling(window=window).mean()


def calculate_stochastic(df, k_window=14, d_window=3):
  low_min = df["Low"].rolling(window=k_window).min()
  high_max = df["High"].rolling(window=k_window).max()
  k_percent = 100 * ((df["Close"] - low_min) / (high_max - low_min))
  d_percent = k_percent.rolling(window=d_window).mean()
  return k_percent, d_percent


def calculate_obv_slope(close, volume, window=10):
  direction = np.sign(close.diff()).fillna(0)
  obv = (direction * volume).cumsum()
  return obv.diff(window)


def analyze_enhanced_sma_strategy(
    tickers,
    interval="1d",
    slope_window=5,
    default_buffer_pct=0.020,
    rvol_threshold=0.80,
):
  results = []
  lookback = "10y" if interval == "1wk" else "2y"

  for ticker in tickers:
    try:
      df = yf.download(
          ticker,
          period=lookback,
          interval=interval,
          auto_adjust=True,
          progress=False,
      )
      if df.empty or len(df) < 205:
        continue

      close = df["Close"].squeeze()
      high = df["High"].squeeze()
      low = df["Low"].squeeze()
      open_p = df["Open"].squeeze()
      volume = df["Volume"].squeeze()

      df_clean = pd.DataFrame(
          {"High": high, "Low": low, "Close": close, "Open": open_p}
      )
      curr_price = float(close.iloc[-1])
      curr_open = float(open_p.iloc[-1])
      curr_volume = float(volume.iloc[-1])

      vol_20_sma = float(volume.rolling(20).mean().iloc[-1])
      rvol = curr_volume / vol_20_sma if vol_20_sma > 0 else 1.0
      rvol_confirmed = rvol >= rvol_threshold

      is_green_day = curr_price >= curr_open
      if rvol >= 1.25 and is_green_day:
        volume_status = "BULLISH (HEAVY BUYING)"
      elif rvol >= 1.25 and not is_green_day:
        volume_status = "BEARISH (HEAVY SELLING)"
      else:
        volume_status = "NORMAL / MODERATE"

      ema20 = close.ewm(span=20, adjust=False).mean()
      sma50 = close.rolling(50).mean()
      ema200 = close.ewm(span=200, adjust=False).mean()

      curr_ema20 = float(ema20.iloc[-1])
      curr_sma50 = float(sma50.iloc[-1])
      curr_ema200 = float(ema200.iloc[-1])
      prev_ema200 = float(ema200.iloc[-(slope_window + 1)])

      atr = float(calculate_atr(df_clean).iloc[-1])
      atr_pct = (atr / curr_price) * 100
      effective_buffer = max(default_buffer_pct, (atr / curr_price) * 1.5)

      upper_th = curr_ema200 * (1.0 + effective_buffer)
      lower_th = curr_ema200 * (1.0 - effective_buffer)

      rsi = float(calculate_rsi(close).iloc[-1])
      adx = float(calculate_adx(df_clean).iloc[-1])
      k_stoch, d_stoch = calculate_stochastic(df_clean)
      curr_stoch = float(k_stoch.iloc[-1])

      obv_slope = float(calculate_obv_slope(close, volume).iloc[-1])
      obv_label = (
          "BULLISH (INFLOW)" if obv_slope > 0 else "BEARISH (OUTFLOW)"
      )

      # --- Bollinger Band Volatility Evaluation ---
      sma20 = close.rolling(20).mean()
      std20 = close.rolling(20).std()
      upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
      lower_bb = float((sma20 - (std20 * 2)).iloc[-1])

      bb_width_series = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
      curr_bb_width = float(bb_width_series.iloc[-1]) * 100
      avg_50_bb_width = float(bb_width_series.rolling(50).mean().iloc[-1]) * 100
      min_50_bb_width = float(bb_width_series.rolling(50).min().iloc[-1]) * 100

      is_squeeze = curr_bb_width <= (min_50_bb_width * 1.15)
      is_expanding = curr_bb_width > (avg_50_bb_width * 1.20)

      if is_squeeze:
        bb_commentary = (
            f"SQUEEZE ({curr_bb_width:.1f}%) - Tight compression before"
            " expansion"
        )
      elif is_expanding:
        bb_commentary = (
            f"EXPANDING HIGH VOLATILITY ({curr_bb_width:.1f}% vs"
            f" avg {avg_50_bb_width:.1f}%) - Wider than normal bands"
        )
      elif curr_price >= upper_bb:
        bb_commentary = (
            f"UPPER BAND STRETCH (${upper_bb:.2f}) - Highly Extended"
        )
      elif curr_price <= lower_bb:
        bb_commentary = (
            f"LOWER BAND STRETCH (${lower_bb:.2f}) - Oversold Conditions"
        )
      else:
        bb_commentary = (
            f"NORMAL RANGE ({curr_bb_width:.1f}% width) - Mid-band:"
            f" ${sma20.iloc[-1]:.2f}"
        )

      ema_diff = ((curr_ema200 - prev_ema200) / prev_ema200) * 100
      slope = (
          "UP" if ema_diff > 0.05 else ("DOWN" if ema_diff < -0.05 else "FLAT")
      )

      is_above_upper = curr_price > upper_th
      is_below_lower = curr_price < lower_th
      is_in_buffer = lower_th <= curr_price <= upper_th

      if is_above_upper and slope in ["UP", "FLAT"]:
        if adx < 20:
          status, signal = "⚪ Consolidating", "NO TREND"
          msg = f"ADX ({adx:.1f}) < 20 indicates non-trending price action."
        elif rsi > 70 or curr_stoch > 80:
          status, signal = "🟡 Overbought", "HOLD"
          msg = (
              f"RSI ({rsi:.1f}) or Stoch ({curr_stoch:.1f}) > thresholds"
              " indicates momentum extension."
          )
        elif obv_slope < 0 and rsi < 50:
          status, signal = "🔴 Divergence", "WARNING"
          msg = "Negative OBV volume outflow signals active distribution."
        elif is_squeeze:
          status, signal = "🟡 Squeeze", "WATCH"
          msg = f"Bollinger Band Squeeze active ({curr_bb_width:.1f}% width)."
        elif (
            curr_price > curr_ema20
            and curr_price > curr_sma50
            and rvol_confirmed
        ):
          status, signal = "🟢 Bullish", "ACCUMULATE"
          msg = f"Confirmed breakout above buffer (RVOL: {rvol:.2f}x)."
        elif curr_price <= curr_ema20 or curr_price <= curr_sma50:
          status, signal = "🟡 Pullback", "WAIT"
          msg = (
              "Macro trend is UP, but price is pulling back below short MAs."
          )
        else:
          status, signal = "🟡 Warning", "LOW VOLUME"
          msg = (
              "Price above buffer, but RVOL"
              f" ({rvol:.2f}x) lacks volume confirmation."
          )
      elif is_below_lower and slope == "DOWN":
        if adx < 20:
          status, signal = "⚪ Consolidating", "WEAK BEAR"
          msg = f"Price below 200 EMA, but weak ADX ({adx:.1f})."
        elif rvol_confirmed:
          status, signal = "🔴 Bearish", "SELL"
          msg = (
              "Selling volume confirmed below buffer"
              f" (RVOL: {rvol:.2f}x)."
          )
        else:
          status, signal = "🔴 Bearish", "SELL LOW VOL"
          msg = (
              "Price below noise buffer with DOWN slope"
              f" (RVOL: {rvol:.2f}x)."
          )
      elif is_in_buffer:
        status, signal = "⚪ Neutral", "BUFFER ZONE"
        msg = (
            "Price within"
            f" +/-{effective_buffer*100:.1f}% noise buffer of 200"
            " EMA."
        )
      else:
        status, signal = "🟡 Warning", "CAUTION"
        msg = (
            "Price crossed EMA but slope"
            f" ({slope}) does not confirm direction."
        )

      pct_ema200 = ((curr_price - curr_ema200) / curr_ema200) * 100.0

      results.append({
          "Ticker": ticker,
          "Status": status,
          "Signal": signal,
          "Price": round(curr_price, 2),
          "200 EMA": round(curr_ema200, 2),
          "% vs 200 EMA": f"{pct_ema200:+.2f}%",
          "RVOL": round(rvol, 2),
          "Volume Signal": volume_status,
          "OBV Signal": obv_label,
          "RSI": round(rsi, 1),
          "Stoch %K": round(curr_stoch, 1),
          "ADX": round(adx, 1),
          "ATR (%)": round(atr_pct, 2),
          "Bollinger Status": bb_commentary,
          "Trigger Rationale": msg,
      })
    except Exception as e:
      st.error(f"Error processing {ticker}: {str(e)}")

  return pd.DataFrame(results)


# --- UI Setup with Interactive URL & State Sync ---
st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.sidebar.header("Screener Configuration")

# 1. Parse URL parameters on initial load
url_params = st.query_params
initial_preset = url_params.get("preset", "Broad Market")
if initial_preset not in PRESET_CONFIGS:
  initial_preset = "Broad Market"

# 2. Setup Session State Callback Functions
def on_preset_change():
  selected = st.session_state["preset_select"]
  st.session_state["ticker_text"] = PRESET_CONFIGS[selected]["tickers"]
  st.query_params["preset"] = selected
  st.query_params["tickers"] = st.session_state["ticker_text"]


def on_ticker_change():
  st.query_params["preset"] = st.session_state["preset_select"]
  st.query_params["tickers"] = st.session_state["ticker_text"]


# 3. Initialize Session State values if not present
if "preset_select" not in st.session_state:
  st.session_state["preset_select"] = initial_preset

if "ticker_text" not in st.session_state:
  st.session_state["ticker_text"] = url_params.get(
      "tickers", PRESET_CONFIGS[initial_preset]["tickers"]
  )

# 4. Render Sidebar Controls with Callbacks
selected_preset = st.sidebar.selectbox(
    "Target Instance Preset:",
    options=list(PRESET_CONFIGS.keys()),
    key="preset_select",
    on_change=on_preset_change,
)

active_defaults = PRESET_CONFIGS[selected_preset]

ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (Editable):",
    key="ticker_text",
    on_change=on_ticker_change,
    help="Edit these tickers freely. Any changes instantly update the browser URL for easy bookmarking.",
)

buffer_setting = st.sidebar.slider(
    "Base Buffer Noise Filter (%)", 1.0, 5.0, active_defaults["buffer"], 0.5
)
rvol_setting = st.sidebar.slider(
    "Min RVOL Breakout Confirmation (x)",
    0.1,
    2.5,
    active_defaults["rvol"],
    0.05,
)
min_atr_setting = st.sidebar.slider(
    "Min Volatility / ATR (%)", 0.0, 5.0, active_defaults["min_atr"], 0.25
)

run_screener = st.sidebar.button(
    "🚀 Run Screener", type="primary", use_container_width=True
)

if run_screener:
  tickers_list = [
      t.strip().upper() for t in ticker_input.split(",") if t.strip()
  ]
  if tickers_list:
    with st.spinner("Analyzing market data..."):
      df_results = analyze_enhanced_sma_strategy(
          tickers_list,
          default_buffer_pct=buffer_setting / 100.0,
          rvol_threshold=rvol_setting,
      )
    if not df_results.empty:
      st.subheader("📊 Unified Trigger & Analysis Matrix")
      st.dataframe(df_results, use_container_width=True, hide_index=True)

      st.divider()
      csv_data = df_results.to_csv(index=False)
      st.download_button(
          "📥 Download Full Table CSV",
          csv_data,
          "200_day_triggers.csv",
          "text/csv",
      )
