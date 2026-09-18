"""
Repository: 200 Day Triggers
Description: Enhanced 200-Day Trend & Volatility Screener using 200 EMA, 50/200 Crosses, 
             and ATR-based Dynamic Noise Buffers.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

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

def analyze_enhanced_sma_strategy(tickers, lookback_period="2y", slope_window=5, default_buffer_pct=0.02):
    """
    Evaluates tickers for the '200 Day Triggers' repository:
    - 200 EMA (Faster reactivity than SMA)
    - 50 / 200 Golden Cross / Death Cross trend detection
    - Volatility-adjusted noise buffer (ATR-based for crypto & high-volatility equities)
    """
    results = []
    
    for ticker in tickers:
        try:
            # Fetch OHLCV price data
            df = yf.download(ticker, period=lookback_period, auto_adjust=True, progress=False)
            
            if df.empty or len(df) < 200 + slope_window:
                print(f"[{REPO_NAME}] Skipping {ticker}: Insufficient data (requires >205 trading days).")
                continue
            
            # Extract closing prices
            close = df['Close']
            curr_price = close.iloc[-1]
            
            # 1. Moving Average Calculations
            ema200 = close.ewm(span=200, adjust=False).mean()
            sma50 = close.rolling(window=50).mean()
            sma200 = close.rolling(window=200).mean()
            
            curr_ema200 = ema200.iloc[-1]
            curr_sma50 = sma50.iloc[-1]
            curr_sma200 = sma200.iloc[-1]
            prev_ema200 = ema200.iloc[-(slope_window + 1)]
            
            # 2. Volatility-based buffer adjustment via ATR
            atr = calculate_atr(df).iloc[-1]
            atr_pct = (atr / curr_price)
            # Use max between default 2% buffer and 1.5 x ATR%
            effective_buffer_pct = max(default_buffer_pct, atr_pct * 1.5)
            
            upper_threshold = curr_ema200 * (1 + effective_buffer_pct)
            lower_threshold = curr_ema200 * (1 - effective_buffer_pct)
            
            # 3. Slope calculation over the slope window
            ema_diff_pct = ((curr_ema200 - prev_ema200) / prev_ema200) * 100
            if ema_diff_pct > 0.05:
                slope = "UP"
            elif ema_diff_pct < -0.05:
                slope = "DOWN"
            else:
                slope = "FLAT"
                
            # 4. Golden / Death Cross Detection
            if curr_sma50 > curr_sma200:
                ma_cross = "GOLDEN CROSS (Bullish)"
            else:
                ma_cross = "DEATH CROSS (Bearish)"
                
            # 5. Trigger Logic with Noise Buffer
            pct_from_ema = ((curr_price - curr_ema200) / curr_ema200) * 100
            
            if curr_price > upper_threshold and slope in ["UP", "FLAT"]:
                signal = "BUY / BULLISH HOLD"
                status = "BULLISH"
                reason = f"Price > {effective_buffer_pct*100:.1f}% buffer above 200 EMA & slope is {slope}."
            elif curr_price < lower_threshold and slope == "DOWN":
                signal = "SELL / CASH OUT"
                status = "BEARISH"
                reason = f"Price < {effective_buffer_pct*100:.1f}% buffer below 200 EMA & slope is DOWN."
            elif lower_threshold <= curr_price <= upper_threshold:
                signal = "NOISE BUFFER ZONE"
                status = "NEUTRAL"
                reason = f"Price within ±{effective_buffer_pct*100:.1f}% noise buffer of 200 EMA; avoid whipsaw."
            else:
                signal = "WATCH / CAUTION"
                status = "WARNING"
                reason = f"Price crossed EMA but slope ({slope}) or buffer does not confirm exit/entry."

            results.append({
                "Repo": REPO_NAME,
                "Ticker": ticker,
                "Price": round(curr_price, 2),
                "200 EMA": round(curr_ema200, 2),
                "Dist EMA (%)": f"{pct_from_ema:+.2f}%",
                "Dynamic Buffer": f"±{effective_buffer_pct*100:.1f}%",
                "EMA Slope": slope,
                "50/200 Trend": ma_cross,
                "Signal": signal,
                "Status": status,
                "Reason": reason
            })
            
        except Exception as e:
            print(f"[{REPO_NAME}] Error processing {ticker}: {e}")
            
    return pd.DataFrame(results)


if __name__ == "__main__":
    watchlist = ["SPY", "QQQ", "GLD", "BTC-USD", "NVDA", "AAPL", "TLT", "IWM"]
    
    print("=" * 110)
    print(f"  REPOSITORY: {REPO_NAME} — SCREENER REPORT ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 110)
    
    df_results = analyze_enhanced_sma_strategy(watchlist)
    
    # Configure display output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    if not df_results.empty:
        print(df_results.drop(columns=["Repo"]).to_string(index=False))
        
        # Save output to CSV with repo branding
        csv_filename = "200_day_triggers_report.csv"
        df_results.to_csv(csv_filename, index=False)
        print("\n" + "=" * 110)
        print(f"Report saved locally to: {csv_filename}")
        print("=" * 110)
    else:
        print("No data retrieved.")
