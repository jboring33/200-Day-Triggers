# 200-Day-Triggers
200 Day buy / sell trigger
# 📖 README: 200-Day Institutional Trend Screener

Welcome to the **200-Day Triggers** trend-following dashboard. This tool is designed to transition your market analysis from basic charting to a disciplined, institutional decision-making framework. It monitors macro trend regimes, applies volatility-scaled noise buffers, and generates clear capital action triggers.

---

## 🚀 Quick Start Guide

1. **Select a Watchlist Preset**: Use the horizontal preset buttons at the top of the main panel to choose a curated asset pool (e.g., *Mag7*, *Sector ETFs*, *Bond ETFs*, *Broad Market*, or your own *Watching* list).
2. **Customize Tickers**: The text box below the preset buttons displays the active tickers. You can add or remove comma-separated stock or ETF symbols freely. All changes are saved automatically and survive browser refreshes.
3. **Run the Screener**: Click the **🚀 Run Screener** button directly below the ticker input box to analyze the secular trends.
4. **Fine-Tune Parameters (Optional)**: Open the collapsible left-side panel (using the double carrots `>>` in the top left) if you wish to adjust the core thresholds:
* **Base Buffer Noise Filter (%)**: Controls the baseline neutral zone width around the 200 SMA.
* **Min RVOL Breakout Confirmation**: Sets the volume expansion requirement for active accumulation.
* **Min Volatility / ATR (%)**: Sets the baseline asset volatility minimum.



---

## 🔍 Special Mode: "Recommendations"

If you select the **Recommendations** preset, the app acts as an autonomous open-world discovery engine:

1. Click the **🔍 Scan & Refresh** button that appears on the left side of the ticker input box.
2. The engine will scan liquid U.S. ETFs to find top-performing structural leaders currently trading above their rising 200-day moving averages.
3. Once populated, click **🚀 Run Screener** to evaluate them.

---

## 📋 Status & Action Classification Matrix

The dashboard evaluates each asset and assigns one of five institutional status actions based on absolute price position relative to the 200 SMA noise bands ($\text{Upper/Lower Threshold} = \text{200 SMA} \times (1 \pm \text{Buffer})$):

* 🟢 **ACCUMULATE**: Price is securely above the upper noise buffer, the 200 SMA slope is **UP** or **FLAT**, and volume/volatility confirm. *Justification: Full institutional confluence and breakout confirmation.*
* 🟢 **HOLD**: Price is above the upper noise buffer and the 200 SMA trend is intact, but volume or volatility indicators fall below strict parameters. *Justification: Secular trend remains healthy, but new capital allocation is paused.*
* ⚪ **BUFFER ZONE**: Price is chopping *inside* the neutral noise band around the 200 SMA. *Justification: Prevents whipsaws and unnecessary turnover during minor consolidations. Capital actions are held neutral.*
* 🟡 **CAUTION**: Price trades outside the noise buffer, but diverges from the moving average slope trajectory (e.g., price dropping below the lower band while the 200 SMA slope is still pointing upward). *Justification: Signals potential regime transition or false breakouts; pause capital actions pending alignment.*
* 🔴 **SELL / ROTATE**: Price has breached the lower noise buffer while the 200 SMA slope has turned **DOWN**. *Justification: Confirmed structural bear regime. Triggers risk mitigation and rotation into safe-harbor assets.*

---

## 📊 Indicator Visual Badging Logic

* **200 SMA Slope**: Evaluated over a 1-month lookback (21 trading days) to filter out short-term wiggles.
* 🟢 **UP** ($\Delta > +0.05\%$)
* ⚪ **FLAT** ($-0.05\% \le \Delta \le +0.05\%$)
* 🔴 **DOWN** ($\Delta < -0.05\%$)


* **Buffer (%)**: Color-coded based on how wide the volatility-scaled noise filter is relative to your preset targets.
* **RVOL (Relative Volume)**: Compares current volume against the 20-day average.
* 🟢 **Green** if meeting or exceeding the parameter threshold; 🟡 **Yellow** if below.


* **ATR (%)**: Measures underlying asset volatility.
* 🟢 **Green** if meeting the minimum volatility threshold; 🔴 **Red** if compressed.



---

*Data powered by Yahoo Finance. All configurations are locally persistent via `watchlist_state.json`.*
