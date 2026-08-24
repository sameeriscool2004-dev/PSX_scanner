"""Read-only Streamlit dashboard for the latest PSX scanner results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "latest_scan.csv"
METADATA_PATH = BASE_DIR / "results" / "latest_scan_metadata.json"
SIGNALS = ["ALL", "EXCEPTIONAL_BUY", "EXCEPTIONAL_SETUP", "EXTENDED_WAIT", "STRONG_BUY", "WATCHLIST", "DEVELOPING"]
DETAIL_COLUMNS = [
    "Close", "SMA20", "SMA50", "SMA200", "EMA9", "EMA20", "EMA50",
    "Distance_From_EMA9", "Distance_From_EMA20", "RSI", "RSI_Status", "Volume",
    "EMA_Aligned", "EMA_Slope_Passed", "Close_Above_EMA9", "EMA9_Extension_Passed",
    "Pullback_Passed", "Bullish_Confirmation_Passed",
    "Volume_Ratio", "Volume_Status", "Resistance", "Setup", "Bullish_Candle",
    "Liquidity_Passed", "Score", "Signal", "Conditions_Partial",
    "Conditions_Failed", "Confirmation_Missing",
    "Buy_Zone_Low", "Buy_Zone_High", "Ideal_Entry", "Stop_Loss",
    "Target_1", "Target_2", "Risk_Reward",
    "Technical_Signal", "Entry_Status", "Action", "Reason",
]

st.set_page_config(page_title="PSX Smart Scanner", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #14231f; --muted: #66756e; --line: #dce7df; --mint: #e8f4ec; --green: #176b45; --gold: #b76b16; --red: #a83b32; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .block-container { max-width: 1180px; padding: 2rem 1rem 3rem; }
    .hero { background: linear-gradient(135deg, #eef8f0 0%, #f8f5e8 100%); border: 1px solid var(--line); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
    .hero h1 { margin: .2rem 0 .4rem; font-size: clamp(1.8rem, 5vw, 3rem); }
    .meta { color: var(--muted); font-size: .9rem; }
    .metric { border: 1px solid var(--line); border-radius: 12px; padding: 1rem; background: white; min-height: 94px; }
    .metric-label { color: var(--muted); font-size: .78rem; font-weight: 600; }
    .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 700; margin-top: .35rem; }
    .signal { border-left: 4px solid var(--line); border-radius: 10px; background: white; padding: .8rem 1rem; margin: .55rem 0; box-shadow: 0 1px 5px rgba(20,35,31,.05); }
    .signal-hot { border-left-color: var(--gold); background: #fffaf0; }
    .signal-strong { border-left-color: var(--green); background: #f1faf3; }
    .stock-line { display: flex; gap: .65rem; align-items: baseline; flex-wrap: wrap; }
    .stock-symbol { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; }
    .stock-score { color: var(--green); font-weight: 700; }
    .stock-sub { color: var(--muted); font-size: .84rem; line-height: 1.55; }
    .badge { font-size: .7rem; font-weight: 700; border-radius: 999px; padding: .18rem .5rem; background: var(--mint); color: var(--green); }
    @media (max-width: 640px) { .block-container { padding: 1rem .7rem 2rem; } .hero { padding: 1.1rem; } .metric-value { font-size: 1.35rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_results() -> tuple[pd.DataFrame | None, dict]:
    if not RESULTS_PATH.exists():
        return None, {}
    frame = pd.read_csv(RESULTS_PATH)
    frame = enrich_results(frame)
    metadata = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return frame, metadata


def rounded_price(price: float) -> float:
    """Use practical precision for PSX price levels."""
    return round(price, 2) if price >= 10 else round(price, 3)


def enrich_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive display-only fields from the already saved scan results."""
    enriched = frame.copy()
    enriched["RSI_Status"] = enriched["RSI"].apply(
        lambda item: "WEAK" if item < 50 else "HEALTHY" if item < 70 else "EXTENDED" if item < 75 else "OVERBOUGHT"
    )
    if "Volume" not in enriched.columns:
        enriched["Volume"] = enriched["Average_Volume_20"] * enriched["Volume_Ratio"]

    def entry_levels(row: pd.Series) -> pd.Series:
        setup = str(row.get("Setup", "NONE"))
        close = float(row.get("Close", 0))
        level = row.get("Resistance")
        if setup not in {"BREAKOUT", "RETEST"} or pd.isna(level) or level <= 0 or close <= 0:
            return pd.Series({column: None for column in ["Buy_Zone_Low", "Buy_Zone_High", "Ideal_Entry", "Stop_Loss", "Target_1", "Target_2", "Risk_Reward"]})

        level = float(level)
        if setup == "RETEST":
            buy_low = level * 0.98
            buy_high = level * 1.02
            support = level
        else:
            buy_low = level
            buy_high = level * 1.02
            support = level
        if buy_high < buy_low:
            return pd.Series({column: None for column in ["Buy_Zone_Low", "Buy_Zone_High", "Ideal_Entry", "Stop_Loss", "Target_1", "Target_2", "Risk_Reward"]})
        ideal_entry = (buy_low + buy_high) / 2
        stop_loss = support * 0.97
        risk = ideal_entry - stop_loss
        if risk <= 0:
            return pd.Series({column: None for column in ["Buy_Zone_Low", "Buy_Zone_High", "Ideal_Entry", "Stop_Loss", "Target_1", "Target_2", "Risk_Reward"]})
        target_1 = ideal_entry + risk * 1.5
        target_2 = ideal_entry + risk * 2.5
        return pd.Series({
            "Buy_Zone_Low": rounded_price(buy_low),
            "Buy_Zone_High": rounded_price(buy_high),
            "Ideal_Entry": rounded_price(ideal_entry),
            "Stop_Loss": rounded_price(stop_loss),
            "Target_1": rounded_price(target_1),
            "Target_2": rounded_price(target_2),
            "Risk_Reward": round((target_2 - ideal_entry) / risk, 2),
        })

    levels = enriched.apply(entry_levels, axis=1)
    enriched = pd.concat([enriched, levels], axis=1)
    def entry_state(row: pd.Series) -> pd.Series:
        setup = str(row.get("Setup", "NONE"))
        close = float(row.get("Close", 0))
        buy_low = row.get("Buy_Zone_Low")
        buy_high = row.get("Buy_Zone_High")
        signal = str(row.get("Signal", "NO_BUY"))
        if signal == "EXCEPTIONAL_SETUP":
            return pd.Series({"Technical_Signal": signal, "Entry_Status": "BELOW_ENTRY_ZONE", "Action": "WAIT_FOR_ENTRY", "Reason": "Exceptional setup confirmed; wait for price to enter the buy zone."})
        if signal == "EXTENDED_WAIT":
            return pd.Series({"Technical_Signal": signal, "Entry_Status": "ABOVE_ENTRY_ZONE", "Action": "WAIT_FOR_PULLBACK", "Reason": "Exceptional setup confirmed; wait for a pullback into the buy zone."})
        if setup not in {"BREAKOUT", "RETEST"}:
            status, action, reason = "SETUP_INVALID", "AVOID", "No valid breakout or retest setup."
        elif pd.isna(buy_low) or pd.isna(buy_high):
            status, action, reason = "UNAVAILABLE", "AVOID", "A valid technical entry zone could not be determined."
        elif close < buy_low:
            status, action, reason = "BELOW_ENTRY_ZONE", "WAIT_FOR_RECLAIM", "Current price is below the valid entry zone."
        elif close > buy_high:
            status, action, reason = "ABOVE_ENTRY_ZONE", "WAIT_FOR_PULLBACK", "Current price is above the preferred entry zone."
        elif signal not in {"EXCEPTIONAL_BUY", "STRONG_BUY"}:
            status, action, reason = "WAITING_FOR_CONFIRMATION", "WAIT_FOR_CONFIRMATION", "Price is in the zone, but technical confirmation is incomplete."
        else:
            status = "READY_TO_BUY"
            action = "BUY_NOW" if signal in {"EXCEPTIONAL_BUY", "STRONG_BUY"} else "BUY_ON_RETEST" if setup == "RETEST" else "BUY_ON_BREAKOUT"
            reason = "Current price is inside the valid technical entry zone."
        return pd.Series({
            "Technical_Signal": signal,
            "Entry_Status": status,
            "Action": action,
            "Reason": reason,
        })

    enriched = pd.concat([enriched, enriched.apply(entry_state, axis=1)], axis=1)
    return enriched


def confirmation_message(row: pd.Series) -> str:
    """Format the actionable warning for a technically promising but unconfirmed row."""
    missing = str(row.get("Confirmation_Missing", ""))
    if row.get("Entry_Status") != "WAITING_FOR_CONFIRMATION" and not missing:
        return ""
    lines = ["**WAIT FOR CONFIRMATION**"]
    if "Volume" in missing:
        lines.append(
            f"- Volume confirmation: 1.5x required (current volume: {float(row['Volume_Ratio']):.2f}x)"
        )
    if "RSI" in missing:
        lines.append("- RSI confirmation: RSI must be healthy, below 70")
    if "Breakout/Retest" in missing:
        lines.append("- Breakout/retest confirmation: valid BREAKOUT or RETEST required")
    if "Pullback/Retest" in missing:
        lines.append("- Pullback/retest confirmation: a recent RETEST is required")
    if "Bullish Confirmation" in missing:
        lines.append("- Bullish candle confirmation: a bullish candle must follow the pullback")
    if "EMA Alignment" in missing or "EMA Slope" in missing:
        lines.append("- EMA confirmation: EMA9 > EMA20 > EMA50 with EMA9 and EMA20 rising")
    if "Close > EMA9" in missing:
        lines.append("- Price confirmation: close must be above EMA9")
    if "EMA Extension" in missing:
        lines.append("- Extension confirmation: close must be within 5% above EMA9")
    if "Trend" in missing:
        lines.append("- Trend confirmation: SMA alignment required")
    if "Liquidity" in missing:
        lines.append("- Liquidity confirmation: liquidity threshold must pass")
    lines.append("**Action:** Wait for a bullish confirmation candle with stronger volume.")
    return "\n".join(lines)


def value(row: pd.Series, column: str) -> str:
    item = row.get(column, "N/A")
    if pd.isna(item):
        return "N/A"
    if column in {"Distance_From_EMA9", "Distance_From_EMA20"}:
        return f"{float(item) * 100:.2f}%"
    if isinstance(item, float):
        return f"{item:.2f}"
    return str(item)


def score_breakdown(row: pd.Series) -> dict[str, str]:
    """Reconstruct the existing saved-score components for display."""
    trend = 0.0
    if all(pd.notna(row.get(name)) for name in ("SMA20", "SMA50", "SMA200")):
        if row["SMA20"] > row["SMA50"] > row["SMA200"]:
            trend = 25.0
        elif row["SMA20"] > row["SMA50"]:
            trend = 12.5
    sma_points = sum(
        5.0 for name in ("SMA20", "SMA50", "SMA200")
        if pd.notna(row.get(name)) and row["Close"] > row[name]
    )
    rsi_points = 0.0 if pd.isna(row.get("RSI")) else max(0.0, min(10.0, (row["RSI"] - 50.0) / 20.0 * 10.0))
    volume_points = max(0.0, min(15.0, row["Volume_Ratio"] / 1.5 * 15.0))
    setup_points = {"BREAKOUT": 20.0, "RETEST": 20.0, "EARLY_BREAKOUT": 10.0}.get(row.get("Setup"), 0.0)
    candle_points = 10.0 if row.get("Bullish_Candle") not in (None, "NONE") else 0.0
    market_points = 5.0 if row.get("Market_Filter") == "BULLISH" else 0.0
    ema_alignment_points = float(row.get("EMA_Alignment", 0.0))
    ema_slope_points = float(row.get("EMA_Slope", 0.0))
    return {
        "Trend": f"{trend:.2f} / 25",
        "Price above SMAs": f"{sma_points:.2f} / 15",
        "RSI": f"{rsi_points:.2f} / 10",
        "Volume": f"{volume_points:.2f} / 15",
        "Breakout/Retest": f"{setup_points:.2f} / 20",
        "Bullish Candle": f"{candle_points:.2f} / 10",
        "Market Filter": f"{market_points:.2f} / 5",
        "EMA Alignment": f"{ema_alignment_points:.2f} / 10",
        "EMA Slope": f"{ema_slope_points:.2f} / 5",
    }


st.markdown(
    '<div class="hero"><div class="eyebrow">PAKISTAN STOCK EXCHANGE</div><h1>PSX SMART SCANNER</h1><div class="meta">Latest saved scan, presented for fast review on desktop or mobile.</div></div>',
    unsafe_allow_html=True,
)

if st.button("↻ Refresh saved results", type="secondary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

results, metadata = load_results()
if results is None:
    st.warning("No saved scan results were found. Run the scanner first, then refresh this dashboard.")
    st.stop()

market = results["Market_Filter"].dropna().iloc[0] if "Market_Filter" in results and not results["Market_Filter"].dropna().empty else "UNKNOWN"
timestamp = metadata.get("scan_timestamp", "Unavailable")
processed = metadata.get("successfully_processed", len(results))
duration = metadata.get("total_scan_duration", "Unavailable")
st.markdown(
    f'<div class="meta"><b>Last scan:</b> {timestamp} &nbsp; <b>Symbols processed:</b> {processed} &nbsp; <b>Market:</b> {market} &nbsp; <b>Duration:</b> {duration}s</div>',
    unsafe_allow_html=True,
)

counts = results["Signal"].value_counts()
cols = st.columns(4)
for column, signal in zip(cols, ["EXCEPTIONAL_BUY", "STRONG_BUY", "WATCHLIST", "DEVELOPING"]):
    with column:
        st.markdown(f'<div class="metric"><div class="metric-label">{signal}</div><div class="metric-value">{int(counts.get(signal, 0))}</div></div>', unsafe_allow_html=True)

st.divider()
st.subheader("🔥 ACTIONABLE BUY OPPORTUNITIES")
actionable = results[results["Signal"].isin(["EXCEPTIONAL_BUY", "STRONG_BUY"])].sort_values("Score", ascending=False)
if actionable.empty:
    st.info("No confirmed actionable buy opportunities in the latest saved scan.")
else:
    for _, row in actionable.iterrows():
        st.markdown(
            f"**{value(row, 'Symbol')}** · {value(row, 'Signal')} · Score {value(row, 'Score')} · "
            f"Close {value(row, 'Close')} · RSI {value(row, 'RSI')} · Volume ratio {value(row, 'Volume_Ratio')}"
        )
        st.caption(
            f"BUY ZONE {value(row, 'Buy_Zone_Low')} - {value(row, 'Buy_Zone_High')} | "
            f"Ideal {value(row, 'Ideal_Entry')} | Stop {value(row, 'Stop_Loss')} | "
            f"T1 {value(row, 'Target_1')} | T2 {value(row, 'Target_2')} | R/R {value(row, 'Risk_Reward')}"
        )
        st.caption(f"Entry status: {value(row, 'Entry_Status')} | Action: {value(row, 'Action')} | {value(row, 'Reason')}")

search = st.text_input("Search PSX symbol", placeholder="e.g. ATRL")
selected = st.selectbox("Show candidates", SIGNALS, index=0)
filtered = results if selected == "ALL" else results[results["Signal"] == selected]
if search.strip():
    filtered = filtered[filtered["Symbol"].astype(str).str.contains(search.strip().upper(), case=False, na=False)]
filtered = filtered.sort_values("Score", ascending=False)
st.subheader(f"{selected.title().replace('_', ' ')} candidates ({len(filtered)})")

if filtered.empty:
    st.info("No candidates match this filter in the latest saved scan.")
else:
    for _, row in filtered.iterrows():
        signal = value(row, "Technical_Signal")
        tone = "signal-hot" if signal == "EXCEPTIONAL_BUY" else "signal-strong" if signal == "STRONG_BUY" else ""
        title = f"<div class='stock-line'><span class='stock-symbol'>{value(row, 'Symbol')}</span><span class='badge'>{signal}</span><span class='stock-score'>Score {value(row, 'Score')}</span></div>"
        summary = (
            f"Close {value(row, 'Close')} &nbsp; | &nbsp; EMA9 {value(row, 'EMA9')} &nbsp; | &nbsp; "
            f"EMA20 {value(row, 'EMA20')} &nbsp; | &nbsp; EMA50 {value(row, 'EMA50')} &nbsp; | &nbsp; "
            f"Distance from EMA9 {value(row, 'Distance_From_EMA9')} &nbsp; | &nbsp; "
            f"Distance from EMA20 {value(row, 'Distance_From_EMA20')} &nbsp; | &nbsp; "
            f"RSI {value(row, 'RSI')} &nbsp; | &nbsp; Volume ratio {value(row, 'Volume_Ratio')} &nbsp; | &nbsp; "
            f"{value(row, 'Volume_Status')} volume &nbsp; | &nbsp; Setup {value(row, 'Setup')}"
        )
        with st.expander(f"{value(row, 'Symbol')}  |  {signal}  |  {value(row, 'Entry_Status')}  |  Score {value(row, 'Score')}"):
            st.markdown(f'<div class="signal {tone}">{title}<div class="stock-sub">{summary}</div></div>', unsafe_allow_html=True)
            message = confirmation_message(row)
            if message:
                st.warning(message)
            st.markdown("**Score breakdown**")
            breakdown_cols = st.columns(2)
            for index, (component, points) in enumerate(score_breakdown(row).items()):
                with breakdown_cols[index % 2]:
                    st.markdown(f"**{component}:** {points}")
            detail_cols = st.columns(2)
            for index, detail in enumerate(DETAIL_COLUMNS):
                with detail_cols[index % 2]:
                    st.markdown(f"**{detail.replace('_', ' ')}:** {value(row, detail)}")
        if not st.session_state.get("expanded_hint"):
            st.session_state.expanded_hint = True

st.caption("Read-only dashboard. Refresh reloads saved files and never starts a new scan.")
print("Run with: streamlit run app.py")
