"""
app.py
======
Dashboard Local Desktop - Decision Support System Saham
"""

import io
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import analysis as A

# ==========================================================================
# CONFIG & CSS
# ==========================================================================

st.set_page_config(
    page_title="Dashboard Saham | Decision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMINAL_CSS = """
<style>
    .stApp { background-color: #0b0e14; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1f2937; }
    div[data-testid="stMetric"] { background-color: #11161f; border: 1px solid #1f2937; border-radius: 8px; padding: 10px 14px; }
    div[data-testid="stMetricValue"] { color: #22d3ee; }
    .signal-buy   { color: #22c55e; font-weight: 700; }
    .signal-watch { color: #eab308; font-weight: 700; }
    .signal-wait  { color: #60a5fa; font-weight: 700; }
    .signal-avoid { color: #ef4444; font-weight: 700; }
    .explain-box {
        background-color: #11161f; border: 1px solid #1f2937; border-left: 4px solid #22d3ee;
        border-radius: 6px; padding: 14px 18px; margin-top: 8px;
    }
    .stock-header { font-size: 26px; font-weight: 800; color: #f5f5f5; }
    .stock-sub { color: #9ca3af; font-size: 14px; }
    thead tr th { background-color: #151b28 !important; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_UP = "#22c55e"
COLOR_DOWN = "#ef4444"
COLOR_ACCENT = "#22d3ee"

RANKING_COLUMNS = [
    "Kode Saham", "Nama Perusahaan", "Overall Score", "Momentum Score",
    "Liquidity Score", "Foreign Score", "Orderbook Score",
    "Bandar Activity Score", "Risk Score", "Signal",
]

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_saham.csv")


# ==========================================================================
# DATA LOADING (CACHED)
# ==========================================================================

@st.cache_data(show_spinner="Crunching the numbers...")
def cached_pipeline(source_data, n_days: int):
    # Pass either string path or BytesIO buffer directly to analysis.py
    raw_df, scored_df, window_dates = A.run_pipeline(source_data, n_days=n_days)
    return raw_df, scored_df, window_dates

def get_data_source(uploaded_file):
    if uploaded_file is not None:
        return io.BytesIO(uploaded_file.getvalue())
    if os.path.exists(DEFAULT_CSV_PATH):
        return DEFAULT_CSV_PATH
    return None

# ==========================================================================
# SIDEBAR
# ==========================================================================

st.sidebar.markdown("## 📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload master_saham.csv (Optional)", type=["csv"],
    help="Defaults to master_saham.csv in the same directory if left empty."
)

if st.sidebar.button("🔄 Refresh Dataset", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

source = get_data_source(uploaded_file)

if source is None:
    st.error("No dataset found. Please upload a file or place master_saham.csv in the root folder.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## ⏱️ Lookback Settings")
lookback_days = st.sidebar.number_input(
    "Evaluation Window (Days)",
    min_value=3, max_value=30, value=A.N_LOOKBACK_DAYS, step=1,
)

raw_df, scored_all, window_dates = cached_pipeline(source, int(lookback_days))
latest_date = window_dates[-1]

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Global Filters")

# Secure max fallback value safely
max_fsell_raw = float(scored_all["Foreign Sell"].max(skipna=True))
max_fsell = max_fsell_raw if pd.notna(max_fsell_raw) else 0.0

min_nilai = st.sidebar.number_input("Min Value (Rp)", min_value=0.0, value=0.0, step=1e9, format="%.0f")
min_volume = st.sidebar.number_input("Min Volume (Shares)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
min_freq = st.sidebar.number_input("Min Frequency", min_value=0.0, value=0.0, step=100.0, format="%.0f")
min_fbuy = st.sidebar.number_input("Min Foreign Buy", min_value=0.0, value=0.0, step=1e6, format="%.0f")
max_fsell_filter = st.sidebar.number_input(
    "Max Foreign Sell", min_value=0.0, value=max_fsell, step=1e6, format="%.0f"
)
min_overall = st.sidebar.slider("Minimum Overall Score", 0, 100, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Ticker Search")
search_query = st.sidebar.text_input("Enter Ticker (e.g., BBCA)", "").upper().strip()


# ==========================================================================
# FILTER EXECUTION
# ==========================================================================

complete = scored_all[scored_all["data_complete"]].copy()

# Filter handling optimized safely
screened = complete[
    (complete["Nilai"].fillna(0) >= min_nilai) &
    (complete["Volume"].fillna(0) >= min_volume) &
    (complete["Frekuensi"].fillna(0) >= min_freq) &
    (complete["Foreign Buy"].fillna(0) >= min_fbuy) &
    (complete["Foreign Sell"].fillna(0) <= max_fsell_filter) &
    (complete["Overall Score"].fillna(0) >= min_overall)
].copy()

screened = screened.sort_values("Overall Score", ascending=False)
top10 = screened.head(10).copy()


# ==========================================================================
# MAIN DASHBOARD HEADER
# ==========================================================================

st.markdown(
    f"<div class='stock-header'>📈 Stock Dashboard — DSS</div>"
    f"<div class='stock-sub'>Auto-lookback {lookback_days} active trading days</div>",
    unsafe_allow_html=True,
)
st.write("")

avg_net_foreign = complete["net_foreign_total"].mean(skipna=True)
advancers = int((complete["close_chg_last"] > 0).sum())
decliners = int((complete["close_chg_last"] < 0).sum())
unchanged = int((complete["close_chg_last"] == 0).sum())
total_moves = max(1, advancers + decliners + unchanged)
breadth_pct = 100 * advancers / total_moves

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest Session", latest_date.strftime("%Y-%m-%d"))
m2.metric("Total Universe", f"{complete.shape[0]:,}".replace(",", "."))
m3.metric("Filtered Candidates", f"{screened.shape[0]:,}".replace(",", "."))
m4.metric("Avg Net Foreign", f"{avg_net_foreign/1e9:,.2f} M".replace(",", "."))
m5.metric("Market Breadth", f"{breadth_pct:.0f}% Up", f"{advancers}↑ / {decliners}↓")

st.markdown("---")

all_codes = sorted(scored_all["Kode Saham"].unique().tolist())
if search_query:
    matched_codes = [c for c in all_codes if search_query in c]
else:
    matched_codes = top10["Kode Saham"].tolist() or all_codes[:10]

left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.subheader("🏆 Top 10 Watchlist")
    if top10.empty:
        st.warning("No stocks passed the current filter criteria.")
    else:
        display_top10 = top10[["Kode Saham", "Nama Perusahaan", "Overall Score", "Signal"]].copy()
        display_top10.insert(0, "Rank", range(1, len(display_top10) + 1))
        st.dataframe(
            display_top10, use_container_width=True, hide_index=True,
            height=min(420, 45 * (len(display_top10) + 1)),
            column_config={
                "Overall Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

    st.markdown("#### Inspect Ticker")
    selected_code = st.selectbox("Kode Saham", matched_codes, index=0, label_visibility="collapsed")


sel_row = scored_all[scored_all["Kode Saham"] == selected_code]

with right_col:
    if sel_row.empty:
        st.info("Select a ticker to load telemetry.")
    else:
        row = sel_row.iloc[0]
        st.subheader(f"{row['Kode Saham']} — {row['Nama Perusahaan']}")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Overall Score", f"{row['Overall Score']:.1f}" if pd.notna(row["Overall Score"]) else "N/A")
        sc2.metric("Signal", row["Signal"])
        sc3.metric("Last Close", f"{row['Penutupan']:,.0f}".replace(",", "."))
        sc4.metric("Change", f"{row['close_chg_last']*100:.2f}%" if pd.notna(row["close_chg_last"]) else "N/A")

        tabs = st.tabs(["💹 Price", "📊 Volume", "🌐 Foreign Flow", "💰 Value", "📗 Orderbook", "🧊 Non Regular", "📋 Synopsis"])
        dates = [d.strftime("%Y-%m-%d") for d in row["_hist_dates"]]

        with tabs[0]:
            fig = go.Figure(data=[go.Candlestick(
                x=dates, open=row["_hist_open"], high=row["_hist_high"],
                low=row["_hist_low"], close=row["_hist_close"],
                increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN, name="Price"
            )])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric(f"{lookback_days}D Return", f"{row['close_chg_window']*100:.2f}%" if pd.notna(row['close_chg_window']) else "N/A")
            colB.metric("Avg Daily Range", f"{row['avg_range_pct']*100:.2f}%" if pd.notna(row['avg_range_pct']) else "N/A")
            colC.metric("Breakout Triggered", "Yes ✅" if row["breakout"] else "No")

        with tabs[1]:
            fig = go.Figure(data=[go.Bar(x=dates, y=row["_hist_volume"], marker_color=COLOR_ACCENT, name="Volume")])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("Last Vol Change", f"{row['vol_chg_last']*100:.1f}%" if pd.notna(row['vol_chg_last']) else "N/A")
            colB.metric("Volume Spike", f"{row['vol_spike']:.2f}x" if pd.notna(row['vol_spike']) else "N/A")
            colC.metric(f"Avg Vol ({lookback_days}D)", f"{row['avg_vol_window']:,.0f}".replace(",", "."))

        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=row["_hist_fbuy"], name="Foreign Buy", marker_color=COLOR_UP))
            # Safely handle NaNs for subtraction in plot
            safe_fsell = [-v if pd.notna(v) else 0 for v in row["_hist_fsell"]]
            fig.add_trace(go.Bar(x=dates, y=safe_fsell, name="Foreign Sell", marker_color=COLOR_DOWN))
            fig.add_trace(go.Scatter(x=dates, y=row["_hist_net_foreign"], name="Net Foreign", mode="lines+markers", line=dict(color=COLOR_ACCENT, width=3)))
            fig.update_layout(template=PLOTLY_TEMPLATE, barmode="relative", height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric(f"Net Foreign ({lookback_days}D)", f"{row['net_foreign_total']/1e9:,.2f} M".replace(",", "."))
            status = "Accumulating 📈" if row["accumulation"] else ("Distributing 📉" if row["distribution"] else "Neutral")
            colB.metric("Flow Status", status)
            colC.metric("Foreign Control", f"{row['foreign_participation']*100:.1f}%" if pd.notna(row['foreign_participation']) else "N/A")

        with tabs[3]:
            fig = go.Figure(data=[go.Bar(x=dates, y=row["_hist_nilai"], marker_color="#a78bfa", name="Value")])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("Last Value Shift", f"{row['nilai_chg_last']*100:.1f}%" if pd.notna(row['nilai_chg_last']) else "N/A")
            colB.metric(f"Avg Value ({lookback_days}D)", f"{row['avg_nilai_window']/1e9:,.2f} M".replace(",", "."))
            colC.metric("Consistency Floor", f"{row['nilai_consistency']/1e9:,.2f} M".replace(",", "."))

        with tabs[4]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Bid Volume", "Offer Volume"], y=[row["Bid Volume"], row["Offer Volume"]], marker_color=[COLOR_UP, COLOR_DOWN]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Total Bid", f"{row['Bid Volume']:,.0f}".replace(",", "."))
            colB.metric("Total Offer", f"{row['Offer Volume']:,.0f}".replace(",", "."))
            colC.metric("Dominance (Bid)", f"{row['bid_offer_dominance']*100:.1f}%" if pd.notna(row['bid_offer_dominance']) else "N/A")
            colD.metric("Spread Gap", f"{row['spread_pct']*100:.2f}%" if pd.notna(row['spread_pct']) else "N/A")

        with tabs[5]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=row["_hist_nr_val"], name="NR Value", marker_color="#f59e0b"))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("NR Value Ratio", f"{row['nr_ratio_last']*100:.1f}%" if pd.notna(row['nr_ratio_last']) else "N/A")
            colB.metric("Block Trade Status", "Detected ⚠️" if row["block_trade_flag"] else "Clear")
            colC.metric("NR Freq Shift", f"{row['nr_freq_trend']:+.0f}" if pd.notna(row['nr_freq_trend']) else "N/A")

        with tabs[6]:
            summary = {
                "Last Price": f"{row['Penutupan']:,.0f}",
                "Volume": f"{row['Volume']:,.0f}",
                "Value": f"{row['Nilai']:,.0f}",
                "Frequency": f"{row['Frekuensi']:,.0f}",
                "Foreign Buy": f"{row['Foreign Buy']:,.0f}",
                "Foreign Sell": f"{row['Foreign Sell']:,.0f}",
                "Free Float (Proxy)": f"{row['free_float_ratio']*100:.1f}%" if pd.notna(row['free_float_ratio']) else "N/A",
                "Turnover Ratio": f"{row['turnover_ratio']*100:.3f}%" if pd.notna(row['turnover_ratio']) else "N/A",
                "Market Cap (Proxy)": f"{row['market_cap_proxy']/1e12:,.2f} T" if pd.notna(row['market_cap_proxy']) else "N/A",
                f"Volatility ({lookback_days}D)": f"{row['volatility']*100:.2f}%" if pd.notna(row['volatility']) else "N/A",
            }
            st.table(pd.DataFrame(summary.items(), columns=["Metric", "Reading"]))

        st.markdown("#### 🧠 Auto-Analysis")
        reasons = A.generate_explanation(row)
        bullet_html = "".join([f"<li>{r}</li>" for r in reasons])
        st.markdown(
            f"<div class='explain-box'><b>{row['Kode Saham']}</b> achieved a score of "
            f"<b>{row['Overall Score']:.1f}</b> ({row['Signal']}) due to:<ul>{bullet_html}</ul></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("📋 Full Database Ranking")

ranking_df = screened.copy()
if search_query:
    ranking_df = ranking_df[ranking_df["Kode Saham"].str.contains(search_query, na=False)]

if ranking_df.empty:
    st.warning("No matches found for your search/filter parameters.")
else:
    show_df = ranking_df[RANKING_COLUMNS].reset_index(drop=True)
    show_df.insert(0, "Rank", range(1, len(show_df) + 1))
    st.dataframe(
        show_df, use_container_width=True, hide_index=True,
        height=min(600, 40 * (len(show_df) + 1)),
        column_config={
            "Overall Score": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f"),
            "Momentum Score": st.column_config.NumberColumn(format="%.1f"),
            "Liquidity Score": st.column_config.NumberColumn(format="%.1f"),
            "Foreign Score": st.column_config.NumberColumn(format="%.1f"),
            "Orderbook Score": st.column_config.NumberColumn(format="%.1f"),
            "Bandar Activity Score": st.column_config.NumberColumn(format="%.1f"),
            "Risk Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )
