"""
app.py
======
Dashboard Local Desktop - Decision Support System Saham

Jalankan dengan:
    streamlit run app.py

Dataset default: master_saham.csv di folder yang sama dengan app.py.
Bisa juga upload file CSV lain lewat sidebar.
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A

# ==========================================================================
# KONFIGURASI HALAMAN & TEMA "TRADING TERMINAL"
# ==========================================================================

st.set_page_config(
    page_title="Dashboard Saham | Decision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMINAL_CSS = """
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1420;
        border-right: 1px solid #1f2937;
    }
    div[data-testid="stMetric"] {
        background-color: #11161f;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #22d3ee;
    }
    .signal-buy   { color: #22c55e; font-weight: 700; }
    .signal-watch { color: #eab308; font-weight: 700; }
    .signal-wait  { color: #60a5fa; font-weight: 700; }
    .signal-avoid { color: #ef4444; font-weight: 700; }
    .explain-box {
        background-color: #11161f;
        border: 1px solid #1f2937;
        border-left: 4px solid #22d3ee;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 8px;
    }
    .stock-header {
        font-size: 26px;
        font-weight: 800;
        color: #f5f5f5;
    }
    .stock-sub {
        color: #9ca3af;
        font-size: 14px;
    }
    thead tr th {
        background-color: #151b28 !important;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_UP = "#22c55e"
COLOR_DOWN = "#ef4444"
COLOR_ACCENT = "#22d3ee"
COLOR_MUTED = "#9ca3af"

SIGNAL_CLASS = {
    "BUY": "signal-buy",
    "WATCH": "signal-watch",
    "WAIT": "signal-wait",
    "AVOID": "signal-avoid",
    "NO DATA": "signal-wait",
}

RANKING_COLUMNS = [
    "Kode Saham", "Nama Perusahaan", "Overall Score", "Momentum Score",
    "Liquidity Score", "Foreign Score", "Orderbook Score",
    "Bandar Activity Score", "Risk Score", "Signal",
]

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_saham.csv")


# ==========================================================================
# DATA LOADING (CACHED)
# ==========================================================================

@st.cache_data(show_spinner="Memuat & menganalisis dataset...")
def cached_pipeline(file_bytes: bytes, n_days: int):
    # n_days ikut menjadi bagian dari cache key, jadi saat lookback diganti
    # di sidebar, Streamlit otomatis menghitung ulang (tidak memakai cache lama).
    buf = io.BytesIO(file_bytes)
    raw_df, scored_df, window_dates = A.run_pipeline(buf, n_days=n_days)
    return raw_df, scored_df, window_dates


def read_source_bytes(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.getvalue()
    if os.path.exists(DEFAULT_CSV_PATH):
        with open(DEFAULT_CSV_PATH, "rb") as f:
            return f.read()
    return None


# ==========================================================================
# SIDEBAR: DATA SOURCE, REFRESH, FILTER, SEARCH
# ==========================================================================

st.sidebar.markdown("## 📂 Sumber Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload master_saham.csv (opsional)", type=["csv"],
    help="Jika tidak diisi, dashboard membaca master_saham.csv di folder yang sama dengan app.py",
)

if st.sidebar.button("🔄 Refresh Dataset", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

source_bytes = read_source_bytes(uploaded_file)

if source_bytes is None:
    st.error(
        "File **master_saham.csv** tidak ditemukan di folder aplikasi, dan tidak ada file yang di-upload.\n\n"
        "Silakan upload file lewat sidebar, atau letakkan master_saham.csv di folder yang sama dengan app.py."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## ⏱️ Lookback")
lookback_days = st.sidebar.number_input(
    "Jumlah hari perdagangan terakhir (lookback)",
    min_value=3,
    max_value=30,
    value=A.N_LOOKBACK_DAYS,
    step=1,
    help=(
        "Minimum 3 hari — beberapa fitur (trend harga, akumulasi/distribusi asing) "
        "butuh minimal 3 hari data untuk terdeteksi."
    ),
)

raw_df, scored_all, window_dates = cached_pipeline(source_bytes, int(lookback_days))
latest_date = window_dates[-1]

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Filter Screening")

max_nilai = float(scored_all["Nilai"].max(skipna=True) or 0)
max_vol = float(scored_all["Volume"].max(skipna=True) or 0)
max_freq = float(scored_all["Frekuensi"].max(skipna=True) or 0)
max_fbuy = float(scored_all["Foreign Buy"].max(skipna=True) or 0)
max_fsell = float(scored_all["Foreign Sell"].max(skipna=True) or 0)

min_nilai = st.sidebar.number_input("Minimum Nilai (Rp)", min_value=0.0, value=0.0, step=1e9, format="%.0f")
min_volume = st.sidebar.number_input("Minimum Volume (lembar)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
min_freq = st.sidebar.number_input("Minimum Frekuensi", min_value=0.0, value=0.0, step=100.0, format="%.0f")
min_fbuy = st.sidebar.number_input("Minimum Foreign Buy (lembar)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
max_fsell_filter = st.sidebar.number_input(
    "Maximum Foreign Sell (lembar)", min_value=0.0, value=float(max_fsell if max_fsell > 0 else 0.0),
    step=1e6, format="%.0f",
    help="Saham dengan Foreign Sell di atas nilai ini akan disembunyikan dari screening",
)
min_overall = st.sidebar.slider("Minimum Overall Score", 0, 100, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Pencarian Kode Saham")
search_query = st.sidebar.text_input("Ketik kode saham (mis. BBCA)", "").upper().strip()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Lookback otomatis: **{lookback_days} hari perdagangan terakhir**\n\n"
    + "\n".join([f"- {d.strftime('%Y-%m-%d')}" for d in window_dates])
)


# ==========================================================================
# APPLY FILTERS -> UNIVERSE SCREENED
# ==========================================================================

complete = scored_all[scored_all["data_complete"]].copy()

screened = complete[
    (complete["Nilai"].fillna(0) >= min_nilai)
    & (complete["Volume"].fillna(0) >= min_volume)
    & (complete["Frekuensi"].fillna(0) >= min_freq)
    & (complete["Foreign Buy"].fillna(0) >= min_fbuy)
    & (complete["Foreign Sell"].fillna(0) <= max_fsell_filter)
    & (complete["Overall Score"].fillna(0) >= min_overall)
].copy()

screened = screened.sort_values("Overall Score", ascending=False)
top10 = screened.head(10).copy()


# ==========================================================================
# HEADER & TOP METRICS
# ==========================================================================

st.markdown(
    f"<div class='stock-header'>📈 Dashboard Saham — Decision Support System</div>"
    f"<div class='stock-sub'>Auto-lookback {lookback_days} hari perdagangan terakhir · Data source: master_saham.csv</div>",
    unsafe_allow_html=True,
)
st.write("")

avg_net_foreign = complete["net_foreign_total"].mean(skipna=True)
advancers = int((complete["close_chg_last"] > 0).sum())
decliners = int((complete["close_chg_last"] < 0).sum())
unchanged = int((complete["close_chg_last"] == 0).sum())
breadth_pct = 100 * advancers / max(1, (advancers + decliners + unchanged))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tanggal Data", latest_date.strftime("%Y-%m-%d"))
m2.metric("Total Saham", f"{complete.shape[0]:,}".replace(",", "."))
m3.metric("Lolos Screening", f"{screened.shape[0]:,}".replace(",", "."))
m4.metric("Rata-rata Net Foreign", f"{avg_net_foreign/1e9:,.2f} M".replace(",", "."))
m5.metric("Market Breadth", f"{breadth_pct:.0f}% naik", f"{advancers}↑ / {decliners}↓ / {unchanged}=")

st.markdown("---")


# ==========================================================================
# TENTUKAN SAHAM YANG DIPILIH (search override top10)
# ==========================================================================

all_codes = sorted(scored_all["Kode Saham"].unique().tolist())
if search_query:
    matched_codes = [c for c in all_codes if search_query in c]
    if not matched_codes:
        st.sidebar.warning(f"Kode mengandung '{search_query}' tidak ditemukan. Menampilkan Top 10.")
        matched_codes = top10["Kode Saham"].tolist() or all_codes[:10]
else:
    matched_codes = top10["Kode Saham"].tolist() or all_codes[:10]


# ==========================================================================
# LAYOUT UTAMA: PANEL KIRI (WATCHLIST) & PANEL KANAN (DETAIL)
# ==========================================================================

left_col, right_col = st.columns([1, 1.6], gap="large")

# -------------------- PANEL KIRI: TOP 10 WATCHLIST -----------------------
with left_col:
    st.subheader("🏆 Top 10 Watchlist")
    if top10.empty:
        st.warning("Tidak ada saham yang lolos filter screening saat ini. Longgarkan filter di sidebar.")
    else:
        display_top10 = top10[["Kode Saham", "Nama Perusahaan", "Overall Score", "Signal"]].copy()
        display_top10.insert(0, "Rank", range(1, len(display_top10) + 1))
        st.dataframe(
            display_top10,
            use_container_width=True,
            hide_index=True,
            height=min(420, 45 * (len(display_top10) + 1)),
            column_config={
                "Overall Score": st.column_config.ProgressColumn(
                    "Overall Score", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )

    st.markdown("#### Pilih Saham untuk Detail")
    default_index = 0
    selected_code = st.selectbox(
        "Kode Saham", matched_codes, index=default_index, label_visibility="collapsed",
    )

# ---------------------- PANEL KANAN: DETAIL SAHAM -------------------------
sel_row = scored_all[scored_all["Kode Saham"] == selected_code]

with right_col:
    if sel_row.empty:
        st.info("Pilih saham di panel kiri untuk melihat detail.")
    else:
        row = sel_row.iloc[0]
        st.subheader(f"{row['Kode Saham']} — {row['Nama Perusahaan']}")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Overall Score", f"{row['Overall Score']:.1f}" if pd.notna(row["Overall Score"]) else "N/A")
        sc2.metric("Signal", row["Signal"])
        sc3.metric("Penutupan", f"{row['Penutupan']:,.0f}".replace(",", "."))
        chg = row["close_chg_last"]
        sc4.metric("Perubahan Terakhir", f"{chg*100:.2f}%" if pd.notna(chg) else "N/A")

        tabs = st.tabs([
            "💹 Harga", "📊 Volume", "🌐 Foreign Flow", "💰 Nilai Transaksi",
            "📗 Bid vs Offer", "🧊 Non Regular", "📋 Ringkasan Metrik",
        ])

        dates = [d.strftime("%Y-%m-%d") for d in row["_hist_dates"]]

        # ---- TAB 1: HARGA (candlestick 3 hari) ----
        with tabs[0]:
            fig = go.Figure(data=[go.Candlestick(
                x=dates,
                open=row["_hist_open"], high=row["_hist_high"],
                low=row["_hist_low"], close=row["_hist_close"],
                increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
                name="Harga",
            )])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric(f"Perubahan {lookback_days} Hari", f"{row['close_chg_3d']*100:.2f}%" if pd.notna(row['close_chg_3d']) else "N/A")
            colB.metric("Avg Range Harian", f"{row['avg_range_pct']*100:.2f}%" if pd.notna(row['avg_range_pct']) else "N/A")
            colC.metric("Breakout", "Ya ✅" if row["breakout"] else "Tidak")

        # ---- TAB 2: VOLUME ----
        with tabs[1]:
            fig = go.Figure(data=[go.Bar(x=dates, y=row["_hist_volume"], marker_color=COLOR_ACCENT, name="Volume")])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("Volume Berubah (Terakhir)", f"{row['vol_chg_last']*100:.1f}%" if pd.notna(row['vol_chg_last']) else "N/A")
            colB.metric("Volume Spike", f"{row['vol_spike']:.2f}x" if pd.notna(row['vol_spike']) else "N/A")
            colC.metric(f"Avg Volume {lookback_days} Hari", f"{row['avg_vol_3d']:,.0f}".replace(",", "."))

        # ---- TAB 3: FOREIGN FLOW ----
        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=row["_hist_fbuy"], name="Foreign Buy", marker_color=COLOR_UP))
            fig.add_trace(go.Bar(x=dates, y=[-v for v in row["_hist_fsell"]], name="Foreign Sell", marker_color=COLOR_DOWN))
            fig.add_trace(go.Scatter(x=dates, y=row["_hist_net_foreign"], name="Net Foreign",
                                      mode="lines+markers", line=dict(color=COLOR_ACCENT, width=3)))
            fig.update_layout(template=PLOTLY_TEMPLATE, barmode="relative", height=380,
                               margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric(f"Net Foreign {lookback_days} Hari", f"{row['net_foreign_total']/1e9:,.2f} M".replace(",", "."))
            status = "Akumulasi 📈" if row["accumulation"] else ("Distribusi 📉" if row["distribution"] else "Netral")
            colB.metric("Status Trend", status)
            colC.metric("Foreign Participation", f"{row['foreign_participation']*100:.1f}%" if pd.notna(row['foreign_participation']) else "N/A")

        # ---- TAB 4: NILAI TRANSAKSI ----
        with tabs[3]:
            fig = go.Figure(data=[go.Bar(x=dates, y=row["_hist_nilai"], marker_color="#a78bfa", name="Nilai")])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("Nilai Berubah (Terakhir)", f"{row['nilai_chg_last']*100:.1f}%" if pd.notna(row['nilai_chg_last']) else "N/A")
            colB.metric(f"Avg Nilai {lookback_days} Hari", f"{row['avg_nilai_3d']/1e9:,.2f} M".replace(",", "."))
            colC.metric("Konsistensi (Nilai Min)", f"{row['nilai_consistency']/1e9:,.2f} M".replace(",", "."))

        # ---- TAB 5: BID VS OFFER ----
        with tabs[4]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Bid Volume", "Offer Volume"], y=[row["Bid Volume"], row["Offer Volume"]],
                                  marker_color=[COLOR_UP, COLOR_DOWN]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Bid", f"{row['Bid']:,.0f}".replace(",", "."))
            colB.metric("Offer", f"{row['Offer']:,.0f}".replace(",", "."))
            colC.metric("Dominance (Bid)", f"{row['bid_offer_dominance']*100:.1f}%" if pd.notna(row['bid_offer_dominance']) else "N/A")
            colD.metric("Spread", f"{row['spread_pct']*100:.2f}%" if pd.notna(row['spread_pct']) else "N/A")

        # ---- TAB 6: NON REGULAR ----
        with tabs[5]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=row["_hist_nr_val"], name="Non Regular Value", marker_color="#f59e0b"))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            colA, colB, colC = st.columns(3)
            colA.metric("NR Value Ratio (Terakhir)", f"{row['nr_ratio_last']*100:.1f}%" if pd.notna(row['nr_ratio_last']) else "N/A")
            colB.metric("Block Trade Terdeteksi", "Ya ⚠️" if row["block_trade_flag"] else "Tidak")
            colC.metric("Trend NR Frekuensi", f"{row['nr_freq_trend']:+.0f}" if pd.notna(row['nr_freq_trend']) else "N/A")

        # ---- TAB 7: RINGKASAN METRIK ----
        with tabs[6]:
            summary = {
                "Penutupan": f"{row['Penutupan']:,.0f}",
                "Volume": f"{row['Volume']:,.0f}",
                "Nilai": f"{row['Nilai']:,.0f}",
                "Frekuensi": f"{row['Frekuensi']:,.0f}",
                "Foreign Buy": f"{row['Foreign Buy']:,.0f}",
                "Foreign Sell": f"{row['Foreign Sell']:,.0f}",
                "Free Float Ratio (proxy)": f"{row['free_float_ratio']*100:.1f}%" if pd.notna(row['free_float_ratio']) else "N/A",
                "Turnover Ratio": f"{row['turnover_ratio']*100:.3f}%" if pd.notna(row['turnover_ratio']) else "N/A",
                "Market Cap (proxy)": f"{row['market_cap_proxy']/1e12:,.2f} T" if pd.notna(row['market_cap_proxy']) else "N/A",
                f"Volatility ({lookback_days} Hari)": f"{row['volatility']*100:.2f}%" if pd.notna(row['volatility']) else "N/A",
            }
            st.table(pd.DataFrame(summary.items(), columns=["Metrik", "Nilai"]))

        # ---- PENJELASAN OTOMATIS ----
        st.markdown("#### 🧠 Penjelasan Otomatis")
        reasons = A.generate_explanation(row)
        bullet_html = "".join([f"<li>{r}</li>" for r in reasons])
        st.markdown(
            f"<div class='explain-box'><b>{row['Kode Saham']}</b> mendapatkan Overall Score "
            f"<b>{row['Overall Score']:.1f}</b> ({row['Signal']}) karena:<ul>{bullet_html}</ul></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")


# ==========================================================================
# TABEL RANKING LENGKAP
# ==========================================================================

st.subheader("📋 Ranking Saham (Hasil Screening)")

ranking_df = screened.copy()
if search_query:
    ranking_df = ranking_df[ranking_df["Kode Saham"].str.contains(search_query, na=False)]

if ranking_df.empty:
    st.warning("Tidak ada saham yang cocok dengan filter/pencarian saat ini.")
else:
    show_df = ranking_df[RANKING_COLUMNS].reset_index(drop=True)
    show_df.insert(0, "Rank", range(1, len(show_df) + 1))
    st.dataframe(
        show_df,
        use_container_width=True,
        hide_index=True,
        height=min(600, 40 * (len(show_df) + 1)),
        column_config={
            "Overall Score": st.column_config.ProgressColumn("Overall Score", min_value=0, max_value=100, format="%.1f"),
            "Momentum Score": st.column_config.NumberColumn(format="%.1f"),
            "Liquidity Score": st.column_config.NumberColumn(format="%.1f"),
            "Foreign Score": st.column_config.NumberColumn(format="%.1f"),
            "Orderbook Score": st.column_config.NumberColumn(format="%.1f"),
            "Bandar Activity Score": st.column_config.NumberColumn(format="%.1f"),
            "Risk Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

st.caption(
    "Overall Opportunity Score menggabungkan Momentum (25%), Liquidity (15%), Foreign Flow (20%), "
    "Orderbook (15%), Bandar Activity (15%), dan Risk (10%) — bukan hanya berdasarkan Volume atau Foreign Flow saja. "
    "Risk Score: semakin tinggi = semakin aman (volatilitas rendah, spread ketat, tidak ada distribusi asing)."
)
