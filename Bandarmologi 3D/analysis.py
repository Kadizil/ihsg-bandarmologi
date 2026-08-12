"""
analysis.py
============
Modul inti Decision Support System untuk Dashboard Saham.

Tanggung jawab modul ini:
1. Memuat dataset master_saham.csv
2. Menentukan window N hari perdagangan terakhir secara OTOMATIS
   (N = LOOKBACK, bisa diubah lewat parameter n_days di run_pipeline /
   build_stock_features, berdasarkan tanggal terbaru yang ada di dataset)
3. Menghitung seluruh fitur analisis dari SEMUA kolom yang tersedia
   (harga, volume, nilai, frekuensi, foreign flow, bid/offer, non-regular,
   shares/free-float)
4. Mengubah fitur mentah menjadi skor 0-100 (percentile rank di dalam
   populasi saham yang sedang di-screening)
5. Menggabungkan sub-skor menjadi Overall Opportunity Score
6. Menentukan sinyal (BUY / WATCH / WAIT / AVOID)
7. Menghasilkan narasi penjelasan otomatis per saham

Modul ini murni pandas/numpy (tidak bergantung pada Streamlit) supaya
mudah di-test terpisah dari UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

DATE_COL = "Tanggal Perdagangan Terakhir"
CODE_COL = "Kode Saham"
NAME_COL = "Nama Perusahaan"

N_LOOKBACK_DAYS = 5  # jumlah hari perdagangan terakhir yang dipakai (otomatis dari data)

# ==========================================
# LOOKBACK CONFIG
# ==========================================

LOOKBACK = N_LOOKBACK_DAYS

RAW_COLUMNS = [
    "No", "Kode Saham", "Nama Perusahaan", "Remarks", "Sebelumnya",
    "Open Price", "Tanggal Perdagangan Terakhir", "First Trade", "Tertinggi",
    "Terendah", "Penutupan", "Selisih", "Volume", "Nilai", "Frekuensi",
    "Index Individual", "Offer", "Offer Volume", "Bid", "Bid Volume",
    "Listed Shares", "Tradeble Shares", "Weight For Index", "Foreign Sell",
    "Foreign Buy", "Non Regular Volume", "Non Regular Value",
    "Non Regular Frequency",
]


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load master_saham.csv dan bersihkan tipe data dasar."""
    df = pd.read_csv(path)

    # Pastikan kolom yang dibutuhkan ada
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[CODE_COL] = df[CODE_COL].astype(str).str.strip()
    df[NAME_COL] = df[NAME_COL].astype(str).str.strip()

    numeric_cols = [c for c in RAW_COLUMNS if c not in
                     (CODE_COL, NAME_COL, "Remarks", DATE_COL, "No")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[DATE_COL, CODE_COL])
    df = df.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    return df


def get_lookback_window(df: pd.DataFrame, n_days: int = N_LOOKBACK_DAYS):
    """Tentukan N hari perdagangan terakhir secara otomatis dari data."""
    all_dates = sorted(df[DATE_COL].unique())
    window_dates = all_dates[-n_days:]
    return window_dates, all_dates


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (N-hari window per saham, N = LOOKBACK/n_days)
# --------------------------------------------------------------------------

def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b == 0, np.nan, a / b)
    return out


def build_stock_features(df: pd.DataFrame, n_days: int = N_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Untuk setiap Kode Saham, ambil n_days terakhir (berdasarkan window global)
    dan hitung seluruh fitur turunan. Mengembalikan 1 baris per saham.

    Saham yang datanya kurang dari n_days hari dalam window akan tetap
    dihitung sebisa mungkin, tetapi diberi flag `data_complete=False` dan
    beberapa fitur trend akan menjadi NaN (otomatis tidak lolos screening
    penuh, tapi tetap terlihat di data mentah).
    """
    window_dates, all_dates = get_lookback_window(df, n_days)
    sub = df[df[DATE_COL].isin(window_dates)].copy()
    sub = sub.sort_values([CODE_COL, DATE_COL])

    rows = []
    for code, g in sub.groupby(CODE_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        n = len(g)
        complete = n >= n_days
        # Ambil n_days terakhir yang tersedia untuk saham ini
        g = g.tail(n_days).reset_index(drop=True)

        name = g[NAME_COL].iloc[-1]
        latest = g.iloc[-1]

        # Helper untuk ambil kolom sebagai list numerik sepanjang window
        def col(c):
            return g[c].to_numpy(dtype=float)

        close = col("Penutupan")
        open_ = col("Open Price")
        high = col("Tertinggi")
        low = col("Terendah")
        vol = col("Volume")
        nilai = col("Nilai")
        freq = col("Frekuensi")
        fbuy = col("Foreign Buy")
        fsell = col("Foreign Sell")
        bid = col("Bid")
        offer = col("Offer")
        bidvol = col("Bid Volume")
        offervol = col("Offer Volume")
        nr_vol = col("Non Regular Volume")
        nr_val = col("Non Regular Value")
        nr_freq = col("Non Regular Frequency")
        listed = col("Listed Shares")
        tradeble = col("Tradeble Shares")
        weight_idx = col("Weight For Index")

        m = len(close)  # jumlah hari aktual yang tersedia (<= n_days)

        # ---------------- HARGA ----------------
        close_chg = np.diff(close) / np.where(close[:-1] == 0, np.nan, close[:-1])
        close_chg_last = close_chg[-1] if len(close_chg) else np.nan
        close_chg_3d = (close[-1] - close[0]) / close[0] if m >= 2 and close[0] != 0 else np.nan
        open_chg_last = _safe_div(open_[-1] - open_[-2], open_[-2]) if m >= 2 else np.nan
        high_chg_last = _safe_div(high[-1] - high[-2], high[-2]) if m >= 2 else np.nan
        low_chg_last = _safe_div(low[-1] - low[-2], low[-2]) if m >= 2 else np.nan

        daily_range_pct = _safe_div(high - low, close)
        avg_range_pct = np.nanmean(daily_range_pct) if m else np.nan

        volatility = np.nanstd(close_chg) if len(close_chg) >= 1 else np.nan

        prior_high = np.max(high[:-1]) if m >= 2 else np.nan
        breakout = bool(m >= 2 and high[-1] > prior_high)
        # Trend dihitung atas SELURUH hari yang tersedia dalam window (dinamis
        # terhadap n_days), bukan hanya 3 hari pertama secara hardcode.
        uptrend_close = bool(m >= 3 and np.all(np.diff(close) > 0))
        downtrend_close = bool(m >= 3 and np.all(np.diff(close) < 0))

        # ---------------- VOLUME ----------------
        vol_chg = np.diff(vol) / np.where(vol[:-1] == 0, np.nan, vol[:-1])
        vol_chg_last = vol_chg[-1] if len(vol_chg) else np.nan
        vol_accel = (vol_chg[-1] - vol_chg[-2]) if len(vol_chg) >= 2 else np.nan
        avg_vol_3d = np.nanmean(vol) if m else np.nan
        avg_vol_prev = np.nanmean(vol[:-1]) if m >= 2 else np.nan
        vol_spike = _safe_div(vol[-1], avg_vol_prev) if m >= 2 else np.nan

        # ---------------- NILAI TRANSAKSI ----------------
        nilai_chg = np.diff(nilai) / np.where(nilai[:-1] == 0, np.nan, nilai[:-1])
        nilai_chg_last = nilai_chg[-1] if len(nilai_chg) else np.nan
        nilai_accel = (nilai_chg[-1] - nilai_chg[-2]) if len(nilai_chg) >= 2 else np.nan
        avg_nilai_3d = np.nanmean(nilai) if m else np.nan
        nilai_consistency = np.min(nilai) if m else np.nan  # konsisten = nilai terkecil tetap besar

        # ---------------- FREKUENSI ----------------
        freq_chg = np.diff(freq) / np.where(freq[:-1] == 0, np.nan, freq[:-1])
        freq_chg_last = freq_chg[-1] if len(freq_chg) else np.nan
        freq_accel = (freq_chg[-1] - freq_chg[-2]) if len(freq_chg) >= 2 else np.nan
        avg_freq_3d = np.nanmean(freq) if m else np.nan

        # ---------------- FOREIGN FLOW ----------------
        net_foreign = fbuy - fsell
        net_foreign_total = np.nansum(net_foreign) if m else np.nan
        net_foreign_last = net_foreign[-1] if m else np.nan
        # Sama seperti trend harga: dihitung atas seluruh window (dinamis
        # terhadap n_days), bukan cuma 3 hari pertama.
        accumulation = bool(m >= 3 and np.all(np.diff(net_foreign) > 0) and net_foreign[-1] > 0)
        distribution = bool(m >= 3 and np.all(np.diff(net_foreign) < 0) and net_foreign[-1] < 0)
        foreign_accel = (net_foreign[-1] - net_foreign[-2]) - (net_foreign[-2] - net_foreign[-3]) if m >= 3 else np.nan
        foreign_intensity = _safe_div(net_foreign_total, avg_nilai_3d) if avg_nilai_3d else np.nan
        foreign_participation = _safe_div(fbuy[-1] + fsell[-1], vol[-1]) if m else np.nan

        # ---------------- BID / OFFER (snapshot hari terakhir) ----------------
        bv, ov = bidvol[-1], offervol[-1]
        bid_offer_dominance = _safe_div(bv, bv + ov) if m else np.nan
        imbalance = bv - ov if m else np.nan
        spread_pct = _safe_div(offer[-1] - bid[-1], close[-1]) if m and close[-1] else np.nan
        support_gap_pct = _safe_div(close[-1] - bid[-1], close[-1]) if m and close[-1] else np.nan
        resistance_gap_pct = _safe_div(offer[-1] - close[-1], close[-1]) if m and close[-1] else np.nan

        # ---------------- NON REGULAR ----------------
        nr_ratio = _safe_div(nr_val, nilai)
        nr_ratio_last = nr_ratio[-1] if m else np.nan
        nr_ratio_first = nr_ratio[0] if m else np.nan
        nr_trend = (nr_ratio_last - nr_ratio_first) if m >= 2 else np.nan
        nr_freq_trend = (nr_freq[-1] - nr_freq[0]) if m >= 2 else np.nan
        block_trade_flag = bool(m and nr_ratio_last is not None and not np.isnan(nr_ratio_last) and nr_ratio_last > 0.15)

        # ---------------- SHARES / FREE FLOAT / INDEX ----------------
        free_float_ratio = _safe_div(weight_idx[-1], listed[-1]) if m and listed[-1] else np.nan
        if free_float_ratio is not None and not np.isnan(free_float_ratio):
            free_float_ratio = min(free_float_ratio, 1.0)  # guard data anomaly (ratio tak boleh > 1)
        turnover_ratio = _safe_div(vol[-1], tradeble[-1]) if m and tradeble[-1] else np.nan
        market_cap_proxy = close[-1] * listed[-1] if m else np.nan
        index_weight_raw = weight_idx[-1] if m else np.nan

        # ---------------- BANDAR / PV DIVERGENCE ----------------
        # Volume naik besar tapi harga nyaris tidak bergerak -> indikasi akumulasi/distribusi diam-diam
        pv_divergence = (vol_chg_last - abs(close_chg_last)) if (
            vol_chg_last is not None and not np.isnan(vol_chg_last) and
            close_chg_last is not None and not np.isnan(close_chg_last)
        ) else np.nan

        rows.append({
            "Kode Saham": code,
            "Nama Perusahaan": name,
            "data_complete": complete,
            "n_hari_tersedia": m,
            "Tanggal Terakhir": g[DATE_COL].iloc[-1],
            # raw latest-day snapshot (untuk filter & tampilan)
            "Penutupan": close[-1] if m else np.nan,
            "Open Price": open_[-1] if m else np.nan,
            "Tertinggi": high[-1] if m else np.nan,
            "Terendah": low[-1] if m else np.nan,
            "Volume": vol[-1] if m else np.nan,
            "Nilai": nilai[-1] if m else np.nan,
            "Frekuensi": freq[-1] if m else np.nan,
            "Foreign Buy": fbuy[-1] if m else np.nan,
            "Foreign Sell": fsell[-1] if m else np.nan,
            "Bid": bid[-1] if m else np.nan,
            "Offer": offer[-1] if m else np.nan,
            "Bid Volume": bv,
            "Offer Volume": ov,
            "Non Regular Volume": nr_vol[-1] if m else np.nan,
            "Non Regular Value": nr_val[-1] if m else np.nan,
            "Non Regular Frequency": nr_freq[-1] if m else np.nan,
            "Listed Shares": listed[-1] if m else np.nan,
            "Tradeble Shares": tradeble[-1] if m else np.nan,
            "Weight For Index": index_weight_raw,
            # data historis mentah sepanjang window n_days (untuk chart)
            "_hist_dates": g[DATE_COL].tolist(),
            "_hist_close": close.tolist(),
            "_hist_open": open_.tolist(),
            "_hist_high": high.tolist(),
            "_hist_low": low.tolist(),
            "_hist_volume": vol.tolist(),
            "_hist_nilai": nilai.tolist(),
            "_hist_freq": freq.tolist(),
            "_hist_fbuy": fbuy.tolist(),
            "_hist_fsell": fsell.tolist(),
            "_hist_net_foreign": net_foreign.tolist(),
            "_hist_nr_val": nr_val.tolist(),
            "_hist_nr_vol": nr_vol.tolist(),
            "_hist_nr_freq": nr_freq.tolist(),
            # ---- fitur turunan ----
            "close_chg_last": close_chg_last,
            "close_chg_3d": close_chg_3d,
            "open_chg_last": open_chg_last,
            "high_chg_last": high_chg_last,
            "low_chg_last": low_chg_last,
            "avg_range_pct": avg_range_pct,
            "volatility": volatility,
            "breakout": breakout,
            "uptrend_close": uptrend_close,
            "downtrend_close": downtrend_close,
            "vol_chg_last": vol_chg_last,
            "vol_accel": vol_accel,
            "avg_vol_3d": avg_vol_3d,
            "vol_spike": vol_spike,
            "nilai_chg_last": nilai_chg_last,
            "nilai_accel": nilai_accel,
            "avg_nilai_3d": avg_nilai_3d,
            "nilai_consistency": nilai_consistency,
            "freq_chg_last": freq_chg_last,
            "freq_accel": freq_accel,
            "avg_freq_3d": avg_freq_3d,
            "net_foreign_total": net_foreign_total,
            "net_foreign_last": net_foreign_last,
            "accumulation": accumulation,
            "distribution": distribution,
            "foreign_accel": foreign_accel,
            "foreign_intensity": foreign_intensity,
            "foreign_participation": foreign_participation,
            "bid_offer_dominance": bid_offer_dominance,
            "imbalance": imbalance,
            "spread_pct": spread_pct,
            "support_gap_pct": support_gap_pct,
            "resistance_gap_pct": resistance_gap_pct,
            "nr_ratio_last": nr_ratio_last,
            "nr_trend": nr_trend,
            "nr_freq_trend": nr_freq_trend,
            "block_trade_flag": block_trade_flag,
            "free_float_ratio": free_float_ratio,
            "turnover_ratio": turnover_ratio,
            "market_cap_proxy": market_cap_proxy,
            "pv_divergence": pv_divergence,
        })

    feat = pd.DataFrame(rows)
    return feat


# --------------------------------------------------------------------------
# 3. SCORING (percentile rank -> 0-100)
# --------------------------------------------------------------------------

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank 0-100, NaN -> diperlakukan sebagai nilai terendah (0)."""
    s = series.copy()
    valid = s.notna()
    ranked = pd.Series(np.nan, index=s.index)
    if valid.sum() > 0:
        r = s[valid].rank(pct=True, ascending=ascending) * 100
        ranked.loc[valid] = r
    ranked = ranked.fillna(0)
    return ranked


def compute_scores(feat: pd.DataFrame) -> pd.DataFrame:
    """Hitung sub-skor (0-100) dan Overall Opportunity Score."""
    f = feat.copy()

    # ---------------- MOMENTUM SCORE ----------------
    p_close3d = _pct_rank(f["close_chg_3d"])
    p_close_last = _pct_rank(f["close_chg_last"])
    p_breakout = f["breakout"].astype(float) * 100
    p_vol_spike = _pct_rank(f["vol_spike"])
    p_vol_accel = _pct_rank(f["vol_accel"])
    p_freq_accel = _pct_rank(f["freq_accel"])
    p_nilai_accel = _pct_rank(f["nilai_accel"])
    f["Momentum Score"] = (
        0.25 * p_close3d + 0.15 * p_close_last + 0.15 * p_breakout +
        0.15 * p_vol_spike + 0.10 * p_vol_accel + 0.10 * p_freq_accel +
        0.10 * p_nilai_accel
    )

    # ---------------- LIQUIDITY SCORE ----------------
    p_avg_nilai = _pct_rank(f["avg_nilai_3d"])
    p_avg_vol = _pct_rank(f["avg_vol_3d"])
    p_turnover = _pct_rank(f["turnover_ratio"])
    p_freefloat = _pct_rank(f["free_float_ratio"])
    p_depth = _pct_rank(f["Bid Volume"] + f["Offer Volume"])
    p_consistency = _pct_rank(f["nilai_consistency"])
    f["Liquidity Score"] = (
        0.30 * p_avg_nilai + 0.20 * p_avg_vol + 0.20 * p_turnover +
        0.10 * p_freefloat + 0.10 * p_depth + 0.10 * p_consistency
    )

    # ---------------- FOREIGN SCORE ----------------
    p_net_total = _pct_rank(f["net_foreign_total"])
    p_net_last = _pct_rank(f["net_foreign_last"])
    p_accum = f["accumulation"].astype(float) * 100
    p_f_accel = _pct_rank(f["foreign_accel"])
    p_f_intensity = _pct_rank(f["foreign_intensity"])
    distribution_penalty = f["distribution"].astype(float) * 25  # kurangi skor jika ada distribusi
    f["Foreign Score"] = (
        0.30 * p_net_total + 0.20 * p_net_last + 0.20 * p_accum +
        0.15 * p_f_accel + 0.15 * p_f_intensity - distribution_penalty
    ).clip(0, 100)

    # ---------------- ORDERBOOK SCORE ----------------
    p_dominance = _pct_rank(f["bid_offer_dominance"])
    p_imbalance = _pct_rank(f["imbalance"])
    p_spread = _pct_rank(f["spread_pct"], ascending=False)  # spread kecil = lebih baik
    p_support = _pct_rank(f["support_gap_pct"], ascending=False)  # dekat support = baik
    f["Orderbook Score"] = (
        0.35 * p_dominance + 0.30 * p_imbalance + 0.20 * p_spread + 0.15 * p_support
    )

    # ---------------- BANDAR ACTIVITY SCORE ----------------
    p_nr_trend = _pct_rank(f["nr_trend"])
    p_nr_freq_trend = _pct_rank(f["nr_freq_trend"])
    p_block = f["block_trade_flag"].astype(float) * 100
    p_pv_div = _pct_rank(f["pv_divergence"])
    f["Bandar Activity Score"] = (
        0.30 * p_nr_trend + 0.20 * p_nr_freq_trend + 0.25 * p_block + 0.25 * p_pv_div
    )

    # ---------------- RISK SCORE (100 = paling aman) ----------------
    p_low_volatility = _pct_rank(f["volatility"], ascending=False)
    p_low_spread = _pct_rank(f["spread_pct"], ascending=False)
    p_high_liquidity = _pct_rank(f["avg_nilai_3d"])
    p_no_distribution = (~f["distribution"]).astype(float) * 100
    p_freefloat_safety = _pct_rank(f["free_float_ratio"])
    f["Risk Score"] = (
        0.30 * p_low_volatility + 0.20 * p_low_spread + 0.20 * p_high_liquidity +
        0.20 * p_no_distribution + 0.10 * p_freefloat_safety
    )

    # ---------------- OVERALL OPPORTUNITY SCORE ----------------
    f["Overall Score"] = (
        0.25 * f["Momentum Score"] +
        0.15 * f["Liquidity Score"] +
        0.20 * f["Foreign Score"] +
        0.15 * f["Orderbook Score"] +
        0.15 * f["Bandar Activity Score"] +
        0.10 * f["Risk Score"]
    ).round(1)

    for c in ["Momentum Score", "Liquidity Score", "Foreign Score",
              "Orderbook Score", "Bandar Activity Score", "Risk Score"]:
        f[c] = f[c].round(1)

    # Saham dengan data tidak lengkap tidak layak masuk ranking utama
    f.loc[~f["data_complete"], "Overall Score"] = np.nan

    # ---------------- SIGNAL ----------------
    def signal_row(row):
        score = row["Overall Score"]
        if pd.isna(score):
            return "NO DATA"
        risky = row["Risk Score"] < 30
        if score >= 75:
            return "WATCH" if risky else "BUY"
        elif score >= 60:
            return "WAIT" if risky else "WATCH"
        elif score >= 45:
            return "WAIT"
        else:
            return "AVOID"

    f["Signal"] = f.apply(signal_row, axis=1)

    return f


# --------------------------------------------------------------------------
# 4. NARASI PENJELASAN OTOMATIS
# --------------------------------------------------------------------------

def generate_explanation(row: pd.Series) -> list[str]:
    """Hasilkan bullet-point alasan otomatis berdasarkan data & skor saham."""
    reasons = []

    # Jumlah hari aktual dalam window lookback saham ini (mengikuti n_days
    # yang dipakai saat run_pipeline / build_stock_features dipanggil).
    n_hari = row.get("n_hari_tersedia")
    n_hari_txt = f"{int(n_hari)} hari" if pd.notna(n_hari) else "beberapa hari"

    # Foreign flow
    if row.get("accumulation"):
        reasons.append(f"Foreign Buy meningkat konsisten selama {n_hari_txt} berturut-turut (akumulasi asing)")
    elif row.get("distribution"):
        reasons.append(f"Terdeteksi distribusi asing konsisten selama {n_hari_txt} berturut-turut (net foreign menurun)")
    elif row.get("net_foreign_last", 0) > 0:
        reasons.append("Net foreign hari terakhir positif (foreign buy > foreign sell)")
    elif row.get("net_foreign_last", 0) < 0:
        reasons.append("Net foreign hari terakhir negatif (foreign sell > foreign buy)")

    # Volume
    vc = row.get("vol_chg_last")
    if vc is not None and not pd.isna(vc):
        if vc > 0.3:
            reasons.append(f"Volume naik signifikan {vc*100:.0f}% dibanding hari sebelumnya")
        elif vc < -0.3:
            reasons.append(f"Volume turun signifikan {vc*100:.0f}% dibanding hari sebelumnya")

    if row.get("vol_spike") is not None and not pd.isna(row.get("vol_spike")) and row["vol_spike"] > 1.5:
        n_prev = int(n_hari) - 1 if pd.notna(n_hari) and n_hari >= 1 else None
        n_prev_txt = f"{n_prev} hari" if n_prev else "hari-hari"
        reasons.append(f"Volume spike {row['vol_spike']:.1f}x di atas rata-rata {n_prev_txt} sebelumnya")

    # Nilai transaksi
    nc = row.get("nilai_chg_last")
    if nc is not None and not pd.isna(nc) and nc > 0.3:
        reasons.append(f"Nilai transaksi meningkat {nc*100:.0f}% dibanding hari sebelumnya")

    # Frekuensi
    fc = row.get("freq_chg_last")
    if fc is not None and not pd.isna(fc) and fc > 0.3:
        reasons.append(f"Frekuensi transaksi meningkat {fc*100:.0f}%, menandakan partisipasi pasar melebar")

    # Bid/offer
    dom = row.get("bid_offer_dominance")
    if dom is not None and not pd.isna(dom):
        if dom > 0.6:
            reasons.append(f"Bid Volume dominan ({dom*100:.0f}% dari total bid+offer) — tekanan beli kuat")
        elif dom < 0.4:
            reasons.append(f"Offer Volume dominan ({(1-dom)*100:.0f}% dari total bid+offer) — tekanan jual kuat")

    # Price action
    if row.get("breakout"):
        n_prior = int(n_hari) - 1 if pd.notna(n_hari) and n_hari >= 1 else None
        n_prior_txt = f"{n_prior} hari" if n_prior else "hari-hari"
        reasons.append(f"Harga breakout dari level tertinggi {n_prior_txt} sebelumnya dalam window")
    if row.get("uptrend_close"):
        reasons.append(f"Harga penutupan naik konsisten selama {n_hari_txt} berturut-turut")
    if row.get("downtrend_close"):
        reasons.append(f"Harga penutupan turun konsisten selama {n_hari_txt} berturut-turut")

    # Non regular / bandar
    if row.get("block_trade_flag"):
        reasons.append("Aktivitas Non Regular (block trade) tinggi, indikasi transaksi besar di luar pasar reguler")
    nrt = row.get("nr_trend")
    if nrt is not None and not pd.isna(nrt) and nrt > 0.02:
        reasons.append(f"Aktivitas Non Regular meningkat dalam window {n_hari_txt} terakhir")

    # Risk warning
    if row.get("Risk Score", 100) < 30:
        reasons.append("⚠ Risk Score rendah — volatilitas tinggi dan/atau spread lebar, perlu kehati-hatian")

    if not reasons:
        reasons.append("Tidak ada sinyal signifikan yang terdeteksi dari kombinasi indikator saat ini")

    return reasons


# --------------------------------------------------------------------------
# 5. PIPELINE UTAMA
# --------------------------------------------------------------------------

def run_pipeline(path: str, n_days: int = N_LOOKBACK_DAYS):
    """Load data -> build features -> compute scores. Return (raw_df, feat_scored_df, window_dates)."""
    raw_df = load_data(path)
    window_dates, all_dates = get_lookback_window(raw_df, n_days)
    feat = build_stock_features(raw_df, n_days)
    scored = compute_scores(feat)
    return raw_df, scored, window_dates
