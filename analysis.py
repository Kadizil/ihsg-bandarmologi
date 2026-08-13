"""
analysis.py
============
Core module of the Decision Support System for the Stock Dashboard.
Optimized with vectorized operations and robust mathematical metrics.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
from typing import Union

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DATE_COL = "Tanggal Perdagangan Terakhir"
CODE_COL = "Kode Saham"
NAME_COL = "Nama Perusahaan"

N_LOOKBACK_DAYS = 5 

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

def load_data(source: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Load master_saham.csv safely accepting both file paths and memory buffers."""
    df = pd.read_csv(source)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

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
    all_dates = sorted(df[DATE_COL].unique())
    window_dates = all_dates[-n_days:] if n_days <= len(all_dates) else all_dates
    return window_dates, all_dates


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Optimized)
# --------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b and not np.isnan(b) and b != 0 else np.nan


def build_stock_features(df: pd.DataFrame, n_days: int = N_LOOKBACK_DAYS) -> pd.DataFrame:
    window_dates, all_dates = get_lookback_window(df, n_days)
    
    # Pre-filter to only the relevant window to save memory
    sub = df[df[DATE_COL].isin(window_dates)].copy()
    sub['net_foreign'] = sub['Foreign Buy'] - sub['Foreign Sell']

    def extract_features(g: pd.DataFrame) -> pd.Series:
        m = len(g)
        last = g.iloc[-1]
        first = g.iloc[0]
        
        close = g["Penutupan"].values
        open_ = g["Open Price"].values
        high = g["Tertinggi"].values
        low = g["Terendah"].values
        vol = g["Volume"].values
        nilai = g["Nilai"].values
        freq = g["Frekuensi"].values
        nf = g["net_foreign"].values
        
        close_chg_last = _safe_div(close[-1] - close[-2], close[-2]) if m >= 2 else np.nan
        close_chg_window = _safe_div(close[-1] - close[0], close[0]) if m >= 2 else np.nan
        
        daily_range_pct = (high - low) / np.where(close == 0, np.nan, close)
        avg_range_pct = np.nanmean(daily_range_pct) if m else np.nan

        # Volatility requires at least 3 days to be statistically meaningful
        volatility = np.nanstd(g["Penutupan"].pct_change().dropna()) if m >= 3 else np.nan

        # Structural breakout check: price exceeds highest point of previous days in window
        prior_high = np.max(high[:-1]) if m >= 2 else np.nan
        breakout = bool(m >= 2 and high[-1] > prior_high and close[-1] > open_[-1])

        # Realistic trend checks
        uptrend_close = bool(m >= 2 and close[-1] > close[0] and close[-1] > np.nanmean(close))
        downtrend_close = bool(m >= 2 and close[-1] < close[0] and close[-1] < np.nanmean(close))

        # Volume dynamics
        vol_chg_last = _safe_div(vol[-1] - vol[-2], vol[-2]) if m >= 2 else np.nan
        avg_vol_window = np.nanmean(vol)
        avg_vol_prev = np.nanmean(vol[:-1]) if m >= 2 else np.nan
        vol_spike = _safe_div(vol[-1], avg_vol_prev) if m >= 2 else np.nan

        # Value dynamics
        nilai_chg_last = _safe_div(nilai[-1] - nilai[-2], nilai[-2]) if m >= 2 else np.nan
        avg_nilai_window = np.nanmean(nilai)
        nilai_consistency = np.min(nilai) if m else np.nan 

        # Frequency dynamics
        freq_chg_last = _safe_div(freq[-1] - freq[-2], freq[-2]) if m >= 2 else np.nan
        avg_freq_window = np.nanmean(freq)

        # Foreign Flow dynamics
        net_foreign_total = np.nansum(nf)
        net_foreign_last = nf[-1]
        
        # Realistic accumulation: Overall positive flow and buying on the last day
        accumulation = bool(m >= 2 and net_foreign_total > 0 and net_foreign_last > 0)
        distribution = bool(m >= 2 and net_foreign_total < 0 and net_foreign_last < 0)
        
        foreign_intensity = _safe_div(net_foreign_total, avg_nilai_window)
        foreign_participation = _safe_div(last["Foreign Buy"] + last["Foreign Sell"], last["Volume"])

        # Orderbook
        bv, ov = last["Bid Volume"], last["Offer Volume"]
        bid_offer_dominance = _safe_div(bv, bv + ov)
        imbalance = bv - ov
        spread_pct = _safe_div(last["Offer"] - last["Bid"], last["Penutupan"])
        
        # Support is more accurately represented by the recent low, not the immediate bid
        recent_low = np.nanmin(low) if m else np.nan
        support_gap_pct = _safe_div(last["Penutupan"] - recent_low, last["Penutupan"])

        # Non Regular
        nr_ratio_last = _safe_div(last["Non Regular Value"], last["Nilai"])
        nr_ratio_first = _safe_div(first["Non Regular Value"], first["Nilai"])
        nr_trend = (nr_ratio_last - nr_ratio_first) if m >= 2 and pd.notna(nr_ratio_last) and pd.notna(nr_ratio_first) else np.nan
        nr_freq_trend = (last["Non Regular Frequency"] - first["Non Regular Frequency"]) if m >= 2 else np.nan
        block_trade_flag = bool(pd.notna(nr_ratio_last) and nr_ratio_last > 0.15)

        # Float / Index
        free_float_ratio = min(_safe_div(last["Weight For Index"], last["Listed Shares"]), 1.0) if pd.notna(last["Listed Shares"]) else np.nan
        turnover_ratio = _safe_div(last["Volume"], last["Tradeble Shares"])
        market_cap_proxy = last["Penutupan"] * last["Listed Shares"] if pd.notna(last["Penutupan"]) else np.nan

        # PV Divergence calculation: Ratio mapping instead of raw subtraction
        # Floor price change at 0.5% to prevent divide-by-zero anomalies
        price_chg_abs = max(abs(close_chg_last), 0.005) if pd.notna(close_chg_last) else np.nan
        pv_divergence = _safe_div(vol_spike, price_chg_abs) if pd.notna(vol_spike) else np.nan

        return pd.Series({
            "Nama Perusahaan": last[NAME_COL],
            "data_complete": m >= n_days,
            "n_hari_tersedia": m,
            "Tanggal Terakhir": last[DATE_COL],
            "Penutupan": last["Penutupan"],
            "Open Price": last["Open Price"],
            "Tertinggi": last["Tertinggi"],
            "Terendah": last["Terendah"],
            "Volume": last["Volume"],
            "Nilai": last["Nilai"],
            "Frekuensi": last["Frekuensi"],
            "Foreign Buy": last["Foreign Buy"],
            "Foreign Sell": last["Foreign Sell"],
            "Bid": last["Bid"],
            "Offer": last["Offer"],
            "Bid Volume": bv,
            "Offer Volume": ov,
            "Non Regular Volume": last["Non Regular Volume"],
            "Non Regular Value": last["Non Regular Value"],
            "Non Regular Frequency": last["Non Regular Frequency"],
            "Listed Shares": last["Listed Shares"],
            "Tradeble Shares": last["Tradeble Shares"],
            "Weight For Index": last["Weight For Index"],
            
            # Chart lists
            "_hist_dates": g[DATE_COL].tolist(),
            "_hist_close": close.tolist(),
            "_hist_open": open_.tolist(),
            "_hist_high": high.tolist(),
            "_hist_low": low.tolist(),
            "_hist_volume": vol.tolist(),
            "_hist_nilai": nilai.tolist(),
            "_hist_freq": freq.tolist(),
            "_hist_fbuy": g["Foreign Buy"].tolist(),
            "_hist_fsell": g["Foreign Sell"].tolist(),
            "_hist_net_foreign": nf.tolist(),
            "_hist_nr_val": g["Non Regular Value"].tolist(),
            "_hist_nr_vol": g["Non Regular Volume"].tolist(),
            "_hist_nr_freq": g["Non Regular Frequency"].tolist(),
            
            # Derived features
            "close_chg_last": close_chg_last,
            "close_chg_window": close_chg_window,
            "avg_range_pct": avg_range_pct,
            "volatility": volatility,
            "breakout": breakout,
            "uptrend_close": uptrend_close,
            "downtrend_close": downtrend_close,
            "vol_chg_last": vol_chg_last,
            "avg_vol_window": avg_vol_window,
            "vol_spike": vol_spike,
            "nilai_chg_last": nilai_chg_last,
            "avg_nilai_window": avg_nilai_window,
            "nilai_consistency": nilai_consistency,
            "freq_chg_last": freq_chg_last,
            "avg_freq_window": avg_freq_window,
            "net_foreign_total": net_foreign_total,
            "net_foreign_last": net_foreign_last,
            "accumulation": accumulation,
            "distribution": distribution,
            "foreign_intensity": foreign_intensity,
            "foreign_participation": foreign_participation,
            "bid_offer_dominance": bid_offer_dominance,
            "imbalance": imbalance,
            "spread_pct": spread_pct,
            "support_gap_pct": support_gap_pct,
            "nr_ratio_last": nr_ratio_last,
            "nr_trend": nr_trend,
            "nr_freq_trend": nr_freq_trend,
            "block_trade_flag": block_trade_flag,
            "free_float_ratio": free_float_ratio,
            "turnover_ratio": turnover_ratio,
            "market_cap_proxy": market_cap_proxy,
            "pv_divergence": pv_divergence,
        })

    feat = sub.groupby(CODE_COL, sort=False).apply(extract_features).reset_index()
    return feat


# --------------------------------------------------------------------------
# 3. SCORING
# --------------------------------------------------------------------------

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank 0-100. NaNs are filled with 50 (neutral) to prevent artificial penalties."""
    s = series.copy()
    valid = s.notna()
    ranked = pd.Series(50.0, index=s.index) # Default neutral
    if valid.sum() > 0:
        r = s[valid].rank(pct=True, ascending=ascending) * 100
        ranked.loc[valid] = r
    return ranked


def compute_scores(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()

    p_close_window = _pct_rank(f["close_chg_window"])
    p_close_last = _pct_rank(f["close_chg_last"])
    p_breakout = f["breakout"].astype(float) * 100
    p_vol_spike = _pct_rank(f["vol_spike"])
    p_vol_accel = _pct_rank(f["vol_chg_last"])
    p_freq_accel = _pct_rank(f["freq_chg_last"])
    p_nilai_accel = _pct_rank(f["nilai_chg_last"])
    
    f["Momentum Score"] = (
        0.25 * p_close_window + 0.15 * p_close_last + 0.15 * p_breakout +
        0.15 * p_vol_spike + 0.10 * p_vol_accel + 0.10 * p_freq_accel +
        0.10 * p_nilai_accel
    )

    p_avg_nilai = _pct_rank(f["avg_nilai_window"])
    p_avg_vol = _pct_rank(f["avg_vol_window"])
    p_turnover = _pct_rank(f["turnover_ratio"])
    p_freefloat = _pct_rank(f["free_float_ratio"])
    p_depth = _pct_rank(f["Bid Volume"] + f["Offer Volume"])
    p_consistency = _pct_rank(f["nilai_consistency"])
    
    f["Liquidity Score"] = (
        0.30 * p_avg_nilai + 0.20 * p_avg_vol + 0.20 * p_turnover +
        0.10 * p_freefloat + 0.10 * p_depth + 0.10 * p_consistency
    )

    p_net_total = _pct_rank(f["net_foreign_total"])
    p_net_last = _pct_rank(f["net_foreign_last"])
    p_accum = f["accumulation"].astype(float) * 100
    p_f_intensity = _pct_rank(f["foreign_intensity"])
    distribution_penalty = f["distribution"].astype(float) * 25 
    
    f["Foreign Score"] = (
        0.35 * p_net_total + 0.25 * p_net_last + 0.25 * p_accum +
        0.15 * p_f_intensity - distribution_penalty
    ).clip(0, 100)

    p_dominance = _pct_rank(f["bid_offer_dominance"])
    p_imbalance = _pct_rank(f["imbalance"])
    p_spread = _pct_rank(f["spread_pct"], ascending=False) 
    p_support = _pct_rank(f["support_gap_pct"], ascending=False) 
    
    f["Orderbook Score"] = (
        0.35 * p_dominance + 0.30 * p_imbalance + 0.20 * p_spread + 0.15 * p_support
    )

    p_nr_trend = _pct_rank(f["nr_trend"])
    p_nr_freq_trend = _pct_rank(f["nr_freq_trend"])
    p_block = f["block_trade_flag"].astype(float) * 100
    p_pv_div = _pct_rank(f["pv_divergence"])
    
    f["Bandar Activity Score"] = (
        0.30 * p_nr_trend + 0.20 * p_nr_freq_trend + 0.25 * p_block + 0.25 * p_pv_div
    )

    p_low_volatility = _pct_rank(f["volatility"], ascending=False)
    p_low_spread = _pct_rank(f["spread_pct"], ascending=False)
    p_high_liquidity = _pct_rank(f["avg_nilai_window"])
    p_no_distribution = (~f["distribution"]).astype(float) * 100
    p_freefloat_safety = _pct_rank(f["free_float_ratio"])
    
    f["Risk Score"] = (
        0.30 * p_low_volatility + 0.20 * p_low_spread + 0.20 * p_high_liquidity +
        0.20 * p_no_distribution + 0.10 * p_freefloat_safety
    )

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

    f.loc[~f["data_complete"], "Overall Score"] = np.nan

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
# 4. EXPLANATION GENERATOR
# --------------------------------------------------------------------------

def generate_explanation(row: pd.Series) -> list[str]:
    reasons = []
    n_hari = row.get("n_hari_tersedia")
    n_hari_txt = f"{int(n_hari)} hari" if pd.notna(n_hari) else "beberapa hari"

    if row.get("accumulation"):
        reasons.append(f"Terdeteksi akumulasi asing solid dalam rentang {n_hari_txt}")
    elif row.get("distribution"):
        reasons.append(f"Terdeteksi distribusi asing (net outflow) dalam rentang {n_hari_txt}")
    elif row.get("net_foreign_last", 0) > 0:
        reasons.append("Net foreign hari terakhir positif (Foreign Buy dominan)")

    vc = row.get("vol_chg_last")
    if pd.notna(vc) and vc > 0.3:
        reasons.append(f"Volume naik signifikan {vc*100:.0f}% dibanding sesi sebelumnya")
    
    if pd.notna(row.get("vol_spike")) and row["vol_spike"] > 1.5:
        reasons.append(f"Volume spike {row['vol_spike']:.1f}x di atas rata-rata")

    if pd.notna(row.get("nilai_chg_last")) and row["nilai_chg_last"] > 0.3:
        reasons.append(f"Nilai transaksi melonjak {row['nilai_chg_last']*100:.0f}%")

    dom = row.get("bid_offer_dominance")
    if pd.notna(dom):
        if dom > 0.6:
            reasons.append(f"Antrean Bid sangat tebal ({dom*100:.0f}% dominasi) — indikasi tahanan harga")
        elif dom < 0.4:
            reasons.append(f"Antrean Offer agresif ({(1-dom)*100:.0f}% dominasi) — tekanan jual kuat")

    if row.get("breakout"):
        reasons.append(f"Terkonfirmasi Breakout menembus high struktural {n_hari_txt} terakhir")
    if row.get("uptrend_close"):
        reasons.append(f"Trend harga harian positif dan berada di atas rata-rata pergerakan")
    
    if row.get("block_trade_flag"):
        reasons.append("Terdeteksi transaksi Non-Regular (Block Trade) bervolume besar")

    if row.get("Risk Score", 100) < 30:
        reasons.append("⚠ Peringatan Risiko: Volatilitas/spread tinggi. Sangat direkomendasikan untuk Wait & See.")

    if not reasons:
        reasons.append("Indikator netral. Tidak ada anomali teknikal/bandarmologi yang mencolok.")

    return reasons


# --------------------------------------------------------------------------
# 5. MAIN PIPELINE
# --------------------------------------------------------------------------

def run_pipeline(source: Union[str, io.BytesIO], n_days: int = N_LOOKBACK_DAYS):
    raw_df = load_data(source)
    window_dates, all_dates = get_lookback_window(raw_df, n_days)
    feat = build_stock_features(raw_df, n_days)
    scored = compute_scores(feat)
    return raw_df, scored, window_dates
