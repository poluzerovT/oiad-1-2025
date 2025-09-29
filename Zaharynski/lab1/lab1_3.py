from pathlib import Path
import numpy as np
import pandas as pd

from lab1_1 import get_sleep_hours
from lab1_2 import run_normality_checks

from scipy.stats import boxcox, yeojohnson
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def iqr_filter(series, k=1.5):
    """
    Удаляет выбросы по IQR. Если нет выбросов (как в данных), серия не меняется.
    """
    q25, q75 = series.quantile([0.25, 0.75])
    iqr = q75 - q25
    low = q25 - k * iqr
    high = q75 + k * iqr
    filtered = series[(series >= low) & (series <= high)].copy()
    print(f"IQR filter: removed {len(series) - len(filtered)} outliers.")
    return filtered

def winsorize_series(series, limits=(0.01, 0.01)):
    arr = series.to_numpy()
    win = winsorize(arr, limits=limits)
    return pd.Series(np.array(win).astype(float), index=series.index)

def log1p_series(series):
    return np.log1p(series)

def boxcox_series(series):
    arr = series.to_numpy()
    if np.any(arr <= 0):
        shift = abs(arr.min()) + 1e-6
        arr = arr + shift
    else:
        shift = 0.0
    transformed, lmbda = boxcox(arr)
    return pd.Series(transformed, index=series.index), lmbda, shift

def yeojohnson_series(series):
    arr = series.to_numpy()
    transformed, lmbda = yeojohnson(arr)
    return pd.Series(transformed, index=series.index), lmbda

def zscore_series(series):
    scaler = StandardScaler()
    transformed = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()
    return pd.Series(transformed, index=series.index)

def minmax_series(series):
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()
    return pd.Series(transformed, index=series.index)

def safe_run_checks(series, name, alpha=0.05, show=False):
    """
    Wrapper: создает папку outputs/<name>/ и вызывает run_normality_checks.
    """
    outdir = Path("outputs/3") / name
    outdir.mkdir(parents=True, exist_ok=True)
    print("\n--- Transform:", name, "-> outputs in", str(outdir.resolve()), "---")
    res = run_normality_checks(series, outdir=str(outdir), alpha=alpha, show=show)
    return res

if __name__ == "__main__":
    try:
        sleep_hours = get_sleep_hours()
        base = sleep_hours.dropna().astype(float).reset_index(drop=True)
        results_summary = []

        print("\n=== Baseline (raw) ===")
        res_raw = safe_run_checks(base, "raw")
        results_summary.append(("raw", res_raw))

        trimmed = iqr_filter(base)
        res_trim = safe_run_checks(trimmed.reset_index(drop=True), "iqr_trimmed")
        results_summary.append(("iqr_trimmed", res_trim))

        wins = winsorize_series(base, limits=(0.01, 0.01))
        res_wins = safe_run_checks(wins.reset_index(drop=True), "winsorize_1pct")
        results_summary.append(("winsorize_1pct", res_wins))

        log1p = log1p_series(base)
        res_log1p = safe_run_checks(pd.Series(log1p).reset_index(drop=True), "log1p")
        results_summary.append(("log1p", res_log1p))

        try:
            bc_series, bc_lambda, bc_shift = boxcox_series(base)
            res_boxcox = safe_run_checks(bc_series.reset_index(drop=True), "boxcox")
            res_boxcox['boxcox_lambda'] = float(bc_lambda)
            res_boxcox['boxcox_shift'] = float(bc_shift)
            results_summary.append(("boxcox", res_boxcox))
        except Exception as e:
            print("Box-Cox failed:", e)

        try:
            yj_series, yj_lambda = yeojohnson_series(base)
            res_yj = safe_run_checks(yj_series.reset_index(drop=True), "yeojohnson")
            res_yj['yeojohnson_lambda'] = float(yj_lambda)
            results_summary.append(("yeojohnson", res_yj))
        except Exception as e:
            print("Yeo-Johnson failed:", e)

        zsc = zscore_series(base)
        res_z = safe_run_checks(zsc.reset_index(drop=True), "zscore")
        results_summary.append(("zscore", res_z))

        mm = minmax_series(base)
        res_mm = safe_run_checks(mm.reset_index(drop=True), "minmax")
        results_summary.append(("minmax", res_mm))

        print("\n\n=== Summary table (Shapiro p, Skew p, Kurt p) ===")
        summary_rows = []
        for name, r in results_summary:
            shapiro_p = r.get("shapiro")[1] if r.get("shapiro") else None
            skew_p = r.get("skew_kurt", {}).get("p_skew")
            kurt_p = r.get("skew_kurt", {}).get("p_kurt")
            notes = ""
            if "boxcox_lambda" in r:
                notes = f"lambda={r['boxcox_lambda']:.4f}, shift={r['boxcox_shift']:.6f}"
            elif "yeojohnson_lambda" in r:
                notes = f"lambda={r['yeojohnson_lambda']:.4f}"
            summary_rows.append({
                "transform": name,
                "shapiro_p": shapiro_p,
                "skew_p": skew_p,
                "kurt_p": kurt_p,
                "notes": notes
            })
        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.sort_values(by=["shapiro_p"], ascending=False).to_string(index=False))
        print("\nПримечание: Выберите трансформацию с наибольшим Shapiro p и незначимыми skew/kurt. Проверьте графики в outputs/<transform>/.")
    except Exception as e:
        print("Ошибка при выполнении lab1_3:", e)