from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab1_1 import get_sleep_hours  # Предполагаем файлы в одной папке

from scipy.stats import norm, chi2 as chi2_dist, shapiro, probplot, skew as _skew, kurtosis as _kurtosis

def combine_bins_until_expected_ok(observed, expected, edges, min_expected=5):
    """
    Объединяет бины до min_expected. Исправлена логика удаления границ edges.
    """
    obs = observed.astype(float).copy()
    exp = expected.astype(float).copy()
    ed = edges.copy()
    while True:
        small = np.where(exp < min_expected)[0]
        if small.size == 0 or len(exp) <= 1:
            break
        i = small[0]
        if i == len(exp) - 1:
            j = i - 1
            obs[j] += obs[i]
            exp[j] += exp[i]
            obs = np.delete(obs, i)
            exp = np.delete(exp, i)
            ed = np.delete(ed, i)
        else:
            obs[i] += obs[i+1]
            exp[i] += exp[i+1]
            obs = np.delete(obs, i + 1)
            exp = np.delete(exp, i + 1)
            ed = np.delete(ed, i + 1)
    return obs, exp, ed

def chi2_normality_test_manual(data, bins='auto', min_expected=5):
    n = len(data)
    if n < 8:
        raise ValueError("Для χ²-теста желательно n >= 8.")
    mu = data.mean()
    sigma = data.std(ddof=1)
    if bins == 'auto':
        m = int(np.sqrt(n))
        if m < 4:
            m = 4
    else:
        m = int(bins)
    counts, edges = np.histogram(data, bins=m)
    cdf_vals = norm.cdf(edges, loc=mu, scale=sigma)
    expected = n * np.diff(cdf_vals)
    observed = counts.astype(float)

    obs_comb, exp_comb, edges_comb = combine_bins_until_expected_ok(observed, expected, edges, min_expected=min_expected)
    m_final = len(exp_comb)
    df_ = m_final - 1 - 2
    if df_ <= 0:
        raise ValueError("Недостаточно степеней свободы после объединения бинов для χ².")
    chi2_stat = float(np.sum((obs_comb - exp_comb)**2 / exp_comb))
    pvalue = float(chi2_dist.sf(chi2_stat, df_))
    return {"chi2": chi2_stat, "df": int(df_), "pvalue": pvalue,
            "observed": obs_comb, "expected": exp_comb, "bin_edges": edges_comb}

def skew_kurtosis_tests(x):
    n = len(x)
    g1 = float(_skew(x, bias=False))
    g2 = float(_kurtosis(x, fisher=True, bias=False))
    SE_skew = np.sqrt(6.0 * n * (n-1) / ((n-2) * (n+1) * (n+3)))
    SE_kurt = np.sqrt(24.0 * n * (n-1)**2 / ((n-3) * (n-2) * (n+3) * (n+5)))
    z_skew = g1 / SE_skew
    z_kurt = g2 / SE_kurt
    p_skew = 2.0 * norm.sf(abs(z_skew))
    p_kurt = 2.0 * norm.sf(abs(z_kurt))
    return {
        "skewness": g1, "SE_skew": SE_skew, "z_skew": z_skew, "p_skew": p_skew,
        "excess_kurtosis": g2, "SE_kurt": SE_kurt, "z_kurt": z_kurt, "p_kurt": p_kurt
    }

def run_normality_checks(sleep_hours: pd.Series, outdir: str = "lab1/outputs/2", alpha: float = 0.05, show: bool = False):
    x = pd.to_numeric(sleep_hours, errors='coerce').dropna().astype(float)
    n = len(x)
    if n == 0:
        raise ValueError("Нет доступных значений в sleep_hours после удаления NaN.")
    print("\n=== Проверка нормальности для Sleep_Hours (n = {}) ===\n".format(n))

    try:
        chi2_res = chi2_normality_test_manual(x, bins='auto', min_expected=5)
        print("Chi-square (manual): chi2 = {:.4f}, df = {}, p = {:.6f}".format(chi2_res['chi2'], chi2_res['df'], chi2_res['pvalue']))
        print("Observed freqs:", np.round(chi2_res['observed'], 3).tolist())
        print("Expected freqs:", np.round(chi2_res['expected'], 3).tolist())
        print("Bin edges:", np.round(chi2_res['bin_edges'], 3).tolist())
    except Exception as e:
        chi2_res = None
        print("Chi-square test error:", e)

    try:
        W, p_sw = shapiro(x)
        print("\nShapiro-Wilk: W = {:.6f}, p = {:.6f}".format(W, p_sw))
    except Exception as e:
        W, p_sw = None, None
        print("\nShapiro-Wilk error:", e)

    skk = skew_kurtosis_tests(x)
    print("\nSkewness (g1) = {:.6f}, SE = {:.6f}, z = {:.4f}, p = {:.6f}".format(skk["skewness"], skk["SE_skew"], skk["z_skew"], skk["p_skew"]))
    print("Excess kurtosis (g2) = {:.6f}, SE = {:.6f}, z = {:.4f}, p = {:.6f}".format(skk["excess_kurtosis"], skk["SE_kurt"], skk["z_kurt"], skk["p_kurt"]))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    probplot(x, dist="norm", plot=plt)
    plt.title("Q-Q plot: Sleep_Hours vs Normal")
    qq_path = outdir / "sleep_hours_qq.png"
    plt.tight_layout()
    plt.savefig(qq_path)
    if show:
        plt.show()
    plt.close()

    print("\n=== Интерпретация (alpha = {}) ===".format(alpha))
    if chi2_res is not None:
        print("Chi2 p = {:.6f} -> {}".format(chi2_res['pvalue'], "reject H0 (not normal)" if chi2_res['pvalue'] < alpha else "do not reject H0"))
    else:
        print("Chi2: not available")
    if p_sw is not None:
        print("Shapiro p = {:.6f} -> {}".format(p_sw, "reject H0 (not normal)" if p_sw < alpha else "do not reject H0"))
    else:
        print("Shapiro: not available")

    print("Skewness p = {:.6f} -> {}".format(skk["p_skew"], "significant skew" if skk["p_skew"] < alpha else "no significant skew"))
    print("Kurtosis p = {:.6f} -> {}".format(skk["p_kurt"], "significant kurtosis" if skk["p_kurt"] < alpha else "no significant kurtosis"))

    print("\nQ-Q plot saved to:", str(qq_path))

    return {
        "chi2": chi2_res,
        "shapiro": (W, p_sw) if W is not None else None,
        "skew_kurt": skk,
        "plots": {"qq": str(qq_path)}
    }

if __name__ == "__main__":
    try:
        sleep_hours = get_sleep_hours()
        run_normality_checks(sleep_hours, outdir="outputs/2", alpha=0.05, show=False)
    except Exception as e:
        print("Ошибка при выполнении lab1_2:", e)