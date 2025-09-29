from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_df(path: str = '../../datasets/teen_phone_addiction_dataset.csv') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {p.resolve()}")
    return pd.read_csv(p)

def get_sleep_hours(df: pd.DataFrame = None, path: str = None) -> pd.Series:
    """
    Возвращает pd.Series с числовыми значениями Sleep_Hours (без NaN).
    Можно передать либо df, либо path к CSV.
    """
    if df is None:
        if path is None:
            path = '../../datasets/teen_phone_addiction_dataset.csv'
        df = load_df(path)
    col_candidates = [c for c in df.columns if 'sleep' in c.lower().strip()]
    if not col_candidates:
        raise KeyError(f"Не найден столбец с 'sleep'. Доступные столбцы: {list(df.columns)}")
    col = col_candidates[0]
    series = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
    return series

def descriptive_stats(sleep_hours: pd.Series) -> dict:
    """
    Возвращает словарь со статистиками и печатает их.
    """
    x = sleep_hours.dropna().astype(float)
    if x.empty:
        raise ValueError("sleep_hours пуст после удаления NaN.")
    q25, q50, q75 = x.quantile([0.25, 0.5, 0.75])
    modes = x.mode().tolist()
    stats = {
        "count": int(x.count()),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "variance_ddof1": float(x.var()),
        "mode": modes if modes else None,
        "median": float(x.median()),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "IQR": float(q75 - q25)
    }
    try:
        from scipy.stats import skew, kurtosis
        stats["skewness"] = float(skew(x, bias=False))
        stats["excess_kurtosis"] = float(kurtosis(x, fisher=True, bias=False))
    except ImportError:
        stats["skewness"] = None
        stats["excess_kurtosis"] = None

    print("=== Descriptive statistics (Sleep_Hours) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    return stats

def plot_hist_cdf(sleep_hours: pd.Series, outdir: str = "outputs", show: bool = False) -> dict:
    """
    Строит и сохраняет гистограмму (с нормальной плотностью) и CDF.
    Q-Q plot перемещен в lab1_2 для пункта II.
    """
    from scipy.stats import norm
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = sleep_hours.dropna().astype(float)


    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=20, density=True, edgecolor='black', alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 300)
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), linewidth=2)
    plt.title("Гистограмма Sleep_Hours + нормальная плотность")
    plt.xlabel("Часы сна")
    plt.ylabel("Плотность")
    hist_path = outdir / "sleep_hours_hist_with_normal.png"
    plt.tight_layout()
    plt.savefig(hist_path)
    if show:
        plt.show()
    plt.close()

    sorted_data = np.sort(x)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(6, 4))
    plt.step(sorted_data, cdf, where='post')
    plt.title("Эмпирическая функция распределения (CDF)")
    plt.xlabel("Часы сна")
    plt.ylabel("F(x)")
    cdf_path = outdir / "sleep_hours_cdf.png"
    plt.tight_layout()
    plt.savefig(cdf_path)
    if show:
        plt.show()
    plt.close()

    paths = {"hist": str(hist_path.resolve()), "cdf": str(cdf_path.resolve())}
    print("Saved plots to:", paths)
    return paths

if __name__ == "__main__":
    try:
        print("Запущен lab1_1.py как скрипт.")
        sh = get_sleep_hours()
        descriptive_stats(sh)
        plot_hist_cdf(sh, outdir="lab1/outputs/1", show=False)
    except Exception as e:
        print("Ошибка при выполнении lab1_1:", e)