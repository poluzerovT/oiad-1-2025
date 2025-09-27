import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from lab1_1 import load_df

def group_and_analyze_sleep(df: pd.DataFrame):
    """
    Группирует данные по 'School_Grade' и вычисляет статистики.
    """
    df['Sleep_Hours'] = pd.to_numeric(df['Sleep_Hours'], errors='coerce')
    df.dropna(subset=['Sleep_Hours', 'School_Grade'], inplace=True)
    grouped_stats = df.groupby('School_Grade')['Sleep_Hours'].agg(['mean', 'var', 'count'])
    print("\n=== Статистики Sleep_Hours по группам School_Grade ===")
    print(grouped_stats)
    return grouped_stats, df

def plot_histograms_by_group(df: pd.DataFrame, stats: pd.DataFrame, outdir: str = "outputs"):
    """
    Строит гистограммы для каждой группы на одном графике.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    grades = df['School_Grade'].unique()
    sns.set(style="whitegrid")
    for grade in sorted(grades):
        subset = df[df['School_Grade'] == grade]
        sns.histplot(subset['Sleep_Hours'], kde=False, label=f'{grade} (n={len(subset)})',
                     alpha=0.5, bins=15, edgecolor='black', zorder=2)
    for grade, row in stats.iterrows():
        plt.axvline(x=row['mean'], color='k', linestyle='--', linewidth=1, label=f'Avg {grade}: {row["mean"]:.2f}')
    plt.title("Гистограммы распределения Sleep_Hours по группам School_Grade")
    plt.xlabel("Часы сна")
    plt.ylabel("Частота")
    plt.legend(title="School Grade")
    plt.tight_layout()
    plot_path = outdir / "sleep_hours_hist_by_grade.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nГистограммы сохранены в {plot_path}")

if __name__ == "__main__":
    try:
        df = load_df()
        grouped_stats, clean_df = group_and_analyze_sleep(df)
        plot_histograms_by_group(clean_df, grouped_stats)
    except Exception as e:
        print(f"Ошибка при выполнении lab1_4: {e}")