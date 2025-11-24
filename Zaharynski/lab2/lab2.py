import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../datasets/students_simple.csv")
x = df.iloc[:, 2].astype(float).values
y = df.iloc[:, 8].astype(float).values
mask = ~np.isnan(x) & ~np.isnan(y)
x, y = x[mask], y[mask]
n = len(y)
print(f"n = {n}")

print("="*70)
print("АНАЛИЗ ЗАВИСИМОСТИ: FRIENDS от INCOME")
print("="*70)
print(f"X (income): {x}")
print(f"Y (friends): {y}\n")

print("\n" + "="*70)
print("1. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("="*70)

pearson_r, pearson_p = pearsonr(x, y)
print(f"\nКорреляция Пирсона:")
print(f"  r = {pearson_r:.4f}")
print(f"  p-value = {pearson_p:.4f}")

n = len(x)
z = np.arctanh(pearson_r)
se = 1/np.sqrt(n-3)
z_crit = 1.96
ci_lower = np.tanh(z - z_crit*se)
ci_upper = np.tanh(z + z_crit*se)
print(f"  Доверительный интервал (95%): [{ci_lower:.4f}, {ci_upper:.4f}]")

# Корреляция Спирмена
spearman_r, spearman_p = spearmanr(x, y)
print(f"\nКорреляция Спирмена:")
print(f"  ρ = {spearman_r:.4f}")
print(f"  p-value = {spearman_p:.4f}")

# Корреляция Кенделла
kendall_r, kendall_p = kendalltau(x, y)
print(f"\nКорреляция Кенделла (Тау):")
print(f"  τ = {kendall_r:.4f}")
print(f"  p-value = {kendall_p:.4f}")

# ===== 2. ВИЗУАЛИЗАЦИЯ =====
print("\n" + "="*70)
print("2. ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма
axes[0].hist(x, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Income', fontsize=11)
axes[0].set_ylabel('Частота', fontsize=11)
axes[0].set_title('Гистограмма распределения Income', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# График рассеяния
axes[1].scatter(x, y, s=100, alpha=0.6, color='coral', edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Income', fontsize=11)
axes[1].set_ylabel('Friends', fontsize=11)
axes[1].set_title('График рассеяния (Scatter Plot)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualization.png', dpi=300, bbox_inches='tight')
plt.show()
print("Графики сохранены в visualization.png")

print("\n" + "="*70)
print("3. УРАВНЕНИЯ РЕГРЕССИИ")
print("="*70)

models = {}

w1_lin, w0_lin = np.polyfit(x, y, 1)
y_pred_lin = w1_lin * x + w0_lin
r2_lin = r2_score(y, y_pred_lin)
models['Линейная'] = {'params': (w1_lin, w0_lin), 'pred': y_pred_lin, 'r2': r2_lin}
print(f"\nЛинейная: y = {w1_lin:.6f}*x + {w0_lin:.6f}")
print(f"  R² = {r2_lin:.4f}")

coeffs_quad = np.polyfit(x, y, 2)
y_pred_quad = np.polyval(coeffs_quad, x)
r2_quad = r2_score(y, y_pred_quad)
models['Квадратичная'] = {'params': coeffs_quad, 'pred': y_pred_quad, 'r2': r2_quad}
print(f"\nКвадратичная: y = {coeffs_quad[0]:.6f}*x² + {coeffs_quad[1]:.6f}*x + {coeffs_quad[2]:.6f}")
print(f"  R² = {r2_quad:.4f}")

X_hyp = 1 / x
w1_hyp, w0_hyp = np.polyfit(X_hyp, y, 1)
y_pred_hyp = w1_hyp / x + w0_hyp
r2_hyp = r2_score(y, y_pred_hyp)
models['Гиперболическая'] = {'params': (w1_hyp, w0_hyp), 'pred': y_pred_hyp, 'r2': r2_hyp}
print(f"\nГиперболическая: y = {w1_hyp:.6f}/x + {w0_hyp:.6f}")
print(f"  R² = {r2_hyp:.4f}")

X_exp = x
Y_exp = np.log(y)
b_exp, a_exp = np.polyfit(X_exp, Y_exp, 1)
w1_exp = np.exp(b_exp)
w0_exp = np.exp(a_exp)
y_pred_exp = (w1_exp ** x) * w0_exp
r2_exp = r2_score(y, y_pred_exp)
models['Показательная'] = {'params': (w1_exp, w0_exp), 'pred': y_pred_exp, 'r2': r2_exp}
print(f"\nПоказательная: y = {w1_exp:.6f}^x * {w0_exp:.6f}")
print(f"  R² = {r2_exp:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_smooth = np.linspace(x.min(), x.max(), 300)

ax = axes[0, 0]
ax.scatter(x, y, s=80, alpha=0.6, color='coral', edgecolor='black')
y_smooth_lin = w1_lin * x_smooth + w0_lin
ax.plot(x_smooth, y_smooth_lin, 'b-', linewidth=2.5, label='Линия регрессии')
ax.set_title(f'Линейная регрессия (R² = {r2_lin:.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Income')
ax.set_ylabel('Friends')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(x, y, s=80, alpha=0.6, color='coral', edgecolor='black')
y_smooth_quad = np.polyval(coeffs_quad, x_smooth)
ax.plot(x_smooth, y_smooth_quad, 'g-', linewidth=2.5, label='Кривая регрессии')
ax.set_title(f'Квадратичная регрессия (R² = {r2_quad:.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Income')
ax.set_ylabel('Friends')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(x, y, s=80, alpha=0.6, color='coral', edgecolor='black')
y_smooth_hyp = w1_hyp / x_smooth + w0_hyp
ax.plot(x_smooth, y_smooth_hyp, 'r-', linewidth=2.5, label='Кривая регрессии')
ax.set_title(f'Гиперболическая регрессия (R² = {r2_hyp:.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Income')
ax.set_ylabel('Friends')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(x, y, s=80, alpha=0.6, color='coral', edgecolor='black')
y_smooth_exp = (w1_exp ** x_smooth) * w0_exp
ax.plot(x_smooth, y_smooth_exp, 'm-', linewidth=2.5, label='Кривая регрессии')
ax.set_title(f'Показательная регрессия (R² = {r2_exp:.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Income')
ax.set_ylabel('Friends')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regressions.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nГрафики регрессий сохранены в regressions.png")

print("\n" + "="*70)
print("4. ПРОВЕРКА УРАВНЕНИЙ РЕГРЕССИИ (КРИТЕРИЙ ФИШЕРА)")
print("="*70)

best_model = max(models.items(), key=lambda x: x[1]['r2'])
worst_model = min(models.items(), key=lambda x: x[1]['r2'])

print(f"\nЛучшая модель: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
print(f"Худшая модель: {worst_model[0]} (R² = {worst_model[1]['r2']:.4f})")

def fisher_test(y_true, y_pred, k):
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    mse = ss_res / (n - k)
    ms_tot = ss_tot / (n - 1)
    F = ms_tot / mse if mse != 0 else np.inf
    p_value = 1 - stats.f.cdf(F, n-k, n-1)
    return F, p_value

print(f"\n{best_model[0]}:")
F_best, p_best = fisher_test(y, best_model[1]['pred'], k=2 if best_model[0]=='Линейная' else 3 if best_model[0]=='Гиперболическая' else 4)
print(f"  F-статистика = {F_best:.4f}")
print(f"  p-value = {p_best:.4f}")
print(f"  Модель {'ЗНАЧИМА' if p_best < 0.05 else 'НЕ значима'} (α = 0.05)")

print(f"\n{worst_model[0]}:")
F_worst, p_worst = fisher_test(y, worst_model[1]['pred'], k=2 if worst_model[0]=='Линейная' else 3 if worst_model[0]=='Гиперболическая' else 4)
print(f"  F-статистика = {F_worst:.4f}")
print(f"  p-value = {p_worst:.4f}")
print(f"  Модель {'ЗНАЧИМА' if p_worst < 0.05 else 'НЕ значима'} (α = 0.05)")
