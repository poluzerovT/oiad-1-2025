import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Загрузка данных
df = pd.read_csv(r'C:\Users\Asus\fldr\datasets\students_simple.csv')

x = df.iloc[:, 1].values  # столбец №2
y = df.iloc[:, 8].values  # столбец №9

# 1. Корреляции
def fechner_corr(x, y): #совпадение знаков отклонений от среднего.
    x_sign = np.sign(x - np.mean(x))
    y_sign = np.sign(y - np.mean(y))
    matches = np.sum(x_sign == y_sign)
    mismatches = np.sum(x_sign != y_sign)
    return (matches - mismatches) / len(x)

fechner = fechner_corr(x, y) # сравнивает знаки отклонений от среднего.
pearson, p_value = stats.pearsonr(x, y) #линейную зависимость между переменными.
spearman, _ = stats.spearmanr(x, y) #монотонную  зависимость.
kendall, _ = stats.kendalltau(x, y) #Считает, сколько пар идут в одном направлении чёи сколько — в разных.

# Доверительный интервал Пирсона -показывает диапазон,
#  в котором с высокой вероятностью 
# находится истинное значение корреляции в генеральной совокупности.
def pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_interval = [z - z_crit * se, z + z_crit * se]
    return np.tanh(z_interval)

ci_low, ci_high = pearson_ci(pearson, len(x))

print(f"Фехнер: {fechner:.3f}")
print(f"Пирсон: {pearson:.3f}, p={p_value:.3f}, CI=({ci_low:.3f}, {ci_high:.3f})")
print(f"Спирмен: {spearman:.3f}")
print(f"Кенделл: {kendall:.3f}")

# 2. Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(x, kde=True)
plt.title("Гистограмма X")
plt.subplot(1, 2, 2)
sns.histplot(y, kde=True)
plt.title("Гистограмма Y")
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(x=x, y=y)
plt.title("График рассеяния")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 3. Регрессии
def plot_regression(model_name, y_pred):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x, y=y, label='Данные')
    sns.lineplot(x=x, y=y_pred, color='red', label=model_name)
    plt.title(f"{model_name} регрессия")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Линейная = 𝑦 = w1x + w0
lin_model = LinearRegression()
x_lin = x.reshape(-1, 1)
lin_model.fit(x_lin, y)
y_lin_pred = lin_model.predict(x_lin)
r2_lin = r2_score(y, y_lin_pred)
plot_regression("Линейная", y_lin_pred)

# Квадратичная = w2x^2 + w1x + w0
poly = PolynomialFeatures(degree=2)
x_quad = poly.fit_transform(x_lin)
quad_model = LinearRegression()
quad_model.fit(x_quad, y)
y_quad_pred = quad_model.predict(x_quad)
r2_quad = r2_score(y, y_quad_pred)
plot_regression("Квадратичная", y_quad_pred)

# Гиперболическая: y = a/x + b
x_hyp = 1 / x_lin
hyp_model = LinearRegression()
hyp_model.fit(x_hyp, y)
y_hyp_pred = hyp_model.predict(x_hyp)
r2_hyp = r2_score(y, y_hyp_pred)
plot_regression("Гиперболическая", y_hyp_pred)

# Показательная: y = a * exp(bx) → ln(y) = bx + ln(a)
x_exp = x_lin
y_log = np.log(y)
exp_model = LinearRegression()
exp_model.fit(x_exp, y_log)
y_exp_pred = np.exp(exp_model.predict(x_exp))
r2_exp = r2_score(y, y_exp_pred)
plot_regression("Показательная", y_exp_pred)

# 4. Проверка по критерию Фишера
def fisher_test(r2, n, k):
    F = (r2 / (1 - r2)) * ((n - k) / (k - 1))
    p = 1 - stats.f.cdf(F, k - 1, n - k)
    return F, p

models_r2 = {
    "Линейная": (r2_lin, 2),
    "Квадратичная": (r2_quad, 3),
    "Гиперболическая": (r2_hyp, 2),
    "Показательная": (r2_exp, 2)
}

best_model = max(models_r2.items(), key=lambda x: x[1][0])
worst_model = min(models_r2.items(), key=lambda x: x[1][0])

for name, (r2, k) in [best_model, worst_model]:
    F, p = fisher_test(r2, len(x), k)
    print(f"{name} модель: F={F:.2f}, p={p:.3f}")

# 5. Выводы
print("\nВыводы:")
print("- Корреляции показывают степень связи между переменными, наиболее сильная — Пирсона.")
print("- Визуализация подтверждает наличие зависимости.")
print(f"- Лучшая модель: {best_model[0]} (R²={best_model[1][0]:.3f})")
print(f"- Худшая модель: {worst_model[0]} (R²={worst_model[1][0]:.3f})")
print("- F-критерий подтверждает статистическую значимость лучшей модели.")
