import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, norm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# НАСТРОЙКИ АНАЛИЗА
# =============================================================================
COLUMN_NAME = 'Sleep_Hours'
COLUMN_DESCRIPTION = 'Часы сна'
ALPHA = 0.05  # Уровень значимости

print(" ЛАБОРАТОРНАЯ РАБОТА №1 - АНАЛИЗ ДАННЫХ")
print("=" * 60)
print(f" Анализируемый показатель: {COLUMN_DESCRIPTION}")
print(f" Столбец данных: {COLUMN_NAME}")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('C:/Users/Asus/my_awesome_project/datasets/teen_phone_addiction_dataset.csv')
data = df[COLUMN_NAME].dropna()  # Удаляем пропущенные значения

# =============================================================================
# I. РАСЧЕТ ХАРАКТЕРИСТИК И ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================
print("\n" + "="*50)
print("I. ОСНОВНЫЕ ХАРАКТЕРИСТИКИ И ГРАФИКИ")
print("="*50)

# Базовые статистики
mean_val = np.mean(data) #среднее
variance_val = np.var(data, ddof=1) #дисперсия 
std_val = np.std(data, ddof=1) #cтандартное отклонение -- корень из дисперсии
mode_val = data.mode().iloc[0] if not data.mode().empty else "Нет моды" #наиболее часто встречающаяся
median_val = np.median(data) #элемент находящийся посередине
q25, q50, q75 = np.quantile(data, [0.25, 0.5, 0.75])
skewness_val = skew(data) #cкос -- положительный нормалный или отрицательный
kurtosis_val = kurtosis(data) #эксцесс - много выбросов = положительный, отрицательный - мало выбросов.
iqr_val = q75 - q25 #IQR показывает, насколько "широко" распределены центральные 50% данных.

print(f"Среднее: {mean_val:.4f}")
print(f"Дисперсия: {variance_val:.4f}")
print(f"Стандартное отклонение: {std_val:.4f}")
print(f"Мода: {mode_val}")
print(f"Медиана: {median_val:.4f}")
print(f"Квантили 0.25: {q25:.4f}, 0.5: {q50:.4f}, 0.75: {q75:.4f}")
print(f"Асимметрия: {skewness_val:.4f}")
print(f"Эксцесс: {kurtosis_val:.4f}")
print(f"Интерквартильный размах: {iqr_val:.4f}")

# Построение графиков
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Гистограмма с нормальной кривой #сколько наблюдений попадает в определенынй интервал
n, bins, patches = ax1.hist(data, bins=15, alpha=0.7, color='skyblue', 
                           edgecolor='black', density=True)
x = np.linspace(data.min(), data.max(), 100)
ax1.plot(x, norm.pdf(x, mean_val, std_val), 'r-', linewidth=2, 
        label='Нормальное распределение')
ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
           label=f'Среднее ({mean_val:.2f})')
ax1.axvline(median_val, color='green', linestyle='--', alpha=0.7, 
           label=f'Медиана ({median_val:.2f})')
ax1.set_xlabel(COLUMN_DESCRIPTION)
ax1.set_ylabel('Плотность вероятности')
ax1.set_title(f'Гистограмма: {COLUMN_DESCRIPTION}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Эмпирическая функция распределения, показывает долю наблюдений, которые меньше или равны заданному значению -> получаем график
sorted_data = np.sort(data)
y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, y_vals, linewidth=2, label='Эмпирическая ФР')
ax2.plot(x, norm.cdf(x, mean_val, std_val), 'r--', alpha=0.7, 
        label='Теоретическая ФР')
ax2.set_xlabel(COLUMN_DESCRIPTION)
ax2.set_ylabel('Вероятность')
ax2.set_title(f'Функция распределения: {COLUMN_DESCRIPTION}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# =============================================================================
# II. ПРОВЕРКА НА НОРМАЛЬНОСТЬ
# =============================================================================
print("\n" + "="*50)
print("II. ПРОВЕРКА НА НОРМАЛЬНОСТЬ")
print("="*50)

# 1. Критерий хи-квадрат
def chi_square_normality_test(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    # Создаем интервалы на основе квантилей нормального распределения
    percentiles = np.linspace(0, 100, 7)  # 6 интервалов
    bounds = np.percentile(data, percentiles)
    # Наблюдаемые частоты
    observed, _ = np.histogram(data, bins=bounds)
    expected = [] #согласно норм распределению
    for i in range(len(bounds)-1): 
        prob = (norm.cdf(bounds[i+1], mean, std) - 
                norm.cdf(bounds[i], mean, std))
        expected.append(prob * n) 
    #Вероятность = ФР(верхняя_граница) - ФР(нижняя_граница)
    expected = np.array(expected) #Если статистика меньше критического значения - распределение нормальное
    
    # Избегаем деления на ноль
    valid = expected > 0
    observed_valid = observed[valid]
    expected_valid = expected[valid]
    
    # Статистика хи-квадрат
    chi2_stat = np.sum((observed_valid - expected_valid)**2 / expected_valid) #Для каждого интервала  (O - E)² / E
    df = len(observed_valid) - 3  # степени свободы
    critical_value = stats.chi2.ppf(1 - alpha, df) # возвращение квантиль распределения хи-квадрат
    
    return chi2_stat, critical_value, chi2_stat < critical_value

chi2_stat, critical_value, chi2_normal = chi_square_normality_test(data)
print(f" Критерий хи-квадрат:")
print(f"   Статистика: {chi2_stat:.4f}")
print(f"   Критическое значение: {critical_value:.4f}")
print(f"   Нормальность: {'Да' if chi2_normal else 'Нет'}")

# 2. Критерий асимметрии и эксцесса
def skewness_kurtosis_test(data, alpha=0.05):
    n = len(data)
    skew_val = skew(data) #aссиметрия
    kurt_val = kurtosis(data) #скос
    
    # Стандартные ошибки
    se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    se_kurt = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) 

#Чнасколько значения асимметрии и эксцесса отличаются от 0.

    #Z = (наблюдаемое - ожидаемое) / стандартная_ошибка
    
    # Z-статистики
    z_skew = skew_val / se_skew
    z_kurt = kurt_val / se_kurt
    
    # Критические значения те проверка гипотезы
    z_critical = norm.ppf(1 - alpha/2)
    
    skew_normal = abs(z_skew) < z_critical
    kurt_normal = abs(z_kurt) < z_critical
    
    return (z_skew, z_kurt, z_critical, skew_normal and kurt_normal)

z_skew, z_kurt, z_critical, skew_kurt_normal = skewness_kurtosis_test(data)
print(f" Критерий асимметрии и эксцесса:")
print(f"   Z-асимметрия: {z_skew:.4f} (критическое: ±{z_critical:.4f})")
print(f"   Z-эксцесс: {z_kurt:.4f} (критическое: ±{z_critical:.4f})")
print(f"   Нормальность: {'Да' if skew_kurt_normal else 'Нет'}")

# 3. Q-Q plot
stats.probplot(data, dist="norm", plot=ax3)
ax3.set_title(f'Q-Q Plot: {COLUMN_DESCRIPTION}')
ax3.grid(True, alpha=0.3)

# 4. Box plot для визуализации выбросов
ax4.boxplot(data, vert=True)
ax4.set_title(f'Box Plot: {COLUMN_DESCRIPTION}')
ax4.set_ylabel('Значения')

plt.tight_layout()
plt.show()

# =============================================================================
# III. ПРЕОБРАЗОВАНИЕ ДАННЫХ
# =============================================================================
print("\n" + "="*50)
print("III. ПРЕОБРАЗОВАНИЕ ДАННЫХ")
print("="*50)

# Удаление выбросов по методу IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]

# Логарифмическое преобразование
data_log = np.log(data + 1)  # +1 чтобы избежать log(0), "Сжимает" большие значения

# Стандартизация
data_standardized = (data - mean_val) / std_val #Приводит данные к среднему = 0 и std = 1

print(f" Исходные данные: {len(data)} записей")
print(f" Без выбросов: {len(data_no_outliers)} записей")
print(f" Удалено выбросов: {len(data) - len(data_no_outliers)}")

# Анализ преобразованных данных - визуализация
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10)) 

# Исходные данные
ax1.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
ax1.set_title('Исходные данные')
ax1.grid(True, alpha=0.3)

# Данные без выбросов
ax2.hist(data_no_outliers, bins=15, alpha=0.7, color='lightgreen', 
         edgecolor='black', density=True)
ax2.set_title('Без выбросов (IQR метод)')
ax2.grid(True, alpha=0.3)

# Логарифмированные данные
ax3.hist(data_log, bins=15, alpha=0.7, color='lightcoral', 
         edgecolor='black', density=True)
ax3.set_title('Логарифмированные данные')
ax3.grid(True, alpha=0.3)

# Стандартизированные данные
ax4.hist(data_standardized, bins=15, alpha=0.7, color='gold', 
         edgecolor='black', density=True)
ax4.set_title('Стандартизированные данные')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Проверка нормальности преобразованных данных
_, _, normal_no_outliers = chi_square_normality_test(data_no_outliers)
_, _, normal_log = chi_square_normality_test(data_log)
_, _, normal_std = chi_square_normality_test(data_standardized)

print(f" Нормальность преобразованных данных:")
print(f"   Исходные: {'Нет' if not chi2_normal else 'Да'}")
print(f"   Без выбросов: {'Нет' if not normal_no_outliers else 'Да'}")
print(f"   Логарифмированные: {'Нет' if not normal_log else 'Да'}")
print(f"   Стандартизированные: {'Нет' if not normal_std else 'Да'}")
# Проверка наличия столбца
print(f"Столбцы в данных: {df.columns.tolist()}")
if 'School_Grade' in df.columns:
    print(f"Уникальные классы: {df['School_Grade'].unique()}")
else:
    print("Столбец 'School_Grade' не найден, используем другой столбец для группировки")
    # Можно использовать другой столбец, например:
    # if 'Grade' in df.columns: ...
# =============================================================================
# IV. ГРУППИРОВКА ПО SCHOOL_GRADE
# =============================================================================
print("\n" + "="*50)
print("IV. АНАЛИЗ ПО ГРУППАМ (School_Grade)")
print("="*50)

if 'School_Grade' in df.columns: #по классам делим
    grades = df['School_Grade'].unique()
    grades.sort()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    group_stats = []
    for grade in grades:
        grade_data = df[df['School_Grade'] == grade][COLUMN_NAME].dropna()
        if len(grade_data) > 0:
            group_stats.append({
                'grade': grade,
                'mean': np.mean(grade_data),
                'variance': np.var(grade_data, ddof=1),
                'count': len(grade_data)
                #инфа для каждой из групп
            })
            
            # Гистограмма для каждой группы
            ax.hist(grade_data, bins=10, alpha=0.6, 
                   label=f'Класс {grade} (n={len(grade_data)})', 
                   density=True)
    
    ax.set_xlabel(COLUMN_DESCRIPTION)
    ax.set_ylabel('Плотность вероятности')
    ax.set_title(f'Распределение {COLUMN_DESCRIPTION} по классам')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Статистики по группам
    print(f" Статистики по классам:")
    for stat in group_stats:
        print(f"   Класс {stat['grade']}: среднее = {stat['mean']:.3f}, "
              f"дисперсия = {stat['variance']:.3f}, n = {stat['count']}")
else:
    print("Столбец 'School_Grade' не найден в данных")

# =============================================================================
# V. ВЫВОДЫ
# =============================================================================
print("\n" + "="*50)
print("V. ОБЩИЕ ВЫВОДЫ")
print("="*50)

print("1.  ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
print(f"   - Распределение {COLUMN_DESCRIPTION.lower()} имеет среднее {mean_val:.2f} и медиану {median_val:.2f}")
print(f"   - Асимметрия ({skewness_val:.2f}) указывает на {'симметричное' if abs(skewness_val) < 0.5 else 'скошенное'} распределение")
print(f"   - Эксцесс ({kurtosis_val:.2f}) показывает {'нормальную' if abs(kurtosis_val) < 0.5 else 'отличную от нормальной'} остроту пика")

print("\n2. ПРОВЕРКА НА НОРМАЛЬНОСТЬ:")
print(f"   - Критерий хи-квадрат: {'нормальное' if chi2_normal else 'не нормальное'}")
print(f"   - Критерий асимметрии и эксцесса: {'нормальное' if skew_kurt_normal else 'не нормальное'}")
print(f"   - Общий вывод: распределение {'является нормальным' if chi2_normal and skew_kurt_normal else 'НЕ является нормальным'}")

print("\n3. ЭФФЕКТ ОТ ПРЕОБРАЗОВАНИЙ:")
improvement = any([normal_no_outliers, normal_log, normal_std])
print(f"   - Преобразования {'улучшили' if improvement else 'не улучшили'} нормальность распределения")
if normal_no_outliers:
    print("   - Наилучший результат: удаление выбросов")
elif normal_log:
    print("   - Наилучший результат: логарифмирование")
elif normal_std:
    print("   - Наилучший результат: стандартизация")
else:
    print("   - Ни одно преобразование не привело к нормальности")

if 'School_Grade' in df.columns and len(group_stats) > 0:
    print("\n4.  РАЗЛИЧИЯ МЕЖДУ ГРУППАМИ:")
    means = [stat['mean'] for stat in group_stats]
    if max(means) - min(means) > 0.5:
        print("   - Обнаружены существенные различия между классами")
    else:
        print("   - Различия между классами незначительны")

print("\n" + "="*60)
print("="*60)