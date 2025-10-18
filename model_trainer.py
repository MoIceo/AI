# model_trainer.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# === КОНФИГУРАЦИЯ ===
DATA_PATH = "data/dataset.csv"
SAVE_PATH = "models/bid_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.15

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === 1. Загрузка и базовая подготовка данных ===
df = pd.read_csv(DATA_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower()
print(f"[ИНФО] Загружено строк: {len(df)}")

# Мягкая фильтрация аномалий
def remove_anomalies(df):
    cond = (
        (df['price_start_local'] > 0) &
        (df['price_bid_local'] > 0) &
        (df['pickup_in_meters'] >= 0) &
        (df['distance_in_meters'] >= 0) &
        (df['pickup_in_seconds'] >= 0) &
        (df['duration_in_seconds'] >= 0)
    )
    cond &= (df['price_start_local'] < 10000) & (df['price_bid_local'] < 10000)
    cond &= (df['distance_in_meters'] < 100000)
    cond &= (df['price_bid_local'] >= df['price_start_local'] * 0.9)
    cond &= (df['price_bid_local'] <= df['price_start_local'] * 2.5)
    
    # Фильтрация строк, где price_start_local и price_bid_local одинаковые, но is_done = cancel
    same_price_cancel_mask = (
        (df['price_start_local'] == df['price_bid_local']) & 
        (df['is_done'] == 'cancel')
    )
    cond &= ~same_price_cancel_mask
    
    return df[cond]

initial_len = len(df)
df = remove_anomalies(df)
print(f"[ИНФО] Удалено аномальных строк: {initial_len - len(df)}")

# === 2. Временные фичи ===
def process_timestamp(ts):
    try:
        date, time = ts.split(" ")
        day, month, year = map(int, date.split("."))
        hour = int(time.split(":")[0])
        return hour, month
    except Exception:
        return 0, 1

df[['hour', 'month']] = df['order_timestamp'].apply(lambda x: pd.Series(process_timestamp(str(x))))
df['hour'].fillna(0, inplace=True)
df['month'].fillna(1, inplace=True)

# Простые временные фичи
df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))
df['is_peak'] = df['is_peak'].astype(int)

# === 3. Цель ===
df['target'] = (df['is_done'] == 'done').astype(int)
print(f"[ИНФО] Успешность заказов: {df['target'].mean():.3f}")

# === 4. Ключевые инсайты и создание фич ===
print(f"\n=== Ключевые инсайты ===")
df['price_ratio'] = df['price_bid_local'] / df['price_start_local']

# Основной инсайт: успешность резко падает после 1.05
df['price_ratio_bucket'] = pd.cut(df['price_ratio'], 
                                 bins=[0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 2.5],
                                 labels=[0, 1, 2, 3, 4, 5, 6, 7])

# Анализ: смотрим реальную успешность по бинам
print("Фактическая успешность по диапазонам коэффициента:")
for bucket in range(8):
    mask = df['price_ratio_bucket'] == bucket
    if mask.sum() > 0:
        low = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3][bucket]
        high = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 2.5][bucket]
        success = df.loc[mask, 'target'].mean()
        count = mask.sum()
        print(f"  {low:.2f}-{high:.2f}: {success:.3f} ({count} заказов)")

# === 5. Создание сфокусированных фич ===
# Гео-фичи
df['pickup_to_trip_ratio'] = df['pickup_in_meters'] / (df['distance_in_meters'] + 1)
df['total_distance_km'] = (df['pickup_in_meters'] + df['distance_in_meters']) / 1000
df['total_time_min'] = (df['pickup_in_seconds'] + df['duration_in_seconds']) / 60

# Ценовые фичи (фокус на ratio)
df['price_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_ratio'] = df['price_bid_local'] / df['price_start_local']
df['price_diff_ratio'] = df['price_diff'] / df['price_start_local']
df['log_price_ratio'] = np.log(df['price_ratio'])

# Экономические фичи
df['base_price_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000 + 1e-6)
df['is_high_base_price'] = (df['base_price_per_km'] > df['base_price_per_km'].median()).astype(int)

# Взаимодействия
df['distance_price_interaction'] = df['distance_in_meters'] * df['price_ratio']
df['peak_high_price_interaction'] = df['is_peak'] * df['is_high_base_price']

# Ограничиваем аномалии
df['base_price_per_km'] = df['base_price_per_km'].clip(5, 100)

# === 6. Выбор самых важных фич ===
features = [
    # Ключевые ценовые фичи
    'price_ratio', 'price_diff_ratio', 'log_price_ratio', 'price_ratio_bucket',
    
    # Географические
    'pickup_in_meters', 'distance_in_meters', 'pickup_to_trip_ratio', 'total_distance_km',
    
    # Временные
    'hour', 'is_peak',
    
    # Экономические
    'base_price_per_km', 'is_high_base_price',
    
    # Взаимодействия
    'distance_price_interaction', 'peak_high_price_interaction'
]

print(f"\n=== Выбранные фичи ===")
print(f"Используется фич: {len(features)}")

# Убедимся, что все фичи существуют
for f in features:
    if f not in df.columns:
        df[f] = 0.0

X = df[features].fillna(0)
y = df['target']

# === 7. Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[ИНФО] Размер обучающей выборки: {len(X_train)}, размер тестовой выборки: {len(X_test)}")

# === 8. Обучение сфокусированной модели ===
print(f"\n=== Обучение модели ===")

# Используем GradientBoosting с регуляризацией
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,  # Ограничиваем глубину против переобучения
    min_samples_split=200,
    min_samples_leaf=100,
    subsample=0.8,
    random_state=RANDOM_STATE
)

# Обучаем и калибруем
calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
calibrated.fit(X_train, y_train)

# === 9. Оценка модели ===
y_proba = calibrated.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

y_pred = (y_proba >= best_threshold).astype(int)

print("\n=== Оценка на тестовых данных ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
print(f"Лучший порог: {best_threshold:.3f}")
print(classification_report(y_test, y_pred))

# === 10. Анализ и валидация вероятностей ===
print("\n=== Валидация вероятностей ===")

# Тестируем на реалистичных ratio
test_ratios = [0.95, 1.0, 1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.25, 1.3, 1.4, 1.5]
probabilities_by_ratio = []
expected_revenues = []

# Создаем реалистичный шаблон
template_idx = X_test.index[0]
template = X_test.loc[template_idx:template_idx].copy()

for ratio in test_ratios:
    X_temp = template.copy()
    
    # Обновляем все price-related фичи
    X_temp['price_ratio'] = ratio
    X_temp['price_diff_ratio'] = ratio - 1
    X_temp['log_price_ratio'] = np.log(ratio)
    
    # Обновляем bucket
    bucket = np.digitize(ratio, [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]) - 1
    X_temp['price_ratio_bucket'] = bucket
    
    # Обновляем interactions
    X_temp['distance_price_interaction'] = X_temp['distance_in_meters'].iloc[0] * ratio
    
    proba = calibrated.predict_proba(X_temp[features])[:, 1][0]
    probabilities_by_ratio.append(proba)
    expected_revenues.append(ratio * proba)

print("Анализ коэффициентов цены:")
print("Коэф. | Вероятность | Ожидаемый доход")
print("-" * 45)
for ratio, prob, revenue in zip(test_ratios, probabilities_by_ratio, expected_revenues):
    print(f"{ratio:5.2f} | {prob:10.3f} | {revenue:15.3f}")

# Находим оптимальный ratio по ожидаемому доходу
best_revenue_idx = np.argmax(expected_revenues)
best_ratio = test_ratios[best_revenue_idx]
best_prob = probabilities_by_ratio[best_revenue_idx]
best_revenue = expected_revenues[best_revenue_idx]

print(f"\nОптимальный коэффициент: {best_ratio:.2f}x")
print(f"Вероятность: {best_prob:.3f}")
print(f"Ожидаемый доход: {best_revenue:.3f}")

# === 11. Сохранение модели ===
meta = {
    'features': features,
    'probability_threshold': float(best_threshold),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'optimal_ratio': float(best_ratio),
    'price_ratio_mapping': dict(zip([float(x) for x in test_ratios], 
                                   [float(x) for x in probabilities_by_ratio])),
    'expected_revenue_mapping': dict(zip([float(x) for x in test_ratios], 
                                        [float(x) for x in expected_revenues]))
}

joblib.dump({'model': calibrated, 'meta': meta}, SAVE_PATH)
print(f"\n[ИНФО] Модель сохранена в: {SAVE_PATH}")

# === 12. Рекомендации по ценообразованию ===
print(f"\n=== Рекомендации по ценообразованию ===")
print("На основе анализа данных и предсказаний модели:")

# Анализируем лучшие стратегии
low_risk_ratio = 1.05  # Минимальное увеличение с хорошей вероятностью
medium_risk_ratio = 1.1  # Умеренное увеличение
high_risk_ratio = best_ratio  # Оптимальное по доходу

print(f"Низкорисковая стратегия: {low_risk_ratio:.1f}x → вер-ть: {probabilities_by_ratio[test_ratios.index(low_risk_ratio)]:.3f}")
print(f"Среднерисковая стратегия: {medium_risk_ratio:.1f}x → вер-ть: {probabilities_by_ratio[test_ratios.index(medium_risk_ratio)]:.3f}")
print(f"Высокорисковая (оптимальная) стратегия: {high_risk_ratio:.2f}x → вер-ть: {best_prob:.3f}")