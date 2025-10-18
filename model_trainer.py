# model_trainer.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE

# === CONFIG ===
DATA_PATH = "data/dataset.csv"
SAVE_PATH = "models/bid_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.15

# === Создаём файл модели, если отсутствует ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === 1. Загрузка данных для обучения ===
df = pd.read_csv(DATA_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower()
print(f"[INFO] Loaded {len(df)} rows")

# === 2. Фильтрация данных ===
initial_len = len(df)
df = df[~((df['price_start_local'] == df['price_bid_local']) & (df['is_done'] == 'cancel'))]
print(f"[INFO] Removed {initial_len - len(df)} rows where start==bid and is_done==cancel")

# === 2.1 Очистка аномалий ===
def remove_anomalies(df):
    cond = (
        (df['price_start_local'] > 0) &
        (df['price_bid_local'] > 0) &
        (df['pickup_in_meters'] >= 0) &
        (df['distance_in_meters'] >= 0) &
        (df['pickup_in_seconds'] >= 0) &
        (df['duration_in_seconds'] >= 0)
    )
    cond &= (df['price_start_local'] < 5000) & (df['price_bid_local'] < 5000)
    cond &= (df['distance_in_meters'] < 50000)
    return df[cond]

before = len(df)
df = remove_anomalies(df)
print(f"[INFO] Removed {before - len(df)} anomaly rows")

# === 3. Временные features ===
def process_timestamp(ts):
    try:
        date, time = ts.split(" ")
        day, month, year = map(int, date.split("."))
        hour = int(time.split(":")[0])
        return hour, month
    except Exception:
        return np.nan, np.nan

df[['hour', 'month']] = df['order_timestamp'].apply(lambda x: pd.Series(process_timestamp(str(x))))
df['hour'].fillna(0, inplace=True)
df['month'].fillna(df['month'].mode()[0], inplace=True)

# === 4. Новые features ===
df['pickup_to_trip_ratio'] = df['pickup_in_meters'] / (df['distance_in_meters'] + 1)
df['price_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_ratio'] = df['price_bid_local'] / (df['price_start_local'] + 1e-6)
df['log_price_ratio'] = np.log1p(df['price_ratio'] - 1)

# === 5. Цель ===
df['target'] = (df['is_done'] == 'done').astype(int)

# === 6. Список Feature ===
features = [
    'pickup_in_meters', 'pickup_in_seconds',
    'distance_in_meters', 'duration_in_seconds',
    'price_start_local', 'price_bid_local',
    'hour', 'month',
    'price_diff', 'price_ratio', 'pickup_to_trip_ratio',
    'log_price_ratio'
]

for f in features:
    if f not in df.columns:
        df[f] = 0.0

X = df[features].fillna(0)
y = df['target']

# === 7. Обучение ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

# === 7.1 Применение SMOTE к обучению ===
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto', k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"[INFO] Applied SMOTE: {len(X_train_res)} samples after balancing (from {len(X_train)})")
print(f"[INFO] Class balance after SMOTE: {np.bincount(y_train_res)}")

# === 8. Постройка конвеера ===
base_clf = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
calibrated = CalibratedClassifierCV(estimator=base_clf, cv=3, method='isotonic')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', calibrated)
])

# === 9. Поиск гиперпараметров ===
param_grid = {
    'clf__estimator__C': [0.01, 0.1, 1.0, 5.0, 10.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"[INFO] Best params: {grid.best_params_}")
best_model = grid.best_estimator_

# === 10. Оценка на тестовом наборе ===
y_proba = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print(f"[INFO] Best threshold for F1: {best_threshold:.3f}")

y_pred = (y_proba >= best_threshold).astype(int)

print("\n=== Test evaluation ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# === 11. Перцентиль соотношений успешных заявок ===
successful = df[df['target'] == 1].copy()
successful['ratio'] = successful['price_bid_local'] / (successful['price_start_local'] + 1e-6)
q25, q50, q75 = successful['ratio'].quantile([0.25, 0.5, 0.75]).values
print(f"[INFO] Price ratio percentiles (q25,q50,q75): {q25:.3f}, {q50:.3f}, {q75:.3f}")

meta = {
    'features': features,
    'ratio_percentiles': {'q25': float(q25), 'q50': float(q50), 'q75': float(q75)}
}

# === 12. Сохранение ===
joblib.dump({'model': best_model, 'meta': meta}, SAVE_PATH)
print(f"[INFO] Saved model + meta to {SAVE_PATH}")