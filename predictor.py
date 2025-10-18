# predictor.py

import pandas as pd
import numpy as np
import joblib
import os

# === 1. Загрузка модели ===

MODEL_PATH = "models/bid_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
meta = model_data["meta"]
features = meta["features"]
ratios = meta["ratio_percentiles"]  # q25, q50, q75

# === 2. Загрузка данных для предсказания ===

INPUT_PATH = "data/test_data.csv"
df = pd.read_csv(INPUT_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower()
print(f"[INFO] Loaded {len(df)} rows for prediction")

# === 3. Временные признаки (только час и месяц) ===

def process_timestamp(ts):
    try:
        date, time = ts.split(" ")
        day, month, year = map(int, date.split("."))
        hour = int(time.split(":")[0])
        return hour, month
    except:
        return 0, 0

df[['hour','month']] = df['order_timestamp'].apply(lambda x: pd.Series(process_timestamp(str(x))))
df[['hour','month']] = df[['hour','month']].fillna(0)

# === 4. Новые признаки ===

df['pickup_to_trip_ratio'] = df['pickup_in_meters'] / (df['distance_in_meters'] + 1)
df['price_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_ratio'] = df['price_bid_local'] / (df['price_start_local'] + 1e-6)

# === 5. Гарантируем наличие всех признаков ===

for f in features:
    if f not in df.columns:
        df[f] = 0.0
X_base = df[features].fillna(0)

# === 6. Генерация трёх ценовых вариантов () ===
def generate_candidate_bids(base_price):
    q_client = base_price * 1.05
    q_medium = base_price * 1.1
    q_driver = base_price * 1.2

    # округление до ближайших 5 ₽
    return [int(round(x/5)*5) for x in [q_client, q_medium, q_driver]]
# повторяем строки и создаём candidate
df_expanded = df.loc[df.index.repeat(3)].copy()
df_expanded = df_expanded.reset_index(drop=True)
df_expanded['original_index'] = np.repeat(df.index.values, 3)
df_expanded['candidate'] = np.tile(['client','medium','driver'], len(df))

# векторное присвоение цен
candidate_prices = np.vstack(df['price_start_local'].apply(generate_candidate_bids).to_numpy())
df_expanded['price_bid_local'] = candidate_prices.flatten()

# пересчёт price_diff и price_ratio
df_expanded['price_diff'] = df_expanded['price_bid_local'] - df_expanded['price_start_local']
df_expanded['price_ratio'] = df_expanded['price_bid_local'] / (df_expanded['price_start_local'] + 1e-6)

# === 7. Векторное предсказание вероятности ===
X_expanded = df_expanded[features]
df_expanded['probability'] = model.predict_proba(X_expanded)[:,1]

# Минимальный порог цены
min_ratio = 0.95
mask_low = df_expanded['price_bid_local'] < df_expanded['price_start_local'] * min_ratio
df_expanded.loc[mask_low, 'probability'] = -1

# Выбираем лучший вариант по вероятности
df_best = df_expanded.loc[df_expanded.groupby('original_index')['probability'].idxmax()]


# === 8. Объединие результатов с исходным DataFrame ===
df_final = df.copy()
df_final['recommended_bid'] = df_best.set_index('original_index')['price_bid_local']
df_final['probability'] = df_best.set_index('original_index')['probability']

# Добавляем три варианта цен и вероятностей
for c in ['client','medium','driver']:
    grp = df_expanded[df_expanded['candidate']==c].groupby('original_index')
    df_final[f'bid_{c}'] = grp['price_bid_local'].first()
    df_final[f'prob_{c}'] = grp['probability'].first()

# === 9. Сохранение в CSV ===

OUTPUT_PATH = "data/new_orders_with_bid_v2_vectorized.csv"
columns_to_save = list(df.columns[:8]) + [
'recommended_bid','probability',
'bid_client','prob_client',
'bid_medium','prob_medium',
'bid_driver','prob_driver'
]
df_final.to_csv(OUTPUT_PATH, sep=';', index=False, columns=columns_to_save)
print(f"[INFO] Saved prediction results → {OUTPUT_PATH}")
