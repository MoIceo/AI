# predictor.py

import pandas as pd
import numpy as np
import joblib
import os


# === 1. Загрузка модели ===
MODEL_PATH = "models/bid_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
meta = model_data["meta"]
features = meta["features"]
print(f"[ИНФО] Загружена модель со следующими признаками: {features}")


# === 2. Загрузка данных для предсказания ===
INPUT_PATH = "data/test_data.csv"
df = pd.read_csv(INPUT_PATH, sep=';')
df = df.drop(columns=["price_bid_local"])
df.columns = df.columns.str.strip().str.lower()
print(f"[ИНФО] Загружено {len(df)} строк для предсказания")


# === 3. Временные признаки ===
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


# === 4. Создание базовых признаков ===
df["pickup_to_trip_ratio"] = df["pickup_in_meters"] / (df["distance_in_meters"] + 1)
df['speed_kmh'] = (df['distance_in_meters'] / 1000) / (df['duration_in_seconds'] / 3600 + 1e-6)
df['pickup_speed_kmh'] = (df['pickup_in_meters'] / 1000) / (df['pickup_in_seconds'] / 3600 + 1e-6)
df['total_time_seconds'] = df['pickup_in_seconds'] + df['duration_in_seconds']
df['time_ratio'] = df['pickup_in_seconds'] / (df['total_time_seconds'] + 1e-6)
df['base_price_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000 + 1e-6)

# Ограничение скоростей и цен
df['speed_kmh'] = df['speed_kmh'].clip(1, 120)
df['pickup_speed_kmh'] = df['pickup_speed_kmh'].clip(1, 120)
df['base_price_per_km'] = df['base_price_per_km'].clip(1, 100)


# === 5. Гарантирование наличия всех признаков ===
for f in features:
    if f not in df.columns:
        df[f] = 0.0


# === 6. Улучшенная генерация ценовых вариантов ===
def generate_psychologically_attractive_bids(base_price):
    """Генерация бидов, психологически привлекательных для пассажиров"""
    
    # Базовые множители
    multipliers = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    
    candidate_bids = []
    
    for multiplier in multipliers:
        raw_bid = base_price * multiplier
        
        # Округление до психологически привлекательных цен
        if raw_bid < 100:
            # Для небольших цен округление до 5
            bid = round(raw_bid / 5) * 5
        elif raw_bid < 500:
            # Для средних цен округление до 10 или 25
            if raw_bid % 100 < 50:
                bid = round(raw_bid / 10) * 10  # 190, 200, 210
            else:
                bid = round(raw_bid / 25) * 25  # 175, 200, 225
        else:
            # Для крупных цен округление до 50
            bid = round(raw_bid / 50) * 50
        
        # Гарантирование, что бид не меньше базовой цены
        bid = max(bid, base_price)
        candidate_bids.append(bid)
    
    # Удаление дубликатов и сортировка
    candidate_bids = sorted(list(set(candidate_bids)))
    
    # Выбор 3 лучших вариантов
    if len(candidate_bids) >= 3:
        # Выбор умеренного, среднего и агрессивного вариантов
        return [candidate_bids[1], candidate_bids[len(candidate_bids)//2], candidate_bids[-1]]
    else:
        # Генерация стандартных вариантов при недостатке вариантов
        return [
            int(round(base_price * 1.05 / 10) * 10),   # +5%
            int(round(base_price * 1.15 / 10) * 10),   # +15%
            int(round(base_price * 1.25 / 10) * 10)    # +25%
        ]


def generate_candidate_bids(base_price):
    """Основная функция генерации бидов с психологически привлекательными ценами"""
    
    # Получение психологически привлекательных бидов
    attractive_bids = generate_psychologically_attractive_bids(base_price)
    
    # Сортировка от меньшего к большему и назначение стратегий
    attractive_bids.sort()
    
    strategies = {
        'client': attractive_bids[0],   # Наиболее консервативный
        'medium': attractive_bids[1],   # Средний
        'driver': attractive_bids[2]    # Наиболее агрессивный
    }
    
    return [strategies['client'], strategies['medium'], strategies['driver']]


# Создание расширенного датафрейма
df_expanded = df.loc[df.index.repeat(3)].copy()
df_expanded = df_expanded.reset_index(drop=True)
df_expanded['original_index'] = np.repeat(df.index.values, 3)
df_expanded['candidate'] = np.tile(['client','medium','driver'], len(df))

# Векторное присвоение цен
candidate_prices = np.vstack(df['price_start_local'].apply(generate_candidate_bids).to_numpy())
df_expanded['price_bid_local'] = candidate_prices.flatten()

# Пересчет price-related признаков
df_expanded['price_diff'] = df_expanded['price_bid_local'] - df_expanded['price_start_local']
df_expanded['price_ratio'] = df_expanded['price_bid_local'] / (df_expanded['price_start_local'] + 1e-6)
df_expanded['price_diff_ratio'] = df_expanded['price_diff'] / (df_expanded['price_start_local'] + 1e-6)
df_expanded['log_price_ratio'] = np.log(df_expanded['price_ratio'] + 1e-6)
df_expanded['bid_price_per_km'] = df_expanded['price_bid_local'] / (df_expanded['distance_in_meters'] / 1000 + 1e-6)
df_expanded['bid_price_per_km'] = df_expanded['bid_price_per_km'].clip(1, 100)


# === 7. Векторное предсказание вероятности ===
X_expanded = df_expanded[features].fillna(0)
df_expanded['probability'] = model.predict_proba(X_expanded)[:, 1]

print(f"[ИНФО] Диапазон вероятностей: {df_expanded['probability'].min():.3f} - {df_expanded['probability'].max():.3f}")


# === 8. Улучшенная оптимизация с учетом психологических факторов ===
df_expanded["expected_revenue"] = df_expanded["price_bid_local"] * df_expanded["probability"]

# Бонус за "круглые" цены (например, 200 вместо 190)
def calculate_psychological_bonus(price):
    """Начисление бонуса за психологически привлекательные цены"""
    last_two_digits = price % 100
    
    if last_two_digits == 0:
        return 1.10  # 200, 300, 400 - наиболее привлекательные
    elif last_two_digits == 50:
        return 1.05  # 250, 350 - также привлекательные
    elif price % 10 == 0:
        return 1.02  # 190, 210 - хорошие
    else:
        return 1.00  # Обычные цены

# Применение психологического бонуса
psychological_bonus = df_expanded['price_bid_local'].apply(calculate_psychological_bonus)
df_expanded["psychologically_adjusted_revenue"] = df_expanded["expected_revenue"] * psychological_bonus

# Штраф за очень низкие вероятности
low_prob_penalty = np.where(df_expanded['probability'] < 0.1, 0.3, 1.0)
df_expanded["final_adjusted_revenue"] = df_expanded["psychologically_adjusted_revenue"] * low_prob_penalty

# Для каждого заказа выбирается бид с максимальным скорректированным доходом
df_best = df_expanded.loc[df_expanded.groupby("original_index")["final_adjusted_revenue"].idxmax()]


# === 9. Объединение результатов ===
df_final = df.copy()
df_final['recommended_bid'] = df_best.set_index('original_index')['price_bid_local']
df_final['probability'] = df_best.set_index('original_index')['probability']

# Добавление трех вариантов цен и вероятностей
for c in ['client','medium','driver']:
    grp = df_expanded[df_expanded['candidate']==c].groupby('original_index')
    df_final[f'bid_{c}'] = grp['price_bid_local'].first()
    df_final[f'prob_{c}'] = grp['probability'].first()


# === 10. Анализ психологически привлекательных цен ===
print(f"\n=== Анализ психологически привлекательных цен ===")

# Анализ количества "круглых" цен среди рекомендованных
def classify_price_attractiveness(price):
    last_two_digits = price % 100
    if last_two_digits == 0:
        return "очень привлекательная"  # 200, 300
    elif last_two_digits == 50:
        return "привлекательная"       # 250, 350
    elif price % 10 == 0:
        return "хорошая"             # 190, 210
    else:
        return "обычная"           # 195, 205

df_final['price_attractiveness'] = df_final['recommended_bid'].apply(classify_price_attractiveness)
attractiveness_counts = df_final['price_attractiveness'].value_counts()

print("Привлекательность рекомендованных цен:")
for category, count in attractiveness_counts.items():
    percentage = count / len(df_final) * 100
    print(f"  {category}: {count} заказов ({percentage:.1f}%)")


# === 11. Сохранение результатов ===
OUTPUT_PATH = "data/recommendation.csv"
columns_to_save = [
    'pickup_in_meters', 'pickup_in_seconds', 'distance_in_meters', 'duration_in_seconds',
    'order_timestamp', 'tender_timestamp', 'price_start_local', 'hour',
    'recommended_bid', 'probability',
    'bid_client', 'prob_client', 'bid_medium', 'prob_medium', 'bid_driver', 'prob_driver'
]

available_columns = [col for col in columns_to_save if col in df_final.columns]
df_final[available_columns].to_csv(OUTPUT_PATH, sep=';', index=False)

print(f"[ИНФО] Результаты предсказания сохранены → {OUTPUT_PATH}")
print(f"[ИНФО] Статистика вероятностей:")
print(f"  Биды клиента:  среднее={df_final['prob_client'].mean():.3f} ± {df_final['prob_client'].std():.3f}")
print(f"  Средние биды: среднее={df_final['prob_medium'].mean():.3f} ± {df_final['prob_medium'].std():.3f}")
print(f"  Биды водителя: среднее={df_final['prob_driver'].mean():.3f} ± {df_final['prob_driver'].std():.3f}")
print(f"  Рекомендованные биды:   среднее={df_final['probability'].mean():.3f} ± {df_final['probability'].std():.3f}")

# Анализ распределения рекомендованных бидов
if 'recommended_bid' in df_final.columns and 'price_start_local' in df_final.columns:
    df_final['recommended_ratio'] = df_final['recommended_bid'] / df_final['price_start_local']
    print(f"\nКоэффициенты рекомендованных бидов:")
    print(f"  Мин: {df_final['recommended_ratio'].min():.3f}")
    print(f"  Среднее: {df_final['recommended_ratio'].mean():.3f}")
    print(f"  Макс: {df_final['recommended_ratio'].max():.3f}")
    
    # Анализ популярных цен
    print(f"\nНаиболее частые рекомендованные биды:")
    top_bids = df_final['recommended_bid'].value_counts().head(10)
    for bid, count in top_bids.items():
        percentage = count / len(df_final) * 100
        print(f"  {bid}₽: {count} заказов ({percentage:.1f}%)")