# predictor.py

import pandas as pd
import numpy as np
import joblib
import os
import json

# === 1. Загрузка модели ===
MODEL_PATH = "models/bid_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
meta = model_data["meta"]
features = meta["features"]
optimal_ratio = meta.get('optimal_ratio', 1.12)  # Берем оптимальное соотношение из метаданных

print(f"[ИНФО] Загружена модель со следующими признаками: {features}")
print(f"[ИНФО] Оптимальное соотношение цены: {optimal_ratio:.2f}x")


# === 2. Загрузка данных для предсказания ===
INPUT_PATH = "data/data.csv"
df = pd.read_csv(INPUT_PATH, sep=';')
#df = df.drop(columns=["price_bid_local"])
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
        return 0, 1  # Исправлено: месяц по умолчанию 1 вместо 0

df[['hour','month']] = df['order_timestamp'].apply(lambda x: pd.Series(process_timestamp(str(x))))
df[['hour','month']] = df[['hour','month']].fillna(0)


# === 4. Создание базовых признаков ===
df["pickup_to_trip_ratio"] = df["pickup_in_meters"] / (df["distance_in_meters"] + 1)
df['total_distance_km'] = (df['pickup_in_meters'] + df['distance_in_meters']) / 1000
df['total_time_min'] = (df['pickup_in_seconds'] + df['duration_in_seconds']) / 60

# Пиковые часы
df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))
df['is_peak'] = df['is_peak'].astype(int)

# Экономические признаки
df['base_price_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000 + 1e-6)
df['is_high_base_price'] = (df['base_price_per_km'] > 20).astype(int)  # Эмпирический порог

# Ограничение аномалий
df['base_price_per_km'] = df['base_price_per_km'].clip(5, 100)


# === 5. Гарантирование наличия всех признаков ===
for f in features:
    if f not in df.columns:
        # Создаем отсутствующие признаки
        if f == 'price_diff_ratio':
            df[f] = 0.0
        elif f == 'log_price_ratio':
            df[f] = 0.0
        elif f == 'price_ratio_category':
            df[f] = 0
        elif f == 'distance_price_interaction':
            df[f] = df['distance_in_meters'] * 1.0  # Базовое значение
        else:
            df[f] = 0.0
print(f"[ИНФО] Все необходимые признаки созданы")


# === 6. Генерация стратегий ценообразования ===
def generate_strategy_bids(base_price, optimal_ratio=1.12):
    """Генерация бидов на основе стратегий с учетом оптимального соотношения"""
    
    # Стратегии основаны на анализе модели
    strategies = {
        'conservative': 1.05,    # Консервативная +5%
        'optimal': optimal_ratio, # Оптимальная (из модели)
        'aggressive': 1.20       # Агрессивная +20%
    }
    
    bids = {}
    for strategy, multiplier in strategies.items():
        raw_bid = base_price * multiplier
        
        # Психологически привлекательное округление
        if raw_bid < 100:
            bid = round(raw_bid / 5) * 5
        elif raw_bid < 500:
            if raw_bid % 100 < 50:
                bid = round(raw_bid / 10) * 10  # 190, 200, 210
            else:
                bid = round(raw_bid / 25) * 25  # 175, 200, 225
        else:
            bid = round(raw_bid / 50) * 50
        
        bids[strategy] = max(int(bid), base_price)  # Гарантия минимальной цены
    
    return [bids['conservative'], bids['optimal'], bids['aggressive']]


# Создание расширенного датафрейма
df_expanded = df.loc[df.index.repeat(3)].copy()
df_expanded = df_expanded.reset_index(drop=True)
df_expanded['original_index'] = np.repeat(df.index.values, 3)
df_expanded['strategy'] = np.tile(['conservative', 'optimal', 'aggressive'], len(df))

# Векторное присвоение цен
candidate_prices = np.vstack(df['price_start_local'].apply(
    lambda x: generate_strategy_bids(x, optimal_ratio)
).to_numpy())
df_expanded['price_bid_local'] = candidate_prices.flatten()


# === 7. Подготовка признаков для предсказания ===
# Обновление price-related признаков
df_expanded['price_ratio'] = df_expanded['price_bid_local'] / df_expanded['price_start_local']
df_expanded['price_diff_ratio'] = df_expanded['price_ratio'] - 1
df_expanded['log_price_ratio'] = np.log(df_expanded['price_ratio'].clip(0.1, 10))

# Категоризация price_ratio (используем bins из модели или стандартные)
ratio_bins = meta.get('ratio_bins', [1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0])
df_expanded['price_ratio_category'] = np.digitize(df_expanded['price_ratio'], ratio_bins) - 1
df_expanded['price_ratio_category'] = df_expanded['price_ratio_category'].clip(0, len(ratio_bins)-2)

# Обновление interactions
df_expanded['distance_price_interaction'] = df_expanded['distance_in_meters'] * df_expanded['price_ratio']

# Гарантия наличия всех признаков в расширенном датафрейме
for f in features:
    if f not in df_expanded.columns:
        df_expanded[f] = 0.0


# === 8. Векторное предсказание вероятности ===
X_expanded = df_expanded[features].fillna(0)
df_expanded['probability'] = model.predict_proba(X_expanded)[:, 1]

print(f"[ИНФО] Диапазон вероятностей: {df_expanded['probability'].min():.3f} - {df_expanded['probability'].max():.3f}")


# === 9. Оптимизация выбора бида ===
df_expanded["expected_revenue"] = df_expanded["price_bid_local"] * df_expanded["probability"]

# Бонус за психологически привлекательные цены
def calculate_psychological_bonus(price):
    last_two_digits = price % 100
    if last_two_digits == 0:
        return 1.10  # 200, 300, 400
    elif last_two_digits == 50:
        return 1.05  # 250, 350
    elif price % 10 == 0:
        return 1.02  # 190, 210
    else:
        return 1.00

psychological_bonus = df_expanded['price_bid_local'].apply(calculate_psychological_bonus)
df_expanded["adjusted_revenue"] = df_expanded["expected_revenue"] * psychological_bonus

# Выбор лучшего бида для каждого заказа
df_best = df_expanded.loc[df_expanded.groupby("original_index")["adjusted_revenue"].idxmax()]


# === 10. Формирование результатов ===
df_final = df.copy()

# Рекомендованный бид
df_final['recommended_bid'] = df_best.set_index('original_index')['price_bid_local']
df_final['probability'] = df_best.set_index('original_index')['probability']
df_final['recommended_strategy'] = df_best.set_index('original_index')['strategy']

# Все стратегии с вероятностями
for strategy in ['conservative', 'optimal', 'aggressive']:
    mask = df_expanded['strategy'] == strategy
    strategy_data = df_expanded[mask].set_index('original_index')
    df_final[f'bid_{strategy}'] = strategy_data['price_bid_local']
    df_final[f'prob_{strategy}'] = strategy_data['probability']

# Дополнительная аналитика
df_final['bid_ratio'] = df_final['recommended_bid'] / df_final['price_start_local']
df_final['expected_revenue'] = df_final['recommended_bid'] * df_final['probability']


# === 11. Анализ результатов ===
print(f"\n=== Анализ результатов предсказания ===")

print(f"Статистика рекомендованных бидов:")
print(f"  Средняя вероятность: {df_final['probability'].mean():.3f} ± {df_final['probability'].std():.3f}")
print(f"  Среднее соотношение: {df_final['bid_ratio'].mean():.3f}x")
print(f"  Диапазон соотношений: {df_final['bid_ratio'].min():.3f}x - {df_final['bid_ratio'].max():.3f}x")

print(f"\nРаспределение стратегий:")
strategy_counts = df_final['recommended_strategy'].value_counts()
for strategy, count in strategy_counts.items():
    percentage = count / len(df_final) * 100
    avg_prob = df_final[df_final['recommended_strategy'] == strategy]['probability'].mean()
    print(f"  {strategy}: {count} заказов ({percentage:.1f}%), средняя вероятность: {avg_prob:.3f}")

print(f"\nСравнение стратегий:")
for strategy in ['conservative', 'optimal', 'aggressive']:
    avg_prob = df_final[f'prob_{strategy}'].mean()
    avg_ratio = (df_final[f'bid_{strategy}'] / df_final['price_start_local']).mean()
    print(f"  {strategy:12}: соотношение {avg_ratio:.3f}x, вероятность {avg_prob:.3f}")


# === 12. Расчет доходов ===
print(f"\n=== Анализ доходности ===")

# Способ 1: Ожидаемая доходность (с учетом вероятности)
print("МЕТОД 1 - Ожидаемая доходность (с учетом вероятностей):")
total_expected_revenue = df_final['expected_revenue'].sum()
total_base_expected = (df_final['price_start_local'] * df_final['prob_conservative']).sum()
revenue_improvement_expected = ((total_expected_revenue / total_base_expected) - 1) * 100

print(f"Ожидаемый доход с рекомендованными бидами: {total_expected_revenue:,.0f} ₽")
print(f"Ожидаемый доход со стартовой ценой: {total_base_expected:,.0f} ₽")
print(f"Улучшение ожидаемого дохода: {revenue_improvement_expected:+.1f}%")

# Способ 2: Потенциальная доходность (если ВСЕ заказы будут приняты)
print("\nМЕТОД 2 - Потенциальная доходность (100% принятие):")
total_potential_revenue = df_final['recommended_bid'].sum()
total_base_potential = df_final['price_start_local'].sum()
revenue_improvement_potential = ((total_potential_revenue / total_base_potential) - 1) * 100

print(f"Потенциальный доход с рекомендованными бидами: {total_potential_revenue:,.0f} ₽")
print(f"Потенциальный доход со стартовой ценой: {total_base_potential:,.0f} ₽")
print(f"Улучшение потенциального дохода: {revenue_improvement_potential:+.1f}%")

# Способ 3: Реалистичная доходность (только высоковероятные заказы)
print("\nМЕТОД 3 - Реалистичная доходность (вероятность > 0.5):")
high_prob_mask = df_final['probability'] > 0.5
high_prob_count = high_prob_mask.sum()

if high_prob_count > 0:
    realistic_revenue = df_final.loc[high_prob_mask, 'recommended_bid'].sum()
    realistic_base = df_final.loc[high_prob_mask, 'price_start_local'].sum()
    revenue_improvement_realistic = ((realistic_revenue / realistic_base) - 1) * 100
    
    print(f"Заказов с вероятностью > 0.5: {high_prob_count} из {len(df_final)} ({high_prob_count/len(df_final)*100:.1f}%)")
    print(f"Доход по высоковероятным заказам: {realistic_revenue:,.0f} ₽")
    print(f"Базовый доход по этим заказам: {realistic_base:,.0f} ₽")
    print(f"Улучшение дохода: {revenue_improvement_realistic:+.1f}%")
else:
    print("Нет заказов с вероятностью > 0.5")

# Способ 4: Анализ по стратегиям
print("\nМЕТОД 4 - Анализ по стратегиям:")
strategies_analysis = []
for strategy in ['conservative', 'optimal', 'aggressive']:
    strategy_data = {
        'name': strategy,
        'avg_bid': df_final[f'bid_{strategy}'].mean(),
        'avg_prob': df_final[f'prob_{strategy}'].mean(),
        'expected_revenue': (df_final[f'bid_{strategy}'] * df_final[f'prob_{strategy}']).sum()
    }
    strategies_analysis.append(strategy_data)

# Сортировка по ожидаемой доходности
strategies_analysis.sort(key=lambda x: x['expected_revenue'], reverse=True)

print("Стратегии по убыванию ожидаемой доходности:")
for i, strategy_data in enumerate(strategies_analysis, 1):
    print(f"  {i}. {strategy_data['name']:12}: {strategy_data['expected_revenue']:,.0f} ₽ "
          f"(бид: {strategy_data['avg_bid']:.0f} ₽, вероятность: {strategy_data['avg_prob']:.3f})")

# Сравнение с фактическим выбором
actual_expected_revenue = (df_final['recommended_bid'] * df_final['probability']).sum()
best_possible_revenue = max([data['expected_revenue'] for data in strategies_analysis])
efficiency_ratio = (actual_expected_revenue / best_possible_revenue) * 100

print(f"\nЭффективность выбора стратегий: {efficiency_ratio:.1f}% "
      f"(фактическая {actual_expected_revenue:,.0f} ₽ vs лучшая {best_possible_revenue:,.0f} ₽)")


# === 13. Сохранение результатов ===
OUTPUT_PATH = "data/recommendation.csv"
output_columns = [
    'pickup_in_meters', 'pickup_in_seconds', 'distance_in_meters', 'duration_in_seconds',
    'order_timestamp', 'tender_timestamp', 'price_start_local', 'hour',
    'recommended_bid', 'probability', 'recommended_strategy', 'bid_ratio', 'expected_revenue',
    'bid_conservative', 'prob_conservative',
    'bid_optimal', 'prob_optimal', 
    'bid_aggressive', 'prob_aggressive'
]

available_columns = [col for col in output_columns if col in df_final.columns]
df_final[available_columns].to_csv(OUTPUT_PATH, sep=';', index=False)

print(f"\n[ИНФО] Результаты предсказания сохранены в {OUTPUT_PATH}")


# === 14. Сохранение результатов в JSON (для API) ===
OUTPUT_JSON_PATH = "data/recommendation.json"

# Создаем список всех рекомендаций в требуемом формате
json_recommendations = []

for idx, row in df_final.iterrows():
    # Создаем список всех трех стратегий
    prices = []
    
    # Добавляем conservative стратегию
    prices.append({
        "amount": int(row['bid_conservative']),
        "success_probability": float(row['prob_conservative']),
        "recommended": (row['recommended_strategy'] == 'conservative')
    })
    
    # Добавляем optimal стратегию
    prices.append({
        "amount": int(row['bid_optimal']),
        "success_probability": float(row['prob_optimal']),
        "recommended": (row['recommended_strategy'] == 'optimal')
    })
    
    # Добавляем aggressive стратегию
    prices.append({
        "amount": int(row['bid_aggressive']),
        "success_probability": float(row['prob_aggressive']),
        "recommended": (row['recommended_strategy'] == 'aggressive')
    })
    
    json_recommendations.append({
        "prices": prices
    })

# Сохраняем в JSON файл
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(json_recommendations, f, ensure_ascii=False, indent=2)

print(f"[ИНФО] JSON результаты сохранены в {OUTPUT_JSON_PATH}")
print(f"[ИНФО] Создано {len(json_recommendations)} JSON записей")

# Финальная сводка
print(f"\n=== Финальная сводка ===")
print(f"Обработано заказов: {len(df_final)}")
print(f"Средняя вероятность принятия: {df_final['probability'].mean():.3f}")
print(f"Среднее повышение цены: {(df_final['bid_ratio'].mean() - 1) * 100:+.1f}%")
print(f"Ожидаемая доходность: {total_expected_revenue:,.0f} ₽")

# Анализ качества рекомендаций
high_confidence = (df_final['probability'] > 0.7).sum()
medium_confidence = ((df_final['probability'] >= 0.3) & (df_final['probability'] <= 0.7)).sum()
low_confidence = (df_final['probability'] < 0.3).sum()

print(f"\nКачество рекомендаций:")
print(f"  Высокая уверенность (вероятность > 0.7): {high_confidence} заказов")
print(f"  Средняя уверенность (0.3-0.7): {medium_confidence} заказов") 
print(f"  Низкая уверенность (< 0.3): {low_confidence} заказов")