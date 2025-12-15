# -*- coding: utf-8 -*-
"""
Генерує артефакти прогнозу поверх моделі:
 - reports/<keyword>_forecast.png
 - reports/<keyword>_forecast.csv
 - reports/<keyword>_validation.csv
 - reports/<keyword>_metrics.json
Працює з GoogleTrendsPredictor із predict.py.
"""
import os, json, argparse
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from modules.core.predict import GoogleTrendsPredictor

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run(keyword: str, outdir: str = "reports", prediction_days: int = 30, seq_length: int = 12) -> int:
    ensure_dir(outdir)

    predictor = GoogleTrendsPredictor(seq_length=seq_length, prediction_days=prediction_days)

    # 1) Дані
    data = predictor.load_data(keyword)
    if data.empty:
        print(f"[WARN] Дані для '{keyword}' не знайдено у db.sqlite3 (is_partial=0).")
        return 1

    # 2) Масштабування і послідовності
    scaled_values = predictor.scaler.fit_transform(data['value'].values.reshape(-1, 1))
    X, y = predictor.create_sequences(scaled_values)
    if len(X) < 2:
        print(f"[WARN] Замало точок для побудови послідовностей: {len(X)}")
        return 2

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train,  y_val  = y[:split_idx], y[split_idx:]

    # 3) Навчання
    predictor.train_model(X_train, y_train, X_val, y_val)

    # 4) Оцінка на валідації (усі метрики: RMSE, MAE, R2, MAPE, sMAPE)
    y_pred_scaled = predictor.model.predict(X_val)
    metrics = predictor.evaluate_model(y_val, y_pred_scaled)

    # інверсія масштабування для таблиць/графіків
    y_pred = predictor.scaler.inverse_transform(y_pred_scaled)
    y_val_inv = predictor.scaler.inverse_transform(y_val)

    # коректне вирівнювання дат валідації з урахуванням seq_length
    val_start_idx = predictor.seq_length + split_idx
    val_len = len(y_pred)
    val_dates = data['date'].iloc[val_start_idx: val_start_idx + val_len]

    # 5) Прогноз у майбутнє
    last_sequence = scaled_values[-predictor.seq_length:].reshape(1, predictor.seq_length, 1)
    future_values = predictor.generate_future_predictions(last_sequence, predictor.prediction_days)
    future_dates  = [data['date'].max() + timedelta(days=i) for i in range(1, predictor.prediction_days + 1)]

    # 6) Збереження артефактів
    base = os.path.join(outdir, keyword.replace(" ", "_"))
    png_path         = base + "_forecast.png"
    csv_forecast_path= base + "_forecast.csv"
    csv_val_path     = base + "_validation.csv"
    json_metrics_path= base + "_metrics.json"

    # графік: історія + валідація + майбутнє
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['value'], label='Фактичні дані', marker='o')
    plt.plot(val_dates.values, y_pred.reshape(-1), label='Прогноз (валідація)', linestyle='--')
    plt.plot(future_dates, future_values, label=f'Прогноз на {predictor.prediction_days} днів', marker='x')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f"Аналіз та прогноз для '{keyword}'")
    plt.xlabel("Дата")
    plt.ylabel("Рейтинг")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # CSV: майбутній прогноз
    pd.DataFrame({'date': future_dates, 'forecast': future_values}).to_csv(
        csv_forecast_path, index=False, encoding='utf-8-sig'
    )

    # CSV: валідація (true vs pred)
    pd.DataFrame({
        'date': pd.to_datetime(val_dates.values),
        'true': y_val_inv.reshape(-1),
        'pred': y_pred.reshape(-1)
    }).to_csv(csv_val_path, index=False, encoding='utf-8-sig')

    # JSON: метрики (усі, як у дипломі)
    with open(json_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'keyword': keyword,
                'prediction_days': predictor.prediction_days,
                'rmse': metrics["RMSE"],
                'mae': metrics["MAE"],
                'r2': metrics["R2"],
                'mape': metrics["MAPE"],
                'smape': metrics["sMAPE"],
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[OK] Збережено:\n - {png_path}\n - {csv_forecast_path}\n - {csv_val_path}\n - {json_metrics_path}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Прогін прогнозу з артефактами у reports/")
    ap.add_argument("keyword", type=str, help="Ключове слово")
    ap.add_argument("-o", "--outdir", default="reports", help="Каталог артефактів")
    ap.add_argument("-p", "--prediction", type=int, default=30, help="К-ть днів прогнозу")
    ap.add_argument("-s", "--seq", type=int, default=12, help="Довжина послідовності")
    args = ap.parse_args()
    raise SystemExit(run(args.keyword, args.outdir, args.prediction, args.seq))
