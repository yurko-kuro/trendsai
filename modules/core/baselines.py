# modules/core/baselines.py
import time
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- метрики (узгодити з predict.py) ---

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

def calc_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }

# --- утиліти ---

def make_lagged_features(series_1d: np.ndarray, seq_length: int):
    """
    series_1d: (N,) у вже масштабованій шкалі (або в оригінальній — не критично для ML)
    Повертає X:(N-seq, seq), y:(N-seq,)
    """
    X, y = [], []
    for i in range(len(series_1d) - seq_length):
        X.append(series_1d[i:i+seq_length])
        y.append(series_1d[i+seq_length])
    return np.asarray(X), np.asarray(y)

# --- базові моделі ---

def naive_forecast(train_series: np.ndarray, steps: int):
    # прогноз = останнє значення train, на всі точки валідації
    last = float(train_series[-1])
    return np.full((steps,), last, dtype=float)

def seasonal_naive_forecast(train_series: np.ndarray, steps: int, season: int = 365):
    """
    Якщо даних < season — fallback на naive.
    Прогноз = значення з лагом season (по колу).
    """
    train_series = np.asarray(train_series).reshape(-1)
    if len(train_series) < season:
        return naive_forecast(train_series, steps)

    preds = []
    for i in range(steps):
        idx = len(train_series) - season + (i % season)
        preds.append(float(train_series[idx]))
    return np.asarray(preds, dtype=float)

# --- ML моделі (lag features) ---

def ridge_model(X_train, y_train, X_val):
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train, y_train)
    return model.predict(X_val)

def rf_model(X_train, y_train, X_val):
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)

# --- LSTM baseline ---

def lstm_model(X_train, y_train, X_val, y_val, seq_length: int):
    """
    X_* очікується у формі (samples, seq, 1)
    """
    tf.keras.utils.set_random_seed(0)

    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        verbose=0,
        shuffle=False,
        callbacks=[es]
    )
    pred = model.predict(X_val, verbose=0).reshape(-1)
    return pred

# --- головний раннер ---

def run_all_baselines(values_orig: np.ndarray, scaler, seq_length: int, split_ratio: float = 0.8, season: int = 365) -> dict:
    """
    values_orig: (N,) у оригінальній шкалі (0..100)
    scaler: RobustScaler з predict.py (буде fit всередині пайплайна, але сюди даємо той самий інстанс)
    Повертає словник результатів по моделях.
    """
    t0 = time.time()

    # 1) scale як у T-GRU
    scaled = scaler.fit_transform(values_orig.reshape(-1, 1)).reshape(-1)

    # 2) supervised (для ML/LSTM)
    X, y = make_lagged_features(scaled, seq_length)
    split = int(split_ratio * len(X))

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # y_val/y_pred -> inverse до оригінальної шкали для метрик
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)

    results = {}

    # NAIVE / SEASONAL NAIVE — у оригінальній шкалі (щоб без плутанини)
    # Для чесності беремо "train tail" у ориг. шкалі, вирівняний як y_val
    # y_val відповідає точкам починаючи з індексу seq_length + split у початковому ряді
    val_len = len(y_val)
    train_end_index = seq_length + split  # індекс у початковому values_orig, де починається валідація
    train_series_orig = values_orig[:train_end_index]
    naive_pred = naive_forecast(train_series_orig, val_len)
    results["Naive"] = calc_metrics(y_val_inv, naive_pred)

    seasonal_pred = seasonal_naive_forecast(train_series_orig, val_len, season=season)
    results["SeasonalNaive"] = calc_metrics(y_val_inv, seasonal_pred)

    # RIDGE (lag) — прогноз у scaled -> inverse
    ridge_pred_scaled = ridge_model(X_train, y_train, X_val)
    ridge_pred = scaler.inverse_transform(ridge_pred_scaled.reshape(-1, 1)).reshape(-1)
    results["RidgeLag"] = calc_metrics(y_val_inv, ridge_pred)

    # RandomForest (lag)
    rf_pred_scaled = rf_model(X_train, y_train, X_val)
    rf_pred = scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).reshape(-1)
    results["RandomForestLag"] = calc_metrics(y_val_inv, rf_pred)

    # LSTM
    X_train_l = X_train.reshape(-1, seq_length, 1)
    X_val_l   = X_val.reshape(-1, seq_length, 1)
    y_train_l = y_train.reshape(-1, 1)
    y_val_l   = y_val.reshape(-1, 1)

    lstm_pred_scaled = lstm_model(X_train_l, y_train_l, X_val_l, y_val_l, seq_length=seq_length)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).reshape(-1)
    results["LSTM"] = calc_metrics(y_val_inv, lstm_pred)

    results["_meta"] = {
        "seq_length": int(seq_length),
        "split_ratio": float(split_ratio),
        "season": int(season),
        "val_len": int(val_len),
        "runtime_sec": float(round(time.time() - t0, 3))
    }
    return results
