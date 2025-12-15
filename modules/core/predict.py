# predict.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# === допоміжні метрики ===

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


# === позиційне кодування ===

class PositionalEncoding(tf.keras.layers.Layer):
    """Легке синусоїдальне позиційне кодування для коротких послідовностей."""
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

    def call(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        dims = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)

        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2 * (dims // 2)) / tf.cast(self.d_model, tf.float32),
        )
        angle_rads = positions * angle_rates  # (seq_len, d_model)

        # парні індекси — sin, непарні — cos
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (seq_len, d_model)
        pos_encoding = tf.expand_dims(pos_encoding, 0)       # (1, seq_len, d_model)

        return x + pos_encoding


# === основний клас прогнозу ===

class GoogleTrendsPredictor:
    def __init__(self, db_path: str = "db.sqlite3", seq_length: int = 12,
                 prediction_days: int = 30):
        self.db_path = db_path
        self.seq_length = seq_length
        self.prediction_days = prediction_days
        self.scaler = RobustScaler()
        self.model = self._build_tgru()

    # === архітектура гібридної моделі Transformer–GRU ===

    def _build_tgru(self):
        """
        Гібридна модель T-GRU:
        - 1 блок Transformer-encoder (MultiHeadAttention + FFN + LayerNorm)
        - 2 шари GRU (128, 64)
        - Dense(32, relu) + Dense(1)
        """
        d_model = 128  # розмір простору ознак після проєкції

        inp = Input(shape=(self.seq_length, 1))

        # 1) Лінійна проєкція в d_model-вимірний простір
        x = Dense(d_model)(inp)

        # 2) Позиційне кодування
        x = PositionalEncoding(d_model)(x)

        # 3) Transformer-encoder block (спрощений, 1 шар)
        attn_out = MultiHeadAttention(
            num_heads=8,
            key_dim=d_model // 8,
        )(x, x)
        attn_out = Dropout(0.1)(attn_out)
        x1 = LayerNormalization(epsilon=1e-6)(x + attn_out)

        ffn = Dense(d_model, activation="relu")(x1)
        ffn = Dense(d_model)(ffn)
        x2 = LayerNormalization(epsilon=1e-6)(x1 + ffn)

        # 4) GRU-шари
        x_gru = GRU(128, activation="tanh", return_sequences=True)(x2)
        x_gru = GRU(64, activation="tanh")(x_gru)

        # 5) Вихідні шари
        x_dense = Dense(32, activation="relu")(x_gru)
        out = Dense(1)(x_dense)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=Huber(delta=1.0),
        )
        return model

    # === робота з даними ===

    def load_data(self, keyword: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                q = """
                SELECT i.date, i.value
                FROM trends_interest i
                JOIN trends_keyword k ON i.keyword_id = k.id
                WHERE k.name = ? AND i.is_partial = 0
                ORDER BY i.date
                """
                return pd.read_sql(q, conn, params=(keyword,), parse_dates=["date"])
        except Exception as e:
            print(f"[DB] Помилка: {e}")
            return pd.DataFrame()

    def create_sequences(self, arr: np.ndarray):
        X, y = [], []
        for i in range(len(arr) - self.seq_length):
            X.append(arr[i : i + self.seq_length])
            y.append(arr[i + self.seq_length])
        return np.array(X), np.array(y)

    # === навчання ===

    def train_model(self, X_train, y_train, X_val, y_val, tag: str = "tgru"):
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                f"best_{tag}.keras",
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        self.model.fit(
            X_train,
            y_train,
            epochs=40,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # важливо для таймсерій
        )

    # === оцінка ===

    def evaluate_model(self, y_val_scaled, y_pred_scaled, keyword: str | None = None):
        """
        keyword залишено опційним лише для сумісності зі старим pipeline,
        всередині не використовується.
        """
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_val_scaled)

        metrics = {
            "RMSE": rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE": mape(y_true, y_pred),
            "sMAPE": smape(y_true, y_pred),
        }
        print("\n📊 Оцінка моделі:", {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    # === генерація майбутніх значень ===

    def generate_future_predictions(self, last_sequence, days_to_predict: int):
        preds = []
        cur = last_sequence.copy()
        for _ in range(days_to_predict):
            nxt = self.model.predict(cur, verbose=0)
            preds.append(self.scaler.inverse_transform(nxt)[0, 0])
            cur = np.append(cur[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
        return preds

    # === візуалізація ===

    def plot_results(self, data, y_pred, future_dates, future_values, keyword: str):
        plt.figure(figsize=(12, 6))
        plt.plot(
            data["date"],
            data["value"],
            label="Фактичні дані",
            marker="o",
        )

        split_idx = int(0.8 * len(data))
        val_dates = data["date"][split_idx:]
        m = min(len(val_dates), len(y_pred))
        plt.plot(
            val_dates[:m],
            y_pred[:m],
            label="Прогноз (валідація)",
            linestyle="--",
        )

        plt.plot(
            future_dates,
            future_values,
            label=f"Прогноз на {len(future_dates)} днів",
            marker="x",
        )

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.legend()
        plt.title(f"'{keyword}' — аналіз і прогноз")
        plt.xlabel("Дата")
        plt.ylabel("Рейтинг")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === головний метод прогнозу (інтерактивний режим view) ===

    def forecast(self, keyword: str):
        data = self.load_data(keyword)
        if data.empty:
            print(f"Дані для '{keyword}' не знайдено.")
            return None

        scaled = self.scaler.fit_transform(
            data["value"].values.reshape(-1, 1)
        )
        X, y = self.create_sequences(scaled)

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.train_model(X_train, y_train, X_val, y_val, tag="tgru")

        y_pred_scaled = self.model.predict(X_val)
        self.evaluate_model(y_val, y_pred_scaled, keyword=keyword)

        last_seq = scaled[-self.seq_length :].reshape(1, self.seq_length, 1)
        future_vals = self.generate_future_predictions(
            last_seq, self.prediction_days
        )
        future_dates = [
            data["date"].max() + timedelta(days=i)
            for i in range(1, self.prediction_days + 1)
        ]

        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        self.plot_results(data, y_pred, future_dates, future_vals, keyword)

        res = pd.DataFrame(
            {
                "Дата": future_dates,
                "Прогнозований рейтинг": future_vals,
            }
        )
        print("\nТаблиця прогнозу по днях:\n", res.to_string(index=False))
        return res
