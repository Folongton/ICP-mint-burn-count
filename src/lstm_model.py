from __future__ import annotations

from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

_EPOCHS = 50


class _ProgressCallback(tf.keras.callbacks.Callback):
    """Keras callback that forwards epoch progress to an external callable."""

    def __init__(self, cb: Callable, total: int) -> None:
        super().__init__()
        self._cb = cb
        self._total = total

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self._cb("training", epoch + 1, self._total)


def run_lstm_forecast(
    df_adj_sorted: pd.DataFrame,
    lookback: int = 30,
    forecast_days: int = 365,
    n_mc_samples: int = 100,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[float], List[float], List[float], List[pd.Timestamp]]:
    """
    Train an LSTM model on ICP supply change history and forecast future values with
    Monte Carlo Dropout confidence intervals.

    Args:
        df_adj_sorted: DataFrame with 'date_dt' (datetime) and 'supply_change' (float) columns.
        lookback:       Number of past days used as the input window for each prediction.
        forecast_days:  Number of days to forecast beyond the last observed date.
        n_mc_samples:   Number of stochastic forward passes for MC Dropout CI estimation.

    Returns:
        Tuple of:
            forecast_mean   – mean predicted supply change, length = forecast_days
            forecast_lower  – lower bound of 95% CI, length = forecast_days
            forecast_upper  – upper bound of 95% CI, length = forecast_days
            forecast_dates  – list of pd.Timestamps for each forecast day

    Raises:
        ValueError: If there are insufficient rows to build at least one training sequence.
    """
    required = {'date_dt', 'supply_change'}
    missing = required - set(df_adj_sorted.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    series = df_adj_sorted['supply_change'].dropna().values.astype(float)

    if len(series) < lookback + 1:
        raise ValueError(
            f"Insufficient data: need at least {lookback + 1} rows, got {len(series)}"
        )

    # Scale to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))  # shape (n, 1)

    # Build supervised sequences
    X_list, y_list = [], []
    for i in range(lookback, len(scaled)):
        X_list.append(scaled[i - lookback : i, 0])
        y_list.append(scaled[i, 0])
    X = np.array(X_list).reshape(-1, lookback, 1)
    y = np.array(y_list)

    # Build model with Dropout for MC Dropout inference
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    _callbacks = [_ProgressCallback(progress_callback, _EPOCHS)] if progress_callback is not None else []
    model.fit(X, y, epochs=_EPOCHS, batch_size=16, validation_split=0.1, verbose=0, callbacks=_callbacks)

    # Monte Carlo Dropout: vectorized — all n_mc_samples run as one batched call per day.
    # 365 model calls instead of 36,500 — ~100x faster on CPU.
    # All samples start from the same window and diverge each step via independent Dropout masks.
    current_windows = np.tile(
        scaled[-lookback:, 0].astype(np.float32), (n_mc_samples, 1)
    )  # shape: (n_mc_samples, lookback)
    all_paths = np.zeros((n_mc_samples, forecast_days), dtype=np.float32)

    for day_idx in range(forecast_days):
        x_batch = current_windows.reshape(n_mc_samples, lookback, 1)
        # One call, batch_size=n_mc_samples; training=True keeps Dropout active
        preds = model(x_batch, training=True).numpy().flatten()  # shape (n_mc_samples,)
        all_paths[:, day_idx] = preds
        # Slide window: drop oldest timestep, append this step's predictions
        current_windows = np.concatenate(
            [current_windows[:, 1:], preds.reshape(-1, 1)], axis=1
        )
        if progress_callback is not None:
            progress_callback("inference", day_idx + 1, forecast_days)

    # Aggregate: mean ± 1.96 × std → 95% CI
    mean_scaled = all_paths.mean(axis=0)       # shape (forecast_days,)
    std_scaled = all_paths.std(axis=0)
    lower_scaled = mean_scaled - 1.96 * std_scaled
    upper_scaled = mean_scaled + 1.96 * std_scaled

    def _inverse(arr: np.ndarray) -> List[float]:
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten().tolist()

    forecast_mean = _inverse(mean_scaled)
    forecast_lower = _inverse(lower_scaled)
    forecast_upper = _inverse(upper_scaled)

    last_date: pd.Timestamp = df_adj_sorted['date_dt'].max()
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    return forecast_mean, forecast_lower, forecast_upper, forecast_dates
