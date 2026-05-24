"""Tests for LSTM forecast model."""

import unittest
import sys
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lstm_model import run_lstm_forecast


def _make_mock_df(n_days: int = 60) -> pd.DataFrame:
    """Create a minimal mock df_adj_sorted-style DataFrame."""
    base_date = datetime(2024, 1, 1)
    dates = pd.to_datetime([base_date + timedelta(days=i) for i in range(n_days)])
    rng = np.random.default_rng(0)
    supply_change = rng.standard_normal(n_days) * 1000
    return pd.DataFrame({'date_dt': dates, 'supply_change': supply_change})


class TestLSTMModel(unittest.TestCase):
    """Use minimal parameters so the full test suite stays fast."""

    LOOKBACK = 5
    FORECAST_DAYS = 10
    N_MC = 5  # Small sample count sufficient for shape/ordering checks

    @classmethod
    def setUpClass(cls):
        """Train once; re-use across all test methods."""
        df = _make_mock_df(60)
        cls.result = run_lstm_forecast(
            df,
            lookback=cls.LOOKBACK,
            forecast_days=cls.FORECAST_DAYS,
            n_mc_samples=cls.N_MC,
        )
        cls.last_observed = df['date_dt'].max()

    def test_forecast_mean_length(self):
        forecast_mean, _, _, _ = self.result
        self.assertEqual(len(forecast_mean), self.FORECAST_DAYS)

    def test_forecast_ci_length(self):
        _, forecast_lower, forecast_upper, _ = self.result
        self.assertEqual(len(forecast_lower), self.FORECAST_DAYS)
        self.assertEqual(len(forecast_upper), self.FORECAST_DAYS)

    def test_forecast_ci_ordering(self):
        """lower <= upper for every forecast step (mean may sit anywhere inside)."""
        _, forecast_lower, forecast_upper, _ = self.result
        for i in range(self.FORECAST_DAYS):
            self.assertLessEqual(
                forecast_lower[i], forecast_upper[i],
                f"CI inverted at step {i}: lower={forecast_lower[i]}, upper={forecast_upper[i]}"
            )

    def test_forecast_dates_length(self):
        _, _, _, forecast_dates = self.result
        self.assertEqual(len(forecast_dates), self.FORECAST_DAYS)

    def test_forecast_dates_sequential(self):
        """Each forecast date should be exactly 1 day after the previous."""
        _, _, _, forecast_dates = self.result
        for i in range(1, len(forecast_dates)):
            delta = (forecast_dates[i] - forecast_dates[i - 1]).days
            self.assertEqual(delta, 1, f"Non-sequential dates at index {i}: gap={delta} days")

    def test_forecast_dates_after_last_observation(self):
        _, _, _, forecast_dates = self.result
        self.assertGreater(forecast_dates[0], self.last_observed)

    def test_insufficient_data_raises(self):
        tiny_df = _make_mock_df(n_days=3)
        with self.assertRaises(ValueError):
            run_lstm_forecast(tiny_df, lookback=5, forecast_days=10, n_mc_samples=2)

    def test_missing_column_raises(self):
        bad_df = _make_mock_df(60).drop(columns=['supply_change'])
        with self.assertRaises(ValueError):
            run_lstm_forecast(bad_df, lookback=5, forecast_days=5, n_mc_samples=2)


if __name__ == '__main__':
    unittest.main()
