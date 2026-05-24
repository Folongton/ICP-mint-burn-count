"""Tests for Prophet forecast model."""

import unittest
import sys
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prophet_model import run_prophet_forecast


def _make_mock_df(n_days: int = 100) -> pd.DataFrame:
    """Create a minimal mock df_adj_sorted-style DataFrame."""
    base_date = datetime(2024, 1, 1)
    dates = pd.to_datetime([base_date + timedelta(days=i) for i in range(n_days)])
    rng = np.random.default_rng(42)
    supply_change = rng.standard_normal(n_days) * 1000 - 500
    return pd.DataFrame({'date_dt': dates, 'supply_change': supply_change})


class TestProphetModel(unittest.TestCase):

    def setUp(self):
        self.df = _make_mock_df(100)
        self.forecast_days = 10
        self.result = run_prophet_forecast(self.df, forecast_days=self.forecast_days)

    def test_forecast_returns_dataframe(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_forecast_columns(self):
        for col in ('ds', 'yhat', 'yhat_lower', 'yhat_upper'):
            self.assertIn(col, self.result.columns, f"Missing column: {col}")

    def test_forecast_length(self):
        last_observed = self.df['date_dt'].max()
        future_rows = self.result[self.result['ds'] > last_observed]
        self.assertEqual(len(future_rows), self.forecast_days)

    def test_forecast_ci_ordering(self):
        """yhat_lower <= yhat <= yhat_upper for every row in the forecast."""
        last_observed = self.df['date_dt'].max()
        future = self.result[self.result['ds'] > last_observed]
        self.assertTrue((future['yhat_lower'] <= future['yhat']).all(),
                        "yhat_lower > yhat for some rows")
        self.assertTrue((future['yhat'] <= future['yhat_upper']).all(),
                        "yhat > yhat_upper for some rows")

    def test_empty_data_raises(self):
        empty_df = pd.DataFrame({'date_dt': [], 'supply_change': []})
        with self.assertRaises(ValueError):
            run_prophet_forecast(empty_df, forecast_days=10)

    def test_missing_column_raises(self):
        bad_df = self.df.drop(columns=['supply_change'])
        with self.assertRaises(ValueError):
            run_prophet_forecast(bad_df, forecast_days=10)


if __name__ == '__main__':
    unittest.main()
