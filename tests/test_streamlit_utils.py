"""Tests for streamlit_utils chart functions (pure Plotly, no Streamlit session needed)."""

import unittest
import sys
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.streamlit_utils import create_interactive_trends_chart


def _make_mock_df_adj_sorted(n_days: int = 180) -> pd.DataFrame:
    """Build a mock processed DataFrame matching the schema expected by chart functions."""
    base_date = datetime(2024, 1, 1)
    date_dts = pd.to_datetime([base_date + timedelta(days=i) for i in range(n_days)])
    rng = np.random.default_rng(7)
    supply_change = rng.standard_normal(n_days) * 1000 - 200

    df = pd.DataFrame({
        'date_dt': date_dts,
        'date_numeric': [d.timestamp() for d in date_dts],
        'supply_change': supply_change,
        'change_7d_avg': pd.Series(supply_change).rolling(window=7).mean().values,
        'change_30d_avg': pd.Series(supply_change).rolling(window=30).mean().values,
    })
    return df.sort_values('date_dt').reset_index(drop=True)


class TestInteractiveTrendsChart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _make_mock_df_adj_sorted(180)
        valid_data = cls.df.dropna(subset=['supply_change', 'date_numeric'])
        cls.slope, cls.intercept, cls.r_value, _, _ = stats.linregress(
            valid_data['date_numeric'], valid_data['supply_change']
        )
        cls.valid_data = valid_data
        cls.fig = create_interactive_trends_chart(
            cls.df, cls.valid_data, cls.slope, cls.intercept, cls.r_value
        )

    def test_returns_figure(self):
        import plotly.graph_objects as go
        self.assertIsInstance(self.fig, go.Figure)

    def test_xaxis_range_is_set(self):
        self.assertIsNotNone(
            self.fig.layout.xaxis.range,
            "xaxis.range should be set for initial 6-month zoom"
        )

    def test_xaxis_range_start_is_approximately_6_months_ago(self):
        """Range start should be within 1 day of last_date - 182 days."""
        last_date = self.df['date_dt'].max()
        expected_start = last_date - timedelta(days=182)
        actual_start = pd.to_datetime(self.fig.layout.xaxis.range[0])
        delta_days = abs((actual_start - expected_start).days)
        self.assertLessEqual(
            delta_days, 1,
            f"Expected x-axis start ~{expected_start.date()}, got {actual_start.date()}"
        )

    def test_xaxis_range_end_is_last_date(self):
        """Range end should be within 1 day of the last observation."""
        last_date = self.df['date_dt'].max()
        actual_end = pd.to_datetime(self.fig.layout.xaxis.range[1])
        delta_days = abs((actual_end - last_date).days)
        self.assertLessEqual(delta_days, 1)

    def test_chart_has_base_traces(self):
        """Figure should include at least the 7-day and 30-day average traces."""
        trace_names = [t.name for t in self.fig.data]
        self.assertIn('7-Day Avg', trace_names)
        self.assertIn('30-Day Avg', trace_names)


if __name__ == '__main__':
    unittest.main()
