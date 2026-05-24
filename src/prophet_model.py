import pandas as pd
from prophet import Prophet


def run_prophet_forecast(df_adj_sorted: pd.DataFrame, forecast_days: int = 365) -> pd.DataFrame:
    """
    Fit a Prophet model on historical ICP supply change data and forecast future values.

    Args:
        df_adj_sorted: DataFrame with 'date_dt' (datetime) and 'supply_change' (float) columns.
        forecast_days: Number of days to forecast beyond the last observed date.

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper (all rows: history + forecast).

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
    """
    required = {'date_dt', 'supply_change'}
    missing = required - set(df_adj_sorted.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    prophet_df = df_adj_sorted[['date_dt', 'supply_change']].dropna().copy()
    prophet_df = prophet_df.rename(columns={'date_dt': 'ds', 'supply_change': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    if prophet_df.empty:
        raise ValueError("No valid data rows after dropping NaN values.")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
