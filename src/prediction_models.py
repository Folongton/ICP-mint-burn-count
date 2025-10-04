import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

def predict_zero_crossing_linear(slope, intercept, current_date_numeric):
    """Method 1: Predict when supply change reaches zero using overall linear trend"""
    if slope >= 0:
        return None, "Slope is positive or zero, supply change won't reach zero"
    
    # Solve for: slope * x + intercept = 0
    zero_crossing_timestamp = -intercept / slope
    zero_crossing_date = pd.to_datetime(zero_crossing_timestamp, unit='s')
    
    return zero_crossing_date, "Success"

def predict_zero_from_recent_trend(quarterly_trends, df_adj_sorted):
    """Method 2: Use the most recent quarterly trend to predict zero crossing"""
    
    if not quarterly_trends:
        return None, "No quarterly trends available"
    
    # Get the most recent quarter's trend
    latest_period = max(quarterly_trends.keys())
    latest_trend = quarterly_trends[latest_period]
    
    # Check if supply change is moving toward zero
    if latest_trend['slope'] >= 0:
        return None, f"Supply change is increasing (slope = {latest_trend['slope']:.2e}), won't reach zero with current trend"
    
    # Get the last data point from the most recent quarter
    quarter_data = df_adj_sorted[df_adj_sorted['year_quarter'] == latest_period].copy()
    quarter_data = quarter_data.dropna(subset=['supply_change', 'date_numeric']).sort_values('date_dt')
    
    if len(quarter_data) == 0:
        return None, "No valid data for recent quarter"
    
    # Use the last point of the quarter for extrapolation
    last_date = quarter_data['date_dt'].iloc[-1]
    last_supply_change = quarter_data['supply_change'].iloc[-1]
    last_timestamp = quarter_data['date_numeric'].iloc[-1]
    
    # Calculate trend line equation: y = mx + b
    slope = latest_trend['slope']
    intercept = last_supply_change - (slope * last_timestamp)
    
    # Solve for when y = 0: 0 = slope * x + intercept
    zero_timestamp = -intercept / slope
    zero_date = pd.to_datetime(zero_timestamp, unit='s')
    
    # Calculate days from last observation
    days_to_zero = (zero_date - last_date).days
    
    return {
        'zero_date': zero_date,
        'days_from_last_observation': days_to_zero,
        'latest_period': latest_period,
        'slope': slope,
        'intercept': intercept,
        'r_squared': latest_trend['r_squared'],
        'last_observation_date': last_date,
        'last_supply_change': last_supply_change
    }, "Success"

def predict_zero_from_moving_average(df_adj_sorted, window=30):
    """Method 3: Predict zero crossing using moving average trend"""
    # Calculate trend from recent moving average
    recent_data = df_adj_sorted.tail(window * 2)  # Use 2x window for stability
    
    # Fit linear trend to recent moving average
    valid_recent = recent_data.dropna(subset=['change_30d_avg', 'date_numeric'])
    if len(valid_recent) < 10:
        return None, "Insufficient data for moving average trend"
    
    slope_ma, intercept_ma, r_value, p_value, std_err = stats.linregress(
        valid_recent['date_numeric'], valid_recent['change_30d_avg']
    )
    
    if slope_ma >= 0:
        return None, f"Moving average trend is positive ({slope_ma:.2e}), won't reach zero"
    
    zero_timestamp = -intercept_ma / slope_ma
    zero_date = pd.to_datetime(zero_timestamp, unit='s')
    
    return {
        'zero_date': zero_date,
        'slope': slope_ma,
        'r_squared': r_value**2,
        'method': 'moving_average'
    }, "Success"



def ensemble_zero_prediction(df_adj_sorted, quarterly_trends, slope, intercept, valid_data):
    """Combine multiple prediction methods for robust predictions"""
    predictions = {}
    methods_info = {}
    
    # Method 1: Overall Linear trend
    try:
        linear_result, linear_msg = predict_zero_crossing_linear(slope, intercept, valid_data['date_numeric'].iloc[-1])
        if linear_result:
            predictions['linear'] = linear_result
            methods_info['linear'] = {
                'slope': slope,
                'r_squared': 0.004,  # From your notebook analysis
                'description': 'Overall dataset linear regression'
            }
    except Exception:
        pass
    
    # Method 2: Recent Quarterly trend
    try:
        quarterly_result, quarterly_msg = predict_zero_from_recent_trend(quarterly_trends, df_adj_sorted)
        if quarterly_result and isinstance(quarterly_result, dict):
            predictions['quarterly'] = quarterly_result['zero_date']
            methods_info['quarterly'] = {
                'slope': quarterly_result['slope'],
                'r_squared': quarterly_result['r_squared'],
                'description': f"Based on {quarterly_result['latest_period']} trend"
            }
    except Exception:
        pass
    
    # Method 3: Moving Average trend
    try:
        ma_result, ma_msg = predict_zero_from_moving_average(df_adj_sorted)
        if ma_result and isinstance(ma_result, dict):
            predictions['moving_average'] = ma_result['zero_date']
            methods_info['moving_average'] = {
                'slope': ma_result['slope'],
                'r_squared': ma_result['r_squared'],
                'description': '30-day moving average trend (60 days of data)'
            }
    except Exception:
        pass
    
    # Method 4: Acceleration model removed - was not providing reliable predictions
    
    return predictions, methods_info