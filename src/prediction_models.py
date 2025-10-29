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

def predict_zero_from_recent_trend(df_adj_sorted):
    """Method 2: Use the most recent quarterly trend (91-day rolling window) to predict zero crossing"""
    from datetime import timedelta
    
    # Calculate most recent 91-day rolling window (Q1)
    last_date = df_adj_sorted['date_dt'].max()
    start_date = last_date - timedelta(days=91)
    
    quarter_data = df_adj_sorted[
        (df_adj_sorted['date_dt'] >= start_date) & 
        (df_adj_sorted['date_dt'] <= last_date)
    ].copy()
    
    quarter_valid = quarter_data.dropna(subset=['supply_change', 'date_numeric'])
    
    if len(quarter_valid) < 10:
        return None, "Insufficient data for recent quarterly trend (Q1)"
    
    # Calculate linear regression for this 91-day window
    slope_q, intercept_q, r_value_q, p_value_q, std_err_q = stats.linregress(
        quarter_valid['date_numeric'], quarter_valid['supply_change']
    )
    
    # Get the last data point from the quarter
    last_point = quarter_valid.iloc[-1]
    last_supply_change = last_point['supply_change']
    last_timestamp = last_point['date_numeric']
    last_obs_date = last_point['date_dt']
    
    # Calculate intercept based on last point
    intercept = last_supply_change - (slope_q * last_timestamp)
    
    # Check if supply change is moving toward zero
    zero_date = None
    days_to_zero = None
    will_cross_zero = False
    
    if slope_q < 0:  # Decreasing - will cross zero
        # Solve for when y = 0: 0 = slope * x + intercept
        zero_timestamp = -intercept / slope_q
        zero_date = pd.to_datetime(zero_timestamp, unit='s')
        days_to_zero = (zero_date - last_obs_date).days
        will_cross_zero = True
    else:
        # Increasing - won't cross zero, but still provide trend info
        # Project 2 years into the future for visualization
        future_days = 730
        future_date = last_obs_date + timedelta(days=future_days)
        zero_date = future_date  # Use as endpoint for visualization
        days_to_zero = future_days
        will_cross_zero = False
    
    return {
        'zero_date': zero_date,
        'days_from_last_observation': days_to_zero,
        'latest_period': 'Q1 (91-day window)',
        'slope': slope_q,
        'intercept': intercept,
        'r_squared': r_value_q**2,
        'last_observation_date': last_obs_date,
        'last_supply_change': last_supply_change,
        'will_cross_zero': will_cross_zero
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
    
    # Get last observation for projection start point
    last_point = valid_recent.iloc[-1]
    last_obs_date = last_point['date_dt']
    
    if slope_ma >= 0:
        # Positive slope - won't cross zero, but provide projection for visualization
        future_days = 730
        future_date = last_obs_date + timedelta(days=future_days)
        return {
            'zero_date': future_date,  # Use as endpoint for visualization
            'slope': slope_ma,
            'intercept': intercept_ma,
            'r_squared': r_value**2,
            'method': 'moving_average',
            'will_cross_zero': False
        }, f"Moving average trend is positive ({slope_ma:.2e}), won't reach zero"
    
    zero_timestamp = -intercept_ma / slope_ma
    zero_date = pd.to_datetime(zero_timestamp, unit='s')
    
    return {
        'zero_date': zero_date,
        'slope': slope_ma,
        'intercept': intercept_ma,
        'r_squared': r_value**2,
        'method': 'moving_average',
        'will_cross_zero': True
    }, "Success"


def ensemble_zero_prediction(df_adj_sorted, slope, intercept, valid_data):
    """Combine multiple prediction methods for robust predictions"""
    predictions = {}
    methods_info = {}
    
    # Get last observation date for extrapolation start point
    last_date = df_adj_sorted['date_dt'].max()
    last_timestamp = df_adj_sorted['date_numeric'].max()
    
    # Method 1: Overall Linear trend
    try:
        linear_result, linear_msg = predict_zero_crossing_linear(slope, intercept, valid_data['date_numeric'].iloc[-1])
        if linear_result:
            predictions['linear'] = linear_result
            # Calculate the supply change at last observation using overall trend
            last_value = slope * last_timestamp + intercept
            methods_info['linear'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': 0.004,  # From your notebook analysis
                'description': 'Overall dataset linear regression',
                'start_date': last_date,
                'start_value': last_value,
                'will_cross_zero': True,
                'projection_end_date': linear_result  # Store the zero crossing date
            }
        else:
            print(f"Linear prediction failed: {linear_msg}")
    except Exception as e:
        print(f"Linear prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Recent Quarterly trend (91-day rolling window)
    try:
        quarterly_result, quarterly_msg = predict_zero_from_recent_trend(df_adj_sorted)
        if quarterly_result and isinstance(quarterly_result, dict):
            # Store method info for visualization (always)
            methods_info['quarterly'] = {
                'slope': quarterly_result['slope'],
                'intercept': quarterly_result['intercept'],
                'r_squared': quarterly_result['r_squared'],
                'description': f"Based on {quarterly_result['latest_period']} trend",
                'start_date': quarterly_result['last_observation_date'],
                'start_value': quarterly_result['last_supply_change'],
                'will_cross_zero': quarterly_result.get('will_cross_zero', True),
                'projection_end_date': quarterly_result['zero_date']  # For visualization
            }
            # Only add to predictions dict if it will actually cross zero
            if quarterly_result.get('will_cross_zero', True):
                predictions['quarterly'] = quarterly_result['zero_date']
                print(f"Quarterly prediction: {quarterly_result['zero_date']}")
            else:
                print(f"Quarterly trend won't cross zero (slope positive), excluded from average but shown on chart")
        else:
            print(f"Quarterly prediction failed: {quarterly_msg}")
    except Exception as e:
        print(f"Quarterly prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Moving Average trend
    try:
        ma_result, ma_msg = predict_zero_from_moving_average(df_adj_sorted)
        if ma_result and isinstance(ma_result, dict):
            # Get the last MA value
            last_ma_value = df_adj_sorted['change_30d_avg'].iloc[-1]
            # Store method info for visualization (always)
            methods_info['moving_average'] = {
                'slope': ma_result['slope'],
                'intercept': ma_result['intercept'],
                'r_squared': ma_result['r_squared'],
                'description': '30-day moving average trend (60 days of data)',
                'start_date': last_date,
                'start_value': last_ma_value,
                'will_cross_zero': ma_result.get('will_cross_zero', True),
                'projection_end_date': ma_result['zero_date']  # For visualization
            }
            # Only add to predictions dict if it will actually cross zero
            if ma_result.get('will_cross_zero', True):
                predictions['moving_average'] = ma_result['zero_date']
                print(f"Moving average prediction: {ma_result['zero_date']}")
            else:
                print(f"Moving average trend won't cross zero (slope positive), excluded from average but shown on chart")
        else:
            print(f"Moving average prediction failed: {ma_msg}")
    except Exception as e:
        print(f"Moving average prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Ensemble predictions generated: {len(predictions)} methods")
    for method, pred_date in predictions.items():
        print(f"  - {method}: {pred_date}")
    
    return predictions, methods_info