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

def predict_zero_with_acceleration(df_adj_sorted):
    """Method 4: Use acceleration (2nd derivative) to improve predictions"""
    # Calculate 2nd derivative (acceleration of supply change)
    if 'supply_change_derivative' not in df_adj_sorted.columns:
        df_adj_sorted['supply_change_derivative'] = np.gradient(df_adj_sorted['supply_change'])
    
    if 'acceleration' not in df_adj_sorted.columns:
        df_adj_sorted['acceleration'] = np.gradient(df_adj_sorted['supply_change_derivative'])
    
    # Get recent values (last 30 days average to reduce noise)
    recent_data = df_adj_sorted.tail(30)
    recent_change = recent_data['supply_change'].mean()
    recent_speed = recent_data['supply_change_derivative'].mean()
    recent_accel = recent_data['acceleration'].mean()
    
    # Check if we're heading toward zero
    if recent_change > 0 and recent_speed >= 0:
        return None, "Supply change is positive and increasing, not heading toward zero"
    
    if recent_change > 0 and recent_speed < 0:
        # Currently positive but decreasing - use simple linear projection
        if recent_speed >= 0:
            return None, "Speed is not negative enough to reach zero"
        
        days_to_zero = -recent_change / recent_speed
        last_date = df_adj_sorted['date_dt'].iloc[-1]
        zero_date = last_date + pd.Timedelta(days=days_to_zero)
        
        return {
            'zero_date': zero_date,
            'method': 'linear_speed',
            'recent_change': recent_change,
            'recent_speed': recent_speed
        }, "Success"
    
    # For more complex cases with acceleration
    if abs(recent_accel) < 1e-10:  # Treat as linear case
        if recent_speed >= 0:
            return None, "Speed is not negative, won't reach zero"
        
        days_to_zero = -recent_change / recent_speed
        last_date = df_adj_sorted['date_dt'].iloc[-1]
        zero_date = last_date + pd.Timedelta(days=days_to_zero)
        
        return {
            'zero_date': zero_date,
            'method': 'acceleration_linear',
            'recent_change': recent_change,
            'recent_speed': recent_speed,
            'recent_accel': recent_accel
        }, "Success"
    
    # Quadratic case: change + speed*t + 0.5*accel*t^2 = 0
    a = 0.5 * recent_accel
    b = recent_speed
    c = recent_change
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None, "No real solution for quadratic equation"
    
    # Take the positive solution (future time)
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Choose the positive solution that's closest to current time
    valid_solutions = [t for t in [t1, t2] if t > 0]
    
    if not valid_solutions:
        return None, "No positive solution found"
    
    t_days = min(valid_solutions)  # Take the earliest positive solution
    
    last_date = df_adj_sorted['date_dt'].iloc[-1]
    zero_date = last_date + pd.Timedelta(days=t_days)
    
    return {
        'zero_date': zero_date,
        'method': 'acceleration_quadratic',
        'recent_change': recent_change,
        'recent_speed': recent_speed,
        'recent_accel': recent_accel,
        'days_to_zero': t_days
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
    
    # Method 4: Acceleration model
    try:
        accel_result, accel_msg = predict_zero_with_acceleration(df_adj_sorted)
        if accel_result and isinstance(accel_result, dict):
            predictions['acceleration'] = accel_result['zero_date']
            methods_info['acceleration'] = {
                'method': accel_result['method'],
                'description': f"Using speed and acceleration ({accel_result['method']})"
            }
    except Exception:
        pass
    
    return predictions, methods_info