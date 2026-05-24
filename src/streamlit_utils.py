import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

def load_data(csv_path):
    """Load and preprocess ICP supply data"""
    df = pd.read_csv(csv_path)
    
    # Process data
    df_adj = df.dropna(subset=['supply_change', 'supply_change_pct']).copy()
    df_adj['total_supply'] = df_adj['total_supply'] / 100_000_000
    df_adj['supply_change'] = df_adj['supply_change'] / 100_000_000
    
    # Add rolling averages
    df_adj['change_7d_avg'] = df_adj['supply_change'].rolling(window=7).mean()
    df_adj['change_30d_avg'] = df_adj['supply_change'].rolling(window=30).mean()
    
    # Convert dates
    df_adj['date_dt'] = pd.to_datetime(df_adj['date'])
    df_adj['date_numeric'] = df_adj['date_dt'].map(pd.Timestamp.timestamp)
    
    return df_adj

def calculate_all_trends(df_adj_sorted, valid_data, slope, intercept, r_value):
    """
    Calculate trends for different time periods using rolling windows.
    
    Windows roll backwards from the last date in the dataset:
    - Monthly: 30-day windows (M1, M2, M3, ...)
    - Quarterly: 91-day windows (Q1, Q2, Q3, ...)
    - Yearly: 365-day windows (Y1, Y2, Y3, ...)
    - Overall: Full dataset
    """
    
    # Get the last date in the dataset
    last_date = df_adj_sorted['date_dt'].max()
    
    # Monthly trends - 30-day rolling windows
    num_months = int((df_adj_sorted['date_dt'].max() - df_adj_sorted['date_dt'].min()).days / 30) + 1
    monthly_trends = {}
    
    for i in range(min(num_months, 24)):  # Limit to 24 months (2 years)
        end_date = last_date - timedelta(days=i * 30)
        start_date = end_date - timedelta(days=30)
        
        month_data = df_adj_sorted[
            (df_adj_sorted['date_dt'] >= start_date) & 
            (df_adj_sorted['date_dt'] <= end_date)
        ].copy()
        
        month_valid = month_data.dropna(subset=['supply_change', 'date_numeric'])
        
        # Require minimum 3 data points
        if len(month_valid) < 3:
            continue
        
        slope_m, intercept_m, r_value_m, p_value_m, std_err_m = stats.linregress(
            month_valid['date_numeric'], month_valid['supply_change']
        )
        
        trend_line_m = slope_m * month_valid['date_numeric'] + intercept_m
        
        period_label = f"M{i+1}"
        monthly_trends[period_label] = {
            'x': month_valid['date_dt'],
            'y': trend_line_m,
            'slope': slope_m,
            'r_squared': r_value_m**2,
            'start_date': start_date,
            'end_date': end_date,
            'period_index': i
        }
    
    # Quarterly trends - 91-day rolling windows
    num_quarters = int((df_adj_sorted['date_dt'].max() - df_adj_sorted['date_dt'].min()).days / 91) + 1
    quarterly_trends = {}
    
    for i in range(min(num_quarters, 12)):  # Limit to 12 quarters (~3 years)
        end_date = last_date - timedelta(days=i * 91)
        start_date = end_date - timedelta(days=91)
        
        quarter_data = df_adj_sorted[
            (df_adj_sorted['date_dt'] >= start_date) & 
            (df_adj_sorted['date_dt'] <= end_date)
        ].copy()
        
        quarter_valid = quarter_data.dropna(subset=['supply_change', 'date_numeric'])
        
        # Require minimum 10 data points
        if len(quarter_valid) < 10:
            continue
        
        slope_q, intercept_q, r_value_q, p_value_q, std_err_q = stats.linregress(
            quarter_valid['date_numeric'], quarter_valid['supply_change']
        )
        
        trend_line_q = slope_q * quarter_valid['date_numeric'] + intercept_q
        
        period_label = f"Q{i+1}"
        quarterly_trends[period_label] = {
            'x': quarter_valid['date_dt'],
            'y': trend_line_q,
            'slope': slope_q,
            'r_squared': r_value_q**2,
            'start_date': start_date,
            'end_date': end_date,
            'period_index': i
        }
    
    # Yearly trends - 365-day rolling windows
    num_years = int((df_adj_sorted['date_dt'].max() - df_adj_sorted['date_dt'].min()).days / 365) + 1
    yearly_trends = {}
    
    for i in range(min(num_years, 5)):  # Limit to 5 years
        end_date = last_date - timedelta(days=i * 365)
        start_date = end_date - timedelta(days=365)
        
        year_data = df_adj_sorted[
            (df_adj_sorted['date_dt'] >= start_date) & 
            (df_adj_sorted['date_dt'] <= end_date)
        ].copy()
        
        year_valid = year_data.dropna(subset=['supply_change', 'date_numeric'])
        
        # Require minimum 30 data points
        if len(year_valid) < 30:
            continue
        
        slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(
            year_valid['date_numeric'], year_valid['supply_change']
        )
        
        trend_line_y = slope_y * year_valid['date_numeric'] + intercept_y
        
        period_label = f"Y{i+1}"
        yearly_trends[period_label] = {
            'x': year_valid['date_dt'],
            'y': trend_line_y,
            'slope': slope_y,
            'r_squared': r_value_y**2,
            'start_date': start_date,
            'end_date': end_date,
            'period_index': i
        }
    
    # Full dataset trend
    trend_line = slope * valid_data['date_numeric'] + intercept
    full_trend = {
        'Overall': {
            'x': valid_data['date_dt'],
            'y': trend_line,
            'slope': slope,
            'r_squared': r_value**2
        }
    }
    
    return monthly_trends, quarterly_trends, yearly_trends, full_trend

def create_interactive_trends_chart(df_adj_sorted, valid_data, slope, intercept, r_value):
    """Create the interactive multi-trend chart with dropdown selector"""
    
    # Calculate all trend types (monthly, quarterly, yearly, full)
    monthly_trends, quarterly_trends, yearly_trends, full_trend = calculate_all_trends(
        df_adj_sorted, valid_data, slope, intercept, r_value
    )
    
    # Create the main figure
    fig = go.Figure()
    
    # Add base traces (7-day and 30-day averages) - these are always visible
    fig.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'], 
        y=df_adj_sorted['change_7d_avg'],
        mode='lines',
        name='7-Day Avg',
        line=dict(color='#FFB74D', width=2),  # Softer orange
        hovertemplate='<b>7-Day Avg</b><br>Date: %{x}<br>Supply Change: %{y:.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'], 
        y=df_adj_sorted['change_30d_avg'],
        mode='lines',
        name='30-Day Avg',
        line=dict(color='#5B9BD5', width=2),  # Softer blue
        hovertemplate='<b>30-Day Avg</b><br>Date: %{x}<br>Supply Change: %{y:.0f}<extra></extra>'
    ))

    # Add trend traces (initially hidden except for quarterly)
    trace_count = 2  # Start after the two base traces

    # Add monthly trend lines (reversed so M1 appears at top of legend)
    monthly_traces = []
    for period, trend_data in sorted(monthly_trends.items(), key=lambda x: x[1]['period_index'], reverse=True):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        # Create hover text with period details
        hover_text = (
            f"<b>{period} (30-day window)</b><br>"
            f"Period: {trend_data['start_date'].strftime('%Y-%m-%d')} to {trend_data['end_date'].strftime('%Y-%m-%d')}<br>"
            f"Date: %{{x}}<br>"
            f"Supply Change: %{{y:.0f}}<br>"
            f"Slope: {trend_data['slope']:.2e}<br>"
            f"R²: {trend_data['r_squared']:.3f}"
        )
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=False,  # Initially hidden
            hovertemplate=hover_text + '<extra></extra>'
        ))
        monthly_traces.append(trace_count)
        trace_count += 1

    # Add quarterly trend lines (reversed so Q1 appears at top of legend)
    quarterly_traces = []
    for period, trend_data in sorted(quarterly_trends.items(), key=lambda x: x[1]['period_index'], reverse=True):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        # Create hover text with period details
        hover_text = (
            f"<b>{period} (91-day window)</b><br>"
            f"Period: {trend_data['start_date'].strftime('%Y-%m-%d')} to {trend_data['end_date'].strftime('%Y-%m-%d')}<br>"
            f"Date: %{{x}}<br>"
            f"Supply Change: %{{y:.0f}}<br>"
            f"Slope: {trend_data['slope']:.2e}<br>"
            f"R²: {trend_data['r_squared']:.3f}"
        )
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=True,  # Initially visible (default)
            hovertemplate=hover_text + '<extra></extra>'
        ))
        quarterly_traces.append(trace_count)
        trace_count += 1

    # Add yearly trend lines (reversed so Y1 appears at top of legend)
    yearly_traces = []
    for period, trend_data in sorted(yearly_trends.items(), key=lambda x: x[1]['period_index'], reverse=True):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        # Create hover text with period details
        hover_text = (
            f"<b>{period} (365-day window)</b><br>"
            f"Period: {trend_data['start_date'].strftime('%Y-%m-%d')} to {trend_data['end_date'].strftime('%Y-%m-%d')}<br>"
            f"Date: %{{x}}<br>"
            f"Supply Change: %{{y:.0f}}<br>"
            f"Slope: {trend_data['slope']:.2e}<br>"
            f"R²: {trend_data['r_squared']:.3f}"
        )
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=False,  # Initially hidden
            hovertemplate=hover_text + '<extra></extra>'
        ))
        yearly_traces.append(trace_count)
        trace_count += 1

    # Add full trend line
    full_traces = []
    for period, trend_data in full_trend.items():
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name='Overall Trend',
            line=dict(color='#BA68C8', width=3),  # Softer purple
            visible=False,  # Initially hidden
            hovertemplate=f'<b>Overall Trend</b><br>Date: %{{x}}<br>Supply Change: %{{y:.0f}}<br>Slope: {trend_data["slope"]:.2e}<extra></extra>'
        ))
        full_traces.append(trace_count)
        trace_count += 1

    # Create visibility arrays for dropdown
    def create_visibility_array(total_traces, base_traces, trend_traces):
        """Create visibility array for dropdown options"""
        visibility = [False] * total_traces
        # Always show base traces (7-day and 30-day averages)
        visibility[0] = True  # 7-day avg
        visibility[1] = True  # 30-day avg
        # Show selected trend traces
        for trace_idx in trend_traces:
            visibility[trace_idx] = True
        return visibility

    total_traces = len(fig.data)
    base_traces = [0, 1]  # 7-day and 30-day averages

    # Create dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": create_visibility_array(total_traces, base_traces, monthly_traces)}],
                        label="Monthly Trends",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": create_visibility_array(total_traces, base_traces, quarterly_traces)}],
                        label="Quarterly Trends",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": create_visibility_array(total_traces, base_traces, yearly_traces)}],
                        label="Yearly Trends",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": create_visibility_array(total_traces, base_traces, full_traces)}],
                        label="Full Trend",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.09,
                yanchor="top"
            ),
        ]
    )

    # Update layout
    fig.update_layout(
        title='ICP Supply Change Over Time - Interactive Trend Analysis',
        xaxis_title='Date',
        yaxis_title='Supply Change',
        height=700,
        hovermode='x unified',
        template='plotly_white'
    )

    # Set x-axis with initial 6-month zoom; full history remains accessible via pan/zoom
    _last_date = df_adj_sorted['date_dt'].max()
    _six_months_ago = _last_date - timedelta(days=182)
    fig.update_xaxes(
        range=[_six_months_ago, _last_date],
        dtick="M1",
        tickformat="%Y-%m-%d",
        tickangle=45
    )

    # Add instructions annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=(
            "Use dropdown menu above to switch between trend periods<br>"
            "<b>Rolling Windows:</b> M1-M24 (30 days), Q1-Q12 (91 days), Y1-Y5 (365 days)<br>"
            "<b>Colors:</b> Red = Positive slopes (increasing), Green = Negative slopes (decreasing)"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, color="black"),
        align="left"
    )
    
    return fig

def create_ensemble_predictions(df_adj_sorted, predictions, methods_info):
    """Create ensemble predictions visualization with extrapolation lines"""
    
    fig_ensemble = go.Figure()
    
    # Add historical data (30-day average)
    fig_ensemble.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'],
        y=df_adj_sorted['change_30d_avg'],
        mode='lines',
        name='30-Day Average (Historical)',
        line=dict(color='#5B9BD5', width=3),
        hovertemplate='<b>30-Day Avg</b><br>Date: %{x}<br>Supply Change: %{y:.0f}<extra></extra>'
    ))
    
    # Colors for each method (removed acceleration/purple)
    colors = {
        'linear': '#FF4444',           # Red
        'quarterly': '#FFB74D',        # Orange
        'moving_average': '#00C853'    # Green
    }
    
    # Add extrapolation lines and prediction points for ALL methods (including positive slopes)
    for method, method_info in methods_info.items():
        method_display = method.replace("_", " ").title()
        color = colors.get(method, '#8D6E63')
        
        # Check if this method will actually cross zero
        will_cross_zero = method_info.get('will_cross_zero', True)
        pred_date = method_info.get('projection_end_date')  # Use this for all methods
        
        # Get method-specific parameters
        start_date = method_info.get('start_date')
        start_value = method_info.get('start_value')
        slope = method_info.get('slope')
        intercept = method_info.get('intercept')
        
        # Skip if we don't have required data
        if pred_date is None or start_date is None or start_value is None or slope is None or intercept is None:
            print(f"Skipping {method} - missing required data")
            continue
        
        # For non-crossing predictions, find the date where y=100,000
        if not will_cross_zero:
            # Linear: solve slope * timestamp + intercept = 100000
            target_y = 100000
            
            if abs(slope) > 1e-10:
                target_timestamp = (target_y - intercept) / slope
                end_date = pd.to_datetime(target_timestamp, unit='s')
            else:
                end_date = pred_date  # Fallback to 2 years
        else:
            # For crossing predictions, use the actual crossing date
            end_date = pred_date
        
        # Create linear extrapolation line
        extrap_dates = pd.date_range(start=start_date, end=end_date, periods=100)
        extrap_timestamps = extrap_dates.map(pd.Timestamp.timestamp)
        extrap_values = slope * extrap_timestamps + intercept
        
        line_style = 'dash'
        line_name = f'{method_display} Extrapolation'
        
        # Add extrapolation line
        fig_ensemble.add_trace(go.Scatter(
            x=extrap_dates,
            y=extrap_values,
            mode='lines',
            name=line_name,
            line=dict(color=color, width=2.5, dash=line_style),
            hovertemplate=f'<b>{method_display} Projection</b><br>Date: %{{x}}<br>Supply Change: %{{y:.0f}}<extra></extra>',
            showlegend=True
        ))
        
        # Create detailed hover text for the marker
        hover_text = f"<b>{method_display}</b><br>"
        if will_cross_zero:
            hover_text += f"<b>Zero Crossing</b><br>"
            hover_text += f"Predicted Date: <b>{pred_date.strftime('%B %d, %Y')}</b><br>"
            hover_text += f"Days from now: <b>{(pred_date - datetime.now()).days}</b><br>"
        else:
            hover_text += f"<b>Trend Projection (No Zero Cross)</b><br>"
            hover_text += f"Direction: <b>Increasing</b><br>"
            hover_text += f"Slope: <b>Positive</b><br>"
        
        if 'description' in method_info:
            hover_text += f"Method: {method_info['description']}<br>"
        if 'r_squared' in method_info:
            hover_text += f"R² = {method_info['r_squared']:.3f}<br>"
        if 'slope' in method_info:
            hover_text += f"Slope: {method_info['slope']:.2e}"
        
        # Add marker at endpoint (star if crosses zero, circle if doesn't)
        marker_symbol = 'star' if will_cross_zero else 'circle'
        marker_size = 18 if will_cross_zero else 12
        
        # For non-crossing, place marker at end of projection line (should be near 100k)
        # For crossing, place at zero
        marker_y = 0 if will_cross_zero else extrap_values[-1]
        marker_x = pred_date if will_cross_zero else end_date
        
        fig_ensemble.add_trace(go.Scatter(
            x=[marker_x],
            y=[marker_y],
            mode='markers',
            name=f'{method_display} {"Crossing" if will_cross_zero else "Endpoint"}',
            marker=dict(color=color, size=marker_size, symbol=marker_symbol, 
                       line=dict(color='white', width=2)),
            hovertemplate=hover_text + '<extra></extra>',
            showlegend=True
        ))
    
    # Calculate crossing dates and all projection dates from methods_info
    crossing_dates = []
    all_projection_dates = []
    
    for method, method_info in methods_info.items():
        proj_date = method_info.get('projection_end_date')
        if proj_date:
            all_projection_dates.append(proj_date)
            if method_info.get('will_cross_zero', True):
                crossing_dates.append(proj_date)
    
    # Add prediction range shading - only for methods that will cross zero
    if len(crossing_dates) > 1:
        min_date = min(crossing_dates)
        max_date = max(crossing_dates)
        fig_ensemble.add_vrect(
            x0=min_date, x1=max_date,
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0,
            annotation_text="Prediction Range",
            annotation_position="bottom left"
        )
    
    # Add zero reference line
    fig_ensemble.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.7, 
                           annotation_text="Zero Target", annotation_position="right")
    
    # Calculate average only from methods that will cross zero
    if len(crossing_dates) > 0:
        avg_date = pd.to_datetime(np.mean([d.timestamp() for d in crossing_dates]), unit='s')
        spread_years = (max(crossing_dates) - min(crossing_dates)).days / 365.25 if len(crossing_dates) > 1 else 0
        avg_text = f"Avg: {avg_date.strftime('%Y-%m-%d')}"
        spread_text = f"Spread: {spread_years:.2f} years<br>"
        range_text = f"Prediction Range: {min(crossing_dates).strftime('%Y-%m-%d')} to {max(crossing_dates).strftime('%Y-%m-%d')}"
    else:
        avg_text = "No zero crossings predicted"
        spread_text = ""
        range_text = "All methods show increasing trends"
    
    fig_ensemble.update_layout(
        title=f'ICP Supply Change - Zero Crossing Predictions with Extrapolation<br><sub>{range_text} ({avg_text})</sub>',
        xaxis_title='Date',
        yaxis_title='Supply Change (ICP)',
        height=700,
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=50, r=50, t=120, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="blue",
            borderwidth=2,
            font=dict(color="black")
        )
    )
    
    # Set x-axis range to include predictions with some padding
    last_historical_date = df_adj_sorted['date_dt'].max()
    if all_projection_dates:
        furthest_prediction = max(all_projection_dates)
        padding_days = (furthest_prediction - last_historical_date).days * 0.1
    else:
        furthest_prediction = last_historical_date + timedelta(days=365)
        padding_days = 30
    
    fig_ensemble.update_xaxes(
        range=[last_historical_date - timedelta(days=180), furthest_prediction + timedelta(days=padding_days)],
        dtick="M3",  # Quarterly ticks
        tickformat="%Y-%m",
        tickangle=45
    )
    
    # Add ensemble info annotation
    num_crossing = len(crossing_dates)
    total_methods = len(methods_info)
    num_non_crossing = total_methods - num_crossing
    
    ensemble_info = f"<b>Ensemble Results ({total_methods} methods)</b><br>"
    if num_crossing > 0:
        ensemble_info += spread_text
        ensemble_info += f"Average: {avg_text}<br>"
        ensemble_info += f"Zero crossings: {num_crossing}<br>"
    if num_non_crossing > 0:
        ensemble_info += f"Non-crossing (increasing): {num_non_crossing}<br>"
    ensemble_info += f"<i>All projections are linear</i>"
    
    fig_ensemble.add_annotation(
        x=0.98, y=0.5,
        xref='paper', yref='paper',
        text=ensemble_info,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="blue",
        borderwidth=2,
        font=dict(size=11, color="black"),
        align="left",
        xanchor="right",
        yanchor="middle"
    )
    
    return fig_ensemble