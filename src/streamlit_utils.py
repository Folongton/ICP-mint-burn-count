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

def calculate_all_trends(df_adj, valid_data, slope, intercept, r_value):
    """Calculate trends for different time periods"""
    
    # Monthly trends
    df_adj['year_month'] = df_adj['date_dt'].dt.to_period('M')
    monthly_trends = {}
    
    for period in df_adj['year_month'].unique():
        if pd.isna(period):
            continue
        
        month_data = df_adj[df_adj['year_month'] == period].copy()
        if len(month_data) < 3:
            continue
        
        month_valid = month_data.dropna(subset=['supply_change', 'date_numeric'])
        if len(month_valid) < 3:
            continue
        
        slope_m, intercept_m, r_value_m, p_value_m, std_err_m = stats.linregress(
            month_valid['date_numeric'], month_valid['supply_change']
        )
        
        trend_line_m = slope_m * month_valid['date_numeric'] + intercept_m
        
        monthly_trends[period] = {
            'x': month_valid['date_dt'],
            'y': trend_line_m,
            'slope': slope_m,
            'r_squared': r_value_m**2
        }
    
    # Quarterly trends (passed in as quarterly_trend_lines parameter)
    # Yearly trends
    df_adj['year'] = df_adj['date_dt'].dt.to_period('Y')
    yearly_trends = {}
    
    for period in df_adj['year'].unique():
        if pd.isna(period):
            continue
        
        year_data = df_adj[df_adj['year'] == period].copy()
        if len(year_data) < 30:
            continue
        
        year_valid = year_data.dropna(subset=['supply_change', 'date_numeric'])
        if len(year_valid) < 30:
            continue
        
        slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(
            year_valid['date_numeric'], year_valid['supply_change']
        )
        
        trend_line_y = slope_y * year_valid['date_numeric'] + intercept_y
        
        yearly_trends[period] = {
            'x': year_valid['date_dt'],
            'y': trend_line_y,
            'slope': slope_y,
            'r_squared': r_value_y**2
        }
    
    # Full dataset trend
    trend_line = slope * valid_data['date_numeric'] + intercept
    full_trend = {
        'overall': {
            'x': valid_data['date_dt'],
            'y': trend_line,
            'slope': slope,
            'r_squared': r_value**2
        }
    }
    
    return monthly_trends, yearly_trends, full_trend

def create_interactive_trends_chart(df_adj_sorted, quarterly_trends, valid_data, slope, intercept, r_value):
    """Create the interactive multi-trend chart with dropdown selector"""
    
    # Calculate all trend types
    monthly_trends, yearly_trends, full_trend = calculate_all_trends(
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

    # Add monthly trend lines
    monthly_traces = []
    for i, (period, trend_data) in enumerate(monthly_trends.items()):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=False,  # Initially hidden
            hovertemplate=f'<b>{period} Trend</b><br>Date: %{{x}}<br>Supply Change: %{{y:.0f}}<br>Slope: {trend_data["slope"]:.2e}<extra></extra>'
        ))
        monthly_traces.append(trace_count)
        trace_count += 1

    # Add quarterly trend lines
    quarterly_traces = []
    for i, (period, trend_data) in enumerate(quarterly_trends.items()):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=True,  # Initially visible (default)
            hovertemplate=f'<b>{period} Trend</b><br>Date: %{{x}}<br>Supply Change: %{{y:.0f}}<br>Slope: {trend_data["slope"]:.2e}<extra></extra>'
        ))
        quarterly_traces.append(trace_count)
        trace_count += 1

    # Add yearly trend lines
    yearly_traces = []
    for i, (period, trend_data) in enumerate(yearly_trends.items()):
        slope_color = '#FF4444' if trend_data['slope'] > 0 else '#00C853'  # More vibrant red/green
        
        fig.add_trace(go.Scatter(
            x=trend_data['x'], 
            y=trend_data['y'],
            mode='lines',
            name=f'{period} Trend',
            line=dict(color=slope_color, width=2),
            visible=False,  # Initially hidden
            hovertemplate=f'<b>{period} Trend</b><br>Date: %{{x}}<br>Supply Change: %{{y:.0f}}<br>Slope: {trend_data["slope"]:.2e}<extra></extra>'
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

    # Set x-axis to show monthly ticks
    fig.update_xaxes(
        dtick="M1",
        tickformat="%Y-%m-%d",
        tickangle=45
    )

    # Add instructions annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text="Use dropdown menu above to switch between trend periods<br>Red = Positive slopes, Green = Negative slopes",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, color="black")
    )
    
    return fig

def create_ensemble_predictions(df_adj_sorted, predictions, methods_info):
    """Create ensemble predictions visualization"""
    
    # Sort predictions by date
    sorted_predictions = sorted([(method, date) for method, date in predictions.items()], key=lambda x: x[1])
    pred_dates = list(predictions.values())
    
    fig_ensemble = go.Figure()
    
    # Add historical data
    fig_ensemble.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'],
        y=df_adj_sorted['change_30d_avg'],
        mode='lines',
        name='30-Day Average',
        line=dict(color='#5B9BD5', width=2)  # Softer blue to match interactive trends
    ))
    
    # Add prediction points
    colors = ['#FF4444', '#FFB74D', '#00C853', '#BA68C8', '#8D6E63']  # More vibrant red, softer orange, more vibrant green, softer purple, softer brown
    for i, (method, pred_date) in enumerate(sorted_predictions):
        method_display = method.replace("_", " ").title()
        
        # Create detailed hover text with proper formatting
        hover_text = f"<b>{method_display} Method</b><br>"
        hover_text += f"Predicted Date: <b>{pred_date.strftime('%B %d, %Y')}</b><br>"
        hover_text += f"Days from now: <b>{(pred_date - datetime.now()).days}</b><br>"
        
        # Add method-specific information if available
        if method in methods_info:
            method_info = methods_info[method]
            if 'description' in method_info:
                hover_text += f"Method: {method_info['description']}<br>"
            if 'r_squared' in method_info:
                hover_text += f"R-squared = {method_info['r_squared']:.3f}<br>"
            if 'slope' in method_info:
                hover_text += f"Slope: {method_info['slope']:.2e}"
        
        fig_ensemble.add_trace(go.Scatter(
            x=[pred_date],
            y=[0],
            mode='markers',
            name=f'{method_display} Prediction',
            marker=dict(color=colors[i % len(colors)], size=15, symbol='star'),  # Slightly larger markers
            hovertemplate=hover_text + '<extra></extra>'  # Custom hover template only
        ))
    
    # Add prediction range
    min_date = min(pred_dates)
    max_date = max(pred_dates)
    fig_ensemble.add_vrect(
        x0=min_date, x1=max_date,
        fillcolor="rgba(255,0,0,0.1)",
        layer="below",
        line_width=0,
        annotation_text="Prediction Range"
    )
    
    # Add zero line
    fig_ensemble.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
    
    # Update layout
    avg_date = pd.to_datetime(np.mean([d.timestamp() for d in pred_dates]), unit='s')
    spread_years = (max(pred_dates) - min(pred_dates)).days / 365.25
    
    fig_ensemble.update_layout(
        title=f'ICP Supply Change - Ensemble Zero Crossing Predictions<br>Range: {min_date.strftime("%Y-%m-%d")} to {max_date.strftime("%Y-%m-%d")} (Avg: {avg_date.strftime("%Y-%m-%d")})',
        xaxis_title='Date',
        yaxis_title='Supply Change',
        height=700,
        hovermode='closest',  # Changed from 'x unified' to allow individual hover boxes
        template='plotly_white',
        # Ensure hover boxes have enough space
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    fig_ensemble.update_xaxes(
        dtick="M6",  # Semi-annual ticks
        tickformat="%Y-%m-%d",
        tickangle=45
    )
    
    # Add ensemble info annotation
    ensemble_info = f"Ensemble Results ({len(predictions)} methods):<br>"
    ensemble_info += f"Spread: {spread_years:.1f} years<br>"
    ensemble_info += f"Average: {avg_date.strftime('%Y-%m-%d')}"
    
    fig_ensemble.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=ensemble_info,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="blue",
        borderwidth=2,
        font=dict(size=12, color="black")
    )
    
    return fig_ensemble