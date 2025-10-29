import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from scipy import stats

# Import our custom modules
from src.streamlit_utils import load_data, create_interactive_trends_chart, create_ensemble_predictions
from src.prediction_models import (
    predict_zero_crossing_linear,
    predict_zero_from_recent_trend,
    predict_zero_from_moving_average,
    ensemble_zero_prediction
)
from src.data_refresh import get_fresh_data

# Page configuration
st.set_page_config(
    page_title="ICP Supply Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.success-box {
    background-color: #d1ecf1;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #17a2b8;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the ICP supply data with automatic refresh"""
    try:
        # Check for fresh data and refresh if needed
        csv_file, was_refreshed = get_fresh_data()
        
        if csv_file is None:
            st.error("❌ No data available. Please check your connection and try again.")
            return None, None, None, None, None
        
        if was_refreshed:
            st.success("🔄 Data successfully updated with latest information!")
            # Clear cache to force reload with new data
            st.cache_data.clear()
        
        if not os.path.exists(csv_file):
            st.error(f"Data file not found: {csv_file}")
            return None, None, None, None, None
        
        df = pd.read_csv(csv_file)
        
        # Process data
        df_adj = df.dropna(subset=['supply_change', 'supply_change_pct']).copy()
        df_adj['total_supply'] = df_adj['total_supply'] / 100_000_000
        df_adj['supply_change'] = df_adj['supply_change'] / 100_000_000
        
        # Add rolling averages
        df_adj['change_7d_avg'] = df_adj['supply_change'].rolling(window=7).mean()
        df_adj['change_30d_avg'] = df_adj['supply_change'].rolling(window=30).mean()
        
        # Convert dates and calculate numerical values for regression
        df_adj['date_dt'] = pd.to_datetime(df_adj['date'])
        df_adj['date_numeric'] = df_adj['date_dt'].map(pd.Timestamp.timestamp)
        
        # Calculate overall trend
        valid_data = df_adj.dropna(subset=['supply_change', 'date_numeric'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data['date_numeric'], valid_data['supply_change'])
        trend_line = slope * valid_data['date_numeric'] + intercept
        
        # Sort data for derivative calculations
        df_adj_sorted = df_adj.sort_values('date_dt').copy()
        
        return df_adj_sorted, valid_data, slope, intercept, r_value
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 ICP Supply Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data_load_state = st.text('Loading data...')
    df_adj_sorted, valid_data, slope, intercept, r_value = load_and_process_data()
    
    if df_adj_sorted is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    data_load_state.text('Data loaded successfully!')
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    
    # Analysis selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["📈 Interactive Trends", "🎯 Zero Crossing Predictions", "📊 Data Overview", "⚡ Speed of Change"]
    )
    
    # Main content area
    if analysis_type == "📊 Data Overview":
        show_data_overview(df_adj_sorted, slope, r_value)
    
    elif analysis_type == "📈 Interactive Trends":
        show_interactive_trends(df_adj_sorted, valid_data, slope, intercept, r_value)
    
    elif analysis_type == "🎯 Zero Crossing Predictions":
        show_ensemble_predictions(df_adj_sorted, slope, intercept, valid_data, r_value)
    
    elif analysis_type == "⚡ Speed of Change":
        show_speed_of_change(df_adj_sorted)

def show_data_overview(df_adj_sorted, slope, r_value):
    """Display data overview and basic statistics"""
    st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df_adj_sorted):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_supply = df_adj_sorted['supply_change'].iloc[-1]
        st.metric("Current Supply Change", f"{current_supply:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Overall Slope", f"{slope:.2e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Value", f"{r_value**2:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Date range
    st.subheader("📅 Data Range")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Start Date:** {df_adj_sorted['date_dt'].min().strftime('%Y-%m-%d')}")
    with col2:
        st.write(f"**End Date:** {df_adj_sorted['date_dt'].max().strftime('%Y-%m-%d')}")
    
    # Recent data table
    st.subheader("📋 Recent Data (Last 10 Days)")
    recent_data = df_adj_sorted[['date', 'supply_change', 'change_7d_avg', 'change_30d_avg']].tail(10)
    recent_data['supply_change'] = recent_data['supply_change'].apply(lambda x: f"{x:,.0f}")
    recent_data['change_7d_avg'] = recent_data['change_7d_avg'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    recent_data['change_30d_avg'] = recent_data['change_30d_avg'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    st.dataframe(recent_data, use_container_width=True)

def show_interactive_trends(df_adj_sorted, valid_data, slope, intercept, r_value):
    """Display the interactive trends analysis"""
    st.markdown('<h2 class="sub-header">📈 Interactive Trend Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Rolling Window Analysis:**
    - **Monthly (M1-M24)**: 30-day windows for short-term trends
    - **Quarterly (Q1-Q12)**: 91-day windows for medium-term trends (default)
    - **Yearly (Y1-Y5)**: 365-day windows for long-term trends
    - **Overall**: Full dataset trend
    
    *Note: M1/Q1/Y1 represent the most recent period, rolling backwards from the last data point.*
    """)
    
    # Create and display the interactive chart
    fig = create_interactive_trends_chart(df_adj_sorted, valid_data, slope, intercept, r_value)
    st.plotly_chart(fig, use_container_width=True)

def show_ensemble_predictions(df_adj_sorted, slope, intercept, valid_data, r_value):
    """Display ensemble prediction analysis"""
    st.markdown('<h2 class="sub-header">🎯 Zero Crossing Predictions</h2>', unsafe_allow_html=True)
    
    # Add explanation
    with st.expander("ℹ️ How Predictions Work", expanded=False):
        st.markdown("""
        ### Prediction Methods:
        
        **1. Linear (Overall Trend)** 🔴
        - Uses the entire dataset's linear regression
        - Most stable but may ignore recent changes
        - Good for long-term baseline
        - Formula: `y = slope × t + intercept`, solve for `y = 0`
        
        **2. Quarterly (Recent 91-Day Trend)** 🟠
        - Based on the most recent 91-day rolling window (Q1)
        - Captures current market momentum
        - More responsive to recent changes
        - *Shows projection even if slope is positive (won't cross zero)*
        - Uses last 91 days to calculate trend
        
        **3. Moving Average (30-Day MA Trend)** 🟢
        - Fits trend to the last 60 days of smoothed data
        - Balances responsiveness and stability
        - Filters out daily noise
        - Reduces impact of outliers
        
        ### Chart Elements:
        - **Solid Blue Line**: Historical 30-day moving average
        - **Dashed Colored Lines**: Linear extrapolations for each method
        - **Stars ⭐**: Predicted zero crossing points
        - **Circles ●**: Projection endpoints at 100k (when trend won't cross zero)
        - **Shaded Area**: Range of uncertainty between earliest and latest predictions
        
        ### Confidence Levels:
        - **High**: All methods agree within 6 months
        - **Moderate**: Methods agree within 1 year
        - **Low**: Spread of 1-2 years
        - **Very Low**: Disagreement >2 years
        """)
    
    # Run ensemble prediction
    with st.spinner('Calculating predictions...'):
        predictions, methods_info = ensemble_zero_prediction(
            df_adj_sorted, slope, intercept, valid_data
        )
    
    # Check if any methods were excluded due to positive slopes
    excluded_methods = [method for method, info in methods_info.items() 
                       if method not in predictions and not info.get('will_cross_zero', True)]
    
    # Display results
    if predictions:
        st.success(f"✅ Generated {len(predictions)} valid predictions")
        if excluded_methods:
            excluded_names = [m.replace('_', ' ').title() for m in excluded_methods]
            st.warning(f"⚠️ Excluded from average: {', '.join(excluded_names)} (positive slope - won't reach zero)")
    else:
        st.warning("⚠️ No zero-crossing predictions available")
        if excluded_methods:
            excluded_names = [m.replace('_', ' ').title() for m in excluded_methods]
            st.info(f"ℹ️ All methods show positive slopes (uptrend): {', '.join(excluded_names)}")
        else:
            st.error("❌ No valid predictions could be generated from any method")
            return
    
    # Create ensemble visualization (show all methods, even with positive slopes)
    fig_ensemble = create_ensemble_predictions(df_adj_sorted, predictions, methods_info)
    st.plotly_chart(fig_ensemble, use_container_width=True)
    
    # Only show prediction details if we have valid predictions
    if predictions:
        # Prediction details
        st.subheader("📋 Prediction Details")
        
        sorted_predictions = sorted([(method, date) for method, date in predictions.items()], key=lambda x: x[1])
        
        prediction_data = []
        for i, (method, pred_date) in enumerate(sorted_predictions):
            days_from_now = (pred_date - pd.Timestamp.now()).days
            years_from_now = days_from_now / 365.25
            method_info = methods_info.get(method, {})
            
            prediction_data.append({
                'Method': method.replace('_', ' ').title(),
                'Predicted Date': pred_date.strftime('%Y-%m-%d'),
                'Days from Now': f"{days_from_now:,}",
                'Years from Now': f"{years_from_now:.1f}",
                'Slope': f"{method_info.get('slope', 'N/A'):.2e}" if 'slope' in method_info else 'N/A',
                'R²': f"{method_info.get('r_squared', 'N/A'):.3f}" if 'r_squared' in method_info else 'N/A'
            })
        
        st.dataframe(pd.DataFrame(prediction_data), use_container_width=True)
        
        # Confidence assessment
        pred_dates = list(predictions.values())
        spread_years = (max(pred_dates) - min(pred_dates)).days / 365.25
        
        st.subheader("🎯 Confidence Assessment")
        
        if spread_years < 0.5:
            st.markdown('<div class="success-box">🟢 <strong>HIGH CONFIDENCE:</strong> All methods agree within 6 months</div>', unsafe_allow_html=True)
        elif spread_years < 1.0:
            st.markdown('<div class="success-box">🟡 <strong>MODERATE CONFIDENCE:</strong> Methods agree within 1 year</div>', unsafe_allow_html=True)
        elif spread_years < 2.0:
            st.markdown('<div class="warning-box">🟠 <strong>LOW CONFIDENCE:</strong> Methods spread across 1-2 years</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">🔴 <strong>VERY LOW CONFIDENCE:</strong> Methods disagree by >2 years</div>', unsafe_allow_html=True)
        
        # Statistics
        avg_date = pd.to_datetime(np.mean([d.timestamp() for d in pred_dates]), unit='s')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Earliest Prediction", min(pred_dates).strftime('%Y-%m-%d'))
        with col2:
            st.metric("Latest Prediction", max(pred_dates).strftime('%Y-%m-%d'))
        with col3:
            st.metric("Average Prediction", avg_date.strftime('%Y-%m-%d'))

def show_speed_of_change(df_adj_sorted):
    """Display speed of change analysis"""
    st.markdown('<h2 class="sub-header">⚡ Speed of Change Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate derivatives
    df_adj_sorted['supply_change_derivative'] = np.gradient(df_adj_sorted['supply_change'])
    df_adj_sorted['derivative_7d_avg'] = df_adj_sorted['supply_change_derivative'].rolling(window=7).mean()
    df_adj_sorted['derivative_30d_avg'] = df_adj_sorted['supply_change_derivative'].rolling(window=30).mean()
    
    # Create speed of change chart
    fig_derivative = go.Figure()
    
    # Add traces
    fig_derivative.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'], 
        y=df_adj_sorted['supply_change_derivative'],
        mode='lines',
        name='Daily Speed of Change',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    fig_derivative.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'], 
        y=df_adj_sorted['derivative_7d_avg'],
        mode='lines',
        name='7-Day Avg Speed',
        line=dict(color='orange', width=2)
    ))
    
    fig_derivative.add_trace(go.Scatter(
        x=df_adj_sorted['date_dt'], 
        y=df_adj_sorted['derivative_30d_avg'],
        mode='lines',
        name='30-Day Avg Speed',
        line=dict(color='blue', width=2)
    ))
    
    # Add zero reference line
    fig_derivative.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # Update layout
    fig_derivative.update_layout(
        title='ICP Supply Change - Speed of Change Analysis (1st Derivative)',
        xaxis_title='Date',
        yaxis_title='Speed of Change (Derivative of Supply Change)',
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_derivative, use_container_width=True)
    
    # Speed statistics
    st.subheader("📊 Speed Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Speed", f"{df_adj_sorted['supply_change_derivative'].mean():.2e}")
    with col2:
        st.metric("Std Dev", f"{df_adj_sorted['supply_change_derivative'].std():.2e}")
    with col3:
        st.metric("Max Acceleration", f"{df_adj_sorted['supply_change_derivative'].max():.2e}")
    with col4:
        st.metric("Max Deceleration", f"{df_adj_sorted['supply_change_derivative'].min():.2e}")

if __name__ == "__main__":
    main()