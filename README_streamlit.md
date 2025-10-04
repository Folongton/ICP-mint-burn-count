# ICP Supply Analysis Streamlit App

This Streamlit application provides interactive analysis of ICP supply change data with trend analysis and zero crossing predictions.

## Features

### 📊 Data Overview
- Key metrics display (total records, current supply change, overall slope, R² value)
- Data range information
- Recent data table (last 10 days)

### 📈 Interactive Trends
- Multi-timeframe trend analysis with dropdown selector:
  - Monthly trends
  - Quarterly trends  
  - Yearly trends
  - Full dataset trend
- Color-coded slope visualization (red = positive, green = negative)
- Interactive Plotly charts with hover information

### 🎯 Zero Crossing Predictions
- Ensemble prediction using multiple methods:
  - Linear extrapolation from overall trend
  - Recent quarterly trend projection
  - Moving average trend analysis
- Confidence assessment based on prediction spread
- Detailed prediction table with dates and statistics

### ⚡ Speed of Change Analysis
- First derivative analysis showing rate of change patterns
- Multiple smoothing levels (daily, 7-day avg, 30-day avg)
- Speed statistics and extremes identification

## Installation

1. Install required packages:
```bash
pip install -r requirements_streamlit.txt
```

2. Ensure your data file exists:
```
output_data/icp_supply_data_2023-10-01_to_2025-10-03_20251003_211226.csv
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## File Structure

```
├── app.py                          # Main Streamlit application
├── src/
│   ├── streamlit_utils.py         # Chart creation and data processing utilities
│   └── prediction_models.py       # Zero crossing prediction algorithms
├── requirements_streamlit.txt      # Python dependencies
└── README_streamlit.md            # This file
```

## Usage

1. **Navigation**: Use the sidebar to select different analysis types
2. **Interactive Charts**: 
   - Use dropdown menus in trend charts to switch between timeframes
   - Hover over data points for detailed information
3. **Predictions**: View ensemble predictions with confidence assessments
4. **Speed Analysis**: Analyze rate of change patterns

## Data Requirements

The app expects a CSV file with the following columns:
- `date`: Date in YYYY-MM-DD format
- `supply_change`: Daily supply change values
- `supply_change_pct`: Supply change percentage
- `total_supply`: Total supply values

## Notes

- Predictions are based on trend extrapolation and should be interpreted with caution
- Low R² values indicate high uncertainty in trend-based predictions
- External factors (policy changes, market events) may significantly alter actual outcomes
- The app automatically handles missing data and calculates rolling averages