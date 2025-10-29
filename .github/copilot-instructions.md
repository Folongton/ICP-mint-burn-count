# ICP Supply Analysis Codebase Instructions

## Architecture Overview

This is a dual-interface Python application for analyzing Internet Computer Protocol (ICP) token supply data:

- **CLI interface** (`main.py`): Data fetching, processing, and export to CSV/JSON/Parquet
- **Streamlit web dashboard** (`app.py`): Interactive visualization with real-time data refresh and predictive modeling

### Core Components

**Data Flow**: `ICPLedgerClient` → `ICPDataProcessor` → Analysis/Visualization
- `src/api_client.py`: Handles API requests to `https://ledger-api.internetcomputer.org/supply/total/series`
- `src/data_processor.py`: Processes API responses (handles both list and dict formats), calculates supply changes
- `src/data_refresh.py`: Manages automatic data updates in Streamlit (checks staleness, merges with existing data)

**Analysis Stack**: Statistical analysis using scipy, predictive modeling with multiple methods
- `src/prediction_models.py`: Zero-crossing predictions using linear regression, quarterly trends, moving averages
- `src/streamlit_utils.py`: Plotly chart generation with quarterly trend overlays

## Development Patterns

### Data Processing Convention
```python
# API responses handled in two formats:
# 1. Direct list: [[timestamp, value], ...]
# 2. Dictionary: {'series': [{'timestamp': x, 'value': y}, ...]}

# Supply values converted from e8s (divide by 100_000_000)
df['total_supply'] = df['total_supply'] / 100_000_000
```

### Error Handling Pattern
All modules use try/except with graceful degradation - CLI prints errors and continues, Streamlit shows error messages but maintains functionality.

### File Naming Convention
Output files use timestamp pattern: `icp_supply_data_{start_date}_to_{end_date}_{YYYYMMDD_HHMMSS}.{ext}`

## Key Workflows

### Running the Application
```bash
# CLI data fetching
python main.py --start-date 2025-09-09 --end-date 2025-09-15 --save-format csv --verbose

# Streamlit dashboard 
streamlit run app.py
```

### Testing Strategy
- Unit tests in `tests/test_basic.py` focus on date utilities and data processing
- Run with: `python -m unittest tests.test_basic -v`
- Mock API responses use specific timestamp/value format for consistency

### Data Refresh Logic (Streamlit Only)
- Automatic refresh if data is >25 hours old
- Incremental updates: fetch only missing dates, merge with existing CSV
- Uses `@st.cache_data` for performance with cache clearing on refresh

## Integration Points

### API Dependencies
- **Primary**: ICP Ledger API (no auth required, rate limits unknown)
- **Fallback**: Application continues with existing data if API fails

### Statistical Analysis
- Uses `scipy.stats.linregress` for trend analysis across multiple timeframes (overall, quarterly, monthly)
- Ensemble prediction methods combine linear regression, recent trends, and moving averages
- R² values track prediction confidence (>0.1 considered "high confidence")

### External Data Format
API returns Unix timestamps with daily step (86400 seconds). All datetime handling uses UTC.

## Streamlit-Specific Patterns

### Page Structure
Multi-tab interface with analysis types: Data Overview, Interactive Trends, Zero Crossing Predictions, Speed of Change

### Chart Generation
Plotly charts with consistent styling via `template='plotly_white'`, interactive hover using `hovermode='x unified'`

### Performance Optimization
- `@st.cache_data` on main data loading function
- Quarterly trend calculations cached in data structure
- Background data refresh with user feedback via status messages