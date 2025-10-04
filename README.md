# ICP Data Analysis App

This application queries the Internet Computer Protocol (ICP) public APIs to gather data and provide analysis of the total supply over time.

## Features

- Query ICP ledger API for total supply data over date ranges
- Date range filtering using YYYY-MM-DD format input
- Data processing with statistical analysis
- Export data to CSV, JSON, or Parquet formats
- Command-line interface with verbose output options

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py --start-date 2025-09-09 --end-date 2025-09-15
```

## Usage Examples

Basic usage:
```bash
python main.py --start-date 2025-09-09 --end-date 2025-09-15
```

With verbose output and JSON format:
```bash
python main.py -s 2025-09-01 -e 2025-09-30 --save-format json --verbose
```

Custom output prefix:
```bash
python main.py --start-date 2025-09-09 --end-date 2025-09-15 --output-prefix icp_supply_sept
```

Run demo:
```bash
python demo.py
```

## Project Structure

```
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── api_client.py      # ICP API client
│   ├── data_processor.py  # Data processing utilities
│   └── date_utils.py      # Date conversion utilities
├── data/                  # Data storage and cache
├── charts/                # Generated charts and visualizations
├── tests/                 # Unit tests
│   └── test_basic.py      # Basic functionality tests
├── main.py               # Main application entry point
├── demo.py               # Demo script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Information

The application uses the ICP Ledger API:

- **Base URL**: `https://ledger-api.internetcomputer.org`
- **Total Supply Series Endpoint**: `/supply/total/series`
- **Parameters**:
  - `step`: Time step in seconds (fixed at 86400 for daily data)
  - `start`: Start timestamp (Unix timestamp)
  - `end`: End timestamp (Unix timestamp)

## Output

The application generates:

1. **Data file** (CSV/JSON/Parquet): Contains timestamp, date, and total supply data
2. **Metrics file** (JSON): Contains statistical analysis including:
   - Supply statistics (min, max, mean, standard deviation)
   - Supply changes (total change, average daily change, volatility)
   - Date range information

## Testing

Run unit tests:
```bash
python -m unittest tests.test_basic -v
```

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - requests
  - pandas
  - matplotlib
  - seaborn
  - plotly
  - python-dateutil
  - pytest