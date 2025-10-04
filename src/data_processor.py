"""
Data processing module for ICP ledger data.
Handles parsing, cleaning, and structuring of API responses.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os

try:
    from .date_utils import timestamp_to_date
except ImportError:
    from date_utils import timestamp_to_date


class ICPDataProcessor:
    """Processes ICP ledger API data for analysis and visualization."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory to store processed data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def process_total_supply_series(self, api_response: Any) -> pd.DataFrame:
        """
        Process total supply series API response into a pandas DataFrame.
        
        Args:
            api_response: Response from get_total_supply_series API call (list or dict)
        
        Returns:
            DataFrame with columns: timestamp, date, total_supply
        
        Raises:
            ValueError: If API response format is invalid
        """
        # Handle both list and dictionary response formats
        if isinstance(api_response, list):
            series_data = api_response
        elif isinstance(api_response, dict) and 'series' in api_response:
            series_data = api_response['series']
        else:
            raise ValueError("API response must be a list or dict with 'series' field")
        
        if not series_data:
            raise ValueError("API response contains empty series data")
        
        # Extract data points
        processed_data = []
        
        for data_point in series_data:
            timestamp = None
            supply_value = None
            
            # Handle different data point formats
            if isinstance(data_point, (list, tuple)) and len(data_point) >= 2:
                # Format: [timestamp, value]
                timestamp = data_point[0]
                supply_value = data_point[1]
            elif isinstance(data_point, dict):
                # Dictionary format - try different possible field names
                for ts_field in ['timestamp', 'time', 't', 'x']:
                    if ts_field in data_point:
                        timestamp = data_point[ts_field]
                        break
                
                for supply_field in ['value', 'supply', 'total_supply', 'amount', 'y']:
                    if supply_field in data_point:
                        supply_value = data_point[supply_field]
                        break
            
            if timestamp is not None and supply_value is not None:
                try:
                    # Convert timestamp to date string
                    date_str = timestamp_to_date(timestamp)
                    
                    processed_data.append({
                        'timestamp': int(timestamp),
                        'date': date_str,
                        'total_supply': float(supply_value)
                    })
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Skipping invalid data point {data_point}: {e}")
                    continue
        
        if not processed_data:
            raise ValueError("No valid data points found in API response")
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp to datetime for better pandas operations
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def calculate_supply_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate key metrics from supply data.
        
        Args:
            df: DataFrame with total supply data
        
        Returns:
            Dictionary containing calculated metrics
        """
        if df.empty:
            return {}
        
        metrics = {
            'total_points': len(df),
            'date_range': {
                'start': df['date'].iloc[0],
                'end': df['date'].iloc[-1]
            },
            'supply_stats': {
                'min': float(df['total_supply'].min()),
                'max': float(df['total_supply'].max()),
                'mean': float(df['total_supply'].mean()),
                'std': float(df['total_supply'].std())
            }
        }
        
        # Calculate supply changes
        if len(df) > 1:
            df['supply_change'] = df['total_supply'].diff()
            df['supply_change_pct'] = df['total_supply'].pct_change() * 100
            
            metrics['supply_changes'] = {
                'total_change': float(df['total_supply'].iloc[-1] - df['total_supply'].iloc[0]),
                'avg_daily_change': float(df['supply_change'].mean()),
                'max_daily_increase': float(df['supply_change'].max()),
                'max_daily_decrease': float(df['supply_change'].min()),
                'volatility': float(df['supply_change'].std())
            }
        
        return metrics
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Filename (without extension)
            format: File format ('csv', 'json', 'parquet')
        
        Returns:
            Full path to saved file
        """
        if format == 'csv':
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = os.path.join(self.data_dir, f"{filename}.json")
            df.to_json(filepath, orient='records', date_format='iso')
        elif format == 'parquet':
            filepath = os.path.join(self.data_dir, f"{filename}.parquet")
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
    
    def load_data(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load processed data from file.
        
        Args:
            filename: Filename (without extension)
            format: File format ('csv', 'json', 'parquet')
        
        Returns:
            Loaded DataFrame
        """
        if format == 'csv':
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            return pd.read_csv(filepath)
        elif format == 'json':
            filepath = os.path.join(self.data_dir, f"{filename}.json")
            return pd.read_json(filepath)
        elif format == 'parquet':
            filepath = os.path.join(self.data_dir, f"{filename}.parquet")
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str) -> str:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary containing metrics
            filename: Filename (without extension)
        
        Returns:
            Full path to saved file
        """
        filepath = os.path.join(self.data_dir, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        return filepath


# Example usage and testing
if __name__ == "__main__":
    # Test data processor with mock data
    processor = ICPDataProcessor()
    
    # Create mock API response (list format)
    mock_response = [
        {'timestamp': 1725840000, 'value': 525000000.0},  # 2025-09-09
        {'timestamp': 1725926400, 'value': 525100000.0},  # 2025-09-10
        {'timestamp': 1726012800, 'value': 525200000.0},  # 2025-09-11
    ]
    
    try:
        print("Testing data processor...")
        
        # Process the data
        df = processor.process_total_supply_series(mock_response)
        print(f"Processed DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Calculate metrics
        metrics = processor.calculate_supply_metrics(df)
        print(f"\nCalculated metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Save data
        csv_path = processor.save_data(df, "test_supply_data", "csv")
        print(f"\nSaved data to: {csv_path}")
        
        # Save metrics
        metrics_path = processor.save_metrics(metrics, "test_metrics")
        print(f"Saved metrics to: {metrics_path}")
    
    except Exception as e:
        print(f"Error testing data processor: {e}")