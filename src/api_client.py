"""
ICP Ledger API client for querying blockchain data.
Handles API requests to the Internet Computer Protocol ledger endpoints.
"""

import requests
from typing import Dict, List, Optional, Any
import json
from urllib.parse import urljoin

try:
    from .date_utils import date_to_timestamp, get_day_step_seconds, validate_date_range
except ImportError:
    from date_utils import date_to_timestamp, get_day_step_seconds, validate_date_range


class ICPLedgerClient:
    """Client for interacting with ICP Ledger API."""
    
    BASE_URL = "https://ledger-api.internetcomputer.org"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the ICP Ledger API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ICP-Data-Analysis-App/1.0.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the ICP Ledger API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
        
        Returns:
            JSON response data
        
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response is not valid JSON
        """
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e
    
    def get_total_supply_series(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get total supply data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            API response containing total supply series data
        
        Raises:
            ValueError: If date range is invalid
            requests.RequestException: If API request fails
        """
        # Validate date range
        start_date, end_date = validate_date_range(start_date, end_date)
        
        # Convert dates to timestamps
        start_timestamp = date_to_timestamp(start_date)
        end_timestamp = date_to_timestamp(end_date)
        
        # Get step size (1 day in seconds)
        step = get_day_step_seconds()
        
        # Prepare API parameters
        params = {
            'step': step,
            'start': start_timestamp,
            'end': end_timestamp
        }
        
        # Make the API request
        endpoint = "/supply/total/series"
        return self._make_request(endpoint, params)
    
    def get_supply_at_time(self, timestamp: int) -> Dict[str, Any]:
        """
        Get total supply at a specific timestamp.
        
        Args:
            timestamp: Unix timestamp
        
        Returns:
            API response containing supply data at the specified time
        
        Raises:
            requests.RequestException: If API request fails
        """
        endpoint = f"/supply/total/{timestamp}"
        return self._make_request(endpoint)
    
    def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Test the client
    client = ICPLedgerClient()
    
    try:
        print("Testing ICP Ledger API client...")
        
        # Test with sample date range
        start_date = "2025-09-09"
        end_date = "2025-09-15"
        
        print(f"Fetching total supply data from {start_date} to {end_date}...")
        
        response = client.get_total_supply_series(start_date, end_date)
        print(f"Response keys: {list(response.keys())}")
        
        if 'series' in response:
            print(f"Number of data points: {len(response['series'])}")
            if response['series']:
                print(f"First data point: {response['series'][0]}")
                print(f"Last data point: {response['series'][-1]}")
    
    except Exception as e:
        print(f"Error testing API client: {e}")
    
    finally:
        client.close()