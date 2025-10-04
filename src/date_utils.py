"""
Date utility functions for ICP data analysis.
Handles conversion between date strings and Unix timestamps.
"""

from datetime import datetime
from typing import Union


def date_to_timestamp(date_str: str) -> int:
    """
    Convert a date string in YYYY-MM-DD format to Unix timestamp.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD' (e.g., '2025-09-09')
    
    Returns:
        Unix timestamp as integer
    
    Raises:
        ValueError: If date string format is invalid
    """
    try:
        # Parse the date string
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Convert to Unix timestamp
        return int(date_obj.timestamp())
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {date_str}") from e


def timestamp_to_date(timestamp: Union[int, float]) -> str:
    """
    Convert Unix timestamp to date string in YYYY-MM-DD format.
    
    Args:
        timestamp: Unix timestamp
    
    Returns:
        Date string in format 'YYYY-MM-DD'
    """
    date_obj = datetime.fromtimestamp(timestamp)
    return date_obj.strftime('%Y-%m-%d')


def get_day_step_seconds() -> int:
    """
    Get the number of seconds in one day (86400).
    This is used as the step parameter for the API calls.
    
    Returns:
        Number of seconds in a day
    """
    return 86400


def validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """
    Validate that start_date is before end_date and both are valid.
    
    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format
    
    Returns:
        Tuple of validated (start_date, end_date)
    
    Raises:
        ValueError: If dates are invalid or start_date >= end_date
    """
    # Validate format by converting to timestamps
    start_ts = date_to_timestamp(start_date)
    end_ts = date_to_timestamp(end_date)
    
    if start_ts >= end_ts:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
    
    return start_date, end_date


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    test_date = "2025-09-09"
    print(f"Date: {test_date}")
    
    timestamp = date_to_timestamp(test_date)
    print(f"Timestamp: {timestamp}")
    
    converted_back = timestamp_to_date(timestamp)
    print(f"Converted back: {converted_back}")
    
    print(f"Day step (seconds): {get_day_step_seconds()}")
    
    # Test validation
    try:
        validate_date_range("2025-09-09", "2025-09-15")
        print("Date range validation: OK")
    except ValueError as e:
        print(f"Date range validation error: {e}")