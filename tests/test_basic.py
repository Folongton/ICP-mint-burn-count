"""
Basic tests for ICP data analysis modules.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.date_utils import date_to_timestamp, timestamp_to_date, validate_date_range
from src.data_processor import ICPDataProcessor


class TestDateUtils(unittest.TestCase):
    """Test date utility functions."""
    
    def test_date_to_timestamp(self):
        """Test date string to timestamp conversion."""
        # Test known date
        timestamp = date_to_timestamp("2025-01-01")
        self.assertIsInstance(timestamp, int)
        self.assertGreater(timestamp, 0)
    
    def test_timestamp_to_date(self):
        """Test timestamp to date string conversion."""
        # Test round-trip conversion
        original_date = "2025-09-09"
        timestamp = date_to_timestamp(original_date)
        converted_date = timestamp_to_date(timestamp)
        self.assertEqual(original_date, converted_date)
    
    def test_validate_date_range(self):
        """Test date range validation."""
        # Valid range
        start, end = validate_date_range("2025-09-09", "2025-09-15")
        self.assertEqual(start, "2025-09-09")
        self.assertEqual(end, "2025-09-15")
        
        # Invalid range (start after end)
        with self.assertRaises(ValueError):
            validate_date_range("2025-09-15", "2025-09-09")


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        """Set up test data processor."""
        self.processor = ICPDataProcessor(data_dir="test_data")
    
    def test_process_total_supply_series(self):
        """Test processing of API response."""
        # Test with dictionary format
        mock_response_dict = {
            'series': [
                {'timestamp': 1725840000, 'value': 525000000.0},
                {'timestamp': 1725926400, 'value': 525100000.0},
            ]
        }
        
        df = self.processor.process_total_supply_series(mock_response_dict)
        
        self.assertEqual(len(df), 2)
        self.assertIn('timestamp', df.columns)
        self.assertIn('date', df.columns)
        self.assertIn('total_supply', df.columns)
        
        # Test with list format
        mock_response_list = [
            {'timestamp': 1725840000, 'value': 525000000.0},
            {'timestamp': 1725926400, 'value': 525100000.0},
        ]
        
        df_list = self.processor.process_total_supply_series(mock_response_list)
        
        self.assertEqual(len(df_list), 2)
        self.assertIn('timestamp', df_list.columns)
        self.assertIn('date', df_list.columns)
        self.assertIn('total_supply', df_list.columns)
    
    def test_calculate_supply_metrics(self):
        """Test metrics calculation."""
        mock_response = [
            {'timestamp': 1725840000, 'value': 525000000.0},
            {'timestamp': 1725926400, 'value': 525100000.0},
        ]
        
        df = self.processor.process_total_supply_series(mock_response)
        metrics = self.processor.calculate_supply_metrics(df)
        
        self.assertIn('total_points', metrics)
        self.assertIn('supply_stats', metrics)
        self.assertEqual(metrics['total_points'], 2)


if __name__ == '__main__':
    unittest.main()