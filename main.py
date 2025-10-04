#!/usr/bin/env python3
"""
ICP Data Analysis Application
Main entry point for querying ICP ledger APIs and generating data analysis.
"""

import argparse
import sys
import os
from datetime import datetime
import traceback

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api_client import ICPLedgerClient
from src.data_processor import ICPDataProcessor
from src.date_utils import validate_date_range


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Query ICP ledger APIs and generate data analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start-date 2025-09-09 --end-date 2025-09-15
  %(prog)s -s 2025-09-01 -e 2025-09-30 --save-format json
  %(prog)s --start-date 2025-09-09 --end-date 2025-09-15 --output-prefix icp_supply_sept
        """
    )
    
    parser.add_argument(
        '-s', '--start-date',
        required=True,
        help='Start date in YYYY-MM-DD format (e.g., 2025-09-09)'
    )
    
    parser.add_argument(
        '-e', '--end-date',
        required=True,
        help='End date in YYYY-MM-DD format (e.g., 2025-09-15)'
    )
    
    parser.add_argument(
        '--save-format',
        choices=['csv', 'json', 'parquet'],
        default='csv',
        help='Output file format (default: csv)'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='icp_supply_data',
        help='Prefix for output files (default: icp_supply_data)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='API request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main application function."""
    args = parse_arguments()
    
    if args.verbose:
        print(f"ICP Data Analysis App - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Validate date range
        if args.verbose:
            print("Validating date range...")
        
        validate_date_range(args.start_date, args.end_date)
        
        # Initialize components
        if args.verbose:
            print("Initializing API client and data processor...")
        
        client = ICPLedgerClient(timeout=args.timeout)
        processor = ICPDataProcessor()
        
        # Fetch data from API
        if args.verbose:
            print("Fetching total supply data from ICP Ledger API...")
        
        api_response = client.get_total_supply_series(args.start_date, args.end_date)
        
        if args.verbose:
            # Handle both list and dict API response formats
            if isinstance(api_response, list):
                data_points = len(api_response)
            else:
                data_points = len(api_response.get('series', []))
            print(f"Received API response with {data_points} data points")
        
        # Process the data
        if args.verbose:
            print("Processing API response data...")
        
        df = processor.process_total_supply_series(api_response)
        
        if args.verbose:
            print(f"Processed {len(df)} data points")
            print(f"Date range in data: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        
        # Calculate metrics
        if args.verbose:
            print("Calculating supply metrics...")
        
        metrics = processor.calculate_supply_metrics(df)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"{args.output_prefix}_{args.start_date}_to_{args.end_date}_{timestamp}"
        metrics_filename = f"{args.output_prefix}_metrics_{args.start_date}_to_{args.end_date}_{timestamp}"
        
        # Save processed data
        if args.verbose:
            print(f"Saving data in {args.save_format} format...")
        
        data_filepath = processor.save_data(df, data_filename, args.save_format)
        metrics_filepath = processor.save_metrics(metrics, metrics_filename)
        
        # Print results
        print("\n" + "="*60)
        print("ICP TOTAL SUPPLY DATA ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\nDate Range: {args.start_date} to {args.end_date}")
        print(f"Data Points: {len(df)}")
        
        if metrics.get('supply_stats'):
            stats = metrics['supply_stats']
            print(f"\nSupply Statistics:")
            print(f"  Minimum: {stats['min']:,.2f}")
            print(f"  Maximum: {stats['max']:,.2f}")
            print(f"  Average: {stats['mean']:,.2f}")
            print(f"  Std Dev: {stats['std']:,.2f}")
        
        if metrics.get('supply_changes'):
            changes = metrics['supply_changes']
            print(f"\nSupply Changes:")
            print(f"  Total Change: {changes['total_change']:,.2f}")
            print(f"  Avg Daily Change: {changes['avg_daily_change']:,.2f}")
        
        print(f"\nOutput Files:")
        print(f"  Data: {data_filepath}")
        print(f"  Metrics: {metrics_filepath}")
        
        print(f"\nData Preview:")
        print(df[['date', 'total_supply']].head())
        
        if len(df) > 5:
            print("...")
            print(df[['date', 'total_supply']].tail(2))
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        try:
            client.close()
        except:
            pass


if __name__ == "__main__":
    main()