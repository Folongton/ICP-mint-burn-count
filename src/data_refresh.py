import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import streamlit as st
from typing import Tuple, Optional

class ICPDataRetriever:
    """Handle ICP supply data retrieval and updates"""
    
    def __init__(self, output_dir: str = "output_data"):
        self.output_dir = output_dir
        self.base_url = "https://ledger-api.internetcomputer.org"
        self.supply_url = f"{self.base_url}/supply/total/series"
        
    def get_latest_csv_file(self) -> Optional[str]:
        """Find the most recent CSV file in output_data directory"""
        if not os.path.exists(self.output_dir):
            return None
            
        csv_files = [f for f in os.listdir(self.output_dir) if f.startswith("icp_supply_data_") and f.endswith(".csv")]
        
        if not csv_files:
            return None
            
        # Sort by modification time to get the most recent
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)), reverse=True)
        return os.path.join(self.output_dir, csv_files[0])
    
    def get_last_data_timestamp(self, csv_path: str) -> Optional[int]:
        """Get the last timestamp from existing CSV data"""
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' not in df.columns or len(df) == 0:
                return None
            
            last_timestamp = df['timestamp'].iloc[-1]
            return int(last_timestamp)
        except Exception as e:
            st.error(f"Error reading last data timestamp: {e}")
            return None
    
    def needs_data_refresh(self, csv_path: str) -> bool:
        """Check if data needs refresh (more than 24 hours old)"""
        if not csv_path or not os.path.exists(csv_path):
            return True
            
        last_timestamp = self.get_last_data_timestamp(csv_path)
        if not last_timestamp:
            return True
            
        # Calculate hours since last data point using timestamps
        current_timestamp = int(datetime.utcnow().timestamp())
        hours_since_last = (current_timestamp - last_timestamp) / 3600
        
        # Data is stale if more than 25 hours old (giving 1 hour buffer for daily updates)
        return hours_since_last > 25
    
    def fetch_icp_supply_data(self, start_date: datetime, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Fetch ICP supply data from ledger API"""
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # Convert dates to timestamps (ledger API expects Unix timestamps)
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Prepare API request parameters
            params = {
                'start': start_timestamp,
                'end': end_timestamp,
                'step': 86400  # Daily data (86400 seconds = 24 hours)
            }
            
            st.info(f"📡 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            
            response = requests.get(self.supply_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                st.warning("No new data available from ledger API")
                return None
            
            # Process the ledger API response (list of [timestamp, value] pairs)
            records = []
            for item in data:
                if len(item) >= 2:
                    timestamp = item[0]
                    supply_value = item[1]
                    date = datetime.utcfromtimestamp(timestamp)
                    
                    records.append({
                        'timestamp': timestamp,
                        'date': date.strftime('%Y-%m-%d'), 
                        'total_supply': float(supply_value),
                        'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
                        'supply_change': 0.0,  # Will be calculated after sorting
                        'supply_change_pct': 0.0  # Will be calculated after sorting
                    })
            
            df = pd.DataFrame(records).sort_values('date')
            
            # Calculate supply changes
            if len(df) > 1:
                df['supply_change'] = df['total_supply'].diff()
                df['supply_change_pct'] = (df['supply_change'] / df['total_supply'].shift(1)) * 100
            else:
                df['supply_change'] = 0
                df['supply_change_pct'] = 0
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return None
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return None
    
    def merge_and_update_data(self, existing_csv: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing CSV and return combined dataset"""
        try:
            # Load existing data
            existing_df = pd.read_csv(existing_csv)
            
            # Convert dates to datetime for comparison
            existing_df['date_dt'] = pd.to_datetime(existing_df['date'])
            new_data['date_dt'] = pd.to_datetime(new_data['date'])
            
            # Find the overlap point - get last date in existing data
            last_existing_date = existing_df['date_dt'].max()
            
            # Filter new data to only include dates after the last existing date
            new_records = new_data[new_data['date_dt'] > last_existing_date].copy()
            
            if len(new_records) == 0:
                st.info("No new records to add - data is already up to date")
                return existing_df.drop('date_dt', axis=1)
            
            # Remove the helper date_dt column before combining
            existing_df = existing_df.drop('date_dt', axis=1)
            new_records = new_records.drop('date_dt', axis=1)
            
            # Combine datasets
            combined_df = pd.concat([existing_df, new_records], ignore_index=True)
            
            # Recalculate supply changes for the entire dataset to ensure consistency
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            combined_df['supply_change'] = combined_df['total_supply'].diff()
            combined_df['supply_change_pct'] = (combined_df['supply_change'] / combined_df['total_supply'].shift(1)) * 100
            
            # Fill NaN values in first row
            combined_df.loc[0, 'supply_change'] = 0
            combined_df.loc[0, 'supply_change_pct'] = 0
            
            st.success(f"✅ Added {len(new_records)} new records to the dataset")
            
            return combined_df
            
        except Exception as e:
            st.error(f"Error merging data: {e}")
            return pd.read_csv(existing_csv)  # Return original data on error
    
    def save_updated_data(self, df: pd.DataFrame, original_csv_path: str) -> str:
        """Save updated dataset to a new CSV file"""
        try:
            # Create new filename with current timestamp
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract date range from data
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            
            new_filename = f"icp_supply_data_{start_date}_to_{end_date}_{timestamp_str}.csv"
            new_filepath = os.path.join(self.output_dir, new_filename)
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save the updated data
            df.to_csv(new_filepath, index=False)
            
            st.success(f"📁 Updated data saved to: {new_filename}")
            
            # Optionally remove the old file to avoid clutter (commented out for safety)
            # if original_csv_path and os.path.exists(original_csv_path):
            #     os.remove(original_csv_path)
            #     st.info(f"Removed old file: {os.path.basename(original_csv_path)}")
            
            return new_filepath
            
        except Exception as e:
            st.error(f"Error saving updated data: {e}")
            return original_csv_path
    
    def refresh_data_if_needed(self) -> Tuple[str, bool]:
        """
        Main method to check and refresh data if needed
        Returns: (csv_file_path, was_refreshed)
        """
        # Find existing CSV file
        existing_csv = self.get_latest_csv_file()
        
        # Check if refresh is needed
        if not self.needs_data_refresh(existing_csv):
            # Show when the last data point is from
            if existing_csv:
                last_timestamp = self.get_last_data_timestamp(existing_csv)
                if last_timestamp:
                    last_date = datetime.utcfromtimestamp(last_timestamp)
                    st.info(f"📊 Data is up to date (last data: {last_date.strftime('%Y-%m-%d %H:%M')} UTC)")
                else:
                    st.info("📊 Data is up to date")
            else:
                st.info("📊 Data is up to date")
            return existing_csv, False
        
        try:
            if existing_csv:
                st.info("🔄 Data is more than 25 hours old. Checking for updates...")
                last_timestamp = self.get_last_data_timestamp(existing_csv)
                
                if last_timestamp:
                    # Convert timestamp to datetime and fetch data from day after last record
                    last_date = datetime.utcfromtimestamp(last_timestamp)
                    start_date = last_date + timedelta(days=1)
                else:
                    # Fallback: fetch last 30 days
                    start_date = datetime.now() - timedelta(days=30)
            else:
                st.info("📥 No existing data found. Fetching initial dataset...")
                # No existing data, fetch last 2 years
                start_date = datetime.now() - timedelta(days=730)
            
            # Fetch new data
            new_data = self.fetch_icp_supply_data(start_date)
            
            if new_data is None or len(new_data) == 0:
                if existing_csv:
                    st.warning("No new data available. Using existing dataset.")
                    return existing_csv, False
                else:
                    st.error("Failed to fetch initial data and no existing data found.")
                    return None, False
            
            if existing_csv:
                # Merge with existing data
                combined_data = self.merge_and_update_data(existing_csv, new_data)
                updated_csv = self.save_updated_data(combined_data, existing_csv)
            else:
                # Save as new file
                updated_csv = self.save_updated_data(new_data, None)
            
            return updated_csv, True
            
        except Exception as e:
            st.error(f"Error during data refresh: {e}")
            if existing_csv:
                st.info("Using existing data due to refresh error")
                return existing_csv, False
            return None, False

def get_fresh_data() -> Tuple[Optional[str], bool]:
    """
    Convenience function to get fresh ICP supply data
    Returns: (csv_file_path, was_refreshed)
    """
    retriever = ICPDataRetriever()
    return retriever.refresh_data_if_needed()