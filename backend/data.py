"""
Data acquisition and preprocessing module for stock forecasting app.
Handles fetching AAPL data from yfinance and data cleaning.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Optional, Tuple


def get_historical_data(start_date: str = '2020-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical AAPL daily OHLC data from yfinance.
    
    Args:
        start_date (str): Start date for data fetch (YYYY-MM-DD format)
        end_date (str, optional): End date for data fetch. If None, uses today.
    
    Returns:
        pd.DataFrame: Cleaned historical data with OHLC, volume, and features
    """
    try:
        # Fetch AAPL data from Yahoo Finance
        ticker = yf.Ticker("AAPL")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download historical data
        data = ticker.history(start=start_date, end=end_date, period='1d')
        
        if data.empty:
            raise ValueError("No data retrieved from yfinance")
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure Date column is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Handle missing values with forward fill
        data = data.fillna(method='ffill')
        
        # Add 7-day moving average feature
        data['ma7'] = data['close'].rolling(window=7, min_periods=1).mean()
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        print(f"Successfully fetched {len(data)} days of AAPL data from {data['date'].min()} to {data['date'].max()}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        raise


def update_latest_data() -> pd.DataFrame:
    """
    Fetch only new AAPL data since the last database entry.
    
    Returns:
        pd.DataFrame: New data since last database entry
    """
    try:
        # Connect to database to get last date
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # Get the latest date from database
        cursor.execute("SELECT MAX(date) FROM prices")
        last_date = cursor.fetchone()[0]
        
        conn.close()
        
        if last_date is None:
            # No data in database, fetch from 2020
            print("No existing data found. Fetching from 2020-01-01")
            return get_historical_data('2020-01-01')
        
        # Convert last_date to datetime and add one day
        last_datetime = pd.to_datetime(last_date)
        start_date = (last_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch new data
        print(f"Fetching new data from {start_date}")
        new_data = get_historical_data(start_date)
        
        return new_data
        
    except Exception as e:
        print(f"Error updating latest data: {e}")
        # Fallback to full data fetch
        return get_historical_data()


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate the quality of fetched data.
    
    Args:
        data (pd.DataFrame): Data to validate
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Check for required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for reasonable price values
        if (data['close'] <= 0).any():
            print("Found invalid price values (<= 0)")
            return False
        
        # Check for reasonable volume values
        if (data['volume'] < 0).any():
            print("Found invalid volume values (< 0)")
            return False
        
        # Check for date consistency
        if not data['date'].is_monotonic_increasing:
            print("Dates are not in chronological order")
            return False
        
        print("Data validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False


def get_data_summary(data: pd.DataFrame) -> dict:
    """
    Get summary statistics for the data.
    
    Args:
        data (pd.DataFrame): Data to summarize
    
    Returns:
        dict: Summary statistics
    """
    try:
        summary = {
            'total_days': len(data),
            'date_range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
            'price_range': f"${data['close'].min():.2f} - ${data['close'].max():.2f}",
            'avg_volume': f"{data['volume'].mean():,.0f}",
            'missing_values': data.isnull().sum().sum()
        }
        
        return summary
        
    except Exception as e:
        print(f"Error generating data summary: {e}")
        return {}


if __name__ == "__main__":
    # Test data fetching
    print("Testing data acquisition...")
    
    # Fetch historical data
    data = get_historical_data()
    
    # Validate data
    if validate_data(data):
        # Print summary
        summary = get_data_summary(data)
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Show last few rows
        print("\nLast 5 rows:")
        print(data.tail())
        
        print("\nData acquisition test completed successfully!")
    else:
        print("Data validation failed!")
