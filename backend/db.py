"""
Database operations module for stock forecasting app.
Handles SQLite database operations for storing historical data and predictions.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db(db_path: str = 'stock_data.db') -> None:
    """
    Initialize the SQLite database with required tables.
    
    Args:
        db_path (str): Path to the SQLite database file
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create prices table for historical OHLC data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                date TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                ma7 REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table for model forecasts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                horizon TEXT NOT NULL,
                model_name TEXT NOT NULL,
                forecast_value REAL NOT NULL,
                confidence_lower REAL,
                confidence_upper REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model_metrics table for performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                rmse REAL NOT NULL,
                mae REAL NOT NULL,
                mape REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def insert_prices(data: pd.DataFrame, db_path: str = 'stock_data.db') -> int:
    """
    Insert or update price data in the database using UPSERT.
    
    Args:
        data (pd.DataFrame): Price data to insert
        db_path (str): Path to the SQLite database file
    
    Returns:
        int: Number of rows inserted/updated
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        data_to_insert = []
        for _, row in data.iterrows():
            data_to_insert.append((
                row['date'].strftime('%Y-%m-%d'),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                float(row.get('ma7', row['close']))  # Use close if ma7 not available
            ))
        
        # UPSERT operation using INSERT OR REPLACE
        cursor.executemany('''
            INSERT OR REPLACE INTO prices 
            (date, open, high, low, close, volume, ma7)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted/updated {rows_affected} price records")
        return rows_affected
        
    except Exception as e:
        logger.error(f"Error inserting prices: {e}")
        raise


def get_latest_prices(days: int = 365, db_path: str = 'stock_data.db') -> pd.DataFrame:
    """
    Get the latest N days of price data from the database.
    
    Args:
        days (int): Number of days to retrieve
        db_path (str): Path to the SQLite database file
    
    Returns:
        pd.DataFrame: Latest price data
    """
    try:
        conn = sqlite3.connect(db_path)
        
        query = '''
            SELECT date, open, high, low, close, volume, ma7
            FROM prices
            ORDER BY date DESC
            LIMIT ?
        '''
        
        data = pd.read_sql_query(query, conn, params=(days,))
        conn.close()
        
        if not data.empty:
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            # Sort by date ascending for time series analysis
            data = data.sort_values('date').reset_index(drop=True)
            
        logger.info(f"Retrieved {len(data)} price records")
        return data
        
    except Exception as e:
        logger.error(f"Error retrieving latest prices: {e}")
        raise


def insert_prediction(date: str, horizon: str, model_name: str, 
                     forecast_value: float, confidence_lower: Optional[float] = None,
                     confidence_upper: Optional[float] = None, 
                     db_path: str = 'stock_data.db') -> int:
    """
    Insert a prediction into the database.
    
    Args:
        date (str): Date for the prediction (YYYY-MM-DD)
        horizon (str): Forecast horizon (e.g., '24hr', '72hr')
        model_name (str): Name of the model (e.g., 'ARIMA', 'LSTM')
        forecast_value (float): Predicted value
        confidence_lower (float, optional): Lower confidence bound
        confidence_upper (float, optional): Upper confidence bound
        db_path (str): Path to the SQLite database file
    
    Returns:
        int: ID of the inserted prediction
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (date, horizon, model_name, forecast_value, confidence_lower, confidence_upper)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, horizon, model_name, forecast_value, confidence_lower, confidence_upper))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted prediction ID {prediction_id} for {model_name}")
        return prediction_id
        
    except Exception as e:
        logger.error(f"Error inserting prediction: {e}")
        raise


def get_predictions(instrument: str = 'AAPL', horizon: str = '24hr', 
                   limit: int = 100, db_path: str = 'stock_data.db') -> pd.DataFrame:
    """
    Get recent predictions from the database.
    
    Args:
        instrument (str): Stock symbol (for future extensibility)
        horizon (str): Forecast horizon to filter by
        limit (int): Maximum number of predictions to retrieve
        db_path (str): Path to the SQLite database file
    
    Returns:
        pd.DataFrame: Recent predictions
    """
    try:
        conn = sqlite3.connect(db_path)
        
        query = '''
            SELECT id, date, horizon, model_name, forecast_value, 
                   confidence_lower, confidence_upper, timestamp
            FROM predictions
            WHERE horizon = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        data = pd.read_sql_query(query, conn, params=(horizon, limit))
        conn.close()
        
        if not data.empty:
            data['date'] = pd.to_datetime(data['date'])
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        logger.info(f"Retrieved {len(data)} predictions for horizon {horizon}")
        return data
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise


def insert_model_metrics(model_name: str, rmse: float, mae: float, mape: float,
                        db_path: str = 'stock_data.db') -> int:
    """
    Insert model performance metrics into the database.
    
    Args:
        model_name (str): Name of the model
        rmse (float): Root Mean Square Error
        mae (float): Mean Absolute Error
        mape (float): Mean Absolute Percentage Error
        db_path (str): Path to the SQLite database file
    
    Returns:
        int: ID of the inserted metrics record
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (model_name, rmse, mae, mape)
            VALUES (?, ?, ?, ?)
        ''', (model_name, rmse, mae, mape))
        
        metrics_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted metrics for {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")
        return metrics_id
        
    except Exception as e:
        logger.error(f"Error inserting model metrics: {e}")
        raise


def get_model_metrics(model_name: Optional[str] = None, 
                     db_path: str = 'stock_data.db') -> pd.DataFrame:
    """
    Get model performance metrics from the database.
    
    Args:
        model_name (str, optional): Filter by specific model name
        db_path (str): Path to the SQLite database file
    
    Returns:
        pd.DataFrame: Model metrics
    """
    try:
        conn = sqlite3.connect(db_path)
        
        if model_name:
            query = '''
                SELECT model_name, rmse, mae, mape, created_at
                FROM model_metrics
                WHERE model_name = ?
                ORDER BY created_at DESC
            '''
            data = pd.read_sql_query(query, conn, params=(model_name,))
        else:
            query = '''
                SELECT model_name, rmse, mae, mape, created_at
                FROM model_metrics
                ORDER BY created_at DESC
            '''
            data = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if not data.empty:
            data['created_at'] = pd.to_datetime(data['created_at'])
        
        logger.info(f"Retrieved metrics for {len(data)} model records")
        return data
        
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {e}")
        raise


def get_database_stats(db_path: str = 'stock_data.db') -> dict:
    """
    Get database statistics.
    
    Args:
        db_path (str): Path to the SQLite database file
    
    Returns:
        dict: Database statistics
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table counts
        cursor.execute("SELECT COUNT(*) FROM prices")
        prices_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_metrics")
        metrics_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
        date_range = cursor.fetchone()
        
        conn.close()
        
        stats = {
            'prices_records': prices_count,
            'predictions_records': predictions_count,
            'metrics_records': metrics_count,
            'date_range': date_range
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}


if __name__ == "__main__":
    # Test database operations
    print("Testing database operations...")
    
    # Initialize database
    init_db()
    
    # Get database stats
    stats = get_database_stats()
    print(f"Database stats: {stats}")
    
    print("Database operations test completed successfully!")
