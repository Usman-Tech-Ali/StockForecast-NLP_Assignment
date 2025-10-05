"""
Stock Market Database Handler
A comprehensive SQLite-based data management system for financial market applications.
Provides efficient storage and retrieval of stock market data, predictions, and metadata.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockMarketDatabase:
    """Stock market database handler with comprehensive data management capabilities."""
    
    def __init__(self, db_path: str = "stock_forecasting.db"):
        """
        Initialize stock market database handler.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.connect()
        self.initialize_tables()
    
    def connect(self):
        """Establish connection to SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.connection = None
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
        return False
    
    def initialize_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            # Datasets table (equivalent to datasets collection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    days INTEGER NOT NULL,
                    records INTEGER NOT NULL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Historical prices table (equivalent to historical_prices collection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, exchange, date)
                )
            ''')
            
            # Predictions table (equivalent to predictions collection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    forecast_horizon INTEGER NOT NULL,
                    predicted_values TEXT NOT NULL,
                    metrics TEXT,
                    y_true TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metadata table (equivalent to metadata collection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    instrument_info TEXT,
                    data_sources TEXT,
                    update_logs TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_symbol ON datasets(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol ON historical_prices(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON metadata(symbol)')
            
            self.connection.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            if self.connection:
                self.connection.rollback()
    
    def save_dataset(self, dataset_data: Dict[str, Any]) -> Optional[int]:
        """
        Save dataset to database.
        
        Args:
            dataset_data (Dict): Dataset information and data
            
        Returns:
            Optional[int]: Dataset ID if successful, None otherwise
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            # Convert data to JSON string for storage
            data_json = json.dumps(dataset_data.get('data', []))
            
            cursor.execute('''
                INSERT INTO datasets (symbol, exchange, days, records, generated_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                dataset_data.get('symbol'),
                dataset_data.get('exchange'),
                dataset_data.get('days'),
                dataset_data.get('records'),
                dataset_data.get('generated_at', datetime.now()),
                data_json
            ))
            
            dataset_id = cursor.lastrowid
            self.connection.commit()
            logger.info(f"Dataset saved successfully with ID: {dataset_id}")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_dataset_by_id(self, dataset_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get dataset by ID.
        
        Args:
            dataset_id (Union[int, str]): Dataset ID
            
        Returns:
            Optional[Dict]: Dataset data if found, None otherwise
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM datasets WHERE id = ?', (int(dataset_id),))
            row = cursor.fetchone()
            
            if row:
                dataset = dict(row)
                # Parse JSON data back to list
                dataset['data'] = json.loads(dataset['data'])
                return dataset
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving dataset: {e}")
            return None
    
    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """
        Get all datasets.
        
        Returns:
            List[Dict]: List of all datasets
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM datasets ORDER BY generated_at DESC')
            rows = cursor.fetchall()
            
            datasets = []
            for row in rows:
                dataset = dict(row)
                # Parse JSON data back to list
                dataset['data'] = json.loads(dataset['data'])
                datasets.append(dataset)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error retrieving all datasets: {e}")
            return []
    
    def get_recent_datasets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent datasets.
        
        Args:
            limit (int): Maximum number of datasets to return
            
        Returns:
            List[Dict]: List of recent datasets
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT id, symbol, exchange, days, records, generated_at 
                FROM datasets 
                ORDER BY generated_at DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
            datasets = []
            for row in rows:
                dataset = dict(row)
                datasets.append({
                    'id': str(dataset['id']),
                    'symbol': dataset['symbol'],
                    'date': dataset['generated_at'][:10],  # Extract date part
                    'records': dataset['records'],
                    'exchange': dataset['exchange']
                })
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error retrieving recent datasets: {e}")
            return []
    
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict]: Latest dataset for symbol
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM datasets 
                WHERE symbol = ? 
                ORDER BY generated_at DESC 
                LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            
            if row:
                dataset = dict(row)
                dataset['data'] = json.loads(dataset['data'])
                return dataset
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving latest data: {e}")
            return None
    
    def save_historical_prices(self, symbol: str, exchange: str, prices: List[Dict[str, Any]]) -> Optional[int]:
        """
        Save historical prices to database.
        
        Args:
            symbol (str): Stock symbol
            exchange (str): Exchange name
            prices (List[Dict]): List of price data
            
        Returns:
            Optional[int]: Number of records inserted
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            if not prices:
                logger.warning("No prices data provided for historical price storage")
                return None
            
            logger.info(f"Saving {len(prices)} historical price records for {symbol}")
            
            cursor = self.connection.cursor()
            
            # Delete existing records for this symbol to avoid duplicates
            cursor.execute('DELETE FROM historical_prices WHERE symbol = ? AND exchange = ?', (symbol, exchange))
            
            # Insert new records
            records_to_insert = []
            for price in prices:
                try:
                    records_to_insert.append((
                        symbol,
                        exchange,
                        price.get('date'),
                        float(price.get('open_price') or price.get('open') or 0),
                        float(price.get('high_price') or price.get('high') or 0),
                        float(price.get('low_price') or price.get('low') or 0),
                        float(price.get('close_price') or price.get('close') or 0),
                        int(price.get('volume') or 0)
                    ))
                except Exception as e:
                    logger.warning(f"Error processing price record: {e}")
                    continue
            
            if records_to_insert:
                cursor.executemany('''
                    INSERT OR REPLACE INTO historical_prices 
                    (symbol, exchange, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', records_to_insert)
                
                self.connection.commit()
                logger.info(f"Successfully saved {len(records_to_insert)} historical price records")
                return len(records_to_insert)
            else:
                logger.warning("No valid price records to save")
                return None
                
        except Exception as e:
            logger.error(f"Error saving historical prices: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_prices(self, symbol: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get historical prices for visualization.
        
        Args:
            symbol (str): Stock symbol
            start_date (Optional[str]): Start date filter
            end_date (Optional[str]): End date filter
            limit (int): Maximum number of records
            
        Returns:
            List[Dict]: List of price records
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            # Build query with optional date filters
            query = 'SELECT * FROM historical_prices WHERE symbol = ?'
            params = [symbol]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY date ASC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            prices = []
            for row in rows:
                price = dict(row)
                prices.append({
                    'symbol': price['symbol'],
                    'date': price['date'],
                    'open': float(price['open_price']),
                    'high': float(price['high_price']),
                    'low': float(price['low_price']),
                    'close': float(price['close_price']),
                    'volume': int(price['volume'])
                })
            
            return prices
            
        except Exception as e:
            logger.error(f"Error retrieving prices: {e}")
            return []
    
    def save_forecast(self, forecast_data: Dict[str, Any]) -> Optional[int]:
        """
        Save forecast to database.
        
        Args:
            forecast_data (Dict): Forecast data
            
        Returns:
            Optional[int]: Forecast ID if successful
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (symbol, model, forecast_horizon, predicted_values, metrics, y_true, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                forecast_data.get('symbol'),
                forecast_data.get('model'),
                forecast_data.get('forecast_horizon'),
                json.dumps(forecast_data.get('predicted_values', [])),
                json.dumps(forecast_data.get('metrics', {})) if forecast_data.get('metrics') else None,
                json.dumps(forecast_data.get('y_true', [])) if forecast_data.get('y_true') else None,
                forecast_data.get('notes'),
                forecast_data.get('created_at', datetime.now())
            ))
            
            forecast_id = cursor.lastrowid
            self.connection.commit()
            logger.info(f"Forecast saved successfully with ID: {forecast_id}")
            return forecast_id
            
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_predictions(self, symbol: Optional[str] = None, horizon: Optional[int] = None,
                       model: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get predictions with optional filters.
        
        Args:
            symbol (Optional[str]): Stock symbol filter
            horizon (Optional[int]): Forecast horizon filter
            model (Optional[str]): Model filter
            limit (int): Maximum number of records
            
        Returns:
            List[Dict]: List of predictions
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            # Build query with optional filters
            query = 'SELECT * FROM predictions WHERE 1=1'
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if horizon:
                query += ' AND forecast_horizon = ?'
                params.append(horizon)
            
            if model:
                query += ' AND model = ?'
                params.append(model)
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            logger.info(f"Querying predictions with filters: symbol={symbol}, horizon={horizon}, model={model}")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                prediction = dict(row)
                # Parse JSON fields back to their original types
                prediction['predicted_values'] = json.loads(prediction['predicted_values'])
                if prediction['metrics']:
                    prediction['metrics'] = json.loads(prediction['metrics'])
                if prediction['y_true']:
                    prediction['y_true'] = json.loads(prediction['y_true'])
                predictions.append(prediction)
            
            logger.info(f"Found {len(predictions)} predictions matching query")
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent predictions.
        
        Args:
            limit (int): Maximum number of predictions
            
        Returns:
            List[Dict]: List of recent predictions
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                prediction = dict(row)
                # Parse JSON fields
                prediction['predicted_values'] = json.loads(prediction['predicted_values'])
                if prediction['metrics']:
                    prediction['metrics'] = json.loads(prediction['metrics'])
                if prediction['y_true']:
                    prediction['y_true'] = json.loads(prediction['y_true'])
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving recent predictions: {e}")
            return []
    
    def upsert_metadata(self, symbol: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Upsert metadata for a symbol.
        
        Args:
            symbol (str): Stock symbol
            metadata (Dict): Metadata information
            
        Returns:
            Optional[Dict]: Updated metadata
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            logger.info(f"Updating metadata for symbol: {symbol}")
            
            cursor = self.connection.cursor()
            
            # Check if metadata exists
            cursor.execute('SELECT * FROM metadata WHERE symbol = ?', (symbol,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing metadata
                cursor.execute('''
                    UPDATE metadata 
                    SET instrument_info = ?, data_sources = ?, update_logs = ?, last_updated = ?
                    WHERE symbol = ?
                ''', (
                    json.dumps(metadata.get('instrument_info', {})),
                    json.dumps(metadata.get('data_sources', {})),
                    json.dumps(metadata.get('update_logs', [])),
                    datetime.now(),
                    symbol
                ))
            else:
                # Insert new metadata
                cursor.execute('''
                    INSERT INTO metadata (symbol, instrument_info, data_sources, update_logs, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    json.dumps(metadata.get('instrument_info', {})),
                    json.dumps(metadata.get('data_sources', {})),
                    json.dumps(metadata.get('update_logs', [])),
                    datetime.now()
                ))
            
            self.connection.commit()
            
            # Return updated metadata
            cursor.execute('SELECT * FROM metadata WHERE symbol = ?', (symbol,))
            updated = cursor.fetchone()
            
            if updated:
                result = dict(updated)
                # Parse JSON fields
                result['instrument_info'] = json.loads(result['instrument_info'])
                result['data_sources'] = json.loads(result['data_sources'])
                result['update_logs'] = json.loads(result['update_logs'])
                logger.info(f"Successfully updated metadata for {symbol}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating metadata for {symbol}: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_metadata(self, symbol: Optional[str] = None) -> Union[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get metadata for a symbol or all symbols.
        
        Args:
            symbol (Optional[str]): Specific symbol to get metadata for
            
        Returns:
            Union[Optional[Dict], List[Dict]]: Metadata for symbol or list of all metadata
        """
        try:
            if not self.connection:
                raise Exception("Database not connected")
            
            cursor = self.connection.cursor()
            
            if symbol:
                cursor.execute('SELECT * FROM metadata WHERE symbol = ?', (symbol,))
                row = cursor.fetchone()
                
                if row:
                    result = dict(row)
                    # Parse JSON fields
                    result['instrument_info'] = json.loads(result['instrument_info'])
                    result['data_sources'] = json.loads(result['data_sources'])
                    result['update_logs'] = json.loads(result['update_logs'])
                    return result
                
                return None
            else:
                cursor.execute('SELECT * FROM metadata ORDER BY last_updated DESC')
                rows = cursor.fetchall()
                
                metadata_list = []
                for row in rows:
                    result = dict(row)
                    # Parse JSON fields
                    result['instrument_info'] = json.loads(result['instrument_info'])
                    result['data_sources'] = json.loads(result['data_sources'])
                    result['update_logs'] = json.loads(result['update_logs'])
                    metadata_list.append(result)
                
                return metadata_list
                
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return None if symbol else []
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """
        Save prediction to database (alias for save_forecast).
        
        Args:
            prediction_data (Dict): Prediction data
            
        Returns:
            Optional[int]: Prediction ID if successful
        """
        return self.save_forecast(prediction_data)
    
    def count_datasets(self) -> int:
        """Count total datasets."""
        try:
            if not self.connection:
                return 0
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM datasets')
            return cursor.fetchone()[0]
            
        except Exception as e:
            logger.error(f"Error counting datasets: {e}")
            return 0
    
    def count_records(self) -> int:
        """Count total records across all datasets."""
        try:
            if not self.connection:
                return 0
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT SUM(records) FROM datasets')
            result = cursor.fetchone()[0]
            return result if result else 0
            
        except Exception as e:
            logger.error(f"Error counting records: {e}")
            return 0
    
    def get_last_generated_date(self) -> Optional[str]:
        """Get last generated date."""
        try:
            if not self.connection:
                return None
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT generated_at FROM datasets ORDER BY generated_at DESC LIMIT 1')
            row = cursor.fetchone()
            
            if row:
                return row[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving last generated date: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Global instance for compatibility with existing code
db_manager = StockMarketDatabase()
