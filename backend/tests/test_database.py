#!/usr/bin/env python3
"""
Unit tests for SQLite database operations in Stock Market Forecasting System.

This module tests all critical database operations including:
- Connection management
- Dataset operations
- Prediction operations
- Historical price operations
- Metadata operations
- Error handling

Author: Stock Market Forecasting Team
Date: October 2025
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
import tempfile
import json

# Fix TensorFlow initialization issue if it appears
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.sqlite_manager import StockMarketDatabase

class TestSQLiteConnection(unittest.TestCase):
    """Test SQLite database connection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db_path = "test_stock_forecasting.db"
    
    @patch('database.sqlite_manager.sqlite3.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful SQLite database connection."""
        print("\n=== Testing Successful SQLite Connection ===")
        
        # Mock successful connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        db = StockMarketDatabase(self.test_db_path)
        
        self.assertIsNotNone(db.connection)
        self.assertTrue(db.test_connection())
        
        print("SQLite connection successful")
    
    @patch('database.sqlite_manager.sqlite3.connect')
    def test_connection_failure(self, mock_connect):
        """Test SQLite connection failure."""
        print("\n=== Testing SQLite Connection Failure ===")
        
        # Mock connection failure
        mock_connect.side_effect = Exception("Connection failed")
        
        db = StockMarketDatabase(self.test_db_path)
        
        self.assertIsNone(db.connection)
        self.assertFalse(db.test_connection())
        
        print("SQLite connection failure handled correctly")
    
    def test_database_initialization(self):
        """Test database initialization with tables."""
        print("\n=== Testing Database Initialization ===")
        
        with patch('database.sqlite_manager.sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            db = StockMarketDatabase(self.test_db_path)
            # SQLite database should initialize tables
            self.assertIsNotNone(db.connection)
            print("Database initialization successful")

class TestSQLiteOperations(unittest.TestCase):
    """Test SQLite database operations."""
    
    def setUp(self):
        """Set up mock SQLite database instance."""
        self.mock_db = StockMarketDatabase("test.db")
        
        # Mock database components
        self.mock_conn = MagicMock()
        self.mock_db.connection = self.mock_conn
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
    
    def test_save_dataset(self):
        """Test saving dataset to database."""
        print("\n=== Testing Save Dataset ===")
        
        dataset_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'days': 30,
            'records': 30,
            'generated_at': datetime.now(),
            'data': []
        }
        
        self.mock_cursor.lastrowid = 1
        
        result = self.mock_db.save_dataset(dataset_data)
        
        self.mock_cursor.execute.assert_called()
        self.mock_conn.commit.assert_called_once()
        self.assertEqual(result, 1)
        
        print("Save dataset working correctly")
    
    def test_save_dataset_no_connection(self):
        """Test save dataset when no database connection."""
        print("\n=== Testing Save Dataset (No Connection) ===")
        
        self.mock_db.connection = None
        
        result = self.mock_db.save_dataset({})
        
        self.assertIsNone(result)
        print("No connection handled correctly")
    
    def test_get_dataset_by_id(self):
        """Test getting dataset by ID."""
        print("\n=== Testing Get Dataset by ID ===")
        
        mock_dataset = {
            'id': 1,
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'records': 30,
            'data': '[]'
        }
        
        self.mock_cursor.fetchone.return_value = mock_dataset
        
        result = self.mock_db.get_dataset_by_id(1)
        
        self.assertEqual(result['symbol'], mock_dataset['symbol'])
        self.mock_cursor.execute.assert_called_once()
        
        print("Get dataset by ID working correctly")
    
    def test_get_all_datasets(self):
        """Test getting all datasets."""
        print("\n=== Testing Get All Datasets ===")
        
        mock_datasets = [
            {'id': 1, 'symbol': 'AAPL', 'data': '[]'},
            {'id': 2, 'symbol': 'MSFT', 'data': '[]'}
        ]
        
        self.mock_cursor.fetchall.return_value = mock_datasets
        
        result = self.mock_db.get_all_datasets()
        
        self.assertEqual(len(result), 2)
        self.mock_cursor.execute.assert_called_once()
        
        print("Get all datasets working correctly")
    
    def test_save_historical_prices(self):
        """Test saving historical prices."""
        print("\n=== Testing Save Historical Prices ===")
        
        prices = [
            {
                'date': '2023-01-01',
                'open_price': 100.0,
                'high_price': 105.0,
                'low_price': 95.0,
                'close_price': 102.0,
                'volume': 1000000
            }
        ]
        
        self.mock_cursor.rowcount = 1
        
        result = self.mock_db.save_historical_prices('AAPL', 'NASDAQ', prices)
        
        self.mock_cursor.execute.assert_called()
        self.mock_conn.commit.assert_called_once()
        self.assertEqual(result, 1)
        
        print("Save historical prices working correctly")
    
    def test_get_prices(self):
        """Test getting prices."""
        print("\n=== Testing Get Prices ===")
        
        mock_prices = [
            {
                'symbol': 'AAPL',
                'date': '2023-01-01',
                'open_price': 100.0,
                'high_price': 105.0,
                'low_price': 98.0,
                'close_price': 102.0,
                'volume': 1000000
            }
        ]
        
        self.mock_cursor.fetchall.return_value = mock_prices
        
        result = self.mock_db.get_prices('AAPL')
        
        self.assertEqual(len(result), 1)
        self.mock_cursor.execute.assert_called_once()
        
        print("Get prices working correctly")
    
    def test_save_forecast(self):
        """Test saving forecast."""
        print("\n=== Testing Save Forecast ===")
        
        forecast_data = {
            'symbol': 'AAPL',
            'model': 'ARIMA',
            'forecast_horizon': 10,
            'predicted_values': [100, 101, 102],
            'metrics': {'rmse': 0.5},
            'y_true': [99, 100, 101],
            'created_at': datetime.now()
        }
        
        self.mock_cursor.lastrowid = 1
        
        result = self.mock_db.save_forecast(forecast_data)
        
        self.mock_cursor.execute.assert_called()
        self.mock_conn.commit.assert_called_once()
        self.assertEqual(result, 1)
        
        print("Save forecast working correctly")
    
    def test_get_predictions(self):
        """Test getting predictions."""
        print("\n=== Testing Get Predictions ===")
        
        mock_predictions = [
            {
                'id': 1,
                'symbol': 'AAPL',
                'model': 'ARIMA',
                'forecast_horizon': 10,
                'predicted_values': '[100, 101, 102]',
                'metrics': '{"rmse": 0.5}',
                'y_true': '[99, 100, 101]',
                'created_at': '2023-01-01'
            }
        ]
        
        self.mock_cursor.fetchall.return_value = mock_predictions
        
        result = self.mock_db.get_predictions()
        
        self.assertEqual(len(result), 1)
        self.mock_cursor.execute.assert_called_once()
        
        print("Get predictions working correctly")
    
    def test_upsert_metadata(self):
        """Test upserting metadata."""
        print("\n=== Testing Upsert Metadata ===")
        
        metadata = {
            'instrument_info': {'symbol': 'AAPL'},
            'data_sources': {'source': 'yahoo'},
            'update_logs': [{'action': 'test'}]
        }
        
        # Mock existing metadata check
        self.mock_cursor.fetchone.return_value = None  # No existing metadata
        
        self.mock_cursor.lastrowid = 1
        
        result = self.mock_db.upsert_metadata('AAPL', metadata)
        
        self.mock_cursor.execute.assert_called()
        self.mock_conn.commit.assert_called_once()
        
        print("Upsert metadata working correctly")
    
    def test_get_metadata(self):
        """Test getting metadata."""
        print("\n=== Testing Get Metadata ===")
        
        mock_metadata = {
            'symbol': 'AAPL',
            'instrument_info': '{"symbol": "AAPL"}',
            'data_sources': '{"source": "yahoo"}',
            'update_logs': '[{"action": "test"}]',
            'last_updated': '2023-01-01',
            'created_at': '2023-01-01'
        }
        
        self.mock_cursor.fetchone.return_value = mock_metadata
        
        result = self.mock_db.get_metadata('AAPL')
        
        self.assertIsNotNone(result)
        self.mock_cursor.execute.assert_called_once()
        
        print("Get metadata working correctly")
    
    def test_count_datasets(self):
        """Test counting datasets."""
        print("\n=== Testing Count Datasets ===")
        
        self.mock_cursor.fetchone.return_value = [5]
        
        result = self.mock_db.count_datasets()
        
        self.assertEqual(result, 5)
        self.mock_cursor.execute.assert_called_once()
        
        print("Count datasets working correctly")
    
    def test_count_records(self):
        """Test counting records."""
        print("\n=== Testing Count Records ===")
        
        self.mock_cursor.fetchone.return_value = [150]
        
        result = self.mock_db.count_records()
        
        self.assertEqual(result, 150)
        self.mock_cursor.execute.assert_called_once()
        
        print("Count records working correctly")
    
    def test_get_last_generated_date(self):
        """Test getting last generated date."""
        print("\n=== Testing Get Last Generated Date ===")
        
        self.mock_cursor.fetchone.return_value = ['2023-01-01']
        
        result = self.mock_db.get_last_generated_date()
        
        self.assertEqual(result, '2023-01-01')
        self.mock_cursor.execute.assert_called_once()
        
        print("Get last generated date working correctly")

class TestSQLiteIntegration(unittest.TestCase):
    """Integration tests for SQLite database."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.db = StockMarketDatabase(self.test_db_path)
    
    def tearDown(self):
        """Clean up test database."""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
            if os.path.exists(self.test_db_path):
                # Wait a bit for file handles to be released
                import time
                time.sleep(0.1)
                os.remove(self.test_db_path)
        except (PermissionError, OSError):
            # File might be locked, skip cleanup
            pass
    
    def test_full_workflow(self):
        """Test complete database workflow."""
        print("\n=== Testing Full Database Workflow ===")
        
        # Test dataset operations
        dataset_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'days': 30,
            'records': 30,
            'generated_at': datetime.now(),
            'data': []
        }
        
        dataset_id = self.db.save_dataset(dataset_data)
        self.assertIsNotNone(dataset_id)
        
        # Test historical prices
        prices = [
            {
                'date': '2023-01-01',
                'open_price': 100.0,
                'high_price': 105.0,
                'low_price': 95.0,
                'close_price': 102.0,
                'volume': 1000000
            }
        ]
        
        result = self.db.save_historical_prices('AAPL', 'NASDAQ', prices)
        self.assertEqual(result, 1)
        
        # Test forecast operations
        forecast_data = {
            'symbol': 'AAPL',
            'model': 'ARIMA',
            'forecast_horizon': 10,
            'predicted_values': [100, 101, 102],
            'metrics': {'rmse': 0.5},
            'y_true': [99, 100, 101],
            'created_at': datetime.now()
        }
        
        forecast_id = self.db.save_forecast(forecast_data)
        self.assertIsNotNone(forecast_id)
        
        # Test metadata operations
        metadata = {
            'instrument_info': {'symbol': 'AAPL'},
            'data_sources': {'source': 'yahoo'},
            'update_logs': [{'action': 'test'}]
        }
        
        metadata_result = self.db.upsert_metadata('AAPL', metadata)
        self.assertIsNotNone(metadata_result)
        
        print("Full database workflow working correctly")

def run_database_tests():
    """Run all database tests."""
    print("=" * 60)
    print("STOCK MARKET FORECASTING SYSTEM - DATABASE TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add connection tests
    test_suite.addTest(unittest.makeSuite(TestSQLiteConnection))
    
    # Add operation tests
    test_suite.addTest(unittest.makeSuite(TestSQLiteOperations))
    
    # Add integration tests
    test_suite.addTest(unittest.makeSuite(TestSQLiteIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATABASE TESTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_database_tests()
    sys.exit(0 if success else 1)