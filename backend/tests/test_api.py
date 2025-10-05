#!/usr/bin/env python3
"""
Unit tests for Flask API endpoints in FinTech DataGen.

This module tests all critical API endpoints including:
- Health check endpoints
- Data generation endpoints
- Forecasting endpoints
- Database query endpoints
- Error handling

Author: FinTech DataGen Team
Date: October 2025
"""

import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Fix TensorFlow initialization issue - set environment variable before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from datetime import datetime

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ALL ML-related modules before importing app
mock_modules = {
    'tensorflow': MagicMock(),
    'tensorflow.keras': MagicMock(),
    'tensorflow.keras.models': MagicMock(),
    'tensorflow.keras.layers': MagicMock(),
    'tensorflow.keras.optimizers': MagicMock(),
    'tensorflow.keras.callbacks': MagicMock(),
    'statsmodels': MagicMock(),
    'statsmodels.tsa': MagicMock(),
    'statsmodels.tsa.arima': MagicMock(),
    'statsmodels.tsa.arima.model': MagicMock(),
    'yfinance': MagicMock(),
    'cryptography': MagicMock(),
    'cryptography.hazmat': MagicMock(),
    'cryptography.hazmat.primitives': MagicMock(),
    'cryptography.hazmat.primitives.padding': MagicMock(),
    'database.mongodb': MagicMock(),
    'ml_models.forecasting': MagicMock(),
    'ml_models.predictor': MagicMock(),
    'ml_models.feature_engineering': MagicMock()
}

with patch.dict('sys.modules', mock_modules):
    from app import app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for Flask API endpoints."""
    
    def setUp(self):
        """Set up test client and mock data."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock database responses
        self.mock_dataset = {
            'id': 1,
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'records': 30,
            'generated_at': datetime.now().isoformat(),
            'data': json.dumps([
                {
                    'symbol': 'AAPL',
                    'exchange': 'NASDAQ',
                    'date': '2023-01-01',
                    'open_price': 100.0,
                    'high_price': 105.0,
                    'low_price': 95.0,
                    'close_price': 102.0,
                    'volume': 1000000,
                    'daily_return': 0.02,
                    'volatility': 0.05,
                    'sma_5': 101.0,
                    'sma_20': 100.5,
                    'rsi': 55.0,
                    'news_headlines': ['Apple stock rises'],
                    'news_sentiment_score': 0.3
                }
            ])
        }
        
        self.mock_prices = [
            {
                'date': '2023-01-01',
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }
        ]
        
        self.mock_prediction = {
            'id': 1,
            'symbol': 'AAPL',
            'model': 'moving_average',
            'forecast_horizon': 5,
            'predicted_values': json.dumps([103.0, 104.0, 105.0, 106.0, 107.0]),
            'metrics': json.dumps({'rmse': 1.5, 'mae': 1.2, 'mape': 1.8}),
            'created_at': datetime.now().isoformat()
        }
        
        # Create a persistent mock database
        self.mock_db = MagicMock()
        # Configure default return values
        self.mock_db.test_connection.return_value = True
        self.mock_db.count_datasets.return_value = 5
        self.mock_db.count_records.return_value = 150
        self.mock_db.get_last_generated_date.return_value = '2023-10-01 12:00:00'
        self.mock_db.get_all_datasets.return_value = [self.mock_dataset]
        self.mock_db.get_prices.return_value = self.mock_prices
        self.mock_db.get_predictions.return_value = [self.mock_prediction]
        self.mock_db.get_dataset_by_id.return_value = self.mock_dataset
        self.mock_db.get_recent_datasets.return_value = [self.mock_dataset]
        self.mock_db.get_recent_predictions.return_value = [self.mock_prediction]
        self.mock_db.get_latest_data.return_value = {'data': [self.mock_dataset]}
        self.mock_db.save_prediction.return_value = None
        self.mock_db.save_forecast.return_value = 1
        
        # Apply the mock globally
        self.db_patcher = patch('app.db', self.mock_db)
        self.db_patcher.start()
        
        # Mock predictor
        self.predictor_patcher = patch('app.predictor')
        self.mock_predictor = self.predictor_patcher.start()
        self.mock_predictor.calculate_accuracy.return_value = 85.5
        self.mock_predictor.predict.return_value = {
            'predicted_price': 105.0,
            'confidence': 0.8,
            'current_price': 102.0,
            'change_percent': 2.94
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.db_patcher.stop()
        self.predictor_patcher.stop()
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        print("\n=== Testing Health Check Endpoint ===")
        
        response = self.client.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['database'], 'connected')
        self.assertIn('timestamp', data)
        self.assertIn('stats', data)
        
        print("Health check endpoint working correctly")
    
    def test_health_check_no_database(self):
        """Test health check when database is not available."""
        print("\n=== Testing Health Check (No Database) ===")
        
        # Temporarily set db to None
        with patch('app.db', None):
            response = self.client.get('/api/health')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertEqual(data['status'], 'healthy')
            self.assertIn('database', data)
            self.assertEqual(data['stats']['totalDatasets'], 0)
            
            print("Health check handles no database correctly")
    
    @patch('app.StockDataCurator')
    @patch('app.db')
    def test_generate_data_endpoint(self, mock_db, mock_curator):
        """Test data generation endpoint."""
        print("\n=== Testing Data Generation Endpoint ===")
        
        # Mock curator
        mock_curator_instance = MagicMock()
        mock_curator.return_value = mock_curator_instance
        
        # Mock curated data
        mock_market_data = MagicMock()
        mock_market_data.symbol = 'AAPL'
        mock_market_data.exchange = 'NASDAQ'
        mock_market_data.date = '2023-01-01'
        mock_market_data.open_price = 100.0
        mock_market_data.high_price = 105.0
        mock_market_data.low_price = 95.0
        mock_market_data.close_price = 102.0
        mock_market_data.volume = 1000000
        mock_market_data.daily_return = 0.02
        mock_market_data.volatility = 0.05
        mock_market_data.sma_5 = 101.0
        mock_market_data.sma_20 = 100.5
        mock_market_data.rsi = 55.0
        mock_market_data.news_headlines = ['Apple stock rises']
        mock_market_data.news_sentiment_score = 0.3
        
        mock_curator_instance.curate_dataset.return_value = [mock_market_data]
        
        # Mock database save
        mock_db.save_dataset.return_value = 1
        mock_db.save_historical_prices.return_value = None
        
        # Test request
        request_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'days': 7
        }
        
        response = self.client.post('/api/generate', 
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertEqual(data['exchange'], 'NASDAQ')
        self.assertEqual(data['days'], 7)
        
        print("Data generation endpoint working correctly")
    
    def test_generate_data_missing_fields(self):
        """Test data generation with missing required fields."""
        print("\n=== Testing Data Generation (Missing Fields) ===")
        
        # Test missing symbol
        request_data = {
            'exchange': 'NASDAQ',
            'days': 7
        }
        
        response = self.client.post('/api/generate',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing fields handled correctly")
    
    def test_generate_data_invalid_days(self):
        """Test data generation with invalid days parameter."""
        print("\n=== Testing Data Generation (Invalid Days) ===")
        
        # Test negative days
        request_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'days': -1
        }
        
        response = self.client.post('/api/generate',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Invalid days handled correctly")
    
    def test_get_prices_endpoint(self):
        """Test get prices endpoint."""
        print("\n=== Testing Get Prices Endpoint ===")
        
        response = self.client.get('/api/prices?symbol=AAPL&limit=100')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertIn('rows', data)
        self.assertGreaterEqual(len(data['rows']), 1)
        
        print("Get prices endpoint working correctly")
    
    def test_get_prices_missing_symbol(self):
        """Test get prices without symbol parameter."""
        print("\n=== Testing Get Prices (Missing Symbol) ===")
        
        response = self.client.get('/api/prices')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing symbol handled correctly")
    
    def test_get_predictions_endpoint(self):
        """Test get predictions endpoint."""
        print("\n=== Testing Get Predictions Endpoint ===")
        
        response = self.client.get('/api/predictions?symbol=AAPL&limit=10')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]['symbol'], 'AAPL')
        
        print("Get predictions endpoint working correctly")
    
    def test_get_predictions_by_models(self):
        """Test get predictions by models endpoint."""
        print("\n=== Testing Get Predictions by Models ===")
        
        response = self.client.get('/api/predictions/by-models?symbol=AAPL&models=ma,arima')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIsInstance(data, list)
        
        print("Get predictions by models working correctly")
    
    def test_get_predictions_by_models_missing_symbol(self):
        """Test get predictions by models without symbol."""
        print("\n=== Testing Get Predictions by Models (Missing Symbol) ===")
        
        response = self.client.get('/api/predictions/by-models')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing symbol handled correctly")
    
    def test_post_prediction_endpoint(self):
        """Test post prediction endpoint."""
        print("\n=== Testing Post Prediction Endpoint ===")
        
        request_data = {
            'symbol': 'AAPL',
            'model': 'moving_average',
            'forecast_horizon': 5,
            'predicted_values': [103.0, 104.0, 105.0, 106.0, 107.0],
            'metrics': {'rmse': 1.5, 'mae': 1.2, 'mape': 1.8}
        }
        
        response = self.client.post('/api/predictions',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        
        self.assertIn('id', data)
        
        print("Post prediction endpoint working correctly")
    
    def test_post_prediction_missing_fields(self):
        """Test post prediction with missing required fields."""
        print("\n=== Testing Post Prediction (Missing Fields) ===")
        
        request_data = {
            'symbol': 'AAPL',
            'model': 'moving_average'
            # Missing forecast_horizon and predicted_values
        }
        
        response = self.client.post('/api/predictions',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing fields handled correctly")
    
    def test_get_datasets_endpoint(self):
        """Test get datasets endpoint."""
        print("\n=== Testing Get Datasets Endpoint ===")
        
        response = self.client.get('/api/datasets')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]['symbol'], 'AAPL')
        
        print("Get datasets endpoint working correctly")
    
    def test_get_dataset_by_id(self):
        """Test get dataset by ID endpoint."""
        print("\n=== Testing Get Dataset by ID ===")
        
        response = self.client.get('/api/datasets/1')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertEqual(data['exchange'], 'NASDAQ')
        
        print("Get dataset by ID working correctly")
    
    def test_get_dataset_not_found(self):
        """Test get dataset when not found."""
        print("\n=== Testing Get Dataset (Not Found) ===")
        
        # Temporarily configure mock to return None for this test
        self.mock_db.get_dataset_by_id.return_value = None
        
        response = self.client.get('/api/datasets/nonexistent_id')
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        # Reset mock for other tests
        self.mock_db.get_dataset_by_id.return_value = self.mock_dataset
        
        print("Dataset not found handled correctly")
    
    def test_analytics_endpoint(self):
        """Test analytics endpoint."""
        print("\n=== Testing Analytics Endpoint ===")
        
        response = self.client.get('/api/analytics')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('datasets', data)
        self.assertIn('predictions', data)
        self.assertIn('accuracy', data)
        
        print("Analytics endpoint working correctly")
    
    def test_predict_endpoint(self):
        """Test predict endpoint."""
        print("\n=== Testing Predict Endpoint ===")
        
        request_data = {'symbol': 'AAPL'}
        
        response = self.client.post('/api/predict',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        
        print("Predict endpoint working correctly")
    
    def test_predict_missing_symbol(self):
        """Test predict endpoint without symbol."""
        print("\n=== Testing Predict (Missing Symbol) ===")
        
        request_data = {}
        
        response = self.client.post('/api/predict',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing symbol handled correctly")

class TestPublicEndpoints(unittest.TestCase):
    """Test public endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Set up mock for public endpoints
        self.mock_prices = [
            {
                'date': '2023-01-01',
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }
        ]
        
        # Create mock database
        self.mock_db = MagicMock()
        self.mock_db.get_prices.return_value = self.mock_prices
        
        # Apply the mock globally
        self.db_patcher = patch('app.db', self.mock_db)
        self.db_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.db_patcher.stop()
    
    def test_get_historical_public(self):
        """Test public historical endpoint."""
        print("\n=== Testing Public Historical Endpoint ===")
        
        response = self.client.get('/get_historical?symbol=AAPL&limit=100')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self
    
    def test_get_historical_missing_symbol(self):
        """Test public historical without symbol."""
        print("\n=== Testing Public Historical (Missing Symbol) ===")
        
        response = self.client.get('/get_historical')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        print("Missing symbol handled correctly")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)