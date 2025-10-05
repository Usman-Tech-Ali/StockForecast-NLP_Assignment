#!/usr/bin/env python3
"""
Unit tests for ML forecasting models in FinTech DataGen.

This module tests all critical ML models including:
- Moving Average Forecaster
- ARIMA Forecaster  
- LSTM Forecaster
- Transformer Forecaster
- Ensemble Average Forecaster

Author: FinTech DataGen Team
Date: October 2025
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.forecasting import (
    MovingAverageForecaster,
    ARIMAForecaster, 
    LSTMForecaster,
    TransformerForecaster,
    EnsembleAverageForecaster,
    compute_performance_metrics,
    split_time_series
)

class TestForecastingModels(unittest.TestCase):
    """Test suite for all forecasting models."""
    
    def setUp(self):
        """Set up test data for all models."""
        # Create synthetic financial time series data
        np.random.seed(42)  # For reproducible tests
        
        # Generate 100 days of synthetic price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create realistic price series with trend and volatility
        base_price = 100
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)  # Random noise
        prices = base_price + trend + noise
        
        self.test_series = pd.Series(prices, index=dates)
        self.train_series, self.test_series_split = split_time_series(self.test_series)
        
        print(f"Test data: {len(self.test_series)} total points")
        print(f"Train data: {len(self.train_series)} points")
        print(f"Test data: {len(self.test_series_split)} points")
    
    def test_moving_average_forecaster(self):
        """Test Moving Average Forecaster functionality."""
        print("\n=== Testing Moving Average Forecaster ===")
        
        # Test different window sizes
        for window in [3, 5, 10]:
            with self.subTest(window=window):
                ma_model = MovingAverageForecaster(window=window)
                
                # Test fitting
                ma_model.fit(self.train_series)
                self.assertIsNotNone(ma_model.history)
                self.assertEqual(len(ma_model.history), len(self.train_series))
                
                # Test prediction
                horizon = 5
                predictions = ma_model.predict(horizon)
                self.assertEqual(len(predictions), horizon)
                self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
                
                # Test evaluation
                eval_result = ma_model.evaluate(self.test_series_split)
                self.assertIn('metrics', eval_result)
                self.assertIn('predicted_values', eval_result)
                self.assertIn('y_true', eval_result)
                self.assertIn('forecast_horizon', eval_result)
                
                # Check metrics are reasonable
                metrics = eval_result['metrics']
                self.assertIn('rmse', metrics)
                self.assertIn('mae', metrics)
                self.assertIn('mape', metrics)
                self.assertGreater(metrics['rmse'], 0)
                self.assertGreater(metrics['mae'], 0)
                
                print(f"Window {window}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    def test_arima_forecaster(self):
        """Test ARIMA Forecaster functionality."""
        print("\n=== Testing ARIMA Forecaster ===")
        
        # Test different ARIMA orders
        orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
        
        for order in orders:
            with self.subTest(order=order):
                arima_model = ARIMAForecaster(order=order)
                
                # Test fitting
                arima_model.fit(self.train_series)
                self.assertIsNotNone(arima_model._fit_result)
                
                # Test prediction
                horizon = 5
                predictions = arima_model.predict(horizon)
                self.assertEqual(len(predictions), horizon)
                self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
                
                # Test evaluation
                eval_result = arima_model.evaluate(self.test_series_split)
                self.assertIn('metrics', eval_result)
                
                metrics = eval_result['metrics']
                self.assertIn('rmse', metrics)
                self.assertIn('mae', metrics)
                self.assertIn('mape', metrics)
                
                print(f"ARIMA{order}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    def test_lstm_forecaster(self):
        """Test LSTM Forecaster functionality."""
        print("\n=== Testing LSTM Forecaster ===")
        
        # Test with minimal epochs for faster testing
        lstm_model = LSTMForecaster(lookback=5, epochs=5, batch_size=8)
        
        # Test fitting
        lstm_model.fit(self.train_series)
        self.assertIsNotNone(lstm_model.model)
        
        # Test prediction
        horizon = 5
        predictions = lstm_model.predict(horizon)
        self.assertEqual(len(predictions), horizon)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
        # Test evaluation
        eval_result = lstm_model.evaluate(self.test_series_split)
        self.assertIn('metrics', eval_result)
        
        metrics = eval_result['metrics']
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        
        print(f"LSTM: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    def test_transformer_forecaster(self):
        """Test Transformer Forecaster functionality."""
        print("\n=== Testing Transformer Forecaster ===")
        
        # Test with minimal parameters for faster testing
        transformer_model = TransformerForecaster(
            lookback=10, 
            d_model=16, 
            num_heads=2, 
            ff_dim=32, 
            epochs=3, 
            batch_size=8
        )
        
        # Test fitting
        transformer_model.fit(self.train_series)
        self.assertIsNotNone(transformer_model.model)
        
        # Test prediction
        horizon = 5
        predictions = transformer_model.predict(horizon)
        self.assertEqual(len(predictions), horizon)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
        # Test evaluation
        eval_result = transformer_model.evaluate(self.test_series_split)
        self.assertIn('metrics', eval_result)
        
        metrics = eval_result['metrics']
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        
        print(f"Transformer: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    def test_ensemble_forecaster(self):
        """Test Ensemble Average Forecaster functionality."""
        print("\n=== Testing Ensemble Average Forecaster ===")
        
        # Create individual forecasters
        ma_model = MovingAverageForecaster(window=5)
        arima_model = ARIMAForecaster(order=(1, 1, 1))
        
        # Create ensemble
        ensemble_model = EnsembleAverageForecaster([ma_model, arima_model])
        
        # Test fitting
        ensemble_model.fit(self.train_series)
        
        # Test prediction
        horizon = 5
        predictions = ensemble_model.predict(horizon)
        self.assertEqual(len(predictions), horizon)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
        # Test evaluation
        eval_result = ensemble_model.evaluate(self.test_series_split)
        self.assertIn('metrics', eval_result)
        
        metrics = eval_result['metrics']
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        
        print(f"Ensemble: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    def test_compute_performance_metrics(self):
        """Test metrics calculation function."""
        print("\n=== Testing Metrics Calculation ===")
        
        # Create test data
        y_true = np.array([100, 105, 110, 115, 120])
        y_pred = np.array([102, 107, 108, 117, 118])
        
        metrics = compute_performance_metrics(y_true, y_pred)
        
        # Check all required metrics are present
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        
        # Check metrics are positive
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['mape'], 0)
        
        print(f"Test metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}")
    
    def test_split_time_series(self):
        """Test train-test split functionality."""
        print("\n=== Testing Train-Test Split ===")
        
        # Test with different test sizes
        for test_size in [5, 10, 15]:
            with self.subTest(test_size=test_size):
                train, test = split_time_series(self.test_series, test_size=test_size)
                
                # Check lengths
                self.assertEqual(len(test), test_size)
                self.assertEqual(len(train), len(self.test_series) - test_size)
                
                # Check no overlap
                train_end = train.index[-1]
                test_start = test.index[0]
                self.assertGreater(test_start, train_end)
                
                print(f"Test size {test_size}: Train={len(train)}, Test={len(test)}")
    
    def test_model_performance_comparison(self):
        """Compare performance of all models."""
        print("\n=== Model Performance Comparison ===")
        
        models = {
            'Moving Average': MovingAverageForecaster(window=5),
            'ARIMA': ARIMAForecaster(order=(1, 1, 1)),
            'LSTM': LSTMForecaster(lookback=5, epochs=3),
            'Transformer': TransformerForecaster(lookback=5, epochs=2, d_model=16)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(self.train_series)
                eval_result = model.evaluate(self.test_series_split)
                results[name] = eval_result['metrics']
                print(f"{name}: RMSE={eval_result['metrics']['rmse']:.4f}")
            except Exception as e:
                print(f"{name}: Failed - {str(e)}")
                results[name] = None
        
        # Ensure at least some models worked
        successful_models = [name for name, result in results.items() if result is not None]
        self.assertGreater(len(successful_models), 0, "At least one model should work")
        
        print(f"\nSuccessful models: {successful_models}")

class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up minimal test data."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        prices = [100 + i + np.random.normal(0, 1) for i in range(20)]
        self.short_series = pd.Series(prices, index=dates)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        print("\n=== Testing Insufficient Data Handling ===")
        
        # Very short series
        short_dates = pd.date_range('2023-01-01', periods=3, freq='D')
        short_prices = [100, 101, 102]
        very_short_series = pd.Series(short_prices, index=short_dates)
        
        # Moving Average should handle this gracefully
        ma_model = MovingAverageForecaster(window=2)
        ma_model.fit(very_short_series)
        predictions = ma_model.predict(2)
        self.assertEqual(len(predictions), 2)
        
        print("Short series handled successfully")
    
    def test_empty_series(self):
        """Test behavior with empty series."""
        print("\n=== Testing Empty Series Handling ===")
        
        empty_series = pd.Series([], dtype=float)
        
        # This should raise an exception
        ma_model = MovingAverageForecaster(window=5)
        with self.assertRaises(Exception):
            ma_model.fit(empty_series)
        
        print("Empty series properly rejected")
    
    def test_constant_series(self):
        """Test behavior with constant price series."""
        print("\n=== Testing Constant Series Handling ===")
        
        # Constant price series
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        constant_prices = [100] * 10
        constant_series = pd.Series(constant_prices, index=dates)
        
        # Moving Average should handle this
        ma_model = MovingAverageForecaster(window=3)
        ma_model.fit(constant_series)
        predictions = ma_model.predict(3)
        self.assertEqual(len(predictions), 3)
        
        print("Constant series handled successfully")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
