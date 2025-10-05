"""
Forecasting models module for stock forecasting app.
Implements ARIMA, LSTM, and ensemble models for AAPL stock prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Traditional models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Neural models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Metrics and utilities
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging

# Import our modules
from db import get_latest_prices, insert_prediction, insert_model_metrics, get_model_metrics
from data import get_historical_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockForecaster:
    """Main class for stock forecasting with multiple models."""
    
    def __init__(self):
        self.arima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.horizon_mapping = {
            '1hr': 1,    # 1 day for daily data
            '3hr': 1,    # 1 day for daily data  
            '24hr': 1,   # 1 day for daily data
            '72hr': 3    # 3 days for daily data
        }
        
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """
        Prepare data for model training and testing.
        
        Args:
            data (pd.DataFrame): Historical price data
            test_size (float): Proportion of data for testing
        
        Returns:
            tuple: (train_data, test_data, train_close, test_close)
        """
        try:
            # Use close prices for forecasting
            close_prices = data['close'].values
            
            # Split data
            split_idx = int(len(close_prices) * (1 - test_size))
            train_close = close_prices[:split_idx]
            test_close = close_prices[split_idx:]
            
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            logger.info(f"Data split: {len(train_close)} training, {len(test_close)} testing samples")
            
            return train_data, test_data, train_close, test_close
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def check_stationarity(self, data: pd.Series) -> bool:
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data (pd.Series): Time series data
        
        Returns:
            bool: True if stationary, False otherwise
        """
        try:
            result = adfuller(data.dropna())
            p_value = result[1]
            
            # If p-value < 0.05, series is stationary
            is_stationary = p_value < 0.05
            
            logger.info(f"ADF test p-value: {p_value:.4f}, Stationary: {is_stationary}")
            return is_stationary
            
        except Exception as e:
            logger.error(f"Error checking stationarity: {e}")
            return False
    
    def train_arima(self, train_close: np.ndarray) -> None:
        """
        Train ARIMA model on historical close prices.
        Using order=(5,1,0) - simple configuration for trending stock data.
        
        Args:
            train_close (np.ndarray): Training close prices
        """
        try:
            logger.info("Training ARIMA model...")
            
            # Fit ARIMA model with order (5,1,0)
            # 5: AR terms, 1: differencing for stationarity, 0: MA terms
            # This is a simple configuration suitable for trending stock data
            self.arima_model = ARIMA(train_close, order=(5, 1, 0))
            self.arima_model = self.arima_model.fit()
            
            logger.info("ARIMA model trained successfully")
            
            # Save model
            joblib.dump(self.arima_model, 'arima_model.pkl')
            logger.info("ARIMA model saved to arima_model.pkl")
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            raise
    
    def forecast_arima(self, horizon_steps: int, data: np.ndarray = None) -> np.ndarray:
        """
        Generate ARIMA forecast for specified horizon.
        
        Args:
            horizon_steps (int): Number of steps to forecast
            data (np.ndarray, optional): Data to use for forecasting
        
        Returns:
            np.ndarray: Forecast values
        """
        try:
            if self.arima_model is None:
                raise ValueError("ARIMA model not trained. Call train_arima() first.")
            
            # Generate forecast
            forecast = self.arima_model.forecast(steps=horizon_steps)
            
            logger.info(f"ARIMA forecast generated for {horizon_steps} steps")
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise
    
    def prepare_lstm_data(self, data: np.ndarray, lookback: int = 60) -> tuple:
        """
        Prepare data for LSTM training with sliding window.
        
        Args:
            data (np.ndarray): Time series data
            lookback (int): Number of previous time steps to use
        
        Returns:
            tuple: (X, y) arrays for LSTM training
        """
        try:
            # Scale data
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
            
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            raise
    
    def train_lstm(self, train_close: np.ndarray, lookback: int = 60) -> None:
        """
        Train LSTM model on historical close prices.
        Using 1 layer, 50 units - captures non-linear patterns in volatile stock data.
        
        Args:
            train_close (np.ndarray): Training close prices
            lookback (int): Number of previous time steps to use
        """
        try:
            logger.info("Training LSTM model...")
            
            # Prepare data for LSTM
            X, y = self.prepare_lstm_data(train_close, lookback)
            
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=False, input_shape=(lookback, 1)),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Compile model
            self.lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            # Using 50 epochs and batch size 32 - reasonable for stock data
            # This captures non-linear patterns in volatile stock data
            history = self.lstm_model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            logger.info("LSTM model trained successfully")
            
            # Save model and scaler
            self.lstm_model.save('lstm_model.h5')
            joblib.dump(self.scaler, 'lstm_scaler.pkl')
            logger.info("LSTM model saved to lstm_model.h5")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def forecast_lstm(self, horizon_steps: int, last_sequence: np.ndarray = None) -> np.ndarray:
        """
        Generate LSTM forecast for specified horizon.
        
        Args:
            horizon_steps (int): Number of steps to forecast
            last_sequence (np.ndarray, optional): Last sequence for forecasting
        
        Returns:
            np.ndarray: Forecast values
        """
        try:
            if self.lstm_model is None:
                raise ValueError("LSTM model not trained. Call train_lstm() first.")
            
            # Load scaler if not already loaded
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = joblib.load('lstm_scaler.pkl')
            
            forecasts = []
            current_sequence = last_sequence.copy() if last_sequence is not None else None
            
            for _ in range(horizon_steps):
                if current_sequence is None:
                    raise ValueError("Last sequence required for LSTM forecasting")
                
                # Scale the sequence
                scaled_sequence = self.scaler.transform(current_sequence.reshape(-1, 1))
                
                # Reshape for LSTM input
                X = scaled_sequence.reshape(1, len(scaled_sequence), 1)
                
                # Make prediction
                prediction = self.lstm_model.predict(X, verbose=0)
                
                # Inverse transform
                forecast_value = self.scaler.inverse_transform(prediction)[0, 0]
                forecasts.append(forecast_value)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], forecast_value)
            
            logger.info(f"LSTM forecast generated for {horizon_steps} steps")
            return np.array(forecasts)
            
        except Exception as e:
            logger.error(f"Error generating LSTM forecast: {e}")
            raise
    
    def forecast_ensemble(self, horizon: str, data: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble forecast combining ARIMA and LSTM.
        
        Args:
            horizon (str): Forecast horizon (e.g., '24hr')
            data (pd.DataFrame): Historical data
        
        Returns:
            np.ndarray: Ensemble forecast values
        """
        try:
            horizon_steps = self.horizon_mapping.get(horizon, 1)
            
            # Get ARIMA forecast
            arima_forecast = self.forecast_arima(horizon_steps, data['close'].values)
            
            # Get LSTM forecast (need last 60 values for sequence)
            if len(data) >= 60:
                last_sequence = data['close'].tail(60).values
                lstm_forecast = self.forecast_lstm(horizon_steps, last_sequence)
            else:
                # Fallback to ARIMA if not enough data for LSTM
                lstm_forecast = arima_forecast
            
            # Simple average ensemble
            ensemble_forecast = (arima_forecast + lstm_forecast) / 2
            
            logger.info(f"Ensemble forecast generated for horizon {horizon}")
            return ensemble_forecast
            
        except Exception as e:
            logger.error(f"Error generating ensemble forecast: {e}")
            raise
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
        """
        Evaluate model performance using RMSE, MAE, and MAPE.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
        
        Returns:
            dict: Performance metrics
        """
        try:
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            metrics = {
                'model': model_name,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            logger.info(f"{model_name} metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")
            
            # Store metrics in database
            insert_model_metrics(model_name, rmse, mae, mape)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def test_models(self) -> None:
        """
        Test all models and generate sample predictions.
        """
        try:
            logger.info("Starting model testing...")
            
            # Get historical data
            data = get_historical_data()
            
            # Prepare data
            train_data, test_data, train_close, test_close = self.prepare_data(data)
            
            # Train models
            self.train_arima(train_close)
            self.train_lstm(train_close)
            
            # Generate predictions for testing
            arima_pred = self.forecast_arima(len(test_close), train_close)
            lstm_pred = self.forecast_lstm(len(test_close), train_close[-60:])
            ensemble_pred = self.forecast_ensemble('24hr', train_data)
            
            # Evaluate models
            arima_metrics = self.evaluate_model(test_close, arima_pred, 'ARIMA')
            lstm_metrics = self.evaluate_model(test_close, lstm_pred, 'LSTM')
            
            # Print results
            print("\n" + "="*50)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*50)
            
            results_df = pd.DataFrame([arima_metrics, lstm_metrics])
            print(results_df.to_string(index=False))
            
            print("\n" + "="*50)
            print("MODEL JUSTIFICATIONS")
            print("="*50)
            print("ARIMA: Suitable for stationary time series with linear trends")
            print("LSTM: Captures non-linear patterns and sequential dependencies")
            print("Ensemble: Combines strengths of both models for robust predictions")
            
            # Generate sample predictions for different horizons
            print("\n" + "="*50)
            print("SAMPLE PREDICTIONS")
            print("="*50)
            
            for horizon in ['24hr', '72hr']:
                try:
                    forecast = self.forecast_ensemble(horizon, data)
                    forecast_date = (datetime.now() + timedelta(days=self.horizon_mapping[horizon])).strftime('%Y-%m-%d')
                    
                    # Insert prediction to database
                    insert_prediction(
                        date=forecast_date,
                        horizon=horizon,
                        model_name='ENSEMBLE',
                        forecast_value=float(forecast[0])
                    )
                    
                    print(f"{horizon} forecast: ${forecast[0]:.2f} (Date: {forecast_date})")
                    
                except Exception as e:
                    logger.error(f"Error generating {horizon} prediction: {e}")
            
            print("\nModel testing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in model testing: {e}")
            raise


def main():
    """Main function to run the forecasting pipeline."""
    try:
        # Initialize forecaster
        forecaster = StockForecaster()
        
        # Test models
        forecaster.test_models()
        
        # Show recent predictions
        print("\n" + "="*50)
        print("RECENT PREDICTIONS FROM DATABASE")
        print("="*50)
        
        predictions = get_predictions(horizon='24hr', limit=5)
        if not predictions.empty:
            print(predictions[['date', 'horizon', 'model_name', 'forecast_value', 'timestamp']].to_string(index=False))
        else:
            print("No predictions found in database")
        
        # Show model metrics
        print("\n" + "="*50)
        print("MODEL METRICS FROM DATABASE")
        print("="*50)
        
        metrics = get_model_metrics()
        if not metrics.empty:
            print(metrics[['model_name', 'rmse', 'mae', 'mape', 'created_at']].to_string(index=False))
        else:
            print("No metrics found in database")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
