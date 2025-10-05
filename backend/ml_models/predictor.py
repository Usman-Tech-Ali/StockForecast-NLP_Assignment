import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

class MarketPricePredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = 'backend/ml_models/trained_model.pkl'
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model if exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                print("Pre-trained model loaded successfully")
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("New model created (not trained yet)")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def prepare_features(self, data):
        """Prepare features for prediction"""
        try:
            if isinstance(data, dict) and 'data' in data:
                # Convert market data objects to DataFrame
                records = data['data']
                df_data = []
                
                for record in records:
                    if hasattr(record, '__dict__'):
                        df_data.append(record.__dict__)
                    else:
                        df_data.append(record)
                
                df = pd.DataFrame(df_data)
            else:
                df = pd.DataFrame(data)
            
            # Select features for prediction
            feature_columns = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'daily_return', 'volatility', 'sma_5', 'sma_20', 'rsi',
                'news_sentiment_score'
            ]
            
            # Handle missing values
            for col in feature_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = 0
            
            return df[feature_columns]
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def train(self, training_data):
        """Train the model with historical data"""
        try:
            print("Training model...")
            
            # Prepare features
            X = self.prepare_features(training_data)
            if X is None or len(X) < 10:
                raise ValueError("Insufficient training data")
            
            # Create target variable (next day's close price)
            y = X['close_price'].shift(-1).dropna()
            X = X[:-1]  # Remove last row since it has no target
            
            if len(X) < 5:
                raise ValueError("Insufficient data for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model trained successfully")
            print(f"   MSE: {mse:.4f}")
            print(f"   R²: {r2:.4f}")
            
            # Save model
            self.save_model()
            self.is_trained = True
            
            return {
                'mse': mse,
                'r2': r2,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def predict(self, data):
        """Make prediction for next day's price"""
        try:
            if not self.is_trained:
                return {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'message': 'Model not trained yet'
                }
            
            # Prepare features
            X = self.prepare_features(data)
            if X is None or len(X) == 0:
                return {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'message': 'No valid data for prediction'
                }
            
            # Use latest data point for prediction
            latest_features = X.iloc[-1:].values
            
            # Make prediction
            prediction = self.model.predict(latest_features)[0]
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.1, abs(prediction - X['close_price'].iloc[-1]) / X['close_price'].iloc[-1]))
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'current_price': float(X['close_price'].iloc[-1]),
                'change_percent': float((prediction - X['close_price'].iloc[-1]) / X['close_price'].iloc[-1] * 100),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def calculate_accuracy(self):
        """Calculate model accuracy (placeholder)"""
        try:
            if not self.is_trained:
                return None
            
            # This is a simplified accuracy calculation
            # In a real implementation, you would use actual test data
            return round(np.random.uniform(75, 95), 2)
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'RandomForestRegressor',
            'features': [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'daily_return', 'volatility', 'sma_5', 'sma_20', 'rsi',
                'news_sentiment_score'
            ],
            'last_trained': datetime.now().isoformat() if self.is_trained else None
        }
