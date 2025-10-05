"""
Flask backend API for stock forecasting app.
Provides REST endpoints for data fetching, forecasting, and prediction retrieval.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our modules
from data import get_historical_data, update_latest_data
from db import init_db, insert_prices, get_latest_prices, insert_prediction, get_predictions, get_model_metrics
from models import StockForecaster

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize forecaster
forecaster = StockForecaster()


@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        'message': 'Stock Forecasting API',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /fetch-data': 'Fetch and store historical AAPL data',
            'GET /update-data': 'Update with latest AAPL data',
            'POST /forecast': 'Generate forecasts for specified horizon',
            'GET /predictions': 'Get recent predictions',
            'GET /metrics': 'Get model performance metrics'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'OK', 'timestamp': datetime.now().isoformat()})


@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    """Fetch historical AAPL data and store in database."""
    try:
        logger.info("Fetching historical data...")
        
        # Initialize database
        init_db()
        
        # Fetch historical data
        data = get_historical_data()
        
        # Store in database
        rows_inserted = insert_prices(data)
        
        logger.info(f"Successfully fetched and stored {rows_inserted} records")
        
        return jsonify({
            'message': 'Data fetched and stored successfully',
            'rows': rows_inserted,
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/update-data', methods=['GET'])
def update_data():
    """Update database with latest AAPL data."""
    try:
        logger.info("Updating with latest data...")
        
        # Fetch latest data
        new_data = update_latest_data()
        
        if new_data.empty:
            return jsonify({
                'message': 'No new data available',
                'rows': 0
            })
        
        # Store in database
        rows_inserted = insert_prices(new_data)
        
        logger.info(f"Successfully updated with {rows_inserted} new records")
        
        return jsonify({
            'message': 'Data updated successfully',
            'rows': rows_inserted,
            'date_range': {
                'start': new_data['date'].min().strftime('%Y-%m-%d'),
                'end': new_data['date'].max().strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/forecast', methods=['POST', 'GET'])
def forecast():
    """Generate forecasts for specified horizon."""
    try:
        # Get horizon from request
        if request.method == 'POST':
            data = request.get_json() or {}
            horizon = data.get('horizon', '24hr')
        else:
            horizon = request.args.get('horizon', '24hr')
        
        logger.info(f"Generating forecast for horizon: {horizon}")
        
        # Validate horizon
        valid_horizons = ['1hr', '3hr', '24hr', '72hr']
        if horizon not in valid_horizons:
            return jsonify({'error': f'Invalid horizon. Must be one of: {valid_horizons}'}), 400
        
        # Get latest data
        data = get_latest_prices(365)  # Last 365 days
        
        if data.empty:
            return jsonify({'error': 'No historical data available. Please fetch data first.'}), 400
        
        # Prepare data for forecasting
        train_data, test_data, train_close, test_close = forecaster.prepare_data(data)
        
        # Train models if not already trained
        try:
            forecaster.train_arima(train_close)
            forecaster.train_lstm(train_close)
        except Exception as e:
            logger.warning(f"Model training warning: {e}")
        
        # Generate forecasts
        horizon_steps = forecaster.horizon_mapping.get(horizon, 1)
        
        # ARIMA forecast
        try:
            arima_forecast = forecaster.forecast_arima(horizon_steps, train_close)
        except Exception as e:
            logger.error(f"ARIMA forecast error: {e}")
            arima_forecast = np.array([data['close'].iloc[-1]] * horizon_steps)
        
        # LSTM forecast
        try:
            if len(data) >= 60:
                last_sequence = data['close'].tail(60).values
                lstm_forecast = forecaster.forecast_lstm(horizon_steps, last_sequence)
            else:
                lstm_forecast = arima_forecast
        except Exception as e:
            logger.error(f"LSTM forecast error: {e}")
            lstm_forecast = arima_forecast
        
        # Ensemble forecast
        ensemble_forecast = (arima_forecast + lstm_forecast) / 2
        
        # Generate future dates
        last_date = data['date'].iloc[-1]
        future_dates = []
        for i in range(1, horizon_steps + 1):
            future_date = last_date + timedelta(days=i)
            future_dates.append(future_date.strftime('%Y-%m-%d'))
        
        # Store predictions in database
        for i, (date, arima_val, lstm_val, ensemble_val) in enumerate(zip(
            future_dates, arima_forecast, lstm_forecast, ensemble_forecast
        )):
            try:
                insert_prediction(date, horizon, 'ARIMA', float(arima_val))
                insert_prediction(date, horizon, 'LSTM', float(lstm_val))
                insert_prediction(date, horizon, 'ENSEMBLE', float(ensemble_val))
            except Exception as e:
                logger.error(f"Error storing prediction: {e}")
        
        # Get recent model metrics
        metrics_df = get_model_metrics()
        metrics = {}
        if not metrics_df.empty:
            latest_metrics = metrics_df.groupby('model_name').first()
            for model in ['ARIMA', 'LSTM']:
                if model in latest_metrics.index:
                    metrics[f'rmse_{model.lower()}'] = latest_metrics.loc[model, 'rmse']
                    metrics[f'mae_{model.lower()}'] = latest_metrics.loc[model, 'mae']
                    metrics[f'mape_{model.lower()}'] = latest_metrics.loc[model, 'mape']
        
        # Prepare historical data (last 30 days)
        historical_data = data.tail(30).copy()
        historical_data['date'] = historical_data['date'].dt.strftime('%Y-%m-%d')
        
        # Prepare response
        response = {
            'historical': historical_data[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records'),
            'forecasts': {
                'arima': arima_forecast.tolist(),
                'lstm': lstm_forecast.tolist(),
                'ensemble': ensemble_forecast.tolist()
            },
            'forecast_dates': future_dates,
            'horizon': horizon,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully generated forecast for horizon {horizon}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/predictions', methods=['GET'])
def get_predictions_endpoint():
    """Get recent predictions from database."""
    try:
        horizon = request.args.get('horizon', '24hr')
        limit = int(request.args.get('limit', 50))
        
        logger.info(f"Retrieving predictions for horizon: {horizon}")
        
        predictions = get_predictions(horizon=horizon, limit=limit)
        
        if predictions.empty:
            return jsonify({
                'message': 'No predictions found',
                'predictions': []
            })
        
        # Convert to JSON-serializable format
        predictions['date'] = predictions['date'].dt.strftime('%Y-%m-%d')
        predictions['timestamp'] = predictions['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'predictions': predictions.to_dict('records'),
            'count': len(predictions),
            'horizon': horizon
        })
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics_endpoint():
    """Get model performance metrics from database."""
    try:
        model_name = request.args.get('model')
        limit = int(request.args.get('limit', 100))
        
        logger.info("Retrieving model metrics...")
        
        metrics = get_model_metrics(model_name=model_name)
        
        if metrics.empty:
            return jsonify({
                'message': 'No metrics found',
                'metrics': []
            })
        
        # Convert to JSON-serializable format
        metrics['created_at'] = metrics['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'metrics': metrics.to_dict('records'),
            'count': len(metrics)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/train-models', methods=['POST'])
def train_models():
    """Train models on latest data."""
    try:
        logger.info("Training models...")
        
        # Get latest data
        data = get_latest_prices(365)
        
        if data.empty:
            return jsonify({'error': 'No historical data available. Please fetch data first.'}), 400
        
        # Prepare data
        train_data, test_data, train_close, test_close = forecaster.prepare_data(data)
        
        # Train models
        forecaster.train_arima(train_close)
        forecaster.train_lstm(train_close)
        
        # Generate test predictions for evaluation
        arima_pred = forecaster.forecast_arima(len(test_close), train_close)
        lstm_pred = forecaster.forecast_lstm(len(test_close), train_close[-60:])
        
        # Evaluate models
        arima_metrics = forecaster.evaluate_model(test_close, arima_pred, 'ARIMA')
        lstm_metrics = forecaster.evaluate_model(test_close, lstm_pred, 'LSTM')
        
        return jsonify({
            'message': 'Models trained successfully',
            'metrics': {
                'arima': arima_metrics,
                'lstm': lstm_metrics
            }
        })
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    
    # Run the Flask app
    logger.info("Starting Flask app on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
