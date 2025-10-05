"""
Test script for Flask API endpoints.
Run this after starting the Flask app to test all endpoints.
"""

import requests
import json
import time
from datetime import datetime


def test_api():
    """Test all API endpoints."""
    base_url = "http://localhost:5000"
    
    print("="*60)
    print("TESTING STOCK FORECASTING API")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Fetch data
    print("\n2. Testing fetch-data endpoint...")
    try:
        response = requests.get(f"{base_url}/fetch-data")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Wait a moment for data processing
    time.sleep(2)
    
    # Test 3: Train models
    print("\n3. Testing train-models endpoint...")
    try:
        response = requests.post(f"{base_url}/train-models")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Wait for model training
    time.sleep(5)
    
    # Test 4: Generate forecast
    print("\n4. Testing forecast endpoint...")
    try:
        response = requests.get(f"{base_url}/forecast?horizon=24hr")
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if 'historical' in result:
            print(f"Historical data points: {len(result['historical'])}")
        if 'forecasts' in result:
            print(f"ARIMA forecast: {result['forecasts']['arima']}")
            print(f"LSTM forecast: {result['forecasts']['lstm']}")
            print(f"Ensemble forecast: {result['forecasts']['ensemble']}")
        if 'metrics' in result:
            print(f"Metrics: {result['metrics']}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Get predictions
    print("\n5. Testing predictions endpoint...")
    try:
        response = requests.get(f"{base_url}/predictions?horizon=24hr&limit=10")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Predictions count: {result.get('count', 0)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Get metrics
    print("\n6. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Metrics count: {result.get('count', 0)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: Update data
    print("\n7. Testing update-data endpoint...")
    try:
        response = requests.get(f"{base_url}/update-data")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("API TESTING COMPLETED")
    print("="*60)


def test_forecast_horizons():
    """Test different forecast horizons."""
    base_url = "http://localhost:5000"
    horizons = ['1hr', '3hr', '24hr', '72hr']
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT FORECAST HORIZONS")
    print("="*60)
    
    for horizon in horizons:
        print(f"\nTesting horizon: {horizon}")
        try:
            response = requests.get(f"{base_url}/forecast?horizon={horizon}")
            if response.status_code == 200:
                result = response.json()
                forecasts = result.get('forecasts', {})
                print(f"  ARIMA: {forecasts.get('arima', [])}")
                print(f"  LSTM: {forecasts.get('lstm', [])}")
                print(f"  Ensemble: {forecasts.get('ensemble', [])}")
            else:
                print(f"  Error: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("HORIZON TESTING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    print("Make sure the Flask app is running on http://localhost:5000")
    print("Run: python app.py")
    print("\nStarting API tests in 3 seconds...")
    time.sleep(3)
    
    # Run tests
    test_api()
    test_forecast_horizons()
    
    print("\nTest completed! Check the Flask app logs for any errors.")
