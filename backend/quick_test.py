"""
Quick test script for the Stock Forecasting Flask application.
Run this after starting the Flask app to test basic functionality.
"""

import requests
import json
import time

def test_flask_app():
    """Test the Flask app endpoints."""
    base_url = "http://localhost:5000"
    
    print("Testing Stock Forecasting Flask App...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing /api/health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Flask app not running")
        print("Please run: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Generate dataset
    print("\n2. Testing /api/generate endpoint...")
    try:
        test_data = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "days": 30
        }
        response = requests.post(f"{base_url}/api/generate", json=test_data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 201:
            result = response.json()
            print(f"Records generated: {result.get('records_count', 0)}")
            print("✅ Dataset generation passed")
        else:
            print(f"❌ Dataset generation failed: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Get datasets
    print("\n3. Testing /api/datasets endpoint...")
    try:
        response = requests.get(f"{base_url}/api/datasets", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Datasets found: {len(result)}")
            if result:
                print(f"Latest dataset: {result[0].get('symbol', 'N/A')} - {result[0].get('records', 0)} records")
            print("✅ Datasets test passed")
        else:
            print(f"❌ Datasets test failed: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Get predictions
    print("\n4. Testing /api/predictions endpoint...")
    try:
        response = requests.get(f"{base_url}/api/predictions?symbol=AAPL&limit=5", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predictions found: {len(result)}")
            print("✅ Predictions test passed")
        else:
            print(f"❌ Predictions failed: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Flask App Testing Completed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    print("Make sure the Flask app is running:")
    print("Run: python app.py")
    print("\nStarting test in 3 seconds...")
    time.sleep(3)
    
    test_flask_app()

