import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, BarChart3, Settings, Activity, DollarSign, Clock } from 'lucide-react';
import axios from 'axios';
import StockChart from './components/StockChart';
import ForecastControls from './components/ForecastControls';
import ModelMetrics from './components/ModelMetrics';
import LoadingSpinner from './components/LoadingSpinner';

const API_BASE_URL = 'http://localhost:5000';

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [forecastHorizon, setForecastHorizon] = useState('24h');
  const [selectedModels, setSelectedModels] = useState(['ma', 'arima', 'lstm']);
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [modelMetrics, setModelMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Available financial instruments
  const instruments = [
    { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ' },
    { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ' },
    { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ' },
    { symbol: 'NFLX', name: 'Netflix Inc.', exchange: 'NASDAQ' }
  ];

  // Forecast horizons
  const horizons = [
    { value: '1h', label: '1 Hour' },
    { value: '3h', label: '3 Hours' },
    { value: '6h', label: '6 Hours' },
    { value: '24h', label: '24 Hours' },
    { value: '72h', label: '72 Hours' },
    { value: '1w', label: '1 Week' }
  ];

  // Available models
  const models = [
    { id: 'ma', name: 'Moving Average', description: 'Simple moving average forecasting' },
    { id: 'arima', name: 'ARIMA', description: 'AutoRegressive Integrated Moving Average' },
    { id: 'lstm', name: 'LSTM', description: 'Long Short-Term Memory neural network' },
    { id: 'transformer', name: 'Transformer', description: 'Transformer-based neural network' }
  ];

  // Fetch historical data
  const fetchHistoricalData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/get_historical`, {
        params: { symbol: selectedSymbol, limit: 100 }
      });
      setHistoricalData(response.data.rows || []);
      setError(null);
    } catch (err) {
      setError('Failed to fetch historical data');
      console.error('Error fetching historical data:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol]);

  // Generate forecast
  const generateForecast = async () => {
    try {
      setLoading(true);
      
      // First generate new data if needed
      await axios.post(`${API_BASE_URL}/api/generate`, {
        symbol: selectedSymbol,
        exchange: 'NASDAQ',
        days: 30
      });

      // Train models and generate forecast
      const forecastResponse = await axios.post(`${API_BASE_URL}/api/forecast/run`, {
        symbol: selectedSymbol,
        models: selectedModels,
        ensemble: useEnsemble,
        preview_horizon_hours: parseInt(forecastHorizon.replace('h', '')) || 24
      });

      setForecastData(forecastResponse.data.results || []);
      setLastUpdated(new Date().toLocaleString());
      setError(null);
    } catch (err) {
      setError('Failed to generate forecast');
      console.error('Error generating forecast:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch model metrics
  const fetchModelMetrics = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/predictions`, {
        params: { symbol: selectedSymbol, limit: 10 }
      });
      setModelMetrics(response.data || []);
    } catch (err) {
      console.error('Error fetching model metrics:', err);
    }
  }, [selectedSymbol]);

  // Initial data load
  useEffect(() => {
    fetchHistoricalData();
    fetchModelMetrics();
  }, [fetchHistoricalData, fetchModelMetrics]);

  // Auto-refresh data every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      fetchHistoricalData();
      fetchModelMetrics();
    }, 300000); // 5 minutes

    return () => clearInterval(interval);
  }, [fetchHistoricalData, fetchModelMetrics]);

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <TrendingUp className="h-8 w-8 text-green-400" />
              <h1 className="text-2xl font-bold text-white">StockForecast AI</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-300">
                <Activity className="h-4 w-4" />
                <span>Real-time Forecasting</span>
              </div>
              {lastUpdated && (
                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <Clock className="h-4 w-4" />
                  <span>Updated: {lastUpdated}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls Section */}
        <div className="mb-8">
          <ForecastControls
            instruments={instruments}
            horizons={horizons}
            models={models}
            selectedSymbol={selectedSymbol}
            forecastHorizon={forecastHorizon}
            selectedModels={selectedModels}
            useEnsemble={useEnsemble}
            onSymbolChange={setSelectedSymbol}
            onHorizonChange={setForecastHorizon}
            onModelsChange={setSelectedModels}
            onEnsembleChange={setUseEnsemble}
            onGenerate={generateForecast}
            loading={loading}
          />
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/20 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <div className="h-4 w-4 bg-red-500 rounded-full"></div>
              <span className="text-red-400 font-medium">Error</span>
            </div>
            <p className="text-red-300 mt-1">{error}</p>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && <LoadingSpinner />}

        {/* Chart Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Chart */}
          <div className="lg:col-span-2">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Price Forecast - {selectedSymbol}</span>
                </h2>
                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <DollarSign className="h-4 w-4" />
                  <span>USD</span>
                </div>
              </div>
              
              <StockChart
                historicalData={historicalData}
                forecastData={forecastData}
                symbol={selectedSymbol}
                horizon={forecastHorizon}
              />
            </div>
          </div>

          {/* Model Metrics */}
          <div className="lg:col-span-1">
            <ModelMetrics
              metrics={modelMetrics}
              selectedModels={selectedModels}
              useEnsemble={useEnsemble}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-400 text-sm">
          <p>CS4063 - Natural Language Processing | Stock Forecasting Application</p>
          <p className="mt-1">Powered by ARIMA, LSTM, Transformer, and Ensemble Models</p>
        </footer>
      </main>
    </div>
  );
}

export default App;