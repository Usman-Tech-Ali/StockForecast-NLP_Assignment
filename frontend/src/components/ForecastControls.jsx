import React from 'react';
import { Play, RefreshCw, CheckSquare, Square, Settings } from 'lucide-react';
import StockSearch from './StockSearch';
import HorizonSearch from './HorizonSearch';

const ForecastControls = ({
  instruments,
  horizons,
  models,
  selectedSymbol,
  forecastHorizon,
  selectedModels,
  useEnsemble,
  onSymbolChange,
  onHorizonChange,
  onModelsChange,
  onEnsembleChange,
  onGenerate,
  loading
}) => {
  const handleModelToggle = (modelId) => {
    if (selectedModels.includes(modelId)) {
      onModelsChange(selectedModels.filter(id => id !== modelId));
    } else {
      onModelsChange([...selectedModels, modelId]);
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center space-x-2">
        <Settings className="h-5 w-5" />
        <span>Forecast Configuration</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Instrument Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Financial Instrument
          </label>
          <StockSearch
            instruments={instruments}
            selectedSymbol={selectedSymbol}
            onSymbolChange={onSymbolChange}
            placeholder="Search for stocks (e.g., AAPL, Apple, NASDAQ)"
          />
        </div>

        {/* Forecast Horizon */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Forecast Horizon
          </label>
          <HorizonSearch
            horizons={horizons}
            selectedHorizon={forecastHorizon}
            onHorizonChange={onHorizonChange}
            placeholder="Search forecast periods (e.g., 24h, 1 week, hours)"
          />
        </div>

        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            ML Models
          </label>
          <div className="space-y-2">
            {models.map((model) => (
              <label key={model.id} className="flex items-center space-x-2 cursor-pointer">
                <button
                  type="button"
                  onClick={() => handleModelToggle(model.id)}
                  className="flex items-center justify-center w-4 h-4 border border-white/30 rounded"
                >
                  {selectedModels.includes(model.id) ? (
                    <CheckSquare className="h-3 w-3 text-green-400" />
                  ) : (
                    <Square className="h-3 w-3 text-gray-400" />
                  )}
                </button>
                <span className="text-sm text-gray-300">{model.name}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Ensemble Option */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Ensemble
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <button
              type="button"
              onClick={() => onEnsembleChange(!useEnsemble)}
              className="flex items-center justify-center w-4 h-4 border border-white/30 rounded"
            >
              {useEnsemble ? (
                <CheckSquare className="h-3 w-3 text-green-400" />
              ) : (
                <Square className="h-3 w-3 text-gray-400" />
              )}
            </button>
            <span className="text-sm text-gray-300">Use Ensemble</span>
          </label>
          <p className="text-xs text-gray-400 mt-1">
            Combine multiple models for better accuracy
          </p>
        </div>
      </div>

      {/* Generate Button */}
      <div className="mt-6 flex justify-center">
        <button
          onClick={onGenerate}
          disabled={loading || selectedModels.length === 0}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-medium px-6 py-3 rounded-lg transition-all duration-200 transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
        >
          {loading ? (
            <RefreshCw className="h-5 w-5 animate-spin" />
          ) : (
            <Play className="h-5 w-5" />
          )}
          <span>{loading ? 'Generating Forecast...' : 'Generate Forecast'}</span>
        </button>
      </div>

      {/* Model Descriptions */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((model) => (
          <div key={model.id} className="bg-white/5 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white">{model.name}</h4>
            <p className="text-xs text-gray-400 mt-1">{model.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ForecastControls;
