import React from 'react';
import { RefreshCw } from 'lucide-react';

const LoadingSpinner = () => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 flex flex-col items-center space-y-4">
        <RefreshCw className="h-8 w-8 text-blue-400 animate-spin" />
        <div className="text-center">
          <h3 className="text-lg font-semibold text-white mb-2">Processing Forecast</h3>
          <p className="text-gray-300 text-sm">
            Training ML models and generating predictions...
          </p>
          <div className="mt-4 space-y-2">
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <span>Fetching historical data</span>
            </div>
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>Training ARIMA model</span>
            </div>
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              <span>Training LSTM model</span>
            </div>
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
              <span>Generating ensemble forecast</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;
