import React from 'react';
import { TrendingUp, TrendingDown, Target, BarChart3 } from 'lucide-react';

const ModelMetrics = ({ metrics, selectedModels, useEnsemble }) => {
  // Process metrics data
  const processedMetrics = metrics.map(metric => ({
    ...metric,
    predicted_values: typeof metric.predicted_values === 'string' 
      ? JSON.parse(metric.predicted_values) 
      : metric.predicted_values,
    metrics: typeof metric.metrics === 'string' 
      ? JSON.parse(metric.metrics) 
      : metric.metrics
  }));

  // Get the latest metrics for each model
  const latestMetrics = processedMetrics.reduce((acc, metric) => {
    if (!acc[metric.model] || new Date(metric.created_at) > new Date(acc[metric.model].created_at)) {
      acc[metric.model] = metric;
    }
    return acc;
  }, {});

  // Sort models by RMSE (lower is better)
  const sortedModels = Object.values(latestMetrics).sort((a, b) => {
    const rmseA = a.metrics?.rmse || Infinity;
    const rmseB = b.metrics?.rmse || Infinity;
    return rmseA - rmseB;
  });

  const getModelIcon = (modelName) => {
    if (modelName.includes('ARIMA')) return <Target className="h-4 w-4" />;
    if (modelName.includes('LSTM')) return <TrendingUp className="h-4 w-4" />;
    if (modelName.includes('Transformer')) return <BarChart3 className="h-4 w-4" />;
    if (modelName.includes('Ensemble')) return <TrendingDown className="h-4 w-4" />;
    return <BarChart3 className="h-4 w-4" />;
  };

  const getModelColor = (modelName) => {
    if (modelName.includes('ARIMA')) return 'text-green-400';
    if (modelName.includes('LSTM')) return 'text-yellow-400';
    if (modelName.includes('Transformer')) return 'text-purple-400';
    if (modelName.includes('Ensemble')) return 'text-red-400';
    return 'text-blue-400';
  };

  const getPerformanceBadge = (rmse) => {
    if (rmse < 5) return { text: 'Excellent', color: 'bg-green-500/20 text-green-400' };
    if (rmse < 10) return { text: 'Good', color: 'bg-yellow-500/20 text-yellow-400' };
    if (rmse < 15) return { text: 'Fair', color: 'bg-orange-500/20 text-orange-400' };
    return { text: 'Poor', color: 'bg-red-500/20 text-red-400' };
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center space-x-2">
        <BarChart3 className="h-5 w-5" />
        <span>Model Performance</span>
      </h2>

      {sortedModels.length === 0 ? (
        <div className="text-center py-8">
          <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-400">No model metrics available</p>
          <p className="text-sm text-gray-500 mt-1">Generate a forecast to see performance metrics</p>
        </div>
      ) : (
        <div className="space-y-4">
          {sortedModels.map((metric, index) => {
            const rmse = metric.metrics?.rmse || 0;
            const mae = metric.metrics?.mae || 0;
            const mape = metric.metrics?.mape || 0;
            const performance = getPerformanceBadge(rmse);
            const isSelected = selectedModels.some(model => 
              metric.model.toLowerCase().includes(model.toLowerCase()) ||
              (model === 'ma' && metric.model.includes('moving_average')) ||
              (model === 'arima' && metric.model.includes('ARIMA')) ||
              (model === 'lstm' && metric.model.includes('LSTM')) ||
              (model === 'transformer' && metric.model.includes('Transformer')) ||
              (model === 'ensemble' && metric.model.includes('Ensemble'))
            );

            return (
              <div
                key={metric.id}
                className={`bg-white/5 rounded-lg p-4 border transition-all duration-200 ${
                  isSelected 
                    ? 'border-blue-500/50 bg-blue-500/10' 
                    : 'border-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {getModelIcon(metric.model)}
                    <span className={`font-medium ${getModelColor(metric.model)}`}>
                      {metric.model}
                    </span>
                    {index === 0 && (
                      <span className="bg-yellow-500/20 text-yellow-400 text-xs px-2 py-1 rounded">
                        Best
                      </span>
                    )}
                  </div>
                  <span className={`text-xs px-2 py-1 rounded ${performance.color}`}>
                    {performance.text}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400 text-xs">RMSE</div>
                    <div className="text-white font-medium">{rmse.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-xs">MAE</div>
                    <div className="text-white font-medium">{mae.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-xs">MAPE</div>
                    <div className="text-white font-medium">{mape.toFixed(1)}%</div>
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>Forecast Horizon: {metric.forecast_horizon} days</span>
                    <span>{new Date(metric.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Model Selection Status */}
      <div className="mt-6 pt-6 border-t border-white/10">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Selected Models</h3>
        <div className="space-y-2">
          {selectedModels.map(model => (
            <div key={model} className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-sm text-gray-300 capitalize">{model}</span>
            </div>
          ))}
          {useEnsemble && (
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span className="text-sm text-gray-300">Ensemble</span>
            </div>
          )}
        </div>
      </div>

      {/* Performance Summary */}
      {sortedModels.length > 0 && (
        <div className="mt-6 pt-6 border-t border-white/10">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Performance Summary</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Best Model:</span>
              <span className="text-white">{sortedModels[0].model}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Best RMSE:</span>
              <span className="text-white">{(sortedModels[0].metrics?.rmse || 0).toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Models Trained:</span>
              <span className="text-white">{sortedModels.length}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelMetrics;
