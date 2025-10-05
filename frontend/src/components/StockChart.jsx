import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

const StockChart = ({ historicalData, forecastData, symbol, horizon }) => {
  const chartData = useMemo(() => {
    if (!historicalData || historicalData.length === 0) {
      return {
        data: [],
        layout: {
          title: `No data available for ${symbol}`,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: 'white' },
          xaxis: { color: 'white' },
          yaxis: { color: 'white' }
        }
      };
    }

    // Process historical data for candlestick chart
    const dates = historicalData.map(item => item.date);
    const open = historicalData.map(item => item.open);
    const high = historicalData.map(item => item.high);
    const low = historicalData.map(item => item.low);
    const close = historicalData.map(item => item.close);

    const data = [
      {
        type: 'candlestick',
        x: dates,
        open: open,
        high: high,
        low: low,
        close: close,
        name: 'Historical',
        increasing: { line: { color: '#00ff88' } },
        decreasing: { line: { color: '#ff4444' } },
        showlegend: true
      }
    ];

    // Add forecast data if available
    if (forecastData && forecastData.length > 0) {
      forecastData.forEach((forecast) => {
        if (forecast.predicted_values && forecast.predicted_values.length > 0) {
          // Generate forecast dates (extend from last historical date)
          const lastDate = new Date(dates[dates.length - 1]);
          const forecastDates = forecast.predicted_values.map((_, i) => {
            const newDate = new Date(lastDate);
            newDate.setDate(newDate.getDate() + i + 1);
            return newDate.toISOString().split('T')[0];
          });

          // Add forecast line
          data.push({
            type: 'scatter',
            mode: 'lines+markers',
            x: [...dates.slice(-5), ...forecastDates], // Show last 5 historical points + forecast
            y: [...close.slice(-5), ...forecast.predicted_values],
            name: forecast.model,
            line: {
              color: getModelColor(forecast.model),
              width: 2,
              dash: 'dash'
            },
            marker: {
              size: 4,
              color: getModelColor(forecast.model)
            },
            showlegend: true
          });

          // Add confidence interval if available
          if (forecast.metrics && forecast.metrics.rmse) {
            const upperBound = forecast.predicted_values.map(val => val + forecast.metrics.rmse);
            const lowerBound = forecast.predicted_values.map(val => val - forecast.metrics.rmse);
            
            data.push({
              type: 'scatter',
              mode: 'lines',
              x: forecastDates,
              y: upperBound,
              name: `${forecast.model} (Upper)`,
              line: { color: getModelColor(forecast.model), width: 1, dash: 'dot' },
              showlegend: false,
              hoverinfo: 'skip'
            });

            data.push({
              type: 'scatter',
              mode: 'lines',
              x: forecastDates,
              y: lowerBound,
              name: `${forecast.model} (Lower)`,
              line: { color: getModelColor(forecast.model), width: 1, dash: 'dot' },
              fill: 'tonexty',
              fillcolor: `rgba(${hexToRgb(getModelColor(forecast.model))}, 0.1)`,
              showlegend: false,
              hoverinfo: 'skip'
            });
          }
        }
      });
    }

    const layout = {
      title: {
        text: `${symbol} Stock Price Forecast (${horizon})`,
        font: { color: 'white', size: 18 }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' },
      xaxis: {
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)',
        title: 'Date'
      },
      yaxis: {
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)',
        title: 'Price (USD)'
      },
      legend: {
        bgcolor: 'rgba(0,0,0,0.5)',
        bordercolor: 'rgba(255,255,255,0.2)',
        borderwidth: 1,
        font: { color: 'white' }
      },
      hovermode: 'x unified',
      showlegend: true,
      margin: { l: 60, r: 30, t: 60, b: 60 }
    };

    const config = {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
      responsive: true
    };

    return { data, layout, config };
  }, [historicalData, forecastData, symbol, horizon]);

  return (
    <div className="w-full h-96">
      <Plot
        data={chartData.data}
        layout={chartData.layout}
        config={chartData.config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
};

// Helper function to get model colors
const getModelColor = (modelName) => {
  const colors = {
    'moving_average': '#3b82f6',      // Blue
    'ARIMA(1, 1, 1)': '#10b981',      // Green
    'LSTM': '#f59e0b',                // Yellow
    'Transformer': '#8b5cf6',         // Purple
    'EnsembleAverage': '#ef4444'      // Red
  };
  return colors[modelName] || '#6b7280'; // Default gray
};

// Helper function to convert hex to rgb
const hexToRgb = (hex) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? 
    `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : 
    '107, 114, 128';
};

export default StockChart;
