import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Traditional models
from statsmodels.tsa.arima.model import ARIMA

# Neural model (Keras)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def split_time_series(series: pd.Series, test_size: int = 10) -> Tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(series):
        test_size = max(1, min(10, len(series) // 4))
    return series.iloc[:-test_size], series.iloc[-test_size:]


def compute_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true, dtype='float64')
    y_pred = np.array(y_pred, dtype='float64')
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # Avoid division by zero in MAPE
    nonzero = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / nonzero)) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ---------- Class-based API ----------

class ForecastingModel:
    def fit(self, train_series: pd.Series) -> None:
        raise NotImplementedError

    def predict(self, horizon: int) -> List[float]:
        raise NotImplementedError

    def evaluate(self, test_series: pd.Series) -> Dict[str, Any]:
        preds = self.predict(len(test_series))
        metrics = compute_performance_metrics(test_series.values, np.array(preds))
        return {
            'forecast_horizon': len(test_series),
            'predicted_values': [float(x) for x in preds],
            'metrics': metrics,
            'y_true': test_series.astype(float).tolist()
        }


class MovingAverageForecaster(ForecastingModel):
    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self.history: List[float] = []

    def fit(self, train_series: pd.Series) -> None:
        if len(train_series) == 0:
            raise ValueError("Cannot fit MovingAverageForecaster with empty series")
        self.history = list(train_series.values)

    def predict(self, horizon: int) -> List[float]:
        preds: List[float] = []
        hist = list(self.history)
        for _ in range(max(0, int(horizon))):
            avg = float(np.mean(hist[-self.window:])) if len(hist) >= self.window else float(np.mean(hist))
            preds.append(avg)
            hist.append(avg)  # naive roll-forward using own prediction
        return preds


class ARIMAForecaster(ForecastingModel):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = tuple(int(x) for x in order)
        self._fit_result = None

    def fit(self, train_series: pd.Series) -> None:
        model = ARIMA(train_series.values, order=self.order)
        self._fit_result = model.fit()

    def predict(self, horizon: int) -> List[float]:
        if self._fit_result is None:
            return []
        preds = self._fit_result.forecast(steps=int(horizon))
        return preds.astype(float).tolist()


class LSTMForecaster(ForecastingModel):
    def __init__(self, lookback: int = 10, epochs: int = 50, batch_size: int = 16, lr: float = 0.01):
        self.lookback = int(lookback)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.min_v: float = 0.0
        self.max_v: float = 1.0
        self.model = None
        self.train_scaled: Optional[np.ndarray] = None
        self.train_all_scaled: Optional[np.ndarray] = None

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        self.min_v = float(np.min(arr))
        self.max_v = float(np.max(arr))
        denom = (self.max_v - self.min_v) if (self.max_v - self.min_v) != 0 else 1.0
        return (arr - self.min_v) / denom

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        denom = (self.max_v - self.min_v) if (self.max_v - self.min_v) != 0 else 1.0
        return (arr - self.min_v) / denom

    def _inv_scale(self, arr: np.ndarray) -> np.ndarray:
        denom = (self.max_v - self.min_v) if (self.max_v - self.min_v) != 0 else 1.0
        return arr * denom + self.min_v

    def fit(self, train_series: pd.Series) -> None:
        train_values = train_series.values.astype('float32')
        train_scaled = self._scale_fit(train_values)
        X_train, y_train = _create_supervised(train_scaled, self.lookback)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) if len(X_train) > 0 else X_train
        model = Sequential([
            LSTM(32, input_shape=(self.lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
        if len(X_train) > 0:
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=callbacks)
        self.model = model
        self.train_all_scaled = train_scaled

    def predict(self, horizon: int) -> List[float]:
        if self.model is None:
            return []
        history = list(self.train_all_scaled) if self.train_all_scaled is not None else []
        preds_scaled: List[float] = []
        for _ in range(max(0, int(horizon))):
            window_seq = np.array(history[-self.lookback:]) if len(history) >= self.lookback else np.array(history)
            if len(window_seq) < self.lookback:
                window_seq = np.pad(window_seq, (self.lookback - len(window_seq), 0), 'edge')
            x = window_seq.reshape((1, self.lookback, 1))
            yhat = float(self.model.predict(x, verbose=0)[0][0])
            preds_scaled.append(yhat)
            history.append(yhat)
        preds = self._inv_scale(np.array(preds_scaled))
        return preds.astype(float).tolist()


class TransformerForecaster(ForecastingModel):
    def __init__(self, lookback: int = 24, d_model: int = 32, num_heads: int = 2, ff_dim: int = 64, epochs: int = 40, batch_size: int = 16, dropout: float = 0.1, lr: float = 0.005):
        self.lookback = int(lookback)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.min_v: float = 0.0
        self.max_v: float = 1.0
        self.model = None
        self.train_all_scaled: Optional[np.ndarray] = None

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        self.min_v = float(np.min(arr))
        self.max_v = float(np.max(arr))
        denom = (self.max_v - self.min_v) if (self.max_v - self.min_v) != 0 else 1.0
        return (arr - self.min_v) / denom

    def _inv_scale(self, arr: np.ndarray) -> np.ndarray:
        denom = (self.max_v - self.min_v) if (self.max_v - self.min_v) != 0 else 1.0
        return arr * denom + self.min_v

    def fit(self, train_series: pd.Series) -> None:
        train_values = train_series.values.astype('float32')
        train_scaled = self._scale_fit(train_values)
        X_train, y_train = _create_supervised(train_scaled, self.lookback)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) if len(X_train) > 0 else X_train

        inp = Input(shape=(self.lookback, 1))
        x = Dense(self.d_model)(inp)
        attn_out = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = Sequential([Dense(self.ff_dim, activation='relu'), Dense(self.d_model)])
        x = LayerNormalization(epsilon=1e-6)(x + ff(x))
        x = Dropout(self.dropout)(x)
        out = Dense(1)(x[:, -1, :])
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
        if len(X_train) > 0:
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=callbacks)
        self.model = model
        self.train_all_scaled = train_scaled

    def predict(self, horizon: int) -> List[float]:
        if self.model is None:
            return []
        history = list(self.train_all_scaled) if self.train_all_scaled is not None else []
        preds_scaled: List[float] = []
        for _ in range(max(0, int(horizon))):
            window_seq = np.array(history[-self.lookback:]) if len(history) >= self.lookback else np.array(history)
            if len(window_seq) < self.lookback:
                window_seq = np.pad(window_seq, (self.lookback - len(window_seq), 0), 'edge')
            x_in = window_seq.reshape((1, self.lookback, 1))
            yhat = float(self.model.predict(x_in, verbose=0)[0])
            preds_scaled.append(yhat)
            history.append(yhat)
        preds = self._inv_scale(np.array(preds_scaled))
        return preds.astype(float).tolist()


class EnsembleAverageForecaster(ForecastingModel):
    def __init__(self, forecasters: List[ForecastingModel]):
        self.forecasters = forecasters

    def fit(self, train_series: pd.Series) -> None:
        for f in self.forecasters:
            f.fit(train_series)

    def predict(self, horizon: int) -> List[float]:
        preds_list = [f.predict(horizon) for f in self.forecasters]
        if not preds_list:
            return []
        # Align lengths
        min_len = min(len(p) for p in preds_list)
        preds_list = [p[:min_len] for p in preds_list]
        avg = np.mean(np.array(preds_list), axis=0)
        return avg.astype(float).tolist()

def simple_moving_average_forecast(series: pd.Series, window: int = 5) -> Dict[str, Any]:
    train, test = split_time_series(series)
    if window < 1:
        window = 1
    history = list(train.values)
    preds: List[float] = []
    for _ in range(len(test)):
        avg = float(np.mean(history[-window:])) if len(history) >= window else float(np.mean(history))
        preds.append(avg)
        history.append(test.iloc[len(preds)-1])
    metrics = compute_performance_metrics(test.values, np.array(preds))
    return {
        'model': 'moving_average',
        'forecast_horizon': len(test),
        'predicted_values': preds,
        'metrics': metrics,
        'y_true': test.values.tolist()
    }


def arima_forecast(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
    train, test = split_time_series(series)
    model = ARIMA(train.values, order=order)
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(test))
    metrics = compute_performance_metrics(test.values, preds)
    return {
        'model': f'ARIMA{order}',
        'forecast_horizon': len(test),
        'predicted_values': preds.astype(float).tolist(),
        'metrics': metrics,
        'y_true': test.values.tolist()
    }


def _create_supervised(sequence: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(sequence) - lookback):
        X.append(sequence[i:i+lookback])
        y.append(sequence[i+lookback])
    return np.array(X), np.array(y)


def lstm_forecast(series: pd.Series, lookback: int = 10, epochs: int = 50, batch_size: int = 16) -> Dict[str, Any]:
    values = series.values.astype('float32')
    train_vals, test_vals = train_test_split_series(series)
    train_values = train_vals.values.astype('float32')
    test_values = test_vals.values.astype('float32')

    # scale optionally (simple min-max)
    min_v = float(np.min(train_values))
    max_v = float(np.max(train_values))
    denom = (max_v - min_v) if (max_v - min_v) != 0 else 1.0
    def scale(arr):
        return (arr - min_v) / denom
    def inv_scale(arr):
        return arr * denom + min_v

    train_scaled = scale(train_values)
    test_scaled = scale(test_values)

    X_train, y_train = _create_supervised(train_scaled, lookback)
    # To predict the next len(test) points, we roll-forward from the tail
    # For evaluation alignment, we generate one-step ahead predictions
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
    if len(X_train) > 0:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    # Rolling one-step predictions over test horizon
    history = list(scale(values))
    preds_scaled: List[float] = []
    for _ in range(len(test_values)):
        window_seq = np.array(history[-lookback:]) if len(history) >= lookback else np.array(history)
        if len(window_seq) < lookback:
            window_seq = np.pad(window_seq, (lookback - len(window_seq), 0), 'edge')
        x = window_seq.reshape((1, lookback, 1))
        yhat = float(model.predict(x, verbose=0)[0][0])
        preds_scaled.append(yhat)
        history.append(test_scaled[len(preds_scaled)-1])

    preds = inv_scale(np.array(preds_scaled))
    metrics = calculate_metrics(test_values, preds)
    return {
        'model': 'LSTM',
        'forecast_horizon': len(test_values),
        'predicted_values': preds.astype(float).tolist(),
        'metrics': metrics,
        'y_true': test_values.astype(float).tolist()
    }


def transformer_forecast(series: pd.Series, lookback: int = 24, d_model: int = 32, num_heads: int = 2, ff_dim: int = 64, epochs: int = 40, batch_size: int = 16, dropout: float = 0.1) -> Dict[str, Any]:
    values = series.values.astype('float32')
    train_vals, test_vals = train_test_split_series(series)
    train_values = train_vals.values.astype('float32')
    test_values = test_vals.values.astype('float32')

    # simple scaling
    min_v = float(np.min(train_values))
    max_v = float(np.max(train_values))
    denom = (max_v - min_v) if (max_v - min_v) != 0 else 1.0
    def scale(arr):
        return (arr - min_v) / denom
    def inv_scale(arr):
        return arr * denom + min_v

    train_scaled = scale(train_values)
    test_scaled = scale(test_values)

    X_train, y_train = _create_supervised(train_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build a minimal Transformer encoder block for regression
    inp = Input(shape=(lookback, 1))
    # Project to d_model
    x = Dense(d_model)(inp)
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = Sequential([Dense(ff_dim, activation='relu'), Dense(d_model)])
    x = LayerNormalization(epsilon=1e-6)(x + ff(x))
    x = Dropout(dropout)(x)
    # Global average over time then dense to 1
    x = Dense(1)(x[:, -1, :])
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
    if len(X_train) > 0:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    # Rolling one-step predictions
    history = list(scale(values))
    preds_scaled: List[float] = []
    for _ in range(len(test_values)):
        window_seq = np.array(history[-lookback:]) if len(history) >= lookback else np.array(history)
        if len(window_seq) < lookback:
            window_seq = np.pad(window_seq, (lookback - len(window_seq), 0), 'edge')
        x_in = window_seq.reshape((1, lookback, 1))
        yhat = float(model.predict(x_in, verbose=0)[0])
        preds_scaled.append(yhat)
        history.append(test_scaled[len(preds_scaled)-1])

    preds = inv_scale(np.array(preds_scaled))
    metrics = calculate_metrics(test_values, preds)
    return {
        'model': 'Transformer',
        'forecast_horizon': len(test_values),
        'predicted_values': preds.astype(float).tolist(),
        'metrics': metrics,
        'y_true': test_values.astype(float).tolist()
    }


