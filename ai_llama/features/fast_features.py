"""
Fast Feature Engineering with Pure NumPy Vectorization

Optimized for ultra-low latency trading decisions.
Target: <1ms feature extraction per symbol
"""

import numpy as np
import numba as nb
from typing import Dict, Any, Optional, Tuple
import time


@nb.jit(nopython=True, cache=True)
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized return calculation"""
    return np.diff(prices) / prices[:-1]


@nb.jit(nopython=True, cache=True)
def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized log return calculation"""
    return np.diff(np.log(prices))


@nb.jit(nopython=True, cache=True)
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using convolution"""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


@nb.jit(nopython=True, cache=True)
def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation"""
    result = np.empty(len(arr) - window + 1)
    for i in range(len(result)):
        result[i] = np.std(arr[i:i+window])
    return result


@nb.jit(nopython=True, cache=True)
def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI calculation"""
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = rolling_mean(gains, period)
    avg_losses = rolling_mean(losses, period)
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


@nb.jit(nopython=True, cache=True)
def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Bollinger Bands calculation"""
    ma = rolling_mean(prices, period)
    std = rolling_std(prices, period)
    
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    
    return upper, ma, lower


@nb.jit(nopython=True, cache=True)
def exponential_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
    """Fast EMA calculation"""
    alpha = 2.0 / (period + 1.0)
    result = np.empty_like(prices)
    result[0] = prices[0]
    
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    
    return result


@nb.jit(nopython=True, cache=True)
def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized MACD calculation"""
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@nb.jit(nopython=True, cache=True)
def momentum(prices: np.ndarray, period: int) -> np.ndarray:
    """Price momentum over period"""
    return prices[period:] / prices[:-period] - 1.0


@nb.jit(nopython=True, cache=True)
def volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling volatility calculation"""
    return rolling_std(returns, window) * np.sqrt(252)  # Annualized


@nb.jit(nopython=True, cache=True)
def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume Weighted Average Price"""
    return np.sum(prices * volumes) / np.sum(volumes)


@nb.jit(nopython=True, cache=True)
def calculate_gap(yesterday_close: float, today_open: float) -> float:
    """Calculate gap percentage"""
    return (today_open - yesterday_close) / yesterday_close


@nb.jit(nopython=True, cache=True)
def price_range(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Price range (high - low)"""
    return high - low


@nb.jit(nopython=True, cache=True)
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range calculation"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # Skip first element due to roll
    tr = np.empty(len(high))
    tr[0] = tr1[0]
    
    for i in range(1, len(high)):
        tr[i] = max(tr1[i], tr2[i], tr3[i])
    
    return tr


class FastFeatureEngine:
    """
    High-performance feature extraction engine
    
    Optimized for <1ms latency per symbol
    Uses aggressive caching and vectorization
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.feature_cache = {}
        self.last_update = {}
        
        # Pre-allocate arrays for better performance
        self.temp_arrays = {
            'returns': np.empty(1000),
            'prices': np.empty(1000),
            'volumes': np.empty(1000)
        }
    
    def extract_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray], 
                        timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract all features for a symbol
        
        Args:
            symbol: Trading symbol
            ohlcv_data: Dict containing 'open', 'high', 'low', 'close', 'volume' arrays
            timestamp: Current timestamp for caching
            
        Returns:
            Dict of extracted features
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = f"{symbol}_{timestamp}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        open_prices = ohlcv_data['open']
        high_prices = ohlcv_data['high']
        low_prices = ohlcv_data['low']
        close_prices = ohlcv_data['close']
        volumes = ohlcv_data['volume']
        
        # Basic features (vectorized)
        returns = calculate_returns(close_prices)
        log_returns = calculate_log_returns(close_prices)
        
        # Technical indicators
        rsi_14 = rsi(close_prices, 14)
        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, 20, 2.0)
        macd_line, macd_signal, macd_hist = macd(close_prices)
        
        # Momentum features
        momentum_5 = momentum(close_prices, 5)
        momentum_20 = momentum(close_prices, 20)
        
        # Volatility features
        vol_5 = volatility(returns, 5)
        vol_20 = volatility(returns, 20)
        
        # Price action features
        price_ranges = price_range(high_prices, low_prices)
        true_ranges = true_range(high_prices, low_prices, close_prices)
        
        # Volume features
        current_vwap = vwap(close_prices[-20:], volumes[-20:]) if len(close_prices) >= 20 else close_prices[-1]
        
        # Gap calculation (if we have previous close)
        gap_percent = 0.0
        if len(close_prices) > 1:
            gap_percent = calculate_gap(close_prices[-2], open_prices[-1])
        
        features = {
            # Price features
            'current_price': close_prices[-1],
            'open_price': open_prices[-1],
            'high_price': high_prices[-1],
            'low_price': low_prices[-1],
            'volume': volumes[-1],
            
            # Returns
            'return_1': returns[-1] if len(returns) > 0 else 0.0,
            'return_5': np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
            'log_return_1': log_returns[-1] if len(log_returns) > 0 else 0.0,
            
            # Technical indicators
            'rsi_14': rsi_14[-1] if len(rsi_14) > 0 else 50.0,
            'bb_position': ((close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) if len(bb_upper) > 0 else 0.5,
            'macd_signal': 1.0 if len(macd_hist) > 0 and macd_hist[-1] > 0 else -1.0,
            
            # Momentum
            'momentum_5': momentum_5[-1] if len(momentum_5) > 0 else 0.0,
            'momentum_20': momentum_20[-1] if len(momentum_20) > 0 else 0.0,
            'momentum_ratio': momentum_5[-1] / momentum_20[-1] if len(momentum_5) > 0 and len(momentum_20) > 0 and momentum_20[-1] != 0 else 1.0,
            
            # Volatility
            'volatility_5': vol_5[-1] if len(vol_5) > 0 else 0.0,
            'volatility_20': vol_20[-1] if len(vol_20) > 0 else 0.0,
            'volatility_ratio': vol_5[-1] / vol_20[-1] if len(vol_5) > 0 and len(vol_20) > 0 and vol_20[-1] != 0 else 1.0,
            
            # Price action
            'price_range': price_ranges[-1],
            'range_percentage': price_ranges[-1] / close_prices[-1],
            'true_range': true_ranges[-1] if len(true_ranges) > 0 else price_ranges[-1],
            
            # Volume
            'vwap': current_vwap,
            'price_vs_vwap': close_prices[-1] / current_vwap - 1.0,
            'volume_ratio': volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0,
            
            # Gap
            'gap_percent': gap_percent,
            'gap_size': abs(gap_percent),
            
            # Meta
            'feature_count': 22,
            'extraction_time_ms': (time.perf_counter() - start_time) * 1000,
            'timestamp': timestamp or time.time()
        }
        
        # Cache result
        if timestamp:
            self.feature_cache[cache_key] = features
            # Clean old cache entries
            if len(self.feature_cache) > self.cache_size:
                oldest_key = min(self.feature_cache.keys(), 
                               key=lambda k: self.feature_cache[k]['timestamp'])
                del self.feature_cache[oldest_key]
        
        return features
    
    def extract_streaming_features(self, symbol: str, new_bar: Dict[str, float], 
                                 history: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Extract features for streaming data (single new bar)
        Optimized for minimal latency
        """
        # Update history with new bar
        updated_history = {}
        for key in history:
            updated_history[key] = np.append(history[key][-99:], new_bar[key])  # Keep last 100 bars
        
        return self.extract_features(symbol, updated_history, time.time())
    
    def get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dict to numpy array for ML models"""
        feature_keys = [
            'return_1', 'return_5', 'log_return_1',
            'rsi_14', 'bb_position', 'macd_signal',
            'momentum_5', 'momentum_20', 'momentum_ratio',
            'volatility_5', 'volatility_20', 'volatility_ratio',
            'range_percentage', 'price_vs_vwap', 'volume_ratio',
            'gap_percent'
        ]
        
        return np.array([features.get(key, 0.0) for key in feature_keys])
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        self.last_update.clear()
