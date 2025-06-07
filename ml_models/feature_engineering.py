#!/usr/bin/env python3

import numpy as np
import talib
from numba import jit
from typing import Dict, List
import os
import sys
import yaml
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import unified system logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.system_logger import get_system_logger

# Initialize component logger
logger = get_system_logger("feature_engineering")

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'ml_models.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

class FeatureEngineer:
    """
    High-performance feature engineering pipeline optimized for real-time trading
    Implements proven features from ML4T with <20ms latency target
    """
    
    def __init__(self, cache_ttl: int = None):
        logger.info("Initializing Feature Engineer", extra={
            "component": "feature_engineering",
            "action": "initialization_start",
            "cache_ttl": cache_ttl
        })
        
        # Load configuration
        feature_config = CONFIG.get('feature_engineering', {})
        performance_config = CONFIG.get('performance', {})
        
        # Setup cache with config values
        cache_ttl = cache_ttl or 300
        self.cache = TTLCache(maxsize=10000, ttl=cache_ttl)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance targets from config
        self.performance_targets = {
            'feature_extraction_ms': performance_config.get('feature_extraction_ms', 50),
            'total_pipeline_ms': performance_config.get('total_pipeline_ms', 100)
        }
        
        # Feature categories from YAML config
        if feature_config and feature_config.get('enabled', True):
            self.feature_config = feature_config
        else:
            self.feature_config = self._get_default_config()
        
        logger.info("Feature Engineer initialized successfully", extra={
            "component": "feature_engineering",
            "action": "initialization_complete",
            "cache_maxsize": 10000,
            "cache_ttl": cache_ttl,
            "thread_pool_workers": 4,
            "performance_targets": self.performance_targets,
            "feature_config_loaded": bool(feature_config),
            "feature_categories": list(self.feature_config.keys())
        })
        
    def _get_default_config(self):
        """Default feature configuration if YAML config is not available"""
        return {
            'price_features': {
                'lagged_returns': [1, 2, 3, 4, 5, 10],  # minutes
                'price_positioning': ['vwap', 'ma20', 'ma50']
            },
            'volume_features': {
                'direction_ratios': ['uptick', 'downtick', 'repeat_uptick', 'repeat_downtick'],
                'volume_patterns': ['surge', 'relative', 'trend']
            },
            'technical_indicators': {
                'momentum': ['RSI', 'STOCHRSI', 'CCI', 'MFI'],
                'volatility': ['ATR', 'NATR', 'BB_WIDTH'],
                'trend': ['MACD', 'BOP', 'ADX']
            },
            'market_context': {
                'time_features': ['minute_of_day', 'session_progress'],
                'regime_features': ['vix_level', 'market_breadth']
            },
            'order_flow': {
                'bid_ask': ['imbalance', 'spread_ratio'],
                'trade_flow': ['at_bid_ratio', 'at_ask_ratio', 'at_mid_ratio']
            }
        }
        
    async def engineer_features(self, market_data: Dict) -> np.ndarray:
        """
        Main feature engineering pipeline
        Target: <20ms total processing time
        """
        import time
        start_time = time.time()
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        logger.debug("Starting feature engineering pipeline", extra={
            "component": "feature_engineering",
            "action": "engineer_features_start",
            "symbol": symbol,
            "timestamp": market_data.get('timestamp', 0),
            "data_keys": list(market_data.keys()),
            "target_time_ms": self.performance_targets['feature_extraction_ms']
        })
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(market_data)
            if cache_key in self.cache:
                logger.debug("Features retrieved from cache", extra={
                    "component": "feature_engineering",
                    "action": "cache_hit",
                    "symbol": symbol,
                    "cache_key": cache_key
                })
                return self.cache[cache_key]
            
            logger.debug("Computing features (cache miss)", extra={
                "component": "feature_engineering",
                "action": "cache_miss",
                "symbol": symbol,
                "cache_key": cache_key
            })
            
            # Parallel feature computation
            compute_start = time.time()
            tasks = [
                self._compute_price_features(market_data),
                self._compute_volume_features(market_data),
                self._compute_technical_indicators(market_data),
                self._compute_market_context(market_data),
                self._compute_order_flow_features(market_data)
            ]
            
            feature_groups = await asyncio.gather(*tasks)
            compute_time = (time.time() - compute_start) * 1000
            
            # Combine all features
            combine_start = time.time()
            features = np.concatenate(feature_groups)
            combine_time = (time.time() - combine_start) * 1000
            
            # Cache result
            self.cache[cache_key] = features
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info("Feature engineering completed", extra={
                "component": "feature_engineering",
                "action": "engineer_features_complete",
                "symbol": symbol,
                "total_features": len(features),
                "feature_groups": [len(group) for group in feature_groups],
                "compute_time_ms": compute_time,
                "combine_time_ms": combine_time,
                "processing_time_ms": processing_time,
                "target_time_ms": self.performance_targets['feature_extraction_ms'],
                "within_target": processing_time < self.performance_targets['feature_extraction_ms'],
                "cache_size": len(self.cache)
            })
            
            return features
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error("Feature engineering pipeline failed", extra={
                "component": "feature_engineering",
                "action": "engineer_features_error",
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time_ms
            })
            return np.zeros(self._get_feature_count())
    
    async def _compute_price_features(self, data: Dict) -> np.ndarray:
        """Compute price-based features (6 features)"""
        try:
            prices = data.get('prices', {})
            current_price = prices.get('close', 0)
            
            features = []
            
            # Lagged returns (5 features)
            for lag in [1, 2, 3, 4, 5]:
                lag_price = prices.get(f'close_{lag}min_ago', current_price)
                ret = (current_price / lag_price - 1) if lag_price > 0 else 0
                features.append(ret)
            
            # Price vs VWAP (1 feature)
            vwap = prices.get('vwap', current_price)
            price_vs_vwap = (current_price / vwap - 1) if vwap > 0 else 0
            features.append(price_vs_vwap)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Price features error: {e}")
            return np.zeros(6, dtype=np.float32)
    
    async def _compute_volume_features(self, data: Dict) -> np.ndarray:
        """Compute volume-based features (4 features)"""
        try:
            volume_data = data.get('volume', {})
            total_volume = volume_data.get('total', 1)
            
            features = []
            
            # Volume direction ratios (2 features)
            uptick_volume = volume_data.get('uptick', 0)
            downtick_volume = volume_data.get('downtick', 0)
            
            uptick_ratio = uptick_volume / total_volume if total_volume > 0 else 0
            downtick_ratio = downtick_volume / total_volume if total_volume > 0 else 0
            
            features.extend([uptick_ratio, downtick_ratio])
            
            # Volume surge (1 feature)
            avg_volume = volume_data.get('avg_20min', total_volume)
            volume_surge = total_volume / avg_volume if avg_volume > 0 else 1
            features.append(volume_surge)
            
            # Relative volume percentile (1 feature)
            volume_percentile = volume_data.get('percentile', 0.5)
            features.append(volume_percentile)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Volume features error: {e}")
            return np.zeros(4, dtype=np.float32)
    
    async def _compute_technical_indicators(self, data: Dict) -> np.ndarray:
        """Compute technical indicators (8 features)"""
        try:
            ohlcv = data.get('ohlcv', {})
            
            # Get price arrays (last 50 periods for indicators)
            high = np.array(ohlcv.get('high', []), dtype=np.float64)
            low = np.array(ohlcv.get('low', []), dtype=np.float64)
            close = np.array(ohlcv.get('close', []), dtype=np.float64)
            np.array(ohlcv.get('volume', []), dtype=np.float64)
            
            if len(close) < 20:  # Need minimum data
                return np.zeros(8, dtype=np.float32)
            
            features = []
            
            # RSI (1 feature)
            rsi = talib.RSI(close, timeperiod=14)
            features.append(rsi[-1] / 100.0 if not np.isnan(rsi[-1]) else 0.5)
            
            # MACD signal (1 feature)
            macd, macd_signal, _ = talib.MACD(close)
            macd_pos = 1 if macd[-1] > macd_signal[-1] else 0
            features.append(macd_pos)
            
            # Bollinger Band position (1 feature)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            bb_pos = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] > bb_lower[-1] else 0.5
            features.append(bb_pos)
            
            # ATR percentile (1 feature)
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_percentile = np.percentile(atr[~np.isnan(atr)], 50) if len(atr[~np.isnan(atr)]) > 0 else 0
            features.append(atr_percentile)
            
            # Stochastic %K (1 feature)
            slowk, _ = talib.STOCH(high, low, close)
            features.append(slowk[-1] / 100.0 if not np.isnan(slowk[-1]) else 0.5)
            
            # Williams %R (1 feature)
            willr = talib.WILLR(high, low, close)
            features.append((willr[-1] + 100) / 100.0 if not np.isnan(willr[-1]) else 0.5)
            
            # CCI (1 feature)
            cci = talib.CCI(high, low, close)
            cci_norm = np.tanh(cci[-1] / 100.0) if not np.isnan(cci[-1]) else 0
            features.append(cci_norm)
            
            # BOP (1 feature)
            bop = talib.BOP(ohlcv.get('open', close), high, low, close)
            features.append(bop[-1] if not np.isnan(bop[-1]) else 0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Technical indicators error: {e}")
            return np.zeros(8, dtype=np.float32)
    
    async def _compute_market_context(self, data: Dict) -> np.ndarray:
        """Compute market context features (4 features)"""
        try:
            context = data.get('market_context', {})
            
            features = []
            
            # VIX level (normalized) (1 feature)
            vix = context.get('vix', 20)
            vix_norm = np.tanh((vix - 20) / 10)  # Normalize around 20
            features.append(vix_norm)
            
            # Market session (1 feature)
            minute_of_day = context.get('minute_of_day', 390)  # Minutes since 9:30
            session_progress = minute_of_day / 390.0  # Normalize to [0,1]
            features.append(session_progress)
            
            # Day of week (1 feature)
            day_of_week = context.get('day_of_week', 2)  # 0=Monday, 4=Friday
            day_norm = day_of_week / 4.0
            features.append(day_norm)
            
            # Market breadth (1 feature)
            breadth = context.get('market_breadth', 0.5)  # SPY/QQQ/IWM correlation
            features.append(breadth)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Market context error: {e}")
            return np.zeros(4, dtype=np.float32)
    
    async def _compute_order_flow_features(self, data: Dict) -> np.ndarray:
        """Compute order flow features (3 features)"""
        try:
            order_flow = data.get('order_flow', {})
            
            features = []
            
            # Bid-ask imbalance (1 feature)
            bid_size = order_flow.get('bid_size', 1)
            ask_size = order_flow.get('ask_size', 1)
            imbalance = (bid_size - ask_size) / (bid_size + ask_size)
            features.append(imbalance)
            
            # Trade at bid vs ask ratio (1 feature)
            trades_at_bid = order_flow.get('trades_at_bid', 0)
            trades_at_ask = order_flow.get('trades_at_ask', 0)
            total_trades = trades_at_bid + trades_at_ask
            bid_ask_ratio = (trades_at_ask - trades_at_bid) / total_trades if total_trades > 0 else 0
            features.append(bid_ask_ratio)
            
            # Spread ratio (1 feature)
            spread = order_flow.get('spread', 0.01)
            mid_price = order_flow.get('mid_price', 100)
            spread_ratio = spread / mid_price if mid_price > 0 else 0
            features.append(spread_ratio)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Order flow features error: {e}")
            return np.zeros(3, dtype=np.float32)
    
    def _generate_cache_key(self, data: Dict) -> str:
        """Generate cache key for feature data"""
        try:
            # Use timestamp and symbol for cache key
            timestamp = data.get('timestamp', 0)
            symbol = data.get('symbol', 'UNKNOWN')
            return f"{symbol}_{timestamp}"
        except:
            return f"default_{hash(str(data))}"
    
    def _get_feature_count(self) -> int:
        """Total number of features: 6 + 4 + 8 + 4 + 3 = 25"""
        return 25
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for interpretability"""
        return [
            # Price features (6)
            'ret_1min', 'ret_2min', 'ret_3min', 'ret_4min', 'ret_5min', 'price_vs_vwap',
            
            # Volume features (4)
            'uptick_ratio', 'downtick_ratio', 'volume_surge', 'volume_percentile',
            
            # Technical indicators (8)
            'rsi_norm', 'macd_signal', 'bb_position', 'atr_percentile', 
            'stoch_k', 'williams_r', 'cci_norm', 'bop',
            
            # Market context (4)
            'vix_norm', 'session_progress', 'day_of_week', 'market_breadth',
            
            # Order flow (3)
            'bid_ask_imbalance', 'bid_ask_trade_ratio', 'spread_ratio'
        ]

@jit(nopython=True)
def fast_returns_calculation(prices: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Numba-optimized returns calculation"""
    n = len(prices)
    n_lags = len(lags)
    returns = np.zeros(n_lags, dtype=np.float32)
    
    current_price = prices[-1]
    
    for i in range(n_lags):
        lag = lags[i]
        if lag < n:
            lag_price = prices[-(lag + 1)]
            if lag_price > 0:
                returns[i] = (current_price / lag_price) - 1.0
    
    return returns

@jit(nopython=True)
def fast_volume_ratios(uptick: float, downtick: float, total: float) -> np.ndarray:
    """Numba-optimized volume ratio calculation"""
    if total > 0:
        return np.array([uptick / total, downtick / total], dtype=np.float32)
    else:
        return np.array([0.0, 0.0], dtype=np.float32)