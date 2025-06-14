"""
Volatility Mean Reversion Features

Specialized feature extraction for volatility-based mean reversion strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from numba import jit


@jit(nopython=True)
def calculate_volatility_metrics(returns: np.ndarray, window: int = 20) -> Tuple[float, float, float]:
    """Calculate volatility metrics with Numba acceleration"""
    
    if len(returns) < window:
        return 0.01, 1.0, 0.0
    
    # Current volatility (recent window)
    recent_returns = returns[-window:]
    current_vol = np.std(recent_returns)
    
    # Historical volatility (longer window)
    if len(returns) >= window * 2:
        historical_returns = returns[:-window]
        historical_vol = np.std(historical_returns)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    
    # Mean reversion tendency
    mean_return = np.mean(recent_returns)
    
    return current_vol, vol_ratio, mean_return


class VolatilityFeatureExtractor:
    """
    Volatility mean reversion feature extraction
    
    Extracts features for volatility-based trading:
    - Volatility regimes and ratios
    - Mean reversion signals
    - Price extremes and reversals
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.cache_ttl = 180  # 3 minutes
    
    def extract_vol_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray], 
                           timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Extract volatility mean reversion features"""
        
        start_time = time.perf_counter()
        
        try:
            features = self._compute_vol_features(ohlcv_data)
            
            features.update({
                'symbol': symbol,
                'extraction_time_ms': (time.perf_counter() - start_time) * 1000,
                'timestamp': timestamp or time.time(),
                'feature_count': len([k for k in features.keys() if not k.startswith('_')])
            })
            
            return features
            
        except Exception as e:
            print(f"Error extracting volatility features for {symbol}: {e}")
            return self._empty_features(symbol)
    
    def _compute_vol_features(self, ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute volatility-specific features"""
        
        closes = ohlcv_data['close']
        highs = ohlcv_data['high']
        lows = ohlcv_data['low']
        volumes = ohlcv_data['volume']
        
        # Calculate returns
        if len(closes) < 2:
            returns = np.array([0.0])
        else:
            returns = np.diff(closes) / closes[:-1]
        
        # Volatility metrics
        current_vol, vol_ratio, mean_return = calculate_volatility_metrics(returns)
        
        # Price extremes
        extremes = self._identify_price_extremes(closes, highs, lows)
        
        # Mean reversion signals
        reversion_signals = self._calculate_reversion_signals(closes, returns, current_vol)
        
        # Volume-volatility relationship
        vol_volume_metrics = self._analyze_volume_volatility(returns, volumes)
        
        return {
            # Core volatility metrics
            'current_volatility': float(current_vol),
            'volatility_ratio': float(vol_ratio),
            'mean_return': float(mean_return),
            
            # Price extremes
            **extremes,
            
            # Reversion signals
            **reversion_signals,
            
            # Volume-volatility
            **vol_volume_metrics,
            
            # Trading signals
            'high_vol_regime': bool(vol_ratio > 1.5),
            'reversion_opportunity': bool(vol_ratio > 1.3 and abs(extremes.get('price_extreme', 0)) > 0.5)
        }
    
    def _identify_price_extremes(self, closes: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray) -> Dict[str, float]:
        """Identify price extremes for mean reversion"""
        
        if len(closes) < 20:
            return {
                'price_extreme': 0.0,
                'extreme_direction': 0.0,
                'bollinger_position': 0.5,
                'price_stretch': 0.0
            }
        
        # Bollinger Band position
        recent_closes = closes[-20:]
        bb_mean = np.mean(recent_closes)
        bb_std = np.std(recent_closes)
        
        current_price = closes[-1]
        
        if bb_std > 0:
            bb_position = (current_price - bb_mean) / (2 * bb_std) + 0.5
            bb_position = np.clip(bb_position, 0.0, 1.0)
        else:
            bb_position = 0.5
        
        # Price extreme calculation
        if bb_position > 0.8:
            price_extreme = (bb_position - 0.5) * 2  # Scale to [-1, 1]
            extreme_direction = 1.0
        elif bb_position < 0.2:
            price_extreme = (bb_position - 0.5) * 2
            extreme_direction = -1.0
        else:
            price_extreme = 0.0
            extreme_direction = 0.0
        
        # Price stretch (distance from recent range)
        if len(closes) >= 10:
            recent_high = np.max(highs[-10:])
            recent_low = np.min(lows[-10:])
            range_size = recent_high - recent_low
            
            if range_size > 0:
                if current_price > recent_high:
                    price_stretch = (current_price - recent_high) / range_size
                elif current_price < recent_low:
                    price_stretch = (recent_low - current_price) / range_size
                else:
                    price_stretch = 0.0
            else:
                price_stretch = 0.0
        else:
            price_stretch = 0.0
        
        return {
            'price_extreme': float(price_extreme),
            'extreme_direction': float(extreme_direction),
            'bollinger_position': float(bb_position),
            'price_stretch': float(price_stretch)
        }
    
    def _calculate_reversion_signals(self, closes: np.ndarray, returns: np.ndarray, 
                                   current_vol: float) -> Dict[str, float]:
        """Calculate mean reversion signals"""
        
        if len(returns) < 10:
            return {
                'reversion_strength': 0.0,
                'momentum_exhaustion': 0.0,
                'vol_mean_reversion': 0.0
            }
        
        # Recent momentum
        recent_momentum = np.sum(returns[-5:]) if len(returns) >= 5 else 0.0
        
        # Momentum exhaustion (high momentum + high volatility)
        momentum_strength = abs(recent_momentum)
        vol_percentile = self._calculate_vol_percentile(returns, current_vol)
        
        momentum_exhaustion = momentum_strength * vol_percentile
        
        # Volatility mean reversion
        if len(returns) >= 20:
            long_term_vol = np.std(returns[-20:])
            vol_mean_reversion = (current_vol - long_term_vol) / long_term_vol if long_term_vol > 0 else 0.0
        else:
            vol_mean_reversion = 0.0
        
        # Overall reversion strength
        reversion_strength = (momentum_exhaustion + abs(vol_mean_reversion)) / 2.0
        
        return {
            'reversion_strength': float(reversion_strength),
            'momentum_exhaustion': float(momentum_exhaustion),
            'vol_mean_reversion': float(vol_mean_reversion)
        }
    
    def _calculate_vol_percentile(self, returns: np.ndarray, current_vol: float) -> float:
        """Calculate current volatility percentile"""
        
        if len(returns) < 50:
            return 0.5
        
        # Rolling volatilities
        vol_window = 10
        rolling_vols = []
        
        for i in range(vol_window, len(returns)):
            window_returns = returns[i-vol_window:i]
            rolling_vols.append(np.std(window_returns))
        
        if not rolling_vols:
            return 0.5
        
        rolling_vols = np.array(rolling_vols)
        percentile = np.sum(rolling_vols <= current_vol) / len(rolling_vols)
        
        return float(percentile)
    
    def _analyze_volume_volatility(self, returns: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Analyze volume-volatility relationship"""
        
        if len(returns) < 10 or len(volumes) < 10:
            return {
                'volume_vol_correlation': 0.0,
                'volume_surge': 0.0,
                'volume_exhaustion': 0.0
            }
        
        # Volume-volatility correlation
        if len(returns) >= len(volumes):
            vol_returns = abs(returns[-len(volumes):])
        else:
            vol_returns = abs(returns)
            volumes = volumes[-len(returns):]
        
        if len(vol_returns) >= 10:
            correlation = np.corrcoef(vol_returns[-10:], volumes[-10:])[0, 1]
            volume_vol_correlation = correlation if not np.isnan(correlation) else 0.0
        else:
            volume_vol_correlation = 0.0
        
        # Volume surge
        if len(volumes) >= 10:
            recent_volume = np.mean(volumes[-3:])
            historical_volume = np.mean(volumes[-10:-3])
            volume_surge = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0.0
        else:
            volume_surge = 0.0
        
        # Volume exhaustion (high volume with decreasing momentum)
        if len(volumes) >= 5 and len(returns) >= 5:
            recent_vol_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
            recent_price_momentum = np.sum(returns[-5:])
            
            if recent_vol_trend > 0 and abs(recent_price_momentum) < 0.01:
                volume_exhaustion = min(recent_vol_trend / np.mean(volumes[-10:]), 1.0) if len(volumes) >= 10 else 0.0
            else:
                volume_exhaustion = 0.0
        else:
            volume_exhaustion = 0.0
        
        return {
            'volume_vol_correlation': float(volume_vol_correlation),
            'volume_surge': float(volume_surge),
            'volume_exhaustion': float(volume_exhaustion)
        }
    
    def _empty_features(self, symbol: str) -> Dict[str, Any]:
        """Return empty features on error"""
        return {
            'symbol': symbol,
            'current_volatility': 0.01,
            'volatility_ratio': 1.0,
            'mean_return': 0.0,
            'price_extreme': 0.0,
            'extreme_direction': 0.0,
            'bollinger_position': 0.5,
            'price_stretch': 0.0,
            'reversion_strength': 0.0,
            'momentum_exhaustion': 0.0,
            'vol_mean_reversion': 0.0,
            'volume_vol_correlation': 0.0,
            'volume_surge': 0.0,
            'volume_exhaustion': 0.0,
            'high_vol_regime': False,
            'reversion_opportunity': False,
            'error': True
        }
    
    def get_vol_signal(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Generate mean reversion signal from volatility features"""
        
        volatility_ratio = features.get('volatility_ratio', 1.0)
        price_extreme = features.get('price_extreme', 0.0)
        reversion_strength = features.get('reversion_strength', 0.0)
        extreme_direction = features.get('extreme_direction', 0.0)
        
        # Mean reversion signal (opposite to extreme direction)
        base_signal = -extreme_direction * min(volatility_ratio / 2.0, 1.0)
        
        # Adjust by reversion strength
        adjusted_signal = base_signal * reversion_strength
        
        # Confidence based on volatility regime
        confidence = min(volatility_ratio / 2.0, 1.0) * reversion_strength
        
        return {
            'signal': float(np.clip(adjusted_signal, -1.0, 1.0)),
            'confidence': float(confidence),
            'strength': float(reversion_strength),
            'vol_ratio': float(volatility_ratio),
            'extreme': float(price_extreme)
        }
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
