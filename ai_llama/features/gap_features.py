"""
Gap & Go Strategy Features

Specialized feature extraction for gap trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from numba import jit


@jit(nopython=True)
def calculate_gap_metrics(close_prices: np.ndarray, open_prices: np.ndarray, 
                         volumes: np.ndarray) -> Tuple[float, float, float]:
    """Calculate gap-specific metrics with Numba acceleration"""
    
    if len(close_prices) < 2 or len(open_prices) < 1:
        return 0.0, 0.0, 0.0
    
    # Current gap
    prev_close = close_prices[-2] if len(close_prices) >= 2 else close_prices[-1]
    current_open = open_prices[-1]
    gap_percent = (current_open - prev_close) / prev_close
    
    # Volume analysis
    if len(volumes) >= 10:
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:-1])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
    else:
        volume_ratio = 1.0
    
    # Gap strength (based on size and volume)
    gap_strength = abs(gap_percent) * min(volume_ratio, 5.0)  # Cap volume impact
    
    return gap_percent, volume_ratio, gap_strength


@jit(nopython=True)
def calculate_premarket_metrics(premarket_volumes: np.ndarray, 
                               regular_volumes: np.ndarray) -> float:
    """Calculate premarket activity metrics"""
    
    if len(premarket_volumes) == 0 or len(regular_volumes) == 0:
        return 1.0
    
    avg_premarket = np.mean(premarket_volumes)
    avg_regular = np.mean(regular_volumes[-5:])  # Last 5 regular sessions
    
    if avg_regular > 0:
        return avg_premarket / avg_regular
    else:
        return 1.0


class GapFeatureExtractor:
    """
    Gap & Go strategy feature extraction
    
    Extracts features specific to gap trading:
    - Gap size and direction
    - Volume confirmation
    - Premarket activity
    - Historical gap performance
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def extract_gap_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray], 
                           premarket_data: Optional[Dict[str, np.ndarray]] = None,
                           timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract comprehensive gap trading features
        
        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV data dictionary
            premarket_data: Optional premarket data
            timestamp: Current timestamp
            
        Returns:
            Gap-specific features
        """
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{symbol}_gap_{timestamp}"
        if cache_key in self.feature_cache:
            cached_entry = self.feature_cache[cache_key]
            if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                return cached_entry['features']
        
        try:
            features = self._compute_gap_features(ohlcv_data, premarket_data)
            
            # Add metadata
            features.update({
                'symbol': symbol,
                'extraction_time_ms': (time.perf_counter() - start_time) * 1000,
                'timestamp': timestamp or time.time(),
                'feature_count': len([k for k in features.keys() if not k.startswith('_')])
            })
            
            # Cache result
            self.feature_cache[cache_key] = {
                'features': features,
                'timestamp': time.time()
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting gap features for {symbol}: {e}")
            return self._empty_features(symbol)
    
    def _compute_gap_features(self, ohlcv_data: Dict[str, np.ndarray], 
                            premarket_data: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Compute gap-specific features"""
        
        # Extract arrays
        opens = ohlcv_data['open']
        highs = ohlcv_data['high']
        lows = ohlcv_data['low']
        closes = ohlcv_data['close']
        volumes = ohlcv_data['volume']
        
        # Basic gap metrics
        gap_percent, volume_ratio, gap_strength = calculate_gap_metrics(closes, opens, volumes)
        
        # Gap direction and size categorization
        gap_direction = 1 if gap_percent > 0 else -1 if gap_percent < 0 else 0
        gap_size_category = self._categorize_gap_size(abs(gap_percent))
        
        # Volume confirmation
        volume_confirmation = self._calculate_volume_confirmation(volumes, volume_ratio)
        
        # Premarket metrics
        premarket_volume_ratio = 1.0
        if premarket_data and 'volume' in premarket_data:
            premarket_volume_ratio = calculate_premarket_metrics(
                premarket_data['volume'], volumes
            )
        
        # Gap quality score
        gap_quality = self._calculate_gap_quality(
            abs(gap_percent), volume_ratio, premarket_volume_ratio
        )
        
        # Technical context
        price_context = self._analyze_price_context(opens, highs, lows, closes)
        
        # Historical gap performance (simplified)
        historical_performance = self._estimate_historical_performance(
            gap_percent, volume_ratio
        )
        
        return {
            # Core gap metrics
            'gap_percent': float(gap_percent),
            'gap_direction': int(gap_direction),
            'gap_size_category': int(gap_size_category),
            'gap_strength': float(gap_strength),
            
            # Volume metrics
            'volume_ratio': float(volume_ratio),
            'volume_confirmation': float(volume_confirmation),
            'premarket_volume_ratio': float(premarket_volume_ratio),
            
            # Quality scores
            'gap_quality': float(gap_quality),
            'historical_performance': float(historical_performance),
            
            # Price context
            **price_context,
            
            # Trading signals
            'is_tradeable_gap': bool(gap_quality > 0.6),
            'expected_direction': int(gap_direction),
            'confidence_score': float(gap_quality * volume_confirmation)
        }
    
    def _categorize_gap_size(self, gap_size: float) -> int:
        """Categorize gap size (0=small, 1=medium, 2=large, 3=extreme)"""
        if gap_size < 0.01:      # < 1%
            return 0
        elif gap_size < 0.03:    # 1-3%
            return 1
        elif gap_size < 0.05:    # 3-5%
            return 2
        else:                    # > 5%
            return 3
    
    def _calculate_volume_confirmation(self, volumes: np.ndarray, volume_ratio: float) -> float:
        """Calculate volume confirmation strength"""
        
        # Base confirmation from volume ratio
        base_confirmation = min(volume_ratio / 2.0, 1.0)  # Cap at 2x normal volume
        
        # Penalty for extremely high volume (might indicate distribution)
        if volume_ratio > 10.0:
            base_confirmation *= 0.7
        
        return base_confirmation
    
    def _calculate_gap_quality(self, gap_size: float, volume_ratio: float, 
                              premarket_volume_ratio: float) -> float:
        """Calculate overall gap quality score"""
        
        # Size component (optimal range: 2-4%)
        if 0.02 <= gap_size <= 0.04:
            size_score = 1.0
        elif 0.01 <= gap_size <= 0.06:
            size_score = 0.8
        elif gap_size <= 0.08:
            size_score = 0.6
        else:
            size_score = 0.4  # Too large might be news-driven
        
        # Volume component
        volume_score = min(volume_ratio / 2.0, 1.0) * 0.9 + 0.1
        
        # Premarket component
        premarket_score = min(premarket_volume_ratio / 1.5, 1.0) * 0.7 + 0.3
        
        # Combined score
        quality_score = (
            0.5 * size_score + 
            0.3 * volume_score + 
            0.2 * premarket_score
        )
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def _analyze_price_context(self, opens: np.ndarray, highs: np.ndarray, 
                              lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
        """Analyze price context around the gap"""
        
        if len(closes) < 5:
            return {
                'recent_volatility': 0.01,
                'price_trend': 0.0,
                'support_resistance': 0.0
            }
        
        # Recent volatility
        recent_returns = np.diff(closes[-5:]) / closes[-5:-1]
        recent_volatility = float(np.std(recent_returns))
        
        # Price trend (last 5 periods)
        if len(closes) >= 5:
            trend_slope = (closes[-1] - closes[-5]) / closes[-5]
            price_trend = float(trend_slope)
        else:
            price_trend = 0.0
        
        # Support/resistance (simplified)
        if len(closes) >= 10:
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            current_price = closes[-1]
            
            # Distance to recent high/low
            max_high = np.max(recent_highs)
            min_low = np.min(recent_lows)
            
            if max_high > min_low:
                support_resistance = (current_price - min_low) / (max_high - min_low)
            else:
                support_resistance = 0.5
        else:
            support_resistance = 0.5
        
        return {
            'recent_volatility': float(recent_volatility),
            'price_trend': float(price_trend),
            'support_resistance': float(support_resistance)
        }
    
    def _estimate_historical_performance(self, gap_percent: float, volume_ratio: float) -> float:
        """Estimate historical performance based on gap characteristics"""
        
        # Simple heuristic based on gap size and volume
        gap_size = abs(gap_percent)
        
        # Optimal gap characteristics for performance
        if 0.02 <= gap_size <= 0.04 and 1.5 <= volume_ratio <= 4.0:
            return 0.8
        elif 0.015 <= gap_size <= 0.06 and 1.2 <= volume_ratio <= 6.0:
            return 0.6
        elif gap_size >= 0.01 and volume_ratio >= 1.1:
            return 0.4
        else:
            return 0.2
    
    def _empty_features(self, symbol: str) -> Dict[str, Any]:
        """Return empty features on error"""
        return {
            'symbol': symbol,
            'gap_percent': 0.0,
            'gap_direction': 0,
            'gap_size_category': 0,
            'gap_strength': 0.0,
            'volume_ratio': 1.0,
            'volume_confirmation': 0.0,
            'premarket_volume_ratio': 1.0,
            'gap_quality': 0.0,
            'historical_performance': 0.0,
            'recent_volatility': 0.01,
            'price_trend': 0.0,
            'support_resistance': 0.5,
            'is_tradeable_gap': False,
            'expected_direction': 0,
            'confidence_score': 0.0,
            'error': True
        }
    
    def get_gap_signal(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signal from gap features"""
        
        gap_quality = features.get('gap_quality', 0.0)
        gap_direction = features.get('gap_direction', 0)
        confidence_score = features.get('confidence_score', 0.0)
        volume_confirmation = features.get('volume_confirmation', 0.0)
        
        # Base signal strength
        signal_strength = gap_quality * volume_confirmation
        
        # Directional signal
        signal = gap_direction * signal_strength
        
        return {
            'signal': float(np.clip(signal, -1.0, 1.0)),
            'confidence': float(confidence_score),
            'strength': float(signal_strength),
            'direction': float(gap_direction),
            'quality': float(gap_quality)
        }
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
