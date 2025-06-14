"""
Opening Range Breakout (ORB) Features

Specialized feature extraction for ORB trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from numba import jit


@jit(nopython=True)
def calculate_orb_metrics(opens: np.ndarray, highs: np.ndarray, 
                         lows: np.ndarray, closes: np.ndarray,
                         volumes: np.ndarray, orb_minutes: int = 30) -> Tuple[float, float, float, float]:
    """Calculate ORB-specific metrics with Numba acceleration"""
    
    if len(opens) < orb_minutes:
        return 0.0, 0.0, 0.0, 0.0
    
    # Opening range (first orb_minutes bars)
    range_high = np.max(highs[:orb_minutes])
    range_low = np.min(lows[:orb_minutes])
    range_size = range_high - range_low
    
    # Current price and breakout status
    current_price = closes[-1]
    
    # Breakout direction and strength
    if current_price > range_high:
        breakout_direction = 1.0
        breakout_strength = (current_price - range_high) / range_size if range_size > 0 else 0.0
    elif current_price < range_low:
        breakout_direction = -1.0
        breakout_strength = (range_low - current_price) / range_size if range_size > 0 else 0.0
    else:
        breakout_direction = 0.0
        breakout_strength = 0.0
    
    # Volume confirmation
    if len(volumes) >= orb_minutes:
        breakout_volume = volumes[-1]
        range_avg_volume = np.mean(volumes[:orb_minutes])
        volume_confirmation = breakout_volume / range_avg_volume if range_avg_volume > 0 else 1.0
    else:
        volume_confirmation = 1.0
    
    return breakout_direction, breakout_strength, volume_confirmation, range_size


class ORBFeatureExtractor:
    """
    Opening Range Breakout feature extraction
    
    Extracts features specific to ORB trading:
    - Opening range metrics
    - Breakout strength and direction
    - Volume confirmation
    - Time-based factors
    """
    
    def __init__(self, orb_minutes: int = 30):
        self.orb_minutes = orb_minutes
        self.feature_cache = {}
        self.cache_ttl = 60  # 1 minute
    
    def extract_orb_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray], 
                           current_time: Optional[float] = None,
                           market_open_time: Optional[float] = None) -> Dict[str, Any]:
        """Extract comprehensive ORB trading features"""
        
        start_time = time.perf_counter()
        
        try:
            features = self._compute_orb_features(ohlcv_data, current_time, market_open_time)
            
            features.update({
                'symbol': symbol,
                'extraction_time_ms': (time.perf_counter() - start_time) * 1000,
                'timestamp': current_time or time.time(),
                'feature_count': len([k for k in features.keys() if not k.startswith('_')])
            })
            
            return features
            
        except Exception as e:
            print(f"Error extracting ORB features for {symbol}: {e}")
            return self._empty_features(symbol)
    
    def _compute_orb_features(self, ohlcv_data: Dict[str, np.ndarray], 
                            current_time: Optional[float],
                            market_open_time: Optional[float]) -> Dict[str, Any]:
        """Compute ORB-specific features"""
        
        opens = ohlcv_data['open']
        highs = ohlcv_data['high']
        lows = ohlcv_data['low']
        closes = ohlcv_data['close']
        volumes = ohlcv_data['volume']
        
        # Basic ORB metrics
        breakout_direction, breakout_strength, volume_confirmation, range_size = calculate_orb_metrics(
            opens, highs, lows, closes, volumes, self.orb_minutes
        )
        
        # Time factors
        time_factors = self._calculate_time_factors(current_time, market_open_time)
        
        # Range quality
        range_quality = self._assess_range_quality(opens, highs, lows, closes, volumes)
        
        # Momentum factors
        momentum = self._calculate_momentum(closes, volumes)
        
        # ORB signal strength
        orb_signal_strength = self._calculate_orb_signal_strength(
            breakout_direction, breakout_strength, volume_confirmation, 
            range_quality, time_factors
        )
        
        return {
            # Core ORB metrics
            'breakout_direction': float(breakout_direction),
            'breakout_strength': float(breakout_strength),
            'range_size': float(range_size),
            'volume_confirmation': float(volume_confirmation),
            
            # Quality metrics
            'range_quality': float(range_quality),
            'orb_signal_strength': float(orb_signal_strength),
            
            # Time factors
            **time_factors,
            
            # Momentum
            **momentum,
            
            # Trading signals
            'is_breakout': bool(abs(breakout_direction) > 0),
            'is_strong_breakout': bool(breakout_strength > 0.5 and volume_confirmation > 1.5),
            'breakout_quality': float(orb_signal_strength)
        }
    
    def _calculate_time_factors(self, current_time: Optional[float], 
                              market_open_time: Optional[float]) -> Dict[str, float]:
        """Calculate time-based factors"""
        
        if not current_time or not market_open_time:
            return {
                'minutes_since_open': 30.0,
                'orb_period_complete': 1.0,
                'time_factor': 1.0
            }
        
        minutes_since_open = (current_time - market_open_time) / 60.0
        orb_period_complete = min(minutes_since_open / self.orb_minutes, 1.0)
        
        # Time factor decreases as day progresses
        if minutes_since_open <= 60:
            time_factor = 1.0
        elif minutes_since_open <= 120:
            time_factor = 0.8
        elif minutes_since_open <= 180:
            time_factor = 0.6
        else:
            time_factor = 0.4
        
        return {
            'minutes_since_open': float(minutes_since_open),
            'orb_period_complete': float(orb_period_complete),
            'time_factor': float(time_factor)
        }
    
    def _assess_range_quality(self, opens: np.ndarray, highs: np.ndarray, 
                            lows: np.ndarray, closes: np.ndarray, 
                            volumes: np.ndarray) -> float:
        """Assess the quality of the opening range"""
        
        if len(closes) < self.orb_minutes:
            return 0.5
        
        # Range characteristics
        range_bars = min(len(closes), self.orb_minutes)
        range_highs = highs[:range_bars]
        range_lows = lows[:range_bars]
        range_volumes = volumes[:range_bars]
        
        # Range tightness (smaller is better for breakouts)
        range_high = np.max(range_highs)
        range_low = np.min(range_lows)
        range_size = range_high - range_low
        
        avg_price = np.mean(closes[:range_bars])
        range_tightness = 1.0 - min(range_size / avg_price, 0.1) * 10  # Normalize
        
        # Volume consistency during range formation
        volume_cv = np.std(range_volumes) / np.mean(range_volumes) if np.mean(range_volumes) > 0 else 1.0
        volume_consistency = 1.0 / (1.0 + volume_cv)
        
        # Overall range quality
        quality = 0.6 * range_tightness + 0.4 * volume_consistency
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def _calculate_momentum(self, closes: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators"""
        
        if len(closes) < 10:
            return {
                'momentum_5': 0.0,
                'momentum_10': 0.0,
                'volume_momentum': 0.0
            }
        
        # Price momentum
        momentum_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0.0
        momentum_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0.0
        
        # Volume momentum
        if len(volumes) >= 5:
            recent_vol = np.mean(volumes[-3:])
            earlier_vol = np.mean(volumes[-8:-3])
            volume_momentum = (recent_vol - earlier_vol) / earlier_vol if earlier_vol > 0 else 0.0
        else:
            volume_momentum = 0.0
        
        return {
            'momentum_5': float(momentum_5),
            'momentum_10': float(momentum_10),
            'volume_momentum': float(volume_momentum)
        }
    
    def _calculate_orb_signal_strength(self, breakout_direction: float, 
                                     breakout_strength: float,
                                     volume_confirmation: float,
                                     range_quality: float,
                                     time_factors: Dict[str, float]) -> float:
        """Calculate overall ORB signal strength"""
        
        # Base strength from breakout
        base_strength = abs(breakout_direction) * min(breakout_strength, 2.0) / 2.0
        
        # Volume confirmation factor
        volume_factor = min(volume_confirmation / 2.0, 1.0)
        
        # Range quality factor
        quality_factor = range_quality
        
        # Time factor
        time_factor = time_factors.get('time_factor', 1.0)
        
        # Combined signal strength
        signal_strength = (
            0.4 * base_strength +
            0.3 * volume_factor +
            0.2 * quality_factor +
            0.1 * time_factor
        )
        
        return float(np.clip(signal_strength, 0.0, 1.0))
    
    def _empty_features(self, symbol: str) -> Dict[str, Any]:
        """Return empty features on error"""
        return {
            'symbol': symbol,
            'breakout_direction': 0.0,
            'breakout_strength': 0.0,
            'range_size': 0.0,
            'volume_confirmation': 1.0,
            'range_quality': 0.5,
            'orb_signal_strength': 0.0,
            'minutes_since_open': 30.0,
            'orb_period_complete': 1.0,
            'time_factor': 1.0,
            'momentum_5': 0.0,
            'momentum_10': 0.0,
            'volume_momentum': 0.0,
            'is_breakout': False,
            'is_strong_breakout': False,
            'breakout_quality': 0.0,
            'error': True
        }
    
    def get_orb_signal(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signal from ORB features"""
        
        breakout_direction = features.get('breakout_direction', 0.0)
        orb_signal_strength = features.get('orb_signal_strength', 0.0)
        volume_confirmation = features.get('volume_confirmation', 1.0)
        time_factor = features.get('time_factor', 1.0)
        
        # Directional signal
        signal = breakout_direction * orb_signal_strength * time_factor
        
        # Confidence based on volume and quality
        confidence = orb_signal_strength * min(volume_confirmation / 2.0, 1.0)
        
        return {
            'signal': float(np.clip(signal, -1.0, 1.0)),
            'confidence': float(confidence),
            'strength': float(orb_signal_strength),
            'direction': float(breakout_direction),
            'time_factor': float(time_factor)
        }
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
