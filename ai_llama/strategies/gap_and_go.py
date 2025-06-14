"""
AI-Enhanced Gap & Go Strategy

Combines traditional gap trading with AI predictions for improved performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import time

from ..features.gap_features import GapFeatureExtractor
from ..models.lag_llama_engine import LagLlamaEngine
from ..models.fast_models import GapQualityClassifier


class AIGapAndGo:
    """
    AI-Enhanced Gap & Go Strategy
    
    Features:
    - AI-powered gap quality assessment
    - Dynamic entry/exit points
    - Risk management integration
    - Performance tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.gap_extractor = GapFeatureExtractor()
        self.gap_classifier = GapQualityClassifier()
        self.lag_llama = LagLlamaEngine()
        
        # Strategy parameters
        self.min_gap_percent = config.get('min_gap_percent', 0.02)
        self.max_gap_percent = config.get('max_gap_percent', 0.08)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.5)
        self.ai_quality_threshold = config.get('ai_quality_threshold', 0.6)
        
        # Position management
        self.active_positions = {}
        self.trade_history = []
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0
        }
    
    def analyze_gap_opportunity(self, symbol: str, ohlcv_data: Dict[str, np.ndarray],
                              premarket_data: Optional[Dict[str, np.ndarray]] = None,
                              current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze gap trading opportunity using AI
        
        Returns comprehensive analysis with trading signals
        """
        
        # Extract gap features
        gap_features = self.gap_extractor.extract_gap_features(
            symbol, ohlcv_data, premarket_data, current_time
        )
        
        # Get AI quality assessment
        gap_quality = self.gap_classifier.classify_gap_quality({
            'gap_percent': gap_features.get('gap_percent', 0.0),
            'volume_ratio': gap_features.get('volume_ratio', 1.0),
            'premarket_volume_ratio': gap_features.get('premarket_volume_ratio', 1.0)
        })
        
        # Get Lag-Llama prediction
        lag_llama_prediction = self.lag_llama.predict(
            ohlcv_data['close'], symbol=symbol, timestamp=current_time
        )
        lag_llama_signals = self.lag_llama.get_trading_signal(lag_llama_prediction)
        
        # Generate trading decision
        trading_decision = self._make_trading_decision(
            gap_features, gap_quality, lag_llama_signals
        )
        
        return {
            'symbol': symbol,
            'gap_features': gap_features,
            'gap_quality': gap_quality,
            'lag_llama_signals': lag_llama_signals,
            'trading_decision': trading_decision,
            'timestamp': current_time or time.time()
        }
    
    def _make_trading_decision(self, gap_features: Dict[str, Any], 
                             gap_quality: float,
                             lag_llama_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision based on all signals"""
        
        gap_percent = gap_features.get('gap_percent', 0.0)
        volume_ratio = gap_features.get('volume_ratio', 1.0)
        gap_direction = gap_features.get('gap_direction', 0)
        
        # Basic gap filters
        gap_size_ok = self.min_gap_percent <= abs(gap_percent) <= self.max_gap_percent
        volume_ok = volume_ratio >= self.min_volume_ratio
        quality_ok = gap_quality >= self.ai_quality_threshold
        
        # AI signal confirmation
        ai_signal = lag_llama_signals.get('signal', 0.0)
        ai_confidence = lag_llama_signals.get('confidence', 0.0)
        
        # Signal alignment
        signal_alignment = np.sign(gap_direction) == np.sign(ai_signal) if ai_signal != 0 else False
        
        # Overall decision
        should_trade = gap_size_ok and volume_ok and quality_ok and signal_alignment
        
        # Position sizing
        if should_trade:
            base_size = 0.02  # 2% of portfolio
            quality_multiplier = gap_quality
            confidence_multiplier = ai_confidence
            position_size = base_size * quality_multiplier * confidence_multiplier
        else:
            position_size = 0.0
        
        return {
            'should_trade': should_trade,
            'direction': gap_direction,
            'position_size': position_size,
            'entry_reason': 'ai_gap_quality' if should_trade else 'no_signal',
            'filters': {
                'gap_size_ok': gap_size_ok,
                'volume_ok': volume_ok,
                'quality_ok': quality_ok,
                'signal_alignment': signal_alignment
            },
            'confidence': gap_quality * ai_confidence
        }
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        
        if self.performance['total_trades'] == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'avg_hold_time': 0.0
            }
        
        win_rate = self.performance['winning_trades'] / self.performance['total_trades']
        avg_pnl = self.performance['total_pnl'] / self.performance['total_trades']
        
        return {
            'total_trades': self.performance['total_trades'],
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': self.performance['total_pnl'],
            'avg_hold_time': self.performance['avg_hold_time']
        }
