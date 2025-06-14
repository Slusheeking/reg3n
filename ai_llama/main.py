"""
AI-Enhanced Trading System Main Engine

Integrates Lag-Llama foundation models with fast traditional models
for ultra-low latency trading decisions.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Import our modules
from config import *
from features.fast_features import FastFeatureEngine
from models.fast_models import FastModelEngine, GapQualityClassifier, BreakoutValidator
from models.lag_llama_engine import LagLlamaEngine
from utils.logger import setup_logging, get_logger, log_trade, log_performance, log_strategy


class AITradingEngine:
    """
    Main AI-Enhanced Trading Engine
    
    Combines:
    - Lag-Llama foundation model for time series forecasting
    - Fast traditional models for real-time signals
    - High-performance feature engineering
    - Multi-strategy execution
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        # Configuration
        self.config = {**globals()}  # Load all config variables
        if config_override:
            self.config.update(config_override)
        
        # Initialize engines
        self.feature_engine = FastFeatureEngine()
        self.fast_model_engine = FastModelEngine()
        self.lag_llama_engine = LagLlamaEngine()
        
        # Strategy-specific models
        self.gap_classifier = GapQualityClassifier()
        self.breakout_validator = BreakoutValidator()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'avg_latency_ms': 0.0,
            'cache_hit_rate': 0.0,
            'model_agreement': 0.0
        }
        
        # Setup comprehensive logging system
        self.logger_system = setup_logging(log_level="INFO", enable_console=True)
        self.logger = get_logger("system")
        
        self.logger.info("AI Trading Engine initialized with comprehensive logging")
    
    def extract_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract features for a symbol"""
        return self.feature_engine.extract_features(symbol, ohlcv_data, time.time())
    
    def generate_ai_signals(self, symbol: str, features: Dict[str, Any], 
                           price_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate AI-enhanced trading signals
        
        Args:
            symbol: Trading symbol
            features: Extracted features
            price_data: Historical price data
            
        Returns:
            Combined AI signals with confidence scores
        """
        start_time = time.perf_counter()
        
        try:
            # Get fast model predictions
            fast_predictions = self.fast_model_engine.predict(features)
            
            # Get Lag-Llama predictions
            lag_llama_prediction = self.lag_llama_engine.predict(
                price_data, symbol=symbol, timestamp=time.time()
            )
            lag_llama_signals = self.lag_llama_engine.get_trading_signal(lag_llama_prediction)
            
            # Combine signals
            combined_signals = self._combine_signals(fast_predictions, lag_llama_signals)
            
            # Add metadata
            combined_signals.update({
                'symbol': symbol,
                'timestamp': time.time(),
                'total_inference_time_ms': (time.perf_counter() - start_time) * 1000,
                'feature_count': features.get('feature_count', 0)
            })
            
            # Log performance metrics
            log_performance("inference_time_ms", combined_signals['total_inference_time_ms'], "ms", 
                          f"AI signal generation for {symbol}")
            log_performance("model_agreement", combined_signals['model_agreement'], "", 
                          f"Model agreement for {symbol}")
            
            self.performance_metrics['total_signals'] += 1
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals for {symbol}: {e}")
            return self._empty_signals(symbol)
    
    def _combine_signals(self, fast_predictions: Dict[str, Any], 
                        lag_llama_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine signals from different models"""
        
        # Extract signals
        fast_signal = fast_predictions.get('combined_signal', 0.0)
        fast_confidence = fast_predictions.get('confidence', 0.0)
        
        lag_llama_signal = lag_llama_signals.get('signal', 0.0)
        lag_llama_confidence = lag_llama_signals.get('confidence', 0.0)
        
        # Weight by confidence
        total_confidence = fast_confidence + lag_llama_confidence
        if total_confidence > 0:
            weighted_signal = (
                fast_signal * fast_confidence + 
                lag_llama_signal * lag_llama_confidence
            ) / total_confidence
        else:
            weighted_signal = 0.0
        
        # Calculate agreement between models
        agreement = 1.0 - abs(fast_signal - lag_llama_signal) / 2.0
        
        # Final confidence based on agreement and individual confidences
        final_confidence = (total_confidence / 2.0) * agreement
        
        return {
            'signal': float(np.clip(weighted_signal, -1.0, 1.0)),
            'confidence': float(final_confidence),
            'fast_signal': float(fast_signal),
            'fast_confidence': float(fast_confidence),
            'lag_llama_signal': float(lag_llama_signal),
            'lag_llama_confidence': float(lag_llama_confidence),
            'model_agreement': float(agreement),
            'signal_strength': float(abs(weighted_signal)),
            'fallback_mode': lag_llama_signals.get('fallback_mode', True)
        }
    
    def _empty_signals(self, symbol: str) -> Dict[str, Any]:
        """Return empty signals on error"""
        return {
            'symbol': symbol,
            'signal': 0.0,
            'confidence': 0.0,
            'fast_signal': 0.0,
            'fast_confidence': 0.0,
            'lag_llama_signal': 0.0,
            'lag_llama_confidence': 0.0,
            'model_agreement': 0.0,
            'signal_strength': 0.0,
            'error': True,
            'timestamp': time.time()
        }
    
    def process_gap_and_go(self, symbol: str, gap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Gap & Go strategy"""
        
        # Extract gap features
        gap_features = {
            'gap_percent': gap_data.get('gap_percent', 0.0),
            'volume_ratio': gap_data.get('volume_ratio', 1.0),
            'premarket_volume_ratio': gap_data.get('premarket_volume_ratio', 1.0)
        }
        
        # Classify gap quality
        gap_quality = self.gap_classifier.classify_gap_quality(gap_features)
        
        # Get AI signals
        ai_signals = self.generate_ai_signals(
            symbol, 
            gap_data.get('features', {}), 
            gap_data.get('price_data', np.array([]))
        )
        
        # Combine gap quality with AI signals
        gap_signal_strength = gap_quality * ai_signals.get('signal_strength', 0.0)
        
        return {
            'strategy': 'gap_and_go',
            'symbol': symbol,
            'gap_quality': gap_quality,
            'gap_signal_strength': gap_signal_strength,
            'ai_signal': ai_signals.get('signal', 0.0),
            'combined_confidence': gap_quality * ai_signals.get('confidence', 0.0),
            'should_trade': gap_quality > STRATEGIES['gap_and_go']['ai_quality_threshold'],
            'timestamp': time.time()
        }
    
    def process_orb(self, symbol: str, orb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Opening Range Breakout strategy"""
        
        # Extract breakout features
        breakout_features = {
            'volume_ratio': orb_data.get('volume_ratio', 1.0),
            'momentum_5': orb_data.get('momentum_5', 0.0),
            'range_expansion': orb_data.get('range_expansion', 1.0)
        }
        
        # Validate breakout
        breakout_quality = self.breakout_validator.validate_breakout(breakout_features)
        
        # Get AI signals
        ai_signals = self.generate_ai_signals(
            symbol, 
            orb_data.get('features', {}), 
            orb_data.get('price_data', np.array([]))
        )
        
        # Combine breakout validation with AI signals
        orb_signal_strength = breakout_quality * ai_signals.get('signal_strength', 0.0)
        
        return {
            'strategy': 'orb',
            'symbol': symbol,
            'breakout_quality': breakout_quality,
            'orb_signal_strength': orb_signal_strength,
            'ai_signal': ai_signals.get('signal', 0.0),
            'combined_confidence': breakout_quality * ai_signals.get('confidence', 0.0),
            'should_trade': breakout_quality > STRATEGIES['orb']['breakout_threshold'],
            'timestamp': time.time()
        }
    
    def process_vol_mean_reversion(self, symbol: str, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Volatility Mean Reversion strategy"""
        
        # Get AI signals
        ai_signals = self.generate_ai_signals(
            symbol, 
            vol_data.get('features', {}), 
            vol_data.get('price_data', np.array([]))
        )
        
        # Check volatility conditions
        volatility_ratio = vol_data.get('volatility_ratio', 1.0)
        reversion_probability = ai_signals.get('lag_llama_confidence', 0.0)
        
        # Mean reversion signal should be opposite to current trend
        mean_reversion_signal = -ai_signals.get('signal', 0.0)
        
        return {
            'strategy': 'vol_mean_reversion',
            'symbol': symbol,
            'volatility_ratio': volatility_ratio,
            'reversion_probability': reversion_probability,
            'mean_reversion_signal': mean_reversion_signal,
            'ai_signal': ai_signals.get('signal', 0.0),
            'combined_confidence': reversion_probability,
            'should_trade': (
                volatility_ratio > STRATEGIES['vol_mean_reversion']['volatility_threshold'] and
                reversion_probability > STRATEGIES['vol_mean_reversion']['reversion_probability_threshold']
            ),
            'timestamp': time.time()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        # Feature engine stats
        feature_stats = self.feature_engine.get_feature_vector({})
        
        # Fast model stats
        fast_model_importance = self.fast_model_engine.get_feature_importance()
        
        # Lag-Llama stats
        lag_llama_stats = self.lag_llama_engine.get_performance_stats()
        
        return {
            'engine_performance': self.performance_metrics,
            'feature_engine': {
                'cache_size': len(getattr(self.feature_engine, 'feature_cache', {}))
            },
            'fast_models': {
                'is_fitted': self.fast_model_engine.is_fitted,
                'feature_importance': fast_model_importance
            },
            'lag_llama': lag_llama_stats,
            'system_info': {
                'total_symbols': len(DATA['symbols']),
                'target_latency_ms': PERFORMANCE['target_latency_ms'],
                'system_ready': self.is_system_ready()
            }
        }
    
    def is_system_ready(self) -> bool:
        """Check if system is ready for trading"""
        return (
            self.feature_engine is not None and
            self.fast_model_engine is not None and
            self.lag_llama_engine is not None
        )
    
    def warmup_system(self):
        """Warmup all components for optimal performance"""
        self.logger.info("Warming up AI Trading Engine...")
        
        # Create dummy data
        dummy_ohlcv = {
            'open': np.random.randn(100) * 0.01 + 100.0,
            'high': np.random.randn(100) * 0.01 + 101.0,
            'low': np.random.randn(100) * 0.01 + 99.0,
            'close': np.random.randn(100) * 0.01 + 100.0,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        # Warmup feature extraction
        for i in range(3):
            features = self.extract_features('WARMUP', dummy_ohlcv)
            self.fast_model_engine.predict(features)
        
        # Warmup Lag-Llama
        self.lag_llama_engine.warmup(dummy_ohlcv['close'])
        
        self.logger.info("System warmup complete")


async def main():
    """Main async entry point"""
    
    # Initialize trading engine
    engine = AITradingEngine()
    
    # Warmup system
    engine.warmup_system()
    
    # Example usage with dummy data
    print("🚀 AI-Enhanced Trading System Ready!")
    print("=" * 50)
    
    # Generate some example predictions
    dummy_ohlcv = {
        'open': np.random.randn(100) * 0.01 + 100.0,
        'high': np.random.randn(100) * 0.01 + 101.0,
        'low': np.random.randn(100) * 0.01 + 99.0,
        'close': np.random.randn(100) * 0.01 + 100.0,
        'volume': np.random.randint(1000, 10000, 100)
    }
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        print(f"\n📊 Processing {symbol}...")
        
        # Extract features
        features = engine.extract_features(symbol, dummy_ohlcv)
        print(f"   ✓ Extracted {features.get('feature_count', 0)} features in {features.get('extraction_time_ms', 0):.2f}ms")
        
        # Generate AI signals
        signals = engine.generate_ai_signals(symbol, features, dummy_ohlcv['close'])
        print(f"   🤖 AI Signal: {signals.get('signal', 0):.3f} (confidence: {signals.get('confidence', 0):.3f})")
        print(f"   ⚡ Total inference: {signals.get('total_inference_time_ms', 0):.2f}ms")
        
        # Example Gap & Go processing
        gap_data = {
            'gap_percent': np.random.uniform(-0.05, 0.05),
            'volume_ratio': np.random.uniform(0.5, 3.0),
            'premarket_volume_ratio': np.random.uniform(0.5, 2.0),
            'features': features,
            'price_data': dummy_ohlcv['close']
        }
        
        gap_result = engine.process_gap_and_go(symbol, gap_data)
        print(f"   📈 Gap Quality: {gap_result.get('gap_quality', 0):.3f} (should trade: {gap_result.get('should_trade', False)})")
        
        # Log strategy decision if should trade
        if gap_result.get('should_trade', False):
            log_strategy("GapAndGo", f"Trade signal generated for {symbol}", level="INFO",
                        gap_quality=gap_result.get('gap_quality', 0),
                        signal_strength=gap_result.get('gap_signal_strength', 0),
                        confidence=gap_result.get('combined_confidence', 0))
    
    # Show performance stats
    print(f"\n📈 Performance Statistics:")
    stats = engine.get_performance_stats()
    
    if not stats['lag_llama'].get('no_data', False):
        print(f"   Average Inference Time: {stats['lag_llama'].get('avg_inference_time_ms', 0):.2f}ms")
        print(f"   P95 Inference Time: {stats['lag_llama'].get('p95_inference_time_ms', 0):.2f}ms")
        print(f"   Cache Hit Rate: {stats['lag_llama'].get('cache_hit_rate', 0)*100:.1f}%")
    
    print(f"   System Ready: {stats['system_info']['system_ready']}")
    print(f"   Fallback Mode: {stats['lag_llama'].get('fallback_mode', True)}")
    
    print("\n✅ Demo complete! System is ready for real trading data.")


if __name__ == "__main__":
    asyncio.run(main())
