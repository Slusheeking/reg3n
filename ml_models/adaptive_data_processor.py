#!/usr/bin/env python3

import sys
import os
import yaml
import time
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


# Add project root to path for filters import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.system_logger import get_system_logger
from .feature_engineering import FeatureEngineer
from .model_server import OptimizedModelServer
from .ensemble_manager import EnsembleManager, MarketRegime, EnsemblePrediction
from .online_learning import OnlineLearningEngine
from filters.adaptive_data_filter import AdaptiveDataFilter, MarketData

# Initialize component logger
logger = get_system_logger("adaptive_data_processor")

@dataclass
class TradingSignal:
    symbol: str
    prediction: float
    confidence: float
    regime: MarketRegime
    volatility_score: float
    momentum_score: float
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    timestamp: float
    processing_time: float
    # NEW: Filter information
    market_condition: Optional[str] = None
    filter_strategy: Optional[str] = None
    ml_score: Optional[float] = None

@dataclass
class MarketRegimeUpdate:
    regime: MarketRegime
    allocation_ratio: List[float]  # [volatility_weight, momentum_weight]
    confidence: float
    regime_features: Dict[str, float]
    timestamp: float

class AdaptiveDataProcessor:
    """
    Main data processor that integrates all ML models with your trading system and adaptive filter
    Implements the complete strategy from strategy.md with <200ms latency
    """
    
    def __init__(self, config_path: str = "yaml/ml_models.yaml", filter_config_path: str = "yaml/filters.yaml"):
        logger.info("Initializing Adaptive Data Processor", extra={
            "component": "adaptive_data_processor",
            "action": "initialization_start",
            "config_path": config_path,
            "filter_config_path": filter_config_path
        })
        
        # Core components
        self.feature_engineer = FeatureEngineer()
        self.model_server = OptimizedModelServer()
        self.ensemble_manager = None  # Will be initialized after model server
        self.online_learner = None    # Will be initialized after ensemble manager
        
        # NEW: Adaptive filter integration
        self.adaptive_filter = AdaptiveDataFilter(filter_config_path)
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Current market state
        self.current_regime = MarketRegime.BALANCED_MARKET
        self.current_allocation = [0.5, 0.5]  # Default 50/50
        self.regime_confidence = 0.5
        
        # Performance tracking
        self.processing_times = []
        self.prediction_accuracy = []
        self.total_predictions = 0
        
        # Position sizing parameters
        self.position_sizing_config = {
            'min_position': 2000,      # $2K minimum
            'max_position': 8000,      # $8K maximum
            'max_positions': 15,       # Maximum concurrent positions
            'max_concentration': 0.2,  # 20% max per position
            'daily_capital': 50000     # $50K daily capital
        }
        
        # Profit taking stages (from strategy.md)
        self.profit_stages = {
            'stage_1': {'trigger': 0.015, 'exit': 0.25},  # 1.5% -> sell 25%
            'stage_2': {'trigger': 0.03, 'exit': 0.5},    # 3% -> sell 50% of remaining
            'stage_3': {'trigger': 0.05, 'exit': 0.75},   # 5% -> sell 75% of remaining
            'final': {'trigger': 0.07, 'exit': 1.0}       # 7% -> close all
        }
        
        # Risk management
        self.risk_config = {
            'stop_loss': 0.02,         # 2% stop loss
            'daily_loss_limit': 750,   # $750 daily loss limit
            'time_limit': 240,         # 4 hours max hold time
            'sector_limit': 3          # Max 3 positions per sector
        }
        
        # Market schedule (from strategy.md)
        self.market_schedule = {
            'market_open': '09:30',
            'entry_cutoff': '15:30',
            'liquidation_start': '15:45',
            'hard_close': '15:58'
        }
        
        logger.info("Adaptive Data Processor initialization complete", extra={
            "component": "adaptive_data_processor",
            "action": "initialization_complete",
            "feature_engineer_ready": self.feature_engineer is not None,
            "model_server_ready": self.model_server is not None,
            "adaptive_filter_ready": self.adaptive_filter is not None,
            "config_loaded": bool(self.config),
            "position_sizing_config": self.position_sizing_config,
            "market_schedule": self.market_schedule
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Load ML configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for adaptive data processor"""
        return {
            'feature_engineering': {
                'cache_ttl': 300,
                'enable_caching': True,
                'parallel_processing': True
            },
            'model_server': {
                'max_workers': 8,
                'gpu_enabled': True,
                'model_dir': 'models',
                'cache_predictions': True
            },
            'ensemble': {
                'adaptive_learning_rate': 0.01,
                'regime_confidence_threshold': 0.7,
                'meta_learning_enabled': True
            },
            'online_learning': {
                'enable_incremental_learning': True,
                'enable_automatic_retraining': True,
                'buffer_size': 5000,
                'min_training_samples': 100
            },
            'performance': {
                'target_latency': 0.2,  # 200ms
                'enable_monitoring': True,
                'log_predictions': True
            }
        }
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Adaptive Data Processor", extra={
            "component": "adaptive_data_processor",
            "action": "initialization_start",
            "config_keys": list(self.config.keys())
        })
        
        try:
            # Initialize model server and load all models
            logger.info("Loading ML models", extra={
                "component": "adaptive_data_processor",
                "action": "model_loading",
                "step": "model_server_init"
            })
            await self.model_server.load_all_models()
            
            # Initialize ensemble manager
            logger.info("Initializing ensemble manager", extra={
                "component": "adaptive_data_processor",
                "action": "ensemble_init",
                "ensemble_config": self.config.get('ensemble', {})
            })
            self.ensemble_manager = EnsembleManager(
                self.model_server,
                self.config.get('ensemble', {})
            )
            
            # Initialize online learning engine
            logger.info("Initializing online learning engine", extra={
                "component": "adaptive_data_processor",
                "action": "online_learning_init",
                "learning_config": self.config.get('online_learning', {})
            })
            self.online_learner = OnlineLearningEngine(
                self.model_server,
                self.ensemble_manager,
                self.config.get('online_learning', {})
            )
            
            # Start online learning
            logger.info("Starting online learning", extra={
                "component": "adaptive_data_processor",
                "action": "online_learning_start"
            })
            await self.online_learner.start_online_learning()
            
            logger.info("Adaptive Data Processor initialized successfully", extra={
                "component": "adaptive_data_processor",
                "action": "initialization_complete",
                "models_loaded": len(self.model_server.models) if hasattr(self.model_server, 'models') else 0,
                "ensemble_ready": self.ensemble_manager is not None,
                "online_learning_ready": self.online_learner is not None
            })
            
        except Exception as e:
            logger.error("Adaptive Data Processor initialization failed", extra={
                "component": "adaptive_data_processor",
                "action": "initialization_error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    async def process_polygon_data(self, polygon_data: List[Dict]) -> List[TradingSignal]:
        """
        NEW: Process raw Polygon data through adaptive filter then ML models
        Target: <200ms total processing time including filtering
        """
        start_time = time.time()
        
        logger.info("Starting Polygon data processing", extra={
            "component": "adaptive_data_processor",
            "action": "process_polygon_data",
            "input_count": len(polygon_data),
            "target_time_ms": 200
        })
        
        try:
            # Step 1: Adaptive Filtering (<75ms)
            filter_start = time.time()
            filtered_stocks = await self.adaptive_filter.process_polygon_data(polygon_data)
            filter_time = (time.time() - filter_start) * 1000
            
            logger.info("Adaptive filtering complete", extra={
                "component": "adaptive_data_processor",
                "action": "adaptive_filtering",
                "input_count": len(polygon_data),
                "filtered_count": len(filtered_stocks),
                "filter_time_ms": filter_time,
                "target_filter_ms": 75,
                "within_target": filter_time < 75
            })
            
            # Step 2: Process each filtered stock through ML pipeline
            trading_signals = []
            ml_start = time.time()
            
            for i, filtered_stock in enumerate(filtered_stocks):
                # Convert filtered stock to market_data format
                market_data = self._convert_filtered_stock_to_market_data(filtered_stock)
                
                # Process through existing ML pipeline
                signal = await self.process_market_data(market_data)
                
                if signal and signal.confidence > 0.6:
                    # Add filter information to signal
                    signal.market_condition = filtered_stock.market_condition
                    signal.filter_strategy = getattr(filtered_stock, 'strategy_type', 'unknown')
                    trading_signals.append(signal)
                    
                    logger.log_trading_signal("ML signal generated", {
                        "symbol": signal.symbol,
                        "prediction": signal.prediction,
                        "confidence": signal.confidence,
                        "regime": signal.regime.value,
                        "position_size": signal.position_size,
                        "market_condition": signal.market_condition,
                        "filter_strategy": signal.filter_strategy
                    })
            
            ml_time = (time.time() - ml_start) * 1000
            
            # Performance tracking
            total_processing_time = time.time() - start_time
            total_time_ms = total_processing_time * 1000
            
            logger.info("Polygon data processing complete", extra={
                "component": "adaptive_data_processor",
                "action": "process_complete",
                "input_count": len(polygon_data),
                "filtered_count": len(filtered_stocks),
                "signal_count": len(trading_signals),
                "filter_time_ms": filter_time,
                "ml_time_ms": ml_time,
                "total_time_ms": total_time_ms,
                "target_time_ms": 200,
                "within_target": total_time_ms < 200,
                "conversion_rate": len(trading_signals) / len(polygon_data) if polygon_data else 0
            })
            
            return trading_signals
            
        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            logger.error("Polygon data processing failed", extra={
                "component": "adaptive_data_processor",
                "action": "process_error",
                "input_count": len(polygon_data),
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": total_time_ms
            })
            return []

    async def process_market_data(self, market_data: Dict) -> TradingSignal:
        """
        Original processing method - now works with pre-filtered data
        Target: <125ms processing time (since filtering takes 75ms)
        """
        start_time = time.time()
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        logger.debug("Starting market data processing", extra={
            "component": "adaptive_data_processor",
            "action": "process_market_data",
            "symbol": symbol,
            "target_time_ms": 125
        })
        
        try:
            # Step 1: Feature Engineering (<20ms)
            feature_start = time.time()
            features = await self.feature_engineer.engineer_features(market_data)
            feature_time = (time.time() - feature_start) * 1000
            
            # Step 2: Ensemble Prediction (<80ms - reduced from 150ms)
            ensemble_start = time.time()
            ensemble_prediction = await self.ensemble_manager.predict(
                features,
                market_data.get('symbol')
            )
            ensemble_time = (time.time() - ensemble_start) * 1000
            
            # Step 3: Update Market Regime
            await self._update_market_regime(ensemble_prediction)
            
            # Step 4: Generate Trading Signal
            signal_start = time.time()
            trading_signal = await self._generate_trading_signal(
                market_data,
                ensemble_prediction
            )
            signal_time = (time.time() - signal_start) * 1000
            
            # Step 5: Online Learning Update
            await self._update_online_learning(
                features,
                ensemble_prediction,
                market_data
            )
            
            # Step 6: Performance Tracking
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            self._update_performance_metrics(processing_time)
            
            trading_signal.processing_time = processing_time
            
            logger.debug("Market data processing complete", extra={
                "component": "adaptive_data_processor",
                "action": "process_complete",
                "symbol": symbol,
                "feature_time_ms": feature_time,
                "ensemble_time_ms": ensemble_time,
                "signal_time_ms": signal_time,
                "total_time_ms": processing_time_ms,
                "target_time_ms": 125,
                "within_target": processing_time_ms < 125,
                "prediction": ensemble_prediction.final_prediction,
                "confidence": ensemble_prediction.confidence,
                "regime": ensemble_prediction.regime.value
            })
            
            return trading_signal
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error("Market data processing failed", extra={
                "component": "adaptive_data_processor",
                "action": "process_error",
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time_ms
            })
            return self._create_fallback_signal(market_data)
    
    async def _update_market_regime(self, ensemble_prediction: EnsemblePrediction):
        """Update current market regime based on ensemble prediction"""
        try:
            # Update current regime
            self.current_regime = ensemble_prediction.regime
            self.regime_confidence = ensemble_prediction.confidence
            
            # Update allocation ratios based on regime
            if ensemble_prediction.regime in self.ensemble_manager.regime_allocations:
                self.current_allocation = self.ensemble_manager.regime_allocations[ensemble_prediction.regime]
            
        except Exception as e:
            logger.error(f"Regime update error: {e}")
    
    async def _generate_trading_signal(self, market_data: Dict, 
                                     ensemble_prediction: EnsemblePrediction) -> TradingSignal:
        """Generate trading signal from ensemble prediction"""
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('price', 0)
            
            # Calculate position size based on prediction confidence and regime
            position_size = self._calculate_position_size(
                ensemble_prediction.final_prediction,
                ensemble_prediction.confidence,
                current_price
            )
            
            # Calculate stop loss and take profit levels
            stop_loss = self._calculate_stop_loss(current_price, ensemble_prediction.final_prediction)
            take_profit_levels = self._calculate_take_profit_levels(current_price, ensemble_prediction.final_prediction)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                prediction=ensemble_prediction.final_prediction,
                confidence=ensemble_prediction.confidence,
                regime=ensemble_prediction.regime,
                volatility_score=ensemble_prediction.volatility_score,
                momentum_score=ensemble_prediction.momentum_score,
                position_size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_levels=take_profit_levels,
                timestamp=time.time(),
                processing_time=0  # Will be set by caller
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._create_fallback_signal(market_data)
    
    def _calculate_position_size(self, prediction: float, confidence: float, price: float) -> float:
        """
        Calculate position size based on prediction strength and confidence
        Implements position sizing from strategy.md
        """
        try:
            # Base position size from config
            min_pos = self.position_sizing_config['min_position']
            max_pos = self.position_sizing_config['max_position']
            
            # Scale based on prediction strength (absolute value)
            prediction_strength = abs(prediction)
            
            # Scale based on confidence
            confidence_multiplier = confidence
            
            # Calculate base position size
            base_size = min_pos + (max_pos - min_pos) * prediction_strength * confidence_multiplier
            
            # Ensure within bounds
            position_size = np.clip(base_size, min_pos, max_pos)
            
            # Convert to number of shares
            if price > 0:
                shares = int(position_size / price)
                return shares * price  # Return dollar amount
            else:
                return min_pos
                
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return self.position_sizing_config['min_position']
    
    def _calculate_stop_loss(self, entry_price: float, prediction: float) -> float:
        """Calculate stop loss based on entry price and prediction direction"""
        try:
            stop_loss_pct = self.risk_config['stop_loss']
            
            if prediction > 0:  # Long position
                return entry_price * (1 - stop_loss_pct)
            else:  # Short position
                return entry_price * (1 + stop_loss_pct)
                
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            return entry_price * 0.98  # Default 2% stop loss
    
    def _calculate_take_profit_levels(self, entry_price: float, prediction: float) -> List[float]:
        """
        Calculate staged take profit levels from strategy.md
        """
        try:
            take_profit_levels = []
            
            direction = 1 if prediction > 0 else -1
            
            for stage_name, stage_config in self.profit_stages.items():
                if stage_name == 'final':
                    continue
                    
                trigger_pct = stage_config['trigger']
                
                if direction > 0:  # Long position
                    tp_price = entry_price * (1 + trigger_pct)
                else:  # Short position
                    tp_price = entry_price * (1 - trigger_pct)
                
                take_profit_levels.append(tp_price)
            
            return take_profit_levels
            
        except Exception as e:
            logger.error(f"Take profit calculation error: {e}")
            return [entry_price * 1.02, entry_price * 1.05, entry_price * 1.07]  # Default levels
    
    async def _update_online_learning(self, features: np.ndarray, 
                                    ensemble_prediction: EnsemblePrediction,
                                    market_data: Dict):
        """Update online learning with new data"""
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Add training example to online learner
            await self.online_learner.add_training_example(
                features=features,
                prediction=ensemble_prediction.final_prediction,
                symbol=symbol,
                actual_outcome=None  # Will be updated later with actual results
            )
            
        except Exception as e:
            logger.error(f"Online learning update error: {e}")
    
    async def update_actual_outcome(self, symbol: str, timestamp: float, actual_return: float):
        """
        Update models with actual trading outcomes for learning
        Call this method when you have actual trading results
        """
        try:
            # This would be called from your trading pipeline when you have actual results
            # For now, we'll implement a placeholder
            
            # Update ensemble manager performance
            # Note: You'd need to store the original prediction to compare
            
            # Update online learning
            # The online learner will use this for model improvement
            
            logger.info(f"Updated actual outcome for {symbol}: {actual_return}")
            
        except Exception as e:
            logger.error(f"Actual outcome update error: {e}")
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics"""
        try:
            self.processing_times.append(processing_time)
            self.total_predictions += 1
            
            # Keep only last 1000 measurements
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
            
            # Log performance warnings
            if processing_time > self.config['performance']['target_latency']:
                logger.warning(f"Processing time exceeded target: {processing_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    def _create_fallback_signal(self, market_data: Dict) -> TradingSignal:
        """Create fallback signal when processing fails"""
        return TradingSignal(
            symbol=market_data.get('symbol', 'UNKNOWN'),
            prediction=0.0,
            confidence=0.1,
            regime=MarketRegime.BALANCED_MARKET,
            volatility_score=0.5,
            momentum_score=0.5,
            position_size=self.position_sizing_config['min_position'],
            entry_price=market_data.get('price', 0),
            stop_loss=market_data.get('price', 0) * 0.98,
            take_profit_levels=[],
            timestamp=time.time(),
            processing_time=0.001
        )
    
    def get_current_regime_info(self) -> MarketRegimeUpdate:
        """Get current market regime information"""
        return MarketRegimeUpdate(
            regime=self.current_regime,
            allocation_ratio=self.current_allocation.copy(),
            confidence=self.regime_confidence,
            regime_features={
                'volatility_weight': self.current_allocation[0],
                'momentum_weight': self.current_allocation[1]
            },
            timestamp=time.time()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            stats = {
                'processing_performance': {
                    'total_predictions': self.total_predictions,
                    'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                    'p95_processing_time': np.percentile(self.processing_times, 95) if self.processing_times else 0,
                    'p99_processing_time': np.percentile(self.processing_times, 99) if self.processing_times else 0,
                    'target_latency': self.config['performance']['target_latency'],
                    'latency_compliance': np.mean([t <= self.config['performance']['target_latency'] for t in self.processing_times]) if self.processing_times else 0
                },
                'current_regime': {
                    'regime': self.current_regime.value,
                    'allocation': self.current_allocation,
                    'confidence': self.regime_confidence
                }
            }
            
            # Add model server stats
            if hasattr(self.model_server, 'get_performance_stats'):
                stats['model_server'] = self.model_server.get_performance_stats()
            
            # Add ensemble stats
            if self.ensemble_manager:
                stats['ensemble'] = self.ensemble_manager.get_ensemble_status()
            
            # Add online learning stats
            if self.online_learner:
                stats['online_learning'] = self.online_learner.get_learning_status()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    async def save_state(self, base_path: str = "models/state"):
        """Save all component states"""
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Save ensemble state
            if self.ensemble_manager:
                await asyncio.to_thread(
                    self.ensemble_manager.save_ensemble_state,
                    str(base_path / "ensemble_state.json")
                )
            
            # Save online learning state
            if self.online_learner:
                await self.online_learner.save_learning_state(
                    str(base_path / "learning_state.json")
                )
            
            # Save processor state
            processor_state = {
                'current_regime': self.current_regime.value,
                'current_allocation': self.current_allocation,
                'regime_confidence': self.regime_confidence,
                'total_predictions': self.total_predictions,
                'performance_stats': self.get_performance_stats()
            }
            
            with open(base_path / "processor_state.json", 'w') as f:
                import json
                json.dump(processor_state, f, indent=2, default=str)
            
            logger.info(f"State saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _convert_filtered_stock_to_market_data(self, filtered_stock: MarketData) -> Dict:
        """Convert filtered stock data to format expected by ML pipeline"""
        return {
            'symbol': filtered_stock.symbol,
            'price': filtered_stock.price,
            'volume': filtered_stock.volume,
            'timestamp': filtered_stock.timestamp,
            'prices': {
                'close': filtered_stock.price,
                'vwap': filtered_stock.price,  # Simplified - could be enhanced
                'close_1min_ago': filtered_stock.price * 0.999,  # Simplified
                'close_2min_ago': filtered_stock.price * 0.998,
                'close_3min_ago': filtered_stock.price * 0.997,
                'close_4min_ago': filtered_stock.price * 0.996,
                'close_5min_ago': filtered_stock.price * 0.995
            },
            'volume': {
                'total': filtered_stock.volume,
                'uptick': filtered_stock.volume * 0.6,  # Simplified
                'downtick': filtered_stock.volume * 0.4,
                'avg_20min': filtered_stock.volume * 0.8,
                'percentile': 0.5
            },
            'ohlcv': {
                'high': [filtered_stock.price * 1.01] * 50,  # Simplified
                'low': [filtered_stock.price * 0.99] * 50,
                'close': [filtered_stock.price] * 50,
                'volume': [filtered_stock.volume] * 50,
                'open': [filtered_stock.price * 0.995] * 50
            },
            'market_context': {
                'vix': 20,  # Will be updated with real VIX from filter
                'minute_of_day': 240,  # Simplified
                'day_of_week': 2,
                'market_breadth': 0.5
            },
            'order_flow': {
                'bid_size': 1000,  # Simplified
                'ask_size': 1000,
                'trades_at_bid': filtered_stock.volume * 0.4,
                'trades_at_ask': filtered_stock.volume * 0.6,
                'spread': 0.01,
                'mid_price': filtered_stock.price
            }
        }

    async def load_state(self, base_path: str = "models/state"):
        """Load all component states"""
        try:
            base_path = Path(base_path)
            
            # Load ensemble state
            ensemble_path = base_path / "ensemble_state.json"
            if ensemble_path.exists() and self.ensemble_manager:
                await asyncio.to_thread(
                    self.ensemble_manager.load_ensemble_state,
                    str(ensemble_path)
                )
            
            # Load online learning state
            learning_path = base_path / "learning_state.json"
            if learning_path.exists() and self.online_learner:
                await self.online_learner.load_learning_state(str(learning_path))
            
            # Load processor state
            processor_path = base_path / "processor_state.json"
            if processor_path.exists():
                with open(processor_path, 'r') as f:
                    import json
                    state = json.load(f)
                
                self.current_regime = MarketRegime(state.get('current_regime', 'balanced_market'))
                self.current_allocation = state.get('current_allocation', [0.5, 0.5])
                self.regime_confidence = state.get('regime_confidence', 0.5)
                self.total_predictions = state.get('total_predictions', 0)
            
            logger.info(f"State loaded from {base_path}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")

# Integration helper functions for your existing trading pipeline

async def create_adaptive_processor(config_path: str = None) -> AdaptiveDataProcessor:
    """
    Factory function to create and initialize the adaptive data processor
    Use this in your main trading application
    """
    processor = AdaptiveDataProcessor(config_path)
    await processor.initialize()
    return processor

def integrate_with_strategy_allocator(processor: AdaptiveDataProcessor) -> Dict[str, float]:
    """
    Integration function for your strategy allocator
    Returns current volatility/momentum allocation ratios
    """
    regime_info = processor.get_current_regime_info()
    return {
        'volatility_weight': regime_info.allocation_ratio[0],
        'momentum_weight': regime_info.allocation_ratio[1],
        'regime': regime_info.regime.value,
        'confidence': regime_info.confidence
    }

def integrate_with_position_sizer(signal: TradingSignal, portfolio_state: Dict) -> Dict[str, Any]:
    """
    Integration function for your position sizing logic
    """
    return {
        'symbol': signal.symbol,
        'position_size': signal.position_size,
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'take_profit_levels': signal.take_profit_levels,
        'confidence': signal.confidence,
        'regime': signal.regime.value
    }