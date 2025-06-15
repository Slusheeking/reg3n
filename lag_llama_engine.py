"""
Lag-Llama Forecasting Engine for Trading System
Main AI/ML component providing probabilistic price forecasts
"""

import torch
import numpy as np
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import orjson as json
import time

# Import Lag-Llama components (adjust imports based on actual installation)
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.common import ListDataset
except ImportError as e:
    logging.error(f"Lag-Llama imports failed: {e}")
    logging.error("Please install lag-llama: pip install lag-llama")

from settings import config
from active_symbols import symbol_manager

logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Container for forecast results"""
    symbol: str
    timestamp: datetime
    horizon_minutes: int
    
    # Price forecasts
    mean_forecast: np.ndarray
    std_forecast: np.ndarray
    quantiles: Dict[str, np.ndarray]
    samples: np.ndarray
    
    # Derived metrics
    direction_probability: float  # P(price goes up)
    magnitude_forecast: float     # Expected absolute change
    volatility_forecast: float    # Predicted volatility
    confidence_score: float       # Model confidence (0-1)
    
    # Trading signals
    long_probability: float       # P(profitable long)
    short_probability: float      # P(profitable short)
    optimal_hold_minutes: int     # Suggested hold time
    
    # Risk metrics
    var_95: float                # Value at Risk (95%)
    expected_shortfall: float    # Expected loss beyond VaR
    max_expected_gain: float     # Maximum expected gain
    max_expected_loss: float     # Maximum expected loss

@dataclass
class MarketForecast:
    """Multi-symbol market forecast"""
    timestamp: datetime
    forecasts: Dict[str, ForecastResult]
    market_regime: str
    overall_confidence: float
    correlation_matrix: Optional[np.ndarray] = None

class LagLlamaEngine:
    """Main Lag-Llama forecasting engine optimized for trading"""
    
    def __init__(self):
        self.config = config.lag_llama
        self.device = torch.device(self.config.device)
        
        # Model components
        self.estimator = None
        self.predictor = None
        self.model_loaded = False
        
        # Data management
        self.price_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.context_length)
        )
        self.volume_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.context_length)
        )
        
        # Forecast cache
        self.forecast_cache: Dict[str, ForecastResult] = {}
        self.cache_expiry_minutes = 5
        
        # Performance tracking
        self.forecast_accuracy: Dict[str, List[float]] = defaultdict(list)
        self.prediction_times: List[float] = []
        
        # Market regime detection
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize model (will be done lazily when needed)
        self._initialization_task = None
    
    async def _ensure_initialized(self):
        """Ensure the model is initialized (lazy initialization)"""
        if self._initialization_task is None:
            self._initialization_task = asyncio.create_task(self._initialize_model())
        await self._initialization_task
    
    async def _initialize_model(self):
        """Initialize Lag-Llama model with GH200 optimizations"""
        
        try:
            logger.info("Initializing Lag-Llama model...")
            
            # Configure for GH200 hardware
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
            
            # Initialize estimator with optimized config
            self.estimator = LagLlamaEstimator(
                ckpt_path=self.config.model_path,
                prediction_length=self.config.prediction_length,
                context_length=self.config.context_length,
                device=self.device,
                batch_size=self.config.batch_size,
                num_parallel_samples=self.config.num_parallel_samples,
                nonnegative_pred_samples=True,
                
                # GH200 optimizations
                rope_scaling={
                    "type": "linear",
                    "factor": self.config.rope_scaling_factor
                }
            )
            
            # Create predictor
            transformation = self.estimator.create_transformation()
            lightning_module = self.estimator.create_lightning_module()
            self.predictor = self.estimator.create_predictor(
                transformation, lightning_module
            )
            
            self.model_loaded = True
            logger.info("Lag-Llama model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Lag-Llama: {e}")
            self.model_loaded = False
    
    def add_price_data(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Add new price/volume data for a symbol"""
        
        self.price_buffers[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        self.volume_buffers[symbol].append({
            'volume': volume,
            'timestamp': timestamp
        })
        
        # Invalidate cache for this symbol
        if symbol in self.forecast_cache:
            del self.forecast_cache[symbol]
    
    def _prepare_dataset(self, symbols: List[str]) -> Optional[Any]: # Changed ListDataset to Any
        """Prepare dataset for Lag-Llama inference"""
        
        dataset_entries = []
        
        for symbol in symbols:
            if len(self.price_buffers[symbol]) < 64:  # Minimum data requirement
                continue
            
            # Extract price series
            prices = [entry['price'] for entry in self.price_buffers[symbol]]
            start_time = self.price_buffers[symbol][0]['timestamp']
            
            # Create dataset entry
            entry = {
                'target': prices,
                'start': start_time,
                'item_id': symbol
            }
            
            dataset_entries.append(entry)
        
        if not dataset_entries:
            return None
        
        return ListDataset(dataset_entries, freq='T')  # Minute frequency
    
    async def generate_forecasts(self, symbols: List[str]) -> Dict[str, ForecastResult]:
        """Generate forecasts for multiple symbols"""
        
        # Ensure model is initialized
        await self._ensure_initialized()
        
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot generate forecasts")
            return {}
        
        start_time = time.time()
        
        try:
            # Prepare dataset
            dataset = self._prepare_dataset(symbols)
            if dataset is None:
                return {}
            
            # Run batch inference with mixed precision
            if not self.predictor:
                logger.error("Predictor not initialized, cannot generate forecasts.")
                return {}
            with torch.cuda.amp.autocast():
                forecasts = list(self.predictor.predict(dataset))
            
            # Process results
            results = {}
            for i, symbol in enumerate(symbols):
                if i >= len(forecasts):
                    continue
                
                forecast = forecasts[i]
                result = self._process_forecast_result(symbol, forecast)
                results[symbol] = result
                
                # Cache result
                self.forecast_cache[symbol] = result
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.info(f"Generated forecasts for {len(results)} symbols in {prediction_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return {}
    
    def _process_forecast_result(self, symbol: str, forecast) -> ForecastResult:
        """Process raw forecast into structured result"""
        
        samples = forecast.samples  # Shape: [num_samples, prediction_length]
        current_price = self.price_buffers[symbol][-1]['price']
        
        # Basic statistics
        mean_forecast = samples.mean(axis=0)
        std_forecast = samples.std(axis=0)
        
        # Quantiles
        quantiles = {
            'q10': np.percentile(samples, 10, axis=0),
            'q25': np.percentile(samples, 25, axis=0),
            'q50': np.percentile(samples, 50, axis=0),
            'q75': np.percentile(samples, 75, axis=0),
            'q90': np.percentile(samples, 90, axis=0)
        }
        
        # Direction probability (probability price goes up)
        direction_probability = (samples[:, 0] > current_price).mean()
        
        # Magnitude forecast (expected absolute percentage change)
        returns = (samples / current_price) - 1
        magnitude_forecast = np.abs(returns).mean()
        
        # Volatility forecast
        volatility_forecast = returns.std()
        
        # Confidence score (inverse of prediction uncertainty)
        prediction_uncertainty = std_forecast[0] / mean_forecast[0] if mean_forecast[0] > 0 else 1.0
        confidence_score = max(0, 1 - prediction_uncertainty)
        
        # Trading probabilities
        long_probability = self._calculate_long_probability(samples, current_price)
        short_probability = self._calculate_short_probability(samples, current_price)
        
        # Optimal hold time
        optimal_hold_minutes = self._calculate_optimal_hold_time(samples, current_price)
        
        # Risk metrics
        returns_1d = returns[:, 0] if returns.shape[1] > 0 else returns.flatten()
        var_95 = np.percentile(returns_1d, 5) * current_price
        expected_shortfall = returns_1d[returns_1d <= np.percentile(returns_1d, 5)].mean() * current_price
        max_expected_gain = np.percentile(returns_1d, 95) * current_price
        max_expected_loss = np.percentile(returns_1d, 5) * current_price
        
        return ForecastResult(
            symbol=symbol,
            timestamp=datetime.now(),
            horizon_minutes=self.config.prediction_length,
            mean_forecast=mean_forecast,
            std_forecast=std_forecast,
            quantiles=quantiles,
            samples=samples,
            direction_probability=direction_probability,
            magnitude_forecast=magnitude_forecast,
            volatility_forecast=volatility_forecast,
            confidence_score=confidence_score,
            long_probability=long_probability,
            short_probability=short_probability,
            optimal_hold_minutes=optimal_hold_minutes,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            max_expected_gain=max_expected_gain,
            max_expected_loss=max_expected_loss
        )
    
    def _calculate_long_probability(self, samples: np.ndarray, current_price: float) -> float:
        """Calculate probability of profitable long trade"""
        
        # Assume 0.5% minimum profit target and 2% stop loss
        profit_target = current_price * 1.005
        stop_loss = current_price * 0.98
        
        # Check different time horizons
        profitable_outcomes = 0
        total_outcomes = 0
        
        for time_idx in range(min(samples.shape[1], 30)):  # Up to 30 minutes
            # Check if price hits profit target before stop loss
            for sample in samples:
                total_outcomes += 1
                
                # Simple check: if price at this time is above target
                if sample[time_idx] >= profit_target:
                    profitable_outcomes += 1
                elif sample[time_idx] <= stop_loss:
                    # Hit stop loss, count as loss
                    pass
                else:
                    # Still in trade, partial credit based on P&L
                    pnl_ratio = (sample[time_idx] - current_price) / current_price
                    if pnl_ratio > 0:
                        profitable_outcomes += pnl_ratio * 2  # Partial credit
        
        return profitable_outcomes / total_outcomes if total_outcomes > 0 else 0.5
    
    def _calculate_short_probability(self, samples: np.ndarray, current_price: float) -> float:
        """Calculate probability of profitable short trade"""
        
        # Assume 0.5% minimum profit target and 2% stop loss for short
        profit_target = current_price * 0.995
        stop_loss = current_price * 1.02
        
        profitable_outcomes = 0
        total_outcomes = 0
        
        for time_idx in range(min(samples.shape[1], 30)):
            for sample in samples:
                total_outcomes += 1
                
                if sample[time_idx] <= profit_target:
                    profitable_outcomes += 1
                elif sample[time_idx] >= stop_loss:
                    pass
                else:
                    pnl_ratio = (current_price - sample[time_idx]) / current_price
                    if pnl_ratio > 0:
                        profitable_outcomes += pnl_ratio * 2
        
        return profitable_outcomes / total_outcomes if total_outcomes > 0 else 0.5
    
    def _calculate_optimal_hold_time(self, samples: np.ndarray, current_price: float) -> int:
        """Calculate optimal hold time in minutes"""
        
        # Find time that maximizes expected profit
        max_expected_profit = 0
        optimal_time = 5  # Default 5 minutes
        
        for time_idx in range(min(samples.shape[1], 120)):  # Up to 2 hours
            expected_price = samples[:, time_idx].mean()
            expected_profit = (expected_price - current_price) / current_price
            
            # Account for time decay and transaction costs
            time_penalty = time_idx * 0.001  # Small penalty for longer holds
            adjusted_profit = expected_profit - time_penalty
            
            if adjusted_profit > max_expected_profit:
                max_expected_profit = adjusted_profit
                optimal_time = time_idx + 1
        
        return max(optimal_time, 5)  # Minimum 5 minutes
    
    async def get_forecast(self, symbol: str, force_refresh: bool = False) -> Optional[ForecastResult]:
        """Get forecast for a single symbol"""
        
        # Check cache first
        if not force_refresh and symbol in self.forecast_cache:
            cached_forecast = self.forecast_cache[symbol]
            age_minutes = (datetime.now() - cached_forecast.timestamp).total_seconds() / 60
            
            if age_minutes < self.cache_expiry_minutes:
                return cached_forecast
        
        # Generate new forecast
        forecasts = await self.generate_forecasts([symbol])
        return forecasts.get(symbol)
    
    async def get_market_forecast(self, symbols: Optional[List[str]] = None) -> MarketForecast:
        """Generate comprehensive market forecast"""
        
        if symbols is None:
            symbols = symbol_manager.get_active_symbols()
        
        # Generate individual forecasts
        forecasts = await self.generate_forecasts(symbols[:20])  # Limit to 20 symbols
        
        # Detect market regime
        market_regime = await self.regime_detector.detect_regime(forecasts)
        
        # Calculate overall confidence
        confidences = [f.confidence_score for f in forecasts.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return MarketForecast(
            timestamp=datetime.now(),
            forecasts=forecasts,
            market_regime=market_regime,
            overall_confidence=float(overall_confidence)
        )
    
    def get_strategy_signals(self, symbol: str, strategy: str) -> Dict:
        """Get strategy-specific signals based on forecasts"""
        
        forecast = self.forecast_cache.get(symbol)
        if not forecast:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No forecast available'}
        
        if strategy == "gap_and_go":
            return self._gap_and_go_signals(symbol, forecast)
        elif strategy == "orb":
            return self._orb_signals(symbol, forecast)
        elif strategy == "mean_reversion":
            return self._mean_reversion_signals(symbol, forecast)
        else:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Unknown strategy'}
    
    def _gap_and_go_signals(self, symbol: str, forecast: ForecastResult) -> Dict:
        """Generate Gap & Go signals from forecast"""
        
        symbol_metrics = symbol_manager.metrics.get(symbol)
        if not symbol_metrics:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        gap_percent = abs(symbol_metrics.gap_percent)
        
        # Check if significant gap
        if gap_percent < config.gap_and_go.min_gap_percent * 100:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Gap too small'}
        
        # Determine direction based on forecast
        if forecast.direction_probability > 0.75 and forecast.confidence_score > 0.7:
            action = 'BUY' if symbol_metrics.gap_percent > 0 else 'SELL'
            
            return {
                'action': action,
                'confidence': forecast.confidence_score * forecast.direction_probability,
                'target_price': forecast.quantiles['q75'][5],  # 5-minute target
                'stop_loss': forecast.quantiles['q25'][0],
                'hold_time': forecast.optimal_hold_minutes,
                'reason': f'Gap continuation probability: {forecast.direction_probability:.2f}'
            }
        
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Low probability/confidence'}
    
    def _orb_signals(self, symbol: str, forecast: ForecastResult) -> Dict:
        """Generate ORB signals from forecast"""
        
        # This would integrate with ORB strategy to provide breakout predictions
        # For now, basic implementation
        
        if forecast.volatility_forecast > 0.02 and forecast.confidence_score > 0.65:
            if forecast.direction_probability > 0.7:
                return {
                    'action': 'BUY',
                    'confidence': forecast.confidence_score,
                    'breakout_probability': forecast.direction_probability,
                    'volatility_expansion': forecast.volatility_forecast
                }
            elif forecast.direction_probability < 0.3:
                return {
                    'action': 'SELL',
                    'confidence': forecast.confidence_score,
                    'breakout_probability': 1 - forecast.direction_probability,
                    'volatility_expansion': forecast.volatility_forecast
                }
        
        return {'action': 'HOLD', 'confidence': 0.0}
    
    def _mean_reversion_signals(self, symbol: str, forecast: ForecastResult) -> Dict:
        """Generate mean reversion signals from forecast"""
        
        # Check for reversion probability
        current_price = self.price_buffers[symbol][-1]['price']
        expected_price = forecast.mean_forecast[10]  # 10-minute ahead
        
        reversion_strength = abs(expected_price - current_price) / current_price
        
        if reversion_strength > 0.01 and forecast.confidence_score > 0.6:
            if expected_price > current_price:
                return {
                    'action': 'BUY',
                    'confidence': forecast.confidence_score,
                    'reversion_target': expected_price,
                    'reversion_strength': reversion_strength
                }
            else:
                return {
                    'action': 'SELL',
                    'confidence': forecast.confidence_score,
                    'reversion_target': expected_price,
                    'reversion_strength': reversion_strength
                }
        
        return {'action': 'HOLD', 'confidence': 0.0}
    
    def track_forecast_accuracy(self, symbol: str, actual_price: float, minutes_ahead: int):
        """Track accuracy of previous forecasts"""
        
        if symbol not in self.forecast_cache:
            return
        
        forecast = self.forecast_cache[symbol]
        if minutes_ahead >= len(forecast.mean_forecast):
            return
        
        predicted_price = forecast.mean_forecast[minutes_ahead]
        accuracy = 1 - abs(predicted_price - actual_price) / actual_price
        
        self.forecast_accuracy[symbol].append(accuracy)
        
        # Keep only recent accuracy scores
        if len(self.forecast_accuracy[symbol]) > 100:
            self.forecast_accuracy[symbol] = self.forecast_accuracy[symbol][-100:]
    
    def get_performance_metrics(self) -> Dict:
        """Get model performance metrics"""
        
        total_forecasts = sum(len(acc) for acc in self.forecast_accuracy.values())
        avg_accuracy = np.mean([
            np.mean(acc) for acc in self.forecast_accuracy.values() if acc
        ]) if self.forecast_accuracy else 0.0
        
        avg_prediction_time = np.mean(self.prediction_times[-100:]) if self.prediction_times else 0.0
        
        return {
            'total_forecasts': total_forecasts,
            'average_accuracy': avg_accuracy,
            'average_prediction_time': avg_prediction_time,
            'symbols_tracked': len(self.forecast_accuracy),
            'cache_size': len(self.forecast_cache),
            'model_loaded': self.model_loaded
        }

class MarketRegimeDetector:
    """Detect current market regime from forecasts"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=50)
    
    async def detect_regime(self, forecasts: Dict[str, ForecastResult]) -> str:
        """Detect current market regime"""
        
        if not forecasts:
            return "unknown"
        
        # Analyze aggregate metrics
        avg_volatility = np.mean([f.volatility_forecast for f in forecasts.values()])
        avg_direction_prob = np.mean([f.direction_probability for f in forecasts.values()])
        avg_confidence = np.mean([f.confidence_score for f in forecasts.values()])
        
        # Classify regime
        if avg_volatility > 0.03:
            regime = "high_volatility"
        elif avg_volatility < 0.01:
            regime = "low_volatility"
        elif avg_direction_prob > 0.65:
            regime = "trending_bull"
        elif avg_direction_prob < 0.35:
            regime = "trending_bear"
        elif avg_confidence > 0.7:
            regime = "range_bound"
        else:
            regime = "uncertain"
        
        self.regime_history.append(regime)
        return regime

# Global instance
lag_llama_engine = LagLlamaEngine()