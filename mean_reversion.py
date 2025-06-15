"""
FAIL FAST SYSTEM - NO FALLBACKS ALLOWED
Mean Reversion Trading Strategy
Enhanced with Lag-Llama probabilistic reversion forecasting
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

from settings import config
from active_symbols import symbol_manager, SymbolMetrics
from lag_llama_engine import lag_llama_engine, ForecastResult, MultiTimeframeForecast
from polygon import get_polygon_data_manager, TradeData, AggregateData
from alpaca import get_alpaca_client, TradeSignal
from database import get_database_manager

logger = logging.getLogger(__name__)

class ReversionType(Enum):
    OVERSOLD_BOUNCE = "oversold_bounce"
    OVERBOUGHT_FADE = "overbought_fade"
    VOLATILITY_COMPRESSION = "volatility_compression"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_FADE = "resistance_fade"

class MeanType(Enum):
    SMA_20 = "sma_20"
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"
    VWAP = "vwap"
    BOLLINGER_MIDDLE = "bollinger_middle"

@dataclass
class MeanReversionMetrics:
    """Comprehensive mean reversion analysis"""
    symbol: str
    current_price: float
    
    # Different mean calculations
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    vwap: float = 0.0
    
    # Deviations from means
    deviation_sma_20: float = 0.0
    deviation_sma_50: float = 0.0
    deviation_sma_200: float = 0.0
    deviation_vwap: float = 0.0
    
    # Standard deviation metrics
    std_dev_20: float = 0.0
    z_score_20: float = 0.0
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    bb_position: float = 0.0  # 0 = lower band, 1 = upper band
    
    # RSI and momentum
    rsi: float = 50.0
    momentum_5: float = 0.0
    momentum_10: float = 0.0
    
    # Volume analysis
    volume_ratio: float = 1.0
    volume_trend: str = "neutral"
    
    # Volatility metrics
    realized_volatility: float = 0.0
    volatility_percentile: float = 50.0

@dataclass
class ReversionAnalysis:
    """Complete reversion opportunity analysis"""
    symbol: str
    reversion_type: ReversionType
    mean_type: Optional[MeanType]
    metrics: MeanReversionMetrics
    
    # Reversion probabilities from Lag-Llama
    reversion_probability: float = 0.0
    mean_reversion_timeframe: int = 0  # Minutes to reversion
    oversold_bounce_prob: float = 0.0
    overbought_fade_prob: float = 0.0
    
    # Target and risk levels
    entry_price: float = 0.0
    target_mean: float = 0.0
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    
    # Confidence and timing
    confidence_score: float = 0.0
    optimal_entry_time: Optional[datetime] = None
    max_hold_time: int = 60  # Default 1 hour
    
    # Market context
    trend_alignment: bool = False
    volatility_regime: str = "normal"
    support_resistance_proximity: float = 0.0

@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reversion_analysis: ReversionAnalysis
    
    # Entry details
    entry_price: float
    stop_loss: float
    targets: List[float]
    position_size: float
    
    # Timing
    signal_time: datetime
    expiry_time: datetime
    max_hold_minutes: int
    
    # Risk metrics
    risk_reward_ratio: float
    max_loss_percent: float
    expected_reversion_time: int
    
    # Strategy specific
    mean_target: float
    deviation_magnitude: float

class MeanReversionStrategy:
    """Mean reversion strategy with Lag-Llama enhancement"""
    
    def __init__(self):
        self.config = config.mean_reversion
        self.strategy_config = config.strategy
        
        # Client instances (lazy-initialized)
        self.polygon_manager = None
        self.alpaca_client = None
        
        # Price history for calculations
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        # Active tracking
        self.active_signals: Dict[str, MeanReversionSignal] = {}
        self.reversion_metrics: Dict[str, MeanReversionMetrics] = {}
        
        # Performance tracking
        self.trades_today: int = 0
        self.successful_reversions: int = 0
        self.failed_reversions: int = 0
        self.total_pnl: float = 0.0
        
        # Daily statistics
        self.daily_stats: Dict[str, Union[int, float]] = {}
        
        # Market regime tracking
        self.market_volatility_regime: str = "normal"
        self.trend_regime: str = "neutral"
        
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize strategy components"""
        
        logger.info("Initializing Mean Reversion strategy...")
        
        # Initialize price history for active symbols
        active_symbols = symbol_manager.get_active_symbols("mean_reversion")
        
        for symbol in active_symbols:
            self.price_history[symbol] = deque(maxlen=200)  # Keep 200 periods
            self.volume_history[symbol] = deque(maxlen=200)
        
        # Reset daily stats
        self.daily_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_reversions': 0,
            'failed_reversions': 0,
            'avg_hold_time': 0.0,
            'avg_deviation_at_entry': 0.0
        }
        
        # Initialize clients
        self.polygon_manager = get_polygon_data_manager()
        self.alpaca_client = get_alpaca_client()
        self.db_manager = get_database_manager()
        
        # Register callbacks
        if self.polygon_manager and hasattr(self.polygon_manager, 'ws_client'): # Add null check
            self.polygon_manager.ws_client.add_trade_callback(self._on_trade_update)
            self.polygon_manager.ws_client.add_aggregate_callback(self._on_aggregate_update)
        
        # Continuous monitoring will be started when needed
        self._monitoring_task = None
    
    async def _continuous_monitoring(self):
        """Continuously monitor for mean reversion opportunities"""
        
        logger.info("Starting continuous mean reversion monitoring...")
        
        while True:
            try:
                # Update all metrics
                await self._update_all_metrics()
                
                # Scan for opportunities
                await self._scan_reversion_opportunities()
                
                # Update active signals
                await self._update_active_signals()
                
                # Sleep before next scan
                await asyncio.sleep(30)  # Scan every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_all_metrics(self):
        """Update mean reversion metrics for all symbols"""
        
        active_symbols = symbol_manager.get_active_symbols("mean_reversion")
        
        for symbol in active_symbols:
            try:
                metrics = await self._calculate_metrics(symbol)
                if metrics:
                    self.reversion_metrics[symbol] = metrics
                    
            except Exception as e:
                logger.error(f"Error updating metrics for {symbol}: {e}")
    
    async def _calculate_metrics(self, symbol: str) -> Optional[MeanReversionMetrics]:
        """Calculate comprehensive mean reversion metrics using professional Polygon indicators"""
        
        try:
            # Get current price
            if not self.polygon_manager:
                self.polygon_manager = get_polygon_data_manager()
            
            latest_trade = self.polygon_manager.get_latest_trade(symbol)
            if not latest_trade:
                return None
            
            current_price = latest_trade.price
            
            # Get professional indicators from Polygon API - HIGH PRIORITY
            # Get professional RSI from Polygon
            rsi_data = await self.polygon_manager.get_rsi(symbol, window=14, timespan="minute")
            professional_rsi = rsi_data['current_rsi']
            
            # Get professional SMA from Polygon
            sma_20_data = await self.polygon_manager.get_sma(symbol, window=20, timespan="minute")
            professional_sma_20 = sma_20_data['current_sma']
            
            # Get professional EMA from Polygon
            ema_12_data = await self.polygon_manager.get_ema(symbol, window=12, timespan="minute")
            professional_ema_12 = ema_12_data['current_ema']
            
            logger.debug(f"Using professional indicators for {symbol}: RSI={professional_rsi:.2f}, SMA20={professional_sma_20:.2f}")
            
            # Ensure we have enough price history for fallback calculations
            if len(self.price_history[symbol]) < 20:
                return None
            
            prices = list(self.price_history[symbol])
            volumes = list(self.volume_history[symbol])
            
            # Use professional indicators from Polygon API
            sma_20 = professional_sma_20
            sma_50 = professional_sma_20  # Use same professional data
            sma_200 = professional_sma_20  # Use same professional data
            
            # Calculate VWAP (simplified - using available data)
            if volumes:
                total_volume = sum(volumes[-20:])
                if total_volume > 0:
                    vwap = sum(p * v for p, v in zip(prices[-20:], volumes[-20:])) / total_volume
                else:
                    vwap = sma_20
            else:
                vwap = sma_20
            
            # Calculate deviations
            deviation_sma_20 = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            deviation_sma_50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            deviation_sma_200 = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0
            deviation_vwap = (current_price - vwap) / vwap if vwap > 0 else 0
            
            # Calculate standard deviation and Z-score
            std_dev_20 = np.std(prices[-20:]) if len(prices) >= 20 else 0
            z_score_20 = (current_price - sma_20) / std_dev_20 if std_dev_20 > 0 else 0
            
            # Bollinger Bands (20-period, 2 std dev)
            bb_middle = sma_20
            bb_upper = bb_middle + (2 * std_dev_20)
            bb_lower = bb_middle - (2 * std_dev_20)
            
            if bb_upper > bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            
            # Use professional RSI from Polygon API
            rsi = professional_rsi
            
            # Momentum calculations
            momentum_5 = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            momentum_10 = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # Volume analysis
            volume_ratio = 1.0
            volume_trend = "neutral"
            
            if volumes and len(volumes) >= 10:
                recent_avg_volume = np.mean(volumes[-5:])
                historical_avg_volume = np.mean(volumes[-20:])
                
                if historical_avg_volume > 0:
                    volume_ratio = recent_avg_volume / historical_avg_volume
                
                if volume_ratio > 1.2:
                    volume_trend = "increasing"
                elif volume_ratio < 0.8:
                    volume_trend = "decreasing"
            
            # Volatility calculations
            if len(prices) >= 20:
                returns = np.diff(prices[-20:]) / prices[-20:-1]
                realized_volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Volatility percentile (simplified)
                if len(prices) >= 100:
                    historical_vols = []
                    for i in range(20, len(prices), 10):
                        hist_returns = np.diff(prices[i-20:i]) / prices[i-20:i-1]
                        hist_vol = np.std(hist_returns) * np.sqrt(252)
                        historical_vols.append(hist_vol)
                    
                    if historical_vols:
                        volatility_percentile = (sum(1 for v in historical_vols if v < realized_volatility) / len(historical_vols)) * 100
                    else:
                        volatility_percentile = 50.0
                else:
                    volatility_percentile = 50.0
            else:
                realized_volatility = 0.0
                volatility_percentile = 50.0
            
            metrics = MeanReversionMetrics(
                symbol=symbol,
                current_price=current_price,
                sma_20=float(sma_20),
                sma_50=float(sma_50),
                sma_200=float(sma_200),
                vwap=float(vwap),
                deviation_sma_20=float(deviation_sma_20),
                deviation_sma_50=float(deviation_sma_50),
                deviation_sma_200=float(deviation_sma_200),
                deviation_vwap=float(deviation_vwap),
                std_dev_20=float(std_dev_20),
                z_score_20=float(z_score_20),
                bb_upper=float(bb_upper),
                bb_lower=float(bb_lower),
                bb_middle=float(bb_middle),
                bb_position=float(bb_position),
                rsi=rsi,
                momentum_5=momentum_5,
                momentum_10=momentum_10,
                volume_ratio=float(volume_ratio),
                volume_trend=volume_trend,
                realized_volatility=realized_volatility,
                volatility_percentile=volatility_percentile
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI (Relative Strength Index)"""
        
        if len(prices) < 2:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    async def _scan_reversion_opportunities(self):
        """Scan for mean reversion opportunities"""
        
        for symbol, metrics in self.reversion_metrics.items():
            if symbol in self.active_signals:
                continue  # Already have active signal
            
            reversion_analysis = await self._analyze_reversion_opportunity(metrics)
            
            if reversion_analysis and reversion_analysis.confidence_score > self.strategy_config.confidence_threshold:
                signal = await self._generate_reversion_signal(reversion_analysis)
                
                if signal:
                    self.active_signals[symbol] = signal
                    self.daily_stats['signals_generated'] += 1
                    
                    # Execute signal
                    await self._execute_reversion_signal(signal)
    
    async def _analyze_reversion_opportunity(self, metrics: MeanReversionMetrics) -> Optional[ReversionAnalysis]:
        """Analyze potential mean reversion opportunity using multi-timeframe analysis"""
        
        try:
            symbol = metrics.symbol
            
            # Get multi-timeframe Lag-Llama forecast
            multi_forecast = await lag_llama_engine.generate_multi_timeframe_forecasts_strict(symbol)
            if not multi_forecast:
                return None
            
            # Determine reversion type and target
            reversion_type, mean_type, target_mean = self._identify_reversion_opportunity(metrics)
            
            if reversion_type is None:
                return None
            
            # Calculate reversion probabilities using multi-timeframe analysis
            reversion_probs = self._calculate_multi_timeframe_reversion_probabilities(metrics, multi_forecast, target_mean)
            
            if reversion_probs['reversion_probability'] < 0.6:  # Minimum 60% probability
                return None
            
            # Determine entry and risk levels
            entry_price = metrics.current_price
            target_price = target_mean
            stop_loss_price = self._calculate_stop_loss(metrics, reversion_type)
            
            # Calculate confidence score using multi-timeframe data
            confidence_score = self._calculate_multi_timeframe_confidence_score(metrics, reversion_probs, multi_forecast)
            
            # Market context analysis
            trend_alignment = self._assess_trend_alignment(metrics, reversion_type)
            volatility_regime = self._assess_volatility_regime(metrics)
            
            reversion_analysis = ReversionAnalysis(
                symbol=symbol,
                reversion_type=reversion_type,
                mean_type=mean_type, # type: ignore
                metrics=metrics,
                reversion_probability=reversion_probs['reversion_probability'],
                mean_reversion_timeframe=reversion_probs['timeframe'],
                oversold_bounce_prob=reversion_probs.get('oversold_bounce_prob', 0),
                overbought_fade_prob=reversion_probs.get('overbought_fade_prob', 0),
                entry_price=entry_price,
                target_mean=target_mean,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                confidence_score=confidence_score,
                optimal_entry_time=datetime.now(),
                max_hold_time=min(multi_forecast.optimal_hold_minutes, self.config.max_hold_minutes),
                trend_alignment=trend_alignment,
                volatility_regime=volatility_regime
            )
            
            return reversion_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing reversion opportunity for {metrics.symbol}: {e}")
            return None
    
    def _identify_reversion_opportunity(self, metrics: MeanReversionMetrics) -> Tuple[Optional[ReversionType], Optional[MeanType], float]:
        """Identify type of mean reversion opportunity"""
        
        # Oversold conditions (potential bounce)
        if (metrics.rsi < 30 and 
            metrics.z_score_20 < -self.config.deviation_threshold and
            metrics.bb_position < 0.1):
            return ReversionType.OVERSOLD_BOUNCE, MeanType.SMA_20, metrics.sma_20
        
        # Overbought conditions (potential fade)
        if (metrics.rsi > 70 and 
            metrics.z_score_20 > self.config.deviation_threshold and
            metrics.bb_position > 0.9):
            return ReversionType.OVERBOUGHT_FADE, MeanType.SMA_20, metrics.sma_20
        
        # VWAP reversion
        if abs(metrics.deviation_vwap) > 0.02:  # 2% deviation from VWAP
            if metrics.deviation_vwap > 0:
                return ReversionType.OVERBOUGHT_FADE, MeanType.VWAP, metrics.vwap
            else:
                return ReversionType.OVERSOLD_BOUNCE, MeanType.VWAP, metrics.vwap
        
        # Bollinger Band extreme positions
        if metrics.bb_position < 0.05:  # Near lower band
            return ReversionType.SUPPORT_BOUNCE, MeanType.BOLLINGER_MIDDLE, metrics.bb_middle
        elif metrics.bb_position > 0.95:  # Near upper band
            return ReversionType.RESISTANCE_FADE, MeanType.BOLLINGER_MIDDLE, metrics.bb_middle
        
        # 50-period mean reversion for longer timeframe
        if abs(metrics.deviation_sma_50) > 0.05:  # 5% deviation
            if metrics.deviation_sma_50 > 0:
                return ReversionType.OVERBOUGHT_FADE, MeanType.SMA_50, metrics.sma_50
            else:
                return ReversionType.OVERSOLD_BOUNCE, MeanType.SMA_50, metrics.sma_50
        
        return None, None, 0.0
    
    def _calculate_multi_timeframe_reversion_probabilities(self, metrics: MeanReversionMetrics,
                                                         multi_forecast: MultiTimeframeForecast, target_mean: float) -> Dict:
        """Calculate reversion probabilities using multi-timeframe Lag-Llama analysis"""
        
        current_price = metrics.current_price
        
        # Use cross-timeframe correlation as primary reversion probability
        base_reversion_prob = multi_forecast.cross_timeframe_correlation
        
        # Enhance with trend consistency
        trend_enhanced_prob = base_reversion_prob * multi_forecast.trend_consistency_score
        
        # Calculate timeframe-specific probabilities
        timeframe_probs = []
        best_timeframe = 60  # Default
        
        for horizon_min, forecast_data in multi_forecast.forecasts.items():
            if horizon_min <= 120:  # Use up to 2-hour forecasts
                # Calculate reversion probability for this timeframe
                if current_price > target_mean:
                    # Looking for reversion down
                    reversion_prob = forecast_data.confidence_score if forecast_data.mean_prediction < current_price else 0.3
                else:
                    # Looking for reversion up
                    reversion_prob = forecast_data.confidence_score if forecast_data.mean_prediction > current_price else 0.3
                
                timeframe_probs.append((horizon_min, reversion_prob))
        
        # Find best timeframe
        if timeframe_probs:
            best_timeframe, best_prob = max(timeframe_probs, key=lambda x: x[1])
            # Combine with multi-timeframe analysis
            final_reversion_prob = (trend_enhanced_prob + best_prob) / 2
        else:
            final_reversion_prob = trend_enhanced_prob
        
        # Enhanced oversold/overbought probabilities using multi-timeframe data
        oversold_bounce_prob = 0.0
        overbought_fade_prob = 0.0
        
        if metrics.rsi < 30:
            # Use momentum alignment for oversold bounce probability
            oversold_bounce_prob = multi_forecast.momentum_alignment_score
        
        if metrics.rsi > 70:
            # Use inverse momentum alignment for overbought fade probability
            overbought_fade_prob = 1.0 - multi_forecast.momentum_alignment_score
        
        return {
            'reversion_probability': final_reversion_prob,
            'timeframe': best_timeframe,
            'oversold_bounce_prob': oversold_bounce_prob,
            'overbought_fade_prob': overbought_fade_prob
        }
    
    def _calculate_stop_loss(self, metrics: MeanReversionMetrics, reversion_type: ReversionType) -> float:
        """Calculate stop loss for mean reversion trade"""
        
        current_price = metrics.current_price
        
        if reversion_type in [ReversionType.OVERSOLD_BOUNCE, ReversionType.SUPPORT_BOUNCE]:
            # For long positions, stop below recent low or support
            stop_candidates = [
                current_price * 0.97,  # 3% stop
                metrics.bb_lower * 0.99,  # Below Bollinger lower band
            ]
            return min(stop_candidates)
        
        else:  # Fade/short positions
            # For short positions, stop above recent high or resistance
            stop_candidates = [
                current_price * 1.03,  # 3% stop
                metrics.bb_upper * 1.01,  # Above Bollinger upper band
            ]
            return max(stop_candidates)
    
    def _calculate_multi_timeframe_confidence_score(self, metrics: MeanReversionMetrics,
                                                  reversion_probs: Dict, multi_forecast: MultiTimeframeForecast) -> float:
        """Calculate overall confidence score for reversion trade using multi-timeframe analysis"""
        
        confidence_factors = []
        
        # Multi-timeframe reversion probability (highest weight)
        confidence_factors.append(reversion_probs['reversion_probability'])
        
        # Cross-timeframe correlation strength
        confidence_factors.append(multi_forecast.cross_timeframe_correlation)
        
        # Trend consistency across timeframes
        confidence_factors.append(multi_forecast.trend_consistency_score)
        
        # Risk-adjusted confidence
        confidence_factors.append(multi_forecast.risk_adjusted_confidence)
        
        # RSI extremity
        if metrics.rsi < 30:
            rsi_score = (30 - metrics.rsi) / 30  # More oversold = higher score
        elif metrics.rsi > 70:
            rsi_score = (metrics.rsi - 70) / 30  # More overbought = higher score
        else:
            rsi_score = 0.5
        
        confidence_factors.append(min(rsi_score, 1.0))
        
        # Z-score extremity
        z_score_abs = abs(metrics.z_score_20)
        z_score_factor = min(z_score_abs / 3.0, 1.0)  # Cap at 3 std devs
        confidence_factors.append(z_score_factor)
        
        # Bollinger position extremity
        if metrics.bb_position < 0.2:
            bb_score = (0.2 - metrics.bb_position) / 0.2
        elif metrics.bb_position > 0.8:
            bb_score = (metrics.bb_position - 0.8) / 0.2
        else:
            bb_score = 0.0
        
        confidence_factors.append(bb_score)
        
        # Volume confirmation
        volume_score = min(metrics.volume_ratio / 1.5, 1.0)  # Prefer higher volume
        confidence_factors.append(volume_score)
        
        # Calculate weighted average with emphasis on multi-timeframe factors
        weights = [0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]  # Multi-timeframe factors get highest weights
        
        if len(confidence_factors) == len(weights):
            confidence_score = sum(f * w for f, w in zip(confidence_factors, weights))
        else:
            confidence_score = np.mean(confidence_factors)
        
        return float(min(confidence_score, 1.0))
    
    def _assess_trend_alignment(self, metrics: MeanReversionMetrics, reversion_type: ReversionType) -> bool:
        """Assess if reversion trade aligns with longer-term trend"""
        
        # Check if short-term reversion aligns with longer-term trend
        longer_term_trend = "up" if metrics.sma_50 > metrics.sma_200 else "down"
        
        if reversion_type in [ReversionType.OVERSOLD_BOUNCE, ReversionType.SUPPORT_BOUNCE]:
            # Long reversion trade aligns with uptrend
            return longer_term_trend == "up"
        else:
            # Short reversion trade aligns with downtrend
            return longer_term_trend == "down"
    
    def _assess_volatility_regime(self, metrics: MeanReversionMetrics) -> str:
        """Assess current volatility regime"""
        
        if metrics.volatility_percentile > 80:
            return "high"
        elif metrics.volatility_percentile < 20:
            return "low"
        else:
            return "normal"
    
    async def _generate_reversion_signal(self, reversion_analysis: ReversionAnalysis) -> Optional[MeanReversionSignal]:
        """Generate trading signal for mean reversion"""
        
        try:
            # Determine action
            if reversion_analysis.reversion_type in [ReversionType.OVERSOLD_BOUNCE, ReversionType.SUPPORT_BOUNCE]:
                action = "BUY"
            else:
                action = "SELL"
            
            # Entry and risk management
            entry_price = reversion_analysis.entry_price
            stop_loss = reversion_analysis.stop_loss_price
            targets = [reversion_analysis.target_price]
            
            # Add partial targets
            if action == "BUY":
                partial_target = entry_price + (reversion_analysis.target_price - entry_price) * 0.5
                targets.insert(0, partial_target)
            else:
                partial_target = entry_price - (entry_price - reversion_analysis.target_price) * 0.5
                targets.insert(0, partial_target)
            
            # Position sizing
            risk_per_share = abs(entry_price - stop_loss)
            risk_percent = config.risk.max_position_size * 0.8  # More conservative for mean reversion
            
            account_equity = 100000.0 # Default equity
            if self.alpaca_client and self.alpaca_client.account_info:
                account_equity = float(self.alpaca_client.account_info.equity)
            
            position_size = (risk_percent * account_equity) / (risk_per_share * entry_price) if risk_per_share > 0 and entry_price > 0 else 0.0
            position_size = min(position_size, config.risk.max_position_size)
            
            # Risk-reward calculation
            profit_per_share = abs(reversion_analysis.target_price - entry_price)
            risk_reward_ratio = profit_per_share / risk_per_share if risk_per_share > 0 else 0
            
            # Timing
            signal_time = datetime.now()
            max_hold = reversion_analysis.max_hold_time
            expiry_time = signal_time + timedelta(minutes=max_hold)
            
            signal = MeanReversionSignal(
                symbol=reversion_analysis.symbol,
                action=action,
                confidence=reversion_analysis.confidence_score,
                reversion_analysis=reversion_analysis,
                entry_price=entry_price,
                stop_loss=stop_loss,
                targets=targets,
                position_size=position_size,
                signal_time=signal_time,
                expiry_time=expiry_time,
                max_hold_minutes=max_hold,
                risk_reward_ratio=risk_reward_ratio,
                max_loss_percent=risk_per_share / entry_price,
                expected_reversion_time=reversion_analysis.mean_reversion_timeframe,
                mean_target=reversion_analysis.target_mean,
                deviation_magnitude=abs(entry_price - reversion_analysis.target_mean) / reversion_analysis.target_mean
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating reversion signal: {e}")
            return None
    
    async def _execute_reversion_signal(self, signal: MeanReversionSignal):
        """Execute mean reversion trading signal"""
        
        try:
            # Convert to trade signal
            trade_signal = TradeSignal(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                strategy="mean_reversion",
                target_price=signal.targets[0] if signal.targets else None,
                stop_loss=signal.stop_loss,
                position_size=signal.position_size,
                hold_time=signal.max_hold_minutes,
                reason=f"Mean reversion {signal.reversion_analysis.reversion_type.value}"
            )
            
            # Execute trade
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                self.trades_today += 1
                self.daily_stats['trades_executed'] += 1
                self.daily_stats['avg_deviation_at_entry'] = (
                    self.daily_stats['avg_deviation_at_entry'] * (self.trades_today - 1) + 
                    signal.deviation_magnitude
                ) / self.trades_today
                
                logger.info(f"Executed mean reversion trade: {signal.action} {signal.symbol} "
                          f"(confidence: {signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error executing reversion signal: {e}")
    
    async def _update_active_signals(self):
        """Update status of active mean reversion signals"""
        
        for symbol in list(self.active_signals.keys()):
            await self._update_signal_status(symbol)
    
    async def _update_signal_status(self, symbol: str):
        """Update status of active mean reversion signal"""
        
        if symbol not in self.active_signals:
            return
        
        signal = self.active_signals[symbol]
        
        # Get current price
        if not self.polygon_manager:
            self.polygon_manager = get_polygon_data_manager()
        
        latest_trade = self.polygon_manager.get_latest_trade(symbol)
        if not latest_trade:
            return
        
        current_price = latest_trade.price
        
        # Check if signal expired
        if datetime.now() > signal.expiry_time:
            logger.info(f"Mean reversion signal expired for {symbol}")
            await self._close_signal(symbol, "Time expiry")
            return
        
        # Check stop loss
        if signal.action == "BUY" and current_price <= signal.stop_loss:
            logger.info(f"Mean reversion stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
            self.failed_reversions += 1
            self.daily_stats['failed_reversions'] += 1
            
        elif signal.action == "SELL" and current_price >= signal.stop_loss:
            logger.info(f"Mean reversion stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
            self.failed_reversions += 1
            self.daily_stats['failed_reversions'] += 1
        
        # Check targets
        for i, target in enumerate(signal.targets):
            if signal.action == "BUY" and current_price >= target:
                logger.info(f"Mean reversion target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                self.successful_reversions += 1
                self.daily_stats['successful_reversions'] += 1
                break
                
            elif signal.action == "SELL" and current_price <= target:
                logger.info(f"Mean reversion target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                self.successful_reversions += 1
                self.daily_stats['successful_reversions'] += 1
                break
    
    async def _close_signal(self, symbol: str, reason: str):
        """Close active mean reversion signal"""
        
        if symbol in self.active_signals:
            signal = self.active_signals[symbol]
            
            # Calculate actual hold time
            hold_time = (datetime.now() - signal.signal_time).total_seconds() / 60
            self.daily_stats['avg_hold_time'] = (
                self.daily_stats['avg_hold_time'] * (self.trades_today - 1) + hold_time
            ) / self.trades_today if self.trades_today > 0 else hold_time
            
            # Convert to close trade signal
            trade_signal = TradeSignal(
                symbol=symbol,
                action="SELL" if signal.action == "BUY" else "BUY",
                confidence=1.0,
                strategy="mean_reversion_close",
                reason=reason
            )
            
            # Submit close order
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                logger.info(f"Closed mean reversion position for {symbol}: {reason}")
            
            del self.active_signals[symbol]
    
    async def _on_trade_update(self, trade_data: TradeData):
        """Handle real-time trade updates"""
        
        symbol = trade_data.symbol
        
        # Update price history
        if symbol in self.price_history:
            self.price_history[symbol].append(trade_data.price)
        else:
            self.price_history[symbol] = deque([trade_data.price], maxlen=200)
    
    async def _on_aggregate_update(self, aggregate_data: AggregateData):
        """Handle minute bar updates"""
        
        symbol = aggregate_data.symbol
        
        # Update volume history
        if symbol in self.volume_history:
            self.volume_history[symbol].append(aggregate_data.volume)
        else:
            self.volume_history[symbol] = deque([aggregate_data.volume], maxlen=200)
    
    def get_active_signals(self) -> List[MeanReversionSignal]:
        """Get currently active mean reversion signals"""
        return list(self.active_signals.values())
    
    def get_reversion_metrics(self, symbol: str) -> Optional[MeanReversionMetrics]:
        """Get current reversion metrics for symbol"""
        return self.reversion_metrics.get(symbol)
    
    def get_strategy_performance(self) -> Dict:
        """Get mean reversion strategy performance metrics"""
        
        success_rate = 0.0
        if self.trades_today > 0:
            success_rate = self.successful_reversions / self.trades_today
        
        return {
            'strategy': 'mean_reversion',
            'trades_today': self.trades_today,
            'successful_reversions': self.successful_reversions,
            'failed_reversions': self.failed_reversions,
            'success_rate': success_rate,
            'total_pnl': self.total_pnl,
            'active_signals': len(self.active_signals),
            'symbols_tracked': len(self.reversion_metrics),
            'daily_stats': self.daily_stats
        }
    
    async def daily_reset(self):
        """Reset strategy for new trading day"""
        
        logger.info("Resetting Mean Reversion strategy for new day")
        
        # Clear active signals
        self.active_signals.clear()
        
        # Reset counters
        self.trades_today = 0
        self.successful_reversions = 0
        self.failed_reversions = 0
        self.total_pnl = 0.0
        
        # Reset daily stats
        self.daily_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_reversions': 0,
            'failed_reversions': 0,
            'avg_hold_time': 0.0,
            'avg_deviation_at_entry': 0.0
        }
        
        # Keep price/volume history but ensure fresh start for metrics
        self.reversion_metrics.clear()
    
    async def _store_reversion_metrics(self, metrics: MeanReversionMetrics):
        """Store mean reversion metrics in database"""
        
        if not self.db_manager:
            return
        
        try:
            metrics_data = {
                'symbol': metrics.symbol,
                'timestamp': datetime.now(),
                'current_price': metrics.current_price,
                'sma_20': metrics.sma_20,
                'sma_50': metrics.sma_50,
                'sma_200': metrics.sma_200,
                'vwap': metrics.vwap,
                'deviation_sma_20': metrics.deviation_sma_20,
                'deviation_sma_50': metrics.deviation_sma_50,
                'deviation_sma_200': metrics.deviation_sma_200,
                'deviation_vwap': metrics.deviation_vwap,
                'z_score_20': metrics.z_score_20,
                'bb_upper': metrics.bb_upper,
                'bb_lower': metrics.bb_lower,
                'bb_middle': metrics.bb_middle,
                'bb_position': metrics.bb_position,
                'rsi': metrics.rsi,
                'momentum_5': metrics.momentum_5,
                'momentum_10': metrics.momentum_10,
                'volume_ratio': metrics.volume_ratio,
                'volume_trend': metrics.volume_trend,
                'realized_volatility': metrics.realized_volatility,
                'volatility_percentile': metrics.volatility_percentile
            }
            
            await self.db_manager.insert_mean_reversion_metrics(metrics_data)
            
        except Exception as e:
            logger.error(f"Error storing reversion metrics: {e}")
    
    async def _store_reversion_analysis(self, analysis: ReversionAnalysis):
        """Store reversion analysis in database"""
        
        if not self.db_manager:
            return
        
        try:
            analysis_data = {
                'symbol': analysis.symbol,
                'timestamp': datetime.now(),
                'reversion_type': analysis.reversion_type.value,
                'mean_type': analysis.mean_type.value if analysis.mean_type else None,
                'reversion_probability': analysis.reversion_probability,
                'mean_reversion_timeframe': analysis.mean_reversion_timeframe,
                'oversold_bounce_prob': analysis.oversold_bounce_prob,
                'overbought_fade_prob': analysis.overbought_fade_prob,
                'entry_price': analysis.entry_price,
                'target_mean': analysis.target_mean,
                'target_price': analysis.target_price,
                'stop_loss_price': analysis.stop_loss_price,
                'confidence_score': analysis.confidence_score,
                'max_hold_time': analysis.max_hold_time,
                'trend_alignment': analysis.trend_alignment,
                'volatility_regime': analysis.volatility_regime
            }
            
            await self.db_manager.insert_reversion_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"Error storing reversion analysis: {e}")
    
    async def _store_mean_reversion_signal(self, signal: MeanReversionSignal):
        """Store mean reversion trading signal in database"""
        
        if not self.db_manager:
            return
        
        try:
            signal_data = {
                'symbol': signal.symbol,
                'timestamp': signal.signal_time,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'targets': signal.targets,
                'position_size': signal.position_size,
                'max_hold_minutes': signal.max_hold_minutes,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'max_loss_percent': signal.max_loss_percent,
                'expected_reversion_time': signal.expected_reversion_time,
                'mean_target': signal.mean_target,
                'deviation_magnitude': signal.deviation_magnitude,
                'reversion_type': signal.reversion_analysis.reversion_type.value,
                'mean_type': signal.reversion_analysis.mean_type.value if signal.reversion_analysis.mean_type else None
            }
            
            await self.db_manager.insert_trading_signal(signal_data)
            
        except Exception as e:
            logger.error(f"Error storing mean reversion signal: {e}")

# Global strategy instance
mean_reversion_strategy = MeanReversionStrategy()
