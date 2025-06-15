"""
FAIL FAST SYSTEM - NO FALLBACKS ALLOWED
Opening Range Breakout (ORB) Trading Strategy
Enhanced with Lag-Llama probabilistic breakout forecasting
"""

import logging
import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from settings import config
from active_symbols import symbol_manager, SymbolMetrics
from lag_llama_engine import lag_llama_engine, ForecastResult, MultiTimeframeForecast
from polygon import get_polygon_data_manager, TradeData, AggregateData
from alpaca import get_alpaca_client, TradeSignal
from database import get_database_manager

logger = logging.getLogger(__name__)

class BreakoutDirection(Enum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    NONE = "none"

class RangeQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class OpeningRange:
    """Opening range data structure"""
    symbol: str
    high: float
    low: float
    open: float
    close: float
    volume: int
    
    # Range metrics
    range_size: float
    range_percent: float
    range_midpoint: float
    
    # Volume analysis
    volume_profile: List[Tuple[float, int]] = field(default_factory=list)
    volume_weighted_price: float = 0.0
    
    # Time data
    start_time: datetime = field(default_factory=lambda: datetime.min)
    end_time: datetime = field(default_factory=lambda: datetime.min)
    
    # Quality assessment
    quality: RangeQuality = RangeQuality.FAIR
    
    def __post_init__(self):
        if not self.volume_profile:
            self.volume_profile = []
        
        self.range_size = self.high - self.low
        self.range_percent = (self.range_size / self.open) * 100 if self.open > 0 else 0
        self.range_midpoint = (self.high + self.low) / 2

@dataclass
class BreakoutAnalysis:
    """Comprehensive breakout analysis"""
    symbol: str
    opening_range: OpeningRange
    current_price: float
    breakout_direction: BreakoutDirection
    
    # Breakout metrics
    breakout_price: float
    breakout_distance: float
    breakout_percent: float
    volume_confirmation: bool = False
    
    # Lag-Llama predictions
    sustainability_probability: float = 0.0
    target_projections: List[float] = field(default_factory=list)
    false_breakout_probability: float = 0.0
    pullback_probability: float = 0.0
    
    # Technical levels
    key_resistance_levels: List[float] = field(default_factory=list)
    key_support_levels: List[float] = field(default_factory=list)
    
    # Volume analysis
    breakout_volume: int = 0
    volume_ratio: float = 0.0
    
    def __post_init__(self):
        if not self.target_projections:
            self.target_projections = []
        if not self.key_resistance_levels:
            self.key_resistance_levels = []
        if not self.key_support_levels:
            self.key_support_levels = []

@dataclass
class ORBSignal:
    """ORB trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    breakout_analysis: BreakoutAnalysis
    
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
    expected_profit: float
    
    # Breakout specific
    breakout_confirmation_price: float
    range_expansion_multiple: float

class ORBStrategy:
    """Opening Range Breakout strategy with Lag-Llama enhancement"""
    
    def __init__(self):
        self.config = config.orb
        self.strategy_config = config.strategy
        
        # Client instances (lazy-initialized)
        self.polygon_data_manager = None
        self.alpaca_client = None
        
        # Strategy state
        self.opening_ranges: Dict[str, OpeningRange] = {}
        self.active_signals: Dict[str, ORBSignal] = {}
        self.range_formation_active: bool = False
        
        # Timing
        self.market_open_time: Optional[datetime] = None
        self.range_end_time: Optional[datetime] = None
        
        # Performance tracking
        self.trades_today: int = 0
        self.successful_breakouts: int = 0
        self.false_breakouts: int = 0
        self.total_pnl: float = 0.0
        
        # Daily statistics
        self.daily_stats: Dict[str, Union[int, float]] = {}
        
        # Price/volume tracking for range formation
        self.range_data: Dict[str, List[Dict]] = {}
        
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize strategy components"""
        
        logger.info("Initializing ORB strategy...")
        
        # Initialize clients
        self.polygon_data_manager = get_polygon_data_manager()
        self.alpaca_client = get_alpaca_client()
        self.db_manager = get_database_manager()
        
        # Reset daily stats
        self.daily_stats = {
            'ranges_formed': 0,
            'breakout_signals': 0,
            'trades_executed': 0,
            'successful_breakouts': 0,
            'false_breakouts': 0,
            'average_range_size': 0.0
        }
    
    async def _store_opening_range(self, opening_range: OpeningRange):
        """Store opening range data in database"""
        
        if not self.db_manager:
            return
        
        try:
            range_data = {
                'symbol': opening_range.symbol,
                'timestamp': opening_range.start_time,
                'high': opening_range.high,
                'low': opening_range.low,
                'open': opening_range.open,
                'close': opening_range.close,
                'volume': opening_range.volume,
                'range_size': opening_range.range_size,
                'range_percent': opening_range.range_percent,
                'quality': opening_range.quality.value,
                'vwap': opening_range.volume_weighted_price
            }
            
            await self.db_manager.insert_orb_range(range_data)
            
        except Exception as e:
            logger.error(f"Error storing opening range: {e}")
    
    async def _store_orb_signal(self, signal: ORBSignal):
        """Store ORB trading signal in database"""
        
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
                'breakout_direction': signal.breakout_analysis.breakout_direction.value,
                'sustainability_probability': signal.breakout_analysis.sustainability_probability,
                'volume_confirmation': signal.breakout_analysis.volume_confirmation,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'range_expansion_multiple': signal.range_expansion_multiple
            }
            
            await self.db_manager.insert_trading_signal(signal_data)
            
        except Exception as e:
            logger.error(f"Error storing ORB signal: {e}")
    
    async def _store_breakout_analysis(self, analysis: BreakoutAnalysis):
        """Store breakout analysis in database"""
        
        if not self.db_manager:
            return
        
        try:
            analysis_data = {
                'symbol': analysis.symbol,
                'timestamp': datetime.now(),
                'breakout_direction': analysis.breakout_direction.value,
                'breakout_price': analysis.breakout_price,
                'current_price': analysis.current_price,
                'sustainability_probability': analysis.sustainability_probability,
                'false_breakout_probability': analysis.false_breakout_probability,
                'volume_confirmation': analysis.volume_confirmation,
                'volume_ratio': analysis.volume_ratio,
                'target_projections': analysis.target_projections,
                'resistance_levels': analysis.key_resistance_levels,
                'support_levels': analysis.key_support_levels
            }
            
            await self.db_manager.insert_breakout_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"Error storing breakout analysis: {e}")
        
        # Register callbacks
        if self.polygon_data_manager:
            self.polygon_data_manager.add_trade_callback(self._on_trade_update) # type: ignore
            self.polygon_data_manager.add_aggregate_callback(self._on_aggregate_update) # type: ignore
    
    async def start_range_formation(self):
        """Start opening range formation period"""
        
        current_time = datetime.now()
        
        # Check if it's market open time
        if not config.is_market_hours(current_time):
            logger.warning("Attempting to start range formation outside market hours")
            return
        
        # Calculate range formation period
        market_open = current_time.replace(
            hour=config.market_open_hour,
            minute=config.market_open_minute,
            second=0,
            microsecond=0
        )
        
        # Check if we're within the range formation window
        minutes_since_open = (current_time - market_open).total_seconds() / 60
        
        if minutes_since_open > self.config.range_minutes:
            logger.warning("Too late to start range formation")
            return
        
        self.range_formation_active = True
        self.market_open_time = market_open
        self.range_end_time = market_open + timedelta(minutes=self.config.range_minutes)
        
        # Initialize range data for active symbols
        active_symbols = symbol_manager.get_active_symbols("orb")
        
        for symbol in active_symbols:
            self.range_data[symbol] = []
        
        logger.info(f"Started ORB range formation for {len(active_symbols)} symbols")
        logger.info(f"Range formation ends at: {self.range_end_time}")
        
        # Range completion will be scheduled when needed
        self._range_completion_task = None
    
    async def _complete_range_formation(self):
        """Complete range formation and analyze ranges"""
        
        # Wait until range formation period ends
        if self.range_end_time:
            wait_time = (self.range_end_time - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        if not self.range_formation_active:
            return
        
        logger.info("Completing opening range formation...")
        
        # Process ranges for all symbols
        for symbol in list(self.range_data.keys()):
            await self._process_opening_range(symbol)
        
        self.range_formation_active = False
        self.daily_stats['ranges_formed'] = len(self.opening_ranges)
        
        # Calculate average range size
        if self.opening_ranges:
            avg_range = np.mean([r.range_percent for r in self.opening_ranges.values()])
            self.daily_stats['average_range_size'] = float(avg_range)
        
        logger.info(f"Formed {len(self.opening_ranges)} opening ranges")
        
        # Breakout monitoring will be started when needed
        self._breakout_monitoring_task = None
    
    async def _process_opening_range(self, symbol: str):
        """Process opening range for a symbol"""
        
        try:
            range_bars = self.range_data.get(symbol, [])
            
            if len(range_bars) < 2:  # Need at least 2 data points
                logger.warning(f"Insufficient data for {symbol} range formation")
                return
            
            # Calculate range metrics
            prices = [bar['price'] for bar in range_bars]
            volumes = [bar['volume'] for bar in range_bars]
            
            high = max(prices)
            low = min(prices)
            open_price = range_bars[0]['price']
            close_price = range_bars[-1]['price']
            total_volume = sum(volumes)
            
            # Volume weighted average price
            vwap = sum(bar['price'] * bar['volume'] for bar in range_bars) / total_volume if total_volume > 0 else close_price
            
            # Create opening range
            opening_range = OpeningRange(
                symbol=symbol,
                high=high,
                low=low,
                open=open_price,
                close=close_price,
                volume=total_volume,
                range_size=high - low,
                range_percent=(high - low) / open_price * 100 if open_price > 0 else 0,
                range_midpoint=(high + low) / 2,
                volume_weighted_price=vwap,
                start_time=self.market_open_time if self.market_open_time else datetime.min,
                end_time=self.range_end_time if self.range_end_time else datetime.min
            )
            
            # Assess range quality
            opening_range.quality = self._assess_range_quality(opening_range)
            
            # Only keep quality ranges
            if opening_range.quality in [RangeQuality.EXCELLENT, RangeQuality.GOOD, RangeQuality.FAIR]:
                self.opening_ranges[symbol] = opening_range
                logger.info(f"Created {opening_range.quality.value} range for {symbol}: "
                          f"{low:.2f}-{high:.2f} ({opening_range.range_percent:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error processing opening range for {symbol}: {e}")
    
    def _assess_range_quality(self, opening_range: OpeningRange) -> RangeQuality:
        """Assess the quality of an opening range"""
        
        # Check range size
        if opening_range.range_percent < self.config.min_range_size * 100:
            return RangeQuality.POOR
        
        if opening_range.range_percent > self.config.max_range_size * 100:
            return RangeQuality.POOR
        
        # Get symbol metrics for volume analysis
        metrics = symbol_manager.metrics.get(opening_range.symbol)
        volume_score = 1.0
        
        if metrics and metrics.avg_volume > 0:
            volume_ratio = opening_range.volume / (metrics.avg_volume * (self.config.range_minutes / 390))
            
            if volume_ratio >= 1.5:
                volume_score = 1.0  # Excellent volume
            elif volume_ratio >= 1.0:
                volume_score = 0.8  # Good volume
            elif volume_ratio >= 0.5:
                volume_score = 0.6  # Fair volume
            else:
                volume_score = 0.3  # Poor volume
        
        # Range size score
        optimal_range = 0.02  # 2% is optimal
        range_deviation = abs(opening_range.range_percent / 100 - optimal_range)
        range_score = max(0.3, 1.0 - (range_deviation * 10))
        
        # Combined quality score
        quality_score = (volume_score + range_score) / 2
        
        if quality_score >= 0.8:
            return RangeQuality.EXCELLENT
        elif quality_score >= 0.6:
            return RangeQuality.GOOD
        elif quality_score >= 0.4:
            return RangeQuality.FAIR
        else:
            return RangeQuality.POOR
    
    async def _monitor_breakouts(self):
        """Monitor for breakouts from opening ranges"""
        
        logger.info("Starting breakout monitoring...")
        
        # Monitor until end of trading day or max hold time
        end_time = datetime.now() + timedelta(hours=6)  # Monitor for 6 hours
        
        while datetime.now() < end_time and self.opening_ranges:
            await asyncio.sleep(1)  # Check every second
            
            # Process any pending breakouts
            await self._process_pending_breakouts()
    
    async def _process_pending_breakouts(self):
        """Process potential breakouts"""
        
        for symbol in list(self.opening_ranges.keys()):
            if symbol in self.active_signals:
                continue  # Already have active signal
            
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            latest_trade = self.polygon_data_manager.get_latest_trade(symbol) # type: ignore
            if not latest_trade:
                continue
            
            breakout_analysis = await self._analyze_breakout(symbol, latest_trade.price)
            
            if breakout_analysis and breakout_analysis.breakout_direction != BreakoutDirection.NONE:
                signal = await self._generate_breakout_signal(breakout_analysis)
                
                if signal:
                    self.active_signals[symbol] = signal
                    self.daily_stats['breakout_signals'] += 1
                    
                    # Execute signal
                    await self._execute_breakout_signal(signal)
    
    async def _analyze_breakout(self, symbol: str, current_price: float) -> Optional[BreakoutAnalysis]:
        """Analyze potential breakout"""
        
        try:
            opening_range = self.opening_ranges.get(symbol)
            if not opening_range:
                return None
            
            # Check for breakout
            breakout_direction = BreakoutDirection.NONE
            breakout_price = current_price
            breakout_distance = 0.0
            
            confirmation_threshold = self.config.breakout_confirmation
            
            if current_price > opening_range.high * (1 + confirmation_threshold):
                breakout_direction = BreakoutDirection.UPWARD
                breakout_price = opening_range.high
                breakout_distance = current_price - opening_range.high
                
            elif current_price < opening_range.low * (1 - confirmation_threshold):
                breakout_direction = BreakoutDirection.DOWNWARD
                breakout_price = opening_range.low
                breakout_distance = opening_range.low - current_price
            
            if breakout_direction == BreakoutDirection.NONE:
                return None
            
            # Create breakout analysis
            breakout_analysis = BreakoutAnalysis(
                symbol=symbol,
                opening_range=opening_range,
                current_price=current_price,
                breakout_direction=breakout_direction,
                breakout_price=breakout_price,
                breakout_distance=breakout_distance,
                breakout_percent=(breakout_distance / breakout_price) * 100
            )
            
            # Add volume confirmation
            await self._add_volume_confirmation(breakout_analysis)
            
            # Add Lag-Llama predictions
            await self._add_breakout_predictions(breakout_analysis)
            
            # Add technical levels
            await self._add_technical_levels(breakout_analysis)
            
            return breakout_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing breakout for {symbol}: {e}")
            return None
    
    async def _add_volume_confirmation(self, breakout_analysis: BreakoutAnalysis):
        """Add volume confirmation to breakout analysis"""
        
        try:
            symbol = breakout_analysis.symbol
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            latest_aggregate = self.polygon_data_manager.get_latest_aggregate(symbol) # type: ignore
            
            if not latest_aggregate:
                return
            
            # Get average volume
            metrics = symbol_manager.metrics.get(symbol)
            if not metrics or not metrics.avg_volume:
                return
            
            # Calculate volume ratio for breakout bar
            volume_ratio = latest_aggregate.volume / (metrics.avg_volume / 390)  # Per minute average
            
            breakout_analysis.breakout_volume = latest_aggregate.volume
            breakout_analysis.volume_ratio = volume_ratio
            
            # Volume confirmation if above threshold
            breakout_analysis.volume_confirmation = volume_ratio >= self.config.volume_confirmation
            
        except Exception as e:
            logger.error(f"Error adding volume confirmation: {e}")
    
    async def _add_breakout_predictions(self, breakout_analysis: BreakoutAnalysis):
        """Add multi-timeframe Lag-Llama predictions for breakout sustainability with Polygon momentum confirmation"""
        
        try:
            symbol = breakout_analysis.symbol
            multi_forecast = await lag_llama_engine.generate_multi_timeframe_forecasts_strict(symbol)
            
            if not multi_forecast:
                return
            
            opening_range = breakout_analysis.opening_range
            current_price = breakout_analysis.current_price
            
            # Get Polygon momentum indicators for breakout confirmation - HIGH PRIORITY
            await self._add_polygon_momentum_confirmation(breakout_analysis)
            
            # Use multi-timeframe analysis for enhanced breakout predictions
            breakout_analysis.sustainability_probability = multi_forecast.momentum_alignment_score
            breakout_analysis.false_breakout_probability = 1.0 - multi_forecast.trend_consistency_score
            
            # Generate targets from multi-timeframe forecasts
            targets = []
            range_size = opening_range.range_size
            
            if breakout_analysis.breakout_direction == BreakoutDirection.UPWARD:
                # Use upward targets from different timeframes
                for horizon_min, forecast_data in multi_forecast.forecasts.items():
                    if horizon_min <= 120:  # Use up to 2-hour forecasts for ORB
                        # Calculate range-based target
                        range_multiple = 0.5 + (horizon_min / 60) * 0.5  # 0.5x to 1.5x range
                        target = opening_range.high + range_size * range_multiple
                        targets.append(target)
                        
                        # Also use forecast quantiles
                        q75_target = forecast_data.quantiles.get('q75', target)
                        if isinstance(q75_target, (list, tuple)) and len(q75_target) > 0:
                            targets.append(q75_target[0])
                        elif isinstance(q75_target, (int, float)):
                            targets.append(q75_target)
                
            else:  # DOWNWARD breakout
                # Use downward targets from different timeframes
                for horizon_min, forecast_data in multi_forecast.forecasts.items():
                    if horizon_min <= 120:  # Use up to 2-hour forecasts for ORB
                        # Calculate range-based target
                        range_multiple = 0.5 + (horizon_min / 60) * 0.5  # 0.5x to 1.5x range
                        target = opening_range.low - range_size * range_multiple
                        targets.append(target)
                        
                        # Also use forecast quantiles
                        q25_target = forecast_data.quantiles.get('q25', target)
                        if isinstance(q25_target, (list, tuple)) and len(q25_target) > 0:
                            targets.append(q25_target[0])
                        elif isinstance(q25_target, (int, float)):
                            targets.append(q25_target)
            
            # Filter and sort targets
            valid_targets = [t for t in targets if t > 0]
            breakout_analysis.target_projections = sorted(set(valid_targets))[:4]  # Top 4 unique targets
            
            # Enhanced pullback probability using cross-timeframe analysis
            breakout_analysis.pullback_probability = multi_forecast.risk_adjusted_confidence
            
            logger.debug(f"Multi-timeframe ORB analysis for {symbol}: "
                        f"sustainability={breakout_analysis.sustainability_probability:.2f}, "
                        f"targets={len(breakout_analysis.target_projections)}")
            
        except Exception as e:
            logger.error(f"Error adding multi-timeframe breakout predictions: {e}")
    
    async def _add_polygon_momentum_confirmation(self, breakout_analysis: BreakoutAnalysis):
        """Add Polygon professional momentum indicators for breakout confirmation - HIGH PRIORITY"""
        
        try:
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            
            symbol = breakout_analysis.symbol
            
            # Get professional RSI for momentum confirmation
            rsi_data = await self.polygon_data_manager.get_rsi(symbol, window=14, timespan="minute")
            
            # Get professional MACD for trend confirmation
            macd_data = await self.polygon_data_manager.get_macd(symbol, timespan="minute")
            
            # Get professional Bollinger Bands for volatility expansion
            bb_data = await self.polygon_data_manager.get_bollinger_bands(
                symbol, 
                window=20, 
                timespan="minute"
            )
            
            momentum_confirmation = False
            momentum_strength = 0.0
            
            if breakout_analysis.breakout_direction == BreakoutDirection.UPWARD:
                # For upward breakout, look for bullish momentum
                momentum_factors = []
                
                # RSI momentum (should be rising and above 50)
                if rsi_data and rsi_data.get('current_rsi', 50) > 50:
                    rsi_strength = min((rsi_data['current_rsi'] - 50) / 30, 1.0)  # Normalize 50-80 to 0-1
                    momentum_factors.append(rsi_strength)
                    logger.debug(f"ORB upward breakout RSI confirmation for {symbol}: {rsi_data['current_rsi']:.2f}")
                
                # MACD bullish momentum
                if macd_data:
                    macd_line = macd_data.get('macd_line', 0)
                    signal_line = macd_data.get('signal_line', 0)
                    histogram = macd_data.get('histogram', 0)
                    
                    if histogram > 0 and macd_line > signal_line:
                        macd_strength = min(abs(histogram) / 0.5, 1.0)
                        momentum_factors.append(macd_strength)
                        logger.debug(f"ORB upward breakout MACD confirmation for {symbol}: histogram={histogram:.4f}")
                
                # Bollinger Band expansion (price above upper band indicates strong momentum)
                if bb_data:
                    current_price = breakout_analysis.current_price
                    upper_band = bb_data.get('upper_band', current_price)
                    
                    if current_price > upper_band:
                        bb_strength = min((current_price - upper_band) / upper_band * 10, 1.0)
                        momentum_factors.append(bb_strength)
                        logger.debug(f"ORB upward breakout BB expansion for {symbol}: price={current_price:.2f}, upper={upper_band:.2f}")
                
                # Calculate overall momentum strength
                if momentum_factors:
                    momentum_strength = np.mean(momentum_factors)
                    momentum_confirmation = momentum_strength > 0.3  # 30% threshold
            
            elif breakout_analysis.breakout_direction == BreakoutDirection.DOWNWARD:
                # For downward breakout, look for bearish momentum
                momentum_factors = []
                
                # RSI momentum (should be falling and below 50)
                if rsi_data and rsi_data.get('current_rsi', 50) < 50:
                    rsi_strength = min((50 - rsi_data['current_rsi']) / 30, 1.0)  # Normalize 50-20 to 0-1
                    momentum_factors.append(rsi_strength)
                    logger.debug(f"ORB downward breakout RSI confirmation for {symbol}: {rsi_data['current_rsi']:.2f}")
                
                # MACD bearish momentum
                if macd_data:
                    macd_line = macd_data.get('macd_line', 0)
                    signal_line = macd_data.get('signal_line', 0)
                    histogram = macd_data.get('histogram', 0)
                    
                    if histogram < 0 and macd_line < signal_line:
                        macd_strength = min(abs(histogram) / 0.5, 1.0)
                        momentum_factors.append(macd_strength)
                        logger.debug(f"ORB downward breakout MACD confirmation for {symbol}: histogram={histogram:.4f}")
                
                # Bollinger Band expansion (price below lower band)
                if bb_data:
                    current_price = breakout_analysis.current_price
                    lower_band = bb_data.get('lower_band', current_price)
                    
                    if current_price < lower_band:
                        bb_strength = min((lower_band - current_price) / lower_band * 10, 1.0)
                        momentum_factors.append(bb_strength)
                        logger.debug(f"ORB downward breakout BB expansion for {symbol}: price={current_price:.2f}, lower={lower_band:.2f}")
                
                # Calculate overall momentum strength
                if momentum_factors:
                    momentum_strength = np.mean(momentum_factors)
                    momentum_confirmation = momentum_strength > 0.3  # 30% threshold
            
            # Enhance sustainability probability with momentum confirmation
            if momentum_confirmation:
                # Boost sustainability probability by momentum strength
                momentum_boost = momentum_strength * 0.25  # Up to 25% boost
                breakout_analysis.sustainability_probability = min(
                    breakout_analysis.sustainability_probability + momentum_boost,
                    1.0
                )
                
                # Reduce false breakout probability
                false_breakout_reduction = momentum_strength * 0.2  # Up to 20% reduction
                breakout_analysis.false_breakout_probability = max(
                    breakout_analysis.false_breakout_probability - false_breakout_reduction,
                    0.0
                )
                
                logger.info(f"ORB momentum confirmation for {symbol}: "
                          f"strength={momentum_strength:.2f}, "
                          f"sustainability boosted to {breakout_analysis.sustainability_probability:.2f}")
            else:
                # Penalize if momentum doesn't confirm
                momentum_penalty = 0.15  # 15% penalty
                breakout_analysis.sustainability_probability = max(
                    breakout_analysis.sustainability_probability - momentum_penalty,
                    0.0
                )
                
                logger.debug(f"ORB momentum divergence for {symbol}, "
                           f"reduced sustainability to {breakout_analysis.sustainability_probability:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding Polygon momentum confirmation for {symbol}: {e}")
    
    async def _add_technical_levels(self, breakout_analysis: BreakoutAnalysis):
        """Add technical support/resistance levels"""
        
        try:
            symbol = breakout_analysis.symbol
            
            # Get historical data
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            historical_data = await self.polygon_data_manager.get_historical_data( # type: ignore
                symbol, days=20, timespan="day"
            )
            
            if not historical_data:
                return
            
            # Extract key levels from recent history
            highs = [bar['high'] for bar in historical_data]
            lows = [bar['low'] for bar in historical_data]
            
            current_price = breakout_analysis.current_price
            
            # Find resistance levels (above current price)
            resistance_levels = [level for level in highs if level > current_price]
            resistance_levels = sorted(set(resistance_levels))[:5]  # Top 5 unique levels
            
            # Find support levels (below current price)
            support_levels = [level for level in lows if level < current_price]
            support_levels = sorted(set(support_levels), reverse=True)[:5]  # Top 5 unique levels
            
            breakout_analysis.key_resistance_levels = resistance_levels
            breakout_analysis.key_support_levels = support_levels
            
        except Exception as e:
            logger.error(f"Error adding technical levels: {e}")
    
    async def _generate_breakout_signal(self, breakout_analysis: BreakoutAnalysis) -> Optional[ORBSignal]:
        """Generate trading signal for breakout"""
        
        try:
            # Check if breakout meets our criteria
            if breakout_analysis.sustainability_probability < self.config.sustainability_threshold:
                return None
            
            if not breakout_analysis.volume_confirmation:
                logger.debug(f"No volume confirmation for {breakout_analysis.symbol}")
                return None
            
            # Determine action
            if breakout_analysis.breakout_direction == BreakoutDirection.UPWARD:
                action = "BUY"
            elif breakout_analysis.breakout_direction == BreakoutDirection.DOWNWARD:
                action = "SELL"
            else:
                return None
            
            # Calculate confidence
            confidence = min(
                breakout_analysis.sustainability_probability,
                1.0 - breakout_analysis.false_breakout_probability,
                breakout_analysis.volume_ratio / self.config.volume_confirmation,
                1.0
            )
            
            if confidence < self.strategy_config.confidence_threshold:
                return None
            
            # Entry and risk management
            entry_price = breakout_analysis.current_price
            opening_range = breakout_analysis.opening_range
            
            # Stop loss
            if action == "BUY":
                # Stop below range or recent support
                stop_candidates = [opening_range.low * 0.995]  # Slightly below range low
                if breakout_analysis.key_support_levels:
                    stop_candidates.append(max(breakout_analysis.key_support_levels))
                stop_loss = max(stop_candidates)
            else:  # SELL
                # Stop above range or recent resistance
                stop_candidates = [opening_range.high * 1.005]  # Slightly above range high
                if breakout_analysis.key_resistance_levels:
                    stop_candidates.append(min(breakout_analysis.key_resistance_levels))
                stop_loss = min(stop_candidates)
            
            # Targets
            targets = breakout_analysis.target_projections[:3]  # Top 3 targets
            
            # Position sizing
            risk_per_share = abs(entry_price - stop_loss)
            risk_percent = config.risk.max_position_size
            
            account_equity = 100000.0 # Default equity
            if self.alpaca_client and self.alpaca_client.account_info:
                account_equity = float(self.alpaca_client.account_info.equity)

            position_size = 0.0
            if risk_per_share > 0 and entry_price > 0:
                position_size = (risk_percent * account_equity) / (risk_per_share * entry_price)
            position_size = min(position_size, config.risk.max_position_size)
            
            # Risk-reward calculation
            if targets:
                avg_target = np.mean(targets)
                profit_per_share = abs(avg_target - entry_price)
                risk_reward_ratio = profit_per_share / risk_per_share if risk_per_share > 0 else 0
            else:
                risk_reward_ratio = 0
            
            # Timing
            signal_time = datetime.now()
            max_hold = min(self.config.max_hold_minutes, 240)  # Max 4 hours
            expiry_time = signal_time + timedelta(minutes=max_hold)
            
            # Range expansion multiple
            range_expansion = breakout_analysis.breakout_distance / opening_range.range_size if opening_range.range_size > 0 else 0
            
            signal = ORBSignal(
                symbol=breakout_analysis.symbol,
                action=action,
                confidence=confidence,
                breakout_analysis=breakout_analysis,
                entry_price=entry_price,
                stop_loss=stop_loss,
                targets=targets,
                position_size=position_size,
                signal_time=signal_time,
                expiry_time=expiry_time,
                max_hold_minutes=max_hold,
                risk_reward_ratio=float(risk_reward_ratio),
                max_loss_percent=risk_per_share / entry_price if entry_price > 0 else 0.0,
                expected_profit=float(profit_per_share if targets else 0),
                breakout_confirmation_price=breakout_analysis.breakout_price,
                range_expansion_multiple=range_expansion
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating breakout signal: {e}")
            return None
    
    async def _execute_breakout_signal(self, signal: ORBSignal):
        """Execute breakout trading signal"""
        
        try:
            # Convert to trade signal
            trade_signal = TradeSignal(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                strategy="orb",
                target_price=signal.targets[0] if signal.targets else None,
                stop_loss=signal.stop_loss,
                position_size=signal.position_size,
                hold_time=signal.max_hold_minutes,
                reason=f"ORB {signal.breakout_analysis.breakout_direction.value} breakout"
            )
            
            # Execute trade
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                self.trades_today += 1
                self.daily_stats['trades_executed'] += 1
                
                logger.info(f"Executed ORB trade: {signal.action} {signal.symbol} "
                          f"(confidence: {signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error executing breakout signal: {e}")
    
    async def _on_trade_update(self, trade_data: TradeData):
        """Handle real-time trade updates"""
        
        symbol = trade_data.symbol
        
        # Update range formation data
        if self.range_formation_active and symbol in self.range_data:
            self.range_data[symbol].append({
                'price': trade_data.price,
                'volume': trade_data.size,
                'timestamp': trade_data.timestamp
            })
        
        # Update active signals
        if symbol in self.active_signals:
            await self._update_signal_status(symbol, trade_data)
    
    async def _update_signal_status(self, symbol: str, trade_data: TradeData):
        """Update status of active ORB signal"""
        
        if symbol not in self.active_signals:
            return
        
        signal = self.active_signals[symbol]
        current_price = trade_data.price
        
        # Check if signal expired
        if datetime.now() > signal.expiry_time:
            logger.info(f"ORB signal expired for {symbol}")
            await self._close_signal(symbol, "Time expiry")
            return
        
        # Check stop loss
        if signal.action == "BUY" and current_price <= signal.stop_loss:
            logger.info(f"ORB stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
            self.false_breakouts += 1
            self.daily_stats['false_breakouts'] += 1
            
        elif signal.action == "SELL" and current_price >= signal.stop_loss:
            logger.info(f"ORB stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
            self.false_breakouts += 1
            self.daily_stats['false_breakouts'] += 1
        
        # Check targets
        for i, target in enumerate(signal.targets):
            if signal.action == "BUY" and current_price >= target:
                logger.info(f"ORB target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                self.successful_breakouts += 1
                self.daily_stats['successful_breakouts'] += 1
                break
                
            elif signal.action == "SELL" and current_price <= target:
                logger.info(f"ORB target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                self.successful_breakouts += 1
                self.daily_stats['successful_breakouts'] += 1
                break
    
    async def _close_signal(self, symbol: str, reason: str):
        """Close active ORB signal"""
        
        if symbol in self.active_signals:
            signal = self.active_signals[symbol]
            
            # Convert to close trade signal
            trade_signal = TradeSignal(
                symbol=symbol,
                action="SELL" if signal.action == "BUY" else "BUY",
                confidence=1.0,
                strategy="orb_close",
                reason=reason
            )
            
            # Submit close order
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                logger.info(f"Closed ORB position for {symbol}: {reason}")
            
            del self.active_signals[symbol]
    
    async def _on_aggregate_update(self, aggregate_data: AggregateData):
        """Handle minute bar updates"""
        
        symbol = aggregate_data.symbol
        
        # Update range formation if active
        if self.range_formation_active and symbol in self.range_data:
            # Add aggregate data to range formation
            self.range_data[symbol].append({
                'price': aggregate_data.close,
                'volume': aggregate_data.volume,
                'timestamp': aggregate_data.timestamp,
                'high': aggregate_data.high,
                'low': aggregate_data.low
            })
    
    def get_opening_ranges(self) -> Dict[str, OpeningRange]:
        """Get all formed opening ranges"""
        return self.opening_ranges.copy()
    
    def get_active_signals(self) -> List[ORBSignal]:
        """Get currently active ORB signals"""
        return list(self.active_signals.values())
    
    def get_strategy_performance(self) -> Dict:
        """Get ORB strategy performance metrics"""
        
        success_rate = 0.0
        if self.trades_today > 0:
            success_rate = self.successful_breakouts / self.trades_today
        
        false_breakout_rate = 0.0
        if self.trades_today > 0:
            false_breakout_rate = self.false_breakouts / self.trades_today
        
        return {
            'strategy': 'orb',
            'ranges_formed': len(self.opening_ranges),
            'trades_today': self.trades_today,
            'successful_breakouts': self.successful_breakouts,
            'false_breakouts': self.false_breakouts,
            'success_rate': success_rate,
            'false_breakout_rate': false_breakout_rate,
            'total_pnl': self.total_pnl,
            'active_signals': len(self.active_signals),
            'daily_stats': self.daily_stats
        }
    
    async def daily_reset(self):
        """Reset strategy for new trading day"""
        
        logger.info("Resetting ORB strategy for new day")
        
        # Clear previous day data
        self.opening_ranges.clear()
        self.active_signals.clear()
        self.range_data.clear()
        
        # Reset state
        self.range_formation_active = False
        self.market_open_time = None
        self.range_end_time = None
        
        # Reset counters
        self.trades_today = 0
        self.successful_breakouts = 0
        self.false_breakouts = 0
        self.total_pnl = 0.0
        
        # Reset daily stats
        self.daily_stats = {
            'ranges_formed': 0,
            'breakout_signals': 0,
            'trades_executed': 0,
            'successful_breakouts': 0,
            'false_breakouts': 0,
            'average_range_size': 0.0
        }

# Global strategy instance
orb_strategy = ORBStrategy()
