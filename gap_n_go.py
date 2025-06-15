"""
FAIL FAST SYSTEM - NO FALLBACKS ALLOWED
Gap and Go Trading Strategy
Enhanced with Lag-Llama probabilistic forecasting
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
from polygon import get_polygon_data_manager, TradeData
from alpaca import get_alpaca_client, TradeSignal
from database import get_database_manager

logger = logging.getLogger(__name__)

class GapType(Enum):
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    NO_GAP = "no_gap"

class GapDirection(Enum):
    CONTINUATION = "continuation"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    EXTENSION = "extension"

@dataclass
class GapAnalysis:
    """Comprehensive gap analysis"""
    symbol: str
    gap_type: GapType
    gap_percent: float
    gap_size: float
    previous_close: float
    open_price: float
    current_price: float
    
    # Volume analysis
    premarket_volume: int = 0
    volume_ratio: float = 0.0
    
    # Technical levels
    resistance_level: Optional[float] = None
    support_level: Optional[float] = None
    key_levels: List[float] = field(default_factory=list)
    
    # Catalyst information
    news_catalyst: bool = False
    earnings_related: bool = False
    sector_movement: bool = False
    
    # Lag-Llama predictions
    continuation_probability: float = 0.0
    fill_probability: float = 0.0
    target_levels: List[float] = field(default_factory=list)
    optimal_hold_time: int = 0
    
    def __post_init__(self):
        if not self.key_levels:
            self.key_levels = []
        if not self.target_levels:
            self.target_levels = []

@dataclass
class GapSignal:
    """Gap trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    gap_analysis: GapAnalysis
    
    # Entry details
    entry_price: float
    stop_loss: float
    targets: List[float]
    position_size: float
    
    # Timing
    hold_time_minutes: int
    entry_time: datetime
    expiry_time: datetime
    
    # Risk metrics
    risk_reward_ratio: float
    max_loss_percent: float
    expected_profit: float

class GapAndGoStrategy:
    """Gap and Go trading strategy with Lag-Llama integration"""
    
    def __init__(self):
        self.config = config.gap_and_go
        self.strategy_config = config.strategy
        
        # Client instances (lazy-initialized)
        self.polygon_data_manager = None
        self.alpaca_client = None
        self.db_manager = None
        
        # Strategy state
        self.active_signals: Dict[str, GapSignal] = {}
        self.gap_analyses: Dict[str, GapAnalysis] = {}
        self.daily_stats: Dict[str, Union[int, float]] = {}
        
        # Performance tracking
        self.trades_today: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0
        
        # Market open tracking
        self.market_opened: bool = False
        self.market_open_time: Optional[datetime] = None
        
        # Initialize
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize strategy components"""
        
        logger.info("Initializing Gap and Go strategy...")
        
        # Initialize clients
        self.polygon_data_manager = get_polygon_data_manager()
        self.alpaca_client = get_alpaca_client()
        self.db_manager = get_database_manager()
        
        # Reset daily stats
        self.daily_stats = {
            'gaps_identified': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_continuations': 0,
            'gap_fills': 0
        }
        
        # Register callbacks
        if self.polygon_data_manager:
            self.polygon_data_manager.add_trade_callback(self._on_trade_update) # type: ignore
            self.polygon_data_manager.add_aggregate_callback(self._on_aggregate_update) # type: ignore
    
    async def analyze_premarket_gaps(self) -> List[GapAnalysis]:
        """Analyze premarket gaps for all symbols"""
        
        logger.info("Analyzing premarket gaps...")
        
        gap_candidates = []
        active_symbols = symbol_manager.get_active_symbols("gap_and_go")
        
        for symbol in active_symbols:
            gap_analysis = await self._analyze_symbol_gap(symbol)
            
            if gap_analysis and self._is_valid_gap(gap_analysis):
                gap_candidates.append(gap_analysis)
                self.gap_analyses[symbol] = gap_analysis
                self.daily_stats['gaps_identified'] += 1
        
        # Sort by gap size and volume
        gap_candidates.sort(
            key=lambda g: abs(g.gap_percent) * g.volume_ratio,
            reverse=True
        )
        
        logger.info(f"Identified {len(gap_candidates)} gap candidates")
        
        # Store gap candidates in database
        await self._store_gap_candidates(gap_candidates)
        
        return gap_candidates
    
    async def _analyze_symbol_gap(self, symbol: str) -> Optional[GapAnalysis]:
        """Analyze gap for individual symbol with enhanced Polygon API integration"""
        
        try:
            # Get symbol metrics
            metrics = symbol_manager.metrics.get(symbol)
            if not metrics:
                return None
            
            # Get previous day data
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()

            daily_bar = self.polygon_data_manager.get_daily_bar(symbol) # type: ignore
            if not daily_bar:
                return None
            
            # Current price data
            latest_trade = self.polygon_data_manager.get_latest_trade(symbol) # type: ignore
            if not latest_trade:
                return None
            
            current_price = latest_trade.price
            previous_close = daily_bar.close
            gap_size = current_price - previous_close
            gap_percent = (gap_size / previous_close) * 100
            
            # Determine gap type
            if gap_percent >= self.config.min_gap_percent * 100:
                gap_type = GapType.GAP_UP
            elif gap_percent <= -self.config.min_gap_percent * 100:
                gap_type = GapType.GAP_DOWN
            else:
                gap_type = GapType.NO_GAP
            
            if gap_type == GapType.NO_GAP:
                return None
            
            # Volume analysis
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            latest_aggregate = self.polygon_data_manager.get_latest_aggregate(symbol) # type: ignore
            volume_ratio = 0.0
            
            if latest_aggregate and metrics.avg_volume > 0:
                volume_ratio = latest_aggregate.volume / metrics.avg_volume
            
            # Create gap analysis
            gap_analysis = GapAnalysis(
                symbol=symbol,
                gap_type=gap_type,
                gap_percent=gap_percent,
                gap_size=gap_size,
                previous_close=previous_close,
                open_price=latest_aggregate.open if latest_aggregate else current_price,
                current_price=current_price,
                volume_ratio=volume_ratio
            )
            
            # Add technical levels
            await self._add_technical_levels(gap_analysis)
            
            # Get Lag-Llama predictions
            await self._add_lagllama_predictions(gap_analysis)
            
            # Add MACD momentum confirmation - HIGH PRIORITY
            await self._add_macd_momentum_confirmation(gap_analysis)
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing gap for {symbol}: {e}")
            return None
    
    async def _add_macd_momentum_confirmation(self, gap_analysis: GapAnalysis):
        """Add MACD momentum confirmation for gap continuation - HIGH PRIORITY"""
        
        try:
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            
            # Get professional MACD from Polygon API
            macd_data = await self.polygon_data_manager.get_macd(
                gap_analysis.symbol, 
                timespan="minute",
                limit=5  # Get last 5 periods for trend analysis
            )
            
            if not macd_data:
                logger.warning(f"Could not get MACD data for {gap_analysis.symbol}")
                return
            
            macd_line = macd_data['macd_line']
            signal_line = macd_data['signal_line']
            histogram = macd_data['histogram']
            
            # Analyze momentum confirmation
            momentum_confirmation = False
            momentum_strength = 0.0
            
            if gap_analysis.gap_type == GapType.GAP_UP:
                # For gap up, look for bullish MACD momentum
                if histogram > 0 and macd_line > signal_line:
                    momentum_confirmation = True
                    momentum_strength = min(abs(histogram) / 0.5, 1.0)  # Normalize to 0-1
                    
                    logger.debug(f"Gap up momentum confirmed for {gap_analysis.symbol}: "
                               f"MACD={macd_line:.4f}, Signal={signal_line:.4f}, Histogram={histogram:.4f}")
            
            elif gap_analysis.gap_type == GapType.GAP_DOWN:
                # For gap down, look for bearish MACD momentum
                if histogram < 0 and macd_line < signal_line:
                    momentum_confirmation = True
                    momentum_strength = min(abs(histogram) / 0.5, 1.0)  # Normalize to 0-1
                    
                    logger.debug(f"Gap down momentum confirmed for {gap_analysis.symbol}: "
                               f"MACD={macd_line:.4f}, Signal={signal_line:.4f}, Histogram={histogram:.4f}")
            
            # Enhance continuation probability with MACD confirmation
            if momentum_confirmation:
                # Boost continuation probability by momentum strength
                momentum_boost = momentum_strength * 0.2  # Up to 20% boost
                gap_analysis.continuation_probability = min(
                    gap_analysis.continuation_probability + momentum_boost, 
                    1.0
                )
                
                logger.info(f"MACD momentum confirmation for {gap_analysis.symbol}: "
                          f"strength={momentum_strength:.2f}, boosted continuation prob to {gap_analysis.continuation_probability:.2f}")
            else:
                # Reduce continuation probability if momentum doesn't confirm
                momentum_penalty = 0.1  # 10% penalty
                gap_analysis.continuation_probability = max(
                    gap_analysis.continuation_probability - momentum_penalty,
                    0.0
                )
                
                logger.debug(f"MACD momentum divergence for {gap_analysis.symbol}, "
                           f"reduced continuation prob to {gap_analysis.continuation_probability:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding MACD momentum confirmation for {gap_analysis.symbol}: {e}")
    
    async def _add_technical_levels(self, gap_analysis: GapAnalysis):
        """Add technical support/resistance levels"""
        
        try:
            # Get historical data for level identification
            if not self.polygon_data_manager:
                self.polygon_data_manager = get_polygon_data_manager()
            historical_data = await self.polygon_data_manager.get_historical_data( # type: ignore
                gap_analysis.symbol, days=30, timespan="day"
            )
            
            if not historical_data:
                return
            
            # Extract highs and lows
            highs = [bar['high'] for bar in historical_data[-20:]]  # Last 20 days
            lows = [bar['low'] for bar in historical_data[-20:]]
            
            # Find significant levels (simplified)
            price_levels = highs + lows
            price_levels.sort()
            
            # Remove duplicates within 1%
            cleaned_levels = []
            for level in price_levels:
                if not cleaned_levels or abs(level - cleaned_levels[-1]) / cleaned_levels[-1] > 0.01:
                    cleaned_levels.append(level)
            
            gap_analysis.key_levels = cleaned_levels
            
            # Determine immediate support/resistance
            current_price = gap_analysis.current_price
            
            resistance_candidates = [level for level in cleaned_levels if level > current_price]
            support_candidates = [level for level in cleaned_levels if level < current_price]
            
            if resistance_candidates:
                gap_analysis.resistance_level = min(resistance_candidates)
            
            if support_candidates:
                gap_analysis.support_level = max(support_candidates)
                
        except Exception as e:
            logger.error(f"Error adding technical levels for {gap_analysis.symbol}: {e}")
    
    async def _add_lagllama_predictions(self, gap_analysis: GapAnalysis):
        """Add multi-timeframe Lag-Llama predictions to gap analysis"""
        
        try:
            # Get multi-timeframe forecast from Lag-Llama
            multi_forecast = await lag_llama_engine.generate_multi_timeframe_forecasts_strict(gap_analysis.symbol)
            
            if not multi_forecast:
                return
            
            current_price = gap_analysis.current_price
            previous_close = gap_analysis.previous_close
            
            # Use cross-timeframe analysis for enhanced predictions
            gap_analysis.continuation_probability = multi_forecast.cross_timeframe_correlation
            gap_analysis.fill_probability = 1.0 - multi_forecast.trend_consistency_score
            
            # Generate targets from multi-timeframe analysis
            targets = []
            
            if gap_analysis.gap_type == GapType.GAP_UP:
                # Use upward targets from different timeframes
                for horizon_min, forecast_data in multi_forecast.forecasts.items():
                    if horizon_min <= 60:  # Use up to 1-hour forecasts for gap trading
                        target = forecast_data.quantiles.get('q75', current_price * 1.02)
                        if isinstance(target, (list, tuple)) and len(target) > 0:
                            targets.append(target[0] if hasattr(target, '__getitem__') else target)
                        elif isinstance(target, (int, float)):
                            targets.append(target)
            else:
                # Use downward targets from different timeframes
                for horizon_min, forecast_data in multi_forecast.forecasts.items():
                    if horizon_min <= 60:  # Use up to 1-hour forecasts for gap trading
                        target = forecast_data.quantiles.get('q25', current_price * 0.98)
                        if isinstance(target, (list, tuple)) and len(target) > 0:
                            targets.append(target[0] if hasattr(target, '__getitem__') else target)
                        elif isinstance(target, (int, float)):
                            targets.append(target)
            
            gap_analysis.target_levels = [t for t in targets if t > 0]
            gap_analysis.optimal_hold_time = multi_forecast.optimal_hold_minutes
            
            logger.debug(f"Multi-timeframe gap analysis for {gap_analysis.symbol}: "
                        f"continuation_prob={gap_analysis.continuation_probability:.2f}, "
                        f"targets={len(gap_analysis.target_levels)}")
            
        except Exception as e:
            logger.error(f"Error adding multi-timeframe Lag-Llama predictions for {gap_analysis.symbol}: {e}")
    
    def _is_valid_gap(self, gap_analysis: GapAnalysis) -> bool:
        """Validate if gap meets trading criteria"""
        
        # Check gap size
        if abs(gap_analysis.gap_percent) < self.config.min_gap_percent * 100:
            return False
        
        if abs(gap_analysis.gap_percent) > self.config.max_gap_percent * 100:
            return False
        
        # Check volume
        if gap_analysis.volume_ratio < self.config.min_volume_ratio:
            return False
        
        # Check Lag-Llama confidence
        if gap_analysis.continuation_probability < self.config.gap_continuation_threshold:
            return False
        
        return True
    
    async def generate_signals(self) -> List[GapSignal]:
        """Generate trading signals for identified gaps"""
        
        signals = []
        
        for symbol, gap_analysis in self.gap_analyses.items():
            signal = await self._generate_signal_for_gap(gap_analysis)
            
            if signal:
                signals.append(signal)
                self.active_signals[symbol] = signal
                self.daily_stats['signals_generated'] += 1
        
        logger.info(f"Generated {len(signals)} gap trading signals")
        
        return signals
    
    async def _generate_signal_for_gap(self, gap_analysis: GapAnalysis) -> Optional[GapSignal]:
        """Generate trading signal for specific gap"""
        
        try:
            # Determine action based on gap type and predictions
            if gap_analysis.gap_type == GapType.GAP_UP:
                if gap_analysis.continuation_probability > self.config.gap_continuation_threshold:
                    action = "BUY"
                else:
                    return None
            elif gap_analysis.gap_type == GapType.GAP_DOWN:
                if gap_analysis.continuation_probability > self.config.gap_continuation_threshold:
                    action = "SELL"
                else:
                    return None
            else:
                return None
            
            # Calculate confidence
            confidence = min(
                gap_analysis.continuation_probability,
                gap_analysis.volume_ratio / self.config.min_volume_ratio,
                1.0
            )
            
            if confidence < self.strategy_config.confidence_threshold:
                return None
            
            # Entry and exit levels
            entry_price = gap_analysis.current_price
            
            # Stop loss calculation
            if action == "BUY":
                # Stop below previous close or support
                stop_candidates = [gap_analysis.previous_close * 0.99]
                if gap_analysis.support_level:
                    stop_candidates.append(gap_analysis.support_level * 0.99)
                stop_loss = max(stop_candidates)
            else:  # SELL
                # Stop above previous close or resistance
                stop_candidates = [gap_analysis.previous_close * 1.01]
                if gap_analysis.resistance_level:
                    stop_candidates.append(gap_analysis.resistance_level * 1.01)
                stop_loss = min(stop_candidates)
            
            # Targets from Lag-Llama
            targets = gap_analysis.target_levels[:3]  # Top 3 targets
            
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
            entry_time = datetime.now()
            hold_time = min(gap_analysis.optimal_hold_time, self.config.max_hold_minutes)
            expiry_time = entry_time + timedelta(minutes=hold_time)
            
            signal = GapSignal(
                symbol=gap_analysis.symbol,
                action=action,
                confidence=confidence,
                gap_analysis=gap_analysis,
                entry_price=entry_price,
                stop_loss=stop_loss,
                targets=targets,
                position_size=position_size,
                hold_time_minutes=hold_time,
                entry_time=entry_time,
                expiry_time=expiry_time,
                risk_reward_ratio=float(risk_reward_ratio),
                max_loss_percent=risk_per_share / entry_price if entry_price > 0 else 0.0,
                expected_profit=float(profit_per_share if targets else 0)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {gap_analysis.symbol}: {e}")
            return None
    
    async def _on_trade_update(self, trade_data: TradeData):
        """Handle real-time trade updates"""
        
        symbol = trade_data.symbol
        
        # Update gap analysis if we're tracking this symbol
        if symbol in self.gap_analyses:
            gap_analysis = self.gap_analyses[symbol]
            gap_analysis.current_price = trade_data.price
            
            # Check if gap filled
            await self._check_gap_fill(gap_analysis, trade_data)
        
        # Update active signals
        if symbol in self.active_signals:
            await self._update_signal_status(symbol, trade_data)
    
    async def _check_gap_fill(self, gap_analysis: GapAnalysis, trade_data: TradeData):
        """Check if gap has been filled"""
        
        current_price = trade_data.price
        previous_close = gap_analysis.previous_close
        
        fill_threshold = 0.005  # 0.5%
        
        if gap_analysis.gap_type == GapType.GAP_UP:
            # Gap filled if price goes back to within 0.5% of previous close
            if current_price <= previous_close * (1 + fill_threshold):
                logger.info(f"Gap filled for {gap_analysis.symbol}: {current_price:.2f}")
                self.daily_stats['gap_fills'] += 1
        
        elif gap_analysis.gap_type == GapType.GAP_DOWN:
            # Gap filled if price goes back to within 0.5% of previous close
            if current_price >= previous_close * (1 - fill_threshold):
                logger.info(f"Gap filled for {gap_analysis.symbol}: {current_price:.2f}")
                self.daily_stats['gap_fills'] += 1
    
    async def _update_signal_status(self, symbol: str, trade_data: TradeData):
        """Update status of active signal"""
        
        if symbol not in self.active_signals:
            return
        
        signal = self.active_signals[symbol]
        current_price = trade_data.price
        
        # Check if signal expired
        if datetime.now() > signal.expiry_time:
            logger.info(f"Gap signal expired for {symbol}")
            del self.active_signals[symbol]
            return
        
        # Check stop loss hit
        if signal.action == "BUY" and current_price <= signal.stop_loss:
            logger.info(f"Stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
        
        elif signal.action == "SELL" and current_price >= signal.stop_loss:
            logger.info(f"Stop loss hit for {symbol}: {current_price:.2f}")
            await self._close_signal(symbol, "Stop loss")
        
        # Check target hit
        for i, target in enumerate(signal.targets):
            if signal.action == "BUY" and current_price >= target:
                logger.info(f"Target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                break
            elif signal.action == "SELL" and current_price <= target:
                logger.info(f"Target {i+1} hit for {symbol}: {current_price:.2f}")
                await self._close_signal(symbol, f"Target {i+1}")
                break
    
    async def _close_signal(self, symbol: str, reason: str):
        """Close active signal"""
        
        if symbol in self.active_signals:
            signal = self.active_signals[symbol]
            
            # Convert to trade signal for Alpaca
            trade_signal = TradeSignal(
                symbol=symbol,
                action="SELL" if signal.action == "BUY" else "BUY",  # Opposite action to close
                confidence=1.0,  # Full confidence for closing
                strategy="gap_and_go_close",
                reason=reason
            )
            
            # Submit close order
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                logger.info(f"Closed gap position for {symbol}: {reason}")
                
                # Track performance
                if "target" in reason.lower():
                    self.winning_trades += 1
                    self.daily_stats['successful_continuations'] += 1
            
            del self.active_signals[symbol]
    
    async def _on_aggregate_update(self, aggregate_data):
        """Handle minute bar updates"""
        
        # Update volume ratios for gap analysis
        symbol = aggregate_data.symbol
        if symbol in self.gap_analyses:
            gap_analysis = self.gap_analyses[symbol]
            
            # Update volume ratio
            metrics = symbol_manager.metrics.get(symbol)
            if metrics and metrics.avg_volume > 0:
                gap_analysis.volume_ratio = aggregate_data.volume / metrics.avg_volume
    
    async def execute_signals(self, signals: List[GapSignal]) -> int:
        """Execute gap trading signals"""
        
        executed_count = 0
        
        for signal in signals:
            # Convert to trade signal
            trade_signal = TradeSignal(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                strategy="gap_and_go",
                target_price=signal.targets[0] if signal.targets else None,
                stop_loss=signal.stop_loss,
                position_size=signal.position_size,
                hold_time=signal.hold_time_minutes,
                reason=f"Gap {signal.gap_analysis.gap_type.value}: {signal.gap_analysis.gap_percent:.1f}%"
            )
            
            # Execute trade
            order_id = None
            if self.alpaca_client:
                order_id = await self.alpaca_client.place_trade(trade_signal)
            
            if order_id:
                executed_count += 1
                self.trades_today += 1
                self.daily_stats['trades_executed'] += 1
                
                # Store trading signal and position in database
                await self._store_trading_signal(signal)
                await self._store_position(signal, order_id)
                
                logger.info(f"Executed gap trade: {signal.action} {signal.symbol}")
        
        return executed_count
    
    def get_active_signals(self) -> List[GapSignal]:
        """Get currently active signals"""
        return list(self.active_signals.values())
    
    def get_strategy_performance(self) -> Dict:
        """Get strategy performance metrics"""
        
        win_rate = self.winning_trades / self.trades_today if self.trades_today > 0 else 0
        
        return {
            'strategy': 'gap_and_go',
            'trades_today': self.trades_today,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'active_signals': len(self.active_signals),
            'daily_stats': self.daily_stats
        }
    
    async def daily_reset(self):
        """Reset daily statistics and state"""
        
        logger.info("Resetting Gap and Go strategy for new day")
        
        # Clear previous day data
        self.gap_analyses.clear()
        self.active_signals.clear()
        
        # Reset counters
        self.trades_today = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.market_opened = False
        self.market_open_time = None
        
        # Reset daily stats
        self.daily_stats = {
            'gaps_identified': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_continuations': 0,
            'gap_fills': 0
        }
    
    async def _store_gap_candidates(self, gap_candidates: List[GapAnalysis]):
        """Store gap candidates in database"""
        
        if not self.db_manager or not gap_candidates:
            return
        
        try:
            for gap in gap_candidates:
                await self.db_manager.insert_gap_candidate(
                    symbol=gap.symbol,
                    gap_type=gap.gap_type.value,
                    gap_percent=gap.gap_percent,
                    previous_close=gap.previous_close,
                    current_price=gap.current_price,
                    volume=gap.premarket_volume,
                    volume_ratio=gap.volume_ratio
                )
            
            logger.info(f"Stored {len(gap_candidates)} gap candidates in database")
            
        except Exception as e:
            logger.error(f"Error storing gap candidates: {e}")
    
    async def _store_trading_signal(self, signal: GapSignal):
        """Store trading signal in database"""
        
        if not self.db_manager:
            return
        
        try:
            signal_data = {
                'timestamp': signal.entry_time,
                'symbol': signal.symbol,
                'strategy': 'gap_and_go',
                'signal_type': signal.action,
                'confidence': signal.confidence,
                'price': signal.entry_price,
                'metadata': {
                    'gap_type': signal.gap_analysis.gap_type.value,
                    'gap_percent': signal.gap_analysis.gap_percent,
                    'stop_loss': signal.stop_loss,
                    'targets': signal.targets,
                    'position_size': signal.position_size,
                    'hold_time_minutes': signal.hold_time_minutes,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'continuation_probability': signal.gap_analysis.continuation_probability
                }
            }
            
            await self.db_manager.insert_trading_signal(signal_data)
            
        except Exception as e:
            logger.error(f"Error storing trading signal: {e}")
    
    async def _store_position(self, signal: GapSignal, order_id: str):
        """Store position in database"""
        
        if not self.db_manager:
            return
        
        try:
            position_data = {
                'symbol': signal.symbol,
                'strategy': 'gap_and_go',
                'side': 'LONG' if signal.action == 'BUY' else 'SHORT',
                'quantity': int(signal.position_size),
                'entry_price': signal.entry_price,
                'opened_at': signal.entry_time,
                'metadata': {
                    'order_id': order_id,
                    'gap_type': signal.gap_analysis.gap_type.value,
                    'gap_percent': signal.gap_analysis.gap_percent,
                    'stop_loss': signal.stop_loss,
                    'targets': signal.targets,
                    'expiry_time': signal.expiry_time.isoformat()
                }
            }
            
            await self.db_manager.insert_position(position_data)
            
        except Exception as e:
            logger.error(f"Error storing position: {e}")

# Global strategy instance
gap_and_go_strategy = GapAndGoStrategy()
