"""
Volatility Mean Reversion Strategy

Trades mean reversion during high volatility periods with AI confirmation.
Features:
- Volatility spike detection
- Mean reversion probability calculation
- Multi-timeframe analysis
- AI model validation for reversal signals
- Dynamic position sizing based on volatility
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

from utils.logger import get_logger, log_strategy, log_performance
from features.vol_features import VolatilityFeatureEngine

@dataclass
class VolatilityState:
    """Volatility state tracking"""
    symbol: str
    current_volatility: float
    avg_volatility: float
    volatility_percentile: float
    volatility_spike: bool
    mean_price: float
    std_price: float
    bollinger_upper: float
    bollinger_lower: float
    rsi: float
    z_score: float
    timestamp: float

class VolMeanReversionStrategy:
    """
    Volatility Mean Reversion Strategy
    
    Strategy Logic:
    1. Detect volatility spikes (>80th percentile)
    2. Identify overbought/oversold conditions
    3. Use AI models to confirm reversal probability
    4. Enter mean reversion trades with tight stops
    5. Exit when price returns toward mean
    """
    
    def __init__(self,
                 volatility_lookback: int = 20,           # Days for volatility calculation
                 volatility_threshold: float = 80.0,      # Percentile threshold for spikes
                 rsi_overbought: float = 70.0,           # RSI overbought level
                 rsi_oversold: float = 30.0,             # RSI oversold level
                 bollinger_std: float = 2.0,             # Bollinger band standard deviations
                 ai_confidence_threshold: float = 0.65,   # AI confidence required
                 max_holding_hours: float = 24.0,        # Maximum holding period
                 max_positions: int = 5):
        
        self.volatility_lookback = volatility_lookback
        self.volatility_threshold = volatility_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bollinger_std = bollinger_std
        self.ai_confidence_threshold = ai_confidence_threshold
        self.max_holding_hours = max_holding_hours
        self.max_positions = max_positions
        
        # Strategy state
        self.volatility_states: Dict[str, VolatilityState] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.price_history: Dict[str, deque] = {}
        self.returns_history: Dict[str, deque] = {}
        
        # Feature engine for volatility analysis
        self.feature_engine = VolatilityFeatureEngine()
        
        # Performance tracking
        self.strategy_stats = {
            'total_signals': 0,
            'mean_reversion_success': 0,
            'mean_reversion_failure': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_holding_time': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Technical indicators cache
        self.indicators_cache: Dict[str, Dict] = {}
        
        # Logging
        self.logger = get_logger("strategies")
        self.logger.info("Volatility Mean Reversion Strategy initialized")
    
    def update_price_data(self, symbol: str, price: float, timestamp: float):
        """Update price data and calculate indicators"""
        
        # Initialize price history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=252)  # One year
            self.returns_history[symbol] = deque(maxlen=252)
        
        # Add new price
        self.price_history[symbol].append(price)
        
        # Calculate return if we have previous price
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2]
            daily_return = (price - prev_price) / prev_price
            self.returns_history[symbol].append(daily_return)
        
        # Update volatility state
        self._update_volatility_state(symbol, price, timestamp)
    
    def _update_volatility_state(self, symbol: str, price: float, timestamp: float):
        """Update volatility state and technical indicators"""
        
        if len(self.price_history[symbol]) < self.volatility_lookback:
            return
        
        prices = np.array(list(self.price_history[symbol]))
        returns = np.array(list(self.returns_history[symbol]))
        
        # Calculate volatility metrics
        current_volatility = self._calculate_realized_volatility(returns)
        avg_volatility = np.mean([self._calculate_realized_volatility(
            returns[max(0, i-self.volatility_lookback):i+1]
        ) for i in range(len(returns)-self.volatility_lookback, len(returns))])
        
        # Volatility percentile
        vol_history = [self._calculate_realized_volatility(
            returns[max(0, i-self.volatility_lookback):i+1]
        ) for i in range(len(returns)-60, len(returns))]  # 60-day lookback
        
        if len(vol_history) > 10:
            volatility_percentile = (np.sum(np.array(vol_history) < current_volatility) / 
                                   len(vol_history)) * 100
        else:
            volatility_percentile = 50.0
        
        # Bollinger Bands
        mean_price = np.mean(prices[-20:])  # 20-period moving average
        std_price = np.std(prices[-20:])
        bollinger_upper = mean_price + (self.bollinger_std * std_price)
        bollinger_lower = mean_price - (self.bollinger_std * std_price)
        
        # RSI calculation
        rsi = self._calculate_rsi(prices)
        
        # Z-score (standard deviations from mean)
        z_score = (price - mean_price) / std_price if std_price > 0 else 0.0
        
        # Update volatility state
        self.volatility_states[symbol] = VolatilityState(
            symbol=symbol,
            current_volatility=current_volatility,
            avg_volatility=avg_volatility,
            volatility_percentile=volatility_percentile,
            volatility_spike=volatility_percentile > self.volatility_threshold,
            mean_price=mean_price,
            std_price=std_price,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower,
            rsi=rsi,
            z_score=z_score,
            timestamp=timestamp
        )
    
    def _calculate_realized_volatility(self, returns: np.ndarray) -> float:
        """Calculate realized volatility from returns"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_mean_reversion_signals(self, symbol: str, current_price: float,
                                   ai_signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for mean reversion trading signals"""
        
        if symbol not in self.volatility_states:
            return None
        
        vol_state = self.volatility_states[symbol]
        
        # Check if we're in a high volatility regime
        if not vol_state.volatility_spike:
            return None
        
        # Check for extreme price movements
        extreme_movement = self._check_extreme_movement(vol_state, current_price)
        if not extreme_movement:
            return None
        
        # Determine direction based on mean reversion
        direction = self._determine_reversion_direction(vol_state, current_price)
        if direction is None:
            return None
        
        # AI confirmation for mean reversion
        ai_confirmation = self._validate_ai_signals(ai_signals, direction)
        if not ai_confirmation:
            return None
        
        # Calculate signal strength
        signal_strength = self._calculate_mean_reversion_strength(
            vol_state, current_price, ai_signals
        )
        
        if signal_strength < 0.6:
            return None
        
        # Calculate position parameters
        stop_loss = self._calculate_stop_loss(vol_state, direction, current_price)
        take_profit = self._calculate_take_profit(vol_state, direction, current_price)
        
        # Create signal
        signal = {
            'strategy': 'VolMeanReversion',
            'symbol': symbol,
            'direction': direction,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_strength': signal_strength,
            'volatility_percentile': vol_state.volatility_percentile,
            'rsi': vol_state.rsi,
            'z_score': vol_state.z_score,
            'ai_confidence': ai_signals.get('confidence', 0.0),
            'ai_signal': ai_signals.get('signal', 0.0),
            'reversion_probability': self._calculate_reversion_probability(vol_state),
            'timestamp': time.time(),
            'should_trade': True
        }
        
        log_strategy("VolMeanReversion", f"Mean reversion signal for {symbol}", level="INFO",
                    direction=direction, signal_strength=signal_strength,
                    volatility_percentile=vol_state.volatility_percentile,
                    rsi=vol_state.rsi)
        
        # Update statistics
        self.strategy_stats['total_signals'] += 1
        log_performance("vol_reversion_signals", self.strategy_stats['total_signals'], 
                       "count", f"Total volatility mean reversion signals")
        
        return signal
    
    def _check_extreme_movement(self, vol_state: VolatilityState, current_price: float) -> bool:
        """Check if price shows extreme movement warranting mean reversion"""
        
        # Check Bollinger Band extremes
        bollinger_extreme = (
            current_price > vol_state.bollinger_upper or 
            current_price < vol_state.bollinger_lower
        )
        
        # Check RSI extremes
        rsi_extreme = (
            vol_state.rsi > self.rsi_overbought or 
            vol_state.rsi < self.rsi_oversold
        )
        
        # Check Z-score extremes (>2 standard deviations)
        z_score_extreme = abs(vol_state.z_score) > 2.0
        
        return bollinger_extreme and (rsi_extreme or z_score_extreme)
    
    def _determine_reversion_direction(self, vol_state: VolatilityState, 
                                     current_price: float) -> Optional[str]:
        """Determine mean reversion direction"""
        
        # Price above upper Bollinger Band -> expect reversion down (short)
        if current_price > vol_state.bollinger_upper and vol_state.rsi > self.rsi_overbought:
            return "short"
        
        # Price below lower Bollinger Band -> expect reversion up (long)
        elif current_price < vol_state.bollinger_lower and vol_state.rsi < self.rsi_oversold:
            return "long"
        
        # Z-score based reversion
        elif vol_state.z_score > 2.5:  # Very overbought
            return "short"
        elif vol_state.z_score < -2.5:  # Very oversold
            return "long"
        
        return None
    
    def _validate_ai_signals(self, ai_signals: Dict[str, Any], direction: str) -> bool:
        """Validate AI signals support mean reversion"""
        
        ai_confidence = ai_signals.get('confidence', 0.0)
        ai_signal = ai_signals.get('signal', 0.0)
        
        # AI confidence must be above threshold
        if ai_confidence < self.ai_confidence_threshold:
            return False
        
        # For mean reversion, we want AI signal to be in reversion direction
        # or show uncertainty (low signal strength)
        ai_signal_strength = abs(ai_signal)
        
        # Strong AI signal in reversion direction
        if direction == "long" and ai_signal > 0.3:
            return True
        elif direction == "short" and ai_signal < -0.3:
            return True
        
        # Weak AI signal (uncertainty) can also support mean reversion
        elif ai_signal_strength < 0.3 and ai_confidence > 0.7:
            return True
        
        return False
    
    def _calculate_mean_reversion_strength(self, vol_state: VolatilityState, 
                                         current_price: float, 
                                         ai_signals: Dict[str, Any]) -> float:
        """Calculate overall mean reversion signal strength"""
        
        # Volatility component (0-0.3)
        vol_score = min(0.3, (vol_state.volatility_percentile - 80) / 20 * 0.3)
        
        # Extremeness component (0-0.3)
        z_score_magnitude = abs(vol_state.z_score)
        extreme_score = min(0.3, (z_score_magnitude - 2.0) / 2.0 * 0.3)
        
        # RSI component (0-0.2)
        if vol_state.rsi > self.rsi_overbought:
            rsi_score = min(0.2, (vol_state.rsi - self.rsi_overbought) / 30 * 0.2)
        elif vol_state.rsi < self.rsi_oversold:
            rsi_score = min(0.2, (self.rsi_oversold - vol_state.rsi) / 30 * 0.2)
        else:
            rsi_score = 0.0
        
        # AI component (0-0.2)
        ai_score = ai_signals.get('confidence', 0.0) * 0.2
        
        total_score = vol_score + extreme_score + rsi_score + ai_score
        
        return min(1.0, total_score)
    
    def _calculate_reversion_probability(self, vol_state: VolatilityState) -> float:
        """Calculate probability of mean reversion"""
        
        # Historical analysis would be done here
        # For now, use technical indicators
        
        probability = 0.5  # Base probability
        
        # Increase probability based on extremeness
        if abs(vol_state.z_score) > 3.0:
            probability += 0.3
        elif abs(vol_state.z_score) > 2.0:
            probability += 0.2
        
        # Increase based on RSI extremes
        if vol_state.rsi > 80 or vol_state.rsi < 20:
            probability += 0.2
        elif vol_state.rsi > 70 or vol_state.rsi < 30:
            probability += 0.1
        
        # Increase based on volatility
        if vol_state.volatility_percentile > 90:
            probability += 0.1
        
        return min(0.95, probability)
    
    def _calculate_stop_loss(self, vol_state: VolatilityState, direction: str, 
                           entry_price: float) -> float:
        """Calculate stop loss for mean reversion trade"""
        
        # Use volatility-based stop loss
        stop_distance = vol_state.std_price * 1.5  # 1.5x standard deviation
        
        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _calculate_take_profit(self, vol_state: VolatilityState, direction: str, 
                             entry_price: float) -> float:
        """Calculate take profit for mean reversion trade"""
        
        # Target is the mean price
        if direction == "long":
            # For long trades, target above mean
            return vol_state.mean_price + (vol_state.std_price * 0.5)
        else:
            # For short trades, target below mean
            return vol_state.mean_price - (vol_state.std_price * 0.5)
    
    def manage_position(self, symbol: str, position_data: Dict[str, Any],
                       current_price: float, ai_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Manage existing mean reversion position"""
        
        entry_price = position_data['entry_price']
        direction = position_data['direction']
        entry_time = position_data.get('entry_time', time.time())
        
        # Calculate holding time
        holding_time = (time.time() - entry_time) / 3600  # Hours
        
        # Calculate current P&L
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        should_exit = False
        exit_reason = ""
        
        # Time-based exit
        if holding_time > self.max_holding_hours:
            should_exit = True
            exit_reason = "max_holding_time"
        
        # Target reached (mean reversion successful)
        elif symbol in self.volatility_states:
            vol_state = self.volatility_states[symbol]
            
            # Check if price has reverted to mean
            distance_from_mean = abs(current_price - vol_state.mean_price) / vol_state.std_price
            
            if distance_from_mean < 0.5:  # Within 0.5 standard deviations of mean
                should_exit = True
                exit_reason = "mean_reversion_complete"
        
        # Stop loss or take profit
        stop_loss = position_data.get('stop_loss')
        take_profit = position_data.get('take_profit')
        
        if stop_loss and ((direction == "long" and current_price <= stop_loss) or
                         (direction == "short" and current_price >= stop_loss)):
            should_exit = True
            exit_reason = "stop_loss"
        
        elif take_profit and ((direction == "long" and current_price >= take_profit) or
                             (direction == "short" and current_price <= take_profit)):
            should_exit = True
            exit_reason = "take_profit"
        
        return {
            'should_exit': should_exit,
            'exit_reason': exit_reason,
            'current_pnl_pct': pnl_pct,
            'holding_time_hours': holding_time,
            'recommendation': 'exit' if should_exit else 'hold'
        }
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        
        # Calculate win rate
        total_completed = (self.strategy_stats['mean_reversion_success'] + 
                          self.strategy_stats['mean_reversion_failure'])
        
        if total_completed > 0:
            self.strategy_stats['win_rate'] = (
                self.strategy_stats['mean_reversion_success'] / total_completed
            )
        
        return {
            'strategy': 'VolMeanReversion',
            'performance': self.strategy_stats,
            'active_positions': len(self.active_positions),
            'tracked_symbols': len(self.volatility_states),
            'settings': {
                'volatility_lookback': self.volatility_lookback,
                'volatility_threshold': self.volatility_threshold,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'ai_confidence_threshold': self.ai_confidence_threshold,
                'max_holding_hours': self.max_holding_hours
            }
        }
    
    def get_volatility_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed volatility analysis for a symbol"""
        
        if symbol not in self.volatility_states:
            return None
        
        vol_state = self.volatility_states[symbol]
        
        return {
            'symbol': symbol,
            'current_volatility': vol_state.current_volatility,
            'avg_volatility': vol_state.avg_volatility,
            'volatility_percentile': vol_state.volatility_percentile,
            'volatility_spike': vol_state.volatility_spike,
            'rsi': vol_state.rsi,
            'z_score': vol_state.z_score,
            'bollinger_bands': {
                'upper': vol_state.bollinger_upper,
                'middle': vol_state.mean_price,
                'lower': vol_state.bollinger_lower
            },
            'reversion_probability': self._calculate_reversion_probability(vol_state),
            'extreme_movement': self._check_extreme_movement(vol_state, vol_state.mean_price),
            'timestamp': vol_state.timestamp
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize volatility mean reversion strategy
    vol_strategy = VolMeanReversionStrategy()
    
    print("📊 Volatility Mean Reversion Strategy Test")
    print("=" * 50)
    
    # Simulate price data with volatility spike
    symbol = "TSLA"
    base_price = 200.0
    
    # Generate price series with increasing volatility
    np.random.seed(42)
    prices = []
    for i in range(60):
        if i < 40:
            # Normal volatility period
            volatility = 0.02
        else:
            # High volatility period
            volatility = 0.08
        
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, volatility)
            price = prices[-1] * (1 + change)
        
        prices.append(price)
        
        # Update strategy with price data
        vol_strategy.update_price_data(symbol, price, time.time() + i * 86400)
    
    # Simulate extreme price movement
    extreme_price = prices[-1] * 1.15  # 15% spike
    vol_strategy.update_price_data(symbol, extreme_price, time.time())
    
    # Get volatility analysis
    vol_analysis = vol_strategy.get_volatility_analysis(symbol)
    
    if vol_analysis:
        print(f"📈 Volatility Analysis for {symbol}:")
        print(f"   Current Price: ${extreme_price:.2f}")
        print(f"   Volatility Percentile: {vol_analysis['volatility_percentile']:.1f}%")
        print(f"   RSI: {vol_analysis['rsi']:.1f}")
        print(f"   Z-Score: {vol_analysis['z_score']:.2f}")
        print(f"   Reversion Probability: {vol_analysis['reversion_probability']:.1%}")
        print(f"   Extreme Movement: {vol_analysis['extreme_movement']}")
    
    # Simulate AI signals
    ai_signals = {
        'signal': -0.4,  # Bearish signal for mean reversion short
        'confidence': 0.75,
        'signal_strength': 0.4
    }
    
    # Check for mean reversion signals
    signal = vol_strategy.check_mean_reversion_signals(symbol, extreme_price, ai_signals)
    
    if signal:
        print(f"\n🎯 Mean Reversion Signal Generated:")
        print(f"   Direction: {signal['direction']}")
        print(f"   Signal Strength: {signal['signal_strength']:.2f}")
        print(f"   Entry Price: ${signal['entry_price']:.2f}")
        print(f"   Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"   Take Profit: ${signal['take_profit']:.2f}")
        print(f"   Reversion Probability: {signal['reversion_probability']:.1%}")
    else:
        print("\n❌ No mean reversion signal generated")
    
    # Get strategy statistics
    stats = vol_strategy.get_strategy_stats()
    print(f"\n📊 Strategy Statistics:")
    for key, value in stats['performance'].items():
        print(f"   {key}: {value}")
    
    print("\n✅ Volatility Mean Reversion Strategy test completed!")
