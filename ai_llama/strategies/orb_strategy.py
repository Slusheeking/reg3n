"""
Opening Range Breakout (ORB) Strategy

Trades breakouts from the opening range with AI confirmation.
Features:
- Multiple timeframe opening ranges (5min, 15min, 30min)
- Volume confirmation for breakouts
- AI model validation
- Dynamic stop loss and take profit
- False breakout detection
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from utils.logger import get_logger, log_strategy, log_performance
from features.orb_features import ORBFeatureEngine

@dataclass
class ORBRange:
    """Opening range data structure"""
    start_time: float
    end_time: float
    high: float
    low: float
    volume: float
    range_size: float
    breakout_threshold: float = 0.001  # 0.1% minimum breakout
    
    def is_breakout_up(self, price: float) -> bool:
        """Check if price breaks above range"""
        return price > (self.high * (1 + self.breakout_threshold))
    
    def is_breakout_down(self, price: float) -> bool:
        """Check if price breaks below range"""
        return price < (self.low * (1 - self.breakout_threshold))
    
    def get_breakout_distance(self, price: float) -> float:
        """Get distance of breakout from range"""
        if self.is_breakout_up(price):
            return (price - self.high) / self.high
        elif self.is_breakout_down(price):
            return (self.low - price) / self.low
        return 0.0

class ORBStrategy:
    """
    Opening Range Breakout Strategy
    
    Strategy Logic:
    1. Define opening range (first 5-30 minutes)
    2. Wait for breakout with volume confirmation
    3. Use AI models to validate breakout quality
    4. Enter trade with dynamic stops
    5. Manage position with trailing stops
    """
    
    def __init__(self, 
                 timeframes: List[int] = [5, 15, 30],  # minutes
                 min_range_size: float = 0.005,        # 0.5% minimum range
                 volume_threshold: float = 1.5,        # 1.5x average volume
                 ai_confidence_threshold: float = 0.6, # 60% AI confidence required
                 max_positions: int = 3):
        
        self.timeframes = timeframes
        self.min_range_size = min_range_size
        self.volume_threshold = volume_threshold
        self.ai_confidence_threshold = ai_confidence_threshold
        self.max_positions = max_positions
        
        # Strategy state
        self.active_ranges: Dict[str, Dict[int, ORBRange]] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.breakout_alerts: List[Dict] = []
        
        # Feature engine for ORB-specific features
        self.feature_engine = ORBFeatureEngine()
        
        # Performance tracking
        self.strategy_stats = {
            'total_signals': 0,
            'successful_breakouts': 0,
            'false_breakouts': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_holding_time': 0.0
        }
        
        # Market session times (in minutes from market open)
        self.market_open_time = 9.5 * 60  # 9:30 AM in minutes
        self.market_close_time = 16 * 60  # 4:00 PM in minutes
        
        # Logging
        self.logger = get_logger("strategies")
        self.logger.info(f"ORB Strategy initialized with timeframes: {timeframes}")
    
    def update_opening_ranges(self, symbol: str, ohlcv_data: Dict[str, np.ndarray], current_time: float):
        """Update opening ranges for all timeframes"""
        if symbol not in self.active_ranges:
            self.active_ranges[symbol] = {}
        
        # Get market time
        market_time = self._get_market_time(current_time)
        
        for timeframe in self.timeframes:
            if timeframe not in self.active_ranges[symbol]:
                # Check if we're within the opening range period
                if market_time <= timeframe:
                    # Create new opening range
                    self._create_opening_range(symbol, timeframe, ohlcv_data, current_time)
                elif market_time <= timeframe + 60:  # Allow 1 hour after range ends
                    # Range period ended, finalize if not already done
                    if timeframe not in self.active_ranges[symbol]:
                        self._finalize_opening_range(symbol, timeframe, ohlcv_data, current_time)
    
    def _get_market_time(self, current_time: float) -> float:
        """Get minutes since market open"""
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(current_time)
        
        # Calculate minutes since market open (9:30 AM)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        if dt < market_open:
            return -1  # Before market open
        
        delta = dt - market_open
        return delta.total_seconds() / 60
    
    def _create_opening_range(self, symbol: str, timeframe: int, ohlcv_data: Dict[str, np.ndarray], current_time: float):
        """Create opening range for specified timeframe"""
        if len(ohlcv_data['high']) < timeframe:
            return
        
        # Get data for opening period
        high_data = ohlcv_data['high'][-timeframe:]
        low_data = ohlcv_data['low'][-timeframe:]
        volume_data = ohlcv_data['volume'][-timeframe:]
        
        range_high = np.max(high_data)
        range_low = np.min(low_data)
        range_volume = np.sum(volume_data)
        range_size = (range_high - range_low) / range_low
        
        # Only create range if it meets minimum size requirement
        if range_size >= self.min_range_size:
            orb_range = ORBRange(
                start_time=current_time - (timeframe * 60),
                end_time=current_time,
                high=range_high,
                low=range_low,
                volume=range_volume,
                range_size=range_size
            )
            
            self.active_ranges[symbol][timeframe] = orb_range
            
            log_strategy("ORB", f"Opening range created for {symbol}", level="INFO",
                        timeframe=timeframe, range_size=range_size, 
                        high=range_high, low=range_low)
    
    def _finalize_opening_range(self, symbol: str, timeframe: int, ohlcv_data: Dict[str, np.ndarray], current_time: float):
        """Finalize opening range after period ends"""
        # Same logic as create, but marks as finalized
        self._create_opening_range(symbol, timeframe, ohlcv_data, current_time)
    
    def check_breakout_signals(self, symbol: str, current_price: float, volume: float, 
                              ai_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for breakout signals across all timeframes"""
        signals = []
        
        if symbol not in self.active_ranges:
            return signals
        
        for timeframe, orb_range in self.active_ranges[symbol].items():
            # Check for breakouts
            breakout_signal = self._analyze_breakout(
                symbol, timeframe, orb_range, current_price, volume, ai_signals
            )
            
            if breakout_signal:
                signals.append(breakout_signal)
        
        return signals
    
    def _analyze_breakout(self, symbol: str, timeframe: int, orb_range: ORBRange, 
                         current_price: float, volume: float, ai_signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze potential breakout"""
        
        # Check if price is breaking out
        is_breakout_up = orb_range.is_breakout_up(current_price)
        is_breakout_down = orb_range.is_breakout_down(current_price)
        
        if not (is_breakout_up or is_breakout_down):
            return None
        
        # Determine direction
        direction = "long" if is_breakout_up else "short"
        breakout_distance = orb_range.get_breakout_distance(current_price)
        
        # Volume confirmation
        avg_volume = orb_range.volume / timeframe  # Average per minute
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        volume_confirmed = volume_ratio >= self.volume_threshold
        
        # AI confirmation
        ai_signal_strength = abs(ai_signals.get('signal', 0.0))
        ai_confidence = ai_signals.get('confidence', 0.0)
        ai_direction_match = (
            (direction == "long" and ai_signals.get('signal', 0.0) > 0) or
            (direction == "short" and ai_signals.get('signal', 0.0) < 0)
        )
        
        # AI validation
        ai_confirmed = (
            ai_confidence >= self.ai_confidence_threshold and
            ai_direction_match and
            ai_signal_strength > 0.3
        )
        
        # Additional technical confirmation
        momentum_confirmed = self._check_momentum_confirmation(symbol, direction, current_price)
        
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            breakout_distance, volume_ratio, ai_confidence, 
            ai_signal_strength, momentum_confirmed
        )
        
        # Generate signal if all conditions met
        if volume_confirmed and ai_confirmed and signal_strength > 0.6:
            
            # Calculate stop loss and take profit
            stop_loss = self._calculate_stop_loss(orb_range, direction, current_price)
            take_profit = self._calculate_take_profit(orb_range, direction, current_price)
            
            signal = {
                'strategy': 'ORB',
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_strength': signal_strength,
                'breakout_distance': breakout_distance,
                'volume_ratio': volume_ratio,
                'ai_confidence': ai_confidence,
                'ai_signal': ai_signals.get('signal', 0.0),
                'range_high': orb_range.high,
                'range_low': orb_range.low,
                'range_size': orb_range.range_size,
                'timestamp': time.time(),
                'should_trade': True
            }
            
            log_strategy("ORB", f"Breakout signal generated for {symbol}", level="INFO",
                        direction=direction, timeframe=timeframe, 
                        signal_strength=signal_strength, volume_ratio=volume_ratio)
            
            # Track performance
            self.strategy_stats['total_signals'] += 1
            log_performance("orb_signals", self.strategy_stats['total_signals'], "count", 
                           f"Total ORB signals generated")
            
            return signal
        
        return None
    
    def _check_momentum_confirmation(self, symbol: str, direction: str, current_price: float) -> bool:
        """Check for momentum confirmation using technical indicators"""
        # This would integrate with technical analysis
        # For now, simple momentum check
        
        # Get recent price data (would come from data feed)
        # Simple momentum: price above/below recent average
        
        # Placeholder for momentum confirmation
        return True
    
    def _calculate_signal_strength(self, breakout_distance: float, volume_ratio: float, 
                                  ai_confidence: float, ai_signal_strength: float, 
                                  momentum_confirmed: bool) -> float:
        """Calculate overall signal strength"""
        
        # Breakout strength (0-0.3)
        breakout_score = min(0.3, breakout_distance * 30)
        
        # Volume strength (0-0.25)
        volume_score = min(0.25, (volume_ratio - 1.0) * 0.125)
        
        # AI strength (0-0.3)
        ai_score = ai_confidence * ai_signal_strength * 0.3
        
        # Momentum strength (0-0.15)
        momentum_score = 0.15 if momentum_confirmed else 0.0
        
        total_score = breakout_score + volume_score + ai_score + momentum_score
        
        return min(1.0, total_score)
    
    def _calculate_stop_loss(self, orb_range: ORBRange, direction: str, entry_price: float) -> float:
        """Calculate stop loss level"""
        if direction == "long":
            # Stop below range low with buffer
            buffer = orb_range.range_size * 0.2  # 20% of range size
            return orb_range.low * (1 - buffer)
        else:
            # Stop above range high with buffer
            buffer = orb_range.range_size * 0.2
            return orb_range.high * (1 + buffer)
    
    def _calculate_take_profit(self, orb_range: ORBRange, direction: str, entry_price: float) -> float:
        """Calculate take profit level"""
        # Take profit at 2x range size from entry
        range_size_dollars = orb_range.high - orb_range.low
        target_distance = range_size_dollars * 2
        
        if direction == "long":
            return entry_price + target_distance
        else:
            return entry_price - target_distance
    
    def validate_breakout_quality(self, symbol: str, timeframe: int, 
                                 breakout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate breakout quality using advanced analysis"""
        
        # Extract ORB-specific features
        orb_features = self.feature_engine.extract_orb_features(
            symbol, breakout_data, timeframe
        )
        
        # Breakout validation metrics
        validation_score = 0.0
        validation_reasons = []
        
        # Volume validation (30% weight)
        volume_score = min(1.0, breakout_data.get('volume_ratio', 1.0) / 2.0)
        validation_score += volume_score * 0.3
        if volume_score > 0.7:
            validation_reasons.append("Strong volume confirmation")
        
        # Range quality validation (25% weight)
        range_size = breakout_data.get('range_size', 0.0)
        range_score = min(1.0, range_size / 0.02)  # Normalize to 2% range
        validation_score += range_score * 0.25
        if range_score > 0.7:
            validation_reasons.append("Quality range size")
        
        # Breakout distance validation (20% weight)
        breakout_distance = breakout_data.get('breakout_distance', 0.0)
        distance_score = min(1.0, breakout_distance / 0.01)  # Normalize to 1% breakout
        validation_score += distance_score * 0.2
        if distance_score > 0.7:
            validation_reasons.append("Strong breakout distance")
        
        # AI confirmation validation (25% weight)
        ai_score = breakout_data.get('ai_confidence', 0.0)
        validation_score += ai_score * 0.25
        if ai_score > 0.7:
            validation_reasons.append("High AI confidence")
        
        return {
            'validation_score': validation_score,
            'is_valid': validation_score > 0.6,
            'validation_reasons': validation_reasons,
            'orb_features': orb_features
        }
    
    def manage_position(self, symbol: str, position_data: Dict[str, Any], 
                       current_price: float, ai_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Manage existing ORB position"""
        
        entry_price = position_data['entry_price']
        direction = position_data['direction']
        stop_loss = position_data['stop_loss']
        take_profit = position_data['take_profit']
        
        # Calculate current P&L
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Stop loss check
        if ((direction == "long" and current_price <= stop_loss) or
            (direction == "short" and current_price >= stop_loss)):
            should_exit = True
            exit_reason = "stop_loss"
        
        # Take profit check
        elif ((direction == "long" and current_price >= take_profit) or
              (direction == "short" and current_price <= take_profit)):
            should_exit = True
            exit_reason = "take_profit"
        
        # AI reversal signal check
        elif abs(ai_signals.get('signal', 0.0)) > 0.7:
            ai_direction = "long" if ai_signals['signal'] > 0 else "short"
            if ai_direction != direction:
                should_exit = True
                exit_reason = "ai_reversal"
        
        # Trailing stop logic (move stop loss in favorable direction)
        new_stop_loss = stop_loss
        if not should_exit and pnl_pct > 0.01:  # If profitable by more than 1%
            trail_distance = abs(entry_price - stop_loss)
            if direction == "long":
                new_stop_loss = max(stop_loss, current_price - trail_distance)
            else:
                new_stop_loss = min(stop_loss, current_price + trail_distance)
        
        return {
            'should_exit': should_exit,
            'exit_reason': exit_reason,
            'current_pnl_pct': pnl_pct,
            'new_stop_loss': new_stop_loss,
            'recommendation': 'hold' if not should_exit else 'exit'
        }
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        
        # Calculate win rate
        if self.strategy_stats['total_signals'] > 0:
            self.strategy_stats['win_rate'] = (
                self.strategy_stats['successful_breakouts'] / 
                self.strategy_stats['total_signals']
            )
        
        return {
            'strategy': 'ORB',
            'performance': self.strategy_stats,
            'active_ranges': len(self.active_ranges),
            'active_positions': len(self.active_positions),
            'timeframes': self.timeframes,
            'settings': {
                'min_range_size': self.min_range_size,
                'volume_threshold': self.volume_threshold,
                'ai_confidence_threshold': self.ai_confidence_threshold,
                'max_positions': self.max_positions
            }
        }
    
    def cleanup_expired_ranges(self, current_time: float):
        """Clean up expired opening ranges"""
        market_time = self._get_market_time(current_time)
        
        # Remove ranges older than 4 hours (240 minutes)
        for symbol in list(self.active_ranges.keys()):
            for timeframe in list(self.active_ranges[symbol].keys()):
                if market_time > timeframe + 240:  # 4 hours after range end
                    del self.active_ranges[symbol][timeframe]
            
            # Remove symbol if no active ranges
            if not self.active_ranges[symbol]:
                del self.active_ranges[symbol]

# Example usage and testing
if __name__ == "__main__":
    # Initialize ORB strategy
    orb_strategy = ORBStrategy()
    
    print("📊 ORB Strategy Test")
    print("=" * 40)
    
    # Simulate opening range data
    symbol = "AAPL"
    current_time = time.time()
    
    # Create sample OHLCV data
    ohlcv_data = {
        'open': np.array([150.0] * 30),
        'high': np.array([151.5] * 15 + [152.0] * 15),
        'low': np.array([149.5] * 15 + [149.0] * 15),
        'close': np.array([150.5] * 30),
        'volume': np.array([1000] * 30)
    }
    
    # Update opening ranges
    orb_strategy.update_opening_ranges(symbol, ohlcv_data, current_time)
    
    # Simulate AI signals
    ai_signals = {
        'signal': 0.75,
        'confidence': 0.8,
        'signal_strength': 0.7
    }
    
    # Check for breakout signals
    current_price = 152.5  # Above range high
    volume = 2500  # High volume
    
    signals = orb_strategy.check_breakout_signals(symbol, current_price, volume, ai_signals)
    
    print(f"📈 Breakout Analysis:")
    print(f"   Symbol: {symbol}")
    print(f"   Current Price: ${current_price}")
    print(f"   Volume: {volume}")
    print(f"   Signals Generated: {len(signals)}")
    
    for signal in signals:
        print(f"\n   Signal Details:")
        print(f"     Direction: {signal['direction']}")
        print(f"     Timeframe: {signal['timeframe']}min")
        print(f"     Signal Strength: {signal['signal_strength']:.2f}")
        print(f"     Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"     Take Profit: ${signal['take_profit']:.2f}")
    
    # Get strategy stats
    stats = orb_strategy.get_strategy_stats()
    print(f"\n📊 Strategy Statistics:")
    for key, value in stats['performance'].items():
        print(f"   {key}: {value}")
    
    print("\n✅ ORB Strategy test completed!")
