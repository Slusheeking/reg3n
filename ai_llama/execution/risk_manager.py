"""
Risk Manager

Comprehensive risk management system for AI trading platform.
Features:
- Position size calculation
- Real-time risk monitoring
- Portfolio exposure limits
- Dynamic risk adjustments
- Stop loss and take profit management
- Maximum drawdown protection
- Volatility-based position sizing
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

from utils.logger import get_logger, log_performance, log_strategy

class RiskLevel(Enum):
    """Risk level definitions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    max_risk: float = 0.02  # 2% max risk per trade
    
    def update_price(self, price: float):
        """Update current price and calculate PnL"""
        self.current_price = price
        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
    
    def get_current_risk(self) -> float:
        """Calculate current risk as percentage of position value"""
        if self.entry_price <= 0:
            return 0.0
        position_value = abs(self.quantity * self.entry_price)
        if position_value <= 0:
            return 0.0
        return abs(self.unrealized_pnl) / position_value

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
class RiskManager:
    """
    Comprehensive Risk Management System
    
    Features:
    - Position sizing based on volatility and correlation
    - Real-time portfolio risk monitoring
    - Dynamic stop loss and take profit levels
    - Exposure limits and concentration risk
    - Correlation-based position limits
    - Maximum drawdown protection
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 max_position_size: float = 0.10,   # 10% max single position
                 max_sector_exposure: float = 0.30, # 30% max sector exposure
                 max_drawdown: float = 0.15,        # 15% max drawdown
                 risk_free_rate: float = 0.02):     # 2% risk-free rate
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Risk metrics
        self.risk_metrics = RiskMetrics()
        self.pnl_history = deque(maxlen=1000)
        self.capital_history = deque(maxlen=1000)
        
        # Correlation matrix and volatility tracking
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        
        # Risk limits and alerts
        self.risk_alerts = []
        self.position_limits = {}
        
        # Performance tracking
        self.daily_returns = deque(maxlen=252)  # One year of trading days
        self.last_capital_update = time.time()
        
        # Logging
        self.logger = get_logger("risk")
        self.logger.info(f"Risk Manager initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              price: float, 
                              volatility: float,
                              confidence: float = 0.5,
                              max_risk_per_trade: float = 0.01) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size using volatility-based sizing
        
        Args:
            symbol: Trading symbol
            price: Current price
            volatility: Estimated volatility
            confidence: Model confidence (0-1)
            max_risk_per_trade: Maximum risk per trade (default 1%)
            
        Returns:
            Tuple of (position_size, sizing_details)
        """
        try:
            # Base position size calculation
            if volatility <= 0 or price <= 0:
                return 0.0, {"error": "Invalid price or volatility"}
            
            # Risk budget for this trade
            risk_budget = min(max_risk_per_trade, self.max_portfolio_risk * 0.5)
            risk_budget *= confidence  # Scale by model confidence
            
            # Volatility-adjusted position size
            # Position size = (Risk Budget * Capital) / (Volatility * Price)
            base_size = (risk_budget * self.current_capital) / (volatility * price)
            
            # Apply portfolio and concentration limits
            max_position_value = self.current_capital * self.max_position_size
            max_size_by_value = max_position_value / price
            
            # Consider correlation with existing positions
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)
            adjusted_size = base_size * correlation_adjustment
            
            # Final position size
            final_size = min(adjusted_size, max_size_by_value)
            
            # Additional safety checks
            if self._check_position_limits(symbol, final_size, price):
                sizing_details = {
                    "base_size": base_size,
                    "max_size_by_value": max_size_by_value,
                    "correlation_adjustment": correlation_adjustment,
                    "final_size": final_size,
                    "risk_budget": risk_budget,
                    "position_value": final_size * price,
                    "position_risk": (final_size * price) / self.current_capital
                }
                
                log_performance("position_size", final_size, "shares", 
                              f"Position sizing for {symbol}")
                
                return final_size, sizing_details
            else:
                return 0.0, {"error": "Position limits exceeded"}
                
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0, {"error": str(e)}
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on correlation with existing positions"""
        if not self.positions or symbol not in self.correlation_matrix:
            return 1.0
        
        # Calculate weighted correlation with existing positions
        total_exposure = 0.0
        weighted_correlation = 0.0
        
        for pos_symbol, position in self.positions.items():
            if pos_symbol == symbol:
                continue
                
            position_weight = abs(position.quantity * position.current_price) / self.current_capital
            correlation = self.correlation_matrix.get(symbol, {}).get(pos_symbol, 0.0)
            
            weighted_correlation += abs(correlation) * position_weight
            total_exposure += position_weight
        
        if total_exposure > 0:
            avg_correlation = weighted_correlation / total_exposure
            # Reduce position size for highly correlated assets
            adjustment = max(0.3, 1.0 - (avg_correlation * 0.7))
            return adjustment
        
        return 1.0
    
    def _check_position_limits(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if position size respects all limits"""
        position_value = abs(quantity * price)
        
        # Check maximum position size
        if position_value > self.current_capital * self.max_position_size:
            self.logger.warning(f"Position size limit exceeded for {symbol}")
            return False
        
        # Check if adding this position would exceed portfolio risk
        current_exposure = sum(abs(pos.quantity * pos.current_price) 
                             for pos in self.positions.values())
        
        if (current_exposure + position_value) > self.current_capital * 0.95:
            self.logger.warning(f"Portfolio exposure limit exceeded")
            return False
        
        return True
    
    def add_position(self, 
                    symbol: str, 
                    side: str, 
                    quantity: float, 
                    price: float,
                    strategy: str = "",
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> bool:
        """Add new position to portfolio"""
        try:
            if symbol in self.positions:
                # Modify existing position
                self._modify_position(symbol, quantity, price)
            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    side=side.lower(),
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    entry_time=time.time(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy=strategy
                )
                
                self.positions[symbol] = position
                
                log_strategy("RiskManager", f"Position opened: {symbol}", level="INFO",
                           side=side, quantity=quantity, price=price, strategy=strategy)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def _modify_position(self, symbol: str, quantity_change: float, price: float):
        """Modify existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate new average price if adding to position
        if (position.side == 'long' and quantity_change > 0) or \
           (position.side == 'short' and quantity_change < 0):
            
            total_cost = position.quantity * position.entry_price + abs(quantity_change) * price
            new_quantity = position.quantity + quantity_change
            
            if new_quantity != 0:
                position.entry_price = total_cost / abs(new_quantity)
                position.quantity = new_quantity
            else:
                # Position closed
                self._close_position(symbol, price)
        else:
            # Reducing position or closing
            if abs(quantity_change) >= abs(position.quantity):
                self._close_position(symbol, price)
            else:
                position.quantity += quantity_change
    
    def update_position_price(self, symbol: str, price: float):
        """Update position with current market price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
            
            # Update price history for volatility calculation
            self.price_history[symbol].append(price)
            
            # Check stop loss and take profit
            self._check_exit_conditions(symbol)
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
    
    def _check_exit_conditions(self, symbol: str):
        """Check if position should be closed due to stop loss or take profit"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        price = position.current_price
        
        should_close = False
        reason = ""
        
        # Check stop loss
        if position.stop_loss is not None:
            if (position.side == 'long' and price <= position.stop_loss) or \
               (position.side == 'short' and price >= position.stop_loss):
                should_close = True
                reason = "stop_loss"
        
        # Check take profit
        if position.take_profit is not None and not should_close:
            if (position.side == 'long' and price >= position.take_profit) or \
               (position.side == 'short' and price <= position.take_profit):
                should_close = True
                reason = "take_profit"
        
        # Check maximum risk per position
        if position.get_current_risk() > position.max_risk:
            should_close = True
            reason = "max_risk_exceeded"
        
        if should_close:
            self.logger.warning(f"Closing position {symbol} due to {reason}")
            self._close_position(symbol, price)
    
    def _close_position(self, symbol: str, price: float):
        """Close position and update metrics"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.update_price(price)
        
        # Calculate realized PnL
        if position.side == 'long':
            position.realized_pnl = (price - position.entry_price) * position.quantity
        else:
            position.realized_pnl = (position.entry_price - price) * position.quantity
        
        # Update capital
        self.current_capital += position.realized_pnl
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update metrics
        self._update_trade_statistics(position)
        
        log_strategy("RiskManager", f"Position closed: {symbol}", level="INFO",
                   realized_pnl=position.realized_pnl, 
                   holding_period=(time.time() - position.entry_time)/3600)
    
    def _update_trade_statistics(self, position: Position):
        """Update trading statistics with closed position"""
        self.risk_metrics.total_trades += 1
        
        if position.realized_pnl > 0:
            self.risk_metrics.winning_trades += 1
            self.risk_metrics.avg_win = (
                (self.risk_metrics.avg_win * (self.risk_metrics.winning_trades - 1) + 
                 position.realized_pnl) / self.risk_metrics.winning_trades
            )
        else:
            self.risk_metrics.losing_trades += 1
            self.risk_metrics.avg_loss = (
                (self.risk_metrics.avg_loss * (self.risk_metrics.losing_trades - 1) + 
                 position.realized_pnl) / self.risk_metrics.losing_trades
            )
        
        # Update win rate
        if self.risk_metrics.total_trades > 0:
            self.risk_metrics.win_rate = self.risk_metrics.winning_trades / self.risk_metrics.total_trades
        
        # Track PnL
        self.pnl_history.append(position.realized_pnl)
        self.capital_history.append(self.current_capital)
        
        # Update daily returns
        current_time = time.time()
        if current_time - self.last_capital_update > 86400:  # 24 hours
            if len(self.capital_history) > 1:
                daily_return = (self.current_capital - self.capital_history[-2]) / self.capital_history[-2]
                self.daily_returns.append(daily_return)
            self.last_capital_update = current_time
    
    def _update_portfolio_metrics(self):
        """Update real-time portfolio risk metrics"""
        # Calculate exposures
        total_long = sum(pos.quantity * pos.current_price 
                        for pos in self.positions.values() if pos.side == 'long')
        total_short = sum(abs(pos.quantity * pos.current_price) 
                         for pos in self.positions.values() if pos.side == 'short')
        
        self.risk_metrics.gross_exposure = (total_long + total_short) / self.current_capital
        self.risk_metrics.net_exposure = (total_long - total_short) / self.current_capital
        self.risk_metrics.total_exposure = max(total_long, total_short) / self.current_capital
        
        # Calculate unrealized PnL
        self.risk_metrics.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.risk_metrics.realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        
        # Calculate current drawdown
        peak_capital = max(self.capital_history) if self.capital_history else self.initial_capital
        current_total = self.current_capital + self.risk_metrics.unrealized_pnl
        self.risk_metrics.current_drawdown = (peak_capital - current_total) / peak_capital
        
        # Update max drawdown
        self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, 
                                           self.risk_metrics.current_drawdown)
        
        # Calculate portfolio volatility and Sharpe ratio
        if len(self.daily_returns) > 10:
            returns_array = np.array(self.daily_returns)
            self.risk_metrics.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            avg_return = np.mean(returns_array) * 252  # Annualized
            
            if self.risk_metrics.volatility > 0:
                self.risk_metrics.sharpe_ratio = (avg_return - self.risk_free_rate) / self.risk_metrics.volatility
            
            # Calculate VaR (95% confidence)
            self.risk_metrics.var_95 = np.percentile(returns_array, 5) * self.current_capital
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment"""
        risk_level = self._assess_risk_level()
        
        return {
            "risk_level": risk_level.value,
            "portfolio_metrics": {
                "current_capital": self.current_capital,
                "total_pnl": self.risk_metrics.realized_pnl + self.risk_metrics.unrealized_pnl,
                "gross_exposure": self.risk_metrics.gross_exposure,
                "net_exposure": self.risk_metrics.net_exposure,
                "current_drawdown": self.risk_metrics.current_drawdown,
                "max_drawdown": self.risk_metrics.max_drawdown,
                "sharpe_ratio": self.risk_metrics.sharpe_ratio,
                "volatility": self.risk_metrics.volatility,
                "var_95": self.risk_metrics.var_95
            },
            "trading_metrics": {
                "total_trades": self.risk_metrics.total_trades,
                "win_rate": self.risk_metrics.win_rate,
                "avg_win": self.risk_metrics.avg_win,
                "avg_loss": self.risk_metrics.avg_loss,
                "active_positions": len(self.positions)
            },
            "risk_alerts": self.risk_alerts,
            "position_summary": {
                symbol: {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "current_risk": pos.get_current_risk()
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def _assess_risk_level(self) -> RiskLevel:
        """Assess current portfolio risk level"""
        risk_score = 0
        
        # Drawdown risk
        if self.risk_metrics.current_drawdown > self.max_drawdown * 0.8:
            risk_score += 30
        elif self.risk_metrics.current_drawdown > self.max_drawdown * 0.5:
            risk_score += 15
        
        # Exposure risk
        if self.risk_metrics.gross_exposure > 0.8:
            risk_score += 25
        elif self.risk_metrics.gross_exposure > 0.5:
            risk_score += 10
        
        # Concentration risk
        if len(self.positions) > 0:
            max_position_size = max(abs(pos.quantity * pos.current_price) / self.current_capital 
                                  for pos in self.positions.values())
            if max_position_size > self.max_position_size * 0.8:
                risk_score += 20
        
        # Volatility risk
        if self.risk_metrics.volatility > 0.25:  # 25% annual volatility
            risk_score += 15
        
        # Return risk level
        if risk_score >= 70:
            return RiskLevel.CRITICAL
        elif risk_score >= 40:
            return RiskLevel.HIGH
        elif risk_score >= 20:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_trade_approval(self, 
                           symbol: str, 
                           side: str, 
                           quantity: float, 
                           price: float) -> Tuple[bool, str]:
        """Check if trade is approved based on risk limits"""
        
        # Check maximum drawdown
        if self.risk_metrics.current_drawdown >= self.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        # Check position size limits
        position_value = abs(quantity * price)
        if position_value > self.current_capital * self.max_position_size:
            return False, "Position size exceeds limit"
        
        # Check portfolio exposure
        current_exposure = sum(abs(pos.quantity * pos.current_price) 
                             for pos in self.positions.values())
        if (current_exposure + position_value) > self.current_capital * 0.95:
            return False, "Portfolio exposure limit exceeded"
        
        # Check if we have enough capital
        if position_value > self.current_capital * 0.1:  # Don't use more than 10% for single trade
            margin_required = position_value * 0.1  # Assume 10:1 leverage
            if margin_required > self.current_capital * 0.1:
                return False, "Insufficient capital"
        
        return True, "Trade approved"
    
    def update_volatility_estimates(self, price_data: Dict[str, List[float]]):
        """Update volatility estimates from price data"""
        for symbol, prices in price_data.items():
            if len(prices) > 10:
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                self.volatility_estimates[symbol] = volatility
    
    def get_stop_loss_level(self, symbol: str, entry_price: float, side: str, atr: float = None) -> float:
        """Calculate dynamic stop loss level"""
        if symbol in self.volatility_estimates:
            volatility = self.volatility_estimates[symbol]
        elif atr:
            volatility = atr / entry_price  # Convert ATR to percentage
        else:
            volatility = 0.02  # Default 2% volatility
        
        # Stop loss at 2x volatility from entry
        stop_distance = min(0.05, volatility * 2)  # Max 5% stop loss
        
        if side.lower() == 'long':
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def get_take_profit_level(self, symbol: str, entry_price: float, side: str, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit level based on stop loss"""
        stop_loss = self.get_stop_loss_level(symbol, entry_price, side)
        stop_distance = abs(entry_price - stop_loss)
        
        if side.lower() == 'long':
            return entry_price + (stop_distance * risk_reward_ratio)
        else:
            return entry_price - (stop_distance * risk_reward_ratio)

# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(initial_capital=100000)
    
    print("🛡️ Risk Manager Test")
    print("=" * 40)
    
    # Test position sizing
    position_size, details = risk_manager.calculate_position_size(
        symbol="AAPL",
        price=150.0,
        volatility=0.25,
        confidence=0.8
    )
    
    print(f"📊 Position Size Calculation:")
    print(f"   Symbol: AAPL")
    print(f"   Position Size: {position_size:.0f} shares")
    print(f"   Position Value: ${position_size * 150:,.2f}")
    print(f"   Details: {details}")
    
    # Test adding positions
    risk_manager.add_position("AAPL", "long", position_size, 150.0, "GapAndGo")
    risk_manager.add_position("MSFT", "long", 100, 415.0, "ORB")
    
    # Update prices
    risk_manager.update_position_price("AAPL", 155.0)
    risk_manager.update_position_price("MSFT", 410.0)
    
    # Get risk assessment
    assessment = risk_manager.get_risk_assessment()
    
    print(f"\n🛡️ Risk Assessment:")
    print(f"   Risk Level: {assessment['risk_level']}")
    print(f"   Current Capital: ${assessment['portfolio_metrics']['current_capital']:,.2f}")
    print(f"   Total PnL: ${assessment['portfolio_metrics']['total_pnl']:,.2f}")
    print(f"   Gross Exposure: {assessment['portfolio_metrics']['gross_exposure']:.1%}")
    print(f"   Current Drawdown: {assessment['portfolio_metrics']['current_drawdown']:.1%}")
    print(f"   Active Positions: {assessment['trading_metrics']['active_positions']}")
    
    print("\n✅ Risk Manager test completed!")
