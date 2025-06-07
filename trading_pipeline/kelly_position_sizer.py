#!/usr/bin/env python3

from typing import Dict
from dataclasses import dataclass
from datetime import datetime, time
import os
import yaml

from utils import get_logger

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'trading_pipeline.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

# Initialize component logger
logger = get_logger("kelly_position_sizer")

@dataclass
class KellyParameters:
    """Kelly Criterion calculation parameters"""
    expected_return: float
    win_probability: float
    loss_probability: float
    profit_loss_ratio: float
    kelly_fraction: float
    position_size: int
    position_value: float

class KellyPositionSizer:
    """
    Kelly Criterion position sizer using ML predictions
    Calculates optimal position sizes for momentum trading strategy
    """
    
    def __init__(self, available_capital: float = 50000):
        logger.startup({
            "available_capital": available_capital,
            "action": "initialization_start"
        })
        
        # Load configuration
        risk_config = CONFIG.get('trading', {}).get('risk_management', {})
        portfolio_config = CONFIG.get('trading', {}).get('portfolio', {})
        
        # Strategy parameters from YAML or defaults
        self.STOP_LOSS_PCT = risk_config.get('stop_loss_percentage', 0.015)  # 1.5% stop loss
        self.TP1_PCT = 0.01         # +1% profit target
        self.TP2_PCT = 0.03         # +3% profit target
        self.TIER_ALLOCATIONS = [0.30, 0.40, 0.30]  # 30%, 40%, 30%
        self.TIME_EXIT = "15:45"    # 3:45 PM ET - close all positions
        
        # Kelly parameters
        self.SAFETY_FACTOR = 0.25   # Use 25% of Kelly for safety (fractional Kelly)
        self.MIN_KELLY_FRACTION = 0.01  # Minimum 1% of capital
        self.MAX_KELLY_FRACTION = risk_config.get('position_size_limit', 0.10)  # Maximum 10% of capital
        self.MIN_POSITION_VALUE = portfolio_config.get('min_position_value', 1000)  # Minimum $1k position
        self.MAX_POSITION_VALUE = risk_config.get('max_position_size', 10000) # Maximum $10k position
        
        # Capital management
        self.available_capital = available_capital
        
        logger.startup({
            "action": "initialization_complete",
            "available_capital": available_capital,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "max_kelly_fraction": self.MAX_KELLY_FRACTION,
            "min_position_value": self.MIN_POSITION_VALUE,
            "max_position_value": self.MAX_POSITION_VALUE,
            "safety_factor": self.SAFETY_FACTOR
        })
    
    def calculate_kelly_position_size(self, signal) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion with ML predictions
        
        Args:
            signal: Trading signal with prediction, confidence, current_price, symbol
            
        Returns:
            Dict with position sizing information
        """
        try:
            # Extract signal parameters
            ml_prediction = getattr(signal, 'prediction', 0.0)
            ml_confidence = getattr(signal, 'confidence', 0.5)
            current_price = getattr(signal, 'current_price', 0.0)
            symbol = getattr(signal, 'symbol', 'UNKNOWN')
            
            if current_price <= 0:
                logger.error(ValueError(f"Invalid price for {symbol}: {current_price}"), {
                    "symbol": symbol,
                    "price": current_price
                })
                return self._create_fallback_position(symbol, current_price)
            
            # Calculate Kelly parameters
            kelly_params = self._calculate_kelly_parameters(ml_prediction, ml_confidence)
            
            # Calculate position size
            position_value = self.available_capital * kelly_params.kelly_fraction
            
            # Apply position limits
            position_value = max(self.MIN_POSITION_VALUE, 
                               min(position_value, self.MAX_POSITION_VALUE))
            
            # Convert to shares
            shares = int(position_value / current_price)
            shares = max(10, shares)  # Minimum 10 shares
            
            # Recalculate actual position value
            actual_position_value = shares * current_price
            
            # Calculate tier quantities
            tier_quantities = self._calculate_tier_quantities(shares)
            
            # Calculate price levels
            prices = self._calculate_price_levels(current_price)
            
            # Calculate time exit
            time_exit = self._calculate_time_exit()
            
            result = {
                'symbol': symbol,
                'entry_price': current_price,
                'total_qty': shares,
                'total_value': actual_position_value,
                'tier_quantities': tier_quantities,
                'prices': prices,
                'time_exit': time_exit,
                'kelly_params': kelly_params,
                'capital_allocation_pct': (actual_position_value / self.available_capital) * 100
            }
            
            logger.log_position_calculation(symbol, "kelly_criterion",
                {"ml_prediction": ml_prediction, "ml_confidence": ml_confidence, "current_price": current_price},
                {"shares": shares, "position_value": actual_position_value, "allocation_pct": result['capital_allocation_pct']})
            
            return result
            
        except Exception as e:
            logger.error(e, {
                "symbol": symbol,
                "operation": "calculate_kelly_position"
            })
            return self._create_fallback_position(symbol, current_price)
    
    def _calculate_kelly_parameters(self, ml_prediction: float, ml_confidence: float) -> KellyParameters:
        """Calculate Kelly Criterion parameters from ML inputs"""
        
        # Expected return from ML signal (prediction weighted by confidence)
        expected_return = ml_prediction * ml_confidence
        
        # Win probability from ML confidence
        # Base 50% + confidence boost (higher confidence = higher win rate)
        win_probability = 0.5 + (ml_confidence * 0.3)
        win_probability = min(win_probability, 0.85)  # Cap at 85%
        win_probability = max(win_probability, 0.55)  # Floor at 55%
        
        # Loss probability
        loss_probability = 1 - win_probability
        
        # Profit/loss ratio
        # Average profit target vs stop loss
        avg_profit_target = (self.TP1_PCT + self.TP2_PCT) / 2  # (1% + 3%) / 2 = 2%
        profit_loss_ratio = avg_profit_target / self.STOP_LOSS_PCT  # 2% / 1.5% = 1.33
        
        # Kelly fraction calculation: f* = (bp - q) / b
        # Where: b = profit_loss_ratio, p = win_probability, q = loss_probability
        kelly_fraction = (profit_loss_ratio * win_probability - loss_probability) / profit_loss_ratio
        
        # Apply safety factor (fractional Kelly)
        kelly_fraction *= self.SAFETY_FACTOR
        
        # Ensure reasonable bounds
        kelly_fraction = max(self.MIN_KELLY_FRACTION, 
                           min(kelly_fraction, self.MAX_KELLY_FRACTION))
        
        return KellyParameters(
            expected_return=expected_return,
            win_probability=win_probability,
            loss_probability=loss_probability,
            profit_loss_ratio=profit_loss_ratio,
            kelly_fraction=kelly_fraction,
            position_size=0,  # Will be calculated later
            position_value=0  # Will be calculated later
        )
    
    def _calculate_tier_quantities(self, total_shares: int) -> Dict[str, int]:
        """Calculate quantities for each profit tier"""
        tier1_qty = int(total_shares * self.TIER_ALLOCATIONS[0])  # 30%
        tier2_qty = int(total_shares * self.TIER_ALLOCATIONS[1])  # 40%
        tier3_qty = total_shares - tier1_qty - tier2_qty         # Remaining 30%
        
        return {
            'tier1': tier1_qty,
            'tier2': tier2_qty,
            'tier3': tier3_qty,
            'total': total_shares
        }
    
    def _calculate_price_levels(self, entry_price: float) -> Dict[str, float]:
        """Calculate stop loss and profit target prices"""
        stop_loss_price = entry_price * (1 - self.STOP_LOSS_PCT)    # -1.5%
        tp1_price = entry_price * (1 + self.TP1_PCT)               # +1%
        tp2_price = entry_price * (1 + self.TP2_PCT)               # +3%
        trail_percent = 2.0  # 2% trailing stop for tier 3
        
        return {
            'stop_loss': round(stop_loss_price, 2),
            'tp1_target': round(tp1_price, 2),
            'tp2_target': round(tp2_price, 2),
            'trail_percent': trail_percent
        }
    
    def _calculate_time_exit(self) -> datetime:
        """Calculate time exit (3:45 PM ET)"""
        today = datetime.now().date()
        time_exit = datetime.combine(today, time(15, 45))  # 3:45 PM ET
        return time_exit
    
    def _create_fallback_position(self, symbol: str, price: float) -> Dict:
        """Create fallback position when Kelly calculation fails"""
        fallback_shares = 50  # Default 50 shares
        fallback_value = fallback_shares * price if price > 0 else 2500
        
        return {
            'symbol': symbol,
            'entry_price': price,
            'total_qty': fallback_shares,
            'total_value': fallback_value,
            'tier_quantities': {
                'tier1': 15,  # 30%
                'tier2': 20,  # 40%
                'tier3': 15,  # 30%
                'total': fallback_shares
            },
            'prices': self._calculate_price_levels(price) if price > 0 else {
                'stop_loss': 0, 'tp1_target': 0, 'tp2_target': 0, 'trail_percent': 2.0
            },
            'time_exit': self._calculate_time_exit(),
            'kelly_params': None,
            'capital_allocation_pct': (fallback_value / self.available_capital) * 100,
            'is_fallback': True
        }
    
    def update_available_capital(self, new_capital: float):
        """Update available capital for position sizing"""
        self.available_capital = new_capital
        logger.config_change("available_capital", self.available_capital, new_capital)
    
    def get_position_sizing_stats(self) -> Dict:
        """Get position sizing statistics and parameters"""
        return {
            'available_capital': self.available_capital,
            'safety_factor': self.SAFETY_FACTOR,
            'kelly_bounds': {
                'min_fraction': self.MIN_KELLY_FRACTION,
                'max_fraction': self.MAX_KELLY_FRACTION
            },
            'position_limits': {
                'min_value': self.MIN_POSITION_VALUE,
                'max_value': self.MAX_POSITION_VALUE
            },
            'strategy_params': {
                'stop_loss_pct': self.STOP_LOSS_PCT,
                'tp1_pct': self.TP1_PCT,
                'tp2_pct': self.TP2_PCT,
                'tier_allocations': self.TIER_ALLOCATIONS
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create position sizer
    kelly_sizer = KellyPositionSizer(available_capital=50000)
    
    # Mock signal for testing
    class MockSignal:
        def __init__(self, symbol, prediction, confidence, current_price):
            self.symbol = symbol
            self.prediction = prediction
            self.confidence = confidence
            self.current_price = current_price
    
    # Test with different signal strengths
    test_signals = [
        MockSignal("AAPL", 0.8, 0.9, 150.0),   # Strong bullish signal
        MockSignal("MSFT", 0.3, 0.6, 300.0),   # Weak bullish signal
        MockSignal("GOOGL", -0.5, 0.7, 2500.0) # Bearish signal
    ]
    
    for signal in test_signals:
        position = kelly_sizer.calculate_kelly_position_size(signal)
        print(f"\n{signal.symbol}: {position['total_qty']} shares, "
              f"${position['total_value']:,.0f} "
              f"({position['capital_allocation_pct']:.1f}% of capital)")