#!/usr/bin/env python3

import logging
import os
import sys
import yaml
from typing import Dict, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import enhanced logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

# Initialize logger
logger = get_system_logger("filters.vix_position_scaler")

@dataclass
class VIXPositionConfig:
    """Configuration for VIX-based position scaling"""
    vix_level: float
    max_positions: int
    position_size: float
    total_capital_target: float
    risk_level: str

class VIXPositionScaler:
    """
    Simple VIX-based position scaling system
    Adjusts portfolio size based on market volatility
    """
    
    def __init__(self, total_capital: float = 50000, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'filters.yaml')
        
        self.config = self._load_config(config_path)
        
        # Load total capital from config or use parameter
        config_capital = self.config.get('vix_position_scaling', {}).get('capital_management', {}).get('total_capital')
        self.total_capital = config_capital if config_capital else total_capital
        
        logger.startup({
            "component": "vix_position_scaler",
            "action": "initialization",
            "total_capital": self.total_capital,
            "config_path": config_path
        })
        
        # Load VIX configs from YAML
        self.vix_configs = self._setup_vix_configs()
        
        self.current_vix = 20.0  # Default VIX level
        self.current_config = self.vix_configs['medium']  # Default to medium
        self.position_history = []
        
        logger.info(f"VIX Position Scaler initialized with ${self.total_capital:,.0f} capital")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load VIX configuration from YAML"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default VIX configuration"""
        return {
            'vix_position_scaling': {
                'thresholds': {
                    'low_vix': 15,
                    'high_vix': 25
                },
                'regimes': {
                    'low': {
                        'max_positions': 8,
                        'position_size': 6250,
                        'risk_level': 'low'
                    },
                    'medium': {
                        'max_positions': 6,
                        'position_size': 8333,
                        'risk_level': 'medium'
                    },
                    'high': {
                        'max_positions': 5,
                        'position_size': 10000,
                        'risk_level': 'high'
                    }
                },
                'capital_management': {
                    'total_capital': 50000
                }
            }
        }
    
    def _setup_vix_configs(self) -> Dict:
        """Setup VIX configurations from YAML"""
        vix_config = self.config.get('vix_position_scaling', {})
        thresholds = vix_config.get('thresholds', {})
        regimes = vix_config.get('regimes', {})
        
        configs = {}
        
        # Setup each regime
        for regime_name, regime_config in regimes.items():
            vix_level = float('inf')  # Default for high regime
            if regime_name == 'low':
                vix_level = thresholds.get('low_vix', 15.0)
            elif regime_name == 'medium':
                vix_level = thresholds.get('high_vix', 25.0)
            
            configs[regime_name] = VIXPositionConfig(
                vix_level=vix_level,
                max_positions=regime_config.get('max_positions', 6),
                position_size=regime_config.get('position_size', 8333),
                total_capital_target=self.total_capital,
                risk_level=regime_config.get('risk_level', regime_name)
            )
        
        # Fallback to defaults if no valid configs
        if not configs:
            configs = {
                'low': VIXPositionConfig(
                    vix_level=15.0,
                    max_positions=8,
                    position_size=6250,
                    total_capital_target=self.total_capital,
                    risk_level='low'
                ),
                'medium': VIXPositionConfig(
                    vix_level=25.0,
                    max_positions=6,
                    position_size=8333,
                    total_capital_target=self.total_capital,
                    risk_level='medium'
                ),
                'high': VIXPositionConfig(
                    vix_level=float('inf'),
                    max_positions=5,
                    position_size=10000,
                    total_capital_target=self.total_capital,
                    risk_level='high'
                )
            }
        
        return configs
    
    def update_vix_level(self, vix_level: float) -> VIXPositionConfig:
        """
        Update VIX level and return appropriate position configuration
        
        Args:
            vix_level: Current VIX level
            
        Returns:
            VIXPositionConfig for current volatility regime
        """
        try:
            self.current_vix = vix_level
            
            # Determine volatility regime
            if vix_level < 15:
                regime = 'low'
            elif vix_level < 25:
                regime = 'medium'
            else:
                regime = 'high'
            
            new_config = self.vix_configs[regime]
            
            # Log regime changes
            if new_config.risk_level != self.current_config.risk_level:
                logger.info(f"VIX regime change: {self.current_config.risk_level} → {new_config.risk_level} "
                           f"(VIX: {vix_level:.1f})")
                logger.info(f"New position limits: {new_config.max_positions} positions × "
                           f"${new_config.position_size:,.0f} = ${new_config.total_capital_target:,.0f}")
            
            self.current_config = new_config
            return new_config
            
        except Exception as e:
            logger.error(f"Error updating VIX level: {e}")
            return self.current_config
    
    def calculate_position_limits(self, current_positions: int = 0) -> Dict:
        """
        Calculate position limits based on current VIX regime
        
        Args:
            current_positions: Number of currently open positions
            
        Returns:
            Dict with position limit information
        """
        try:
            config = self.current_config
            
            # Calculate available position slots
            available_slots = max(0, config.max_positions - current_positions)
            
            # Calculate capital allocation
            capital_per_position = min(config.position_size, self.total_capital / config.max_positions)
            total_capital_allocated = current_positions * capital_per_position
            available_capital = self.total_capital - total_capital_allocated
            
            return {
                'vix_level': self.current_vix,
                'risk_regime': config.risk_level,
                'max_positions': config.max_positions,
                'current_positions': current_positions,
                'available_slots': available_slots,
                'position_size_target': capital_per_position,
                'total_capital_allocated': total_capital_allocated,
                'available_capital': available_capital,
                'can_add_position': available_slots > 0 and available_capital >= capital_per_position,
                'position_utilization_pct': (current_positions / config.max_positions) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating position limits: {e}")
            return {}
    
    def should_accept_new_position(self, current_positions: int, 
                                 kelly_position_size: float) -> Tuple[bool, Dict]:
        """
        Determine if we should accept a new position based on VIX limits
        
        Args:
            current_positions: Number of current positions
            kelly_position_size: Position size from Kelly calculation
            
        Returns:
            Tuple of (should_accept, position_info)
        """
        try:
            limits = self.calculate_position_limits(current_positions)
            
            # Check if we can add another position
            if not limits.get('can_add_position', False):
                return False, {
                    'reason': 'position_limit_reached',
                    'max_positions': limits.get('max_positions', 0),
                    'current_positions': current_positions,
                    'vix_regime': limits.get('risk_regime', 'unknown')
                }
            
            # Adjust Kelly position size to VIX limits if needed
            vix_position_size = limits.get('position_size_target', kelly_position_size)
            final_position_size = min(kelly_position_size, vix_position_size)
            
            # Check if we have enough capital
            available_capital = limits.get('available_capital', 0)
            if final_position_size > available_capital:
                return False, {
                    'reason': 'insufficient_capital',
                    'required': final_position_size,
                    'available': available_capital,
                    'vix_regime': limits.get('risk_regime', 'unknown')
                }
            
            return True, {
                'approved': True,
                'kelly_size': kelly_position_size,
                'vix_adjusted_size': final_position_size,
                'size_adjustment': final_position_size / kelly_position_size if kelly_position_size > 0 else 1.0,
                'vix_regime': limits.get('risk_regime', 'unknown'),
                'position_slot': current_positions + 1,
                'max_positions': limits.get('max_positions', 0)
            }
            
        except Exception as e:
            logger.error(f"Error checking position acceptance: {e}")
            return False, {'reason': 'error', 'error': str(e)}
    
    def get_optimal_position_size(self, kelly_size: float, current_positions: int) -> float:
        """
        Get optimal position size considering both Kelly and VIX constraints
        
        Args:
            kelly_size: Kelly Criterion position size
            current_positions: Number of current positions
            
        Returns:
            Optimal position size
        """
        try:
            limits = self.calculate_position_limits(current_positions)
            vix_target_size = limits.get('position_size_target', kelly_size)
            
            # Use the smaller of Kelly or VIX target
            optimal_size = min(kelly_size, vix_target_size)
            
            logger.debug(f"Position sizing: Kelly=${kelly_size:,.0f}, VIX=${vix_target_size:,.0f}, "
                        f"Optimal=${optimal_size:,.0f}")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size: {e}")
            return kelly_size
    
    def get_vix_stats(self) -> Dict:
        """Get VIX position scaling statistics"""
        return {
            'current_vix': self.current_vix,
            'current_regime': self.current_config.risk_level,
            'max_positions': self.current_config.max_positions,
            'target_position_size': self.current_config.position_size,
            'total_capital': self.total_capital,
            'vix_thresholds': {
                'low_vix': '< 15',
                'medium_vix': '15-25',
                'high_vix': '> 25'
            },
            'position_configs': {
                regime: {
                    'max_positions': config.max_positions,
                    'position_size': config.position_size,
                    'risk_level': config.risk_level
                }
                for regime, config in self.vix_configs.items()
            }
        }
    
    def log_position_decision(self, decision: Dict, symbol: str = None):
        """Log position scaling decision for tracking"""
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'vix_level': self.current_vix,
            'regime': self.current_config.risk_level,
            'decision': decision
        }
        
        self.position_history.append(log_entry)
        
        # Keep only last 100 decisions
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
        
        # Log the decision
        if decision.get('approved'):
            logger.info(f"Position approved for {symbol}: "
                       f"${decision.get('vix_adjusted_size', 0):,.0f} "
                       f"({decision.get('vix_regime', 'unknown')} VIX regime)")
        else:
            logger.info(f"Position rejected for {symbol}: "
                       f"{decision.get('reason', 'unknown')} "
                       f"({decision.get('vix_regime', 'unknown')} VIX regime)")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create VIX position scaler
    vix_scaler = VIXPositionScaler(total_capital=50000)
    
    # Test different VIX scenarios
    test_scenarios = [
        {'vix': 12, 'scenario': 'Low volatility'},
        {'vix': 18, 'scenario': 'Normal volatility'},
        {'vix': 30, 'scenario': 'High volatility'},
        {'vix': 45, 'scenario': 'Crisis volatility'}
    ]
    
    print("VIX Position Scaling Test:")
    print("=" * 50)
    
    for scenario in test_scenarios:
        vix_level = scenario['vix']
        config = vix_scaler.update_vix_level(vix_level)
        
        print(f"\n{scenario['scenario']} (VIX: {vix_level})")
        print(f"  Risk Regime: {config.risk_level}")
        print(f"  Max Positions: {config.max_positions}")
        print(f"  Position Size: ${config.position_size:,.0f}")
        
        # Test position acceptance with different current position counts
        for current_pos in [0, 3, 6, 8]:
            kelly_size = 7500  # Example Kelly size
            accept, info = vix_scaler.should_accept_new_position(current_pos, kelly_size)
            
            if accept:
                print(f"    {current_pos} positions: ✓ Accept ${info['vix_adjusted_size']:,.0f}")
            else:
                print(f"    {current_pos} positions: ✗ Reject ({info['reason']})")
    
    # Print overall stats
    print("\nVIX Scaler Stats:")
    stats = vix_scaler.get_vix_stats()
    print(f"  Current VIX: {stats['current_vix']}")
    print(f"  Current Regime: {stats['current_regime']}")
    print(f"  Max Positions: {stats['max_positions']}")