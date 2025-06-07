#!/usr/bin/env python3

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, time
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger

# Project imports
from .kelly_position_sizer import KellyPositionSizer
from .alpaca_rest_api import AlpacaRESTClient
from filters import MomentumConsistencyFilter, VIXPositionScaler, EntryTimingOptimizer

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
logger = get_logger("momentum_execution_engine")

class MomentumExecutionEngine:
    """
    Complete momentum trading execution engine
    Combines Kelly position sizing with Alpaca order execution
    """
    
    def __init__(self, alpaca_client: Optional[AlpacaRESTClient] = None,
                 initial_capital: float = 50000):
        """
        Initialize enhanced momentum execution engine with filters
        
        Args:
            alpaca_client: Alpaca REST client (will create if None)
            initial_capital: Starting capital for Kelly calculations
        """
        # Initialize Alpaca client
        self.alpaca_client = alpaca_client or AlpacaRESTClient()
        
        # Initialize Kelly position sizer
        self.kelly_sizer = KellyPositionSizer(available_capital=initial_capital)
        
        # Initialize high-impact filters
        self.momentum_filter = MomentumConsistencyFilter()
        self.vix_scaler = VIXPositionScaler(total_capital=initial_capital)
        self.timing_optimizer = EntryTimingOptimizer()
        
        # Execution tracking
        self.executed_trades = []
        self.failed_trades = []
        self.total_capital_deployed = 0.0
        
        # Load configuration
        trading_config = CONFIG.get('trading', {})
        portfolio_config = trading_config.get('portfolio', {})
        strategy_config = CONFIG.get('strategy', {})
        
        # Market timing from config or defaults
        self.market_open = time(9, 30)   # 9:30 AM ET
        self.entry_cutoff = time(15, 30) # 3:30 PM ET (stop new entries)
        self.time_exit = time(15, 45)    # 3:45 PM ET (close all positions)
        
        # Strategy parameters from YAML
        self.max_positions = portfolio_config.get('max_positions', 20)
        self.signal_confidence_threshold = strategy_config.get('signal_confidence_threshold', 0.7)
        self.execution_delay_ms = strategy_config.get('execution_delay_ms', 100)
        
        logger.info("Momentum Execution Engine initialized successfully", extra={
            "component": "momentum_execution_engine",
            "action": "initialization_complete",
            "initial_capital": initial_capital,
            "kelly_safety_factor": self.kelly_sizer.SAFETY_FACTOR,
            "filters_enabled": {
                "momentum_consistency": True,
                "vix_position_scaling": True,
                "entry_timing": True
            },
            "market_timing": {
                "market_open": str(self.market_open),
                "entry_cutoff": str(self.entry_cutoff),
                "time_exit": str(self.time_exit)
            },
            "strategy_config": {
                "max_positions": self.max_positions,
                "signal_confidence_threshold": self.signal_confidence_threshold,
                "execution_delay_ms": self.execution_delay_ms
            }
        })
    
    async def execute_momentum_trade(self, signal, market_data: List[Dict] = None,
                                   current_vix: float = 20.0) -> Optional[Dict]:
        """
        Execute complete momentum trade with enhanced filtering
        
        Args:
            signal: Trading signal with prediction, confidence, symbol, current_price
            market_data: Market data for momentum consistency filter
            current_vix: Current VIX level for position scaling
            
        Returns:
            Dict with execution results or None if failed
        """
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        
        logger.info("Starting momentum trade execution", extra={
            "component": "momentum_execution_engine",
            "action": "trade_execution_start",
            "symbol": symbol,
            "prediction": getattr(signal, 'prediction', None),
            "confidence": getattr(signal, 'confidence', None),
            "current_price": getattr(signal, 'current_price', None),
            "vix_level": current_vix
        })
        
        try:
            # FILTER 1: Entry Timing Validation
            timing_valid, timing_info = self.timing_optimizer.validate_trade_entry(symbol)
            
            logger.info("Entry timing filter result", extra={
                "component": "momentum_execution_engine",
                "action": "entry_timing_filter",
                "symbol": symbol,
                "timing_valid": timing_valid,
                "timing_info": timing_info
            })
            
            if not timing_valid:
                logger.info("Entry timing blocked", extra={
                    "component": "momentum_execution_engine",
                    "action": "trade_blocked",
                    "symbol": symbol,
                    "filter": "entry_timing",
                    "reason": timing_info.get('reason', 'unknown')
                })
                return None
            
            # FILTER 2: Momentum Consistency Check (if market data available)
            if market_data:
                consistent_stocks = self.momentum_filter.filter_consistent_momentum_stocks(market_data)
                stock_symbols = [stock.get('symbol') for stock in consistent_stocks]
                momentum_passed = symbol in stock_symbols
                
                logger.info("Momentum consistency filter result", extra={
                    "component": "momentum_execution_engine",
                    "action": "momentum_consistency_filter",
                    "symbol": symbol,
                    "momentum_passed": momentum_passed,
                    "total_consistent_stocks": len(consistent_stocks),
                    "symbol_in_list": momentum_passed
                })
                
                if not momentum_passed:
                    logger.info("Momentum consistency failed", extra={
                        "component": "momentum_execution_engine",
                        "action": "trade_blocked",
                        "symbol": symbol,
                        "filter": "momentum_consistency",
                        "reason": "not in top decile for both 6M and 5M"
                    })
                    return None
                    
                logger.info("Momentum consistency passed", extra={
                    "component": "momentum_execution_engine",
                    "action": "filter_passed",
                    "symbol": symbol,
                    "filter": "momentum_consistency",
                    "reason": "top decile in both periods"
                })
            
            # FILTER 3: VIX Position Scaling
            self.vix_scaler.update_vix_level(current_vix)
            current_positions = len(self.executed_trades)
            
            # Calculate Kelly position size
            logger.info("Calculating Kelly position", extra={
                "component": "momentum_execution_engine",
                "action": "kelly_calculation",
                "symbol": symbol,
                "signal_prediction": getattr(signal, 'prediction', None),
                "signal_confidence": getattr(signal, 'confidence', None),
                "current_price": getattr(signal, 'current_price', None)
            })
            
            kelly_order_package = self.kelly_sizer.calculate_kelly_position_size(signal)
            
            if not kelly_order_package or kelly_order_package.get('is_fallback'):
                logger.warning(f"Kelly calculation failed for {symbol}, using fallback")
            
            # Apply VIX position scaling
            kelly_size = kelly_order_package['total_value']
            vix_accept, vix_info = self.vix_scaler.should_accept_new_position(current_positions, kelly_size)
            
            logger.info("VIX position scaling filter result", extra={
                "component": "momentum_execution_engine",
                "action": "vix_position_scaling_filter",
                "symbol": symbol,
                "vix_accept": vix_accept,
                "vix_info": vix_info
            })
            if not vix_accept:
                logger.info(f"VIX position scaling blocked {symbol}: {vix_info.get('reason', 'unknown')}", extra={
                    "component": "momentum_execution_engine",
                    "symbol": symbol,
                    "reason": vix_info.get('reason', 'unknown'),
                    "action": "vix_blocked"
                })
                self.vix_scaler.log_position_decision(vix_info, symbol)
                return None
            
            # Adjust position size based on VIX scaling
            final_position_size = vix_info.get('vix_adjusted_size', kelly_size)
            if final_position_size != kelly_size:
                # Recalculate order package with VIX-adjusted size
                adjustment_factor = final_position_size / kelly_size
                kelly_order_package['total_value'] = final_position_size
                kelly_order_package['total_qty'] = int(kelly_order_package['total_qty'] * adjustment_factor)
                
                # Adjust tier quantities
                for tier in kelly_order_package['tier_quantities']:
                    if tier != 'total':
                        kelly_order_package['tier_quantities'][tier] = int(
                            kelly_order_package['tier_quantities'][tier] * adjustment_factor
                        )
                
                logger.info(f"VIX scaling applied to {symbol}: "
                           f"${kelly_size:,.0f} → ${final_position_size:,.0f} "
                           f"({vix_info.get('vix_regime', 'unknown')} regime)")
            
            # Log successful filter passage
            self.vix_scaler.log_position_decision(vix_info, symbol)
            
            # Validate final position size
            if not self._validate_position_size(kelly_order_package):
                logger.warning(f"Final position size validation failed for {symbol}", extra={
                    "component": "momentum_execution_engine",
                    "symbol": symbol,
                    "issue": "position_validation_failed"
                })
                return None
            
            # Submit complete trade package to Alpaca
            logger.info(f"Submitting enhanced momentum trade package for {symbol}", extra={
                "component": "momentum_execution_engine",
                "symbol": symbol,
                "action": "submitting_trade_package"
            })
            submitted_orders = self.alpaca_client.submit_momentum_trade_package(kelly_order_package)
            
            if submitted_orders:
                # Track successful execution
                execution_result = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'order_package': kelly_order_package,
                    'submitted_orders': submitted_orders,
                    'execution_status': 'submitted',
                    'kelly_params': kelly_order_package.get('kelly_params'),
                    'capital_deployed': kelly_order_package['total_value'],
                    'filter_results': {
                        'timing_window': timing_info.get('window_name', 'unknown'),
                        'momentum_consistent': True if market_data else 'not_checked',
                        'vix_regime': vix_info.get('vix_regime', 'unknown'),
                        'vix_adjustment': vix_info.get('size_adjustment', 1.0)
                    }
                }
                
                self.executed_trades.append(execution_result)
                self.total_capital_deployed += kelly_order_package['total_value']
                
                # Update available capital
                remaining_capital = self.kelly_sizer.available_capital - self.total_capital_deployed
                self.kelly_sizer.update_available_capital(max(remaining_capital, 10000))  # Keep min $10k
                
                logger.info(f"Enhanced momentum trade executed for {symbol}: "
                           f"{kelly_order_package['total_qty']} shares, "
                           f"${kelly_order_package['total_value']:,.0f} "
                           f"({timing_info.get('window_name', 'unknown')} window, "
                           f"{vix_info.get('vix_regime', 'unknown')} VIX)")
                
                return execution_result
            else:
                # Track failed execution
                self.failed_trades.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'reason': 'order_submission_failed',
                    'order_package': kelly_order_package
                })
                
                logger.error(f"Failed to submit momentum trade for {symbol}", extra={
                    "component": "momentum_execution_engine",
                    "symbol": symbol,
                    "error": "trade_submission_failed"
                })
                return None
                
        except Exception as e:
            logger.error(f"Error executing momentum trade for {symbol}: {e}", extra={
                "component": "momentum_execution_engine",
                "symbol": symbol,
                "error": str(e)
            })
            self.failed_trades.append({
                'symbol': getattr(signal, 'symbol', 'UNKNOWN'),
                'timestamp': datetime.now(),
                'reason': f'execution_error: {str(e)}',
                'signal': signal
            })
            return None
    
    async def execute_multiple_trades(self, signals: List) -> Dict[str, Any]:
        """
        Execute multiple momentum trades in sequence
        
        Args:
            signals: List of trading signals
            
        Returns:
            Dict with execution summary
        """
        execution_summary = {
            'total_signals': len(signals),
            'successful_executions': 0,
            'failed_executions': 0,
            'total_capital_deployed': 0.0,
            'executed_symbols': [],
            'failed_symbols': []
        }
        
        logger.info(f"Executing {len(signals)} momentum trades")
        
        for signal in signals:
            result = await self.execute_momentum_trade(signal)
            
            if result:
                execution_summary['successful_executions'] += 1
                execution_summary['total_capital_deployed'] += result['capital_deployed']
                execution_summary['executed_symbols'].append(result['symbol'])
            else:
                execution_summary['failed_executions'] += 1
                execution_summary['failed_symbols'].append(getattr(signal, 'symbol', 'UNKNOWN'))
            
            # Small delay between executions to avoid rate limits
            await asyncio.sleep(0.1)
        
        logger.info(f"Execution complete: {execution_summary['successful_executions']}/{len(signals)} trades executed, "
                   f"${execution_summary['total_capital_deployed']:,.0f} deployed", extra={
            "component": "momentum_execution_engine",
            "successful_executions": execution_summary['successful_executions'],
            "total_signals": len(signals),
            "capital_deployed": execution_summary['total_capital_deployed'],
            "action": "execution_complete"
        })
        
        return execution_summary
    
    def _is_valid_entry_time(self) -> bool:
        """Check if current time is valid for new entries"""
        current_time = datetime.now().time()
        return self.market_open <= current_time <= self.entry_cutoff
    
    def _validate_position_size(self, order_package: Dict) -> bool:
        """Validate position size before execution"""
        try:
            # Check minimum position value
            if order_package['total_value'] < 1000:
                logger.warning(f"Position value too small: ${order_package['total_value']:.0f}", extra={
                    "component": "momentum_execution_engine",
                    "position_value": order_package['total_value'],
                    "validation": "value_too_small"
                })
                return False
            
            # Check maximum position value
            if order_package['total_value'] > 10000:
                logger.warning(f"Position value too large: ${order_package['total_value']:.0f}", extra={
                    "component": "momentum_execution_engine",
                    "position_value": order_package['total_value'],
                    "validation": "value_too_large"
                })
                return False
            
            # Check minimum shares
            if order_package['total_qty'] < 10:
                logger.warning(f"Share quantity too small: {order_package['total_qty']}", extra={
                    "component": "momentum_execution_engine",
                    "total_qty": order_package['total_qty'],
                    "validation": "qty_too_small"
                })
                return False
            
            # Check capital allocation
            if order_package['capital_allocation_pct'] > 20:  # Max 20% per position
                logger.warning(f"Capital allocation too high: {order_package['capital_allocation_pct']:.1f}%", extra={
                    "component": "momentum_execution_engine",
                    "allocation_pct": order_package['capital_allocation_pct'],
                    "validation": "allocation_too_high"
                })
                return False
            
            # Check tier quantities
            tier_qtys = order_package['tier_quantities']
            if tier_qtys['tier1'] < 1 or tier_qtys['tier2'] < 1 or tier_qtys['tier3'] < 1:
                logger.warning(f"Tier quantities too small: {tier_qtys}", extra={
                    "component": "momentum_execution_engine",
                    "tier_qtys": tier_qtys,
                    "validation": "tier_qtys_too_small"
                })
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Position validation error: {e}", extra={
                "component": "momentum_execution_engine",
                "error": str(e),
                "validation": "validation_error"
            })
            return False
    
    async def get_account_status(self) -> Dict[str, Any]:
        """Get current account status and available capital"""
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()
            
            if account and positions is not None:
                account_status = {
                    'equity': float(account.get('equity', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'cash': float(account.get('cash', 0)),
                    'portfolio_value': float(account.get('portfolio_value', 0)),
                    'num_positions': len(positions),
                    'day_trade_count': int(account.get('daytrade_count', 0)),
                    'pattern_day_trader': account.get('pattern_day_trader', False)
                }
                
                # Update Kelly sizer with current buying power
                if account_status['buying_power'] > 0:
                    self.kelly_sizer.update_available_capital(account_status['buying_power'])
                
                return account_status
            else:
                logger.error("Failed to get account information", extra={
                    "component": "momentum_execution_engine",
                    "error": "account_info_failed"
                })
                return {}
                
        except Exception as e:
            logger.error(f"Error getting account status: {e}", extra={
                "component": "momentum_execution_engine",
                "error": str(e)
            })
            return {}
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics"""
        total_trades = len(self.executed_trades) + len(self.failed_trades)
        success_rate = (len(self.executed_trades) / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'execution_summary': {
                'total_trades_attempted': total_trades,
                'successful_executions': len(self.executed_trades),
                'failed_executions': len(self.failed_trades),
                'success_rate_pct': success_rate,
                'total_capital_deployed': self.total_capital_deployed
            },
            'kelly_stats': self.kelly_sizer.get_position_sizing_stats(),
            'alpaca_stats': self.alpaca_client.get_stats(),
            'recent_executions': self.executed_trades[-5:] if self.executed_trades else [],
            'recent_failures': self.failed_trades[-5:] if self.failed_trades else []
        }
    
    async def close_all_positions_at_time_exit(self) -> Dict[str, Any]:
        """Close all positions at time exit (3:45 PM ET)"""
        try:
            current_time = datetime.now().time()
            
            if current_time < self.time_exit:
                logger.info(f"Time exit not reached yet (current: {current_time}, exit: {self.time_exit})", extra={
                    "component": "momentum_execution_engine",
                    "current_time": str(current_time),
                    "exit_time": str(self.time_exit),
                    "action": "time_exit_check"
                })
                return {'status': 'not_time_yet'}
            
            logger.info("Time exit reached - closing all positions", extra={
                "component": "momentum_execution_engine",
                "action": "time_exit_triggered"
            })
            
            # Get current positions
            positions = self.alpaca_client.get_positions()
            
            if not positions:
                logger.info("No positions to close", extra={
                    "component": "momentum_execution_engine",
                    "action": "no_positions_to_close"
                })
                return {'status': 'no_positions', 'closed_positions': 0}
            
            # Close all positions
            closed_positions = []
            for position in positions:
                try:
                    result = self.alpaca_client.close_position(position.symbol)
                    if result:
                        closed_positions.append(position.symbol)
                        logger.info(f"Closed position: {position.symbol}", extra={
                            "component": "momentum_execution_engine",
                            "symbol": position.symbol,
                            "action": "position_closed"
                        })
                    else:
                        logger.error(f"Failed to close position: {position.symbol}", extra={
                            "component": "momentum_execution_engine",
                            "symbol": position.symbol,
                            "error": "position_close_failed"
                        })
                except Exception as e:
                    logger.error(f"Error closing position {position.symbol}: {e}", extra={
                        "component": "momentum_execution_engine",
                        "symbol": position.symbol,
                        "error": str(e)
                    })
            
            return {
                'status': 'completed',
                'total_positions': len(positions),
                'closed_positions': len(closed_positions),
                'closed_symbols': closed_positions
            }
            
        except Exception as e:
            logger.error(f"Error during time exit: {e}", extra={
                "component": "momentum_execution_engine",
                "error": str(e),
                "action": "time_exit_error"
            })
            return {'status': 'error', 'error': str(e)}

# Example usage and testing
async def main():
    """Example usage of momentum execution engine"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create execution engine
        engine = MomentumExecutionEngine(initial_capital=50000)
        
        # Get account status
        account_status = await engine.get_account_status()
        logger.info(f"Account status: {account_status}", extra={
            "component": "momentum_execution_engine",
            "account_status": account_status
        })
        
        # Mock signal for testing
        class MockSignal:
            def __init__(self, symbol, prediction, confidence, current_price):
                self.symbol = symbol
                self.prediction = prediction
                self.confidence = confidence
                self.current_price = current_price
        
        # Test signals
        test_signals = [
            MockSignal("AAPL", 0.8, 0.9, 150.0),   # Strong bullish
            MockSignal("MSFT", 0.6, 0.7, 300.0),   # Medium bullish
        ]
        
        # Execute trades (in paper trading mode)
        execution_summary = await engine.execute_multiple_trades(test_signals)
        logger.info(f"Execution summary: {execution_summary}", extra={
            "component": "momentum_execution_engine",
            "execution_summary": execution_summary
        })
        
        # Get execution stats
        stats = engine.get_execution_stats()
        logger.info(f"Execution stats: {stats}", extra={
            "component": "momentum_execution_engine",
            "execution_stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error in main: {e}", extra={
            "component": "momentum_execution_engine",
            "error": str(e)
        })

if __name__ == "__main__":
    asyncio.run(main())