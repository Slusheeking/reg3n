#!/usr/bin/env python3

"""
Simulated Alpaca Executor for Production-Native Backtesting
Mirrors alpaca_momentum_executor.py interface without actual trading
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # ERROR
    YELLOW = '\033[93m'   # WARNING
    BLUE = '\033[94m'     # DEBUG
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # Reset to default

class SystemLogger:
    def __init__(self, name="simulated_executor"):
        self.name = name
        self.color_map = {
            'ERROR': Colors.RED,
            'WARNING': Colors.YELLOW,
            'DEBUG': Colors.BLUE,
            'INFO': Colors.WHITE
        }
        
        # Create logs directory and file
        import os
        self.log_dir = '/home/ubuntu/reg3n-1/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, 'backtesting.log')
        
    def _format_message(self, level: str, message: str, colored: bool = True) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if colored:
            color = self.color_map.get(level, Colors.WHITE)
            return f"[{timestamp}] - {color}{level}{Colors.RESET} - [{self.name}]: {message}"
        else:
            return f"[{timestamp}] - {level} - [{self.name}]: {message}"
    
    def _write_to_file(self, level: str, message: str):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(self._format_message(level, str(message), colored=False) + '\n')
        except Exception:
            pass  # Fail silently
        
    def info(self, message, extra=None):
        print(self._format_message("INFO", str(message)))
        self._write_to_file("INFO", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("INFO", f"    Extra: {extra}")
    
    def debug(self, message, extra=None):
        print(self._format_message("DEBUG", str(message)))
        self._write_to_file("DEBUG", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("DEBUG", f"    Extra: {extra}")
    
    def warning(self, message, extra=None):
        print(self._format_message("WARNING", str(message)))
        self._write_to_file("WARNING", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("WARNING", f"    Extra: {extra}")
    
    def error(self, message, extra=None):
        print(self._format_message("ERROR", str(message)))
        self._write_to_file("ERROR", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("ERROR", f"    Extra: {extra}")

logger = SystemLogger()

class OrderRequest:
    """Simulated order request structure matching production"""
    def __init__(self, symbol, qty, side, type='market', time_in_force='day', 
                 limit_price=None, stop_price=None, client_order_id=None):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.type = type
        self.time_in_force = time_in_force
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.client_order_id = client_order_id or f"{side}_{symbol}_{int(time.time() * 1000)}"

class OrderResponse:
    """Simulated order response matching production"""
    def __init__(self, order_id, symbol, status, filled_qty, avg_fill_price, 
                 timestamp, client_order_id=None):
        self.order_id = order_id
        self.symbol = symbol
        self.status = status
        self.filled_qty = filled_qty
        self.avg_fill_price = avg_fill_price
        self.timestamp = timestamp
        self.client_order_id = client_order_id

class SimulatedPosition:
    """Track simulated position state"""
    def __init__(self, symbol, qty, avg_price, timestamp):
        self.symbol = symbol
        self.qty = qty
        self.avg_price = avg_price
        self.timestamp = timestamp
        self.unrealized_pnl = 0.0
        self.market_value = qty * avg_price

class SimulatedTrade:
    """Record of simulated trade execution"""
    def __init__(self, symbol, side, qty, price, timestamp, order_type='market', prediction=None, confidence=None):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.timestamp = timestamp
        self.order_type = order_type
        self.trade_value = qty * price
        self.commission = self._calculate_commission(qty, price)
        self.trade_id = f"{symbol}_{side}_{int(timestamp * 1000)}"
        self.prediction = prediction
        self.confidence = confidence
        self.prediction_correct = None  # Will be updated when we know the outcome
        self.pnl = 0.0  # Will be updated when the trade is closed
        self.entry_trade_id = None  # Link to the corresponding entry trade for sells
        self.is_processed = False  # Track if this trade has been processed for P&L
    
    def _calculate_commission(self, qty, price):
        """Calculate realistic commission (Alpaca is commission-free for stocks)"""
        return 0.0  # Alpaca doesn't charge commissions for stock trades

class SimulatedAlpacaExecutor:
    """
    Simulated execution engine that mirrors alpaca_momentum_executor.py
    Provides realistic trade simulation without actual API calls
    """
    
    def __init__(self, initial_capital=50000, slippage_bps=2, latency_ms=50):
        # Portfolio state
        self.initial_capital = initial_capital
        self.cash_available = initial_capital
        self.portfolio_value = initial_capital
        self.positions = {}  # symbol -> SimulatedPosition
        self.daily_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Trading parameters
        self.slippage_bps = slippage_bps  # Basis points of slippage
        self.latency_ms = latency_ms  # Simulated execution latency
        
        # Order tracking
        self.pending_orders = {}  # order_id -> OrderRequest
        self.executed_orders = {}  # order_id -> OrderResponse
        self.trades = []  # List of SimulatedTrade objects
        self.order_counter = 0
        
        # Enhanced trade tracking
        self.open_positions_trades = {}  # symbol -> list of entry trades
        self.completed_trades = []  # List of completed trade pairs with P&L
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_commission = 0.0
        
        # Daily tracking for aggressive strategy
        self.daily_target = 1000  # $1000/day target
        self.daily_trades = 0
        self.current_positions = 0
        self.max_daily_positions = 20
        
        # Risk management
        self.max_position_value = initial_capital * 0.1  # Max 10% per position
        self.max_total_exposure = initial_capital * 0.8  # Max 80% total exposure
        
        logger.info(f"Simulated Alpaca Executor initialized with ${initial_capital:,} capital")
        logger.info(f"Slippage: {slippage_bps} bps, Latency: {latency_ms}ms")
    
    def _generate_order_id(self):
        """Generate unique order ID"""
        self.order_counter += 1
        return f"SIM_{int(time.time())}_{self.order_counter}"
    
    def _calculate_slippage(self, price, side, qty):
        """Calculate realistic slippage based on order size and market conditions"""
        base_slippage = price * (self.slippage_bps / 10000)  # Convert bps to decimal
        
        # Increase slippage for larger orders
        size_multiplier = min(2.0, 1.0 + (qty / 1000))  # Up to 2x for large orders
        
        # Apply slippage direction
        if side == 'buy':
            return price + (base_slippage * size_multiplier)
        else:
            return price - (base_slippage * size_multiplier)
    
    async def _simulate_execution_delay(self):
        """Simulate realistic execution latency"""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
    
    async def execute_trade(self, symbol, position_size, price, ml_prediction):
        """
        Execute trade for Polygon client integration
        Mirrors the production alpaca_momentum_executor.py interface exactly
        """
        try:
            # Skip if position size is zero or invalid
            if not position_size or abs(position_size) < 1:
                return None
            
            # Determine side based on position size
            side = 'buy' if position_size > 0 else 'sell'
            qty = abs(position_size)
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Log trade execution
            confidence = ml_prediction.get('confidence', 0.5)
            prediction = ml_prediction.get('prediction', 0.0)
            
            logger.info(f"SIMULATING TRADE: {symbol} {side} {qty} shares @ ${price:.2f} "
                       f"(confidence: {confidence:.2f}, prediction: {prediction:.2f})")
            
            # Execute the simulated order
            order_response = await self.submit_market_order(symbol, qty, side, price)
            
            return order_response
            
        except Exception as e:
            logger.error(f"Simulated trade execution failed for {symbol}: {e}")
            return None
    
    async def execute_momentum_trade(self, signal, market_data=None, current_vix=20.0):
        """
        Execute complete momentum trade with enhanced filtering
        Mirrors production alpaca_momentum_executor.py method exactly
        """
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        current_price = getattr(signal, 'current_price', 100.0)
        confidence = getattr(signal, 'confidence', 0.5)
        
        logger.info(f"Starting simulated momentum trade execution for {symbol}")
        
        try:
            # Check daily limits
            if self.daily_trades >= self.max_daily_positions:
                logger.warning(f"Daily position limit reached: {self.daily_trades}/{self.max_daily_positions}")
                return None
            
            if self.current_positions >= 20:
                logger.warning(f"Max concurrent positions reached: {self.current_positions}/20")
                return None
            
            # Calculate position size (simplified Kelly-like calculation)
            position_value = min(
                self.max_position_value,
                self.cash_available * 0.2,  # Max 20% per trade
                confidence * 4000  # Scale with confidence
            )
            
            if position_value < 1000:  # Minimum position size
                logger.warning(f"Position size too small: ${position_value:.0f}")
                return None
            
            shares = int(position_value / current_price)
            if shares < 10:  # Minimum share count
                logger.warning(f"Share count too small: {shares}")
                return None
            
            # Create simulated Kelly order package
            kelly_order_package = {
                'symbol': symbol,
                'total_qty': shares,
                'total_value': shares * current_price,
                'tier_quantities': {
                    'tier1': int(shares * 0.5),   # 50% for quick exit
                    'tier2': int(shares * 0.3),   # 30% for secondary exit
                    'tier3': int(shares * 0.2),   # 20% for trailing stop
                    'total': shares
                },
                'prices': {
                    'entry_price': current_price,  # Add actual entry price
                    'stop_loss': round(current_price * 0.995, 2),  # 0.5% stop loss
                    'tp1_target': round(current_price * 1.005, 2),  # 0.5% take profit
                    'tp2_target': round(current_price * 1.01, 2),   # 1% take profit
                    'trail_percent': 1.0
                }
            }
            
            # Submit simulated momentum trade package with prediction data
            submitted_orders = await self.submit_aggressive_momentum_package(
                kelly_package=kelly_order_package,
                prediction=getattr(signal, 'prediction', 0.0),
                confidence=confidence
            )
            
            if submitted_orders:
                # Track successful execution
                execution_result = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'order_package': kelly_order_package,
                    'submitted_orders': submitted_orders,
                    'execution_status': 'simulated',
                    'capital_deployed': kelly_order_package['total_value'],
                    'confidence': confidence,
                    'current_price': current_price
                }
                
                self.daily_trades += 1
                self.current_positions += 1
                
                logger.info(f"Simulated momentum trade executed for {symbol}: "
                           f"{shares} shares, ${kelly_order_package['total_value']:,.0f}")
                
                return execution_result
            else:
                logger.error(f"Failed to submit simulated momentum trade for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing simulated momentum trade for {symbol}: {e}")
            return None
    
    async def submit_aggressive_momentum_package(self, kelly_package, prediction=None, confidence=None):
        """
        Submit simulated bracket order with built-in stop-loss and take-profit tiers
        Mirrors production method exactly
        """
        symbol = kelly_package['symbol']
        logger.info(f"Submitting momentum package for {symbol} with prediction: {prediction}, confidence: {confidence}")
        
        try:
            # Simulate execution delay
            await self._simulate_execution_delay()
            
            # Submit main entry order with prediction data - use actual market price
            entry_price = kelly_package['prices'].get('entry_price')
            if entry_price is None:
                logger.error(f"No entry price provided in kelly_package for {symbol}")
                return []
            
            entry_order = await self.submit_market_order(
                symbol=symbol,
                qty=kelly_package['total_qty'],
                side='buy',
                current_price=entry_price,
                prediction=prediction,
                confidence=confidence
            )
            
            submitted_orders = [entry_order.order_id] if entry_order else []
            
            # Submit tier exit orders (simulated)
            if entry_order and entry_order.status == 'filled':
                # Tier 1: Quick profit taking
                tier1_order = await self.submit_limit_order(
                    symbol=symbol,
                    qty=kelly_package['tier_quantities']['tier1'],
                    limit_price=kelly_package['prices']['tp1_target'],
                    side='sell'
                )
                if tier1_order:
                    submitted_orders.append(tier1_order.order_id)
                
                # Tier 2: Secondary profit taking
                tier2_order = await self.submit_limit_order(
                    symbol=symbol,
                    qty=kelly_package['tier_quantities']['tier2'],
                    limit_price=kelly_package['prices']['tp2_target'],
                    side='sell'
                )
                if tier2_order:
                    submitted_orders.append(tier2_order.order_id)
                
                # Stop loss order
                stop_order = await self.submit_stop_order(
                    symbol=symbol,
                    qty=kelly_package['total_qty'],
                    stop_price=kelly_package['prices']['stop_loss'],
                    side='sell'
                )
                if stop_order:
                    submitted_orders.append(stop_order.order_id)
            
            logger.info(f"Simulated bracket order package submitted: {len(submitted_orders)} orders for {symbol}")
            return submitted_orders
            
        except Exception as e:
            logger.error(f"Simulated bracket order error for {symbol}: {e}")
            return []
    
    async def submit_market_order(self, symbol, qty, side, current_price=None, prediction=None, confidence=None):
        """Submit simulated market order with prediction data"""
        try:
            await self._simulate_execution_delay()
            
            order_id = self._generate_order_id()
            
            # Use provided price - this should NEVER be None in production
            if current_price is None:
                logger.error(f"No current price provided for {symbol} - this indicates a data flow bug!")
                return None
            
            # Calculate execution price with slippage
            execution_price = self._calculate_slippage(current_price, side, qty)
            
            # Check if we have enough cash for buy orders
            if side == 'buy':
                required_cash = qty * execution_price
                if required_cash > self.cash_available:
                    logger.warning(f"Insufficient cash for {symbol}: need ${required_cash:,.0f}, have ${self.cash_available:,.0f}")
                    return None
            
            # Create order response
            order_response = OrderResponse(
                order_id=order_id,
                symbol=symbol,
                status='filled',
                filled_qty=qty,
                avg_fill_price=execution_price,
                timestamp=time.time()
            )
            
            # Update portfolio state
            await self._process_fill(order_response, side)
            
            # Ensure prediction and confidence have valid values
            safe_prediction = prediction if prediction is not None else 0.0
            safe_confidence = confidence if confidence is not None else 0.5
            
            # Record trade with prediction data
            trade = SimulatedTrade(
                symbol=symbol,
                side=side,
                qty=qty,
                price=execution_price,
                timestamp=time.time(),
                order_type='market',
                prediction=safe_prediction,
                confidence=safe_confidence
            )
            
            # Log trade with prediction data
            logger.info(f"Trade recorded with prediction: {safe_prediction:.4f}, confidence: {safe_confidence:.4f}")
            self.trades.append(trade)
            self.trade_count += 1
            
            logger.info(f"✓ Simulated market order: {side} {qty} {symbol} @ ${execution_price:.2f}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Simulated market order failed: {e}")
            return None
    
    async def submit_limit_order(self, symbol, qty, limit_price, side='sell'):
        """Submit simulated limit order"""
        try:
            await self._simulate_execution_delay()
            
            order_id = self._generate_order_id()
            
            # Simulate limit order fill probability (simplified)
            fill_probability = 0.7  # 70% chance of fill for simulation
            
            if np.random.random() < fill_probability:
                status = 'filled'
                filled_qty = qty
                avg_fill_price = limit_price
                
                # Update portfolio state
                order_response = OrderResponse(
                    order_id=order_id,
                    symbol=symbol,
                    status=status,
                    filled_qty=filled_qty,
                    avg_fill_price=avg_fill_price,
                    timestamp=time.time()
                )
                
                await self._process_fill(order_response, side)
                
                # Record trade
                trade = SimulatedTrade(symbol, side, qty, avg_fill_price, time.time(), 'limit')
                self.trades.append(trade)
                
                logger.info(f"✓ Simulated limit order filled: {side} {qty} {symbol} @ ${avg_fill_price:.2f}")
            else:
                status = 'pending'
                filled_qty = 0
                avg_fill_price = 0
                
                order_response = OrderResponse(
                    order_id=order_id,
                    symbol=symbol,
                    status=status,
                    filled_qty=filled_qty,
                    avg_fill_price=avg_fill_price,
                    timestamp=time.time()
                )
                
                logger.info(f"○ Simulated limit order pending: {side} {qty} {symbol} @ ${limit_price:.2f}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Simulated limit order failed: {e}")
            return None
    
    async def submit_stop_order(self, symbol, qty, stop_price, side='sell'):
        """Submit simulated stop order"""
        try:
            await self._simulate_execution_delay()
            
            order_id = self._generate_order_id()
            
            # For simulation, assume stop orders are placed but not immediately filled
            order_response = OrderResponse(
                order_id=order_id,
                symbol=symbol,
                status='pending',
                filled_qty=0,
                avg_fill_price=0,
                timestamp=time.time()
            )
            
            logger.info(f"○ Simulated stop order placed: {side} {qty} {symbol} @ ${stop_price:.2f}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Simulated stop order failed: {e}")
            return None
    
    async def _process_fill(self, order_response, side):
        """Process order fill and update portfolio state with proper trade tracking"""
        symbol = order_response.symbol
        qty = order_response.filled_qty
        price = order_response.avg_fill_price
        
        if side == 'buy':
            # Buy order - add to position, reduce cash
            trade_value = qty * price
            self.cash_available -= trade_value
            
            if symbol in self.positions:
                # Add to existing position
                existing_pos = self.positions[symbol]
                total_qty = existing_pos.qty + qty
                total_value = (existing_pos.qty * existing_pos.avg_price) + trade_value
                new_avg_price = total_value / total_qty
                
                self.positions[symbol] = SimulatedPosition(symbol, total_qty, new_avg_price, time.time())
            else:
                # New position
                self.positions[symbol] = SimulatedPosition(symbol, qty, price, time.time())
            
            # Track this as an entry trade
            if symbol not in self.open_positions_trades:
                self.open_positions_trades[symbol] = []
            
            # Find the corresponding buy trade in our trades list and ensure it's properly tracked
            buy_trade_found = False
            for trade in reversed(self.trades):  # Search from most recent
                if (trade.symbol == symbol and trade.side == 'buy' and
                    not trade.is_processed and abs(trade.qty - qty) < 0.01):
                    # Don't match on price since slippage can cause small differences
                    self.open_positions_trades[symbol].append(trade)
                    trade.is_processed = True
                    buy_trade_found = True
                    break
            
            if not buy_trade_found:
                logger.warning(f"Could not find matching buy trade for {symbol} {qty} shares @ ${price:.2f}")
            
            logger.debug(f"Position updated: {symbol} +{qty} shares @ ${price:.2f}")
            
        elif side == 'sell':
            # Sell order - reduce position, add cash
            trade_value = qty * price
            self.cash_available += trade_value
            
            if symbol in self.positions and symbol in self.open_positions_trades:
                existing_pos = self.positions[symbol]
                
                # Match this sell with the oldest unmatched buy trades (FIFO)
                remaining_sell_qty = qty
                total_pnl = 0.0
                trades_closed = 0
                
                while remaining_sell_qty > 0 and self.open_positions_trades[symbol]:
                    entry_trade = self.open_positions_trades[symbol][0]
                    
                    # Calculate how much of this entry trade to close
                    close_qty = min(remaining_sell_qty, entry_trade.qty)
                    
                    # Calculate P&L for this portion
                    cost_basis = entry_trade.price * close_qty
                    sale_value = price * close_qty
                    pnl = sale_value - cost_basis
                    total_pnl += pnl
                    trades_closed += 1
                    
                    # Create completed trade record with all required fields
                    completed_trade = {
                        'symbol': symbol,
                        'entry_price': entry_trade.price,
                        'exit_price': price,
                        'qty': close_qty,
                        'pnl': pnl,
                        'entry_timestamp': entry_trade.timestamp,
                        'exit_timestamp': time.time(),
                        'prediction': entry_trade.prediction if entry_trade.prediction is not None else 0.0,
                        'confidence': entry_trade.confidence if entry_trade.confidence is not None else 0.5,
                        'prediction_correct': pnl > 0
                    }
                    self.completed_trades.append(completed_trade)
                    
                    # Update the entry trade
                    entry_trade.qty -= close_qty
                    if entry_trade.qty <= 0:
                        self.open_positions_trades[symbol].pop(0)
                    
                    remaining_sell_qty -= close_qty
                
                # Update performance metrics
                self.realized_pnl += total_pnl
                self.daily_pnl += total_pnl
                
                # Update win/loss counts based on overall P&L
                if total_pnl > 0:
                    self.win_count += 1
                    logger.info(f"Trade for {symbol} marked as successful, PnL: ${total_pnl:.2f}")
                elif total_pnl < 0:
                    self.loss_count += 1
                    logger.info(f"Trade for {symbol} marked as failed, PnL: ${total_pnl:.2f}")
                else:
                    # Break-even trades count as failed for conservative approach
                    self.loss_count += 1
                    logger.info(f"Trade for {symbol} marked as failed, PnL: ${total_pnl:.2f}")
                
                # Update the sell trade record with P&L - ensure we find the right trade
                sell_trade_found = False
                for trade in reversed(self.trades):
                    if (trade.symbol == symbol and trade.side == 'sell' and
                        not trade.is_processed and abs(trade.qty - qty) < 0.01):
                        trade.pnl = total_pnl
                        trade.prediction_correct = total_pnl > 0
                        trade.is_processed = True
                        sell_trade_found = True
                        break
                
                if not sell_trade_found:
                    logger.warning(f"Could not find matching sell trade for {symbol} {qty} shares @ ${price:.2f}")
                
                # Update position
                remaining_qty = existing_pos.qty - qty
                if remaining_qty <= 0:
                    # Position closed
                    del self.positions[symbol]
                    if symbol in self.open_positions_trades and not self.open_positions_trades[symbol]:
                        del self.open_positions_trades[symbol]
                    self.current_positions = max(0, self.current_positions - 1)
                    logger.debug(f"Position closed: {symbol} P&L: ${total_pnl:.2f}")
                else:
                    # Partial close
                    self.positions[symbol] = SimulatedPosition(
                        symbol, remaining_qty, existing_pos.avg_price, time.time()
                    )
                    logger.debug(f"Position reduced: {symbol} -{qty} shares, P&L: ${total_pnl:.2f}")
            else:
                # No matching position found - this shouldn't happen in normal operation
                logger.error(f"Attempted to sell {symbol} but no position or entry trades found")
                # Create a dummy completed trade to avoid missing outcomes
                completed_trade = {
                    'symbol': symbol,
                    'entry_price': price,  # Use sell price as fallback
                    'exit_price': price,
                    'qty': qty,
                    'pnl': 0.0,  # No P&L since no position
                    'entry_timestamp': time.time() - 1,  # Slightly earlier timestamp
                    'exit_timestamp': time.time(),
                    'prediction': 0.0,
                    'confidence': 0.5,
                    'prediction_correct': False
                }
                self.completed_trades.append(completed_trade)
                self.loss_count += 1
        
        # Update portfolio value
        self._update_portfolio_value()
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        position_value = sum(pos.qty * pos.avg_price for pos in self.positions.values())
        self.portfolio_value = self.cash_available + position_value
    
    async def get_account_status(self):
        """Get current simulated account status"""
        self._update_portfolio_value()
        
        daily_progress_pct = (self.daily_pnl / self.daily_target) * 100
        
        return {
            'equity': self.portfolio_value,
            'buying_power': self.cash_available,
            'cash': self.cash_available,
            'portfolio_value': self.portfolio_value,
            'num_positions': len(self.positions),
            'day_trade_count': self.daily_trades,
            'pattern_day_trader': True,
            'daily_pnl': self.daily_pnl,
            'daily_target': self.daily_target,
            'target_progress_pct': daily_progress_pct,
            'target_achieved': self.daily_pnl >= self.daily_target,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        win_rate = (self.win_count / max(1, self.win_count + self.loss_count)) * 100
        
        return {
            # Trading performance
            "total_trades": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate_pct": win_rate,
            "daily_pnl": self.daily_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            
            # Portfolio metrics
            "portfolio_value": self.portfolio_value,
            "cash_available": self.cash_available,
            "positions_count": len(self.positions),
            "daily_trades": self.daily_trades,
            
            # Strategy metrics
            "daily_target": self.daily_target,
            "target_progress_pct": (self.daily_pnl / self.daily_target) * 100,
            "target_achieved": self.daily_pnl >= self.daily_target,
            
            # Execution metrics
            "avg_slippage_bps": self.slippage_bps,
            "avg_latency_ms": self.latency_ms,
            "total_commission": self.total_commission,
            
            # Risk metrics
            "max_position_value": self.max_position_value,
            "current_exposure": self.portfolio_value - self.cash_available,
            "exposure_pct": ((self.portfolio_value - self.cash_available) / self.portfolio_value) * 100
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking for new trading day - preserve completed trades"""
        # Log current state before reset
        logger.info(f"Before reset: {len(self.completed_trades)} completed trades, {self.win_count} wins, {self.loss_count} losses")
        
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_positions = 0
        self.win_count = 0
        self.loss_count = 0
        self.realized_pnl = 0.0
        
        # DON'T reset completed trades - they need to persist for analytics
        # Only reset open positions for new day
        self.open_positions_trades = {}
        # Keep completed_trades for analytics collection
        
        logger.info(f"Daily tracking reset - Ready for new ${self.daily_target} target day")
        logger.info(f"Preserved {len(self.completed_trades)} completed trades for analytics")
    
    def get_positions(self):
        """Get current positions"""
        return {symbol: {
            'qty': pos.qty,
            'avg_price': pos.avg_price,
            'market_value': pos.market_value,
            'timestamp': pos.timestamp
        } for symbol, pos in self.positions.items()}
    
    def get_trades_history(self):
        """Get trade history with prediction correctness"""
        return [{
            'symbol': trade.symbol,
            'side': trade.side,
            'qty': trade.qty,
            'price': trade.price,
            'timestamp': trade.timestamp,
            'trade_value': trade.trade_value,
            'commission': trade.commission,
            'trade_id': trade.trade_id,
            'prediction': trade.prediction,
            'confidence': trade.confidence,
            'prediction_correct': trade.prediction_correct,
            'pnl': trade.pnl
        } for trade in self.trades]
    
    def get_completed_trades(self):
        """Get completed trade pairs with proper P&L calculation"""
        return self.completed_trades.copy()
    
    def clear_completed_trades(self):
        """Clear completed trades after they've been processed by analytics"""
        cleared_count = len(self.completed_trades)
        self.completed_trades.clear()
        logger.info(f"Cleared {cleared_count} completed trades from executor")
        return cleared_count
    
    def get_trade_outcome_summary(self):
        """Get summary of trade outcomes for debugging"""
        total_trades = len(self.trades)
        trades_with_outcomes = sum(1 for trade in self.trades if trade.prediction_correct is not None)
        completed_trades_count = len(self.completed_trades)
        
        return {
            'total_trades': total_trades,
            'trades_with_outcomes': trades_with_outcomes,
            'trades_without_outcomes': total_trades - trades_with_outcomes,
            'completed_trades': completed_trades_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_simulated_executor():
        print("Testing Simulated Alpaca Executor...")
        
        executor = SimulatedAlpacaExecutor(initial_capital=50000)
        
        # Test market order
        order = await executor.submit_market_order('AAPL', 100, 'buy', 150.0)
        print(f"Market order result: {order.status if order else 'Failed'}")
        
        # Test account status
        account = await executor.get_account_status()
        print(f"Account status: Cash=${account['cash']:,.0f}, Portfolio=${account['portfolio_value']:,.0f}")
        
        # Test performance stats
        stats = executor.get_performance_stats()
        print(f"Performance: {stats['total_trades']} trades, ${stats['daily_pnl']:.2f} P&L")
        
        print("Simulated Executor test completed")
    
    # Run test
    asyncio.run(test_simulated_executor())