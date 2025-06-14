"""
Order Management System

Intelligent order execution with risk controls, slippage optimization, and performance tracking.
Features:
- Smart order routing and execution
- Real-time position tracking
- Risk limit enforcement
- Execution cost analysis
- Order book analysis and timing
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from ..data.alpaca_client import AlpacaClient, Order, OrderSide, OrderType, TimeInForce, OrderStatus, Position
from ..data.data_pipeline import DataPipeline, MarketData, Quote, Trade

class ExecutionStyle(Enum):
    AGGRESSIVE = "aggressive"  # Market orders, immediate execution
    PASSIVE = "passive"       # Limit orders, better prices
    SMART = "smart"          # AI-driven execution optimization
    TWAP = "twap"           # Time-weighted average price
    VWAP = "vwap"           # Volume-weighted average price

@dataclass
class OrderRequest:
    """Order request with execution parameters"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    execution_style: ExecutionStyle = ExecutionStyle.SMART
    max_slippage_bps: int = 50  # Maximum slippage in basis points
    urgency: str = "normal"     # low, normal, high
    client_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['time_in_force'] = self.time_in_force.value
        data['execution_style'] = self.execution_style.value
        return data

@dataclass
class ExecutionReport:
    """Execution performance report"""
    order_id: str
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    avg_fill_price: float
    benchmark_price: float
    slippage_bps: float
    execution_time_ms: float
    market_impact_bps: float
    commission: float
    total_cost: float
    execution_quality: str  # excellent, good, fair, poor
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OrderManager:
    """
    Intelligent order management system
    
    Features:
    - Smart execution algorithms
    - Real-time risk monitoring
    - Execution cost analysis
    - Position tracking
    - Market impact minimization
    """
    
    def __init__(self, 
                 alpaca_client: AlpacaClient,
                 data_pipeline: DataPipeline,
                 max_position_size: float = 100000,  # Max position value
                 max_order_value: float = 50000,     # Max single order value
                 max_daily_loss: float = 5000):      # Max daily loss limit
        
        self.alpaca = alpaca_client
        self.data_pipeline = data_pipeline
        
        # Risk limits
        self.max_position_size = max_position_size
        self.max_order_value = max_order_value
        self.max_daily_loss = max_daily_loss
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.execution_reports: List[ExecutionReport] = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
        self.daily_trades = 0
        
        # Market data for execution decisions
        self.market_data: Dict[str, MarketData] = {}
        self.order_books: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Set up data callbacks
        self._setup_market_data_callbacks()
    
    def _setup_market_data_callbacks(self):
        """Set up market data callbacks for execution decisions"""
        
        async def handle_quote(quote):
            """Update market data with latest quotes"""
            self.market_data[quote.symbol] = MarketData(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                price=(quote.bid + quote.ask) / 2,
                volume=0,  # Not available in quotes
                bid=quote.bid,
                ask=quote.ask,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                spread=quote.spread,
                source="polygon_quote"
            )
            
            # Update order book
            self.order_books[quote.symbol] = {
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'spread': quote.spread,
                'timestamp': quote.timestamp
            }
        
        async def handle_trade(trade):
            """Update market data with latest trades"""
            self.market_data[trade.symbol] = MarketData(
                symbol=trade.symbol,
                timestamp=trade.timestamp,
                price=trade.price,
                volume=trade.size,
                source="polygon_trade"
            )
        
        # Register callbacks
        self.data_pipeline.add_callback('quote', handle_quote)
        self.data_pipeline.add_callback('trade', handle_trade)
    
    async def submit_order(self, request: OrderRequest) -> Optional[Order]:
        """
        Submit order with intelligent execution
        
        Args:
            request: Order request parameters
            
        Returns:
            Order object if successful, None if rejected
        """
        try:
            # Pre-trade risk checks
            if not await self._pre_trade_risk_check(request):
                self.logger.warning(f"Order rejected by risk check: {request.symbol}")
                return None
            
            # Get optimal execution parameters
            execution_params = await self._optimize_execution(request)
            
            # Submit order to broker
            order = await self.alpaca.rest.submit_order(
                symbol=request.symbol,
                qty=request.quantity,
                side=request.side,
                order_type=execution_params['order_type'],
                time_in_force=request.time_in_force,
                limit_price=execution_params.get('limit_price'),
                stop_price=execution_params.get('stop_price'),
                extended_hours=False,
                client_order_id=request.client_order_id
            )
            
            # Track order
            self.active_orders[order.id] = order
            
            # Start monitoring
            asyncio.create_task(self._monitor_order(order, request))
            
            self.logger.info(f"Order submitted: {order.id} {request.symbol} {request.side.value} {request.quantity}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        try:
            await self.alpaca.rest.cancel_order(order_id)
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all orders for symbol or all orders"""
        try:
            if symbol:
                cancelled = 0
                for order_id, order in list(self.active_orders.items()):
                    if order.symbol == symbol:
                        if await self.cancel_order(order_id):
                            cancelled += 1
                return cancelled
            else:
                await self.alpaca.rest.cancel_all_orders()
                cancelled = len(self.active_orders)
                self.active_orders.clear()
                return cancelled
                
        except Exception as e:
            self.logger.error(f"Failed to cancel orders: {e}")
            return 0
    
    async def _pre_trade_risk_check(self, request: OrderRequest) -> bool:
        """Pre-trade risk validation"""
        try:
            # Get current account info
            account = await self.alpaca.rest.get_account()
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
                return False
            
            # Get current price for value calculation
            current_price = self.data_pipeline.get_latest_price(request.symbol)
            if not current_price:
                self.logger.warning(f"No current price available for {request.symbol}")
                return False
            
            # Check order value limit
            order_value = request.quantity * current_price
            if order_value > self.max_order_value:
                self.logger.warning(f"Order value ${order_value:.2f} exceeds limit ${self.max_order_value}")
                return False
            
            # Check position size limit
            current_position = await self.alpaca.rest.get_position(request.symbol)
            current_qty = current_position.qty if current_position else 0
            
            if request.side == OrderSide.BUY:
                new_qty = current_qty + request.quantity
            else:
                new_qty = current_qty - request.quantity
            
            new_position_value = abs(new_qty * current_price)
            if new_position_value > self.max_position_size:
                self.logger.warning(f"Position value ${new_position_value:.2f} would exceed limit ${self.max_position_size}")
                return False
            
            # Check buying power
            required_buying_power = order_value * 0.5  # Assume 50% margin requirement
            if required_buying_power > account.buying_power:
                self.logger.warning(f"Insufficient buying power: ${account.buying_power:.2f} < ${required_buying_power:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk check failed: {e}")
            return False
    
    async def _optimize_execution(self, request: OrderRequest) -> Dict[str, Any]:
        """Optimize execution parameters based on market conditions"""
        
        # Get market data
        market_data = self.market_data.get(request.symbol)
        order_book = self.order_books.get(request.symbol)
        
        # Default to request parameters
        params = {
            'order_type': request.order_type,
            'limit_price': request.limit_price,
            'stop_price': request.stop_price
        }
        
        if not market_data or not order_book:
            return params
        
        # Execution style optimization
        if request.execution_style == ExecutionStyle.AGGRESSIVE:
            # Use market orders for immediate execution
            params['order_type'] = OrderType.MARKET
            
        elif request.execution_style == ExecutionStyle.PASSIVE:
            # Use limit orders at or better than current market
            params['order_type'] = OrderType.LIMIT
            
            if request.side == OrderSide.BUY:
                # Bid for better price
                params['limit_price'] = order_book['bid']
            else:
                # Ask for better price
                params['limit_price'] = order_book['ask']
                
        elif request.execution_style == ExecutionStyle.SMART:
            # AI-driven optimization based on market conditions
            params = await self._smart_execution_optimization(request, market_data, order_book)
        
        return params
    
    async def _smart_execution_optimization(self, 
                                          request: OrderRequest,
                                          market_data: MarketData,
                                          order_book: Dict) -> Dict[str, Any]:
        """Smart execution using AI and market microstructure analysis"""
        
        # Calculate market conditions
        spread_bps = (order_book['spread'] / market_data.price) * 10000
        liquidity_ratio = min(order_book['bid_size'], order_book['ask_size']) / request.quantity
        
        # Market impact estimation
        estimated_impact_bps = self._estimate_market_impact(request, order_book)
        
        # Decision logic
        if request.urgency == "high" or estimated_impact_bps < 5:
            # High urgency or low impact - use market order
            return {'order_type': OrderType.MARKET}
        
        elif spread_bps > 20 or liquidity_ratio < 0.5:
            # Wide spread or low liquidity - use aggressive limit
            if request.side == OrderSide.BUY:
                limit_price = order_book['ask'] - (order_book['spread'] * 0.25)
            else:
                limit_price = order_book['bid'] + (order_book['spread'] * 0.25)
            
            return {
                'order_type': OrderType.LIMIT,
                'limit_price': limit_price
            }
        
        else:
            # Normal conditions - use passive limit
            if request.side == OrderSide.BUY:
                limit_price = order_book['bid'] + (order_book['spread'] * 0.1)
            else:
                limit_price = order_book['ask'] - (order_book['spread'] * 0.1)
            
            return {
                'order_type': OrderType.LIMIT,
                'limit_price': limit_price
            }
    
    def _estimate_market_impact(self, request: OrderRequest, order_book: Dict) -> float:
        """Estimate market impact in basis points"""
        
        # Simple market impact model
        # Impact = k * (order_size / average_size)^0.5 * spread
        
        if request.side == OrderSide.BUY:
            available_size = order_book['ask_size']
        else:
            available_size = order_book['bid_size']
        
        if available_size == 0:
            return 100  # High impact if no liquidity
        
        size_ratio = request.quantity / available_size
        spread_bps = (order_book['spread'] / ((order_book['bid'] + order_book['ask']) / 2)) * 10000
        
        # Market impact model: impact increases with square root of size ratio
        impact_bps = 0.5 * (size_ratio ** 0.5) * spread_bps
        
        return min(impact_bps, 100)  # Cap at 100 bps
    
    async def _monitor_order(self, order: Order, request: OrderRequest):
        """Monitor order execution and generate reports"""
        
        start_time = time.time()
        benchmark_price = self.data_pipeline.get_latest_price(order.symbol)
        
        try:
            while order.id in self.active_orders:
                # Check order status
                updated_order = await self.alpaca.rest.get_order(order.id)
                self.active_orders[order.id] = updated_order
                
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED]:
                    # Order completed
                    del self.active_orders[order.id]
                    self.completed_orders.append(updated_order)
                    
                    # Generate execution report
                    if updated_order.status == OrderStatus.FILLED:
                        report = await self._generate_execution_report(
                            updated_order, request, benchmark_price, start_time
                        )
                        self.execution_reports.append(report)
                        
                        # Update daily stats
                        self.daily_volume += updated_order.filled_qty * float(report.avg_fill_price)
                        self.daily_trades += 1
                    
                    break
                
                # Wait before next check
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error monitoring order {order.id}: {e}")
    
    async def _generate_execution_report(self, 
                                       order: Order,
                                       request: OrderRequest, 
                                       benchmark_price: float,
                                       start_time: float) -> ExecutionReport:
        """Generate detailed execution report"""
        
        # Calculate execution metrics
        avg_fill_price = float(order.filled_qty * benchmark_price) / order.filled_qty if order.filled_qty > 0 else 0
        
        # Slippage calculation
        if request.side == OrderSide.BUY:
            slippage_bps = ((avg_fill_price - benchmark_price) / benchmark_price) * 10000
        else:
            slippage_bps = ((benchmark_price - avg_fill_price) / benchmark_price) * 10000
        
        # Execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Market impact (simplified)
        market_impact_bps = abs(slippage_bps) * 0.7  # Assume 70% of slippage is market impact
        
        # Commission (estimate)
        commission = max(order.filled_qty * 0.005, 1.0)  # $0.005 per share, min $1
        
        # Total cost
        total_cost = commission + (abs(slippage_bps) / 10000 * order.filled_qty * avg_fill_price)
        
        # Execution quality
        if abs(slippage_bps) < 5:
            quality = "excellent"
        elif abs(slippage_bps) < 15:
            quality = "good"
        elif abs(slippage_bps) < 30:
            quality = "fair"
        else:
            quality = "poor"
        
        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            requested_qty=request.quantity,
            filled_qty=order.filled_qty,
            avg_fill_price=avg_fill_price,
            benchmark_price=benchmark_price,
            slippage_bps=slippage_bps,
            execution_time_ms=execution_time_ms,
            market_impact_bps=market_impact_bps,
            commission=commission,
            total_cost=total_cost,
            execution_quality=quality
        )
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        return await self.alpaca.rest.get_positions()
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return await self.alpaca.rest.get_position(symbol)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get active orders"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        
        if not self.execution_reports:
            return {}
        
        reports = self.execution_reports
        
        # Calculate aggregate stats
        avg_slippage = np.mean([r.slippage_bps for r in reports])
        avg_execution_time = np.mean([r.execution_time_ms for r in reports])
        total_commission = sum([r.commission for r in reports])
        total_cost = sum([r.total_cost for r in reports])
        
        # Quality distribution
        quality_counts = {}
        for report in reports:
            quality_counts[report.execution_quality] = quality_counts.get(report.execution_quality, 0) + 1
        
        return {
            'total_orders': len(reports),
            'avg_slippage_bps': avg_slippage,
            'avg_execution_time_ms': avg_execution_time,
            'total_commission': total_commission,
            'total_execution_cost': total_cost,
            'daily_volume': self.daily_volume,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'execution_quality_distribution': quality_counts,
            'active_orders_count': len(self.active_orders)
        }

# Example usage
async def example_usage():
    """Example of order management system usage"""
    
    from ..data.alpaca_client import AlpacaClient
    from ..data.data_pipeline import DataPipeline
    
    # Initialize components
    alpaca = AlpacaClient("API_KEY", "SECRET_KEY", paper_trading=True)
    pipeline = DataPipeline(polygon_api_key="POLYGON_KEY")
    
    # Initialize order manager
    order_manager = OrderManager(alpaca, pipeline)
    
    try:
        # Start systems
        await alpaca.start()
        await pipeline.start(['AAPL', 'TSLA', 'SPY'])
        
        print("🚀 Order management system started")
        
        # Example order requests
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                execution_style=ExecutionStyle.SMART,
                urgency="normal"
            ),
            OrderRequest(
                symbol="TSLA", 
                side=OrderSide.SELL,
                quantity=5,
                execution_style=ExecutionStyle.PASSIVE,
                urgency="low"
            )
        ]
        
        # Submit orders
        for request in orders:
            order = await order_manager.submit_order(request)
            if order:
                print(f"📝 Order submitted: {order.id} {request.symbol}")
            else:
                print(f"❌ Order rejected: {request.symbol}")
        
        # Monitor for a while
        await asyncio.sleep(30)
        
        # Get stats
        stats = order_manager.get_execution_stats()
        print(f"\n📊 Execution Stats:")
        print(f"   Total Orders: {stats.get('total_orders', 0)}")
        print(f"   Avg Slippage: {stats.get('avg_slippage_bps', 0):.1f} bps")
        print(f"   Avg Execution Time: {stats.get('avg_execution_time_ms', 0):.1f} ms")
        print(f"   Active Orders: {stats.get('active_orders_count', 0)}")
        
        # Get positions
        positions = await order_manager.get_positions()
        print(f"\n💼 Current Positions:")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.qty} shares @ ${pos.current_price:.2f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await alpaca.stop()
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())
