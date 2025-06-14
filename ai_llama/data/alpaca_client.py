"""
Alpaca Trading API Client

Ultra-fast trade execution and portfolio management with Alpaca Markets.
Features:
- Real-time order execution
- Portfolio management and tracking
- Risk monitoring and position limits
- Paper trading and live trading support
- WebSocket streaming for account updates
"""

import asyncio
import aiohttp
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"    # Good till canceled
    IOC = "ioc"    # Immediate or cancel
    FOK = "fok"    # Fill or kill

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PENDING_REVIEW = "pending_review"
    HELD = "held"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"

@dataclass
class AlpacaConfig:
    """Configuration for Alpaca client"""
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    data_url: str = "https://data.alpaca.markets"
    websocket_url: str = "wss://stream.data.alpaca.markets/v2"
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: int = 30
    paper_trading: bool = True

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    qty: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    lastday_price: float
    change_today: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Order:
    """Trading order"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    filled_qty: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    replaced_at: Optional[datetime] = None
    replaced_by: Optional[str] = None
    replaces: Optional[str] = None
    asset_id: Optional[str] = None
    asset_class: str = "us_equity"
    notional: Optional[float] = None
    qty_to_fill: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    hwm: Optional[float] = None
    extended_hours: bool = False
    legs: Optional[List] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enum values to strings
        data['side'] = self.side.value if isinstance(self.side, OrderSide) else self.side
        data['order_type'] = self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type
        data['status'] = self.status.value if isinstance(self.status, OrderStatus) else self.status
        data['time_in_force'] = self.time_in_force.value if isinstance(self.time_in_force, TimeInForce) else self.time_in_force
        return data

@dataclass
class Account:
    """Account information"""
    id: str
    account_number: str
    status: str
    currency: str
    buying_power: float
    regt_buying_power: float
    daytrading_buying_power: float
    effective_buying_power: float
    cash: float
    accrued_fees: float
    pending_transfer_out: float
    pending_transfer_in: float
    portfolio_value: float
    pattern_day_trader: bool
    trade_suspended_by_user: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    shorting_enabled: bool
    long_market_value: float
    short_market_value: float
    equity: float
    last_equity: float
    multiplier: str
    initial_margin: float
    maintenance_margin: float
    last_maintenance_margin: float
    sma: float
    daytrade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AlpacaRESTClient:
    """
    High-performance Alpaca REST API client
    
    Features:
    - Async HTTP requests for maximum throughput
    - Comprehensive order management
    - Portfolio tracking and risk monitoring
    - Paper and live trading support
    """
    
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Authentication headers
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key,
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers
        )
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          data: Dict[str, Any] = None,
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated API request with error handling"""
        if not self.session:
            await self.connect()
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url, params=params) as response:
                        return await self._handle_response(response)
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data, params=params) as response:
                        return await self._handle_response(response)
                elif method.upper() == "PATCH":
                    async with self.session.patch(url, json=data, params=params) as response:
                        return await self._handle_response(response)
                elif method.upper() == "DELETE":
                    async with self.session.delete(url, params=params) as response:
                        return await self._handle_response(response)
                        
            except Exception as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise Exception(f"Failed to make request to {endpoint} after {self.config.max_retries} attempts")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle API response"""
        if response.status in [200, 201, 204]:
            if response.content_type == 'application/json':
                return await response.json()
            else:
                return {"status": "success"}
        elif response.status == 401:
            raise Exception("Authentication failed - check API keys")
        elif response.status == 403:
            raise Exception("Access forbidden - check permissions")
        elif response.status == 429:
            # Rate limited
            retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
            self.logger.warning(f"Rate limited, waiting {retry_after}s")
            await asyncio.sleep(retry_after)
            raise Exception("Rate limited")
        else:
            error_text = await response.text()
            raise Exception(f"API error {response.status}: {error_text}")
    
    # Account Management
    async def get_account(self) -> Account:
        """Get account information"""
        data = await self._make_request("GET", "/v2/account")
        
        return Account(
            id=data['id'],
            account_number=data['account_number'],
            status=data['status'],
            currency=data['currency'],
            buying_power=float(data['buying_power']),
            regt_buying_power=float(data['regt_buying_power']),
            daytrading_buying_power=float(data['daytrading_buying_power']),
            effective_buying_power=float(data['effective_buying_power']),
            cash=float(data['cash']),
            accrued_fees=float(data['accrued_fees']),
            pending_transfer_out=float(data.get('pending_transfer_out', 0)),
            pending_transfer_in=float(data.get('pending_transfer_in', 0)),
            portfolio_value=float(data['portfolio_value']),
            pattern_day_trader=data['pattern_day_trader'],
            trade_suspended_by_user=data['trade_suspended_by_user'],
            trading_blocked=data['trading_blocked'],
            transfers_blocked=data['transfers_blocked'],
            account_blocked=data['account_blocked'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            shorting_enabled=data['shorting_enabled'],
            long_market_value=float(data['long_market_value']),
            short_market_value=float(data['short_market_value']),
            equity=float(data['equity']),
            last_equity=float(data['last_equity']),
            multiplier=data['multiplier'],
            initial_margin=float(data['initial_margin']),
            maintenance_margin=float(data['maintenance_margin']),
            last_maintenance_margin=float(data['last_maintenance_margin']),
            sma=float(data['sma']),
            daytrade_count=int(data['daytrade_count'])
        )
    
    # Position Management
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        data = await self._make_request("GET", "/v2/positions")
        
        positions = []
        for pos_data in data:
            position = Position(
                symbol=pos_data['symbol'],
                qty=float(pos_data['qty']),
                side=pos_data['side'],
                market_value=float(pos_data['market_value']),
                cost_basis=float(pos_data['cost_basis']),
                unrealized_pl=float(pos_data['unrealized_pl']),
                unrealized_plpc=float(pos_data['unrealized_plpc']),
                current_price=float(pos_data['current_price']),
                lastday_price=float(pos_data['lastday_price']),
                change_today=float(pos_data['change_today'])
            )
            positions.append(position)
        
        return positions
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            data = await self._make_request("GET", f"/v2/positions/{symbol}")
            
            return Position(
                symbol=data['symbol'],
                qty=float(data['qty']),
                side=data['side'],
                market_value=float(data['market_value']),
                cost_basis=float(data['cost_basis']),
                unrealized_pl=float(data['unrealized_pl']),
                unrealized_plpc=float(data['unrealized_plpc']),
                current_price=float(data['current_price']),
                lastday_price=float(data['lastday_price']),
                change_today=float(data['change_today'])
            )
        except Exception:
            # Position doesn't exist
            return None
    
    async def close_position(self, symbol: str, qty: Optional[float] = None) -> Order:
        """Close position (or partial position)"""
        endpoint = f"/v2/positions/{symbol}"
        params = {}
        if qty is not None:
            params['qty'] = str(qty)
        
        data = await self._make_request("DELETE", endpoint, params=params)
        return self._parse_order(data)
    
    async def close_all_positions(self, cancel_orders: bool = True) -> List[Order]:
        """Close all positions"""
        params = {'cancel_orders': str(cancel_orders).lower()}
        data = await self._make_request("DELETE", "/v2/positions", params=params)
        
        orders = []
        for order_data in data:
            orders.append(self._parse_order(order_data))
        
        return orders
    
    # Order Management
    async def submit_order(self,
                         symbol: str,
                         qty: float,
                         side: OrderSide,
                         order_type: OrderType = OrderType.MARKET,
                         time_in_force: TimeInForce = TimeInForce.DAY,
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         trail_price: Optional[float] = None,
                         trail_percent: Optional[float] = None,
                         extended_hours: bool = False,
                         client_order_id: Optional[str] = None) -> Order:
        """Submit a new order"""
        
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": time_in_force.value,
            "extended_hours": extended_hours
        }
        
        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)
        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)
        if trail_price is not None:
            order_data["trail_price"] = str(trail_price)
        if trail_percent is not None:
            order_data["trail_percent"] = str(trail_percent)
        if client_order_id is not None:
            order_data["client_order_id"] = client_order_id
        
        data = await self._make_request("POST", "/v2/orders", data=order_data)
        return self._parse_order(data)
    
    async def get_orders(self, 
                        status: Optional[str] = None,
                        limit: int = 50,
                        after: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        direction: str = "desc",
                        nested: bool = True,
                        symbols: Optional[List[str]] = None) -> List[Order]:
        """Get orders with filters"""
        
        params = {
            "limit": limit,
            "direction": direction,
            "nested": str(nested).lower()
        }
        
        if status:
            params["status"] = status
        if after:
            params["after"] = after.isoformat()
        if until:
            params["until"] = until.isoformat()
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        data = await self._make_request("GET", "/v2/orders", params=params)
        
        orders = []
        for order_data in data:
            orders.append(self._parse_order(order_data))
        
        return orders
    
    async def get_order(self, order_id: str) -> Order:
        """Get specific order by ID"""
        data = await self._make_request("GET", f"/v2/orders/{order_id}")
        return self._parse_order(data)
    
    async def cancel_order(self, order_id: str) -> None:
        """Cancel specific order"""
        await self._make_request("DELETE", f"/v2/orders/{order_id}")
    
    async def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        await self._make_request("DELETE", "/v2/orders")
    
    async def replace_order(self,
                          order_id: str,
                          qty: Optional[float] = None,
                          time_in_force: Optional[TimeInForce] = None,
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          trail: Optional[float] = None,
                          client_order_id: Optional[str] = None) -> Order:
        """Replace an existing order"""
        
        replace_data = {}
        
        if qty is not None:
            replace_data["qty"] = str(qty)
        if time_in_force is not None:
            replace_data["time_in_force"] = time_in_force.value
        if limit_price is not None:
            replace_data["limit_price"] = str(limit_price)
        if stop_price is not None:
            replace_data["stop_price"] = str(stop_price)
        if trail is not None:
            replace_data["trail"] = str(trail)
        if client_order_id is not None:
            replace_data["client_order_id"] = client_order_id
        
        data = await self._make_request("PATCH", f"/v2/orders/{order_id}", data=replace_data)
        return self._parse_order(data)
    
    def _parse_order(self, data: Dict[str, Any]) -> Order:
        """Parse order data from API response"""
        
        return Order(
            id=data['id'],
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            qty=float(data['qty']),
            filled_qty=float(data['filled_qty']),
            status=OrderStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
            submitted_at=datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00')) if data.get('submitted_at') else None,
            filled_at=datetime.fromisoformat(data['filled_at'].replace('Z', '+00:00')) if data.get('filled_at') else None,
            expired_at=datetime.fromisoformat(data['expired_at'].replace('Z', '+00:00')) if data.get('expired_at') else None,
            canceled_at=datetime.fromisoformat(data['canceled_at'].replace('Z', '+00:00')) if data.get('canceled_at') else None,
            failed_at=datetime.fromisoformat(data['failed_at'].replace('Z', '+00:00')) if data.get('failed_at') else None,
            replaced_at=datetime.fromisoformat(data['replaced_at'].replace('Z', '+00:00')) if data.get('replaced_at') else None,
            replaced_by=data.get('replaced_by'),
            replaces=data.get('replaces'),
            asset_id=data.get('asset_id'),
            asset_class=data.get('asset_class', 'us_equity'),
            notional=float(data['notional']) if data.get('notional') else None,
            qty_to_fill=float(data['qty']) - float(data['filled_qty']),
            limit_price=float(data['limit_price']) if data.get('limit_price') else None,
            stop_price=float(data['stop_price']) if data.get('stop_price') else None,
            trail_price=float(data['trail_price']) if data.get('trail_price') else None,
            trail_percent=float(data['trail_percent']) if data.get('trail_percent') else None,
            hwm=float(data['hwm']) if data.get('hwm') else None,
            extended_hours=data.get('extended_hours', False),
            legs=data.get('legs'),
            time_in_force=TimeInForce(data.get('time_in_force', 'day'))
        )
    
    # Portfolio Analytics
    async def get_portfolio_history(self, 
                                  period: str = "1M",
                                  timeframe: str = "1D",
                                  extended_hours: bool = False) -> Dict[str, Any]:
        """Get portfolio history"""
        params = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": str(extended_hours).lower()
        }
        
        return await self._make_request("GET", "/v2/account/portfolio/history", params=params)

class AlpacaWebSocketClient:
    """
    High-performance Alpaca WebSocket client for real-time account updates
    """
    
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.websocket = None
        self.callbacks = {
            'trade_updates': [],
            'account_updates': [],
            'error': []
        }
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def connect(self):
        """Connect to Alpaca WebSocket"""
        try:
            ws_url = f"{self.config.websocket_url}/account"
            self.websocket = await websockets.connect(
                ws_url,
                timeout=self.config.connection_timeout
            )
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.config.api_key,
                "secret": self.config.secret_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await self.websocket.recv()
            auth_data = json.loads(response)
            
            if auth_data[0].get('T') != 'success':
                raise Exception(f"Authentication failed: {auth_data}")
            
            # Subscribe to trade updates
            subscribe_message = {
                "action": "listen",
                "data": {
                    "streams": ["trade_updates"]
                }
            }
            await self.websocket.send(json.dumps(subscribe_message))
            
            self.logger.info("Connected to Alpaca WebSocket")
            self.running = True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while self.running:
                message = await self.websocket.recv()
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            if self.running:
                await self._reconnect()
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
            if self.running:
                await self._reconnect()
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            for event in data:
                event_type = event.get('T')
                
                if event_type == 'trade_update':
                    for callback in self.callbacks['trade_updates']:
                        try:
                            await callback(event)
                        except Exception as e:
                            self.logger.error(f"Trade update callback error: {e}")
                
                elif event_type == 'account_update':
                    for callback in self.callbacks['account_updates']:
                        try:
                            await callback(event)
                        except Exception as e:
                            self.logger.error(f"Account update callback error: {e}")
                            
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            for callback in self.callbacks['error']:
                try:
                    await callback(e)
                except:
                    pass
    
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket"""
        self.logger.info("Attempting to reconnect...")
        
        for attempt in range(self.config.max_retries):
            try:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                await self.connect()
                self.logger.info("Reconnected successfully")
                return
                
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self.logger.error("Failed to reconnect after maximum attempts")
        self.running = False

class AlpacaClient:
    """
    Combined Alpaca client with REST and WebSocket capabilities
    """
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        config = AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=paper_trading
        )
        
        # Use paper trading URL if specified
        if paper_trading:
            config.base_url = "https://paper-api.alpaca.markets"
        else:
            config.base_url = "https://api.alpaca.markets"
        
        self.rest = AlpacaRESTClient(config)
        self.websocket = AlpacaWebSocketClient(config)
        self.logger = logging.getLogger(__name__)
        self.paper_trading = paper_trading
    
    async def start(self):
        """Start both REST and WebSocket clients"""
        await self.rest.connect()
        await self.websocket.connect()
        
        mode = "PAPER" if self.paper_trading else "LIVE"
        self.logger.info(f"Connected to Alpaca in {mode} trading mode")
    
    async def stop(self):
        """Stop both clients"""
        await self.rest.disconnect()
        await self.websocket.disconnect()
    
    def add_trade_update_callback(self, callback: Callable):
        """Add callback for trade updates"""
        self.websocket.add_callback('trade_updates', callback)
    
    def add_account_update_callback(self, callback: Callable):
        """Add callback for account updates"""
        self.websocket.add_callback('account_updates', callback)
    
    # Quick access methods
    async def buy_market(self, symbol: str, qty: float, extended_hours: bool = False) -> Order:
        """Place market buy order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            extended_hours=extended_hours
        )
    
    async def sell_market(self, symbol: str, qty: float, extended_hours: bool = False) -> Order:
        """Place market sell order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            extended_hours=extended_hours
        )
    
    async def buy_limit(self, symbol: str, qty: float, limit_price: float, 
                       time_in_force: TimeInForce = TimeInForce.DAY) -> Order:
        """Place limit buy order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    async def sell_limit(self, symbol: str, qty: float, limit_price: float,
                        time_in_force: TimeInForce = TimeInForce.DAY) -> Order:
        """Place limit sell order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    async def place_stop_loss(self, symbol: str, qty: float, stop_price: float) -> Order:
        """Place stop loss order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=stop_price
        )
    
    async def place_trailing_stop(self, symbol: str, qty: float, trail_percent: float) -> Order:
        """Place trailing stop order"""
        return await self.rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            trail_percent=trail_percent
        )

# Example usage
async def example_usage():
    """Example of how to use the Alpaca client"""
    
    # Initialize client (paper trading)
    client = AlpacaClient(
        api_key="YOUR_API_KEY",
        secret_key="YOUR_SECRET_KEY",
        paper_trading=True
    )
    
    # Define callbacks for real-time updates
    async def handle_trade_update(update):
        print(f"Trade Update: {update}")
    
    async def handle_account_update(update):
        print(f"Account Update: {update}")
    
    # Set up callbacks
    client.add_trade_update_callback(handle_trade_update)
    client.add_account_update_callback(handle_account_update)
    
    try:
        # Start the client
        await client.start()
        print("🚀 Connected to Alpaca (Paper Trading)")
        
        # Get account info
        account = await client.rest.get_account()
        print(f"💰 Account Info:")
        print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Day Trades: {account.daytrade_count}")
        
        # Get current positions
        positions = await client.rest.get_positions()
        print(f"\n📊 Current Positions ({len(positions)}):")
        for pos in positions:
            pnl_color = "🟢" if pos.unrealized_pl >= 0 else "🔴"
            print(f"   {pos.symbol}: {pos.qty} shares @ ${pos.current_price:.2f}")
            print(f"      {pnl_color} P&L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_plpc:.2%})")
        
        # Get recent orders
        orders = await client.rest.get_orders(limit=10)
        print(f"\n📋 Recent Orders ({len(orders)}):")
        for order in orders[:5]:  # Show last 5
            status_emoji = "✅" if order.status == OrderStatus.FILLED else "⏳" if order.status in [OrderStatus.NEW, OrderStatus.ACCEPTED] else "❌"
            print(f"   {status_emoji} {order.symbol}: {order.side.value.upper()} {order.qty} @ {order.order_type.value.upper()}")
            print(f"      Status: {order.status.value}, Filled: {order.filled_qty}/{order.qty}")
        
        # Example trades (commented out for safety)
        """
        # Buy 10 shares of AAPL at market
        buy_order = await client.buy_market("AAPL", 10)
        print(f"📈 Market Buy Order: {buy_order.id}")
        
        # Place a limit sell order
        sell_order = await client.sell_limit("AAPL", 5, 150.00)
        print(f"📉 Limit Sell Order: {sell_order.id}")
        
        # Place a stop loss
        stop_order = await client.place_stop_loss("AAPL", 5, 140.00)
        print(f"🛑 Stop Loss Order: {stop_order.id}")
        """
        
        print("\n📡 Listening for real-time updates... (Press Ctrl+C to stop)")
        
        # Listen for real-time updates (run this in background)
        await client.websocket.listen()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping trading session...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.stop()
        print("🔌 Disconnected from Alpaca")

async def demo_portfolio_management():
    """Demo advanced portfolio management features"""
    
    client = AlpacaClient(
        api_key="YOUR_API_KEY",
        secret_key="YOUR_SECRET_KEY",
        paper_trading=True
    )
    
    try:
        await client.start()
        print("🏦 Portfolio Management Demo")
        
        # Get comprehensive account info
        account = await client.rest.get_account()
        
        print(f"\n💼 Account Summary:")
        print(f"   Account Number: {account.account_number}")
        print(f"   Status: {account.status}")
        print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Day Trading BP: ${account.daytrading_buying_power:,.2f}")
        print(f"   Pattern Day Trader: {'Yes' if account.pattern_day_trader else 'No'}")
        print(f"   Day Trade Count: {account.daytrade_count}")
        
        # Portfolio performance
        portfolio_history = await client.rest.get_portfolio_history(period="1M", timeframe="1D")
        if portfolio_history.get('equity'):
            equity_values = portfolio_history['equity']
            if len(equity_values) > 1:
                total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
                print(f"   30-Day Return: {total_return:+.2f}%")
        
        # Position analysis
        positions = await client.rest.get_positions()
        if positions:
            print(f"\n📊 Position Analysis:")
            total_value = sum(pos.market_value for pos in positions)
            total_pnl = sum(pos.unrealized_pl for pos in positions)
            
            print(f"   Total Positions Value: ${total_value:,.2f}")
            print(f"   Total Unrealized P&L: ${total_pnl:,.2f}")
            
            # Top positions
            sorted_positions = sorted(positions, key=lambda x: abs(x.market_value), reverse=True)
            print(f"\n🔝 Top Positions:")
            for i, pos in enumerate(sorted_positions[:5], 1):
                allocation = pos.market_value / account.portfolio_value * 100
                pnl_color = "🟢" if pos.unrealized_pl >= 0 else "🔴"
                print(f"   {i}. {pos.symbol}: ${pos.market_value:,.2f} ({allocation:.1f}%)")
                print(f"      {pnl_color} P&L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_plpc:.2%})")
        else:
            print("\n📊 No current positions")
        
        # Order analysis
        orders = await client.rest.get_orders(limit=50)
        if orders:
            print(f"\n📋 Order Analysis:")
            filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
            pending_orders = [o for o in orders if o.status in [OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]]
            
            print(f"   Recent Orders: {len(orders)}")
            print(f"   Filled Orders: {len(filled_orders)}")
            print(f"   Pending Orders: {len(pending_orders)}")
            
            if pending_orders:
                print(f"\n⏳ Pending Orders:")
                for order in pending_orders[:5]:
                    print(f"   {order.symbol}: {order.side.value.upper()} {order.qty} @ {order.order_type.value.upper()}")
                    if order.limit_price:
                        print(f"      Limit: ${order.limit_price:.2f}")
                    if order.stop_price:
                        print(f"      Stop: ${order.stop_price:.2f}")
        
        # Risk metrics
        print(f"\n⚠️  Risk Metrics:")
        if account.long_market_value > 0:
            leverage = account.long_market_value / account.equity
            print(f"   Leverage: {leverage:.2f}x")
        
        cash_allocation = account.cash / account.portfolio_value * 100
        print(f"   Cash Allocation: {cash_allocation:.1f}%")
        
        if account.daytrading_buying_power > 0:
            bp_usage = (account.daytrading_buying_power - account.buying_power) / account.daytrading_buying_power * 100
            print(f"   Buying Power Usage: {bp_usage:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "portfolio":
        asyncio.run(demo_portfolio_management())
    else:
        asyncio.run(example_usage())
