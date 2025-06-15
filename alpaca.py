"""
Custom High-Performance Alpaca Trading Client
Replaces alpaca-trade-api and alpaca-py with optimized aiohttp implementation
"""

import asyncio
import aiohttp
import websockets
from websockets import client, exceptions as websocket_exceptions
import logging
import time
import base64
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum
import orjson  # Fast JSON library
from urllib.parse import urlencode

from settings import config
from active_symbols import symbol_manager

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

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
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

@dataclass
class TradeSignal:
    """Trading signal from strategies"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    strategy: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    hold_time: Optional[int] = None
    reason: str = ""
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PositionInfo:
    """Enhanced position information"""
    symbol: str
    qty: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    
    # Strategy tracking
    strategy: str = "unknown"
    entry_time: Optional[datetime] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Performance metrics
    max_unrealized_pl: float = 0.0
    min_unrealized_pl: float = 0.0
    hold_duration: Optional[timedelta] = None

@dataclass
class OrderInfo:
    """Enhanced order information"""
    id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    created_at: datetime
    
    # Strategy context
    strategy: str = "unknown"
    confidence: float = 0.0
    original_signal: Optional[TradeSignal] = None

@dataclass
class AccountInfo:
    """Account information"""
    id: str
    account_number: str
    status: str
    currency: str
    buying_power: float
    regt_buying_power: float
    daytrading_buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    last_equity: float
    multiplier: int
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    sma: float
    daytrade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    trade_suspended_by_user: bool
    crypto_status: str

class AlpacaHTTPClient:
    """High-performance HTTP client for Alpaca REST API"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Connection pooling and optimization
        self.connector = aiohttp.TCPConnector(
            limit=50,  # Total connection pool size
            limit_per_host=20,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Timeout configuration
        self.timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Custom-Alpaca-Client/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.secret_key
                },
                json_serialize=lambda obj: orjson.dumps(obj).decode()
            )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.connector:
            await self.connector.close()
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict[str, Any]] = None, 
                          data: Optional[Dict[str, Any]] = None) -> Dict:
        """Make HTTP request with error handling"""
        if not self.session:
            await self.initialize()
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(3):  # Retry logic
            try:
                kwargs = {}
                if params:
                    kwargs['params'] = params
                if data:
                    kwargs['json'] = data
                
                assert self.session is not None
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status in [200, 201]:
                        return await response.json(loads=orjson.loads)
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 422:  # Unprocessable Entity
                        error_data = await response.json(loads=orjson.loads)
                        raise ValueError(f"API Error: {error_data}")
                    else:
                        response.raise_for_status()
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
            
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        raise Exception(f"Failed to {method} {endpoint} after 3 attempts")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Make GET request"""
        return await self._make_request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict:
        """Make POST request"""
        return await self._make_request("POST", endpoint, data=data)
    
    async def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict:
        """Make PATCH request"""
        return await self._make_request("PATCH", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict:
        """Make DELETE request"""
        return await self._make_request("DELETE", endpoint)

class AlpacaWebSocketClient:
    """High-performance WebSocket client for Alpaca real-time updates"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Determine WebSocket URL based on base URL
        if "paper" in base_url:
            self.websocket_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.websocket_url = "wss://api.alpaca.markets/stream"
        
        self.websocket: Optional[client.WebSocketClientProtocol] = None
        self.connected = False
        
        # Message processing
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.processing_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.trade_update_callbacks: List[Callable] = []
        self.account_update_callbacks: List[Callable] = []
        
        # Performance tracking
        self.messages_received = 0
        self.connection_start_time: Optional[datetime] = None
        self.last_heartbeat = time.time()
        
        # Reconnection logic
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
    
    async def connect(self):
        """Connect to WebSocket with automatic reconnection"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to Alpaca WebSocket (attempt {self.reconnect_attempts + 1})")
                
                # Connection with optimized settings
                self.websocket = await client.connect(
                    self.websocket_url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=2**20,    # 1MB max message size
                    compression=None   # Disable compression for speed
                )
                
                self.connected = True
                self.connection_start_time = datetime.now()
                self.reconnect_attempts = 0
                
                # Authenticate
                await self._authenticate()
                
                # Start message processing
                self.processing_task = asyncio.create_task(self._process_messages())
                
                # Start heartbeat monitoring
                asyncio.create_task(self._heartbeat_monitor())
                
                logger.info("Alpaca WebSocket connected and authenticated")
                return
                
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error(f"WebSocket connection failed: {e}")
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    wait_time = self.reconnect_delay * (2 ** min(self.reconnect_attempts, 5))
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max reconnection attempts reached")
                    raise
    
    async def _authenticate(self):
        """Authenticate WebSocket connection"""
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        
        if self.websocket:
            await self.websocket.send(orjson.dumps(auth_message).decode())
        
            # Wait for auth response
            response = await self.websocket.recv()
            auth_data = orjson.loads(response)
        else:
            raise ConnectionError("WebSocket not connected for authentication")
        
        if auth_data.get("T") == "success":
            logger.info("Alpaca WebSocket authentication successful")
            
            # Subscribe to trade updates
            await self._subscribe_to_updates()
        else:
            raise Exception(f"Alpaca WebSocket authentication failed: {auth_data}")
    
    async def _subscribe_to_updates(self):
        """Subscribe to trade and account updates"""
        subscribe_message = {
            "action": "listen",
            "data": {
                "streams": ["trade_updates", "account_updates"]
            }
        }
        
        if self.websocket:
            await self.websocket.send(orjson.dumps(subscribe_message).decode())
            logger.info("Subscribed to Alpaca trade and account updates")
        else:
            logger.warning("Cannot subscribe, WebSocket not connected.")
    
    async def _process_messages(self):
        """Process incoming WebSocket messages"""
        if not self.websocket:
            logger.error("WebSocket not connected, cannot process messages.")
            return
        try:
            async for message in self.websocket:
                self.messages_received += 1
                self.last_heartbeat = time.time()
                
                try:
                    # Use orjson for fast parsing
                    data = orjson.loads(message)
                    
                    # Queue message for processing to avoid blocking
                    if not self.message_queue.full():
                        await self.message_queue.put(data)
                    else:
                        logger.warning("Message queue full, dropping message")
                        
                except orjson.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websocket_exceptions.ConnectionClosed:
            logger.warning("Alpaca WebSocket connection closed")
            self.connected = False
            await self._reconnect()
        except Exception as e:
            logger.error(f"Alpaca WebSocket processing error: {e}")
            self.connected = False
            await self._reconnect()
    
    async def _handle_messages(self):
        """Handle messages from queue"""
        while True:
            try:
                data = await self.message_queue.get()
                await self._handle_message(data)
                    
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    async def _handle_message(self, message: Dict):
        """Handle individual WebSocket message"""
        msg_type = message.get("T")
        
        if msg_type == "trade_update":
            await self._handle_trade_update(message)
        elif msg_type == "account_update":
            await self._handle_account_update(message)
        elif msg_type in ["success", "error"]:
            logger.debug(f"Status message: {message}")
    
    async def _handle_trade_update(self, message: Dict):
        """Handle trade update message"""
        # Call registered callbacks
        for callback in self.trade_update_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Trade update callback error: {e}")
    
    async def _handle_account_update(self, message: Dict):
        """Handle account update message"""
        # Call registered callbacks
        for callback in self.account_update_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Account update callback error: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor connection health"""
        while self.connected:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if time.time() - self.last_heartbeat > 60:  # No message for 60 seconds
                logger.warning("No heartbeat received, reconnecting...")
                await self._reconnect()
                break
    
    async def _reconnect(self):
        """Reconnect WebSocket"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
        
        await asyncio.sleep(5)  # Wait before reconnecting
        await self.connect()
    
    def add_trade_update_callback(self, callback: Callable):
        """Add callback for trade updates"""
        self.trade_update_callbacks.append(callback)
    
    def add_account_update_callback(self, callback: Callable):
        """Add callback for account updates"""
        self.account_update_callbacks.append(callback)
    
    async def close(self):
        """Close WebSocket connection"""
        self.connected = False
        if self.processing_task:
            self.processing_task.cancel()
        if self.websocket:
            await self.websocket.close()

class AlpacaTradingClient:
    """Main Alpaca trading client"""
    
    def __init__(self):
        self.config = config.alpaca
        
        # Initialize HTTP client
        self.http_client = AlpacaHTTPClient(
            self.config.api_key,
            self.config.secret_key,
            self.config.base_url
        )
        
        # Initialize WebSocket client
        self.ws_client = AlpacaWebSocketClient(
            self.config.api_key,
            self.config.secret_key,
            self.config.base_url
        )
        
        # Position and order tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.open_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []
        
        # Account information
        self.account_info: Optional[AccountInfo] = None
        self.buying_power: float = 0.0
        
        # Risk management
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.max_daily_loss_hit: bool = False
        
        # Performance tracking
        self.total_trades_today: int = 0
        self.winning_trades_today: int = 0
        self.total_fees_today: float = 0.0
        
        # Message processing task
        self.message_handler_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the Alpaca trading client"""
        logger.info("Initializing Alpaca trading client...")
        
        try:
            # Initialize HTTP client
            await self.http_client.initialize()
            
            # Test API connection
            await self._test_connection()
            
            # Load account information
            await self._load_account_info()
            
            # Load current positions
            await self._load_positions()
            
            # Load open orders
            await self._load_open_orders()
            
            # Connect WebSocket
            await self.ws_client.connect()
            
            # Start message handler
            self.message_handler_task = asyncio.create_task(self.ws_client._handle_messages())
            
            # Set up callbacks
            self.ws_client.add_trade_update_callback(self._on_trade_update)
            self.ws_client.add_account_update_callback(self._on_account_update)
            
            # Start periodic updates
            asyncio.create_task(self._periodic_updates())
            
            logger.info("Alpaca trading client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca trading client: {e}")
            raise
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            account_data = await self.http_client.get("/v2/account")
            logger.info(f"Connected to Alpaca account: {account_data.get('id', 'unknown')}")
            
            # Check if account can trade
            if account_data.get('trading_blocked'):
                raise Exception("Account trading is blocked")
            
        except Exception as e:
            logger.error(f"Alpaca API connection test failed: {e}")
            raise
    
    async def _load_account_info(self):
        """Load account information"""
        try:
            account_data = await self.http_client.get("/v2/account")
            
            self.account_info = AccountInfo(
                id=account_data.get('id', ''),
                account_number=account_data.get('account_number', ''),
                status=account_data.get('status', ''),
                currency=account_data.get('currency', 'USD'),
                buying_power=float(account_data.get('buying_power', 0)),
                regt_buying_power=float(account_data.get('regt_buying_power', 0)),
                daytrading_buying_power=float(account_data.get('daytrading_buying_power', 0)),
                cash=float(account_data.get('cash', 0)),
                portfolio_value=float(account_data.get('portfolio_value', 0)),
                equity=float(account_data.get('equity', 0)),
                last_equity=float(account_data.get('last_equity', 0)),
                multiplier=int(account_data.get('multiplier', 1)),
                long_market_value=float(account_data.get('long_market_value', 0)),
                short_market_value=float(account_data.get('short_market_value', 0)),
                initial_margin=float(account_data.get('initial_margin', 0)),
                maintenance_margin=float(account_data.get('maintenance_margin', 0)),
                sma=float(account_data.get('sma', 0)),
                daytrade_count=int(account_data.get('daytrade_count', 0)),
                pattern_day_trader=account_data.get('pattern_day_trader', False),
                trading_blocked=account_data.get('trading_blocked', False),
                transfers_blocked=account_data.get('transfers_blocked', False),
                account_blocked=account_data.get('account_blocked', False),
                created_at=datetime.fromisoformat(account_data.get('created_at', '').replace('Z', '+00:00')),
                trade_suspended_by_user=account_data.get('trade_suspended_by_user', False),
                crypto_status=account_data.get('crypto_status', '')
            )
            
            if self.account_info:
                self.buying_power = self.account_info.buying_power
                logger.info(f"Account equity: ${self.account_info.equity:,.2f}")
                logger.info(f"Buying power: ${self.buying_power:,.2f}")
            
        except Exception as e:
            logger.error(f"Error loading account info: {e}")
            raise
    
    async def _load_positions(self):
        """Load current positions"""
        try:
            positions_data = await self.http_client.get("/v2/positions")
            
            for position_data in positions_data:
                position_info = PositionInfo(
                    symbol=position_data.get('symbol', ''),
                    qty=float(position_data.get('qty', 0)),
                    side=position_data.get('side', ''),
                    market_value=float(position_data.get('market_value', 0)),
                    cost_basis=float(position_data.get('cost_basis', 0)),
                    unrealized_pl=float(position_data.get('unrealized_pl', 0)),
                    unrealized_plpc=float(position_data.get('unrealized_plpc', 0)),
                    current_price=float(position_data.get('current_price', 0))
                )
                
                self.positions[position_info.symbol] = position_info
            
            logger.info(f"Loaded {len(self.positions)} existing positions")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def _load_open_orders(self):
        """Load open orders"""
        try:
            orders_data = await self.http_client.get("/v2/orders", params={"status": "open"})
            
            for order_data in orders_data:
                order_info = OrderInfo(
                    id=order_data.get('id', ''),
                    symbol=order_data.get('symbol', ''),
                    side=order_data.get('side', ''),
                    qty=float(order_data.get('qty', 0)),
                    order_type=order_data.get('order_type', ''),
                    status=order_data.get('status', ''),
                    filled_qty=float(order_data.get('filled_qty', 0)),
                    filled_avg_price=float(order_data.get('filled_avg_price', 0)) if order_data.get('filled_avg_price') else None,
                    created_at=datetime.fromisoformat(order_data.get('created_at', '').replace('Z', '+00:00'))
                )
                
                self.open_orders[order_info.id] = order_info
            
            logger.info(f"Loaded {len(self.open_orders)} open orders")
            
        except Exception as e:
            logger.error(f"Error loading open orders: {e}")
    
    async def _periodic_updates(self):
        """Periodic updates of positions and orders"""
        while True:
            try:
                await self._update_positions()
                await self._update_orders()
                await self._update_account()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _update_positions(self):
        """Update position information"""
        try:
            positions_data = await self.http_client.get("/v2/positions")
            
            # Update existing positions
            current_symbols = set()
            
            for position_data in positions_data:
                symbol = position_data.get('symbol', '')
                current_symbols.add(symbol)
                
                if symbol in self.positions:
                    # Update existing position
                    self.positions[symbol].qty = float(position_data.get('qty', 0))
                    self.positions[symbol].market_value = float(position_data.get('market_value', 0))
                    self.positions[symbol].unrealized_pl = float(position_data.get('unrealized_pl', 0))
                    self.positions[symbol].unrealized_plpc = float(position_data.get('unrealized_plpc', 0))
                    self.positions[symbol].current_price = float(position_data.get('current_price', 0))
                    
                    # Update min/max unrealized P&L
                    current_pl = self.positions[symbol].unrealized_pl
                    self.positions[symbol].max_unrealized_pl = max(
                        self.positions[symbol].max_unrealized_pl, current_pl
                    )
                    self.positions[symbol].min_unrealized_pl = min(
                        self.positions[symbol].min_unrealized_pl, current_pl
                    )
                else:
                    # New position
                    position_info = PositionInfo(
                        symbol=symbol,
                        qty=float(position_data.get('qty', 0)),
                        side=position_data.get('side', ''),
                        market_value=float(position_data.get('market_value', 0)),
                        cost_basis=float(position_data.get('cost_basis', 0)),
                        unrealized_pl=float(position_data.get('unrealized_pl', 0)),
                        unrealized_plpc=float(position_data.get('unrealized_plpc', 0)),
                        current_price=float(position_data.get('current_price', 0)),
                        entry_time=datetime.now()
                    )
                    
                    self.positions[symbol] = position_info
            
            # Remove closed positions
            closed_symbols = set(self.positions.keys()) - current_symbols
            for symbol in closed_symbols:
                logger.info(f"Position closed: {symbol}")
                del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _update_orders(self):
        """Update order status"""
        try:
            orders_data = await self.http_client.get("/v2/orders", params={"status": "open"})
            current_order_ids = {order.get('id') for order in orders_data}
            
            # Update existing orders
            for order_data in orders_data:
                order_id = order_data.get('id')
                if order_id in self.open_orders:
                    self.open_orders[order_id].status = order_data.get('status', '')
                    self.open_orders[order_id].filled_qty = float(order_data.get('filled_qty', 0))
                    self.open_orders[order_id].filled_avg_price = (
                        float(order_data.get('filled_avg_price', 0)) if order_data.get('filled_avg_price') else None
                    )
            
            # Remove filled/canceled orders
            completed_orders = set(self.open_orders.keys()) - current_order_ids
            for order_id in completed_orders:
                order_info = self.open_orders[order_id]
                self.order_history.append(order_info)
                del self.open_orders[order_id]
                
                logger.info(f"Order completed: {order_id} ({order_info.symbol})")
            
        except Exception as e:
            logger.error(f"Error updating orders: {e}")
    
    async def _update_account(self):
        """Update account information"""
        try:
            account_data = await self.http_client.get("/v2/account")
            
            if self.account_info:
                self.account_info.buying_power = float(account_data.get('buying_power', 0))
                self.account_info.equity = float(account_data.get('equity', 0))
                self.account_info.portfolio_value = float(account_data.get('portfolio_value', 0))
                
                self.buying_power = self.account_info.buying_power
                
                # Calculate daily P&L
                self.daily_pnl = sum(pos.unrealized_pl for pos in self.positions.values())
                
                # Check risk limits
                if self.account_info.equity > 0 and self.daily_pnl < -config.risk.max_daily_loss * self.account_info.equity:
                    if not self.max_daily_loss_hit:
                        logger.warning("Daily loss limit hit!")
                        self.max_daily_loss_hit = True
                        await self._emergency_close_all()
            
        except Exception as e:
            logger.error(f"Error updating account: {e}")
    
    async def _on_trade_update(self, message: Dict):
        """Handle trade update from WebSocket"""
        logger.debug(f"Trade update: {message}")
        # Process trade updates for real-time order status changes
    
    async def _on_account_update(self, message: Dict):
        """Handle account update from WebSocket"""
        logger.debug(f"Account update: {message}")
        # Process account updates for real-time balance changes
    
    async def place_trade(self, signal: TradeSignal) -> Optional[str]:
        """Place a trade based on signal"""
        
        if not self._validate_trade_signal(signal):
            return None
        
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {signal.symbol}: {position_size}")
                return None
            
            # Get current price to calculate shares
            current_price = await self._get_current_price(signal.symbol)
            if not current_price or current_price <= 0: # Added check for current_price > 0
                logger.error(f"Cannot get valid current price for {signal.symbol}")
                return None
            
            qty = int(position_size / current_price)
            if qty <= 0:
                logger.warning(f"Calculated quantity too small: {qty}")
                return None
            
            # Prepare order parameters
            order_data = {
                'symbol': signal.symbol,
                'side': signal.action.lower(),
                'type': 'market',  # Default to market orders for now
                'qty': str(qty),
                'time_in_force': 'day'
            }
            
            # Add stop loss if provided
            if signal.stop_loss:
                # TODO: Implement bracket orders with stop loss
                pass
            
            # Submit order
            order_response = await self.http_client.post("/v2/orders", data=order_data)
            
            # Track order
            order_info = OrderInfo(
                id=order_response.get('id', ''),
                symbol=order_response.get('symbol', ''),
                side=order_response.get('side', ''),
                qty=float(order_response.get('qty', 0)),
                order_type=order_response.get('order_type', ''),
                status=order_response.get('status', ''),
                filled_qty=0,
                filled_avg_price=None,
                created_at=datetime.fromisoformat(order_response.get('created_at', '').replace('Z', '+00:00')),
                strategy=signal.strategy,
                confidence=signal.confidence,
                original_signal=signal
            )
            
            self.open_orders[order_info.id] = order_info
            self.total_trades_today += 1
            
            logger.info(f"Order placed: {signal.action} {qty} {signal.symbol} (ID: {order_info.id})")
            
            return order_info.id
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None
    
    def _validate_trade_signal(self, signal: TradeSignal) -> bool:
        """Validate trade signal before execution"""
        
        # Check if we're allowed to trade
        if self.max_daily_loss_hit:
            logger.warning("Daily loss limit hit, rejecting trade")
            return False
        
        # Check confidence threshold
        if signal.confidence < config.strategy.confidence_threshold:
            logger.warning(f"Signal confidence too low: {signal.confidence}")
            return False
        
        # Check if we already have a position
        if signal.symbol in self.positions:
            existing_pos = self.positions[signal.symbol]
            
            # Don't add to existing position in same direction
            if (signal.action == "BUY" and existing_pos.side == "long") or \
               (signal.action == "SELL" and existing_pos.side == "short"):
                logger.warning(f"Already have {existing_pos.side} position in {signal.symbol}")
                return False
        
        # Check daily trade limit
        if self.total_trades_today >= config.alpaca.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check symbol is tradeable
        if signal.symbol not in symbol_manager.get_active_symbols():
            logger.warning(f"Symbol {signal.symbol} not in active symbols")
            return False
        
        return True
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate position size based on risk management"""
        
        if signal.position_size:
            # Use signal's suggested size
            if self.account_info:
                return signal.position_size * self.account_info.equity
            return 0.0 # Should not happen if initialized correctly
        
        # Default position sizing
        base_size = 0.0
        if self.account_info:
            base_size = config.risk.max_position_size * self.account_info.equity
        
        # Adjust for confidence
        confidence_multiplier = min(signal.confidence * 2, 1.0)
        
        # Adjust for volatility
        symbol_metrics = symbol_manager.metrics.get(signal.symbol)
        volatility_multiplier = 1.0
        if symbol_metrics and hasattr(symbol_metrics, 'volatility') and symbol_metrics.volatility > 0:
            # Reduce size for high volatility
            volatility_multiplier = min(1.0, 0.02 / symbol_metrics.volatility)
        
        return base_size * confidence_multiplier * volatility_multiplier
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        
        try:
            # Get latest trade
            trade_data = await self.http_client.get(f"/v2/stocks/{symbol}/trades/latest")
            if trade_data and trade_data.get('trade'):
                return float(trade_data['trade'].get('p', 0))
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        
        return None
    
    async def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close position for symbol"""
        
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return False
        
        try:
            position = self.positions[symbol]
            
            # Determine side for closing order
            close_side = "sell" if position.side == "long" else "buy"
            
            order_data = {
                'symbol': symbol,
                'side': close_side,
                'type': 'market',
                'qty': str(abs(int(position.qty))),
                'time_in_force': 'day'
            }
            
            order_response = await self.http_client.post("/v2/orders", data=order_data)
            
            logger.info(f"Closing position: {symbol} ({reason}) - Order ID: {order_response.get('id')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def close_all_positions(self, reason: str = "Close all") -> int:
        """Close all open positions"""
        
        closed_count = 0
        
        for symbol in list(self.positions.keys()):
            if await self.close_position(symbol, reason):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} positions")
        return closed_count
    
    async def _emergency_close_all(self):
        """Emergency close all positions due to risk limits"""
        
        logger.warning("EMERGENCY: Closing all positions due to risk limits")
        
        # Cancel all open orders first
        await self.cancel_all_orders("Emergency risk management")
        
        # Close all positions
        await self.close_all_positions("Emergency risk management")
    
    async def cancel_all_orders(self, reason: str = "Cancel all") -> int:
        """Cancel all open orders"""
        
        canceled_count = 0
        
        for order_id in list(self.open_orders.keys()):
            try:
                await self.http_client.delete(f"/v2/orders/{order_id}")
                canceled_count += 1
                logger.info(f"Canceled order {order_id} ({reason})")
                
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")
        
        return canceled_count
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pl = sum(pos.unrealized_pl for pos in self.positions.values())
        
        win_rate = 0.0
        if self.total_trades_today > 0:
            win_rate = self.winning_trades_today / self.total_trades_today
        
        equity = 0.0
        if self.account_info:
            equity = self.account_info.equity

        return {
            'account_equity': equity,
            'buying_power': self.buying_power,
            'positions_count': len(self.positions),
            'open_orders_count': len(self.open_orders),
            'total_position_value': total_value,
            'total_unrealized_pl': total_unrealized_pl,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.total_trades_today,
            'win_rate_today': win_rate,
            'max_daily_loss_hit': self.max_daily_loss_hit
        }
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position info for symbol"""
        return self.positions.get(symbol)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get open orders, optionally filtered by symbol"""
        
        if symbol:
            return [order for order in self.open_orders.values() if order.symbol == symbol]
        
        return list(self.open_orders.values())
    
    async def cleanup(self):
        """Cleanup connections and resources"""
        
        logger.info("Cleaning up Alpaca connections...")
        
        if self.message_handler_task:
            self.message_handler_task.cancel()
        
        await self.ws_client.close()
        await self.http_client.close()
        
        logger.info("Alpaca cleanup completed")

# Global instance (lazy-initialized)
alpaca_client = None

def get_alpaca_client():
    """Get or create the global alpaca trading client"""
    global alpaca_client
    if alpaca_client is None:
        alpaca_client = AlpacaTradingClient()
    return alpaca_client
