"""
Custom High-Performance Polygon.io Client
Replaces polygon-api-client with optimized aiohttp and websockets implementation
"""

import asyncio
import aiohttp
import websockets
from websockets import client, exceptions as websocket_exceptions
import logging
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Callable, Any, Set, Union, Awaitable, cast
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import orjson  # Fast JSON library
from urllib.parse import urlencode

from settings import config
from active_symbols import symbol_manager
from lag_llama_engine import lag_llama_engine
from database import get_database_manager

logger = logging.getLogger(__name__)

@dataclass
class TradeData:
    """Trade data structure"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    conditions: Optional[List[str]] = None
    exchange: Optional[int] = None

@dataclass
class QuoteData:
    """Quote data structure"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    exchange: Optional[int] = None

@dataclass
class AggregateData:
    """Aggregate (minute bar) data structure"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    timestamp: datetime

@dataclass
class DailyBarData:
    """Daily bar data structure"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    date: date

class RateLimiter:
    """Advanced rate limiter with burst handling"""
    
    def __init__(self, calls_per_minute: int = 5, burst_size: int = 10):
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size
        self.calls = deque()
        self.burst_calls = 0
        self.last_reset = time.time()
    
    async def acquire(self):
        """Acquire permission to make API call"""
        now = time.time()
        
        # Reset burst counter every minute
        if now - self.last_reset >= 60:
            self.burst_calls = 0
            self.last_reset = now
            self.calls.clear()
        
        # Remove old calls from sliding window
        while self.calls and now - self.calls[0] >= 60:
            self.calls.popleft()
        
        # Check if we can make the call
        if len(self.calls) >= self.calls_per_minute:
            # Calculate wait time
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.debug(f"Rate limit hit, waiting {wait_time*1000:.0f}ms")
                await asyncio.sleep(wait_time)
        
        # Check burst limit
        if self.burst_calls >= self.burst_size:
            await asyncio.sleep(1)  # Brief pause for burst protection
            self.burst_calls = 0
        
        # Record the call
        self.calls.append(now)
        self.burst_calls += 1

class PolygonHTTPClient:
    """High-performance HTTP client for Polygon REST API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(calls_per_minute=5, burst_size=10)
        
        # Connection pooling and optimization
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Connections per host
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
                    'User-Agent': 'Custom-Polygon-Client/1.0',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
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
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Make HTTP request with rate limiting and error handling"""
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self.initialize()
        
        # Add API key to params
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(3):  # Retry logic
            try:
                assert self.session is not None
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json(loads=orjson.loads)
                        return data
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
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
        
        raise Exception(f"Failed to fetch {endpoint} after 3 attempts")
    
    async def get_market_status(self) -> Dict:
        """Get market status"""
        return await self._make_request("/v1/marketstatus/now")
    
    async def get_aggregates(self, symbol: str, multiplier: int, timespan: str,
                           from_date: str, to_date: str) -> Dict:
        """Get aggregate bars"""
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        return await self._make_request(endpoint, params)
    
    async def get_multi_timeframe_aggregates(self, symbol: str, timeframes: List[int],
                                           from_date: str, to_date: str) -> Dict[int, Dict]:
        """Get aggregate bars for multiple timeframes (5, 15, 30, 60, 120 minutes)"""
        results = {}
        
        for timeframe in timeframes:
            try:
                data = await self.get_aggregates(symbol, timeframe, "minute", from_date, to_date)
                results[timeframe] = data
                
                # Small delay between requests to respect rate limits
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching {timeframe}min data for {symbol}: {e}")
                results[timeframe] = {}
        
        return results
    
    async def get_daily_bars(self, symbol: str, date_str: str) -> Dict:
        """Get daily bars for specific date"""
        return await self.get_aggregates(symbol, 1, "day", date_str, date_str)
    
    async def get_trades(self, symbol: str, timestamp: Optional[str] = None, limit: int = 1000) -> Dict:
        """Get recent trades"""
        endpoint = f"/v3/trades/{symbol}"
        params: Dict[str, Any] = {'limit': limit}
        if timestamp:
            params['timestamp'] = timestamp
        return await self._make_request(endpoint, params)
    
    async def get_quotes(self, symbol: str, timestamp: Optional[str] = None, limit: int = 1000) -> Dict:
        """Get recent quotes"""
        endpoint = f"/v3/quotes/{symbol}"
        params: Dict[str, Any] = {'limit': limit}
        if timestamp:
            params['timestamp'] = timestamp
        return await self._make_request(endpoint, params)

class PolygonWebSocketClient:
    """High-performance WebSocket client for real-time data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.indices_websocket_url = "wss://socket.polygon.io/indices"
        self.websocket: Optional[client.WebSocketClientProtocol] = None
        self.connected = False
        self.subscribed_symbols: Set[str] = set()
        
        # Message processing
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.processing_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.trade_callbacks: List[Callable[[TradeData], Awaitable[None]]] = []
        self.quote_callbacks: List[Callable[[QuoteData], Awaitable[None]]] = []
        self.aggregate_callbacks: List[Callable[[AggregateData], Awaitable[None]]] = []
        
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
                logger.info(f"Connecting to Polygon WebSocket (attempt {self.reconnect_attempts + 1})")
                
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
                
                logger.info("WebSocket connected and authenticated")
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
            "params": self.api_key
        }
        
        if self.websocket:
            await self.websocket.send(orjson.dumps(auth_message).decode())
        
            # Wait for auth response
            response = await self.websocket.recv()
            auth_data = orjson.loads(response)
        else:
            raise ConnectionError("WebSocket is not connected for authentication")
        
        if auth_data[0].get("status") in ["auth_success", "connected"]:
            logger.info("WebSocket authentication successful")
        else:
            raise Exception(f"WebSocket authentication failed: {auth_data}")
    
    async def _process_messages(self):
        """Process incoming WebSocket messages with high performance"""
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
            logger.warning("WebSocket connection closed")
            self.connected = False
            await self._reconnect()
        except Exception as e:
            logger.error(f"WebSocket processing error: {e}")
            self.connected = False
            await self._reconnect()
    
    async def _handle_messages(self):
        """Handle messages from queue"""
        while True:
            try:
                data = await self.message_queue.get()
                
                # Handle different message types
                for item in data:
                    await self._handle_message(item)
                    
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    async def _handle_message(self, message: Dict):
        """Handle individual WebSocket message"""
        msg_type = message.get("ev")
        
        if msg_type == "T":  # Trade
            await self._handle_trade_message(message)
        elif msg_type == "Q":  # Quote
            await self._handle_quote_message(message)
        elif msg_type in ["A", "AM"]:  # Aggregate
            await self._handle_aggregate_message(message)
        elif msg_type == "V":  # Index value
            await self._handle_index_message(message)
        elif msg_type == "status":
            logger.debug(f"Status message: {message}")
    
    async def _handle_trade_message(self, message: Dict):
        """Handle trade message"""
        symbol = message.get("sym")
        if not symbol:
            return
        
        trade_data = TradeData(
            symbol=symbol,
            price=message.get("p", 0.0),
            size=message.get("s", 0),
            timestamp=datetime.fromtimestamp(message.get("t", 0) / 1000),
            conditions=message.get("c", []),
            exchange=message.get("x")
        )
        
        # Update symbol metrics
        symbol_manager.update_symbol_metrics(
            symbol,
            price=trade_data.price,
            volume=trade_data.size,
            last_updated=trade_data.timestamp
        )
        
        # Add to Lag-Llama price buffer
        lag_llama_engine.add_price_data(
            symbol, trade_data.price, trade_data.size, trade_data.timestamp
        )
        
        # Call registered callbacks
        for callback in self.trade_callbacks:
            try:
                await callback(trade_data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    async def _handle_quote_message(self, message: Dict):
        """Handle quote message"""
        symbol = message.get("sym")
        if not symbol:
            return
        
        quote_data = QuoteData(
            symbol=symbol,
            bid=message.get("bp", 0.0),
            ask=message.get("ap", 0.0),
            bid_size=message.get("bs", 0),
            ask_size=message.get("as", 0),
            timestamp=datetime.fromtimestamp(message.get("t", 0) / 1000),
            exchange=message.get("x")
        )
        
        # Call registered callbacks
        for callback in self.quote_callbacks:
            try:
                await callback(quote_data)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")
    
    async def _handle_aggregate_message(self, message: Dict):
        """Handle aggregate (minute bar) message"""
        symbol = message.get("sym")
        if not symbol:
            return
        
        aggregate_data = AggregateData(
            symbol=symbol,
            open=message.get("o", 0.0),
            high=message.get("h", 0.0),
            low=message.get("l", 0.0),
            close=message.get("c", 0.0),
            volume=message.get("v", 0),
            vwap=message.get("vw", 0.0),
            timestamp=datetime.fromtimestamp(message.get("s", 0) / 1000)
        )
        
        # Call registered callbacks
        for callback in self.aggregate_callbacks:
            try:
                await callback(aggregate_data)
            except Exception as e:
                logger.error(f"Aggregate callback error: {e}")
    
    async def _handle_index_message(self, message: Dict):
        """Handle index value message"""
        symbol = message.get("sym")
        value = message.get("val")
        timestamp = datetime.fromtimestamp(message.get("t", 0) / 1000)
        logger.debug(f"Index {symbol}: {value} at {timestamp}")
    
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
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        if not self.connected:
            logger.warning("Not connected, cannot subscribe")
            return
        
        subscriptions = []
        for symbol in symbols:
            if config.polygon.subscribe_trades:
                subscriptions.append(f"T.{symbol}")
            if config.polygon.subscribe_quotes:
                subscriptions.append(f"Q.{symbol}")
            if config.polygon.subscribe_aggregates:
                subscriptions.append(f"A.{symbol}")
                subscriptions.append(f"AM.{symbol}")
        
        if subscriptions:
            sub_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            if self.websocket:
                await self.websocket.send(orjson.dumps(sub_message).decode())
                self.subscribed_symbols.update(symbols)
                logger.info(f"Subscribed to {len(subscriptions)} data streams")
            else:
                logger.warning("Cannot subscribe, WebSocket not connected.")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not self.connected:
            return
        
        subscriptions = []
        for symbol in symbols:
            subscriptions.extend([f"T.{symbol}", f"Q.{symbol}", f"A.{symbol}", f"AM.{symbol}"])
        
        unsub_message = {
            "action": "unsubscribe",
            "params": ",".join(subscriptions)
        }
        
        if self.websocket:
            await self.websocket.send(orjson.dumps(unsub_message).decode())
            self.subscribed_symbols.difference_update(symbols)
            logger.info(f"Unsubscribed from {symbols}")
        else:
            logger.warning("Cannot unsubscribe, WebSocket not connected.")
    
    def add_trade_callback(self, callback: Callable[[TradeData], Awaitable[None]]):
        """Add callback for trade data"""
        self.trade_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable[[QuoteData], Awaitable[None]]):
        """Add callback for quote data"""
        self.quote_callbacks.append(callback)
    
    def add_aggregate_callback(self, callback: Callable[[AggregateData], Awaitable[None]]):
        """Add callback for aggregate data"""
        self.aggregate_callbacks.append(callback)
    
    async def close(self):
        """Close WebSocket connection"""
        self.connected = False
        if self.processing_task:
            self.processing_task.cancel()
        if self.websocket:
            await self.websocket.close()

class PolygonDataManager:
    """Main Polygon data manager combining HTTP and WebSocket clients"""
    
    def __init__(self):
        self.config = config.polygon
        self.http_client = PolygonHTTPClient(self.config.api_key)
        self.ws_client = PolygonWebSocketClient(self.config.api_key)
        self.db_manager = None
        
        # Data storage
        self.latest_trades: Dict[str, TradeData] = {}
        self.latest_quotes: Dict[str, QuoteData] = {}
        self.latest_aggregates: Dict[str, AggregateData] = {}
        self.daily_bars: Dict[str, DailyBarData] = {}
        
        # Historical data cache
        self.historical_cache: Dict[str, tuple] = {}
        
        # Message processing task
        self.message_handler_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the Polygon data manager"""
        logger.info("Initializing Polygon data manager...")
        
        try:
            # Initialize HTTP client
            await self.http_client.initialize()
            
            # Initialize database manager
            self.db_manager = get_database_manager()
            
            # Test connection
            await self._test_connection()
            
            # Load previous day data
            await self._load_previous_day_data()
            
            # Connect WebSocket
            await self.ws_client.connect()
            
            # Start message handler
            self.message_handler_task = asyncio.create_task(self.ws_client._handle_messages())
            
            # Subscribe to symbols
            symbols = symbol_manager.get_active_symbols()
            await self.ws_client.subscribe(symbols)
            
            # Set up callbacks
            self.ws_client.add_trade_callback(cast(Callable[[TradeData], Awaitable[None]], self._on_trade))
            self.ws_client.add_quote_callback(cast(Callable[[QuoteData], Awaitable[None]], self._on_quote))
            self.ws_client.add_aggregate_callback(cast(Callable[[AggregateData], Awaitable[None]], self._on_aggregate))
            
            logger.info("Polygon data manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polygon data manager: {e}")
            raise
    
    async def _test_connection(self):
        """Test REST API connection"""
        try:
            status = await self.http_client.get_market_status()
            logger.info(f"Market status: {status.get('market', 'unknown')}")
        except Exception as e:
            logger.error(f"Polygon API connection test failed: {e}")
            raise
    
    async def _load_previous_day_data(self):
        """Load previous day closing data for gap calculations"""
        logger.info("Loading previous day data...")
        
        symbols = symbol_manager.get_active_symbols()
        
        # Get previous trading day
        today = date.today()
        previous_day = today - timedelta(days=1)
        
        # Handle weekends
        while previous_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            previous_day -= timedelta(days=1)
        
        try:
            for symbol in symbols:
                data = await self.http_client.get_daily_bars(symbol, previous_day.strftime("%Y-%m-%d"))
                
                if data.get('results') and len(data['results']) > 0:
                    bar = data['results'][0]
                    
                    daily_bar = DailyBarData(
                        symbol=symbol,
                        open=float(bar.get('o', 0.0)),
                        high=float(bar.get('h', 0.0)),
                        low=float(bar.get('l', 0.0)),
                        close=float(bar.get('c', 0.0)),
                        volume=int(bar.get('v', 0)),
                        vwap=float(bar.get('vw', 0.0)),
                        date=previous_day
                    )
                    
                    self.daily_bars[symbol] = daily_bar
                    
                    # Update symbol metrics with previous close
                    symbol_manager.update_symbol_metrics(
                        symbol,
                        prev_close=bar.get('c', 0.0)
                    )
            
            logger.info(f"Loaded previous day data for {len(self.daily_bars)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading previous day data: {e}")
    
    async def _on_trade(self, trade_data: TradeData):
        """Handle trade data"""
        self.latest_trades[trade_data.symbol] = trade_data
        
        # Calculate gap if we have previous close
        if trade_data.symbol in self.daily_bars:
            prev_close = self.daily_bars[trade_data.symbol].close
            if hasattr(symbol_manager.metrics.get(trade_data.symbol), 'update_gap_metrics'):
                symbol_manager.metrics[trade_data.symbol].update_gap_metrics(
                    trade_data.price, prev_close
                )
    
    async def _on_quote(self, quote_data: QuoteData):
        """Handle quote data"""
        self.latest_quotes[quote_data.symbol] = quote_data
    
    async def _on_aggregate(self, aggregate_data: AggregateData):
        """Handle aggregate data"""
        self.latest_aggregates[aggregate_data.symbol] = aggregate_data
        
        # Update volume metrics
        if aggregate_data.symbol in symbol_manager.metrics:
            avg_volume = symbol_manager.metrics[aggregate_data.symbol].avg_volume or 1000000
            if hasattr(symbol_manager.metrics[aggregate_data.symbol], 'update_volume_metrics'):
                symbol_manager.metrics[aggregate_data.symbol].update_volume_metrics(
                    aggregate_data.volume, avg_volume
                )
    
    async def get_historical_data(self, symbol: str, days: int = 30,
                                timespan: str = "minute") -> List[Dict]:
        """Get historical data for a symbol"""
        cache_key = f"{symbol}_{days}_{timespan}"
        
        # Check cache first
        if cache_key in self.historical_cache:
            cache_time, data = self.historical_cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=1):
                return data
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            response = await self.http_client.get_aggregates(
                symbol, 1, timespan,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            data = []
            if response.get('results'):
                for bar in response['results']:
                    data.append({
                        'timestamp': datetime.fromtimestamp(bar.get('t', 0) / 1000),
                        'open': bar.get('o', 0.0),
                        'high': bar.get('h', 0.0),
                        'low': bar.get('l', 0.0),
                        'close': bar.get('c', 0.0),
                        'volume': bar.get('v', 0),
                        'vwap': bar.get('vw', 0.0)
                    })
            
            # Cache the data
            self.historical_cache[cache_key] = (datetime.now(), data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def get_multi_timeframe_historical_data(self, symbol: str, days: int = 30) -> Dict[int, List[Dict]]:
        """Get historical data for multiple timeframes (5, 15, 30, 60, 120 minutes) - HIGH PRIORITY for Lag-Llama"""
        
        # Standard timeframes for multi-timeframe analysis
        timeframes = [5, 15, 30, 60, 120]
        results = {}
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            # Get data for all timeframes
            multi_data = await self.http_client.get_multi_timeframe_aggregates(
                symbol, timeframes,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            # Process each timeframe
            for timeframe_min, response in multi_data.items():
                data = []
                if response.get('results'):
                    for bar in response['results']:
                        data.append({
                            'timestamp': datetime.fromtimestamp(bar.get('t', 0) / 1000),
                            'open': bar.get('o', 0.0),
                            'high': bar.get('h', 0.0),
                            'low': bar.get('l', 0.0),
                            'close': bar.get('c', 0.0),
                            'volume': bar.get('v', 0),
                            'vwap': bar.get('vw', 0.0),
                            'timeframe_minutes': timeframe_min
                        })
                
                results[timeframe_min] = data
                
                # Cache each timeframe separately
                cache_key = f"{symbol}_{days}_{timeframe_min}min"
                self.historical_cache[cache_key] = (datetime.now(), data)
            
            logger.info(f"Retrieved multi-timeframe data for {symbol}: {[f'{tf}min({len(data)})' for tf, data in results.items()]}")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe historical data for {symbol}: {e}")
            return {}
    
    async def get_gap_candidates(self, min_gap_percent: float = 2.0, 
                               min_volume_ratio: float = 1.5) -> List[Dict]:
        """Get gap candidates from Polygon grouped daily data"""
        try:
            # Get previous trading day
            today = date.today()
            yesterday = today - timedelta(days=1)
            
            # Handle weekends
            while yesterday.weekday() >= 5:
                yesterday -= timedelta(days=1)
            
            # Get grouped daily aggregates
            response = await self.http_client._make_request(
                f"/v2/aggs/grouped/locale/us/market/stocks/{yesterday.strftime('%Y-%m-%d')}"
            )
            
            gap_candidates = []
            
            if response.get('results'):
                for result in response['results']:
                    symbol = result.get('T')
                    if not symbol:
                        continue
                    
                    prev_close = float(result.get('c', 0))
                    volume = int(result.get('v', 0))
                    
                    # Get current price
                    try:
                        current_data = await self.http_client._make_request(
                            f"/v2/last/trade/{symbol}"
                        )
                        
                        if current_data.get('results'):
                            current_price = float(current_data['results'].get('p', prev_close))
                        else:
                            current_price = prev_close
                    except:
                        current_price = prev_close
                    
                    # Calculate gap
                    gap_percent = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                    
                    # Estimate volume ratio (using 30-day average as baseline)
                    avg_volume = volume  # Simplified - could fetch historical average
                    volume_ratio = 1.0  # Simplified
                    
                    if abs(gap_percent) >= min_gap_percent:
                        gap_candidates.append({
                            'symbol': symbol,
                            'gap_percent': gap_percent,
                            'prev_close': prev_close,
                            'current_price': current_price,
                            'volume': volume,
                            'volume_ratio': volume_ratio
                        })
            
            # Sort by gap size
            gap_candidates.sort(key=lambda x: abs(x['gap_percent']), reverse=True)
            return gap_candidates
            
        except Exception as e:
            logger.error(f"Error fetching gap candidates: {e}")
            return []
    
    async def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed ticker information including market cap"""
        try:
            response = await self.http_client._make_request(f"/v3/reference/tickers/{symbol}")
            
            if response.get('results'):
                details = response['results']
                return {
                    'symbol': symbol,
                    'name': details.get('name'),
                    'market_cap': details.get('market_cap'),
                    'sector': details.get('sic_description'),
                    'industry': details.get('sic_code'),
                    'description': details.get('description'),
                    'homepage_url': details.get('homepage_url'),
                    'total_employees': details.get('total_employees'),
                    'list_date': details.get('list_date'),
                    'locale': details.get('locale'),
                    'primary_exchange': details.get('primary_exchange'),
                    'type': details.get('type'),
                    'currency_name': details.get('currency_name'),
                    'cik': details.get('cik'),
                    'composite_figi': details.get('composite_figi'),
                    'share_class_figi': details.get('share_class_figi')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {e}")
            return None
    
    async def get_earnings_calendar(self, symbols: List[str], days_ahead: int = 7) -> List[Dict]:
        """Get earnings calendar information from news data"""
        try:
            today = date.today()
            future_date = today + timedelta(days=days_ahead)
            
            earnings_info = []
            
            for symbol in symbols:
                try:
                    # Get recent news
                    response = await self.http_client._make_request(
                        f"/v2/reference/news",
                        params={
                            'ticker': symbol,
                            'published_utc.gte': today.strftime('%Y-%m-%d'),
                            'published_utc.lte': future_date.strftime('%Y-%m-%d'),
                            'limit': 10
                        }
                    )
                    
                    if response.get('results'):
                        for article in response['results']:
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            
                            # Check for earnings keywords
                            earnings_keywords = [
                                'earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4',
                                'financial results', 'revenue', 'eps', 'guidance',
                                'earnings call', 'earnings report'
                            ]
                            
                            if any(keyword in title or keyword in description for keyword in earnings_keywords):
                                earnings_info.append({
                                    'symbol': symbol,
                                    'title': article.get('title'),
                                    'published_utc': article.get('published_utc'),
                                    'article_url': article.get('article_url'),
                                    'description': article.get('description')
                                })
                                break  # Found earnings news for this symbol
                
                except Exception as e:
                    logger.warning(f"Error getting earnings info for {symbol}: {e}")
                    continue
            
            return earnings_info
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    # ========== NEW HIGH-PRIORITY POLYGON API ENDPOINTS ==========
    
    async def get_market_movers(self, direction: str = "gainers", include_otc: bool = False) -> List[Dict]:
        """Get top market movers (gainers/losers) - HIGHEST PRIORITY for gap detection"""
        try:
            params = {}
            if include_otc:
                params['include_otc'] = 'true'
            
            response = await self.http_client._make_request(
                f"/v2/snapshot/locale/us/markets/stocks/{direction}",
                params=params
            )
            
            movers = []
            if response.get('results'):
                for ticker_data in response['results']:
                    if ticker_data.get('ticker'):
                        mover_info = {
                            'symbol': ticker_data['ticker'],
                            'price': ticker_data.get('value', 0.0),
                            'change': ticker_data.get('todaysChange', 0.0),
                            'change_percent': ticker_data.get('todaysChangePerc', 0.0),
                            'volume': ticker_data.get('day', {}).get('v', 0) if ticker_data.get('day') else 0,
                            'prev_close': ticker_data.get('prevDay', {}).get('c', 0.0) if ticker_data.get('prevDay') else 0.0,
                            'last_updated': ticker_data.get('updated', 0)
                        }
                        movers.append(mover_info)
            
            logger.info(f"Retrieved {len(movers)} market {direction}")
            return movers
            
        except Exception as e:
            logger.error(f"Error fetching market {direction}: {e}")
            return []
    
    async def get_rsi(self, symbol: str, window: int = 14, timespan: str = "day", 
                     limit: int = 10) -> Optional[Dict]:
        """Get RSI indicator - HIGH PRIORITY for mean reversion strategy"""
        try:
            params = {
                'timespan': timespan,
                'window': window,
                'series_type': 'close',
                'order': 'desc',
                'limit': limit
            }
            
            response = await self.http_client._make_request(
                f"/v1/indicators/rsi/{symbol}",
                params=params
            )
            
            if response.get('results') and response['results'].get('values'):
                values = response['results']['values']
                return {
                    'symbol': symbol,
                    'current_rsi': values[0].get('value', 0.0) if values else 0.0,
                    'values': values,
                    'window': window,
                    'timespan': timespan
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching RSI for {symbol}: {e}")
            return None
    
    async def get_macd(self, symbol: str, short_window: int = 12, long_window: int = 26,
                      signal_window: int = 9, timespan: str = "day", limit: int = 10) -> Optional[Dict]:
        """Get MACD indicator - HIGH PRIORITY for momentum confirmation"""
        try:
            params = {
                'timespan': timespan,
                'short_window': short_window,
                'long_window': long_window,
                'signal_window': signal_window,
                'series_type': 'close',
                'order': 'desc',
                'limit': limit
            }
            
            response = await self.http_client._make_request(
                f"/v1/indicators/macd/{symbol}",
                params=params
            )
            
            if response.get('results') and response['results'].get('values'):
                values = response['results']['values']
                current = values[0] if values else {}
                
                return {
                    'symbol': symbol,
                    'macd_line': current.get('value', 0.0),
                    'signal_line': current.get('signal', 0.0),
                    'histogram': current.get('histogram', 0.0),
                    'values': values,
                    'short_window': short_window,
                    'long_window': long_window,
                    'signal_window': signal_window
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching MACD for {symbol}: {e}")
            return None
    
    async def get_sma(self, symbol: str, window: int = 20, timespan: str = "day",
                     limit: int = 10) -> Optional[Dict]:
        """Get Simple Moving Average - MEDIUM PRIORITY"""
        try:
            params = {
                'timespan': timespan,
                'window': window,
                'series_type': 'close',
                'order': 'desc',
                'limit': limit
            }
            
            response = await self.http_client._make_request(
                f"/v1/indicators/sma/{symbol}",
                params=params
            )
            
            if response.get('results') and response['results'].get('values'):
                values = response['results']['values']
                return {
                    'symbol': symbol,
                    'current_sma': values[0].get('value', 0.0) if values else 0.0,
                    'values': values,
                    'window': window,
                    'timespan': timespan
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching SMA for {symbol}: {e}")
            return None
    
    async def get_ema(self, symbol: str, window: int = 20, timespan: str = "day",
                     limit: int = 10) -> Optional[Dict]:
        """Get Exponential Moving Average - MEDIUM PRIORITY"""
        try:
            params = {
                'timespan': timespan,
                'window': window,
                'series_type': 'close',
                'order': 'desc',
                'limit': limit
            }
            
            response = await self.http_client._make_request(
                f"/v1/indicators/ema/{symbol}",
                params=params
            )
            
            if response.get('results') and response['results'].get('values'):
                values = response['results']['values']
                return {
                    'symbol': symbol,
                    'current_ema': values[0].get('value', 0.0) if values else 0.0,
                    'values': values,
                    'window': window,
                    'timespan': timespan
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching EMA for {symbol}: {e}")
            return None
    
    async def get_bollinger_bands(self, symbol: str, window: int = 20, timespan: str = "day",
                                limit: int = 10) -> Optional[Dict]:
        """Get Bollinger Bands - MEDIUM PRIORITY for volatility analysis"""
        try:
            params = {
                'timespan': timespan,
                'window': window,
                'series_type': 'close',
                'order': 'desc',
                'limit': limit
            }
            
            response = await self.http_client._make_request(
                f"/v1/indicators/bb/{symbol}",
                params=params
            )
            
            if response.get('results') and response['results'].get('values'):
                values = response['results']['values']
                current = values[0] if values else {}
                
                return {
                    'symbol': symbol,
                    'upper_band': current.get('upper_band', 0.0),
                    'middle_band': current.get('middle_band', 0.0),
                    'lower_band': current.get('lower_band', 0.0),
                    'values': values,
                    'window': window,
                    'timespan': timespan
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Bollinger Bands for {symbol}: {e}")
            return None
    
    async def get_multi_timeframe_indicators(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Dict]:
        """Get professional indicators across multiple timeframes - ULTRA HIGH PRIORITY for multi-timeframe analysis"""
        
        if timeframes is None:
            timeframes = ["minute", "hour"]  # Focus on intraday timeframes
        
        results = {}
        
        try:
            logger.info(f"Fetching multi-timeframe indicators for {symbol} across {timeframes}")
            
            for timespan in timeframes:
                timeframe_data = {}
                
                # Batch fetch all indicators for this timeframe
                tasks = [
                    self.get_rsi(symbol, window=14, timespan=timespan),
                    self.get_macd(symbol, timespan=timespan),
                    self.get_sma(symbol, window=20, timespan=timespan),
                    self.get_ema(symbol, window=12, timespan=timespan),
                    self.get_bollinger_bands(symbol, window=20, timespan=timespan)
                ]
                
                # Execute all requests concurrently
                rsi_data, macd_data, sma_data, ema_data, bb_data = await asyncio.gather(
                    *tasks, return_exceptions=True
                )
                
                # Process results
                timeframe_data['rsi'] = rsi_data if not isinstance(rsi_data, Exception) else None
                timeframe_data['macd'] = macd_data if not isinstance(macd_data, Exception) else None
                timeframe_data['sma_20'] = sma_data if not isinstance(sma_data, Exception) else None
                timeframe_data['ema_12'] = ema_data if not isinstance(ema_data, Exception) else None
                timeframe_data['bollinger_bands'] = bb_data if not isinstance(bb_data, Exception) else None
                timeframe_data['timestamp'] = datetime.now()
                
                results[timespan] = timeframe_data
                
                # Rate limiting between timeframes
                await asyncio.sleep(1)
            
            logger.info(f"Retrieved multi-timeframe indicators for {symbol}: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe indicators for {symbol}: {e}")
            return {}
    
    async def get_grouped_daily_aggs(self, date_str: Optional[str] = None) -> Dict:
        """Get grouped daily aggregates for all stocks - MEDIUM PRIORITY for batch updates"""
        try:
            if date_str is None:
                # Get previous trading day
                today = date.today()
                target_date = today - timedelta(days=1)
                
                # Handle weekends
                while target_date.weekday() >= 5:
                    target_date -= timedelta(days=1)
                
                date_str = target_date.strftime("%Y-%m-%d")
            
            response = await self.http_client._make_request(
                f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching grouped daily aggregates: {e}")
            return {}
    
    async def get_snapshot_ticker(self, symbol: str) -> Optional[Dict]:
        """Get snapshot for specific ticker - MEDIUM PRIORITY for targeted updates"""
        try:
            response = await self.http_client._make_request(
                f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            )
            
            if response.get('results'):
                ticker_data = response['results']
                return {
                    'symbol': symbol,
                    'price': ticker_data.get('value', 0.0),
                    'change': ticker_data.get('todaysChange', 0.0),
                    'change_percent': ticker_data.get('todaysChangePerc', 0.0),
                    'volume': ticker_data.get('day', {}).get('v', 0) if ticker_data.get('day') else 0,
                    'prev_close': ticker_data.get('prevDay', {}).get('c', 0.0) if ticker_data.get('prevDay') else 0.0,
                    'day_data': ticker_data.get('day', {}),
                    'prev_day_data': ticker_data.get('prevDay', {}),
                    'last_quote': ticker_data.get('lastQuote', {}),
                    'last_trade': ticker_data.get('lastTrade', {}),
                    'last_updated': ticker_data.get('updated', 0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
            return None
    
    # ========== ENHANCED GAP DETECTION METHODS ==========
    
    async def get_enhanced_gap_candidates(self, min_gap_percent: float = 2.0,
                                        min_volume_ratio: float = 1.5) -> List[Dict]:
        """Enhanced gap detection using market movers API - ULTRA HIGH PRIORITY"""
        try:
            logger.info("Fetching enhanced gap candidates using market movers...")
            
            # Get both gainers and losers
            gainers = await self.get_market_movers("gainers")
            losers = await self.get_market_movers("losers")
            
            all_movers = gainers + losers
            gap_candidates = []
            
            # Get our active symbols for filtering
            active_symbols = set(symbol_manager.get_active_symbols())
            
            for mover in all_movers:
                symbol = mover['symbol']
                
                # Only analyze symbols we're tracking
                if symbol not in active_symbols:
                    continue
                
                gap_percent = mover['change_percent']
                volume = mover['volume']
                
                # Check gap criteria
                if abs(gap_percent) >= min_gap_percent:
                    # Get additional data for volume ratio calculation
                    try:
                        # Get historical average volume (simplified - using current volume as baseline)
                        avg_volume = symbol_manager.metrics.get(symbol, type('obj', (object,), {'avg_volume': volume})).avg_volume or volume
                        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                        
                        if volume_ratio >= min_volume_ratio:
                            gap_candidate = {
                                'symbol': symbol,
                                'gap_percent': gap_percent,
                                'volume_ratio': volume_ratio,
                                'price': mover['price'],
                                'prev_close': mover['prev_close'],
                                'volume': volume,
                                'change': mover['change'],
                                'direction': 'up' if gap_percent > 0 else 'down',
                                'last_updated': mover['last_updated']
                            }
                            gap_candidates.append(gap_candidate)
                            
                            # Update symbol metrics immediately
                            symbol_manager.update_symbol_metrics(
                                symbol,
                                price=mover['price'],
                                prev_close=mover['prev_close'],
                                gap_percent=gap_percent,
                                volume=volume,
                                volume_ratio=volume_ratio
                            )
                    
                    except Exception as e:
                        logger.warning(f"Error processing gap candidate {symbol}: {e}")
                        continue
            
            # Sort by absolute gap percentage
            gap_candidates.sort(key=lambda x: abs(x['gap_percent']), reverse=True)
            
            logger.info(f"Found {len(gap_candidates)} enhanced gap candidates")
            
            # Store gap candidates in database
            await self._store_gap_candidates(gap_candidates)
            
            return gap_candidates
            
        except Exception as e:
            logger.error(f"Error fetching enhanced gap candidates: {e}")
            return []
    
    async def get_professional_indicators_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get professional indicators for multiple symbols - BATCH OPTIMIZATION"""
        try:
            logger.info(f"Fetching professional indicators for {len(symbols)} symbols...")
            
            indicators_data = {}
            
            # Process symbols in batches to avoid overwhelming the API
            batch_size = 5
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Fetch indicators for this batch
                batch_tasks = []
                for symbol in batch:
                    batch_tasks.extend([
                        self.get_rsi(symbol),
                        self.get_macd(symbol),
                        self.get_sma(symbol, window=20),
                        self.get_ema(symbol, window=12)
                    ])
                
                # Execute batch requests concurrently
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, symbol in enumerate(batch):
                    base_idx = j * 4
                    rsi_data = results[base_idx] if not isinstance(results[base_idx], Exception) else None
                    macd_data = results[base_idx + 1] if not isinstance(results[base_idx + 1], Exception) else None
                    sma_data = results[base_idx + 2] if not isinstance(results[base_idx + 2], Exception) else None
                    ema_data = results[base_idx + 3] if not isinstance(results[base_idx + 3], Exception) else None
                    
                    indicators_data[symbol] = {
                        'rsi': rsi_data,
                        'macd': macd_data,
                        'sma_20': sma_data,
                        'ema_12': ema_data,
                        'timestamp': datetime.now()
                    }
                
                # Rate limiting between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(2)  # 2 second pause between batches
            
            logger.info(f"Retrieved professional indicators for {len(indicators_data)} symbols")
            return indicators_data
            
        except Exception as e:
            logger.error(f"Error fetching professional indicators batch: {e}")
            return {}
    
    async def get_market_regime_indicators(self) -> Dict:
        """Get market regime indicators using major indices - MEDIUM PRIORITY"""
        try:
            logger.info("Analyzing market regime using index indicators...")
            
            # Major indices to analyze
            indices = ['SPY', 'QQQ', 'IWM', 'DIA']
            
            regime_data = {}
            
            # Get MACD for each index
            for index in indices:
                macd_data = await self.get_macd(index)
                if macd_data:
                    regime_data[index] = {
                        'macd_line': macd_data['macd_line'],
                        'signal_line': macd_data['signal_line'],
                        'histogram': macd_data['histogram'],
                        'momentum': 'bullish' if macd_data['histogram'] > 0 else 'bearish'
                    }
            
            # Determine overall market regime
            bullish_count = sum(1 for data in regime_data.values() if data.get('momentum') == 'bullish')
            total_indices = len(regime_data)
            
            if bullish_count >= total_indices * 0.75:
                overall_regime = 'strong_bullish'
            elif bullish_count >= total_indices * 0.5:
                overall_regime = 'bullish'
            elif bullish_count <= total_indices * 0.25:
                overall_regime = 'strong_bearish'
            else:
                overall_regime = 'bearish'
            
            regime_analysis = {
                'overall_regime': overall_regime,
                'bullish_indices': bullish_count,
                'total_indices': total_indices,
                'bullish_percentage': (bullish_count / total_indices) * 100 if total_indices > 0 else 0,
                'indices_data': regime_data,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Market regime: {overall_regime} ({bullish_count}/{total_indices} indices bullish)")
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {}
    
    def get_latest_trade(self, symbol: str) -> Optional[TradeData]:
        """Get latest trade for symbol"""
        return self.latest_trades.get(symbol)
    
    def get_latest_quote(self, symbol: str) -> Optional[QuoteData]:
        """Get latest quote for symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_latest_aggregate(self, symbol: str) -> Optional[AggregateData]:
        """Get latest aggregate for symbol"""
        return self.latest_aggregates.get(symbol)
    
    def get_daily_bar(self, symbol: str) -> Optional[DailyBarData]:
        """Get previous day bar for symbol"""
        return self.daily_bars.get(symbol)
    
    async def subscribe_symbol(self, symbol: str):
        """Subscribe to a new symbol"""
        await self.ws_client.subscribe([symbol])
    
    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol"""
        await self.ws_client.unsubscribe([symbol])
    
    def get_connection_stats(self) -> Dict:
        """Get connection and performance statistics"""
        uptime = None
        if self.ws_client.connection_start_time:
            uptime = datetime.now() - self.ws_client.connection_start_time
        
        return {
            'connected': self.ws_client.connected,
            'uptime': str(uptime) if uptime else None,
            'messages_received': self.ws_client.messages_received,
            'subscribed_symbols': len(self.ws_client.subscribed_symbols),
            'symbols_with_trades': len(self.latest_trades),
            'symbols_with_quotes': len(self.latest_quotes),
            'cache_entries': len(self.historical_cache)
        }
    
    async def cleanup(self):
        """Cleanup connections and resources"""
        logger.info("Cleaning up Polygon connections...")
        
        if self.message_handler_task:
            self.message_handler_task.cancel()
        
        await self.ws_client.close()
        await self.http_client.close()
        
        logger.info("Polygon cleanup completed")
    
    async def _store_gap_candidates(self, gap_candidates: List[Dict]):
        """Store gap candidates in database"""
        
        if not self.db_manager or not gap_candidates:
            return
        
        try:
            for gap in gap_candidates:
                await self.db_manager.insert_gap_candidate(
                    symbol=gap['symbol'],
                    gap_type=gap['direction'],
                    gap_percent=gap['gap_percent'],
                    previous_close=gap['prev_close'],
                    current_price=gap['price'],
                    volume=gap['volume'],
                    volume_ratio=gap['volume_ratio']
                )
            
            logger.info(f"Stored {len(gap_candidates)} gap candidates in database")
            
        except Exception as e:
            logger.error(f"Error storing gap candidates: {e}")
    
    async def _store_market_data(self, symbol: str, data_type: str, data: Dict):
        """Store market data in database"""
        
        if not self.db_manager:
            return
        
        try:
            market_data = {
                'symbol': symbol,
                'data_type': data_type,
                'timestamp': datetime.now(),
                'data': data
            }
            
            await self.db_manager.insert_market_data(market_data)
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def _store_earnings_calendar(self, earnings_data: List[Dict]):
        """Store earnings calendar data in database"""
        
        if not self.db_manager or not earnings_data:
            return
        
        try:
            for earnings in earnings_data:
                await self.db_manager.insert_earnings_event(
                    symbol=earnings['symbol'],
                    event_date=earnings.get('published_utc'),
                    event_type='earnings',
                    description=earnings.get('title', ''),
                    metadata={
                        'article_url': earnings.get('article_url'),
                        'description': earnings.get('description')
                    }
                )
            
            logger.info(f"Stored {len(earnings_data)} earnings events in database")
            
        except Exception as e:
            logger.error(f"Error storing earnings calendar: {e}")

# Global instance (lazy-initialized)
polygon_data_manager = None

def get_polygon_data_manager():
    """Get or create the global polygon data manager"""
    global polygon_data_manager
    if polygon_data_manager is None:
        polygon_data_manager = PolygonDataManager()
    return polygon_data_manager
