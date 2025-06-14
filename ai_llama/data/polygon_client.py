"""
Polygon.io REST and WebSocket Client

Ultra-fast market data integration with Polygon.io API.
Features:
- Real-time market data via WebSocket
- Historical data via REST API
- Rate limiting and connection management
- Automatic reconnection and error handling
"""

import asyncio
import aiohttp
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from urllib.parse import urlencode

@dataclass
class PolygonConfig:
    """Configuration for Polygon.io client"""
    api_key: str
    base_url: str = "https://api.polygon.io"
    websocket_url: str = "wss://socket.polygon.io"
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests_per_minute: int = 5000  # Adjust based on your plan
    connection_timeout: int = 30

@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    timestamp: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'spread': self.spread
        }

@dataclass
class Trade:
    """Real-time trade data"""
    symbol: str
    timestamp: int
    price: float
    size: int
    conditions: List[int]
    exchange: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.price,
            'size': self.size,
            'conditions': self.conditions,
            'exchange': self.exchange
        }

@dataclass
class Aggregate:
    """Aggregate bar data"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    transactions: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'transactions': self.transactions
        }

class PolygonRESTClient:
    """
    High-performance Polygon.io REST API client
    
    Features:
    - Async HTTP requests for maximum throughput
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Response caching
    """
    
    def __init__(self, config: PolygonConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_times = []
        self.last_request_time = 0
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we're hitting rate limit
        if len(self.request_times) >= self.config.rate_limit_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated API request with error handling"""
        if not self.session:
            await self.connect()
        
        self._check_rate_limit()
        
        # Add API key to parameters
        params = params or {}
        params['apikey'] = self.config.api_key
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        self.logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        self.logger.error(f"API error {response.status}: {await response.text()}")
                        
            except Exception as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise Exception(f"Failed to fetch data from {endpoint} after {self.config.max_retries} attempts")
    
    async def get_aggregates(self, 
                           symbol: str, 
                           multiplier: int = 1,
                           timespan: str = "minute",
                           from_date: str = None,
                           to_date: str = None,
                           limit: int = 120) -> List[Aggregate]:
        """
        Get aggregate bars for a stock
        
        Args:
            symbol: Stock symbol
            multiplier: Multiplier for timespan (1, 5, 15, etc.)
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results to return
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit
        }
        
        data = await self._make_request(endpoint, params)
        
        aggregates = []
        if 'results' in data:
            for result in data['results']:
                agg = Aggregate(
                    symbol=symbol,
                    timestamp=result['t'],
                    open=result['o'],
                    high=result['h'],
                    low=result['l'],
                    close=result['c'],
                    volume=result['v'],
                    vwap=result.get('vw'),
                    transactions=result.get('n')
                )
                aggregates.append(agg)
        
        return aggregates
    
    async def get_previous_close(self, symbol: str) -> Optional[Aggregate]:
        """Get previous trading day's close for a symbol"""
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        
        data = await self._make_request(endpoint)
        
        if 'results' in data and data['results']:
            result = data['results'][0]
            return Aggregate(
                symbol=symbol,
                timestamp=result['T'],
                open=result['o'],
                high=result['h'],
                low=result['l'],
                close=result['c'],
                volume=result['v'],
                vwap=result.get('vw'),
                transactions=result.get('n')
            )
        
        return None
    
    async def get_trades(self, 
                        symbol: str, 
                        timestamp: str,
                        limit: int = 1000) -> List[Trade]:
        """Get trades for a symbol on a specific day"""
        endpoint = f"/v3/trades/{symbol}"
        
        params = {
            'timestamp': timestamp,
            'limit': limit,
            'sort': 'timestamp'
        }
        
        data = await self._make_request(endpoint, params)
        
        trades = []
        if 'results' in data:
            for result in data['results']:
                trade = Trade(
                    symbol=symbol,
                    timestamp=result['participant_timestamp'],
                    price=result['price'],
                    size=result['size'],
                    conditions=result.get('conditions', []),
                    exchange=result.get('exchange', 0)
                )
                trades.append(trade)
        
        return trades
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        endpoint = "/v1/marketstatus/now"
        return await self._make_request(endpoint)
    
    async def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a ticker"""
        endpoint = f"/v3/reference/tickers/{symbol}"
        return await self._make_request(endpoint)

class PolygonWebSocketClient:
    """
    High-performance Polygon.io WebSocket client
    
    Features:
    - Real-time quotes and trades
    - Automatic reconnection
    - Multiple subscription management
    - Callback-based event handling
    """
    
    def __init__(self, config: PolygonConfig):
        self.config = config
        self.websocket = None
        self.subscriptions = set()
        self.callbacks = {
            'quote': [],
            'trade': [],
            'aggregate': [],
            'status': [],
            'error': []
        }
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            self.websocket = await websockets.connect(
                f"{self.config.websocket_url}/stocks",
                timeout=self.config.connection_timeout
            )
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.config.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await self.websocket.recv()
            auth_data = json.loads(response)
            
            if auth_data[0].get('status') != 'auth_success':
                raise Exception(f"Authentication failed: {auth_data}")
            
            self.logger.info("Connected to Polygon WebSocket")
            self.running = True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
    
    async def subscribe(self, symbols: List[str], data_types: List[str] = None):
        """
        Subscribe to real-time data for symbols
        
        Args:
            symbols: List of stock symbols
            data_types: List of data types ['Q' for quotes, 'T' for trades, 'A' for aggregates]
        """
        if data_types is None:
            data_types = ['Q', 'T']  # Quotes and trades by default
        
        for data_type in data_types:
            for symbol in symbols:
                subscription = f"{data_type}.{symbol}"
                self.subscriptions.add(subscription)
        
        subscribe_message = {
            "action": "subscribe",
            "params": ",".join(self.subscriptions)
        }
        
        await self.websocket.send(json.dumps(subscribe_message))
        self.logger.info(f"Subscribed to {len(self.subscriptions)} streams")
    
    async def unsubscribe(self, symbols: List[str], data_types: List[str] = None):
        """Unsubscribe from symbols"""
        if data_types is None:
            data_types = ['Q', 'T']
        
        to_remove = set()
        for data_type in data_types:
            for symbol in symbols:
                subscription = f"{data_type}.{symbol}"
                to_remove.add(subscription)
        
        self.subscriptions -= to_remove
        
        unsubscribe_message = {
            "action": "unsubscribe", 
            "params": ",".join(to_remove)
        }
        
        await self.websocket.send(json.dumps(unsubscribe_message))
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            for event in data:
                event_type = event.get('ev')
                
                if event_type == 'Q':  # Quote
                    quote = Quote(
                        symbol=event['sym'],
                        timestamp=event['t'],
                        bid=event['bp'],
                        ask=event['ap'],
                        bid_size=event['bs'],
                        ask_size=event['as'],
                        spread=event['ap'] - event['bp']
                    )
                    
                    for callback in self.callbacks['quote']:
                        try:
                            await callback(quote)
                        except Exception as e:
                            self.logger.error(f"Quote callback error: {e}")
                
                elif event_type == 'T':  # Trade
                    trade = Trade(
                        symbol=event['sym'],
                        timestamp=event['t'],
                        price=event['p'],
                        size=event['s'],
                        conditions=event.get('c', []),
                        exchange=event.get('x', 0)
                    )
                    
                    for callback in self.callbacks['trade']:
                        try:
                            await callback(trade)
                        except Exception as e:
                            self.logger.error(f"Trade callback error: {e}")
                
                elif event_type == 'A':  # Aggregate
                    aggregate = Aggregate(
                        symbol=event['sym'],
                        timestamp=event['s'],
                        open=event['o'],
                        high=event['h'],
                        low=event['l'],
                        close=event['c'],
                        volume=event['v'],
                        vwap=event.get('vw')
                    )
                    
                    for callback in self.callbacks['aggregate']:
                        try:
                            await callback(aggregate)
                        except Exception as e:
                            self.logger.error(f"Aggregate callback error: {e}")
                
                elif event_type == 'status':
                    for callback in self.callbacks['status']:
                        try:
                            await callback(event)
                        except Exception as e:
                            self.logger.error(f"Status callback error: {e}")
                            
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            for callback in self.callbacks['error']:
                try:
                    await callback(e)
                except:
                    pass
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while self.running:
                message = await self.websocket.recv()
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            if self.running:
                # Attempt reconnection
                await self._reconnect()
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
            if self.running:
                await self._reconnect()
    
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket"""
        self.logger.info("Attempting to reconnect...")
        
        for attempt in range(self.config.max_retries):
            try:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                await self.connect()
                
                # Re-subscribe to previous subscriptions
                if self.subscriptions:
                    subscribe_message = {
                        "action": "subscribe",
                        "params": ",".join(self.subscriptions)
                    }
                    await self.websocket.send(json.dumps(subscribe_message))
                
                self.logger.info("Reconnected successfully")
                return
                
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self.logger.error("Failed to reconnect after maximum attempts")
        self.running = False

class PolygonClient:
    """
    Combined Polygon.io client with both REST and WebSocket capabilities
    """
    
    def __init__(self, api_key: str):
        config = PolygonConfig(api_key=api_key)
        self.rest = PolygonRESTClient(config)
        self.websocket = PolygonWebSocketClient(config)
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start both REST and WebSocket clients"""
        await self.rest.connect()
        await self.websocket.connect()
    
    async def stop(self):
        """Stop both clients"""
        await self.rest.disconnect()
        await self.websocket.disconnect()
    
    def add_quote_callback(self, callback: Callable):
        """Add callback for real-time quotes"""
        self.websocket.add_callback('quote', callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add callback for real-time trades"""
        self.websocket.add_callback('trade', callback)
    
    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        await self.websocket.subscribe(symbols, ['Q', 'T', 'A'])
    
    async def get_historical_data(self, 
                                symbol: str, 
                                days: int = 5, 
                                timespan: str = "minute") -> pd.DataFrame:
        """Get historical data as pandas DataFrame"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        aggregates = await self.rest.get_aggregates(
            symbol=symbol,
            multiplier=1,
            timespan=timespan,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d'),
            limit=5000
        )
        
        if not aggregates:
            return pd.DataFrame()
        
        data = [agg.to_dict() for agg in aggregates]
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        return df

    async def get_indices_data(self, 
                             indices: List[str] = None, 
                             days: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for major indices
        
        Args:
            indices: List of index symbols (defaults to major indices)
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping index symbols to DataFrames
        """
        if indices is None:
            # Major US indices
            indices = [
                'SPY',    # S&P 500 ETF
                'QQQ',    # Nasdaq-100 ETF
                'IWM',    # Russell 2000 ETF
                'DIA',    # Dow Jones ETF
                'VTI',    # Total Stock Market ETF
                'VXX',    # Volatility ETF
                'TLT',    # 20+ Year Treasury ETF
                'GLD',    # Gold ETF
                'USO',    # Oil ETF
                'UUP',    # Dollar ETF
            ]
        
        results = {}
        
        for index in indices:
            try:
                data = await self.get_historical_data(index, days=days)
                if not data.empty:
                    results[index] = data
                    self.logger.info(f"Retrieved {len(data)} bars for index {index}")
                else:
                    self.logger.warning(f"No data found for index {index}")
            except Exception as e:
                self.logger.error(f"Failed to get data for index {index}: {e}")
        
        return results
    
    async def get_sector_etfs_data(self, days: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for sector ETFs
        
        Returns:
            Dictionary mapping sector ETF symbols to DataFrames
        """
        sector_etfs = [
            'XLK',  # Technology
            'XLF',  # Financials
            'XLV',  # Health Care
            'XLE',  # Energy
            'XLI',  # Industrials
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLB',  # Materials
            'XLU',  # Utilities
            'XLRE', # Real Estate
            'XLC',  # Communication Services
        ]
        
        results = {}
        
        for etf in sector_etfs:
            try:
                data = await self.get_historical_data(etf, days=days)
                if not data.empty:
                    results[etf] = data
                    self.logger.info(f"Retrieved {len(data)} bars for sector ETF {etf}")
            except Exception as e:
                self.logger.error(f"Failed to get data for sector ETF {etf}: {e}")
        
        return results
    
    async def subscribe_to_market_overview(self):
        """Subscribe to comprehensive market overview including stocks and indices"""
        
        # Major indices
        indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VXX']
        
        # Major stocks
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        # Sector ETFs
        sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY']
        
        all_symbols = indices + stocks + sectors
        
        await self.subscribe_to_symbols(all_symbols)
        self.logger.info(f"Subscribed to market overview: {len(all_symbols)} symbols")
        self.logger.info(f"Indices: {indices}")
        self.logger.info(f"Stocks: {stocks}")
        self.logger.info(f"Sectors: {sectors}")

# Market data categories for easy access
MARKET_CATEGORIES = {
    'major_indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
    'volatility_indices': ['VXX', 'UVXY', 'SVXY'],
    'commodities': ['GLD', 'SLV', 'USO', 'UNG'],
    'currencies': ['UUP', 'FXE', 'FXY', 'EWZ'],
    'bonds': ['TLT', 'IEF', 'SHY', 'JNK', 'HYG'],
    'mega_cap_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
    'growth_stocks': ['NFLX', 'CRM', 'ADBE', 'PYPL', 'ROKU', 'ZM', 'PTON'],
    'value_stocks': ['BRK.B', 'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V'],
    'sector_etfs': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE', 'XLC'],
    'meme_stocks': ['GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV'],
    'chinese_stocks': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI'],
    'crypto_related': ['COIN', 'MSTR', 'SQ', 'RIOT', 'MARA']
}

def get_symbols_by_category(categories: List[str]) -> List[str]:
    """Get symbols for specific market categories"""
    symbols = []
    for category in categories:
        if category in MARKET_CATEGORIES:
            symbols.extend(MARKET_CATEGORIES[category])
    return list(set(symbols))  # Remove duplicates

# Example usage
async def example_usage():
    """Example of comprehensive market data usage including stocks and indices"""
    
    # Initialize client
    client = PolygonClient(api_key="YOUR_API_KEY")
    
    # Define callbacks for real-time data
    async def handle_quote(quote: Quote):
        symbol_type = "INDEX" if quote.symbol in MARKET_CATEGORIES['major_indices'] else "STOCK"
        print(f"{symbol_type} Quote: {quote.symbol} Bid: ${quote.bid:.2f} Ask: ${quote.ask:.2f} Spread: ${quote.spread:.3f}")
    
    async def handle_trade(trade: Trade):
        symbol_type = "INDEX" if trade.symbol in MARKET_CATEGORIES['major_indices'] else "STOCK"
        print(f"{symbol_type} Trade: {trade.symbol} Price: ${trade.price:.2f} Size: {trade.size:,}")
    
    async def handle_aggregate(agg: Aggregate):
        symbol_type = "INDEX" if agg.symbol in MARKET_CATEGORIES['major_indices'] else "STOCK"
        print(f"{symbol_type} Bar: {agg.symbol} O: ${agg.open:.2f} H: ${agg.high:.2f} L: ${agg.low:.2f} C: ${agg.close:.2f} V: {agg.volume:,}")
    
    # Set up callbacks
    client.add_quote_callback(handle_quote)
    client.add_trade_callback(handle_trade)
    client.websocket.add_callback('aggregate', handle_aggregate)
    
    try:
        # Start the client
        await client.start()
        print("🚀 Connected to Polygon.io")
        
        # Subscribe to comprehensive market overview
        await client.subscribe_to_market_overview()
        print("📊 Subscribed to market overview")
        
        # Get historical data for major indices
        print("\n📈 Getting historical data for major indices...")
        indices_data = await client.get_indices_data(days=2)
        for symbol, data in indices_data.items():
            print(f"   {symbol}: {len(data)} bars, Latest: ${data['close'].iloc[-1]:.2f}")
        
        # Get sector ETF data
        print("\n🏭 Getting sector ETF data...")
        sector_data = await client.get_sector_etfs_data(days=1)
        for symbol, data in sector_data.items():
            print(f"   {symbol}: {len(data)} bars, Latest: ${data['close'].iloc[-1]:.2f}")
        
        # Get specific category data
        print("\n💎 Getting mega-cap stocks data...")
        mega_caps = get_symbols_by_category(['mega_cap_stocks'])
        for symbol in mega_caps[:3]:  # First 3 for demo
            data = await client.get_historical_data(symbol, days=1)
            if not data.empty:
                print(f"   {symbol}: {len(data)} bars, Latest: ${data['close'].iloc[-1]:.2f}")
        
        print("\n🎯 Available market categories:")
        for category, symbols in MARKET_CATEGORIES.items():
            print(f"   {category}: {len(symbols)} symbols")
        
        print("\n📡 Listening for real-time data... (Press Ctrl+C to stop)")
        
        # Listen for real-time data (run this in the background)
        await client.websocket.listen()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping data feed...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.stop()
        print("🔌 Disconnected from Polygon.io")

async def demo_trading_universe():
    """Demo showing how to build a comprehensive trading universe"""
    
    client = PolygonClient(api_key="YOUR_API_KEY")
    
    try:
        await client.start()
        
        print("🌍 Building comprehensive trading universe...")
        
        # Get data for different asset classes
        categories_to_fetch = [
            'major_indices',
            'mega_cap_stocks', 
            'sector_etfs',
            'volatility_indices',
            'commodities'
        ]
        
        universe = {}
        
        for category in categories_to_fetch:
            symbols = MARKET_CATEGORIES[category]
            print(f"\n📊 Fetching {category} ({len(symbols)} symbols)...")
            
            for symbol in symbols:
                try:
                    data = await client.get_historical_data(symbol, days=1)
                    if not data.empty:
                        universe[symbol] = {
                            'category': category,
                            'data': data,
                            'current_price': data['close'].iloc[-1],
                            'volume': data['volume'].iloc[-1],
                            'daily_change': (data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1] * 100
                        }
                        print(f"   ✅ {symbol}: ${data['close'].iloc[-1]:.2f} ({universe[symbol]['daily_change']:+.2f}%)")
                except Exception as e:
                    print(f"   ❌ {symbol}: Failed - {e}")
        
        print(f"\n🎯 Trading universe built: {len(universe)} symbols")
        
        # Sort by volume (liquidity)
        sorted_by_volume = sorted(universe.items(), key=lambda x: x[1]['volume'], reverse=True)
        print(f"\n💧 Top 10 by volume:")
        for symbol, data in sorted_by_volume[:10]:
            print(f"   {symbol}: {data['volume']:,} shares, ${data['current_price']:.2f}")
        
        # Sort by daily change
        sorted_by_change = sorted(universe.items(), key=lambda x: abs(x[1]['daily_change']), reverse=True)
        print(f"\n📈 Top 10 by daily movement:")
        for symbol, data in sorted_by_change[:10]:
            print(f"   {symbol}: {data['daily_change']:+.2f}%, ${data['current_price']:.2f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_trading_universe())
    else:
        asyncio.run(example_usage())
