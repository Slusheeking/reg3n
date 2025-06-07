#!/usr/bin/env python3

import os
import asyncio
import json
import logging
import time
import websockets
import aiohttp
import yaml
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Import enhanced logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

# Load environment variables
load_dotenv()

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'data_pipeline.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

# Initialize logger
logger = get_system_logger("data_pipeline.polygon_websocket")


@dataclass
class MarketData:
    """Container for real-time market data"""
    symbol: str
    timestamp: float
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    data_type: str = "trade"  # trade, quote, aggregate


class ConnectionHealthMonitor:
    """Monitor WebSocket connection health using YAML config"""
    
    def __init__(self):
        health_config = CONFIG.get('health_monitoring', {})
        self.heartbeat_interval = health_config.get('heartbeat_interval', 30)
        self.max_reconnect_attempts = health_config.get('max_reconnect_attempts', 10)
        self.data_timeout_seconds = health_config.get('data_timeout_seconds', 60)
        self.last_heartbeat = time.time()
        self.last_data_received = time.time()
        self.connection_status = "disconnected"
        self.reconnect_count = 0
        self.total_messages_received = 0
        
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = time.time()
        
    def update_data_received(self):
        """Update data received timestamp"""
        self.last_data_received = time.time()
        self.total_messages_received += 1
        
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        now = time.time()
        heartbeat_ok = (now - self.last_heartbeat) < (self.heartbeat_interval * 2)
        data_flow_ok = (now - self.last_data_received) < self.data_timeout_seconds
        return heartbeat_ok and data_flow_ok and self.connection_status == "connected"
    
    def get_status(self) -> Dict:
        """Get connection status"""
        now = time.time()
        return {
            "status": self.connection_status,
            "last_heartbeat_seconds_ago": now - self.last_heartbeat,
            "last_data_seconds_ago": now - self.last_data_received,
            "reconnect_count": self.reconnect_count,
            "total_messages": self.total_messages_received,
            "is_healthy": self.is_healthy()
        }


class DataValidator:
    """Validate incoming market data using YAML config"""
    
    def __init__(self):
        validation_config = CONFIG.get('data_processing', {}).get('validation', {})
        self.enabled = validation_config.get('enabled', True)
        price_range = validation_config.get('price_range', {})
        volume_range = validation_config.get('volume_range', {})
        
        self.min_price = price_range.get('min', 0.01)
        self.max_price = price_range.get('max', 1000000)
        self.min_volume = volume_range.get('min', 0)
        self.max_volume = volume_range.get('max', 1000000000)
    
    def validate_trade_data(self, data: Dict) -> bool:
        """Validate trade data structure"""
        if not self.enabled:
            return True
        required_fields = ['sym', 'p', 's', 't']  # symbol, price, size, timestamp
        return all(field in data for field in required_fields)
    
    def validate_quote_data(self, data: Dict) -> bool:
        """Validate quote data structure"""
        if not self.enabled:
            return True
        required_fields = ['sym', 'bp', 'ap', 't']  # symbol, bid_price, ask_price, timestamp
        return all(field in data for field in required_fields)
    
    def sanitize_price(self, price: Any) -> Optional[float]:
        """Sanitize price data using config ranges"""
        if not self.enabled:
            try:
                return float(price)
            except (ValueError, TypeError):
                return None
        
        try:
            price_float = float(price)
            if self.min_price <= price_float <= self.max_price:
                return price_float
            return None
        except (ValueError, TypeError):
            return None
    
    def sanitize_volume(self, volume: Any) -> Optional[int]:
        """Sanitize volume data using config ranges"""
        if not self.enabled:
            try:
                return int(volume)
            except (ValueError, TypeError):
                return None
        
        try:
            volume_int = int(volume)
            if self.min_volume <= volume_int <= self.max_volume:
                return volume_int
            return None
        except (ValueError, TypeError):
            return None


class SymbolManager:
    """Manages all tradeable symbols using YAML config and A100 optimizations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.all_symbols: Set[str] = set()
        self.active_symbols: Set[str] = set()
        self.symbol_metadata: Dict[str, dict] = {}
        
        # Load config
        symbols_config = CONFIG.get('data_processing', {}).get('symbols', {})
        self.fetch_all = symbols_config.get('fetch_all', True)
        self.auto_filter = symbols_config.get('auto_filter', False)
        self.batch_processing = symbols_config.get('batch_processing', True)
        self.max_symbols = symbols_config.get('max_symbols', None)
        
        # A100 optimizations
        a100_config = CONFIG.get('a100_optimizations', {})
        self.a100_enabled = a100_config.get('enabled', True)
        self.batch_multiplier = a100_config.get('batch_size_multiplier', 10)
        
    
    async def fetch_all_symbols(self) -> Set[str]:
        """Fetch ALL available symbols from Polygon REST API with pagination"""
        logger.info("Starting symbol fetch from Polygon API")
        logger.log_data_flow("fetch", "symbols", data_type="api_request")
        
        # Get API config from YAML
        api_config = CONFIG.get('api', {}).get('polygon', {})
        rate_config = CONFIG.get('rate_limiting', {})
        
        base_url = api_config.get('base_url', 'https://api.polygon.io')
        min_request_interval = rate_config.get('min_request_interval', 0.01)
        
        try:
            url = f"{base_url}/v3/reference/tickers"
            params = {
                'apikey': self.api_key,
                'market': 'stocks',
                'active': 'true',
                'limit': 1000  # Maximum per request
            }
            
            all_symbols = set()
            next_url = None
            page_count = 0
            
            async with aiohttp.ClientSession() as session:
                while True:
                    if next_url:
                        request_url = next_url + f"&apikey={self.api_key}"
                        request_params = None
                    else:
                        request_url = url
                        request_params = params
                        
                    async with session.get(request_url, params=request_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            page_count += 1
                            
                            if 'results' in data:
                                for ticker in data['results']:
                                    symbol = ticker.get('ticker')
                                    if symbol:
                                        all_symbols.add(symbol)
                                        self.symbol_metadata[symbol] = ticker
                            
                            # Check for pagination
                            if 'next_url' in data and data['next_url']:
                                next_url = data['next_url']
                                logger.debug(f"Fetched page {page_count}, got {len(data.get('results', []))} symbols, continuing...")
                                logger.log_data_flow("fetch", "pagination", data_size=len(data.get('results', [])))
                                await asyncio.sleep(min_request_interval * 10)  # Use YAML config for rate limiting
                            else:
                                break
                        else:
                            logger.log_api_response(response.status)
                            logger.error(Exception(f"Failed to fetch symbols: {response.status}"),
                                                 {"status_code": response.status})
                            break
            
            self.all_symbols = all_symbols
            logger.log_data_flow("fetch", "completed", data_size=len(all_symbols))
            logger.info(f"Fetched {len(all_symbols)} total symbols from Polygon across {page_count} pages")
            return all_symbols
            
        except Exception as e:
            logger.error(e, {"operation": "fetch_symbols", "page_count": page_count})
            return set()
    
    def get_all_symbols_unfiltered(self) -> Set[str]:
        """Return all symbols without any filtering - optimized for A100"""
        self.active_symbols = self.all_symbols.copy()
        logger.info(f"Using ALL {len(self.active_symbols)} symbols without filtering (A100 optimized)")
        return self.active_symbols
    
    def get_all_symbols_list(self) -> List[str]:
        """Get all symbols as a list - optimized for batch processing"""
        return sorted(list(self.active_symbols))
    
    def get_symbol_batches(self, batch_size: int = 1000) -> List[List[str]]:
        """Get symbols in batches for optimized subscription - A100 can handle larger batches"""
        symbols_list = self.get_all_symbols_list()
        batches = []
        for i in range(0, len(symbols_list), batch_size):
            batch = symbols_list[i:i + batch_size]
            batches.append(batch)
        logger.info(f"Created {len(batches)} batches of up to {batch_size} symbols each")
        return batches


class PolygonWebSocketManager:
    """Pure WebSocket manager using YAML config with health monitoring and data validation"""
    
    
    def __init__(self, api_key: str, symbols: List[str], data_callback: Callable):
        logger.startup({
            "symbols_count": len(symbols),
            "has_api_key": bool(api_key),
            "websocket_url": "wss://socket.polygon.io/stocks"
        })
        
        logger.log_data_flow("initialization", "websocket_manager", data_size=len(symbols))
        
        self.api_key = api_key
        self.symbols = symbols
        self.data_callback = data_callback
        self.websocket = None
        self.symbol_manager = SymbolManager(api_key)
        self.health_monitor = ConnectionHealthMonitor()
        self.data_validator = DataValidator()
        self.is_connected = False
        
        # Load config
        api_config = CONFIG.get('api', {}).get('polygon', {})
        health_config = CONFIG.get('health_monitoring', {})
        rate_config = CONFIG.get('rate_limiting', {})
        
        self.websocket_url = api_config.get('websocket_url', "wss://socket.polygon.io/stocks")
        self.timeout = api_config.get('timeout', 15)
        self.max_retries = api_config.get('max_retries', 5)
        
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = health_config.get('max_reconnect_attempts', 10)
        self.reconnect_backoff_base = health_config.get('reconnect_backoff_base', 1.0)
        self.reconnect_backoff_max = health_config.get('reconnect_backoff_max', 60.0)
        
        self.rate_limiter = {
            "last_request": 0,
            "min_interval": rate_config.get('min_request_interval', 0.01)
        }
        
        # A100 optimizations
        a100_config = CONFIG.get('a100_optimizations', {})
        self.a100_enabled = a100_config.get('enabled', True)
        self.batch_size = rate_config.get('websocket_subscriptions_per_batch', 1000)
        
    async def initialize_symbols(self):
        """Initialize and fetch ALL available symbols - A100 optimized"""
        await self.symbol_manager.fetch_all_symbols()
        all_symbols = self.symbol_manager.get_all_symbols_unfiltered()
        
        # If no specific symbols provided, use ALL symbols (A100 can handle it)
        if not self.symbols:
            self.symbols = self.symbol_manager.get_all_symbols_list()
            logger.info(f"A100 OPTIMIZED: Auto-selected ALL {len(self.symbols)} symbols for processing")
        else:
            # Validate provided symbols exist (but don't filter them)
            valid_symbols = [s for s in self.symbols if s in all_symbols]
            invalid_symbols = [s for s in self.symbols if s not in all_symbols]
            
            if invalid_symbols:
                logger.warning(f"Invalid symbols (not found in Polygon): {invalid_symbols}")
            
            self.symbols = valid_symbols
            logger.info(f"Using {len(self.symbols)} provided symbols")
    
    
    async def connect(self):
        """Establish pure WebSocket connection to Polygon with health monitoring"""
        logger.connection("connecting", {
            "websocket_url": self.websocket_url,
            "reconnect_attempts": self.reconnect_attempts
        })
        
        logger.log_data_flow("connection", "websocket", data_type="attempt")
        
        try:
            # Rate limiting check
            await self._rate_limit_check()
            
            # Connect to Polygon WebSocket
            self.websocket = await websockets.connect(self.websocket_url)
            
            logger.connection("established", {"websocket_url": self.websocket_url})
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            logger.log_data_flow("authentication", "sent", data_type="auth_message")
            
            # Wait for auth response
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            
            if auth_data.get("status") == "auth_success":
                self.is_connected = True
                self.health_monitor.connection_status = "connected"
                self.health_monitor.reconnect_count = self.reconnect_attempts
                self.reconnect_attempts = 0
                
                logger.connection("authenticated", {
                    "auth_status": "success",
                    "reconnect_attempts_reset": True
                })
                logger.log_data_flow("authentication", "success", data_type="auth_response")
                
                # Subscribe to symbols
                await self._subscribe_to_symbols()
                
                # Start message handling
                asyncio.create_task(self._message_handler())
                asyncio.create_task(self._heartbeat_handler())
                
            else:
                auth_error = Exception(f"Authentication failed: {auth_data.get('status', 'unknown')}")
                logger.error(auth_error, {"auth_data": auth_data})
                await self._handle_reconnection()
            
        except Exception as e:
            logger.error(e, {"websocket_url": self.websocket_url})
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()
    
    async def _subscribe_to_symbols(self):
        """Subscribe to symbols using YAML config - A100 optimized batching"""
        if not self.symbols:
            logger.error(Exception("No symbols to subscribe to"), {})
            return
        
        # Get data processing config
        data_config = CONFIG.get('data_processing', {}).get('storage', {})
        enable_trades = data_config.get('enable_trades', True)
        enable_quotes = data_config.get('enable_quotes', True)
        
        # Build subscriptions based on config
        all_subscriptions = []
        if enable_trades:
            trade_subscriptions = [f"T.{symbol}" for symbol in self.symbols]
            all_subscriptions.extend(trade_subscriptions)
        if enable_quotes:
            quote_subscriptions = [f"Q.{symbol}" for symbol in self.symbols]
            all_subscriptions.extend(quote_subscriptions)
        
        if not all_subscriptions:
            logger.warning("No subscriptions enabled in config")
            return
        
        logger.info(f"YAML CONFIG: Subscribing to {len(all_subscriptions)} total subscriptions in batches of {self.batch_size}")
        logger.info(f"A100 OPTIMIZED: Trades={enable_trades}, Quotes={enable_quotes}")
        
        for i in range(0, len(all_subscriptions), self.batch_size):
            batch = all_subscriptions[i:i + self.batch_size]
            
            # Create subscription message
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(batch)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to batch {i//self.batch_size + 1}/{(len(all_subscriptions) + self.batch_size - 1)//self.batch_size}: {len(batch)} subscriptions")
            
            # Rate limiting using YAML config
            rate_config = CONFIG.get('rate_limiting', {})
            min_interval = rate_config.get('min_request_interval', 0.01)
            await asyncio.sleep(min_interval * 5)  # 5x the minimum interval for batch subscriptions
        
        logger.info(f"A100 OPTIMIZED: Successfully subscribed to {len(self.symbols)} symbols for trades and quotes")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages with validation"""
        try:
            async for message in self.websocket:
                self.health_monitor.update_data_received()
                
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if isinstance(data, list):
                        for item in data:
                            await self._process_message(item)
                    else:
                        await self._process_message(data)
                        
                except json.JSONDecodeError as e:
                    logger.error(e, {"operation": "parse_json_message"})
                except Exception as e:
                    logger.error(e, {"operation": "process_message"})
                    
        except websockets.exceptions.ConnectionClosed:
            logger.connection("closed", {})
            self.is_connected = False
            self.health_monitor.connection_status = "disconnected"
            await self._handle_reconnection()
        except Exception as e:
            logger.error(e, {"operation": "message_handler"})
            self.is_connected = False
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()
    
    
    async def _process_message(self, data: Dict):
        """Process individual message with data validation"""
        try:
            msg_type = data.get('ev')  # Event type
            symbol = data.get('sym', 'UNKNOWN')
            
            logger.log_data_flow("processing", "message",
                                     data_sample={"type": msg_type, "symbol": symbol})
            
            if msg_type == 'T':  # Trade
                if self.data_validator.validate_trade_data(data):
                    price = self.data_validator.sanitize_price(data.get('p'))
                    volume = self.data_validator.sanitize_volume(data.get('s'))
                    
                    if price is not None and volume is not None:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=data.get('t', 0) / 1000,  # Convert to seconds
                            price=price,
                            volume=volume,
                            data_type='trade'
                        )
                        
                        logger.log_data_flow("processing", "trade_data",
                                                 data_sample={"symbol": symbol, "price": price, "volume": volume})
                        
                        self.data_callback(market_data, 'trade')
                        
            elif msg_type == 'Q':  # Quote
                if self.data_validator.validate_quote_data(data):
                    bid_price = self.data_validator.sanitize_price(data.get('bp'))
                    ask_price = self.data_validator.sanitize_price(data.get('ap'))
                    
                    if bid_price is not None and ask_price is not None:
                        mid_price = (bid_price + ask_price) / 2
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=data.get('t', 0) / 1000,
                            price=mid_price,
                            volume=0,
                            bid=bid_price,
                            ask=ask_price,
                            bid_size=data.get('bs', 0),
                            ask_size=data.get('as', 0),
                            data_type='quote'
                        )
                        
                        logger.log_data_flow("processing", "quote_data",
                                                 data_sample={"symbol": symbol, "bid": bid_price, "ask": ask_price})
                        
                        self.data_callback(market_data, 'quote')
                        
            elif msg_type == 'status':
                logger.log_data_flow("processing", "status_message", data_sample=data)
                logger.info(f"Status message: {data}")
                
        except Exception as e:
            logger.error(e, {
                "message_type": data.get('ev', 'unknown'),
                "symbol": data.get('sym', 'unknown'),
                "operation": "process_message"
            })
    
    async def _heartbeat_handler(self):
        """Handle heartbeat to maintain connection health"""
        while self.is_connected:
            try:
                # Send ping to keep connection alive
                if self.websocket and not self.websocket.closed:
                    await self.websocket.ping()
                    self.health_monitor.update_heartbeat()
                    
                await asyncio.sleep(self.health_monitor.heartbeat_interval)
                
            except Exception as e:
                logger.error(e, {"operation": "heartbeat"})
                break
    
    async def _rate_limit_check(self):
        """Check rate limiting before making requests"""
        now = time.time()
        time_since_last = now - self.rate_limiter["last_request"]
        
        if time_since_last < self.rate_limiter["min_interval"]:
            sleep_time = self.rate_limiter["min_interval"] - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.rate_limiter["last_request"] = time.time()
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(Exception("Max reconnection attempts reached. Stopping."),
                                 {"max_attempts": self.max_reconnect_attempts})
            return
        
        self.reconnect_attempts += 1
        # Use YAML config for exponential backoff
        backoff_time = self.reconnect_backoff_base * (2 ** self.reconnect_attempts)
        wait_time = min(backoff_time, self.reconnect_backoff_max)
        
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time}s")
        await asyncio.sleep(wait_time)
        await self.connect()
    
    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.is_connected = False
            self.health_monitor.connection_status = "disconnected"
            logger.connection("disconnected", {})
    
    def get_health_status(self) -> Dict:
        """Get connection health status"""
        return self.health_monitor.get_status()
    
    def get_connection_stats(self) -> Dict:
        """Get comprehensive connection statistics"""
        health_status = self.get_health_status()
        return {
            "health": health_status,
            "symbols_subscribed": len(self.symbols),
            "reconnect_attempts": self.reconnect_attempts,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "is_connected": self.is_connected,
            "websocket_url": self.websocket_url,
            "batch_size_optimized": self.batch_size,  # From YAML config
            "a100_optimized": True
        }


class RealTimeDataFeed:
    """Main class for managing real-time data feed"""
    
    
    def __init__(self, api_key: str, symbols: Optional[List[str]] = None):
        logger.startup({
            "symbols_provided": len(symbols) if symbols else 0,
            "has_api_key": bool(api_key)
        })
        
        logger.log_data_flow("initialization", "realtime_feed",
                                 data_size=len(symbols) if symbols else 0)
        
        self.api_key = api_key
        self.symbols = symbols or []
        self.ws_manager = None
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.latest_data: Dict[str, MarketData] = {}
        
        logger.info(f"Real-Time Data Feed initialized with {len(self.symbols)} symbols")
        
    async def start(self):
        """Start the real-time data feed"""
        self.ws_manager = PolygonWebSocketManager(
            api_key=self.api_key,
            symbols=self.symbols,
            data_callback=self._process_market_data
        )
        
        # Initialize symbols (fetch all if none provided)
        await self.ws_manager.initialize_symbols()
        
        # Connect and start streaming
        await self.ws_manager.connect()
        
        logger.info("Real-time data feed started")
    
    
    def _process_market_data(self, data: MarketData, data_type: str):
        """Process incoming market data"""
        symbol = data.symbol
        
        logger.log_data_flow("processing", "market_data",
                                 data_sample={"symbol": symbol, "type": data_type, "price": data.price})
        
        # Update latest data
        self.latest_data[symbol] = data
        
        # Add to buffer
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Keep only last N data points per symbol (from YAML config)
        data_config = CONFIG.get('data_processing', {}).get('storage', {})
        buffer_size = data_config.get('buffer_size', 1000)
        
        buffer_size_before = len(self.data_buffer[symbol])
        if len(self.data_buffer[symbol]) > buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol][-buffer_size:]
            
        if buffer_size_before > buffer_size:
            logger.log_data_flow("buffering", "trimmed",
                                     data_size=len(self.data_buffer[symbol]))
        
        # Log periodic updates
        if int(time.time()) % 10 == 0:  # Every 10 seconds
            logger.data_processing(symbol, data_type, 1)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for a symbol"""
        return self.latest_data.get(symbol)
    
    def get_symbol_buffer(self, symbol: str) -> List[MarketData]:
        """Get data buffer for a symbol"""
        return self.data_buffer.get(symbol, [])
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols being tracked"""
        return list(self.latest_data.keys())
    
    async def stop(self):
        """Stop the data feed"""
        if self.ws_manager:
            await self.ws_manager.disconnect()
        logger.info("Real-time data feed stopped")


# Example usage
async def main():
    """Example usage of the real-time data feed"""
    
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable required")
    
    # Create data feed (will auto-fetch all symbols if none provided)
    data_feed = RealTimeDataFeed(api_key=api_key)
    
    try:
        # Start the feed
        await data_feed.start()
        
        # Run for a while
        await asyncio.sleep(60)  # Run for 1 minute
        
        # Print some stats
        symbols = data_feed.get_all_symbols()
        print(f"Tracking {len(symbols)} symbols")
        
        for symbol in symbols[:10]:  # Show first 10
            latest = data_feed.get_latest_data(symbol)
            if latest:
                print(f"{symbol}: ${latest.price:.2f} at {datetime.fromtimestamp(latest.timestamp)}")
    
    finally:
        await data_feed.stop()


if __name__ == "__main__":
    logger.info("Starting Polygon WebSocket example")
    asyncio.run(main())