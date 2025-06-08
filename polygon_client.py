#!/usr/bin/env python3
"""Ultra-optimized Polygon client for HFT with zero-copy operations and minimal latency."""

# ULTRA-FAST IMPORTS - Pre-loaded for maximum speed
import asyncio
import json
import os
import time
import aiohttp
import websockets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional, Any

# Pre-import datetime for performance
import datetime as _dt
_strftime = _dt.datetime.now().strftime

# HARDCODED ULTRA-FAST SETTINGS FOR MAXIMUM HFT SPEED
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_WEBSOCKET_URL = "wss://socket.polygon.io/stocks"
API_TIMEOUT = 15
MAX_RETRIES = 5
REST_BATCH_SIZE = 50
WEBSOCKET_SUBSCRIPTIONS_PER_BATCH = 50000  # Single batch for ultra-low latency
MIN_REQUEST_INTERVAL = 0.01
HEARTBEAT_INTERVAL = 30
MAX_RECONNECT_ATTEMPTS = 10
DATA_TIMEOUT_SECONDS = 60
RECONNECT_BACKOFF_BASE = 1.0
RECONNECT_BACKOFF_MAX = 60.0
VALIDATION_ENABLED = True
PRICE_MIN = 0.01
PRICE_MAX = 1000000
VOLUME_MIN = 0
VOLUME_MAX = 1000000000
FETCH_ALL_SYMBOLS = True
AUTO_FILTER_SYMBOLS = False
BATCH_PROCESSING_ENABLED = True
MAX_SYMBOLS = 15000
BUFFER_SIZE = 2000
ENABLE_QUOTES = True                    # Q.{symbol} - Essential for bid/ask spreads
ENABLE_SECOND_AGGREGATES = True         # AS.{symbol} - Essential for intraday data
ENABLE_TRADES = False                   # T.{symbol} - High volume, disabled for production
ENABLE_MINUTE_AGGREGATES = False        # A.{symbol} - Redundant with second aggregates
ENABLE_MARKET_SNAPSHOTS = True          # Essential for current market state
ENABLE_DAILY_BARS = True               # Essential for daily OHLCV data
ENABLE_GROUPED_DAILY_BARS = True       # Essential for full market snapshot
ENABLE_MARKET_MOVERS = True            # Essential for market breadth analysis
A100_OPTIMIZATIONS_ENABLED = True
BATCH_SIZE_MULTIPLIER = 20  # Increased for better batching
METRICS_ENABLED = True
ULTRA_LOW_LATENCY_MODE = True
ZERO_COPY_MANDATORY = True
MEMORY_POOL_SIZE = 15000
SUBSCRIPTION_BATCH_SIZE = 2000
CPU_AFFINITY_ENABLED = True
MEMORY_PREFETCH_ENABLED = True
GPU_AVAILABLE = False

# Pre-compiled timestamp format for ultra-fast logging
_LOG_FORMAT = "%Y-%m-%d %H:%M:%S"

class FastLogger:
    """Ultra-fast logger with pre-compiled formats and minimal overhead."""
    
    __slots__ = ('name', 'colors', 'log_dir', 'log_file')
    
    def __init__(self, name="polygon_client"):
        self.name = name
        # ANSI color codes for terminal output
        self.colors = {
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'WHITE': '\033[97m',
            'RESET': '\033[0m'
        }
        
        # Create logs directory if it doesn't exist
        self.log_dir = "/home/ubuntu/reg3n-1/logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_file = os.path.join(self.log_dir, "backtesting.log")
    
    def _log(self, level, message, color_code, extra=None):
        """Unified logging method with colored output and file logging."""
        timestamp = _strftime(_LOG_FORMAT)
        formatted_message = f"[{timestamp}] - {color_code}{level}{self.colors['RESET']} - [{self.name}]: {message}"
        
        # Print to console with colors
        print(formatted_message)
        
        # Write to file without colors
        file_message = f"[{timestamp}] - {level} - [{self.name}]: {message}"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(file_message + '\n')
                if extra:
                    f.write(f"    Extra: {extra}\n")
        except Exception:
            pass  # Fail silently to avoid disrupting performance
        
        if extra:
            print(f"    Extra: {extra}")
    
    def info(self, message, extra=None):
        self._log("INFO", message, self.colors['WHITE'], extra)
    
    def debug(self, message, extra=None):
        self._log("DEBUG", message, self.colors['BLUE'], extra)
    
    def warning(self, message, extra=None):
        self._log("WARNING", message, self.colors['YELLOW'], extra)
    
    def error(self, message, extra=None):
        self._log("ERROR", message, self.colors['RED'], extra)
    
    def critical(self, message, extra=None):
        self._log("CRITICAL", message, self.colors['RED'], extra)

class MarketData:
    """Ultra-fast market data container with __slots__ for memory efficiency."""
    
    __slots__ = ('symbol', 'timestamp', 'price', 'volume', 'bid', 'ask', 'bid_size', 'ask_size', 'data_type')
    
    def __init__(self, symbol, timestamp, price, volume, bid=None, ask=None, bid_size=None, ask_size=None, data_type="trade"):
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price
        self.volume = volume
        self.bid = bid
        self.ask = ask
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.data_type = data_type

class AggregateData:
    """Ultra-fast aggregate data container with __slots__ for memory efficiency."""
    
    __slots__ = ('symbol', 'timestamp', 'open', 'high', 'low', 'close', 'price', 'volume', 'vwap', 'data_type')
    
    def __init__(self, symbol, timestamp, open_price, high_price, low_price, close_price, volume, vwap=None, data_type="aggregate"):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.price = close_price  # For compatibility
        self.volume = volume
        self.vwap = vwap
        self.data_type = data_type

class ConnectionHealthMonitor:
    """Ultra-fast connection health monitor with minimal overhead."""
    
    __slots__ = ('logger', 'heartbeat_interval', 'max_reconnect_attempts', 'data_timeout_seconds',
                 'last_heartbeat', 'last_data_received', 'connection_status', 'reconnect_count', 'total_messages_received')
    
    def __init__(self):
        self.logger = FastLogger(name="ConnectionHealthMonitor")
        self.heartbeat_interval = HEARTBEAT_INTERVAL
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self.data_timeout_seconds = DATA_TIMEOUT_SECONDS
        self.last_heartbeat = time.time()
        self.last_data_received = time.time()
        self.connection_status = "disconnected"
        self.reconnect_count = 0
        self.total_messages_received = 0
        self.logger.info("ConnectionHealthMonitor initialized.")

    def update_heartbeat(self):
        self.last_heartbeat = time.time()

    def update_data_received(self):
        self.last_data_received = time.time()
        self.total_messages_received += 1

    def is_healthy(self):
        now = time.time()
        heartbeat_ok = (now - self.last_heartbeat) < (self.heartbeat_interval * 2)
        data_flow_ok = (now - self.last_data_received) < self.data_timeout_seconds
        is_healthy = heartbeat_ok and data_flow_ok and self.connection_status == "connected"
        
        if not is_healthy:
            self.logger.warning(
                f"Connection unhealthy. Status: {self.connection_status}, "
                f"Heartbeat: {now - self.last_heartbeat:.2f}s ago, "
                f"Data: {now - self.last_data_received:.2f}s ago"
            )
        
        return is_healthy

    def get_status(self):
        now = time.time()
        return {
            "status": self.connection_status,
            "last_heartbeat_seconds_ago": now - self.last_heartbeat,
            "last_data_seconds_ago": now - self.last_data_received,
            "reconnect_count": self.reconnect_count,
            "total_messages": self.total_messages_received,
            "is_healthy": self.is_healthy(),
        }

class FastDataValidator:
    """Ultra-fast data validator with minimal validation overhead."""
    
    __slots__ = ('enabled', 'min_price', 'max_price', 'min_volume', 'max_volume')
    
    def __init__(self):
        self.enabled = VALIDATION_ENABLED
        self.min_price = PRICE_MIN
        self.max_price = PRICE_MAX
        self.min_volume = VOLUME_MIN
        self.max_volume = VOLUME_MAX

    def validate_quote_data(self, data):
        """Fast quote validation with early returns."""
        if not self.enabled:
            return True
        return all(field in data for field in ("sym", "bp", "ap", "t"))

    def sanitize_price(self, price):
        """Ultra-fast price sanitization."""
        if not self.enabled:
            try:
                return float(price)
            except (ValueError, TypeError):
                return None
        
        try:
            price_float = float(price)
            return price_float if self.min_price <= price_float <= self.max_price else None
        except (ValueError, TypeError):
            return None

    def sanitize_volume(self, volume):
        """Ultra-fast volume sanitization."""
        if not self.enabled:
            try:
                return int(volume)
            except (ValueError, TypeError):
                return None
        
        try:
            volume_int = int(volume)
            return volume_int if self.min_volume <= volume_int <= self.max_volume else None
        except (ValueError, TypeError):
            return None

class SymbolManager:
    """Ultra-fast symbol manager with optimized batch processing."""
    
    __slots__ = ('logger', 'api_key', 'all_symbols', 'active_symbols', 'symbol_metadata',
                 'fetch_all', 'auto_filter', 'batch_processing', 'max_symbols', 'a100_enabled', 'batch_multiplier', '_validation_engine')
    
    def __init__(self, api_key):
        self.logger = FastLogger(name="SymbolManager")
        self.api_key = api_key
        self.all_symbols = set()
        self.active_symbols = set()
        self.symbol_metadata = {}
        self.fetch_all = FETCH_ALL_SYMBOLS
        self.auto_filter = AUTO_FILTER_SYMBOLS
        self.batch_processing = BATCH_PROCESSING_ENABLED
        self.max_symbols = MAX_SYMBOLS
        self.a100_enabled = A100_OPTIMIZATIONS_ENABLED
        self.batch_multiplier = BATCH_SIZE_MULTIPLIER
        # Initialize TensorRT validation engine attribute first
        self._validation_engine = None
        self.logger.info("SymbolManager initialized.")

    async def fetch_all_symbols(self):
        """TensorRT-accelerated symbol fetching using full market snapshots with ultra-fast batch validation."""
        try:
            self.logger.info("Fetching symbols using TensorRT-accelerated full market snapshots...")
            
            # Ensure _validation_engine attribute exists before any processing
            if not hasattr(self, '_validation_engine'):
                self._validation_engine = None
                self.logger.debug("Initialized _validation_engine attribute")
            
            # Use direct API call instead of creating temporary client
            snapshot_data = await self._get_full_market_snapshot_direct()
            
            if snapshot_data and len(snapshot_data) > 0:
                self.logger.info(f"🚀 Starting TensorRT batch validation of {len(snapshot_data)} symbols...")
                
                # Use TensorRT-accelerated batch validation for ultra-fast processing
                valid_snapshot_data = self._batch_validate_symbols_tensorrt(snapshot_data)
                
                discovered_symbols = set()
                valid_symbols = len(valid_snapshot_data)
                
                # Process validated symbols
                for ticker, data in valid_snapshot_data.items():
                    discovered_symbols.add(ticker)
                    self.symbol_metadata[ticker] = {
                        'price': data.get('day', {}).get('c', 0),
                        'volume': data.get('day', {}).get('v', 0),
                        'market_cap': data.get('market_cap', 0),
                        'last_updated': time.time(),
                        'source': 'tensorrt_snapshot'
                    }
                
                if len(discovered_symbols) > 1000:  # Ensure we have a reasonable number of symbols
                    self.all_symbols = discovered_symbols
                    self.logger.info(f"✅ TensorRT-accelerated symbol discovery: {len(discovered_symbols)} symbols from snapshots ({valid_symbols} valid)")
                    return discovered_symbols
                else:
                    self.logger.warning(f"TensorRT validation returned insufficient symbols ({len(discovered_symbols)}), falling back to pagination...")
            else:
                self.logger.warning("Snapshot method returned no data, falling back to pagination...")
            
            # Fallback to pagination if snapshots fail or return insufficient data
            return await self._fetch_symbols_pagination_fallback()

        except Exception as e:
            self.logger.error(f"Critical error in TensorRT symbol fetching: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            self.logger.info("Attempting pagination fallback...")
            return await self._fetch_symbols_pagination_fallback()

    def _is_valid_trading_symbol(self, snapshot_data):
        """TensorRT-accelerated symbol validation for ultra-fast batch processing."""
        try:
            if not snapshot_data:
                return False
                
            day_data = snapshot_data.get('day', {})
            if not day_data:
                # Try alternative data structure
                price = snapshot_data.get('value', 0) or snapshot_data.get('c', 0)
                volume = snapshot_data.get('volume', 0) or snapshot_data.get('v', 0)
            else:
                price = day_data.get('c', 0)
                volume = day_data.get('v', 0)
            
            # Convert to float/int safely
            try:
                price = float(price) if price else 0.0
                volume = int(volume) if volume else 0
            except (ValueError, TypeError):
                return False
            
            # Validation criteria: reasonable price range and minimum volume
            is_valid = 1.0 <= price <= 1000.0 and volume >= 100000
            
            return is_valid
            
        except Exception as e:
            # Log validation errors for debugging but don't fail the entire process
            self.logger.debug(f"Symbol validation error: {e}")
            return False

    def _batch_validate_symbols_tensorrt(self, snapshot_data):
        """TensorRT-accelerated batch symbol validation for ultra-fast processing."""
        try:
            import numpy as np
            
            # Ensure validation engine attribute exists at the very beginning
            if not hasattr(self, '_validation_engine') or self._validation_engine is None:
                self._validation_engine = None
                self.logger.debug("Ensured _validation_engine attribute exists")
            
            # Extract all price and volume data for batch processing
            symbols = list(snapshot_data.keys())
            prices = []
            volumes = []
            
            for symbol in symbols:
                data = snapshot_data[symbol]
                day_data = data.get('day', {})
                
                if day_data:
                    price = day_data.get('c', 0)
                    volume = day_data.get('v', 0)
                else:
                    price = data.get('value', 0) or data.get('c', 0)
                    volume = data.get('volume', 0) or data.get('v', 0)
                
                try:
                    prices.append(float(price) if price else 0.0)
                    volumes.append(int(volume) if volume else 0)
                except (ValueError, TypeError):
                    prices.append(0.0)
                    volumes.append(0)
            
            # Convert to numpy arrays for TensorRT processing
            price_array = np.array(prices, dtype=np.float32)
            volume_array = np.array(volumes, dtype=np.float32)
            
            # TensorRT-accelerated batch validation
            try:
                # Ensure validation engine attribute exists and initialize if needed
                if not hasattr(self, '_validation_engine') or self._validation_engine is None:
                    self._validation_engine = self._create_validation_tensorrt_engine()
                
                # Check if TensorRT engine is available
                if hasattr(self, '_validation_engine') and self._validation_engine is not None:
                    # Use TensorRT for ultra-fast batch processing
                    valid_mask = self._tensorrt_batch_validate(price_array, volume_array)
                    
                    # Return valid symbols
                    valid_symbols = {}
                    for i, symbol in enumerate(symbols):
                        if valid_mask[i]:
                            valid_symbols[symbol] = snapshot_data[symbol]
                    
                    self.logger.info(f"🚀 TensorRT batch validation: {len(valid_symbols)}/{len(symbols)} symbols passed filter")
                    return valid_symbols
                else:
                    # TensorRT not available, fall back to CPU
                    raise Exception("TensorRT validation engine not available")
                
            except Exception as e:
                self.logger.warning(f"TensorRT batch validation failed: {e}, falling back to CPU")
                # Fallback to CPU-based batch processing
                return self._cpu_batch_validate(snapshot_data, price_array, volume_array, symbols)
                
        except Exception as e:
            self.logger.error(f"Batch validation error: {e}")
            # Fallback to individual validation
            return {symbol: data for symbol, data in snapshot_data.items()
                   if self._is_valid_trading_symbol(data)}

    def _tensorrt_batch_validate(self, prices, volumes):
        """TensorRT-accelerated batch validation using GPU with separate inputs."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            import numpy as np
            
            # Ensure validation engine is initialized
            if not hasattr(self, '_validation_engine') or self._validation_engine is None:
                self._validation_engine = None
            
            # Create TensorRT validation engine if not exists
            if self._validation_engine is None:
                self._validation_engine = self._create_validation_tensorrt_engine()
            
            if self._validation_engine is None:
                self.logger.warning("Failed to build TensorRT validation engine")
                raise Exception("TensorRT engine not available")
            
            # Prepare input data for fixed batch size (16384)
            engine_batch_size = 16384
            actual_batch_size = len(prices)
            
            if actual_batch_size > engine_batch_size:
                # Process in chunks if data is larger than engine batch size
                results = []
                for i in range(0, actual_batch_size, engine_batch_size):
                    chunk_prices = prices[i:i + engine_batch_size]
                    chunk_volumes = volumes[i:i + engine_batch_size]
                    chunk_result = self._process_tensorrt_chunk(chunk_prices, chunk_volumes, engine_batch_size)
                    results.extend(chunk_result[:len(chunk_prices)])
                return np.array(results, dtype=bool)
            else:
                # Pad data to match engine batch size
                padded_prices = np.zeros(engine_batch_size, dtype=np.float32)
                padded_volumes = np.zeros(engine_batch_size, dtype=np.float32)
                padded_prices[:actual_batch_size] = prices
                padded_volumes[:actual_batch_size] = volumes
                
                result = self._process_tensorrt_chunk(padded_prices, padded_volumes, engine_batch_size)
                return result[:actual_batch_size]
            
        except Exception as e:
            self.logger.debug(f"TensorRT validation error: {e}")
            raise

    def _process_tensorrt_chunk(self, prices, volumes, batch_size):
        """Process a single chunk through TensorRT engine."""
        try:
            import pycuda.driver as cuda
            import numpy as np
            
            # Ensure validation engine exists
            if not hasattr(self, '_validation_engine') or self._validation_engine is None:
                self._validation_engine = None
            
            if self._validation_engine is None:
                self._validation_engine = self._create_validation_tensorrt_engine()
            
            if self._validation_engine is None:
                raise Exception("TensorRT validation engine not available")
            
            # Allocate GPU memory for separate inputs
            d_prices = cuda.mem_alloc(prices.nbytes)
            d_volumes = cuda.mem_alloc(volumes.nbytes)
            d_output = cuda.mem_alloc(batch_size * np.dtype(np.bool_).itemsize)
            
            # Copy inputs to GPU
            cuda.memcpy_htod(d_prices, prices)
            cuda.memcpy_htod(d_volumes, volumes)
            
            # Create execution context
            context = self._validation_engine.create_execution_context()
            
            # Run inference with separate inputs
            context.execute_v2([int(d_prices), int(d_volumes), int(d_output)])
            
            # Copy result back to CPU
            output = np.empty(batch_size, dtype=np.bool_)
            cuda.memcpy_dtoh(output, d_output)
            
            # Clean up GPU memory
            d_prices.free()
            d_volumes.free()
            d_output.free()
            
            return output
            
        except Exception as e:
            self.logger.debug(f"TensorRT chunk processing error: {e}")
            raise

    def _create_validation_tensorrt_engine(self):
        """Create simplified TensorRT engine for symbol validation using fixed batch size."""
        try:
            import tensorrt as trt
            import numpy as np
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Use fixed batch size to avoid dynamic shape issues
            batch_size = 16384  # Fixed batch size for better optimization
            
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Define separate inputs for price and volume to avoid slice operations
            price_input = network.add_input("price", trt.float32, (batch_size,))
            volume_input = network.add_input("volume", trt.float32, (batch_size,))
            
            # Create constants for validation thresholds
            price_min_const = network.add_constant((1,), np.array([1.0], dtype=np.float32))
            price_max_const = network.add_constant((1,), np.array([1000.0], dtype=np.float32))
            volume_min_const = network.add_constant((1,), np.array([100000.0], dtype=np.float32))
            
            # Price validation: 1.0 <= price <= 1000.0
            price_ge_min = network.add_elementwise(
                price_input,
                price_min_const.get_output(0),
                trt.ElementWiseOperation.GREATER
            )
            price_le_max = network.add_elementwise(
                price_input,
                price_max_const.get_output(0),
                trt.ElementWiseOperation.LESS
            )
            
            # Volume validation: volume >= 100000
            volume_ge_min = network.add_elementwise(
                volume_input,
                volume_min_const.get_output(0),
                trt.ElementWiseOperation.GREATER
            )
            
            # Combine all conditions with AND operations
            price_valid = network.add_elementwise(
                price_ge_min.get_output(0),
                price_le_max.get_output(0),
                trt.ElementWiseOperation.AND
            )
            all_valid = network.add_elementwise(
                price_valid.get_output(0),
                volume_ge_min.get_output(0),
                trt.ElementWiseOperation.AND
            )
            
            # Mark output
            all_valid.get_output(0).name = "output"
            network.mark_output(all_valid.get_output(0))
            
            # Build engine with FP16 instead of INT8 to avoid calibration issues
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
            config.set_flag(trt.BuilderFlag.FP16)  # Use FP16 instead of INT8
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            
            # Create optimization profile for fixed batch size
            profile = builder.create_optimization_profile()
            profile.set_shape("price", (batch_size,), (batch_size,), (batch_size,))
            profile.set_shape("volume", (batch_size,), (batch_size,), (batch_size,))
            config.add_optimization_profile(profile)
            
            # Build and serialize engine
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                self.logger.warning("Failed to build TensorRT validation engine")
                return None
            
            # Deserialize engine
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            validation_engine = runtime.deserialize_cuda_engine(engine)
            
            if validation_engine is None:
                self.logger.warning("Failed to deserialize TensorRT validation engine")
                return None
            
            self.logger.info(f"🚀 TensorRT FP16 symbol validation engine created (batch_size: {batch_size})")
            return validation_engine
            
        except ImportError as e:
            self.logger.warning(f"TensorRT or PyCUDA not available: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to create TensorRT validation engine: {e}")
            return None

    def _cpu_batch_validate(self, snapshot_data, price_array, volume_array, symbols):
        """CPU-based batch validation fallback."""
        try:
            import numpy as np
            
            # Vectorized validation using NumPy
            price_valid = (price_array >= 1.0) & (price_array <= 1000.0)
            volume_valid = volume_array >= 100000
            valid_mask = price_valid & volume_valid
            
            # Return valid symbols
            valid_symbols = {}
            for i, symbol in enumerate(symbols):
                if valid_mask[i]:
                    valid_symbols[symbol] = snapshot_data[symbol]
            
            self.logger.info(f"⚡ CPU batch validation: {len(valid_symbols)}/{len(symbols)} symbols passed filter")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"CPU batch validation error: {e}")
            # Final fallback to individual validation
            return {symbol: data for symbol, data in snapshot_data.items()
                   if self._is_valid_trading_symbol(data)}

    async def _get_full_market_snapshot_direct(self):
        """Direct API call for full market snapshot using correct response format."""
        try:
            self.logger.info("Fetching full market snapshot from Polygon API...")
            
            # Use the correct snapshot endpoint with proper response handling
            url = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {
                "apikey": self.api_key,
                "include_otc": "false"  # Exclude OTC securities for cleaner data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Use correct response format: 'tickers' array, not 'results'
                        tickers = data.get("tickers", [])
                        count = data.get("count", 0)
                        status = data.get("status", "")
                        
                        self.logger.info(f"Snapshot API response: status={status}, count={count}, tickers={len(tickers)}")
                        
                        if tickers and len(tickers) > 0:
                            snapshot_data = {}
                            valid_tickers = 0
                            
                            for item in tickers:
                                ticker = item.get("ticker")
                                if ticker:
                                    snapshot_data[ticker] = item
                                    valid_tickers += 1
                            
                            self.logger.info(f"✅ Successfully retrieved {len(snapshot_data)} symbols from full market snapshot")
                            return snapshot_data
                        else:
                            self.logger.warning(f"Snapshot API returned no tickers. Status: {status}, Count: {count}")
                            
                            # Log the full response for debugging
                            self.logger.debug(f"Full response: {data}")
                            
                    else:
                        self.logger.warning(f"Snapshot API returned status {response.status}")
                        response_text = await response.text()
                        self.logger.debug(f"Response text: {response_text[:500]}")
            
            return None
                    
        except Exception as e:
            self.logger.error(f"Snapshot API critical error: {e}")
            return None

    async def _fetch_symbols_pagination_fallback(self):
        """Robust fallback pagination method with comprehensive error handling."""
        self.logger.info("Starting pagination fallback for symbol fetching...")
        
        try:
            url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
            params = {
                "apikey": self.api_key,
                "market": "stocks",
                "active": "true",
                "limit": 1000,
            }

            all_symbols = set()
            next_url = None
            page_count = 0
            max_pages = 50  # Limit to prevent infinite loops

            async with aiohttp.ClientSession() as session:
                while page_count < max_pages:
                    try:
                        if next_url:
                            request_url = next_url + f"&apikey={self.api_key}"
                            request_params = None
                        else:
                            request_url = url
                            request_params = params

                        async with session.get(request_url, params=request_params,
                                             timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as response:
                            if response.status == 200:
                                data = await response.json()
                                page_count += 1

                                if "results" in data and data["results"]:
                                    batch_symbols = 0
                                    for ticker in data["results"]:
                                        symbol = ticker.get("ticker")
                                        if symbol:
                                            all_symbols.add(symbol)
                                            batch_symbols += 1
                                            # Store metadata with source tracking
                                            self.symbol_metadata[symbol] = {
                                                **ticker,
                                                'source': 'pagination',
                                                'last_updated': time.time()
                                            }
                                    
                                    self.logger.debug(f"Pagination page {page_count}: {batch_symbols} symbols, total: {len(all_symbols)}")

                                    if "next_url" in data and data["next_url"]:
                                        next_url = data["next_url"]
                                        await asyncio.sleep(MIN_REQUEST_INTERVAL * 10)  # Rate limiting
                                    else:
                                        self.logger.info(f"Pagination complete: {len(all_symbols)} symbols from {page_count} pages")
                                        break
                                else:
                                    self.logger.warning(f"Empty results on page {page_count}, stopping pagination")
                                    break
                            elif response.status == 429:
                                self.logger.warning("Rate limit hit during pagination, waiting...")
                                await asyncio.sleep(60)  # Wait 1 minute for rate limit reset
                                continue
                            else:
                                self.logger.error(f"Pagination failed with status {response.status} on page {page_count}")
                                break
                                
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout on pagination page {page_count}, retrying...")
                        await asyncio.sleep(5)
                        continue
                    except Exception as e:
                        self.logger.error(f"Error on pagination page {page_count}: {e}")
                        break

            if len(all_symbols) > 0:
                self.all_symbols = all_symbols
                self.logger.info(f"✅ Pagination fallback successful: {len(all_symbols)} symbols retrieved")
                return all_symbols
            else:
                self.logger.error("Pagination fallback failed to retrieve any symbols")
                return set()

        except Exception as e:
            self.logger.error(f"Critical error in pagination fallback: {e}")
            # Return a minimal set of common symbols as last resort
            fallback_symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "IWM"}
            self.logger.warning(f"Using emergency fallback symbols: {fallback_symbols}")
            self.all_symbols = fallback_symbols
            return fallback_symbols

    def get_all_symbols_unfiltered(self):
        """Return all symbols without filtering - optimized for A100."""
        self.active_symbols = self.all_symbols.copy()
        return self.active_symbols

    def get_all_symbols_list(self):
        """Get all symbols as sorted list for batch processing."""
        return sorted(list(self.active_symbols))

    def get_symbol_batches(self, batch_size=1000):
        """Get symbols in optimized batches for A100."""
        symbols_list = self.get_all_symbols_list()
        return [symbols_list[i:i + batch_size] for i in range(0, len(symbols_list), batch_size)]
    
    def cleanup(self):
        """Cleanup TensorRT resources to prevent memory leaks."""
        try:
            if hasattr(self, '_validation_engine') and self._validation_engine is not None:
                # Clean up TensorRT engine
                del self._validation_engine
                self._validation_engine = None
                self.logger.debug("TensorRT validation engine cleaned up")
        except Exception as e:
            self.logger.debug(f"Error during TensorRT cleanup: {e}")

class PolygonClient:
    """Ultra-optimized Polygon client for HFT with zero-copy operations."""

    def __init__(self, api_key=None, symbols=None, data_callback=None, enable_filtering=True, memory_pools=None,
                 portfolio_manager=None, ml_bridge=None):
        self.logger = FastLogger(name="PolygonClient")
        self.api_key = api_key or "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57"

        if not self.api_key:
            msg = "Polygon API key must be provided directly"
            self.logger.critical(msg)
            raise ValueError(msg)
        self.logger.info("Polygon API key loaded.")

        # Unified architecture integration
        self.portfolio_manager = portfolio_manager
        self.ml_bridge = ml_bridge

        # Initialize zero-copy memory pools if not provided
        if memory_pools is None and ZERO_COPY_MANDATORY:
            memory_pools = self._create_zero_copy_memory_pools()
            self.logger.info("Created mandatory zero-copy memory pools")

        # REST API Configuration
        self.base_url = POLYGON_BASE_URL
        self.timeout = API_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.batch_size = REST_BATCH_SIZE
        self.min_request_interval = MIN_REQUEST_INTERVAL

        # Initialize REST session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # WebSocket Configuration
        self.websocket_url = POLYGON_WEBSOCKET_URL
        self.symbols = symbols or []
        self.data_callback = data_callback
        self.websocket = None
        self.is_connected = False

        # Initialize components
        self.symbol_manager = SymbolManager(self.api_key)
        self.health_monitor = ConnectionHealthMonitor()
        self.data_validator = FastDataValidator()

        # Unified data storage with zero-copy optimization
        self.latest_data = {}
        self.filtered_data = {}
        self.latest_aggregates = {}
        self.enable_filtering = enable_filtering
        
        # Zero-copy memory pools
        self.memory_pools = memory_pools or {}
        self.zero_copy_enabled = bool(memory_pools)

        # WebSocket reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self.reconnect_backoff_base = RECONNECT_BACKOFF_BASE
        self.reconnect_backoff_max = RECONNECT_BACKOFF_MAX

        # Rate limiting
        self.rate_limiter = {"last_request": 0}

        # Unified statistics tracking
        self.stats = {
            "requests": {"total": 0, "successful": 0, "failed": 0},
            "performance": {"total_response_time_ms": 0.0},
            "config_loaded": True,
        }

        # Performance tracking
        self.filter_stats = {
            'total_processed': 0,
            'filtered_passed': 0,
            'filter_processing_times': [],
            'zero_copy_enabled': self.zero_copy_enabled
        }

        # A100 optimizations
        self.a100_enabled = A100_OPTIMIZATIONS_ENABLED
        self.websocket_batch_size = WEBSOCKET_SUBSCRIPTIONS_PER_BATCH

        # Initialize adaptive data filter if filtering is enabled
        if self.enable_filtering:
            try:
                from adaptive_data_filter import AdaptiveDataFilter
                self.adaptive_filter = AdaptiveDataFilter(
                    memory_pools=self.memory_pools,
                    polygon_client=self
                )
                self.logger.info("Adaptive data filter initialized with TensorRT acceleration")
                self.logger.info("Production filter system integrated for dataset creation compatibility")
            except ImportError as e:
                self.logger.warning(f"Adaptive filter not available: {e}")
                self.adaptive_filter = None
        else:
            self.adaptive_filter = None

        self.logger.info(f"Ultra-optimized PolygonClient initialized with filtering {'enabled' if self.enable_filtering else 'disabled'}, zero-copy {'enabled' if self.zero_copy_enabled else 'disabled'}")

    # =============================================================================
    # OPTIMIZED REST API METHODS
    # =============================================================================

    def _make_request(self, endpoint, params=None):
        """Ultra-fast HTTP request with minimal overhead."""
        if not self.api_key:
            return None

        url = f"{self.base_url}{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        start_time = time.time()
        self.stats["requests"]["total"] += 1

        try:
            response = self.session.get(url, params=request_params, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            self.stats["performance"]["total_response_time_ms"] += response_time

            if response.status_code == 200:
                self.stats["requests"]["successful"] += 1
                return response.json()
            else:
                self.stats["requests"]["failed"] += 1
                self.logger.warning(f"Request to {endpoint} failed with status {response.status_code}")
                return None

        except Exception as e:
            self.stats["requests"]["failed"] += 1
            self.logger.error(f"Request to {endpoint} failed: {e}")
            return None

    async def _rate_limit_check(self):
        """Ultra-fast rate limiting check."""
        now = time.time()
        time_since_last = now - self.rate_limiter["last_request"]

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.rate_limiter["last_request"] = time.time()

    def get_market_status(self):
        """Get current market status."""
        try:
            return self._make_request("/v1/marketstatus/now")
        except Exception as e:
            self.logger.error(f"Error getting market status: {e}")
            return None

    def get_single_snapshot(self, ticker):
        """Get snapshot for a single ticker."""
        try:
            response = self._make_request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
            return response.get("results", response) if response else None
        except Exception as e:
            self.logger.error(f"Error getting single snapshot for {ticker}: {e}")
            return None

    def get_market_movers(self, direction="gainers"):
        """Get market movers with fast validation."""
        try:
            if direction == "gainers":
                endpoint = "/v2/snapshot/locale/us/markets/stocks/gainers"
            elif direction == "losers":
                endpoint = "/v2/snapshot/locale/us/markets/stocks/losers"
            else:
                return None

            response = self._make_request(endpoint)
            return response.get("results", []) if response else None
        except Exception as e:
            self.logger.error(f"Error getting market movers for {direction}: {e}")
            return None

    def get_full_market_snapshot(self, limit=50):
        """Get full market snapshot with optimized processing."""
        try:
            response = self._make_request("/v2/snapshot/locale/us/markets/stocks/tickers")
            if not response:
                return None

            # Fast dictionary comprehension
            results = response.get("results", [])
            return {item.get("ticker"): item for item in results[:limit] if item.get("ticker")}
        except Exception as e:
            self.logger.error(f"Error getting full market snapshot: {e}")
            return None

    def get_grouped_daily_bars(self, date: str):
        """Get grouped daily bars for all symbols on a specific date - production VIX source."""
        try:
            endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
            response = self._make_request(endpoint, params={"adjusted": "true"})
            
            if not response:
                return None
            
            results = response.get("results", [])
            if not results:
                return None
            
            # Convert to symbol-keyed dictionary for easy access
            grouped_data = {}
            for bar in results:
                symbol = bar.get("T")  # Ticker symbol
                if symbol:
                    grouped_data[symbol] = {
                        "symbol": symbol,
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "vwap": bar.get("vw"),
                        "timestamp": bar.get("t"),
                        "transactions": bar.get("n", 0)
                    }
            
            self.logger.debug(f"Retrieved grouped daily bars for {len(grouped_data)} symbols on {date}")
            return grouped_data
            
        except Exception as e:
            self.logger.error(f"Error getting grouped daily bars for {date}: {e}")
            return None

    def get_enhanced_market_movers(self, direction="both"):
        """Enhanced market movers for trading candidate identification."""
        try:
            movers_data = {}
            
            if direction in ["gainers", "both"]:
                gainers = self.get_market_movers("gainers")
                if gainers:
                    movers_data['gainers'] = [
                        {
                            'symbol': stock.get('ticker'),
                            'price': stock.get('value', 0),
                            'change_percent': stock.get('change_percentage', 0),
                            'volume': stock.get('session', {}).get('volume', 0),
                            'market_cap': stock.get('market_cap', 0),
                            'momentum_score': min(stock.get('change_percentage', 0) / 10.0, 1.0)
                        }
                        for stock in gainers[:50]
                    ]
            
            if direction in ["losers", "both"]:
                losers = self.get_market_movers("losers")
                if losers:
                    movers_data['losers'] = [
                        {
                            'symbol': stock.get('ticker'),
                            'price': stock.get('value', 0),
                            'change_percent': stock.get('change_percentage', 0),
                            'volume': stock.get('session', {}).get('volume', 0),
                            'market_cap': stock.get('market_cap', 0),
                            'momentum_score': max(stock.get('change_percentage', 0) / 10.0, -1.0)
                        }
                        for stock in losers[:50]
                    ]
            
            return movers_data
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced market movers: {e}")
            return {}

    def get_enhanced_market_breadth(self):
        """Enhanced market breadth analysis for regime detection."""
        try:
            etf_symbols = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE"]
            breadth_data = self.get_market_breadth_data(etf_symbols)
            
            if not breadth_data:
                return self._get_default_breadth_data()
            
            advancing = declining = total_volume = advancing_volume = 0
            
            for symbol, data in breadth_data.items():
                if not data:
                    continue
                    
                day_data = data.get('day', {})
                change = day_data.get('c', 0) - day_data.get('o', 0)
                volume = day_data.get('v', 0)
                
                total_volume += volume
                
                if change > 0:
                    advancing += 1
                    advancing_volume += volume
                elif change < 0:
                    declining += 1
            
            total_stocks = advancing + declining
            advance_decline_ratio = advancing / max(total_stocks, 1)
            volume_ratio = advancing_volume / max(total_volume, 1)
            
            market_strength = "strong" if advance_decline_ratio > 0.6 else \
                            "weak" if advance_decline_ratio < 0.4 else "neutral"
            
            return {
                'advance_decline_ratio': advance_decline_ratio,
                'advancing_volume_ratio': volume_ratio,
                'advancing_count': advancing,
                'declining_count': declining,
                'total_count': total_stocks,
                'market_strength': market_strength,
                'breadth_score': (advance_decline_ratio + volume_ratio) / 2,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced market breadth: {e}")
            return self._get_default_breadth_data()
    
    def _get_default_breadth_data(self):
        """Default market breadth data when API fails."""
        return {
            'advance_decline_ratio': 0.5,
            'advancing_volume_ratio': 0.5,
            'advancing_count': 0,
            'declining_count': 0,
            'total_count': 0,
            'market_strength': 'neutral',
            'breadth_score': 0.5,
            'timestamp': time.time()
        }

    def get_market_breadth_data(self, symbols=None):
        """Get market breadth data for regime detection."""
        try:
            if not symbols:
                symbols = ["SPY", "QQQ", "IWM"]

            breadth_data = {}
            
            for symbol in symbols:
                try:
                    snapshot = self.get_single_snapshot(symbol)
                    if snapshot:
                        breadth_data[symbol] = snapshot
                except Exception as e:
                    self.logger.warning(f"Failed to get snapshot for {symbol}: {e}")

            return breadth_data if breadth_data else None
        except Exception as e:
            self.logger.error(f"Error getting market breadth data: {e}")
            return None

    # =============================================================================
    # OPTIMIZED WEBSOCKET STREAMING METHODS
    # =============================================================================

    async def initialize_symbols(self):
        """Initialize and fetch ALL available symbols - A100 optimized."""
        self.logger.info("Initializing symbols...")
        await self.symbol_manager.fetch_all_symbols()
        all_symbols = self.symbol_manager.get_all_symbols_unfiltered()
        self.logger.info(f"Fetched {len(all_symbols)} total symbols.")

        if not self.symbols:
            self.symbols = self.symbol_manager.get_all_symbols_list()
            self.logger.info(f"No specific symbols provided, tracking all {len(self.symbols)} symbols.")
        else:
            initial_symbol_count = len(self.symbols)
            valid_symbols = [s for s in self.symbols if s in all_symbols]
            self.symbols = valid_symbols
            if len(valid_symbols) < initial_symbol_count:
                self.logger.warning(f"Some provided symbols are not valid. Tracking {len(valid_symbols)} out of {initial_symbol_count} symbols.")
            else:
                self.logger.info(f"Tracking {len(self.symbols)} specified symbols.")

    async def connect(self):
        """Establish WebSocket connection with ultra-fast processing."""
        self.logger.info("Attempting to connect to WebSocket...")
        try:
            await self._rate_limit_check()

            self.websocket = await websockets.connect(self.websocket_url)
            self.logger.info(f"Connected to {self.websocket_url}")

            # Authenticate
            auth_message = {"action": "auth", "params": self.api_key}
            await self.websocket.send(json.dumps(auth_message))

            # Wait for auth response
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)

            # Handle both single object and list responses
            if isinstance(auth_data, list):
                auth_data = auth_data[0] if auth_data else {}

            if auth_data.get("status") == "auth_success":
                self.is_connected = True
                self.health_monitor.connection_status = "connected"
                self.health_monitor.reconnect_count = self.reconnect_attempts
                self.reconnect_attempts = 0
                self.logger.info("WebSocket authenticated successfully.")

                # Subscribe to symbols
                await self._subscribe_to_symbols()

                # Start message handling
                asyncio.create_task(self._message_handler())
                asyncio.create_task(self._heartbeat_handler())
                self.logger.info("Message and heartbeat handlers started.")

            else:
                # Check if this is actually a successful connection with wrong status parsing
                message = auth_data.get('message', '')
                if 'Connected Successfully' in message or 'success' in message.lower():
                    self.is_connected = True
                    self.health_monitor.connection_status = "connected"
                    self.health_monitor.reconnect_count = self.reconnect_attempts
                    self.reconnect_attempts = 0
                    self.logger.info("WebSocket authenticated successfully (parsed from message).")
                    
                    # Subscribe to symbols
                    await self._subscribe_to_symbols()
                    
                    # Start message handling
                    asyncio.create_task(self._message_handler())
                    asyncio.create_task(self._heartbeat_handler())
                    self.logger.info("Message and heartbeat handlers started.")
                else:
                    self.logger.error(f"Authentication failed: {auth_data.get('message', 'Unknown error')}")
                    await self._handle_reconnection()

        except Exception as e:
            self.logger.error(f"Error during WebSocket connection: {e}")
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()

    async def _subscribe_to_symbols(self):
        """Ultra-optimized subscription to essential WebSocket streams only."""
        if not self.symbols:
            return

        all_subscriptions = []
        
        # Subscribe to enabled streams
        if ENABLE_SECOND_AGGREGATES:
            all_subscriptions.extend([f"AS.{symbol}" for symbol in self.symbols])
            
        if ENABLE_QUOTES:
            all_subscriptions.extend([f"Q.{symbol}" for symbol in self.symbols])
            
        if ENABLE_TRADES:
            all_subscriptions.extend([f"T.{symbol}" for symbol in self.symbols])
            
        if ENABLE_MINUTE_AGGREGATES:
            all_subscriptions.extend([f"A.{symbol}" for symbol in self.symbols])

        if not all_subscriptions:
            self.logger.warning("No subscriptions enabled - check configuration")
            return

        # Calculate optimization metrics
        original_stream_count = len(self.symbols) * 4  # T, Q, A, AS
        optimized_stream_count = len(all_subscriptions)
        reduction_pct = ((original_stream_count - optimized_stream_count) / original_stream_count) * 100
        
        self.logger.info(f"🚀 OPTIMIZED WebSocket: {optimized_stream_count} subscriptions ({reduction_pct:.1f}% reduction)")

        # ULTRA-LOW LATENCY: Single batch subscription
        if len(all_subscriptions) <= self.websocket_batch_size:
            subscribe_message = {"action": "subscribe", "params": ",".join(all_subscriptions)}
            await self.websocket.send(json.dumps(subscribe_message))
            self.logger.info(f"🚀 SINGLE BATCH: Subscribed to {len(all_subscriptions)} streams in ONE batch (ultra-low latency)")
        else:
            # Fallback to multiple batches if needed
            self.logger.warning(f"⚠️ Using multiple batches: {len(all_subscriptions)} > {self.websocket_batch_size}")
            for i in range(0, len(all_subscriptions), self.websocket_batch_size):
                batch = all_subscriptions[i : i + self.websocket_batch_size]
                subscribe_message = {"action": "subscribe", "params": ",".join(batch)}
                await self.websocket.send(json.dumps(subscribe_message))
                self.logger.info(f"Subscribed to {len(batch)} streams in batch {i//self.websocket_batch_size + 1}")
                await asyncio.sleep(MIN_REQUEST_INTERVAL * 2)

    async def _message_handler(self):
        """Ultra-fast message handler with minimal overhead."""
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
                    self.logger.error(f"Failed to decode JSON message: {message}. Error: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed during message handling: {e}")
            self.is_connected = False
            self.health_monitor.connection_status = "disconnected"
            await self._handle_reconnection()
        except Exception as e:
            self.logger.error(f"Unexpected error in message handler: {e}")
            self.is_connected = False
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()

    async def _process_message(self, data):
        """Ultra-fast message processing with minimal validation overhead."""
        try:
            msg_type = data.get("ev")
            symbol = data.get("sym", "UNKNOWN")

            if msg_type == "Q":  # Quote
                if self.data_validator.validate_quote_data(data):
                    bid_price = self.data_validator.sanitize_price(data.get("bp"))
                    ask_price = self.data_validator.sanitize_price(data.get("ap"))

                    if bid_price is not None and ask_price is not None:
                        mid_price = (bid_price + ask_price) / 2
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=data.get("t", 0) / 1000,
                            price=mid_price,
                            volume=0,
                            bid=bid_price,
                            ask=ask_price,
                            bid_size=data.get("bs", 0),
                            ask_size=data.get("as", 0),
                            data_type="quote",
                        )

                        self._process_market_data(market_data, "quote")

            elif msg_type == "AS":  # 1-second Aggregate
                if self._validate_aggregate_data(data):
                    open_price = self.data_validator.sanitize_price(data.get("o"))
                    high_price = self.data_validator.sanitize_price(data.get("h"))
                    low_price = self.data_validator.sanitize_price(data.get("l"))
                    close_price = self.data_validator.sanitize_price(data.get("c"))
                    volume = self.data_validator.sanitize_volume(data.get("v"))
                    vwap = self.data_validator.sanitize_price(data.get("vw"))

                    if all(x is not None for x in [open_price, high_price, low_price, close_price, volume]):
                        aggregate_data = AggregateData(
                            symbol=symbol,
                            timestamp=data.get("t", 0) / 1000,
                            open_price=open_price,
                            high_price=high_price,
                            low_price=low_price,
                            close_price=close_price,
                            volume=volume,
                            vwap=vwap,
                            data_type="second_aggregate",
                        )

                        self._process_market_data(aggregate_data, "second_aggregate")

        except Exception as e:
            self.logger.error(f"Error processing market data for symbol {symbol}: {e}")

    def _validate_aggregate_data(self, data):
        """Ultra-fast aggregate data validation."""
        if not self.data_validator.enabled:
            return True
        return all(field in data for field in ("sym", "o", "h", "l", "c", "v", "t"))

    async def _heartbeat_handler(self):
        """Ultra-fast heartbeat handler."""
        while self.is_connected:
            try:
                if self.websocket and hasattr(self.websocket, 'ping'):
                    await self.websocket.ping()
                    self.health_monitor.update_heartbeat()

                await asyncio.sleep(self.health_monitor.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error in heartbeat handler: {e}")
                break

    async def _handle_reconnection(self):
        """Handle automatic reconnection with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
            return

        self.reconnect_attempts += 1
        backoff_time = self.reconnect_backoff_base * (2**self.reconnect_attempts)
        wait_time = min(backoff_time, self.reconnect_backoff_max)
        self.logger.warning(
            f"Attempting to reconnect in {wait_time:.2f} seconds "
            f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )

        await asyncio.sleep(wait_time)
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self.reconnect_attempts} failed: {e}")

    # =============================================================================
    # ULTRA-FAST DATA PROCESSING
    # =============================================================================

    def _process_market_data(self, data, data_type):
        """Ultra-fast market data processing with integrated pipeline."""
        symbol = data.symbol

        # Zero-copy processing if enabled
        if self.zero_copy_enabled:
            self._update_memory_pools_zero_copy(data, data_type)
        
        # Update latest data with minimal overhead
        if data_type in ["minute_aggregate", "second_aggregate"]:
            self.latest_aggregates[symbol] = data
        
        self.latest_data[symbol] = data

        # Integrated processing pipeline: Filter → Features → ML → Kelly → Execute
        try:
            # Step 1: Apply real-time filtering
            filtered_data = self._apply_filtering(data, data_type)
            if not filtered_data:
                return  # Data didn't pass filter
            
            # Step 2: Feature engineering
            features = self._extract_features(filtered_data, data_type)
            
            # Step 3: ML prediction
            ml_prediction = self._get_ml_prediction(filtered_data, features)
            
            # Step 4: Kelly position sizing
            position_size = self._calculate_position_size(filtered_data, ml_prediction)
            
            # Step 5: Execute trade if conditions met
            if position_size and abs(position_size) > 0:
                asyncio.create_task(self._execute_trade(filtered_data, position_size, ml_prediction))
            
        except Exception as e:
            self.logger.error(f"Integrated pipeline error for {symbol}: {e}")

        # Call unified data callback if provided
        if self.data_callback:
            try:
                self.data_callback(data, data_type)
            except Exception as e:
                self.logger.error(f"Data callback error for {symbol}: {e}")

    def _apply_filtering(self, data, data_type):
        """Apply adaptive filtering with fallback."""
        symbol = data.symbol
        
        if self.enable_filtering and self.adaptive_filter:
            filter_start = time.time()
            try:
                # Convert to format expected by adaptive filter
                polygon_data = [{
                    'symbol': data.symbol,
                    'price': data.price,
                    'volume': data.volume,
                    'timestamp': data.timestamp,
                    'market_cap': getattr(data, 'market_cap', 1000000000),
                    'daily_change': getattr(data, 'daily_change', 0),
                    'volatility': getattr(data, 'volatility', 0.02),
                    'momentum_score': getattr(data, 'momentum_score', 0)
                }]
                
                # Process through adaptive filter (async call in sync context)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task for async processing
                        asyncio.create_task(self.adaptive_filter.process_polygon_data(polygon_data))
                        filtered_results = []
                    else:
                        filtered_results = asyncio.run(self.adaptive_filter.process_polygon_data(polygon_data))
                except RuntimeError:
                    filtered_results = []
                
                if filtered_results:
                    filtered_data = {
                        'symbol': data.symbol,
                        'price': data.price,
                        'volume': data.volume,
                        'timestamp': data.timestamp,
                        'filtered': True,
                        'ml_score': getattr(filtered_results[0], 'ml_score', 0.8),
                        'market_condition': getattr(filtered_results[0], 'market_condition', 'unknown')
                    }
                    self.filtered_data[symbol] = filtered_data
                    self.filter_stats['filtered_passed'] += 1
                    
                    filter_time = time.time() - filter_start
                    self.filter_stats['filter_processing_times'].append(filter_time)
                    
                    # Keep only last 100 processing times
                    if len(self.filter_stats['filter_processing_times']) > 100:
                        self.filter_stats['filter_processing_times'] = self.filter_stats['filter_processing_times'][-100:]
                    
                    return filtered_data
                
                self.filter_stats['total_processed'] += 1
                return None  # Didn't pass filter
                    
            except Exception as e:
                self.logger.warning(f"Adaptive filtering failed for {symbol}: {e}")
                
        elif self.enable_filtering:
            # Fallback simple filtering
            if data.price > 1.0 and data.price < 1000.0 and data.volume > 100000:
                filtered_data = {
                    'symbol': data.symbol,
                    'price': data.price,
                    'volume': data.volume,
                    'timestamp': data.timestamp,
                    'filtered': True
                }
                self.filtered_data[symbol] = filtered_data
                self.filter_stats['filtered_passed'] += 1
                self.filter_stats['total_processed'] += 1
                return filtered_data
            
            self.filter_stats['total_processed'] += 1
            return None
        
        # No filtering enabled, pass through
        return {
            'symbol': data.symbol,
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp,
            'filtered': False
        }

    def _extract_features(self, filtered_data, data_type):
        """Extract features using feature engineering system."""
        try:
            if hasattr(self, 'feature_engine') and self.feature_engine:
                # Use integrated feature engine
                features = self.feature_engine.extract_features(filtered_data, data_type)
                return features
            else:
                # Fallback basic features
                return {
                    'price': filtered_data['price'],
                    'volume': filtered_data['volume'],
                    'timestamp': filtered_data['timestamp'],
                    'price_momentum': 0.0,
                    'volume_momentum': 0.0,
                    'volatility': 0.02
                }
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {filtered_data['symbol']}: {e}")
            return None

    def _get_ml_prediction(self, filtered_data, features):
        """Get ML prediction using ensemble system."""
        try:
            if hasattr(self, 'ml_system') and self.ml_system and features:
                # Use integrated ML system
                prediction = self.ml_system.predict(features)
                
                # Cache prediction in ML bridge
                if self.ml_bridge:
                    symbol_idx = self._get_symbol_index(filtered_data['symbol'])
                    if symbol_idx >= 0:
                        self.ml_bridge.cache_ml_prediction(
                            symbol_idx,
                            prediction.get('prediction', 0.0),
                            prediction.get('confidence', 0.5),
                            prediction.get('regime', 0),
                            prediction.get('quality_score', 1.0)
                        )
                
                return prediction
            else:
                # Fallback basic prediction
                return {
                    'prediction': 0.0,
                    'confidence': 0.5,
                    'regime': 0,
                    'quality_score': 0.5
                }
        except Exception as e:
            self.logger.warning(f"ML prediction failed for {filtered_data['symbol']}: {e}")
            return None

    def _calculate_position_size(self, filtered_data, ml_prediction):
        """Calculate position size using Kelly criterion."""
        try:
            if hasattr(self, 'kelly_sizer') and self.kelly_sizer and ml_prediction:
                # Use integrated Kelly sizer
                position_size = self.kelly_sizer.calculate_position_size(
                    filtered_data, ml_prediction
                )
                return position_size
            else:
                # Fallback basic position sizing
                if ml_prediction and ml_prediction.get('confidence', 0) > 0.6:
                    return 100  # Basic 100 share position
                return 0
        except Exception as e:
            self.logger.warning(f"Position sizing failed for {filtered_data['symbol']}: {e}")
            return 0

    async def _execute_trade(self, filtered_data, position_size, ml_prediction):
        """Execute trade using execution engine."""
        try:
            if hasattr(self, 'executor') and self.executor:
                # Use integrated execution engine
                await self.executor.execute_trade(
                    symbol=filtered_data['symbol'],
                    position_size=position_size,
                    price=filtered_data['price'],
                    ml_prediction=ml_prediction
                )
            else:
                # Log trade that would be executed
                self.logger.info(f"TRADE SIGNAL: {filtered_data['symbol']} "
                               f"size={position_size} price={filtered_data['price']} "
                               f"confidence={ml_prediction.get('confidence', 0)}")
        except Exception as e:
            self.logger.error(f"Trade execution failed for {filtered_data['symbol']}: {e}")

    def _get_symbol_index(self, symbol):
        """Get symbol index for memory pool operations."""
        try:
            if self.memory_pools and 'symbol_to_index' in self.memory_pools:
                symbol_to_index = self.memory_pools['symbol_to_index']
                return symbol_to_index.get(symbol, -1)
            return -1
        except Exception:
            return -1

    def _update_memory_pools_zero_copy(self, data, data_type):
        """Ultra-fast zero-copy memory pool updates."""
        try:
            if not self.memory_pools:
                return
            
            symbol = data.symbol
            
            # Get memory pool references
            market_data_pool = self.memory_pools.get('market_data_pool')
            symbol_to_index = self.memory_pools.get('symbol_to_index', {})
            index_to_symbol = self.memory_pools.get('index_to_symbol', [])
            active_symbols_mask = self.memory_pools.get('active_symbols_mask')
            
            if market_data_pool is None:
                return
            
            # Get or create symbol index
            symbol_idx = self._get_symbol_index_zero_copy(symbol, symbol_to_index, index_to_symbol)
            
            if symbol_idx >= 0 and symbol_idx < len(market_data_pool):
                # Update market data pool directly (zero-copy)
                market_data_pool[symbol_idx, 0] = data.price
                market_data_pool[symbol_idx, 1] = data.volume
                market_data_pool[symbol_idx, 2] = data.timestamp
                
                if hasattr(data, 'bid') and data.bid:
                    market_data_pool[symbol_idx, 3] = data.bid
                if hasattr(data, 'ask') and data.ask:
                    market_data_pool[symbol_idx, 4] = data.ask
                
                # Mark as active
                if active_symbols_mask is not None and symbol_idx < len(active_symbols_mask):
                    active_symbols_mask[symbol_idx] = True
                
        except Exception as e:
            self.logger.error(f"Zero-copy memory pool update failed for {data.symbol}: {e}")

    def _get_symbol_index_zero_copy(self, symbol, symbol_to_index, index_to_symbol):
        """Ultra-fast symbol index lookup/creation."""
        if symbol in symbol_to_index:
            return symbol_to_index[symbol]
        
        # Find next available slot
        for i in range(len(index_to_symbol)):
            if index_to_symbol[i] == '' or index_to_symbol[i] == symbol:
                symbol_to_index[symbol] = i
                index_to_symbol[i] = symbol
                return i
        
        return -1  # Pool full

    # =============================================================================
    # OPTIMIZED DATA ACCESS METHODS
    # =============================================================================

    def get_latest_data(self, symbol):
        """Get latest data for a symbol."""
        return self.latest_data.get(symbol)

    def get_all_symbols(self):
        """Get all symbols being tracked."""
        return list(self.latest_data.keys())

    def get_filtered_data(self, symbol):
        """Get filtered data for a symbol."""
        return self.filtered_data.get(symbol)

    def get_all_filtered_symbols(self):
        """Get all symbols that passed filtering."""
        return list(self.filtered_data.keys())

    def get_latest_aggregate(self, symbol):
        """Get latest aggregate data for a symbol."""
        return self.latest_aggregates.get(symbol)

    def get_all_aggregate_symbols(self):
        """Get all symbols with aggregate data."""
        return list(self.latest_aggregates.keys())

    def get_filter_performance_stats(self):
        """Get real-time filter performance statistics."""
        if not self.enable_filtering:
            return {"filtering_enabled": False}

        processing_times = self.filter_stats['filter_processing_times']
        total_processed = self.filter_stats['total_processed']
        filtered_passed = self.filter_stats['filtered_passed']

        return {
            "filtering_enabled": True,
            "total_processed": total_processed,
            "filtered_passed": filtered_passed,
            "filter_pass_rate": (filtered_passed / max(total_processed, 1)) * 100,
            "avg_filter_time_ms": (sum(processing_times) / max(len(processing_times), 1)) * 1000,
            "p95_filter_time_ms": (sorted(processing_times)[int(len(processing_times) * 0.95)] if processing_times else 0) * 1000,
            "symbols_tracked": len(self.latest_data),
            "symbols_filtered": len(self.filtered_data)
        }

    def get_stats(self):
        """Get client statistics."""
        total_requests = self.stats["requests"]["total"]
        if total_requests > 0:
            success_rate = (self.stats["requests"]["successful"] / total_requests) * 100
            avg_response_time = (
                self.stats["performance"]["total_response_time_ms"] / total_requests
            )
        else:
            success_rate = 0
            avg_response_time = 0

        return {
            "requests": {
                "total": total_requests,
                "successful": self.stats["requests"]["successful"],
                "failed": self.stats["requests"]["failed"],
                "success_rate_pct": success_rate,
            },
            "performance": {"avg_response_time_ms": avg_response_time},
        }

    def is_healthy(self):
        """Check if client is healthy."""
        stats = self.get_stats()
        if stats["requests"]["total"] == 0:
            return True
        return stats["requests"]["success_rate_pct"] > 50

    def get_health_status(self):
        """Get connection health status."""
        return self.health_monitor.get_status()

    def get_connection_stats(self):
        """Get comprehensive connection statistics."""
        health_status = self.get_health_status()
        return {
            "health": health_status,
            "symbols_subscribed": len(self.symbols),
            "reconnect_attempts": self.reconnect_attempts,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "is_connected": self.is_connected,
            "websocket_url": self.websocket_url,
            "batch_size_optimized": self.websocket_batch_size,
            "a100_optimized": True,
        }

    async def disconnect(self):
        """Gracefully disconnect from WebSocket."""
        if self.websocket:
            self.logger.info("Disconnecting from WebSocket...")
            try:
                # Check if websocket is still open before closing
                if hasattr(self.websocket, 'close') and not getattr(self.websocket, 'closed', True):
                    await self.websocket.close()
                elif hasattr(self.websocket, 'close'):
                    # Try to close anyway, ignore errors if already closed
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass  # Already closed or connection lost
            except Exception as e:
                self.logger.debug(f"Error during websocket close: {e}")
            finally:
                self.is_connected = False
                self.health_monitor.connection_status = "disconnected"
                self.websocket = None
                self.logger.info("WebSocket disconnected.")

    async def stop(self):
        """Stop the data feed."""
        self.logger.info("Stopping real-time data feed...")
        await self.disconnect()
        
        # Cleanup TensorRT resources
        if hasattr(self, 'symbol_manager') and self.symbol_manager:
            self.symbol_manager.cleanup()
        
        # Log final filter stats if enabled
        if self.enable_filtering:
            stats = self.get_filter_performance_stats()
            self.logger.info(f"Final filter stats: {stats}")
        
        self.logger.info("Real-time data feed stopped.")

    def _create_zero_copy_memory_pools(self):
        """Create zero-copy memory pools for ultra-low latency processing."""
        try:
            import numpy as np
            
            pool_size = MEMORY_POOL_SIZE
            self.logger.info(f"Creating zero-copy memory pools for {pool_size} symbols")
            
            # Pre-allocate memory pools
            memory_pools = {
                'market_data_pool': np.zeros((pool_size, 8), dtype=np.float64),
                'symbol_to_index': {},
                'index_to_symbol': [''] * pool_size,
                'active_symbols_mask': np.zeros(pool_size, dtype=bool),
                'filter_results_pool': np.zeros((pool_size, 4), dtype=np.float64),
                'aggregate_pool': np.zeros((pool_size, 10), dtype=np.float64),
            }
            
            self.logger.info("Zero-copy memory pools created successfully")
            return memory_pools
            
        except ImportError:
            self.logger.warning("NumPy not available, falling back to standard memory allocation")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to create zero-copy memory pools: {e}")
            return {}

    async def start(self):
        """Start the unified Polygon client with comprehensive auto-initialization."""
        self.logger.info("🚀 Starting ultra-optimized Polygon client with auto-initialization...")
        
        try:
            # Step 1: Auto-initialize all components
            await self._auto_initialize_components()
            
            # Step 2: Initialize all symbols
            await self.initialize_symbols()
            self.logger.info(f"✓ Initialized {len(self.symbols)} symbols for tracking")
            
            # Step 3: Connect to WebSocket and start streaming
            await self.connect()
            self.logger.info("✓ WebSocket connection established and streaming started")
            
            # Step 4: Start continuous operation with integrated pipeline
            self.logger.info("✓ Polygon client started successfully - processing real-time data")
            self.logger.info("🎯 Integrated pipeline: Polygon → Filter → ML → Kelly → Alpaca")
            
            # Keep running until interrupted
            try:
                while self.is_connected:
                    await asyncio.sleep(1)
                    
                    # Log periodic stats
                    if hasattr(self, '_last_stats_log'):
                        if time.time() - self._last_stats_log > 60:  # Every minute
                            self._log_performance_stats()
                    else:
                        self._last_stats_log = time.time()
                        
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested")
                
        except Exception as e:
            self.logger.error(f"Error starting Polygon client: {e}")
            raise
        finally:
            await self.stop()

    async def _auto_initialize_components(self):
        """Auto-initialize all trading system components."""
        self.logger.info("🔧 Auto-initializing trading system components...")
        
        # Create unified memory pools if not provided
        if not self.memory_pools:
            self.memory_pools = self._create_unified_memory_pools()
            self.zero_copy_enabled = bool(self.memory_pools)
            self.logger.info("✓ Created unified zero-copy memory pools")
        
        # Initialize portfolio manager if not provided
        if not self.portfolio_manager:
            self.portfolio_manager = self._create_portfolio_manager()
            self.logger.info("✓ Created unified portfolio state manager")
        
        # Initialize ML bridge if not provided
        if not self.ml_bridge:
            self.ml_bridge = self._create_ml_bridge()
            self.logger.info("✓ Created ML prediction bridge")
        
        # Auto-initialize ML ensemble system
        if not hasattr(self, 'ml_system') or not self.ml_system:
            self.ml_system = await self._auto_initialize_ml_system()
            if self.ml_system:
                self.logger.info("✓ ML ensemble system auto-initialized")
        
        # Auto-initialize Kelly position sizer
        if not hasattr(self, 'kelly_sizer') or not self.kelly_sizer:
            self.kelly_sizer = await self._auto_initialize_kelly_sizer()
            if self.kelly_sizer:
                self.logger.info("✓ Kelly position sizer auto-initialized")
        
        # Auto-initialize execution engine
        if not hasattr(self, 'executor') or not self.executor:
            self.executor = await self._auto_initialize_executor()
            if self.executor:
                self.logger.info("✓ Execution engine auto-initialized")
        
        # Auto-initialize feature engineering
        if not hasattr(self, 'feature_engine') or not self.feature_engine:
            self.feature_engine = await self._auto_initialize_feature_engine()
            if self.feature_engine:
                self.logger.info("✓ Feature engineering system auto-initialized")
        
        # Wire all components together
        await self._wire_components()
        self.logger.info("✓ All components wired together for unified data flow")

    async def _auto_initialize_ml_system(self):
        """Auto-initialize ML ensemble system."""
        try:
            from ml_ensemble_system import UltraFastMLEnsembleSystem
            ml_system = UltraFastMLEnsembleSystem(
                gpu_enabled=True,
                memory_pools=self.memory_pools
            )
            return ml_system
        except ImportError as e:
            self.logger.warning(f"ML ensemble system not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize ML ensemble system: {e}")
            return None

    async def _auto_initialize_kelly_sizer(self):
        """Auto-initialize Kelly position sizer."""
        try:
            from kelly_position_sizer import UltraFastKellyPositionSizer
            kelly_sizer = UltraFastKellyPositionSizer(
                available_capital=50000.0,
                memory_pools=self.memory_pools
            )
            return kelly_sizer
        except ImportError as e:
            self.logger.warning(f"Kelly position sizer not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize Kelly position sizer: {e}")
            return None

    async def _auto_initialize_executor(self):
        """Auto-initialize execution engine."""
        try:
            from alpaca_momentum_executor import UltraFastAlpacaMomentumExecutor
            executor = UltraFastAlpacaMomentumExecutor(
                initial_capital=50000.0,
                memory_pools=self.memory_pools
            )
            return executor
        except ImportError as e:
            self.logger.warning(f"Execution engine not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize execution engine: {e}")
            return None

    async def _auto_initialize_feature_engine(self):
        """Auto-initialize feature engineering system."""
        try:
            from feature_engineering import UltraFastFeatureEngineering
            feature_engine = UltraFastFeatureEngineering(
                memory_pools=self.memory_pools,
                polygon_client=self
            )
            return feature_engine
        except ImportError as e:
            self.logger.warning(f"Feature engineering system not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize feature engineering system: {e}")
            return None

    async def _wire_components(self):
        """Wire all components together for unified data flow."""
        try:
            # Wire ML system
            if self.ml_system:
                self.ml_system.ml_bridge = self.ml_bridge
                self.ml_system.portfolio_manager = self.portfolio_manager
                if hasattr(self, 'feature_engine') and self.feature_engine:
                    self.ml_system.feature_engine = self.feature_engine
            
            # Wire Kelly sizer
            if self.kelly_sizer:
                self.kelly_sizer.ml_bridge = self.ml_bridge
                self.kelly_sizer.portfolio_manager = self.portfolio_manager
                if self.ml_system:
                    self.kelly_sizer.ml_system = self.ml_system
            
            # Wire executor
            if self.executor:
                self.executor.portfolio_manager = self.portfolio_manager
                if self.kelly_sizer:
                    self.executor.kelly_sizer = self.kelly_sizer
                if self.ml_system:
                    self.executor.ml_system = self.ml_system
            
            # Wire feature engine
            if hasattr(self, 'feature_engine') and self.feature_engine:
                self.feature_engine.polygon_client = self
                self.feature_engine.portfolio_manager = self.portfolio_manager
            
            # Wire adaptive filter
            if self.adaptive_filter:
                self.adaptive_filter.polygon_client = self
                self.adaptive_filter.portfolio_manager = self.portfolio_manager
                self.adaptive_filter.ml_bridge = self.ml_bridge
            
            self.logger.info("✓ Component wiring completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error wiring components: {e}")

    def _create_unified_memory_pools(self):
        """Create comprehensive unified memory pools for all components."""
        try:
            import numpy as np
            
            pool_size = 15000  # Support up to 15,000 symbols
            self.logger.info(f"Creating unified zero-copy memory pools for {pool_size} symbols")
            
            memory_pools = {
                # Polygon client pools
                'market_data_pool': np.zeros((pool_size, 8), dtype=np.float64),
                'symbol_to_index': {},
                'index_to_symbol': [''] * pool_size,
                'active_symbols_mask': np.zeros(pool_size, dtype=bool),
                'filter_results_pool': np.zeros((pool_size, 4), dtype=np.float64),
                'aggregate_pool': np.zeros((pool_size, 10), dtype=np.float64),
                
                # Feature engineering pools
                'feature_pool': np.zeros((pool_size, 15), dtype=np.float32),
                'price_feature_pool': np.zeros((pool_size, 6), dtype=np.float32),
                'volume_feature_pool': np.zeros((pool_size, 4), dtype=np.float32),
                'technical_feature_pool': np.zeros((pool_size, 8), dtype=np.float32),
                'context_feature_pool': np.zeros((pool_size, 4), dtype=np.float32),
                'orderflow_feature_pool': np.zeros((pool_size, 3), dtype=np.float32),
                
                # ML ensemble pools
                'prediction_pool': np.zeros((pool_size, 8), dtype=np.float32),
                'ml_ready_mask': np.zeros(pool_size, dtype=bool),
                'confidence_pool': np.zeros((pool_size, 4), dtype=np.float32),
                'regime_pool': np.zeros((pool_size, 3), dtype=np.int32),
                'ml_prediction_cache': np.zeros((pool_size, 6), dtype=np.float32),
                
                # Kelly position sizer pools
                'position_pool': np.zeros((pool_size, 8), dtype=np.float64),
                'tier_pool': np.zeros((pool_size, 4), dtype=np.int32),
                'price_pool': np.zeros((pool_size, 4), dtype=np.float64),
                'kelly_results_pool': np.zeros((pool_size, 6), dtype=np.float64),
                
                # Execution engine pools
                'order_pool': np.zeros((pool_size, 10), dtype=np.float64),
                'execution_pool': np.zeros((pool_size, 8), dtype=np.float64),
                'pnl_pool': np.zeros((pool_size, 6), dtype=np.float64),
                
                # Portfolio state management pools
                'portfolio_state_pool': np.zeros((1, 12), dtype=np.float64),
                'position_state_pool': np.zeros((pool_size, 8), dtype=np.float64),
                'cash_flow_pool': np.zeros((pool_size, 4), dtype=np.float64),
                'risk_metrics_pool': np.zeros((1, 8), dtype=np.float64),
                
                # Shared metadata
                'symbol_metadata_pool': np.zeros((pool_size, 12), dtype=np.float64),
                'timestamp_pool': np.zeros((pool_size, 4), dtype=np.float64),
                'status_pool': np.zeros((pool_size, 8), dtype=np.int32),
                'performance_metrics_pool': np.zeros((pool_size, 6), dtype=np.float64),
            }
            
            self.logger.info(f"✓ Unified zero-copy memory pools created: {len(memory_pools)} pools")
            self.logger.info(f"✓ Total memory allocated: ~{(sum(pool.nbytes for pool in memory_pools.values() if hasattr(pool, 'nbytes')) / 1024 / 1024):.1f} MB")
            
            return memory_pools
            
        except ImportError:
            self.logger.warning("NumPy not available, falling back to standard memory allocation")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to create unified zero-copy memory pools: {e}")
            return {}

    def _create_portfolio_manager(self):
        """Create unified portfolio state manager."""
        try:
            class UnifiedPortfolioStateManager:
                def __init__(self, memory_pools, initial_capital=50000.0):
                    self.memory_pools = memory_pools
                    self.initial_capital = initial_capital
                    
                    # Initialize portfolio state in memory pool
                    if 'portfolio_state_pool' in memory_pools:
                        portfolio_pool = memory_pools['portfolio_state_pool']
                        portfolio_pool[0, 0] = initial_capital  # cash_available
                        portfolio_pool[0, 1] = initial_capital  # portfolio_value
                        portfolio_pool[0, 2] = 0.0  # daily_pnl
                        portfolio_pool[0, 3] = 0.0  # current_positions
                        portfolio_pool[0, 4] = 0.0  # daily_trades
                        portfolio_pool[0, 5] = 1000.0  # daily_target
                        portfolio_pool[0, 6] = 0.0  # total_exposure
                        portfolio_pool[0, 7] = 0.0  # unrealized_pnl
                        portfolio_pool[0, 8] = 0.0  # realized_pnl
                        portfolio_pool[0, 9] = time.time()  # last_update
                        portfolio_pool[0, 10] = 0.0  # max_drawdown
                        portfolio_pool[0, 11] = 0.0  # risk_score
                
                def get_portfolio_state(self):
                    if 'portfolio_state_pool' not in self.memory_pools:
                        return self._get_fallback_state()
                    
                    portfolio_pool = self.memory_pools['portfolio_state_pool']
                    return {
                        'cash_available': float(portfolio_pool[0, 0]),
                        'portfolio_value': float(portfolio_pool[0, 1]),
                        'daily_pnl': float(portfolio_pool[0, 2]),
                        'current_positions': int(portfolio_pool[0, 3]),
                        'daily_trades': int(portfolio_pool[0, 4]),
                        'daily_target': float(portfolio_pool[0, 5]),
                        'total_exposure': float(portfolio_pool[0, 6]),
                        'unrealized_pnl': float(portfolio_pool[0, 7]),
                        'realized_pnl': float(portfolio_pool[0, 8]),
                        'last_update': float(portfolio_pool[0, 9]),
                        'max_drawdown': float(portfolio_pool[0, 10]),
                        'risk_score': float(portfolio_pool[0, 11])
                    }
                
                def update_portfolio_state(self, **kwargs):
                    if 'portfolio_state_pool' not in self.memory_pools:
                        return
                    
                    portfolio_pool = self.memory_pools['portfolio_state_pool']
                    field_mapping = {
                        'cash_available': 0, 'portfolio_value': 1, 'daily_pnl': 2,
                        'current_positions': 3, 'daily_trades': 4, 'daily_target': 5,
                        'total_exposure': 6, 'unrealized_pnl': 7, 'realized_pnl': 8,
                        'max_drawdown': 10, 'risk_score': 11
                    }
                    
                    for field, value in kwargs.items():
                        if field in field_mapping:
                            portfolio_pool[0, field_mapping[field]] = float(value)
                    
                    portfolio_pool[0, 9] = time.time()  # Always update timestamp
                
                def _get_fallback_state(self):
                    return {
                        'cash_available': self.initial_capital,
                        'portfolio_value': self.initial_capital,
                        'daily_pnl': 0.0,
                        'current_positions': 0,
                        'daily_trades': 0,
                        'daily_target': 1000.0,
                        'total_exposure': 0.0,
                        'unrealized_pnl': 0.0,
                        'realized_pnl': 0.0,
                        'last_update': time.time(),
                        'max_drawdown': 0.0,
                        'risk_score': 0.0
                    }
            
            return UnifiedPortfolioStateManager(self.memory_pools, initial_capital=50000.0)
            
        except Exception as e:
            self.logger.error(f"Failed to create portfolio manager: {e}")
            return None

    def _create_ml_bridge(self):
        """Create ML prediction bridge."""
        try:
            class MLPredictionBridge:
                def __init__(self, memory_pools):
                    self.memory_pools = memory_pools
                    self.prediction_cache_ttl = 1.0  # 1 second TTL for predictions
                
                def cache_ml_prediction(self, symbol_idx, prediction, confidence, regime, quality_score=1.0):
                    if 'ml_prediction_cache' not in self.memory_pools:
                        return
                    
                    cache_pool = self.memory_pools['ml_prediction_cache']
                    if symbol_idx < len(cache_pool):
                        cache_pool[symbol_idx, 0] = symbol_idx
                        cache_pool[symbol_idx, 1] = prediction
                        cache_pool[symbol_idx, 2] = confidence
                        cache_pool[symbol_idx, 3] = regime
                        cache_pool[symbol_idx, 4] = time.time()  # timestamp
                        cache_pool[symbol_idx, 5] = quality_score
                
                def get_ml_prediction(self, symbol_idx):
                    if 'ml_prediction_cache' not in self.memory_pools:
                        return None
                    
                    cache_pool = self.memory_pools['ml_prediction_cache']
                    if symbol_idx >= len(cache_pool):
                        return None
                    
                    # Check TTL
                    timestamp = cache_pool[symbol_idx, 4]
                    if time.time() - timestamp > self.prediction_cache_ttl:
                        return None
                    
                    return {
                        'prediction': float(cache_pool[symbol_idx, 1]),
                        'confidence': float(cache_pool[symbol_idx, 2]),
                        'regime': int(cache_pool[symbol_idx, 3]),
                        'timestamp': timestamp,
                        'quality_score': float(cache_pool[symbol_idx, 5])
                    }
            
            return MLPredictionBridge(self.memory_pools)
            
        except Exception as e:
            self.logger.error(f"Failed to create ML bridge: {e}")
            return None

    def _log_performance_stats(self):
        """Log performance statistics for monitoring."""
        try:
            stats = self.get_stats()
            health = self.get_health_status()
            filter_stats = self.get_filter_performance_stats()
            
            self.logger.info(f"Performance Stats - Requests: {stats['requests']['total']}, "
                           f"Success Rate: {stats['requests']['success_rate_pct']:.1f}%, "
                           f"Avg Response: {stats['performance']['avg_response_time_ms']:.2f}ms")
            
            self.logger.info(f"Health Stats - Connected: {health['is_healthy']}, "
                           f"Messages: {health['total_messages']}, "
                           f"Reconnects: {health['reconnect_count']}")
            
            if filter_stats.get('filtering_enabled'):
                self.logger.info(f"Filter Stats - Processed: {filter_stats['total_processed']}, "
                               f"Pass Rate: {filter_stats['filter_pass_rate']:.1f}%, "
                               f"Avg Filter Time: {filter_stats['avg_filter_time_ms']:.3f}ms")
            
            self._last_stats_log = time.time()
            
        except Exception as e:
            self.logger.warning(f"Error logging performance stats: {e}")



# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for existing polygon_websocket.py usage
class RealTimeDataFeed(PolygonClient):
    """Backward compatibility alias for RealTimeDataFeed."""
    pass

# Alias for existing polygon_rest_api.py usage
class PolygonRESTClient(PolygonClient):
    """Backward compatibility alias for PolygonRESTClient."""
    pass