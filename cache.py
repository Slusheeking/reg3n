"""
High-Performance Tiered Cache System for GH200
Optimized for ultra-low latency trading data access with intelligent routing
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque, Set, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import orjson

from settings import config

logger = logging.getLogger(__name__)

class DataSource(Enum):
    L1_CACHE = "l1_cache"
    L2_CACHE = "l2_cache"
    DATABASE = "database"

@dataclass
class DataQuality:
    """Data quality metrics for intelligent routing"""
    freshness_score: float  # 0-1, based on age
    completeness_score: float  # 0-1, based on data availability
    latency_score: float  # 0-1, based on access speed
    consistency_score: float  # 0-1, based on data integrity
    
    @property
    def total_score(self) -> float:
        """Weighted total quality score"""
        return (
            self.freshness_score * 0.3 +
            self.completeness_score * 0.4 +
            self.latency_score * 0.2 +
            self.consistency_score * 0.1
        )

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    last_cleanup: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class RingBuffer:
    """High-performance ring buffer for time-series data"""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = np.empty(maxsize, dtype=object)
        self.head = 0
        self.size = 0
        self.lock = threading.RLock()
    
    def append(self, item: Any):
        """Add item to ring buffer"""
        with self.lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.maxsize
            if self.size < self.maxsize:
                self.size += 1
    
    def get_latest(self, count: int = 1) -> List[Any]:
        """Get latest N items"""
        with self.lock:
            if self.size == 0:
                return []
            
            count = min(count, self.size)
            items = []
            
            for i in range(count):
                idx = (self.head - 1 - i) % self.maxsize
                if idx < 0:
                    idx += self.maxsize
                items.append(self.buffer[idx])
            
            return items
    
    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        with self.lock:
            if self.size == 0:
                return []
            
            items = []
            start_idx = (self.head - self.size) % self.maxsize
            
            for i in range(self.size):
                idx = (start_idx + i) % self.maxsize
                items.append(self.buffer[idx])
            
            return items
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.head = 0
            self.size = 0

@dataclass
class PriceData:
    """Price data structure optimized for memory"""
    timestamp: float  # Unix timestamp for speed
    price: float
    volume: int
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp),
            'price': self.price,
            'volume': self.volume
        }

@dataclass
class BarData:
    """OHLCV bar data structure"""
    timestamp: float
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    trade_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'trade_count': self.trade_count
        }

@dataclass
class IndicatorData:
    """Technical indicator data"""
    timestamp: float
    symbol: str
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None

class SymbolCache:
    """Cache for individual symbol data"""
    
    def __init__(self, symbol: str, max_ticks: int = 10000, max_bars: int = 3600):
        self.symbol = symbol
        self.max_ticks = max_ticks
        self.max_bars = max_bars
        
        # Ring buffers for different data types
        self.ticks = RingBuffer(max_ticks)  # Trade ticks
        self.quotes = RingBuffer(max_ticks)  # Quote updates
        self.second_bars = RingBuffer(max_bars)  # Second-level bars
        self.minute_bars = RingBuffer(390)  # Minute bars (6.5 hours)
        
        # Latest values for instant access
        self.latest_price: Optional[float] = None
        self.latest_volume: Optional[int] = None
        self.latest_bid: Optional[float] = None
        self.latest_ask: Optional[float] = None
        self.latest_timestamp: Optional[float] = None
        
        # Indicators cache
        self.latest_indicators: Optional[IndicatorData] = None
        self.indicators_history = RingBuffer(100)  # Last 100 indicator updates
        
        # Performance metrics
        self.last_update = time.time()
        self.update_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_tick(self, price: float, volume: int, timestamp: Optional[float] = None):
        """Add trade tick"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            tick_data = PriceData(timestamp, price, volume)
            self.ticks.append(tick_data)
            
            # Update latest values
            self.latest_price = price
            self.latest_volume = volume
            self.latest_timestamp = timestamp
            self.last_update = time.time()
            self.update_count += 1
    
    def add_quote(self, bid: float, ask: float, bid_size: int, ask_size: int, 
                  timestamp: Optional[float] = None):
        """Add quote update"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            quote_data = {
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'bid_size': bid_size,
                'ask_size': ask_size
            }
            self.quotes.append(quote_data)
            
            # Update latest values
            self.latest_bid = bid
            self.latest_ask = ask
            self.latest_timestamp = timestamp
    
    def add_bar(self, bar_data: BarData, bar_type: str = 'second'):
        """Add OHLCV bar"""
        with self.lock:
            if bar_type == 'second':
                self.second_bars.append(bar_data)
            elif bar_type == 'minute':
                self.minute_bars.append(bar_data)
            
            # Update latest price from bar
            self.latest_price = bar_data.close
            self.latest_timestamp = bar_data.timestamp
    
    def add_indicators(self, indicators: IndicatorData):
        """Add technical indicators"""
        with self.lock:
            self.latest_indicators = indicators
            self.indicators_history.append(indicators)
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price (thread-safe)"""
        with self.lock:
            return self.latest_price
    
    def get_latest_quote(self) -> Optional[Dict]:
        """Get latest bid/ask"""
        with self.lock:
            if self.latest_bid is not None and self.latest_ask is not None:
                return {
                    'bid': self.latest_bid,
                    'ask': self.latest_ask,
                    'spread': self.latest_ask - self.latest_bid,
                    'mid': (self.latest_bid + self.latest_ask) / 2
                }
            return None
    
    def get_recent_ticks(self, count: int = 100) -> List[Dict]:
        """Get recent trade ticks"""
        with self.lock:
            ticks = self.ticks.get_latest(count)
            return [tick.to_dict() for tick in ticks if tick is not None]
    
    def get_recent_bars(self, count: int = 60, bar_type: str = 'second') -> List[Dict]:
        """Get recent bars"""
        with self.lock:
            if bar_type == 'second':
                bars = self.second_bars.get_latest(count)
            else:
                bars = self.minute_bars.get_latest(count)
            
            return [bar.to_dict() for bar in bars if bar is not None]
    
    def get_price_series(self, count: int = 512) -> np.ndarray:
        """Get price series as numpy array for Lag-Llama"""
        with self.lock:
            ticks = self.ticks.get_latest(count)
            if not ticks:
                return np.array([])
            
            prices = [tick.price for tick in ticks if tick is not None]
            return np.array(prices, dtype=np.float32)
    
    def get_volume_series(self, count: int = 512) -> np.ndarray:
        """Get volume series as numpy array"""
        with self.lock:
            ticks = self.ticks.get_latest(count)
            if not ticks:
                return np.array([])
            
            volumes = [tick.volume for tick in ticks if tick is not None]
            return np.array(volumes, dtype=np.int32)
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        # Rough estimation
        tick_size = 32  # PriceData size
        quote_size = 40  # Quote dict size
        bar_size = 64   # BarData size
        
        return (
            self.ticks.size * tick_size +
            self.quotes.size * quote_size +
            self.second_bars.size * bar_size +
            self.minute_bars.size * bar_size
        )

class TradingCache:
    """FAIL FAST Cache System - NO FALLBACKS - Data MUST be pre-loaded"""
    
    def __init__(self):
        # Symbol caches - MANDATORY data presence
        self.symbol_caches: Dict[str, SymbolCache] = {}
        self.stats = CacheStats()
        
        # FAIL FAST validation flags
        self.cache_warmed: bool = False
        self.mandatory_symbols: Set[str] = set()
        self.required_data_length: int = 512  # Minimum required for Lag-Llama
        
        # Market regime cache
        self.market_regime: Optional[Dict] = None
        self.market_regime_history = RingBuffer(100)
        
        # Strategy state cache
        self.strategy_signals: Dict[str, Any] = {}
        self.portfolio_state: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, Any] = {}
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache")
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start background cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    # ========== FAIL FAST METHODS - NO FALLBACKS ==========
    
    def set_mandatory_symbols(self, symbols: List[str]):
        """Set symbols that MUST have data before system can operate"""
        with self.lock:
            self.mandatory_symbols = set(symbols)
            logger.info(f"Set {len(symbols)} mandatory symbols for FAIL FAST validation")
    
    async def mandatory_cache_warm(self, database_client) -> bool:
        """MANDATORY cache warming - system CANNOT operate without this"""
        logger.info("Starting MANDATORY cache warming - FAIL FAST mode")
        
        if not self.mandatory_symbols:
            raise RuntimeError("No mandatory symbols set - cannot warm cache")
        
        success_count = 0
        failed_symbols = []
        
        for symbol in self.mandatory_symbols:
            try:
                # Get historical data from database
                bars = await database_client.get_realtime_bars(symbol, hours=2)
                
                if len(bars) < self.required_data_length:
                    failed_symbols.append(f"{symbol}: only {len(bars)} bars, need {self.required_data_length}")
                    continue
                
                # Populate cache with historical data
                cache = self.get_symbol_cache(symbol)
                for bar in bars:
                    bar_data = BarData(
                        timestamp=bar['time'].timestamp(),
                        symbol=symbol,
                        open=float(bar['open']),
                        high=float(bar['high']),
                        low=float(bar['low']),
                        close=float(bar['close']),
                        volume=int(bar['volume']),
                        vwap=float(bar.get('vwap', 0)),
                        trade_count=int(bar.get('trade_count', 0))
                    )
                    cache.add_bar(bar_data, 'second')
                
                # Validate cache has sufficient data
                price_series = cache.get_price_series(self.required_data_length)
                if len(price_series) < self.required_data_length:
                    failed_symbols.append(f"{symbol}: cache validation failed")
                    continue
                
                success_count += 1
                logger.debug(f"Successfully warmed cache for {symbol}: {len(price_series)} data points")
                
            except Exception as e:
                failed_symbols.append(f"{symbol}: {str(e)}")
        
        if failed_symbols:
            error_msg = f"FAIL FAST: Cache warming failed for symbols: {failed_symbols}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.cache_warmed = True
        logger.info(f"MANDATORY cache warming completed successfully for {success_count} symbols")
        return True
    
    def validate_cache_readiness(self) -> bool:
        """STRICT validation - system MUST have all required data"""
        if not self.cache_warmed:
            raise RuntimeError("FAIL FAST: Cache not warmed - system cannot operate")
        
        missing_symbols = []
        insufficient_data = []
        
        for symbol in self.mandatory_symbols:
            if symbol not in self.symbol_caches:
                missing_symbols.append(symbol)
                continue
            
            cache = self.symbol_caches[symbol]
            price_series = cache.get_price_series(self.required_data_length)
            
            if len(price_series) < self.required_data_length:
                insufficient_data.append(f"{symbol}: {len(price_series)}/{self.required_data_length}")
        
        if missing_symbols:
            raise RuntimeError(f"FAIL FAST: Missing symbol caches: {missing_symbols}")
        
        if insufficient_data:
            raise RuntimeError(f"FAIL FAST: Insufficient data: {insufficient_data}")
        
        logger.info("Cache readiness validation PASSED - system ready for operation")
        return True
    
    def _cleanup_old_data(self):
        """Clean up old data to manage memory"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        with self.lock:
            # Clean up inactive symbols
            inactive_symbols = []
            for symbol, cache in self.symbol_caches.items():
                if cache.last_update < cutoff_time:
                    inactive_symbols.append(symbol)
            
            for symbol in inactive_symbols:
                del self.symbol_caches[symbol]
                self.stats.evictions += 1
            
            self.last_cleanup = current_time
            self.stats.last_cleanup = datetime.fromtimestamp(current_time)
            
            if inactive_symbols:
                logger.info(f"Cleaned up {len(inactive_symbols)} inactive symbol caches")
    
    def get_symbol_cache(self, symbol: str) -> SymbolCache:
        """Get or create symbol cache"""
        with self.lock:
            if symbol not in self.symbol_caches:
                self.symbol_caches[symbol] = SymbolCache(symbol)
                logger.debug(f"Created cache for symbol {symbol}")
            
            return self.symbol_caches[symbol]
    
    # ========== PRICE DATA METHODS ==========
    
    def add_trade(self, symbol: str, price: float, volume: int, 
                  timestamp: Optional[float] = None):
        """Add trade data"""
        cache = self.get_symbol_cache(symbol)
        cache.add_tick(price, volume, timestamp)
        self.stats.hits += 1
    
    def add_quote(self, symbol: str, bid: float, ask: float, 
                  bid_size: int, ask_size: int, timestamp: Optional[float] = None):
        """Add quote data"""
        cache = self.get_symbol_cache(symbol)
        cache.add_quote(bid, ask, bid_size, ask_size, timestamp)
    
    def add_bar(self, symbol: str, timestamp: float, open_price: float, 
                high: float, low: float, close: float, volume: int,
                vwap: float = 0.0, trade_count: int = 0, bar_type: str = 'second'):
        """Add OHLCV bar"""
        bar_data = BarData(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            trade_count=trade_count
        )
        
        cache = self.get_symbol_cache(symbol)
        cache.add_bar(bar_data, bar_type)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        if symbol in self.symbol_caches:
            self.stats.hits += 1
            return self.symbol_caches[symbol].get_latest_price()
        
        self.stats.misses += 1
        return None
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for symbol"""
        if symbol in self.symbol_caches:
            self.stats.hits += 1
            return self.symbol_caches[symbol].get_latest_quote()
        
        self.stats.misses += 1
        return None
    
    def get_price_series(self, symbol: str, count: int = 512) -> np.ndarray:
        """Get price series for Lag-Llama - FAIL FAST, NO FALLBACKS"""
        # STRICT: Cache must be warmed before any operations
        if not self.cache_warmed:
            raise RuntimeError(f"FAIL FAST: Cache not warmed - cannot get price series for {symbol}")
        
        # STRICT: Symbol must be in mandatory symbols
        if symbol not in self.mandatory_symbols:
            raise RuntimeError(f"FAIL FAST: Symbol {symbol} not in mandatory symbols list")
        
        # STRICT: Symbol cache must exist
        if symbol not in self.symbol_caches:
            raise RuntimeError(f"FAIL FAST: No cache found for mandatory symbol {symbol}")
        
        # Get price series
        price_series = self.symbol_caches[symbol].get_price_series(count)
        
        # STRICT: Must have sufficient data
        if len(price_series) < count:
            raise RuntimeError(
                f"FAIL FAST: Insufficient data for {symbol}: "
                f"got {len(price_series)}, required {count}"
            )
        
        self.stats.hits += 1
        return price_series
    
    def get_volume_series(self, symbol: str, count: int = 512) -> np.ndarray:
        """Get volume series"""
        if symbol in self.symbol_caches:
            return self.symbol_caches[symbol].get_volume_series(count)
        
        return np.array([])
    
    def get_recent_bars(self, symbol: str, count: int = 60, 
                       bar_type: str = 'second') -> List[Dict]:
        """Get recent bars for symbol"""
        if symbol in self.symbol_caches:
            self.stats.hits += 1
            return self.symbol_caches[symbol].get_recent_bars(count, bar_type)
        
        self.stats.misses += 1
        return []
    
    # ========== INDICATOR METHODS ==========
    
    def add_indicators(self, symbol: str, timestamp: float, **indicators):
        """Add technical indicators"""
        indicator_data = IndicatorData(
            timestamp=timestamp,
            symbol=symbol,
            **indicators
        )
        
        cache = self.get_symbol_cache(symbol)
        cache.add_indicators(indicator_data)
    
    def get_latest_indicators(self, symbol: str) -> Optional[IndicatorData]:
        """Get latest indicators for symbol"""
        if symbol in self.symbol_caches:
            self.stats.hits += 1
            return self.symbol_caches[symbol].latest_indicators
        
        self.stats.misses += 1
        return None
    
    # ========== MARKET REGIME METHODS ==========
    
    def update_market_regime(self, regime_data: Dict[str, Any]):
        """Update market regime data"""
        with self.lock:
            self.market_regime = regime_data
            self.market_regime_history.append(regime_data)
    
    def get_market_regime(self) -> Optional[Dict]:
        """Get current market regime"""
        with self.lock:
            return self.market_regime
    
    def get_market_regime_history(self, count: int = 10) -> List[Dict]:
        """Get market regime history"""
        with self.lock:
            return self.market_regime_history.get_latest(count)
    
    # ========== STRATEGY STATE METHODS ==========
    
    def update_strategy_signal(self, strategy: str, symbol: str, signal_data: Dict):
        """Update strategy signal"""
        with self.lock:
            key = f"{strategy}_{symbol}"
            self.strategy_signals[key] = {
                'timestamp': time.time(),
                'strategy': strategy,
                'symbol': symbol,
                **signal_data
            }
    
    def get_strategy_signals(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy signals"""
        with self.lock:
            if strategy:
                return {k: v for k, v in self.strategy_signals.items() 
                       if v.get('strategy') == strategy}
            return self.strategy_signals.copy()
    
    def update_portfolio_state(self, state_data: Dict[str, Any]):
        """Update portfolio state"""
        with self.lock:
            self.portfolio_state.update(state_data)
            self.portfolio_state['last_update'] = time.time()
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get portfolio state"""
        with self.lock:
            return self.portfolio_state.copy()
    
    def update_risk_metrics(self, metrics: Dict[str, Any]):
        """Update risk metrics"""
        with self.lock:
            self.risk_metrics.update(metrics)
            self.risk_metrics['last_update'] = time.time()
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        with self.lock:
            return self.risk_metrics.copy()
    
    # ========== BATCH OPERATIONS ==========
    
    def get_multi_symbol_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get data for multiple symbols efficiently"""
        result = {}
        
        for symbol in symbols:
            if symbol in self.symbol_caches:
                cache = self.symbol_caches[symbol]
                result[symbol] = {
                    'latest_price': cache.get_latest_price(),
                    'latest_quote': cache.get_latest_quote(),
                    'latest_indicators': cache.latest_indicators,
                    'last_update': cache.last_update
                }
                self.stats.hits += 1
            else:
                self.stats.misses += 1
        
        return result
    
    def get_lag_llama_context(self, symbols: List[str], 
                             context_length: int = 512) -> Dict[str, np.ndarray]:
        """Get price context for multiple symbols for Lag-Llama"""
        result = {}
        
        for symbol in symbols:
            price_series = self.get_price_series(symbol, context_length)
            if len(price_series) > 0:
                result[symbol] = price_series
        
        return result
    
    # ========== PERFORMANCE METHODS ==========
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_memory = sum(
                cache.get_memory_usage() 
                for cache in self.symbol_caches.values()
            )
            
            self.stats.memory_usage_bytes = total_memory
            
            return {
                'hit_rate': self.stats.hit_rate,
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'memory_usage_mb': total_memory / (1024 * 1024),
                'symbol_count': len(self.symbol_caches),
                'last_cleanup': self.stats.last_cleanup,
                'active_symbols': list(self.symbol_caches.keys())
            }
    
    def get_symbol_stats(self, symbol: str) -> Optional[Dict]:
        """Get statistics for specific symbol"""
        if symbol in self.symbol_caches:
            cache = self.symbol_caches[symbol]
            return {
                'symbol': symbol,
                'tick_count': cache.ticks.size,
                'quote_count': cache.quotes.size,
                'second_bar_count': cache.second_bars.size,
                'minute_bar_count': cache.minute_bars.size,
                'latest_price': cache.latest_price,
                'latest_timestamp': cache.latest_timestamp,
                'update_count': cache.update_count,
                'memory_usage_bytes': cache.get_memory_usage()
            }
        return None
    
    def clear_symbol(self, symbol: str):
        """Clear cache for specific symbol"""
        with self._lock:
            if symbol in self.symbol_caches:
                del self.symbol_caches[symbol]
    
    def clear_all(self):
        """Clear all cached data"""
        with self._lock:
            self.symbol_caches.clear()
            self.strategy_signals.clear()
            self.portfolio_state.clear()
            self.stats = CacheStats()
    
    async def cleanup(self):
        """Cleanup cache resources"""
        self.clear_all()

# Global cache instance
trading_cache = None

def get_trading_cache() -> TradingCache:
    """Get or create the global trading cache"""
    global trading_cache
    if trading_cache is None:
        trading_cache = TradingCache()
        logger.info("Trading cache initialized")
    return trading_cache
