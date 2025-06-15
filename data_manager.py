"""
Ultra-Low Latency FAIL FAST Data Manager
Streamlined for speed with minimal overhead
"""

import asyncio
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
import numpy as np
from dataclasses import dataclass

from cache import get_trading_cache, TradingCache
from database import get_database_client, TimescaleDBClient
from settings import config

logger = logging.getLogger(__name__)

class DataManagerError(Exception):
    """Base exception for data manager errors"""
    pass

class SystemNotReadyError(DataManagerError):
    """Raised when system is not ready for operations"""
    pass

@dataclass
class DataManagerConfig:
    """Minimal data manager configuration"""
    cache_enabled: bool = True
    database_enabled: bool = True
    persistence_interval: int = 60
    batch_size: int = 1000
    
class DataManager:
    """Ultra-Low Latency FAIL FAST Data Manager"""
    
    def __init__(self, config: DataManagerConfig = None):
        self.config = config or DataManagerConfig()
        self.cache: Optional[TradingCache] = None
        self.database: Optional[TimescaleDBClient] = None
        
        # FAIL FAST state tracking
        self.system_ready: bool = False
        self.cache_warmed: bool = False
        self.mandatory_symbols: List[str] = []
        
        # Minimal persistence tracking
        self.pending_bars: List[Dict] = []
        self.last_persistence = time.time()
        
        # Simplified metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'database_writes': 0
        }
        
        # Background tasks
        self._persistence_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self, mandatory_symbols: List[str] = None):
        """Initialize data manager with FAIL FAST cache warming"""
        try:
            logger.info("Initializing FAIL FAST data manager...")
            
            # Initialize cache
            if self.config.cache_enabled:
                self.cache = get_trading_cache()
                logger.info("Cache initialized")
            else:
                raise SystemNotReadyError("Cache is mandatory for FAIL FAST system")
            
            # Initialize database
            if self.config.database_enabled:
                self.database = get_database_client()
                await self.database.initialize()
                logger.info("Database initialized")
            else:
                raise SystemNotReadyError("Database is mandatory for FAIL FAST system")
            
            # MANDATORY: Set symbols and warm cache
            if mandatory_symbols:
                self.mandatory_symbols = mandatory_symbols
                await self._mandatory_cache_warming()
            else:
                logger.warning("No mandatory symbols provided - system may fail during operation")
            
            # Start background persistence task
            self._running = True
            self._persistence_task = asyncio.create_task(self._persistence_worker())
            
            self.system_ready = True
            logger.info("FAIL FAST data manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAIL FAST data manager: {e}")
            raise
    
    async def _mandatory_cache_warming(self):
        """MANDATORY cache warming - streamlined for speed"""
        if not self.cache or not self.database:
            raise SystemNotReadyError("Cache and database must be initialized")
        
        logger.info(f"Starting cache warming for {len(self.mandatory_symbols)} symbols")
        start_time = time.time()
        
        try:
            # Set mandatory symbols in cache
            self.cache.set_mandatory_symbols(self.mandatory_symbols)
            
            # Perform mandatory cache warming
            await self.cache.mandatory_cache_warm(self.database)
            
            # Validate cache readiness
            self.cache.validate_cache_readiness()
            
            self.cache_warmed = True
            warming_time = (time.time() - start_time) * 1000
            
            logger.info(f"Cache warming completed in {warming_time:.0f}ms")
            
        except Exception as e:
            raise SystemNotReadyError(f"FAIL FAST: Cache warming failed: {e}")
    
    async def _persistence_worker(self):
        """Simplified background worker for persisting data"""
        while self._running:
            try:
                await asyncio.sleep(self.config.persistence_interval)
                await self._persist_pending_data()
            except Exception as e:
                logger.error(f"Persistence worker error: {e}")
    
    async def _persist_pending_data(self):
        """Persist pending data - streamlined"""
        if not self.database or not self.config.database_enabled:
            return
        
        try:
            # Persist bars only (most critical for Lag-Llama)
            if self.pending_bars:
                await self.database.insert_realtime_bars(self.pending_bars[:self.config.batch_size])
                self.pending_bars = self.pending_bars[self.config.batch_size:]
                self.metrics['database_writes'] += 1
            
            self.last_persistence = time.time()
            
        except Exception as e:
            logger.error(f"Error persisting data: {e}")
    
    # ========== PRICE DATA METHODS ==========
    
    async def add_trade(self, symbol: str, price: float, volume: int, 
                       timestamp: Optional[datetime] = None, persist: bool = True):
        """Add trade data to cache and optionally persist"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to cache
        if self.cache:
            self.cache.add_trade(symbol, price, volume, timestamp.timestamp())
        
        # Queue for persistence
        if persist and self.database:
            trade_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'price': price,
                'volume': volume
            }
            # Note: This would be used for trade execution records, not raw ticks
    
    async def add_quote(self, symbol: str, bid: float, ask: float, 
                       bid_size: int, ask_size: int, timestamp: Optional[datetime] = None):
        """Add quote data to cache"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.cache:
            self.cache.add_quote(symbol, bid, ask, bid_size, ask_size, timestamp.timestamp())
    
    async def add_bar(self, symbol: str, timestamp: datetime, open_price: float,
                     high: float, low: float, close: float, volume: int,
                     vwap: float = 0.0, trade_count: int = 0, 
                     bar_type: str = 'second', persist: bool = True):
        """Add OHLCV bar to cache and optionally persist"""
        
        # Add to cache
        if self.cache:
            self.cache.add_bar(
                symbol, timestamp.timestamp(), open_price, high, low, 
                close, volume, vwap, trade_count, bar_type
            )
        
        # Queue for persistence (only second-level bars)
        if persist and self.database and bar_type == 'second':
            bar_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': vwap,
                'trade_count': trade_count
            }
            self.pending_bars.append(bar_data)
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price - optimized for speed"""
        if self.cache:
            price = self.cache.get_latest_price(symbol)
            if price is not None:
                self.metrics['cache_hits'] += 1
                return price
            self.metrics['cache_misses'] += 1
        return None
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote - optimized for speed"""
        if self.cache:
            quote = self.cache.get_latest_quote(symbol)
            if quote is not None:
                self.metrics['cache_hits'] += 1
                return quote
            self.metrics['cache_misses'] += 1
        return None
    
    async def get_price_series(self, symbol: str, count: int = 512) -> np.ndarray:
        """Get price series for Lag-Llama - FASTEST PATH"""
        # STRICT: System must be ready
        if not self.system_ready:
            raise SystemNotReadyError("FAIL FAST: System not ready")
        
        # STRICT: Cache must be available and warmed
        if not self.cache or not self.cache_warmed:
            raise SystemNotReadyError("FAIL FAST: Cache not warmed")
        
        # Get data from cache - FASTEST PATH
        try:
            series = self.cache.get_price_series(symbol, count)
            self.metrics['cache_hits'] += 1
            return series
        except Exception as e:
            self.metrics['cache_misses'] += 1
            raise SystemNotReadyError(f"FAIL FAST: Cache failed for {symbol}: {e}")
    
    # ========== INDICATOR METHODS ==========
    
    async def add_indicators(self, symbol: str, timestamp: datetime, 
                           persist: bool = True, **indicators):
        """Add technical indicators"""
        
        # Add to cache
        if self.cache:
            self.cache.add_indicators(symbol, timestamp.timestamp(), **indicators)
        
        # Queue for persistence
        if persist and self.database:
            indicator_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                **indicators
            }
            self.pending_indicators.append(indicator_data)
    
    async def get_latest_indicators(self, symbol: str) -> Optional[Dict]:
        """Get latest indicators for symbol"""
        if self.cache:
            indicators = self.cache.get_latest_indicators(symbol)
            if indicators is not None:
                self.metrics['cache_hits'] += 1
                return {
                    'timestamp': datetime.fromtimestamp(indicators.timestamp),
                    'symbol': indicators.symbol,
                    'rsi': indicators.rsi,
                    'macd_line': indicators.macd_line,
                    'macd_signal': indicators.macd_signal,
                    'macd_histogram': indicators.macd_histogram,
                    'sma_20': indicators.sma_20,
                    'sma_50': indicators.sma_50,
                    'sma_200': indicators.sma_200,
                    'ema_12': indicators.ema_12,
                    'ema_26': indicators.ema_26,
                    'bollinger_upper': indicators.bollinger_upper,
                    'bollinger_lower': indicators.bollinger_lower,
                    'bollinger_middle': indicators.bollinger_middle
                }
            self.metrics['cache_misses'] += 1
        
        # Fallback to database
        if self.database:
            indicators = await self.database.get_latest_indicators(symbol)
            if indicators:
                self.metrics['database_reads'] += 1
                return indicators
        
        return None
    
    # ========== MARKET REGIME METHODS ==========
    
    async def update_market_regime(self, regime_data: Dict[str, Any], persist: bool = True):
        """Update market regime data"""
        
        # Add timestamp if not present
        if 'timestamp' not in regime_data:
            regime_data['timestamp'] = datetime.now()
        
        # Update cache
        if self.cache:
            self.cache.update_market_regime(regime_data)
        
        # Persist to database
        if persist and self.database:
            await self.database.insert_market_regime(regime_data)
    
    async def get_market_regime(self) -> Optional[Dict]:
        """Get current market regime"""
        if self.cache:
            regime = self.cache.get_market_regime()
            if regime is not None:
                self.metrics['cache_hits'] += 1
                return regime
            self.metrics['cache_misses'] += 1
        
        # Fallback to database
        if self.database:
            regime = await self.database.get_latest_market_regime()
            if regime:
                self.metrics['database_reads'] += 1
                return regime
        
        return None
    
    # ========== STRATEGY STATE METHODS ==========
    
    async def update_strategy_signal(self, strategy: str, symbol: str, signal_data: Dict):
        """Update strategy signal"""
        if self.cache:
            self.cache.update_strategy_signal(strategy, symbol, signal_data)
    
    async def get_strategy_signals(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy signals"""
        if self.cache:
            return self.cache.get_strategy_signals(strategy)
        return {}
    
    async def update_portfolio_state(self, state_data: Dict[str, Any]):
        """Update portfolio state"""
        if self.cache:
            self.cache.update_portfolio_state(state_data)
    
    async def get_portfolio_state(self) -> Dict[str, Any]:
        """Get portfolio state"""
        if self.cache:
            return self.cache.get_portfolio_state()
        return {}
    
    async def update_risk_metrics(self, metrics: Dict[str, Any]):
        """Update risk metrics"""
        if self.cache:
            self.cache.update_risk_metrics(metrics)
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        if self.cache:
            return self.cache.get_risk_metrics()
        return {}
    
    # ========== BATCH OPERATIONS ==========
    
    async def get_multi_symbol_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get data for multiple symbols - optimized"""
        if self.cache:
            return self.cache.get_multi_symbol_data(symbols)
        return {}
    
    async def get_lag_llama_context(self, symbols: List[str],
                                   context_length: int = 512) -> Dict[str, np.ndarray]:
        """Get price context for multiple symbols - OPTIMIZED for Lag-Llama"""
        if self.cache:
            return self.cache.get_lag_llama_context(symbols, context_length)
        return {}
    
    # ========== TRADE EXECUTION METHODS ==========
    
    async def record_trade_execution(self, trade_data: Dict[str, Any]):
        """Record trade execution"""
        if self.database:
            await self.database.insert_trade_execution(trade_data)
            self.metrics['database_writes'] += 1
    
    async def get_trade_executions(self, symbol: Optional[str] = None,
                                  strategy: Optional[str] = None,
                                  days: int = 7) -> List[Dict]:
        """Get trade execution history"""
        if self.database:
            self.metrics['database_reads'] += 1
            return await self.database.get_trade_executions(symbol, strategy, days)
        return []
    
    # ========== GAP CANDIDATES METHODS ==========
    
    async def update_gap_candidates(self, candidates: List[Dict[str, Any]]):
        """Update gap candidates"""
        if self.database:
            await self.database.insert_gap_candidates(candidates)
            self.metrics['database_writes'] += 1
    
    async def get_gap_candidates(self, date: Optional[date] = None,
                                min_gap_percent: float = 2.0) -> List[Dict]:
        """Get gap candidates"""
        if self.database:
            self.metrics['database_reads'] += 1
            return await self.database.get_gap_candidates(date, min_gap_percent)
        return []
    
    # ========== DAILY OPERATIONS ==========
    
    async def end_of_day_processing(self):
        """End of day data processing"""
        logger.info("Starting end-of-day processing...")
        
        try:
            # Force persistence of all pending data
            await self._persist_pending_data()
            
            # Generate daily summaries from cache
            if self.cache and self.database:
                daily_bars = []
                current_date = datetime.now().date()
                
                for symbol in self.cache.symbol_caches.keys():
                    # Get all minute bars for the day
                    minute_bars = self.cache.get_recent_bars(symbol, 390, 'minute')
                    
                    if minute_bars:
                        # Calculate daily summary
                        opens = [bar['open'] for bar in minute_bars]
                        highs = [bar['high'] for bar in minute_bars]
                        lows = [bar['low'] for bar in minute_bars]
                        closes = [bar['close'] for bar in minute_bars]
                        volumes = [bar['volume'] for bar in minute_bars]
                        
                        daily_bar = {
                            'date': current_date,
                            'symbol': symbol,
                            'open': opens[0] if opens else 0,
                            'high': max(highs) if highs else 0,
                            'low': min(lows) if lows else 0,
                            'close': closes[-1] if closes else 0,
                            'volume': sum(volumes),
                            'vwap': sum(bar.get('vwap', 0) * bar['volume'] for bar in minute_bars) / sum(volumes) if volumes else 0
                        }
                        daily_bars.append(daily_bar)
                
                # Insert daily summaries
                if daily_bars:
                    await self.database.insert_daily_bars(daily_bars)
                    logger.info(f"Inserted {len(daily_bars)} daily bar summaries")
            
            # Clear old cache data
            if self.cache:
                # Keep only recent data in cache
                cutoff_time = time.time() - 3600  # 1 hour
                symbols_to_clear = []
                
                for symbol, cache in self.cache.symbol_caches.items():
                    if cache.last_update < cutoff_time:
                        symbols_to_clear.append(symbol)
                
                for symbol in symbols_to_clear:
                    self.cache.clear_symbol(symbol)
                
                logger.info(f"Cleared {len(symbols_to_clear)} inactive symbols from cache")
            
            logger.info("End-of-day processing completed")
            
        except Exception as e:
            logger.error(f"Error in end-of-day processing: {e}")
    
    # ========== PERFORMANCE METHODS ==========
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get data manager performance statistics"""
        stats = {
            'data_manager': self.metrics.copy(),
            'cache': None,
            'database': None,
            'pending_data': {
                'bars': len(self.pending_bars),
                'trades': len(self.pending_trades),
                'indicators': len(self.pending_indicators)
            }
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_cache_stats()
        
        if self.database:
            stats['database'] = await self.database.get_database_stats()
        
        return stats
    
    async def cleanup(self):
        """Cleanup data manager resources"""
        logger.info("Cleaning up data manager...")
        
        # Stop background tasks
        self._running = False
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
        
        # Final persistence
        await self._persist_pending_data()
        
        # Cleanup components
        if self.cache:
            await self.cache.cleanup()
        
        if self.database:
            await self.database.cleanup()
        
        logger.info("Data manager cleanup completed")

# Global data manager instance
data_manager = None

def get_data_manager() -> DataManager:
    """Get or create the global data manager"""
    global data_manager
    if data_manager is None:
        data_manager = DataManager()
    return data_manager

async def initialize_data_manager():
    """Initialize the global data manager"""
    dm = get_data_manager()
    await dm.initialize()
    return dm
