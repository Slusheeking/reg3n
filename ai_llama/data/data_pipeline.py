"""
Real-time Data Pipeline

Orchestrates data flow between market data sources, AI models, and trading strategies.
Features:
- Unified data interface for multiple sources
- Real-time data normalization and validation
- Feature extraction pipeline integration
- AI model data preparation
- Latency optimization and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

from .polygon_client import PolygonClient, Quote, Trade, Aggregate
from .alpaca_client import AlpacaClient
from ..features.fast_features import extract_fast_features
from ..features.gap_features import calculate_gap_features
from ..features.orb_features import calculate_orb_features
from ..features.vol_features import calculate_volume_features

@dataclass
class MarketData:
    """Unified market data structure"""
    symbol: str
    timestamp: int
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    spread: Optional[float] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MarketDataBar:
    """OHLCV bar data"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    transactions: Optional[int] = None
    source: str = "polygon"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FeatureSet:
    """Extracted features for a symbol"""
    symbol: str
    timestamp: int
    features: Dict[str, float]
    price_series: np.ndarray
    volume_series: np.ndarray
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['price_series'] = self.price_series.tolist()
        data['volume_series'] = self.volume_series.tolist()
        return data

class DataBuffer:
    """High-performance circular buffer for market data"""
    
    def __init__(self, symbol: str, max_size: int = 1000):
        self.symbol = symbol
        self.max_size = max_size
        self.quotes = deque(maxlen=max_size)
        self.trades = deque(maxlen=max_size)
        self.bars = deque(maxlen=max_size)
        self.last_update = 0
        
    def add_quote(self, quote: Quote):
        """Add quote to buffer"""
        self.quotes.append(quote)
        self.last_update = time.time()
    
    def add_trade(self, trade: Trade):
        """Add trade to buffer"""
        self.trades.append(trade)
        self.last_update = time.time()
    
    def add_bar(self, bar: Union[Aggregate, MarketDataBar]):
        """Add bar to buffer"""
        self.bars.append(bar)
        self.last_update = time.time()
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price from trades or quotes"""
        if self.trades:
            return self.trades[-1].price
        elif self.quotes:
            return (self.quotes[-1].bid + self.quotes[-1].ask) / 2
        elif self.bars:
            return self.bars[-1].close
        return None
    
    def get_price_series(self, length: int = 100) -> np.ndarray:
        """Get price series from bars"""
        if not self.bars:
            return np.array([])
        
        prices = [bar.close for bar in list(self.bars)[-length:]]
        return np.array(prices)
    
    def get_volume_series(self, length: int = 100) -> np.ndarray:
        """Get volume series from bars"""
        if not self.bars:
            return np.array([])
        
        volumes = [bar.volume for bar in list(self.bars)[-length:]]
        return np.array(volumes)
    
    def get_ohlcv_df(self, length: int = 100) -> pd.DataFrame:
        """Get OHLCV data as DataFrame"""
        if not self.bars:
            return pd.DataFrame()
        
        bars_list = list(self.bars)[-length:]
        data = []
        
        for bar in bars_list:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': getattr(bar, 'vwap', None)
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
        
        return df

class DataPipeline:
    """
    Real-time data pipeline orchestrator
    
    Features:
    - Multi-source data aggregation
    - Real-time feature extraction
    - Latency monitoring
    - Data validation and cleanup
    - Symbol universe management
    """
    
    def __init__(self, 
                 polygon_api_key: str = None,
                 alpaca_api_key: str = None,
                 alpaca_secret_key: str = None):
        
        # Initialize clients
        self.polygon_client = PolygonClient(polygon_api_key) if polygon_api_key else None
        self.alpaca_client = AlpacaClient(alpaca_api_key, alpaca_secret_key) if alpaca_api_key and alpaca_secret_key else None
        
        # Data buffers for each symbol
        self.buffers: Dict[str, DataBuffer] = {}
        
        # Callbacks for data events
        self.callbacks = {
            'quote': [],
            'trade': [],
            'bar': [],
            'features': [],
            'error': []
        }
        
        # Performance tracking
        self.stats = {
            'messages_processed': 0,
            'avg_latency_ms': 0.0,
            'features_extracted': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Feature extraction
        self.feature_extraction_enabled = True
        self.feature_history_length = 200
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self, symbols: List[str] = None):
        """Start the data pipeline"""
        self.stats['start_time'] = time.time()
        
        # Default symbol universe
        if symbols is None:
            symbols = [
                # Major indices
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
                # Major stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                # Sector ETFs
                'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY'
            ]
        
        # Initialize buffers
        for symbol in symbols:
            self.buffers[symbol] = DataBuffer(symbol)
        
        # Start data sources
        tasks = []
        
        if self.polygon_client:
            tasks.append(self._start_polygon_feeds(symbols))
        
        if self.alpaca_client:
            tasks.append(self._start_alpaca_feeds())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        self.running = True
        self.logger.info(f"Data pipeline started with {len(symbols)} symbols")
    
    async def stop(self):
        """Stop the data pipeline"""
        self.running = False
        
        if self.polygon_client:
            await self.polygon_client.stop()
        
        if self.alpaca_client:
            await self.alpaca_client.stop()
        
        self.executor.shutdown(wait=True)
        self.logger.info("Data pipeline stopped")
    
    async def _start_polygon_feeds(self, symbols: List[str]):
        """Start Polygon data feeds"""
        if not self.polygon_client:
            return
        
        # Set up callbacks
        self.polygon_client.add_quote_callback(self._handle_quote)
        self.polygon_client.add_trade_callback(self._handle_trade)
        self.polygon_client.websocket.add_callback('aggregate', self._handle_aggregate)
        
        # Start client and subscribe
        await self.polygon_client.start()
        await self.polygon_client.subscribe_to_symbols(symbols)
        
        # Get historical data for feature extraction
        await self._load_historical_data(symbols)
        
        self.logger.info(f"Polygon feeds started for {len(symbols)} symbols")
    
    async def _start_alpaca_feeds(self):
        """Start Alpaca data feeds"""
        if not self.alpaca_client:
            return
        
        # Set up callbacks for account updates
        self.alpaca_client.add_trade_update_callback(self._handle_trade_update)
        self.alpaca_client.add_account_update_callback(self._handle_account_update)
        
        await self.alpaca_client.start()
        self.logger.info("Alpaca feeds started")
    
    async def _load_historical_data(self, symbols: List[str]):
        """Load historical data for feature extraction"""
        self.logger.info("Loading historical data for feature extraction...")
        
        for symbol in symbols:
            try:
                # Get 2 days of minute data
                df = await self.polygon_client.get_historical_data(symbol, days=2, timespan="minute")
                
                if not df.empty:
                    # Convert to bars and add to buffer
                    for _, row in df.iterrows():
                        bar = MarketDataBar(
                            symbol=symbol,
                            timestamp=int(row.name.timestamp() * 1000),
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            vwap=row.get('vwap'),
                            source="polygon_historical"
                        )
                        self.buffers[symbol].add_bar(bar)
                    
                    self.logger.debug(f"Loaded {len(df)} historical bars for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to load historical data for {symbol}: {e}")
        
        self.logger.info("Historical data loading completed")
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for data events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def _handle_quote(self, quote: Quote):
        """Handle incoming quote"""
        start_time = time.time()
        
        try:
            # Add to buffer
            if quote.symbol in self.buffers:
                self.buffers[quote.symbol].add_quote(quote)
            
            # Trigger callbacks
            for callback in self.callbacks['quote']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(quote)
                    else:
                        callback(quote)
                except Exception as e:
                    self.logger.error(f"Quote callback error: {e}")
            
            # Update stats
            self._update_latency_stats(start_time)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Quote handling error: {e}")
    
    async def _handle_trade(self, trade: Trade):
        """Handle incoming trade"""
        start_time = time.time()
        
        try:
            # Add to buffer
            if trade.symbol in self.buffers:
                self.buffers[trade.symbol].add_trade(trade)
            
            # Trigger callbacks
            for callback in self.callbacks['trade']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trade)
                    else:
                        callback(trade)
                except Exception as e:
                    self.logger.error(f"Trade callback error: {e}")
            
            # Update stats
            self._update_latency_stats(start_time)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Trade handling error: {e}")
    
    async def _handle_aggregate(self, aggregate: Aggregate):
        """Handle incoming aggregate bar"""
        start_time = time.time()
        
        try:
            # Convert to MarketDataBar
            bar = MarketDataBar(
                symbol=aggregate.symbol,
                timestamp=aggregate.timestamp,
                open=aggregate.open,
                high=aggregate.high,
                low=aggregate.low,
                close=aggregate.close,
                volume=aggregate.volume,
                vwap=aggregate.vwap,
                transactions=aggregate.transactions,
                source="polygon_realtime"
            )
            
            # Add to buffer
            if bar.symbol in self.buffers:
                self.buffers[bar.symbol].add_bar(bar)
            
            # Extract features if enabled
            if self.feature_extraction_enabled:
                await self._extract_features(bar.symbol)
            
            # Trigger callbacks
            for callback in self.callbacks['bar']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(bar)
                    else:
                        callback(bar)
                except Exception as e:
                    self.logger.error(f"Bar callback error: {e}")
            
            # Update stats
            self._update_latency_stats(start_time)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Aggregate handling error: {e}")
    
    async def _handle_trade_update(self, update):
        """Handle Alpaca trade update"""
        self.logger.debug(f"Trade update: {update}")
    
    async def _handle_account_update(self, update):
        """Handle Alpaca account update"""
        self.logger.debug(f"Account update: {update}")
    
    async def _extract_features(self, symbol: str):
        """Extract features for a symbol"""
        try:
            buffer = self.buffers.get(symbol)
            if not buffer:
                return
            
            # Get price and volume series
            price_series = buffer.get_price_series(self.feature_history_length)
            volume_series = buffer.get_volume_series(self.feature_history_length)
            
            if len(price_series) < 20:  # Need minimum data
                return
            
            # Get OHLCV data
            df = buffer.get_ohlcv_df(self.feature_history_length)
            if df.empty:
                return
            
            # Extract features in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                self.executor,
                self._compute_features,
                symbol,
                price_series,
                volume_series,
                df
            )
            
            if features:
                # Create feature set
                feature_set = FeatureSet(
                    symbol=symbol,
                    timestamp=int(time.time() * 1000),
                    features=features,
                    price_series=price_series[-100:],  # Keep last 100 points
                    volume_series=volume_series[-100:]
                )
                
                # Trigger feature callbacks
                for callback in self.callbacks['features']:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(feature_set)
                        else:
                            callback(feature_set)
                    except Exception as e:
                        self.logger.error(f"Feature callback error: {e}")
                
                self.stats['features_extracted'] += 1
        
        except Exception as e:
            self.logger.error(f"Feature extraction error for {symbol}: {e}")
    
    def _compute_features(self, 
                         symbol: str,
                         price_series: np.ndarray, 
                         volume_series: np.ndarray,
                         df: pd.DataFrame) -> Dict[str, float]:
        """Compute all features (runs in thread pool)"""
        try:
            features = {}
            
            # Fast features
            fast_features = extract_fast_features(price_series, volume_series)
            features.update(fast_features)
            
            # Strategy-specific features
            if len(df) >= 50:
                # Gap features
                gap_features = calculate_gap_features(
                    df['close'].values,
                    df['high'].values,
                    df['low'].values,
                    df['volume'].values
                )
                features.update(gap_features)
                
                # ORB features
                orb_features = calculate_orb_features(
                    df['close'].values,
                    df['high'].values,
                    df['low'].values,
                    df['volume'].values
                )
                features.update(orb_features)
                
                # Volume features
                vol_features = calculate_volume_features(
                    df['close'].values,
                    df['volume'].values
                )
                features.update(vol_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature computation error: {e}")
            return {}
    
    def _update_latency_stats(self, start_time: float):
        """Update latency statistics"""
        latency_ms = (time.time() - start_time) * 1000
        
        # Update rolling average
        alpha = 0.1  # Smoothing factor
        if self.stats['avg_latency_ms'] == 0:
            self.stats['avg_latency_ms'] = latency_ms
        else:
            self.stats['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.stats['avg_latency_ms']
            )
        
        self.stats['messages_processed'] += 1
    
    def get_buffer(self, symbol: str) -> Optional[DataBuffer]:
        """Get data buffer for symbol"""
        return self.buffers.get(symbol)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        buffer = self.buffers.get(symbol)
        return buffer.get_latest_price() if buffer else None
    
    def get_price_series(self, symbol: str, length: int = 100) -> np.ndarray:
        """Get price series for symbol"""
        buffer = self.buffers.get(symbol)
        return buffer.get_price_series(length) if buffer else np.array([])
    
    def get_ohlcv_data(self, symbol: str, length: int = 100) -> pd.DataFrame:
        """Get OHLCV data for symbol"""
        buffer = self.buffers.get(symbol)
        return buffer.get_ohlcv_df(length) if buffer else pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        runtime_seconds = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'runtime_seconds': runtime_seconds,
            'symbols_tracked': len(self.buffers),
            'messages_processed': self.stats['messages_processed'],
            'features_extracted': self.stats['features_extracted'],
            'avg_latency_ms': self.stats['avg_latency_ms'],
            'messages_per_second': self.stats['messages_processed'] / max(runtime_seconds, 1),
            'errors': self.stats['errors'],
            'active_buffers': {
                symbol: {
                    'quotes': len(buffer.quotes),
                    'trades': len(buffer.trades),
                    'bars': len(buffer.bars),
                    'last_update': buffer.last_update
                }
                for symbol, buffer in self.buffers.items()
            }
        }

# Global pipeline instance
_pipeline = None

def get_pipeline(polygon_api_key: str = None,
                alpaca_api_key: str = None, 
                alpaca_secret_key: str = None) -> DataPipeline:
    """Get global data pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline(polygon_api_key, alpaca_api_key, alpaca_secret_key)
    return _pipeline

# Example usage
async def example_usage():
    """Example of how to use the data pipeline"""
    
    # Initialize pipeline
    pipeline = DataPipeline(
        polygon_api_key="YOUR_POLYGON_API_KEY",
        alpaca_api_key="YOUR_ALPACA_API_KEY",
        alpaca_secret_key="YOUR_ALPACA_SECRET_KEY"
    )
    
    # Define callbacks
    async def handle_quote(quote):
        print(f"📊 Quote: {quote.symbol} ${quote.bid:.2f} x ${quote.ask:.2f}")
    
    async def handle_trade(trade):
        print(f"💰 Trade: {trade.symbol} ${trade.price:.2f} x {trade.size}")
    
    async def handle_bar(bar):
        print(f"📈 Bar: {bar.symbol} O:${bar.open:.2f} H:${bar.high:.2f} L:${bar.low:.2f} C:${bar.close:.2f} V:{bar.volume:,}")
    
    async def handle_features(feature_set):
        print(f"🧠 Features: {feature_set.symbol} - {len(feature_set.features)} features extracted")
        print(f"   Latest price: ${feature_set.price_series[-1]:.2f}")
        print(f"   Key features: {dict(list(feature_set.features.items())[:3])}")
    
    # Set up callbacks
    pipeline.add_callback('quote', handle_quote)
    pipeline.add_callback('trade', handle_trade)
    pipeline.add_callback('bar', handle_bar)
    pipeline.add_callback('features', handle_features)
    
    try:
        # Start pipeline
        symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ']
        await pipeline.start(symbols)
        print(f"🚀 Data pipeline started with {len(symbols)} symbols")
        
        # Run for demo
        print("📡 Streaming live data... (Press Ctrl+C to stop)")
        
        # Monitor stats
        start_time = time.time()
        while True:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            stats = pipeline.get_statistics()
            print(f"\n📊 Pipeline Stats:")
            print(f"   Runtime: {stats['runtime_seconds']:.0f}s")
            print(f"   Messages: {stats['messages_processed']}")
            print(f"   Features: {stats['features_extracted']}")
            print(f"   Avg Latency: {stats['avg_latency_ms']:.1f}ms")
            print(f"   Rate: {stats['messages_per_second']:.1f} msg/s")
            print(f"   Errors: {stats['errors']}")
            
            # Show latest prices
            print(f"\n💰 Latest Prices:")
            for symbol in symbols:
                price = pipeline.get_latest_price(symbol)
                if price:
                    print(f"   {symbol}: ${price:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopping pipeline...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await pipeline.stop()
        print("🔌 Pipeline stopped")

if __name__ == "__main__":
    asyncio.run(example_usage())
