
import sys
import os
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import enhanced logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

# Initialize logger for the package
logger = get_system_logger("data_pipeline")

# Log package initialization
logger.startup({
    "component": "data_pipeline",
    "action": "package_initialization",
    "adaptive_filter_check": "pending"
})

logger.log_data_flow("initialization", "package", data_type="startup")

# Import core data pipeline components
from .polygon_websocket import RealTimeDataFeed, MarketData
from .polygon_rest_api import PolygonRESTClient

# Try to import adaptive_data_filter from filters directory
try:
    # Import from filters directory since that's where it actually exists
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'filters'))
    from adaptive_data_filter import (
        AdaptiveDataFilter,
        MarketConditionScanner,
        AdaptiveStockFilter,
        MLReadyFilter,
        MarketCondition
    )
    ADAPTIVE_FILTER_AVAILABLE = True
    logger.info("Adaptive data filter imported successfully")
    logger.log_data_flow("import", "adaptive_filter", data_sample="success")
except ImportError as e:
    logger.error(e, {
        "adaptive_filter_available": False,
        "attempted_import": "filters.adaptive_data_filter"
    })
    ADAPTIVE_FILTER_AVAILABLE = False
    
    # Create placeholder classes with proper interfaces
    class AdaptiveDataFilter:
        def __init__(self, *args, **kwargs):
            logger.warning("Using placeholder AdaptiveDataFilter - adaptive filter not available")
            
        async def process_polygon_data(self, data: List[Dict]) -> List[Dict]:
            logger.debug("Placeholder filter: returning unfiltered data")
            return data
            
        def get_filter_stats(self) -> Dict:
            return {"placeholder": True, "filtered_count": 0}
    
    class MarketConditionScanner:
        def __init__(self):
            self.current_condition = "unknown"
    
    class AdaptiveStockFilter:
        def __init__(self, *args, **kwargs):
            pass
    
    class MLReadyFilter:
        def __init__(self):
            pass
    
    class MarketCondition:
        def __init__(self, condition: str = "unknown", **kwargs):
            self.condition = condition

__all__ = [
    'RealTimeDataFeed',
    'MarketData',
    'PolygonRESTClient',
    'AdaptiveDataFilter',
    'MarketConditionScanner',
    'AdaptiveStockFilter',
    'MLReadyFilter',
    'MarketCondition',
    'DataPipelineManager',
    'DataPipelineConfig'
]


class DataPipelineConfig:
    """Configuration management for data pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'data_pipeline.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                logger.log_data_flow("load", "config", data_size=len(config.keys()) if config else 0)
                return config or {}
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(e, {
                "config_path": self.config_path,
                "operation": "load_config"
            })
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data_processing": {
                "batch_size": 1000,
                "max_symbols": 5000,
                "buffer_size": 10000,
                "enable_filtering": True
            },
            "performance": {
                "max_processing_time_ms": 5000,
                "memory_limit_mb": 1024,
                "enable_metrics": True
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay_ms": 1000,
                "enable_fallback": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


class DataPipelineManager:
    """
    Main data pipeline manager that orchestrates all data components
    Enhanced with better error handling, configuration management, and performance monitoring
    """
    
    
    def __init__(self,
                 polygon_api_key: str,
                 filter_config_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 symbols: Optional[List[str]] = None):
        """
        Initialize DataPipelineManager with enhanced configuration
        
        Args:
            polygon_api_key: Polygon.io API key
            filter_config_path: Path to adaptive filter configuration
            config_path: Path to main pipeline configuration
            symbols: Optional list of symbols to track
        """
        logger.startup({
            "filter_config_path": filter_config_path,
            "config_path": config_path,
            "symbols_count": len(symbols) if symbols else 0,
            "adaptive_filter_available": ADAPTIVE_FILTER_AVAILABLE
        })
        
        logger.log_data_flow("initialization", "manager",
                                 data_size=len(symbols) if symbols else 0)
        
        # Validate API key
        if not polygon_api_key:
            raise ValueError("polygon_api_key is required")
        
        self.polygon_api_key = polygon_api_key
        self.symbols = symbols or []
        
        # Load configuration
        self.config = DataPipelineConfig(config_path)
        
        # Initialize components with error handling
        try:
            self.websocket_feed = RealTimeDataFeed(api_key=polygon_api_key, symbols=self.symbols)
            logger.info(f"WebSocket feed initialized with {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(e, {"component": "websocket_feed"})
            raise
        
        try:
            self.rest_client = PolygonRESTClient(api_key=polygon_api_key)
            logger.info("REST client initialized")
        except Exception as e:
            logger.error(e, {"component": "rest_client"})
            raise
        
        # Initialize adaptive filter with proper configuration
        filter_config_path = filter_config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'yaml', 'adaptive_filter_config.yaml'
        )
        
        try:
            if ADAPTIVE_FILTER_AVAILABLE:
                self.adaptive_filter = AdaptiveDataFilter(filter_config_path)
                logger.info(f"Adaptive filter initialized with config: {filter_config_path}")
            else:
                self.adaptive_filter = AdaptiveDataFilter()  # Placeholder
                logger.warning("Placeholder filter initialized - adaptive filtering unavailable")
        except Exception as e:
            logger.error(e, {"component": "adaptive_filter"})
            # Use placeholder on error
            self.adaptive_filter = AdaptiveDataFilter()
        
        # Pipeline state and metrics
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_processing_time = 0.0
        self.performance_metrics = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "avg_processing_time_ms": 0.0,
            "last_error": None
        }
        
        # Configuration-driven settings
        self.batch_size = self.config.get('data_processing.batch_size', 1000)
        self.max_symbols = self.config.get('data_processing.max_symbols', 5000)
        self.enable_filtering = self.config.get('data_processing.enable_filtering', True)
        self.max_processing_time = self.config.get('performance.max_processing_time_ms', 5000) / 1000.0
        
        logger.info("DataPipelineManager initialized successfully")
        logger.log_data_flow("initialization", "complete",
                                 data_size=self.max_symbols)
        
    
    async def start_pipeline(self) -> bool:
        """
        Start the complete data pipeline with enhanced error handling
        
        Returns:
            bool: True if pipeline started successfully, False otherwise
        """
        logger.info(f"Starting data pipeline with {len(self.symbols)} symbols")
        logger.log_data_flow("startup", "pipeline", data_size=len(self.symbols))
        
        try:
            # Start WebSocket feed with timeout
            start_time = time.time()
            await asyncio.wait_for(self.websocket_feed.start(), timeout=30.0)
            
            self.is_running = True
            startup_time = (time.time() - start_time) * 1000
            
            logger.info(f"Data pipeline started successfully in {startup_time:.2f}ms")
            logger.log_performance("start_pipeline", startup_time)
            
            return True
            
        except asyncio.TimeoutError:
            timeout_error = Exception("Pipeline startup timeout")
            logger.error(timeout_error, {"timeout_seconds": 30, "operation": "start_pipeline"})
            return False
        except Exception as e:
            logger.error(e, {"operation": "start_pipeline"})
            self.error_count += 1
            return False
    
    
    async def get_filtered_data(self, max_symbols: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get filtered data ready for ML processing with enhanced performance monitoring
        
        Args:
            max_symbols: Maximum number of symbols to process (overrides config)
            
        Returns:
            List of filtered market data dictionaries
        """
        start_time = time.time()
        
        logger.log_data_flow("processing", "filtered_data",
                                 data_size=max_symbols or self.max_symbols)
        
        if not self.is_running:
            logger.warning("Pipeline not running, returning empty data")
            return []
        
        try:
            # Collect raw data from WebSocket with configurable limits
            symbol_limit = max_symbols or self.max_symbols
            raw_data = await self._collect_websocket_data(symbol_limit)
            
            logger.log_data_flow("collection", "raw_data", data_size=len(raw_data))
            
            # Process through adaptive filter if enabled and available
            if self.enable_filtering and ADAPTIVE_FILTER_AVAILABLE and hasattr(self.adaptive_filter, 'process_polygon_data'):
                try:
                    filtered_data = await asyncio.wait_for(
                        self.adaptive_filter.process_polygon_data(raw_data),
                        timeout=self.max_processing_time
                    )
                    logger.debug(f"Data filtered: {len(raw_data)} -> {len(filtered_data)} records")
                    logger.log_data_flow("filtering", "adaptive",
                                             data_size=len(filtered_data))
                except asyncio.TimeoutError:
                    timeout_error = Exception("Filter processing timeout")
                    logger.error(timeout_error, {
                        "timeout_seconds": self.max_processing_time,
                        "raw_data_count": len(raw_data)
                    })
                    filtered_data = raw_data  # Fallback to raw data
                except Exception as e:
                    logger.error(e, {"operation": "adaptive_filtering"})
                    filtered_data = raw_data  # Fallback to raw data
            else:
                logger.debug("Using raw data (filtering disabled or unavailable)")
                filtered_data = raw_data
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(len(raw_data), len(filtered_data), processing_time, True)
            
            logger.debug(f"Data filtering completed: {len(filtered_data)} records in {processing_time:.2f}ms")
            logger.log_performance("get_filtered_data", processing_time)
            
            return filtered_data
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(0, 0, processing_time, False)
            logger.error(e, {"operation": "get_filtered_data"})
            return []
    
    async def _collect_websocket_data(self, symbol_limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Collect data from WebSocket feed with enhanced data enrichment
        
        Args:
            symbol_limit: Maximum number of symbols to collect data for
            
        Returns:
            List of market data dictionaries
        """
        try:
            symbols = self.websocket_feed.get_all_symbols()
            limited_symbols = symbols[:symbol_limit]
            
            logger.log_data_flow("collection", "websocket_data",
                                     data_size=len(limited_symbols))
            
            data = []
            processed_symbols = 0
            
            for symbol in limited_symbols:
                try:
                    latest = self.websocket_feed.get_latest_data(symbol)
                    if latest:
                        # Enhanced data structure with better market data
                        market_data = {
                            'symbol': symbol,
                            'price': latest.price,
                            'volume': latest.volume,
                            'timestamp': latest.timestamp,
                            'bid': latest.bid,
                            'ask': latest.ask,
                            'data_type': getattr(latest, 'data_type', 'trade'),
                            # Enhanced fields with better defaults
                            'market_cap': self._estimate_market_cap(symbol, latest.price),
                            'daily_change': self._calculate_daily_change(symbol, latest.price),
                            'volatility': self._estimate_volatility(symbol),
                            'momentum_score': self._calculate_momentum_score(symbol, latest.price),
                            'spread': (latest.ask - latest.bid) if (latest.ask and latest.bid) else 0.0,
                            'collection_time': time.time()
                        }
                        data.append(market_data)
                        processed_symbols += 1
                        
                except Exception as e:
                    logger.error(e, {
                        "symbol": symbol,
                        "operation": "collect_symbol_data"
                    })
                    continue
            
            success_rate = (processed_symbols / len(limited_symbols)) * 100 if limited_symbols else 0
            logger.debug(f"WebSocket data collection completed: {processed_symbols} symbols, {success_rate:.1f}% success rate")
            logger.log_data_flow("collection", "completed", data_size=len(data))
            
            return data
            
        except Exception as e:
            logger.error(e, {"operation": "collect_websocket_data"})
            return []
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap (simplified implementation)"""
        # This is a simplified estimation - in production, you'd use actual shares outstanding
        if price > 100:
            return 10_000_000_000  # Large cap
        elif price > 20:
            return 2_000_000_000   # Mid cap
        else:
            return 500_000_000     # Small cap
    
    def _calculate_daily_change(self, symbol: str, current_price: float) -> float:
        """Calculate daily change (simplified implementation)"""
        # In production, you'd compare with previous close
        # For now, return a small random-like value based on symbol hash
        return (hash(symbol) % 100 - 50) / 10000.0  # -0.5% to +0.5%
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility (simplified implementation)"""
        # In production, you'd calculate from historical data
        return 0.02 + (hash(symbol) % 50) / 10000.0  # 0.02 to 0.07
    
    def _calculate_momentum_score(self, symbol: str, price: float) -> float:
        """Calculate momentum score (simplified implementation)"""
        # In production, you'd use technical indicators
        return (hash(symbol + str(int(price))) % 100 - 50) / 5000.0  # -0.01 to +0.01
    
    def _update_performance_metrics(self, input_count: int, output_count: int,
                                  processing_time_ms: float, success: bool):
        """Update performance metrics"""
        self.processed_count += input_count
        self.last_processing_time = processing_time_ms
        
        self.performance_metrics["total_processed"] += input_count
        if success:
            self.performance_metrics["successful_processed"] += input_count
        else:
            self.performance_metrics["failed_processed"] += input_count
            self.error_count += 1
        
        # Update average processing time
        total_successful = self.performance_metrics["successful_processed"]
        if total_successful > 0:
            current_avg = self.performance_metrics["avg_processing_time_ms"]
            self.performance_metrics["avg_processing_time_ms"] = (
                (current_avg * (total_successful - input_count) + processing_time_ms) / total_successful
            )
    
    async def stop_pipeline(self) -> bool:
        """
        Stop the data pipeline gracefully
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        logger.info("Stopping data pipeline")
        
        try:
            if self.websocket_feed:
                await self.websocket_feed.stop()
            
            self.is_running = False
            
            logger.info(f"Data pipeline stopped successfully. Processed {self.processed_count} total records")
            
            return True
            
        except Exception as e:
            logger.error(e, {"operation": "stop_pipeline"})
            return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline performance statistics
        
        Returns:
            Dictionary containing detailed pipeline statistics
        """
        try:
            filter_stats = {}
            if hasattr(self.adaptive_filter, 'get_filter_stats'):
                try:
                    filter_stats = self.adaptive_filter.get_filter_stats()
                except Exception as e:
                    logger.error(e, {"operation": "get_filter_stats"})
                    filter_stats = {"error": str(e)}
            
            websocket_stats = {}
            if hasattr(self.websocket_feed, 'get_connection_stats'):
                try:
                    websocket_stats = self.websocket_feed.get_connection_stats()
                except Exception as e:
                    logger.error(e, {"operation": "get_websocket_stats"})
                    websocket_stats = {"error": str(e)}
            
            return {
                'pipeline': {
                    'is_running': self.is_running,
                    'processed_count': self.processed_count,
                    'error_count': self.error_count,
                    'last_processing_time_ms': self.last_processing_time,
                    'symbols_tracked': len(self.symbols),
                    'adaptive_filter_available': ADAPTIVE_FILTER_AVAILABLE,
                    'filtering_enabled': self.enable_filtering
                },
                'performance': self.performance_metrics,
                'configuration': {
                    'batch_size': self.batch_size,
                    'max_symbols': self.max_symbols,
                    'max_processing_time_ms': self.max_processing_time * 1000
                },
                'filter_stats': filter_stats,
                'websocket_stats': websocket_stats,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(e, {"operation": "get_pipeline_stats"})
            return {
                'error': str(e),
                'timestamp': time.time(),
                'is_running': self.is_running
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the pipeline
        
        Returns:
            Dictionary containing health status information
        """
        health_status = {
            'overall_health': 'unknown',
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check WebSocket feed health
            if hasattr(self.websocket_feed, 'get_health_status'):
                ws_health = self.websocket_feed.get_health_status()
                health_status['components']['websocket'] = ws_health
            else:
                health_status['components']['websocket'] = {'status': 'unknown'}
            
            # Check REST client health
            if hasattr(self.rest_client, 'is_healthy'):
                rest_healthy = self.rest_client.is_healthy()
                health_status['components']['rest_client'] = {'healthy': rest_healthy}
            else:
                health_status['components']['rest_client'] = {'status': 'unknown'}
            
            # Check adaptive filter health
            health_status['components']['adaptive_filter'] = {
                'available': ADAPTIVE_FILTER_AVAILABLE,
                'enabled': self.enable_filtering
            }
            
            # Determine overall health
            ws_healthy = health_status['components']['websocket'].get('is_healthy', False)
            rest_healthy = health_status['components']['rest_client'].get('healthy', False)
            
            if ws_healthy and rest_healthy and self.is_running:
                health_status['overall_health'] = 'healthy'
            elif self.is_running:
                health_status['overall_health'] = 'degraded'
            else:
                health_status['overall_health'] = 'unhealthy'
            
            logger.debug(f"Health check completed: {health_status['overall_health']}")
            
        except Exception as e:
            logger.error(e, {"operation": "health_check"})
            health_status['overall_health'] = 'error'
            health_status['error'] = str(e)
        
        return health_status