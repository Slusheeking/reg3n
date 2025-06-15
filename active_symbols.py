"""
Active Symbols Management for Trading System
Manages watchlists, symbol rotation, and real-time symbol performance
"""

import orjson
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Optional
from datetime import datetime, time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SymbolStatus(Enum):
    ACTIVE = "active"
    MONITORING = "monitoring"
    HALTED = "halted"
    EXCLUDED = "excluded"
    PREMARKET = "premarket"
    AFTERHOURS = "afterhours"

class SymbolCategory(Enum):
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    MEME_STOCK = "meme_stock"
    ETF = "etf"
    INDEX = "index"
    HIGH_VOLUME = "high_volume"
    GAP_CANDIDATE = "gap_candidate"
    EARNINGS = "earnings"

@dataclass
class SymbolMetrics:
    """Real-time metrics for each symbol"""
    symbol: str
    price: float = 0.0
    volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 0.0
    
    # Gap metrics
    prev_close: float = 0.0
    gap_percent: float = 0.0
    gap_size: float = 0.0
    
    # Volatility metrics
    atr: float = 0.0
    daily_range: float = 0.0
    volatility: float = 0.0
    
    # Market data
    market_cap: Optional[float] = None
    avg_daily_volume: Optional[int] = None
    sector: Optional[str] = None
    
    # Performance tracking
    pnl_today: float = 0.0
    trades_today: int = 0
    win_rate: float = 0.0
    last_updated: Optional[datetime] = None
    
    def update_gap_metrics(self, current_price: float, previous_close: float):
        """Update gap-related metrics"""
        self.price = current_price
        self.prev_close = previous_close
        self.gap_size = current_price - previous_close
        self.gap_percent = (self.gap_size / previous_close) * 100 if previous_close > 0 else 0.0
    
    def update_volume_metrics(self, current_volume: int, average_volume: int):
        """Update volume-related metrics"""
        self.volume = current_volume
        self.avg_volume = average_volume
        self.volume_ratio = current_volume / average_volume if average_volume > 0 else 0.0
    
    def is_gap_candidate(self, min_gap_percent: float = 2.0) -> bool:
        """Check if symbol qualifies for gap trading"""
        return abs(self.gap_percent) >= min_gap_percent and self.volume_ratio >= 1.5

@dataclass
class SymbolConfig:
    """Configuration for each symbol"""
    symbol: str
    category: SymbolCategory
    status: SymbolStatus = SymbolStatus.ACTIVE
    max_position_size: float = 0.05  # 5% default
    strategies_enabled: List[str] = field(default_factory=list)
    last_trade_time: Optional[datetime] = None
    
    # Strategy-specific settings
    gap_trading_enabled: bool = True
    orb_trading_enabled: bool = True
    mean_reversion_enabled: bool = True
    
    # Risk overrides
    risk_multiplier: float = 1.0  # Adjust default risk for this symbol
    volatility_cap: float = 0.1   # 10% max daily volatility
    
    def __post_init__(self):
        if not self.strategies_enabled:
            self.strategies_enabled = ["gap_and_go", "orb", "mean_reversion"]

class SymbolManager:
    """Manages active trading symbols and their configurations"""
    
    def __init__(self, config_file: str = "symbols_config.json"):
        self.config_file = config_file
        self.symbols: Dict[str, SymbolConfig] = {}
        self.metrics: Dict[str, SymbolMetrics] = {}
        self.watchlists: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.symbol_performance: Dict[str, Dict] = {}
        
        # Initialize default watchlists
        self._initialize_default_watchlists()
        self._load_symbol_configs()
    
    def _initialize_default_watchlists(self):
        """Set up default watchlists"""
        
        # Large cap tech stocks
        self.watchlists["mega_cap"] = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"
        ]
        
        # High volume movers
        self.watchlists["high_volume"] = [
            "SPY", "QQQ", "IWM", "AMD", "BABA", "NIO", "PLTR", "AMC", "GME"
        ]
        
        # Gap trading candidates (updated daily)
        self.watchlists["gap_candidates"] = []
        
        # Earnings plays (updated weekly)
        self.watchlists["earnings"] = []
        
        # Sector ETFs
        self.watchlists["sector_etfs"] = [
            "XLK", "XLF", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLRE"
        ]
        
        # Market indices
        self.watchlists["indices"] = [
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO"
        ]
        
        # Momentum stocks
        self.watchlists["momentum"] = [
            "ROKU", "SHOP", "SQ", "PYPL", "ZOOM", "SNOW", "CRWD", "OKTA"
        ]
    
    def _load_symbol_configs(self):
        """Load symbol configurations from file"""
        try:
            with open(self.config_file, 'r') as f:
                data = orjson.loads(f.read())
                
            for symbol_data in data.get('symbols', []):
                config = SymbolConfig(**symbol_data)
                self.symbols[config.symbol] = config
                
                # Initialize metrics
                self.metrics[config.symbol] = SymbolMetrics(symbol=config.symbol)
                
        except FileNotFoundError:
            logger.info(f"Config file {self.config_file} not found, using defaults")
            self._create_default_configs()
        except Exception as e:
            logger.error(f"Error loading symbol configs: {e}")
            self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default symbol configurations"""
        
        # Configure all watchlist symbols
        all_symbols = set()
        for watchlist in self.watchlists.values():
            all_symbols.update(watchlist)
        
        for symbol in all_symbols:
            category = self._determine_category(symbol)
            self.symbols[symbol] = SymbolConfig(
                symbol=symbol,
                category=category,
                status=SymbolStatus.ACTIVE
            )
            self.metrics[symbol] = SymbolMetrics(symbol=symbol)
        
        self.save_configs()
    
    def _determine_category(self, symbol: str) -> SymbolCategory:
        """Determine category for a symbol"""
        
        # ETFs and indices
        if symbol in ["SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO"]:
            return SymbolCategory.INDEX
        
        if symbol.startswith("XL") or symbol in ["GLD", "SLV", "TLT", "HYG"]:
            return SymbolCategory.ETF
        
        # Meme stocks
        if symbol in ["GME", "AMC", "BBBY", "CLOV"]:
            return SymbolCategory.MEME_STOCK
        
        # Large cap
        if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]:
            return SymbolCategory.LARGE_CAP
        
        # Default to mid cap
        return SymbolCategory.MID_CAP
    
    def add_symbol(self, symbol: str, category: SymbolCategory = SymbolCategory.MID_CAP) -> bool:
        """Add a new symbol to active trading"""
        
        if symbol in self.symbols:
            logger.warning(f"Symbol {symbol} already exists")
            return False
        
        self.symbols[symbol] = SymbolConfig(
            symbol=symbol,
            category=category,
            status=SymbolStatus.ACTIVE
        )
        
        self.metrics[symbol] = SymbolMetrics(symbol=symbol)
        
        logger.info(f"Added symbol {symbol} with category {category}")
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from active trading"""
        
        if symbol not in self.symbols:
            return False
        
        self.symbols[symbol].status = SymbolStatus.EXCLUDED
        logger.info(f"Excluded symbol {symbol} from trading")
        return True
    
    def update_symbol_metrics(self, symbol: str, **kwargs):
        """Update real-time metrics for a symbol"""
        
        if symbol not in self.metrics:
            self.metrics[symbol] = SymbolMetrics(symbol=symbol)
        
        metrics = self.metrics[symbol]
        
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        metrics.last_updated = datetime.now()
    
    def get_active_symbols(self, strategy: Optional[str] = None) -> List[str]:
        """Get list of active symbols, optionally filtered by strategy"""
        
        active_symbols = []
        
        for symbol, config in self.symbols.items():
            if config.status != SymbolStatus.ACTIVE:
                continue
            
            if strategy and strategy not in config.strategies_enabled:
                continue
            
            active_symbols.append(symbol)
        
        return active_symbols
    
    def get_gap_candidates(self, min_gap_percent: float = 2.0) -> List[str]:
        """Get symbols that are gapping significantly"""
        
        candidates = []
        
        for symbol in self.get_active_symbols("gap_and_go"):
            metrics = self.metrics.get(symbol)
            if metrics and metrics.is_gap_candidate(min_gap_percent):
                candidates.append(symbol)
        
        # Sort by gap size (absolute value)
        candidates.sort(key=lambda s: abs(self.metrics[s].gap_percent), reverse=True)
        
        return candidates
    
    def get_high_volume_symbols(self, min_volume_ratio: float = 2.0) -> List[str]:
        """Get symbols with unusually high volume"""
        
        high_volume = []
        
        for symbol in self.get_active_symbols():
            metrics = self.metrics.get(symbol)
            if metrics and metrics.volume_ratio >= min_volume_ratio:
                high_volume.append(symbol)
        
        # Sort by volume ratio
        high_volume.sort(key=lambda s: self.metrics[s].volume_ratio, reverse=True)
        
        return high_volume
    
    def get_symbols_by_category(self, category: SymbolCategory) -> List[str]:
        """Get symbols by category"""
        
        return [
            symbol for symbol, config in self.symbols.items()
            if config.category == category and config.status == SymbolStatus.ACTIVE
        ]
    
    def update_watchlist(self, watchlist_name: str, symbols: List[str]):
        """Update a watchlist with new symbols"""
        
        self.watchlists[watchlist_name] = symbols
        
        # Add any new symbols to our tracking
        for symbol in symbols:
            if symbol not in self.symbols:
                self.add_symbol(symbol)
    
    def get_watchlist(self, name: str) -> List[str]:
        """Get symbols from a specific watchlist"""
        return self.watchlists.get(name, [])
    
    def analyze_symbol_performance(self, symbol: str) -> Dict:
        """Analyze performance metrics for a symbol"""
        
        if symbol not in self.metrics:
            return {}
        
        metrics = self.metrics[symbol]
        config = self.symbols.get(symbol)
        
        return {
            "symbol": symbol,
            "current_price": metrics.price,
            "gap_percent": metrics.gap_percent,
            "volume_ratio": metrics.volume_ratio,
            "volatility": metrics.volatility,
            "pnl_today": metrics.pnl_today,
            "trades_today": metrics.trades_today,
            "win_rate": metrics.win_rate,
            "category": config.category.value if config else "unknown",
            "status": config.status.value if config else "unknown"
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of all active symbols"""
        
        summary = {
            "total_symbols": len(self.get_active_symbols()),
            "gap_candidates": len(self.get_gap_candidates()),
            "high_volume": len(self.get_high_volume_symbols()),
            "categories": {},
            "total_pnl": 0.0,
            "total_trades": 0
        }
        
        # Category breakdown
        for category in SymbolCategory:
            symbols = self.get_symbols_by_category(category)
            summary["categories"][category.value] = len(symbols)
        
        # Aggregate P&L and trades
        for metrics in self.metrics.values():
            summary["total_pnl"] += metrics.pnl_today
            summary["total_trades"] += metrics.trades_today
        
        return summary
    
    def save_configs(self):
        """Save current symbol configurations to file"""
        
        data = {
            "symbols": [asdict(config) for config in self.symbols.values()],
            "watchlists": self.watchlists,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.config_file, 'wb') as f: # orjson.dumps returns bytes
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2, default=str))
            logger.info("Symbol configurations saved")
        except Exception as e:
            logger.error(f"Error saving configs: {e}")
    
    async def daily_symbol_refresh(self):
        """Daily refresh of symbol lists and metrics"""
        
        logger.info("Starting daily symbol refresh")
        
        # Reset daily metrics
        for metrics in self.metrics.values():
            metrics.pnl_today = 0.0
            metrics.trades_today = 0
        
        # TODO: Fetch new gap candidates from Polygon
        # TODO: Update earnings calendar
        # TODO: Refresh market cap data
        
        self.save_configs()
        logger.info("Daily symbol refresh completed")

# Global symbol manager instance
symbol_manager = SymbolManager()

# Convenience functions
def get_active_symbols(strategy: Optional[str] = None) -> List[str]:
    """Get active symbols for trading"""
    return symbol_manager.get_active_symbols(strategy)

def get_gap_candidates(min_gap_percent: float = 2.0) -> List[str]:
    """Get gap trading candidates"""
    return symbol_manager.get_gap_candidates(min_gap_percent)

def update_symbol_metrics(symbol: str, **kwargs):
    """Update metrics for a symbol"""
    symbol_manager.update_symbol_metrics(symbol, **kwargs)