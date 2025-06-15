"""
Trading System Configuration Settings
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MarketRegime(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"

class StrategyType(Enum):
    GAP_AND_GO = "gap_and_go"
    ORB = "orb"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class LagLlamaConfig:
    """Lag-Llama Model Configuration"""
    model_path: str = "lag-llama.ckpt"
    context_length: int = 512
    prediction_length: int = 64
    batch_size: int = 32
    device: str = "cuda:0"
    precision: str = "bf16"
    num_parallel_samples: int = 100
    rope_scaling_factor: float = 8.0
    attention_optimization: str = "flash_attention_2"
    memory_efficient: bool = True
    
    # Forecasting horizons (minutes)
    forecast_horizons: List[int] = field(default_factory=lambda: [5, 15, 30, 60, 120])
    
    def __post_init__(self):
        if not self.forecast_horizons: # Ensure it's not empty, if default_factory was just list
            self.forecast_horizons = [5, 15, 30, 60, 120]

@dataclass
class PolygonConfig:
    """Polygon.io API Configuration"""
    api_key: str = os.getenv('POLYGON_API_KEY', '')
    base_url: str = "https://api.polygon.io"
    websocket_url: str = "wss://socket.polygon.io/stocks"
    max_retries: int = 3
    timeout: int = 30
    rate_limit_delay: float = 0.1
    
    # Data subscriptions
    subscribe_trades: bool = True
    subscribe_quotes: bool = True
    subscribe_aggregates: bool = True
    subscribe_indices: bool = True

@dataclass
class AlpacaConfig:
    """Alpaca API Configuration"""
    api_key: str = os.getenv('APCA_API_KEY_ID', os.getenv('ALPACA_API_KEY', ''))
    secret_key: str = os.getenv('APCA_API_SECRET_KEY', os.getenv('ALPACA_SECRET_KEY', ''))
    base_url: str = os.getenv('ALPACA_BASE_URL', "https://paper-api.alpaca.markets")  # Paper trading
    websocket_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    max_retries: int = 3
    timeout: int = 10
    
    # Trading settings
    paper_trading: bool = True
    max_positions: int = 10
    max_daily_trades: int = 50
    pdt_compliant: bool = True

@dataclass
class RiskManagement:
    """Risk Management Configuration"""
    max_portfolio_risk: float = 0.10  # 10% max portfolio risk
    max_position_size: float = 0.05   # 5% max single position
    max_daily_loss: float = 0.03      # 3% max daily loss
    max_drawdown: float = 0.15        # 15% max drawdown
    
    # Position sizing
    kelly_fraction_cap: float = 0.25  # Cap Kelly criterion at 25%
    volatility_lookback: int = 20     # Days for volatility calculation
    correlation_threshold: float = 0.7 # Max correlation between positions
    
    # Stop losses
    dynamic_stops: bool = True
    atr_stop_multiplier: float = 2.0
    time_stop_minutes: int = 240      # 4 hours max hold

@dataclass
class StrategyConfig:
    """Strategy-Specific Configuration"""
    enabled_strategies: List[StrategyType] = field(default_factory=list)
    confidence_threshold: float = 0.65
    min_signal_strength: float = 0.7
    signal_timeout_minutes: int = 15
    
    # Strategy weights
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    # Market regime adaptations
    regime_adjustments: Dict[MarketRegime, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.enabled_strategies:
            self.enabled_strategies = [
                StrategyType.GAP_AND_GO,
                StrategyType.ORB,
                StrategyType.MEAN_REVERSION
            ]
        
        if not self.strategy_weights:
            self.strategy_weights = {
                "gap_and_go": 0.4,
                "orb": 0.35,
                "mean_reversion": 0.25
            }
        
        if not self.regime_adjustments:
            self.regime_adjustments = {
                MarketRegime.TRENDING_BULL: {
                    "gap_and_go": 1.2,
                    "orb": 1.1,
                    "mean_reversion": 0.8
                },
                MarketRegime.TRENDING_BEAR: {
                    "gap_and_go": 1.1,
                    "orb": 0.9,
                    "mean_reversion": 1.0
                },
                MarketRegime.RANGE_BOUND: {
                    "gap_and_go": 0.7,
                    "orb": 1.3,
                    "mean_reversion": 1.4
                },
                MarketRegime.HIGH_VOLATILITY: {
                    "gap_and_go": 0.8,
                    "orb": 0.8,
                    "mean_reversion": 0.6
                }
            }

@dataclass
class GapAndGoConfig:
    """Gap and Go Strategy Configuration"""
    min_gap_percent: float = 0.02     # 2% minimum gap
    max_gap_percent: float = 0.15     # 15% maximum gap
    min_volume_ratio: float = 2.0     # 2x average volume
    gap_continuation_threshold: float = 0.75  # 75% probability
    max_hold_minutes: int = 120       # 2 hours max hold
    target_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    
    def __post_init__(self):
        if not self.target_multipliers: # Ensure it's not empty
            self.target_multipliers = [0.5, 1.0, 1.5]  # Gap fill ratios

@dataclass
class ORBConfig:
    """Opening Range Breakout Configuration"""
    range_minutes: int = 15           # 15-minute opening range
    min_range_size: float = 0.005     # 0.5% minimum range
    max_range_size: float = 0.04      # 4% maximum range
    breakout_confirmation: float = 0.01  # 1% beyond range
    volume_confirmation: float = 1.5   # 1.5x average volume
    sustainability_threshold: float = 0.75  # 75% probability
    max_hold_minutes: int = 180       # 3 hours max hold

@dataclass
class MeanReversionConfig:
    """Mean Reversion Strategy Configuration"""
    lookback_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    deviation_threshold: float = 2.0  # Standard deviations
    reversion_probability: float = 0.70  # 70% probability
    max_hold_minutes: int = 60        # 1 hour max hold
    volatility_filter: bool = True
    trend_filter: bool = True
    
    def __post_init__(self):
        if not self.lookback_periods: # Ensure it's not empty
            self.lookback_periods = [20, 50, 200]  # Different mean periods

@dataclass
class SystemConfig:
    """Main System Configuration"""
    # Component configs
    lag_llama: LagLlamaConfig = LagLlamaConfig()
    polygon: PolygonConfig = PolygonConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    risk: RiskManagement = RiskManagement()
    strategy: StrategyConfig = StrategyConfig()
    
    # Strategy-specific configs
    gap_and_go: GapAndGoConfig = GapAndGoConfig()
    orb: ORBConfig = ORBConfig()
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    
    # System settings
    update_frequency_seconds: int = 30
    forecast_update_frequency: int = 60
    performance_tracking: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Market hours (Eastern Time)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    
    # Pre/post market trading
    premarket_start: int = 4  # 4 AM ET
    afterhours_end: int = 20  # 8 PM ET
    
    def is_market_hours(self, current_time) -> bool:
        """Check if current time is within market hours"""
        import datetime
        
        if isinstance(current_time, datetime.datetime):
            hour = current_time.hour
            minute = current_time.minute
            
            market_open = hour > self.market_open_hour or (
                hour == self.market_open_hour and minute >= self.market_open_minute
            )
            market_close = hour < self.market_close_hour or (
                hour == self.market_close_hour and minute <= self.market_close_minute
            )
            
            return market_open and market_close
        
        return False
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check API keys
        if not self.polygon.api_key:
            errors.append("Polygon API key not set")
        
        if not self.alpaca.api_key or not self.alpaca.secret_key:
            errors.append("Alpaca API credentials not set")
        
        # Validate risk parameters
        if self.risk.max_position_size > self.risk.max_portfolio_risk:
            errors.append("Max position size cannot exceed max portfolio risk")
        
        # Validate strategy weights
        total_weight = sum(self.strategy.strategy_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Strategy weights sum to {total_weight}, should be 1.0")
        
        if errors:
            for error in errors:
                print(f"Config Error: {error}")
            return False
        
        return True

# Global config instance
config = SystemConfig()

# Environment-specific overrides
if os.getenv('TRADING_ENV') == 'live':
    config.alpaca.base_url = "https://api.alpaca.markets"
    config.alpaca.paper_trading = False
    config.risk.max_daily_loss = 0.02  # More conservative in live trading

if os.getenv('TRADING_ENV') == 'development':
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.performance_tracking = True