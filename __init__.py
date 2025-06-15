"""
AI-Enhanced Trading System
A sophisticated trading system powered by Lag-Llama AI forecasting
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"
__description__ = "AI-Enhanced Trading System with Lag-Llama Forecasting"

# Core components
from settings import config
from active_symbols import symbol_manager
from lag_llama_engine import lag_llama_engine, LagLlamaEngine, ForecastResult, MarketForecast
from alpaca import get_alpaca_client, AlpacaTradingClient
from polygon import get_polygon_data_manager, PolygonDataManager

# Trading strategies
from gap_n_go import gap_and_go_strategy, GapAndGoStrategy
from orb import orb_strategy, ORBStrategy
from mean_reversion import mean_reversion_strategy, MeanReversionStrategy

# Main application
from main import TradingSystemOrchestrator

__all__ = [
    # Core components
    "config",
    "symbol_manager", 
    "lag_llama_engine",
    "LagLlamaEngine",
    "ForecastResult",
    "MarketForecast",
    "get_alpaca_client",
    "AlpacaTradingClient",
    "get_polygon_data_manager",
    "PolygonDataManager",
    
    # Trading strategies
    "gap_and_go_strategy",
    "GapAndGoStrategy",
    "orb_strategy", 
    "ORBStrategy",
    "mean_reversion_strategy",
    "MeanReversionStrategy",
    
    # Main application
    "TradingSystemOrchestrator",
]

# Package metadata
__package_info__ = {
    "name": "ai-trading-system",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "python_requires": ">=3.10",
    "dependencies": [
        "torch>=2.1.0",
        "gluonts>=0.14.0", 
        "pytorch-lightning>=2.0.0",
        "aiohttp>=3.9.0",
        "websockets>=12.0",
        "orjson>=3.9.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ]
}
