"""
Data Management Module

Real-time and historical data processing for trading.
"""

from .polygon_client import PolygonDataClient
from .alpaca_client import AlpacaDataClient
from .data_pipeline import DataPipeline

__all__ = [
    'PolygonDataClient',
    'AlpacaDataClient', 
    'DataPipeline'
]
