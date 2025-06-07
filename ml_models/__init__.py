"""
ML Models Package for Adaptive Day Trading System
Comprehensive ensemble of all ML4T models with parallel inference and online learning
"""

from utils.system_logger import get_system_logger

# Package imports
from .ensemble_manager import EnsembleManager
from .model_server import OptimizedModelServer
from .online_learning import OnlineLearningEngine
from .feature_engineering import FeatureEngineer
from .adaptive_data_processor import AdaptiveDataProcessor, TradingSignal, MarketRegimeUpdate

# Initialize package logger
logger = get_system_logger("ml_models")
logger.info("Initializing ML Models package", extra={
    "component": "ml_models",
    "action": "package_init",
    "modules": ["EnsembleManager", "OptimizedModelServer", "OnlineLearningEngine", "FeatureEngineer", "AdaptiveDataProcessor"]
})

__all__ = [
    'EnsembleManager',
    'OptimizedModelServer',
    'OnlineLearningEngine',
    'FeatureEngineer',
    'AdaptiveDataProcessor',
    'TradingSignal',
    'MarketRegimeUpdate'
]

logger.info("ML Models package initialization complete", extra={
    "component": "ml_models",
    "action": "package_ready",
    "exported_classes": len(__all__)
})