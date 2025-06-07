
import sys
import os

# Import enhanced logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

# Package imports
from .momentum_consistency_filter import MomentumConsistencyFilter
from .vix_position_scaler import VIXPositionScaler
from .entry_timing_optimizer import EntryTimingOptimizer
from .adaptive_data_filter import AdaptiveDataFilter

# Initialize logger for the package
logger = get_system_logger("filters")

logger.startup({
    "component": "filters",
    "action": "package_initialization"
})

__all__ = [
    'MomentumConsistencyFilter',
    'VIXPositionScaler',
    'EntryTimingOptimizer',
    'AdaptiveDataFilter'
]