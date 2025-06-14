"""
Feature Engineering Module

High-performance vectorized feature extraction for trading strategies.
"""

from .fast_features import FastFeatureEngine
from .gap_features import GapFeatureExtractor
from .orb_features import ORBFeatureExtractor
from .vol_features import VolatilityFeatureExtractor

__all__ = [
    'FastFeatureEngine',
    'GapFeatureExtractor', 
    'ORBFeatureExtractor',
    'VolatilityFeatureExtractor'
]
