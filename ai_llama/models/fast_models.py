"""
Fast Traditional Models

Optimized traditional ML models for ultra-low latency trading.
Target: <3ms inference per symbol
"""

import numpy as np
import numba as nb
from typing import Dict, Any, Optional, Tuple, List
import time


@nb.jit(nopython=True, cache=True)
def linear_predict(weights: np.ndarray, features: np.ndarray, bias: float) -> float:
    """Ultra-fast linear prediction"""
    return np.dot(weights, features) + bias


@nb.jit(nopython=True, cache=True)
def sigmoid(x: float) -> float:
    """Fast sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


@nb.jit(nopython=True, cache=True)
def tanh_activation(x: float) -> float:
    """Fast tanh activation"""
    return np.tanh(np.clip(x, -250, 250))


@nb.jit(nopython=True, cache=True)
def simple_momentum_signal(returns: np.ndarray, threshold: float = 0.01) -> float:
    """Simple momentum signal"""
    if len(returns) < 5:
        return 0.0
    
    recent_momentum = np.mean(returns[-5:])
    if recent_momentum > threshold:
        return 1.0
    elif recent_momentum < -threshold:
        return -1.0
    else:
        return 0.0


@nb.jit(nopython=True, cache=True)
def mean_reversion_signal(price: float, ma: float, std: float, threshold: float = 2.0) -> float:
    """Mean reversion signal based on z-score"""
    if std == 0:
        return 0.0
    
    z_score = (price - ma) / std
    
    if z_score > threshold:
        return -1.0  # Overbought, sell
    elif z_score < -threshold:
        return 1.0   # Oversold, buy
    else:
        return 0.0


@nb.jit(nopython=True, cache=True)
def rsi_signal(rsi: float, oversold: float = 30.0, overbought: float = 70.0) -> float:
    """RSI-based signal"""
    if rsi < oversold:
        return 1.0   # Buy signal
    elif rsi > overbought:
        return -1.0  # Sell signal
    else:
        return 0.0


@nb.jit(nopython=True, cache=True)
def bollinger_signal(price: float, bb_upper: float, bb_lower: float, bb_middle: float) -> float:
    """Bollinger Bands signal"""
    bb_width = bb_upper - bb_lower
    if bb_width == 0:
        return 0.0
    
    bb_position = (price - bb_lower) / bb_width
    
    if bb_position > 0.8:
        return -1.0  # Near upper band, sell
    elif bb_position < 0.2:
        return 1.0   # Near lower band, buy
    else:
        return 0.0


class FastLinearModel:
    """Ultra-fast linear model for trading signals"""
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model using normal equation"""
        # Add bias term
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Normal equation: w = (X^T X)^-1 X^T y
        try:
            coeffs = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.weights = coeffs[:-1]
            self.bias = coeffs[-1]
            self.is_fitted = True
        except np.linalg.LinAlgError:
            # Fallback to Ridge regression
            alpha = 0.01
            A = X_with_bias.T @ X_with_bias + alpha * np.eye(X_with_bias.shape[1])
            coeffs = np.linalg.solve(A, X_with_bias.T @ y)
            self.weights = coeffs[:-1]
            self.bias = coeffs[-1]
            self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction"""
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        
        if X.ndim == 1:
            return linear_predict(self.weights, X, self.bias)
        else:
            return np.array([linear_predict(self.weights, x, self.bias) for x in X])


class FastModelEngine:
    """
    Fast model engine for trading signals
    
    Combines multiple simple models for robust predictions
    Target: <3ms total inference time
    """
    
    def __init__(self):
        self.models = {
            'momentum': FastLinearModel(),
            'mean_reversion': FastLinearModel(),
            'trend': FastLinearModel()
        }
        
        self.model_weights = {
            'momentum': 0.4,
            'mean_reversion': 0.3,
            'trend': 0.3
        }
        
        self.is_fitted = False
        self.feature_means = None
        self.feature_stds = None
        
    def _normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features for stable training"""
        if fit:
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0) + 1e-8
        
        if self.feature_means is not None:
            return (features - self.feature_means) / self.feature_stds
        return features
    
    def fit(self, features: np.ndarray, returns: np.ndarray):
        """
        Fit fast models on historical data
        
        Args:
            features: Feature matrix (n_samples, n_features)
            returns: Future returns (n_samples,)
        """
        # Normalize features
        features_norm = self._normalize_features(features, fit=True)
        
        # Create different target variables for different strategies
        momentum_targets = np.sign(returns)  # Direction prediction
        mean_reversion_targets = -np.sign(returns)  # Contrarian prediction
        trend_targets = returns  # Magnitude prediction
        
        # Fit models
        self.models['momentum'].fit(features_norm, momentum_targets)
        self.models['mean_reversion'].fit(features_norm, mean_reversion_targets)
        self.models['trend'].fit(features_norm, trend_targets)
        
        self.is_fitted = True
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Fast prediction for trading signals
        
        Args:
            features: Feature dictionary from FastFeatureEngine
            
        Returns:
            Dict with prediction results
        """
        start_time = time.perf_counter()
        
        if not self.is_fitted:
            return {
                'momentum_signal': 0.0,
                'mean_reversion_signal': 0.0,
                'trend_signal': 0.0,
                'combined_signal': 0.0,
                'confidence': 0.0,
                'inference_time_ms': 0.0
            }
        
        # Extract feature vector
        feature_vector = np.array([
            features.get('return_1', 0.0),
            features.get('return_5', 0.0),
            features.get('rsi_14', 50.0),
            features.get('bb_position', 0.5),
            features.get('momentum_5', 0.0),
            features.get('momentum_20', 0.0),
            features.get('volatility_5', 0.0),
            features.get('volatility_20', 0.0),
            features.get('volume_ratio', 1.0),
            features.get('price_vs_vwap', 0.0)
        ])
        
        # Normalize features
        feature_vector_norm = self._normalize_features(feature_vector.reshape(1, -1))[0]
        
        # Get model predictions
        momentum_pred = self.models['momentum'].predict(feature_vector_norm)
        mean_reversion_pred = self.models['mean_reversion'].predict(feature_vector_norm)
        trend_pred = self.models['trend'].predict(feature_vector_norm)
        
        # Simple rule-based signals for validation
        simple_momentum = simple_momentum_signal(
            np.array([features.get('return_1', 0.0), features.get('return_5', 0.0)])
        )
        
        simple_mean_reversion = rsi_signal(features.get('rsi_14', 50.0))
        
        # Combine predictions
        combined_signal = (
            momentum_pred * self.model_weights['momentum'] +
            mean_reversion_pred * self.model_weights['mean_reversion'] +
            np.tanh(trend_pred) * self.model_weights['trend']
        )
        
        # Calculate confidence based on agreement between models
        signals = [momentum_pred, mean_reversion_pred, np.tanh(trend_pred)]
        confidence = 1.0 - np.std(signals) / (np.mean(np.abs(signals)) + 1e-8)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'momentum_signal': float(momentum_pred),
            'mean_reversion_signal': float(mean_reversion_pred),
            'trend_signal': float(trend_pred),
            'combined_signal': float(combined_signal),
            'simple_momentum': float(simple_momentum),
            'simple_mean_reversion': float(simple_mean_reversion),
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'model_agreement': np.corrcoef(signals)[0, 1] if len(set(signals)) > 1 else 1.0
        }
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update model weights based on recent performance"""
        total_performance = sum(performance_metrics.values())
        if total_performance > 0:
            for model_name in self.model_weights:
                if model_name in performance_metrics:
                    self.model_weights[model_name] = performance_metrics[model_name] / total_performance
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from fitted models"""
        if not self.is_fitted:
            return {}
        
        importance = {}
        feature_names = [
            'return_1', 'return_5', 'rsi_14', 'bb_position',
            'momentum_5', 'momentum_20', 'volatility_5', 'volatility_20',
            'volume_ratio', 'price_vs_vwap'
        ]
        
        for model_name, model in self.models.items():
            if model.weights is not None:
                importance[model_name] = dict(zip(feature_names, np.abs(model.weights)))
        
        return importance


class GapQualityClassifier:
    """Fast classifier for gap quality assessment"""
    
    def __init__(self):
        self.threshold_volume = 1.5  # Volume ratio threshold
        self.threshold_size = 0.02   # Minimum gap size
        
    def classify_gap_quality(self, gap_features: Dict[str, float]) -> float:
        """
        Classify gap quality (0.0 to 1.0)
        
        Args:
            gap_features: Dict with gap-related features
            
        Returns:
            Quality score (0.0 = poor, 1.0 = excellent)
        """
        gap_size = abs(gap_features.get('gap_percent', 0.0))
        volume_ratio = gap_features.get('volume_ratio', 1.0)
        premarket_volume = gap_features.get('premarket_volume_ratio', 1.0)
        
        # Size component (0.2 to 0.8 weight)
        size_score = min(gap_size / 0.05, 1.0)  # Max score at 5% gap
        
        # Volume component
        volume_score = min(volume_ratio / self.threshold_volume, 1.0)
        
        # Premarket activity component
        premarket_score = min(premarket_volume / 2.0, 1.0)
        
        # Combined score
        quality_score = (
            size_score * 0.4 +
            volume_score * 0.3 +
            premarket_score * 0.3
        )
        
        return np.clip(quality_score, 0.0, 1.0)


class BreakoutValidator:
    """Fast validator for breakout quality"""
    
    def __init__(self):
        self.volume_threshold = 1.2
        self.momentum_threshold = 0.01
        
    def validate_breakout(self, breakout_features: Dict[str, float]) -> float:
        """
        Validate breakout quality (0.0 to 1.0)
        
        Args:
            breakout_features: Dict with breakout-related features
            
        Returns:
            Validation score (0.0 = weak, 1.0 = strong)
        """
        volume_ratio = breakout_features.get('volume_ratio', 1.0)
        momentum = abs(breakout_features.get('momentum_5', 0.0))
        range_expansion = breakout_features.get('range_expansion', 1.0)
        
        # Volume confirmation
        volume_score = min(volume_ratio / self.volume_threshold, 1.0)
        
        # Momentum confirmation
        momentum_score = min(momentum / self.momentum_threshold, 1.0)
        
        # Range expansion
        range_score = min(range_expansion / 1.5, 1.0)
        
        # Combined validation score
        validation_score = (
            volume_score * 0.4 +
            momentum_score * 0.4 +
            range_score * 0.2
        )
        
        return np.clip(validation_score, 0.0, 1.0)
