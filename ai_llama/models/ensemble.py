"""
Advanced 4-Model Ensemble System

Combines all foundation models with fast traditional models for ultimate AI trading.
Dynamic weighting, confidence scoring, and model agreement analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import all model engines
from .fast_models import FastModelEngine, GapQualityClassifier, BreakoutValidator
from .lag_llama_engine import LagLlamaEngine
from .chronos_engine import ChronosEngine, get_chronos_engine
from .timesfm_engine import TimesFMEngine, get_timesfm_engine

@dataclass
class EnsembleConfig:
    """Configuration for 4-model ensemble"""
    # Model weights (dynamically adjusted)
    fast_model_weight: float = 0.25     # Speed priority
    lag_llama_weight: float = 0.25      # General AI patterns
    chronos_weight: float = 0.25        # Amazon's validation
    timesfm_weight: float = 0.25        # Google's context + covariates
    
    # Performance tracking
    track_performance: bool = True
    weight_update_frequency: int = 50   # Update weights every N predictions
    
    # Agreement thresholds
    high_agreement_threshold: float = 0.8
    low_agreement_threshold: float = 0.4
    
    # Inference settings
    parallel_inference: bool = True
    max_workers: int = 4
    
    # Model fallback settings
    enable_fallback: bool = True
    min_working_models: int = 2

@dataclass
class EnsembleResult:
    """Result from 4-model ensemble prediction"""
    signal: float                                # Final ensemble signal (-1 to 1)
    confidence: float                            # Ensemble confidence (0 to 1)
    model_agreement: float                       # Agreement between models (0 to 1)
    individual_predictions: Dict[str, float]     # Each model's prediction
    individual_confidences: Dict[str, float]     # Each model's confidence
    inference_time_ms: float                     # Total inference time
    model_weights: Dict[str, float]              # Current model weights
    participating_models: List[str]              # Models that contributed
    signal_strength: float                       # Absolute signal strength

class ModelEnsemble:
    """
    Ultimate 4-Model Ensemble System for HFT Trading
    
    Architecture:
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Fast Models   │   │   Lag-Llama     │   │    Chronos      │   │    TimesFM      │
    │   (3-8ms)       │   │   (100-300ms)   │   │   (20-50ms)     │   │   (50-100ms)    │
    │                 │   │                 │   │                 │   │                 │
    │ • Gap Quality   │   │ • Uncertainty   │   │ • Zero-shot     │   │ • Covariates    │
    │ • ORB Validation│   │ • GluonTS base  │   │ • T5-based      │   │ • Finetuning    │ 
    │ • Vol Analysis  │   │ • Probabilistic │   │ • Chronos-Bolt  │   │ • 2048 context  │
    └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
              │                     │                     │                     │
              └─────────────────────┼─────────────────────┼─────────────────────┘
                                    │                     │
                            ┌─────────────────────────────────────┐
                            │       Dynamic Ensemble Engine       │
                            │                                     │
                            │ • Weighted voting                   │
                            │ • Confidence scoring               │
                            │ • Agreement analysis               │
                            │ • Performance tracking             │
                            │ • Automatic fallback               │
                            │ • Position size optimization       │
                            └─────────────────────────────────────┘
    
    Features:
    - All 4 models running in parallel
    - Dynamic weight adjustment based on performance
    - Model agreement scoring for confidence validation
    - Automatic fallback when models fail
    - Strategy-specific optimizations
    - Real-time performance monitoring
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        
        # Initialize all model engines
        self.fast_engine = FastModelEngine()
        self.lag_llama_engine = LagLlamaEngine()
        self.chronos_engine = get_chronos_engine()
        self.timesfm_engine = get_timesfm_engine()
        
        # Model weights (dynamically updated)
        self.weights = {
            'fast_models': self.config.fast_model_weight,
            'lag_llama': self.config.lag_llama_weight,
            'chronos': self.config.chronos_weight,
            'timesfm': self.config.timesfm_weight
        }
        
        # Performance tracking for all models
        self.performance_history = {
            'fast_models': {'predictions': [], 'accuracy': [], 'confidence': []},
            'lag_llama': {'predictions': [], 'accuracy': [], 'confidence': []},
            'chronos': {'predictions': [], 'accuracy': [], 'confidence': []},
            'timesfm': {'predictions': [], 'accuracy': [], 'confidence': []}
        }
        
        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.successful_predictions = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized 4-Model Ensemble System")
        
    def predict(self, 
                features: Dict[str, float],
                price_series: Optional[np.ndarray] = None,
                volume_series: Optional[np.ndarray] = None,
                strategy_type: str = "gap_and_go",
                market_covariates: Optional[Dict[str, np.ndarray]] = None) -> EnsembleResult:
        """
        Generate ensemble prediction from all 4 models
        
        Args:
            features: Extracted features dictionary
            price_series: Price time series for foundation models
            volume_series: Volume data for enhanced analysis
            strategy_type: Trading strategy type
            market_covariates: Market microstructure data for TimesFM
            
        Returns:
            EnsembleResult with combined 4-model prediction
        """
        start_time = time.time()
        
        # Store individual predictions and confidences
        predictions = {}
        confidences = {}
        inference_times = {}
        
        if self.config.parallel_inference:
            # Parallel inference for maximum speed
            predictions, confidences, inference_times = self._parallel_inference(
                features, price_series, volume_series, strategy_type, market_covariates
            )
        else:
            # Sequential inference
            predictions, confidences, inference_times = self._sequential_inference(
                features, price_series, volume_series, strategy_type, market_covariates
            )
        
        # Calculate ensemble signal using weighted average
        ensemble_signal = self._calculate_weighted_signal(predictions)
        
        # Calculate model agreement
        agreement = self._calculate_agreement(predictions)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            confidences, agreement
        )
        
        total_inference_time = time.time() - start_time
        
        # Update performance tracking
        self._update_performance_tracking(predictions, confidences)
        
        # Update model weights if needed
        if self.prediction_count % self.config.weight_update_frequency == 0:
            self._update_model_weights()
        
        self.prediction_count += 1
        self.total_inference_time += total_inference_time
        
        # Filter participating models
        participating_models = [k for k, v in predictions.items() if v != 0.0]
        
        result = EnsembleResult(
            signal=ensemble_signal,
            confidence=ensemble_confidence,
            model_agreement=agreement,
            individual_predictions=predictions,
            individual_confidences=confidences,
            inference_time_ms=total_inference_time * 1000,
            model_weights=self.weights.copy(),
            participating_models=participating_models,
            signal_strength=abs(ensemble_signal)
        )
        
        self.logger.debug(
            f"4-Model Ensemble: signal={ensemble_signal:.3f}, "
            f"confidence={ensemble_confidence:.3f}, "
            f"agreement={agreement:.3f}, "
            f"time={total_inference_time*1000:.1f}ms, "
            f"models={len(participating_models)}"
        )
        
        return result
    
    def _parallel_inference(self, 
                          features: Dict[str, float],
                          price_series: Optional[np.ndarray],
                          volume_series: Optional[np.ndarray],
                          strategy_type: str,
                          market_covariates: Optional[Dict[str, np.ndarray]]) -> Tuple[Dict, Dict, Dict]:
        """Run all 4 models in parallel for maximum speed"""
        
        predictions = {}
        confidences = {}
        inference_times = {}
        
        def run_fast_models():
            try:
                start = time.time()
                result = self.fast_engine.predict_features(features)
                inference_times['fast_models'] = (time.time() - start) * 1000
                return 'fast_models', result.get('combined_signal', 0.0), result.get('confidence', 0.0)
            except Exception as e:
                self.logger.error(f"Fast models failed: {e}")
                return 'fast_models', 0.0, 0.0
        
        def run_lag_llama():
            try:
                if price_series is not None:
                    start = time.time()
                    result = self.lag_llama_engine.predict(price_series)
                    inference_times['lag_llama'] = (time.time() - start) * 1000
                    signal = self.lag_llama_engine.get_trading_signal(result)
                    return 'lag_llama', signal.get('signal', 0.0), signal.get('confidence', 0.0)
                else:
                    return 'lag_llama', 0.0, 0.0
            except Exception as e:
                self.logger.error(f"Lag-Llama failed: {e}")
                return 'lag_llama', 0.0, 0.0
        
        def run_chronos():
            try:
                if price_series is not None:
                    start = time.time()
                    result = self.chronos_engine.predict(price_series)
                    inference_times['chronos'] = (time.time() - start) * 1000
                    # Convert forecasting result to trading signal
                    signal = self._chronos_to_signal(result, price_series[-1])
                    return 'chronos', signal, result.confidence
                else:
                    return 'chronos', 0.0, 0.0
            except Exception as e:
                self.logger.error(f"Chronos failed: {e}")
                return 'chronos', 0.0, 0.0
        
        def run_timesfm():
            try:
                if price_series is not None:
                    start = time.time()
                    result = self.timesfm_engine.predict(
                        price_series, 
                        covariates=market_covariates
                    )
                    inference_times['timesfm'] = (time.time() - start) * 1000
                    # Convert forecasting result to trading signal
                    signal = self._timesfm_to_signal(result, price_series[-1])
                    return 'timesfm', signal, result.confidence
                else:
                    return 'timesfm', 0.0, 0.0
            except Exception as e:
                self.logger.error(f"TimesFM failed: {e}")
                return 'timesfm', 0.0, 0.0
        
        # Run all models in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(run_fast_models),
                executor.submit(run_lag_llama),
                executor.submit(run_chronos),
                executor.submit(run_timesfm)
            ]
            
            for future in futures:
                try:
                    model_name, prediction, confidence = future.result(timeout=5.0)
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                except Exception as e:
                    self.logger.error(f"Model execution failed: {e}")
        
        return predictions, confidences, inference_times
    
    def _sequential_inference(self, 
                            features: Dict[str, float],
                            price_series: Optional[np.ndarray],
                            volume_series: Optional[np.ndarray],
                            strategy_type: str,
                            market_covariates: Optional[Dict[str, np.ndarray]]) -> Tuple[Dict, Dict, Dict]:
        """Run models sequentially (fallback mode)"""
        
        predictions = {}
        confidences = {}
        inference_times = {}
        
        # Fast Models
        try:
            start = time.time()
            result = self.fast_engine.predict_features(features)
            predictions['fast_models'] = result.get('combined_signal', 0.0)
            confidences['fast_models'] = result.get('confidence', 0.0)
            inference_times['fast_models'] = (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Fast models failed: {e}")
            predictions['fast_models'] = 0.0
            confidences['fast_models'] = 0.0
        
        # Lag-Llama
        try:
            if price_series is not None:
                start = time.time()
                result = self.lag_llama_engine.predict(price_series)
                signal = self.lag_llama_engine.get_trading_signal(result)
                predictions['lag_llama'] = signal.get('signal', 0.0)
                confidences['lag_llama'] = signal.get('confidence', 0.0)
                inference_times['lag_llama'] = (time.time() - start) * 1000
            else:
                predictions['lag_llama'] = 0.0
                confidences['lag_llama'] = 0.0
        except Exception as e:
            self.logger.error(f"Lag-Llama failed: {e}")
            predictions['lag_llama'] = 0.0
            confidences['lag_llama'] = 0.0
        
        # Chronos
        try:
            if price_series is not None:
                start = time.time()
                result = self.chronos_engine.predict(price_series)
                signal = self._chronos_to_signal(result, price_series[-1])
                predictions['chronos'] = signal
                confidences['chronos'] = result.confidence
                inference_times['chronos'] = (time.time() - start) * 1000
            else:
                predictions['chronos'] = 0.0
                confidences['chronos'] = 0.0
        except Exception as e:
            self.logger.error(f"Chronos failed: {e}")
            predictions['chronos'] = 0.0
            confidences['chronos'] = 0.0
        
        # TimesFM
        try:
            if price_series is not None:
                start = time.time()
                result = self.timesfm_engine.predict(
                    price_series, 
                    covariates=market_covariates
                )
                signal = self._timesfm_to_signal(result, price_series[-1])
                predictions['timesfm'] = signal
                confidences['timesfm'] = result.confidence
                inference_times['timesfm'] = (time.time() - start) * 1000
            else:
                predictions['timesfm'] = 0.0
                confidences['timesfm'] = 0.0
        except Exception as e:
            self.logger.error(f"TimesFM failed: {e}")
            predictions['timesfm'] = 0.0
            confidences['timesfm'] = 0.0
        
        return predictions, confidences, inference_times
    
    def _chronos_to_signal(self, result, current_price: float) -> float:
        """Convert Chronos forecasting result to trading signal"""
        try:
            # Get first prediction point
            predicted_price = result.point_forecast[0]
            
            # Convert to percentage change
            pct_change = (predicted_price - current_price) / current_price
            
            # Normalize to [-1, 1] signal
            return float(np.clip(pct_change * 10, -1.0, 1.0))
        except:
            return 0.0
    
    def _timesfm_to_signal(self, result, current_price: float) -> float:
        """Convert TimesFM forecasting result to trading signal"""
        try:
            # Get first prediction point
            predicted_price = result.point_forecast[0]
            
            # Convert to percentage change
            pct_change = (predicted_price - current_price) / current_price
            
            # Normalize to [-1, 1] signal
            return float(np.clip(pct_change * 10, -1.0, 1.0))
        except:
            return 0.0
    
    def _calculate_weighted_signal(self, predictions: Dict[str, float]) -> float:
        """Calculate ensemble signal using dynamic weights"""
        
        if not predictions:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            if model_name in self.weights and prediction != 0.0:
                weight = self.weights[model_name]
                weighted_sum += prediction * weight
                total_weight += weight
        
        if total_weight > 0:
            return float(np.clip(weighted_sum / total_weight, -1.0, 1.0))
        else:
            return 0.0
    
    def _calculate_agreement(self, predictions: Dict[str, float]) -> float:
        """Calculate agreement between models"""
        
        valid_predictions = [p for p in predictions.values() if p != 0.0]
        
        if len(valid_predictions) < 2:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(valid_predictions)):
            for j in range(i + 1, len(valid_predictions)):
                diff = abs(valid_predictions[i] - valid_predictions[j])
                agreement = 1.0 - min(diff / 2.0, 1.0)
                agreements.append(agreement)
        
        return float(np.mean(agreements)) if agreements else 1.0
    
    def _calculate_ensemble_confidence(self, 
                                     confidences: Dict[str, float], 
                                     agreement: float) -> float:
        """Calculate overall ensemble confidence"""
        
        valid_confidences = [c for c in confidences.values() if c > 0.0]
        
        if not valid_confidences:
            return 0.0
        
        # Base confidence from individual models
        avg_confidence = np.mean(valid_confidences)
        
        # Boost confidence with agreement
        ensemble_confidence = avg_confidence * (0.7 + 0.3 * agreement)
        
        return float(np.clip(ensemble_confidence, 0.0, 1.0))
    
    def _update_performance_tracking(self, 
                                   predictions: Dict[str, float], 
                                   confidences: Dict[str, float]):
        """Update performance tracking for all models"""
        
        for model_name in predictions:
            if model_name in self.performance_history:
                self.performance_history[model_name]['predictions'].append(predictions[model_name])
                self.performance_history[model_name]['confidence'].append(confidences.get(model_name, 0.0))
                
                # Keep only recent history
                for key in ['predictions', 'confidence']:
                    if len(self.performance_history[model_name][key]) > 1000:
                        self.performance_history[model_name][key] = \
                            self.performance_history[model_name][key][-1000:]
    
    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        
        performance_scores = {}
        
        for model_name in self.weights:
            if model_name in self.performance_history:
                history = self.performance_history[model_name]
                
                if len(history['confidence']) >= 10:
                    # Calculate recent performance
                    recent_confidence = np.mean(history['confidence'][-50:])
                    recent_predictions = history['predictions'][-50:]
                    
                    # Score based on confidence and consistency
                    consistency = 1.0 - np.std(recent_predictions) if len(recent_predictions) > 1 else 1.0
                    performance_score = 0.7 * recent_confidence + 0.3 * consistency
                    
                    performance_scores[model_name] = max(0.1, performance_score)
                else:
                    # Not enough history, keep current weight
                    performance_scores[model_name] = self.weights[model_name]
            else:
                performance_scores[model_name] = self.weights[model_name]
        
        # Normalize scores to weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for model_name in self.weights:
                self.weights[model_name] = performance_scores[model_name] / total_score
        
        self.logger.debug(f"Updated model weights: {self.weights}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        
        stats = {
            'total_predictions': self.prediction_count,
            'avg_inference_time_ms': (self.total_inference_time / max(self.prediction_count, 1)) * 1000,
            'current_weights': self.weights.copy(),
            'model_performance': {}
        }
        
        for model_name in self.weights:
            if model_name in self.performance_history:
                history = self.performance_history[model_name]
                if history['confidence']:
                    stats['model_performance'][model_name] = {
                        'avg_confidence': float(np.mean(history['confidence'])),
                        'prediction_count': len(history['predictions']),
                        'current_weight': self.weights[model_name]
                    }
                else:
                    stats['model_performance'][model_name] = {
                        'avg_confidence': 0.0,
                        'prediction_count': 0,
                        'current_weight': self.weights[model_name]
                    }
        
        return stats
    
    def reset_performance(self):
        """Reset all performance tracking"""
        for model_name in self.performance_history:
            self.performance_history[model_name] = {
                'predictions': [], 
                'accuracy': [], 
                'confidence': []
            }
        self.prediction_count = 0
        self.total_inference_time = 0.0

# Global ensemble instance
_ensemble = None

def get_ensemble(config: EnsembleConfig = None) -> ModelEnsemble:
    """Get global ensemble instance"""
    global _ensemble
    if _ensemble is None:
        _ensemble = ModelEnsemble(config)
    return _ensemble
