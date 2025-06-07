#!/usr/bin/env python3

import os
import sys
import json
import yaml
import time
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

# ML Libraries

# Import unified system logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.system_logger import get_system_logger

# Initialize component logger
logger = get_system_logger("ensemble_manager")

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'ml_models.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

class MarketRegime(Enum):
    HIGH_VOLATILITY = "high_volatility"
    MEDIUM_VOLATILITY = "medium_volatility" 
    BALANCED_MARKET = "balanced_market"
    TRENDING_MOMENTUM = "trending_momentum"
    STRONG_MOMENTUM = "strong_momentum"

@dataclass
class ModelPrediction:
    model_name: str
    prediction: float
    confidence: float
    inference_time: float
    timestamp: float

@dataclass
class EnsemblePrediction:
    final_prediction: float
    regime: MarketRegime
    volatility_score: float
    momentum_score: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    confidence: float
    processing_time: float

class EnsembleManager:
    """
    Advanced ensemble manager that combines all ML4T models
    Features adaptive weighting, regime-aware predictions, and online learning
    """
    
    def __init__(self, model_server, config: Dict = None):
        self.model_server = model_server
        # Load config from YAML if not provided
        yaml_config = CONFIG.get('ensemble', {})
        self.config = config or yaml_config or self._get_default_config()
        
        logger.info("Initializing Ensemble Manager", extra={
            "component": "ensemble_manager",
            "action": "initialization_start",
            "config_source": "yaml" if yaml_config else "default",
            "config_keys": list(self.config.keys())
        })
        
        # Model organization
        self.model_groups = {
            'regime_models': ['lightgbm_regime', 'xgboost_ensemble', 'logistic_direction'],
            'volatility_models': ['lightgbm_volatility', 'garch_volatility', 'lstm_returns'],
            'momentum_models': ['lightgbm_momentum', 'cnn_patterns', 'feedforward_nn'],
            'risk_models': ['var_multivariate', 'ridge_factors', 'dqn_agent'],
            'meta_models': ['ensemble_meta']
        }
        
        # Adaptive weights (start equal, evolve based on performance)
        self.model_weights = self._initialize_weights()
        
        # Performance tracking
        self.performance_history = {model: deque(maxlen=1000) for model in self._get_all_models()}
        self.regime_history = deque(maxlen=100)
        
        # Regime allocation ratios (from strategy.md)
        self.regime_allocations = {
            MarketRegime.HIGH_VOLATILITY: [0.8, 0.2],      # 80% volatility, 20% momentum
            MarketRegime.MEDIUM_VOLATILITY: [0.6, 0.4],    # 60% volatility, 40% momentum
            MarketRegime.BALANCED_MARKET: [0.5, 0.5],      # 50% volatility, 50% momentum
            MarketRegime.TRENDING_MOMENTUM: [0.3, 0.7],    # 30% volatility, 70% momentum
            MarketRegime.STRONG_MOMENTUM: [0.2, 0.8]       # 20% volatility, 80% momentum
        }
        
        # Meta-learning components
        self.meta_learner = None
        self.feature_importance = {}
        
        # Performance metrics
        self.ensemble_metrics = {
            'total_predictions': 0,
            'avg_processing_time': 0,
            'regime_accuracy': 0,
            'prediction_accuracy': 0
        }
        
        logger.info("Ensemble Manager initialization complete", extra={
            "component": "ensemble_manager",
            "action": "initialization_complete",
            "model_groups": {group: len(models) for group, models in self.model_groups.items()},
            "total_models": len(self._get_all_models()),
            "regime_allocations": self.regime_allocations,
            "performance_thresholds": {
                "regime_confidence_threshold": self.config.get('regime_confidence_threshold', 0.7),
                "prediction_confidence_threshold": self.config.get('prediction_confidence_threshold', 0.6)
            },
            "meta_learning_enabled": self.config.get('meta_learning_enabled', True)
        })
    
    def _get_default_config(self) -> Dict:
        """Default configuration for ensemble manager"""
        return {
            'enabled': True,
            'combination_method': 'weighted_average',
            'confidence_threshold': 0.6,
            'min_models_agreement': 2,
            'adaptive_learning_rate': 0.01,
            'weight_decay': 0.99,
            'min_weight': 0.01,
            'max_weight': 0.5,
            'regime_confidence_threshold': 0.7,
            'prediction_confidence_threshold': 0.6,
            'performance_window': 100,
            'meta_learning_enabled': True
        }
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize equal weights for all models"""
        all_models = self._get_all_models()
        equal_weight = 1.0 / len(all_models)
        return {model: equal_weight for model in all_models}
    
    def _get_all_models(self) -> List[str]:
        """Get list of all model names"""
        all_models = []
        for group in self.model_groups.values():
            all_models.extend(group)
        return list(set(all_models))  # Remove duplicates
    
    async def predict(self, features: np.ndarray, symbol: str = None) -> EnsemblePrediction:
        """
        Main ensemble prediction method
        Target: <150ms total processing time
        """
        start_time = time.time()
        
        logger.debug("Starting ensemble prediction", extra={
            "component": "ensemble_manager",
            "action": "predict_start",
            "symbol": symbol,
            "feature_count": len(features),
            "target_time_ms": 150
        })
        
        try:
            # Step 1: Get predictions from all models (parallel)
            model_start = time.time()
            individual_predictions = await self.model_server.predict_all_models(features, symbol)
            model_time = (time.time() - model_start) * 1000
            
            logger.debug("Model predictions complete", extra={
                "component": "ensemble_manager",
                "action": "model_predictions",
                "symbol": symbol,
                "model_count": len(individual_predictions),
                "model_time_ms": model_time,
                "predictions": individual_predictions
            })
            
            # Step 2: Detect market regime
            regime_start = time.time()
            regime = self._detect_market_regime(individual_predictions, features)
            regime_time = (time.time() - regime_start) * 1000
            
            # Step 3: Compute category scores
            score_start = time.time()
            volatility_score = self._compute_volatility_score(individual_predictions)
            momentum_score = self._compute_momentum_score(individual_predictions)
            score_time = (time.time() - score_start) * 1000
            
            # Step 4: Apply regime-based allocation
            regime_allocation = self.regime_allocations[regime]
            
            # Step 5: Combine predictions using adaptive weights
            combine_start = time.time()
            final_prediction = self._combine_predictions(
                individual_predictions,
                regime_allocation,
                volatility_score,
                momentum_score
            )
            combine_time = (time.time() - combine_start) * 1000
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(individual_predictions, regime)
            
            # Step 7: Update performance tracking
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            self._update_metrics(processing_time)
            
            # Create ensemble prediction
            ensemble_pred = EnsemblePrediction(
                final_prediction=final_prediction,
                regime=regime,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                individual_predictions=individual_predictions,
                model_weights=self.model_weights.copy(),
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info("Ensemble prediction complete", extra={
                "component": "ensemble_manager",
                "action": "predict_complete",
                "symbol": symbol,
                "final_prediction": final_prediction,
                "confidence": confidence,
                "regime": regime.value,
                "volatility_score": volatility_score,
                "momentum_score": momentum_score,
                "model_time_ms": model_time,
                "regime_time_ms": regime_time,
                "score_time_ms": score_time,
                "combine_time_ms": combine_time,
                "total_time_ms": processing_time_ms,
                "target_time_ms": 150,
                "within_target": processing_time_ms < 150,
                "model_count": len(individual_predictions)
            })
            
            return ensemble_pred
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error("Ensemble prediction failed", extra={
                "component": "ensemble_manager",
                "action": "predict_error",
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time_ms
            })
            return self._create_fallback_prediction()
    
    def _detect_market_regime(self, predictions: Dict[str, float], features: np.ndarray) -> MarketRegime:
        """
        Detect current market regime using regime models
        """
        try:
            # Get regime model predictions
            regime_scores = {}
            
            for model_name in self.model_groups['regime_models']:
                if model_name in predictions:
                    regime_scores[model_name] = predictions[model_name]
            
            if not regime_scores:
                return MarketRegime.BALANCED_MARKET
            
            # Combine regime predictions (weighted average)
            regime_prediction = 0
            total_weight = 0
            
            for model_name, score in regime_scores.items():
                weight = self.model_weights.get(model_name, 0.33)
                regime_prediction += score * weight
                total_weight += weight
            
            if total_weight > 0:
                regime_prediction /= total_weight
            
            # Map prediction to regime (based on thresholds)
            if regime_prediction > 0.8:
                regime = MarketRegime.STRONG_MOMENTUM
            elif regime_prediction > 0.6:
                regime = MarketRegime.TRENDING_MOMENTUM
            elif regime_prediction > 0.4:
                regime = MarketRegime.BALANCED_MARKET
            elif regime_prediction > 0.2:
                regime = MarketRegime.MEDIUM_VOLATILITY
            else:
                regime = MarketRegime.HIGH_VOLATILITY
            
            # Update regime history
            self.regime_history.append(regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegime.BALANCED_MARKET
    
    def _compute_volatility_score(self, predictions: Dict[str, float]) -> float:
        """Compute volatility score from volatility models"""
        try:
            volatility_scores = []
            
            for model_name in self.model_groups['volatility_models']:
                if model_name in predictions:
                    score = predictions[model_name]
                    weight = self.model_weights.get(model_name, 0.33)
                    volatility_scores.append(score * weight)
            
            if volatility_scores:
                return np.mean(volatility_scores)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Volatility score error: {e}")
            return 0.5
    
    def _compute_momentum_score(self, predictions: Dict[str, float]) -> float:
        """Compute momentum score from momentum models"""
        try:
            momentum_scores = []
            
            for model_name in self.model_groups['momentum_models']:
                if model_name in predictions:
                    score = predictions[model_name]
                    weight = self.model_weights.get(model_name, 0.33)
                    momentum_scores.append(score * weight)
            
            if momentum_scores:
                return np.mean(momentum_scores)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Momentum score error: {e}")
            return 0.5
    
    def _combine_predictions(self, predictions: Dict[str, float], regime_allocation: List[float], 
                           volatility_score: float, momentum_score: float) -> float:
        """
        Combine all predictions using regime-based allocation and adaptive weights
        """
        try:
            # Apply regime allocation
            vol_weight, mom_weight = regime_allocation
            
            # Base prediction from volatility and momentum scores
            base_prediction = (volatility_score * vol_weight + momentum_score * mom_weight)
            
            # Meta-model enhancement (if available)
            meta_prediction = predictions.get('ensemble_meta', base_prediction)
            
            # Risk model adjustment
            risk_adjustment = self._compute_risk_adjustment(predictions)
            
            # Combine with adaptive weighting
            final_prediction = (
                base_prediction * 0.6 +      # 60% from regime-based allocation
                meta_prediction * 0.3 +      # 30% from meta-model
                risk_adjustment * 0.1        # 10% from risk adjustment
            )
            
            # Ensure prediction is in reasonable range
            final_prediction = np.clip(final_prediction, -1.0, 1.0)
            
            return float(final_prediction)
            
        except Exception as e:
            logger.error(f"Prediction combination error: {e}")
            return 0.0
    
    def _compute_risk_adjustment(self, predictions: Dict[str, float]) -> float:
        """Compute risk adjustment from risk models"""
        try:
            risk_scores = []
            
            for model_name in self.model_groups['risk_models']:
                if model_name in predictions:
                    score = predictions[model_name]
                    weight = self.model_weights.get(model_name, 0.33)
                    risk_scores.append(score * weight)
            
            if risk_scores:
                risk_adjustment = np.mean(risk_scores)
                # Risk adjustment should be conservative (reduce extreme predictions)
                return risk_adjustment * 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Risk adjustment error: {e}")
            return 0.0
    
    def _calculate_confidence(self, predictions: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate confidence in the ensemble prediction"""
        try:
            # Confidence based on prediction agreement
            pred_values = list(predictions.values())
            if len(pred_values) < 2:
                return 0.5
            
            # Calculate standard deviation of predictions
            pred_std = np.std(pred_values)
            
            # Lower std = higher confidence
            agreement_confidence = 1.0 / (1.0 + pred_std)
            
            # Confidence based on regime stability
            regime_confidence = self._calculate_regime_confidence()
            
            # Confidence based on recent performance
            performance_confidence = self._calculate_performance_confidence()
            
            # Combine confidences
            total_confidence = (
                agreement_confidence * 0.4 +
                regime_confidence * 0.3 +
                performance_confidence * 0.3
            )
            
            return float(np.clip(total_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _calculate_regime_confidence(self) -> float:
        """Calculate confidence based on regime stability"""
        try:
            if len(self.regime_history) < 5:
                return 0.5
            
            # Check regime consistency over last 5 predictions
            recent_regimes = list(self.regime_history)[-5:]
            most_common = max(set(recent_regimes), key=recent_regimes.count)
            consistency = recent_regimes.count(most_common) / len(recent_regimes)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Regime confidence error: {e}")
            return 0.5
    
    def _calculate_performance_confidence(self) -> float:
        """Calculate confidence based on recent model performance"""
        try:
            # Get recent performance for all models
            recent_performances = []
            
            for model_name in self._get_all_models():
                if model_name in self.performance_history:
                    history = self.performance_history[model_name]
                    if len(history) > 0:
                        recent_perf = np.mean(list(history)[-10:])  # Last 10 predictions
                        recent_performances.append(recent_perf)
            
            if recent_performances:
                avg_performance = np.mean(recent_performances)
                return float(np.clip(avg_performance, 0.0, 1.0))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Performance confidence error: {e}")
            return 0.5
    
    async def update_performance(self, prediction: EnsemblePrediction, actual_outcome: float):
        """
        Update model performance based on actual outcomes
        Implements online learning for adaptive weights
        """
        try:
            # Calculate prediction error
            prediction_error = abs(prediction.final_prediction - actual_outcome)
            
            # Update individual model performance
            for model_name, model_pred in prediction.individual_predictions.items():
                model_error = abs(model_pred - actual_outcome)
                model_accuracy = 1.0 / (1.0 + model_error)  # Convert error to accuracy
                
                # Update performance history
                if model_name in self.performance_history:
                    self.performance_history[model_name].append(model_accuracy)
            
            # Update adaptive weights
            await self._update_adaptive_weights()
            
            # Update ensemble metrics
            ensemble_accuracy = 1.0 / (1.0 + prediction_error)
            self._update_ensemble_metrics(ensemble_accuracy)
            
        except Exception as e:
            logger.error(f"Performance update error: {e}")
    
    async def _update_adaptive_weights(self):
        """Update model weights based on recent performance"""
        try:
            learning_rate = self.config['adaptive_learning_rate']
            weight_decay = self.config['weight_decay']
            min_weight = self.config['min_weight']
            max_weight = self.config['max_weight']
            
            # Calculate new weights based on performance
            new_weights = {}
            total_performance = 0
            
            for model_name in self._get_all_models():
                if model_name in self.performance_history:
                    history = self.performance_history[model_name]
                    if len(history) > 0:
                        # Use recent performance (last 50 predictions)
                        recent_performance = np.mean(list(history)[-50:])
                        total_performance += recent_performance
                        new_weights[model_name] = recent_performance
                    else:
                        new_weights[model_name] = 0.5  # Default
                else:
                    new_weights[model_name] = 0.5
            
            # Normalize weights
            if total_performance > 0:
                for model_name in new_weights:
                    new_weights[model_name] /= total_performance
            
            # Apply learning rate and constraints
            for model_name in self.model_weights:
                if model_name in new_weights:
                    # Exponential moving average update
                    old_weight = self.model_weights[model_name]
                    new_weight = new_weights[model_name]
                    
                    updated_weight = (
                        old_weight * (1 - learning_rate) + 
                        new_weight * learning_rate
                    )
                    
                    # Apply weight decay and constraints
                    updated_weight *= weight_decay
                    updated_weight = np.clip(updated_weight, min_weight, max_weight)
                    
                    self.model_weights[model_name] = updated_weight
            
            # Renormalize to ensure weights sum to 1
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
        except Exception as e:
            logger.error(f"Adaptive weights update error: {e}")
    
    def _update_metrics(self, processing_time: float):
        """Update ensemble performance metrics"""
        self.ensemble_metrics['total_predictions'] += 1
        
        # Update average processing time
        total_preds = self.ensemble_metrics['total_predictions']
        current_avg = self.ensemble_metrics['avg_processing_time']
        
        self.ensemble_metrics['avg_processing_time'] = (
            (current_avg * (total_preds - 1) + processing_time) / total_preds
        )
    
    def _update_ensemble_metrics(self, accuracy: float):
        """Update ensemble accuracy metrics"""
        total_preds = self.ensemble_metrics['total_predictions']
        current_acc = self.ensemble_metrics['prediction_accuracy']
        
        self.ensemble_metrics['prediction_accuracy'] = (
            (current_acc * (total_preds - 1) + accuracy) / total_preds
        )
    
    def _create_fallback_prediction(self) -> EnsemblePrediction:
        """Create fallback prediction when ensemble fails"""
        return EnsemblePrediction(
            final_prediction=0.0,
            regime=MarketRegime.BALANCED_MARKET,
            volatility_score=0.5,
            momentum_score=0.5,
            individual_predictions={},
            model_weights=self.model_weights.copy(),
            confidence=0.1,
            processing_time=0.001
        )
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and performance"""
        return {
            'model_weights': self.model_weights.copy(),
            'performance_metrics': self.ensemble_metrics.copy(),
            'regime_history': list(self.regime_history)[-10:],  # Last 10 regimes
            'model_count': len(self.model_weights),
            'total_predictions': self.ensemble_metrics['total_predictions'],
            'avg_processing_time': self.ensemble_metrics['avg_processing_time'],
            'prediction_accuracy': self.ensemble_metrics['prediction_accuracy']
        }
    
    def save_ensemble_state(self, filepath: str):
        """Save ensemble state for persistence"""
        try:
            state = {
                'model_weights': self.model_weights,
                'performance_history': {
                    model: list(history) for model, history in self.performance_history.items()
                },
                'ensemble_metrics': self.ensemble_metrics,
                'regime_history': list(self.regime_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving ensemble state: {e}")
    
    def load_ensemble_state(self, filepath: str):
        """Load ensemble state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.model_weights = state.get('model_weights', self.model_weights)
            
            # Restore performance history
            perf_history = state.get('performance_history', {})
            for model, history in perf_history.items():
                if model in self.performance_history:
                    self.performance_history[model] = deque(history, maxlen=1000)
            
            self.ensemble_metrics = state.get('ensemble_metrics', self.ensemble_metrics)
            
            # Restore regime history
            regime_history = state.get('regime_history', [])
            self.regime_history = deque(
                [MarketRegime(r) for r in regime_history], 
                maxlen=100
            )
            
        except Exception as e:
            logger.error(f"Error loading ensemble state: {e}")