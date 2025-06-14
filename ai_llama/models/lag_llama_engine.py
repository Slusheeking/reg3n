"""
Lag-Llama Foundation Model Engine

Real integration with Lag-Llama for zero-shot time series forecasting.
Optimized for trading with aggressive caching and batch processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import time
import sys
import os

# Add the local lag_llama modules to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    import torch
    LAGLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Lag-Llama dependencies not available: {e}")
    LAGLLAMA_AVAILABLE = False


class LagLlamaEngine:
    """
    Production Lag-Llama engine for trading
    
    Integrates the real Lag-Llama foundation model with:
    - Aggressive caching for low latency
    - Batch processing for efficiency  
    - Trading-optimized data conversion
    - Uncertainty quantification
    """
    
    def __init__(self, 
                 ckpt_path: Optional[str] = None,
                 context_length: int = 64,
                 prediction_length: int = 5,
                 batch_size: int = 32,
                 device: str = 'auto'):
        
        if not LAGLLAMA_AVAILABLE:
            print("Warning: Lag-Llama dependencies not available. Running in fallback mode.")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not self.fallback_mode else 'cpu'
        else:
            self.device = torch.device(device) if not self.fallback_mode else 'cpu'
        
        # Model and predictor
        self.estimator = None
        self.predictor = None
        self.ckpt_path = ckpt_path
        self.is_loaded = False
        
        # Caching for performance
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize model if checkpoint provided and not in fallback mode
        if ckpt_path and not self.fallback_mode:
            self.load_model(ckpt_path)
    
    def load_model(self, ckpt_path: str):
        """Load pre-trained Lag-Llama model"""
        if self.fallback_mode:
            print("Cannot load model: running in fallback mode")
            return
            
        try:
            print(f"Loading Lag-Llama model from {ckpt_path}")
            
            self.estimator = LagLlamaEstimator(
                ckpt_path=ckpt_path,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                # Optimized for inference
                batch_size=self.batch_size,
                num_parallel_samples=100,
                # Disable augmentations for inference
                aug_prob=0.0,
                device=self.device
            )
            
            # Create predictor
            transformation = self.estimator.create_transformation()
            lightning_module = self.estimator.create_lightning_module()
            lightning_module.eval()
            
            self.predictor = self.estimator.create_predictor(
                transformation=transformation,
                module=lightning_module
            )
            
            self.is_loaded = True
            print(f"Lag-Llama model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Lag-Llama model: {e}")
            self.is_loaded = False
            self.fallback_mode = True
    
    def _fallback_prediction(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction using simple statistical methods"""
        
        if len(data) < 2:
            return self._empty_prediction()
        
        # Simple trend-based prediction
        recent_returns = np.diff(data[-10:]) / data[-10:-1] if len(data) >= 10 else np.diff(data) / data[:-1]
        
        # Calculate trend
        trend = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
        
        # Generate simple predictions
        last_price = data[-1]
        predictions = []
        
        for i in range(self.prediction_length):
            # Add some random walk with drift
            random_component = np.random.normal(0, volatility * 0.5)
            next_price = last_price * (1 + trend + random_component)
            predictions.append(next_price)
            last_price = next_price
        
        # Convert to returns
        current_price = data[-1]
        prediction_returns = [(p - current_price) / current_price for p in predictions]
        
        return {
            'mean': prediction_returns,
            'std': [volatility] * self.prediction_length,
            'quantiles': {
                '0.1': [r - 1.28 * volatility for r in prediction_returns],
                '0.25': [r - 0.67 * volatility for r in prediction_returns],
                '0.5': prediction_returns,
                '0.75': [r + 0.67 * volatility for r in prediction_returns],
                '0.9': [r + 1.28 * volatility for r in prediction_returns],
            },
            'uncertainty': float(volatility),
            'confidence': float(1.0 / (1.0 + volatility)),
            'direction': float(np.sign(trend)),
            'magnitude': float(abs(trend)),
            'trend': float(trend),
            'fallback_mode': True
        }
    
    def _numpy_to_gluonts_dataset(self, 
                                  data: np.ndarray, 
                                  freq: str = '1min',
                                  start_time: Optional[pd.Timestamp] = None) -> ListDataset:
        """Convert numpy array to GluonTS dataset format"""
        
        if start_time is None:
            start_time = pd.Timestamp('2024-01-01 09:30:00')
        
        # Create dataset entry
        dataset_entry = {
            FieldName.TARGET: data.tolist(),
            FieldName.START: start_time
        }
        
        return ListDataset([dataset_entry], freq=freq)
    
    def _extract_prediction_results(self, forecast) -> Dict[str, Any]:
        """Extract results from GluonTS forecast object"""
        try:
            # Get samples
            samples = forecast.samples  # Shape: [num_samples, prediction_length]
            
            # Calculate statistics
            mean_prediction = np.mean(samples, axis=0)
            std_prediction = np.std(samples, axis=0)
            
            # Calculate quantiles for uncertainty
            quantiles = {
                '0.1': np.quantile(samples, 0.1, axis=0),
                '0.25': np.quantile(samples, 0.25, axis=0),
                '0.5': np.quantile(samples, 0.5, axis=0),  # median
                '0.75': np.quantile(samples, 0.75, axis=0),
                '0.9': np.quantile(samples, 0.9, axis=0),
            }
            
            # Calculate overall uncertainty (average std across prediction horizon)
            uncertainty = np.mean(std_prediction)
            
            # Calculate confidence (inverse of uncertainty, normalized)
            confidence = 1.0 / (1.0 + uncertainty)
            
            return {
                'mean': mean_prediction.tolist(),
                'std': std_prediction.tolist(),
                'quantiles': {k: v.tolist() for k, v in quantiles.items()},
                'uncertainty': float(uncertainty),
                'confidence': float(confidence),
                'direction': float(np.sign(mean_prediction[0])),  # Next step direction
                'magnitude': float(abs(mean_prediction[0])),      # Next step magnitude
                'trend': float(np.mean(np.diff(mean_prediction))), # Overall trend
                'fallback_mode': False
            }
            
        except Exception as e:
            print(f"Error extracting prediction results: {e}")
            return self._empty_prediction()
    
    def _empty_prediction(self) -> Dict[str, Any]:
        """Return empty prediction when model fails"""
        return {
            'mean': [0.0] * self.prediction_length,
            'std': [0.0] * self.prediction_length,
            'quantiles': {
                '0.1': [0.0] * self.prediction_length,
                '0.25': [0.0] * self.prediction_length,
                '0.5': [0.0] * self.prediction_length,
                '0.75': [0.0] * self.prediction_length,
                '0.9': [0.0] * self.prediction_length,
            },
            'uncertainty': 1.0,
            'confidence': 0.0,
            'direction': 0.0,
            'magnitude': 0.0,
            'trend': 0.0,
            'fallback_mode': True
        }
    
    def predict(self, 
                data: np.ndarray, 
                symbol: str = 'UNKNOWN',
                timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate prediction for single time series
        
        Args:
            data: Historical price data (numpy array)
            symbol: Trading symbol for caching
            timestamp: Current timestamp for caching
            
        Returns:
            Prediction results with uncertainty quantification
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = f"{symbol}_{len(data)}_{timestamp}"
        if cache_key in self.prediction_cache:
            cache_entry = self.prediction_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                self.cache_hits += 1
                return cache_entry['prediction']
            else:
                # Remove stale entry
                del self.prediction_cache[cache_key]
        
        self.cache_misses += 1
        
        # Use fallback if model not loaded or in fallback mode
        if self.fallback_mode or not self.is_loaded:
            results = self._fallback_prediction(data)
        else:
            try:
                # Ensure minimum length
                if len(data) < self.context_length:
                    # Pad with the first value
                    padding = np.full(self.context_length - len(data), data[0])
                    data = np.concatenate([padding, data])
                
                # Use only the most recent context_length points
                data = data[-self.context_length:]
                
                # Convert to GluonTS format
                dataset = self._numpy_to_gluonts_dataset(data)
                
                # Generate predictions
                forecasts = list(self.predictor.predict(dataset))
                
                if len(forecasts) > 0:
                    results = self._extract_prediction_results(forecasts[0])
                else:
                    results = self._fallback_prediction(data)
                    
            except Exception as e:
                print(f"Error in Lag-Llama prediction: {e}, using fallback")
                results = self._fallback_prediction(data)
        
        # Add timing info
        inference_time = (time.perf_counter() - start_time) * 1000
        results['inference_time_ms'] = inference_time
        results['cache_hit'] = False
        
        # Cache result
        if timestamp:
            self.prediction_cache[cache_key] = {
                'prediction': results,
                'timestamp': time.time()
            }
            
            # Clean cache if too large
            if len(self.prediction_cache) > self.max_cache_size:
                oldest_key = min(self.prediction_cache.keys(),
                               key=lambda k: self.prediction_cache[k]['timestamp'])
                del self.prediction_cache[oldest_key]
        
        # Track performance
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        return results
    
    def get_trading_signal(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert Lag-Llama prediction to trading signals
        
        Args:
            prediction: Prediction results from predict()
            
        Returns:
            Trading signals and confidence
        """
        try:
            direction = prediction.get('direction', 0.0)
            confidence = prediction.get('confidence', 0.0)
            uncertainty = prediction.get('uncertainty', 1.0)
            magnitude = prediction.get('magnitude', 0.0)
            trend = prediction.get('trend', 0.0)
            
            # Base signal from direction and confidence
            base_signal = direction * confidence
            
            # Adjust signal strength based on uncertainty
            uncertainty_factor = 1.0 / (1.0 + uncertainty)
            adjusted_signal = base_signal * uncertainty_factor
            
            # Trend confirmation
            trend_confirmation = 1.0 if (direction * trend) > 0 else 0.5
            final_signal = adjusted_signal * trend_confirmation
            
            return {
                'signal': float(np.clip(final_signal, -1.0, 1.0)),
                'confidence': float(confidence),
                'uncertainty': float(uncertainty),
                'direction': float(direction),
                'magnitude': float(magnitude),
                'trend': float(trend),
                'signal_strength': float(abs(final_signal)),
                'fallback_mode': prediction.get('fallback_mode', True)
            }
            
        except Exception as e:
            print(f"Error generating trading signal: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'uncertainty': 1.0,
                'direction': 0.0,
                'magnitude': 0.0,
                'trend': 0.0,
                'signal_strength': 0.0,
                'fallback_mode': True
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {'no_data': True}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'p50_inference_time_ms': np.percentile(self.inference_times, 50),
            'p95_inference_time_ms': np.percentile(self.inference_times, 95),
            'p99_inference_time_ms': np.percentile(self.inference_times, 99),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            'cache_size': len(self.prediction_cache),
            'total_predictions': self.cache_hits + self.cache_misses,
            'model_loaded': self.is_loaded,
            'fallback_mode': self.fallback_mode,
            'device': str(self.device)
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def warmup(self, dummy_data: Optional[np.ndarray] = None):
        """Warmup model for faster inference"""
        print("Warming up Lag-Llama model...")
        
        if dummy_data is None:
            # Create dummy data
            dummy_data = np.random.randn(self.context_length) * 0.01 + 100.0
        
        # Run a few dummy predictions
        for i in range(3):
            self.predict(dummy_data, symbol='WARMUP', timestamp=time.time())
        
        print("Model warmup complete")
