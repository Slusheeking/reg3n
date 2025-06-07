#!/usr/bin/env python3

import numpy as np
import asyncio
import os
import yaml
import sys
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import pickle
from cachetools import TTLCache
import time

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import tensorflow as tf
import torch
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from arch import arch_model
import statsmodels.api as sm

# Import unified system logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.system_logger import get_system_logger

# Initialize component logger
logger = get_system_logger("model_server")

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

class OptimizedModelServer:
    """
    High-performance model server supporting all ML4T models
    Target: <100ms parallel inference across all models
    """
    
    def __init__(self, model_dir: str = None, max_workers: int = None):
        logger.info("Initializing Optimized Model Server", extra={
            "component": "model_server",
            "action": "initialization_start",
            "model_dir": model_dir,
            "max_workers": max_workers
        })
        
        # Load configuration
        server_config = CONFIG.get('model_server', {})
        storage_config = CONFIG.get('storage', {})
        gpu_config = CONFIG.get('gpu', {})
        
        self.model_dir = Path(model_dir or storage_config.get('model_path', 'models'))
        self.max_workers = max_workers or server_config.get('workers', 8)
        
        # Thread pools for different model types
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gpu_executor = ThreadPoolExecutor(max_workers=2)
        
        # Caching systems from config
        cache_size = server_config.get('model_cache_size', 5)
        self.prediction_cache = TTLCache(maxsize=5000, ttl=60)  # 1-minute cache
        self.model_cache = TTLCache(maxsize=cache_size * 10, ttl=3600)  # 1-hour cache
        
        # Model registry
        self.models = {}
        self.model_metadata = {}
        
        # GPU memory management
        self.gpu_enabled = gpu_config.get('enabled', False)
        self.gpu_available = self._check_gpu_availability() and self.gpu_enabled
        if self.gpu_available:
            self._configure_gpu_memory()
        
        # Performance tracking
        self.inference_times = {}
        self.prediction_counts = {}
        
        # Performance targets from config
        performance_config = CONFIG.get('performance', {})
        self.target_inference_ms = performance_config.get('model_inference_ms', 10)
        self.target_ensemble_ms = performance_config.get('ensemble_prediction_ms', 25)
        
        logger.info("Optimized Model Server initialization complete", extra={
            "component": "model_server",
            "action": "initialization_complete",
            "model_dir": str(self.model_dir),
            "max_workers": self.max_workers,
            "gpu_enabled": self.gpu_enabled,
            "gpu_available": self.gpu_available,
            "target_inference_ms": self.target_inference_ms,
            "target_ensemble_ms": self.target_ensemble_ms,
            "cache_sizes": {
                "prediction_cache": self.prediction_cache.maxsize,
                "model_cache": self.model_cache.maxsize
            }
        })
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for TensorFlow and PyTorch"""
        try:
            tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
            torch_gpu = torch.cuda.is_available()
            return tf_gpu or torch_gpu
        except:
            return False
    
    def _configure_gpu_memory(self):
        """Configure GPU memory growth to prevent OOM"""
        try:
            # TensorFlow GPU config
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # PyTorch GPU config
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"GPU configuration warning: {e}")
    
    async def load_all_models(self):
        """Load all pretrained models from ML4T repository"""
        logger.info("Loading all ML models", extra={
            "component": "model_server",
            "action": "load_models_start",
            "model_dir": str(self.model_dir),
            "max_workers": self.max_workers,
            "gpu_enabled": self.gpu_available
        })
        
        start_time = time.time()
        
        model_loaders = [
            self._load_gradient_boosting_models(),
            self._load_time_series_models(),
            self._load_deep_learning_models(),
            self._load_linear_models(),
            self._load_ensemble_models()
        ]
        
        await asyncio.gather(*model_loaders)
        
        load_time = time.time() - start_time
        
        logger.info("All ML models loaded successfully", extra={
            "component": "model_server",
            "action": "load_models_complete",
            "models_loaded": len(self.models),
            "model_types": list(set(meta.get('type', 'unknown') for meta in self.model_metadata.values())),
            "load_time_seconds": load_time,
            "gpu_models": len([m for m in self.model_metadata.values() if m.get('device') == 'gpu']),
            "cpu_models": len([m for m in self.model_metadata.values() if m.get('device') == 'cpu'])
        })
    
    async def _load_gradient_boosting_models(self):
        """Load LightGBM, XGBoost, CatBoost models"""
        try:
            # LightGBM models
            lgb_models = {
                'lightgbm_regime': 'regime_detector.txt',
                'lightgbm_volatility': 'volatility_scorer.txt', 
                'lightgbm_momentum': 'momentum_scorer.txt',
                'lightgbm_intraday': 'intraday_predictor.txt'
            }
            
            for name, filename in lgb_models.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    self.models[name] = lgb.Booster(model_file=str(model_path))
                    self.model_metadata[name] = {
                        'type': 'lightgbm',
                        'inference_time': 0.01,  # 10ms target
                        'device': 'cpu'
                    }
                else:
                    # Create and train basic model if not exists
                    self.models[name] = self._create_default_lightgbm()
                    self.model_metadata[name] = {
                        'type': 'lightgbm',
                        'inference_time': 0.01,
                        'device': 'cpu',
                        'status': 'default'
                    }
            
            # XGBoost models
            xgb_models = {
                'xgboost_ensemble': 'xgboost_meta.json',
                'xgboost_backup': 'xgboost_backup.json'
            }
            
            for name, filename in xgb_models.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    self.models[name] = xgb.Booster()
                    self.models[name].load_model(str(model_path))
                else:
                    self.models[name] = self._create_default_xgboost()
                
                self.model_metadata[name] = {
                    'type': 'xgboost',
                    'inference_time': 0.015,  # 15ms target
                    'device': 'cpu'
                }
            
            # CatBoost model
            cb_path = self.model_dir / 'catboost_model.cbm'
            if cb_path.exists():
                self.models['catboost_backup'] = cb.CatBoostRegressor().load_model(str(cb_path))
            else:
                self.models['catboost_backup'] = self._create_default_catboost()
            
            self.model_metadata['catboost_backup'] = {
                'type': 'catboost',
                'inference_time': 0.02,  # 20ms target
                'device': 'cpu'
            }
            
        except Exception as e:
            logger.error(f"Error loading gradient boosting models: {e}")
    
    async def _load_time_series_models(self):
        """Load GARCH, ARIMA, VAR models"""
        try:
            # GARCH volatility model
            garch_path = self.model_dir / 'garch_volatility.pkl'
            if garch_path.exists():
                with open(garch_path, 'rb') as f:
                    self.models['garch_volatility'] = pickle.load(f)
            else:
                self.models['garch_volatility'] = self._create_default_garch()
            
            # ARIMA trend model
            arima_path = self.model_dir / 'arima_trend.pkl'
            if arima_path.exists():
                with open(arima_path, 'rb') as f:
                    self.models['arima_trend'] = pickle.load(f)
            else:
                self.models['arima_trend'] = self._create_default_arima()
            
            # VAR multivariate model
            var_path = self.model_dir / 'var_multivariate.pkl'
            if var_path.exists():
                with open(var_path, 'rb') as f:
                    self.models['var_multivariate'] = pickle.load(f)
            else:
                self.models['var_multivariate'] = self._create_default_var()
            
            # Update metadata
            for model_name in ['garch_volatility', 'arima_trend', 'var_multivariate']:
                self.model_metadata[model_name] = {
                    'type': 'time_series',
                    'inference_time': 0.005,  # 5ms target
                    'device': 'cpu'
                }
                
        except Exception as e:
            logger.error(f"Error loading time series models: {e}")
    
    async def _load_deep_learning_models(self):
        """Load TensorFlow and PyTorch models"""
        try:
            # TensorFlow models
            tf_models = {
                'lstm_returns': 'lstm_returns.h5',
                'cnn_patterns': 'cnn_patterns.h5',
                'feedforward_nn': 'feedforward_nn.h5',
                'dqn_agent': 'dqn_trading_agent.h5'
            }
            
            for name, filename in tf_models.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    self.models[name] = tf.keras.models.load_model(str(model_path))
                else:
                    self.models[name] = self._create_default_tensorflow_model(name)
                
                self.model_metadata[name] = {
                    'type': 'tensorflow',
                    'inference_time': 0.05,  # 50ms target
                    'device': 'gpu' if self.gpu_available else 'cpu'
                }
            
            # PyTorch models (if needed)
            # Can be added here for specific PyTorch implementations
            
        except Exception as e:
            logger.error(f"Error loading deep learning models: {e}")
    
    async def _load_linear_models(self):
        """Load scikit-learn linear models"""
        try:
            linear_models = {
                'ridge_factors': 'ridge_factors.pkl',
                'lasso_selection': 'lasso_selection.pkl',
                'logistic_direction': 'logistic_direction.pkl'
            }
            
            for name, filename in linear_models.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(str(model_path))
                else:
                    self.models[name] = self._create_default_linear_model(name)
                
                self.model_metadata[name] = {
                    'type': 'linear',
                    'inference_time': 0.001,  # 1ms target
                    'device': 'cpu'
                }
                
        except Exception as e:
            logger.error(f"Error loading linear models: {e}")
    
    async def _load_ensemble_models(self):
        """Load ensemble meta-models"""
        try:
            # Meta-ensemble model
            meta_path = self.model_dir / 'ensemble_meta.txt'
            if meta_path.exists():
                self.models['ensemble_meta'] = lgb.Booster(model_file=str(meta_path))
            else:
                self.models['ensemble_meta'] = self._create_default_lightgbm()
            
            self.model_metadata['ensemble_meta'] = {
                'type': 'ensemble',
                'inference_time': 0.01,  # 10ms target
                'device': 'cpu'
            }
            
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
    
    async def predict_all_models(self, features: np.ndarray, symbol: str = None) -> Dict[str, float]:
        """
        Parallel prediction across all models
        Target: <100ms total time
        """
        start_time = time.time()
        
        logger.debug("Starting parallel model predictions", extra={
            "component": "model_server",
            "action": "predict_all_start",
            "symbol": symbol,
            "feature_count": len(features),
            "target_time_ms": 100,
            "total_models": len(self.models)
        })
        
        # Check cache
        cache_key = self._generate_prediction_cache_key(features, symbol)
        if cache_key in self.prediction_cache:
            logger.debug("Predictions retrieved from cache", extra={
                "component": "model_server",
                "action": "cache_hit",
                "symbol": symbol,
                "cache_key": cache_key
            })
            return self.prediction_cache[cache_key]
        
        # Group models by execution type
        cpu_models = []
        gpu_models = []
        
        for name, metadata in self.model_metadata.items():
            if metadata['device'] == 'gpu':
                gpu_models.append(name)
            else:
                cpu_models.append(name)
        
        logger.debug("Model execution grouping", extra={
            "component": "model_server",
            "action": "model_grouping",
            "symbol": symbol,
            "cpu_models": len(cpu_models),
            "gpu_models": len(gpu_models),
            "cpu_model_names": cpu_models,
            "gpu_model_names": gpu_models
        })
        
        # Execute predictions in parallel
        execution_start = time.time()
        tasks = []
        
        # CPU models
        for model_name in cpu_models:
            task = self.cpu_executor.submit(self._predict_single_model, model_name, features)
            tasks.append((model_name, task))
        
        # GPU models (batched)
        if gpu_models:
            gpu_task = self.gpu_executor.submit(self._predict_gpu_batch, gpu_models, features)
            tasks.append(('gpu_batch', gpu_task))
        
        # Collect results
        predictions = {}
        timeouts = 0
        errors = 0
        
        for identifier, task in tasks:
            try:
                if identifier == 'gpu_batch':
                    gpu_results = task.result(timeout=0.08)  # 80ms timeout
                    predictions.update(gpu_results)
                else:
                    result = task.result(timeout=0.05)  # 50ms timeout
                    predictions[identifier] = result
            except Exception as e:
                if 'timeout' in str(e).lower():
                    timeouts += 1
                else:
                    errors += 1
                logger.warning("Model prediction failed", extra={
                    "component": "model_server",
                    "action": "prediction_error",
                    "model": identifier,
                    "symbol": symbol,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                predictions[identifier] = 0.0
        
        execution_time = (time.time() - execution_start) * 1000
        
        # Cache results
        total_time = time.time() - start_time
        total_time_ms = total_time * 1000
        self.prediction_cache[cache_key] = predictions
        
        # Update performance metrics
        self._update_performance_metrics(total_time, len(predictions))
        
        logger.info("Parallel model predictions complete", extra={
            "component": "model_server",
            "action": "predict_all_complete",
            "symbol": symbol,
            "successful_predictions": len([p for p in predictions.values() if p != 0.0]),
            "total_models": len(self.models),
            "execution_time_ms": execution_time,
            "total_time_ms": total_time_ms,
            "target_time_ms": 100,
            "within_target": total_time_ms < 100,
            "timeouts": timeouts,
            "errors": errors,
            "cache_size": len(self.prediction_cache)
        })
        
        return predictions
    
    def _predict_single_model(self, model_name: str, features: np.ndarray) -> float:
        """Predict with a single model"""
        try:
            model = self.models[model_name]
            model_type = self.model_metadata[model_name]['type']
            
            if model_type == 'lightgbm':
                return float(model.predict(features.reshape(1, -1))[0])
            
            elif model_type == 'xgboost':
                dmatrix = xgb.DMatrix(features.reshape(1, -1))
                return float(model.predict(dmatrix)[0])
            
            elif model_type == 'catboost':
                return float(model.predict(features.reshape(1, -1))[0])
            
            elif model_type == 'linear':
                return float(model.predict(features.reshape(1, -1))[0])
            
            elif model_type == 'time_series':
                return self._predict_time_series(model_name, model, features)
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Single model prediction error for {model_name}: {e}")
            return 0.0
    
    def _predict_gpu_batch(self, model_names: List[str], features: np.ndarray) -> Dict[str, float]:
        """Batch prediction for GPU models"""
        try:
            results = {}
            
            # Prepare batch input
            batch_features = features.reshape(1, -1)
            
            for model_name in model_names:
                model = self.models[model_name]
                
                if hasattr(model, 'predict'):
                    prediction = model.predict(batch_features, verbose=0)
                    if isinstance(prediction, np.ndarray):
                        results[model_name] = float(prediction[0])
                    else:
                        results[model_name] = float(prediction)
                else:
                    results[model_name] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"GPU batch prediction error: {e}")
            return {name: 0.0 for name in model_names}
    
    def _predict_time_series(self, model_name: str, model: Any, features: np.ndarray) -> float:
        """Predict with time series models"""
        try:
            if 'garch' in model_name:
                # GARCH volatility forecast
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(horizon=1)
                    return float(forecast.variance.iloc[-1, 0])
                else:
                    return 0.02  # Default volatility
            
            elif 'arima' in model_name:
                # ARIMA trend forecast
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(steps=1)
                    return float(forecast[0])
                else:
                    return 0.0
            
            elif 'var' in model_name:
                # VAR multivariate forecast
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(steps=1)
                    return float(forecast[0, 0])  # First variable
                else:
                    return 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Time series prediction error for {model_name}: {e}")
            return 0.0
    
    def _generate_prediction_cache_key(self, features: np.ndarray, symbol: str = None) -> str:
        """Generate cache key for predictions"""
        try:
            feature_hash = hash(features.tobytes())
            timestamp = int(time.time() / 60)  # Minute-level caching
            return f"{symbol}_{timestamp}_{feature_hash}"
        except:
            return f"default_{int(time.time())}"
    
    def _update_performance_metrics(self, inference_time: float, model_count: int):
        """Update performance tracking metrics"""
        time.time()
        
        if 'total_inference_time' not in self.inference_times:
            self.inference_times['total_inference_time'] = []
        
        self.inference_times['total_inference_time'].append(inference_time)
        
        # Keep only last 1000 measurements
        if len(self.inference_times['total_inference_time']) > 1000:
            self.inference_times['total_inference_time'] = self.inference_times['total_inference_time'][-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times.get('total_inference_time'):
            return {}
        
        times = self.inference_times['total_inference_time']
        
        return {
            'avg_inference_time': np.mean(times),
            'p95_inference_time': np.percentile(times, 95),
            'p99_inference_time': np.percentile(times, 99),
            'total_predictions': len(times),
            'models_loaded': len(self.models),
            'cache_hit_ratio': len(self.prediction_cache) / max(len(times), 1)
        }
    
    # Default model creators (for when pretrained models don't exist)
    def _create_default_lightgbm(self) -> lgb.Booster:
        """Create default LightGBM model"""
        # Create minimal training data
        X = np.random.randn(100, 25)
        y = np.random.randn(100)
        
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 16,
            'learning_rate': 0.1,
            'verbose': -1
        }
        
        return lgb.train(params, train_data, num_boost_round=10)
    
    def _create_default_xgboost(self) -> xgb.Booster:
        """Create default XGBoost model"""
        X = np.random.randn(100, 25)
        y = np.random.randn(100)
        
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.1
        }
        
        return xgb.train(params, dtrain, num_boost_round=10)
    
    def _create_default_catboost(self) -> cb.CatBoostRegressor:
        """Create default CatBoost model"""
        model = cb.CatBoostRegressor(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            verbose=False
        )
        
        X = np.random.randn(100, 25)
        y = np.random.randn(100)
        model.fit(X, y)
        
        return model
    
    def _create_default_tensorflow_model(self, model_name: str) -> tf.keras.Model:
        """Create default TensorFlow model"""
        if 'lstm' in model_name:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, input_shape=(1, 25)),
                tf.keras.layers.Dense(1)
            ])
        elif 'cnn' in model_name:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(32, 3, input_shape=(25, 1)),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(1)
            ])
        else:  # feedforward
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(25,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with dummy data
        X = np.random.randn(100, 25)
        y = np.random.randn(100)
        
        if 'lstm' in model_name:
            X = X.reshape(100, 1, 25)
        elif 'cnn' in model_name:
            X = X.reshape(100, 25, 1)
        
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def _create_default_linear_model(self, model_name: str):
        """Create default linear model"""
        X = np.random.randn(100, 25)
        y = np.random.randn(100)
        
        if 'ridge' in model_name:
            model = Ridge(alpha=1.0)
        elif 'lasso' in model_name:
            model = Lasso(alpha=1.0)
        else:  # logistic
            y = (y > 0).astype(int)
            model = LogisticRegression()
        
        model.fit(X, y)
        return model
    
    def _create_default_garch(self):
        """Create default GARCH model"""
        # Generate dummy returns data
        returns = np.random.randn(100) * 0.02
        
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            return fitted_model
        except:
            # Return simple volatility estimator if GARCH fails
            return lambda: 0.02
    
    def _create_default_arima(self):
        """Create default ARIMA model"""
        # Generate dummy time series
        ts = np.cumsum(np.random.randn(100) * 0.01)
        
        try:
            model = sm.tsa.ARIMA(ts, order=(1, 1, 1))
            fitted_model = model.fit()
            return fitted_model
        except:
            # Return simple trend estimator if ARIMA fails
            return lambda: 0.0
    
    def _create_default_var(self):
        """Create default VAR model"""
        # Generate dummy multivariate time series
        data = np.random.randn(100, 3)
        
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            model = VAR(data)
            fitted_model = model.fit(maxlags=1)
            return fitted_model
        except:
            # Return simple multivariate estimator if VAR fails
            return lambda: np.array([0.0, 0.0, 0.0])