#!/usr/bin/env python3

import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor
import json

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
import os
import yaml

from utils.system_logger import get_system_logger

# Initialize component logger
logger = get_system_logger("online_learning")

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

@dataclass
class TrainingExample:
    features: np.ndarray
    target: float
    timestamp: float
    symbol: str
    actual_outcome: Optional[float] = None
    prediction_error: Optional[float] = None

@dataclass
class ModelPerformance:
    model_name: str
    accuracy: float
    mse: float
    information_coefficient: float
    prediction_count: int
    last_updated: float

class OnlineLearningEngine:
    """
    Online learning system that continuously improves models during trading
    Features incremental learning, performance monitoring, and automatic retraining
    """
    
    def __init__(self, model_server, ensemble_manager, config: Dict = None):
        self.model_server = model_server
        self.ensemble_manager = ensemble_manager
        # Load config from YAML if not provided
        yaml_config = CONFIG.get('online_learning', {})
        self.config = config or yaml_config or self._get_default_config()
        
        logger.info("Initializing Online Learning Engine", extra={
            "component": "online_learning",
            "action": "initialization_start",
            "config_source": "yaml" if yaml_config else "default",
            "config_keys": list(self.config.keys())
        })
        
        # Training data buffers
        buffer_size = self.config.get('buffer_size', 5000)
        self.training_buffers = {
            model_name: deque(maxlen=buffer_size)
            for model_name in self._get_trainable_models()
        }
        
        # Performance tracking
        self.model_performance = {}
        self.performance_history = deque(maxlen=10000)
        
        # Update frequencies (number of examples before retraining)
        self.update_frequencies = {
            'lightgbm_regime': 100,
            'lightgbm_volatility': 100,
            'lightgbm_momentum': 100,
            'lightgbm_intraday': 200,
            'xgboost_ensemble': 200,
            'lstm_returns': 500,
            'cnn_patterns': 500,
            'feedforward_nn': 300,
            'dqn_agent': 50,
            'ensemble_meta': 150
        }
        
        # Learning rates for different model types (use YAML config if available)
        base_lr = self.config.get('learning_rate', 0.01)
        self.learning_rates = {
            'lightgbm': base_lr,
            'xgboost': base_lr,
            'tensorflow': base_lr * 0.1,  # Lower for neural networks
            'pytorch': base_lr * 0.1,
            'ensemble': base_lr * 0.5
        }
        
        # Background training thread
        self.training_executor = ThreadPoolExecutor(max_workers=2)
        self.training_queue = asyncio.Queue(maxsize=1000)
        self.is_training = False
        
        # Model backup system
        self.model_backups = {}
        self.backup_frequency = self.config['backup_frequency']
        
        # Performance thresholds for retraining
        self.performance_thresholds = {
            'min_accuracy': 0.52,
            'min_ic': 0.01,
            'max_mse': 0.1,
            'performance_decay_threshold': 0.05
        }
        
        logger.info("Online Learning Engine initialization complete", extra={
            "component": "online_learning",
            "action": "initialization_complete",
            "trainable_models": len(self._get_trainable_models()),
            "buffer_size": self.config['buffer_size'],
            "min_training_samples": self.config['min_training_samples'],
            "update_frequencies": self.update_frequencies,
            "learning_rates": self.learning_rates,
            "performance_thresholds": self.performance_thresholds,
            "incremental_learning": self.config.get('enable_incremental_learning', True),
            "automatic_retraining": self.config.get('enable_automatic_retraining', True)
        })
        
    def _get_default_config(self) -> Dict:
        """Default configuration for online learning"""
        return {
            'enabled': True,
            'update_frequency': 'daily',
            'learning_rate': 0.01,
            'batch_size': 1000,
            'memory_window': 252,
            'performance_tracking': True,
            'model_decay_factor': 0.95,
            'retraining_threshold': 0.1,
            'buffer_size': 5000,
            'min_training_samples': 100,
            'backup_frequency': 1000,
            'performance_window': 500,
            'max_training_time': 30.0,  # seconds
            'enable_incremental_learning': True,
            'enable_automatic_retraining': True,
            'validation_split': 0.2
        }
    
    def _get_trainable_models(self) -> List[str]:
        """Get list of models that support online learning"""
        return [
            'lightgbm_regime', 'lightgbm_volatility', 'lightgbm_momentum', 'lightgbm_intraday',
            'xgboost_ensemble', 'lstm_returns', 'cnn_patterns', 'feedforward_nn',
            'dqn_agent', 'ensemble_meta'
        ]
    
    async def start_online_learning(self):
        """Start the online learning background process"""
        logger.info("Starting online learning engine", extra={
            "component": "online_learning",
            "action": "start_engine",
            "buffer_size": self.config['buffer_size'],
            "min_training_samples": self.config['min_training_samples'],
            "trainable_models": len(self._get_trainable_models()),
            "incremental_learning": self.config['enable_incremental_learning'],
            "automatic_retraining": self.config['enable_automatic_retraining']
        })
        
        # Start background training loop
        asyncio.create_task(self._training_loop())
        
        # Initialize model performance tracking
        await self._initialize_performance_tracking()
        
        logger.info("Online learning engine started successfully", extra={
            "component": "online_learning",
            "action": "engine_ready",
            "trainable_models": self._get_trainable_models(),
            "update_frequencies": self.update_frequencies,
            "performance_thresholds": self.performance_thresholds
        })
    
    async def add_training_example(self, features: np.ndarray, prediction: float, 
                                 symbol: str, actual_outcome: Optional[float] = None):
        """
        Add a new training example to the learning system
        """
        try:
            example = TrainingExample(
                features=features.copy(),
                target=prediction,
                timestamp=time.time(),
                symbol=symbol,
                actual_outcome=actual_outcome
            )
            
            if actual_outcome is not None:
                example.prediction_error = abs(prediction - actual_outcome)
            
            # Add to training queue
            await self.training_queue.put(example)
            
            # Update performance if we have actual outcome
            if actual_outcome is not None:
                await self._update_model_performance(example)
            
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
    
    async def _training_loop(self):
        """Background training loop"""
        while True:
            try:
                # Get training examples from queue
                examples = []
                
                # Collect batch of examples (non-blocking)
                for _ in range(min(50, self.training_queue.qsize())):
                    try:
                        example = await asyncio.wait_for(self.training_queue.get(), timeout=0.1)
                        examples.append(example)
                    except asyncio.TimeoutError:
                        break
                
                if examples:
                    await self._process_training_batch(examples)
                
                # Check for retraining needs
                await self._check_retraining_needs()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_training_batch(self, examples: List[TrainingExample]):
        """Process a batch of training examples"""
        try:
            # Group examples by model type
            model_examples = {model: [] for model in self._get_trainable_models()}
            
            for example in examples:
                # Add to all relevant model buffers
                for model_name in self._get_trainable_models():
                    model_examples[model_name].append(example)
                    self.training_buffers[model_name].append(example)
            
            # Trigger incremental updates if enabled
            if self.config['enable_incremental_learning']:
                await self._incremental_updates(model_examples)
            
        except Exception as e:
            logger.error(f"Error processing training batch: {e}")
    
    async def _incremental_updates(self, model_examples: Dict[str, List[TrainingExample]]):
        """Perform incremental model updates"""
        try:
            update_tasks = []
            
            for model_name, examples in model_examples.items():
                if len(examples) > 0:
                    # Check if model needs update
                    buffer_size = len(self.training_buffers[model_name])
                    update_freq = self.update_frequencies.get(model_name, 100)
                    
                    if buffer_size >= update_freq:
                        task = self._incremental_update_model(model_name)
                        update_tasks.append(task)
            
            # Execute updates in parallel
            if update_tasks:
                await asyncio.gather(*update_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error in incremental updates: {e}")
    
    async def _incremental_update_model(self, model_name: str):
        """Perform incremental update for a specific model"""
        try:
            if self.is_training:
                return  # Skip if already training
            
            self.is_training = True
            
            # Get training data from buffer
            buffer = self.training_buffers[model_name]
            if len(buffer) < self.config['min_training_samples']:
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(list(buffer))
            
            if len(X) == 0:
                return
            
            # Determine model type and update accordingly
            model_type = self._get_model_type(model_name)
            
            if model_type == 'lightgbm':
                await self._update_lightgbm_model(model_name, X, y)
            elif model_type == 'xgboost':
                await self._update_xgboost_model(model_name, X, y)
            elif model_type == 'tensorflow':
                await self._update_tensorflow_model(model_name, X, y)
            elif model_type == 'pytorch':
                await self._update_pytorch_model(model_name, X, y)
            
            # Clear part of buffer to prevent memory issues
            self._trim_buffer(model_name)
            
            logger.info(f"Incremental update completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Error in incremental update for {model_name}: {e}")
        finally:
            self.is_training = False
    
    async def _update_lightgbm_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Update LightGBM model incrementally"""
        try:
            current_model = self.model_server.models.get(model_name)
            if current_model is None:
                return
            
            # Create new training dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Incremental training parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': self.learning_rates['lightgbm'],
                'num_leaves': 16,
                'verbose': -1
            }
            
            # Continue training from existing model
            updated_model = lgb.train(
                params,
                train_data,
                num_boost_round=10,  # Small number for incremental update
                init_model=current_model,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)]
            )
            
            # Update model in server
            self.model_server.models[model_name] = updated_model
            
        except Exception as e:
            logger.error(f"LightGBM update error for {model_name}: {e}")
    
    async def _update_xgboost_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Update XGBoost model incrementally"""
        try:
            current_model = self.model_server.models.get(model_name)
            if current_model is None:
                return
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y)
            
            # Incremental training parameters
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': self.learning_rates['xgboost'],
                'max_depth': 3,
                'subsample': 0.8
            }
            
            # Continue training
            updated_model = xgb.train(
                params,
                dtrain,
                num_boost_round=10,
                xgb_model=current_model
            )
            
            # Update model in server
            self.model_server.models[model_name] = updated_model
            
        except Exception as e:
            logger.error(f"XGBoost update error for {model_name}: {e}")
    
    async def _update_tensorflow_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Update TensorFlow model incrementally"""
        try:
            current_model = self.model_server.models.get(model_name)
            if current_model is None:
                return
            
            # Prepare data for TensorFlow
            if 'lstm' in model_name:
                X = X.reshape(X.shape[0], 1, X.shape[1])
            elif 'cnn' in model_name:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Set learning rate
            current_model.optimizer.learning_rate = self.learning_rates['tensorflow']
            
            # Incremental training
            current_model.fit(
                X, y,
                epochs=1,
                batch_size=min(32, len(X)),
                verbose=0,
                validation_split=0.2
            )
            
        except Exception as e:
            logger.error(f"TensorFlow update error for {model_name}: {e}")
    
    async def _update_pytorch_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Update PyTorch model incrementally"""
        try:
            # PyTorch models would be implemented here
            # For now, we'll skip as the main models are TensorFlow-based
            pass
            
        except Exception as e:
            logger.error(f"PyTorch update error for {model_name}: {e}")
    
    def _prepare_training_data(self, examples: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from examples"""
        try:
            # Filter examples with actual outcomes
            valid_examples = [ex for ex in examples if ex.actual_outcome is not None]
            
            if not valid_examples:
                return np.array([]), np.array([])
            
            # Extract features and targets
            X = np.array([ex.features for ex in valid_examples])
            y = np.array([ex.actual_outcome for ex in valid_examples])
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name"""
        if 'lightgbm' in model_name:
            return 'lightgbm'
        elif 'xgboost' in model_name:
            return 'xgboost'
        elif any(x in model_name for x in ['lstm', 'cnn', 'feedforward', 'dqn']):
            return 'tensorflow'
        elif 'pytorch' in model_name:
            return 'pytorch'
        else:
            return 'unknown'
    
    def _trim_buffer(self, model_name: str):
        """Trim training buffer to prevent memory issues"""
        try:
            buffer = self.training_buffers[model_name]
            if len(buffer) > self.config['buffer_size'] * 0.8:
                # Remove oldest 20% of examples
                trim_count = int(len(buffer) * 0.2)
                for _ in range(trim_count):
                    if buffer:
                        buffer.popleft()
                        
        except Exception as e:
            logger.error(f"Error trimming buffer for {model_name}: {e}")
    
    async def _update_model_performance(self, example: TrainingExample):
        """Update model performance metrics"""
        try:
            if example.actual_outcome is None or example.prediction_error is None:
                return
            
            # Calculate performance metrics
            accuracy = 1.0 / (1.0 + example.prediction_error)
            mse = example.prediction_error ** 2
            
            # Update performance history
            self.performance_history.append({
                'timestamp': example.timestamp,
                'symbol': example.symbol,
                'accuracy': accuracy,
                'mse': mse,
                'prediction_error': example.prediction_error
            })
            
            # Update model-specific performance
            for model_name in self._get_trainable_models():
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = ModelPerformance(
                        model_name=model_name,
                        accuracy=accuracy,
                        mse=mse,
                        information_coefficient=0.0,
                        prediction_count=1,
                        last_updated=time.time()
                    )
                else:
                    perf = self.model_performance[model_name]
                    # Exponential moving average update
                    alpha = 0.1
                    perf.accuracy = perf.accuracy * (1 - alpha) + accuracy * alpha
                    perf.mse = perf.mse * (1 - alpha) + mse * alpha
                    perf.prediction_count += 1
                    perf.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    async def _check_retraining_needs(self):
        """Check if any models need full retraining"""
        try:
            if not self.config['enable_automatic_retraining']:
                return
            
            current_time = time.time()
            
            for model_name, performance in self.model_performance.items():
                # Check performance thresholds
                needs_retraining = (
                    performance.accuracy < self.performance_thresholds['min_accuracy'] or
                    performance.mse > self.performance_thresholds['max_mse'] or
                    (current_time - performance.last_updated) > 3600  # 1 hour without update
                )
                
                if needs_retraining:
                    await self._schedule_full_retraining(model_name)
            
        except Exception as e:
            logger.error(f"Error checking retraining needs: {e}")
    
    async def _schedule_full_retraining(self, model_name: str):
        """Schedule full retraining for a model"""
        try:
            logger.info(f"Scheduling full retraining for {model_name}")
            
            # Create backup of current model
            await self._backup_model(model_name)
            
            # Schedule retraining task
            self.training_executor.submit(self._full_retrain_model, model_name)
            
            # Don't wait for completion to avoid blocking
            
        except Exception as e:
            logger.error(f"Error scheduling retraining for {model_name}: {e}")
    
    def _full_retrain_model(self, model_name: str):
        """Perform full retraining of a model"""
        try:
            logger.info(f"Starting full retraining for {model_name}")
            
            # Get all training data
            buffer = self.training_buffers[model_name]
            X, y = self._prepare_training_data(list(buffer))
            
            if len(X) < self.config['min_training_samples']:
                logger.warning(f"Insufficient data for retraining {model_name}")
                return
            
            # Split data
            split_idx = int(len(X) * (1 - self.config['validation_split']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Retrain based on model type
            model_type = self._get_model_type(model_name)
            
            if model_type == 'lightgbm':
                self._retrain_lightgbm(model_name, X_train, y_train, X_val, y_val)
            elif model_type == 'xgboost':
                self._retrain_xgboost(model_name, X_train, y_train, X_val, y_val)
            elif model_type == 'tensorflow':
                self._retrain_tensorflow(model_name, X_train, y_train, X_val, y_val)
            
            logger.info(f"Full retraining completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Error in full retraining for {model_name}: {e}")
    
    def _retrain_lightgbm(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray):
        """Retrain LightGBM model from scratch"""
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'verbose': -1
            }
            
            new_model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Update model in server
            self.model_server.models[model_name] = new_model
            
        except Exception as e:
            logger.error(f"LightGBM retraining error for {model_name}: {e}")
    
    def _retrain_xgboost(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray):
        """Retrain XGBoost model from scratch"""
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            new_model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Update model in server
            self.model_server.models[model_name] = new_model
            
        except Exception as e:
            logger.error(f"XGBoost retraining error for {model_name}: {e}")
    
    def _retrain_tensorflow(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray):
        """Retrain TensorFlow model from scratch"""
        try:
            # Prepare data based on model type
            if 'lstm' in model_name:
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            elif 'cnn' in model_name:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            # Get current model architecture
            current_model = self.model_server.models.get(model_name)
            if current_model is None:
                return
            
            # Create new model with same architecture
            new_model = tf.keras.models.clone_model(current_model)
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train new model
            new_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
                ]
            )
            
            # Update model in server
            self.model_server.models[model_name] = new_model
            
        except Exception as e:
            logger.error(f"TensorFlow retraining error for {model_name}: {e}")
    
    async def _backup_model(self, model_name: str):
        """Create backup of current model"""
        try:
            current_model = self.model_server.models.get(model_name)
            if current_model is None:
                return
            
            backup_key = f"{model_name}_{int(time.time())}"
            self.model_backups[backup_key] = current_model
            
            # Limit number of backups
            if len(self.model_backups) > 10:
                oldest_key = min(self.model_backups.keys())
                del self.model_backups[oldest_key]
            
        except Exception as e:
            logger.error(f"Error backing up model {model_name}: {e}")
    
    async def _initialize_performance_tracking(self):
        """Initialize performance tracking for all models"""
        try:
            for model_name in self._get_trainable_models():
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = ModelPerformance(
                        model_name=model_name,
                        accuracy=0.5,
                        mse=0.1,
                        information_coefficient=0.0,
                        prediction_count=0,
                        last_updated=time.time()
                    )
        except Exception as e:
            logger.error(f"Error initializing performance tracking: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current online learning status"""
        try:
            return {
                'is_training': self.is_training,
                'training_queue_size': self.training_queue.qsize(),
                'buffer_sizes': {
                    model: len(buffer) for model, buffer in self.training_buffers.items()
                },
                'model_performance': {
                    model: {
                        'accuracy': perf.accuracy,
                        'mse': perf.mse,
                        'prediction_count': perf.prediction_count,
                        'last_updated': perf.last_updated
                    }
                    for model, perf in self.model_performance.items()
                },
                'total_examples': len(self.performance_history),
                'backup_count': len(self.model_backups)
            }
        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
            return {}
    
    async def save_learning_state(self, filepath: str):
        """Save online learning state"""
        try:
            state = {
                'model_performance': {
                    model: {
                        'accuracy': perf.accuracy,
                        'mse': perf.mse,
                        'information_coefficient': perf.information_coefficient,
                        'prediction_count': perf.prediction_count,
                        'last_updated': perf.last_updated
                    }
                    for model, perf in self.model_performance.items()
                },
                'performance_history': list(self.performance_history),
                'update_frequencies': self.update_frequencies,
                'learning_rates': self.learning_rates
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    async def load_learning_state(self, filepath: str):
        """Load online learning state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore model performance
            perf_data = state.get('model_performance', {})
            for model_name, perf_dict in perf_data.items():
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name,
                    accuracy=perf_dict['accuracy'],
                    mse=perf_dict['mse'],
                    information_coefficient=perf_dict['information_coefficient'],
                    prediction_count=perf_dict['prediction_count'],
                    last_updated=perf_dict['last_updated']
                )
            
            # Restore performance history
            history = state.get('performance_history', [])
            self.performance_history = deque(history, maxlen=10000)
            
            # Restore frequencies and rates
            self.update_frequencies.update(state.get('update_frequencies', {}))
            self.learning_rates.update(state.get('learning_rates', {}))
            
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")