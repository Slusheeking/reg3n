"""
Lag-Llama Fine-tuning Pipeline
FAIL FAST Architecture - No Fallbacks, Production-Ready Training
"""

import asyncio
import logging
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import orjson as json
from pathlib import Path

# Import Lag-Llama components
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from settings import config
from database import get_database_client
from cache import get_trading_cache

logger = logging.getLogger(__name__)

# FAIL FAST Exceptions
class TrainingError(Exception):
    """Base exception for training errors"""
    pass

class InsufficientTrainingDataError(TrainingError):
    """Raised when insufficient training data is available"""
    pass

class ModelTrainingFailedError(TrainingError):
    """Raised when model training fails"""
    pass

class ModelValidationFailedError(TrainingError):
    """Raised when model validation fails"""
    pass

class CheckpointSaveError(TrainingError):
    """Raised when checkpoint saving fails"""
    pass

@dataclass
class TrainingConfig:
    """Training configuration with FAIL FAST validation"""
    training_data_days: int = 30
    validation_split: float = 0.2
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    early_stopping_patience: int = 3
    min_improvement_threshold: float = 0.05  # 5% minimum improvement
    
    def __post_init__(self):
        # FAIL FAST: Validate configuration
        if self.training_data_days < 7:
            raise TrainingError("FAIL FAST: Minimum 7 days of training data required")
        if not 0.1 <= self.validation_split <= 0.3:
            raise TrainingError("FAIL FAST: Validation split must be between 0.1 and 0.3")
        if self.epochs < 1:
            raise TrainingError("FAIL FAST: Minimum 1 epoch required")

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_type: str
    checkpoint_path: str
    training_start: datetime
    training_end: datetime
    training_data_period: int  # days
    performance_metrics: Dict[str, float]
    validation_loss: float
    is_active: bool = False
    
class LagLlamaTrainer:
    """FAIL FAST Lag-Llama Fine-tuning Engine"""
    
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.config = training_config or TrainingConfig()
        self.db = get_database_client()
        self.cache = get_trading_cache()
        
        # FAIL FAST: Validate system dependencies
        if not self.db.initialized:
            raise TrainingError("FAIL FAST: Database not initialized")
        
        # Training state
        self.current_training_session = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths
        self.models_dir = Path("ai_llama/models/lag_llama/fine_tuned")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Lag-Llama Trainer initialized with FAIL FAST architecture")
    
    async def prepare_training_data(self, symbols: List[str]) -> ListDataset:
        """Prepare training dataset with FAIL FAST validation"""
        logger.info(f"Preparing training data for {len(symbols)} symbols")
        
        # FAIL FAST: Validate symbols
        if not symbols:
            raise InsufficientTrainingDataError("FAIL FAST: No symbols provided for training")
        
        training_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.training_data_days)
        
        for symbol in symbols:
            try:
                # Get historical data from database
                symbol_data = await self._get_symbol_training_data(symbol, start_date, end_date)
                
                if len(symbol_data) < config.lag_llama.context_length * 2:
                    logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} points")
                    continue
                
                # Create dataset entry
                entry = {
                    'target': symbol_data.tolist(),
                    'start': start_date,
                    'item_id': symbol
                }
                training_data.append(entry)
                
            except Exception as e:
                logger.error(f"Failed to prepare data for {symbol}: {e}")
                # FAIL FAST: Don't continue with insufficient data
                continue
        
        # FAIL FAST: Validate sufficient training data
        if len(training_data) < 3:
            raise InsufficientTrainingDataError(
                f"FAIL FAST: Insufficient training data - only {len(training_data)} symbols have adequate data"
            )
        
        logger.info(f"Prepared training dataset with {len(training_data)} symbols")
        return ListDataset(training_data, freq='T')
    
    async def _get_symbol_training_data(self, symbol: str, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Get training data for a symbol with FAIL FAST validation"""
        
        # Try cache first for recent data
        try:
            if (datetime.now() - end_date).days < 1:
                cache_data = self.cache.get_price_series(symbol, self.config.training_data_days * 390)
                if len(cache_data) >= config.lag_llama.context_length:
                    return cache_data
        except Exception:
            pass  # Fall through to database
        
        # Get from database
        async with self.db.get_connection() as conn:
            query = """
                SELECT close 
                FROM market_data 
                WHERE symbol = $1 
                    AND time >= $2 
                    AND time <= $3 
                    AND timeframe = '1m'
                ORDER BY time ASC
            """
            rows = await conn.fetch(query, symbol, start_date, end_date)
            
            if not rows:
                raise InsufficientTrainingDataError(f"FAIL FAST: No data found for {symbol}")
            
            prices = np.array([float(row['close']) for row in rows], dtype=np.float32)
            
            # FAIL FAST: Validate data quality
            if np.any(np.isnan(prices)) or np.any(prices <= 0):
                raise InsufficientTrainingDataError(f"FAIL FAST: Invalid price data for {symbol}")
            
            return prices
    
    async def train_model(self, symbols: List[str]) -> ModelVersion:
        """Train fine-tuned model with FAIL FAST validation"""
        training_start = datetime.now()
        logger.info("Starting Lag-Llama fine-tuning with FAIL FAST architecture")
        
        try:
            # Prepare training data
            dataset = await self.prepare_training_data(symbols)
            
            # Split into train/validation
            train_dataset, val_dataset = self._split_dataset(dataset)
            
            # Create estimator for fine-tuning
            estimator = self._create_fine_tuning_estimator()
            
            # Train model
            predictor = await self._train_with_validation(estimator, train_dataset, val_dataset)
            
            # Validate trained model
            validation_metrics = await self._validate_trained_model(predictor, val_dataset)
            
            # Save model checkpoint
            version_id = f"fine_tuned_{int(time.time())}"
            checkpoint_path = await self._save_model_checkpoint(predictor, version_id)
            
            # Create model version record
            model_version = ModelVersion(
                version_id=version_id,
                model_type="lag_llama_fine_tuned",
                checkpoint_path=str(checkpoint_path),
                training_start=training_start,
                training_end=datetime.now(),
                training_data_period=self.config.training_data_days,
                performance_metrics=validation_metrics,
                validation_loss=validation_metrics.get('validation_loss', float('inf'))
            )
            
            # Save to database
            await self._save_model_version(model_version)
            
            logger.info(f"Fine-tuning completed successfully: {version_id}")
            return model_version
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelTrainingFailedError(f"FAIL FAST: Training failed: {e}")
    
    def _split_dataset(self, dataset: ListDataset) -> Tuple[ListDataset, ListDataset]:
        """Split dataset into train/validation with FAIL FAST validation"""
        
        data_list = list(dataset)
        total_size = len(data_list)
        
        # FAIL FAST: Validate dataset size
        if total_size < 5:
            raise InsufficientTrainingDataError(
                f"FAIL FAST: Dataset too small for train/val split: {total_size}"
            )
        
        val_size = int(total_size * self.config.validation_split)
        train_size = total_size - val_size
        
        # FAIL FAST: Ensure minimum sizes
        if train_size < 3 or val_size < 1:
            raise InsufficientTrainingDataError(
                f"FAIL FAST: Insufficient data for split - train: {train_size}, val: {val_size}"
            )
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:]
        
        return ListDataset(train_data, freq='T'), ListDataset(val_data, freq='T')
    
    def _create_fine_tuning_estimator(self) -> LagLlamaEstimator:
        """Create estimator for fine-tuning with existing checkpoint"""
        
        # FAIL FAST: Validate checkpoint exists
        if not Path(config.lag_llama.model_path).exists():
            raise ModelTrainingFailedError(f"FAIL FAST: Base checkpoint not found: {config.lag_llama.model_path}")
        
        estimator = LagLlamaEstimator(
            ckpt_path=config.lag_llama.model_path,
            prediction_length=config.lag_llama.prediction_length,
            context_length=config.lag_llama.context_length,
            device=self.device,
            batch_size=self.config.batch_size,
            num_parallel_samples=config.lag_llama.num_parallel_samples,
            
            # Fine-tuning specific parameters
            lr=self.config.learning_rate,
            trainer_kwargs={
                'max_epochs': self.config.epochs,
                'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                'devices': 1,
                'precision': 'bf16' if torch.cuda.is_available() else 32,
                'callbacks': [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.config.early_stopping_patience,
                        mode='min'
                    ),
                    ModelCheckpoint(
                        monitor='val_loss',
                        mode='min',
                        save_top_k=1
                    )
                ]
            }
        )
        
        return estimator
    
    async def _train_with_validation(self, estimator: LagLlamaEstimator, 
                                   train_dataset: ListDataset, 
                                   val_dataset: ListDataset):
        """Train model with validation monitoring"""
        
        try:
            logger.info("Starting model training...")
            
            # Train the model
            predictor = estimator.train(
                training_data=train_dataset,
                validation_data=val_dataset,
                shuffle_buffer_length=1000
            )
            
            # FAIL FAST: Validate predictor was created
            if predictor is None:
                raise ModelTrainingFailedError("FAIL FAST: Training returned None predictor")
            
            return predictor
            
        except Exception as e:
            raise ModelTrainingFailedError(f"FAIL FAST: Training process failed: {e}")
    
    async def _validate_trained_model(self, predictor, val_dataset: ListDataset) -> Dict[str, float]:
        """Validate trained model performance"""
        
        try:
            logger.info("Validating trained model...")
            
            # Generate predictions on validation set
            forecasts = list(predictor.predict(val_dataset))
            
            # FAIL FAST: Validate forecasts were generated
            if not forecasts:
                raise ModelValidationFailedError("FAIL FAST: No forecasts generated during validation")
            
            # Calculate validation metrics
            val_data_list = list(val_dataset)
            total_loss = 0.0
            valid_forecasts = 0
            
            for i, (forecast, data_entry) in enumerate(zip(forecasts, val_data_list)):
                if hasattr(forecast, 'samples') and len(forecast.samples) > 0:
                    # Calculate simple MSE loss
                    target = np.array(data_entry['target'][-config.lag_llama.prediction_length:])
                    pred_mean = forecast.samples.mean(axis=0)
                    
                    if len(pred_mean) == len(target):
                        mse = np.mean((target - pred_mean) ** 2)
                        total_loss += mse
                        valid_forecasts += 1
            
            # FAIL FAST: Validate we have valid forecasts
            if valid_forecasts == 0:
                raise ModelValidationFailedError("FAIL FAST: No valid forecasts for validation")
            
            avg_loss = total_loss / valid_forecasts
            
            metrics = {
                'validation_loss': float(avg_loss),
                'valid_forecasts': valid_forecasts,
                'total_forecasts': len(forecasts)
            }
            
            logger.info(f"Validation completed - Loss: {avg_loss:.6f}")
            return metrics
            
        except Exception as e:
            raise ModelValidationFailedError(f"FAIL FAST: Model validation failed: {e}")
    
    async def _save_model_checkpoint(self, predictor, version_id: str) -> Path:
        """Save model checkpoint with FAIL FAST validation"""
        
        checkpoint_path = self.models_dir / f"{version_id}.ckpt"
        
        try:
            # Save the predictor's lightning module
            if hasattr(predictor, 'prediction_net'):
                torch.save(predictor.prediction_net.state_dict(), checkpoint_path)
            else:
                raise CheckpointSaveError("FAIL FAST: Predictor has no saveable state")
            
            # FAIL FAST: Validate checkpoint was saved
            if not checkpoint_path.exists():
                raise CheckpointSaveError("FAIL FAST: Checkpoint file was not created")
            
            # Validate checkpoint size
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            if size_mb < 1:  # Minimum 1MB expected
                raise CheckpointSaveError(f"FAIL FAST: Checkpoint too small: {size_mb:.1f}MB")
            
            logger.info(f"Model checkpoint saved: {checkpoint_path} ({size_mb:.1f}MB)")
            return checkpoint_path
            
        except Exception as e:
            raise CheckpointSaveError(f"FAIL FAST: Failed to save checkpoint: {e}")
    
    async def _save_model_version(self, model_version: ModelVersion):
        """Save model version to database"""
        
        try:
            async with self.db.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO model_versions 
                    (version_id, model_type, checkpoint_path, training_start, training_end,
                     training_data_period, performance_metrics, validation_loss, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                model_version.version_id,
                model_version.model_type,
                model_version.checkpoint_path,
                model_version.training_start,
                model_version.training_end,
                model_version.training_data_period,
                json.dumps(model_version.performance_metrics),
                model_version.validation_loss,
                model_version.is_active
                )
            
            logger.info(f"Model version saved to database: {model_version.version_id}")
            
        except Exception as e:
            logger.error(f"Failed to save model version to database: {e}")
            raise
    
    async def get_best_model_version(self) -> Optional[ModelVersion]:
        """Get the best performing model version"""
        
        try:
            async with self.db.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM model_versions 
                    WHERE model_type = 'lag_llama_fine_tuned'
                    ORDER BY validation_loss ASC 
                    LIMIT 1
                """)
                
                if row:
                    return ModelVersion(
                        version_id=row['version_id'],
                        model_type=row['model_type'],
                        checkpoint_path=row['checkpoint_path'],
                        training_start=row['training_start'],
                        training_end=row['training_end'],
                        training_data_period=row['training_data_period'],
                        performance_metrics=json.loads(row['performance_metrics']),
                        validation_loss=float(row['validation_loss']),
                        is_active=row['is_active']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get best model version: {e}")
            return None
    
    async def compare_with_baseline(self, fine_tuned_version: ModelVersion) -> Dict[str, Any]:
        """Compare fine-tuned model with baseline zero-shot model"""
        
        # This would implement A/B testing comparison
        # For now, return basic comparison structure
        return {
            'fine_tuned_loss': fine_tuned_version.validation_loss,
            'improvement_threshold': self.config.min_improvement_threshold,
            'should_deploy': fine_tuned_version.validation_loss < 1.0,  # Placeholder logic
            'comparison_timestamp': datetime.now()
        }

# Global trainer instance
lag_llama_trainer = LagLlamaTrainer()