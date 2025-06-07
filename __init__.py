
# Standard library imports
from utils.system_logger import ComponentLogger

# Core package imports
from . import data_pipeline
from . import ml_models
from . import trading_pipeline
from . import filters
from . import utils
from . import yaml
from . import tests
from . import examples

# Main system components
from .data_pipeline import DataPipelineManager
from .trading_pipeline import TradingPipelineManager
from .ml_models import EnsembleManager
from .utils import UtilityManager
from .yaml import ConfigManager

__version__ = "1.0.0"
__author__ = "REG3N Development Team"
__description__ = "Advanced Algorithmic Trading System with ML4T Integration"

# Initialize package logger
logger = ComponentLogger("reg3n_system")
logger.info("Initializing REG3N Trading System", extra={
    "component": "reg3n_system",
    "action": "package_init",
    "version": __version__,
    "author": __author__
})

__all__ = [
    # Package modules
    'data_pipeline',
    'ml_models', 
    'trading_pipeline',
    'filters',
    'utils',
    'yaml',
    'tests',
    'examples',
    
    # Main managers
    'DataPipelineManager',
    'TradingPipelineManager',
    'EnsembleManager',
    'UtilityManager',
    'ConfigManager',
    'REG3NSystem'
]

class REG3NSystem:
    """
    Main system orchestrator for the REG3N trading platform
    Coordinates all subsystems: data pipeline, ML models, trading execution
    """
    
    def __init__(self, 
                 polygon_api_key: str,
                 alpaca_api_key: str, 
                 alpaca_secret_key: str,
                 paper_trading: bool = True,
                 config_dir: str = "yaml"):
        
        # API credentials
        self.polygon_api_key = polygon_api_key
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.paper_trading = paper_trading
        
        # Initialize core managers
        self.config_manager = ConfigManager(config_dir)
        self.utils_manager = UtilityManager()
        self.logger = ComponentLogger("reg3n_system_instance")
        
        self.logger.info("Initializing REG3N System instance", extra={
            "component": "reg3n_system",
            "action": "system_init",
            "paper_trading": paper_trading,
            "config_dir": config_dir,
            "api_keys_present": {
                "polygon": bool(polygon_api_key),
                "alpaca_api": bool(alpaca_api_key),
                "alpaca_secret": bool(alpaca_secret_key)
            }
        })
        
        # Initialize subsystems
        self.data_pipeline = DataPipelineManager(
            polygon_api_key=polygon_api_key,
            filter_config_path=f"{config_dir}/adaptive_filter_config.yaml"
        )
        
        self.trading_pipeline = TradingPipelineManager(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            paper_trading=paper_trading
        )
        
        self.ml_ensemble = EnsembleManager()
        
        # System state
        self.is_running = False
        self.system_stats = {
            'start_time': None,
            'processed_signals': 0,
            'executed_trades': 0,
            'system_errors': 0
        }
        
        self.logger.info("REG3N System initialized successfully", extra={
            "component": "reg3n_system",
            "action": "system_ready",
            "subsystems": {
                "data_pipeline": bool(self.data_pipeline),
                "trading_pipeline": bool(self.trading_pipeline),
                "ml_ensemble": bool(self.ml_ensemble)
            }
        })
    
    async def start_system(self):
        """Start the complete REG3N trading system"""
        try:
            self.logger.info("Starting REG3N trading system...", component="SYSTEM")
            
            # Start data pipeline
            await self.data_pipeline.start_pipeline()
            self.logger.info("Data pipeline started", component="DATA")
            
            # Start trading pipeline  
            await self.trading_pipeline.start_trading()
            self.logger.info("Trading pipeline started", component="TRADING")
            
            # Initialize ML ensemble
            await self.ml_ensemble.initialize_ensemble()
            self.logger.info("ML ensemble initialized", component="ML")
            
            self.is_running = True
            self.system_stats['start_time'] = self.utils_manager.get_current_timestamp()
            
            self.logger.info("REG3N system fully operational", component="SYSTEM")
            
        except Exception as e:
            self.system_stats['system_errors'] += 1
            self.logger.error(f"Failed to start REG3N system: {str(e)}", component="SYSTEM", error=e)
            raise
    
    async def process_trading_cycle(self):
        """Execute one complete trading cycle"""
        if not self.is_running:
            return {'status': 'error', 'message': 'System not running'}
        
        try:
            # Get filtered market data
            market_data = await self.data_pipeline.get_filtered_data()
            
            if not market_data:
                return {'status': 'no_data', 'message': 'No market data available'}
            
            # Generate ML predictions
            predictions = await self.ml_ensemble.generate_predictions(market_data)
            
            # Execute trades based on predictions
            trade_results = []
            for prediction in predictions:
                if prediction.get('confidence', 0) > 0.7:  # High confidence threshold
                    trade_signal = {
                        'symbol': prediction['symbol'],
                        'side': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'expected_return': prediction.get('expected_return', 0.01),
                        'risk': prediction.get('risk', 0.02)
                    }
                    
                    result = await self.trading_pipeline.execute_trade_signal(trade_signal)
                    trade_results.append(result)
                    
                    if result.get('status') == 'success':
                        self.system_stats['executed_trades'] += 1
            
            self.system_stats['processed_signals'] += len(predictions)
            
            return {
                'status': 'success',
                'market_data_count': len(market_data),
                'predictions_count': len(predictions),
                'trades_executed': len([r for r in trade_results if r.get('status') == 'success']),
                'trade_results': trade_results
            }
            
        except Exception as e:
            self.system_stats['system_errors'] += 1
            self.logger.error(f"Trading cycle error: {str(e)}", component="SYSTEM", error=e)
            return {'status': 'error', 'message': str(e)}
    
    async def stop_system(self):
        """Stop the REG3N trading system"""
        try:
            self.logger.info("Stopping REG3N trading system...", component="SYSTEM")
            
            # Stop subsystems
            await self.trading_pipeline.stop_trading()
            await self.data_pipeline.stop_pipeline()
            
            self.is_running = False
            self.logger.info("REG3N system stopped", component="SYSTEM")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}", component="SYSTEM", error=e)
            raise
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'system_info': {
                'version': __version__,
                'is_running': self.is_running,
                'paper_trading': self.paper_trading,
                'stats': self.system_stats
            },
            'data_pipeline': self.data_pipeline.get_pipeline_stats(),
            'trading_pipeline': self.trading_pipeline.get_trading_stats(),
            'ml_ensemble': self.ml_ensemble.get_ensemble_stats() if hasattr(self.ml_ensemble, 'get_ensemble_stats') else {}
        }