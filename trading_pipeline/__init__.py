"""
Trading Pipeline Package
Handles order execution, position sizing, and trading strategy implementation
"""

from utils import get_logger

# Package imports
from .alpaca_websocket import AlpacaWebSocketClient
from .alpaca_rest_api import AlpacaRESTClient
from .momentum_execution_engine import MomentumExecutionEngine
from .kelly_position_sizer import KellyPositionSizer

# Initialize package logger
logger = get_logger("trading_pipeline")
logger.startup({
    "action": "package_init",
    "modules": ["AlpacaWebSocketClient", "AlpacaRESTClient", "MomentumExecutionEngine", "KellyPositionSizer"]
})

__all__ = [
    'AlpacaWebSocketClient',
    'AlpacaRESTClient',
    'MomentumExecutionEngine',
    'KellyPositionSizer',
    'TradingPipelineManager'
]

logger.startup({
    "action": "package_ready",
    "exported_classes": len(__all__)
})

class TradingPipelineManager:
    """
    Main trading pipeline manager that orchestrates all trading components
    """
    
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, paper_trading: bool = True):
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.paper_trading = paper_trading
        
        # Initialize components
        self.websocket_client = AlpacaWebSocketClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
            paper=paper_trading
        )
        self.rest_client = AlpacaRESTClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
            paper=paper_trading
        )
        self.execution_engine = MomentumExecutionEngine(
            alpaca_client=self.rest_client
        )
        self.position_sizer = KellyPositionSizer()
        
        # Pipeline state
        self.is_active = False
        self.executed_trades = 0
        
    async def start_trading(self):
        """Start the trading pipeline"""
        await self.websocket_client.connect()
        await self.execution_engine.start()
        self.is_active = True
        
    async def execute_trade_signal(self, signal: dict) -> dict:
        """Execute a trade signal through the pipeline"""
        if not self.is_active:
            return {'status': 'error', 'message': 'Pipeline not active'}
        
        # Calculate position size using Kelly criterion
        position_size = self.position_sizer.calculate_position_size(
            signal.get('confidence', 0.5),
            signal.get('expected_return', 0.01),
            signal.get('risk', 0.02)
        )
        
        # Execute trade through momentum engine
        result = await self.execution_engine.execute_momentum_trade(
            symbol=signal['symbol'],
            side=signal['side'],
            quantity=position_size,
            confidence=signal.get('confidence', 0.5)
        )
        
        if result.get('status') == 'success':
            self.executed_trades += 1
            
        return result
    
    async def stop_trading(self):
        """Stop the trading pipeline"""
        await self.execution_engine.stop()
        await self.websocket_client.disconnect()
        self.is_active = False
    
    def get_trading_stats(self) -> dict:
        """Get trading pipeline statistics"""
        return {
            'is_active': self.is_active,
            'executed_trades': self.executed_trades,
            'execution_stats': self.execution_engine.get_execution_stats() if hasattr(self.execution_engine, 'get_execution_stats') else {},
            'position_sizer_stats': self.position_sizer.get_stats() if hasattr(self.position_sizer, 'get_stats') else {}
        }