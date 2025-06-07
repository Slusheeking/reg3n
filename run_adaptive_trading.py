#!/usr/bin/env python3

import asyncio
import os
import sys
import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project imports
from utils import get_system_logger
from ml_models.adaptive_data_processor import AdaptiveDataProcessor
from data_pipeline.polygon_websocket import RealTimeDataFeed
from trading_pipeline.alpaca_rest_api import AlpacaRESTClient
from trading_pipeline.momentum_execution_engine import MomentumExecutionEngine

# Initialize enhanced system logger
logger = get_system_logger("adaptive_trading_system")

class AdaptiveTradingSystem:
    """Main trading system that orchestrates all components - SINGLETON WebSocket connections"""
    
    def __init__(self):
        # API keys from environment
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not all([self.polygon_api_key, self.alpaca_api_key, self.alpaca_secret_key]):
            raise ValueError("Missing required API keys in environment variables")
        
        # SINGLETON WebSocket connections - only create once
        self._polygon_websocket = None
        self._alpaca_websocket = None
        
        # Core components
        self.ml_processor = None
        self.alpaca_client = None
        self.momentum_engine = None
        
        # System state
        self.running = False
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        
        logger.startup({
            "api_keys_present": {
                "polygon": bool(self.polygon_api_key),
                "alpaca_api": bool(self.alpaca_api_key),
                "alpaca_secret": bool(self.alpaca_secret_key)
            },
            "websocket_pattern": "singleton",
            "component": "adaptive_trading_system"
        })
        
    async def initialize(self):
        """Initialize all system components with singleton WebSocket connections"""
        logger.log_data_flow("initialization", "adaptive_trading_system")
        
        try:
            # Initialize ML processor with adaptive filter
            logger.log_data_flow("ml_processor_init", "adaptive_filter", data_size=2)
            self.ml_processor = AdaptiveDataProcessor(
                config_path="yaml/ml_config.yaml",
                filter_config_path="yaml/adaptive_filter_config.yaml"
            )
            await self.ml_processor.initialize()
            
            # Initialize SINGLETON Polygon WebSocket (only create once)
            if self._polygon_websocket is None:
                logger.connection("creating", {
                    "connection_type": "polygon_websocket",
                    "pattern": "singleton"
                })
                self._polygon_websocket = RealTimeDataFeed(
                    api_key=self.polygon_api_key,
                    symbols=None  # Will auto-fetch all symbols
                )
            else:
                logger.connection("reusing", {
                    "connection_type": "polygon_websocket",
                    "pattern": "singleton"
                })
            
            # Initialize Alpaca REST client (not WebSocket)
            logger.log_data_flow("alpaca_client_init", "rest_client")
            self.alpaca_client = AlpacaRESTClient(
                api_key=self.alpaca_api_key,
                secret_key=self.alpaca_secret_key,
                base_url="https://paper-api.alpaca.markets"  # Paper trading
            )
            
            # Initialize Momentum Execution Engine with Kelly Criterion
            account = self.alpaca_client.get_account()
            initial_capital = float(account.get('buying_power', 50000)) if account else 50000
            
            logger.log_data_flow("momentum_engine_init", "execution_engine", data_size=int(initial_capital))
            
            self.momentum_engine = MomentumExecutionEngine(
                alpaca_client=self.alpaca_client,
                initial_capital=initial_capital
            )
            
            # Initialize SINGLETON Alpaca WebSocket if needed (for order updates)
            if self._alpaca_websocket is None:
                logger.connection("skipped", {
                    "connection_type": "alpaca_websocket",
                    "reason": "using_rest_api_only"
                })
                # Only create if we need real-time order updates
                # For now, we'll use REST API only
                self._alpaca_websocket = None
            
            logger.log_data_flow("initialization", "complete", data_size=int(initial_capital))
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(e, {
                "operation": "initialization",
                "component": "adaptive_trading_system"
            })
            raise
    
    async def start_trading(self):
        """Start the main trading loop with singleton WebSocket"""
        logger.log_data_flow("startup", "trading_system")
        
        try:
            # Start SINGLETON Polygon WebSocket data feed
            await self._polygon_websocket.start()
            logger.connection("started", {"connection_type": "polygon_websocket"})
            
            # Set running flag
            self.running = True
            
            # Start main processing loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(e, {"operation": "start_trading"})
            await self.shutdown()
    
    async def _main_trading_loop(self):
        """Main trading loop that processes data and executes trades every 2 minutes"""
        
        logger.log_data_flow("trading_loop", "start", data_size=120)
        
        loop_count = 0
        
        while self.running:
            try:
                loop_count += 1
                logger.log_data_flow("trading_cycle", "start", data_size=loop_count)
                
                # Collect fresh Polygon data
                polygon_data = await self._collect_polygon_data()
                
                if polygon_data:
                    # Process through adaptive filter + ML models
                    trading_signals = await self.ml_processor.process_polygon_data(polygon_data)
                    
                    # Execute trades for valid signals
                    if trading_signals:
                        await self._execute_trading_signals(trading_signals)
                        self.total_signals_generated += len(trading_signals)
                
                # Log comprehensive system status every cycle
                await self._log_system_status()
                
                # Check for time-based exits every cycle
                await self._check_time_exit()
                
                logger.log_data_flow("trading_cycle", "complete", data_size=loop_count)
                
                # Wait 2 minutes before next cycle
                await asyncio.sleep(120)  # 2-minute intervals as requested
                
            except Exception as e:
                logger.error(e, {"operation": "trading_loop", "cycle": loop_count})
                await asyncio.sleep(30)  # Wait 30 seconds on error before retrying
    
    async def _collect_polygon_data(self) -> list:
        """Collect and format data from SINGLETON Polygon WebSocket"""
        
        if not self._polygon_websocket:
            return []
        
        # Get all symbols being tracked from singleton WebSocket
        symbols = self._polygon_websocket.get_all_symbols()
        
        polygon_data = []
        
        # Limit to first 1000 symbols for performance
        for symbol in symbols[:1000]:
            latest_data = self._polygon_websocket.get_latest_data(symbol)
            
            if latest_data:
                # Format for adaptive filter
                formatted_data = {
                    'symbol': symbol,
                    'price': latest_data.price,
                    'volume': latest_data.volume,
                    'timestamp': latest_data.timestamp,
                    'bid': latest_data.bid,
                    'ask': latest_data.ask,
                    'market_cap': self._estimate_market_cap(symbol, latest_data.price),
                    'daily_change': self._calculate_daily_change(symbol),
                    'volatility': self._calculate_volatility(symbol),
                    'momentum_score': self._calculate_momentum(symbol)
                }
                polygon_data.append(formatted_data)
        
        # Add market indicators
        market_indicators = await self._get_market_indicators()
        polygon_data.extend(market_indicators)
        
        return polygon_data
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap (simplified)"""
        # Large caps
        large_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        if symbol in large_caps:
            return 1000000000000  # $1T
        
        # Assume mid-cap for others
        return 10000000000  # $10B
    
    def _calculate_daily_change(self, symbol: str) -> float:
        """Calculate daily price change using singleton WebSocket"""
        try:
            symbol_buffer = self._polygon_websocket.get_symbol_buffer(symbol)
            if len(symbol_buffer) >= 2:
                current = symbol_buffer[-1].price
                previous = symbol_buffer[0].price
                if previous > 0:
                    return (current - previous) / previous
            return 0.0
        except:
            return 0.0
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility using singleton WebSocket"""
        try:
            symbol_buffer = self._polygon_websocket.get_symbol_buffer(symbol)
            if len(symbol_buffer) >= 10:
                prices = [data.price for data in symbol_buffer[-10:]]
                returns = []
                for i in range(1, len(prices)):
                    if prices[i-1] > 0:
                        ret = (prices[i] - prices[i-1]) / prices[i-1]
                        returns.append(ret)
                if returns:
                    import numpy as np
                    return float(np.std(returns))
            return 0.02
        except:
            return 0.02
    
    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum score using singleton WebSocket"""
        try:
            symbol_buffer = self._polygon_websocket.get_symbol_buffer(symbol)
            if len(symbol_buffer) >= 20:
                recent = [data.price for data in symbol_buffer[-20:]]
                older = [data.price for data in symbol_buffer[-40:-20]] if len(symbol_buffer) >= 40 else recent
                
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                
                if older_avg > 0:
                    return (recent_avg - older_avg) / older_avg
            return 0.0
        except:
            return 0.0
    
    async def _get_market_indicators(self) -> list:
        """Get VIX and SPY data from singleton WebSocket"""
        indicators = []
        
        # VIX
        vix_data = self._polygon_websocket.get_latest_data('VIX')
        if vix_data:
            indicators.append({
                'symbol': 'VIX',
                'price': vix_data.price,
                'volume': vix_data.volume,
                'timestamp': vix_data.timestamp,
                'market_cap': 0,
                'daily_change': self._calculate_daily_change('VIX')
            })
        
        # SPY
        spy_data = self._polygon_websocket.get_latest_data('SPY')
        if spy_data:
            indicators.append({
                'symbol': 'SPY',
                'price': spy_data.price,
                'volume': spy_data.volume,
                'timestamp': spy_data.timestamp,
                'market_cap': 0,
                'daily_change': self._calculate_daily_change('SPY')
            })
        
        return indicators
    
    async def _execute_trading_signals(self, trading_signals: list):
        """Execute momentum trades using Kelly Criterion and Alpaca integration"""
        
        if not trading_signals:
            return
        
        try:
            # Filter for bullish signals only (momentum strategy is long-biased)
            bullish_signals = [s for s in trading_signals if s.prediction > 0]
            
            if not bullish_signals:
                logger.log_data_flow("signal_filter", "no_bullish_signals")
                return
            
            logger.log_data_flow("trade_execution", "momentum_trades", data_size=len(bullish_signals))
            
            # Execute trades using momentum engine
            execution_summary = await self.momentum_engine.execute_multiple_trades(bullish_signals)
            
            # Update trade statistics
            self.total_trades_executed += execution_summary['successful_executions']
            
            # Log execution results
            logger.performance({
                "successful_executions": execution_summary['successful_executions'],
                "total_signals": len(bullish_signals),
                "capital_deployed": execution_summary['total_capital_deployed']
            })
            
            if execution_summary['failed_symbols']:
                logger.warning(f"Failed executions: {execution_summary['failed_symbols']}")
            
        except Exception as e:
            logger.error(e, {"operation": "execute_trading_signals"})
    
    async def _log_system_status(self):
        """Log comprehensive system performance status"""
        
        try:
            # Get filter stats
            filter_stats = self.ml_processor.adaptive_filter.get_filter_stats()
            
            # Get ML processor stats
            self.ml_processor.get_performance_stats()
            
            # Get momentum execution stats
            momentum_stats = self.momentum_engine.get_execution_stats()
            
            # Get account status
            account_status = await self.momentum_engine.get_account_status()
            
            # Log comprehensive performance metrics
            system_metrics = {
                "signals_generated": self.total_signals_generated,
                "trades_executed": self.total_trades_executed,
                "success_rate_pct": momentum_stats['execution_summary']['success_rate_pct'],
                "capital_deployed": momentum_stats['execution_summary']['total_capital_deployed'],
                "market_condition": filter_stats['current_condition'],
                "filter_efficiency_pct": filter_stats['filter_efficiency'],
                "avg_processing_time_s": filter_stats['avg_processing_time'],
                "available_capital": momentum_stats['kelly_stats']['available_capital'],
                "kelly_safety_factor": momentum_stats['kelly_stats']['safety_factor']
            }
            
            if account_status:
                system_metrics.update({
                    "account_equity": account_status.get('equity', 0),
                    "buying_power": account_status.get('buying_power', 0),
                    "current_positions": account_status.get('num_positions', 0)
                })
            
            logger.performance(system_metrics)
            
        except Exception as e:
            logger.error(e, {"operation": "log_system_status"})
    
    async def _check_time_exit(self):
        """Check and execute time-based position exits at 3:45 PM"""
        try:
            from datetime import datetime, time
            current_time = datetime.now().time()
            time_exit = time(15, 45)  # 3:45 PM ET
            
            # Check if it's time to exit (within 1 minute window)
            if current_time >= time_exit and current_time <= time(15, 46):
                logger.log_data_flow("time_exit", "window_reached")
                exit_result = await self.momentum_engine.close_all_positions_at_time_exit()
                
                if exit_result['status'] == 'completed':
                    logger.performance({"time_exit_closed_positions": exit_result['closed_positions']})
                elif exit_result['status'] == 'no_positions':
                    logger.log_data_flow("time_exit", "no_positions")
                else:
                    logger.warning(f"Time exit status: {exit_result}")
                    
        except Exception as e:
            logger.error(e, {"operation": "check_time_exit"})
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.log_data_flow("shutdown", "trading_system")
        
        self.running = False
        
        # Shutdown singleton Polygon WebSocket
        if self._polygon_websocket:
            await self._polygon_websocket.stop()
        
        # Shutdown singleton Alpaca WebSocket if exists
        if self._alpaca_websocket:
            await self._alpaca_websocket.stop()
        
        if self.ml_processor:
            await self.ml_processor.save_state()
        
        logger.log_data_flow("shutdown", "complete")

def setup_logging():
    """Setup logging configuration - now handled by enhanced system logger"""
    # Enhanced logging is automatically configured via YAML
    # This function is kept for compatibility but no longer needed
    pass

def signal_handler(trading_system):
    """Handle shutdown signals"""
    def handler(signum, frame):
        logger.log_data_flow("signal_handler", "shutdown_signal", data_size=signum)
        asyncio.create_task(trading_system.shutdown())
    return handler

async def main():
    """Main function"""
    
    # Setup logging
    setup_logging()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.log_data_flow("main", "startup")
    
    # Create trading system
    trading_system = AdaptiveTradingSystem()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(trading_system))
    signal.signal(signal.SIGTERM, signal_handler(trading_system))
    
    try:
        # Initialize and start
        await trading_system.initialize()
        await trading_system.start_trading()
        
    except KeyboardInterrupt:
        logger.log_data_flow("main", "keyboard_interrupt")
    except Exception as e:
        logger.error(e, {"operation": "main"})
    finally:
        await trading_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())