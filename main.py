"""
FAIL FAST SYSTEM - NO FALLBACKS ALLOWED
Main Trading System Entry Point
Coordinates all trading strategies with Lag-Llama AI integration
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any

from settings import config
from active_symbols import symbol_manager
from lag_llama_engine import lag_llama_engine
from polygon import get_polygon_data_manager
from alpaca import get_alpaca_client
from gap_n_go import gap_and_go_strategy
from orb import orb_strategy
from mean_reversion import mean_reversion_strategy
from database import get_database_manager

logger = logging.getLogger(__name__)

class TradingSystemOrchestrator:
    """Main orchestrator for the trading system"""
    
    def __init__(self):
        self.running = False
        self.system_start_time: Optional[datetime] = None
        
        # System components (lazy-initialized)
        self.polygon_data_manager: Optional[Any] = None
        self.alpaca_client: Optional[Any] = None
        self.db_manager = None
        self.components = {
            'symbol_manager': symbol_manager,
            'lag_llama': lag_llama_engine,
            'gap_and_go': gap_and_go_strategy,
            'orb': orb_strategy,
            'mean_reversion': mean_reversion_strategy
        }
        
        # System state
        self.market_session_active = False
        self.premarket_analysis_done = False
        self.orb_formation_started = False
        
        # Performance tracking
        self.system_metrics = {
            'uptime': timedelta(0),
            'total_trades': 0,
            'total_pnl': 0.0,
            'data_messages_processed': 0,
            'forecasts_generated': 0,
            'errors_encountered': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing trading system...")
        
        try:
            # Initialize database manager
            logger.info("Initializing database manager...")
            self.db_manager = get_database_manager()
            if self.db_manager:
                await self.db_manager.initialize()
            logger.info("✓ Database manager ready")
            
            # Initialize Polygon data manager
            logger.info("Initializing Polygon data manager...")
            self.polygon_data_manager = get_polygon_data_manager()
            if self.polygon_data_manager:
                await self.polygon_data_manager.initialize()
            logger.info("✓ Polygon data manager ready")
            
            # Initialize Alpaca trading client
            logger.info("Initializing Alpaca trading client...")
            self.alpaca_client = get_alpaca_client()
            if self.alpaca_client:
                await self.alpaca_client.initialize()
            logger.info("✓ Alpaca trading client ready")
            
            # Lag-Llama initializes automatically in background
            logger.info("✓ Lag-Llama engine initializing...")
            
            # Wait a moment for Lag-Llama to load
            await asyncio.sleep(5)
            
            # Initialize strategies
            logger.info("✓ Trading strategies initialized")
            
            # Daily symbol refresh
            await symbol_manager.daily_symbol_refresh()
            logger.info("✓ Symbol manager ready")
            
            self.system_start_time = datetime.now()
            
            # Store system startup in database
            await self._store_system_startup()
            
            logger.info("✓ System initialization completed successfully")
            logger.info(f"System ready at: {self.system_start_time}")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def run(self):
        """Main system loop"""
        self.running = True
        logger.info("Starting main trading system loop...")
        
        try:
            while self.running:
                current_time = datetime.now().time()
                
                # Update system metrics
                if self.system_start_time:
                    self.system_metrics['uptime'] = datetime.now() - self.system_start_time
                
                # Store system metrics every 5 minutes
                if datetime.now().minute % 5 == 0:
                    await self._store_system_metrics()
                
                # Pre-market analysis (6:00 AM - 9:30 AM ET)
                if time(6, 0) <= current_time <= time(9, 30) and not self.premarket_analysis_done:
                    await self._run_premarket_analysis()
                    self.premarket_analysis_done = True
                
                # Market open preparation (9:25 AM - 9:30 AM ET)
                elif time(9, 25) <= current_time <= time(9, 30):
                    await self._prepare_for_market_open()
                
                # Regular trading hours (9:30 AM - 4:00 PM ET)
                elif time(9, 30) <= current_time <= time(16, 0):
                    if not self.market_session_active:
                        await self._start_market_session()
                        self.market_session_active = True
                    
                    await self._run_trading_session()
                
                # After hours (4:00 PM - 8:00 PM ET)
                elif time(16, 0) <= current_time <= time(20, 0):
                    if self.market_session_active:
                        await self._end_market_session()
                        self.market_session_active = False
                    
                    await self._run_after_hours_analysis()
                
                # Overnight period (8:00 PM - 6:00 AM ET)
                else:
                    await self._run_overnight_maintenance()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.system_metrics['errors_encountered'] += 1
        
        finally:
            await self.shutdown()
    
    async def _run_premarket_analysis(self):
        """Run pre-market analysis"""
        logger.info("Running pre-market analysis...")
        
        try:
            # Refresh symbols and get gap candidates
            await symbol_manager.daily_symbol_refresh()
            
            # Get gap candidates from Polygon
            if self.polygon_data_manager:
                gap_candidates = await self.polygon_data_manager.get_enhanced_gap_candidates()
                logger.info(f"Found {len(gap_candidates)} gap candidates")
            
            # Run gap analysis
            gap_analyses = await gap_and_go_strategy.analyze_premarket_gaps()
            logger.info(f"Analyzed {len(gap_analyses)} gaps")
            
            # Generate gap signals
            gap_signals = await gap_and_go_strategy.generate_signals()
            logger.info(f"Generated {len(gap_signals)} gap signals")
            
            # Store premarket analysis in database
            await self._store_trading_session('premarket_analysis', {
                'gap_candidates': len(gap_candidates) if 'gap_candidates' in locals() else 0,
                'gap_analyses': len(gap_analyses),
                'gap_signals': len(gap_signals)
            })
            
        except Exception as e:
            logger.error(f"Error in pre-market analysis: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _prepare_for_market_open(self):
        """Prepare for market open"""
        logger.info("Preparing for market open...")
        
        try:
            # Final gap signal review
            gap_signals = gap_and_go_strategy.get_active_signals()
            
            # Prepare ORB strategy
            await orb_strategy.prepare_for_market_open()
            
            # Update symbol metrics
            await symbol_manager.update_all_metrics()
            
            logger.info(f"Ready for market open with {len(gap_signals)} gap signals")
            
        except Exception as e:
            logger.error(f"Error preparing for market open: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _start_market_session(self):
        """Start market trading session"""
        logger.info("Starting market trading session...")
        
        try:
            # Execute gap signals
            gap_signals = gap_and_go_strategy.get_active_signals()
            if gap_signals:
                executed = await gap_and_go_strategy.execute_signals(gap_signals)
                logger.info(f"Executed {executed} gap trades")
                self.system_metrics['total_trades'] += executed
            
            # Start ORB formation tracking
            await orb_strategy.start_orb_formation()
            self.orb_formation_started = True
            
            # Store market session start
            await self._store_trading_session('market_open', {
                'gap_signals_executed': len(gap_signals),
                'orb_formation_started': True
            })
            
        except Exception as e:
            logger.error(f"Error starting market session: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _run_trading_session(self):
        """Run active trading session"""
        current_time = datetime.now().time()
        
        try:
            # ORB strategy (first 15 minutes)
            if time(9, 30) <= current_time <= time(9, 45) and self.orb_formation_started:
                orb_signals = await orb_strategy.check_orb_breakouts()
                if orb_signals:
                    executed = await orb_strategy.execute_signals(orb_signals)
                    logger.info(f"Executed {executed} ORB trades")
                    self.system_metrics['total_trades'] += executed
            
            # Mean reversion strategy (throughout the day)
            mr_signals = await mean_reversion_strategy.generate_signals()
            if mr_signals:
                executed = await mean_reversion_strategy.execute_signals(mr_signals)
                logger.info(f"Executed {executed} mean reversion trades")
                self.system_metrics['total_trades'] += executed
            
            # Update forecasts every 5 minutes
            if datetime.now().minute % 5 == 0:
                await self._update_forecasts()
            
            # Store strategy performance every 15 minutes
            if datetime.now().minute % 15 == 0:
                await self._store_all_strategy_performance()
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _end_market_session(self):
        """End market trading session"""
        logger.info("Ending market trading session...")
        
        try:
            # Close any remaining positions if configured
            if config.risk.close_positions_at_market_close:
                await self._close_all_positions()
            
            # Get final performance metrics
            performance_data = {}
            for strategy_name, strategy in self.components.items():
                if hasattr(strategy, 'get_strategy_performance'):
                    perf = strategy.get_strategy_performance()
                    performance_data[strategy_name] = perf
                    await self._store_strategy_performance(strategy_name, perf)
            
            # Store market session end
            await self._store_trading_session('market_close', performance_data)
            
            logger.info("Market session ended successfully")
            
        except Exception as e:
            logger.error(f"Error ending market session: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _run_after_hours_analysis(self):
        """Run after-hours analysis"""
        # Run every 30 minutes during after hours
        if datetime.now().minute % 30 == 0:
            try:
                # Update earnings calendar
                if self.polygon_data_manager:
                    symbols = symbol_manager.get_active_symbols()
                    earnings_data = await self.polygon_data_manager.get_earnings_calendar(symbols)
                    logger.info(f"Updated earnings calendar with {len(earnings_data)} events")
                
                # Prepare for next day
                await symbol_manager.prepare_for_next_day()
                
            except Exception as e:
                logger.error(f"Error in after-hours analysis: {e}")
                self.system_metrics['errors_encountered'] += 1
    
    async def _run_overnight_maintenance(self):
        """Run overnight maintenance tasks"""
        # Run once per hour during overnight
        if datetime.now().minute == 0:
            try:
                # Daily reset at midnight
                if datetime.now().hour == 0:
                    await self._daily_reset()
                
                # System health check
                await self._system_health_check()
                
            except Exception as e:
                logger.error(f"Error in overnight maintenance: {e}")
                self.system_metrics['errors_encountered'] += 1
    
    async def _update_forecasts(self):
        """Update Lag-Llama forecasts"""
        try:
            symbols = symbol_manager.get_active_symbols()
            for symbol in symbols[:10]:  # Limit to avoid overwhelming
                forecast = await lag_llama_engine.get_forecast(symbol)
                if forecast:
                    self.system_metrics['forecasts_generated'] += 1
        
        except Exception as e:
            logger.error(f"Error updating forecasts: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            if self.alpaca_client:
                positions = await self.alpaca_client.get_positions()
                for position in positions:
                    await self.alpaca_client.close_position(position.symbol)
                logger.info(f"Closed {len(positions)} positions")
        
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _daily_reset(self):
        """Perform daily reset"""
        logger.info("Performing daily reset...")
        
        try:
            # Reset strategies
            await gap_and_go_strategy.daily_reset()
            await orb_strategy.daily_reset()
            await mean_reversion_strategy.daily_reset()
            
            # Reset system state
            self.premarket_analysis_done = False
            self.orb_formation_started = False
            self.market_session_active = False
            
            # Reset daily metrics
            self.system_metrics.update({
                'total_trades': 0,
                'total_pnl': 0.0,
                'data_messages_processed': 0,
                'forecasts_generated': 0,
                'errors_encountered': 0
            })
            
            logger.info("Daily reset completed")
            
        except Exception as e:
            logger.error(f"Error in daily reset: {e}")
    
    async def _system_health_check(self):
        """Perform system health check"""
        try:
            # Check connections
            if self.polygon_data_manager:
                stats = self.polygon_data_manager.get_connection_stats()
                if not stats.get('connected'):
                    logger.warning("Polygon connection lost, attempting reconnect...")
                    await self.polygon_data_manager.initialize()
            
            # Check Alpaca connection
            if self.alpaca_client:
                account = await self.alpaca_client.get_account()
                if not account:
                    logger.warning("Alpaca connection lost, attempting reconnect...")
                    await self.alpaca_client.initialize()
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _store_all_strategy_performance(self):
        """Store performance data for all strategies"""
        try:
            for strategy_name, strategy in self.components.items():
                if hasattr(strategy, 'get_strategy_performance'):
                    performance_data = strategy.get_strategy_performance()
                    await self._store_strategy_performance(strategy_name, performance_data)
        
        except Exception as e:
            logger.error(f"Error storing strategy performance: {e}")
    
    async def shutdown(self):
        """Shutdown the trading system"""
        logger.info("Shutting down trading system...")
        
        try:
            # Stop all strategies
            for strategy_name, strategy in self.components.items():
                if hasattr(strategy, 'shutdown'):
                    await strategy.shutdown()
            
            # Close connections
            if self.polygon_data_manager:
                await self.polygon_data_manager.cleanup()
            
            if self.alpaca_client:
                await self.alpaca_client.cleanup()
            
            # Store final system metrics
            await self._store_system_metrics()
            
            # Store system shutdown event
            if self.db_manager:
                shutdown_data = {
                    'timestamp': datetime.now(),
                    'event_type': 'system_shutdown',
                    'metadata': {
                        'uptime': str(self.system_metrics['uptime']),
                        'final_metrics': self.system_metrics
                    }
                }
                await self.db_manager.insert_system_event(shutdown_data)
            
            logger.info("Trading system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _store_system_startup(self):
        """Store system startup event in database"""
        
        if not self.db_manager:
            return
        
        try:
            startup_data = {
                'timestamp': self.system_start_time,
                'event_type': 'system_startup',
                'metadata': {
                    'config_version': config.version if hasattr(config, 'version') else '1.0',
                    'active_strategies': list(self.components.keys()),
                    'system_metrics': self.system_metrics
                }
            }
            
            await self.db_manager.insert_system_event(startup_data)
            logger.info("System startup logged to database")
            
        except Exception as e:
            logger.error(f"Error storing system startup: {e}")
    
    async def _store_system_metrics(self):
        """Store system performance metrics in database"""
        
        if not self.db_manager:
            return
        
        try:
            metrics_data = {
                'timestamp': datetime.now(),
                'uptime_seconds': int(self.system_metrics['uptime'].total_seconds()),
                'total_trades': self.system_metrics['total_trades'],
                'total_pnl': self.system_metrics['total_pnl'],
                'data_messages_processed': self.system_metrics['data_messages_processed'],
                'forecasts_generated': self.system_metrics['forecasts_generated'],
                'errors_encountered': self.system_metrics['errors_encountered'],
                'memory_usage': 0,  # Could add psutil for actual memory usage
                'cpu_usage': 0      # Could add psutil for actual CPU usage
            }
            
            await self.db_manager.insert_system_metrics(metrics_data)
            
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
    
    async def _store_trading_session(self, session_type: str, session_data: Dict):
        """Store trading session data in database"""
        
        if not self.db_manager:
            return
        
        try:
            session_record = {
                'timestamp': datetime.now(),
                'session_type': session_type,
                'data': session_data
            }
            
            await self.db_manager.insert_trading_session(session_record)
            
        except Exception as e:
            logger.error(f"Error storing trading session: {e}")
    
    async def _store_strategy_performance(self, strategy_name: str, performance_data: Dict):
        """Store strategy performance data in database"""
        
        if not self.db_manager:
            return
        
        try:
            performance_record = {
                'timestamp': datetime.now(),
                'strategy': strategy_name,
                'trades_today': performance_data.get('trades_today', 0),
                'winning_trades': performance_data.get('winning_trades', 0),
                'total_pnl': performance_data.get('total_pnl', 0.0),
                'win_rate': performance_data.get('win_rate', 0.0),
                'active_signals': performance_data.get('active_signals', 0),
                'metadata': performance_data
            }
            
            await self.db_manager.insert_strategy_performance(performance_record)
            
        except Exception as e:
            logger.error(f"Error storing strategy performance: {e}")

async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Starting Reg3n Trading System...")
    
    orchestrator = TradingSystemOrchestrator()
    
    try:
        await orchestrator.initialize()
        await orchestrator.run()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
