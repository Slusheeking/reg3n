"""
Main Trading System Orchestrator
Coordinates Polygon data, Lag-Llama forecasting, trading strategies, and Alpaca execution
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
import orjson as json
from dataclasses import asdict

# Import all system components
from settings import config
from active_symbols import symbol_manager
from lag_llama_engine import lag_llama_engine
from polygon import get_polygon_data_manager
from alpaca import get_alpaca_client
from gap_n_go import gap_and_go_strategy
from orb import orb_strategy
from mean_reversion import mean_reversion_strategy

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemOrchestrator:
    """Main orchestrator for the complete trading system"""
    
    def __init__(self):
        self.running = False
        self.system_start_time: Optional[datetime] = None
        
        # System components (lazy-initialized)
        self.polygon_data_manager: Optional[Any] = None
        self.alpaca_client: Optional[Any] = None
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
    
    async def initialize_system(self):
        """Initialize all system components"""
        
        logger.info("=" * 60)
        logger.info("INITIALIZING TRADING SYSTEM")
        logger.info("=" * 60)
        
        try:
            # Validate configuration
            if not config.validate_config():
                raise Exception("Configuration validation failed")
            
            logger.info("✓ Configuration validated")
            
            # Initialize components in order
            await self._initialize_components()
            
            # Load historical data and prepare for trading
            await self._prepare_for_trading()
            
            self.system_start_time = datetime.now()
            logger.info("✓ System initialization completed successfully")
            logger.info(f"System ready at: {self.system_start_time}")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all system components"""
        
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
    
    async def _prepare_for_trading(self):
        """Prepare system for active trading"""
        
        logger.info("Preparing system for trading...")
        
        # Load active symbols
        active_symbols = symbol_manager.get_active_symbols()
        logger.info(f"Tracking {len(active_symbols)} active symbols")
        
        # Subscribe to real-time data
        if self.polygon_data_manager is not None:
            for symbol in active_symbols:
                await self.polygon_data_manager.subscribe_symbol(symbol)
        
        logger.info("✓ Real-time data subscriptions active")
        
        # Pre-load some historical data for Lag-Llama
        logger.info("Pre-loading historical data...")
        await self._preload_historical_data(active_symbols[:10])  # Load top 10 symbols
        
        logger.info("✓ Historical data preloaded")
    
    async def _preload_historical_data(self, symbols: List[str]):
        """Preload historical data for key symbols"""
        
        for symbol in symbols:
            try:
                # Get recent historical data
                if self.polygon_data_manager is not None:
                    historical_data = await self.polygon_data_manager.get_historical_data(
                        symbol, days=5, timespan="minute"
                    )
                else:
                    continue
                
                # Add to Lag-Llama price buffers
                for bar in historical_data[-100:]:  # Last 100 minutes
                    lag_llama_engine.add_price_data(
                        symbol, 
                        bar['close'], 
                        bar['volume'], 
                        bar['timestamp']
                    )
                
            except Exception as e:
                logger.warning(f"Could not preload data for {symbol}: {e}")
    
    async def run_trading_system(self):
        """Main trading system loop"""
        
        logger.info("=" * 60)
        logger.info("STARTING TRADING SYSTEM")
        logger.info("=" * 60)
        
        self.running = True
        
        try:
            # Start all concurrent tasks
            tasks = [
                # Market session management
                self._market_session_manager(),
                
                # Strategy execution
                self._strategy_coordinator(),
                
                # System monitoring
                self._system_monitor(),
                
                # Performance tracking
                self._performance_tracker(),
                
                # Risk monitoring
                self._risk_monitor(),
                
                # Periodic maintenance
                self._periodic_maintenance()
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Critical error in trading system: {e}")
            await self._emergency_shutdown()
        
        finally:
            await self._cleanup_system()
    
    async def _market_session_manager(self):
        """Manage different market sessions and trading phases"""
        
        logger.info("Starting market session manager...")
        
        while self.running:
            try:
                current_time = datetime.now()
                current_hour = current_time.hour
                current_minute = current_time.minute
                
                # Pre-market phase (4:00 AM - 9:30 AM ET)
                if 4 <= current_hour < 9 or (current_hour == 9 and current_minute < 30):
                    await self._handle_premarket_session()
                
                # Market open phase (9:30 AM - 4:00 PM ET)
                elif config.is_market_hours(current_time):
                    await self._handle_market_session()
                
                # After-hours phase (4:00 PM - 8:00 PM ET)
                elif 16 <= current_hour < 20:
                    await self._handle_afterhours_session()
                
                # Overnight phase (8:00 PM - 4:00 AM ET)
                else:
                    await self._handle_overnight_session()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in market session manager: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _handle_premarket_session(self):
        """Handle pre-market trading session"""
        
        if not self.premarket_analysis_done:
            logger.info("Starting pre-market analysis...")
            
            # Analyze gaps for Gap & Go strategy
            gap_candidates = await gap_and_go_strategy.analyze_premarket_gaps()
            
            if gap_candidates:
                logger.info(f"Found {len(gap_candidates)} gap candidates")
                
                # Generate and queue gap signals
                gap_signals = await gap_and_go_strategy.generate_signals()
                logger.info(f"Generated {len(gap_signals)} gap signals")
            
            # Update symbol metrics with gap information
            for symbol in symbol_manager.get_active_symbols():
                metrics = symbol_manager.metrics.get(symbol)
                if metrics and abs(metrics.gap_percent) > 2.0:
                    logger.info(f"{symbol}: Gap {metrics.gap_percent:.1f}%")
            
            self.premarket_analysis_done = True
            logger.info("✓ Pre-market analysis completed")
    
    async def _handle_market_session(self):
        """Handle regular market session"""
        
        if not self.market_session_active:
            logger.info("Market session started - activating strategies")
            self.market_session_active = True
            
            # Execute any queued Gap & Go signals
            gap_signals = gap_and_go_strategy.get_active_signals()
            if gap_signals:
                executed = await gap_and_go_strategy.execute_signals(gap_signals)
                logger.info(f"Executed {executed} gap trades at market open")
        
        # Start ORB formation at market open
        current_time = datetime.now()
        if (current_time.hour == 9 and current_time.minute == 30 and 
            not self.orb_formation_started):
            
            logger.info("Starting Opening Range Breakout formation")
            await orb_strategy.start_range_formation()
            self.orb_formation_started = True
    
    async def _handle_afterhours_session(self):
        """Handle after-hours session"""
        
        if self.market_session_active:
            logger.info("Market session ended - wrapping up trades")
            
            # Close any remaining day trades
            await self._close_day_trades()
            
            # Generate end-of-day reports
            await self._generate_daily_reports()
            
            self.market_session_active = False
    
    async def _handle_overnight_session(self):
        """Handle overnight session"""
        
        # Minimal activity - just system maintenance
        if self.premarket_analysis_done:
            # Reset for next day
            await self._reset_daily_state()
    
    async def _strategy_coordinator(self):
        """Coordinate execution of all trading strategies"""
        
        logger.info("Starting strategy coordinator...")
        
        while self.running:
            try:
                if self.market_session_active:
                    # Update Lag-Llama forecasts
                    await self._update_forecasts()
                    
                    # Monitor active positions and signals
                    await self._monitor_active_trades()
                    
                    # Check for new opportunities
                    await self._scan_new_opportunities()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in strategy coordinator: {e}")
                await asyncio.sleep(60)
    
    async def _update_forecasts(self):
        """Update Lag-Llama forecasts for active symbols"""
        
        try:
            # Get symbols with active positions or signals
            active_symbols = set()
            
            # Add symbols with positions
            if self.alpaca_client is not None and hasattr(self.alpaca_client, 'positions'):
                for symbol in self.alpaca_client.positions.keys():
                    active_symbols.add(symbol)
            
            # Add symbols with active signals
            for strategy in [gap_and_go_strategy, orb_strategy, mean_reversion_strategy]:
                for signal in strategy.get_active_signals():
                    active_symbols.add(signal.symbol)
            
            # Add top gap candidates
            gap_candidates = symbol_manager.get_gap_candidates()[:5]
            active_symbols.update(gap_candidates)
            
            # Generate forecasts
            if active_symbols:
                forecasts = await lag_llama_engine.generate_forecasts(list(active_symbols))
                self.system_metrics['forecasts_generated'] += len(forecasts)
                
        except Exception as e:
            logger.error(f"Error updating forecasts: {e}")
            self.system_metrics['errors_encountered'] += 1
    
    async def _monitor_active_trades(self):
        """Monitor active trades and signals"""
        
        try:
            # Get portfolio summary
            if self.alpaca_client is not None:
                portfolio = self.alpaca_client.get_portfolio_summary()
                
                # Update system metrics
                self.system_metrics['total_trades'] = portfolio['daily_trades']
                self.system_metrics['total_pnl'] = portfolio['daily_pnl']
                
                # Check for risk limit violations
                if portfolio['max_daily_loss_hit']:
                    logger.warning("Daily loss limit hit - emergency shutdown initiated")
                    await self._emergency_shutdown()
            
        except Exception as e:
            logger.error(f"Error monitoring trades: {e}")
    
    async def _scan_new_opportunities(self):
        """Scan for new trading opportunities"""
        
        try:
            # Mean reversion opportunities are scanned continuously
            # Gap & Go opportunities are identified pre-market
            # ORB opportunities are identified after range formation
            
            # Just ensure strategies are running their scans
            pass
            
        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
    
    async def _system_monitor(self):
        """Monitor system health and performance"""
        
        logger.info("Starting system monitor...")
        
        while self.running:
            try:
                await self._check_system_health()
                await self._log_system_status()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(600)
    
    async def _check_system_health(self):
        """Check health of all system components"""
        
        health_status = {}
        
        # Check Polygon connection
        if self.polygon_data_manager is not None:
            polygon_stats = self.polygon_data_manager.get_connection_stats()
            health_status['polygon'] = polygon_stats['connected']
        else:
            health_status['polygon'] = False
        
        # Check Lag-Llama performance
        lagllama_metrics = lag_llama_engine.get_performance_metrics()
        health_status['lag_llama'] = lagllama_metrics['model_loaded']
        
        # Check Alpaca connection
        if self.alpaca_client is not None:
            portfolio = self.alpaca_client.get_portfolio_summary()
            health_status['alpaca'] = portfolio['account_equity'] > 0
        else:
            health_status['alpaca'] = False
        
        # Log any unhealthy components
        for component, healthy in health_status.items():
            if not healthy:
                logger.warning(f"Component {component} is unhealthy")
    
    async def _log_system_status(self):
        """Log comprehensive system status"""
        
        if self.system_start_time:
            self.system_metrics['uptime'] = datetime.now() - self.system_start_time
        
        # Get component statistics
        if self.polygon_data_manager is not None:
            polygon_stats = self.polygon_data_manager.get_connection_stats()
        else:
            polygon_stats = {'messages_received': 0}
            
        lagllama_metrics = lag_llama_engine.get_performance_metrics()
        
        if self.alpaca_client is not None:
            portfolio = self.alpaca_client.get_portfolio_summary()
        else:
            portfolio = {'account_equity': 0, 'daily_pnl': 0, 'daily_trades': 0}
        
        # Log summary
        logger.info(f"System Status - Uptime: {self.system_metrics['uptime']}")
        logger.info(f"Portfolio: ${portfolio['account_equity']:,.2f} | "
                   f"P&L: ${portfolio['daily_pnl']:,.2f} | "
                   f"Trades: {portfolio['daily_trades']}")
        logger.info(f"Data: {polygon_stats['messages_received']} msgs | "
                   f"Forecasts: {self.system_metrics['forecasts_generated']} | "
                   f"Errors: {self.system_metrics['errors_encountered']}")
    
    async def _performance_tracker(self):
        """Track and analyze system performance"""
        
        logger.info("Starting performance tracker...")
        
        while self.running:
            try:
                await self._analyze_strategy_performance()
                await self._track_forecast_accuracy()
                
                await asyncio.sleep(900)  # Every 15 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracker: {e}")
                await asyncio.sleep(1800)
    
    async def _analyze_strategy_performance(self):
        """Analyze performance of each strategy"""
        
        strategies = {
            'gap_and_go': gap_and_go_strategy,
            'orb': orb_strategy,
            'mean_reversion': mean_reversion_strategy
        }
        
        for name, strategy in strategies.items():
            try:
                performance = strategy.get_strategy_performance()
                
                if performance['trades_today'] > 0:
                    logger.info(f"{name.upper()}: "
                              f"{performance['trades_today']} trades, "
                              f"${performance.get('total_pnl', 0):.2f} P&L")
                
            except Exception as e:
                logger.error(f"Error analyzing {name} performance: {e}")
    
    async def _track_forecast_accuracy(self):
        """Track Lag-Llama forecast accuracy"""
        
        try:
            # This would compare Lag-Llama predictions with actual outcomes
            # Implementation depends on having stored historical forecasts
            pass
            
        except Exception as e:
            logger.error(f"Error tracking forecast accuracy: {e}")
    
    async def _risk_monitor(self):
        """Monitor risk metrics and enforce limits"""
        
        logger.info("Starting risk monitor...")
        
        while self.running:
            try:
                await self._check_risk_limits()
                await self._monitor_position_correlation()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(300)
    
    async def _check_risk_limits(self):
        """Check various risk limits"""
        
        if self.alpaca_client is not None:
            portfolio = self.alpaca_client.get_portfolio_summary()
            
            # Check daily loss limit
            daily_loss_limit = config.risk.max_daily_loss * portfolio['account_equity']
            if portfolio['daily_pnl'] < -daily_loss_limit:
                logger.warning("Daily loss limit exceeded!")
                await self._emergency_shutdown()
            
            # Check position concentration
            if portfolio['positions_count'] > config.alpaca.max_positions:
                logger.warning("Too many open positions")
            
            # Check total exposure
            if portfolio['total_position_value'] > portfolio['account_equity'] * 0.95:
                logger.warning("High portfolio exposure")
    
    async def _monitor_position_correlation(self):
        """Monitor correlation between positions"""
        
        try:
            # This would calculate correlation between current positions
            # and warn if portfolio is too concentrated in correlated assets
            pass
            
        except Exception as e:
            logger.error(f"Error monitoring correlation: {e}")
    
    async def _periodic_maintenance(self):
        """Perform periodic system maintenance"""
        
        logger.info("Starting periodic maintenance...")
        
        while self.running:
            try:
                # Clean up old data
                await self._cleanup_old_data()
                
                # Update symbol lists
                await self._update_symbol_lists()
                
                # Save system state
                await self._save_system_state()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in periodic maintenance: {e}")
                await asyncio.sleep(7200)
    
    async def _cleanup_old_data(self):
        """Clean up old data and logs"""
        
        try:
            # Clear old forecast cache entries
            # Clear old performance data
            # Rotate log files if needed
            pass
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")
    
    async def _update_symbol_lists(self):
        """Update active symbol lists"""
        
        try:
            # Refresh gap candidates
            # Update high volume symbols
            # Check for new earnings plays
            pass
            
        except Exception as e:
            logger.error(f"Error updating symbol lists: {e}")
    
    async def _save_system_state(self):
        """Save current system state"""
        
        try:
            portfolio = {}
            if self.alpaca_client is not None:
                portfolio = self.alpaca_client.get_portfolio_summary()
            
            system_state = {
                'timestamp': datetime.now().isoformat(),
                'uptime': str(self.system_metrics['uptime']),
                'metrics': self.system_metrics,
                'active_symbols': symbol_manager.get_active_symbols(),
                'portfolio': portfolio
            }
            
            with open('system_state.json', 'wb') as f: # orjson.dumps returns bytes
                f.write(json.dumps(system_state, option=json.OPT_INDENT_2, default=str))
                
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    async def _close_day_trades(self):
        """Close any remaining day trades"""
        
        logger.info("Closing day trades before market close...")
        
        try:
            # Close positions opened today
            if self.alpaca_client is not None:
                closed_count = await self.alpaca_client.close_all_positions("End of day")
                
                if closed_count > 0:
                    logger.info(f"Closed {closed_count} day trade positions")
            
        except Exception as e:
            logger.error(f"Error closing day trades: {e}")
    
    async def _generate_daily_reports(self):
        """Generate end-of-day reports"""
        
        logger.info("Generating daily reports...")
        
        try:
            # Strategy performance reports
            strategies = {
                'gap_and_go': gap_and_go_strategy.get_strategy_performance(),
                'orb': orb_strategy.get_strategy_performance(),
                'mean_reversion': mean_reversion_strategy.get_strategy_performance()
            }
            
            # Portfolio performance
            portfolio = {}
            if self.alpaca_client:
                portfolio = self.alpaca_client.get_portfolio_summary()
            
            # System performance
            system_performance = {
                'uptime': str(self.system_metrics['uptime']),
                'total_forecasts': self.system_metrics['forecasts_generated'],
                'total_errors': self.system_metrics['errors_encountered']
            }
            
            # Save daily report
            daily_report = {
                'date': datetime.now().date().isoformat(),
                'portfolio': portfolio,
                'strategies': strategies,
                'system': system_performance
            }
            
            report_filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_filename, 'wb') as f: # orjson.dumps returns bytes
                f.write(json.dumps(daily_report, option=json.OPT_INDENT_2, default=str))
            
            logger.info(f"Daily report saved: {report_filename}")
            
        except Exception as e:
            logger.error(f"Error generating daily reports: {e}")
    
    async def _reset_daily_state(self):
        """Reset state for new trading day"""
        
        logger.info("Resetting daily state...")
        
        try:
            # Reset strategy states
            await gap_and_go_strategy.daily_reset()
            await orb_strategy.daily_reset()
            await mean_reversion_strategy.daily_reset()
            
            # Reset system flags
            self.market_session_active = False
            self.premarket_analysis_done = False
            self.orb_formation_started = False
            
            # Reset symbol manager
            await symbol_manager.daily_symbol_refresh()
            
            logger.info("✓ Daily state reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting daily state: {e}")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        
        logger.warning("EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Cancel all open orders
            if self.alpaca_client:
                await self.alpaca_client.cancel_all_orders("Emergency shutdown")
            
            # Close all positions if needed
            if self.alpaca_client:
                await self.alpaca_client.close_all_positions("Emergency shutdown")
            
            # Save emergency state
            await self._save_system_state()
            
            self.running = False
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
    
    async def _cleanup_system(self):
        """Clean up system resources"""
        
        logger.info("Cleaning up system resources...")
        
        try:
            # Cleanup each component
            if self.polygon_data_manager:
                await self.polygon_data_manager.cleanup()
            if self.alpaca_client:
                await self.alpaca_client.cleanup()
            
            # Save final state
            await self._save_system_state()
            
            logger.info("✓ System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        
        logger.info(f"Received signal {signum} - initiating graceful shutdown...")
        self.running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Give tasks time to finish
        await asyncio.sleep(5)
        
        await self._cleanup_system()
        
        logger.info("System shutdown completed")

async def main():
    """Main entry point"""
    
    print("""
    ╔═════════════════════════════════════════════════════╗
    ║              LAG-LLAMA TRADING SYSTEM v1.0          ║
    ║                                                     ║
    ║  Polygon Data → Lag-Llama AI → Strategies → Alpaca  ║
    ║                                                     ║
    ║  Strategies: Gap & Go | ORB | Mean Reversion        ║
    ║  Hardware: GH200 Grace Hopper Superchip             ║
    ╚═════════════════════════════════════════════════════╝
    """)
    
    orchestrator = TradingSystemOrchestrator()
    
    try:
        # Initialize system
        await orchestrator.initialize_system()
        
        # Run trading system
        await orchestrator.run_trading_system()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    # Set event loop policy for better performance on Linux
    if sys.platform.startswith('linux'):
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            # uvloop not available, use default policy
            pass
    
    # Run the main trading system
    asyncio.run(main())
