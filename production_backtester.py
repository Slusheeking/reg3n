#!/usr/bin/env python3

"""
Production-Native Backtester
Uses existing production components with historical data for perfect parity
"""

import asyncio
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

# Import historical data adapter and simulated executor
from historical_data_adapter import HistoricalDataAdapter
from simulated_executor import SimulatedAlpacaExecutor

# Import production components directly (no modifications needed)
from polygon_client import PolygonClient
from adaptive_data_filter import AdaptiveDataFilter
from feature_engineering import FeatureEngineer
from ml_ensemble_system import UltraFastMLEnsembleSystem
from kelly_position_sizer import UltraFastKellyPositionSizer
from enhanced_backtest_analytics import EnhancedBacktestAnalytics as BacktestAnalytics

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # ERROR
    YELLOW = '\033[93m'   # WARNING
    BLUE = '\033[94m'     # DEBUG
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # Reset to default

class SystemLogger:
    def __init__(self, name="production_backtester"):
        self.name = name
        self.color_map = {
            'ERROR': Colors.RED,
            'WARNING': Colors.YELLOW,
            'DEBUG': Colors.BLUE,
            'INFO': Colors.WHITE
        }
        
        # Create logs directory and file
        self.log_dir = '/home/ubuntu/reg3n-1/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, 'backtesting.log')
        
    def _format_message(self, level: str, message: str, colored: bool = True) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if colored:
            color = self.color_map.get(level, Colors.WHITE)
            return f"[{timestamp}] - {color}{level}{Colors.RESET} - [{self.name}]: {message}"
        else:
            return f"[{timestamp}] - {level} - [{self.name}]: {message}"
    
    def _write_to_file(self, level: str, message: str):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(self._format_message(level, str(message), colored=False) + '\n')
        except Exception:
            pass  # Fail silently
        
    def info(self, message, extra=None):
        print(self._format_message("INFO", str(message)))
        self._write_to_file("INFO", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("INFO", f"    Extra: {extra}")
    
    def debug(self, message, extra=None):
        print(self._format_message("DEBUG", str(message)))
        self._write_to_file("DEBUG", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("DEBUG", f"    Extra: {extra}")
    
    def warning(self, message, extra=None):
        print(self._format_message("WARNING", str(message)))
        self._write_to_file("WARNING", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("WARNING", f"    Extra: {extra}")
    
    def error(self, message, extra=None):
        print(self._format_message("ERROR", str(message)))
        self._write_to_file("ERROR", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("ERROR", f"    Extra: {extra}")

logger = SystemLogger()

class MockSignal:
    """Mock signal object for production compatibility"""
    def __init__(self, symbol, current_price, confidence, prediction=1.0):
        self.symbol = symbol
        self.current_price = current_price
        self.confidence = confidence
        self.prediction = prediction

class ProductionBacktester:
    """
    Production-native backtesting system that uses existing production components
    with historical data feeds for perfect parity
    """
    
    def __init__(self, polygon_api_key: str, initial_capital: float = 50000, 
                 symbols: List[str] = None, config: Dict = None):
        self.polygon_api_key = polygon_api_key
        self.initial_capital = initial_capital
        self.symbols = symbols or self._get_default_symbols()
        self.config = config or self._get_default_config()
        
        # Initialize historical data adapter
        self.data_adapter = HistoricalDataAdapter(polygon_api_key)
        
        # Initialize simulated executor
        self.executor = SimulatedAlpacaExecutor(
            initial_capital=initial_capital,
            slippage_bps=self.config.get('slippage_bps', 2),
            latency_ms=self.config.get('latency_ms', 50)
        )
        
        # Initialize production components (unchanged from production)
        self._initialize_production_components()
        
        # Backtest state
        self.backtest_results = {}
        self.daily_results = []
        self.current_date = None
        self.backtest_start_time = None
        self.backtest_end_time = None
        
        # Performance tracking
        self.total_trades = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.processing_times = []
        
        # Initialize comprehensive analytics system
        self.analytics = BacktestAnalytics(save_detailed_logs=True)
        
        logger.info(f"Production Backtester initialized with {len(self.symbols)} symbols")
        logger.info(f"Initial capital: ${initial_capital:,}")
        logger.info("✓ Comprehensive analytics system initialized")
    
    def _initialize_production_components(self):
        """Initialize production components exactly as in production system"""
        try:
            # Initialize Polygon client in historical mode
            self.polygon_client = PolygonClient(
                api_key=self.polygon_api_key,
                symbols=None,
                data_callback=None,
                enable_filtering=True,
                memory_pools=None,
                portfolio_manager=None,
                ml_bridge=None
            )
            
            # Initialize adaptive data filter (production component)
            self.adaptive_filter = AdaptiveDataFilter(
                polygon_client=self.polygon_client,
                portfolio_manager=None,  # Will be set up later if needed
                ml_bridge=None
            )
            
            # Initialize feature engineer (production component)
            self.feature_engineer = FeatureEngineer(
                portfolio_manager=None,
                ml_bridge=None
            )
            
            # Initialize ML ensemble system (production component)
            self.ml_system = UltraFastMLEnsembleSystem(
                gpu_enabled=True,
                memory_pools=None
            )
            
            # Initialize Kelly position sizer (production component)
            self.kelly_sizer = UltraFastKellyPositionSizer(
                available_capital=self.initial_capital
            )
            
            # Connect components (production architecture)
            self.ml_system.feature_engineer = self.feature_engineer
            self.kelly_sizer.ml_bridge = self.ml_system
            self.executor.kelly_sizer = self.kelly_sizer
            
            logger.info("✓ All production components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize production components: {e}")
            raise
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbol list for backtesting"""
        return [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            # Large cap other sectors
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA',
            # Mid cap growth
            'CRM', 'ADBE', 'NFLX', 'PYPL', 'SHOP', 'SQ', 'ROKU',
            # ETFs for market context
            'SPY', 'QQQ', 'IWM', 'VIX'
        ]
    
    def _get_default_config(self) -> Dict:
        """Get default backtesting configuration"""
        return {
            'slippage_bps': 2,  # 2 basis points slippage
            'latency_ms': 50,   # 50ms execution latency
            'commission': 0.0,  # Alpaca is commission-free
            'min_confidence': 0.6,  # Minimum ML confidence threshold
            'max_positions': 20,    # Maximum concurrent positions
            'position_size_method': 'kelly',  # Use Kelly criterion
            'risk_management': True,  # Enable risk management
            'market_hours_only': True,  # Trade only during market hours
            'enable_logging': True,   # Enable detailed logging
            'save_results': True      # Save backtest results
        }
    
    async def run_backtest(self, start_date: str, end_date: str, 
                          save_results: bool = True) -> Dict:
        """
        Run complete backtest using production pipeline
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        logger.info(f"Starting production backtest: {start_date} to {end_date}")
        logger.info(f"Symbols: {len(self.symbols)} | Capital: ${self.initial_capital:,}")
        
        self.backtest_start_time = time.time()
        
        try:
            # Reset state
            self._reset_backtest_state()
            
            # Run backtest day by day
            async for daily_data in self.data_adapter.simulate_realtime_feed(
                start_date, end_date, self.symbols
            ):
                await self._process_daily_data(daily_data)
            
            # Finalize results
            self.backtest_end_time = time.time()
            results = await self._finalize_backtest_results()
            
            # Save results if requested
            if save_results:
                await self._save_backtest_results(results, start_date, end_date)
                
                # Generate comprehensive analytics report and charts
                logger.info("Generating comprehensive analytics report with charts...")
                analytics_report = self.analytics.generate_comprehensive_report()
                
                # Save analytics report (overwrites previous report)
                analytics_filename = "latest_analytics_report.json"
                analytics_filepath = os.path.join('backtest_results', analytics_filename)
                with open(analytics_filepath, 'w') as f:
                    json.dump(analytics_report, f, indent=2, default=str)
                logger.info(f"Analytics report saved to: {analytics_filepath}")
                
                # Generate PNG reports (overwrites previous reports)
                png_reports = self.analytics.generate_png_reports("backtest_reports")
                if png_reports:
                    logger.info(f"📊 Generated {len(png_reports)} PNG reports in backtest_reports/")
                    logger.info("🎯 PNG reports updated with latest backtest results!")
                    
                    # Log each report generated
                    for report_name, filepath in png_reports.items():
                        logger.info(f"  ✓ {report_name}: {filepath}")
                else:
                    logger.warning("⚠️ No PNG reports were generated")
                
                # Print analytics summary
                self.analytics.print_summary()
            
            logger.info(f"Backtest completed successfully")
            logger.info(f"Total return: {results['performance']['total_return_pct']:.2f}%")
            logger.info(f"Sharpe ratio: {results['performance']['sharpe_ratio']:.2f}")
            logger.info(f"Max drawdown: {results['performance']['max_drawdown_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _process_daily_data(self, daily_data: Dict):
        """Process one day of historical data through production pipeline"""
        date = daily_data['date']
        market_data = daily_data['data']
        
        self.current_date = date
        logger.info(f"Processing {date} - {len(market_data)} symbols")
        
        daily_start_time = time.time()
        
        try:
            # Get market context (VIX, SPY data)
            vix_level = self.data_adapter.get_vix_data(date)
            spy_data = self.data_adapter.get_spy_data(date)
            
            # Convert market data to format expected by production components
            polygon_data = self._convert_to_polygon_format(market_data, vix_level, spy_data)
            
            logger.info(f"=== PRODUCTION BACKTESTER DEBUG for {date} ===")
            logger.info(f"Raw market data: {len(market_data)} symbols")
            logger.info(f"Converted polygon data: {len(polygon_data)} items")
            
            # Debug: Log sample data being passed to filter
            if polygon_data:
                sample = polygon_data[0]
                logger.info(f"Sample polygon data: {sample.get('symbol')} - price: {sample.get('price')}, volume: {sample.get('volume')}, market_cap: {sample.get('market_cap')}")
            
            # STEP 1: Adaptive Data Filter (production component)
            # Set the current backtest date for proper VIX data lookup
            self.adaptive_filter.current_backtest_date = date
            
            filter_start = time.time()
            filtered_data = await self.adaptive_filter.process_polygon_data(polygon_data)
            filter_time = (time.time() - filter_start) * 1000
            
            # Record filter latency and accuracy
            self.analytics.record_component_latency("adaptive_filter", filter_time)
            filter_accuracy = len(filtered_data) / max(1, len(polygon_data))  # Filter pass rate
            self.analytics.record_system_metric("filter_pass_rate", filter_accuracy)
            
            logger.info(f"Adaptive filter result: {len(polygon_data)} → {len(filtered_data)} stocks ({filter_time:.2f}ms)")
            
            if not filtered_data:
                logger.warning(f"No stocks passed adaptive filter for {date}")
                logger.warning(f"This suggests the filter is not receiving valid data or the filtering criteria are too strict")
                return
            
            # STEP 2: Feature Engineering (production component)
            feature_start = time.time()
            features_batch = await self.feature_engineer.engineer_features_batch(filtered_data)
            feature_time = (time.time() - feature_start) * 1000
            
            # Record feature engineering latency
            self.analytics.record_component_latency("feature_engineering", feature_time)
            
            logger.debug(f"Feature engineering: {len(filtered_data)} stocks ({feature_time:.2f}ms)")
            
            # STEP 3: ML Predictions (production component)
            ml_start = time.time()
            predictions = await self.ml_system.predict_batch(features_batch)
            ml_time = (time.time() - ml_start) * 1000
            
            # Record ML prediction latency and track individual predictions
            self.analytics.record_component_latency("ml_prediction", ml_time)
            
            # Track individual model predictions if available
            for i, (stock_data, prediction) in enumerate(zip(filtered_data, predictions)):
                symbol = stock_data.get('symbol', f'STOCK_{i}') if hasattr(stock_data, 'get') else getattr(stock_data, 'symbol', f'STOCK_{i}')
                pred_value = prediction.get('prediction', 0.0) if hasattr(prediction, 'get') else getattr(prediction, 'prediction', 0.0)
                confidence = prediction.get('confidence', 0.5) if hasattr(prediction, 'get') else getattr(prediction, 'confidence', 0.5)
                
                # Record ensemble prediction
                self.analytics.record_ensemble_prediction(
                    symbol=symbol,
                    individual_predictions={'ensemble': pred_value},
                    individual_confidences={'ensemble': confidence},
                    ensemble_prediction=pred_value,
                    ensemble_confidence=confidence,
                    ensemble_time_ms=ml_time / len(predictions) if predictions else ml_time
                )
            
            logger.debug(f"ML predictions: {len(predictions)} predictions ({ml_time:.2f}ms)")
            
            # STEP 4: Position Sizing (production component)
            kelly_start = time.time()
            position_signals = await self._calculate_position_sizes(filtered_data, predictions)
            kelly_time = (time.time() - kelly_start) * 1000
            
            # Record Kelly sizing latency and decisions
            self.analytics.record_component_latency("kelly_sizing", kelly_time)
            
            # Track position sizing decisions
            for signal in position_signals:
                self.analytics.record_position_sizing(
                    symbol=signal['symbol'],
                    kelly_fraction=signal.get('kelly_result', {}).get('kelly_fraction', 0),
                    position_size=signal['position_size'],
                    confidence=signal['confidence'],
                    sizing_time_ms=kelly_time / len(position_signals) if position_signals else kelly_time
                )
            
            logger.debug(f"Kelly sizing: {len(position_signals)} positions ({kelly_time:.2f}ms)")
            
            # STEP 5: Trade Execution (simulated)
            execution_start = time.time()
            executed_trades = await self._execute_trades(position_signals, vix_level)
            execution_time = (time.time() - execution_start) * 1000
            
            # Record trade execution latency
            self.analytics.record_component_latency("trade_execution", execution_time)
            
            logger.debug(f"Trade execution: {len(executed_trades)} trades ({execution_time:.2f}ms)")
            
            # Calculate end-to-end latency
            total_time_ms = (time.time() - daily_start_time) * 1000
            self.analytics.record_end_to_end_latency(total_time_ms)
            
            # Record daily performance
            daily_performance = await self._record_daily_performance(date, {
                'filter_time_ms': filter_time,
                'feature_time_ms': feature_time,
                'ml_time_ms': ml_time,
                'kelly_time_ms': kelly_time,
                'execution_time_ms': execution_time,
                'total_time_ms': total_time_ms,
                'stocks_filtered': len(filtered_data),
                'predictions_made': len(predictions),
                'trades_executed': len(executed_trades),
                'vix_level': vix_level
            })
            
            # Record daily performance in analytics
            account_status = await self.executor.get_account_status()
            self.analytics.record_daily_performance(
                date=date,
                portfolio_value=account_status['portfolio_value'],
                daily_pnl=account_status['daily_pnl'],
                trades_count=len(executed_trades),
                positions_count=account_status['num_positions'],
                processing_metrics={
                    'filter_time_ms': filter_time,
                    'feature_time_ms': feature_time,
                    'ml_time_ms': ml_time,
                    'kelly_time_ms': kelly_time,
                    'execution_time_ms': execution_time,
                    'total_time_ms': total_time_ms
                }
            )
            
            self.daily_results.append(daily_performance)
            
        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
    
    def _convert_to_polygon_format(self, market_data: Dict, vix_level: float, spy_data: Dict) -> List[Dict]:
        """Convert historical data to format expected by production components"""
        polygon_data = []
        
        for symbol, data in market_data.items():
            # Ensure we have valid price and volume data
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            market_cap = data.get('market_cap', 0)
            
            # Skip symbols with invalid data
            if price <= 0 or volume <= 0:
                logger.debug(f"Skipping {symbol} - invalid price ({price}) or volume ({volume})")
                continue
                
            # Format data exactly like production Polygon WebSocket streams
            formatted_data = {
                'symbol': symbol,
                'aggregates': data.get('aggregates', []),
                'quotes': data.get('quotes', []),
                'timestamp': data.get('timestamp', time.time()),
                'price': price,
                'volume': volume,
                'bid': data.get('bid', price * 0.999),  # Estimate bid if not available
                'ask': data.get('ask', price * 1.001),  # Estimate ask if not available
                'high': data.get('high', price),
                'low': data.get('low', price),
                'open': data.get('open', price),
                'vwap': data.get('vwap', price),
                'market_cap': market_cap,
                'daily_change': data.get('daily_change', 0),
                'volatility': data.get('volatility', 0.02),
                'momentum_score': data.get('momentum_score', 0),
                'data_type': 'historical'
            }
            
            # Add market context
            formatted_data['market_context'] = {
                'vix': vix_level,
                'spy_change': spy_data.get('daily_change', 0),
                'volume_ratio': min(max(volume / 1000000, 0.1), 5.0)  # Normalize volume ratio
            }
            
            polygon_data.append(formatted_data)
        
        logger.debug(f"Converted {len(market_data)} market data items to {len(polygon_data)} polygon format items")
        return polygon_data
    
    async def _calculate_position_sizes(self, filtered_data: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Calculate position sizes using production Kelly sizer"""
        position_signals = []
        
        for i, (stock_data, prediction) in enumerate(zip(filtered_data, predictions)):
            try:
                # Handle both dict and MarketData object types
                if hasattr(stock_data, 'get'):
                    symbol = stock_data.get('symbol', f'STOCK_{i}')
                    current_price = stock_data.get('price', 100.0)
                    market_context = stock_data.get('market_context', {})
                    market_cap = stock_data.get('market_cap', 1000000000)
                else:
                    symbol = getattr(stock_data, 'symbol', f'STOCK_{i}')
                    current_price = getattr(stock_data, 'price', 100.0)
                    market_context = getattr(stock_data, 'market_context', {})
                    market_cap = getattr(stock_data, 'market_cap', 1000000000)
                
                if hasattr(prediction, 'get'):
                    confidence = prediction.get('confidence', 0.5)
                else:
                    confidence = getattr(prediction, 'confidence', 0.5)
                
                # Skip low confidence predictions
                if confidence < self.config.get('min_confidence', 0.6):
                    continue
                
                # Get VIX level from market context
                if hasattr(market_context, 'get'):
                    vix_level = market_context.get('vix', 20.0)
                else:
                    vix_level = getattr(market_context, 'vix', 20.0) if hasattr(market_context, 'vix') else 20.0
                
                # Use production Kelly sizer
                kelly_result = self.kelly_sizer.calculate_aggressive_position_size(
                    symbol=symbol,
                    current_price=current_price,
                    confidence=confidence,
                    vix_level=vix_level,
                    market_cap=market_cap
                )
                
                if kelly_result and kelly_result['total_qty'] > 0:
                    position_signals.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'position_size': kelly_result['total_qty'],
                        'position_value': kelly_result['total_value'],
                        'confidence': confidence,
                        'prediction': prediction.get('prediction', 0.0) if hasattr(prediction, 'get') else getattr(prediction, 'prediction', 0.0),
                        'kelly_result': kelly_result
                    })
                    
            except Exception as e:
                symbol_name = stock_data.get('symbol', 'UNKNOWN') if hasattr(stock_data, 'get') else getattr(stock_data, 'symbol', 'UNKNOWN')
                logger.error(f"Position sizing error for {symbol_name}: {e}")
        
        return position_signals
    
    async def _execute_trades(self, position_signals: List[Dict], vix_level: float) -> List[Dict]:
        """Execute trades using simulated executor"""
        executed_trades = []
        
        for signal_data in position_signals:
            try:
                # Validate signal data
                required_fields = ['symbol', 'current_price', 'confidence', 'prediction', 'position_size']
                missing_fields = [field for field in required_fields if field not in signal_data]
                
                if missing_fields:
                    logger.error(f"Missing required fields for {signal_data.get('symbol', 'UNKNOWN')}: {missing_fields}")
                    continue
                
                # Ensure all values are valid
                symbol = signal_data['symbol']
                current_price = float(signal_data['current_price']) if signal_data['current_price'] is not None else 0.0
                confidence = float(signal_data['confidence']) if signal_data['confidence'] is not None else 0.5
                prediction = float(signal_data['prediction']) if signal_data['prediction'] is not None else 0.0
                position_size = int(signal_data['position_size']) if signal_data['position_size'] is not None else 0
                
                # Skip invalid trades
                if current_price <= 0 or position_size <= 0:
                    logger.warning(f"Skipping invalid trade for {symbol}: price={current_price}, size={position_size}")
                    continue
                
                # Create mock signal object for production compatibility
                signal = MockSignal(
                    symbol=symbol,
                    current_price=current_price,
                    confidence=confidence,
                    prediction=prediction
                )
                
                # Execute using production momentum trade logic
                execution_result = await self.executor.execute_momentum_trade(
                    signal=signal,
                    market_data=None,
                    current_vix=vix_level
                )
                
                if execution_result:
                    # Create trade record with prediction correctness tracking
                    trade_record = {
                        'symbol': symbol,
                        'execution_result': execution_result,
                        'signal_data': signal_data,
                        'timestamp': time.time(),
                        'prediction': prediction,
                        'confidence': confidence,
                        'prediction_correct': None  # Will be updated later when we know the outcome
                    }
                    
                    executed_trades.append(trade_record)
                    self.total_trades += 1
                    
                    # Record trade in analytics system with proper data structure
                    analytics_trade_record = {
                        'symbol': symbol,
                        'timestamp': time.time(),
                        'prediction': prediction,
                        'confidence': confidence,
                        'entry_price': current_price,
                        'position_size': position_size,
                        'prediction_correct': None,  # Will be updated later
                        'pnl': 0.0  # Will be updated later
                    }
                    self.analytics.record_trade(analytics_trade_record)
                    
                else:
                    logger.warning(f"Trade execution failed for {symbol} - no execution result returned")
                    
            except Exception as e:
                symbol_name = signal_data.get('symbol', 'UNKNOWN') if isinstance(signal_data, dict) else 'UNKNOWN'
                logger.error(f"Trade execution error for {symbol_name}: {e}")
                # Continue with other trades even if one fails
                continue
        
        logger.info(f"Successfully executed {len(executed_trades)} out of {len(position_signals)} potential trades")
        return executed_trades
    
    async def _record_daily_performance(self, date: str, metrics: Dict) -> Dict:
        """Record daily performance metrics"""
        account_status = await self.executor.get_account_status()
        
        # CRITICAL: Collect ALL trade data BEFORE any reset operations
        trade_summary = self.executor.get_trade_outcome_summary()
        completed_trades = self.executor.get_completed_trades()
        all_trades = self.executor.get_trades_history()
        
        logger.info(f"Daily performance for {date}: {trade_summary}")
        logger.info(f"Collecting {len(completed_trades)} completed trades before reset")
        
        # Update analytics with completed trades immediately
        for completed_trade in completed_trades:
            try:
                self.analytics.update_trade_outcome(
                    symbol=completed_trade['symbol'],
                    prediction_correct=completed_trade['prediction_correct'],
                    pnl=completed_trade.get('pnl', 0.0),
                    entry_price=completed_trade['entry_price'],
                    exit_price=completed_trade['exit_price'],
                    entry_timestamp=completed_trade['entry_timestamp'],
                    exit_timestamp=completed_trade['exit_timestamp']
                )
            except Exception as e:
                logger.error(f"Error updating trade outcome for {completed_trade.get('symbol', 'UNKNOWN')}: {e}")
        
        # Also update analytics with any trades that have outcomes but weren't in completed_trades
        for trade in all_trades:
            if (trade.get('prediction_correct') is not None and
                trade.get('symbol') and trade.get('pnl') is not None):
                try:
                    # Check if this trade is already in analytics
                    trade_exists = False
                    for analytics_trade in self.analytics.trades:
                        if (analytics_trade.get('symbol') == trade.get('symbol') and
                            abs(analytics_trade.get('timestamp', 0) - trade.get('timestamp', 0)) < 1):
                            trade_exists = True
                            # Update if not already updated
                            if analytics_trade.get('prediction_correct') is None:
                                analytics_trade['prediction_correct'] = trade.get('prediction_correct')
                                analytics_trade['pnl'] = trade.get('pnl', 0.0)
                            break
                    
                    if not trade_exists:
                        # Add missing trade to analytics
                        self.analytics.record_trade({
                            'symbol': trade.get('symbol'),
                            'timestamp': trade.get('timestamp', time.time()),
                            'prediction': trade.get('prediction', 0.0),
                            'confidence': trade.get('confidence', 0.5),
                            'entry_price': trade.get('price', 0.0),
                            'position_size': trade.get('qty', 0),
                            'prediction_correct': trade.get('prediction_correct'),
                            'pnl': trade.get('pnl', 0.0)
                        })
                except Exception as e:
                    logger.error(f"Error processing trade {trade.get('symbol', 'UNKNOWN')}: {e}")
        
        # Clear completed trades from executor after processing
        cleared_count = self.executor.clear_completed_trades()
        logger.info(f"Cleared {cleared_count} completed trades from executor")
        
        daily_performance = {
            'date': date,
            'portfolio_value': account_status['portfolio_value'],
            'cash_available': account_status['cash'],
            'daily_pnl': account_status['daily_pnl'],
            'positions_count': account_status['num_positions'],
            'trades_count': account_status['day_trade_count'],
            'processing_metrics': metrics,
            'account_status': account_status,
            'trade_summary': trade_summary
        }
        
        # Reset daily tracking in executor for next day AFTER all data collection
        self.executor.reset_daily_tracking()
        
        return daily_performance
    
    async def _finalize_backtest_results(self) -> Dict:
        """Finalize and compile backtest results"""
        final_account = await self.executor.get_account_status()
        executor_stats = self.executor.get_performance_stats()
        
        # Update prediction outcomes based on trade results
        trade_history = self._update_prediction_outcomes()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Compile comprehensive results
        results = {
            'backtest_info': {
                'start_time': self.backtest_start_time,
                'end_time': self.backtest_end_time,
                'duration_seconds': self.backtest_end_time - self.backtest_start_time,
                'symbols_count': len(self.symbols),
                'symbols': self.symbols,
                'initial_capital': self.initial_capital,
                'config': self.config
            },
            'final_portfolio': {
                'portfolio_value': final_account['portfolio_value'],
                'cash_available': final_account['cash'],
                'positions': self.executor.get_positions(),
                'total_return': final_account['portfolio_value'] - self.initial_capital,
                'total_return_pct': ((final_account['portfolio_value'] / self.initial_capital) - 1) * 100
            },
            'performance': performance_metrics,
            'trading_stats': executor_stats,
            'daily_results': self.daily_results,
            'component_performance': {
                'total_trades': self.total_trades,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
                'win_rate_pct': (self.successful_predictions / max(1, self.total_trades)) * 100
            }
        }
        
        return results
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.daily_results:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'annualized_return': 0.0,
                'annualized_return_pct': 0.0,
                'volatility': 0.0,
                'volatility_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate_pct': 0.0,
                'total_trading_days': 0,
                'avg_daily_return_pct': 0.0,
                'best_day_pct': 0.0,
                'worst_day_pct': 0.0
            }
        
        # Extract daily portfolio values
        portfolio_values = [day['portfolio_value'] for day in self.daily_results]
        daily_returns = []
        
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'annualized_return': 0.0,
                'annualized_return_pct': 0.0,
                'volatility': 0.0,
                'volatility_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate_pct': 0.0,
                'total_trading_days': 0,
                'avg_daily_return_pct': 0.0,
                'best_day_pct': 0.0,
                'worst_day_pct': 0.0
            }
        
        daily_returns = np.array(daily_returns)
        
        # Calculate key metrics
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free rate
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate win rate
        winning_days = sum(1 for ret in daily_returns if ret > 0)
        win_rate = (winning_days / len(daily_returns)) * 100
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate,
            'total_trading_days': len(daily_returns),
            'avg_daily_return_pct': np.mean(daily_returns) * 100,
            'best_day_pct': np.max(daily_returns) * 100,
            'worst_day_pct': np.min(daily_returns) * 100
        }
    
    async def _save_backtest_results(self, results: Dict, start_date: str, end_date: str):
        """Save backtest results to file"""
        try:
            # Use consistent filename that overwrites previous results
            filename = "latest_backtest_results.json"
            
            # Create results directory if it doesn't exist
            os.makedirs('backtest_results', exist_ok=True)
            filepath = os.path.join('backtest_results', filename)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert numpy types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            serializable_results = recursive_convert(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
    
    def _update_prediction_outcomes(self):
        """Update prediction outcomes based on trade results"""
        # Get all trades from executor
        trade_history = self.executor.get_trades_history()
        completed_trades = self.executor.get_completed_trades()
        
        logger.info(f"Processing trade outcomes: {len(trade_history)} total trades, {len(completed_trades)} completed trades")
        
        # Update analytics with completed trade data
        for completed_trade in completed_trades:
            try:
                self.analytics.update_trade_outcome(
                    symbol=completed_trade['symbol'],
                    prediction_correct=completed_trade['prediction_correct'],
                    pnl=completed_trade.get('pnl', 0.0),
                    entry_price=completed_trade['entry_price'],
                    exit_price=completed_trade['exit_price'],
                    entry_timestamp=completed_trade['entry_timestamp'],
                    exit_timestamp=completed_trade['exit_timestamp']
                )
            except Exception as e:
                logger.error(f"Error updating completed trade outcome for {completed_trade.get('symbol', 'UNKNOWN')}: {e}")
        
        # Process all trades from trade history to ensure no trades are missed
        trades_with_outcomes = 0
        trades_without_outcomes = 0
        
        for trade in trade_history:
            if trade.get('prediction_correct') is not None:
                trades_with_outcomes += 1
                # Find corresponding trade in analytics and update it
                trade_updated = False
                for analytics_trade in self.analytics.trades:
                    if (analytics_trade.get('symbol') == trade.get('symbol') and
                        analytics_trade.get('prediction_correct') is None and
                        abs(analytics_trade.get('timestamp', 0) - trade.get('timestamp', 0)) < 2):
                        analytics_trade['prediction_correct'] = trade.get('prediction_correct')
                        analytics_trade['pnl'] = trade.get('pnl', 0.0)
                        trade_updated = True
                        break
                
                if not trade_updated:
                    # Add missing trade to analytics if it has an outcome
                    try:
                        self.analytics.record_trade({
                            'symbol': trade.get('symbol'),
                            'timestamp': trade.get('timestamp', time.time()),
                            'prediction': trade.get('prediction', 0.0),
                            'confidence': trade.get('confidence', 0.5),
                            'entry_price': trade.get('price', 0.0),
                            'position_size': trade.get('qty', 0),
                            'prediction_correct': trade.get('prediction_correct'),
                            'pnl': trade.get('pnl', 0.0)
                        })
                    except Exception as e:
                        logger.error(f"Error adding missing trade to analytics: {e}")
            else:
                trades_without_outcomes += 1
        
        # Count analytics trades without outcomes and force-classify them
        analytics_trades_without_outcomes = 0
        for analytics_trade in self.analytics.trades:
            if analytics_trade.get('prediction_correct') is None:
                analytics_trades_without_outcomes += 1
                # Force classify based on P&L if available, otherwise as loss
                pnl = analytics_trade.get('pnl', 0.0)
                if pnl != 0:
                    analytics_trade['prediction_correct'] = pnl > 0
                    logger.warning(f"Force-classified analytics trade {analytics_trade.get('symbol', 'UNKNOWN')} based on P&L: {pnl}")
                else:
                    analytics_trade['prediction_correct'] = False
                    analytics_trade['pnl'] = 0.0
                    logger.warning(f"Force-classified analytics trade {analytics_trade.get('symbol', 'UNKNOWN')} as loss (no P&L data)")
        
        # Update successful/failed prediction counts from executor
        self.successful_predictions = self.executor.win_count
        self.failed_predictions = self.executor.loss_count
        
        logger.info(f"Trade outcome summary:")
        logger.info(f"  - Executor trades with outcomes: {trades_with_outcomes}")
        logger.info(f"  - Executor trades without outcomes: {trades_without_outcomes}")
        logger.info(f"  - Analytics trades without outcomes (force-classified): {analytics_trades_without_outcomes}")
        logger.info(f"  - Final counts: {self.successful_predictions} successful, {self.failed_predictions} failed")
        logger.info(f"  - Win rate: {(self.successful_predictions / max(1, self.total_trades)) * 100:.2f}%")
        
        return trade_history
    
    def _reset_backtest_state(self):
        """Reset backtest state for new run"""
        self.backtest_results = {}
        self.daily_results = []
        self.current_date = None
        self.total_trades = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.processing_times = []
        
        # Reset executor
        self.executor.reset_daily_tracking()
        
        logger.info("Backtest state reset")
    
    def _validate_system_integrity(self) -> bool:
        """Validate system integrity and data consistency"""
        try:
            # Check executor state
            executor_trades = self.executor.get_trades_history()
            completed_trades = self.executor.get_completed_trades()
            analytics_trades = self.analytics.trades
            
            logger.info(f"System integrity check:")
            logger.info(f"  - Executor trades: {len(executor_trades)}")
            logger.info(f"  - Completed trades: {len(completed_trades)}")
            logger.info(f"  - Analytics trades: {len(analytics_trades)}")
            
            # Check for trades without outcomes
            trades_without_outcomes = 0
            for trade in executor_trades:
                if trade.get('prediction_correct') is None:
                    trades_without_outcomes += 1
            
            analytics_without_outcomes = 0
            for trade in analytics_trades:
                if trade.get('prediction_correct') is None:
                    analytics_without_outcomes += 1
            
            logger.info(f"  - Executor trades without outcomes: {trades_without_outcomes}")
            logger.info(f"  - Analytics trades without outcomes: {analytics_without_outcomes}")
            
            # Check for None values in critical fields
            none_predictions = sum(1 for trade in executor_trades if trade.get('prediction') is None)
            none_confidences = sum(1 for trade in executor_trades if trade.get('confidence') is None)
            
            logger.info(f"  - Trades with None predictions: {none_predictions}")
            logger.info(f"  - Trades with None confidences: {none_confidences}")
            
            # System is healthy if we have minimal issues
            is_healthy = (trades_without_outcomes < len(executor_trades) * 0.1 and  # Less than 10% without outcomes
                         analytics_without_outcomes < len(analytics_trades) * 0.1 and
                         none_predictions == 0 and none_confidences == 0)
            
            logger.info(f"  - System integrity: {'HEALTHY' if is_healthy else 'ISSUES DETECTED'}")
            return is_healthy
            
        except Exception as e:
            logger.error(f"Error during system integrity check: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    async def test_production_backtester():
        print("Testing Production Backtester...")
        
        # Get API key from environment
        api_key = os.getenv('POLYGON_API_KEY', 'Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57')
        if not api_key:
            print("Please set POLYGON_API_KEY environment variable")
            return
        
        # Initialize backtester
        backtester = ProductionBacktester(
            polygon_api_key=api_key,
            initial_capital=50000,
            symbols=['AAPL', 'TSLA', 'NVDA']  # Small test set
        )
        
        # Run short backtest (avoid MLK Day holiday on 2024-01-15)
        results = await backtester.run_backtest(
            start_date='2024-01-16',
            end_date='2024-01-22',  # One week test
            save_results=True
        )
        
        print(f"Backtest completed:")
        print(f"Total return: {results['performance']['total_return_pct']:.2f}%")
        print(f"Sharpe ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"Total trades: {results['trading_stats']['total_trades']}")
        
        print("Production Backtester test completed")
    
    # Run test
    asyncio.run(test_production_backtester())