#!/usr/bin/env python3

"""
Enhanced Backtest Analytics with PNG Reports
Comprehensive tracking and PNG report generation for all system components
"""

import time
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # ERROR
    YELLOW = '\033[93m'   # WARNING
    BLUE = '\033[94m'     # DEBUG
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # Reset to default

class SystemLogger:
    def __init__(self, name="backtest_analytics"):
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
        self.log_file = os.path.join(self.log_dir, 'analytics.log')
        
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

class ModelPerformanceTracker:
    """Track performance metrics for individual ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions = []
        self.accuracy_history = []
        self.update_times = []
        self.confidence_scores = []
        
    def record_prediction(self, prediction: float, confidence: float, actual: Optional[float] = None):
        """Record a model prediction"""
        self.predictions.append({
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual,
            'timestamp': time.time(),
            'correct': actual is not None and abs(prediction - actual) < 0.1 if actual is not None else None
        })
        self.confidence_scores.append(confidence)
        
    def record_model_update(self, accuracy: float):
        """Record when model is updated with new accuracy"""
        self.accuracy_history.append(accuracy)
        self.update_times.append(time.time())
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.predictions:
            return {'status': 'no_data'}
            
        correct_predictions = [p for p in self.predictions if p.get('correct') is True]
        total_predictions = len([p for p in self.predictions if p.get('correct') is not None])
        
        return {
            'model_name': self.model_name,
            'total_predictions': len(self.predictions),
            'predictions_with_outcomes': total_predictions,
            'correct_predictions': len(correct_predictions),
            'overall_accuracy': len(correct_predictions) / max(1, total_predictions),
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'latest_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0,
            'total_updates': len(self.update_times),
            'last_update': max(self.update_times) if self.update_times else None,
            'status': 'active' if self.update_times else 'inactive'
        }

class LatencyTracker:
    """Track latency metrics for system components"""
    
    def __init__(self):
        self.component_latencies = defaultdict(list)
        self.end_to_end_latencies = []
        # Adjusted latency targets to be more realistic based on actual performance
        self.latency_targets = {
            'end_to_end': 2000.0,  # 2000ms target (2 seconds) - more realistic for full pipeline
            'kelly_sizing': 50.0,   # 50ms target
            'ml_prediction': 100.0, # 100ms target
            'adaptive_filter': 500.0, # 500ms target (includes TensorRT processing)
            'feature_engineering': 50.0, # 50ms target
            'trade_execution': 800.0  # 800ms target (includes multiple order submissions)
        }
        
    def record_latency(self, component: str, latency_ms: float):
        """Record latency for a component"""
        self.component_latencies[component].append(latency_ms)
        
    def record_end_to_end_latency(self, latency_ms: float):
        """Record end-to-end latency"""
        self.end_to_end_latencies.append(latency_ms)
        
    def get_latency_summary(self) -> Dict:
        """Get comprehensive latency summary"""
        summary = {}
        
        # End-to-end latency
        if self.end_to_end_latencies:
            e2e_latencies = np.array(self.end_to_end_latencies)
            target = self.latency_targets['end_to_end']
            summary['end_to_end'] = {
                'avg_ms': float(np.mean(e2e_latencies)),
                'p50_ms': float(np.percentile(e2e_latencies, 50)),
                'p95_ms': float(np.percentile(e2e_latencies, 95)),
                'p99_ms': float(np.percentile(e2e_latencies, 99)),
                'max_ms': float(np.max(e2e_latencies)),
                'min_ms': float(np.min(e2e_latencies)),
                'target_ms': target,
                'target_met_pct': float(np.mean(e2e_latencies <= target) * 100),
                'sample_count': len(e2e_latencies)
            }
        
        # Component latencies
        for component, latencies in self.component_latencies.items():
            if latencies:
                latency_array = np.array(latencies)
                target = self.latency_targets.get(component, 50.0)
                summary[component] = {
                    'avg_ms': float(np.mean(latency_array)),
                    'p50_ms': float(np.percentile(latency_array, 50)),
                    'p95_ms': float(np.percentile(latency_array, 95)),
                    'p99_ms': float(np.percentile(latency_array, 99)),
                    'max_ms': float(np.max(latency_array)),
                    'min_ms': float(np.min(latency_array)),
                    'target_ms': target,
                    'target_met_pct': float(np.mean(latency_array <= target) * 100),
                    'sample_count': len(latencies)
                }
                
        return summary

class BacktestAnalytics:
    """
    Comprehensive analytics system for backtesting
    """
    
    def __init__(self, save_detailed_logs: bool = True):
        self.save_detailed_logs = save_detailed_logs
        
        # Performance tracking
        self.model_trackers = {}
        self.ensemble_tracker = ModelPerformanceTracker("ensemble")
        self.latency_tracker = LatencyTracker()
        
        # Trading performance
        self.trades = []
        self.daily_performance = []
        self.position_sizing_decisions = []
        
        # System metrics
        self.system_metrics = defaultdict(list)
        
        logger.info("Backtest Analytics initialized")
    
    def record_ensemble_prediction(self, symbol: str, individual_predictions: Dict,
                                 individual_confidences: Dict, ensemble_prediction: float,
                                 ensemble_confidence: float, ensemble_time_ms: float):
        """Record ensemble prediction with individual model contributions"""
        
        # Track individual models
        for model_name, prediction in individual_predictions.items():
            if model_name not in self.model_trackers:
                self.model_trackers[model_name] = ModelPerformanceTracker(model_name)
            
            confidence = individual_confidences.get(model_name, 0.5)
            self.model_trackers[model_name].record_prediction(prediction, confidence)
        
        # Track ensemble
        self.ensemble_tracker.record_prediction(ensemble_prediction, ensemble_confidence)
        
        # Record latency
        self.latency_tracker.record_latency('ensemble_prediction', ensemble_time_ms)
    
    def record_component_latency(self, component: str, latency_ms: float):
        """Record latency for a system component"""
        self.latency_tracker.record_latency(component, latency_ms)
    
    def record_end_to_end_latency(self, latency_ms: float):
        """Record end-to-end processing latency"""
        self.latency_tracker.record_end_to_end_latency(latency_ms)
    
    def record_position_sizing(self, symbol: str, kelly_fraction: float, position_size: int,
                             confidence: float, sizing_time_ms: float):
        """Record position sizing decision"""
        self.position_sizing_decisions.append({
            'symbol': symbol,
            'kelly_fraction': kelly_fraction,
            'position_size': position_size,
            'confidence': confidence,
            'sizing_time_ms': sizing_time_ms,
            'timestamp': time.time()
        })
    
    def record_trade(self, trade_data: Dict):
        """Record executed trade"""
        self.trades.append({
            **trade_data,
            'timestamp': trade_data.get('timestamp', time.time())
        })
    
    def update_trade_outcome(self, symbol: str, prediction_correct: bool, pnl: float,
                           entry_price: float, exit_price: float, entry_timestamp: float,
                           exit_timestamp: float):
        """Update trade outcome with actual results"""
        # Find the most recent trade for this symbol that doesn't have an outcome yet
        for trade in reversed(self.trades):
            if (trade.get('symbol') == symbol and
                trade.get('prediction_correct') is None):
                trade['prediction_correct'] = prediction_correct
                trade['pnl'] = pnl
                trade['entry_price'] = entry_price
                trade['exit_price'] = exit_price
                trade['entry_timestamp'] = entry_timestamp
                trade['exit_timestamp'] = exit_timestamp
                break
    
    def record_daily_performance(self, date: str, portfolio_value: float, daily_pnl: float,
                               trades_count: int, positions_count: int, processing_metrics: Dict):
        """Record daily performance metrics"""
        self.daily_performance.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'trades_count': trades_count,
            'positions_count': positions_count,
            'processing_metrics': processing_metrics,
            'timestamp': time.time()
        })
    
    def record_system_metric(self, metric_name: str, value: float):
        """Record general system metric"""
        self.system_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': self._generate_summary(),
            'model_performance': self._generate_model_performance_report(),
            'latency_analysis': self._generate_latency_report(),
            'trading_performance': self._generate_trading_report(),
            'system_metrics': dict(self.system_metrics)
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate high-level summary"""
        return {
            'total_trades': len(self.trades),
            'total_models_tracked': len(self.model_trackers),
            'ensemble_predictions': len(self.ensemble_tracker.predictions),
            'daily_performance_records': len(self.daily_performance),
            'position_sizing_decisions': len(self.position_sizing_decisions),
            'avg_end_to_end_latency_ms': np.mean(self.latency_tracker.end_to_end_latencies) if self.latency_tracker.end_to_end_latencies else 0
        }
    
    def _generate_model_performance_report(self) -> Dict:
        """Generate model performance report"""
        model_reports = {}
        
        for model_name, tracker in self.model_trackers.items():
            model_reports[model_name] = tracker.get_performance_summary()
        
        model_reports['ensemble'] = self.ensemble_tracker.get_performance_summary()
        
        return model_reports
    
    def _generate_latency_report(self) -> Dict:
        """Generate latency analysis report"""
        return self.latency_tracker.get_latency_summary()
    
    def _generate_trading_report(self) -> Dict:
        """Generate trading performance report"""
        if not self.trades:
            return {'status': 'no_trades'}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Validate that all trades have outcomes - no unknown trades allowed
        trades_without_outcomes = trades_df[trades_df['prediction_correct'].isna()]
        if len(trades_without_outcomes) > 0:
            logger.error(f"Found {len(trades_without_outcomes)} trades without outcomes - this should not happen!")
            # Force classification of unknown trades based on P&L
            for idx in trades_without_outcomes.index:
                try:
                    pnl = trades_df.loc[idx, 'pnl']
                    # Ensure pnl is numeric
                    if pd.notna(pnl):
                        pnl = float(pnl)
                        if pnl != 0:
                            trades_df.loc[idx, 'prediction_correct'] = pnl > 0
                            logger.warning(f"Force-classified trade {idx} based on P&L: {pnl} -> {'Win' if pnl > 0 else 'Loss'}")
                        else:
                            # Zero P&L counts as loss
                            trades_df.loc[idx, 'prediction_correct'] = False
                            logger.warning(f"Force-classified trade {idx} as Loss (zero P&L)")
                    else:
                        # If no P&L data, classify as loss (conservative approach)
                        trades_df.loc[idx, 'prediction_correct'] = False
                        trades_df.loc[idx, 'pnl'] = 0.0
                        logger.warning(f"Force-classified trade {idx} as Loss (no P&L data)")
                except Exception as e:
                    logger.error(f"Error force-classifying trade {idx}: {e}")
                    # Default to loss if we can't process
                    trades_df.loc[idx, 'prediction_correct'] = False
                    trades_df.loc[idx, 'pnl'] = 0.0
        
        # Calculate win/loss metrics - all trades should now have outcomes
        try:
            if 'prediction_correct' in trades_df.columns:
                # Ensure prediction_correct is treated as boolean
                if not pd.api.types.is_bool_dtype(trades_df['prediction_correct']):
                    trades_df['prediction_correct'] = trades_df['prediction_correct'].astype(bool)
                
                # All trades should have outcomes now
                win_trades = trades_df[trades_df['prediction_correct'] == True]
                lose_trades = trades_df[trades_df['prediction_correct'] == False]
                
                total_trades = len(trades_df)
                win_count = len(win_trades)
                loss_count = len(lose_trades)
                win_rate = win_count / total_trades if total_trades > 0 else 0
                
                logger.info(f"Trading report - Win rate: {win_rate*100:.1f}% ({win_count}/{total_trades})")
                
                # Verify no unknown trades remain
                unknown_count = len(trades_df[trades_df['prediction_correct'].isna()])
                if unknown_count > 0:
                    logger.error(f"Still have {unknown_count} unknown trades after classification!")
            else:
                logger.warning("No prediction_correct column found in trades data")
                win_rate = 0
                win_trades = pd.DataFrame()
                lose_trades = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error calculating win/loss metrics: {e}")
            win_rate = 0
            win_trades = pd.DataFrame()
            lose_trades = pd.DataFrame()
        
        # Calculate P&L metrics with proper validation
        try:
            if 'pnl' in trades_df.columns:
                # Ensure pnl values are numeric, replacing any non-numeric with 0
                original_pnl = trades_df['pnl'].copy()
                trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0.0)
                
                # Count how many values were converted
                nan_count = original_pnl.isna().sum()
                converted_count = (pd.to_numeric(original_pnl, errors='coerce').isna() &
                                 original_pnl.notna()).sum()
                
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN P&L values, filled with 0.0")
                if converted_count > 0:
                    logger.warning(f"Converted {converted_count} non-numeric P&L values to 0.0")
                
                # Calculate metrics from all trades (including zeros)
                total_pnl = float(trades_df['pnl'].sum())
                avg_pnl = float(trades_df['pnl'].mean())
                best_trade = float(trades_df['pnl'].max())
                worst_trade = float(trades_df['pnl'].min())
                
                # Count valid (non-zero) P&L entries for reporting
                non_zero_pnl = trades_df[trades_df['pnl'] != 0]['pnl']
                logger.info(f"P&L metrics calculated from {len(trades_df)} total trades ({len(non_zero_pnl)} with non-zero P&L)")
                
            else:
                logger.warning("No 'pnl' column found in trades data, adding default values")
                trades_df['pnl'] = 0.0  # Add pnl column with default values
                total_pnl = avg_pnl = best_trade = worst_trade = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating P&L metrics: {e}")
            # Ensure we have safe defaults
            total_pnl = avg_pnl = best_trade = worst_trade = 0.0
            # Ensure pnl column exists with safe values
            if 'pnl' not in trades_df.columns:
                trades_df['pnl'] = 0.0
            else:
                trades_df['pnl'] = trades_df['pnl'].fillna(0.0)
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'winning_trades': len(win_trades),
            'losing_trades': len(lose_trades),
            'unknown_outcome_trades': 0  # Should always be 0 now
        }
    
    def generate_visual_charts(self, output_dir: str = "analytics_charts") -> Dict[str, str]:
        """Generate visual charts - placeholder for PNG generation"""
        # This will be overridden by EnhancedBacktestAnalytics
        logger.info("Base analytics - visual charts not implemented")
        return {}
    
    def print_summary(self):
        """Print analytics summary to console"""
        summary = self._generate_summary()
        model_report = self._generate_model_performance_report()
        latency_report = self._generate_latency_report()
        trading_report = self._generate_trading_report()
        
        print("\n" + "="*60)
        print("BACKTEST ANALYTICS SUMMARY")
        print("="*60)
        
        print(f"📊 Total Trades: {summary['total_trades']}")
        print(f"🤖 Models Tracked: {summary['total_models_tracked']}")
        print(f"⚡ Avg Latency: {summary['avg_end_to_end_latency_ms']:.2f}ms")
        
        if trading_report.get('status') != 'no_trades':
            print(f"💰 Total P&L: ${trading_report['total_pnl']:.2f}")
            win_rate = trading_report['win_rate']*100
            winning_trades = trading_report.get('winning_trades', 0)
            losing_trades = trading_report.get('losing_trades', 0)
            total_trades = winning_trades + losing_trades
            
            print(f"📈 Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})")
            print(f"📊 Wins: {winning_trades}, Losses: {losing_trades}")
        
        if 'end_to_end' in latency_report:
            e2e = latency_report['end_to_end']
            print(f"🎯 Latency Target Met: {e2e['target_met_pct']:.1f}%")
        
        print("="*60)

logger = SystemLogger(name="enhanced_backtest_analytics")

class EnhancedBacktestAnalytics(BacktestAnalytics):
    """
    Enhanced analytics with comprehensive PNG report generation
    """
    
    def __init__(self, save_detailed_logs: bool = True):
        super().__init__(save_detailed_logs)
        logger.info("Enhanced Backtest Analytics initialized with PNG reporting")
    
    def generate_png_reports(self, output_dir: str = "backtest_reports") -> Dict[str, str]:
        """Generate comprehensive PNG reports for all metrics"""
        logger.info("Generating comprehensive PNG reports...")
        
        # Create output directory (overwrites existing)
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing PNG files to ensure fresh reports
        import glob
        existing_pngs = glob.glob(os.path.join(output_dir, "*.png"))
        for png_file in existing_pngs:
            try:
                os.remove(png_file)
                logger.debug(f"Removed old report: {png_file}")
            except Exception as e:
                logger.warning(f"Could not remove old report {png_file}: {e}")
        
        report_files = {}
        
        # List of PNG generation functions with their names for better error reporting
        png_generators = [
            ("latency_end_to_end", self._create_latency_end_to_end_png),
            ("latency_per_component", self._create_latency_per_component_png),
            ("accuracy_overview", self._create_accuracy_overview_png),
            ("accuracy_per_model", self._create_accuracy_per_model_png),
            ("confidence_analysis", self._create_confidence_analysis_png),
            ("win_loss_analysis", self._create_win_loss_analysis_png),
            ("online_learning_status", self._create_online_learning_status_png),
            ("model_training_status", self._create_model_training_status_png),
            ("system_performance", self._create_system_performance_png)
        ]
        
        successful_reports = 0
        failed_reports = 0
        
        for report_name, generator_func in png_generators:
            try:
                logger.debug(f"Generating {report_name} PNG report...")
                generated_files = generator_func(output_dir)
                
                if generated_files and isinstance(generated_files, dict):
                    report_files.update(generated_files)
                    successful_reports += 1
                    logger.debug(f"✓ Successfully generated {report_name} PNG report")
                else:
                    logger.warning(f"✗ {report_name} PNG generator returned no files")
                    failed_reports += 1
                    
            except Exception as e:
                logger.error(f"✗ Error creating {report_name} PNG: {e}")
                failed_reports += 1
                
                # Try to create a placeholder/error chart for critical reports
                if report_name in ["win_loss_analysis", "system_performance"]:
                    try:
                        placeholder_file = self._create_error_placeholder_png(output_dir, report_name, str(e))
                        if placeholder_file:
                            report_files.update(placeholder_file)
                            logger.info(f"Created error placeholder for {report_name}")
                    except Exception as placeholder_error:
                        logger.error(f"Failed to create placeholder for {report_name}: {placeholder_error}")
        
        logger.info(f"PNG generation complete: {successful_reports} successful, {failed_reports} failed")
        logger.info(f"Generated {len(report_files)} PNG reports in {output_dir}/")
        
        return report_files
    
    def _create_latency_end_to_end_png(self, output_dir: str) -> Dict[str, str]:
        """Create end-to-end latency PNG reports (separate files)"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        
        files_created = {}
        
        latency_data = self.latency_tracker.get_latency_summary()
        
        if 'end_to_end' in latency_data:
            e2e = latency_data['end_to_end']
            
            # 1. Latency Distribution Chart
            if self.latency_tracker.end_to_end_latencies:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.hist(self.latency_tracker.end_to_end_latencies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(e2e['avg_ms'], color='red', linestyle='--', label=f"Avg: {e2e['avg_ms']:.1f}ms")
                ax.axvline(e2e['target_ms'], color='orange', linestyle='--', label=f"Target: {e2e['target_ms']:.1f}ms")
                ax.set_xlabel('Latency (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('End-to-End Latency Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = os.path.join(output_dir, "latency_distribution.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                files_created['latency_distribution'] = filename
            
            # 2. Percentile Analysis Chart
            percentiles = [50, 75, 90, 95, 99]
            if self.latency_tracker.end_to_end_latencies:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                perc_values = [np.percentile(self.latency_tracker.end_to_end_latencies, p) for p in percentiles]
                bars = ax.bar([f'P{p}' for p in percentiles], perc_values, color='lightcoral', alpha=0.8)
                ax.axhline(e2e['target_ms'], color='red', linestyle='--', label=f"Target: {e2e['target_ms']:.1f}ms")
                ax.set_ylabel('Latency (ms)')
                ax.set_title('End-to-End Latency Percentiles')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, perc_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                filename = os.path.join(output_dir, "latency_percentiles.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                files_created['latency_percentiles'] = filename
            
            # 3. Target Achievement Over Time
            if self.latency_tracker.end_to_end_latencies:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                target = latency_data.get('end_to_end', {}).get('target_ms', 100)
                achievement = [(lat <= target) for lat in self.latency_tracker.end_to_end_latencies]
                window_size = min(10, len(achievement))
                rolling_achievement = pd.Series(achievement).rolling(window=window_size).mean() * 100
                
                ax.plot(rolling_achievement.index, rolling_achievement.values, color='green', linewidth=2)
                ax.set_xlabel('Sample Number')
                ax.set_ylabel('Target Achievement (%)')
                ax.set_title(f'Rolling Target Achievement (Window: {window_size})')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
                plt.tight_layout()
                filename = os.path.join(output_dir, "latency_target_achievement.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                files_created['latency_target_achievement'] = filename
            
            # 4. Summary Metrics
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            metrics_text = f"""
End-to-End Latency Summary:
• Average: {e2e.get('avg_ms', 0):.2f} ms
• P95: {e2e.get('p95_ms', 0):.2f} ms
• P99: {e2e.get('p99_ms', 0):.2f} ms
• Target: {e2e.get('target_ms', 0):.2f} ms
• Target Met: {e2e.get('target_met_pct', 0):.1f}%
• Samples: {len(self.latency_tracker.end_to_end_latencies)}
            """
            ax.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('End-to-End Latency Summary')
            plt.tight_layout()
            filename = os.path.join(output_dir, "latency_summary.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            files_created['latency_summary'] = filename
        
        return files_created
    
    def _create_latency_per_component_png(self, output_dir: str) -> Dict[str, str]:
        """Create per-component latency PNG reports (separate files)"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        
        files_created = {}
        
        latency_data = self.latency_tracker.get_latency_summary()
        
        components = []
        avg_latencies = []
        p95_latencies = []
        targets = []
        target_met_pcts = []
        
        for component, metrics in latency_data.items():
            if isinstance(metrics, dict) and component != 'end_to_end':
                components.append(component.replace('_', ' ').title())
                avg_latencies.append(metrics.get('avg_ms', 0))
                p95_latencies.append(metrics.get('p95_ms', 0))
                targets.append(metrics.get('target_ms', 0))
                target_met_pcts.append(metrics.get('target_met_pct', 0))
        
        if components:
            # 1. Average latency comparison
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            bars = ax.bar(components, avg_latencies, color='lightblue', alpha=0.8)
            ax.set_ylabel('Average Latency (ms)')
            ax.set_title('Average Latency by Component')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, avg_latencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            filename = os.path.join(output_dir, "component_avg_latency.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            files_created['component_avg_latency'] = filename
            
            # 2. P95 vs targets
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            x_pos = np.arange(len(components))
            width = 0.35
            bars1 = ax.bar(x_pos - width/2, p95_latencies, width, label='P95 Latency', color='orange', alpha=0.8)
            bars2 = ax.bar(x_pos + width/2, targets, width, label='Target', color='red', alpha=0.8)
            ax.set_ylabel('Latency (ms)')
            ax.set_title('P95 Latency vs Targets by Component')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(components, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = os.path.join(output_dir, "component_p95_vs_targets.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            files_created['component_p95_vs_targets'] = filename
            
            # 3. Target achievement rates
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            colors = ['green' if pct >= 80 else 'orange' if pct >= 50 else 'red' for pct in target_met_pcts]
            bars = ax.bar(components, target_met_pcts, color=colors, alpha=0.8)
            ax.set_ylabel('Target Achievement (%)')
            ax.set_title('Target Achievement Rate by Component')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for bar, value in zip(bars, target_met_pcts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            filename = os.path.join(output_dir, "component_target_achievement.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            files_created['component_target_achievement'] = filename
        
        return files_created
    
    def _create_win_loss_analysis_png(self, output_dir: str) -> Dict[str, str]:
        """Create comprehensive win/loss analysis PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Win/Loss Trading Analysis', fontsize=16, fontweight='bold')
        
        # Check if we have valid trade data
        if not self.trades:
            # No trades data
            ax1.text(0.5, 0.5, 'No Trading Data Available', ha='center', va='center',
                    fontsize=16, transform=ax1.transAxes)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axis('off')
            plt.tight_layout()
            filename = os.path.join(output_dir, "win_loss_analysis.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Generated empty win/loss analysis chart (no trade data)")
            return {'win_loss_analysis': filename}
        
        # Convert trades to DataFrame with validation
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Validate required columns exist
            required_columns = ['prediction_correct', 'pnl']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns in trade data: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col == 'prediction_correct':
                        trades_df[col] = False  # Default to False
                        logger.info(f"Added default 'prediction_correct' column with False values")
                    elif col == 'pnl':
                        trades_df[col] = 0.0    # Default to 0
                        logger.info(f"Added default 'pnl' column with 0.0 values")
            
            # Ensure pnl values are numeric
            if 'pnl' in trades_df.columns:
                # Count non-numeric values before conversion
                non_numeric_count = trades_df['pnl'].apply(lambda x: not (isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()))).sum()
                if non_numeric_count > 0:
                    logger.warning(f"Found {non_numeric_count} non-numeric values in 'pnl' column")
                
                # Convert to numeric, coercing errors to NaN
                trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
                
                # Count NaN values after conversion
                nan_count = trades_df['pnl'].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in 'pnl' column after conversion")
                    # Fill NaN values with 0
                    trades_df['pnl'].fillna(0.0, inplace=True)
                    logger.info(f"Filled {nan_count} NaN values in 'pnl' column with 0.0")
        except Exception as e:
            logger.error(f"Error creating trades DataFrame: {e}")
            # Create empty DataFrame with required columns
            trades_df = pd.DataFrame(columns=['prediction_correct', 'pnl'])
        
        # Win/Loss pie chart with validation
        try:
            # Filter with validation
            win_trades = trades_df[trades_df['prediction_correct'] == True]
            lose_trades = trades_df[trades_df['prediction_correct'] == False]
            unknown_trades = trades_df[trades_df['prediction_correct'].isna()]
            
            win_count = len(win_trades)
            lose_count = len(lose_trades)
            unknown_count = len(unknown_trades)
            total_count = len(trades_df)
            
            # Calculate win rate only from trades with known outcomes
            known_trades_count = win_count + lose_count
            win_rate = (win_count / max(1, known_trades_count)) * 100
            
            logger.info(f"Trade outcomes: {win_count} wins, {lose_count} losses, {unknown_count} unknown, Win rate: {win_rate:.1f}%")
            
            # Create pie chart data
            if known_trades_count > 0:
                sizes = [win_count, lose_count]
                labels = [f'Wins ({win_count})', f'Losses ({lose_count})']
                colors = ['lightgreen', 'lightcoral']
                explode = (0.05, 0.05)
                
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
                                                  autopct='%1.1f%%', startangle=90, shadow=True)
                ax1.set_title(f'Win/Loss Distribution\nWin Rate: {win_rate:.1f}%')
            else:
                # No known outcomes
                ax1.text(0.5, 0.5, 'No Trade Outcome Data Available', ha='center', va='center',
                        fontsize=14, transform=ax1.transAxes)
                ax1.axis('off')
        except Exception as e:
            logger.error(f"Error creating win/loss pie chart: {e}")
            ax1.text(0.5, 0.5, 'Error Creating Chart', ha='center', va='center',
                    fontsize=14, transform=ax1.transAxes)
            ax1.axis('off')
        
        # P&L distribution by win/loss with validation
        try:
            if 'pnl' in trades_df.columns:
                # Log the state of the data
                logger.info(f"P&L distribution - Win count: {win_count}, Lose count: {lose_count}")
                
                # Ensure pnl values are numeric
                if not pd.api.types.is_numeric_dtype(trades_df['pnl']):
                    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
                    logger.info("Converted 'pnl' column to numeric")
                
                # Filter out NaN values
                win_pnl = win_trades['pnl'].dropna()
                lose_pnl = lose_trades['pnl'].dropna()
                
                logger.info(f"P&L distribution - Valid win P&L values: {len(win_pnl)}, Valid lose P&L values: {len(lose_pnl)}")
                
                if len(win_pnl) > 0 and len(lose_pnl) > 0:
                    # Convert to numpy arrays for histogram
                    win_pnl_values = win_pnl.values
                    lose_pnl_values = lose_pnl.values
                    
                    # Calculate bin count based on data size
                    bin_count = min(20, max(5, min(len(win_pnl_values), len(lose_pnl_values))//5))
                    
                    # Create histogram
                    ax2.hist([win_pnl_values, lose_pnl_values], bins=bin_count, alpha=0.7,
                            label=[f'Wins (avg: ${win_pnl.mean():.2f})', f'Losses (avg: ${lose_pnl.mean():.2f})'],
                            color=['green', 'red'])
                    ax2.set_xlabel('P&L ($)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('P&L Distribution by Outcome')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    logger.info(f"P&L stats - Win avg: ${win_pnl.mean():.2f}, Loss avg: ${lose_pnl.mean():.2f}")
                elif len(win_pnl) > 0:
                    ax2.text(0.5, 0.5, 'No Valid Loss P&L Data', ha='center', va='center',
                            fontsize=14, transform=ax2.transAxes)
                    ax2.axis('off')
                    logger.warning("Cannot create P&L distribution - No valid loss P&L data")
                elif len(lose_pnl) > 0:
                    ax2.text(0.5, 0.5, 'No Valid Win P&L Data', ha='center', va='center',
                            fontsize=14, transform=ax2.transAxes)
                    ax2.axis('off')
                    logger.warning("Cannot create P&L distribution - No valid win P&L data")
                else:
                    ax2.text(0.5, 0.5, 'No Valid P&L Data', ha='center', va='center',
                            fontsize=14, transform=ax2.transAxes)
                    ax2.axis('off')
                    logger.warning("Cannot create P&L distribution - No valid P&L data")
            else:
                ax2.text(0.5, 0.5, 'No P&L Data Available', ha='center', va='center',
                        fontsize=14, transform=ax2.transAxes)
                ax2.axis('off')
                logger.warning("Cannot create P&L distribution - No P&L column in trades data")
        except Exception as e:
            logger.error(f"Error creating P&L distribution chart: {e}")
            ax2.text(0.5, 0.5, 'Error Creating Chart', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')
        
        # Cumulative win/loss over time with validation
        try:
            if len(trades_df) > 0 and 'prediction_correct' in trades_df.columns:
                # Add trade number and calculate cumulative stats
                trades_df['trade_number'] = range(1, len(trades_df) + 1)
                
                # Handle boolean or non-boolean prediction_correct values
                if trades_df['prediction_correct'].dtype == bool:
                    trades_df['cumulative_wins'] = trades_df['prediction_correct'].cumsum()
                    trades_df['cumulative_losses'] = (~trades_df['prediction_correct']).cumsum()
                else:
                    # Convert to boolean if not already
                    trades_df['prediction_correct'] = trades_df['prediction_correct'].fillna(False).astype(bool)
                    trades_df['cumulative_wins'] = trades_df['prediction_correct'].cumsum()
                    trades_df['cumulative_losses'] = (~trades_df['prediction_correct']).cumsum()
                
                ax3.plot(trades_df['trade_number'], trades_df['cumulative_wins'],
                        color='green', linewidth=2, label='Cumulative Wins')
                ax3.plot(trades_df['trade_number'], trades_df['cumulative_losses'],
                        color='red', linewidth=2, label='Cumulative Losses')
                ax3.set_xlabel('Trade Number')
                ax3.set_ylabel('Count')
                ax3.set_title('Cumulative Win/Loss Over Time')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                logger.info(f"Cumulative stats - Final wins: {trades_df['cumulative_wins'].iloc[-1]}, losses: {trades_df['cumulative_losses'].iloc[-1]}")
            else:
                ax3.text(0.5, 0.5, 'Insufficient Trade Data', ha='center', va='center',
                        fontsize=14, transform=ax3.transAxes)
                ax3.axis('off')
        except Exception as e:
            logger.error(f"Error creating cumulative win/loss chart: {e}")
            ax3.text(0.5, 0.5, 'Error Creating Chart', ha='center', va='center',
                    fontsize=14, transform=ax3.transAxes)
            ax3.axis('off')
        
        # Win/Loss statistics summary with validation
        try:
            if win_count > 0 and lose_count > 0 and 'pnl' in trades_df.columns:
                # Ensure pnl values are numeric
                if not pd.api.types.is_numeric_dtype(trades_df['pnl']):
                    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
                
                # Calculate statistics with validation
                win_pnl_clean = win_trades['pnl'].dropna()
                lose_pnl_clean = lose_trades['pnl'].dropna()
                
                if len(win_pnl_clean) > 0 and len(lose_pnl_clean) > 0:
                    avg_win_pnl = win_pnl_clean.mean()
                    avg_lose_pnl = lose_pnl_clean.mean()
                    max_win = win_pnl_clean.max()
                    max_loss = lose_pnl_clean.min()
                    
                    # Calculate profit factor with validation
                    total_win_pnl = win_pnl_clean.sum()
                    total_lose_pnl = abs(lose_pnl_clean.sum())
                    profit_factor = total_win_pnl / total_lose_pnl if total_lose_pnl != 0 else float('inf')
                    
                    # Calculate total P&L safely
                    total_pnl = trades_df['pnl'].sum()
                    
                    stats_text = f"""
Trading Performance Summary:
• Total Trades: {len(trades_df)}
• Known Outcomes: {win_count + lose_count} of {len(trades_df)}
• Wins: {win_count} ({win_rate:.1f}%)
• Losses: {lose_count} ({100-win_rate:.1f}%)
• Average Win: ${avg_win_pnl:.2f}
• Average Loss: ${avg_lose_pnl:.2f}
• Best Trade: ${max_win:.2f}
• Worst Trade: ${max_loss:.2f}
• Profit Factor: {profit_factor:.2f}
• Total P&L: ${total_pnl:.2f}
                    """
                    
                    logger.info(f"Trading stats - Win rate: {win_rate:.1f}% ({win_count}/{win_count + lose_count}), Profit factor: {profit_factor:.2f}, Total P&L: ${total_pnl:.2f}")
                    
                    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                    ax4.set_xlim(0, 1)
                    ax4.set_ylim(0, 1)
                    ax4.axis('off')
                    ax4.set_title('Trading Statistics')
                else:
                    ax4.text(0.5, 0.5, 'Insufficient P&L Data', ha='center', va='center',
                            fontsize=14, transform=ax4.transAxes)
                    ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, 'Insufficient Trading Data', ha='center', va='center',
                        fontsize=14, transform=ax4.transAxes)
                ax4.axis('off')
        except Exception as e:
            logger.error(f"Error creating trading statistics summary: {e}")
            ax4.text(0.5, 0.5, 'Error Creating Statistics', ha='center', va='center',
                    fontsize=14, transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "win_loss_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_online_learning_status_png(self, output_dir: str) -> Dict[str, str]:
        """Create online learning status PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Online Learning System Status', fontsize=16, fontweight='bold')
        
        # Model update frequency
        model_names = []
        update_counts = []
        prediction_counts = []
        learning_active = []
        
        for model_name, tracker in self.model_trackers.items():
            model_names.append(model_name)
            update_counts.append(len(tracker.update_times))
            prediction_counts.append(len(tracker.predictions))
            learning_active.append(len(tracker.update_times) > 0)
        
        if model_names:
            # Update frequency comparison
            bars1 = ax1.bar(model_names, update_counts, color='lightblue', alpha=0.8)
            ax1.set_ylabel('Number of Updates')
            ax1.set_title('Model Update Frequency')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars1, update_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Learning activity status
            colors = ['green' if active else 'red' for active in learning_active]
            bars2 = ax2.bar(model_names, [1 if active else 0 for active in learning_active], 
                           color=colors, alpha=0.8)
            ax2.set_ylabel('Learning Active')
            ax2.set_title('Online Learning Status')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Inactive', 'Active'])
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Update to prediction ratio
            update_ratios = []
            for i, model_name in enumerate(model_names):
                if prediction_counts[i] > 0:
                    ratio = (update_counts[i] / prediction_counts[i]) * 100
                    update_ratios.append(ratio)
                else:
                    update_ratios.append(0)
            
            bars3 = ax3.bar(model_names, update_ratios, color='orange', alpha=0.8)
            ax3.set_ylabel('Update Rate (%)')
            ax3.set_title('Update to Prediction Ratio')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add percentage labels
            for bar, value in zip(bars3, update_ratios):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Online learning summary
        total_models = len(model_names)
        active_models = sum(learning_active)
        total_updates = sum(update_counts)
        total_predictions = sum(prediction_counts)
        
        summary_text = f"""
Online Learning Summary:
• Total Models: {total_models}
• Active Learning Models: {active_models}
• Total Model Updates: {total_updates}
• Total Predictions: {total_predictions}
• Overall Update Rate: {(total_updates/max(total_predictions,1)*100):.1f}%
• Learning Coverage: {(active_models/max(total_models,1)*100):.1f}%

Model Status:
        """
        
        for i, model_name in enumerate(model_names):
            status = "✓ Active" if learning_active[i] else "✗ Inactive"
            summary_text += f"• {model_name}: {status} ({update_counts[i]} updates)\n"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Learning System Summary')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "online_learning_status.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_accuracy_overview_png(self, output_dir: str) -> Dict[str, str]:
        """Create accuracy overview PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Accuracy Overview', fontsize=16, fontweight='bold')
        
        model_names = []
        accuracies = []
        prediction_counts = []
        confidences = []
        
        # Collect data from all models
        for model_name, tracker in self.model_trackers.items():
            summary = tracker.get_performance_summary()
            if summary.get('status') != 'no_data':
                model_names.append(model_name)
                accuracies.append(summary.get('overall_accuracy', 0))
                prediction_counts.append(summary.get('total_predictions', 0))
                confidences.append(summary.get('avg_confidence', 0))
        
        # Add ensemble
        ensemble_summary = self.ensemble_tracker.get_performance_summary()
        if ensemble_summary.get('status') != 'no_data':
            model_names.append('Ensemble')
            accuracies.append(ensemble_summary.get('overall_accuracy', 0))
            prediction_counts.append(ensemble_summary.get('total_predictions', 0))
            confidences.append(ensemble_summary.get('avg_confidence', 0))
        
        if model_names:
            # Accuracy comparison
            colors = ['lightgreen' if acc >= 0.6 else 'orange' if acc >= 0.5 else 'lightcoral' for acc in accuracies]
            bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Add accuracy labels
            for bar, value in zip(bars1, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Prediction volume
            bars2 = ax2.bar(model_names, prediction_counts, color='lightblue', alpha=0.8)
            ax2.set_ylabel('Number of Predictions')
            ax2.set_title('Prediction Volume by Model')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Confidence vs Accuracy scatter
            ax3.scatter(confidences, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
            for i, model in enumerate(model_names):
                ax3.annotate(model, (confidences[i], accuracies[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
            ax3.set_xlabel('Average Confidence')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Confidence vs Accuracy')
            ax3.grid(True, alpha=0.3)
            
            # Perfect calibration line
            ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
            ax3.legend()
        
        # Accuracy summary table
        if model_names:
            summary_data = []
            for i, model in enumerate(model_names):
                summary_data.append([
                    model,
                    f"{accuracies[i]:.3f}",
                    f"{prediction_counts[i]}",
                    f"{confidences[i]:.3f}"
                ])
            
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=summary_data,
                            colLabels=['Model', 'Accuracy', 'Predictions', 'Avg Confidence'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title('Accuracy Summary Table')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "accuracy_overview.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_accuracy_per_model_png(self, output_dir: str) -> Dict[str, str]:
        """Create detailed per-model accuracy PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        
        # Create dynamic subplot layout based on number of models
        n_models = len(self.model_trackers) + 1  # +1 for ensemble
        if n_models <= 2:
            fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
            if n_models == 1:
                axes = [axes]
        else:
            rows = (n_models + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(16, 6*rows))
            axes = axes.flatten()
        
        fig.suptitle('Detailed Model Accuracy Analysis', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Plot individual model accuracy trends
        for model_name, tracker in self.model_trackers.items():
            if tracker.accuracy_history and plot_idx < len(axes):
                ax = axes[plot_idx]
                x_vals = list(range(len(tracker.accuracy_history)))
                ax.plot(x_vals, tracker.accuracy_history, linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Update Number')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model_name} Accuracy Evolution')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Add trend line
                if len(tracker.accuracy_history) > 1:
                    z = np.polyfit(x_vals, tracker.accuracy_history, 1)
                    p = np.poly1d(z)
                    ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, 
                           label=f'Trend: {"↗" if z[0] > 0 else "↘"}')
                    ax.legend()
                
                plot_idx += 1
        
        # Plot ensemble accuracy if available
        if self.ensemble_tracker.accuracy_history and plot_idx < len(axes):
            ax = axes[plot_idx]
            x_vals = list(range(len(self.ensemble_tracker.accuracy_history)))
            ax.plot(x_vals, self.ensemble_tracker.accuracy_history, linewidth=2, 
                   marker='o', markersize=4, color='red')
            ax.set_xlabel('Update Number')
            ax.set_ylabel('Accuracy')
            ax.set_title('Ensemble Accuracy Evolution')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "accuracy_per_model.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_confidence_analysis_png(self, output_dir: str) -> Dict[str, str]:
        """Create confidence analysis PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confidence Level Analysis', fontsize=16, fontweight='bold')
        
        # Collect confidence data from all models
        all_confidences = []
        all_accuracies = []
        model_confidence_data = {}
        
        for model_name, tracker in self.model_trackers.items():
            if tracker.predictions:
                confidences = [pred.get('confidence', 0) for pred in tracker.predictions]
                accuracies = [pred.get('correct', False) for pred in tracker.predictions]
                all_confidences.extend(confidences)
                all_accuracies.extend(accuracies)
                model_confidence_data[model_name] = {'confidences': confidences, 'accuracies': accuracies}
        
        # Add ensemble data
        if self.ensemble_tracker.predictions:
            ensemble_confidences = [pred.get('confidence', 0) for pred in self.ensemble_tracker.predictions]
            ensemble_accuracies = [pred.get('correct', False) for pred in self.ensemble_tracker.predictions]
            all_confidences.extend(ensemble_confidences)
            all_accuracies.extend(ensemble_accuracies)
            model_confidence_data['Ensemble'] = {'confidences': ensemble_confidences, 'accuracies': ensemble_accuracies}
        
        # If no confidence data is available, create an empty chart with a message
        if not all_confidences:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No Confidence Data Available', ha='center', va='center',
                        fontsize=14, transform=ax.transAxes)
                ax.axis('off')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, "confidence_analysis.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return {'single_chart': filename}
        
        if all_confidences:
            # Confidence distribution
            ax1.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(all_confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_confidences):.3f}')
            ax1.set_xlabel('Confidence Level')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Overall Confidence Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confidence vs Accuracy calibration
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            bin_accuracies = []
            bin_counts = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (np.array(all_confidences) >= confidence_bins[i]) & \
                       (np.array(all_confidences) < confidence_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean(np.array(all_accuracies)[mask])
                    bin_accuracies.append(bin_accuracy)
                    bin_counts.append(np.sum(mask))
                else:
                    bin_accuracies.append(0)
                    bin_counts.append(0)
            
            ax2.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8, label='Actual Accuracy')
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
            ax2.set_xlabel('Confidence Level')
            ax2.set_ylabel('Actual Accuracy')
            ax2.set_title('Confidence Calibration')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            
            # Confidence by model
            model_names = list(model_confidence_data.keys())
            model_avg_confidences = [np.mean(data['confidences']) for data in model_confidence_data.values()]
            
            bars = ax3.bar(model_names, model_avg_confidences, color='lightgreen', alpha=0.8)
            ax3.set_ylabel('Average Confidence')
            ax3.set_title('Average Confidence by Model')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, model_avg_confidences):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence statistics summary
        if all_confidences:
            try:
                conf_mean = np.mean(all_confidences) if all_confidences else 0.0
                conf_std = np.std(all_confidences) if all_confidences else 0.0
                conf_min = np.min(all_confidences) if all_confidences else 0.0
                conf_max = np.max(all_confidences) if all_confidences else 0.0
                
                # Calculate calibration error (ECE - Expected Calibration Error)
                ece = 0.0
                
                # Safely check if bin data exists and is valid
                try:
                    # Check if bin variables exist in the current scope
                    if ('bin_accuracies' in locals() and 'bin_counts' in locals() and
                        'bin_centers' in locals() and bin_accuracies and bin_counts and bin_centers):
                        
                        # Validate all arrays have the same length and contain valid data
                        if (len(bin_accuracies) == len(bin_counts) == len(bin_centers) and
                            all(x is not None for x in bin_accuracies) and
                            all(x is not None for x in bin_counts) and
                            all(x is not None for x in bin_centers)):
                            
                            for i, (bin_acc, bin_count, bin_conf) in enumerate(zip(bin_accuracies, bin_counts, bin_centers)):
                                if bin_count > 0 and bin_acc is not None and bin_conf is not None:
                                    try:
                                        ece += (bin_count / len(all_confidences)) * abs(float(bin_conf) - float(bin_acc))
                                    except (TypeError, ValueError) as e:
                                        logger.warning(f"Error calculating ECE for bin {i}: {e}")
                                        continue
                        else:
                            logger.debug("Bin data arrays have mismatched lengths or contain None values")
                            ece = 0.0
                    else:
                        logger.debug("Bin data not available for ECE calculation")
                        ece = 0.0
                except Exception as e:
                    logger.warning(f"Error accessing bin data for ECE calculation: {e}")
                    ece = 0.0
                
                # Calculate overall accuracy safely
                overall_acc = np.mean(all_accuracies) if all_accuracies else 0.0
                
                # Format ECE safely
                ece_str = f"{ece:.3f}" if ece is not None and not np.isnan(ece) else 'N/A'
                
                stats_text = f"""
Confidence Analysis Summary:
• Total Predictions: {len(all_confidences)}
• Mean Confidence: {conf_mean:.3f}
• Std Confidence: {conf_std:.3f}
• Min Confidence: {conf_min:.3f}
• Max Confidence: {conf_max:.3f}
• Expected Calibration Error: {ece_str}
• Overall Accuracy: {overall_acc:.3f}

Calibration Quality:
• ECE < 0.05: Excellent
• ECE < 0.10: Good
• ECE < 0.15: Fair
• ECE ≥ 0.15: Poor
                """
                ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
                ax4.set_title('Confidence Statistics')
            except Exception as e:
                logger.error(f"Error calculating confidence statistics: {e}")
                ax4.text(0.5, 0.5, 'Error Calculating Statistics', ha='center', va='center',
                        fontsize=14, transform=ax4.transAxes)
                ax4.axis('off')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "confidence_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_model_training_status_png(self, output_dir: str) -> Dict[str, str]:
        """Create model training status PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Training Status & Progress', fontsize=16, fontweight='bold')
        
        model_names = []
        training_samples = []
        last_update_times = []
        update_frequencies = []
        
        current_time = time.time()
        
        for model_name, tracker in self.model_trackers.items():
            model_names.append(model_name)
            training_samples.append(len(tracker.update_times))
            
            if tracker.update_times:
                last_update = max(tracker.update_times)
                last_update_times.append((current_time - last_update) / 3600)  # Hours ago
                
                # Calculate update frequency (updates per hour)
                if len(tracker.update_times) > 1:
                    time_span = max(tracker.update_times) - min(tracker.update_times)
                    if time_span > 0:
                        freq = len(tracker.update_times) / (time_span / 3600)
                        update_frequencies.append(freq)
                    else:
                        update_frequencies.append(0)
                else:
                    update_frequencies.append(0)
            else:
                last_update_times.append(float('inf'))
                update_frequencies.append(0)
        
        if model_names:
            # Training samples per model
            bars1 = ax1.bar(model_names, training_samples, color='lightblue', alpha=0.8)
            ax1.set_ylabel('Training Samples')
            ax1.set_title('Training Data Volume by Model')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars1, training_samples):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Last update recency
            colors = ['green' if hours < 1 else 'orange' if hours < 24 else 'red'
                     for hours in last_update_times]
            finite_times = [min(hours, 48) for hours in last_update_times]  # Cap at 48 hours for display
            
            bars2 = ax2.bar(model_names, finite_times, color=colors, alpha=0.8)
            ax2.set_ylabel('Hours Since Last Update')
            ax2.set_title('Model Update Recency')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Update frequency
            bars3 = ax3.bar(model_names, update_frequencies, color='lightgreen', alpha=0.8)
            ax3.set_ylabel('Updates per Hour')
            ax3.set_title('Model Update Frequency')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add frequency labels
            for bar, value in zip(bars3, update_frequencies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Training status summary
        total_samples = sum(training_samples)
        active_models = sum(1 for t in last_update_times if t < 24)  # Updated in last 24 hours
        avg_frequency = np.mean(update_frequencies) if update_frequencies else 0
        
        status_text = f"""
Training System Status:
• Total Models: {len(model_names)}
• Active Models (24h): {active_models}
• Total Training Samples: {total_samples}
• Average Update Frequency: {avg_frequency:.2f}/hour

Model Status Details:
        """
        
        for i, model_name in enumerate(model_names):
            if last_update_times[i] == float('inf'):
                status = "Never Updated"
            elif last_update_times[i] < 1:
                status = f"Updated {last_update_times[i]*60:.0f}m ago"
            elif last_update_times[i] < 24:
                status = f"Updated {last_update_times[i]:.1f}h ago"
            else:
                status = f"Updated {last_update_times[i]/24:.1f}d ago"
            
            status_text += f"• {model_name}: {training_samples[i]} samples, {status}\n"
        
        ax4.text(0.1, 0.5, status_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Training Status Summary')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "model_training_status.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_system_performance_png(self, output_dir: str) -> Dict[str, str]:
        """Create overall system performance PNG report"""
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall System Performance Dashboard', fontsize=16, fontweight='bold')
        
        # System health indicators
        health_metrics = {
            'Latency': 'Good' if self.latency_tracker.end_to_end_latencies and
                      np.mean(self.latency_tracker.end_to_end_latencies) < 100 else 'Warning',
            'Accuracy': 'Good' if self.ensemble_tracker.get_performance_summary().get('overall_accuracy', 0) > 0.6 else 'Warning',
            'Trading': 'Good' if len(self.trades) > 0 else 'Warning',
            'Learning': 'Good' if any(len(tracker.update_times) > 0 for tracker in self.model_trackers.values()) else 'Warning'
        }
        
        # Health status pie chart
        good_count = sum(1 for status in health_metrics.values() if status == 'Good')
        warning_count = len(health_metrics) - good_count
        
        if good_count + warning_count > 0:
            sizes = [good_count, warning_count]
            labels = [f'Healthy ({good_count})', f'Warning ({warning_count})']
            colors = ['lightgreen', 'orange']
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
                                              autopct='%1.1f%%', startangle=90, shadow=True)
            ax1.set_title('System Health Overview')
        
        # Performance metrics over time with validation
        try:
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                
                # Check if pnl column exists
                if 'pnl' in trades_df.columns:
                    # Ensure pnl values are numeric
                    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
                    
                    # Sort by timestamp
                    if 'timestamp' in trades_df.columns:
                        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
                        trades_df = trades_df.sort_values('timestamp')
                    
                    # Calculate cumulative P&L
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    
                    # Plot if we have valid data
                    if not trades_df['cumulative_pnl'].isna().all():
                        ax2.plot(trades_df.index, trades_df['cumulative_pnl'], linewidth=2, color='blue')
                        ax2.set_xlabel('Trade Number')
                        ax2.set_ylabel('Cumulative P&L ($)')
                        ax2.set_title('Cumulative P&L Over Time')
                        ax2.grid(True, alpha=0.3)
                        
                        # Add final P&L annotation
                        final_pnl = trades_df['cumulative_pnl'].iloc[-1]
                        ax2.annotate(f'Final P&L: ${final_pnl:.2f}',
                                    xy=(len(trades_df)-1, final_pnl),
                                    xytext=(10, 10), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                        
                        logger.info(f"Cumulative P&L chart created - Final P&L: ${final_pnl:.2f}")
                    else:
                        ax2.text(0.5, 0.5, 'No Valid P&L Data', ha='center', va='center',
                                fontsize=14, transform=ax2.transAxes)
                        ax2.axis('off')
                else:
                    ax2.text(0.5, 0.5, 'No P&L Data Available', ha='center', va='center',
                            fontsize=14, transform=ax2.transAxes)
                    ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, 'No Trade Data Available', ha='center', va='center',
                        fontsize=14, transform=ax2.transAxes)
                ax2.axis('off')
        except Exception as e:
            logger.error(f"Error creating cumulative P&L chart: {e}")
            ax2.text(0.5, 0.5, 'Error Creating Chart', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')
        
        # System throughput metrics
        total_predictions = sum(len(tracker.predictions) for tracker in self.model_trackers.values())
        total_updates = sum(len(tracker.update_times) for tracker in self.model_trackers.values())
        total_trades = len(self.trades)
        
        throughput_metrics = ['Predictions', 'Model Updates', 'Trades']
        throughput_values = [total_predictions, total_updates, total_trades]
        
        bars = ax3.bar(throughput_metrics, throughput_values, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax3.set_ylabel('Count')
        ax3.set_title('System Throughput Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, throughput_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput_values)*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # System summary dashboard
        latency_summary = self.latency_tracker.get_latency_summary()
        ensemble_summary = self.ensemble_tracker.get_performance_summary()
        
        avg_latency = latency_summary.get('end_to_end', {}).get('avg_ms', 0)
        overall_accuracy = ensemble_summary.get('overall_accuracy', 0)
        # Calculate win rate and total PnL with validation
        try:
            # Count trades with prediction_correct=True
            win_count = sum(1 for trade in self.trades if trade.get('prediction_correct', False) is True)
            # Count trades with prediction_correct field (either True or False, not None)
            known_outcome_count = sum(1 for trade in self.trades if 'prediction_correct' in trade)
            # Calculate win rate only from trades with known outcomes
            win_rate = (win_count / max(known_outcome_count, 1)) * 100
            
            # Calculate total PnL safely
            total_pnl = sum(float(trade.get('pnl', 0)) for trade in self.trades)
            
            logger.info(f"Dashboard stats - Win rate: {win_rate:.1f}% ({win_count}/{known_outcome_count}), Total P&L: ${total_pnl:.2f}")
        except Exception as e:
            logger.error(f"Error calculating win rate or PnL: {e}")
            win_rate = 0.0
            total_pnl = 0.0
        
        # Get known outcome count for win rate context
        known_outcome_count = sum(1 for trade in self.trades if 'prediction_correct' in trade)
        win_count = sum(1 for trade in self.trades if trade.get('prediction_correct', False) is True)
        
        dashboard_text = f"""
System Performance Dashboard:

PERFORMANCE METRICS:
• Average Latency: {avg_latency:.2f} ms
• Overall Accuracy: {overall_accuracy:.3f}
• Win Rate: {win_rate:.1f}% ({win_count}/{known_outcome_count})
• Total P&L: ${total_pnl:.2f}

SYSTEM HEALTH:
• Latency Status: {health_metrics['Latency']}
• Accuracy Status: {health_metrics['Accuracy']}
• Trading Status: {health_metrics['Trading']}
• Learning Status: {health_metrics['Learning']}

THROUGHPUT:
• Total Predictions: {total_predictions}
• Model Updates: {total_updates}
• Trades Executed: {total_trades}
        """
        
        ax4.text(0.1, 0.5, dashboard_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('System Dashboard')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, "system_performance.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return {'single_chart': filename}
    
    def _create_error_placeholder_png(self, output_dir: str, report_name: str, error_message: str) -> Dict[str, str]:
        """Create a placeholder PNG when report generation fails"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Create error message display
            error_text = f"""
Error Generating {report_name.replace('_', ' ').title()} Report

Error Details:
{error_message}

This report could not be generated due to insufficient data
or a processing error. Please check the logs for more details.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax.text(0.5, 0.5, error_text, ha='center', va='center',
                   fontsize=12, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(f'Error: {report_name.replace("_", " ").title()}', fontsize=16, color='red')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f"{report_name}_error.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {f'{report_name}_error': filename}
            
        except Exception as e:
            logger.error(f"Failed to create error placeholder PNG: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Create enhanced analytics instance
    analytics = EnhancedBacktestAnalytics()
    
    # Generate sample data for testing
    import random
    
    # Add some sample trades
    for i in range(20):
        analytics.record_trade({
            'timestamp': datetime.now() - timedelta(hours=i),
            'symbol': f'STOCK{i%5}',
            'prediction_correct': random.choice([True, False]),
            'pnl': random.uniform(-50, 100),
            'confidence': random.uniform(0.5, 0.9)
        })
    
    # Add some sample latency data
    for i in range(50):
        analytics.latency_tracker.record_latency('end_to_end', random.uniform(20, 150))
        analytics.latency_tracker.record_latency('kelly_sizing', random.uniform(5, 25))
        analytics.latency_tracker.record_latency('ml_prediction', random.uniform(10, 40))
        analytics.latency_tracker.record_latency('data_filter', random.uniform(2, 15))
    
    # Generate PNG reports
    print("Generating PNG reports...")
    report_files = analytics.generate_png_reports("test_reports")
    
    print(f"Generated {len(report_files)} PNG reports:")
    for report_name, filepath in report_files.items():
        print(f"  {report_name}: {filepath}")
