#!/usr/bin/env python3

import sys
import os
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

logger = get_system_logger("filters.entry_timing_optimizer")

@dataclass
class TradingWindow:
    """Trading window configuration"""
    name: str
    start_time: time
    end_time: time
    expected_win_rate: float
    description: str

class EntryTimingOptimizer:
    """
    Simple entry timing optimizer
    Only allows trades during proven high win-rate windows
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'filters.yaml')
        
        self.config = self._load_config(config_path)
        
        # Load trading windows from config or use defaults
        self.trading_windows = self._setup_trading_windows()
        
        # Load avoid windows from config or use defaults
        self.avoid_windows = self._setup_avoid_windows()
        
        self.entry_history = []
        self.window_stats = {window.name: {'attempts': 0, 'accepted': 0} for window in self.trading_windows}
        
        logger.startup({
            "optimal_windows_count": len(self.trading_windows),
            "avoid_windows_count": len(self.avoid_windows),
            "config_path": config_path
        })
        self._log_trading_windows()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load timing configuration from YAML"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(e, {"operation": "load_config", "path": config_path})
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default timing configuration"""
        return {
            'entry_timing': {
                'optimal_windows': [
                    {
                        'name': 'morning_momentum',
                        'start_time': '10:00',
                        'end_time': '10:30',
                        'expected_win_rate': 0.75,
                        'description': 'Post-opening range establishment'
                    },
                    {
                        'name': 'afternoon_initiation',
                        'start_time': '14:00',
                        'end_time': '14:30',
                        'expected_win_rate': 0.70,
                        'description': 'Afternoon momentum initiation'
                    },
                    {
                        'name': 'end_day_acceleration',
                        'start_time': '15:30',
                        'end_time': '16:00',
                        'expected_win_rate': 0.80,
                        'description': 'End-of-day momentum acceleration'
                    }
                ],
                'avoid_windows': [
                    {
                        'name': 'market_open',
                        'start_time': '09:30',
                        'end_time': '10:00',
                        'reason': 'Opening volatility and noise'
                    },
                    {
                        'name': 'lunch_lull',
                        'start_time': '11:30',
                        'end_time': '13:30',
                        'reason': 'Low volume and momentum'
                    }
                ]
            }
        }
    
    def _setup_trading_windows(self) -> List[TradingWindow]:
        """Setup trading windows from config"""
        windows_config = self.config.get('entry_timing', {}).get('optimal_windows', [])
        windows = []
        
        for window_config in windows_config:
            try:
                start_time = time.fromisoformat(window_config['start_time'])
                end_time = time.fromisoformat(window_config['end_time'])
                
                window = TradingWindow(
                    name=window_config['name'],
                    start_time=start_time,
                    end_time=end_time,
                    expected_win_rate=window_config['expected_win_rate'],
                    description=window_config['description']
                )
                windows.append(window)
            except Exception as e:
                logger.error(e, {"operation": "setup_trading_window", "window": window_config})
        
        # Fallback to defaults if no valid windows
        if not windows:
            windows = [
                TradingWindow(
                    name="morning_momentum",
                    start_time=time(10, 0),
                    end_time=time(10, 30),
                    expected_win_rate=0.75,
                    description="Post-opening range establishment"
                ),
                TradingWindow(
                    name="afternoon_initiation",
                    start_time=time(14, 0),
                    end_time=time(14, 30),
                    expected_win_rate=0.70,
                    description="Afternoon momentum initiation"
                ),
                TradingWindow(
                    name="end_day_acceleration",
                    start_time=time(15, 30),
                    end_time=time(16, 0),
                    expected_win_rate=0.80,
                    description="End-of-day momentum acceleration"
                )
            ]
        
        return windows
    
    def _setup_avoid_windows(self) -> List[Dict]:
        """Setup avoid windows from config"""
        avoid_config = self.config.get('entry_timing', {}).get('avoid_windows', [])
        avoid_windows = []
        
        for avoid in avoid_config:
            try:
                start_time = time.fromisoformat(avoid['start_time'])
                end_time = time.fromisoformat(avoid['end_time'])
                
                avoid_windows.append({
                    'name': avoid['name'],
                    'start': start_time,
                    'end': end_time,
                    'reason': avoid['reason']
                })
            except Exception as e:
                logger.error(e, {"operation": "setup_avoid_window", "window": avoid})
        
        # Fallback to defaults if no valid windows
        if not avoid_windows:
            avoid_windows = [
                {
                    'name': 'market_open',
                    'start': time(9, 30),
                    'end': time(10, 0),
                    'reason': 'Opening volatility and noise'
                },
                {
                    'name': 'lunch_lull',
                    'start': time(11, 30),
                    'end': time(13, 30),
                    'reason': 'Low volume and momentum'
                }
            ]
        
        return avoid_windows
    
    def _log_trading_windows(self):
        """Log the configured trading windows"""
        logger.info("Optimal trading windows configuration:")
        
        total_trading_minutes = 0
        for window in self.trading_windows:
            # Calculate window duration
            start_minutes = window.start_time.hour * 60 + window.start_time.minute
            end_minutes = window.end_time.hour * 60 + window.end_time.minute
            duration_minutes = end_minutes - start_minutes
            total_trading_minutes += duration_minutes
            
            logger.info(f"  {window.name}: {window.start_time.strftime('%H:%M')}-"
                       f"{window.end_time.strftime('%H:%M')} "
                       f"({duration_minutes}min, {window.expected_win_rate:.0%} win rate) - {window.description}")
        
        logger.info(f"Total optimal trading time: {total_trading_minutes} minutes ({total_trading_minutes/60:.1f} hours)")
        
        # Log avoid windows
        logger.info("Avoid windows configuration:")
        total_avoid_minutes = 0
        for avoid in self.avoid_windows:
            start_minutes = avoid['start'].hour * 60 + avoid['start'].minute
            end_minutes = avoid['end'].hour * 60 + avoid['end'].minute
            duration_minutes = end_minutes - start_minutes
            total_avoid_minutes += duration_minutes
            
            logger.info(f"  {avoid['name']}: {avoid['start'].strftime('%H:%M')}-"
                       f"{avoid['end'].strftime('%H:%M')} "
                       f"({duration_minutes}min) - {avoid['reason']}")
        
        logger.info(f"Total avoid time: {total_avoid_minutes} minutes ({total_avoid_minutes/60:.1f} hours)")
        
        # Calculate coverage statistics
        market_hours = 6.5 * 60  # 9:30 AM to 4:00 PM = 6.5 hours
        optimal_coverage = (total_trading_minutes / market_hours) * 100
        avoid_coverage = (total_avoid_minutes / market_hours) * 100
        
        logger.info(f"Market coverage: {optimal_coverage:.1f}% optimal, {avoid_coverage:.1f}% avoid, "
                   f"{100 - optimal_coverage - avoid_coverage:.1f}% neutral")
    
    def is_valid_entry_time(self, check_time: Optional[datetime] = None) -> Tuple[bool, Dict]:
        """
        Check if current time is valid for new entries
        
        Args:
            check_time: Time to check (defaults to current time)
            
        Returns:
            Tuple of (is_valid, window_info)
        """
        try:
            if check_time is None:
                check_time = datetime.now()
            
            current_time = check_time.time()
            logger.debug(f"Checking entry time validity for {current_time.strftime('%H:%M:%S')}")
            
            # Check if we're in an optimal trading window
            for window in self.trading_windows:
                if window.start_time <= current_time <= window.end_time:
                    logger.debug(f"Time {current_time.strftime('%H:%M:%S')} is within optimal window: {window.name}")
                    return True, {
                        'valid': True,
                        'window_name': window.name,
                        'window_description': window.description,
                        'expected_win_rate': window.expected_win_rate,
                        'time_remaining': self._calculate_time_remaining(current_time, window.end_time),
                        'current_time': current_time.strftime('%H:%M:%S')
                    }
            
            # Check if we're in an avoid window
            for avoid in self.avoid_windows:
                if avoid['start'] <= current_time <= avoid['end']:
                    logger.debug(f"Time {current_time.strftime('%H:%M:%S')} is within avoid window: {avoid['name']} - {avoid['reason']}")
                    return False, {
                        'valid': False,
                        'reason': 'avoid_window',
                        'avoid_window': avoid['name'],
                        'avoid_reason': avoid['reason'],
                        'next_window': self._get_next_trading_window(current_time),
                        'current_time': current_time.strftime('%H:%M:%S')
                    }
            
            # Not in any window (general avoid time)
            logger.debug(f"Time {current_time.strftime('%H:%M:%S')} is outside all optimal windows")
            return False, {
                'valid': False,
                'reason': 'outside_optimal_windows',
                'next_window': self._get_next_trading_window(current_time),
                'current_time': current_time.strftime('%H:%M:%S')
            }
            
        except Exception as e:
            logger.error(e, {"operation": "is_valid_entry_time"})
            return False, {'valid': False, 'reason': 'error', 'error': str(e)}
    
    def _calculate_time_remaining(self, current_time: time, end_time: time) -> str:
        """Calculate time remaining in current window"""
        try:
            # Convert to datetime for calculation
            today = datetime.now().date()
            current_dt = datetime.combine(today, current_time)
            end_dt = datetime.combine(today, end_time)
            
            if end_dt < current_dt:  # Handle day boundary
                end_dt = end_dt.replace(day=end_dt.day + 1)
                logger.debug(f"Time calculation crossed day boundary: {current_time} -> {end_time}")
            
            remaining = end_dt - current_dt
            minutes = int(remaining.total_seconds() / 60)
            
            logger.debug(f"Time remaining calculation: {current_time} to {end_time} = {minutes} minutes")
            return f"{minutes} minutes"
            
        except Exception as e:
            logger.error(e, {"operation": "_calculate_time_remaining", "current_time": str(current_time), "end_time": str(end_time)})
            return "unknown"
    
    def _get_next_trading_window(self, current_time: time) -> Optional[Dict]:
        """Get the next available trading window"""
        try:
            logger.debug(f"Finding next trading window after {current_time.strftime('%H:%M:%S')}")
            
            for window in self.trading_windows:
                if current_time < window.start_time:
                    logger.debug(f"Next window found: {window.name} at {window.start_time.strftime('%H:%M')}")
                    return {
                        'name': window.name,
                        'start_time': window.start_time.strftime('%H:%M'),
                        'end_time': window.end_time.strftime('%H:%M'),
                        'description': window.description,
                        'expected_win_rate': window.expected_win_rate
                    }
            
            # If past all windows today, return first window of next day
            first_window = self.trading_windows[0]
            logger.debug(f"Past all windows today, next window is tomorrow: {first_window.name}")
            return {
                'name': first_window.name + '_next_day',
                'start_time': first_window.start_time.strftime('%H:%M'),
                'end_time': first_window.end_time.strftime('%H:%M'),
                'description': f"Next day: {first_window.description}",
                'expected_win_rate': first_window.expected_win_rate
            }
            
        except Exception as e:
            logger.error(e, {"operation": "_get_next_trading_window", "current_time": str(current_time)})
            return None
    
    def validate_trade_entry(self, symbol: str, check_time: Optional[datetime] = None) -> Tuple[bool, Dict]:
        """
        Validate if a trade entry should be allowed based on timing
        
        Args:
            symbol: Stock symbol for logging
            check_time: Time to check (defaults to current time)
            
        Returns:
            Tuple of (should_allow, timing_info)
        """
        try:
            logger.debug(f"Validating trade entry for {symbol} at {check_time or 'current time'}")
            is_valid, window_info = self.is_valid_entry_time(check_time)
            
            # Update statistics
            if is_valid:
                window_name = window_info.get('window_name', 'unknown')
                old_attempts = self.window_stats[window_name]['attempts']
                old_accepted = self.window_stats[window_name]['accepted']
                
                self.window_stats[window_name]['attempts'] += 1
                self.window_stats[window_name]['accepted'] += 1
                
                logger.debug(f"Updated stats for {window_name}: attempts {old_attempts} -> {self.window_stats[window_name]['attempts']}, "
                           f"accepted {old_accepted} -> {self.window_stats[window_name]['accepted']}")
            else:
                logger.debug(f"Entry rejected for {symbol}: {window_info.get('reason', 'unknown')}")
            
            # Log entry decision
            self._log_entry_decision(symbol, is_valid, window_info)
            
            return is_valid, window_info
            
        except Exception as e:
            logger.error(e, {"operation": "validate_trade_entry", "symbol": symbol})
            return False, {'valid': False, 'reason': 'error', 'error': str(e)}
    
    def _log_entry_decision(self, symbol: str, allowed: bool, info: Dict):
        """Log entry timing decision"""
        timestamp = datetime.now()
        
        entry_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'allowed': allowed,
            'info': info
        }
        
        self.entry_history.append(entry_record)
        
        # Keep only last 100 entries
        history_before = len(self.entry_history)
        if len(self.entry_history) > 100:
            self.entry_history = self.entry_history[-100:]
            logger.debug(f"Entry history trimmed from {history_before} to {len(self.entry_history)} entries")
        
        # Log the decision with enhanced context
        if allowed:
            window_name = info.get('window_name', 'unknown')
            win_rate = info.get('expected_win_rate', 0)
            time_remaining = info.get('time_remaining', 'unknown')
            
            logger.info(f"Entry ALLOWED for {symbol}: {window_name} window "
                       f"({win_rate:.0%} win rate, {time_remaining} remaining)")
            
            # Log additional context for allowed entries
            logger.debug(f"Entry context for {symbol}: window_description='{info.get('window_description', 'N/A')}', "
                        f"current_time={info.get('current_time', 'N/A')}")
            
            logger.log_filter_decision("entry_timing", symbol, True, info)
        else:
            reason = info.get('reason', 'unknown')
            next_window = info.get('next_window', {})
            
            logger.info(f"Entry BLOCKED for {symbol}: {reason}. "
                       f"Next window: {next_window.get('name', 'unknown')} "
                       f"at {next_window.get('start_time', 'unknown')}")
            
            # Log additional context for blocked entries
            if reason == 'avoid_window':
                avoid_window = info.get('avoid_window', 'unknown')
                avoid_reason = info.get('avoid_reason', 'unknown')
                logger.debug(f"Entry blocked in avoid window '{avoid_window}': {avoid_reason}")
            elif reason == 'outside_optimal_windows':
                logger.debug(f"Entry blocked - outside all optimal trading windows at {info.get('current_time', 'N/A')}")
            
            logger.log_filter_decision("entry_timing", symbol, False, info)
        
        # Log running statistics periodically
        if len(self.entry_history) % 10 == 0:  # Every 10 entries
            total_attempts = sum(stats['attempts'] for stats in self.window_stats.values())
            total_accepted = sum(stats['accepted'] for stats in self.window_stats.values())
            acceptance_rate = (total_accepted / max(total_attempts, 1)) * 100
            logger.debug(f"Running stats: {total_attempts} attempts, {total_accepted} accepted ({acceptance_rate:.1f}%)")
    
    def get_current_window_info(self) -> Optional[Dict]:
        """Get information about current trading window"""
        is_valid, info = self.is_valid_entry_time()
        
        if is_valid:
            logger.debug(f"Current window info: {info.get('window_name', 'unknown')} - {info.get('time_remaining', 'unknown')} remaining")
            return info
        else:
            logger.debug(f"No current valid window: {info.get('reason', 'unknown')}")
            return None
    
    def get_timing_stats(self) -> Dict:
        """Get entry timing statistics"""
        total_attempts = sum(stats['attempts'] for stats in self.window_stats.values())
        total_accepted = sum(stats['accepted'] for stats in self.window_stats.values())
        acceptance_rate = (total_accepted / max(total_attempts, 1)) * 100
        
        logger.debug(f"Timing stats requested: {total_attempts} attempts, {total_accepted} accepted, {acceptance_rate:.1f}% rate")
        
        stats_data = {
            'total_entry_attempts': total_attempts,
            'total_entries_accepted': total_accepted,
            'acceptance_rate_pct': acceptance_rate,
            'window_stats': {
                name: {
                    'attempts': stats['attempts'],
                    'accepted': stats['accepted'],
                    'acceptance_rate': (stats['accepted'] / max(stats['attempts'], 1)) * 100
                }
                for name, stats in self.window_stats.items()
            },
            'optimal_windows': [
                {
                    'name': window.name,
                    'time_range': f"{window.start_time.strftime('%H:%M')}-{window.end_time.strftime('%H:%M')}",
                    'expected_win_rate': window.expected_win_rate,
                    'description': window.description
                }
                for window in self.trading_windows
            ],
            'recent_entries': self.entry_history[-10:] if self.entry_history else []
        }
        
        # Log performance metrics
        logger.performance(stats_data)
        return stats_data
    
    def get_next_entry_opportunity(self) -> Optional[Dict]:
        """Get next entry opportunity information"""
        current_time = datetime.now().time()
        logger.debug(f"Getting next entry opportunity from {current_time.strftime('%H:%M:%S')}")
        
        next_window = self._get_next_trading_window(current_time)
        is_currently_valid = self.is_valid_entry_time()[0]
        
        if next_window:
            opportunity = {
                'next_window': next_window,
                'current_time': current_time.strftime('%H:%M:%S'),
                'is_currently_valid': is_currently_valid
            }
            logger.debug(f"Next opportunity: {next_window['name']} at {next_window['start_time']}, currently valid: {is_currently_valid}")
            return opportunity
        
        logger.warning("No next entry opportunity found")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Create entry timing optimizer
    timing_optimizer = EntryTimingOptimizer()
    logger.info("Starting Entry Timing Optimizer example")
    
    # Test different times throughout the day
    test_times = [
        time(9, 45),   # Market open (should be blocked)
        time(10, 15),  # Morning window (should be allowed)
        time(12, 0),   # Lunch lull (should be blocked)
        time(14, 15),  # Afternoon window (should be allowed)
        time(15, 45),  # End-day window (should be allowed)
        time(16, 30),  # After hours (should be blocked)
    ]
    
    print("Entry Timing Test:")
    print("=" * 50)
    
    for test_time in test_times:
        test_datetime = datetime.combine(datetime.now().date(), test_time)
        is_valid, info = timing_optimizer.validate_trade_entry("TEST", test_datetime)
        
        status = "✓ ALLOWED" if is_valid else "✗ BLOCKED"
        print(f"{test_time.strftime('%H:%M')} - {status}")
        
        if is_valid:
            print(f"  Window: {info.get('window_name', 'unknown')}")
            print(f"  Win Rate: {info.get('expected_win_rate', 0):.0%}")
            print(f"  Time Left: {info.get('time_remaining', 'unknown')}")
        else:
            print(f"  Reason: {info.get('reason', 'unknown')}")
            next_window = info.get('next_window', {})
            if next_window:
                print(f"  Next: {next_window.get('name', 'unknown')} at {next_window.get('start_time', 'unknown')}")
        print()
    
    # Print timing stats
    stats = timing_optimizer.get_timing_stats()
    print("Timing Statistics:")
    print(f"  Total Attempts: {stats['total_entry_attempts']}")
    print(f"  Acceptance Rate: {stats['acceptance_rate_pct']:.1f}%")