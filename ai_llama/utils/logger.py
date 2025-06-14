"""
System Logger

Centralized logging system for the AI trading platform.
Features:
- Multi-level logging (trade, performance, error, debug)
- File rotation and archiving
- Real-time log monitoring
- Trade audit trails
- Performance metrics logging
- Error tracking and alerting
"""

import logging
import logging.handlers
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from collections import deque, defaultdict

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: str
    logger: str
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

class TradingFormatter(logging.Formatter):
    """Custom formatter for trading logs"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created)
        
        # Color coding for console output
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[92m',     # Green
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',    # Red
            'CRITICAL': '\033[95m'  # Magenta
        }
        reset_color = '\033[0m'
        
        # Get color for level
        color = colors.get(record.levelname, '')
        
        # Format message based on logger type
        if hasattr(record, 'trade_data'):
            # Trade-specific formatting
            trade_data = record.trade_data
            formatted = f"{color}[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {record.levelname:<8}{reset_color} "
            formatted += f"💰 TRADE: {trade_data.get('symbol', 'N/A')} {trade_data.get('side', 'N/A')} "
            formatted += f"{trade_data.get('quantity', 0)} @ ${trade_data.get('price', 0):.2f} | {record.getMessage()}"
            
        elif hasattr(record, 'performance_data'):
            # Performance-specific formatting
            perf_data = record.performance_data
            formatted = f"{color}[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {record.levelname:<8}{reset_color} "
            formatted += f"📊 PERF: {perf_data.get('metric', 'N/A')}={perf_data.get('value', 0)} "
            formatted += f"({perf_data.get('unit', '')}) | {record.getMessage()}"
            
        elif hasattr(record, 'market_data'):
            # Market data formatting
            market_data = record.market_data
            formatted = f"{color}[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {record.levelname:<8}{reset_color} "
            formatted += f"📈 MARKET: {market_data.get('symbol', 'N/A')} ${market_data.get('price', 0):.2f} "
            formatted += f"| {record.getMessage()}"
            
        else:
            # Standard formatting
            formatted = f"{color}[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {record.levelname:<8}{reset_color} "
            formatted += f"{record.name:<20} | {record.getMessage()}"
            
            # Add file/line info for errors
            if record.levelno >= logging.ERROR:
                formatted += f" [{record.filename}:{record.lineno}]"
        
        return formatted

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            extra=getattr(record, 'extra_data', None)
        )
        return log_entry.to_json()

class SystemLogger:
    """
    Centralized system logger for the trading platform
    
    Features:
    - Multiple log levels and categories
    - File rotation and archiving
    - Real-time monitoring
    - Trade audit trails
    - Performance metrics
    - Error tracking
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 10,
                 enable_console: bool = True,
                 enable_json: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_json = enable_json
        
        # Performance tracking
        self.performance_metrics = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.log_counts = defaultdict(int)
        
        # Create loggers
        self.loggers = {}
        self._setup_loggers()
        
        # Log monitoring
        self.recent_logs = deque(maxlen=1000)
        self.error_logs = deque(maxlen=100)
        
    def _setup_loggers(self):
        """Set up different logger categories"""
        
        # Main system logger
        self.system_logger = self._create_logger(
            'system', 
            'system.log',
            console=self.enable_console
        )
        
        # Trade execution logger
        self.trade_logger = self._create_logger(
            'trades',
            'trades.log',
            console=True  # Always show trades on console
        )
        
        # Market data logger
        self.market_logger = self._create_logger(
            'market_data',
            'market_data.log',
            console=False  # Too verbose for console
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'performance',
            'performance.log',
            console=False
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'errors',
            'errors.log',
            console=True,
            level=logging.ERROR
        )
        
        # Strategy logger
        self.strategy_logger = self._create_logger(
            'strategies',
            'strategies.log',
            console=True
        )
        
        # AI model logger
        self.ai_logger = self._create_logger(
            'ai_models',
            'ai_models.log',
            console=False
        )
        
        # Store loggers for easy access
        self.loggers = {
            'system': self.system_logger,
            'trades': self.trade_logger,
            'market': self.market_logger,
            'performance': self.performance_logger,
            'errors': self.error_logger,
            'strategies': self.strategy_logger,
            'ai': self.ai_logger
        }
    
    def _create_logger(self, 
                      name: str, 
                      filename: str,
                      console: bool = True,
                      level: int = None) -> logging.Logger:
        """Create a logger with file and console handlers"""
        
        logger = logging.getLogger(f"ai_llama.{name}")
        logger.setLevel(level or self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_path = self.log_dir / filename
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(level or self.log_level)
        file_handler.setFormatter(TradingFormatter())
        logger.addHandler(file_handler)
        
        # JSON file handler for structured logs
        if self.enable_json:
            json_path = self.log_dir / f"{name}_structured.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            json_handler.setLevel(level or self.log_level)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
        
        # Console handler
        if console and self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level or self.log_level)
            console_handler.setFormatter(TradingFormatter())
            logger.addHandler(console_handler)
        
        # Add custom filter for log monitoring
        logger.addFilter(self._log_filter)
        
        return logger
    
    def _log_filter(self, record):
        """Custom filter to track logs for monitoring"""
        
        # Track log counts
        self.log_counts[record.levelname] += 1
        
        # Store recent logs
        self.recent_logs.append({
            'timestamp': record.created,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        })
        
        # Store error logs separately
        if record.levelno >= logging.ERROR:
            self.error_counts[record.name] += 1
            self.error_logs.append({
                'timestamp': record.created,
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'filename': record.filename,
                'lineno': record.lineno
            })
        
        return True
    
    # Convenience methods for different log types
    def log_trade(self, message: str, trade_data: Dict[str, Any], level: str = "INFO"):
        """Log trade-related events"""
        logger = self.trade_logger
        log_level = getattr(logging, level.upper())
        
        # Create log record with trade data
        record = logger.makeRecord(
            logger.name, log_level, __file__, 0, message, (), None
        )
        record.trade_data = trade_data
        logger.handle(record)
    
    def log_market_data(self, message: str, market_data: Dict[str, Any], level: str = "DEBUG"):
        """Log market data events"""
        logger = self.market_logger
        log_level = getattr(logging, level.upper())
        
        record = logger.makeRecord(
            logger.name, log_level, __file__, 0, message, (), None
        )
        record.market_data = market_data
        logger.handle(record)
    
    def log_performance(self, metric: str, value: float, unit: str = "", message: str = ""):
        """Log performance metrics"""
        logger = self.performance_logger
        
        # Store for tracking
        self.performance_metrics[metric].append({
            'timestamp': time.time(),
            'value': value,
            'unit': unit
        })
        
        # Keep only recent metrics
        if len(self.performance_metrics[metric]) > 1000:
            self.performance_metrics[metric].popleft()
        
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.performance_data = {
            'metric': metric,
            'value': value,
            'unit': unit
        }
        logger.handle(record)
    
    def log_strategy(self, strategy_name: str, message: str, level: str = "INFO", **kwargs):
        """Log strategy events"""
        logger = self.strategy_logger
        log_level = getattr(logging, level.upper())
        
        formatted_message = f"[{strategy_name}] {message}"
        if kwargs:
            formatted_message += f" | Data: {kwargs}"
        
        getattr(logger, level.lower())(formatted_message)
    
    def log_ai_model(self, model_name: str, message: str, level: str = "INFO", **kwargs):
        """Log AI model events"""
        logger = self.ai_logger
        log_level = getattr(logging, level.upper())
        
        formatted_message = f"[{model_name}] {message}"
        if kwargs:
            formatted_message += f" | Data: {kwargs}"
        
        getattr(logger, level.lower())(formatted_message)
    
    def get_logger(self, category: str) -> logging.Logger:
        """Get logger by category"""
        return self.loggers.get(category, self.system_logger)
    
    def get_recent_logs(self, count: int = 50) -> List[Dict]:
        """Get recent log entries"""
        return list(self.recent_logs)[-count:]
    
    def get_error_logs(self, count: int = 20) -> List[Dict]:
        """Get recent error logs"""
        return list(self.error_logs)[-count:]
    
    def get_performance_stats(self, metric: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if metric:
            if metric not in self.performance_metrics:
                return {}
            
            values = [m['value'] for m in self.performance_metrics[metric]]
            return {
                'metric': metric,
                'count': len(values),
                'latest': values[-1] if values else 0,
                'avg': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            }
        else:
            return {
                metric: self.get_performance_stats(metric)
                for metric in self.performance_metrics.keys()
            }
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'log_counts': dict(self.log_counts),
            'error_counts': dict(self.error_counts),
            'recent_logs_count': len(self.recent_logs),
            'error_logs_count': len(self.error_logs),
            'performance_metrics_count': len(self.performance_metrics),
            'loggers': list(self.loggers.keys())
        }
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.system_logger.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    self.system_logger.error(f"Failed to delete log file {log_file}: {e}")

# Global logger instance
_system_logger = None

def get_logger(category: str = "system") -> logging.Logger:
    """Get logger instance by category"""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    return _system_logger.get_logger(category)

def setup_logging(log_dir: str = "logs", 
                 log_level: str = "INFO",
                 enable_console: bool = True) -> SystemLogger:
    """Set up global logging system"""
    global _system_logger
    _system_logger = SystemLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console
    )
    return _system_logger

def log_trade(message: str, **trade_data):
    """Quick trade logging function"""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    _system_logger.log_trade(message, trade_data)

def log_performance(metric: str, value: float, unit: str = "", message: str = ""):
    """Quick performance logging function"""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    _system_logger.log_performance(metric, value, unit, message)

def log_strategy(strategy_name: str, message: str, level: str = "INFO", **kwargs):
    """Quick strategy logging function"""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    _system_logger.log_strategy(strategy_name, message, level, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logger_system = setup_logging(log_level="DEBUG")
    
    # Test different log types
    print("🔧 Testing system logger...")
    
    # System logs
    system_logger = get_logger("system")
    system_logger.info("System initialized successfully")
    system_logger.warning("This is a warning message")
    system_logger.error("This is an error message")
    
    # Trade logs
    log_trade("Order executed successfully", 
              symbol="AAPL", side="BUY", quantity=100, price=150.25)
    
    # Performance logs
    log_performance("latency_ms", 2.5, "ms", "Data pipeline latency measured")
    log_performance("throughput", 1500, "msg/s", "Message processing rate")
    
    # Strategy logs
    log_strategy("GapAndGo", "Position opened", level="INFO", 
                entry_price=145.30, size=50, confidence=0.85)
    
    # AI model logs
    ai_logger = get_logger("ai")
    ai_logger.info("Model prediction completed: 85% confidence")
    
    # Show statistics
    print("\n📊 Logger Statistics:")
    stats = logger_system.get_log_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n📈 Performance Stats:")
    perf_stats = logger_system.get_performance_stats()
    for metric, stats in perf_stats.items():
        print(f"   {metric}: {stats}")
    
    print("\n📋 Recent Logs:")
    recent = logger_system.get_recent_logs(5)
    for log in recent:
        timestamp = datetime.fromtimestamp(log['timestamp'])
        print(f"   [{timestamp.strftime('%H:%M:%S')}] {log['level']}: {log['message']}")
