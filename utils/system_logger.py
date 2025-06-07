"""
Unified System Logging for reg3n Trading System
Provides centralized logging configuration and utilities for all components
"""

import logging
import logging.handlers
import os
import sys
import yaml
import inspect
import traceback
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[97m',     # White
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        if self.use_colors and record.levelname in self.COLORS:
            # Add color to the entire log message
            colored_record = logging.makeLogRecord(record.__dict__)
            colored_record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            colored_record.name = f"{self.COLORS[record.levelname]}{record.name}{self.COLORS['RESET']}"
            colored_record.msg = f"{self.COLORS[record.levelname]}{record.msg}{self.COLORS['RESET']}"
            return super().format(colored_record)
        return super().format(record)


def get_caller_info(skip_frames=2):
    """Get detailed information about the caller including file, function, and line number"""
    try:
        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is None:
                break
            frame = frame.f_back
        
        if frame is None:
            return {
                'file': 'unknown',
                'function': 'unknown',
                'line': 0,
                'module': 'unknown'
            }
        
        filename = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Get relative path from project root
        try:
            relative_path = os.path.relpath(filename, os.path.dirname(os.path.dirname(__file__)))
        except ValueError:
            relative_path = os.path.basename(filename)
        
        # Get module name
        module_name = os.path.splitext(os.path.basename(filename))[0]
        
        return {
            'file': relative_path,
            'function': function_name,
            'line': line_number,
            'module': module_name,
            'full_path': filename
        }
    except Exception:
        return {
            'file': 'unknown',
            'function': 'unknown',
            'line': 0,
            'module': 'unknown'
        }


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'utils.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}


class SystemLogger:
    """
    Unified system logger with YAML configuration and colored output
    Handles logging for WebSocket, REST API, and trading components
    """
    
    def __init__(self, name: str = "reg3n_system"):
        self.config = load_config()
        self.log_config = self.config.get('logging', {})
        self.name = name
        self.logger = None
        self.logging_enabled = self._check_logging_enabled()
        
        if self.logging_enabled:
            self._setup_logging()
        else:
            self._setup_null_logger()
    
    def _check_logging_enabled(self) -> bool:
        """Check if logging is enabled for this component"""
        # Check master logging switch
        if not self.log_config.get('enabled', True):
            return False
        
        # Check component-specific logging
        components = self.log_config.get('components', {})
        component_name = self.name.replace('reg3n_', '')
        
        return components.get(component_name, True)
    
    def _setup_null_logger(self):
        """Setup a null logger that doesn't output anything"""
        self.logger = logging.getLogger(self.name)
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
    
    def _setup_logging(self):
        """Setup logging configuration from YAML with colored output"""
        # Get logging configuration
        log_level = self.log_config.get('level', 'INFO')
        log_format = self.log_config.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_logging = self.log_config.get('file_logging', True)
        console_logging = self.log_config.get('console_logging', True)
        colored_output = self.log_config.get('colored_output', True)
        log_rotation = self.log_config.get('log_rotation', True)
        max_log_size_mb = self.log_config.get('max_log_size_mb', 100)
        backup_count = self.log_config.get('backup_count', 5)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colored output
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            if colored_output:
                # Use colored formatter for console
                console_formatter = ColoredFormatter(log_format, use_colors=True)
            else:
                # Use regular formatter for console
                console_formatter = logging.Formatter(log_format)
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (always without colors)
        if file_logging:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"{self.name}.log"
            
            if log_rotation:
                # Rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_log_size_mb * 1024 * 1024,  # Convert MB to bytes
                    backupCount=backup_count
                )
            else:
                # Regular file handler
                file_handler = logging.FileHandler(log_file)
            
            file_handler.setLevel(getattr(logging, log_level.upper()))
            # File logs should never have colors
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger
    
    def log_system_startup(self, component: str, config_summary: Dict[str, Any]):
        """Log system startup information"""
        self.logger.info(f"=== {component} STARTUP ===")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"Component: {component}")
        
        for key, value in config_summary.items():
            self.logger.info(f"Config - {key}: {value}")
        
        self.logger.info(f"=== {component} STARTUP COMPLETE ===")
    
    def log_performance_metrics(self, component: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.logger.info(f"=== {component} PERFORMANCE METRICS ===")
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
            else:
                self.logger.info(f"{metric_name}: {metric_value}")
    
    def log_error_with_context(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with additional context and precise location tracking"""
        caller_info = get_caller_info(skip_frames=3)
        
        self.logger.error(f"=== {component} ERROR ===")
        self.logger.error(f"Location: {caller_info['file']}:{caller_info['line']} in {caller_info['function']}()")
        self.logger.error(f"Module: {caller_info['module']}")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        
        if context:
            self.logger.error("Error Context:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")
        
        # Log stack trace
        self.logger.error("Stack Trace:")
        self.logger.error(traceback.format_exc())
    
    def log_connection_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """Log connection status changes"""
        self.logger.info(f"{component} Connection Status: {status}")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_data_processing(self, component: str, symbol: str, data_type: str, count: int):
        """Log data processing events"""
        self.logger.debug(f"{component} - Processed {count} {data_type} records for {symbol}")
    
    def log_rate_limiting(self, component: str, action: str, delay_ms: float):
        """Log rate limiting events"""
        self.logger.debug(f"{component} - Rate limiting: {action}, delay: {delay_ms:.2f}ms")
    
    def log_config_change(self, component: str, setting: str, old_value: Any, new_value: Any):
        """Log configuration changes"""
        self.logger.info(f"{component} - Config change: {setting} changed from {old_value} to {new_value}")


class ComponentLogger:
    """
    Component-specific logger wrapper with enable/disable controls
    Provides easy access to system logging for individual components
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.system_logger = SystemLogger(f"reg3n_{component_name}")
        self.logger = self.system_logger.get_logger()
        self.logging_enabled = self.system_logger.logging_enabled
    
    def _log_if_enabled(self, log_func, *args, **kwargs):
        """Only log if logging is enabled for this component"""
        if self.logging_enabled:
            return log_func(*args, **kwargs)
        return None
    
    def startup(self, config_summary: Dict[str, Any]):
        """Log component startup"""
        if self.logging_enabled:
            self.system_logger.log_system_startup(self.component_name, config_summary)
    
    def performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        if self.logging_enabled:
            self.system_logger.log_performance_metrics(self.component_name, metrics)
    
    def error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context and precise location tracking"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            
            # Add location info to context
            enhanced_context = context or {}
            enhanced_context.update({
                'error_location': {
                    'file': caller_info['file'],
                    'function': caller_info['function'],
                    'line': caller_info['line'],
                    'module': caller_info['module']
                }
            })
            
            self.system_logger.log_error_with_context(self.component_name, error, enhanced_context)
    
    def connection(self, status: str, details: Dict[str, Any] = None):
        """Log connection status"""
        if self.logging_enabled:
            self.system_logger.log_connection_status(self.component_name, status, details)
    
    def data_processing(self, symbol: str, data_type: str, count: int):
        """Log data processing"""
        if self.logging_enabled:
            self.system_logger.log_data_processing(self.component_name, symbol, data_type, count)
    
    def rate_limiting(self, action: str, delay_ms: float):
        """Log rate limiting"""
        if self.logging_enabled:
            self.system_logger.log_rate_limiting(self.component_name, action, delay_ms)
    
    def config_change(self, setting: str, old_value: Any, new_value: Any):
        """Log configuration changes"""
        if self.logging_enabled:
            self.system_logger.log_config_change(self.component_name, setting, old_value, new_value)
    
    def info(self, message: str):
        """Log info message with location tracking"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.info(f"{self.component_name} - {message} [{caller_info['file']}:{caller_info['line']}]")
    
    def debug(self, message: str):
        """Log debug message with location tracking"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.debug(f"{self.component_name} - {message} [{caller_info['file']}:{caller_info['line']}]")
    
    def warning(self, message: str):
        """Log warning message with location tracking"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.warning(f"{self.component_name} - {message} [{caller_info['file']}:{caller_info['line']}]")
    
    def critical(self, message: str):
        """Log critical message with location tracking"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.critical(f"{self.component_name} - {message} [{caller_info['file']}:{caller_info['line']}]")
    
    # Trading-specific logging methods
    def log_trading_signal(self, signal_type: str, symbol: str, signal_data: dict):
        """Log trading signal processing"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.info(f"{self.component_name} - TRADING_SIGNAL {signal_type} {symbol}: {signal_data} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_position_calculation(self, symbol: str, calculation_type: str, inputs: dict, outputs: dict):
        """Log position size calculations"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            self.logger.info(f"{self.component_name} - POSITION_CALC {symbol} {calculation_type}: inputs={inputs} -> outputs={outputs} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_order_submission(self, symbol: str, order_type: str, order_data: dict, result: dict = None):
        """Log order submissions"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            result_str = f" -> {result}" if result else ""
            self.logger.info(f"{self.component_name} - ORDER_SUBMIT {symbol} {order_type}: {order_data}{result_str} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_filter_decision(self, filter_name: str, symbol: str, decision: bool, criteria: dict):
        """Log filter decisions"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            decision_str = "PASS" if decision else "REJECT"
            self.logger.info(f"{self.component_name} - FILTER {filter_name} {symbol}: {decision_str} | {criteria} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_api_request(self, method: str, url: str, params: dict = None):
        """Log API requests"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            params_str = f" params={params}" if params else ""
            self.logger.debug(f"{self.component_name} - API_REQUEST {method} {url}{params_str} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_api_response(self, status_code: int, response_size: int = None, response_time: float = None):
        """Log API responses"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            size_str = f" ({response_size} bytes)" if response_size else ""
            time_str = f" [{response_time:.3f}ms]" if response_time else ""
            self.logger.debug(f"{self.component_name} - API_RESPONSE {status_code}{size_str}{time_str} [{caller_info['file']}:{caller_info['line']}]")
    
    def log_data_flow(self, operation: str, data_type: str, data_size: int = None):
        """Log data flow operations"""
        if self.logging_enabled:
            caller_info = get_caller_info(skip_frames=2)
            size_str = f" ({data_size} items)" if data_size else ""
            self.logger.debug(f"{self.component_name} - DATA_FLOW {operation}: {data_type}{size_str} [{caller_info['file']}:{caller_info['line']}]")


# Convenience functions for quick access
def get_logger(component_name: str) -> ComponentLogger:
    """Get a component logger"""
    return ComponentLogger(component_name)


def get_websocket_logger() -> ComponentLogger:
    """Get WebSocket component logger"""
    return ComponentLogger("websocket")


def get_rest_api_logger() -> ComponentLogger:
    """Get REST API component logger"""
    return ComponentLogger("rest_api")




def get_system_logger(component_name: str = "system") -> ComponentLogger:
    """Get system-wide logger with optional component name"""
    return ComponentLogger(component_name)


# Example usage
if __name__ == "__main__":
    # Test the logging system
    logger = get_system_logger()
    
    logger.startup({
        "version": "1.0.0",
        "environment": "development",
        "a100_optimized": True
    })
    
    logger.info("System logging test completed successfully")
    
    # Test component loggers
    ws_logger = get_websocket_logger()
    ws_logger.connection("connected", {"symbols": 5000, "batch_size": 1000})
    
    api_logger = get_rest_api_logger()
    api_logger.performance({
        "requests_per_second": 95.5,
        "avg_response_time_ms": 125.3,
        "success_rate": 99.8
    })