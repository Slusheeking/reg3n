#!/usr/bin/env python3

import os
import time
from datetime import datetime

# =============================================================================
# HARDCODED SYSTEM LOGGER FOR MAXIMUM SPEED - NO EXTERNAL DEPENDENCIES
# =============================================================================

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # ERROR
    YELLOW = '\033[93m'   # WARNING
    BLUE = '\033[94m'     # DEBUG
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # Reset to default

class SystemLogger:
    """Ultra-fast system logger with hardcoded configuration for maximum HFT performance"""
    
    def __init__(self, log_file='system_log.log', name='system_logger'):
        self.name = name
        self.log_file = log_file
        
        # Hardcoded log directory
        self.log_dir = 'logs'
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Hardcoded log path
        self.log_path = os.path.join(self.log_dir, log_file)
        
        # Color mapping for different log levels
        self.color_map = {
            'ERROR': Colors.RED,
            'WARNING': Colors.YELLOW,
            'DEBUG': Colors.BLUE,
            'INFO': Colors.WHITE,
            'CRITICAL': Colors.RED
        }
        
        # Initialize with hardcoded settings for maximum speed
        self._init_logger()
    
    def _init_logger(self):
        """Initialize logger with hardcoded settings for maximum performance"""
        # No complex logging setup - direct file and console output for speed
        pass
    
    def _format_message(self, level: str, message: str, colored: bool = True) -> str:
        """Format log message with hardcoded format for maximum speed"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if colored:
            color = self.color_map.get(level, Colors.WHITE)
            return f"[{timestamp}] - {color}{level}{Colors.RESET} - [{self.name}]: {message}"
        else:
            return f"[{timestamp}] - {level} - [{self.name}]: {message}"
    
    def _write_log(self, level: str, message: str):
        """Write log message to both file and console for maximum speed"""
        # Console output with colors
        colored_message = self._format_message(level, message, colored=True)
        print(colored_message)
        
        # File output without colors
        plain_message = self._format_message(level, message, colored=False)
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(plain_message + '\n')
        except Exception:
            # Fail silently for maximum speed - don't let logging slow down trading
            pass
    
    def info(self, message: str, extra: dict = None):
        """Log info message with hardcoded format"""
        if extra:
            message = f"{message} | {extra}"
        self._write_log("INFO", str(message))
    
    def warning(self, message: str, extra: dict = None):
        """Log warning message with hardcoded format"""
        if extra:
            message = f"{message} | {extra}"
        self._write_log("WARNING", str(message))
    
    def error(self, message: str, extra: dict = None):
        """Log error message with hardcoded format"""
        if extra:
            message = f"{message} | {extra}"
        self._write_log("ERROR", str(message))
    
    def debug(self, message: str, extra: dict = None):
        """Log debug message with hardcoded format"""
        if extra:
            message = f"{message} | {extra}"
        self._write_log("DEBUG", str(message))
    
    def critical(self, message: str, extra: dict = None):
        """Log critical message with hardcoded format"""
        if extra:
            message = f"{message} | {extra}"
        self._write_log("CRITICAL", str(message))
    
    # Additional methods for trading system compatibility
    def startup(self, data: dict):
        """Log startup information"""
        self.info(f"STARTUP: {data}")
    
    def log_data_flow(self, operation: str, status: str, data_size: int = 0):
        """Log data flow information"""
        self.info(f"DATA_FLOW: {operation} -> {status} (size: {data_size})")
    
    def connection(self, status: str, data: dict):
        """Log connection information"""
        self.info(f"CONNECTION: {status} -> {data}")
    
    def performance(self, metrics: dict):
        """Log performance metrics"""
        self.info(f"PERFORMANCE: {metrics}")

# Hardcoded factory function for maximum speed
def get_system_logger(name: str = "system_logger") -> SystemLogger:
    """Get system logger instance with hardcoded configuration"""
    return SystemLogger(name=name)

# Example usage (for testing purposes, can be removed later)
if __name__ == "__main__":
    logger = SystemLogger()
    logger.info("SystemLogger initialized with hardcoded configuration.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    logger.critical("This is a critical message.")
    
    # Test additional methods
    logger.startup({"component": "test", "status": "initialized"})
    logger.log_data_flow("test_operation", "completed", 100)
    logger.connection("established", {"type": "websocket", "endpoint": "test"})
    logger.performance({"latency_ms": 0.5, "throughput": 1000})