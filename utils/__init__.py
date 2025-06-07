"""
Utilities Package
Common utilities and helper functions for the trading system
"""

from .system_logger import (
    SystemLogger,
    ComponentLogger,
    get_logger,
    get_websocket_logger,
    get_rest_api_logger,
    get_system_logger
)

__all__ = [
    'SystemLogger',
    'ComponentLogger',
    'get_logger',
    'get_websocket_logger',
    'get_rest_api_logger',
    'get_system_logger',
    'UtilityManager'
]

class UtilityManager:
    """
    Central utility manager for common system operations
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = SystemLogger(level=log_level)
        
    def get_logger(self) -> SystemLogger:
        """Get the system logger instance"""
        return self.logger
    
    def log_info(self, message: str, component: str = "SYSTEM"):
        """Log an info message"""
        self.logger.info(message, component=component)
    
    def log_error(self, message: str, component: str = "SYSTEM", error: Exception = None):
        """Log an error message"""
        self.logger.error(message, component=component, error=error)
    
    def log_warning(self, message: str, component: str = "SYSTEM"):
        """Log a warning message"""
        self.logger.warning(message, component=component)
    
    def log_debug(self, message: str, component: str = "SYSTEM"):
        """Log a debug message"""
        self.logger.debug(message, component=component)