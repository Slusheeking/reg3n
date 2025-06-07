"""
YAML Configuration Package
Configuration management and YAML utilities for the trading system
"""

import yaml
import os
from typing import Dict, Any, Optional

__all__ = [
    'ConfigManager',
    'YAMLLoader',
    'ConfigValidator'
]

class YAMLLoader:
    """
    YAML file loader with error handling
    """
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML file and return parsed content"""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {file_path}: {str(e)}")
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
        """Save data to YAML file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                yaml.safe_dump(data, file, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            raise IOError(f"Failed to save YAML file {file_path}: {str(e)}")

class ConfigValidator:
    """
    Configuration validator for YAML configs
    """
    
    @staticmethod
    def validate_data_pipeline_config(config: Dict[str, Any]) -> bool:
        """Validate data pipeline configuration"""
        required_fields = ['polygon_api', 'filters', 'processing']
        return all(field in config for field in required_fields)
    
    @staticmethod
    def validate_trade_pipeline_config(config: Dict[str, Any]) -> bool:
        """Validate trade pipeline configuration"""
        required_fields = ['alpaca_api', 'execution', 'risk_management']
        return all(field in config for field in required_fields)
    
    @staticmethod
    def validate_adaptive_filter_config(config: Dict[str, Any]) -> bool:
        """Validate adaptive filter configuration"""
        required_fields = ['market_conditions', 'filters', 'thresholds']
        return all(field in config for field in required_fields)

class ConfigManager:
    """
    Central configuration manager for all YAML configs
    """
    
    def __init__(self, config_dir: str = "yaml"):
        self.config_dir = config_dir
        self.configs = {}
        self.loader = YAMLLoader()
        self.validator = ConfigValidator()
    
    def load_config(self, config_name: str, validate: bool = True) -> Dict[str, Any]:
        """Load a specific configuration file"""
        file_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        config = self.loader.load_yaml(file_path)
        
        if validate:
            self._validate_config(config_name, config)
        
        self.configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get a loaded configuration"""
        return self.configs.get(config_name)
    
    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """Reload a configuration from file"""
        return self.load_config(config_name, validate=True)
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        file_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        success = self.loader.save_yaml(config_data, file_path)
        
        if success:
            self.configs[config_name] = config_data
        
        return success
    
    def _validate_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Validate configuration based on its type"""
        validation_map = {
            'data_pipeline': self.validator.validate_data_pipeline_config,
            'trade_pipeline': self.validator.validate_trade_pipeline_config,
            'adaptive_filter_config': self.validator.validate_adaptive_filter_config
        }
        
        validator_func = validation_map.get(config_name)
        if validator_func and not validator_func(config):
            raise ValueError(f"Invalid configuration format for {config_name}")
    
    def list_configs(self) -> list:
        """List all available configuration files"""
        if not os.path.exists(self.config_dir):
            return []
        
        yaml_files = [f[:-5] for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
        return yaml_files
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self.configs.copy()