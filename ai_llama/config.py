"""
Configuration for AI-Enhanced Trading System
Optimized for NVIDIA GH200 Grace Hopper ARM64 Architecture
"""
import os
import platform
import subprocess
from typing import Dict, Any, Optional

def detect_gh200() -> Dict[str, bool]:
    """Detect NVIDIA GH200 Grace Hopper Superchip"""
    is_arm64 = platform.machine().lower() in ['aarch64', 'arm64']
    is_grace = False
    has_h100 = False
    large_memory = False
    
    if is_arm64:
        # Check for NVIDIA Grace CPU
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            is_grace = 'NVIDIA' in cpu_info or 'Grace' in cpu_info
        except:
            pass
        
        # Check for H100 GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, check=True)
            gpu_names = result.stdout.strip()
            has_h100 = 'H100' in gpu_names
        except:
            pass
        
        # Check memory (>400GB indicates GH200)
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb // (1024 * 1024)
                        large_memory = mem_gb > 400
                        break
        except:
            pass
    
    return {
        'is_arm64': is_arm64,
        'is_grace': is_grace,
        'has_h100': has_h100,
        'large_memory': large_memory,
        'is_gh200': is_grace and has_h100 and large_memory
    }

# Architecture Detection
ARCH_INFO = detect_gh200()
IS_GH200 = ARCH_INFO['is_gh200']
IS_ARM64 = ARCH_INFO['is_arm64']

print(f"🔍 Architecture Detection:")
print(f"   ARM64: {IS_ARM64}")
print(f"   GH200: {IS_GH200}")
if IS_GH200:
    print(f"   🔥 NVIDIA GH200 Grace Hopper detected - Enabling optimizations!")

# API Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# GH200 Architecture Configuration
GH200_CONFIG: Dict[str, Any] = {
    'detected': IS_GH200,
    'arm64': IS_ARM64,
    'optimizations_enabled': IS_GH200,
    'multi_model_parallel': IS_GH200,
    'large_memory_mode': ARCH_INFO.get('large_memory', False),
    'gpu_acceleration': ARCH_INFO.get('has_h100', False),
    
    # Performance settings based on architecture
    'omp_threads': 144 if IS_GH200 else (8 if IS_ARM64 else 4),
    'max_workers': 32 if IS_GH200 else (8 if IS_ARM64 else 4),
    'batch_size_multiplier': 4 if IS_GH200 else (2 if IS_ARM64 else 1),
    'cache_size_multiplier': 10 if IS_GH200 else (2 if IS_ARM64 else 1),
    
    # Memory settings
    'unified_memory_gb': 480 if IS_GH200 else (64 if IS_ARM64 else 32),
    'model_cache_size_gb': 50 if IS_GH200 else (8 if IS_ARM64 else 4),
    'feature_cache_size_mb': 5000 if IS_GH200 else (1000 if IS_ARM64 else 500),
}

# Model Configuration
MODELS: Dict[str, Any] = {
    'lag_llama': {
        'context_length': 64,
        'prediction_length': 5,
        'update_frequency': 30,  # seconds
        'confidence_threshold': 0.7,
        'batch_size': 32,
        'cache_ttl': 300  # 5 minutes
    },
    'fast_models': {
        'momentum_window': 20,
        'volatility_window': 60,
        'rsi_period': 14,
        'bollinger_period': 20,
        'bollinger_std': 2.0
    }
}

# Strategy Configuration
STRATEGIES: Dict[str, Any] = {
    'gap_and_go': {
        'enabled': True,
        'min_gap_percent': 0.02,
        'max_gap_percent': 0.10,
        'ai_quality_threshold': 0.6,
        'volume_threshold': 100000,
        'premarket_start': '04:00',
        'market_open': '09:30',
        'trading_window': 60  # minutes after open
    },
    'orb': {
        'enabled': True,
        'base_range_minutes': 5,
        'dynamic_range': True,
        'min_range_minutes': 3,
        'max_range_minutes': 10,
        'breakout_threshold': 0.01,
        'volume_confirmation': True,
        'trading_window': 30  # minutes after open
    },
    'vol_mean_reversion': {
        'enabled': True,
        'volatility_threshold': 2.0,
        'reversion_probability_threshold': 0.7,
        'max_position_time': 300,  # seconds
        'stop_loss_percent': 0.02
    }
}

# Risk Management
RISK_MANAGEMENT: Dict[str, Any] = {
    'max_position_size': 0.02,  # 2% of portfolio per position
    'max_total_exposure': 0.20,  # 20% total exposure
    'max_daily_loss': 0.05,  # 5% max daily loss
    'var_confidence': 0.95,
    'correlation_threshold': 0.7,
    'uncertainty_scaling': True
}

# Performance Configuration
PERFORMANCE: Dict[str, Any] = {
    'target_latency_ms': 8,
    'cache_warmup': True,
    'vectorization': True,
    'numba_compilation': True,
    'batch_processing': True,
    'async_execution': True
}

# Data Configuration
DATA: Dict[str, Any] = {
    'symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
    ],
    'timeframes': ['1min', '5min'],
    'history_days': 30,
    'real_time': True,
    'news_enabled': True,
    'level2_data': False
}

# Logging Configuration
LOGGING: Dict[str, Any] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'ai_trading.log',
    'max_size': '100MB',
    'backup_count': 5,
    'structured': True
}

# Monitoring Configuration
MONITORING: Dict[str, Any] = {
    'metrics_enabled': True,
    'latency_percentiles': [50, 90, 95, 99],
    'alert_thresholds': {
        'latency_p95_ms': 15,
        'error_rate_percent': 1.0,
        'model_accuracy_drop': 0.05
    },
    'dashboard_update_seconds': 5
}
