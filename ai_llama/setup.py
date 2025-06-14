"""
AI-Enhanced Trading System - Complete Setup Script

This script automatically:
1. Installs all Python dependencies
2. Clones all foundation model repositories  
3. Downloads all pretrained models and LLMs
4. Sets up the environment and configuration
5. Tests the system components

Usage:
    python setup.py                    # Full installation
    python setup.py --download-models  # Download models only
    python setup.py --clone-repos      # Clone repositories only
    python setup.py --test             # Test system only
"""

import subprocess
import sys
import os
import argparse
import shutil
import json
from pathlib import Path
import urllib.request
import time

class SetupManager:
    """Comprehensive setup manager for AI trading system"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.ai_llama_dir = Path(__file__).parent
        self.models_dir = self.ai_llama_dir / "models"
        
        # Repository URLs
        self.repositories = {
            'lag-llama': 'https://github.com/time-series-foundation-models/lag-llama.git',
            'chronos-forecasting': 'https://github.com/amazon-science/chronos-forecasting.git', 
            'timesfm': 'https://github.com/google-research/timesfm.git'
        }
        
        # HuggingFace models to download
        self.hf_models = {
            'chronos': [
                'amazon/chronos-t5-small'  # Optimal balance for trading: speed + accuracy
            ],
            'timesfm': [
                'google/timesfm-1.0-200m'
            ]
        }
        
        print("🚀 AI-Enhanced Trading System Setup Manager")
        print("=" * 50)
        
        # Detect system architecture
        import platform
        self.arch = platform.machine().lower()
        self.is_arm64 = self.arch in ['aarch64', 'arm64']
        self.is_gh200 = self.is_arm64 and self._detect_gh200()
        
        if self.is_gh200:
            print("🔥 NVIDIA GH200 detected - Enabling ultra-performance mode!")
        elif self.is_arm64:
            print("🔧 ARM64 architecture detected - Using optimized builds")
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Check git
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            print("✅ Git available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Git not found. Please install git first.")
            return False
        
        # Check internet connection
        try:
            urllib.request.urlopen('https://google.com', timeout=5)
            print("✅ Internet connection available")
        except:
            print("❌ No internet connection")
            return False
        
        # GH200 specific checks
        if self.is_gh200:
            self._check_gh200_capabilities()
        
        return True
    
    def _detect_gh200(self):
        """Detect if running on GH200"""
        try:
            # Check for NVIDIA Grace CPU + H100 GPU combination
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            
            # Check for Grace CPU indicators
            is_grace = 'NVIDIA' in cpu_info or 'Grace' in cpu_info
            
            # Check for H100 GPU
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, check=True)
                gpu_names = result.stdout.strip()
                has_h100 = 'H100' in gpu_names
                
                return is_grace and has_h100
            except:
                return False
                
        except:
            return False
    
    def _check_gh200_capabilities(self):
        """Check GH200 specific capabilities"""
        print("🔥 GH200 System Analysis:")
        
        # Check memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb // (1024 * 1024)
                    print(f"   💾 Memory: {mem_gb}GB unified memory detected")
                    if mem_gb > 400:
                        print("   ✅ Sufficient memory for all models simultaneously")
                    break
        except:
            pass
        
        # Check GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            gpu_memory = int(result.stdout.strip())
            print(f"   🎮 GPU Memory: {gpu_memory//1024}GB H100 detected")
            print("   ✅ Ultra-high performance inference capable")
        except:
            pass
        
        # Recommend optimizations
        print("   🚀 Enabling GH200 optimizations:")
        print("   - Parallel model inference")
        print("   - Large batch processing")  
        print("   - Multi-strategy execution")
    
    def install_requirements(self):
        """Install all Python requirements"""
        print("\n📦 Installing Python dependencies...")
        
        requirements_file = self.ai_llama_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            print("   ⬆️  Upgrading pip...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL)
            
            # Install requirements
            print("   📋 Installing requirements...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            
            # Install additional packages for model downloads
            print("   🤗 Installing HuggingFace CLI...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"
            ])
            
            print("✅ All Python dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
    
    def clone_repositories(self):
        """Clone all foundation model repositories"""
        print("\n📂 Cloning foundation model repositories...")
        
        for repo_name, repo_url in self.repositories.items():
            repo_path = self.root_dir / repo_name
            
            if repo_path.exists():
                print(f"   ⚠️  {repo_name} already exists, skipping...")
                continue
            
            try:
                print(f"   📥 Cloning {repo_name}...")
                subprocess.check_call([
                    'git', 'clone', repo_url, str(repo_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                print(f"   ✅ {repo_name} cloned successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Error cloning {repo_name}: {e}")
                return False
        
        print("✅ All repositories cloned successfully!")
        return True
    
    def install_repositories(self):
        """Install the cloned repositories"""
        print("\n🔧 Installing foundation model packages...")
        
        installations = {
            'lag-llama': self.root_dir / 'lag-llama',
            'chronos-forecasting': self.root_dir / 'chronos-forecasting',
            'timesfm': self.root_dir / 'timesfm'
        }
        
        for repo_name, repo_path in installations.items():
            if not repo_path.exists():
                print(f"   ⚠️  {repo_name} not found, skipping installation...")
                continue
            
            try:
                print(f"   📦 Installing {repo_name}...")
                
                # Change to repository directory and install
                original_cwd = os.getcwd()
                os.chdir(repo_path)
                
                # Install in development mode
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-e", "."
                ], stdout=subprocess.DEVNULL)
                
                os.chdir(original_cwd)
                print(f"   ✅ {repo_name} installed successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Error installing {repo_name}: {e}")
                os.chdir(original_cwd)
                return False
        
        print("✅ All foundation model packages installed!")
        return True
    
    def download_huggingface_models(self):
        """Download all HuggingFace models"""
        print("\n🤗 Downloading HuggingFace models...")
        
        # Create models cache directory
        cache_dir = self.models_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, models in self.hf_models.items():
            print(f"   📥 Downloading {model_type} models...")
            
            for model_name in models:
                try:
                    print(f"      🔄 Downloading {model_name}...")
                    
                    # Use huggingface-cli to download
                    subprocess.check_call([
                        'huggingface-cli', 'download', model_name,
                        '--cache-dir', str(cache_dir),
                        '--local-dir', str(cache_dir / model_name.replace('/', '_'))
                    ], stdout=subprocess.DEVNULL)
                    
                    print(f"      ✅ {model_name} downloaded")
                    
                except subprocess.CalledProcessError as e:
                    print(f"      ❌ Error downloading {model_name}: {e}")
                    # Continue with other models
                    continue
        
        print("✅ HuggingFace models downloaded!")
        return True
    
    def download_lag_llama_models(self):
        """Download Lag-Llama pretrained models"""
        print("\n🦙 Setting up Lag-Llama models...")
        
        try:
            # Test Lag-Llama installation and download models
            test_script = """
import sys
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    print("✅ Lag-Llama package available")
    
    # Try to load pretrained model (this will download if needed)
    estimator = LagLlamaEstimator.from_pretrained("lag-llama")
    print("✅ Lag-Llama pretrained model loaded successfully")
    
except Exception as e:
    print(f"❌ Error with Lag-Llama: {e}")
    sys.exit(1)
"""
            
            # Write and execute test script
            test_file = self.ai_llama_dir / "test_lag_llama.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            subprocess.check_call([sys.executable, str(test_file)])
            
            # Clean up test file
            test_file.unlink()
            
            print("✅ Lag-Llama models ready!")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Error setting up Lag-Llama models")
            return False
    
    def setup_model_symlinks(self):
        """Create symbolic links for easy model access"""
        print("\n🔗 Setting up model symbolic links...")
        
        # Create symlinks to the cloned repositories
        symlinks = [
            (self.root_dir / 'lag-llama', self.models_dir / 'lag_llama'),
            (self.root_dir / 'chronos-forecasting', self.models_dir / 'chronos'),
            (self.root_dir / 'timesfm', self.models_dir / 'timesfm')
        ]
        
        for source, target in symlinks:
            if source.exists() and not target.exists():
                try:
                    # Create symlink (works on Unix-like systems)
                    target.symlink_to(source)
                    print(f"   ✅ Created symlink: {target.name} -> {source.name}")
                except OSError:
                    # Fallback: copy for Windows or if symlinks not supported
                    shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.git*'))
                    print(f"   ✅ Copied: {source.name} -> {target.name}")
        
        return True
    
    def setup_environment(self):
        """Setup environment configuration"""
        print("\n🔧 Setting up environment configuration...")
        
        # Create comprehensive .env.example
        env_example = """# AI-Enhanced Trading System Configuration
# Copy this file to .env and fill in your actual API keys

# =============================================================================
# MARKET DATA APIs
# =============================================================================

# Polygon.io (Primary data source)
POLYGON_API_KEY=your_polygon_api_key_here

# Alpaca Trading API
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading by default

# Alternative Data Sources (optional)
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
# IEX_CLOUD_API_KEY=your_iex_cloud_key
# BINANCE_API_KEY=your_binance_key
# BINANCE_SECRET_KEY=your_binance_secret

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model paths (automatically set by setup)
LAG_LLAMA_MODEL_PATH=./models/lag_llama
CHRONOS_MODEL_PATH=./models/chronos
TIMESFM_MODEL_PATH=./models/timesfm

# HuggingFace settings
HF_CACHE_DIR=./models/cache
HF_HOME=./models/cache
TRANSFORMERS_CACHE=./models/cache

        # Model preferences
        DEFAULT_CHRONOS_MODEL=amazon/chronos-t5-small
        DEFAULT_TIMESFM_MODEL=google/timesfm-1.0-200m

# =============================================================================
# GH200 ARCHITECTURE OPTIMIZATIONS
# =============================================================================

# GH200 Grace Hopper Superchip Detection
GH200_DETECTED={"true" if self.is_gh200 else "false"}
ARM64_DETECTED={"true" if self.is_arm64 else "false"}

# GH200 Performance Settings (144 ARM cores + H100 GPU)
OMP_NUM_THREADS={"144" if self.is_gh200 else "8"}
CUDA_VISIBLE_DEVICES=0
JAX_PLATFORMS=cuda
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# GH200 Memory Optimization (480GB unified memory)
TRANSFORMERS_CACHE=./models/cache
JAX_ENABLE_X64=true
PYTORCH_ENABLE_MPS_FALLBACK={"false" if self.is_gh200 else "true"}

# GH200 Multi-Model Parallel Processing
ENABLE_MODEL_PARALLEL={"true" if self.is_gh200 else "false"}
MAX_CONCURRENT_MODELS={"3" if self.is_gh200 else "1"}
BATCH_SIZE_MULTIPLIER={"4" if self.is_gh200 else "1"}

# =============================================================================
# SYSTEM CONFIGURATION  
# =============================================================================

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# Performance (GH200 optimized)
MAX_WORKERS={"32" if self.is_gh200 else "8"}
ENABLE_GPU={"true" if self.is_gh200 else "false"}
CACHE_SIZE={"10000" if self.is_gh200 else "1000"}

# Redis (for production caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Risk Management
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=5000
MAX_POSITIONS=5

# Strategy Settings
ENABLE_GAP_AND_GO=true
ENABLE_ORB=true
ENABLE_VOL_MEAN_REVERSION=true

# AI Enhancement
AI_CONFIDENCE_THRESHOLD=0.7
ENABLE_ENSEMBLE=true
MODEL_UPDATE_FREQUENCY=30

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Debug mode
DEBUG=false
ENABLE_PROFILING=false
SAVE_PREDICTIONS=false

# Testing
ENABLE_PAPER_TRADING=true
BACKTEST_MODE=false
"""
        
        env_file = self.root_dir / '.env.example'
        with open(env_file, 'w') as f:
            f.write(env_example)
        
        print("   ✅ Created comprehensive .env.example")
        
        # Interactive API key setup
        self.setup_interactive_env()
        
        # Create config template
        config_template = {
            "models": {
                "lag_llama": {
                    "enabled": True,
                    "model_path": "./models/lag_llama",
                    "context_length": 64,
                    "prediction_length": 5
                },
                "chronos": {
                    "enabled": True,
                    "model_name": "amazon/chronos-t5-small",
                    "context_length": 512,
                    "prediction_length": 24
                },
                "timesfm": {
                    "enabled": True,
                    "model_name": "google/timesfm-1.0-200m",
                    "context_len": 512,
                    "horizon_len": 128
                }
            },
            "strategies": {
                "gap_and_go": {
                    "enabled": True,
                    "min_gap_percent": 0.02,
                    "max_gap_percent": 0.10
                },
                "orb": {
                    "enabled": True,
                    "base_range_minutes": 5,
                    "dynamic_range": True
                },
                "vol_mean_reversion": {
                    "enabled": True,
                    "volatility_threshold": 2.0
                }
            }
        }
        
        config_file = self.ai_llama_dir / 'config_template.json'
        with open(config_file, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        print("   ✅ Created configuration template")
        
        # Create directories
        directories = [
            self.root_dir / 'logs',
            self.models_dir / 'cache',
            self.ai_llama_dir / 'data' / 'cache'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("   ✅ Created necessary directories")
        return True
    
    def setup_interactive_env(self):
        """Interactive API key setup and .env file creation"""
        print("\n🔑 Interactive API Key Setup")
        print("=" * 40)
        
        # Check if .env already exists
        env_file = self.root_dir / '.env'
        if env_file.exists():
            print("   ⚠️  .env file already exists!")
            response = input("   Do you want to overwrite it? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("   ⏭️  Skipping API key setup")
                return True
        
        print("\n📡 Please enter your API keys:")
        print("   (Leave blank to skip - you can add them later in .env)")
        
        # Get API keys from user
        api_keys = {}
        
        # Polygon API Key
        print("\n🔹 Polygon.io API Key:")
        print("   Get yours at: https://polygon.io/dashboard")
        polygon_key = input("   Enter Polygon API key: ").strip()
        api_keys['POLYGON_API_KEY'] = polygon_key if polygon_key else 'your_polygon_api_key_here'
        
        # Alpaca API Key
        print("\n🔹 Alpaca Trading API:")
        print("   Get yours at: https://app.alpaca.markets/account/keys")
        alpaca_key = input("   Enter Alpaca API key: ").strip()
        api_keys['ALPACA_API_KEY'] = alpaca_key if alpaca_key else 'your_alpaca_api_key_here'
        
        # Alpaca Secret Key
        alpaca_secret = input("   Enter Alpaca Secret key: ").strip()
        api_keys['ALPACA_SECRET_KEY'] = alpaca_secret if alpaca_secret else 'your_alpaca_secret_key_here'
        
        # Ask about paper trading
        print("\n📝 Trading Mode:")
        use_paper = input("   Use paper trading? (Y/n): ").strip().lower()
        if use_paper in ['', 'y', 'yes']:
            alpaca_url = 'https://paper-api.alpaca.markets'
            print("   ✅ Paper trading mode enabled (recommended for testing)")
        else:
            alpaca_url = 'https://api.alpaca.markets'
            print("   ⚠️  Live trading mode - USE WITH CAUTION!")
        
        # Create .env file content
        env_content = f"""# AI-Enhanced Trading System Configuration
# Generated by setup script on {time.strftime('%Y-%m-%d %H:%M:%S')}

# =============================================================================
# MARKET DATA APIs
# =============================================================================

# Polygon.io (Primary data source)
POLYGON_API_KEY={api_keys['POLYGON_API_KEY']}

# Alpaca Trading API
ALPACA_API_KEY={api_keys['ALPACA_API_KEY']}
ALPACA_SECRET_KEY={api_keys['ALPACA_SECRET_KEY']}
ALPACA_BASE_URL={alpaca_url}

# Alternative Data Sources (optional)
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
# IEX_CLOUD_API_KEY=your_iex_cloud_key
# BINANCE_API_KEY=your_binance_key
# BINANCE_SECRET_KEY=your_binance_secret

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model paths (automatically set by setup)
LAG_LLAMA_MODEL_PATH=./models/lag_llama
CHRONOS_MODEL_PATH=./models/chronos
TIMESFM_MODEL_PATH=./models/timesfm

# HuggingFace settings
HF_CACHE_DIR=./models/cache
HF_HOME=./models/cache
TRANSFORMERS_CACHE=./models/cache

# Model preferences
DEFAULT_CHRONOS_MODEL=amazon/chronos-t5-small
DEFAULT_TIMESFM_MODEL=google/timesfm-1.0-200m

# =============================================================================
# SYSTEM CONFIGURATION  
# =============================================================================

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# Performance
MAX_WORKERS=4
ENABLE_GPU=false
CACHE_SIZE=1000

# Redis (for production caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Risk Management
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=5000
MAX_POSITIONS=5

# Strategy Settings
ENABLE_GAP_AND_GO=true
ENABLE_ORB=true
ENABLE_VOL_MEAN_REVERSION=true

# AI Enhancement
AI_CONFIDENCE_THRESHOLD=0.7
ENABLE_ENSEMBLE=true
MODEL_UPDATE_FREQUENCY=30

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Debug mode
DEBUG=false
ENABLE_PROFILING=false
SAVE_PREDICTIONS=false

# Testing
ENABLE_PAPER_TRADING={"true" if use_paper in ['', 'y', 'yes'] else "false"}
BACKTEST_MODE=false
"""
        
        # Write .env file
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print(f"\n   ✅ Created .env file with your API keys")
            
            # Validate keys
            missing_keys = []
            if api_keys['POLYGON_API_KEY'] == 'your_polygon_api_key_here':
                missing_keys.append('Polygon.io')
            if api_keys['ALPACA_API_KEY'] == 'your_alpaca_api_key_here':
                missing_keys.append('Alpaca')
            
            if missing_keys:
                print(f"   ⚠️  Missing API keys: {', '.join(missing_keys)}")
                print(f"   📝 You can add them later by editing: {env_file}")
            else:
                print("   🎉 All API keys configured!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating .env file: {e}")
            return False
    
    def test_system(self):
        """Test all system components"""
        print("\n🧪 Testing system components...")
        
        test_results = {}
        
        # Test basic imports
        print("   📚 Testing imports...")
        try:
            from features.fast_features import extract_fast_features
            from models.fast_models import FastModelEngine
            test_results['basic_imports'] = True
            print("      ✅ Basic imports successful")
        except Exception as e:
            print(f"      ❌ Basic imports failed: {e}")
            test_results['basic_imports'] = False
        
        # Test foundation models
        print("   🤖 Testing foundation models...")
        
        # Test Lag-Llama
        try:
            from models.lag_llama_engine import LagLlamaEngine
            lag_llama = LagLlamaEngine()
            test_results['lag_llama'] = True
            print("      ✅ Lag-Llama engine initialized")
        except Exception as e:
            print(f"      ❌ Lag-Llama test failed: {e}")
            test_results['lag_llama'] = False
        
        # Test Chronos
        try:
            from models.chronos_engine import ChronosEngine
            chronos = ChronosEngine()
            test_results['chronos'] = True
            print("      ✅ Chronos engine initialized")
        except Exception as e:
            print(f"      ❌ Chronos test failed: {e}")
            test_results['chronos'] = False
        
        # Test TimesFM
        try:
            from models.timesfm_engine import TimesFMEngine
            timesfm = TimesFMEngine()
            test_results['timesfm'] = True
            print("      ✅ TimesFM engine initialized")
        except Exception as e:
            print(f"      ❌ TimesFM test failed: {e}")
            test_results['timesfm'] = False
        
        # Test ensemble
        try:
            from models.ensemble import ModelEnsemble
            ensemble = ModelEnsemble()
            test_results['ensemble'] = True
            print("      ✅ Model ensemble initialized")
        except Exception as e:
            print(f"      ❌ Ensemble test failed: {e}")
            test_results['ensemble'] = False
        
        # Test strategies
        print("   📈 Testing trading strategies...")
        try:
            from strategies.gap_and_go import GapAndGoStrategy
            from strategies.orb_strategy import ORBStrategy
            from strategies.vol_mean_reversion import VolMeanReversionStrategy
            
            gap_strategy = GapAndGoStrategy()
            orb_strategy = ORBStrategy()
            vol_strategy = VolMeanReversionStrategy()
            
            test_results['strategies'] = True
            print("      ✅ All strategies initialized")
        except Exception as e:
            print(f"      ❌ Strategy test failed: {e}")
            test_results['strategies'] = False
        
        # Summary
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"\n📊 Test Results: {passed}/{total} components working")
        
        if passed == total:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️  Some components have issues")
            for component, status in test_results.items():
                if not status:
                    print(f"   ❌ {component}")
            return False
    
    def create_download_script(self):
        """Create standalone model download script"""
        print("\n📝 Creating model download script...")
        
        download_script = '''#!/usr/bin/env python3
"""
Standalone Model Download Script
Run this script to download all AI models after setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def download_models():
    """Download all foundation models"""
    print("🤗 Downloading AI models...")
    
    # HuggingFace models - optimized for trading
    models = [
        "amazon/chronos-t5-small",  # Optimal balance: speed + accuracy for trading
        "google/timesfm-1.0-200m"   # Google's foundation model
    ]
    
    for model in models:
        print(f"📥 Downloading {model}...")
        try:
            subprocess.check_call([
                sys.executable, "-c",
                f"from transformers import AutoTokenizer, AutoModel; "
                f"AutoTokenizer.from_pretrained('{model}'); "
                f"AutoModel.from_pretrained('{model}')"
            ])
            print(f"✅ {model} downloaded")
        except Exception as e:
            print(f"❌ Error downloading {model}: {e}")
    
    # Lag-Llama
    print("🦙 Setting up Lag-Llama...")
    try:
        subprocess.check_call([
            sys.executable, "-c",
            "from lag_llama.gluon.estimator import LagLlamaEstimator; "
            "LagLlamaEstimator.from_pretrained('lag-llama')"
        ])
        print("✅ Lag-Llama ready")
    except Exception as e:
        print(f"❌ Error with Lag-Llama: {e}")
    
    print("🎉 Model download completed!")

if __name__ == "__main__":
    download_models()
'''
        
        script_file = self.ai_llama_dir / 'download_models.py'
        with open(script_file, 'w') as f:
            f.write(download_script)
        
        # Make executable on Unix systems
        try:
            os.chmod(script_file, 0o755)
        except:
            pass
        
        print("   ✅ Created download_models.py script")
        return True
    
    def run_full_setup(self):
        """Run complete setup process"""
        start_time = time.time()
        
        if not self.check_prerequisites():
            return False
        
        steps = [
            ("Installing Python dependencies", self.install_requirements),
            ("Cloning repositories", self.clone_repositories),
            ("Installing foundation packages", self.install_repositories),
            ("Downloading HuggingFace models", self.download_huggingface_models),
            ("Setting up Lag-Llama", self.download_lag_llama_models),
            ("Creating model symlinks", self.setup_model_symlinks),
            ("Setting up environment", self.setup_environment),
            ("Creating download script", self.create_download_script),
            ("Testing system", self.test_system)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"🔄 {step_name}...")
            
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Setup completed successfully in {elapsed:.1f} seconds!")
        
        self.print_next_steps()
        return True
    
    def print_next_steps(self):
        """Print next steps for user"""
        print("\n📋 Next Steps:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Edit .env with your API keys:")
        print("   nano .env  # or your preferred editor")
        print("\n3. Run the trading system:")
        print("   cd ai_llama")
        print("   python main.py")
        print("\n4. Optional - Download additional models:")
        print("   python download_models.py")
        print("\n5. Check the documentation:")
        print("   cat README.md")
        print("   cat QUICK_START.md")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="AI Trading System Setup")
    parser.add_argument('--download-models', action='store_true',
                       help='Download models only')
    parser.add_argument('--clone-repos', action='store_true',
                       help='Clone repositories only')
    parser.add_argument('--test', action='store_true',
                       help='Test system only')
    parser.add_argument('--requirements', action='store_true',
                       help='Install requirements only')
    
    args = parser.parse_args()
    
    setup = SetupManager()
    
    if args.download_models:
        setup.download_huggingface_models()
        setup.download_lag_llama_models()
    elif args.clone_repos:
        setup.clone_repositories()
        setup.install_repositories()
    elif args.test:
        setup.test_system()
    elif args.requirements:
        setup.install_requirements()
    else:
        # Full setup
        setup.run_full_setup()

if __name__ == "__main__":
    main()
