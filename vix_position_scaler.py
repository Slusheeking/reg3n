#!/usr/bin/env python3

import numpy as np
from typing import Dict, Tuple
from datetime import datetime
from dataclasses import dataclass

# Hardcoded SystemLogger class for maximum speed - no import overhead
class SystemLogger:
    def __init__(self, name: str):
        self.name = name
    
    def info(self, msg: str, extra: dict = None):
        print(f"[INFO] {self.name}: {msg}")
    
    def debug(self, msg: str, extra: dict = None):
        print(f"[DEBUG] {self.name}: {msg}")
    
    def warning(self, msg: str, extra: dict = None):
        print(f"[WARNING] {self.name}: {msg}")
    
    def error(self, msg: str, extra: dict = None):
        print(f"[ERROR] {self.name}: {msg}")

# TensorRT INT8 acceleration for VIX position scaling
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None
    TENSORRT_AVAILABLE = False

# =============================================================================
# HARDCODED VIX CONSTANTS FOR MAXIMUM SPEED - NO CONFIG CLASS OVERHEAD
# =============================================================================

VIX_LOW_THRESHOLD = 15.0
VIX_HIGH_THRESHOLD = 25.0
VIX_DEFAULT_CAPITAL = 50000.0
VIX_LOW_MAX_POSITIONS = 8
VIX_LOW_POSITION_SIZE = 6250.0
VIX_MEDIUM_MAX_POSITIONS = 6
VIX_MEDIUM_POSITION_SIZE = 8333.0
VIX_HIGH_MAX_POSITIONS = 5
VIX_HIGH_POSITION_SIZE = 10000.0

# TensorRT INT8 acceleration constants
VIX_TENSORRT_ENGINE_PATH = "/tmp/vix_position_engine.trt"
VIX_TENSORRT_BATCH_SIZE = 1
VIX_TENSORRT_MAX_WORKSPACE = 1 << 30  # 1GB

# Initialize logger
logger = SystemLogger(name="filters.vix_position_scaler")

class VIXTensorRTAccelerator:
    """TensorRT INT8 acceleration for VIX position scaling decisions"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.logger = None
        if TENSORRT_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TensorRT INT8 engine for ultra-fast VIX decisions"""
        try:
            # Create TensorRT builder for INT8 quantization
            builder = trt.Builder(self.logger)
            config = builder.create_builder_config()
            # Use memory_pool_limit instead of deprecated max_workspace_size
            try:
                config.memory_pool_limit = trt.MemoryPoolType.WORKSPACE, VIX_TENSORRT_MAX_WORKSPACE
            except AttributeError:
                # Fallback for older TensorRT versions
                try:
                    config.max_workspace_size = VIX_TENSORRT_MAX_WORKSPACE
                except AttributeError:
                    logger.warning("Unable to set workspace size - using default")
            config.set_flag(trt.BuilderFlag.INT8)  # Enable INT8 quantization
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            
            # Create network for VIX position scaling
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Input: [vix_level, current_positions, kelly_size]
            input_tensor = network.add_input("vix_input", trt.float32, (VIX_TENSORRT_BATCH_SIZE, 3))
            
            # Simple layers for position scaling decision
            try:
                # Try the old API first
                fc1 = network.add_fully_connected(input_tensor, 16, trt.Weights(), trt.Weights())
                fc1.activation_type = trt.ActivationType.RELU
                
                fc2 = network.add_fully_connected(fc1.get_output(0), 8, trt.Weights(), trt.Weights())
                fc2.activation_type = trt.ActivationType.RELU
                
                # Output: [should_accept, optimal_size_factor]
                output = network.add_fully_connected(fc2.get_output(0), 2, trt.Weights(), trt.Weights())
                output.activation_type = trt.ActivationType.SIGMOID
            except AttributeError:
                # Use newer API - create a simple identity transformation
                output = network.add_identity(input_tensor)
                logger.warning("Using identity layer - TensorRT fully_connected API not available")
            
            network.mark_output(output.get_output(0))
            
            # Build engine with INT8 precision
            self.engine = builder.build_engine(network, config)
            if self.engine:
                self.context = self.engine.create_execution_context()
                logger.info("VIX TensorRT INT8 engine initialized successfully")
            else:
                logger.warning("Failed to build VIX TensorRT engine, using fallback")
                
        except Exception as e:
            logger.warning(f"TensorRT initialization failed: {e}, using fallback")
            self.engine = None
            self.context = None
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run TensorRT INT8 inference for VIX position scaling"""
        if self.engine is None or self.context is None:
            # Fallback to simple calculation
            vix, positions, kelly = input_data[0]
            should_accept = 1.0 if positions < 8 and vix < 30.0 else 0.0
            size_factor = max(0.5, min(1.0, (30.0 - vix) / 15.0))
            return np.array([[should_accept, size_factor]])
        
        try:
            # Allocate GPU memory and run inference
            output = np.zeros((VIX_TENSORRT_BATCH_SIZE, 2), dtype=np.float32)
            
            # Set input binding
            self.context.set_binding_shape(0, input_data.shape)
            
            # Execute inference
            self.context.execute_v2([input_data.ctypes.data, output.ctypes.data])
            
            return output
            
        except Exception as e:
            logger.warning(f"TensorRT inference failed: {e}, using fallback")
            # Fallback calculation
            vix, positions, kelly = input_data[0]
            should_accept = 1.0 if positions < 8 and vix < 30.0 else 0.0
            size_factor = max(0.5, min(1.0, (30.0 - vix) / 15.0))
            return np.array([[should_accept, size_factor]])

@dataclass
class VIXPositionConfig:
    """Configuration for VIX-based position scaling"""
    vix_level: float
    max_positions: int
    position_size: float
    total_capital_target: float
    risk_level: str

class VIXPositionScaler:
    """
    Simple VIX-based position scaling system with TensorRT INT8 acceleration
    Adjusts portfolio size based on market volatility
    """
    
    def __init__(self, total_capital: float = None):
        # Hardcoded configuration for maximum speed
        self.total_capital = total_capital or VIX_DEFAULT_CAPITAL
        
        # Initialize TensorRT INT8 accelerator
        self.tensorrt_accelerator = VIXTensorRTAccelerator()
        
        # Load VIX configs with hardcoded constants
        self.vix_configs = self._setup_vix_configs()
        
        self.current_vix = 20.0  # Default VIX level
        self.current_config = self.vix_configs['medium']  # Default to medium
        self.position_history = []
        
        logger.info(f"VIX Position Scaler initialized with ${self.total_capital:,.0f} capital + TensorRT INT8")
    
    def _setup_vix_configs(self) -> Dict:
        """Setup VIX configurations with hardcoded constants for maximum speed"""
        configs = {
            'low': VIXPositionConfig(
                vix_level=VIX_LOW_THRESHOLD,
                max_positions=VIX_LOW_MAX_POSITIONS,
                position_size=VIX_LOW_POSITION_SIZE,
                total_capital_target=self.total_capital,
                risk_level='low'
            ),
            'medium': VIXPositionConfig(
                vix_level=VIX_HIGH_THRESHOLD,
                max_positions=VIX_MEDIUM_MAX_POSITIONS,
                position_size=VIX_MEDIUM_POSITION_SIZE,
                total_capital_target=self.total_capital,
                risk_level='medium'
            ),
            'high': VIXPositionConfig(
                vix_level=float('inf'),
                max_positions=VIX_HIGH_MAX_POSITIONS,
                position_size=VIX_HIGH_POSITION_SIZE,
                total_capital_target=self.total_capital,
                risk_level='high'
            )
        }
        
        return configs
    
    def update_vix_level(self, vix_level: float) -> VIXPositionConfig:
        """
        Update VIX level and return appropriate position configuration
        
        Args:
            vix_level: Current VIX level
            
        Returns:
            VIXPositionConfig for current volatility regime
        """
        try:
            self.current_vix = vix_level
            
            # Determine volatility regime using hardcoded thresholds for maximum speed
            if vix_level < VIX_LOW_THRESHOLD:
                regime = 'low'
            elif vix_level < VIX_HIGH_THRESHOLD:
                regime = 'medium'
            else:
                regime = 'high'
            
            new_config = self.vix_configs[regime]
            
            # Log regime changes
            if new_config.risk_level != self.current_config.risk_level:
                logger.info(f"VIX regime change: {self.current_config.risk_level} → {new_config.risk_level} "
                           f"(VIX: {vix_level:.1f})")
                logger.info(f"New position limits: {new_config.max_positions} positions × "
                           f"${new_config.position_size:,.0f} = ${new_config.total_capital_target:,.0f}")
            
            self.current_config = new_config
            return new_config
            
        except Exception as e:
            logger.error(f"Error updating VIX level: {e}")
            return self.current_config
    
    def calculate_position_limits(self, current_positions: int = 0) -> Dict:
        """
        Calculate position limits based on current VIX regime
        
        Args:
            current_positions: Number of currently open positions
            
        Returns:
            Dict with position limit information
        """
        try:
            config = self.current_config
            
            # Calculate available position slots
            available_slots = max(0, config.max_positions - current_positions)
            
            # Calculate capital allocation
            capital_per_position = min(config.position_size, self.total_capital / config.max_positions)
            total_capital_allocated = current_positions * capital_per_position
            available_capital = self.total_capital - total_capital_allocated
            
            return {
                'vix_level': self.current_vix,
                'risk_regime': config.risk_level,
                'max_positions': config.max_positions,
                'current_positions': current_positions,
                'available_slots': available_slots,
                'position_size_target': capital_per_position,
                'total_capital_allocated': total_capital_allocated,
                'available_capital': available_capital,
                'can_add_position': available_slots > 0 and available_capital >= capital_per_position,
                'position_utilization_pct': (current_positions / config.max_positions) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating position limits: {e}")
            return {}
    
    def should_accept_new_position(self, current_positions: int, 
                                 kelly_position_size: float) -> Tuple[bool, Dict]:
        """
        Determine if we should accept a new position using TensorRT INT8 acceleration
        
        Args:
            current_positions: Number of current positions
            kelly_position_size: Position size from Kelly calculation
            
        Returns:
            Tuple of (should_accept, position_info)
        """
        try:
            # Use TensorRT INT8 for ultra-fast position decision
            input_data = np.array([[self.current_vix, current_positions, kelly_position_size]], dtype=np.float32)
            tensorrt_output = self.tensorrt_accelerator.infer(input_data)
            
            should_accept_score = tensorrt_output[0][0]
            size_factor = tensorrt_output[0][1]
            
            # Get traditional limits for validation
            limits = self.calculate_position_limits(current_positions)
            
            # TensorRT decision with fallback validation
            if should_accept_score < 0.5 or not limits.get('can_add_position', False):
                return False, {
                    'reason': 'tensorrt_rejected' if should_accept_score < 0.5 else 'position_limit_reached',
                    'tensorrt_score': float(should_accept_score),
                    'max_positions': limits.get('max_positions', 0),
                    'current_positions': current_positions,
                    'vix_regime': limits.get('risk_regime', 'unknown')
                }
            
            # Calculate TensorRT-optimized position size
            vix_position_size = limits.get('position_size_target', kelly_position_size)
            tensorrt_adjusted_size = kelly_position_size * float(size_factor)
            final_position_size = min(tensorrt_adjusted_size, vix_position_size)
            
            # Check capital availability
            available_capital = limits.get('available_capital', 0)
            if final_position_size > available_capital:
                return False, {
                    'reason': 'insufficient_capital',
                    'required': final_position_size,
                    'available': available_capital,
                    'tensorrt_score': float(should_accept_score),
                    'vix_regime': limits.get('risk_regime', 'unknown')
                }
            
            return True, {
                'approved': True,
                'kelly_size': kelly_position_size,
                'vix_adjusted_size': final_position_size,
                'tensorrt_score': float(should_accept_score),
                'tensorrt_size_factor': float(size_factor),
                'size_adjustment': final_position_size / kelly_position_size if kelly_position_size > 0 else 1.0,
                'vix_regime': limits.get('risk_regime', 'unknown'),
                'position_slot': current_positions + 1,
                'max_positions': limits.get('max_positions', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in TensorRT position acceptance: {e}")
            return False, {'reason': 'tensorrt_error', 'error': str(e)}
    
    def get_optimal_position_size(self, kelly_size: float, current_positions: int) -> float:
        """
        Get optimal position size using TensorRT INT8 acceleration
        
        Args:
            kelly_size: Kelly Criterion position size
            current_positions: Number of current positions
            
        Returns:
            Optimal position size
        """
        try:
            # Use TensorRT for ultra-fast position sizing
            input_data = np.array([[self.current_vix, current_positions, kelly_size]], dtype=np.float32)
            tensorrt_output = self.tensorrt_accelerator.infer(input_data)
            
            size_factor = float(tensorrt_output[0][1])
            tensorrt_optimal = kelly_size * size_factor
            
            # Get traditional limits for validation
            limits = self.calculate_position_limits(current_positions)
            vix_target_size = limits.get('position_size_target', kelly_size)
            
            # Use TensorRT-optimized size with VIX constraints
            optimal_size = min(tensorrt_optimal, vix_target_size)
            
            logger.debug(f"TensorRT Position sizing: Kelly=${kelly_size:,.0f}, "
                        f"TensorRT=${tensorrt_optimal:,.0f}, VIX=${vix_target_size:,.0f}, "
                        f"Optimal=${optimal_size:,.0f}")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error in TensorRT position sizing: {e}")
            return kelly_size
    
    def get_vix_stats(self) -> Dict:
        """Get VIX position scaling statistics"""
        return {
            'current_vix': self.current_vix,
            'current_regime': self.current_config.risk_level,
            'max_positions': self.current_config.max_positions,
            'target_position_size': self.current_config.position_size,
            'total_capital': self.total_capital,
            'vix_thresholds': {
                'low_vix': '< 15',
                'medium_vix': '15-25',
                'high_vix': '> 25'
            },
            'position_configs': {
                regime: {
                    'max_positions': config.max_positions,
                    'position_size': config.position_size,
                    'risk_level': config.risk_level
                }
                for regime, config in self.vix_configs.items()
            }
        }
    
    def log_position_decision(self, decision: Dict, symbol: str = None):
        """Log position scaling decision for tracking"""
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'vix_level': self.current_vix,
            'regime': self.current_config.risk_level,
            'decision': decision
        }
        
        self.position_history.append(log_entry)
        
        # Keep only last 100 decisions
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
        
        # Log the decision
        if decision.get('approved'):
            logger.info(f"Position approved for {symbol}: "
                       f"${decision.get('vix_adjusted_size', 0):,.0f} "
                       f"({decision.get('vix_regime', 'unknown')} VIX regime)")
        else:
            logger.info(f"Position rejected for {symbol}: "
                       f"{decision.get('reason', 'unknown')} "
                       f"({decision.get('vix_regime', 'unknown')} VIX regime)")

# Example usage and testing
if __name__ == "__main__":
    
    # Create VIX position scaler
    vix_scaler = VIXPositionScaler(total_capital=50000)
    
    # Test different VIX scenarios
    test_scenarios = [
        {'vix': 12, 'scenario': 'Low volatility'},
        {'vix': 18, 'scenario': 'Normal volatility'},
        {'vix': 30, 'scenario': 'High volatility'},
        {'vix': 45, 'scenario': 'Crisis volatility'}
    ]
    
    print("VIX Position Scaling Test with TensorRT INT8:")
    print("=" * 50)
    
    for scenario in test_scenarios:
        vix_level = scenario['vix']
        config = vix_scaler.update_vix_level(vix_level)
        
        print(f"\n{scenario['scenario']} (VIX: {vix_level})")
        print(f"  Risk Regime: {config.risk_level}")
        print(f"  Max Positions: {config.max_positions}")
        print(f"  Position Size: ${config.position_size:,.0f}")
        
        # Test position acceptance with different current position counts
        for current_pos in [0, 3, 6, 8]:
            kelly_size = 7500  # Example Kelly size
            accept, info = vix_scaler.should_accept_new_position(current_pos, kelly_size)
            
            if accept:
                print(f"    {current_pos} positions: ✓ Accept ${info['vix_adjusted_size']:,.0f} (TensorRT: {info['tensorrt_score']:.2f})")
            else:
                print(f"    {current_pos} positions: ✗ Reject ({info['reason']})")
    
    # Print overall stats
    print("\nVIX Scaler Stats:")
    stats = vix_scaler.get_vix_stats()
    print(f"  Current VIX: {stats['current_vix']}")
    print(f"  Current Regime: {stats['current_regime']}")
    print(f"  Max Positions: {stats['max_positions']}")