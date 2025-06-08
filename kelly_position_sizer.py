#!/usr/bin/env python3

# ULTRA-FAST KELLY POSITION SIZER WITH ZERO-COPY OPERATIONS
# Enhanced for real-time position sizing with aggressive Kelly criterion

# ULTRA-LOW LATENCY KELLY POSITION SIZER WITH ZERO-COPY OPERATIONS
# All lookup tables hardcoded directly for maximum HFT speed
# Zero import overhead - everything self-contained

import time
import numpy as np  # For zero-copy memory operations
from typing import Dict, List, Optional, Any

import os

# Hardcoded SystemLogger class for maximum speed (no imports)
class SystemLogger:
    def __init__(self, name="kelly_position_sizer"):
        self.name = name
        # ANSI color codes for terminal output
        self.colors = {
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'WHITE': '\033[97m',
            'RESET': '\033[0m'
        }
        
        # Create logs directory if it doesn't exist
        self.log_dir = "/home/ubuntu/reg3n-1/logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_file = os.path.join(self.log_dir, "backtesting.log")
        
    def _log(self, level, message, color_code, extra=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] - {color_code}{level}{self.colors['RESET']} - [{self.name}]: {message}"
        
        # Print to console with colors
        print(formatted_message)
        
        # Write to file without colors
        file_message = f"[{timestamp}] - {level} - [{self.name}]: {message}"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(file_message + '\n')
                if extra:
                    f.write(f"    Extra: {extra}\n")
        except Exception:
            pass  # Fail silently to avoid disrupting performance
        
        if extra:
            print(f"    Extra: {extra}")
        
    def info(self, message, extra=None):
        self._log("INFO", message, self.colors['WHITE'], extra)
    
    def debug(self, message, extra=None):
        self._log("DEBUG", message, self.colors['BLUE'], extra)
    
    def warning(self, message, extra=None):
        self._log("WARNING", message, self.colors['YELLOW'], extra)
    
    def error(self, message, extra=None):
        self._log("ERROR", message, self.colors['RED'], extra)

# =============================================================================
# CONSOLIDATED KELLY LOOKUP TABLES - ALL METHODS HARDCODED FOR MAXIMUM SPEED
# =============================================================================

# BINARY PACKED KELLY LOOKUP - BIT OPERATIONS FOR MAXIMUM SPEED
# Binary packed Kelly positions (position_pct * 100 as 16-bit integers)
KELLY_BINARY_LOOKUP = [
    # Win Rate 50% - positions 0.5% to 1.8%
    [50, 40, 30, 20, 10, 80, 60, 50, 30, 20, 120, 90, 70, 50, 30, 150, 110, 80, 60, 40, 180, 130, 100, 70, 50],
    
    # Win Rate 52% - positions 1.2% to 5.0%
    [120, 100, 80, 60, 40, 240, 190, 150, 110, 80, 360, 290, 230, 170, 120, 430, 340, 270, 200, 140, 500, 400, 320, 240, 160],
    
    # Win Rate 54% - positions 2.4% to 10.0%
    [240, 190, 150, 110, 80, 480, 380, 300, 230, 150, 720, 580, 460, 340, 230, 860, 690, 550, 410, 280, 1000, 800, 640, 480, 320],
    
    # Win Rate 56% - positions 3.6% to 15.0%
    [360, 290, 230, 170, 120, 720, 580, 460, 340, 230, 1080, 860, 690, 520, 350, 1290, 1030, 830, 620, 410, 1500, 1200, 960, 720, 480],
    
    # Win Rate 58% - positions 4.8% to 20.0%
    [480, 380, 300, 230, 150, 960, 770, 610, 460, 310, 1440, 1150, 920, 690, 460, 1730, 1380, 1100, 830, 550, 2000, 1600, 1280, 960, 640],
    
    # Win Rate 60% - positions 6.0% to 25.0%
    [600, 480, 380, 290, 190, 1200, 960, 770, 580, 380, 1800, 1440, 1150, 860, 580, 2160, 1730, 1380, 1040, 690, 2500, 2000, 1600, 1200, 800],
    
    # Win Rate 62% - positions 7.2% to 30.0% (capped)
    [720, 580, 460, 340, 230, 1440, 1150, 920, 690, 460, 2160, 1730, 1380, 1040, 690, 2590, 2070, 1660, 1240, 830, 3000, 2400, 1920, 1440, 960],
    
    # Win Rate 64% - positions 8.4% to 30.0% (capped)
    [840, 670, 540, 400, 270, 1680, 1340, 1070, 800, 540, 2520, 2020, 1610, 1210, 810, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960],
    
    # Win Rate 66% - positions 9.6% to 30.0% (capped)
    [960, 770, 610, 460, 310, 1920, 1540, 1230, 920, 610, 2880, 2300, 1840, 1380, 920, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960],
    
    # Win Rate 68% - positions 10.8% to 30.0% (capped)
    [1080, 860, 690, 520, 350, 2160, 1730, 1380, 1040, 690, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960],
    
    # Win Rate 70% - positions 12.0% to 30.0% (capped)
    [1200, 960, 770, 580, 380, 2400, 1920, 1540, 1150, 770, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960, 3000, 2400, 1920, 1440, 960]
]

# KELLY FRACTION LOOKUP TABLES - PRE-COMPUTED FOR MAXIMUM SPEED
KELLY_WIN_RATE_LOOKUP = {
    0.50: [0.50, 0.000, 0.0],   # Break-even, no edge
    0.51: [0.51, 0.020, 2.0],   # Minimal edge
    0.52: [0.52, 0.040, 4.0],   # Small edge
    0.53: [0.53, 0.060, 6.0],   # Growing edge
    0.54: [0.54, 0.080, 8.0],   # Decent edge
    0.55: [0.55, 0.100, 10.0],  # Good edge
    0.56: [0.56, 0.120, 12.0],  # Strong edge
    0.57: [0.57, 0.140, 14.0],  # Very strong edge
    0.58: [0.58, 0.160, 16.0],  # Excellent edge
    0.59: [0.59, 0.180, 18.0],  # Outstanding edge
    0.60: [0.60, 0.200, 20.0],  # Exceptional edge
    0.61: [0.61, 0.220, 22.0],  # Rare edge
    0.62: [0.62, 0.240, 24.0],  # Extreme edge
    0.63: [0.63, 0.260, 26.0],  # Maximum practical edge
    0.64: [0.64, 0.280, 28.0],  # Theoretical edge
    0.65: [0.65, 0.300, 30.0],  # Cap at 30% for safety
    0.66: [0.66, 0.300, 30.0],  # Safety cap
    0.67: [0.67, 0.300, 30.0],  # Safety cap
    0.68: [0.68, 0.300, 30.0],  # Safety cap
    0.69: [0.69, 0.300, 30.0],  # Safety cap
    0.70: [0.70, 0.300, 30.0],  # Safety cap
    0.75: [0.75, 0.300, 30.0],  # Safety cap
    0.80: [0.80, 0.300, 30.0],  # Safety cap
    0.85: [0.85, 0.300, 30.0],  # Safety cap
    0.90: [0.90, 0.300, 30.0],  # Safety cap
    0.95: [0.95, 0.300, 30.0],  # Safety cap
}

# Confidence Level Multipliers (0.1 to 1.0 in 0.1 increments)
CONFIDENCE_MULTIPLIERS = {
    0.1: 0.10,  # Very low confidence - 10% of Kelly
    0.2: 0.20,  # Low confidence - 20% of Kelly
    0.3: 0.35,  # Below average confidence - 35% of Kelly
    0.4: 0.50,  # Average confidence - 50% of Kelly
    0.5: 0.65,  # Above average confidence - 65% of Kelly
    0.6: 0.75,  # Good confidence - 75% of Kelly
    0.7: 0.85,  # High confidence - 85% of Kelly
    0.8: 0.90,  # Very high confidence - 90% of Kelly
    0.9: 0.95,  # Excellent confidence - 95% of Kelly
    1.0: 1.00,  # Maximum confidence - 100% of Kelly
}

# Volatility Adjustment Factors (VIX levels)
VIX_ADJUSTMENT_FACTORS = {
    10.0: 1.50,  # Very low volatility - increase position
    12.0: 1.30,  # Low volatility - increase position
    15.0: 1.20,  # Below normal volatility - slight increase
    18.0: 1.10,  # Normal volatility - slight increase
    20.0: 1.00,  # Average volatility - no adjustment
    22.0: 0.90,  # Above average volatility - slight decrease
    25.0: 0.80,  # High volatility - decrease position
    30.0: 0.70,  # Very high volatility - significant decrease
    35.0: 0.60,  # Extreme volatility - major decrease
    40.0: 0.50,  # Crisis volatility - half position
    50.0: 0.30,  # Panic volatility - minimal position
    60.0: 0.20,  # Market crash - emergency position
    70.0: 0.10,  # Black swan - survival mode
}

# PRE-COMPUTED KELLY POSITION SIZE ARRAYS - INSTANT ARRAY INDEXING
# Array dimensions: Win rates 50-70% (11 values) × Confidence 20-100% (5 values) × VIX 10-50 (5 values)
KELLY_POSITION_ARRAY = [
    # Win Rate 50% (index 0) - No edge, minimal positions
    [
        [0.5, 0.4, 0.3, 0.2, 0.1],  # Confidence 20%
        [0.8, 0.6, 0.5, 0.3, 0.2],  # Confidence 40%
        [1.2, 0.9, 0.7, 0.5, 0.3],  # Confidence 60%
        [1.5, 1.1, 0.8, 0.6, 0.4],  # Confidence 80%
        [1.8, 1.3, 1.0, 0.7, 0.5],  # Confidence 100%
    ],
    # Win Rate 52% (index 1) - Small edge
    [
        [1.2, 1.0, 0.8, 0.6, 0.4],  # Confidence 20%
        [2.4, 1.9, 1.5, 1.1, 0.8],  # Confidence 40%
        [3.6, 2.9, 2.3, 1.7, 1.2],  # Confidence 60%
        [4.3, 3.4, 2.7, 2.0, 1.4],  # Confidence 80%
        [5.0, 4.0, 3.2, 2.4, 1.6],  # Confidence 100%
    ],
    # Win Rate 54% (index 2) - Decent edge
    [
        [2.4, 1.9, 1.5, 1.1, 0.8],  # Confidence 20%
        [4.8, 3.8, 3.0, 2.3, 1.5],  # Confidence 40%
        [7.2, 5.8, 4.6, 3.4, 2.3],  # Confidence 60%
        [8.6, 6.9, 5.5, 4.1, 2.8],  # Confidence 80%
        [10.0, 8.0, 6.4, 4.8, 3.2],  # Confidence 100%
    ],
    # Win Rate 56% (index 3) - Strong edge
    [
        [3.6, 2.9, 2.3, 1.7, 1.2],  # Confidence 20%
        [7.2, 5.8, 4.6, 3.4, 2.3],  # Confidence 40%
        [10.8, 8.6, 6.9, 5.2, 3.5],  # Confidence 60%
        [12.9, 10.3, 8.3, 6.2, 4.1],  # Confidence 80%
        [15.0, 12.0, 9.6, 7.2, 4.8],  # Confidence 100%
    ],
    # Win Rate 58% (index 4) - Excellent edge
    [
        [4.8, 3.8, 3.0, 2.3, 1.5],  # Confidence 20%
        [9.6, 7.7, 6.1, 4.6, 3.1],  # Confidence 40%
        [14.4, 11.5, 9.2, 6.9, 4.6],  # Confidence 60%
        [17.3, 13.8, 11.0, 8.3, 5.5],  # Confidence 80%
        [20.0, 16.0, 12.8, 9.6, 6.4],  # Confidence 100%
    ],
    # Win Rate 60% (index 5) - Exceptional edge
    [
        [6.0, 4.8, 3.8, 2.9, 1.9],  # Confidence 20%
        [12.0, 9.6, 7.7, 5.8, 3.8],  # Confidence 40%
        [18.0, 14.4, 11.5, 8.6, 5.8],  # Confidence 60%
        [21.6, 17.3, 13.8, 10.4, 6.9],  # Confidence 80%
        [25.0, 20.0, 16.0, 12.0, 8.0],  # Confidence 100%
    ],
    # Win Rate 62% (index 6) - Rare edge
    [
        [7.2, 5.8, 4.6, 3.4, 2.3],  # Confidence 20%
        [14.4, 11.5, 9.2, 6.9, 4.6],  # Confidence 40%
        [21.6, 17.3, 13.8, 10.4, 6.9],  # Confidence 60%
        [25.9, 20.7, 16.6, 12.4, 8.3],  # Confidence 80%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 100%
    ],
    # Win Rate 64% (index 7) - Extreme edge
    [
        [8.4, 6.7, 5.4, 4.0, 2.7],  # Confidence 20%
        [16.8, 13.4, 10.7, 8.0, 5.4],  # Confidence 40%
        [25.2, 20.2, 16.1, 12.1, 8.1],  # Confidence 60%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 80%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 100% (capped)
    ],
    # Win Rate 66% (index 8) - Maximum practical edge
    [
        [9.6, 7.7, 6.1, 4.6, 3.1],  # Confidence 20%
        [19.2, 15.4, 12.3, 9.2, 6.1],  # Confidence 40%
        [28.8, 23.0, 18.4, 13.8, 9.2],  # Confidence 60%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 80% (capped)
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 100% (capped)
    ],
    # Win Rate 68% (index 9) - Theoretical edge
    [
        [10.8, 8.6, 6.9, 5.2, 3.5],  # Confidence 20%
        [21.6, 17.3, 13.8, 10.4, 6.9],  # Confidence 40%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 60% (capped)
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 80% (capped)
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 100% (capped)
    ],
    # Win Rate 70% (index 10) - Safety cap
    [
        [12.0, 9.6, 7.7, 5.8, 3.8],  # Confidence 20%
        [24.0, 19.2, 15.4, 11.5, 7.7],  # Confidence 40%
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 60% (capped)
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 80% (capped)
        [30.0, 24.0, 19.2, 14.4, 9.6],  # Confidence 100% (capped)
    ]
]

# Market cap adjustment factors (packed as 8-bit integers, factor * 100)
MARKET_CAP_FACTORS = [120, 110, 100, 90, 70]  # 1.2x, 1.1x, 1.0x, 0.9x, 0.7x

# Time of day adjustment factors (packed as 8-bit integers, factor * 100)
TIME_FACTORS = [70, 100, 100, 80]  # 0.7x, 1.0x, 1.0x, 0.8x

# Market cap multipliers for position size adjustment
MARKET_CAP_MULTIPLIERS = [
    1.2,  # $1T+ mega cap
    1.1,  # $100B+ large cap
    1.0,  # $10B+ mid cap
    0.9,  # $1B+ small cap
    0.7   # <$1B micro cap
]

# Time of day multipliers (market hours)
TIME_MULTIPLIERS = [
    0.7,  # Market open (9:30-10:30)
    1.0,  # Mid morning (10:30-12:00)
    1.0,  # Afternoon (12:00-15:00)
    0.8   # Power hour (15:00-16:00)
]

# HARDCODED ULTRA-FAST SETTINGS FOR AGGRESSIVE $1000/DAY STRATEGY
AVAILABLE_CAPITAL = 50000
DAILY_TARGET = 1000  # $1000/day target
AGGRESSIVE_POSITION_MIN = 2000  # $2000 minimum per position
AGGRESSIVE_POSITION_MAX = 4000  # $4000 maximum per position
STOP_LOSS_PCT = 0.005  # Tighter 0.5% stops for quick exits
TP1_PCT = 0.005  # Quick 0.5% take profits
TP2_PCT = 0.01   # Secondary 1% take profits
SAFETY_FACTOR = 0.8  # More aggressive (was 0.25)
MIN_POSITION_VALUE = 2000  # Increased minimum
MAX_POSITION_VALUE = 4000  # Aggressive maximum
MIN_SHARES = 10
MAX_DAILY_POSITIONS = 20  # Maximum 20 positions per day
TARGET_TRADES_PER_DAY = 15  # Target 15 successful trades

# Initialize component logger
logger = SystemLogger(name="kelly_position_sizer")

# =============================================================================
# ULTRA-FAST KELLY LOOKUP FUNCTIONS - ALL METHODS CONSOLIDATED
# =============================================================================

class UltraFastKellyResult:
    """Ultra-fast Kelly result optimized for sub-microsecond speed"""
    def __init__(self, symbol, total_qty, total_value, tier_quantities, prices, kelly_fraction, confidence_tier, processing_time_ms):
        self.symbol = symbol
        self.total_qty = total_qty
        self.total_value = total_value
        self.tier_quantities = tier_quantities
        self.prices = prices
        self.kelly_fraction = kelly_fraction
        self.confidence_tier = confidence_tier  # 0=low, 1=medium, 2=high
        self.processing_time_ms = processing_time_ms

def binary_kelly_lookup(win_rate: float, confidence: float, vix_level: float = 20.0,
                       market_cap: float = 10000000000, available_capital: float = 50000.0) -> float:
    """Ultra-fast binary Kelly position sizing - returns position size in dollars"""
    
    # Convert to indices using bit operations (fastest possible)
    win_rate_int = int(win_rate * 100)
    win_idx = max(0, min(10, (win_rate_int - 50) >> 1))  # Divide by 2 using bit shift
    
    confidence_int = int(confidence * 100)
    conf_idx = max(0, min(4, (confidence_int - 20) // 20))
    
    vix_int = int(vix_level)
    vix_idx = max(0, min(4, (vix_int - 10) // 10))
    
    # Calculate array index using bit operations
    array_idx = (conf_idx << 2) + vix_idx  # conf_idx * 4 + vix_idx using bit shift
    
    # Binary lookup (single memory access)
    base_position_packed = KELLY_BINARY_LOOKUP[win_idx][array_idx]
    
    # Market cap factor (bit operations for speed)
    if market_cap >= 1000000000000:  # $1T+
        market_factor = MARKET_CAP_FACTORS[0]
    elif market_cap >= 100000000000:  # $100B+
        market_factor = MARKET_CAP_FACTORS[1]
    elif market_cap >= 10000000000:   # $10B+
        market_factor = MARKET_CAP_FACTORS[2]
    elif market_cap >= 1000000000:    # $1B+
        market_factor = MARKET_CAP_FACTORS[3]
    else:
        market_factor = MARKET_CAP_FACTORS[4]
    
    # Apply factors using integer math for speed
    adjusted_position_packed = (base_position_packed * market_factor) // 100
    
    # Convert back to percentage and apply limits
    final_position_pct = adjusted_position_packed / 100.0
    final_position_pct = min(final_position_pct, 30.0)  # Safety cap
    final_position_pct = max(final_position_pct, 0.5)   # Minimum position
    
    # Calculate position in dollars
    return final_position_pct * available_capital / 100.0

def ultra_fast_kelly_lookup(win_rate: float, confidence: float, vix_level: float = 20.0,
                           market_cap: float = 10000000000, available_capital: float = 50000.0) -> float:
    """Ultra-fast Kelly array lookup - returns position size in dollars"""
    
    # Get array indices (3 integer operations)
    win_rate_pct = max(50, min(70, int(win_rate * 100)))
    win_idx = (win_rate_pct - 50) // 2
    
    confidence_pct = max(20, min(100, int(confidence * 100)))
    conf_idx = (confidence_pct - 20) // 20
    
    vix_rounded = max(10, min(50, int(vix_level / 10) * 10))
    vix_idx = (vix_rounded - 10) // 10
    
    # Array lookup (single memory access)
    base_position_pct = KELLY_POSITION_ARRAY[win_idx][conf_idx][vix_idx]
    
    # Apply market cap multiplier
    if market_cap >= 1000000000000:  # $1T+
        market_multiplier = MARKET_CAP_MULTIPLIERS[0]
    elif market_cap >= 100000000000:  # $100B+
        market_multiplier = MARKET_CAP_MULTIPLIERS[1]
    elif market_cap >= 10000000000:   # $10B+
        market_multiplier = MARKET_CAP_MULTIPLIERS[2]
    elif market_cap >= 1000000000:    # $1B+
        market_multiplier = MARKET_CAP_MULTIPLIERS[3]
    else:  # <$1B
        market_multiplier = MARKET_CAP_MULTIPLIERS[4]
    
    # Final position calculation
    final_position_pct = base_position_pct * market_multiplier
    final_position_pct = min(final_position_pct, 30.0)  # Safety cap
    final_position_pct = max(final_position_pct, 0.5)   # Minimum position
    
    return final_position_pct * available_capital / 100

def get_ultra_fast_kelly_position(win_rate: float, confidence: float, vix_level: float = 20.0,
                                 market_cap: float = 10000000000, available_capital: float = 50000.0) -> dict:
    """Ultra-fast Kelly position sizing with O(1) lookup - returns detailed info"""
    
    # Base Kelly fraction lookup (O(1))
    win_rate_rounded = round(win_rate, 2)
    if win_rate_rounded < 0.50:
        win_rate_rounded = 0.50
    elif win_rate_rounded > 0.95:
        win_rate_rounded = 0.95
    
    base_kelly_data = KELLY_WIN_RATE_LOOKUP.get(win_rate_rounded, KELLY_WIN_RATE_LOOKUP[0.55])
    base_kelly_pct = base_kelly_data[1] * 100  # Convert to percentage
    
    # Confidence adjustment (O(1))
    confidence_rounded = round(confidence, 1)
    confidence_multiplier = CONFIDENCE_MULTIPLIERS.get(confidence_rounded, 0.65)
    adjusted_kelly_pct = base_kelly_pct * confidence_multiplier
    
    # VIX volatility adjustment (O(1))
    vix_rounded = round(vix_level / 5) * 5  # Round to nearest 5
    vix_factor = VIX_ADJUSTMENT_FACTORS.get(vix_rounded, 1.0)
    volatility_adjusted_pct = adjusted_kelly_pct * vix_factor
    
    # Market cap risk adjustment (O(1))
    market_cap_factor = 1.0
    if market_cap >= 1000000000000:  # $1T+
        market_cap_factor = 1.2
    elif market_cap >= 100000000000:  # $100B+
        market_cap_factor = 1.1
    elif market_cap >= 10000000000:   # $10B+
        market_cap_factor = 1.0
    elif market_cap >= 1000000000:    # $1B+
        market_cap_factor = 0.9
    else:
        market_cap_factor = 0.7
    
    market_cap_adjusted_pct = volatility_adjusted_pct * market_cap_factor
    
    # Apply safety factor and maximum limits
    final_kelly_pct = market_cap_adjusted_pct * SAFETY_FACTOR
    final_position_pct = min(final_kelly_pct, 30.0)
    
    # Ensure minimum position size for valid signals
    if final_position_pct < 0.5:
        final_position_pct = 0.5
    
    # Calculate position in dollars
    position_dollars = final_position_pct * available_capital / 100
    
    return {
        'position_pct': final_position_pct,
        'position_dollars': position_dollars,
        'base_kelly_pct': base_kelly_pct,
        'confidence_multiplier': confidence_multiplier,
        'vix_factor': vix_factor,
        'market_cap_factor': market_cap_factor,
        'safety_factor': SAFETY_FACTOR,
        'win_rate_used': win_rate_rounded,
    }

# =============================================================================
# MAIN KELLY POSITION SIZER CLASS - CONSOLIDATED ALL METHODS
# =============================================================================

class UltraFastKellyPositionSizer:
    """
    Ultra-fast Kelly position sizer integrated with Polygon client
    Provides real-time position sizing for live trading
    Ultra-fast Kelly position sizer with all lookup methods consolidated
    Target: Sub-microsecond position sizing for maximum HFT speed
    """
    
    def __init__(self, available_capital=None, gpu_enabled=False, memory_pools=None):
        # Hardcoded config values for maximum speed
        self.available_capital = available_capital or AVAILABLE_CAPITAL
        self.initial_capital = self.available_capital
        
        # Unified architecture integration
        self.memory_pools = memory_pools or {}
        self.ml_bridge = None
        self.portfolio_manager = None
        self.ml_system = None
        self.zero_copy_enabled = bool(memory_pools)
        self.gpu_enabled = False  # Pure lookup architecture
        
        # Zero-copy memory pools
        self.memory_pools = memory_pools or self._create_zero_copy_memory_pools()
        self.zero_copy_enabled = bool(self.memory_pools)
        
        # ML prediction bridge and portfolio manager (injected by orchestrator)
        self.ml_bridge = None
        self.portfolio_manager = None
        
        logger.info(f"Initializing Aggressive Kelly Position Sizer (capital: ${self.available_capital:,}, Target: ${DAILY_TARGET}/day)")
        if self.zero_copy_enabled:
            logger.info("Zero-copy memory pools enabled for sub-microsecond position sizing")
        
        # Aggressive $1000/day strategy constants
        self.DAILY_TARGET = DAILY_TARGET
        self.AGGRESSIVE_POSITION_MIN = AGGRESSIVE_POSITION_MIN
        self.AGGRESSIVE_POSITION_MAX = AGGRESSIVE_POSITION_MAX
        self.STOP_LOSS_PCT = STOP_LOSS_PCT
        self.TP1_PCT = TP1_PCT
        self.TP2_PCT = TP2_PCT
        self.SAFETY_FACTOR = SAFETY_FACTOR
        self.MIN_POSITION_VALUE = MIN_POSITION_VALUE
        self.MAX_POSITION_VALUE = MAX_POSITION_VALUE
        self.MIN_SHARES = MIN_SHARES
        self.MAX_DAILY_POSITIONS = MAX_DAILY_POSITIONS
        self.TARGET_TRADES_PER_DAY = TARGET_TRADES_PER_DAY
        
        # Daily tracking for $1000 target
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_positions = []
        self.current_positions = 0
        self.cash_available = self.available_capital
        self.portfolio_value = self.available_capital
        
        # Aggressive position sizing parameters
        self.base_position_size = self.DAILY_TARGET / self.TARGET_TRADES_PER_DAY  # ~$67 per trade target
        self.position_multiplier = 30  # Scale up for 0.5% profits: $67 * 30 = ~$2000

# Pre-computed tier allocations
        self.TIER_ALLOCATIONS = [0.30, 0.40, 0.30]
        
        # Initialize ultra-fast lookup tables
        self._init_lookup_tables()
        
        # Pre-computed price level multipliers
        self.price_multipliers = {
            'stop_loss': 1.0 - self.STOP_LOSS_PCT,
            'tp1_target': 1.0 + self.TP1_PCT,
            'tp2_target': 1.0 + self.TP2_PCT,
            'trail_percent': 2.0
        }
        
        # Performance tracking
        self.stats = {
            'calculations_made': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0,
            'lookup_hits': 0,
            'binary_calculations': 0,
            'array_calculations': 0,
            'table_calculations': 0
        }
        
        logger.info(f"✓ Consolidated Kelly Position Sizer initialized (all lookup methods available)")
    
    def _create_zero_copy_memory_pools(self):
        """Create zero-copy memory pools for ultra-fast position sizing."""
        try:
            pool_size = 1000  # Support up to 1000 concurrent position calculations
            logger.info(f"Creating zero-copy memory pools for {pool_size} position calculations")
            
            # Pre-allocate memory pools for position sizing
            memory_pools = {
                'position_pool': np.zeros((pool_size, 8), dtype=np.float64),  # symbol_idx, price, confidence, vix, market_cap, position_size, shares, value
                'tier_pool': np.zeros((pool_size, 4), dtype=np.int32),  # tier1, tier2, tier3, total
                'price_pool': np.zeros((pool_size, 4), dtype=np.float64),  # stop_loss, tp1, tp2, trail_pct
                'symbol_to_index': {},
                'index_to_symbol': [''] * pool_size,
                'active_positions_mask': np.zeros(pool_size, dtype=bool),
                'kelly_results_pool': np.zeros((pool_size, 6), dtype=np.float64),  # kelly_fraction, confidence_tier, processing_time, daily_pnl, target_progress, cash_available
            }
            
            logger.info("Zero-copy memory pools created successfully for Kelly position sizing")
            return memory_pools
            
        except ImportError:
            logger.warning("NumPy not available, falling back to standard memory allocation")
            return {}
        except Exception as e:
            logger.error(f"Failed to create zero-copy memory pools: {e}")
            return {}
    
    def _init_lookup_tables(self):
        """Initialize ultra-fast lookup tables for sub-nanosecond Kelly calculations"""
        
        # Pre-compute tier quantities for common share counts using zero-copy arrays
        self.tier_qty_lookup = {}
        
        # Convert to NumPy arrays for zero-copy operations
        if self.zero_copy_enabled:
            shares_array = np.arange(10, 1001, 10, dtype=np.int32)
            tier1_array = (shares_array * 0.30).astype(np.int32)
            tier2_array = (shares_array * 0.40).astype(np.int32)
            tier3_array = shares_array - tier1_array - tier2_array
            
            for i, shares in enumerate(shares_array):
                self.tier_qty_lookup[shares] = {
                    'tier1': int(tier1_array[i]),
                    'tier2': int(tier2_array[i]),
                    'tier3': int(tier3_array[i]),
                    'total': shares
                }
        else:
            # Fallback to original method
            for shares in range(10, 1001, 10):
                tier1 = int(shares * 0.30)
                tier2 = int(shares * 0.40)
                tier3 = shares - tier1 - tier2
                
                self.tier_qty_lookup[shares] = {
                    'tier1': tier1,
                    'tier2': tier2,
                    'tier3': tier3,
                    'total': shares
                }
        
        logger.info(f"✓ All Kelly lookup tables initialized (tier lookup: {len(self.tier_qty_lookup)} entries, zero-copy: {self.zero_copy_enabled})")
    
    def calculate_position_size(self, filtered_data, ml_prediction):
        """
        Calculate position size for Polygon client integration
        Takes filtered data and ML prediction, returns position size
        """
        try:
            # Extract data from inputs - handle both dict and object types
            if hasattr(filtered_data, 'get'):
                symbol = filtered_data.get('symbol', 'UNKNOWN')
                current_price = filtered_data.get('price', 0)
            else:
                symbol = getattr(filtered_data, 'symbol', 'UNKNOWN')
                current_price = getattr(filtered_data, 'price', 0)
            
            # Extract ML prediction data - handle both dict and object types
            if hasattr(ml_prediction, 'get'):
                confidence = ml_prediction.get('confidence', 0.5)
                prediction = ml_prediction.get('prediction', 0.0)
                regime = ml_prediction.get('regime', 0)
            else:
                confidence = getattr(ml_prediction, 'confidence', 0.5)
                prediction = getattr(ml_prediction, 'prediction', 0.0)
                regime = getattr(ml_prediction, 'regime', 0)
            
            # Skip if confidence too low or price invalid
            if confidence < 0.6 or current_price <= 0:
                return 0
            
            # Skip if prediction is too weak
            if abs(prediction) < 0.3:
                return 0
            
            # Get portfolio state if available
            available_capital = self.available_capital
            if self.portfolio_manager:
                portfolio_state = self.portfolio_manager.get_portfolio_state()
                available_capital = portfolio_state.get('cash_available', self.available_capital)
            
            # Calculate position size using existing method
            position_result = self.calculate_position_ultra_fast(
                symbol=symbol,
                current_price=current_price,
                confidence=confidence,
                vix_level=20.0,  # Default VIX
                market_cap=1000000000,  # Default market cap
                time_hour=12.0  # Default time
            )
            
            # Extract position size
            if isinstance(position_result, dict):
                total_qty = position_result.get('total_qty', 0)
            else:
                total_qty = position_result if position_result else 0
            
            # Apply prediction direction
            if prediction < 0:
                total_qty = -abs(total_qty)  # Short position
            else:
                total_qty = abs(total_qty)   # Long position
            
            # Apply regime adjustments
            if regime == -1:  # Bear market
                total_qty = int(total_qty * 0.7)  # Reduce size
            elif regime == 1:  # Bull market
                total_qty = int(total_qty * 1.2)  # Increase size
            
            # Final safety checks
            max_position_value = available_capital * 0.1  # Max 10% per position
            max_shares = int(max_position_value / current_price)
            total_qty = max(-max_shares, min(max_shares, total_qty))
            
            return total_qty
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed for {filtered_data.get('symbol', 'UNKNOWN')}: {e}")
            return 0

    def calculate_aggressive_position_size(self, symbol, current_price, confidence,
                                         vix_level=20.0, market_cap=10000000000, time_hour=12.0):
        """
        Calculate aggressive position size for $1000/day strategy
        Target: $2000-4000 positions with 0.5-1% take profits
        """
        start_time = time.perf_counter()
        
        try:
            # Skip confidence check for backtesting - generate positions for all valid signals
            if current_price <= 0:
                return None
            
            # Calculate remaining target for the day
            remaining_target = max(100, self.DAILY_TARGET - self.daily_pnl)  # Minimum $100 target
            remaining_trades = max(1, self.TARGET_TRADES_PER_DAY - self.daily_trades)
            target_per_trade = remaining_target / remaining_trades
            
            # Aggressive position sizing based on daily progress
            if self.daily_pnl < self.DAILY_TARGET * 0.3:  # First 30% of target
                # Aggressive phase - larger positions
                base_position = self.AGGRESSIVE_POSITION_MAX
            elif self.daily_pnl < self.DAILY_TARGET * 0.7:  # Middle 40% of target
                # Steady phase - medium positions
                base_position = (self.AGGRESSIVE_POSITION_MIN + self.AGGRESSIVE_POSITION_MAX) / 2
            else:  # Final 30% of target
                # Conservative phase - smaller positions to preserve gains
                base_position = self.AGGRESSIVE_POSITION_MIN
            
            # Adjust for confidence and market conditions
            confidence_multiplier = max(0.5, 0.7 + (confidence * 0.6))  # 0.5 to 1.3 range
            vix_multiplier = max(0.5, min(1.5, 25.0 / max(vix_level, 10.0)))  # Inverse VIX scaling
            
            # Calculate final position size
            position_dollars = base_position * confidence_multiplier * vix_multiplier
            
            # Ensure within bounds and available cash
            position_dollars = max(self.AGGRESSIVE_POSITION_MIN,
                                 min(self.AGGRESSIVE_POSITION_MAX, position_dollars))
            position_dollars = min(position_dollars, self.available_capital * 0.15)  # Max 15% of capital per trade
            
            # Calculate shares - ensure minimum viable position
            shares = max(self.MIN_SHARES, int(position_dollars / current_price))
            
            # Ensure minimum position value for meaningful trades
            min_position_value = 1000  # $1000 minimum
            if shares * current_price < min_position_value:
                shares = max(self.MIN_SHARES, int(min_position_value / current_price))
            
            actual_position_value = shares * current_price
            
            # Calculate tier quantities for aggressive exits
            tier_quantities = {
                'tier1': int(shares * 0.5),   # 50% for quick 0.5% exit
                'tier2': int(shares * 0.3),   # 30% for 1% exit
                'tier3': int(shares * 0.2),   # 20% for trailing stop
                'total': shares
            }
            
            # Aggressive price targets for quick profits
            prices = {
                'stop_loss': round(current_price * (1.0 - self.STOP_LOSS_PCT), 2),
                'tp1_target': round(current_price * (1.0 + self.TP1_PCT), 2),    # 0.5% quick exit
                'tp2_target': round(current_price * (1.0 + self.TP2_PCT), 2),    # 1% secondary exit
                'trail_percent': 1.0  # Tight 1% trailing stop
            }
            
            processing_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            
            # Create result with aggressive parameters
            result = {
                'symbol': symbol,
                'total_qty': shares,
                'total_value': actual_position_value,
                'tier_quantities': tier_quantities,
                'prices': prices,
                'kelly_fraction': actual_position_value / self.available_capital,
                'confidence_tier': 2 if confidence > 0.8 else 1 if confidence > 0.6 else 0,
                'processing_time_ms': processing_time / 1000,
                'daily_progress': {
                    'current_pnl': self.daily_pnl,
                    'target_remaining': remaining_target,
                    'trades_today': self.daily_trades,
                    'target_per_trade': target_per_trade
                },
                'position_rationale': {
                    'base_size': base_position,
                    'confidence_mult': confidence_multiplier,
                    'vix_mult': vix_multiplier,
                    'phase': 'aggressive' if self.daily_pnl < self.DAILY_TARGET * 0.3 else
                            'steady' if self.daily_pnl < self.DAILY_TARGET * 0.7 else 'conservative'
                }
            }
            
            logger.debug(f"Aggressive Kelly: {symbol} ${actual_position_value:,.0f} "
                        f"({result['position_rationale']['phase']} phase, "
                        f"${self.daily_pnl:.0f}/${self.DAILY_TARGET} daily)")
            
            return result
            
        except Exception as e:
            logger.error(f"Aggressive position sizing error for {symbol}: {e}")
            return None
    
    def update_daily_progress(self, trade_pnl, position_closed=False):
        """Update daily P&L and position tracking"""
        self.daily_pnl += trade_pnl
        
        if position_closed:
            self.current_positions = max(0, self.current_positions - 1)
        else:
            self.daily_trades += 1
            self.current_positions += 1
        
        # Update cash available (simplified)
        self.cash_available = self.initial_capital + self.daily_pnl - (self.current_positions * 3000)  # Estimate
        
        logger.info(f"Daily progress: ${self.daily_pnl:.0f}/${self.DAILY_TARGET} "
                   f"({self.daily_trades} trades, {self.current_positions} open)")
        
        return {
            'daily_pnl': self.daily_pnl,
            'target_progress_pct': (self.daily_pnl / self.DAILY_TARGET) * 100,
            'trades_today': self.daily_trades,
            'open_positions': self.current_positions,
            'cash_available': self.cash_available,
            'target_achieved': self.daily_pnl >= self.DAILY_TARGET
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking for new trading day"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_positions = []
        self.current_positions = 0
        self.cash_available = self.available_capital
        logger.info(f"Daily tracking reset - Ready for new $1000 target day")
    
    def get_daily_target_status(self):
        """Get current status toward daily $1000 target"""
        progress_pct = (self.daily_pnl / self.DAILY_TARGET) * 100
        remaining = max(0, self.DAILY_TARGET - self.daily_pnl)
        trades_remaining = max(0, self.TARGET_TRADES_PER_DAY - self.daily_trades)
        
        return {
            'target': self.DAILY_TARGET,
            'current_pnl': self.daily_pnl,
            'progress_pct': progress_pct,
            'remaining_target': remaining,
            'trades_today': self.daily_trades,
            'trades_remaining': trades_remaining,
            'avg_per_trade_needed': remaining / trades_remaining if trades_remaining > 0 else 0,
            'on_track': progress_pct >= (self.daily_trades / self.TARGET_TRADES_PER_DAY) * 100,
            'target_achieved': self.daily_pnl >= self.DAILY_TARGET
        }
    
    def calculate_position_ultra_fast(self, symbol, current_price, confidence,
                                     vix_level=20.0, market_cap=10000000000, time_hour=12.0, method='binary'):
        """
        Ultra-fast position calculation using specified lookup method
        Methods: 'binary', 'array', 'table'
        """
        start_time = time.perf_counter()
        
        try:
            # Input validation (ultra-fast)
            if current_price <= 0:
                return self._create_fallback_result(symbol, current_price, start_time)
            
            # Choose lookup method
            # Get real ML prediction if available
            ml_confidence = confidence  # Default to provided confidence
            if self.ml_bridge:
                # Try to get symbol index from memory pools
                symbol_to_index = self.memory_pools.get('symbol_to_index', {})
                symbol_idx = symbol_to_index.get(symbol, -1)
                if symbol_idx >= 0:
                    ml_prediction = self.ml_bridge.get_ml_prediction(symbol_idx)
                    if ml_prediction:
                        ml_confidence = ml_prediction['confidence']
                        logger.debug(f"Using ML confidence {ml_confidence:.3f} for {symbol} (was {confidence:.3f})")
            
            # Sync with portfolio manager for real-time cash
            available_capital = self.available_capital
            if self.portfolio_manager:
                portfolio_state = self.portfolio_manager.get_portfolio_state()
                available_capital = portfolio_state['cash_available']
            
            if method == 'binary':
                position_dollars = binary_kelly_lookup(
                    win_rate=0.5 + (ml_confidence * 0.2),  # Use real ML confidence
                    confidence=ml_confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=available_capital
                )
                self.stats['binary_calculations'] += 1
            elif method == 'array':
                position_dollars = ultra_fast_kelly_lookup(
                    win_rate=0.5 + (ml_confidence * 0.2),
                    confidence=ml_confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=available_capital
                )
                self.stats['array_calculations'] += 1
            else:  # table method
                result_dict = get_ultra_fast_kelly_position(
                    win_rate=0.5 + (ml_confidence * 0.2),
                    confidence=ml_confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=available_capital
                )
                position_dollars = result_dict['position_dollars']
                self.stats['table_calculations'] += 1
            
            # Calculate shares using integer operations
            shares = max(self.MIN_SHARES, int(position_dollars / current_price))
            
            # Lookup tier quantities (O(1) operation)
            shares_rounded = (shares // 10) * 10  # Round to nearest 10
            tier_quantities = self.tier_qty_lookup.get(
                shares_rounded, 
                self._calculate_tier_quantities_fast(shares)
            )
            
            # Calculate price levels using pre-computed multipliers
            prices = {
                'stop_loss': round(current_price * self.price_multipliers['stop_loss'], 2),
                'tp1_target': round(current_price * self.price_multipliers['tp1_target'], 2),
                'tp2_target': round(current_price * self.price_multipliers['tp2_target'], 2),
                'trail_percent': self.price_multipliers['trail_percent']
            }
            
            # Calculate actual position value
            actual_position_value = shares * current_price
            kelly_fraction = position_dollars / self.available_capital
            
            # Confidence tier (fast classification)
            if confidence < 0.4:
                confidence_tier = 0  # Low
            elif confidence < 0.7:
                confidence_tier = 1  # Medium
            else:
                confidence_tier = 2  # High
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            self.stats['calculations_made'] += 1
            self.stats['total_time_ms'] += processing_time / 1000  # Convert to ms for compatibility
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['calculations_made']
            self.stats['lookup_hits'] += 1
            
            result = UltraFastKellyResult(
                symbol=symbol,
                total_qty=shares,
                total_value=actual_position_value,
                tier_quantities=tier_quantities,
                prices=prices,
                kelly_fraction=kelly_fraction,
                confidence_tier=confidence_tier,
                processing_time_ms=processing_time / 1000  # Convert to ms
            )
            
            logger.debug(f"Kelly {method}: {symbol} {shares} shares @ {kelly_fraction:.3f} ({processing_time:.1f}μs)")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Kelly calculation error for {symbol}: {e}")
            return self._create_fallback_result(symbol, current_price, start_time)
    
    def calculate_batch_ultra_fast(self, batch_data, method='binary'):
        """
        Ultra-fast batch calculation for multiple positions using zero-copy operations
        Target: Sub-microsecond per position
        """
        start_time = time.perf_counter()
        
        try:
            # Zero-copy batch processing if enabled
            if self.zero_copy_enabled and len(batch_data) > 1:
                return self._calculate_batch_zero_copy(batch_data, method, start_time)
            
            # Fallback to individual calculations
            results = []
            for data in batch_data:
                if len(data) >= 6:
                    symbol, price, confidence, vix, market_cap, time_hour = data[:6]
                else:
                    symbol, price, confidence = data[:3]
                    vix, market_cap, time_hour = 20.0, 10000000000, 12.0
                
                result = self.calculate_position_ultra_fast(
                    symbol, price, confidence, vix, market_cap, time_hour, method
                )
                results.append(result)
            
            batch_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            avg_time_per_position = batch_time / len(batch_data) if batch_data else 0
            
            logger.info(f"✓ Batch Kelly {method} calculation: {len(batch_data)} positions ({avg_time_per_position:.1f}μs avg)")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Batch Kelly calculation error (batch size: {len(batch_data)}): {e}")
            
            # Fallback to individual calculations
            return [self._create_fallback_result(data[0], data[1], start_time)
                   for data in batch_data]
    
    def _calculate_batch_zero_copy(self, batch_data, method, start_time):
        """Zero-copy batch position calculation for maximum performance."""
        try:
            batch_size = len(batch_data)
            
            # Get memory pool references
            position_pool = self.memory_pools.get('position_pool')
            tier_pool = self.memory_pools.get('tier_pool')
            price_pool = self.memory_pools.get('price_pool')
            kelly_results_pool = self.memory_pools.get('kelly_results_pool')
            
            if position_pool is None:
                logger.warning("Zero-copy memory pools not available, falling back to standard processing")
                return []
            
            # Prepare batch input data using vectorized operations
            symbols = []
            prices = np.zeros(batch_size, dtype=np.float64)
            confidences = np.zeros(batch_size, dtype=np.float64)
            vix_levels = np.zeros(batch_size, dtype=np.float64)
            market_caps = np.zeros(batch_size, dtype=np.float64)
            
            for i, data in enumerate(batch_data):
                if len(data) >= 6:
                    symbol, price, confidence, vix, market_cap, time_hour = data[:6]
                else:
                    symbol, price, confidence = data[:3]
                    vix, market_cap, time_hour = 20.0, 10000000000, 12.0
                
                symbols.append(symbol)
                prices[i] = price
                confidences[i] = confidence
                vix_levels[i] = vix
                market_caps[i] = market_cap
            
            # Vectorized Kelly calculations
            win_rates = 0.5 + (confidences * 0.2)  # Convert confidence to win rate
            
            # Vectorized position sizing using NumPy operations
            if method == 'binary':
                position_dollars = self._binary_kelly_vectorized(win_rates, confidences, vix_levels, market_caps)
            elif method == 'array':
                position_dollars = self._array_kelly_vectorized(win_rates, confidences, vix_levels, market_caps)
            else:
                position_dollars = self._table_kelly_vectorized(win_rates, confidences, vix_levels, market_caps)
            
            # Vectorized share calculations
            shares = np.maximum(self.MIN_SHARES, (position_dollars / prices).astype(np.int32))
            actual_values = shares * prices
            kelly_fractions = position_dollars / self.available_capital
            
            # Vectorized tier calculations
            tier1_shares = (shares * 0.30).astype(np.int32)
            tier2_shares = (shares * 0.40).astype(np.int32)
            tier3_shares = shares - tier1_shares - tier2_shares
            
            # Vectorized price level calculations
            stop_losses = prices * (1.0 - self.STOP_LOSS_PCT)
            tp1_targets = prices * (1.0 + self.TP1_PCT)
            tp2_targets = prices * (1.0 + self.TP2_PCT)
            
            # Update memory pools directly (zero-copy)
            for i in range(min(batch_size, len(position_pool))):
                # Update position pool
                position_pool[i, 0] = i  # symbol index
                position_pool[i, 1] = prices[i]
                position_pool[i, 2] = confidences[i]
                position_pool[i, 3] = vix_levels[i]
                position_pool[i, 4] = market_caps[i]
                position_pool[i, 5] = position_dollars[i]
                position_pool[i, 6] = shares[i]
                position_pool[i, 7] = actual_values[i]
                
                # Update tier pool
                tier_pool[i, 0] = tier1_shares[i]
                tier_pool[i, 1] = tier2_shares[i]
                tier_pool[i, 2] = tier3_shares[i]
                tier_pool[i, 3] = shares[i]
                
                # Update price pool
                price_pool[i, 0] = stop_losses[i]
                price_pool[i, 1] = tp1_targets[i]
                price_pool[i, 2] = tp2_targets[i]
                price_pool[i, 3] = self.price_multipliers['trail_percent']
                
                # Update kelly results pool
                kelly_results_pool[i, 0] = kelly_fractions[i]
                kelly_results_pool[i, 1] = 2 if confidences[i] > 0.8 else 1 if confidences[i] > 0.6 else 0
                kelly_results_pool[i, 2] = 0.0  # Will be set below
                kelly_results_pool[i, 3] = self.daily_pnl
                kelly_results_pool[i, 4] = (self.daily_pnl / self.DAILY_TARGET) * 100
                kelly_results_pool[i, 5] = self.cash_available
            
            # Create results from memory pools (zero-copy access)
            results = []
            processing_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            processing_time_per_position = processing_time / batch_size
            
            for i in range(batch_size):
                # Update processing time in memory pool
                if i < len(kelly_results_pool):
                    kelly_results_pool[i, 2] = processing_time_per_position / 1000  # Convert to ms
                
                result = UltraFastKellyResult(
                    symbol=symbols[i],
                    total_qty=int(shares[i]),
                    total_value=float(actual_values[i]),
                    tier_quantities={
                        'tier1': int(tier1_shares[i]),
                        'tier2': int(tier2_shares[i]),
                        'tier3': int(tier3_shares[i]),
                        'total': int(shares[i])
                    },
                    prices={
                        'stop_loss': round(float(stop_losses[i]), 2),
                        'tp1_target': round(float(tp1_targets[i]), 2),
                        'tp2_target': round(float(tp2_targets[i]), 2),
                        'trail_percent': self.price_multipliers['trail_percent']
                    },
                    kelly_fraction=float(kelly_fractions[i]),
                    confidence_tier=int(kelly_results_pool[i, 1]) if i < len(kelly_results_pool) else 1,
                    processing_time_ms=processing_time_per_position / 1000
                )
                results.append(result)
            
            # Update statistics
            self.stats['calculations_made'] += batch_size
            self.stats['total_time_ms'] += processing_time / 1000
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['calculations_made']
            self.stats['lookup_hits'] += batch_size
            
            logger.info(f"✓ Zero-copy batch Kelly calculation: {batch_size} positions ({processing_time_per_position:.1f}μs avg)")
            
            return results
            
        except Exception as e:
            logger.error(f"Zero-copy batch calculation failed: {e}")
            return []
    
    def _binary_kelly_vectorized(self, win_rates, confidences, vix_levels, market_caps):
        """Vectorized binary Kelly lookup for batch processing."""
        # Convert to indices using vectorized operations
        win_rate_ints = (win_rates * 100).astype(np.int32)
        win_indices = np.maximum(0, np.minimum(10, (win_rate_ints - 50) // 2))
        
        confidence_ints = (confidences * 100).astype(np.int32)
        conf_indices = np.maximum(0, np.minimum(4, (confidence_ints - 20) // 20))
        
        vix_ints = vix_levels.astype(np.int32)
        vix_indices = np.maximum(0, np.minimum(4, (vix_ints - 10) // 10))
        
        # Vectorized market cap factors
        market_cap_factors = np.ones_like(market_caps)
        market_cap_factors[market_caps >= 1000000000000] = 1.2  # $1T+
        market_cap_factors[(market_caps >= 100000000000) & (market_caps < 1000000000000)] = 1.1  # $100B+
        market_cap_factors[(market_caps >= 10000000000) & (market_caps < 100000000000)] = 1.0   # $10B+
        market_cap_factors[(market_caps >= 1000000000) & (market_caps < 10000000000)] = 0.9     # $1B+
        market_cap_factors[market_caps < 1000000000] = 0.7  # <$1B
        
        # Vectorized position calculations
        position_percentages = np.zeros_like(win_rates)
        for i in range(len(win_rates)):
            # Lookup from binary table
            array_idx = conf_indices[i] * 4 + vix_indices[i]
            if win_indices[i] < len(KELLY_BINARY_LOOKUP) and array_idx < len(KELLY_BINARY_LOOKUP[0]):
                base_position_packed = KELLY_BINARY_LOOKUP[win_indices[i]][array_idx]
                adjusted_position_packed = (base_position_packed * market_cap_factors[i] * 100) // 100
                position_percentages[i] = adjusted_position_packed / 100.0
        
        # Apply limits
        position_percentages = np.minimum(position_percentages, 30.0)
        position_percentages = np.maximum(position_percentages, 0.5)
        
        return position_percentages * self.available_capital / 100.0
    
    def _array_kelly_vectorized(self, win_rates, confidences, vix_levels, market_caps):
        """Vectorized array Kelly lookup for batch processing."""
        # Similar to binary but using KELLY_POSITION_ARRAY
        win_rate_pcts = np.maximum(50, np.minimum(70, (win_rates * 100).astype(np.int32)))
        win_indices = (win_rate_pcts - 50) // 2
        
        confidence_pcts = np.maximum(20, np.minimum(100, (confidences * 100).astype(np.int32)))
        conf_indices = (confidence_pcts - 20) // 20
        
        vix_rounded = np.maximum(10, np.minimum(50, (vix_levels / 10).astype(np.int32) * 10))
        vix_indices = (vix_rounded - 10) // 10
        
        # Vectorized market cap multipliers
        market_cap_multipliers = np.ones_like(market_caps)
        market_cap_multipliers[market_caps >= 1000000000000] = 1.2
        market_cap_multipliers[(market_caps >= 100000000000) & (market_caps < 1000000000000)] = 1.1
        market_cap_multipliers[(market_caps >= 10000000000) & (market_caps < 100000000000)] = 1.0
        market_cap_multipliers[(market_caps >= 1000000000) & (market_caps < 10000000000)] = 0.9
        market_cap_multipliers[market_caps < 1000000000] = 0.7
        
        # Vectorized position calculations
        position_percentages = np.zeros_like(win_rates)
        for i in range(len(win_rates)):
            if (win_indices[i] < len(KELLY_POSITION_ARRAY) and
                conf_indices[i] < len(KELLY_POSITION_ARRAY[0]) and
                vix_indices[i] < len(KELLY_POSITION_ARRAY[0][0])):
                base_position_pct = KELLY_POSITION_ARRAY[win_indices[i]][conf_indices[i]][vix_indices[i]]
                position_percentages[i] = base_position_pct * market_cap_multipliers[i]
        
        # Apply limits
        position_percentages = np.minimum(position_percentages, 30.0)
        position_percentages = np.maximum(position_percentages, 0.5)
        
        return position_percentages * self.available_capital / 100.0
    
    def _table_kelly_vectorized(self, win_rates, confidences, vix_levels, market_caps):
        """Vectorized table Kelly lookup for batch processing."""
        # Simplified vectorized version of table lookup
        base_kelly_pcts = win_rates * 20.0  # Simplified calculation
        
        # Vectorized confidence multipliers
        confidence_multipliers = 0.7 + (confidences * 0.6)
        
        # Vectorized VIX factors
        vix_factors = np.maximum(0.5, np.minimum(1.5, 25.0 / vix_levels))
        
        # Vectorized market cap factors
        market_cap_factors = np.ones_like(market_caps)
        market_cap_factors[market_caps >= 1000000000000] = 1.2
        market_cap_factors[(market_caps >= 100000000000) & (market_caps < 1000000000000)] = 1.1
        market_cap_factors[(market_caps >= 10000000000) & (market_caps < 100000000000)] = 1.0
        market_cap_factors[(market_caps >= 1000000000) & (market_caps < 10000000000)] = 0.9
        market_cap_factors[market_caps < 1000000000] = 0.7
        
        # Combined calculation
        final_position_pcts = (base_kelly_pcts * confidence_multipliers *
                              vix_factors * market_cap_factors * self.SAFETY_FACTOR)
        
        # Apply limits
        final_position_pcts = np.minimum(final_position_pcts, 30.0)
        final_position_pcts = np.maximum(final_position_pcts, 0.5)
        
        return final_position_pcts * self.available_capital / 100.0
    
    def _calculate_tier_quantities_fast(self, total_shares):
        """Fast tier quantity calculation without numpy"""
        tier1 = int(total_shares * 0.30)
        tier2 = int(total_shares * 0.40)
        tier3 = total_shares - tier1 - tier2  # Ensure total matches
        
        return {
            'tier1': tier1,
            'tier2': tier2,
            'tier3': tier3,
            'total': total_shares
        }
    
    def _create_fallback_result(self, symbol, price, start_time):
        """Create fallback result for error cases"""
        processing_time = (time.perf_counter() - start_time) * 1000
        
        fallback_shares = 50
        fallback_value = fallback_shares * price if price > 0 else 2500
        
        return UltraFastKellyResult(
            symbol=symbol,
            total_qty=fallback_shares,
            total_value=fallback_value,
            tier_quantities={
                'tier1': 15, 'tier2': 20, 'tier3': 15, 'total': fallback_shares
            },
            prices={
                'stop_loss': price * 0.985 if price > 0 else 0,
                'tp1_target': price * 1.01 if price > 0 else 0,
                'tp2_target': price * 1.03 if price > 0 else 0,
                'trail_percent': 2.0
            },
            kelly_fraction=0.05,
            confidence_tier=1,
            processing_time_ms=processing_time
        )
    
    def update_capital_fast(self, new_capital):
        """Fast capital update (lookup tables don't need regeneration)"""
        if abs(new_capital - self.available_capital) / self.available_capital > 0.1:  # 10% change
            self.available_capital = new_capital
            logger.info(f"✓ Capital updated to ${new_capital:,} (lookup tables unchanged)")
    
    def get_performance_stats(self):
        """Get consolidated Kelly position sizer performance statistics with daily target tracking"""
        daily_status = self.get_daily_target_status()
        
        return {
            "calculations_made": self.stats['calculations_made'],
            "avg_time_ms": self.stats['avg_time_ms'],
            "avg_time_microseconds": self.stats['avg_time_ms'] * 1000,
            "target_time_ms": 0.001,  # 1 microsecond target
            "performance_ratio": (
                0.001 / self.stats['avg_time_ms']
                if self.stats['avg_time_ms'] > 0 else float('inf')
            ),
            "lookup_hits": self.stats['lookup_hits'],
            "binary_calculations": self.stats['binary_calculations'],
            "array_calculations": self.stats['array_calculations'],
            "table_calculations": self.stats['table_calculations'],
            "binary_lookup_size": len(KELLY_BINARY_LOOKUP) * len(KELLY_BINARY_LOOKUP[0]),
            "array_lookup_size": len(KELLY_POSITION_ARRAY) * len(KELLY_POSITION_ARRAY[0]) * len(KELLY_POSITION_ARRAY[0][0]),
            "table_lookup_size": len(KELLY_WIN_RATE_LOOKUP),
            "tier_lookup_size": len(self.tier_qty_lookup),
            "available_capital": self.available_capital,
            "gpu_enabled": self.gpu_enabled,
            "lookup_methods": ["binary", "array", "table"],
            
            # Aggressive $1000/day strategy metrics
            "daily_target": self.DAILY_TARGET,
            "daily_pnl": self.daily_pnl,
            "daily_progress_pct": daily_status['progress_pct'],
            "trades_today": self.daily_trades,
            "open_positions": self.current_positions,
            "cash_available": self.cash_available,
            "target_achieved": daily_status['target_achieved'],
            "aggressive_position_range": f"${self.AGGRESSIVE_POSITION_MIN}-${self.AGGRESSIVE_POSITION_MAX}",
            "take_profit_targets": f"{self.TP1_PCT*100:.1f}%-{self.TP2_PCT*100:.1f}%",
            "stop_loss_pct": f"{self.STOP_LOSS_PCT*100:.1f}%",
            "max_daily_positions": self.MAX_DAILY_POSITIONS,
            "target_trades_per_day": self.TARGET_TRADES_PER_DAY
        }
    
    def is_performance_target_met(self):
        """Check if performance target of <1 microsecond is being met"""
        return (self.stats['avg_time_ms'] < 0.001
                if self.stats['avg_time_ms'] > 0 else False)
    
    def validate_daily_target_strategy(self) -> Dict:
        """Validate that Kelly sizing strategy is optimized for $1000+ daily target"""
        validation_results = {
            'daily_target': self.DAILY_TARGET,
            'current_progress': self.daily_pnl,
            'progress_pct': (self.daily_pnl / self.DAILY_TARGET) * 100,
            'target_achieved': self.daily_pnl >= self.DAILY_TARGET,
            'position_sizing_optimal': False,
            'risk_management_active': False,
            'aggressive_strategy_enabled': True,
            'validation_passed': False
        }
        
        try:
            # Check position sizing optimization
            if hasattr(self, 'stats') and self.stats['calculations_made'] > 0:
                avg_position_size = self.stats.get('avg_position_size', 0)
                validation_results['avg_position_size'] = avg_position_size
                validation_results['position_sizing_optimal'] = (
                    self.AGGRESSIVE_POSITION_MIN <= avg_position_size <= self.AGGRESSIVE_POSITION_MAX
                )
            
            # Check risk management
            validation_results['risk_management_active'] = (
                self.current_positions <= self.MAX_POSITIONS and
                self.daily_trades <= self.MAX_DAILY_TRADES
            )
            
            # Check aggressive strategy parameters
            validation_results['aggressive_params'] = {
                'min_position': self.AGGRESSIVE_POSITION_MIN,
                'max_position': self.AGGRESSIVE_POSITION_MAX,
                'max_positions': self.MAX_POSITIONS,
                'max_daily_trades': self.MAX_DAILY_TRADES,
                'current_positions': self.current_positions,
                'daily_trades': self.daily_trades
            }
            
            # Performance validation
            validation_results['performance_metrics'] = {
                'avg_time_ms': self.stats.get('avg_time_ms', 0),
                'calculations_made': self.stats.get('calculations_made', 0),
                'target_time_met': self.is_performance_target_met()
            }
            
            # Overall validation
            validation_results['validation_passed'] = (
                validation_results['position_sizing_optimal'] and
                validation_results['risk_management_active'] and
                validation_results['performance_metrics']['target_time_met']
            )
            
            return validation_results
            
        except Exception as e:
            validation_results['error'] = str(e)
            return validation_results
    
    def optimize_for_daily_target(self, current_time_of_day: float = None) -> Dict:
        """Optimize position sizing strategy based on time of day and progress toward daily target"""
        try:
            if current_time_of_day is None:
                import datetime
                now = datetime.datetime.now()
                # Convert to fraction of trading day (9:30 AM - 4:00 PM EST)
                market_open = 9.5  # 9:30 AM
                market_close = 16.0  # 4:00 PM
                current_hour = now.hour + now.minute / 60.0
                
                if current_hour < market_open:
                    current_time_of_day = 0.0
                elif current_hour > market_close:
                    current_time_of_day = 1.0
                else:
                    current_time_of_day = (current_hour - market_open) / (market_close - market_open)
            
            progress_pct = (self.daily_pnl / self.DAILY_TARGET) * 100
            time_remaining = 1.0 - current_time_of_day
            
            optimization_strategy = {
                'time_of_day': current_time_of_day,
                'progress_pct': progress_pct,
                'time_remaining': time_remaining,
                'recommended_strategy': 'conservative'
            }
            
            # Determine optimal strategy based on progress and time
            if progress_pct >= 100:
                # Target achieved - conservative approach
                optimization_strategy['recommended_strategy'] = 'conservative'
                optimization_strategy['position_multiplier'] = 0.5
                optimization_strategy['risk_level'] = 'low'
                
            elif progress_pct >= 70:
                # Close to target - moderate approach
                optimization_strategy['recommended_strategy'] = 'moderate'
                optimization_strategy['position_multiplier'] = 0.75
                optimization_strategy['risk_level'] = 'medium'
                
            elif time_remaining < 0.3 and progress_pct < 50:
                # Late in day, behind target - aggressive approach
                optimization_strategy['recommended_strategy'] = 'aggressive'
                optimization_strategy['position_multiplier'] = 1.5
                optimization_strategy['risk_level'] = 'high'
                
            else:
                # Normal trading - standard approach
                optimization_strategy['recommended_strategy'] = 'standard'
                optimization_strategy['position_multiplier'] = 1.0
                optimization_strategy['risk_level'] = 'medium'
            
            # Calculate recommended position sizes
            base_min = self.AGGRESSIVE_POSITION_MIN
            base_max = self.AGGRESSIVE_POSITION_MAX
            multiplier = optimization_strategy['position_multiplier']
            
            optimization_strategy['recommended_position_range'] = {
                'min': int(base_min * multiplier),
                'max': int(base_max * multiplier)
            }
            
            return optimization_strategy
            
        except Exception as e:
            return {'error': f'Error optimizing for daily target: {e}'}

# =============================================================================
# GLOBAL FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Global instance for maximum speed
GLOBAL_KELLY_SIZER = UltraFastKellyPositionSizer()

def calculate_kelly_position(symbol, price, confidence, vix_level=20.0, market_cap=10000000000, method='binary'):
    """Global function for ultra-fast Kelly position sizing"""
    return GLOBAL_KELLY_SIZER.calculate_position_ultra_fast(
        symbol, price, confidence, vix_level, market_cap, method=method
    )

if __name__ == "__main__":
    # Test all consolidated lookup methods
    logger.info("Testing consolidated Kelly lookup methods...")
    
    # Performance test for all methods
    import time
    
    test_data = [
        ('AAPL', 150.0, 0.8, 22.0, 3000000000000),
        ('TSLA', 200.0, 0.7, 25.0, 800000000000),
        ('NVDA', 400.0, 0.9, 18.0, 2000000000000),
    ]
    
    methods = ['binary', 'array', 'table']
    
    for method in methods:
        start_time = time.perf_counter()
        
        # Test 1000 lookups for each method
        for i in range(1000):
            for symbol, price, confidence, vix, market_cap in test_data:
                result = calculate_kelly_position(symbol, price, confidence, vix, market_cap, method)
        
        end_time = time.perf_counter()
        avg_time_microseconds = (end_time - start_time) * 1000000 / (1000 * len(test_data))
        
        logger.info(f"{method.upper()} method average time: {avg_time_microseconds:.2f} microseconds")
    
    # Display final statistics
    stats = GLOBAL_KELLY_SIZER.get_performance_stats()
    logger.info(f"Consolidated Kelly sizer stats: {stats}")
    
    logger.info("All Kelly lookup methods consolidated and ready for maximum HFT speed!")
