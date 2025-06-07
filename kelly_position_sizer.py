#!/usr/bin/env python3

# ULTRA-LOW LATENCY KELLY POSITION SIZER WITH CONSOLIDATED LOOKUP TABLES
# All lookup tables hardcoded directly for maximum HFT speed
# Zero import overhead - everything self-contained

import time

# Hardcoded SystemLogger class for maximum speed (no imports)
class SystemLogger:
    def __init__(self, name="kelly_position_sizer"):
        self.name = name
        
    def info(self, message, extra=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] INFO [{self.name}]: {message}")
        if extra:
            print(f"    Extra: {extra}")
    
    def debug(self, message, extra=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] DEBUG [{self.name}]: {message}")
        if extra:
            print(f"    Extra: {extra}")
    
    def warning(self, message, extra=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] WARNING [{self.name}]: {message}")
        if extra:
            print(f"    Extra: {extra}")
    
    def error(self, message, extra=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ERROR [{self.name}]: {message}")
        if extra:
            print(f"    Extra: {extra}")

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
    Ultra-fast Kelly position sizer with all lookup methods consolidated
    Target: Sub-microsecond position sizing for maximum HFT speed
    """
    
    def __init__(self, available_capital=None, gpu_enabled=False):
        # Hardcoded config values for maximum speed
        self.available_capital = available_capital or AVAILABLE_CAPITAL
        self.initial_capital = self.available_capital
        self.gpu_enabled = False  # Pure lookup architecture
        
        logger.info(f"Initializing Aggressive Kelly Position Sizer (capital: ${self.available_capital:,}, Target: ${DAILY_TARGET}/day)")
        
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
    
    def _init_lookup_tables(self):
        """Initialize ultra-fast lookup tables for sub-nanosecond Kelly calculations"""
        
        # Pre-compute tier quantities for common share counts (faster than numpy)
        self.tier_qty_lookup = {}
        for shares in range(10, 1001, 10):  # 10 to 1000 shares in increments of 10
            tier1 = int(shares * 0.30)
            tier2 = int(shares * 0.40)
            tier3 = shares - tier1 - tier2  # Ensure total matches
            
            self.tier_qty_lookup[shares] = {
                'tier1': tier1,
                'tier2': tier2,
                'tier3': tier3,
                'total': shares
            }
        
        logger.info(f"✓ All Kelly lookup tables initialized (tier lookup: {len(self.tier_qty_lookup)} entries)")
    
    def calculate_aggressive_position_size(self, symbol, current_price, confidence,
                                         vix_level=20.0, market_cap=10000000000, time_hour=12.0):
        """
        Calculate aggressive position size for $1000/day strategy
        Target: $2000-4000 positions with 0.5-1% take profits
        """
        start_time = time.perf_counter()
        
        try:
            # Check daily limits
            if self.daily_trades >= self.MAX_DAILY_POSITIONS:
                logger.warning(f"Daily position limit reached: {self.daily_trades}/{self.MAX_DAILY_POSITIONS}")
                return None
            
            if self.current_positions >= 20:  # Max concurrent positions
                logger.warning(f"Max concurrent positions reached: {self.current_positions}/20")
                return None
            
            # Calculate remaining target for the day
            remaining_target = max(0, self.DAILY_TARGET - self.daily_pnl)
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
            confidence_multiplier = 0.7 + (confidence * 0.6)  # 0.7 to 1.3 range
            vix_multiplier = max(0.5, min(1.5, 25.0 / vix_level))  # Inverse VIX scaling
            
            # Calculate final position size
            position_dollars = base_position * confidence_multiplier * vix_multiplier
            
            # Ensure within bounds and available cash
            position_dollars = max(self.AGGRESSIVE_POSITION_MIN,
                                 min(self.AGGRESSIVE_POSITION_MAX, position_dollars))
            position_dollars = min(position_dollars, self.cash_available * 0.2)  # Max 20% of cash per trade
            
            # Calculate shares
            shares = max(self.MIN_SHARES, int(position_dollars / current_price))
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
            if method == 'binary':
                position_dollars = binary_kelly_lookup(
                    win_rate=0.5 + (confidence * 0.2),  # Convert confidence to win rate
                    confidence=confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=self.available_capital
                )
                self.stats['binary_calculations'] += 1
            elif method == 'array':
                position_dollars = ultra_fast_kelly_lookup(
                    win_rate=0.5 + (confidence * 0.2),
                    confidence=confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=self.available_capital
                )
                self.stats['array_calculations'] += 1
            else:  # table method
                result_dict = get_ultra_fast_kelly_position(
                    win_rate=0.5 + (confidence * 0.2),
                    confidence=confidence,
                    vix_level=vix_level,
                    market_cap=market_cap,
                    available_capital=self.available_capital
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
        Ultra-fast batch calculation for multiple positions
        Target: Sub-microsecond per position
        """
        start_time = time.perf_counter()
        results = []
        
        try:
            # Batch calculation using specified method
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
