# Simple HFT System

A streamlined, fail-fast HFT trading system that runs completely separate from the main complex system.

## Overview

This simplified system focuses on the core trading pipeline:
**Polygon WebSocket → Feature Engineering → Model Predictions → Alpaca Bracket Orders**

### Key Design Principles

- **Fail-Fast**: Any error immediately stops the system to prevent bad trades
- **Single-Threaded**: No complex threading or synchronization
- **Fixed Weights**: No online learning - models use static, pre-tuned weights
- **Simple Error Handling**: Clear error propagation and system shutdown
- **No State Management**: Minimal state tracking

## System Architecture

```
Polygon WebSocket (Real-time data)
    ↓
Feature Calculator (10 features)
    ↓
Linear Model (Momentum & Mean Reversion)
    ↓
Signal Selection (Best signal > 65% confidence)
    ↓
Alpaca Bracket Orders (Entry + Take Profit + Stop Loss)
```

## Files Created

### Core System Files
- `simple_main.cpp` - Main entry point with simple loop
- `simple_hft.hpp` - Core HFT engine coordinator
- `simple_config.hpp` - Configuration using existing API keys and symbols
- `simple_models.hpp` - Fixed-weight prediction models and feature calculator
- `simple_polygon.hpp` - Basic Polygon WebSocket client
- `simple_alpaca.hpp` - Basic Alpaca HTTP client for bracket orders

### Build & Run Scripts
- `build_simple.sh` - Build script with dependency checking
- `run_simple.sh` - Run script with pre-flight checks and logging

## Configuration

### Trading Parameters (Fixed)
- **Confidence Threshold**: 65% minimum to trade
- **Position Size**: 2% of portfolio per trade
- **Take Profit**: 0.8% target
- **Stop Loss**: 1.5% limit
- **Max Positions**: 5 concurrent positions
- **Portfolio Value**: $50,000 (fixed)

### Symbols
Uses all 25 symbols from existing config:
SPY, NVDA, QQQ, TSLA, AAPL, MSFT, META, AMZN, GOOGL, AMD, TQQQ, SOXL, BAC, COIN, UNH, WMT, V, IWM, GLD, XOM, VXX, TLT, EEM, DIA, ARKK

### API Keys
Reuses existing API keys from `config.hpp`:
- Polygon API: `Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57`
- Alpaca API Key: `PKOA0DZRDVPMC7V6A4EU`
- Alpaca Secret: `1BM2AiVp8N6Glbc5fm14up7KEh1V8KNFleD5jgYu`

## Installation & Setup

### 1. Install Dependencies
```bash
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev rapidjson-dev libwebsocketpp-dev libssl-dev libboost-all-dev
```

### 2. Build the System
```bash
chmod +x build_simple.sh
./build_simple.sh
```

### 3. Run the System

#### Test Mode (Recommended First)
```bash
chmod +x run_simple.sh
./run_simple.sh --test                    # Run for 60 seconds
./run_simple.sh --test --duration 300     # Run for 5 minutes
```

#### Production Mode
```bash
./run_simple.sh                           # Run indefinitely with logging
./run_simple.sh --no-log                  # Run without log file
```

## Features Calculated

The system calculates 10 features from market data:

1. **ret_1s** - 1-second return
2. **ret_5s** - 5-second return  
3. **ret_15s** - 15-second return
4. **spread** - Bid-ask spread as % of mid price
5. **volume_intensity** - Current vs average volume
6. **price_acceleration** - Second derivative of price
7. **momentum_persistence** - Trend consistency
8. **quote_intensity** - Bid size ratio
9. **trade_size_surprise** - Volume vs recent average
10. **depth_imbalance** - Bid vs ask size imbalance

## Model Strategy

### Linear Model with Fixed Weights

**Momentum Strategy:**
- Positive weights on returns (0.35, 0.25, 0.15)
- Negative weight on spread (-0.20)
- Positive weights on momentum indicators

**Mean Reversion Strategy:**
- Negative weights on returns (-0.35, -0.25, -0.18)
- Positive weight on spread (0.15)
- Mixed weights for reversion signals

**Signal Selection:**
- Both strategies run in parallel
- Best signal (highest confidence) is chosen
- Must exceed 65% confidence threshold
- Rate limited: 30 seconds between signals per symbol

## Bracket Order Structure

Each trade places a 3-leg bracket order:
1. **Entry Order**: Limit order at current market price
2. **Take Profit**: Limit order at +0.8% from entry
3. **Stop Loss**: Stop-limit order at -1.5% from entry

## Error Handling & Fail-Fast

### System Shutdown Triggers
- 3+ errors from any component
- Lost connection to Polygon WebSocket
- Failed authentication with Polygon or Alpaca
- Any unhandled exception in main loop

### Risk Controls
- Maximum 5 concurrent positions
- No duplicate positions in same symbol
- Minimum buying power checks
- Invalid feature validation (no NaN/Inf values)

## Logging

### Console Output
- Real-time status messages
- Trade confirmations
- Error messages
- Periodic statistics

### Log Files
Saved in `logs/` directory with timestamp:
- `logs/simple_hft_YYYYMMDD_HHMMSS.log`

### Monitoring Commands
```bash
# Monitor live log
tail -f logs/simple_hft_*.log

# Search for errors
grep ERROR logs/simple_hft_*.log

# Check trade results
grep "Order successful\|Order failed" logs/simple_hft_*.log
```

## Differences from Main System

### Removed Complexity
- ❌ CPU optimization and thread pinning
- ❌ Online learning engines
- ❌ Market regime detection  
- ❌ Advanced risk management
- ❌ Memory pools and NUMA optimization
- ❌ Multi-threading
- ❌ Order queues and priorities
- ❌ Performance tracking
- ❌ WebSocket reconnection logic
- ❌ Complex error recovery

### Simplified Components
- ✅ Basic Polygon WebSocket (fail-fast on disconnect)
- ✅ Simple feature calculation (10 features only)
- ✅ Fixed-weight model predictions
- ✅ Direct Alpaca HTTP calls (synchronous)
- ✅ Basic error logging
- ✅ Single-threaded execution

## Testing Strategy

1. **Start with Test Mode**: Run `--test` first to verify connectivity
2. **Monitor Logs**: Watch for any errors or unexpected behavior
3. **Check Account**: Verify Alpaca paper trading account status
4. **Gradual Rollout**: Start with short test periods, increase duration
5. **Compare with Main System**: Validate against existing complex system

## Troubleshooting

### Common Issues

**Build Failures:**
- Install missing dependencies listed in build error
- Ensure C++17 compiler support

**Connection Errors:**
- Check internet connectivity
- Verify API keys are correct and active
- Ensure Polygon/Alpaca services are operational

**No Signals Generated:**
- Check that symbols have sufficient market data
- Verify confidence threshold isn't too high
- Monitor feature calculation for valid values

**Order Failures:**
- Check Alpaca account buying power
- Verify market hours (system works in extended hours)
- Review position limits (max 5 positions)

### Debug Commands
```bash
# Test connectivity only
curl -H "Authorization: Bearer POLYGON_API_KEY" "https://api.polygon.io/v1/last/stocks/SPY"

# Check Alpaca account
curl -H "APCA-API-KEY-ID: ALPACA_KEY" -H "APCA-API-SECRET-KEY: ALPACA_SECRET" "https://paper-api.alpaca.markets/v2/account"

# Monitor system resources
top -p $(pgrep simple_hft)
```

## Performance Expectations

This simplified system prioritizes:
- **Reliability** over speed
- **Simplicity** over optimization  
- **Fail-fast behavior** over recovery
- **Clear debugging** over complex features

Expected performance:
- Message processing: ~1000-5000 msg/sec
- Signal generation: ~1-10 signals/minute
- Order placement: <1 second per order
- Memory usage: <100MB
- CPU usage: <10% single core

## Next Steps

After successful testing:
1. Monitor performance vs main system
2. Tune model weights based on results
3. Add specific error recovery if needed
4. Consider additional risk controls
5. Evaluate for production deployment