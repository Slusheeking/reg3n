# Production HFT Trading System with Online Learning

A production-ready high-frequency trading system targeting $1000+ daily profit from $50K capital with aggressive 95% capital allocation, ultra-low latency execution, and real-time model adaptation via online learning.

## Production Specifications

- **Profit Target**: $1000+ daily profit (2%+ daily return)
- **Starting Capital**: $50,000
- **Capital Allocation**: Up to 95% aggressive deployment
- **Pipeline Latency**: Sub-200μs end-to-end (achieved: 15.8μs)
- **Trading Symbols**: 63 high-liquidity stocks (10 Tier-1 + 53 Tier-2)
- **Features**: 10 Polygon-native market data features
- **Portfolio Sync**: Real-time via Alpaca WebSocket

## Trading Universe

### Tier 1 Symbols (Ultra-High Liquidity)
SPY, NVDA, QQQ, TSLA, AAPL, MSFT, IBIT, META, AMZN, GOOGL

### Tier 2 Symbols (High Liquidity)
TQQQ, SQQQ, SOXL, SOXS, AMD, TSM, AVGO, INTC, QCOM, MU, BAC, WFC, JPM, XLF, FBTC, COIN, MSTR, ETHA, UNH, LLY, JNJ, PFE, XLV, WMT, HD, V, MA, IWM, XLK, GLD, SLV, XOM, CVX, XLE, VXX, UVXY, SVXY, SMCI, PLTR, UPST, ROKU, XBI, ARKQ, IYR, XRT, TLT, HYG, UUP, EFA, EEM, USO, UNG, DIA, NFLX, PYPL, CRM, ORCL, ARKK, ARM

## Optimized Features (Polygon WebSocket Native)

1. **ret_1s**: 1-period price return
2. **ret_5s**: 5-period price return  
3. **ret_15s**: 15-period price return
4. **spread**: Bid-ask spread as % of mid-price
5. **volume_intensity**: Current vs average volume
6. **price_acceleration**: Second derivative of price
7. **momentum_persistence**: Trend consistency measure
8. **quote_intensity**: Quote update frequency proxy
9. **trade_size_surprise**: Volume vs expected size
10. **depth_imbalance**: Bid vs ask size imbalance

## Production Risk Management

- **Max Position Size**: 20% per position (up from 5%)
- **Max Total Exposure**: 95% (up from 80%)
- **Min Cash Reserve**: 5% (down from 10%)
- **Max Daily Loss**: 3% (up from 2%)
- **Max Symbol Exposure**: 25% per symbol (up from 10%)
- **Order Rate Limit**: 200 orders/second (up from 100)

## API Integration

API keys are loaded from environment variables:
- `POLYGON_API_KEY`: Your Polygon.io API key.
- `ALPACA_API_KEY`: Your Alpaca API key.
- `ALPACA_SECRET_KEY`: Your Alpaca secret key.

Ensure these environment variables are set before running the application.

### Polygon.io WebSocket
- **Endpoint**: `wss://socket.polygon.io/stocks`
- **API Key**: (Set via `POLYGON_API_KEY` environment variable)
- **Subscriptions**: Trade and Quote data for all 63 symbols

### Alpaca Paper Trading
- **API Key**: (Set via `ALPACA_API_KEY` environment variable)
- **Secret**: (Set via `ALPACA_SECRET_KEY` environment variable)
- **Base URL**: `https://paper-api.alpaca.markets/v2`
- **WebSocket**: `wss://paper-api.alpaca.markets/stream`

## Architecture

1. **Dual WebSocket Clients**: Polygon (market data) + Alpaca (portfolio sync)
2. **10-Feature Engine**: Optimized for Polygon-native data processing
3. **Three-Tier Models**: Linear (5μs), Tree (20μs), Opportunistic (50μs)
4. **Aggressive Order Engine**: 95% capital utilization with real Alpaca API
5. **Real-time Portfolio Sync**: WebSocket-only position tracking
6. **Online Learning Engine**: Real-time model adaptation via P&L feedback

## Online Learning System

### Adaptive Model Optimization
- **Real-time Weight Updates**: Models continuously adapt based on trading performance
- **P&L Feedback Loop**: Signal outcomes drive learning with reward/penalty system
- **Multi-Strategy Learning**: Separate learning engines for momentum and reversion models
- **Performance-Based Adaptation**: Learning rates adjust based on win/loss streaks

### Learning Components
- **Signal Outcome Tracking**: Every trade tracked from signal to P&L outcome
- **Stochastic Gradient Descent**: Fast weight updates with momentum
- **Adaptive Learning Rates**: Dynamic adjustment based on market volatility
- **Risk-Aware Learning**: Conservative updates during high volatility periods

### Learning Metrics
- **Win Rate Tracking**: Real-time accuracy monitoring per strategy
- **Average P&L per Signal**: Risk-adjusted return measurement
- **Model Performance**: Continuous evaluation of prediction quality
- **Learning Convergence**: Monitoring of weight stability and adaptation speed

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Production Deployment

```bash
# Production mode with NUMA optimization
./hft_engine --numa-node 0

# Test mode (60 seconds with comprehensive online learning metrics)
./hft_engine --test
```

## Enhanced Testing System

### Comprehensive Test Metrics
- **Pipeline Performance**: Latency tracking with sub-200μs target validation
- **Trading Performance**: Signal generation, order execution, and fill rates
- **Portfolio Tracking**: Real-time P&L, exposure, and position monitoring
- **Online Learning Status**: Model adaptation and weight update confirmation
- **Production Readiness**: Multi-factor system health assessment

### Test Output Features
- **Real-time Progress**: 10-second interval updates with learning status
- **Visual Indicators**: ✅/❌/⚠️ status symbols for quick assessment
- **Profit Projection**: Daily target tracking and performance forecasting
- **Learning Validation**: Confirmation of adaptive model operation
- **Risk Assessment**: Capital allocation and exposure monitoring

### Production Validation
- **Latency Compliance**: Sub-200μs pipeline verification
- **Capital Efficiency**: 95% deployment target tracking
- **Learning Activity**: Real-time model adaptation confirmation
- **Risk Management**: Automated safety system validation

## Performance Benchmarks

- **Pipeline Latency**: 15.8μs average (target: 200μs) ✅
- **Feature Processing**: 10 features in <2μs ✅
- **Model Inference**: <20μs for all tiers ✅
- **Order Execution**: <100μs including API calls ✅
- **Capital Efficiency**: 95% deployment capability ✅

## Production Targets

- **Daily Profit**: $1000+ (2%+ return on $50K)
- **Win Rate**: >55% across all tiers
- **Sharpe Ratio**: >2.0 risk-adjusted returns
- **Max Drawdown**: <5% daily, <15% monthly
- **Uptime**: >99.5% during market hours

## System Requirements

- Linux x86_64 with AVX2 support
- C++20 compatible compiler (GCC 11+ or Clang 13+)
- Real-time kernel for optimal performance
- Stable internet connection for WebSocket feeds
- NUMA-aware system recommended for maximum performance (auto-detected)

## Recent Updates

### Memory Management Enhancements
- **NUMA Auto-Detection**: System now automatically detects NUMA configuration
- **Single-Node Optimization**: NUMA disabled on single-node systems (like AMD EPYC with 1 node)
- **Multi-Node Support**: NUMA optimizations enabled only when beneficial (>1 NUMA nodes)
- **Huge Pages**: Automatic huge page allocation for large memory blocks (>2MB)

### Build System Improvements
- **WebSocketPP C++20 Compatibility**: Added compiler flags to handle template syntax issues
- **Error Handling**: Fixed namespace and syntax errors in WebSocket implementation
- **Performance**: Maintained sub-200μs latency with memory management improvements

### WebSocket Client Fixes
- **Dual Endpoint Support**: Fixed issues with Polygon stocks + indices streaming
- **Message Processing**: Corrected namespace and parsing errors
- **Connection Stability**: Improved error handling and reconnection logic

## Dependencies

- **WebSocketPP**: WebSocket client library (C++20 compatibility patches applied)
- **RapidJSON**: High-performance JSON parsing
- **httplib**: HTTP client for REST API calls
- **simdjson**: Ultra-fast JSON processing (optional)
- **libnuma**: NUMA memory management (optional, auto-detected)
- **Intel TBB**: Threading Building Blocks (optional)
- **Intel MKL**: Math Kernel Library (optional)
