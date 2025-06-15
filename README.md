# Lag-Llama Trading System

A sophisticated algorithmic trading system that combines **Lag-Llama AI forecasting** with professional trading strategies, real-time market data, and automated execution.

## 🚀 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GH200 TRADING SYSTEM                        │
├─────────────────┬──────────────────┬──────────────────┬─────────┤
│   DATA LAYER    │   AI/ML LAYER    │  STRATEGY LAYER  │ EXECUTION│
│                 │                  │                  │  LAYER  │
│ ┌─────────────┐ │ ┌──────────────┐ │ ┌──────────────┐ │ ┌─────┐ │
│ │  POLYGON.IO │ │ │  LAG-LLAMA   │ │ │ GAP & GO     │ │ │ALPACA│ │
│ │             │ │ │              │ │ │              │ │ │     │ │
│ │ • REST API  │ │ │ • Forecasts  │ │ │ • Entry Sigs │ │ │• API│ │
│ │ • WebSocket │►│ │ • Confidence │►│ │ • Position   │►│ │• WS │ │
│ │ • Level 1   │ │ │ • Quantiles  │ │ │   Sizing     │ │ │• OMS│ │
│ │ • Indices   │ │ │ • Risk Metrics│ │ │              │ │ │     │ │
│ │             │ │ │              │ │ │ ┌──────────────┤ │ │     │ │
│ └─────────────┘ │ └──────────────┘ │ │ │ ORB STRATEGY │ │ │     │ │
│                 │                  │ │ │              │ │ │     │ │
│                 │                  │ │ │ • Breakouts  │ │ │     │ │
│                 │                  │ │ │ • Range Det. │ │ │     │ │
│                 │                  │ │ └──────────────┤ │ │     │ │
│                 │                  │ │ ┌──────────────┤ │ │     │ │
│                 │                  │ │ │MEAN REVERSION│ │ │     │ │
│                 │                  │ │ │              │ │ │     │ │
│                 │                  │ │ │ • Volatility │ │ │     │ │
│                 │                  │ │ │ • Pairs      │ │ │     │ │
│                 │                  │ └─┴──────────────┘ │ └─────┘ │
└─────────────────┴──────────────────┴──────────────────┴─────────┘
```

## ✨ Key Features

### 🤖 AI-Powered Forecasting
- **Lag-Llama Foundation Model**: First open-source time series foundation model
- **Probabilistic Predictions**: Get probability distributions, not just point estimates
- **Multi-Timeframe Analysis**: 5-minute to 2-hour forecasts
- **Confidence Scoring**: Know when to trust the predictions

### 📊 Professional Trading Strategies
- **Gap & Go**: Trade gap continuations with AI confirmation
- **Opening Range Breakout (ORB)**: Enhanced breakout detection with sustainability forecasting
- **Mean Reversion**: Volatility-based reversion with optimal timing

### 🔧 Enterprise-Grade Infrastructure
- **GH200 Optimized**: Designed for NVIDIA Grace Hopper Superchip
- **Real-Time Data**: Polygon.io professional market data
- **Institutional Execution**: Alpaca Markets API integration
- **Risk Management**: Dynamic position sizing and stop losses

### 📈 Advanced Analytics
- **Position Sizing**: Kelly Criterion with confidence adjustments
- **Risk Metrics**: VaR, Expected Shortfall, correlation monitoring
- **Performance Tracking**: Real-time P&L and strategy analytics

## 🛠️ Installation

### Prerequisites

- **Hardware**: NVIDIA GH200 Grace Hopper Superchip (or H100/A100)
- **OS**: Ubuntu 22.04+ (recommended)
- **Python**: 3.10+
- **CUDA**: 12.1+
- **Memory**: 32GB+ RAM recommended

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/lag-llama-trading
cd lag-llama-trading
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Install Lag-Llama (if not in PyPI)
git clone https://github.com/time-series-foundation-models/lag-llama.git
cd lag-llama
pip install -e .
cd ..
```

### 3. Download Model

```bash
# Download Lag-Llama model checkpoint
wget https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt
```

### 4. Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit with your API keys
nano .env
```

Add your API credentials:
```env
# Polygon.io
POLYGON_API_KEY=your_polygon_api_key

# Alpaca Markets
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Trading Environment
TRADING_ENV=paper  # or 'live' for real trading
```

## 🚀 Quick Start

### 1. Test Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import polygon; print('Polygon OK')"
python -c "import alpaca_trade_api; print('Alpaca OK')"
```

### 2. Configure Settings

Edit `settings.py` to customize:

```python
# Key settings to adjust
config.lag_llama.context_length = 512  # Increase for more history
config.risk.max_position_size = 0.05   # 5% max position size
config.gap_and_go.min_gap_percent = 0.02  # 2% minimum gap
```

### 3. Run the System

```bash
# Paper trading (safe)
TRADING_ENV=paper python main.py

# Live trading (real money - be careful!)
TRADING_ENV=live python main.py
```

## 📋 File Structure

```
REG3N
├── main.py
├── settings.py
├── active_symbols.py
├── lag_llama.py
├── polygon_client.py
├── alpaca_client.py
├── gap_n_go.py
├── orb.py
├── mean_reversion.py
├── requirements.txt
├── README.md
├── .env
└── logs/
    ├── trading_system.log
    └── daily_reports/
```

## 🎯 Trading Strategies

### Gap & Go Strategy

**Concept**: Trade stocks that gap significantly at market open
- **Entry**: Gap >2% with volume confirmation + Lag-Llama continuation probability >75%
- **Exit**: Target levels based on gap size or time-based exit
- **Risk**: Stop below previous day's close

```python
# Example Gap & Go signal
{
    'symbol': 'AAPL',
    'action': 'BUY',
    'confidence': 0.85,
    'gap_percent': 3.2,
    'continuation_probability': 0.78,
    'target_price': 185.50,
    'stop_loss': 179.25
}
```

### Opening Range Breakout (ORB)

**Concept**: Trade breakouts from the first 15 minutes of trading
- **Entry**: Price breaks above/below opening range with volume + Lag-Llama sustainability >75%
- **Exit**: Range expansion targets (0.5x, 1x, 1.5x range size)
- **Risk**: Stop on opposite side of range

```python
# Example ORB signal
{
    'symbol': 'TSLA',
    'action': 'BUY',
    'opening_range': {'high': 240.50, 'low': 238.20},
    'breakout_price': 240.75,
    'sustainability_probability': 0.82,
    'targets': [242.65, 243.80, 245.95]
}
```

### Mean Reversion Strategy

**Concept**: Trade overbought/oversold conditions back to mean
- **Entry**: RSI <30 or >70 + Z-score >2 + Lag-Llama reversion probability >70%
- **Exit**: Return to 20-period SMA or Bollinger middle band
- **Risk**: Stop beyond recent extreme

```python
# Example Mean Reversion signal
{
    'symbol': 'SPY',
    'action': 'BUY',
    'rsi': 25.3,
    'z_score': -2.4,
    'reversion_probability': 0.74,
    'target_mean': 445.20,
    'expected_reversion_time': 35  # minutes
}
```

## 📊 System Monitoring

### Real-Time Dashboard

The system provides comprehensive logging:

```
2024-01-15 09:30:00 - System Status - Uptime: 2:30:15
2024-01-15 09:30:00 - Portfolio: $100,000.00 | P&L: $1,250.00 | Trades: 12
2024-01-15 09:30:00 - Data: 15,432 msgs | Forecasts: 347 | Errors: 0
2024-01-15 09:30:00 - GAP_AND_GO: 5 trades, $650.00 P&L
2024-01-15 09:30:00 - ORB: 4 trades, $425.00 P&L
2024-01-15 09:30:00 - MEAN_REVERSION: 3 trades, $175.00 P&L
```

### Daily Reports

Automatic daily reports saved to `daily_report_YYYYMMDD.json`:

```json
{
  "date": "2024-01-15",
  "portfolio": {
    "account_equity": 101250.00,
    "daily_pnl": 1250.00,
    "daily_trades": 12,
    "win_rate": 0.75
  },
  "strategies": {
    "gap_and_go": {"trades": 5, "pnl": 650.00, "win_rate": 0.80},
    "orb": {"trades": 4, "pnl": 425.00, "win_rate": 0.75},
    "mean_reversion": {"trades": 3, "pnl": 175.00, "win_rate": 0.67}
  }
}
```

## ⚙️ Configuration

### Risk Management

```python
# In settings.py
config.risk.max_portfolio_risk = 0.10    # 10% max portfolio risk
config.risk.max_position_size = 0.05     # 5% max single position
config.risk.max_daily_loss = 0.03        # 3% max daily loss
config.risk.max_drawdown = 0.15          # 15% max drawdown
```

### Strategy Tuning

```python
# Gap & Go
config.gap_and_go.min_gap_percent = 0.02           # 2% minimum gap
config.gap_and_go.gap_continuation_threshold = 0.75 # 75% probability

# ORB
config.orb.range_minutes = 15                      # 15-minute range
config.orb.sustainability_threshold = 0.75         # 75% sustainability

# Mean Reversion
config.mean_reversion.deviation_threshold = 2.0    # 2 standard deviations
config.mean_reversion.reversion_probability = 0.70 # 70% reversion prob
```

### Lag-Llama Optimization

```python
# For GH200 hardware
config.lag_llama.context_length = 512        # Use more context
config.lag_llama.batch_size = 32            # Larger batches
config.lag_llama.rope_scaling_factor = 8.0  # Scale for longer context
```

## 🔒 Risk Management

### Built-in Safeguards

1. **Position Limits**: Maximum 5% per position, 10% total risk
2. **Daily Loss Limit**: Automatic shutdown at 3% daily loss
3. **Dynamic Stops**: Volatility-adjusted stop losses
4. **Correlation Monitoring**: Prevent over-concentration
5. **Emergency Shutdown**: Immediate closure on risk violations

### Risk Monitoring

```python
# Real-time risk checks
if daily_pnl < -max_daily_loss * account_equity:
    emergency_shutdown()

if position_correlation > 0.7:
    warn_high_correlation()

if vix > 30:
    reduce_position_sizes()
```

## 📈 Performance Optimization

### GH200 Specific

- **Unified Memory**: Direct CPU-GPU data sharing
- **Mixed Precision**: BF16 for faster inference
- **Batch Processing**: Process multiple symbols simultaneously
- **Context Scaling**: Use RoPE scaling for longer sequences

### General Optimization

- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Reuse WebSocket connections
- **Data Caching**: Cache forecasts and historical data
- **Memory Management**: Efficient buffer management

## 🚨 Important Disclaimers

### ⚠️ Trading Risks

- **Past Performance**: Does not guarantee future results
- **Market Risk**: All trading involves risk of loss
- **System Risk**: Technology failures can cause losses
- **Model Risk**: AI predictions may be incorrect

### 🧪 Testing Recommendations

1. **Start with Paper Trading**: Test extensively before live trading
2. **Small Position Sizes**: Begin with minimal risk
3. **Monitor Closely**: Watch system behavior carefully
4. **Backtest Thoroughly**: Validate strategies on historical data

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [Polygon.io API Docs](https://polygon.io/docs)
- [Alpaca API Docs](https://alpaca.markets/docs)

### Community
- [GitHub Issues](https://github.com/your-repo/lag-llama-trading/issues)
- [Discord Server](https://discord.gg/your-server)

### Emergency Contact
- **System Issues**: Create a GitHub issue
- **Trading Issues**: Contact your broker directly
- **Critical Bugs**: Email support@your-domain.com

---

## 🎉 Get Started

Ready to revolutionize your trading with AI? 

```bash
git clone https://github.com/your-repo/lag-llama-trading
cd lag-llama-trading
pip install -r requirements.txt
python main.py
```

**Happy Trading! 🚀📈**

---

*Disclaimer: This software is for educational and research purposes. Trading involves substantial risk. The authors are not responsible for any financial losses incurred through use of this system.*