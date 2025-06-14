#!/usr/bin/env python3
"""
Test script for the comprehensive logging system

Demonstrates all logging features:
- Trade logging with structured data
- Performance metrics logging
- Strategy decision logging
- Error handling and logging
- Log monitoring and statistics
"""

import time
import numpy as np
import asyncio
from utils.logger import setup_logging, get_logger, log_trade, log_performance, log_strategy

def test_basic_logging():
    """Test basic logging functionality"""
    print("🔧 Testing Basic Logging...")
    
    # Setup logging
    logger_system = setup_logging(log_level="DEBUG", enable_console=True)
    
    # Get different logger categories
    system_logger = get_logger("system")
    trade_logger = get_logger("trades")
    strategy_logger = get_logger("strategies")
    error_logger = get_logger("errors")
    
    # Test different log levels
    system_logger.info("System initialization completed")
    system_logger.warning("This is a warning message")
    system_logger.debug("Debug information for development")
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        error_logger.error(f"Caught test error: {e}")
    
    print("   ✓ Basic logging test completed")
    return logger_system

def test_trade_logging():
    """Test trade-specific logging"""
    print("\n💰 Testing Trade Logging...")
    
    # Simulate various trade scenarios
    trades = [
        {"symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 150.25, "strategy": "GapAndGo"},
        {"symbol": "MSFT", "side": "SELL", "quantity": 50, "price": 415.80, "strategy": "ORB"},
        {"symbol": "GOOGL", "side": "BUY", "quantity": 25, "price": 2750.00, "strategy": "MeanReversion"},
        {"symbol": "NVDA", "side": "SELL", "quantity": 75, "price": 875.50, "strategy": "GapAndGo"},
    ]
    
    for trade in trades:
        log_trade(f"Order executed: {trade['strategy']} strategy", **trade)
        time.sleep(0.1)  # Small delay to show timing
    
    print("   ✓ Trade logging test completed")

def test_performance_logging():
    """Test performance metrics logging"""
    print("\n📊 Testing Performance Logging...")
    
    # Simulate various performance metrics
    metrics = [
        ("latency_ms", np.random.uniform(1.0, 5.0), "ms", "Signal generation latency"),
        ("throughput", np.random.uniform(800, 1200), "signals/sec", "Signal processing rate"),
        ("accuracy", np.random.uniform(0.75, 0.95), "", "Model prediction accuracy"),
        ("sharpe_ratio", np.random.uniform(1.5, 3.0), "", "Strategy performance"),
        ("max_drawdown", np.random.uniform(0.02, 0.08), "", "Risk metric"),
        ("cache_hit_rate", np.random.uniform(0.80, 0.98), "", "Feature cache efficiency"),
    ]
    
    for metric, value, unit, description in metrics:
        log_performance(metric, value, unit, description)
        time.sleep(0.05)
    
    print("   ✓ Performance logging test completed")

def test_strategy_logging():
    """Test strategy decision logging"""
    print("\n📈 Testing Strategy Logging...")
    
    strategies = [
        ("GapAndGo", "Signal generated", "INFO", {
            "gap_percent": 0.035, "volume_ratio": 2.1, "confidence": 0.82
        }),
        ("ORB", "Breakout detected", "INFO", {
            "range_expansion": 1.5, "volume_surge": 3.2, "momentum": 0.67
        }),
        ("MeanReversion", "Reversion opportunity", "INFO", {
            "volatility_ratio": 2.8, "reversion_prob": 0.74, "zscore": -2.1
        }),
        ("GapAndGo", "Risk threshold exceeded", "WARNING", {
            "gap_percent": 0.12, "risk_score": 0.95, "action": "skip"
        }),
        ("ORB", "False breakout detected", "WARNING", {
            "volume_ratio": 0.8, "momentum_decay": 0.3, "action": "exit"
        }),
    ]
    
    for strategy_name, message, level, data in strategies:
        log_strategy(strategy_name, message, level=level, **data)
        time.sleep(0.1)
    
    print("   ✓ Strategy logging test completed")

def test_ai_model_logging():
    """Test AI model logging"""
    print("\n🤖 Testing AI Model Logging...")
    
    ai_logger = get_logger("ai")
    
    models = [
        ("LagLlama", "Prediction completed", {"confidence": 0.87, "inference_time_ms": 12.5}),
        ("Chronos", "Forecast generated", {"horizon": 60, "mape": 0.045}),
        ("TimesFM", "Pattern detected", {"pattern_strength": 0.72, "lookback": 120}),
        ("FastModels", "Ensemble prediction", {"models_count": 5, "agreement": 0.91}),
    ]
    
    for model_name, message, data in models:
        formatted_message = f"[{model_name}] {message}"
        if data:
            formatted_message += f" | Data: {data}"
        ai_logger.info(formatted_message)
        time.sleep(0.1)
    
    print("   ✓ AI model logging test completed")

def test_market_data_logging():
    """Test market data logging"""
    print("\n📈 Testing Market Data Logging...")
    
    logger_system = setup_logging()
    
    # Simulate market data updates
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    
    for symbol in symbols:
        price = np.random.uniform(100, 300)
        volume = np.random.randint(1000000, 5000000)
        
        logger_system.log_market_data(
            f"Price update received",
            {"symbol": symbol, "price": price, "volume": volume},
            level="DEBUG"
        )
        time.sleep(0.05)
    
    print("   ✓ Market data logging test completed")

def test_error_scenarios():
    """Test error logging scenarios"""
    print("\n🚨 Testing Error Scenarios...")
    
    error_logger = get_logger("errors")
    
    # Simulate various error conditions
    errors = [
        ("Connection timeout to data provider", "ERROR"),
        ("Invalid order parameters detected", "ERROR"),
        ("Model inference failed", "CRITICAL"),
        ("Risk limit exceeded", "WARNING"),
        ("API rate limit reached", "WARNING"),
    ]
    
    for error_msg, level in errors:
        if level == "ERROR":
            error_logger.error(error_msg)
        elif level == "CRITICAL":
            error_logger.critical(error_msg)
        elif level == "WARNING":
            error_logger.warning(error_msg)
        time.sleep(0.1)
    
    print("   ✓ Error scenario testing completed")

def test_log_monitoring():
    """Test log monitoring and statistics"""
    print("\n📋 Testing Log Monitoring...")
    
    logger_system = setup_logging()
    
    # Generate some activity
    for i in range(10):
        log_trade(f"Test trade {i}", symbol="TEST", side="BUY", quantity=100, price=150.0)
        log_performance("test_metric", np.random.uniform(1, 10), "units", f"Test metric {i}")
        if i % 3 == 0:
            get_logger("errors").error(f"Test error {i}")
    
    # Get statistics
    stats = logger_system.get_log_statistics()
    print("   📊 Log Statistics:")
    for key, value in stats.items():
        print(f"      {key}: {value}")
    
    # Get performance stats
    perf_stats = logger_system.get_performance_stats("test_metric")
    if perf_stats:
        print(f"\n   📈 Performance Stats for 'test_metric':")
        for key, value in perf_stats.items():
            print(f"      {key}: {value}")
    
    # Get recent logs
    recent_logs = logger_system.get_recent_logs(5)
    print(f"\n   📋 Recent Logs (last 5):")
    for log_entry in recent_logs[-5:]:
        timestamp = time.strftime('%H:%M:%S', time.localtime(log_entry['timestamp']))
        print(f"      [{timestamp}] {log_entry['level']}: {log_entry['message']}")
    
    # Get error logs
    error_logs = logger_system.get_error_logs(3)
    if error_logs:
        print(f"\n   🚨 Recent Errors (last 3):")
        for error_entry in error_logs[-3:]:
            timestamp = time.strftime('%H:%M:%S', time.localtime(error_entry['timestamp']))
            print(f"      [{timestamp}] {error_entry['level']}: {error_entry['message']}")
    
    print("   ✓ Log monitoring test completed")

async def test_high_frequency_logging():
    """Test high-frequency logging performance"""
    print("\n⚡ Testing High-Frequency Logging Performance...")
    
    logger_system = setup_logging()
    
    # Simulate high-frequency trading logging
    start_time = time.perf_counter()
    num_logs = 1000
    
    for i in range(num_logs):
        if i % 100 == 0:
            # Trade logs (less frequent)
            log_trade(f"HFT trade {i}", symbol="HFT", side="BUY", quantity=100, price=150.0)
        
        if i % 10 == 0:
            # Performance logs (moderate frequency)
            log_performance("hft_latency", np.random.uniform(0.1, 2.0), "ms", "HFT latency")
        
        # Strategy logs (high frequency)
        if i % 50 == 0:
            log_strategy("HFT", f"Signal {i}", level="DEBUG", signal_strength=np.random.uniform(0, 1))
    
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    
    print(f"   ⚡ Logged {num_logs} entries in {total_time:.2f}ms")
    print(f"   📊 Average time per log: {total_time/num_logs:.3f}ms")
    print("   ✓ High-frequency logging test completed")

def main():
    """Main test function"""
    print("🚀 Comprehensive Logging System Test")
    print("=" * 50)
    
    # Run all tests
    logger_system = test_basic_logging()
    test_trade_logging()
    test_performance_logging()
    test_strategy_logging()
    test_ai_model_logging()
    test_market_data_logging()
    test_error_scenarios()
    test_log_monitoring()
    
    # Run async test
    asyncio.run(test_high_frequency_logging())
    
    print("\n✅ All logging tests completed successfully!")
    print("\n📂 Check the 'logs' directory for generated log files:")
    print("   - system.log (main system logs)")
    print("   - trades.log (trade execution logs)")
    print("   - performance.log (performance metrics)")
    print("   - strategies.log (strategy decisions)")
    print("   - errors.log (error logs)")
    print("   - ai_models.log (AI model logs)")
    print("   - market_data.log (market data logs)")
    print("   - *_structured.jsonl (JSON structured logs)")
    
    print(f"\n📊 Final System Statistics:")
    final_stats = logger_system.get_log_statistics()
    for key, value in final_stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
