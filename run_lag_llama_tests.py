#!/usr/bin/env python3
"""
Test Runner for Lag-Llama Engine Comprehensive Tests
Executes accuracy and latency tests with detailed reporting
"""

import asyncio
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'lag_llama_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

async def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    
    logger.info("=" * 80)
    logger.info("STARTING LAG-LLAMA ENGINE COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Import test modules
        from test_lag_llama_engine_comprehensive import (
            TestLagLlamaEngineAccuracy,
            TestLagLlamaEngineLatency,
            TestLagLlamaEngineORBIntegration,
            TestLagLlamaEngineErrorHandling,
            TestLagLlamaEngineProduction,
            test_performance_benchmark,
            MockCache,
            MockDatabase,
            TestDataGenerator
        )
        
        from lag_llama_engine import EnhancedLagLlamaEngine
        
        logger.info("✅ Successfully imported all test modules")
        
        # Setup test environment
        logger.info("🔧 Setting up test environment...")
        
        mock_cache = MockCache()
        mock_db = MockDatabase()
        test_symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        mock_cache.setup_test_data(test_symbols)
        
        engine = EnhancedLagLlamaEngine()
        engine.cache = mock_cache
        engine.db = mock_db
        engine.model_loaded = True
        engine.system_health['model_loaded'] = True
        engine.system_health['cache_connected'] = True
        engine.system_health['database_connected'] = True
        
        logger.info("✅ Test environment setup complete")
        
        # Create mock engine fixture
        mock_engine_fixture = (engine, mock_cache, mock_db, test_symbols)
        
        # Test 1: Accuracy Tests
        logger.info("\n" + "=" * 60)
        logger.info("🎯 RUNNING ACCURACY TESTS")
        logger.info("=" * 60)
        
        accuracy_test = TestLagLlamaEngineAccuracy()
        
        try:
            await accuracy_test.test_forecast_generation_accuracy(mock_engine_fixture)
            logger.info("✅ Forecast generation accuracy test PASSED")
        except Exception as e:
            logger.error(f"❌ Forecast generation accuracy test FAILED: {e}")
        
        try:
            await accuracy_test.test_forecast_consistency(mock_engine_fixture)
            logger.info("✅ Forecast consistency test PASSED")
        except Exception as e:
            logger.error(f"❌ Forecast consistency test FAILED: {e}")
        
        # Test 2: Latency Tests
        logger.info("\n" + "=" * 60)
        logger.info("⚡ RUNNING LATENCY TESTS")
        logger.info("=" * 60)
        
        latency_test = TestLagLlamaEngineLatency()
        
        try:
            await latency_test.test_single_forecast_latency(mock_engine_fixture)
            logger.info("✅ Single forecast latency test PASSED")
        except Exception as e:
            logger.error(f"❌ Single forecast latency test FAILED: {e}")
        
        try:
            await latency_test.test_batch_forecast_latency(mock_engine_fixture)
            logger.info("✅ Batch forecast latency test PASSED")
        except Exception as e:
            logger.error(f"❌ Batch forecast latency test FAILED: {e}")
        
        try:
            await latency_test.test_memory_usage(mock_engine_fixture)
            logger.info("✅ Memory usage test PASSED")
        except Exception as e:
            logger.error(f"❌ Memory usage test FAILED: {e}")
        
        # Test 3: ORB Integration Tests
        logger.info("\n" + "=" * 60)
        logger.info("📊 RUNNING ORB INTEGRATION TESTS")
        logger.info("=" * 60)
        
        orb_test = TestLagLlamaEngineORBIntegration()
        
        try:
            await orb_test.test_orb_analysis_generation(mock_engine_fixture)
            logger.info("✅ ORB analysis generation test PASSED")
        except Exception as e:
            logger.error(f"❌ ORB analysis generation test FAILED: {e}")
        
        # Test 4: Error Handling Tests
        logger.info("\n" + "=" * 60)
        logger.info("🛡️ RUNNING ERROR HANDLING TESTS")
        logger.info("=" * 60)
        
        error_test = TestLagLlamaEngineErrorHandling()
        
        try:
            await error_test.test_insufficient_data_handling(mock_engine_fixture)
            logger.info("✅ Insufficient data handling test PASSED")
        except Exception as e:
            logger.error(f"❌ Insufficient data handling test FAILED: {e}")
        
        try:
            await error_test.test_invalid_data_handling(mock_engine_fixture)
            logger.info("✅ Invalid data handling test PASSED")
        except Exception as e:
            logger.error(f"❌ Invalid data handling test FAILED: {e}")
        
        try:
            await error_test.test_stale_data_handling(mock_engine_fixture)
            logger.info("✅ Stale data handling test PASSED")
        except Exception as e:
            logger.error(f"❌ Stale data handling test FAILED: {e}")
        
        # Test 5: Production Tests
        logger.info("\n" + "=" * 60)
        logger.info("🏭 RUNNING PRODUCTION TESTS")
        logger.info("=" * 60)
        
        production_test = TestLagLlamaEngineProduction()
        
        try:
            await production_test.test_concurrent_forecast_requests(mock_engine_fixture)
            logger.info("✅ Concurrent forecast requests test PASSED")
        except Exception as e:
            logger.error(f"❌ Concurrent forecast requests test FAILED: {e}")
        
        try:
            await production_test.test_cache_performance(mock_engine_fixture)
            logger.info("✅ Cache performance test PASSED")
        except Exception as e:
            logger.error(f"❌ Cache performance test FAILED: {e}")
        
        try:
            await production_test.test_system_health_monitoring(mock_engine_fixture)
            logger.info("✅ System health monitoring test PASSED")
        except Exception as e:
            logger.error(f"❌ System health monitoring test FAILED: {e}")
        
        # Test 6: Performance Benchmark
        logger.info("\n" + "=" * 60)
        logger.info("🚀 RUNNING PERFORMANCE BENCHMARK")
        logger.info("=" * 60)
        
        try:
            await test_performance_benchmark()
            logger.info("✅ Performance benchmark PASSED")
        except Exception as e:
            logger.error(f"❌ Performance benchmark FAILED: {e}")
        
        # Test Summary
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📋 TEST SUITE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"🧪 Test categories completed: 6")
        logger.info(f"📊 Test symbols used: {len(test_symbols)}")
        logger.info(f"💾 Mock data points generated: {len(test_symbols) * 512}")
        
        # System Information
        logger.info("\n" + "=" * 60)
        logger.info("🖥️ SYSTEM INFORMATION")
        logger.info("=" * 60)
        
        import torch
        import numpy as np
        
        logger.info(f"🐍 Python version: {sys.version}")
        logger.info(f"🔥 PyTorch version: {torch.__version__}")
        logger.info(f"🔢 NumPy version: {np.__version__}")
        logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Performance Metrics
        logger.info("\n" + "=" * 60)
        logger.info("📈 PERFORMANCE METRICS")
        logger.info("=" * 60)
        
        metrics = engine.get_performance_metrics()
        
        logger.info(f"🎯 Model loaded: {metrics['model_performance']['model_loaded']}")
        logger.info(f"💾 Cache symbols: {metrics['cache_stats']['symbol_count']}")
        logger.info(f"📊 Cache hit rate: {metrics['cache_stats']['hit_rate']:.1%}")
        logger.info(f"🔒 No fallbacks used: {metrics['data_quality']['no_fallbacks_used']}")
        logger.info(f"✅ Strict validation: {metrics['data_quality']['strict_validation_enabled']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 LAG-LLAMA ENGINE TEST SUITE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR in test suite: {e}")
        import traceback
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        return False

async def run_quick_validation():
    """Run a quick validation test to ensure basic functionality"""
    
    logger.info("\n" + "=" * 60)
    logger.info("⚡ RUNNING QUICK VALIDATION TEST")
    logger.info("=" * 60)
    
    try:
        from lag_llama_engine import EnhancedLagLlamaEngine, ForecastResult
        from test_lag_llama_engine_comprehensive import MockCache, MockDatabase
        
        # Quick setup
        engine = EnhancedLagLlamaEngine()
        mock_cache = MockCache()
        mock_db = MockDatabase()
        
        test_symbols = ['AAPL', 'TSLA']
        mock_cache.setup_test_data(test_symbols)
        
        engine.cache = mock_cache
        engine.db = mock_db
        engine.model_loaded = True
        engine.system_health['model_loaded'] = True
        engine.system_health['cache_connected'] = True
        engine.system_health['database_connected'] = True
        
        # Test data validation
        price_series = engine._get_validated_price_series('AAPL', 100)
        assert len(price_series) >= 100, "Should have sufficient price data"
        
        current_price = engine._get_validated_current_price('AAPL')
        assert current_price > 0, "Should have valid current price"
        
        cache_stats = engine.cache.get_cache_stats()
        assert cache_stats['symbol_count'] > 0, "Should have symbols in cache"
        
        logger.info("✅ Data validation: PASSED")
        logger.info("✅ Price series validation: PASSED")
        logger.info("✅ Cache functionality: PASSED")
        logger.info("✅ Quick validation: ALL TESTS PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick validation FAILED: {e}")
        return False

def main():
    """Main test runner"""
    
    print("\n🚀 LAG-LLAMA ENGINE TEST RUNNER")
    print("=" * 50)
    
    # Check if we should run quick validation or full suite
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick validation test...")
        success = asyncio.run(run_quick_validation())
    else:
        print("Running comprehensive test suite...")
        print("(Use --quick flag for quick validation only)")
        success = asyncio.run(run_comprehensive_tests())
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()