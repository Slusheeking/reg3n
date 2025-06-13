#include "simple_hft.hpp"
#include <iostream>
#include <signal.h>
#include <chrono>
#include <thread>

using namespace simple_hft;

// Global engine instance for signal handling
std::unique_ptr<SimpleHFTEngine> g_simple_engine;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << " - shutting down gracefully...\n";
    if (g_simple_engine) {
        g_simple_engine->stop();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    std::cout << "=== SIMPLE HFT SYSTEM ===" << std::endl;
    std::cout << "Streamlined: Polygon → Features → Models → Alpaca" << std::endl;
    std::cout << "No complex optimizations, online learning disabled" << std::endl;
    std::cout << "Fixed weights, fail-fast design" << std::endl;
    std::cout << "Symbols: " << SimpleConfig::SYMBOLS.size() << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Parse command line arguments
    bool test_mode = false;
    bool liquidate_mode = false;
    int test_duration = 60; // Default 60 seconds for test mode
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            test_mode = true;
        } else if (std::string(argv[i]) == "--duration" && i + 1 < argc) {
            test_duration = std::atoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--liquidate") {
            liquidate_mode = true;
        } else if (std::string(argv[i]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --test              Run in test mode\n";
            std::cout << "  --duration N        Test duration in seconds (default: 60)\n";
            std::cout << "  --liquidate         Cancel all orders and close all positions\n";
            std::cout << "  --help              Show this help\n";
            return 0;
        }
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // Create and initialize the simple HFT engine
        std::cout << "Creating Simple HFT Engine..." << std::endl;
        g_simple_engine = std::make_unique<SimpleHFTEngine>();
        
        std::cout << "Initializing Simple HFT Engine..." << std::endl;
        if (!g_simple_engine->initialize()) {
            std::cerr << "❌ Failed to initialize Simple HFT Engine!" << std::endl;
            return 1;
        }
        
        // Handle liquidation mode
        if (liquidate_mode) {
            std::cout << "🚨 LIQUIDATION MODE ACTIVATED 🚨" << std::endl;
            bool liquidation_success = g_simple_engine->liquidate_all();
            
            // Clean shutdown regardless of liquidation result
            g_simple_engine->stop();
            g_simple_engine.reset();
            
            if (liquidation_success) {
                std::cout << "✅ Liquidation completed successfully" << std::endl;
                return 0;
            } else {
                std::cerr << "❌ Liquidation failed" << std::endl;
                return 1;
            }
        }
        
        std::cout << "Starting Simple HFT Engine..." << std::endl;
        if (!g_simple_engine->start()) {
            std::cerr << "❌ Failed to start Simple HFT Engine!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Simple HFT Engine started successfully!" << std::endl;
        std::cout << "Press Ctrl+C to stop..." << std::endl;
        
        if (test_mode) {
            std::cout << "🧪 Running in test mode for " << test_duration << " seconds..." << std::endl;
            
            auto start_time = std::chrono::steady_clock::now();
            auto end_time = start_time + std::chrono::seconds(test_duration);
            
            int stats_interval = std::max(10, test_duration / 6); // Print stats 6 times during test
            auto last_stats_time = start_time;
            
            while (std::chrono::steady_clock::now() < end_time && g_simple_engine->is_running()) {
                if (!g_simple_engine->process_cycle()) {
                    std::cerr << "❌ Engine stopped due to critical error" << std::endl;
                    break;
                }
                
                // Print stats periodically
                auto now = std::chrono::steady_clock::now();
                auto time_since_stats = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
                if (time_since_stats.count() >= stats_interval) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                    std::cout << "\n--- Test Progress: " << elapsed.count() << "/" << test_duration << "s ---" << std::endl;
                    g_simple_engine->print_stats();
                    last_stats_time = now;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            std::cout << "\n🧪 Test completed!" << std::endl;
            g_simple_engine->print_stats();
            
        } else {
            // Production mode - run indefinitely
            std::cout << "🚀 Running in production mode..." << std::endl;
            
            auto last_stats_time = std::chrono::steady_clock::now();
            const int STATS_INTERVAL_SECONDS = 300; // Print stats every 5 minutes
            
            while (g_simple_engine->is_running()) {
                if (!g_simple_engine->process_cycle()) {
                    std::cerr << "❌ Engine stopped due to critical error" << std::endl;
                    break;
                }
                
                // Print stats periodically
                auto now = std::chrono::steady_clock::now();
                auto time_since_stats = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
                if (time_since_stats.count() >= STATS_INTERVAL_SECONDS) {
                    std::cout << "\n--- Periodic Status Update ---" << std::endl;
                    g_simple_engine->print_stats();
                    last_stats_time = now;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    // Graceful shutdown
    std::cout << "\nShutting down Simple HFT Engine..." << std::endl;
    if (g_simple_engine) {
        g_simple_engine->stop();
        g_simple_engine.reset();
    }
    
    std::cout << "✅ Simple HFT System shutdown complete." << std::endl;
    return 0;
}