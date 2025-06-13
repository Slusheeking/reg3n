#pragma once

#include "simple_config.hpp"
#include "simple_models.hpp"
#include "simple_polygon.hpp"
#include "simple_alpaca.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <iomanip>
#include <thread>
#include <memory>

namespace simple_hft {

// Simple logging utility
class SimpleLogger {
private:
    std::ofstream log_file_;
    
public:
    SimpleLogger() {
        if (SimpleConfig::ENABLE_VERBOSE_LOGGING) {
            log_file_.open(SimpleConfig::LOG_FILE, std::ios::app);
        }
    }
    
    ~SimpleLogger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    void log(const std::string& level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // Console output
        std::cout << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                  << "] [" << level << "] " << message << std::endl;
        
        // File output
        if (log_file_.is_open()) {
            log_file_ << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                      << "] [" << level << "] " << message << std::endl;
            log_file_.flush();
        }
    }
    
    void info(const std::string& message) { log("INFO", message); }
    void warn(const std::string& message) { log("WARN", message); }
    void error(const std::string& message) { log("ERROR", message); }
    void debug(const std::string& message) { 
        if (SimpleConfig::ENABLE_VERBOSE_LOGGING) {
            log("DEBUG", message); 
        }
    }
};

// Main simplified HFT engine - FAIL FAST design
class SimpleHFTEngine {
private:
    // Components
    std::unique_ptr<SimplePolygonClient> polygon_client_;
    std::unique_ptr<SimpleAlpacaClient> alpaca_client_;
    std::unique_ptr<SimpleLinearModel> model_;
    std::unique_ptr<SimpleFeatureCalculator> feature_calculator_;
    std::unique_ptr<SimpleLogger> logger_;
    
    // State
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    std::atomic<int> total_errors_;
    
    // Performance tracking
    std::atomic<uint64_t> messages_processed_;
    std::atomic<uint64_t> signals_generated_;
    std::atomic<uint64_t> orders_placed_;
    std::atomic<uint64_t> orders_successful_;
    
    // Current market prices (for order placement)
    std::unordered_map<std::string, double> current_prices_;
    
    // Last processing time for each symbol (to avoid too frequent trading)
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_processed_;
    
    // Latency tracking
    std::atomic<uint64_t> total_processing_time_us_;
    std::atomic<uint64_t> total_order_time_us_;
    std::atomic<uint64_t> total_calculation_time_us_;
    std::atomic<uint64_t> total_e2e_cycle_time_us_;
    std::atomic<uint64_t> processed_count_;
    std::atomic<uint64_t> order_count_;
    std::atomic<uint64_t> calculation_count_;
    std::atomic<uint64_t> e2e_cycle_count_;
    
    // Position tracking
    std::unordered_map<std::string, int> open_positions_;

public:
    SimpleHFTEngine()
        : running_(false), initialized_(false), total_errors_(0),
          messages_processed_(0), signals_generated_(0),
          orders_placed_(0), orders_successful_(0),
          total_processing_time_us_(0), total_order_time_us_(0),
          total_calculation_time_us_(0), total_e2e_cycle_time_us_(0),
          processed_count_(0), order_count_(0), calculation_count_(0), e2e_cycle_count_(0) {
        
        logger_ = std::make_unique<SimpleLogger>();
        logger_->info("SimpleHFTEngine created");
    }
    
    ~SimpleHFTEngine() {
        stop();
    }
    
    // Initialize all components - FAIL FAST on any error
    bool initialize() {
        try {
            logger_->info("Initializing SimpleHFTEngine...");
            
            // Initialize components
            polygon_client_ = std::make_unique<SimplePolygonClient>();
            alpaca_client_ = std::make_unique<SimpleAlpacaClient>();
            model_ = std::make_unique<SimpleLinearModel>();
            feature_calculator_ = std::make_unique<SimpleFeatureCalculator>();
            
            // Test Alpaca connection first
            logger_->info("Testing Alpaca connection...");
            if (!alpaca_client_->test_connection()) {
                logger_->error("CRITICAL: Failed to connect to Alpaca API");
                return false;
            }
            logger_->info("Alpaca connection successful");
            
            // Load current portfolio data
            logger_->info("Loading portfolio data...");
            AccountInfo account = alpaca_client_->get_account_info();
            if (account.valid) {
                logger_->info("Account Info:");
                logger_->info("  Portfolio Value: $" + std::to_string(account.portfolio_value));
                logger_->info("  Buying Power: $" + std::to_string(account.buying_power));
                logger_->info("  Cash: $" + std::to_string(account.cash));
            } else {
                logger_->warn("Failed to load account information");
            }
            
            // Load existing positions
            int position_count = alpaca_client_->get_position_count();
            if (position_count >= 0) {
                logger_->info("Current positions: " + std::to_string(position_count));
                if (position_count > 0) {
                    logger_->info("Existing positions will be tracked for risk management");
                }
            } else {
                logger_->warn("Failed to load position data");
            }
            
            // Set up Polygon callback
            polygon_client_->set_message_callback([this](const MarketMessage& msg) {
                this->process_market_message(msg);
            });
            
            // Connect to Polygon
            logger_->info("Connecting to Polygon WebSocket...");
            if (!polygon_client_->connect()) {
                logger_->error("CRITICAL: Failed to connect to Polygon WebSocket");
                return false;
            }
            
            // Subscribe to symbols
            logger_->info("Subscribing to market data...");
            if (!polygon_client_->subscribe_to_symbols()) {
                logger_->error("CRITICAL: Failed to subscribe to symbols");
                return false;
            }
            
            logger_->info("Model info: " + model_->get_model_info());
            
            initialized_ = true;
            logger_->info("SimpleHFTEngine initialized successfully");
            return true;
            
        } catch (const std::exception& e) {
            logger_->error("CRITICAL: Initialization failed: " + std::string(e.what()));
            increment_error_count();
            return false;
        }
    }
    
    // Start the engine
    bool start() {
        if (!initialized_) {
            logger_->error("Cannot start: Engine not initialized");
            return false;
        }
        
        logger_->info("Starting SimpleHFTEngine...");
        running_ = true;
        
        logger_->info("=== SIMPLE HFT ENGINE STARTED ===");
        logger_->info("Symbols: " + std::to_string(SimpleConfig::SYMBOLS.size()));
        logger_->info("Confidence threshold: " + std::to_string(SimpleConfig::CONFIDENCE_THRESHOLD));
        logger_->info("Position size: $" + std::to_string(SimpleConfig::POSITION_SIZE_DOLLARS));
        logger_->info("Max positions: " + std::to_string(SimpleConfig::MAX_POSITIONS));
        logger_->info("=====================================");
        
        return true;
    }
    
    // Stop the engine
    void stop() {
        if (running_) {
            logger_->info("Stopping SimpleHFTEngine...");
            running_ = false;
            
            if (polygon_client_) {
                polygon_client_->disconnect();
            }
            
            print_final_stats();
            logger_->info("SimpleHFTEngine stopped");
        }
    }
    
    // Main processing loop - call this continuously
    bool process_cycle() {
        if (!running_) {
            return false;
        }
        
        // Check for critical errors (fail-fast)
        if (check_critical_errors()) {
            logger_->error("CRITICAL: Too many errors detected. Shutting down.");
            stop();
            return false;
        }
        
        // The actual processing happens in the Polygon callback
        // This method is mainly for error checking and status updates
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Don't spin too fast
        return true;
    }
    
    // Cancel all orders and liquidate all positions
    bool liquidate_all() {
        if (!alpaca_client_) {
            logger_->error("Alpaca client not initialized");
            return false;
        }
        
        logger_->info("🚨 EMERGENCY LIQUIDATION INITIATED 🚨");
        
        // Stop all signal generation immediately
        running_.store(false);
        
        // Disconnect from market data to stop signals
        if (polygon_client_) {
            polygon_client_.reset();
        }
        
        if (alpaca_client_->cancel_all_orders_and_liquidate()) {
            // Clear our position tracking
            open_positions_.clear();
            logger_->info("✅ Complete liquidation successful - portfolio reset to cash");
            return true;
        } else {
            logger_->error("❌ Liquidation failed");
            return false;
        }
    }
    
    // Get current statistics
    void print_stats() const {
        logger_->info("=== CURRENT STATS ===");
        logger_->info("Messages processed: " + std::to_string(messages_processed_.load()));
        logger_->info("Signals generated: " + std::to_string(signals_generated_.load()));
        logger_->info("Orders placed: " + std::to_string(orders_placed_.load()));
        logger_->info("Orders successful: " + std::to_string(orders_successful_.load()));
        logger_->info("Total errors: " + std::to_string(total_errors_.load()));
        logger_->info("Polygon connected: " + std::string(polygon_client_ && polygon_client_->is_connected() ? "YES" : "NO"));
        logger_->info("Polygon authenticated: " + std::string(polygon_client_ && polygon_client_->is_authenticated() ? "YES" : "NO"));
        logger_->info("====================");
    }
    
    bool is_running() const { return running_.load(); }
    bool is_initialized() const { return initialized_.load(); }
    
private:
    // Process incoming market message - CORE TRADING LOGIC
    void process_market_message(const MarketMessage& msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto e2e_start_time = start_time; // Track end-to-end latency
        
        try {
            messages_processed_.fetch_add(1);
            
            if (!msg.valid) {
                return;
            }
            
            // Update feature calculator with new data
            if (msg.type == "trade" && msg.price > 0 && msg.volume > 0) {
                feature_calculator_->update_price_data(msg.symbol, msg.price, msg.volume);
                current_prices_[msg.symbol] = msg.price; // Store current price for orders
                
            } else if (msg.type == "quote" && msg.bid > 0 && msg.ask > 0) {
                feature_calculator_->update_quote_data(msg.symbol, msg.bid, msg.ask, 
                                                      msg.bid_size, msg.ask_size);
            }
            
            // Only process trades for signal generation (quotes are just for features)
            if (msg.type != "trade") {
                return;
            }
            
            // Rate limiting: don't process the same symbol too frequently
            auto now = std::chrono::steady_clock::now();
            auto it = last_processed_.find(msg.symbol);
            if (it != last_processed_.end()) {
                auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(now - it->second);
                if (time_since_last.count() < 30) { // Wait at least 30 seconds between signals for same symbol
                    return;
                }
            }
            
            // Check if we have sufficient data for features
            if (!feature_calculator_->has_sufficient_data(msg.symbol)) {
                return;
            }
            
            // Calculate features
            FeatureVector features = feature_calculator_->calculate_features(msg.symbol);
            if (!features.valid) {
                return;
            }
            
            // Get prediction from model
            SimpleSignal signal = model_->predict(msg.symbol, features);
            
            if (signal.direction != 0 && signal.confidence > SimpleConfig::CONFIDENCE_THRESHOLD) {
                signals_generated_.fetch_add(1);
                last_processed_[msg.symbol] = now; // Update last processed time
                
                logger_->info("SIGNAL: " + signal.symbol + " " + signal.strategy + 
                             " direction=" + std::to_string(signal.direction) + 
                             " confidence=" + std::to_string(signal.confidence));
                
                // Place bracket order
                place_order(signal, msg.price);
            }
            
            // Track processing latency
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            total_processing_time_us_.fetch_add(duration_us);
            processed_count_.fetch_add(1);
            
        } catch (const std::exception& e) {
            logger_->error("Error processing market message: " + std::string(e.what()));
            increment_error_count();
        }
    }
    
    // Place bracket order
    void place_order(const SimpleSignal& signal, double current_price) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            orders_placed_.fetch_add(1);
            
            logger_->info("Placing bracket order for " + signal.symbol +
                         " at $" + std::to_string(current_price));
            
            OrderResult result = alpaca_client_->place_bracket_order(signal, current_price);
            
            // Track order placement latency
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            total_order_time_us_.fetch_add(duration_us);
            order_count_.fetch_add(1);
            
            if (result.success) {
                orders_successful_.fetch_add(1);
                logger_->info("✅ Order successful: " + result.order_id);
            }
            // Note: Order failures are not logged to reduce noise
            // They could be due to market conditions, risk limits, etc.
            
        } catch (const std::exception& e) {
            logger_->error("Error placing order: " + std::string(e.what()));
            increment_error_count();
        }
    }
    
    // Check for critical errors that should shut down the system
    bool check_critical_errors() {
        int total_errors = total_errors_.load();
        int polygon_errors = polygon_client_ ? polygon_client_->get_error_count() : 0;
        int alpaca_errors = alpaca_client_ ? alpaca_client_->get_error_count() : 0;
        
        // Check individual component error limits
        if (polygon_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            logger_->error("CRITICAL: Polygon client has too many errors (" + 
                          std::to_string(polygon_errors) + ")");
            return true;
        }
        
        if (alpaca_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            logger_->error("CRITICAL: Alpaca client has too many errors (" + 
                          std::to_string(alpaca_errors) + ")");
            return true;
        }
        
        // Check total system errors
        if (total_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            logger_->error("CRITICAL: System has too many total errors (" + 
                          std::to_string(total_errors) + ")");
            return true;
        }
        
        // Check connection status
        if (polygon_client_ && !polygon_client_->is_connected()) {
            logger_->error("CRITICAL: Lost connection to Polygon");
            return true;
        }
        
        return false;
    }
    
    void increment_error_count() {
        int current_errors = total_errors_.fetch_add(1) + 1;
        if (current_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            logger_->error("CRITICAL: System error limit reached (" + 
                          std::to_string(current_errors) + ")");
        }
    }
    
    void print_final_stats() const {
        logger_->info("=== FINAL STATISTICS ===");
        logger_->info("Total messages processed: " + std::to_string(messages_processed_.load()));
        logger_->info("Total signals generated: " + std::to_string(signals_generated_.load()));
        logger_->info("Total orders placed: " + std::to_string(orders_placed_.load()));
        logger_->info("Total successful orders: " + std::to_string(orders_successful_.load()));
        logger_->info("Total errors: " + std::to_string(total_errors_.load()));
        
        uint64_t orders = orders_placed_.load();
        uint64_t successful = orders_successful_.load();
        if (orders > 0) {
            double success_rate = (double)successful / orders * 100.0;
            logger_->info("Order success rate: " + std::to_string(success_rate) + "%");
        }
        
        uint64_t messages = messages_processed_.load();
        uint64_t signals = signals_generated_.load();
        if (messages > 0) {
            double signal_rate = (double)signals / messages * 100.0;
            logger_->info("Signal generation rate: " + std::to_string(signal_rate) + "%");
        }
        
        logger_->info("========================");
    }
};

} // namespace simple_hft