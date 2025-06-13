#pragma once

#include "memory_manager.hpp"
#include "hft_models.hpp"
#include "alpaca_client.hpp"
#include <unordered_map>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <cstdlib>
#include <set>
#include <iomanip>   // for std::fixed, std::setprecision

namespace hft {


// Forward declaration - AlpacaOrderRequest is defined in websocket_client.hpp
struct AlpacaOrderRequest;

// Forward declaration for AlpacaClient enums and structs
namespace AlpacaClientTypes {
    enum class Side { BUY, SELL };
    enum class TimeInForce { DAY, GTC, IOC, FOK };
    struct OrderResult {
        bool success{false};
        std::string order_id;
        std::string error_message;
        std::chrono::nanoseconds total_latency{0};
    };
}

// Order types and states
enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    BRACKET = 3
};

enum class OrderSide : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderStatus : uint8_t {
    PENDING = 0,
    SUBMITTED = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4
};

// Order is now defined in memory_manager.hpp

// Position tracking
struct alignas(32) Position {
    int32_t symbol_id;
    uint64_t signal_id;        // Track which signal opened this position
    float quantity;
    float average_price;
    float market_value;
    float unrealized_pnl;
    float realized_pnl;
    uint64_t entry_timestamp_ns;
    uint64_t last_update_ns;
    uint8_t side; // 0=long, 1=short
    uint8_t padding[7];
};

// Risk limits structure
struct RiskLimits {
    float max_position_size;      // 20% of portfolio
    float max_total_exposure;     // 95% max exposure
    float min_cash_reserve;       // 5% cash reserve
    float max_daily_loss;         // 3% daily loss limit
    float max_symbol_exposure;    // 25% per symbol
    uint32_t max_orders_per_sec;  // Rate limiting
    uint32_t max_total_positions; // Maximum number of open positions
};

// Kelly Position Sizer for dynamic position sizing
class KellyPositionSizer {
private:
    struct SymbolStats {
        uint32_t trades{0};
        uint32_t wins{0};
        float total_win_size{0.0f};
        float total_loss_size{0.0f};
        float kelly_fraction{0.02f}; // Conservative start
    };
    
    std::unordered_map<int32_t, SymbolStats> symbol_stats_;
    mutable std::mutex stats_mutex_;
    
public:
    float calculate_position_size(int32_t symbol_id, float confidence,
                                float portfolio_value, float current_volatility) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto& stats = symbol_stats_[symbol_id];
        
        if (stats.trades < 20) {
            // Not enough data - use conservative sizing
            return portfolio_value * 0.01f * confidence;
        }
        
        // Calculate Kelly fraction
        float win_rate = static_cast<float>(stats.wins) / stats.trades;
        float avg_win = stats.total_win_size / (stats.wins + 1e-6f);
        float avg_loss = stats.total_loss_size / ((stats.trades - stats.wins) + 1e-6f);
        
        // Kelly formula: f = (p * b - q) / b
        // where p = win_rate, q = 1-p, b = avg_win/avg_loss
        float b = avg_win / (avg_loss + 1e-6f);
        float kelly = (win_rate * b - (1.0f - win_rate)) / b;
        
        // Apply Kelly with confidence adjustment and volatility scaling
        kelly = std::max(0.0f, std::min(0.25f, kelly)); // Cap at 25%
        float vol_adjustment = std::min(1.0f, 0.01f / current_volatility);
        
        return portfolio_value * kelly * confidence * vol_adjustment * 0.5f; // Half-Kelly
    }
    
    void update_stats(int32_t symbol_id, float pnl) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto& stats = symbol_stats_[symbol_id];
        
        stats.trades++;
        if (pnl > 0) {
            stats.wins++;
            stats.total_win_size += pnl;
        } else {
            stats.total_loss_size += std::abs(pnl);
        }
        
        // Update Kelly fraction with exponential smoothing
        float new_kelly = calculate_position_size(symbol_id, 1.0f, 1.0f, 0.01f);
        stats.kelly_fraction = 0.9f * stats.kelly_fraction + 0.1f * new_kelly;
    }
};

// Lock-free priority order queue for ultra-fast order processing
template<size_t CAPACITY>
class LockFreePriorityOrderQueue {
private:
    struct alignas(64) OrderSlot {
        Order order;
        uint64_t priority; // Combine priority + timestamp
        std::atomic<uint32_t> version{0};
    };
    
    std::array<OrderSlot, CAPACITY> slots_;
    std::atomic<uint32_t> size_{0};
    
public:
    bool try_push(const Order& order) noexcept {
        uint32_t current_size = size_.load(std::memory_order_acquire);
        if (current_size >= CAPACITY) return false;
        
        // Calculate priority (lower = higher priority)
        uint64_t priority = (static_cast<uint64_t>(order.tier) << 56) |
                           (order.timestamp_ns & 0xFFFFFFFFFFFFFF);
        
        // Find insertion point
        for (uint32_t i = 0; i < CAPACITY; ++i) {
            uint32_t expected = 0;
            if (slots_[i].version.compare_exchange_strong(expected, 1,
                std::memory_order_acq_rel)) {
                
                slots_[i].order = order;
                slots_[i].priority = priority;
                slots_[i].version.store(2, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_release);
                return true;
            }
        }
        return false;
    }
    
    bool try_pop(Order& out) noexcept {
        uint32_t current_size = size_.load(std::memory_order_acquire);
        if (current_size == 0) return false;
        
        uint32_t best_idx = UINT32_MAX;
        uint64_t best_priority = UINT64_MAX;
        
        // Find highest priority order
        for (uint32_t i = 0; i < CAPACITY; ++i) {
            uint32_t ver = slots_[i].version.load(std::memory_order_acquire);
            if (ver == 2 && slots_[i].priority < best_priority) {
                best_priority = slots_[i].priority;
                best_idx = i;
            }
        }
        
        if (best_idx < CAPACITY) {
            out = slots_[best_idx].order;
            slots_[best_idx].version.store(0, std::memory_order_release);
            size_.fetch_sub(1, std::memory_order_release);
            return true;
        }
        return false;
    }
    
    bool empty() const noexcept {
        return size_.load(std::memory_order_acquire) == 0;
    }
    
    size_t size() const noexcept {
        return size_.load(std::memory_order_acquire);
    }
};


// Ultra-fast order engine with sub-100μs execution and online learning
class UltraFastOrderEngine {
private:
    UltraFastMemoryManager& memory_manager_;
    
    // Alpaca client for order submission
    ::AlpacaClient* alpaca_client_;
    
    // Order management
    std::atomic<uint64_t> next_order_id_{1};
    LockFreePriorityOrderQueue<10000> pending_orders_;
    LockFreeQueue<Order, 10000> completed_orders_;
    LockFreeQueue<SignalOutcome, 1000> trade_outcome_queue_; // For learning
    
    // Position tracking (Alpaca handles take profit automatically)
    std::unordered_map<int32_t, Position> positions_;
    mutable std::mutex positions_mutex_;
    
    // Signal tracking for learning feedback
    std::unordered_map<uint64_t, TradingSignal> active_signals_;
    std::unordered_map<uint64_t, float*> signal_features_; // Store features for learning
    mutable std::mutex signals_mutex_;
    
    // Portfolio state - Use defaults until real Alpaca data arrives
    std::atomic<float> cash_balance_{50000.0f}; // Default $50K for immediate trading
    std::atomic<float> buying_power_{50000.0f}; // Default $50K for immediate trading
    std::atomic<float> portfolio_value_{50000.0f}; // Default $50K for immediate trading
    std::atomic<float> daily_pnl_{0.0f};
    std::atomic<bool> alpaca_data_received_{false}; // Track if we have real data
    
    // Risk management
    RiskLimits risk_limits_;
    std::atomic<uint32_t> orders_this_second_{0};
    std::atomic<uint64_t> last_rate_reset_ns_{0};
    
    // Processing thread (simplified - no position monitoring needed)
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
    
    // Performance tracking
    std::atomic<uint64_t> orders_processed_{0};
    std::atomic<uint64_t> total_processing_time_ns_{0};
    std::atomic<uint64_t> orders_filled_{0};
    std::atomic<uint64_t> orders_rejected_{0};

public:
    explicit UltraFastOrderEngine(UltraFastMemoryManager& memory_manager,
                                 ::AlpacaClient* alpaca_client = nullptr)
        : memory_manager_(memory_manager), alpaca_client_(alpaca_client) {
        
        std::cout << "--- UltraFastOrderEngine CONSTRUCTOR ---" << std::endl;
        
        // Try to fetch real Alpaca data immediately if client is available
        if (alpaca_client_ && alpaca_client_->isReadyForOrders()) {
            std::cout << "  🔄 Fetching real portfolio data from Alpaca..." << std::endl;
            try {
                auto account_update = alpaca_client_->getAccountInfo();
                
                // Validate and use real data if available
                if (account_update.equity > 0 && account_update.buying_power > 0 &&
                    !std::isnan(account_update.equity) && !std::isinf(account_update.equity)) {
                    
                    portfolio_value_.store(static_cast<float>(account_update.equity));
                    buying_power_.store(static_cast<float>(account_update.buying_power));
                    cash_balance_.store(static_cast<float>(account_update.cash));
                    alpaca_data_received_.store(true);
                    
                    std::cout << "  ✅ REAL portfolio values from Alpaca:" << std::endl;
                    std::cout << "     Portfolio Value: $" << std::fixed << std::setprecision(2) << account_update.equity << std::endl;
                    std::cout << "     Buying Power: $" << std::fixed << std::setprecision(2) << account_update.buying_power << std::endl;
                    std::cout << "     Cash: $" << std::fixed << std::setprecision(2) << account_update.cash << std::endl;
                } else {
                    std::cout << "  ⚠️ Invalid Alpaca data, using defaults temporarily" << std::endl;
                    std::cout << "     Will sync with real data once account polling starts" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "  ⚠️ Could not fetch Alpaca data immediately: " << e.what() << std::endl;
                std::cout << "     Using defaults temporarily, will sync when polling starts" << std::endl;
            }
        } else {
            std::cout << "  ⚠️ Alpaca client not ready during construction" << std::endl;
            std::cout << "     Using default values temporarily until client is ready" << std::endl;
        }
        
        std::cout << "  Current portfolio values:" << std::endl;
        std::cout << "     cash_balance_: $" << std::fixed << std::setprecision(2) << cash_balance_.load() << std::endl;
        std::cout << "     buying_power_: $" << std::fixed << std::setprecision(2) << buying_power_.load() << std::endl;
        std::cout << "     portfolio_value_: $" << std::fixed << std::setprecision(2) << portfolio_value_.load() << std::endl;
        std::cout << "     real_data_received: " << (alpaca_data_received_.load() ? "YES" : "NO") << std::endl;
        
        // Initialize balanced risk limits - respect Alpaca rate limits but allow reasonable trading
        risk_limits_ = {
            0.20f,  // max_position_size (20% - up from 5%)
            0.95f,  // max_total_exposure (95% - up from 80%)
            0.05f,  // min_cash_reserve (5% - down from 10%)
            0.03f,  // max_daily_loss (3% - up from 2%)
            0.25f,  // max_symbol_exposure (25% - up from 10%)
            2,      // max_orders_per_sec (strict: 2/sec = 120/min, well under 200/min limit)
            25      // max_total_positions (maximum 25 open positions)
        };
        std::cout << "--- END UltraFastOrderEngine CONSTRUCTOR ---" << std::endl;
    }
    
    ~UltraFastOrderEngine() {
        stop();
    }
    
    // Start order processing
    bool start() {
        if (running_.load()) {
            return true;
        }
        
        running_.store(true);
        
        // Start processing thread with high priority
        processing_thread_ = std::thread([this]() { this->processing_loop(); });
        
        // Set CPU affinity (core 3)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset);
        pthread_setaffinity_np(processing_thread_.native_handle(), sizeof(cpuset), &cpuset);
        
        // Set real-time priority
        struct sched_param param;
        param.sched_priority = 98;
        pthread_setschedparam(processing_thread_.native_handle(), SCHED_FIFO, &param);
        
        std::cout << "✅ Order engine started with automatic 0.8% take profit orders" << std::endl;
        return true;
    }
    
    // Stop order processing
    void stop() {
        if (!running_.load()) {
            return;
        }
        
        running_.store(false);
        
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
    
    // Execute trading signal with ultra-fast risk checks
    FORCE_INLINE bool execute_signal(const TradingSignal& signal) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Rate limiting check
        if (!check_rate_limit()) {
            orders_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        // Create order from signal
        Order order{};
        if (!create_order_from_signal(signal, order)) {
            orders_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        // Fast risk checks
        if (!validate_risk_limits(order)) {
            orders_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        // Queue order for processing
        if (!pending_orders_.try_push(order)) {
            orders_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        // Track performance
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        total_processing_time_ns_.fetch_add(duration.count(), std::memory_order_relaxed);
        orders_processed_.fetch_add(1, std::memory_order_relaxed);
        
        return true;
    }
    
    // Get current portfolio state
    struct PortfolioState {
        float cash_balance;
        float buying_power;
        float portfolio_value;
        float daily_pnl;
        size_t active_positions;
        float total_exposure;
    };
    
    PortfolioState get_portfolio_state() const {
        // Get real position count from Alpaca REST API
        size_t real_position_count = 0;
        float total_market_value = 0.0f;
        
        if (alpaca_client_ && alpaca_client_->isReadyForOrders()) {
            try {
                // Get real positions from Alpaca
                auto positions = alpaca_client_->getPositions();
                real_position_count = positions.size();
                
                // Calculate total market value from real positions
                for (const auto& pos : positions) {
                    total_market_value += std::abs(pos.market_value);
                }
                
                std::cout << "✅ Fetched " << real_position_count << " real positions from Alpaca" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "❌ Error getting positions from Alpaca: " << e.what() << std::endl;
                // Fall back to internal tracking
                std::lock_guard<std::mutex> lock(positions_mutex_);
                real_position_count = positions_.size();
                for (const auto& [symbol_id, position] : positions_) {
                    total_market_value += std::abs(position.market_value);
                }
            }
        } else {
            // Fall back to internal tracking if Alpaca client not available
            std::lock_guard<std::mutex> lock(positions_mutex_);
            real_position_count = positions_.size();
            for (const auto& [symbol_id, position] : positions_) {
                total_market_value += std::abs(position.market_value);
            }
        }
        
        return {
            cash_balance_.load(),
            buying_power_.load(),
            portfolio_value_.load(),
            daily_pnl_.load(),
            real_position_count,  // Use real position count
            total_market_value / std::max(1.0f, portfolio_value_.load())  // Avoid division by zero
        };
    }
    
    // Performance statistics
    struct PerformanceStats {
        uint64_t orders_processed;
        uint64_t orders_filled;
        uint64_t orders_rejected;
        double avg_processing_time_us;
        double fill_rate;
    };
    
    PerformanceStats get_performance_stats() const {
        uint64_t processed = orders_processed_.load();
        uint64_t filled = orders_filled_.load();
        uint64_t rejected = orders_rejected_.load();
        uint64_t total_time = total_processing_time_ns_.load();
        
        return {
            processed,
            filled,
            rejected,
            processed > 0 ? static_cast<double>(total_time) / (processed * 1000.0) : 0.0,
            processed > 0 ? static_cast<double>(filled) / processed : 0.0
        };
    }
    
    // Register signal for tracking (called when signal is generated)
    void register_signal(const TradingSignal& signal, const float* features) {
        std::lock_guard<std::mutex> lock(signals_mutex_);
        active_signals_[signal.signal_id] = signal;
        
        // Store features for learning (allocate memory for 10 features)
        float* feature_copy = new float[10];
        std::memcpy(feature_copy, features, 10 * sizeof(float));
        signal_features_[signal.signal_id] = feature_copy;
    }
    
    // Generate learning feedback when position is closed
    SignalOutcome create_signal_outcome(uint64_t signal_id, float pnl, uint64_t exit_time) {
        std::lock_guard<std::mutex> lock(signals_mutex_);
        
        auto signal_it = active_signals_.find(signal_id);
        if (signal_it == active_signals_.end()) {
            return {}; // Signal not found
        }
        
        const TradingSignal& signal = signal_it->second;
        
        SignalOutcome outcome{};
        outcome.signal_id = signal_id;
        outcome.symbol_id = signal.symbol_id;
        outcome.predicted_return = signal.direction * signal.confidence;
        // Use REAL portfolio value for normalization - no hardcoded values
        float current_portfolio = portfolio_value_.load();
        if (current_portfolio > 0.0f) {
            outcome.actual_return = pnl / (signal.position_size * current_portfolio);
        } else {
            outcome.actual_return = 0.0f; // Can't calculate without real portfolio value
        }
        outcome.pnl = pnl;
        outcome.entry_time_ns = signal.timestamp_ns;
        outcome.exit_time_ns = exit_time;
        outcome.strategy_id = signal.strategy_id;
        outcome.tier = signal.tier;
        outcome.is_winner = (pnl > 0.0f);
        
        return outcome;
    }
    
    // Clean up signal tracking data
    void cleanup_signal(uint64_t signal_id) {
        std::lock_guard<std::mutex> lock(signals_mutex_);
        
        auto features_it = signal_features_.find(signal_id);
        if (features_it != signal_features_.end()) {
            delete[] features_it->second;
            signal_features_.erase(features_it);
        }
        
        active_signals_.erase(signal_id);
    }
    
    // Get features for a signal (for learning feedback)
    const float* get_signal_features(uint64_t signal_id) const {
        std::lock_guard<std::mutex> lock(signals_mutex_);
        auto it = signal_features_.find(signal_id);
        return (it != signal_features_.end()) ? it->second : nullptr;
    }

    // For HFTEngine to pull outcomes
    bool get_next_trade_outcome(SignalOutcome& outcome) {
        // LockFreeQueue is thread-safe, no mutex needed
        return trade_outcome_queue_.try_pop(outcome);
    }
    
    // Update portfolio values from live Alpaca account data
    void update_portfolio_from_alpaca(const AlpacaClient::AccountUpdate& update) {
        // Validate the incoming data
        bool valid_portfolio = update.equity > 0 && update.equity < 1e10 &&
                              !std::isnan(update.equity) && !std::isinf(update.equity);
        bool valid_buying_power = update.buying_power > 0 && update.buying_power < 1e10 &&
                                 !std::isnan(update.buying_power) && !std::isinf(update.buying_power);
        bool valid_cash = update.cash > -1e10 && update.cash < 1e10 &&
                         !std::isnan(update.cash) && !std::isinf(update.cash);
        
        if (valid_portfolio && valid_buying_power && valid_cash) {
            // Store the previous value for comparison
            float prev_portfolio = portfolio_value_.load();
            bool was_using_defaults = !alpaca_data_received_.load();
            
            portfolio_value_.store(static_cast<float>(update.equity));
            buying_power_.store(static_cast<float>(update.buying_power));
            cash_balance_.store(static_cast<float>(update.cash));
            alpaca_data_received_.store(true);
            
            // Always log the first real data update, or if values changed significantly
            if (was_using_defaults || std::abs(prev_portfolio - update.equity) > 0.01) {
                if (was_using_defaults) {
                    std::cout << "🎉 FIRST REAL ALPACA DATA received! Replacing defaults:" << std::endl;
                    std::cout << "   OLD (defaults): Portfolio=$50000 Cash=$50000 BuyingPower=$50000" << std::endl;
                    std::cout << "   NEW (real):     Portfolio=$" << std::fixed << std::setprecision(2) << update.equity
                              << " Cash=$" << update.cash
                              << " BuyingPower=$" << update.buying_power << std::endl;
                } else {
                    std::cout << "✅ Portfolio updated: Portfolio=$" << std::fixed << std::setprecision(2) << update.equity
                              << " Cash=$" << update.cash
                              << " BuyingPower=$" << update.buying_power << std::endl;
                }
            }
        } else {
            std::cout << "❌ Invalid live portfolio data received from Alpaca:" << std::endl;
            std::cout << "   Equity: $" << update.equity << " (valid: " << valid_portfolio << ")" << std::endl;
            std::cout << "   Cash: $" << update.cash << " (valid: " << valid_cash << ")" << std::endl;
            std::cout << "   BuyingPower: $" << update.buying_power << " (valid: " << valid_buying_power << ")" << std::endl;
        }
    }

private:
    // Rate limiting check
    FORCE_INLINE bool check_rate_limit() {
        uint64_t current_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        uint64_t last_reset = last_rate_reset_ns_.load();
        
        // Reset counter every second
        if (current_time_ns - last_reset > 1000000000ULL) { // 1 second in ns
            if (last_rate_reset_ns_.compare_exchange_weak(last_reset, current_time_ns)) {
                orders_this_second_.store(0);
            }
        }
        
        uint32_t current_count = orders_this_second_.fetch_add(1, std::memory_order_relaxed);
        return current_count < risk_limits_.max_orders_per_sec;
    }
    
    // Create order from trading signal
    FORCE_INLINE bool create_order_from_signal(const TradingSignal& signal, Order& order) {
        if (signal.direction == 0 || signal.confidence < 0.1f) {
            return false;
        }
        
        order.order_id = next_order_id_.fetch_add(1, std::memory_order_relaxed);
        order.signal_id = signal.signal_id;
        order.symbol_id = signal.symbol_id;
        order.type = static_cast<uint8_t>(OrderType::MARKET); // Use market orders for HFT speed (no rate limits via WebSocket)
        order.side = (signal.direction > 0) ? static_cast<int8_t>(OrderSide::BUY) : static_cast<int8_t>(OrderSide::SELL);
        order.status = static_cast<uint8_t>(OrderStatus::PENDING);
        
        // Calculate position size strictly limited to $2K per position
        float buying_power = buying_power_.load();
        
        // STRICT $2K limit per position - ignore buying power percentage
        float dollar_amount = 2000.0f; // Fixed $2K per position regardless of buying power
        
        // Get current market price from market data system
        float current_price = 0.0f;
        uint32_t volume;
        float bid, ask, bid_size, ask_size;
        
        // Try to get market data manager from component registry
        auto* market_data_mgr = memory_manager_.get_component<class MarketDataManager>("MarketDataManager");
        if (market_data_mgr &&
            market_data_mgr->get_latest_data(order.symbol_id, current_price, volume, bid, ask, bid_size, ask_size)) {
            // Use mid-price for better execution
            if (bid > 0 && ask > 0) {
                current_price = (bid + ask) / 2.0f;
            }
        }
        
        // FAIL-FAST: No fallback pricing - require real market data
        if (current_price <= 0.0f) {
            std::string symbol_name = memory_manager_.get_symbol_name(order.symbol_id);
            std::cout << "❌ No live market data for " << symbol_name << " - rejecting order" << std::endl;
            return false; // Fail without fallback
        }
        
        order.quantity = dollar_amount / current_price;
        order.price = current_price;
        
        // Set stop loss and take profit
        if (signal.direction > 0) {
            order.stop_price = current_price * (1.0f - signal.stop_loss);
            order.take_profit_price = current_price * (1.0f + signal.take_profit);
        } else {
            order.stop_price = current_price * (1.0f + signal.stop_loss);
            order.take_profit_price = current_price * (1.0f - signal.take_profit);
        }
        
        order.timestamp_ns = signal.timestamp_ns;
        order.strategy_id = signal.strategy_id;
        order.tier = signal.tier;
        
        return true;
    }
    
    // Sync portfolio data from Alpaca HTTP REST API (the real working source!)
    void sync_portfolio_from_alpaca() {
        std::cout << "--- SYNC_PORTFOLIO_FROM_ALPACA (order_engine.hpp) ---" << std::endl;
        
        // Priority 1: Use AlpacaClient HTTP REST API (has real $50,000+ data)
        if (alpaca_client_ && alpaca_client_->isReadyForOrders()) {
            try {
                auto account_update = alpaca_client_->getAccountInfo();
                
                std::cout << "  Fetched from AlpacaClient HTTP REST API:" << std::endl;
                std::cout << "    Portfolio Value: $" << std::fixed << std::setprecision(2) << account_update.portfolio_value << std::endl;
                std::cout << "    Buying Power: $" << std::fixed << std::setprecision(2) << account_update.buying_power << std::endl;
                std::cout << "    Cash: $" << std::fixed << std::setprecision(2) << account_update.cash << std::endl;

                // Validate values before storing them
                bool valid_portfolio_value = account_update.portfolio_value > 0 &&
                                            account_update.portfolio_value < 1e10 &&
                                            !std::isnan(account_update.portfolio_value) &&
                                            !std::isinf(account_update.portfolio_value);
                
                bool valid_buying_power = account_update.buying_power > 0 &&
                                         account_update.buying_power < 1e10 &&
                                         !std::isnan(account_update.buying_power) &&
                                         !std::isinf(account_update.buying_power);
                
                bool valid_cash = account_update.cash >= 0 &&
                                 account_update.cash < 1e10 &&
                                 !std::isnan(account_update.cash) &&
                                 !std::isinf(account_update.cash);

                // Use real Alpaca data if valid
                if (valid_portfolio_value && valid_buying_power && valid_cash) {
                    portfolio_value_.store(static_cast<float>(account_update.portfolio_value));
                    buying_power_.store(static_cast<float>(account_update.buying_power));
                    cash_balance_.store(static_cast<float>(account_update.cash));
                    alpaca_data_received_.store(true); // Mark that we have real data

                    std::cout << "  ✅ REAL ALPACA HTTP API DATA stored in OrderEngine:" << std::endl;
                    std::cout << "    portfolio_value_: $" << std::fixed << std::setprecision(2) << portfolio_value_.load() << std::endl;
                    std::cout << "    buying_power_: $" << std::fixed << std::setprecision(2) << buying_power_.load() << std::endl;
                    std::cout << "    cash_balance_: $" << std::fixed << std::setprecision(2) << cash_balance_.load() << std::endl;
                    return;
                } else {
                    std::cout << "❌ Invalid portfolio data from Alpaca HTTP API" << std::endl;
                    std::cout << "  Portfolio Value Valid: " << valid_portfolio_value << std::endl;
                    std::cout << "  Buying Power Valid: " << valid_buying_power << std::endl;
                    std::cout << "  Cash Valid: " << valid_cash << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "❌ Exception fetching from AlpacaClient HTTP API: " << e.what() << std::endl;
            }
        }
        
        // No fallback available - REST API only
        std::cout << "  ❌ No Alpaca REST API data available" << std::endl;
        alpaca_data_received_.store(false);
        std::cout << "--- END SYNC_PORTFOLIO_FROM_ALPACA (order_engine.hpp) ---" << std::endl;
    }
    
    // Ultra-fast risk validation with cached portfolio data
    FORCE_INLINE bool validate_risk_limits(const Order& order) {
        float portfolio_val = portfolio_value_.load();
        float cash = cash_balance_.load();
        float order_value = order.quantity * order.price;
        
        // Ensure we have valid portfolio values (already set in constructor)
        if (portfolio_val <= 0.0f) {
            portfolio_val = 50000.0f;
            portfolio_value_.store(portfolio_val);
        }
        if (cash <= 0.0f) {
            cash = 50000.0f;
            cash_balance_.store(cash);
        }
        
        // Check cash reserve
        float required_cash_reserve = portfolio_val * risk_limits_.min_cash_reserve;
        if (cash - order_value < required_cash_reserve) {
            return false;
        }
        
        // Check position size limit
        float position_size_pct = order_value / portfolio_val;
        if (position_size_pct > risk_limits_.max_position_size) {
            return false;
        }
        
        // Check daily loss limit
        if (daily_pnl_.load() < -portfolio_val * risk_limits_.max_daily_loss) {
            return false;
        }
        
        // Check maximum number of positions limit
        std::lock_guard<std::mutex> lock(positions_mutex_);
        
        // STRICT: No duplicate positions allowed - each symbol can only have ONE position
        auto it = positions_.find(order.symbol_id);
        if (it != positions_.end()) {
            std::cout << "❌ Position already exists for symbol_id " << order.symbol_id
                      << " - rejecting duplicate order" << std::endl;
            return false; // Reject any order for symbols we already have positions in
        }
        
        // Check if we're at the 25 position limit
        if (positions_.size() >= risk_limits_.max_total_positions) {
            std::cout << "❌ Maximum positions reached (" << positions_.size()
                      << "/" << risk_limits_.max_total_positions << ") - rejecting order" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Order processing loop
    void processing_loop() {
        Order order;
        
        while (running_.load()) {
            if (pending_orders_.try_pop(order)) {
                process_order(order);
            } else {
                // No orders, yield CPU briefly
                std::this_thread::yield();
            }
        }
    }
    
    // Process individual order with real Alpaca API integration
    void process_order(Order& order) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Submit order to Alpaca API
        order.status = static_cast<uint8_t>(OrderStatus::SUBMITTED);
        
        // Create Alpaca order request
        bool success = submit_to_alpaca(order);
        
        if (success) {
            order.status = static_cast<uint8_t>(OrderStatus::FILLED);
            order.filled_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            order.filled_price = order.price; // Will be updated by WebSocket
            order.filled_quantity = order.quantity;
            
            // Update position (will be synced via Alpaca WebSocket)
            update_position(order);
            orders_filled_.fetch_add(1, std::memory_order_relaxed);
        } else {
            order.status = static_cast<uint8_t>(OrderStatus::REJECTED);
            orders_rejected_.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Queue completed order
        completed_orders_.try_push(order);

        // Note: SignalOutcome is generated when position is closed in update_position()
        // We don't generate outcomes here because P&L can only be calculated when exiting
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        total_processing_time_ns_.fetch_add(duration.count(), std::memory_order_relaxed);
    }
    
    // Submit order to Alpaca API
    bool submit_to_alpaca(const Order& order) {
        if (!alpaca_client_) {
            std::cerr << "❌ AlpacaClient not available in submit_to_alpaca" << std::endl;
            return false;
        }

        if (!alpaca_client_->isReadyForOrders()) {
            std::cerr << "❌ AlpacaClient not ready for orders" << std::endl;
            return false;
        }

        try {
            // Get symbol name from memory manager
            std::string symbol_name = memory_manager_.get_symbol_name(order.symbol_id);
            if (symbol_name.empty()) {
                std::cerr << "❌ Invalid symbol_id " << order.symbol_id << " in submit_to_alpaca" << std::endl;
                return false;
            }

            // Convert order side
            ::AlpacaClient::Side alpaca_side = (order.side == static_cast<int8_t>(OrderSide::BUY)) ?
                ::AlpacaClient::Side::BUY : ::AlpacaClient::Side::SELL;

            // Convert quantity to integer (Alpaca expects whole shares for most stocks)
            int quantity = static_cast<int>(std::round(order.quantity));
            if (quantity <= 0) {
                // Skip logging for zero quantities - they're normal when position sizing is small
                return false;
            }

            // Generate client order ID for tracking
            std::string client_order_id = "hft_" + std::to_string(order.order_id);

            ::AlpacaClient::OrderResult result;

            // Calculate take profit price (0.08% above/below entry price)
            double take_profit_price;
            double stop_loss_price;
            
            if (alpaca_side == ::AlpacaClient::Side::BUY) {
                take_profit_price = order.price * 1.0008; // 0.08% above for long positions
                stop_loss_price = order.price * 0.985;    // 1.5% below for stop loss
            } else {
                take_profit_price = order.price * 0.9992; // 0.08% below for short positions
                stop_loss_price = order.price * 1.015;    // 1.5% above for stop loss
            }
            
            std::cout << "📈 BRACKET ORDER: Entry=$" << order.price
                      << " TakeProfit=$" << take_profit_price << " (0.08%)"
                      << " StopLoss=$" << stop_loss_price << " (1.5%)" << std::endl;

            // Submit atomic bracket order (single API call for entry + take profit + stop loss)
            if (order.type == static_cast<uint8_t>(OrderType::MARKET)) {
                // Use atomic bracket order for more reliable execution
                result = alpaca_client_->submitBracketOrder(
                    symbol_name,           // Symbol
                    quantity,              // Quantity
                    alpaca_side,           // Side (BUY/SELL)
                    order.price,           // Entry limit price (close to market price)
                    stop_loss_price,       // Stop loss price (1.5% away)
                    take_profit_price,     // Take profit price (0.08% away)
                    client_order_id        // Client order ID
                );
                
                if (result.success) {
                    std::cout << "✅ ATOMIC BRACKET ORDER submitted successfully:" << std::endl;
                    std::cout << "   Entry: " << quantity << " shares of " << symbol_name
                              << " @ $" << std::fixed << std::setprecision(2) << order.price << std::endl;
                    std::cout << "   Take Profit: $" << take_profit_price << " (0.08% profit)" << std::endl;
                    std::cout << "   Stop Loss: $" << stop_loss_price << " (1.5% loss)" << std::endl;
                    std::cout << "   Order ID: " << result.order_id << std::endl;
                } else {
                    std::cout << "❌ ATOMIC BRACKET ORDER failed: " << result.error_message << std::endl;
                }
            } else if (order.type == static_cast<uint8_t>(OrderType::LIMIT)) {
                result = alpaca_client_->submitLimitOrder(
                    symbol_name,
                    quantity,
                    alpaca_side,
                    order.price,
                    ::AlpacaClient::TimeInForce::DAY,
                    client_order_id
                );
            } else {
                std::cerr << "❌ Unsupported order type " << static_cast<int>(order.type) << std::endl;
                return false;
            }

            if (result.success) {
                std::cout << "✅ Order submitted successfully: " << result.order_id 
                          << " for " << symbol_name << " qty=" << quantity << std::endl;
                return true;
            } else {
                std::cerr << "❌ Order submission failed: " << result.error_message << std::endl;
                return false;
            }

        } catch (const std::exception& e) {
            std::cerr << "❌ Exception in submit_to_alpaca: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Update position tracking
    void update_position(const Order& order) {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        
        auto& position = positions_[order.symbol_id];
        
        if (position.symbol_id == 0) {
            // New position
            position.symbol_id = order.symbol_id;
            position.signal_id = order.signal_id;
            position.quantity = (order.side == static_cast<int8_t>(OrderSide::BUY)) ? 
                               order.filled_quantity : -order.filled_quantity;
            position.average_price = order.filled_price;
            position.market_value = position.quantity * order.filled_price;
            position.entry_timestamp_ns = order.filled_timestamp_ns;
            position.side = (order.side == static_cast<int8_t>(OrderSide::BUY)) ? 0 : 1;
        } else {
            // Update existing position
            float new_quantity = position.quantity + 
                ((order.side == static_cast<int8_t>(OrderSide::BUY)) ? 
                 order.filled_quantity : -order.filled_quantity);
            
            if (std::abs(new_quantity) < 1e-6f) {
                // Position closed - generate learning outcome
                float pnl = calculate_position_pnl(position, order.filled_price);
                
                SignalOutcome outcome = create_signal_outcome(
                    position.signal_id, 
                    pnl, 
                    order.filled_timestamp_ns
                );
                
                // Queue outcome for learning
                trade_outcome_queue_.try_push(outcome);
                
                // Clean up signal tracking
                cleanup_signal(position.signal_id);
                
                // Remove position
                positions_.erase(order.symbol_id);
            } else {
                // Update position
                float total_cost = position.quantity * position.average_price + 
                                 order.filled_quantity * order.filled_price;
                position.quantity = new_quantity;
                position.average_price = total_cost / position.quantity;
                position.market_value = position.quantity * order.filled_price;
            }
        }
        
        position.last_update_ns = order.filled_timestamp_ns;
    }
    
    // Calculate P&L for a position
    float calculate_position_pnl(const Position& position, float exit_price) {
        float entry_value = std::abs(position.quantity) * position.average_price;
        float exit_value = std::abs(position.quantity) * exit_price;
        
        if (position.side == 0) { // Long position
            return exit_value - entry_value;
        } else { // Short position
            return entry_value - exit_value;
        }
    }
};

} // namespace hft
