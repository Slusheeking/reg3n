#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>
#include <vector>
#include <random>
#include "config.hpp"
#include "beast_http_client.hpp"

/**
 * @brief Ultra-low latency Alpaca trading client with Boost.Beast HTTP implementation
 * 
 * This class provides:
 * - Modern Boost.Beast HTTP client for order submission
 * - Rate limit management (180 requests/minute)
 * - Priority order queuing system
 * - Sub-10ms order submission latency
 * - Comprehensive performance monitoring
 * - 5-minute reconciliation system
 * - No Alpaca SDK dependencies
 */
class AlpacaClient {
public:
    /**
     * @brief Order side enumeration
     */
    enum class Side {
        BUY,
        SELL
    };

    /**
     * @brief Order type enumeration
     */
    enum class Type {
        MARKET,
        LIMIT,
        BRACKET
    };

    /**
     * @brief Time in force enumeration
     */
    enum class TimeInForce {
        DAY,
        GTC,  // Good Till Canceled
        IOC,  // Immediate or Cancel
        FOK   // Fill or Kill
    };

    /**
     * @brief Order priority levels
     */
    enum class Priority {
        CRITICAL = 1,  // Emergency exits, stop losses
        HIGH = 2,      // High-confidence signals
        NORMAL = 3,    // Regular trading signals
        LOW = 4,       // Opportunistic trades
        BATCH = 5      // Batch operations
    };

    /**
     * @brief Order result structure with comprehensive metrics
     */
    struct OrderResult {
        bool success{false};
        std::string order_id;
        std::string client_order_id;
        std::string error_message;
        std::chrono::nanoseconds total_latency{0};
        std::chrono::nanoseconds queue_time{0};
        std::chrono::nanoseconds network_latency{0};
        std::chrono::nanoseconds processing_time{0};
        int retry_count{0};
        std::chrono::steady_clock::time_point submission_time;
        unsigned int http_status_code{0};
    };

    /**
     * @brief Trade update structure for reconciliation
     */
    struct TradeUpdate {
        std::string order_id;
        std::string client_order_id;
        std::string symbol;
        std::string side;
        int quantity{0};
        double price{0.0};
        double filled_qty{0.0};
        double filled_avg_price{0.0};
        std::string status;
        std::chrono::steady_clock::time_point timestamp;
        std::string raw_json;
    };

    /**
     * @brief Account update structure
     */
    struct AccountUpdate {
        std::string account_id;
        std::string status;
        double buying_power{0.0};
        double equity{0.0};
        double cash{0.0};
        double portfolio_value{0.0};
        double day_trade_buying_power{0.0};
        bool pattern_day_trader{false};
        bool trading_blocked{false};
        bool transfers_blocked{false};
        bool account_blocked{false};
        int daytrade_count{0};
        double realized_pnl{0.0};
        double unrealized_pnl{0.0};
        std::chrono::steady_clock::time_point timestamp;
        std::string raw_json;
    };

    /**
     * @brief Position structure
     */
    struct Position {
        std::string symbol;
        std::string asset_id;
        int quantity{0};
        double avg_entry_price{0.0};
        double market_value{0.0};
        double cost_basis{0.0};
        double unrealized_pl{0.0};
        double unrealized_plpc{0.0};  // Unrealized P&L percentage
        double current_price{0.0};
        std::string side;  // "long" or "short"
        std::chrono::steady_clock::time_point timestamp;
    };

    /**
     * @brief Rate limiter statistics
     */
    struct RateLimitStats {
        int requests_in_current_window{0};
        int max_requests_per_window{180};
        std::chrono::milliseconds time_until_reset{0};
        double utilization_percentage{0.0};
        bool is_throttling{false};
    };

    /**
     * @brief Connection pool statistics
     */
    struct ConnectionStats {
        size_t total_connections{0};
        size_t active_connections{0};
        size_t available_connections{0};
        uint64_t total_requests_served{0};
        std::chrono::nanoseconds avg_connection_time{0};
    };

    /**
     * @brief Comprehensive performance metrics
     */
    struct PerformanceMetrics {
        uint64_t total_orders_submitted{0};
        uint64_t successful_orders{0};
        uint64_t failed_orders{0};
        uint64_t retried_orders{0};
        std::chrono::nanoseconds min_latency{0};
        std::chrono::nanoseconds max_latency{0};
        std::chrono::nanoseconds avg_latency{0};
        std::chrono::nanoseconds p95_latency{0};
        std::chrono::nanoseconds p99_latency{0};
        double success_rate{0.0};
        RateLimitStats rate_limit_stats;
        ConnectionStats connection_stats;
    };

    /**
     * @brief Reconciliation statistics
     */
    struct ReconciliationStats {
        uint64_t trades_reconciled{0};
        uint64_t trades_pending{0};
        uint64_t reconciliation_errors{0};
        std::chrono::steady_clock::time_point last_reconciliation;
        std::chrono::milliseconds avg_reconciliation_time{0};
    };

    /**
     * @brief Constructor
     */
    explicit AlpacaClient(const hft::APIConfig& config);

    /**
     * @brief Destructor - ensures proper shutdown
     */
    ~AlpacaClient();

    // Move semantics for unique_ptr members and threads
    AlpacaClient(AlpacaClient&&) noexcept = default;
    AlpacaClient& operator=(AlpacaClient&&) noexcept = default;
    
    // Delete copy semantics
    AlpacaClient(const AlpacaClient&) = delete;
    AlpacaClient& operator=(const AlpacaClient&) = delete;

    /**
     * @brief Initialize the client with Beast HTTP optimizations
     * @return true if successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Submit a market order with priority queuing
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param priority Order priority level
     * @param client_order_id Optional client order ID for tracking
     * @return Future<OrderResult> for async processing
     */
    std::future<OrderResult> submitMarketOrderAsync(const std::string& symbol,
                                                   int quantity,
                                                   Side side,
                                                   Priority priority = Priority::NORMAL,
                                                   const std::string& client_order_id = "");

    /**
     * @brief Submit a limit order with priority queuing
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param limit_price Limit price
     * @param priority Order priority level
     * @param tif Time in force
     * @param client_order_id Optional client order ID for tracking
     * @return Future<OrderResult> for async processing
     */
    std::future<OrderResult> submitLimitOrderAsync(const std::string& symbol,
                                                  int quantity,
                                                  Side side,
                                                  double limit_price,
                                                  Priority priority = Priority::NORMAL,
                                                  TimeInForce tif = TimeInForce::DAY,
                                                  const std::string& client_order_id = "");

    /**
     * @brief Submit a market order (synchronous version)
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param client_order_id Optional client order ID for tracking
     * @return OrderResult with success status and timing
     */
    OrderResult submitMarketOrder(const std::string& symbol,
                                 int quantity,
                                 Side side,
                                 const std::string& client_order_id = "");

    /**
     * @brief Submit a limit order (synchronous version)
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param limit_price Limit price
     * @param tif Time in force
     * @param client_order_id Optional client order ID for tracking
     * @return OrderResult with success status and timing
     */
    OrderResult submitLimitOrder(const std::string& symbol,
                                int quantity,
                                Side side,
                                double limit_price,
                                TimeInForce tif = TimeInForce::DAY,
                                const std::string& client_order_id = "");

    /**
     * @brief Submit a bracket order (entry + take profit + stop loss) - async version
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param limit_price Entry limit price
     * @param stop_loss_price Stop loss trigger price
     * @param take_profit_price Take profit limit price
     * @param priority Order priority level
     * @param client_order_id Optional client order ID for tracking
     * @return Future<OrderResult> for async processing
     */
    std::future<OrderResult> submitBracketOrderAsync(const std::string& symbol,
                                                    int quantity,
                                                    Side side,
                                                    double limit_price,
                                                    double stop_loss_price,
                                                    double take_profit_price,
                                                    Priority priority = Priority::NORMAL,
                                                    const std::string& client_order_id = "");

    /**
     * @brief Submit a bracket order (entry + take profit + stop loss) - synchronous version
     * @param symbol Stock symbol
     * @param quantity Number of shares
     * @param side Buy or sell
     * @param limit_price Entry limit price
     * @param stop_loss_price Stop loss trigger price
     * @param take_profit_price Take profit limit price
     * @param client_order_id Optional client order ID for tracking
     * @return OrderResult with success status and timing
     */
    OrderResult submitBracketOrder(const std::string& symbol,
                                  int quantity,
                                  Side side,
                                  double limit_price,
                                  double stop_loss_price,
                                  double take_profit_price,
                                  const std::string& client_order_id = "");

    /**
     * @brief Cancel an order
     * @param order_id Order ID to cancel
     * @return OrderResult with cancellation status
     */
    OrderResult cancelOrder(const std::string& order_id);

    /**
     * @brief Cancel all orders
     * @return Number of orders canceled
     */
    int cancelAllOrders();

    /**
     * @brief Check if market is open
     * @return true if market is open
     */
    bool isMarketOpen();

    /**
     * @brief Get account buying power
     * @return Available buying power
     */
    double getBuyingPower();

    /**
     * @brief Get account information
     * @return AccountUpdate structure with current account data
     */
    AccountUpdate getAccountInfo();

    /**
     * @brief Get all positions
     * @return Vector of Position structures
     */
    std::vector<Position> getPositions();

    /**
     * @brief Get position count
     * @return Number of active positions
     */
    int getPositionCount();

    /**
     * @brief Start account data polling (replaces WebSocket for reliability)
     * @param on_account_update Callback for account updates
     * @param poll_interval_ms Polling interval in milliseconds
     * @return true if polling started successfully
     */
    bool startAccountPolling(std::function<void(const AccountUpdate&)> on_account_update,
                           int poll_interval_ms = 5000);

    /**
     * @brief Stop account data polling
     */
    void stopAccountPolling();

    /**
     * @brief Start 5-minute reconciliation with Alpaca
     * @param on_trade_update Callback for trade updates during reconciliation
     * @return true if reconciliation started successfully
     */
    bool startReconciliation(std::function<void(const TradeUpdate&)> on_trade_update);

    /**
     * @brief Stop reconciliation
     */
    void stopReconciliation();

    /**
     * @brief Force immediate reconciliation
     * @return Number of trades reconciled
     */
    int forceReconciliation();

    /**
     * @brief Get comprehensive performance metrics
     * @return PerformanceMetrics structure
     */
    PerformanceMetrics getPerformanceMetrics() const;

    /**
     * @brief Get current rate limit status
     * @return RateLimitStats structure
     */
    RateLimitStats getRateLimitStats() const;

    /**
     * @brief Get connection pool status
     * @return ConnectionStats structure
     */
    ConnectionStats getConnectionStats() const;

    /**
     * @brief Get reconciliation statistics
     * @return ReconciliationStats structure
     */
    ReconciliationStats getReconciliationStats() const;

    /**
     * @brief Check if client is ready to submit orders
     * @return true if ready, false if throttled or error
     */
    bool isReadyForOrders() const;

    /**
     * @brief Get estimated time until next order can be submitted
     * @return Duration until next available slot
     */
    std::chrono::milliseconds getTimeUntilNextOrder() const;

private:
    /**
     * @brief Priority Order Queue System using Beast HTTP client
     */
    struct PriorityOrder {
        std::string symbol;
        int quantity;
        Side side;
        Type type;
        double limit_price;
        double take_profit_price{0.0};
        double stop_loss_price{0.0};
        TimeInForce tif;
        Priority priority;
        std::string client_order_id;
        std::chrono::steady_clock::time_point queue_time;
        std::promise<OrderResult> result_promise;
        
        // Move-only semantics for std::promise
        PriorityOrder() = default;
        PriorityOrder(const PriorityOrder&) = delete;
        PriorityOrder& operator=(const PriorityOrder&) = delete;
        PriorityOrder(PriorityOrder&&) = default;
        PriorityOrder& operator=(PriorityOrder&&) = default;
        
        bool operator<(const PriorityOrder& other) const {
            if (priority != other.priority) {
                return static_cast<int>(priority) > static_cast<int>(other.priority);
            }
            return queue_time > other.queue_time; // Earlier orders first
        }
    };

    // Comparator for unique_ptr<PriorityOrder>
    struct PriorityOrderCompare {
        bool operator()(const std::unique_ptr<PriorityOrder>& a, 
                       const std::unique_ptr<PriorityOrder>& b) const {
            return *a < *b;
        }
    };

    class OrderQueue {
    public:
        OrderQueue(hft::BeastHttpClient& http_client);
        ~OrderQueue();
        
        std::future<OrderResult> submitOrder(PriorityOrder order);
        void start();
        void stop();
        
    private:
        void processOrders();
        OrderResult executeOrder(const PriorityOrder& order);
        
        hft::BeastHttpClient& http_client_;
        // Use unique_ptr to handle non-copyable PriorityOrder
        std::priority_queue<std::unique_ptr<PriorityOrder>, 
                           std::vector<std::unique_ptr<PriorityOrder>>, 
                           PriorityOrderCompare> pending_orders_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        std::thread processing_thread_;
        std::atomic<bool> running_{false};
    };

    /**
     * @brief Performance Tracker
     */
    class PerformanceTracker {
    public:
        void recordOrder(const OrderResult& result);
        PerformanceMetrics getMetrics() const;
        void reset();
        
    private:
        mutable std::mutex metrics_mutex_;
        std::vector<std::chrono::nanoseconds> latency_samples_;
        std::atomic<uint64_t> total_orders_{0};
        std::atomic<uint64_t> successful_orders_{0};
        std::atomic<uint64_t> failed_orders_{0};
        std::atomic<uint64_t> retried_orders_{0};
    };

    /**
     * @brief Reconciliation Manager for 5-minute trade reconciliation
     */
    class ReconciliationManager {
    public:
        ReconciliationManager(hft::BeastHttpClient& http_client);
        ~ReconciliationManager();
        
        bool start(std::function<void(const TradeUpdate&)> callback);
        void stop();
        int forceReconciliation();
        ReconciliationStats getStats() const;
        
    private:
        void reconciliationLoop();
        std::vector<TradeUpdate> fetchTradeUpdates();
        
        hft::BeastHttpClient& http_client_;
        std::function<void(const TradeUpdate&)> trade_callback_;
        std::thread reconciliation_thread_;
        std::atomic<bool> running_{false};
        
        mutable std::mutex stats_mutex_;
        std::atomic<uint64_t> trades_reconciled_{0};
        std::atomic<uint64_t> trades_pending_{0};
        std::atomic<uint64_t> reconciliation_errors_{0};
        std::chrono::steady_clock::time_point last_reconciliation_;
        std::atomic<uint64_t> total_reconciliation_time_ms_{0};
        std::atomic<uint64_t> reconciliation_count_{0};
    };

    /**
     * @brief Account Polling Manager
     */
    class AccountPollingManager {
    public:
        AccountPollingManager(hft::BeastHttpClient& http_client);
        ~AccountPollingManager();
        
        bool start(std::function<void(const AccountUpdate&)> callback, int interval_ms);
        void stop();
        AccountUpdate fetchAccountUpdate();
        
    private:
        void pollingLoop();
        
        hft::BeastHttpClient& http_client_;
        std::function<void(const AccountUpdate&)> account_callback_;
        std::thread polling_thread_;
        std::atomic<bool> running_{false};
        std::atomic<int> poll_interval_ms_{5000};
    };

    // Convert enums to strings
    std::string sideToString(Side side) const;
    std::string typeToString(Type type) const;
    std::string tifToString(TimeInForce tif) const;

    // Generate unique client order ID
    std::string generateClientOrderId();

    // JSON parsing helpers
    bool parseOrderResponse(const std::string& json, OrderResult& result);
    bool parseAccountResponse(const std::string& json, AccountUpdate& update);
    bool parseTradeResponse(const std::string& json, std::vector<TradeUpdate>& trades);

    // Configuration
    hft::APIConfig config_;
    
    // Boost.Beast HTTP client
    std::unique_ptr<boost::asio::io_context> io_context_;
    std::unique_ptr<hft::BeastHttpClient> http_client_;
    std::thread io_thread_;
    
    // Optimization components
    std::unique_ptr<OrderQueue> order_queue_;
    std::unique_ptr<PerformanceTracker> performance_tracker_;
    std::unique_ptr<ReconciliationManager> reconciliation_manager_;
    std::unique_ptr<AccountPollingManager> account_polling_manager_;
    
    // Thread safety
    mutable std::mutex client_mutex_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    
    // Random number generator for client order IDs
    std::random_device rd_;
    std::mt19937 gen_;
    
    // Market status caching
    mutable std::chrono::steady_clock::time_point last_market_check_;
    mutable bool cached_market_open_ = false;
    
    // Buying power caching
    mutable std::chrono::steady_clock::time_point last_buying_power_check_;
    mutable double cached_buying_power_ = 0.0;
    
    // Current account cache
    mutable AccountUpdate current_account_;
};
