#include "alpaca_client.hpp"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <immintrin.h> // For _mm_pause
#include <stdexcept> // For std::stod, std::runtime_error

// Constructor
AlpacaClient::AlpacaClient(const hft::APIConfig& config) 
    : config_(config), 
      gen_(rd_()) {
    
    io_context_ = std::make_unique<boost::asio::io_context>();
    http_client_ = std::make_unique<hft::BeastHttpClient>(*io_context_, "paper-api.alpaca.markets", "443", 180);
    
    // API KEYS
    std::string api_key = "PKOA0DZRDVPMC7V6A4EU";
    std::string secret_key = "1BM2AiVp8N6Glbc5fm14up7KEh1V8KNFleD5jgYu";
    
    std::cout << "🔑 Using API keys:" << std::endl;
    std::cout << "   API Key: " << api_key << std::endl;
    std::cout << "   Secret Key: " << secret_key.substr(0, 8) << "..." << std::endl;
    
    http_client_->set_default_header("APCA-API-KEY-ID", api_key);
    http_client_->set_default_header("APCA-API-SECRET-KEY", secret_key);
    http_client_->set_default_header("Content-Type", "application/json");
    http_client_->set_user_agent("HFT-Beast-Client/1.0");
    
    order_queue_ = std::make_unique<OrderQueue>(*http_client_);
    performance_tracker_ = std::make_unique<PerformanceTracker>();
    reconciliation_manager_ = std::make_unique<ReconciliationManager>(*http_client_);
    account_polling_manager_ = std::make_unique<AccountPollingManager>(*http_client_);
    
    std::cout << "🚀 AlpacaClient initialized with Beast HTTP optimizations" << std::endl;
}

AlpacaClient::~AlpacaClient() {
    if (running_.load()) {
        if (order_queue_) order_queue_->stop();
        if (reconciliation_manager_) reconciliation_manager_->stop();
        if (account_polling_manager_) account_polling_manager_->stop();
        
        running_.store(false);
        
        if (io_thread_.joinable()) {
            if (io_context_ && !io_context_->stopped()) {
                io_context_->stop();
            }
            io_thread_.join();
        }
    }
    std::cout << "🛑 AlpacaClient destroyed safely" << std::endl;
}

bool AlpacaClient::initialize() {
    std::lock_guard<std::mutex> lock(client_mutex_);
    if (initialized_.load()) return true;
    
    try {
        if (!http_client_ || !http_client_->initialize()) {
            std::cerr << "❌ Failed to initialize Beast HTTP client" << std::endl;
            return false;
        }
        http_client_->start();
        
        io_thread_ = std::thread([this]() {
            try {
                std::cerr << "🔍 DEBUG (io_thread_): IO context thread started" << std::endl;
                if (io_context_) {
                    std::cerr << "🔍 DEBUG (io_thread_): Creating work_guard." << std::endl;
                    auto work_guard = boost::asio::make_work_guard(*io_context_);
                    std::cerr << "🔍 DEBUG (io_thread_): Starting io_context->run()..." << std::endl;
                    io_context_->run(); // This is the blocking call
                    std::cerr << "🔍 DEBUG (io_thread_): io_context->run() completed." << std::endl;
                } else {
                    std::cerr << "❌ ERROR (io_thread_): io_context_ is null!" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ IO context thread error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ IO context thread caught unknown exception!" << std::endl;
            }
            std::cerr << "🔍 DEBUG (io_thread_): IO context thread exiting." << std::endl;
        });
        
        if (order_queue_) order_queue_->start();
        
        running_.store(true);
        initialized_.store(true);
        std::cout << "✅ AlpacaClient initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ AlpacaClient initialization failed: " << e.what() << std::endl;
        return false;
    }
}

// OrderQueue Implementation
AlpacaClient::OrderQueue::OrderQueue(hft::BeastHttpClient& http_client) : http_client_(http_client) {
    std::cout << "🚀 Beast HTTP order queue initialized" << std::endl;
}
AlpacaClient::OrderQueue::~OrderQueue() { stop(); std::cout << "🛑 Beast HTTP order queue destroyed" << std::endl; }
void AlpacaClient::OrderQueue::start() {
    if (running_.load()) return;
    running_.store(true);
    processing_thread_ = std::thread(&OrderQueue::processOrders, this);
    std::cout << "🚀 Beast HTTP order queue started" << std::endl;
}
void AlpacaClient::OrderQueue::stop() {
    if (!running_.load()) return;
    running_.store(false);
    queue_cv_.notify_all();
    if (processing_thread_.joinable()) processing_thread_.join();
    std::cout << "🛑 Beast HTTP order queue stopped" << std::endl;
}
void AlpacaClient::OrderQueue::processOrders() {
    std::cout << "🚀 Ultra-fast order processing started" << std::endl;
    while (running_.load(std::memory_order_relaxed)) {
        PriorityOrder order_container;
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !pending_orders_.empty() || !running_.load(); });
        if (!running_.load()) break;
        if (!pending_orders_.empty()) {
            std::unique_ptr<PriorityOrder> order_ptr = std::move(const_cast<std::unique_ptr<PriorityOrder>&>(pending_orders_.top()));
            pending_orders_.pop();
            lock.unlock();
            order_container = std::move(*order_ptr);
        } else { continue; }
        
        auto processing_start = std::chrono::steady_clock::now();
        auto queue_duration = processing_start - order_container.queue_time;
        try {
            OrderResult result = executeOrder(order_container);
            result.queue_time = queue_duration;
            result.processing_time = std::chrono::steady_clock::now() - processing_start;
            result.total_latency = result.queue_time + result.network_latency + result.processing_time;
            order_container.result_promise.set_value(std::move(result));
        } catch (const std::exception& e) {
            OrderResult error_result;
            error_result.success = false;
            error_result.error_message = e.what();
            error_result.queue_time = queue_duration;
            error_result.processing_time = std::chrono::steady_clock::now() - processing_start;
            order_container.result_promise.set_value(std::move(error_result));
        }
    }
    std::cout << "🛑 Ultra-fast order processing stopped" << std::endl;
}
std::future<AlpacaClient::OrderResult> AlpacaClient::OrderQueue::submitOrder(PriorityOrder order) {
    auto future = order.result_promise.get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_orders_.push(std::make_unique<PriorityOrder>(std::move(order)));
    }
    queue_cv_.notify_one();
    return future;
}
AlpacaClient::OrderResult AlpacaClient::OrderQueue::executeOrder(const PriorityOrder& order) {
    auto start_time = std::chrono::steady_clock::now();
    OrderResult result;
    result.client_order_id = order.client_order_id;
    result.submission_time = start_time;
    try {
        rapidjson::Document doc;
        doc.SetObject();
        auto& allocator = doc.GetAllocator();
        doc.AddMember("symbol", rapidjson::Value(order.symbol.c_str(), allocator), allocator);
        doc.AddMember("qty", rapidjson::Value(std::to_string(order.quantity).c_str(), allocator), allocator);
        doc.AddMember("side", rapidjson::Value(order.side == Side::BUY ? "buy" : "sell", allocator), allocator);
        doc.AddMember("time_in_force", rapidjson::Value("day", allocator), allocator);
        if (!order.client_order_id.empty()) {
             doc.AddMember("client_order_id", rapidjson::Value(order.client_order_id.c_str(), allocator), allocator);
        }
        
        if (order.type == Type::BRACKET) {
            // Bracket order implementation
            doc.AddMember("type", rapidjson::Value("limit", allocator), allocator);
            doc.AddMember("order_class", rapidjson::Value("bracket", allocator), allocator);
            
            // Main order limit price
            std::ostringstream limit_oss;
            limit_oss << std::fixed << std::setprecision(2) << order.limit_price;
            doc.AddMember("limit_price", rapidjson::Value(limit_oss.str().c_str(), allocator), allocator);
            
            // Take profit leg
            rapidjson::Value take_profit(rapidjson::kObjectType);
            std::ostringstream tp_oss;
            tp_oss << std::fixed << std::setprecision(2) << order.take_profit_price;
            take_profit.AddMember("limit_price", rapidjson::Value(tp_oss.str().c_str(), allocator), allocator);
            doc.AddMember("take_profit", take_profit, allocator);
            
            // Stop loss leg
            rapidjson::Value stop_loss(rapidjson::kObjectType);
            std::ostringstream sl_oss;
            sl_oss << std::fixed << std::setprecision(2) << order.stop_loss_price;
            stop_loss.AddMember("stop_price", rapidjson::Value(sl_oss.str().c_str(), allocator), allocator);
            // For stop-limit orders, set limit price slightly worse than stop price
            double stop_limit_price = order.side == Side::BUY ?
                order.stop_loss_price * 1.001 : order.stop_loss_price * 0.999;
            std::ostringstream sll_oss;
            sll_oss << std::fixed << std::setprecision(2) << stop_limit_price;
            stop_loss.AddMember("limit_price", rapidjson::Value(sll_oss.str().c_str(), allocator), allocator);
            doc.AddMember("stop_loss", stop_loss, allocator);
        } else {
            // Regular market/limit order
            doc.AddMember("type", rapidjson::Value(order.type == Type::MARKET ? "market" : "limit", allocator), allocator);
            if (order.type == Type::LIMIT) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << order.limit_price;
                doc.AddMember("limit_price", rapidjson::Value(oss.str().c_str(), allocator), allocator);
            }
        }
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        http_client_.async_post("/v2/orders", buffer.GetString(),
            [&promise](hft::HttpResponse response) { promise.set_value(std::move(response)); });
        
        auto http_response = future.get();
        result.http_status_code = http_response.status_code;
        result.network_latency = http_response.latency;
        if (http_response.success && (http_response.status_code == 200 || http_response.status_code == 201)) {
            rapidjson::Document response_doc;
            response_doc.Parse(http_response.body.c_str());
            if (!response_doc.HasParseError() && response_doc.HasMember("id") && response_doc["id"].IsString()) {
                result.success = true;
                result.order_id = response_doc["id"].GetString();
            } else {
                result.success = false;
                result.error_message = "Failed to parse order response JSON. Body: " + http_response.body;
            }
        } else {
            result.success = false;
            result.error_message = http_response.error_message.empty() ? 
                ("HTTP error " + std::to_string(http_response.status_code) + ". Body: " + http_response.body) :
                http_response.error_message;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception in executeOrder: ") + e.what();
    }
    return result;
}

// Public API Methods
std::future<AlpacaClient::OrderResult> AlpacaClient::submitMarketOrderAsync(
    const std::string& symbol, int quantity, Side side, Priority priority, const std::string& client_order_id) {
    PriorityOrder order;
    order.symbol = symbol; order.quantity = quantity; order.side = side; order.type = Type::MARKET;
    order.priority = priority; order.client_order_id = client_order_id.empty() ? generateClientOrderId() : client_order_id;
    order.queue_time = std::chrono::steady_clock::now();
    return order_queue_->submitOrder(std::move(order));
}
std::future<AlpacaClient::OrderResult> AlpacaClient::submitLimitOrderAsync(
    const std::string& symbol, int quantity, Side side, double limit_price, 
    Priority priority, TimeInForce tif, const std::string& client_order_id) {
    PriorityOrder order;
    order.symbol = symbol; order.quantity = quantity; order.side = side; order.type = Type::LIMIT;
    order.limit_price = limit_price; order.tif = tif; order.priority = priority;
    order.client_order_id = client_order_id.empty() ? generateClientOrderId() : client_order_id;
    order.queue_time = std::chrono::steady_clock::now();
    return order_queue_->submitOrder(std::move(order));
}
AlpacaClient::OrderResult AlpacaClient::submitMarketOrder(
    const std::string& symbol, int quantity, Side side, const std::string& client_order_id) {
    if (!initialized_.load()) { OrderResult r; r.success = false; r.error_message = "Client not initialized"; return r; }
    auto future = submitMarketOrderAsync(symbol, quantity, side, Priority::NORMAL, client_order_id);
    return future.get();
}
AlpacaClient::OrderResult AlpacaClient::submitLimitOrder(
    const std::string& symbol, int quantity, Side side, double limit_price, 
    TimeInForce tif, const std::string& client_order_id) {
    if (!initialized_.load()) { OrderResult r; r.success = false; r.error_message = "Client not initialized"; return r; }
    auto future = submitLimitOrderAsync(symbol, quantity, side, limit_price, Priority::NORMAL, tif, client_order_id);
    return future.get();
}

std::future<AlpacaClient::OrderResult> AlpacaClient::submitBracketOrderAsync(
    const std::string& symbol, int quantity, Side side, double limit_price,
    double stop_loss_price, double take_profit_price, Priority priority, const std::string& client_order_id) {
    
    PriorityOrder order;
    order.symbol = symbol;
    order.quantity = quantity;
    order.side = side;
    order.type = Type::BRACKET;
    order.limit_price = limit_price;
    order.stop_loss_price = stop_loss_price;
    order.take_profit_price = take_profit_price;
    order.priority = priority;
    order.client_order_id = client_order_id.empty() ? generateClientOrderId() : client_order_id;
    order.queue_time = std::chrono::steady_clock::now();
    
    return order_queue_->submitOrder(std::move(order));
}

AlpacaClient::OrderResult AlpacaClient::submitBracketOrder(
    const std::string& symbol, int quantity, Side side, double limit_price,
    double stop_loss_price, double take_profit_price, const std::string& client_order_id) {
    
    if (!initialized_.load()) {
        OrderResult r;
        r.success = false;
        r.error_message = "Client not initialized";
        return r;
    }
    
    auto future = submitBracketOrderAsync(symbol, quantity, side, limit_price,
                                         stop_loss_price, take_profit_price,
                                         Priority::NORMAL, client_order_id);
    return future.get();
}

// Helper methods
std::string AlpacaClient::generateClientOrderId() {
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    std::stringstream ss;
    ss << "hftb_" << dist(gen_);
    std::string full_id = ss.str();
    return full_id.length() > 48 ? full_id.substr(0, 48) : full_id;
}
std::string AlpacaClient::sideToString(Side side) const { return side == Side::BUY ? "buy" : "sell"; }
std::string AlpacaClient::typeToString(Type type) const { return type == Type::MARKET ? "market" : "limit"; }
std::string AlpacaClient::tifToString(TimeInForce tif) const {
    switch (tif) {
        case TimeInForce::DAY: return "day"; case TimeInForce::GTC: return "gtc";
        case TimeInForce::IOC: return "ioc"; case TimeInForce::FOK: return "fok";
        default: return "day";
    }
}

// Performance Tracker Implementation
void AlpacaClient::PerformanceTracker::recordOrder(const OrderResult& result) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    total_orders_.fetch_add(1);
    if (result.success) successful_orders_.fetch_add(1); else failed_orders_.fetch_add(1);
    if (result.retry_count > 0) retried_orders_.fetch_add(1);
    latency_samples_.push_back(result.total_latency);
    if (latency_samples_.size() > 1000) latency_samples_.erase(latency_samples_.begin(), latency_samples_.begin() + (latency_samples_.size() - 1000));
}
AlpacaClient::PerformanceMetrics AlpacaClient::PerformanceTracker::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    PerformanceMetrics metrics;
    metrics.total_orders_submitted = total_orders_.load();
    metrics.successful_orders = successful_orders_.load();
    metrics.failed_orders = failed_orders_.load();
    metrics.retried_orders = retried_orders_.load();
    
    if (metrics.total_orders_submitted > 0) {
        metrics.success_rate = static_cast<double>(metrics.successful_orders) / metrics.total_orders_submitted;
    } else {
        metrics.success_rate = 0.0;
    }
    
    if (!latency_samples_.empty()) {
        auto sorted_samples = latency_samples_; // Make a copy for sorting
        std::sort(sorted_samples.begin(), sorted_samples.end());
        
        metrics.min_latency = sorted_samples.front();
        metrics.max_latency = sorted_samples.back();
        
        std::chrono::nanoseconds total_duration_ns(0);
        for(const auto& dur : sorted_samples) {
            total_duration_ns += dur;
        }
        metrics.avg_latency = total_duration_ns / sorted_samples.size();
        
        size_t s_size = sorted_samples.size();
        if (s_size > 0) {
            // Percentile calculation: index = p * (N-1)
            // For p95, index is 0.95 * (N-1)
            // For p99, index is 0.99 * (N-1)
            // Ensure index is clamped to [0, N-1]
            
            size_t p95_idx = static_cast<size_t>(0.95 * (s_size -1));
            metrics.p95_latency = sorted_samples[std::min(p95_idx, s_size - 1)];
            
            size_t p99_idx = static_cast<size_t>(0.99 * (s_size-1));
            metrics.p99_latency = sorted_samples[std::min(p99_idx, s_size - 1)];
        } else {
            metrics.p95_latency = std::chrono::nanoseconds(0);
            metrics.p99_latency = std::chrono::nanoseconds(0);
        }
    } else {
        metrics.min_latency = std::chrono::nanoseconds(0);
        metrics.max_latency = std::chrono::nanoseconds(0);
        metrics.avg_latency = std::chrono::nanoseconds(0);
        metrics.p95_latency = std::chrono::nanoseconds(0);
        metrics.p99_latency = std::chrono::nanoseconds(0);
    }
    return metrics;
}
void AlpacaClient::PerformanceTracker::reset() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    total_orders_.store(0); successful_orders_.store(0); failed_orders_.store(0);
    retried_orders_.store(0); latency_samples_.clear();
}

// Reconciliation Manager Implementation
AlpacaClient::ReconciliationManager::ReconciliationManager(hft::BeastHttpClient& http_client)
    : http_client_(http_client), last_reconciliation_(std::chrono::steady_clock::now()) {}
AlpacaClient::ReconciliationManager::~ReconciliationManager() { stop(); }
bool AlpacaClient::ReconciliationManager::start(std::function<void(const TradeUpdate&)> callback) {
    if (running_.load()) return true;
    trade_callback_ = callback; running_.store(true);
    reconciliation_thread_ = std::thread(&ReconciliationManager::reconciliationLoop, this);
    std::cout << "🚀 Reconciliation manager started (5-minute intervals)" << std::endl; return true;
}
void AlpacaClient::ReconciliationManager::stop() {
    if (!running_.load()) return; running_.store(false);
    if (reconciliation_thread_.joinable()) reconciliation_thread_.join();
    std::cout << "🛑 Reconciliation manager stopped" << std::endl;
}
void AlpacaClient::ReconciliationManager::reconciliationLoop() {
    while (running_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        try {
            auto trades = fetchTradeUpdates();
            for (const auto& trade : trades) {
                if (trade_callback_) trade_callback_(trade);
                trades_reconciled_.fetch_add(1);
            }
            last_reconciliation_ = std::chrono::steady_clock::now();
            auto reconciliation_time = std::chrono::duration_cast<std::chrono::milliseconds>(last_reconciliation_ - start_time);
            total_reconciliation_time_ms_.fetch_add(reconciliation_time.count());
            reconciliation_count_.fetch_add(1);
        } catch (const std::exception& e) {
            reconciliation_errors_.fetch_add(1);
            std::cerr << "❌ Reconciliation error: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::minutes(5));
    }
}
std::vector<AlpacaClient::TradeUpdate> AlpacaClient::ReconciliationManager::fetchTradeUpdates() {
    std::vector<TradeUpdate> trades;
    try {
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        auto now_sys = std::chrono::system_clock::now();
        auto yesterday_sys = now_sys - std::chrono::hours(24);
        std::time_t yesterday_time_t = std::chrono::system_clock::to_time_t(yesterday_sys);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&yesterday_time_t), "%Y-%m-%dT%H:%M:%SZ");
        std::string after_timestamp_str = ss.str();
        std::string query_params = "/v2/orders?status=filled&limit=500&direction=desc&after=" + after_timestamp_str;
        
        http_client_.async_get(query_params, [&promise](hft::HttpResponse response) {
            promise.set_value(std::move(response));
        });
        auto http_response = future.get();
        if (http_response.success && http_response.status_code == 200) {
            rapidjson::Document doc;
            doc.Parse(http_response.body.c_str());
            if (!doc.HasParseError() && doc.IsArray()) {
                for (rapidjson::SizeType i = 0; i < doc.Size(); ++i) {
                    const auto& order_json = doc[i];
                    TradeUpdate trade;
                    if (order_json.HasMember("id") && order_json["id"].IsString()) trade.order_id = order_json["id"].GetString();
                    if (order_json.HasMember("symbol") && order_json["symbol"].IsString()) trade.symbol = order_json["symbol"].GetString();
                    if (order_json.HasMember("side") && order_json["side"].IsString()) trade.side = order_json["side"].GetString();
                    if (order_json.HasMember("filled_qty") && order_json["filled_qty"].IsString()) trade.quantity = static_cast<int>(std::stod(order_json["filled_qty"].GetString()));
                    if (order_json.HasMember("filled_avg_price") && order_json["filled_avg_price"].IsString()) trade.price = std::stod(order_json["filled_avg_price"].GetString());
                    if (order_json.HasMember("status") && order_json["status"].IsString()) trade.status = order_json["status"].GetString();
                    trade.timestamp = std::chrono::steady_clock::now();
                    trades.push_back(trade);
                }
            } else { std::cerr << "❌ Failed to parse trade updates JSON. Body: " << http_response.body << std::endl; }
        } else { std::cerr << "❌ Failed to fetch trade updates: HTTP " << http_response.status_code << ", Error: " << http_response.error_message << ", Body: " << http_response.body << std::endl; }
    } catch (const std::exception& e) { std::cerr << "❌ Exception in fetchTradeUpdates: " << e.what() << std::endl; }
    return trades;
}
AlpacaClient::ReconciliationStats AlpacaClient::ReconciliationManager::getStats() const {
    ReconciliationStats stats;
    stats.trades_reconciled = trades_reconciled_.load();
    stats.trades_pending = trades_pending_.load();
    stats.reconciliation_errors = reconciliation_errors_.load();
    stats.last_reconciliation = last_reconciliation_;
    auto total_time_ms = total_reconciliation_time_ms_.load();
    auto count = reconciliation_count_.load();
    if (count > 0) stats.avg_reconciliation_time = std::chrono::milliseconds(total_time_ms / count);
    return stats;
}

// Account Polling Manager Implementation
AlpacaClient::AccountPollingManager::AccountPollingManager(hft::BeastHttpClient& http_client) : http_client_(http_client) {}
AlpacaClient::AccountPollingManager::~AccountPollingManager() { stop(); }
bool AlpacaClient::AccountPollingManager::start(std::function<void(const AccountUpdate&)> callback, int interval_ms) {
    if (running_.load()) return true;
    account_callback_ = callback; poll_interval_ms_.store(interval_ms); running_.store(true);
    polling_thread_ = std::thread(&AccountPollingManager::pollingLoop, this);
    std::cout << "🚀 Account polling started (" << interval_ms << "ms intervals)" << std::endl; return true;
}
void AlpacaClient::AccountPollingManager::stop() {
    if (!running_.load()) return; running_.store(false);
    if (polling_thread_.joinable()) polling_thread_.join();
    std::cout << "🛑 Account polling stopped" << std::endl;
}
void AlpacaClient::AccountPollingManager::pollingLoop() {
    while (running_.load()) {
        try {
            std::cout << "🔄 Starting account polling fetch..." << std::endl;
            auto account_update = fetchAccountUpdate();
            std::cout << "✅ Account polling fetch completed - equity: $" << account_update.equity
                      << " cash: $" << account_update.cash
                      << " buying_power: $" << account_update.buying_power << std::endl;
            if (account_callback_) {
                std::cout << "🔄 Calling account update callback..." << std::endl;
                account_callback_(account_update);
                std::cout << "✅ Account update callback completed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ Account polling error: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms_.load()));
    }
}
AlpacaClient::AccountUpdate AlpacaClient::AccountPollingManager::fetchAccountUpdate() {
    AccountUpdate update; update.timestamp = std::chrono::steady_clock::now();
    
    std::cout << "🔍 DEBUG: Starting fetchAccountUpdate..." << std::endl;
    
    try {
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        
        std::cout << "🔍 DEBUG: Making async_get request to /v2/account..." << std::endl;
        
        http_client_.async_get("/v2/account", [&promise](hft::HttpResponse response) {
            std::cout << "🔍 DEBUG: HTTP response received - Status: " << response.status_code 
                      << ", Success: " << (response.success ? "true" : "false") << std::endl;
            promise.set_value(std::move(response));
        });
        
        auto http_response = future.get();
        
        std::cout << "🔍 DEBUG: HTTP Response Details:" << std::endl;
        std::cout << "   Status Code: " << http_response.status_code << std::endl;
        std::cout << "   Success: " << (http_response.success ? "true" : "false") << std::endl;
        std::cout << "   Error Message: " << http_response.error_message << std::endl;
        std::cout << "   Body Length: " << http_response.body.length() << std::endl;
        std::cout << "   Body (first 500 chars): " << http_response.body.substr(0, 500) << std::endl;
        
        if (http_response.success && http_response.status_code == 200) {
            std::cout << "🔍 DEBUG: Parsing JSON response..." << std::endl;
            rapidjson::Document doc;
            doc.Parse(http_response.body.c_str());
            
            if (!doc.HasParseError()) {
                std::cout << "🔍 DEBUG: JSON parsed successfully" << std::endl;
                
                // Debug: Print all available fields
                std::cout << "🔍 DEBUG: Available JSON fields:" << std::endl;
                for (auto& member : doc.GetObject()) {
                    std::cout << "   Field: " << member.name.GetString() << std::endl;
                }
                
                if (doc.HasMember("buying_power") && doc["buying_power"].IsString()) {
                    update.buying_power = std::stod(doc["buying_power"].GetString());
                    std::cout << "🔍 DEBUG: Parsed buying_power: $" << update.buying_power << std::endl;
                } else {
                    std::cout << "🔍 DEBUG: buying_power field missing or not string" << std::endl;
                }
                
                if (doc.HasMember("portfolio_value") && doc["portfolio_value"].IsString()) {
                    update.equity = std::stod(doc["portfolio_value"].GetString());
                    std::cout << "🔍 DEBUG: Parsed portfolio_value: $" << update.equity << std::endl;
                } else {
                    std::cout << "🔍 DEBUG: portfolio_value field missing or not string" << std::endl;
                }
                
                if (doc.HasMember("cash") && doc["cash"].IsString()) {
                    update.cash = std::stod(doc["cash"].GetString());
                    std::cout << "🔍 DEBUG: Parsed cash: $" << update.cash << std::endl;
                } else {
                    std::cout << "🔍 DEBUG: cash field missing or not string" << std::endl;
                }
                
                std::cout << "🔍 DEBUG: Final AccountUpdate values:" << std::endl;
                std::cout << "   Equity: $" << update.equity << std::endl;
                std::cout << "   Cash: $" << update.cash << std::endl;
                std::cout << "   Buying Power: $" << update.buying_power << std::endl;
                
            } else { 
                std::cerr << "❌ Failed to parse account update JSON. Parse error: " << doc.GetParseError() << std::endl;
                std::cerr << "   Body: " << http_response.body << std::endl; 
            }
        } else { 
            std::cerr << "❌ Failed to fetch account update: HTTP " << http_response.status_code 
                      << ", Error: " << http_response.error_message 
                      << ", Body: " << http_response.body << std::endl; 
        }
    } catch (const std::exception& e) { 
        std::cerr << "❌ Exception in fetchAccountUpdate: " << e.what() << std::endl; 
    }
    
    std::cout << "🔍 DEBUG: fetchAccountUpdate completed" << std::endl;
    return update;
}

// Additional API methods
bool AlpacaClient::isMarketOpen() {
    auto now = std::chrono::steady_clock::now();
    if (initialized_.load() && http_client_ && (now - last_market_check_ < std::chrono::minutes(1))) {
        return cached_market_open_;
    }
    if (!http_client_) {
        std::cerr << "❌ HTTP client not initialized in AlpacaClient::isMarketOpen" << std::endl;
        return cached_market_open_; 
    }
    try {
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        http_client_->async_get("/v2/clock", [&promise](hft::HttpResponse response) {
            promise.set_value(std::move(response));
        });
        auto http_response = future.get(); 
        if (http_response.success && http_response.status_code == 200) {
            rapidjson::Document doc;
            doc.Parse(http_response.body.c_str());
            if (!doc.HasParseError() && doc.HasMember("is_open") && doc["is_open"].IsBool()) {
                cached_market_open_ = doc["is_open"].GetBool();
                last_market_check_ = std::chrono::steady_clock::now();
                return cached_market_open_;
            } else { std::cerr << "❌ Failed to parse market clock JSON. Body: " << http_response.body << std::endl; }
        } else { std::cerr << "❌ Failed to fetch market clock: HTTP " << http_response.status_code << ", Error: " << http_response.error_message << ", Body: " << http_response.body << std::endl; }
    } catch (const std::exception& e) { std::cerr << "❌ Exception in isMarketOpen: " << e.what() << std::endl; }
    return cached_market_open_; 
}

double AlpacaClient::getBuyingPower() {
    auto now = std::chrono::steady_clock::now();
    if (initialized_.load() && http_client_ && (now - last_buying_power_check_ < std::chrono::seconds(5))) {
        return cached_buying_power_;
    }
    if (!http_client_) {
        std::cerr << "❌ HTTP client not initialized in AlpacaClient::getBuyingPower" << std::endl;
        return cached_buying_power_;
    }
    try {
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        http_client_->async_get("/v2/account", [&promise](hft::HttpResponse response) {
            promise.set_value(std::move(response));
        });
        auto http_response = future.get();
        if (http_response.success && http_response.status_code == 200) {
            rapidjson::Document doc;
            doc.Parse(http_response.body.c_str());
            if (!doc.HasParseError() && doc.HasMember("buying_power") && doc["buying_power"].IsString()) {
                cached_buying_power_ = std::stod(doc["buying_power"].GetString());
                last_buying_power_check_ = std::chrono::steady_clock::now();
                // current_account_ = account_polling_manager_->fetchAccountUpdate(); // This might be redundant
                return cached_buying_power_;
            } else { std::cerr << "❌ Failed to parse buying power from account JSON. Body: " << http_response.body << std::endl; }
        } else { std::cerr << "❌ Failed to fetch account for buying power: HTTP " << http_response.status_code << ", Error: " << http_response.error_message << ", Body: " << http_response.body << std::endl; }
    } catch (const std::exception& e) { std::cerr << "❌ Exception in getBuyingPower: " << e.what() << std::endl; }
    return cached_buying_power_;
}

AlpacaClient::AccountUpdate AlpacaClient::getAccountInfo() {
    if (account_polling_manager_ && initialized_.load()) {
        auto now = std::chrono::steady_clock::now();
        if (current_account_.timestamp.time_since_epoch().count() != 0 && 
            (now - current_account_.timestamp < std::chrono::seconds(1))) {
            return current_account_;
        }
    }
    if (account_polling_manager_ && initialized_.load() && http_client_) {
         AccountUpdate fresh_update = account_polling_manager_->fetchAccountUpdate();
         current_account_ = fresh_update; 
         return fresh_update;
    }
    std::cerr << "⚠️ getAccountInfo called before polling manager fully active or client not initialized." << std::endl;
    return current_account_; 
}

std::vector<AlpacaClient::Position> AlpacaClient::getPositions() {
    std::vector<Position> positions;
    
    if (!initialized_.load() || !http_client_) {
        std::cerr << "❌ Client not initialized in getPositions" << std::endl;
        return positions;
    }
    
    try {
        std::promise<hft::HttpResponse> promise;
        auto future = promise.get_future();
        
        http_client_->async_get("/v2/positions", [&promise](hft::HttpResponse response) {
            promise.set_value(std::move(response));
        });
        
        auto http_response = future.get();
        
        if (http_response.success && http_response.status_code == 200) {
            rapidjson::Document doc;
            doc.Parse(http_response.body.c_str());
            
            if (!doc.HasParseError() && doc.IsArray()) {
                for (rapidjson::SizeType i = 0; i < doc.Size(); ++i) {
                    const auto& pos_json = doc[i];
                    Position pos;
                    
                    if (pos_json.HasMember("symbol") && pos_json["symbol"].IsString()) {
                        pos.symbol = pos_json["symbol"].GetString();
                    }
                    if (pos_json.HasMember("asset_id") && pos_json["asset_id"].IsString()) {
                        pos.asset_id = pos_json["asset_id"].GetString();
                    }
                    if (pos_json.HasMember("qty") && pos_json["qty"].IsString()) {
                        pos.quantity = std::stoi(pos_json["qty"].GetString());
                    }
                    if (pos_json.HasMember("avg_entry_price") && pos_json["avg_entry_price"].IsString()) {
                        pos.avg_entry_price = std::stod(pos_json["avg_entry_price"].GetString());
                    }
                    if (pos_json.HasMember("market_value") && pos_json["market_value"].IsString()) {
                        pos.market_value = std::stod(pos_json["market_value"].GetString());
                    }
                    if (pos_json.HasMember("cost_basis") && pos_json["cost_basis"].IsString()) {
                        pos.cost_basis = std::stod(pos_json["cost_basis"].GetString());
                    }
                    if (pos_json.HasMember("unrealized_pl") && pos_json["unrealized_pl"].IsString()) {
                        pos.unrealized_pl = std::stod(pos_json["unrealized_pl"].GetString());
                    }
                    if (pos_json.HasMember("unrealized_plpc") && pos_json["unrealized_plpc"].IsString()) {
                        pos.unrealized_plpc = std::stod(pos_json["unrealized_plpc"].GetString());
                    }
                    if (pos_json.HasMember("current_price") && pos_json["current_price"].IsString()) {
                        pos.current_price = std::stod(pos_json["current_price"].GetString());
                    }
                    if (pos_json.HasMember("side") && pos_json["side"].IsString()) {
                        pos.side = pos_json["side"].GetString();
                    }
                    
                    pos.timestamp = std::chrono::steady_clock::now();
                    positions.push_back(pos);
                }
                
                std::cout << "✅ Fetched " << positions.size() << " positions from Alpaca" << std::endl;
            } else {
                std::cerr << "❌ Failed to parse positions JSON or not an array" << std::endl;
            }
        } else {
            std::cerr << "❌ Failed to fetch positions: HTTP " << http_response.status_code 
                      << ", Error: " << http_response.error_message << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in getPositions: " << e.what() << std::endl;
    }
    
    return positions;
}

int AlpacaClient::getPositionCount() {
    auto positions = getPositions();
    return static_cast<int>(positions.size());
}

bool AlpacaClient::startAccountPolling(std::function<void(const AccountUpdate&)> on_account_update, int poll_interval_ms) {
    if (!account_polling_manager_ || !initialized_.load()) return false;
    
    // Enforce minimum 30-second polling to respect 200 req/min rate limit
    int safe_interval = std::max(poll_interval_ms, 30000);
    if (safe_interval > poll_interval_ms) {
        std::cout << "⚠️ Increased polling interval from " << poll_interval_ms << "ms to " << safe_interval << "ms for rate limit compliance" << std::endl;
    }
    
    return account_polling_manager_->start(
        [this, user_callback = std::move(on_account_update)](const AccountUpdate& update) {
            this->current_account_ = update;
            if (user_callback) user_callback(update);
        },
        safe_interval
    );
}
void AlpacaClient::stopAccountPolling() { if (account_polling_manager_) account_polling_manager_->stop(); }
bool AlpacaClient::startReconciliation(std::function<void(const TradeUpdate&)> on_trade_update) {
     if (!reconciliation_manager_ || !initialized_.load()) return false;
    return reconciliation_manager_->start(on_trade_update);
}
void AlpacaClient::stopReconciliation() { if (reconciliation_manager_) reconciliation_manager_->stop(); }
AlpacaClient::PerformanceMetrics AlpacaClient::getPerformanceMetrics() const {
    if (!performance_tracker_) return {};
    return performance_tracker_->getMetrics();
}
bool AlpacaClient::isReadyForOrders() const {
    return initialized_.load() && running_.load() && http_client_ && order_queue_;
}
std::chrono::milliseconds AlpacaClient::getTimeUntilNextOrder() const {
    if (http_client_) {
        // This would require BeastHttpClient to expose RateLimiter stats.
        // For now, returning 0.
    }
    return std::chrono::milliseconds(0);
}
