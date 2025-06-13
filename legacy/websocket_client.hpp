#pragma once

#include "memory_manager.hpp"
#include "market_data.hpp"
#include "config.hpp"
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <rapidjson/document.h>

namespace hft {

// Market data message types
enum class MessageType : uint8_t {
    TRADE = 1,
    QUOTE = 2,
    AGGREGATE = 3
};

// Parsed market data for HFT engine
struct ParsedMessage {
    MessageType type;
    int32_t symbol_id;
    uint64_t timestamp_ns;
    float price;
    uint32_t volume;
    float bid, ask;
    uint16_t bid_size, ask_size;
};

// Performance statistics
struct WebSocketPerformanceStats {
    uint64_t messages_received;
    uint64_t messages_processed;
    uint64_t connection_attempts;
    uint64_t successful_connections;
    bool is_connected;
    bool is_authenticated;
    double avg_processing_time_us;
    std::chrono::steady_clock::time_point last_message_time;
};

// Polygon WebSocket client for real-time market data
class PolygonWebSocketClient {
public:
    using client = websocketpp::client<websocketpp::config::asio_tls_client>;
    using message_ptr = websocketpp::config::asio_tls_client::message_type::ptr;
    using context_ptr = websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>;
    
    explicit PolygonWebSocketClient(const std::string& api_key);
    ~PolygonWebSocketClient();
    
    // Connection management
    bool connect();
    bool authenticate();
    bool subscribe_to_symbols(const std::vector<std::string>& symbols);
    void disconnect();
    
    // Status
    bool is_connected() const { return connected_.load(); }
    bool is_authenticated() const { return authenticated_.load(); }
    
    // Callbacks
    void set_message_callback(std::function<void(const std::string&, const std::string&, 
                                                double, uint64_t, double, double)> callback);
    
    // Performance
    WebSocketPerformanceStats get_performance_stats() const;

private:
    // WebSocket event handlers
    context_ptr on_tls_init(websocketpp::connection_hdl hdl);
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_fail(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, message_ptr msg);
    
    // Message processing
    void process_message(const std::string& message);
    bool send_auth_message();
    bool send_subscription_message(const std::vector<std::string>& symbols);
    
    // Connection state
    client ws_client_;
    websocketpp::connection_hdl connection_hdl_;
    std::thread ws_thread_;
    
    // Configuration
    std::string api_key_;
    std::string ws_url_;
    
    // State
    std::atomic<bool> connected_{false};
    std::atomic<bool> authenticated_{false};
    std::atomic<bool> running_{false};
    
    // Callback
    std::function<void(const std::string&, const std::string&, double, uint64_t, double, double)> message_callback_;
    
    // Performance tracking
    mutable std::mutex stats_mutex_;
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_processed_{0};
    std::atomic<uint64_t> connection_attempts_{0};
    std::atomic<uint64_t> successful_connections_{0};
    std::chrono::steady_clock::time_point last_message_time_;
    
    // Heartbeat mechanism
    std::thread heartbeat_thread_;
    std::atomic<bool> heartbeat_running_{false};
    void start_heartbeat();
    void stop_heartbeat();
    bool send_ping();
};

// Main HFT WebSocket client
class UltraFastWebSocketClient {
public:
    explicit UltraFastWebSocketClient(MarketDataManager& market_data,
                                    UltraFastMemoryManager& memory_manager,
                                    const APIConfig& config = APIConfig{});
    ~UltraFastWebSocketClient();
    
    // Lifecycle
    bool start();
    void stop();
    
    // Callbacks
    void set_market_data_callback(std::function<void(const ParsedMessage&)> callback);
    
    // Performance
    WebSocketPerformanceStats get_performance_stats() const;

private:
    // Symbol management
    void initialize_all_symbols();
    void setup_market_data_callback();
    
    // Message conversion
    ParsedMessage convert_to_parsed_message(const std::string& symbol, const std::string& type,
                                          double price, uint64_t timestamp, double bid, double ask);
    
    // Components
    std::unique_ptr<PolygonWebSocketClient> polygon_client_;
    MarketDataManager& market_data_;
    UltraFastMemoryManager& memory_manager_;
    
    // Symbol mapping
    std::unordered_map<std::string, int32_t> symbol_map_;
    std::vector<std::string> all_symbols_;
    
    // State
    std::atomic<bool> running_{false};
    std::function<void(const ParsedMessage&)> market_data_callback_;
    
    // Performance tracking
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_processed_{0};
};

} // namespace hft
