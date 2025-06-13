#include "websocket_client.hpp"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <openssl/ssl.h>
#include <iostream>
#include <sstream>

namespace hft {

// PolygonWebSocketClient Implementation
PolygonWebSocketClient::PolygonWebSocketClient(const std::string& api_key) 
    : api_key_(api_key), ws_url_("wss://socket.polygon.io/stocks") {
    
    std::cout << "🔧 Initializing Polygon WebSocket client with API key: " 
              << api_key.substr(0, 8) << "..." << std::endl;
    
    // Configure WebSocket client
    ws_client_.init_asio();
    ws_client_.set_access_channels(websocketpp::log::alevel::none);
    ws_client_.set_error_channels(websocketpp::log::elevel::none);
    
    // Set TLS init handler
    ws_client_.set_tls_init_handler([this](websocketpp::connection_hdl hdl) {
        return this->on_tls_init(hdl);
    });
    
    // Set event handlers
    ws_client_.set_open_handler([this](websocketpp::connection_hdl hdl) {
        this->on_open(hdl);
    });
    
    ws_client_.set_close_handler([this](websocketpp::connection_hdl hdl) {
        this->on_close(hdl);
    });
    
    ws_client_.set_fail_handler([this](websocketpp::connection_hdl hdl) {
        this->on_fail(hdl);
    });
    
    ws_client_.set_message_handler([this](websocketpp::connection_hdl hdl, message_ptr msg) {
        this->on_message(hdl, msg);
    });
    
    last_message_time_ = std::chrono::steady_clock::now();
}

PolygonWebSocketClient::~PolygonWebSocketClient() {
    disconnect();
}

PolygonWebSocketClient::context_ptr PolygonWebSocketClient::on_tls_init(websocketpp::connection_hdl hdl) {
    context_ptr ctx = websocketpp::lib::make_shared<websocketpp::lib::asio::ssl::context>(
        websocketpp::lib::asio::ssl::context::tlsv12);
    
    try {
        ctx->set_options(websocketpp::lib::asio::ssl::context::default_workarounds |
                        websocketpp::lib::asio::ssl::context::no_sslv2 |
                        websocketpp::lib::asio::ssl::context::no_sslv3 |
                        websocketpp::lib::asio::ssl::context::single_dh_use);
        
        // Minimal SSL verification for speed
        ctx->set_verify_mode(websocketpp::lib::asio::ssl::verify_none);
        
    } catch (std::exception& e) {
        std::cerr << "❌ TLS context error: " << e.what() << std::endl;
    }
    
    return ctx;
}

bool PolygonWebSocketClient::connect() {
    try {
        // CORRECT: Connect to URL without API key (auth comes after)
        std::cout << "🔌 Connecting to Polygon WebSocket: " << ws_url_ << std::endl;
        
        connection_attempts_.fetch_add(1);
        
        websocketpp::lib::error_code ec;
        client::connection_ptr con = ws_client_.get_connection(ws_url_, ec);
        
        if (ec) {
            std::cerr << "❌ Connection error: " << ec.message() << std::endl;
            return false;
        }
        
        connection_hdl_ = con->get_handle();
        ws_client_.connect(con);
        
        // Start WebSocket client in separate thread
        running_.store(true);
        ws_thread_ = std::thread([this]() {
            try {
                std::cout << "🔧 Starting WebSocket client thread..." << std::endl;
                ws_client_.run();
                std::cout << "🔧 WebSocket client thread ended" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ WebSocket thread error: " << e.what() << std::endl;
            }
        });
        
        // Wait for connection with timeout
        auto start = std::chrono::steady_clock::now();
        while (!connected_.load() && running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                std::cerr << "❌ Connection timeout" << std::endl;
                return false;
            }
        }
        
        if (connected_.load()) {
            successful_connections_.fetch_add(1);
            
            // Start heartbeat to keep connection alive
            start_heartbeat();
            
            return true;
        }
        
        return false;
        
    } catch (std::exception& e) {
        std::cerr << "❌ Connect exception: " << e.what() << std::endl;
        return false;
    }
}

bool PolygonWebSocketClient::authenticate() {
    if (!connected_.load()) {
        std::cerr << "❌ Not connected, cannot authenticate" << std::endl;
        return false;
    }
    
    // CORRECT: Send auth message after connection per official docs
    std::cout << "🔐 Sending authentication message..." << std::endl;
    
    if (!send_auth_message()) {
        std::cerr << "❌ Failed to send authentication message" << std::endl;
        return false;
    }
    
    // Wait for auth_success response
    auto start = std::chrono::steady_clock::now();
    while (!authenticated_.load() && connected_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
            std::cerr << "❌ Authentication timeout after 10 seconds" << std::endl;
            return false;
        }
    }
    
    if (authenticated_.load()) {
        std::cout << "✅ Authentication successful!" << std::endl;
        return true;
    } else {
        std::cerr << "❌ Authentication failed" << std::endl;
        return false;
    }
}

bool PolygonWebSocketClient::subscribe_to_symbols(const std::vector<std::string>& symbols) {
    if (!authenticated_.load()) {
        std::cerr << "❌ Not authenticated, cannot subscribe" << std::endl;
        return false;
    }
    
    std::cout << "📡 Subscribing to " << symbols.size() << " symbols..." << std::endl;
    
    return send_subscription_message(symbols);
}

void PolygonWebSocketClient::disconnect() {
    if (running_.load()) {
        std::cout << "🔌 Disconnecting Polygon WebSocket..." << std::endl;
        
        // Set flags to stop processing
        running_.store(false);
        connected_.store(false);
        authenticated_.store(false);
        
        // Stop heartbeat first
        stop_heartbeat();
        
        try {
            // Send close frame if connection is still active
            if (connection_hdl_.lock()) {
                std::cout << "🔧 Sending WebSocket close frame..." << std::endl;
                websocketpp::lib::error_code ec;
                ws_client_.close(connection_hdl_, websocketpp::close::status::normal, "Clean shutdown", ec);
                if (ec) {
                    std::cout << "⚠️ WebSocket close error: " << ec.message() << std::endl;
                } else {
                    std::cout << "✅ WebSocket close frame sent" << std::endl;
                }
                
                // Give time for close handshake
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Stop the WebSocket client event loop
            std::cout << "🔧 Stopping WebSocket client..." << std::endl;
            ws_client_.stop();
            
        } catch (const std::exception& e) {
            std::cout << "⚠️ Error during WebSocket shutdown: " << e.what() << std::endl;
        } catch (...) {
            std::cout << "⚠️ Unknown error during WebSocket shutdown" << std::endl;
        }
        
        // Wait for background thread to finish gracefully
        if (ws_thread_.joinable()) {
            std::cout << "🔧 Waiting for WebSocket thread to finish..." << std::endl;
            ws_thread_.join();
            std::cout << "✅ WebSocket thread finished" << std::endl;
        }
        
        std::cout << "✅ Polygon WebSocket disconnected cleanly" << std::endl;
    } else {
        std::cout << "ℹ️ Polygon WebSocket was not running" << std::endl;
    }
}

void PolygonWebSocketClient::set_message_callback(std::function<void(const std::string&, const std::string&, 
                                                                   double, uint64_t, double, double)> callback) {
    message_callback_ = callback;
}

void PolygonWebSocketClient::on_open(websocketpp::connection_hdl hdl) {
    std::cout << "✅ Connected to Polygon WebSocket" << std::endl;
    connected_.store(true);
}

void PolygonWebSocketClient::on_close(websocketpp::connection_hdl hdl) {
    std::cout << "❌ Polygon WebSocket connection closed" << std::endl;
    connected_.store(false);
    authenticated_.store(false);
}

void PolygonWebSocketClient::on_fail(websocketpp::connection_hdl hdl) {
    std::cout << "❌ Polygon WebSocket connection failed" << std::endl;
    connected_.store(false);
    authenticated_.store(false);
}

void PolygonWebSocketClient::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
    try {
        messages_received_.fetch_add(1);
        last_message_time_ = std::chrono::steady_clock::now();
        
        std::string payload = msg->get_payload();
        process_message(payload);
        
        messages_processed_.fetch_add(1);
        
    } catch (std::exception& e) {
        std::cerr << "❌ Message processing error: " << e.what() << std::endl;
    }
}

void PolygonWebSocketClient::process_message(const std::string& message) {
    try {
        rapidjson::Document doc;
        doc.Parse(message.c_str());
        
        if (doc.HasParseError()) {
            std::cerr << "❌ JSON parse error in message: " << message.substr(0, 200) << std::endl;
            return;
        }
        
        // Process ALL messages - both single objects and arrays
        std::vector<rapidjson::Value*> items_to_process;
        
        if (doc.IsArray()) {
            for (auto& item : doc.GetArray()) {
                if (item.IsObject()) {
                    items_to_process.push_back(&item);
                }
            }
        } else if (doc.IsObject()) {
            items_to_process.push_back(&doc);
        } else {
            return;
        }
        
        // Process each item
        for (auto* item : items_to_process) {
            // Handle status messages (connection, subscription confirmations)
            if (item->HasMember("ev") && item->HasMember("status")) {
                std::string ev = (*item)["ev"].GetString();
                std::string status = (*item)["status"].GetString();
                
                if (ev == "status" && status == "connected") {
                    std::cout << "✅ Polygon connection confirmed!" << std::endl;
                    continue;
                } else if (ev == "status" && status == "auth_success") {
                    std::cout << "✅ Polygon authentication successful!" << std::endl;
                    authenticated_.store(true);
                    continue;
                } else if (ev == "status" && status == "success") {
                    std::cout << "✅ Polygon subscription successful!" << std::endl;
                    continue;
                }
            }
            
            // Handle legacy status format (single object with "status" field)
            if (item->HasMember("status") && !item->HasMember("ev")) {
                std::string status = (*item)["status"].GetString();
                
                if (status == "connected") {
                    std::cout << "✅ Polygon connection confirmed!" << std::endl;
                    continue;
                } else if (status == "auth_success") {
                    std::cout << "✅ Polygon authentication successful!" << std::endl;
                    authenticated_.store(true);
                    continue;
                } else if (status == "success") {
                    std::cout << "✅ Polygon subscription successful!" << std::endl;
                    continue;
                }
            }
            
            // Handle market data messages silently
            if (item->HasMember("ev")) {
                std::string event_type = (*item)["ev"].GetString();
                
                if (item->HasMember("sym")) {
                    std::string symbol = (*item)["sym"].GetString();
                    
                    double price = 0.0;
                    uint64_t timestamp = 0;
                    double bid = 0.0, ask = 0.0;
                    
                    // Extract data based on event type
                    if (event_type == "T") {  // Trade
                        if (item->HasMember("p")) price = (*item)["p"].GetDouble();
                        if (item->HasMember("t")) timestamp = (*item)["t"].GetUint64();
                    } else if (event_type == "Q") {  // Quote
                        if (item->HasMember("bp")) bid = (*item)["bp"].GetDouble();
                        if (item->HasMember("ap")) ask = (*item)["ap"].GetDouble();
                        if (item->HasMember("t")) timestamp = (*item)["t"].GetUint64();
                        price = (bid > 0 && ask > 0) ? (bid + ask) / 2.0 : 0.0;
                    }
                    
                    // Forward valid data to callback
                    if ((price > 0 || (bid > 0 && ask > 0)) && timestamp > 0) {
                        if (message_callback_) {
                            message_callback_(symbol, event_type, price, timestamp, bid, ask);
                        }
                    }
                }
            }
        }
        
    } catch (std::exception& e) {
        std::cerr << "❌ Message processing exception: " << e.what() << std::endl;
    }
}

bool PolygonWebSocketClient::send_auth_message() {
    try {
        // CORRECT: Official Polygon.io auth format per documentation
        std::string auth_message = "{\"action\":\"auth\",\"params\":\"" + api_key_ + "\"}";
        
        std::cout << "🔐 Sending auth message: " << auth_message << std::endl;
        
        websocketpp::lib::error_code ec;
        ws_client_.send(connection_hdl_, auth_message, websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "❌ Send auth error: " << ec.message() << std::endl;
            return false;
        }
        
        std::cout << "✅ Auth message sent successfully" << std::endl;
        return true;
        
    } catch (std::exception& e) {
        std::cerr << "❌ Auth message exception: " << e.what() << std::endl;
        return false;
    }
}

bool PolygonWebSocketClient::send_subscription_message(const std::vector<std::string>& symbols) {
    try {
        // CRITICAL FIX: Polygon.io expects params as comma-separated string, not array
        std::cout << "🔍 DEBUG: Subscribing to " << symbols.size() << " symbols" << std::endl;
        
        // Build comma-separated string for both trades and quotes
        std::string params_str = "";
        bool first = true;
        
        for (const auto& symbol : symbols) {
            if (!first) params_str += ",";
            params_str += "T." + symbol;  // Trade channel
            params_str += ",Q." + symbol; // Quote channel
            first = false;
        }
        
        // Build JSON message with string params (not array)
        std::string message = "{\"action\":\"subscribe\",\"params\":\"" + params_str + "\"}";
        
        std::cout << "📡 Subscription message: " << message.substr(0, 200) << "..." << std::endl;
        std::cout << "🔍 DEBUG: Message length: " << message.length() << " characters" << std::endl;
        std::cout << "🔍 DEBUG: Total channels: " << (symbols.size() * 2) << " (trades + quotes)" << std::endl;
        
        websocketpp::lib::error_code ec;
        ws_client_.send(connection_hdl_, message, websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "❌ Send subscription error: " << ec.message() << std::endl;
            return false;
        }
        
        std::cout << "✅ Subscription message sent successfully" << std::endl;
        std::cout << "🔍 DEBUG: Waiting for subscription response and market data..." << std::endl;
        
        return true;
        
    } catch (std::exception& e) {
        std::cerr << "❌ Subscription message exception: " << e.what() << std::endl;
        return false;
    }
}

// Heartbeat mechanism implementation
void PolygonWebSocketClient::start_heartbeat() {
    heartbeat_running_.store(true);
    heartbeat_thread_ = std::thread([this]() {
        std::cout << "🔄 Starting WebSocket heartbeat (ping every 30s)" << std::endl;
        
        while (heartbeat_running_.load() && connected_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            if (!heartbeat_running_.load() || !connected_.load()) {
                break;
            }
            
            // Check if we've received messages recently
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_message_time_).count();
            
            if (time_since_last > 60) {  // No messages for 60 seconds
                std::cout << "⚠️ No messages received for " << time_since_last << "s, sending ping" << std::endl;
                if (!send_ping()) {
                    std::cout << "❌ Ping failed, connection may be dead" << std::endl;
                }
            } else {
                std::cout << "💓 Heartbeat: Connection alive (last message " << time_since_last << "s ago)" << std::endl;
            }
        }
        
        std::cout << "🔄 WebSocket heartbeat stopped" << std::endl;
    });
}

void PolygonWebSocketClient::stop_heartbeat() {
    if (heartbeat_running_.load()) {
        heartbeat_running_.store(false);
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
    }
}

bool PolygonWebSocketClient::send_ping() {
    try {
        if (!connected_.load()) return false;
        
        // Send a ping frame
        websocketpp::lib::error_code ec;
        ws_client_.ping(connection_hdl_, "ping", ec);
        
        if (ec) {
            std::cerr << "❌ Ping error: " << ec.message() << std::endl;
            return false;
        }
        
        std::cout << "📡 Ping sent successfully" << std::endl;
        return true;
        
    } catch (std::exception& e) {
        std::cerr << "❌ Ping exception: " << e.what() << std::endl;
        return false;
    }
}

WebSocketPerformanceStats PolygonWebSocketClient::get_performance_stats() const {
    WebSocketPerformanceStats stats;
    stats.messages_received = messages_received_.load();
    stats.messages_processed = messages_processed_.load();
    stats.connection_attempts = connection_attempts_.load();
    stats.successful_connections = successful_connections_.load();
    stats.is_connected = connected_.load();
    stats.is_authenticated = authenticated_.load();
    stats.avg_processing_time_us = 1.0; // Placeholder
    stats.last_message_time = last_message_time_;
    return stats;
}

// UltraFastWebSocketClient Implementation
UltraFastWebSocketClient::UltraFastWebSocketClient(MarketDataManager& market_data,
                                                 UltraFastMemoryManager& memory_manager,
                                                 const APIConfig& config)
    : market_data_(market_data), memory_manager_(memory_manager) {
    
    std::cout << "🚀 Initializing UltraFastWebSocketClient..." << std::endl;
    
    // Create Polygon client with hardcoded API key (SINGLETON PATTERN)
    std::string api_key = "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57";
    
    // Ensure only ONE Polygon client exists per UltraFastWebSocketClient
    if (polygon_client_) {
        std::cout << "⚠️ Polygon client already exists - disconnecting old connection" << std::endl;
        polygon_client_->disconnect();
        polygon_client_.reset();
    }
    
    polygon_client_ = std::make_unique<PolygonWebSocketClient>(api_key);
    std::cout << "✅ Created single Polygon WebSocket client instance" << std::endl;
    
    // Initialize all symbols from config
    initialize_all_symbols();
    
    // Setup market data callback
    setup_market_data_callback();
    
    std::cout << "✅ UltraFastWebSocketClient initialized with " << all_symbols_.size() << " symbols" << std::endl;
}

UltraFastWebSocketClient::~UltraFastWebSocketClient() {
    stop();
}

bool UltraFastWebSocketClient::start() {
    std::cout << "🚀 Starting UltraFastWebSocketClient..." << std::endl;
    
    if (!polygon_client_) {
        std::cerr << "❌ Polygon client not initialized" << std::endl;
        return false;
    }
    
    // Connect to Polygon
    if (!polygon_client_->connect()) {
        std::cerr << "❌ Failed to connect to Polygon" << std::endl;
        return false;
    }
    
    // Authenticate
    if (!polygon_client_->authenticate()) {
        std::cerr << "❌ Failed to authenticate with Polygon" << std::endl;
        return false;
    }
    
    // Subscribe to all symbols from config for production
    if (!polygon_client_->subscribe_to_symbols(all_symbols_)) {
        std::cerr << "❌ Failed to subscribe to symbols" << std::endl;
        return false;
    }
    
    running_.store(true);
    
    std::cout << "✅ UltraFastWebSocketClient started successfully" << std::endl;
    std::cout << "📊 Subscribed to " << all_symbols_.size() << " symbols for real-time data" << std::endl;
    
    return true;
}

void UltraFastWebSocketClient::stop() {
    if (running_.load()) {
        running_.store(false);
        
        if (polygon_client_) {
            polygon_client_->disconnect();
        }
        
        std::cout << "🛑 UltraFastWebSocketClient stopped" << std::endl;
    }
}

void UltraFastWebSocketClient::set_market_data_callback(std::function<void(const ParsedMessage&)> callback) {
    market_data_callback_ = callback;
}

void UltraFastWebSocketClient::initialize_all_symbols() {
    std::cout << "🔧 Initializing symbols from config..." << std::endl;
    
    all_symbols_.clear();
    symbol_map_.clear();
    
    int32_t symbol_id = 0;
    
    // Add Tier 1 symbols
    for (size_t i = 0; i < APIConfig::NUM_TIER_1_SYMBOLS; ++i) {
        std::string symbol = APIConfig::TIER_1_SYMBOLS[i];
        symbol_map_[symbol] = symbol_id;
        all_symbols_.push_back(symbol);
        market_data_.activate_symbol(symbol_id);
        symbol_id++;
    }
    
    // Add Tier 2 symbols
    for (size_t i = 0; i < APIConfig::NUM_TIER_2_SYMBOLS; ++i) {
        std::string symbol = APIConfig::TIER_2_SYMBOLS[i];
        symbol_map_[symbol] = symbol_id;
        all_symbols_.push_back(symbol);
        market_data_.activate_symbol(symbol_id);
        symbol_id++;
    }
    
    std::cout << "✅ Initialized " << all_symbols_.size() << " symbols:" << std::endl;
    std::cout << "   - Tier 1: " << APIConfig::NUM_TIER_1_SYMBOLS << " symbols" << std::endl;
    std::cout << "   - Tier 2: " << APIConfig::NUM_TIER_2_SYMBOLS << " symbols" << std::endl;
}

void UltraFastWebSocketClient::setup_market_data_callback() {
    if (!polygon_client_) return;
    
    polygon_client_->set_message_callback([this](const std::string& symbol, const std::string& type,
                                                double price, uint64_t timestamp, double bid, double ask) {
        try {
            messages_received_.fetch_add(1);
            
            // Get symbol ID
            auto it = symbol_map_.find(symbol);
            if (it == symbol_map_.end()) {
                return; // Unknown symbol
            }
            
            int32_t symbol_id = it->second;
            
            // Update market data manager
            if (type == "T") {  // Trade
                market_data_.update_trade(symbol_id, static_cast<float>(price), 100, timestamp * 1000000);
            } else if (type == "Q" && bid > 0 && ask > 0) {  // Quote
                market_data_.update_quote(symbol_id, static_cast<float>(bid), static_cast<float>(ask), 100.0f, 100.0f);
            }
            
            // Forward to HFT engine callback
            if (market_data_callback_) {
                ParsedMessage msg = convert_to_parsed_message(symbol, type, price, timestamp, bid, ask);
                market_data_callback_(msg);
            }
            
            messages_processed_.fetch_add(1);
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Market data callback error: " << e.what() << std::endl;
        }
    });
}

ParsedMessage UltraFastWebSocketClient::convert_to_parsed_message(const std::string& symbol, const std::string& type,
                                                                double price, uint64_t timestamp, double bid, double ask) {
    ParsedMessage msg;
    
    // Set message type
    if (type == "T") {
        msg.type = MessageType::TRADE;
    } else if (type == "Q") {
        msg.type = MessageType::QUOTE;
    } else {
        msg.type = MessageType::AGGREGATE;
    }
    
    // Set symbol ID
    auto it = symbol_map_.find(symbol);
    msg.symbol_id = (it != symbol_map_.end()) ? it->second : -1;
    
    // Set data
    msg.timestamp_ns = timestamp * 1000000; // Convert to nanoseconds
    msg.price = static_cast<float>(price);
    msg.volume = 100; // Default volume
    msg.bid = static_cast<float>(bid);
    msg.ask = static_cast<float>(ask);
    msg.bid_size = 100;
    msg.ask_size = 100;
    
    return msg;
}

WebSocketPerformanceStats UltraFastWebSocketClient::get_performance_stats() const {
    if (polygon_client_) {
        auto stats = polygon_client_->get_performance_stats();
        stats.messages_received = messages_received_.load();
        stats.messages_processed = messages_processed_.load();
        return stats;
    }
    
    WebSocketPerformanceStats stats;
    stats.messages_received = messages_received_.load();
    stats.messages_processed = messages_processed_.load();
    return stats;
}

} // namespace hft
