#pragma once

#include "simple_config.hpp"
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <functional>
#include <string>
#include <atomic>
#include <thread>
#include <iostream>
#include <unordered_map>

namespace simple_hft {

// Market data message from Polygon
struct MarketMessage {
    std::string symbol;
    std::string type;       // "trade" or "quote"
    double price;
    double volume;
    double bid, ask;
    double bid_size, ask_size;
    uint64_t timestamp;
    bool valid;
    
    MarketMessage() : price(0), volume(0), bid(0), ask(0), bid_size(0), ask_size(0), 
                     timestamp(0), valid(false) {}
};

// Simple Polygon WebSocket client - FAIL FAST design
class SimplePolygonClient {
public:
    using client = websocketpp::client<websocketpp::config::asio_tls_client>;
    using message_ptr = websocketpp::config::asio_tls_client::message_type::ptr;
    using context_ptr = websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>;
    
    explicit SimplePolygonClient() 
        : connected_(false), authenticated_(false), running_(false), error_count_(0) {
        
        // Configure WebSocket client
        ws_client_.clear_access_channels(websocketpp::log::alevel::all);
        ws_client_.clear_error_channels(websocketpp::log::elevel::all);
        ws_client_.set_access_channels(websocketpp::log::alevel::connect);
        ws_client_.set_access_channels(websocketpp::log::alevel::disconnect);
        
        ws_client_.init_asio();
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
    }
    
    ~SimplePolygonClient() {
        disconnect();
    }
    
    // Set callback for market data messages
    void set_message_callback(std::function<void(const MarketMessage&)> callback) {
        message_callback_ = callback;
    }
    
    // Connect and authenticate - FAIL FAST on any error
    bool connect() {
        try {
            std::cout << "Connecting to Polygon WebSocket..." << std::endl;
            
            websocketpp::lib::error_code ec;
            auto con = ws_client_.get_connection(SimpleConfig::POLYGON_WS_URL, ec);
            
            if (ec) {
                std::cerr << "Connection creation error: " << ec.message() << std::endl;
                return false;
            }
            
            connection_hdl_ = con->get_handle();
            ws_client_.connect(con);
            
            // Start the WebSocket thread
            ws_thread_ = std::thread([this]() {
                try {
                    ws_client_.run();
                } catch (const std::exception& e) {
                    std::cerr << "WebSocket thread error: " << e.what() << std::endl;
                    this->increment_error_count();
                }
            });
            
            running_ = true;
            
            // Wait for connection with timeout
            int timeout_count = 0;
            while (!connected_ && timeout_count < 50) { // 5 second timeout
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                timeout_count++;
            }
            
            if (!connected_) {
                std::cerr << "Connection timeout" << std::endl;
                return false;
            }
            
            std::cout << "Connected to Polygon WebSocket" << std::endl;
            return authenticate();
            
        } catch (const std::exception& e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            increment_error_count();
            return false;
        }
    }
    
    // Authenticate with Polygon
    bool authenticate() {
        try {
            std::cout << "Authenticating with Polygon..." << std::endl;
            
            // Simple string-based auth message for Polygon WebSocket
            std::string auth_json = R"({"action":"auth","params":")" + std::string(SimpleConfig::POLYGON_API_KEY) + R"("})";
            
            websocketpp::lib::error_code ec;
            ws_client_.send(connection_hdl_, auth_json,
                           websocketpp::frame::opcode::text, ec);
            
            if (ec) {
                std::cerr << "Auth send error: " << ec.message() << std::endl;
                increment_error_count();
                return false;
            }
            
            // Wait for authentication with timeout
            int timeout_count = 0;
            while (!authenticated_ && timeout_count < 100) { // 10 second timeout
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                timeout_count++;
            }
            
            if (!authenticated_) {
                std::cerr << "Authentication timeout" << std::endl;
                increment_error_count();
                return false;
            }
            
            std::cout << "Authenticated with Polygon" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Authentication error: " << e.what() << std::endl;
            increment_error_count();
            return false;
        }
    }
    
    // Subscribe to all symbols
    bool subscribe_to_symbols() {
        try {
            std::cout << "Subscribing to symbols..." << std::endl;
            
            // CRITICAL: Polygon.io expects params as comma-separated string, not array
            std::string params_str = "";
            bool first = true;
            
            for (const auto& symbol : SimpleConfig::SYMBOLS) {
                if (!first) params_str += ",";
                params_str += "T." + symbol;  // Trade channel
                params_str += ",Q." + symbol; // Quote channel
                first = false;
            }
            
            // Build JSON message with string params (not array)
            std::string sub_json = "{\"action\":\"subscribe\",\"params\":\"" + params_str + "\"}";
            
            websocketpp::lib::error_code ec;
            ws_client_.send(connection_hdl_, sub_json,
                           websocketpp::frame::opcode::text, ec);
            
            if (ec) {
                std::cerr << "Subscription send error: " << ec.message() << std::endl;
                increment_error_count();
                return false;
            }
            
            std::cout << "Subscribed to " << SimpleConfig::SYMBOLS.size() << " symbols" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Subscription error: " << e.what() << std::endl;
            increment_error_count();
            return false;
        }
    }
    
    void disconnect() {
        if (running_) {
            running_ = false;
            connected_ = false;
            authenticated_ = false;
            
            if (ws_thread_.joinable()) {
                ws_client_.stop();
                ws_thread_.join();
            }
        }
    }
    
    bool is_connected() const { return connected_.load(); }
    bool is_authenticated() const { return authenticated_.load(); }
    int get_error_count() const { return error_count_.load(); }
    
private:
    // TLS context initialization (matches legacy working code)
    context_ptr on_tls_init(websocketpp::connection_hdl hdl) {
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
            std::cerr << "TLS context error: " << e.what() << std::endl;
        }
        
        return ctx;
    }

    void on_open(websocketpp::connection_hdl hdl) {
        std::cout << "WebSocket connection opened" << std::endl;
        connected_ = true;
    }
    
    void on_close(websocketpp::connection_hdl hdl) {
        std::cout << "WebSocket connection closed" << std::endl;
        connected_ = false;
        authenticated_ = false;
        increment_error_count();
    }
    
    void on_fail(websocketpp::connection_hdl hdl) {
        std::cerr << "WebSocket connection failed" << std::endl;
        connected_ = false;
        authenticated_ = false;
        increment_error_count();
    }
    
    void on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
        try {
            const std::string& payload = msg->get_payload();
            
            // Parse JSON message
            rapidjson::Document doc;
            if (doc.Parse(payload.c_str()).HasParseError()) {
                return; // Skip invalid JSON
            }
            
            // Handle different message types
            if (doc.IsArray() && doc.Size() > 0) {
                // Process each array item - could be control or market data
                for (auto& item : doc.GetArray()) {
                    if (item.IsObject()) {
                        // Check if it's a control message first
                        if (item.HasMember("ev") && item.HasMember("status")) {
                            process_control_message_from_value(item);
                        } else if (item.HasMember("status") && !item.HasMember("ev")) {
                            process_control_message_from_value(item);
                        } else {
                            // Market data message
                            process_market_data(item);
                        }
                    }
                }
            } else if (doc.IsObject()) {
                // Control messages (auth response, status, etc.)
                process_control_message(doc);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Message processing error: " << e.what() << std::endl;
            // Don't increment error count for individual message errors
        }
    }
    
    void process_control_message(const rapidjson::Document& doc) {
        process_control_message_from_value(doc);
    }
    
    void process_control_message_from_value(const rapidjson::Value& value) {
        // Handle status messages (connection, subscription confirmations)
        if (value.HasMember("ev") && value.HasMember("status")) {
            std::string ev = value["ev"].GetString();
            std::string status = value["status"].GetString();
            
            if (ev == "status" && status == "connected") {
                std::cout << "✅ Polygon connection confirmed!" << std::endl;
            } else if (ev == "status" && status == "auth_success") {
                std::cout << "✅ Polygon authentication successful!" << std::endl;
                authenticated_ = true;
            } else if (ev == "status" && status == "success") {
                // Subscription successful (silent)
            }
        }
        
        // Handle legacy status format (single object with "status" field)
        if (value.HasMember("status") && !value.HasMember("ev")) {
            std::string status = value["status"].GetString();
            
            if (status == "connected") {
                std::cout << "✅ Polygon connection confirmed!" << std::endl;
            } else if (status == "auth_success") {
                std::cout << "✅ Polygon authentication successful!" << std::endl;
                authenticated_ = true;
            } else if (status == "success") {
                // Subscription successful (silent)
            } else if (status.find("error") != std::string::npos) {
                std::cerr << "❌ Polygon error: " << status << std::endl;
                increment_error_count();
            }
        }
    }
    
    void process_market_data(const rapidjson::Value& item) {
        if (!item.IsObject() || !item.HasMember("ev") || !item.HasMember("sym")) {
            return;
        }
        
        std::string event_type = item["ev"].GetString();
        std::string symbol = item["sym"].GetString();
        
        MarketMessage msg;
        msg.symbol = symbol;
        msg.valid = true;
        
        if (event_type == "T") {
            // Trade message
            msg.type = "trade";
            
            if (item.HasMember("p") && item["p"].IsNumber()) {
                msg.price = item["p"].GetDouble();
            }
            
            if (item.HasMember("s") && item["s"].IsNumber()) {
                msg.volume = item["s"].GetDouble();
            }
            
            if (item.HasMember("t") && item["t"].IsNumber()) {
                msg.timestamp = item["t"].GetUint64();
            }
            
        } else if (event_type == "Q") {
            // Quote message
            msg.type = "quote";
            
            if (item.HasMember("bp") && item["bp"].IsNumber()) {
                msg.bid = item["bp"].GetDouble();
            }
            
            if (item.HasMember("ap") && item["ap"].IsNumber()) {
                msg.ask = item["ap"].GetDouble();
            }
            
            if (item.HasMember("bs") && item["bs"].IsNumber()) {
                msg.bid_size = item["bs"].GetDouble();
            }
            
            if (item.HasMember("as") && item["as"].IsNumber()) {
                msg.ask_size = item["as"].GetDouble();
            }
            
            if (item.HasMember("t") && item["t"].IsNumber()) {
                msg.timestamp = item["t"].GetUint64();
            }
        }
        
        // Send to callback if valid
        if (msg.valid && message_callback_) {
            message_callback_(msg);
        }
    }
    
    void increment_error_count() {
        int current_errors = error_count_.fetch_add(1) + 1;
        if (current_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            std::cerr << "CRITICAL: Maximum errors reached (" << current_errors 
                      << "). System will shutdown." << std::endl;
            // The main system should monitor error_count and shutdown
        }
    }
    
    // WebSocket client
    client ws_client_;
    websocketpp::connection_hdl connection_hdl_;
    std::thread ws_thread_;
    
    // State
    std::atomic<bool> connected_;
    std::atomic<bool> authenticated_;
    std::atomic<bool> running_;
    std::atomic<int> error_count_;
    
    // Callback
    std::function<void(const MarketMessage&)> message_callback_;
};

} // namespace simple_hft