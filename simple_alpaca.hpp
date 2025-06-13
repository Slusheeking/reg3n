#pragma once

#include "simple_config.hpp"
#include "simple_models.hpp"
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace simple_hft {

// Order result from Alpaca
struct OrderResult {
    bool success;
    std::string order_id;
    std::string error_message;
    std::string symbol;
    double quantity;
    
    OrderResult() : success(false), quantity(0.0) {}
};

// Account information from Alpaca
struct AccountInfo {
    double buying_power;
    double portfolio_value;
    double cash;
    int position_count;
    bool valid;
    
    AccountInfo() : buying_power(0), portfolio_value(0), cash(0), position_count(0), valid(false) {}
};

// Simple callback for libcurl
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append((char*)contents, total_size);
    return total_size;
}

// Simple Alpaca HTTP client - FAIL FAST design
class SimpleAlpacaClient {
private:
    CURL* curl_;
    struct curl_slist* headers_;
    std::atomic<int> error_count_;
    std::unordered_map<std::string, int> open_positions_; // symbol -> position count
    
    // Round price according to Alpaca's sub-penny rules
    double round_price_for_alpaca(double price) {
        if (price >= 1.0) {
            // Prices >= $1.00: Max 2 decimal places (cents)
            return std::round(price * 100.0) / 100.0;
        } else {
            // Prices < $1.00: Max 4 decimal places
            return std::round(price * 10000.0) / 10000.0;
        }
    }
    
public:
    SimpleAlpacaClient() : curl_(nullptr), headers_(nullptr), error_count_(0) {
        // Initialize curl
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_ = curl_easy_init();
        
        if (!curl_) {
            throw std::runtime_error("Failed to initialize CURL");
        }
        
        // Set up headers with API keys
        std::string auth_header = "APCA-API-KEY-ID: " + std::string(SimpleConfig::ALPACA_API_KEY);
        std::string secret_header = "APCA-API-SECRET-KEY: " + std::string(SimpleConfig::ALPACA_SECRET_KEY);
        
        headers_ = curl_slist_append(headers_, "Content-Type: application/json");
        headers_ = curl_slist_append(headers_, auth_header.c_str());
        headers_ = curl_slist_append(headers_, secret_header.c_str());
        
        // Set common curl options
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 10L); // 10 second timeout
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYHOST, 2L);
        
        std::cout << "Alpaca client initialized" << std::endl;
    }
    
    ~SimpleAlpacaClient() {
        if (headers_) {
            curl_slist_free_all(headers_);
        }
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
        curl_global_cleanup();
    }
    
    // Get account information - FAIL FAST on error
    AccountInfo get_account_info() {
        AccountInfo info;
        
        try {
            std::string url = std::string(SimpleConfig::ALPACA_BASE_URL) + "/account";
            std::string response;
            
            curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
            curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl_);
            
            if (res != CURLE_OK) {
                std::cerr << "Account info request failed: " << curl_easy_strerror(res) << std::endl;
                increment_error_count();
                return info;
            }
            
            // Parse response
            rapidjson::Document doc;
            if (doc.Parse(response.c_str()).HasParseError()) {
                std::cerr << "Failed to parse account response" << std::endl;
                increment_error_count();
                return info;
            }
            
            if (doc.HasMember("buying_power") && doc["buying_power"].IsString()) {
                info.buying_power = std::stod(doc["buying_power"].GetString());
            }
            
            if (doc.HasMember("portfolio_value") && doc["portfolio_value"].IsString()) {
                info.portfolio_value = std::stod(doc["portfolio_value"].GetString());
            }
            
            if (doc.HasMember("cash") && doc["cash"].IsString()) {
                info.cash = std::stod(doc["cash"].GetString());
            }
            
            info.valid = true;
            return info;
            
        } catch (const std::exception& e) {
            std::cerr << "Account info error: " << e.what() << std::endl;
            increment_error_count();
            return info;
        }
    }
    
    // Get current positions count
    int get_position_count() {
        try {
            std::string url = std::string(SimpleConfig::ALPACA_BASE_URL) + "/positions";
            std::string response;
            
            curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
            curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl_);
            
            if (res != CURLE_OK) {
                std::cerr << "Positions request failed: " << curl_easy_strerror(res) << std::endl;
                increment_error_count();
                return -1;
            }
            
            // Parse response
            rapidjson::Document doc;
            if (doc.Parse(response.c_str()).HasParseError()) {
                std::cerr << "Failed to parse positions response" << std::endl;
                increment_error_count();
                return -1;
            }
            
            if (doc.IsArray()) {
                // Update our position tracking
                open_positions_.clear();
                for (auto& pos : doc.GetArray()) {
                    if (pos.HasMember("symbol") && pos["symbol"].IsString()) {
                        std::string symbol = pos["symbol"].GetString();
                        open_positions_[symbol] = 1; // Mark as having position
                    }
                }
                return doc.Size();
            }
            
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Position count error: " << e.what() << std::endl;
            increment_error_count();
            return -1;
        }
    }
    
    // Cancel all open orders and liquidate all positions
    bool cancel_all_orders_and_liquidate() {
        try {
            std::cout << "🔄 Starting complete liquidation..." << std::endl;
            
            // Step 1: Cancel all open orders
            std::cout << "📋 Cancelling all open orders..." << std::endl;
            std::string url = std::string(SimpleConfig::ALPACA_BASE_URL) + "/orders";
            std::string response;
            
            curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, "");
            curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl_);
            
            if (res != CURLE_OK) {
                std::cerr << "❌ Failed to cancel orders: " << curl_easy_strerror(res) << std::endl;
                return false;
            }
            std::cout << "✅ All orders cancelled" << std::endl;
            
            // Step 2: Close all positions
            std::cout << "💰 Liquidating all positions..." << std::endl;
            url = std::string(SimpleConfig::ALPACA_BASE_URL) + "/positions?cancel_orders=true";
            response.clear();
            
            curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, "");
            curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
            
            res = curl_easy_perform(curl_);
            
            // Reset to POST for future requests
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "POST");
            
            if (res != CURLE_OK) {
                std::cerr << "❌ Failed to liquidate positions: " << curl_easy_strerror(res) << std::endl;
                return false;
            }
            
            std::cout << "✅ All positions liquidated successfully" << std::endl;
            std::cout << "🧹 Portfolio reset to cash-only" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error during liquidation: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Check if we can place an order (position limits, etc.)
    bool can_place_order(const std::string& symbol, const AccountInfo& account) {
        // Check max positions limit
        if (open_positions_.size() >= SimpleConfig::MAX_POSITIONS) {
            return false;
        }
        
        // Check if we already have a position in this symbol
        if (open_positions_.find(symbol) != open_positions_.end()) {
            return false;
        }
        
        // Check buying power
        double order_value = SimpleConfig::POSITION_SIZE_DOLLARS;
        if (account.buying_power < order_value) {
            return false;
        }
        
        return true;
    }
    
    // Place bracket order - FAIL FAST on error
    OrderResult place_bracket_order(const SimpleSignal& signal, double current_price) {
        OrderResult result;
        result.symbol = signal.symbol;
        
        try {
            // Get account info first
            AccountInfo account = get_account_info();
            if (!account.valid) {
                result.error_message = "Failed to get account information";
                increment_error_count();
                return result;
            }
            
            // Check if we can place the order
            if (!can_place_order(signal.symbol, account)) {
                result.error_message = "Order rejected by risk checks";
                return result; // Not an error, just a business rule
            }
            
            // Calculate quantities and prices (signal.position_size is now fixed dollar amount)
            double order_value = signal.position_size;
            int quantity = static_cast<int>(order_value / current_price);
            
            if (quantity <= 0) {
                result.error_message = "Invalid quantity calculated";
                return result;
            }
            
            // Calculate bracket prices
            double take_profit_price, stop_loss_price;
            if (signal.direction == 1) { // Buy order
                take_profit_price = current_price * (1.0 + SimpleConfig::TAKE_PROFIT_PCT);
                stop_loss_price = current_price * (1.0 - SimpleConfig::STOP_LOSS_PCT);
            } else { // Sell order (short)
                take_profit_price = current_price * (1.0 - SimpleConfig::TAKE_PROFIT_PCT);
                stop_loss_price = current_price * (1.0 + SimpleConfig::STOP_LOSS_PCT);
            }
            
            // Round all prices according to Alpaca's sub-penny rules
            double rounded_current_price = round_price_for_alpaca(current_price);
            double rounded_take_profit = round_price_for_alpaca(take_profit_price);
            double rounded_stop_loss = round_price_for_alpaca(stop_loss_price);
            
            result.quantity = quantity;
            
            // Build bracket order JSON
            rapidjson::Document order_doc;
            order_doc.SetObject();
            auto& allocator = order_doc.GetAllocator();
            
            order_doc.AddMember("symbol", rapidjson::Value(signal.symbol.c_str(), allocator), allocator);
            order_doc.AddMember("qty", quantity, allocator);
            order_doc.AddMember("side", rapidjson::Value(signal.direction == 1 ? "buy" : "sell", allocator), allocator);
            order_doc.AddMember("type", "market", allocator);
            order_doc.AddMember("time_in_force", "day", allocator);
            order_doc.AddMember("order_class", "bracket", allocator);
            
            // Take profit leg
            rapidjson::Value take_profit(rapidjson::kObjectType);
            take_profit.AddMember("limit_price", rounded_take_profit, allocator);
            order_doc.AddMember("take_profit", take_profit, allocator);
            
            // Stop loss leg
            rapidjson::Value stop_loss(rapidjson::kObjectType);
            stop_loss.AddMember("stop_price", rounded_stop_loss, allocator);
            stop_loss.AddMember("limit_price", rounded_stop_loss, allocator);
            order_doc.AddMember("stop_loss", stop_loss, allocator);
            
            // Convert to JSON string
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            order_doc.Accept(writer);
            std::string json_data = buffer.GetString();
            
            // Send order
            std::string url = std::string(SimpleConfig::ALPACA_BASE_URL) + "/orders";
            std::string response;
            
            curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_data.c_str());
            curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl_);
            
            if (res != CURLE_OK) {
                result.error_message = "Order request failed: " + std::string(curl_easy_strerror(res));
                increment_error_count();
                return result;
            }
            
            // Parse response
            rapidjson::Document response_doc;
            if (response_doc.Parse(response.c_str()).HasParseError()) {
                result.error_message = "Failed to parse order response";
                increment_error_count();
                return result;
            }
            
            // Check for success
            if (response_doc.HasMember("id") && response_doc["id"].IsString()) {
                result.success = true;
                result.order_id = response_doc["id"].GetString();
                
                // Track the new position
                open_positions_[signal.symbol] = 1;
                
                std::cout << "✅ Bracket order placed successfully:" << std::endl;
                std::cout << "   Symbol: " << signal.symbol << std::endl;
                std::cout << "   Quantity: " << quantity << std::endl;
                std::cout << "   Entry: Market Order (immediate fill)" << std::endl;
                std::cout << "   Take Profit: $" << rounded_take_profit << std::endl;
                std::cout << "   Stop Loss: $" << rounded_stop_loss << std::endl;
                std::cout << "   Strategy: " << signal.strategy << std::endl;
                std::cout << "   Order ID: " << result.order_id << std::endl;
                
            } else if (response_doc.HasMember("message")) {
                result.error_message = "Order rejected: " + std::string(response_doc["message"].GetString());
            } else {
                result.error_message = "Unknown order response: " + response;
            }
            
            return result;
            
        } catch (const std::exception& e) {
            result.error_message = "Order placement error: " + std::string(e.what());
            increment_error_count();
            return result;
        }
    }
    
    // Check if we have too many errors (for fail-fast)
    bool has_too_many_errors() const {
        return error_count_.load() >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN;
    }
    
    int get_error_count() const {
        return error_count_.load();
    }
    
    // Test connection to Alpaca
    bool test_connection() {
        AccountInfo info = get_account_info();
        return info.valid;
    }

private:
    void increment_error_count() {
        int current_errors = error_count_.fetch_add(1) + 1;
        if (current_errors >= SimpleConfig::MAX_ERRORS_BEFORE_SHUTDOWN) {
            std::cerr << "CRITICAL: Alpaca client maximum errors reached (" << current_errors 
                      << "). System should shutdown." << std::endl;
        }
    }
};

} // namespace simple_hft