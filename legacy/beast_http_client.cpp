#include "beast_http_client.hpp"
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ssl.hpp>
#include <iostream>
#include <sstream>

namespace hft {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;

//==============================================================================
// Simple BeastHttpClient Implementation - Works like curl
//==============================================================================

BeastHttpClient::BeastHttpClient(net::io_context& ioc, const std::string& host, 
                                const std::string& port, uint32_t rate_limit)
    : ioc_(ioc), host_(host), port_(port), user_agent_("HFT-Beast-Client/1.0") {
    
    ssl_ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    
    // Set default headers
    default_headers_["User-Agent"] = user_agent_;
    default_headers_["Connection"] = "keep-alive";
    default_headers_["Accept"] = "application/json";
    default_headers_["Content-Type"] = "application/json";
}

BeastHttpClient::~BeastHttpClient() {
    stop();
}

bool BeastHttpClient::initialize() {
    try {
        // Disable SSL verification - just like curl -k
        ssl_ctx_->set_verify_mode(ssl::verify_none);
        std::cout << "✅ BeastHttpClient initialized for " << host_ << ":" << port_ << " (SSL verification disabled)" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize BeastHttpClient: " << e.what() << std::endl;
        return false;
    }
}

void BeastHttpClient::start() {
    running_.store(true);
    std::cout << "✅ BeastHttpClient started" << std::endl;
}

void BeastHttpClient::stop() {
    running_.store(false);
    std::cout << "🛑 BeastHttpClient stopped" << std::endl;
}

void BeastHttpClient::set_default_header(const std::string& name, const std::string& value) {
    default_headers_[name] = value;
}

void BeastHttpClient::set_user_agent(const std::string& user_agent) {
    user_agent_ = user_agent;
    default_headers_["User-Agent"] = user_agent;
}

void BeastHttpClient::set_ssl_verification(bool verify) {
    // Ignored - always disabled for simplicity
}

// Simple synchronous HTTP request
HttpResponse BeastHttpClient::make_simple_request(http::verb method, const std::string& target, 
                                                  const std::string& body,
                                                  const std::unordered_map<std::string, std::string>& headers) {
    HttpResponse response;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Resolve the host
        tcp::resolver resolver(ioc_);
        auto const results = resolver.resolve(host_, port_);
        
        // Create SSL stream
        ssl::stream<tcp::socket> stream(ioc_, *ssl_ctx_);
        
        // Set SNI hostname
        if (!SSL_set_tlsext_host_name(stream.native_handle(), host_.c_str())) {
            throw std::runtime_error("Failed to set SNI hostname");
        }
        
        // Connect to the first endpoint
        beast::get_lowest_layer(stream).connect(results.begin()->endpoint());
        
        // Perform SSL handshake
        stream.handshake(ssl::stream_base::client);
        
        // Build HTTP request
        http::request<http::string_body> req{method, target, 11};
        req.set(http::field::host, host_);
        
        // Add default headers
        for (const auto& [name, value] : default_headers_) {
            req.set(name, value);
        }
        
        // Add custom headers
        for (const auto& [name, value] : headers) {
            req.set(name, value);
        }
        
        // Set body if provided
        if (!body.empty()) {
            req.body() = body;
            req.prepare_payload();
        }
        
        // Send the request
        http::write(stream, req);
        
        // Receive the response
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);
        
        // Fill response
        response.success = true;
        response.status_code = res.result_int();
        response.body = res.body();
        
        // Copy headers
        for (const auto& field : res) {
            response.headers[std::string(field.name_string())] = std::string(field.value());
        }
        
        // Graceful shutdown
        beast::error_code ec;
        stream.shutdown(ec);
        
        auto end_time = std::chrono::steady_clock::now();
        response.latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        std::cout << "✅ " << method << " " << target << " -> HTTP " << response.status_code 
                  << " (" << response.latency.count() / 1000000.0 << "ms)" << std::endl;
        
        // Debug: print response body
        if (!response.body.empty() && response.body.size() < 500) {
            std::cout << "📄 Response: " << response.body << std::endl;
        }
        
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = e.what();
        std::cerr << "❌ Request failed: " << e.what() << std::endl;
    }
    
    return response;
}

void BeastHttpClient::async_get(const std::string& target,
    std::function<void(HttpResponse)> callback,
    const std::unordered_map<std::string, std::string>& headers,
    std::chrono::milliseconds timeout) {
    
    // For simplicity, just do synchronous request and call callback
    auto response = make_simple_request(http::verb::get, target, "", headers);
    callback(response);
}

void BeastHttpClient::async_post(const std::string& target,
    const std::string& body,
    std::function<void(HttpResponse)> callback,
    const std::unordered_map<std::string, std::string>& headers,
    std::chrono::milliseconds timeout) {
    
    // For simplicity, just do synchronous request and call callback
    auto response = make_simple_request(http::verb::post, target, body, headers);
    callback(response);
}

void BeastHttpClient::async_delete(const std::string& target,
    std::function<void(HttpResponse)> callback,
    const std::unordered_map<std::string, std::string>& headers,
    std::chrono::milliseconds timeout) {
    
    // For simplicity, just do synchronous request and call callback
    auto response = make_simple_request(http::verb::delete_, target, "", headers);
    callback(response);
}

void BeastHttpClient::async_put(const std::string& target,
    const std::string& body,
    std::function<void(HttpResponse)> callback,
    const std::unordered_map<std::string, std::string>& headers,
    std::chrono::milliseconds timeout) {
    
    // For simplicity, just do synchronous request and call callback
    auto response = make_simple_request(http::verb::put, target, body, headers);
    callback(response);
}

BeastHttpClient::PerformanceStats BeastHttpClient::get_performance_stats() const {
    return PerformanceStats{};
}

void BeastHttpClient::reset_stats() {
    // No stats in simple version
}

// Stubs for removed complex features
void BeastHttpClient::register_io_thread(std::thread::id thread_id) {}
bool BeastHttpClient::is_io_context_thread() const { return false; }
void BeastHttpClient::validate_thread_safety() const {}

//==============================================================================
// Factory Function
//==============================================================================

std::unique_ptr<BeastHttpClient> create_alpaca_client(net::io_context& ioc) {
    std::cout << "🏗️  Creating simple Alpaca client for paper-api.alpaca.markets" << std::endl;
    
    auto client = std::make_unique<BeastHttpClient>(ioc, "paper-api.alpaca.markets", "443", 180);
    
    if (!client->initialize()) {
        std::cerr << "❌ Failed to initialize Alpaca client" << std::endl;
        return nullptr;
    }
    
    std::cout << "✅ Simple Alpaca client created successfully" << std::endl;
    return client;
}

} // namespace hft
