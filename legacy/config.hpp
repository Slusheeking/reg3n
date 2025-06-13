#pragma once

#include <string>
#include <cstdlib> // For std::getenv
#include <stdexcept> // For std::runtime_error

namespace hft {

// Helper function to get environment variables
inline std::string get_env_var(const char* name, const char* default_val = nullptr) {
    const char* value = std::getenv(name);
    if (value) {
        return std::string(value);
    }
    if (default_val) {
        return std::string(default_val);
    }
    throw std::runtime_error(std::string("Environment variable not set: ") + name);
}

// API Configuration
struct APIConfig {
    // Polygon API
    std::string POLYGON_API_KEY;
    static constexpr const char* POLYGON_BASE_URL = "https://api.polygon.io";
    
    // Alpaca API (Paper Trading)
    std::string ALPACA_API_KEY;
    std::string ALPACA_SECRET_KEY;
    static constexpr const char* ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2";
    // Fixed Alpaca WebSocket URLs for Paper Trading
    // Updated Alpaca WebSocket URLs for Paper Trading API v2
    static constexpr const char* ALPACA_WS_URL_ACCOUNT = "wss://paper-api.alpaca.markets/stream";
    static constexpr const char* ALPACA_WS_URL_MARKET = "wss://stream.data.alpaca.markets/v2/iex";

    APIConfig() {
        // Hardcoded API keys for production deployment
        POLYGON_API_KEY = "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57";
        ALPACA_API_KEY = "PKOA0DZRDVPMC7V6A4EU";
        ALPACA_SECRET_KEY = "1BM2AiVp8N6Glbc5fm14up7KEh1V8KNFleD5jgYu";
    }
    
    // Trading Configuration - Aggressive Production Settings
    static constexpr double MAX_POSITION_SIZE = 10000.0;  // $10K max per position
    static constexpr double MAX_PORTFOLIO_RISK = 0.95;    // 95% max portfolio risk (AGGRESSIVE)
    static constexpr double STOP_LOSS_PCT = 0.008;        // 0.8% stop loss (tighter)
    static constexpr double TAKE_PROFIT_PCT = 0.025;      // 2.5% take profit (higher target)
    
    // Production Risk Limits - Optimized for $1000+ Daily Profit
    static constexpr double MAX_POSITION_SIZE_PCT = 0.20; // 20% per position (up from 5%)
    static constexpr double MAX_TOTAL_EXPOSURE = 0.95;    // 95% max exposure
    static constexpr double MIN_CASH_RESERVE = 0.05;      // 5% cash reserve (down from 10%)
    static constexpr double MAX_DAILY_LOSS_PCT = 0.03;    // 3% daily loss limit
    static constexpr double MAX_SYMBOL_EXPOSURE = 0.25;   // 25% per symbol (up from 10%)
    static constexpr uint32_t MAX_ORDERS_PER_SEC = 200;   // Increased rate limit
    
    // Performance Targets
    static constexpr double TARGET_LATENCY_US = 200.0;    // 200μs target
    static constexpr double MAX_LATENCY_US = 500.0;       // 500μs max acceptable
    
    // Critical Configuration Updates
    static constexpr double POSITION_SIZE_MULTIPLIER = 0.5;  // Half-Kelly multiplier
    static constexpr double MAX_CORRELATED_EXPOSURE = 0.6;   // 60% max correlated positions
    static constexpr double VOLATILITY_SCALING_FACTOR = 2.0; // Vol-based size adjustment
    static constexpr uint32_t MIN_TRADES_FOR_KELLY = 20;     // Minimum trades before using Kelly
    
    // PRODUCTION SYMBOLS - Exactly 25 symbols for optimal diversification
    // Each position limited to $2K for total $50K capital allocation
    static constexpr const char* TIER_1_SYMBOLS[] = {
        "SPY",    // SPDR S&P 500 ETF - #1 liquidity
        "NVDA",   // NVIDIA - AI leader
        "QQQ",    // Nasdaq 100 ETF
        "TSLA",   // Tesla - High volatility
        "AAPL",   // Apple - Mega cap
        "MSFT",   // Microsoft - Mega cap
        "META",   // Meta Platforms
        "AMZN",   // Amazon
        "GOOGL",  // Alphabet
        "AMD",    // Advanced Micro Devices
        "TQQQ",   // 3x Nasdaq ETF
        "SOXL",   // 3x Semiconductor ETF
        "BAC",    // Bank of America
        "COIN",   // Coinbase
        "UNH",    // UnitedHealth
        "WMT",    // Walmart
        "V",      // Visa
        "IWM",    // Russell 2000 ETF
        "GLD",    // Gold ETF
        "XOM",    // ExxonMobil
        "VXX",    // Volatility ETF
        "TLT",    // Treasury bonds
        "EEM",    // Emerging markets
        "DIA",    // Dow Jones ETF
        "ARKK"    // ARK Innovation ETF
    };
    
    // No Tier 2 symbols - using single tier for simplicity
    static constexpr const char* TIER_2_SYMBOLS[] = {};
    
    static constexpr size_t NUM_TIER_1_SYMBOLS = sizeof(TIER_1_SYMBOLS) / sizeof(TIER_1_SYMBOLS[0]);
    static constexpr size_t NUM_TIER_2_SYMBOLS = 0; // No Tier 2 symbols
    static constexpr size_t TOTAL_SYMBOLS = 25; // Exactly 25 symbols
    
    // Validate symbol count at compile time
    static_assert(NUM_TIER_1_SYMBOLS == 25, "Must have exactly 25 Tier 1 symbols");
};

} // namespace hft
