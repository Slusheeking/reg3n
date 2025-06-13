#pragma once

#include <string>
#include <vector>
#include <array>

namespace simple_hft {

// Simple configuration - reuse existing API keys and symbols
struct SimpleConfig {
    // API Keys (from existing config.hpp)
    static constexpr const char* POLYGON_API_KEY = "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57";
    static constexpr const char* ALPACA_API_KEY = "PKOA0DZRDVPMC7V6A4EU";
    static constexpr const char* ALPACA_SECRET_KEY = "1BM2AiVp8N6Glbc5fm14up7KEh1V8KNFleD5jgYu";
    
    // API URLs
    static constexpr const char* POLYGON_WS_URL = "wss://socket.polygon.io/stocks";
    static constexpr const char* ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2";
    
    // All symbols from legacy config (no tiers, all symbols)
    static const std::vector<std::string> SYMBOLS;
    
    // Trading parameters - FIXED VALUES (no online learning)
    static constexpr double CONFIDENCE_THRESHOLD = 0.80;    // Higher threshold for better quality signals
    static constexpr double POSITION_SIZE_DOLLARS = 2000.0; // Fixed $2000 per trade
    static constexpr double TAKE_PROFIT_PCT = 0.002;        // 0.2% take profit (very tight scalping)
    static constexpr double STOP_LOSS_PCT = 0.012;          // 1.2% stop loss (wider to avoid noise)
    static constexpr int MAX_POSITIONS = 24;                // Maximum concurrent positions
    static constexpr int PORTFOLIO_VALUE = 50000;           // Fixed portfolio value
    
    // Feature engineering parameters
    static constexpr int FEATURE_COUNT = 10;
    static constexpr int LOOKBACK_PERIOD = 100;             // Price history lookback
    
    // Error handling
    static constexpr int MAX_ERRORS_BEFORE_SHUTDOWN = 3;    // Fail-fast after 3 errors
    static constexpr int RECONNECT_ATTEMPTS = 2;            // Limited reconnection attempts
    
    // Logging
    static constexpr bool ENABLE_VERBOSE_LOGGING = true;
    static constexpr const char* LOG_FILE = "simple_hft.log";
};

// All symbols from legacy config (complete list, no tiers)
const std::vector<std::string> SimpleConfig::SYMBOLS = {
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

} // namespace simple_hft