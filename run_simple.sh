#!/bin/bash

echo "=== Simple HFT System Runner ==="

# Make sure the system is built first
if [ ! -f "build_simple/simple_hft" ]; then
    echo "Simple HFT system not found. Building it first..."
    chmod +x build_simple.sh
    ./build_simple.sh
    
    if [ $? -ne 0 ]; then
        echo "❌ Build failed. Cannot run."
        exit 1
    fi
fi

# Parse command line arguments
TEST_MODE=""
DURATION=""
LOG_TO_FILE=true

for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE="--test"
            shift
            ;;
        --duration)
            DURATION="--duration $2"
            shift 2
            ;;
        --no-log)
            LOG_TO_FILE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --test              Run in test mode (60 seconds)"
            echo "  --duration N        Test duration in seconds"
            echo "  --no-log           Don't save logs to file"
            echo "  --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                          # Production mode with logging"
            echo "  $0 --test                   # Test mode for 60 seconds"
            echo "  $0 --test --duration 300    # Test mode for 5 minutes"
            echo "  $0 --no-log                # Production mode without log file"
            exit 0
            ;;
        *)
            # Unknown option, pass it through
            ;;
    esac
done

# Create logs directory
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/simple_hft_${TIMESTAMP}.log"

# Pre-flight checks
echo "🔍 Pre-flight checks..."

# Check internet connectivity
if ! ping -c 1 google.com >/dev/null 2>&1; then
    echo "❌ No internet connection detected"
    exit 1
fi

# Check if Polygon and Alpaca endpoints are reachable
echo "   Checking Polygon API accessibility..."
if ! curl -s --connect-timeout 5 "https://api.polygon.io" >/dev/null; then
    echo "⚠️  Warning: Cannot reach Polygon API"
fi

echo "   Checking Alpaca API accessibility..."
if ! curl -s --connect-timeout 5 "https://paper-api.alpaca.markets" >/dev/null; then
    echo "⚠️  Warning: Cannot reach Alpaca API"
fi

echo "✅ Pre-flight checks completed"
echo ""

# Show configuration
echo "🚀 Starting Simple HFT System..."
echo "   Mode: $([ -n "$TEST_MODE" ] && echo "TEST" || echo "PRODUCTION")"
echo "   Symbols: 25 (SPY, NVDA, QQQ, TSLA, AAPL, etc.)"
echo "   Max Positions: 5"
echo "   Position Size: 2% per trade"
echo "   Take Profit: 0.8%"
echo "   Stop Loss: 1.5%"
if [ "$LOG_TO_FILE" = true ]; then
    echo "   Log File: $LOG_FILE"
fi
echo ""

# Run the system
if [ "$LOG_TO_FILE" = true ]; then
    echo "📝 Starting with logging to $LOG_FILE"
    echo "   Use 'tail -f $LOG_FILE' in another terminal to monitor"
    echo ""
    
    # Run with logging
    ./build_simple/simple_hft $TEST_MODE $DURATION 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "📋 Session completed. Logs saved to: $LOG_FILE"
    
    # Show summary from log
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📊 Session Summary:"
        grep -E "(Messages processed|Signals generated|Orders placed|Orders successful)" "$LOG_FILE" | tail -4
    fi
    
else
    echo "🖥️  Starting without file logging (console only)"
    echo ""
    
    # Run without logging to file
    ./build_simple/simple_hft $TEST_MODE $DURATION
fi

echo ""
echo "✅ Simple HFT System session ended"