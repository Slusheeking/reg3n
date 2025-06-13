we#!/bin/bash

echo "Building Simple HFT System..."

# Check if required libraries are available
echo "Checking dependencies..."

# Check for required development packages
MISSING_DEPS=""

# Check for libcurl
if ! pkg-config --exists libcurl; then
    echo "❌ libcurl-dev not found"
    MISSING_DEPS="$MISSING_DEPS libcurl4-openssl-dev"
fi

# Check for rapidjson
if ! find /usr/include -name "rapidjson" -type d 2>/dev/null | grep -q rapidjson; then
    echo "❌ rapidjson not found"
    MISSING_DEPS="$MISSING_DEPS rapidjson-dev"
fi

# Check for websocketpp
if ! find /usr/include -name "websocketpp" -type d 2>/dev/null | grep -q websocketpp; then
    echo "❌ websocketpp not found"
    MISSING_DEPS="$MISSING_DEPS libwebsocketpp-dev"
fi

# Check for OpenSSL
if ! pkg-config --exists openssl; then
    echo "❌ openssl not found"
    MISSING_DEPS="$MISSING_DEPS libssl-dev"
fi

# Check for Boost
if ! find /usr/include -name "boost" -type d 2>/dev/null | grep -q boost; then
    echo "❌ boost not found"
    MISSING_DEPS="$MISSING_DEPS libboost-all-dev"
fi

if [ ! -z "$MISSING_DEPS" ]; then
    echo ""
    echo "Missing dependencies. Please install them with:"
    echo "sudo apt-get update"
    echo "sudo apt-get install $MISSING_DEPS"
    echo ""
    exit 1
fi

echo "✅ All dependencies found"

# Create build directory
mkdir -p build_simple
cd build_simple

# Compile the simple HFT system
echo "Compiling simple_hft..."

g++ -std=c++17 -O3 -DNDEBUG \
    -I/usr/include/rapidjson \
    -I/usr/include/websocketpp \
    -I/usr/include/boost \
    -I.. \
    ../simple_main.cpp \
    -lcurl \
    -lssl \
    -lcrypto \
    -lpthread \
    -lboost_system \
    -lboost_thread \
    -o simple_hft

if [ $? -eq 0 ]; then
    echo "✅ Simple HFT system compiled successfully!"
    echo ""
    echo "Executable location: build_simple/simple_hft"
    echo ""
    echo "Usage:"
    echo "  ./build_simple/simple_hft           # Run in production mode"
    echo "  ./build_simple/simple_hft --test    # Run in test mode (60 seconds)"
    echo "  ./build_simple/simple_hft --test --duration 120  # Test for 120 seconds"
    echo ""
    echo "To run with logging:"
    echo "  ./build_simple/simple_hft 2>&1 | tee simple_hft_run.log"
    echo ""
else
    echo "❌ Compilation failed!"
    exit 1
fi