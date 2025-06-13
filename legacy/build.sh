#!/bin/bash

# Ultra-Fast HFT Build Script
# Optimized for A100 x86 systems with sub-200μs target latency

set -e

echo "=== Ultra-Fast HFT Build System ==="
echo "Target: Sub-200μs end-to-end pipeline"
echo "===================================="

# Parse command line arguments
BUILD_TYPE="Release"
NUMA_NODE=""
ENABLE_PGO=false
CLEAN_BUILD=false
RUN_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --numa-node)
            NUMA_NODE="$2"
            shift 2
            ;;
        --pgo)
            ENABLE_PGO=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug         Build in debug mode"
            echo "  --numa-node N   Optimize for NUMA node N"
            echo "  --pgo           Enable Profile-Guided Optimization"
            echo "  --clean         Clean build directory"
            echo "  --test          Run performance tests after build"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# System checks
echo "Checking system requirements..."

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.20+"
    exit 1
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "Error: No C++ compiler found. Please install GCC or Clang"
    exit 1
fi

# Check CPU features
echo "Detecting CPU features..."
if grep -q avx512f /proc/cpuinfo; then
    echo "✓ AVX-512 support detected"
else
    echo "⚠ AVX-512 not available - using AVX2"
fi

if grep -q numa /proc/cpuinfo; then
    echo "✓ NUMA support detected"
else
    echo "⚠ NUMA not detected"
fi

# Check for NUMA libraries
if ldconfig -p | grep -q libnuma; then
    echo "✓ NUMA libraries found"
else
    echo "⚠ NUMA libraries not found - install libnuma-dev for optimal performance"
fi

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring build ($BUILD_TYPE mode)..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

if [ "$ENABLE_PGO" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PGO=ON"
fi

# Use Ninja if available for faster builds
if command -v ninja &> /dev/null; then
    CMAKE_ARGS="$CMAKE_ARGS -GNinja"
    echo "Using Ninja build system"
fi

cmake .. $CMAKE_ARGS

# Build
echo "Building HFT engine..."
if command -v ninja &> /dev/null && [ -f build.ninja ]; then
    ninja -j$(nproc)
else
    make -j$(nproc)
fi

echo "✓ Build completed successfully!"

# Set optimal system configuration
echo "Applying system optimizations..."

# Set CPU governor to performance mode
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "Setting CPU governor to performance mode..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo tee $cpu > /dev/null 2>&1 || true
    done
fi

# Disable CPU frequency scaling
if [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "Disabling CPU turbo boost for consistent latency..."
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1 || true
fi

# Set process priority limits
echo "Configuring real-time scheduling limits..."
if [ -w /etc/security/limits.conf ]; then
    sudo bash -c 'cat >> /etc/security/limits.conf << EOF
# HFT real-time scheduling limits
* soft rtprio 99
* hard rtprio 99
* soft memlock unlimited
* hard memlock unlimited
EOF' || true
fi

# Configure huge pages
echo "Configuring huge pages..."
if [ -w /proc/sys/vm/nr_hugepages ]; then
    echo 1024 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1 || true
fi

# Run performance tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "Running performance tests..."
    
    if [ -n "$NUMA_NODE" ]; then
        echo "Testing with NUMA node $NUMA_NODE..."
        ./hft_engine --test --numa-node $NUMA_NODE
    else
        echo "Testing without NUMA optimization..."
        ./hft_engine --test
    fi
fi

echo ""
echo "=== Build Summary ==="
echo "Build type: $BUILD_TYPE"
echo "Executable: $(pwd)/hft_engine"
echo "Target latency: <200μs end-to-end"

if [ -n "$NUMA_NODE" ]; then
    echo "NUMA node: $NUMA_NODE"
fi

echo ""
echo "=== Usage ==="
echo "Run HFT engine:"
echo "  ./hft_engine"
echo ""
echo "Run with NUMA optimization:"
echo "  ./hft_engine --numa-node 0"
echo ""
echo "Run performance test:"
echo "  ./hft_engine --test"
echo ""
echo "=== Performance Tips ==="
echo "1. Run on isolated CPU cores for best latency"
echo "2. Use NUMA node binding for memory locality"
echo "3. Disable CPU frequency scaling"
echo "4. Enable huge pages for large allocations"
echo "5. Run with real-time scheduling priority"
echo ""
echo "Build completed successfully! 🚀"