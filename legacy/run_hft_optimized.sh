#!/bin/bash

# Production HFT Engine Runner with Full System Optimization
# This script initializes the system for ultra-low latency trading and runs the HFT engine
# Usage: ./run_hft_optimized.sh [options]

set -euo pipefail

# Configuration
HFT_ENGINE="./build/hft_engine"
OPTIMIZE_SCRIPT="./optimize_system.sh"
LOG_DIR="/var/log/hft"
PID_FILE="/var/run/hft_engine.pid"
NUMA_NODE=0
ENGINE_ARGS=""
DRY_RUN=false
VERBOSE=false

# Create log directory
mkdir -p "${LOG_DIR}" || sudo mkdir -p "${LOG_DIR}" 2>/dev/null || true

# Logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "${LOG_DIR}/hft_engine.log" 2>/dev/null || echo "[$timestamp] $*"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production HFT Engine Runner with System Optimization

OPTIONS:
    -n, --numa-node NODE     NUMA node to bind to (default: 0)
    -d, --dry-run           Show what would be done without executing
    -v, --verbose           Enable verbose output
    -t, --test              Run in test mode
    -b, --benchmark TYPE    Run specific benchmark (pipeline_latency, memory_allocation, model_inference, websocket_throughput)
    --duration SECS         Test duration in seconds (default: 60)
    --no-optimize           Skip system optimization
    --no-kernel-module      Skip kernel module loading
    --args "ARGS"           Additional arguments to pass to HFT engine
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run with default settings
    $0 -n 1 -v              # Run on NUMA node 1 with verbose output
    $0 -t --duration 30     # Run test for 30 seconds
    $0 -b pipeline_latency  # Run pipeline latency benchmark
    $0 --dry-run            # Show what would be executed

REQUIREMENTS:
    - Root privileges for system optimization
    - HFT engine binary in current directory
    - optimize_system.sh script in current directory

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--numa-node)
                NUMA_NODE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -t|--test)
                ENGINE_ARGS="$ENGINE_ARGS --test"
                shift
                ;;
            -b|--benchmark)
                ENGINE_ARGS="$ENGINE_ARGS --test --benchmark $2"
                shift 2
                ;;
            --duration)
                ENGINE_ARGS="$ENGINE_ARGS --duration $2"
                shift 2
                ;;
            --no-optimize)
                SKIP_OPTIMIZE=true
                shift
                ;;
            --no-kernel-module)
                SKIP_KERNEL_MODULE=true
                shift
                ;;
            --args)
                ENGINE_ARGS="$ENGINE_ARGS $2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if HFT engine binary exists
    if [[ ! -f "$HFT_ENGINE" ]]; then
        log "ERROR: HFT engine binary not found at $HFT_ENGINE"
        log "Please build the project first: cmake --build . --target hft_engine"
        exit 1
    fi
    
    # Check if engine is executable
    if [[ ! -x "$HFT_ENGINE" ]]; then
        log "ERROR: HFT engine binary is not executable"
        chmod +x "$HFT_ENGINE" 2>/dev/null || {
            log "ERROR: Cannot make HFT engine executable"
            exit 1
        }
    fi
    
    # Check if optimization script exists (optional)
    if [[ ! -f "$OPTIMIZE_SCRIPT" ]] && [[ "${SKIP_OPTIMIZE:-false}" != "true" ]]; then
        log "WARNING: System optimization script not found at $OPTIMIZE_SCRIPT"
        log "System optimizations will be skipped"
        SKIP_OPTIMIZE=true
    fi
    
    # Check for root privileges if optimization is needed
    if [[ "${SKIP_OPTIMIZE:-false}" != "true" ]] && [[ $EUID -ne 0 ]]; then
        log "WARNING: Root privileges required for system optimization"
        log "Running without system optimization (use sudo for full optimization)"
        SKIP_OPTIMIZE=true
    fi
    
    log "Prerequisites check completed"
}

# Initialize system optimizations
init_system_optimization() {
    if [[ "${SKIP_OPTIMIZE:-false}" == "true" ]]; then
        log "Skipping system optimization"
        return 0
    fi
    
    log "Initializing system optimizations..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would execute: sudo $OPTIMIZE_SCRIPT start"
        return 0
    fi
    
    if [[ -x "$OPTIMIZE_SCRIPT" ]]; then
        if sudo "$OPTIMIZE_SCRIPT" start; then
            log "✅ System optimization completed successfully"
        else
            log "⚠️ System optimization failed, continuing anyway"
        fi
    else
        log "⚠️ Optimization script not executable, skipping"
    fi
}

# Load kernel module
load_kernel_module() {
    if [[ "${SKIP_KERNEL_MODULE:-false}" == "true" ]]; then
        log "Skipping kernel module loading"
        return 0
    fi
    
    log "Loading HFT kernel module..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would load kernel module if available"
        return 0
    fi
    
    # Check if module exists
    if [[ -f "kernel_module/hft_kernel_module.ko" ]]; then
        if sudo insmod kernel_module/hft_kernel_module.ko 2>/dev/null; then
            log "✅ HFT kernel module loaded successfully"
        else
            log "⚠️ Failed to load HFT kernel module, continuing anyway"
        fi
    elif lsmod | grep -q hft_kernel_module; then
        log "✅ HFT kernel module already loaded"
    else
        log "ℹ️ HFT kernel module not available"
    fi
}

# Set up NUMA optimization
setup_numa() {
    log "Setting up NUMA optimization for node $NUMA_NODE..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would bind to NUMA node $NUMA_NODE"
        return 0
    fi
    
    # Check if NUMA is available
    if command -v numactl >/dev/null 2>&1; then
        log "✅ NUMA control available"
    else
        log "⚠️ numactl not found, NUMA optimization disabled"
        log "Install with: sudo apt-get install numactl"
        return 0
    fi
    
    # Validate NUMA node
    local max_node=$(numactl --hardware | grep "available:" | awk '{print $2}' | head -n1)
    max_node=$((max_node - 1))
    
    if [[ $NUMA_NODE -gt $max_node ]]; then
        log "⚠️ NUMA node $NUMA_NODE not available (max: $max_node), using node 0"
        NUMA_NODE=0
    fi
    
    log "✅ NUMA optimization configured for node $NUMA_NODE"
}

# Run the HFT engine
run_hft_engine() {
    log "Starting HFT engine..."
    log "Command: $HFT_ENGINE --numa-node $NUMA_NODE $ENGINE_ARGS"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would execute HFT engine with NUMA binding"
        return 0
    fi
    
    # Create PID file
    echo $$ > "$PID_FILE" 2>/dev/null || true
    
    # Set up signal handlers for graceful shutdown
    trap 'cleanup_and_exit' SIGTERM SIGINT
    
    # Run with NUMA binding if available
    if command -v numactl >/dev/null 2>&1; then
        log "Running with NUMA node $NUMA_NODE binding..."
        if [[ "$VERBOSE" == "true" ]]; then
            numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$HFT_ENGINE" --numa-node "$NUMA_NODE" $ENGINE_ARGS 2>&1
        else
            numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$HFT_ENGINE" --numa-node "$NUMA_NODE" $ENGINE_ARGS 2>&1 | tee -a "${LOG_DIR}/hft_engine.log"
        fi
    else
        log "Running without NUMA binding..."
        if [[ "$VERBOSE" == "true" ]]; then
            "$HFT_ENGINE" --numa-node "$NUMA_NODE" $ENGINE_ARGS 2>&1
        else
            "$HFT_ENGINE" --numa-node "$NUMA_NODE" $ENGINE_ARGS 2>&1 | tee -a "${LOG_DIR}/hft_engine.log"
        fi
    fi
    
    local exit_code=$?
    log "HFT engine exited with code $exit_code"
    return $exit_code
}

# Cleanup function
cleanup_and_exit() {
    log "Shutting down HFT engine..."
    
    # Remove PID file
    rm -f "$PID_FILE" 2>/dev/null || true
    
    # Stop system optimizations if we started them
    if [[ "${SKIP_OPTIMIZE:-false}" != "true" ]] && [[ -x "$OPTIMIZE_SCRIPT" ]] && [[ "$DRY_RUN" != "true" ]]; then
        log "Cleaning up system optimizations..."
        sudo "$OPTIMIZE_SCRIPT" stop 2>/dev/null || true
    fi
    
    # Unload kernel module if we loaded it
    if [[ "${SKIP_KERNEL_MODULE:-false}" != "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        if lsmod | grep -q hft_kernel_module; then
            log "Unloading HFT kernel module..."
            sudo rmmod hft_kernel_module 2>/dev/null || true
        fi
    fi
    
    log "Cleanup completed"
    exit 0
}

# Show system status
show_status() {
    log "=== HFT System Status ==="
    
    # Engine status
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log "HFT Engine: RUNNING (PID: $(cat "$PID_FILE"))"
    else
        log "HFT Engine: STOPPED"
    fi
    
    # System optimization status
    if [[ -x "$OPTIMIZE_SCRIPT" ]]; then
        "$OPTIMIZE_SCRIPT" status 2>/dev/null || log "System optimization status: UNKNOWN"
    else
        log "System optimization: NOT AVAILABLE"
    fi
    
    # Kernel module status
    if lsmod | grep -q hft_kernel_module; then
        log "Kernel module: LOADED"
        if [[ -f /proc/hft_stats ]]; then
            log "Kernel stats: AVAILABLE at /proc/hft_stats"
        fi
    else
        log "Kernel module: NOT LOADED"
    fi
    
    # NUMA status
    if command -v numactl >/dev/null 2>&1; then
        log "NUMA: AVAILABLE"
        log "NUMA topology: $(numactl --hardware | grep available)"
    else
        log "NUMA: NOT AVAILABLE"
    fi
    
    log "=========================="
}

# Main execution function
main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Show banner
    log "🚀 HFT Production Engine Runner v1.0"
    log "Target: Sub-200μs pipeline latency"
    log "NUMA Node: $NUMA_NODE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "🔍 DRY RUN MODE - No changes will be made"
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Show current status if verbose
    if [[ "$VERBOSE" == "true" ]]; then
        show_status
    fi
    
    # Initialize system
    init_system_optimization
    load_kernel_module
    setup_numa
    
    # Run the engine
    log "🎯 Starting ultra-low latency HFT engine..."
    run_hft_engine
}

# Execute main function with all arguments
main "$@"
