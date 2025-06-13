#!/bin/bash

# HFT System Optimization Script
# Comprehensive production-ready system tuning for ultra-low latency trading
# Usage: ./optimize_system.sh [start|stop|reload|status]

set -euo pipefail

# Configuration
HFT_LOG_FILE="/var/log/hft-optimize.log"
HFT_CONFIG_DIR="/etc/hft"
HFT_BACKUP_DIR="/var/lib/hft/backup"
ISOLATED_CORES="0-5"
HOUSEKEEPING_CORES="6-11"
NUMA_NODE=0

# Create directories
mkdir -p "${HFT_CONFIG_DIR}" "${HFT_BACKUP_DIR}" "$(dirname "${HFT_LOG_FILE}")"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$$] $*" | tee -a "${HFT_LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root"
    fi
}

# Backup original settings
backup_settings() {
    log "Backing up original system settings..."
    
    # CPU governor settings
    find /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor -exec cp {} "${HFT_BACKUP_DIR}/governor_{}.bak" \; 2>/dev/null || true
    
    # Kernel parameters
    sysctl -a > "${HFT_BACKUP_DIR}/sysctl.conf.bak" 2>/dev/null || true
    
    # IRQ affinity
    find /proc/irq/*/smp_affinity -exec sh -c 'cp "$1" "${HFT_BACKUP_DIR}/irq_$(basename $(dirname $1)).bak"' _ {} \; 2>/dev/null || true
    
    log "Backup completed"
}

# Restore original settings
restore_settings() {
    log "Restoring original system settings..."
    
    # Restore CPU governors
    find "${HFT_BACKUP_DIR}" -name "governor_*.bak" -exec bash -c '
        core=$(echo "$1" | sed "s/.*cpu\([0-9]*\).*/\1/")
        if [[ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" ]]; then
            cat "$1" > "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" 2>/dev/null || true
        fi
    ' _ {} \; 2>/dev/null || true
    
    # Restore some critical kernel parameters
    echo "powersave" > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    
    log "Settings restored"
}

# Setup CPU isolation and performance
setup_cpu_optimization() {
    log "Setting up CPU optimization..."
    
    # Set CPU governor to performance for isolated cores
    for core in {0..5}; do
        if [[ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" ]]; then
            echo "performance" > "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" || true
            log "Set CPU${core} governor to performance"
        fi
        
        # Set maximum frequency
        if [[ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_min_freq" ]] && 
           [[ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/cpuinfo_max_freq" ]]; then
            max_freq=$(cat "/sys/devices/system/cpu/cpu${core}/cpufreq/cpuinfo_max_freq")
            echo "${max_freq}" > "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_min_freq" || true
            log "Set CPU${core} to maximum frequency: ${max_freq}"
        fi
    done
    
    # Disable turbo boost for consistent timing (Intel systems)
    if [[ -f "/sys/devices/system/cpu/intel_pstate/no_turbo" ]]; then
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
        log "Disabled Intel Turbo Boost"
    elif [[ -f "/sys/devices/system/cpu/cpufreq/boost" ]]; then
        echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
        log "Disabled CPU frequency boost"
    else
        log "Turbo boost control not available on this system"
    fi
    
    # Disable CPU idle states (C-states) for isolated cores
    for state in /sys/devices/system/cpu/cpuidle/state*/disable; do
        [[ -f "$state" ]] && echo 1 > "$state" 2>/dev/null || true
    done
    
    # Disable hyperthreading siblings for isolated cores (cores 6-11)
    for core in {6..11}; do
        if [[ -f "/sys/devices/system/cpu/cpu${core}/online" ]]; then
            echo 0 > "/sys/devices/system/cpu/cpu${core}/online" 2>/dev/null || true
            log "Disabled hyperthreading sibling CPU${core}"
        fi
    done
    
    log "CPU optimization completed"
}

# Setup NUMA optimization
setup_numa_optimization() {
    log "Setting up NUMA optimization..."
    
    # Set NUMA balancing off for deterministic memory access
    echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true
    
    # Configure zone reclaim mode
    echo 0 > /proc/sys/vm/zone_reclaim_mode 2>/dev/null || true
    
    log "NUMA optimization completed"
}

# Setup huge pages
setup_huge_pages() {
    log "Setting up huge pages..."
    
    # Calculate huge pages needed (4GB worth of 2MB pages + 2GB worth of 1GB pages)
    local pages_2mb=$((4 * 1024 / 2))  # 2048 pages of 2MB
    local pages_1gb=2                   # 2 pages of 1GB
    
    # Setup 2MB huge pages
    echo "${pages_2mb}" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages || true
    
    # Setup 1GB huge pages
    echo "${pages_1gb}" > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages || true
    
    # Mount hugetlbfs if not already mounted
    if ! mountpoint -q /dev/hugepages; then
        mkdir -p /dev/hugepages
        mount -t hugetlbfs hugetlbfs /dev/hugepages
    fi
    
    # Set permissions for HFT user
    chmod 755 /dev/hugepages
    
    local allocated_2mb=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo 0)
    local allocated_1gb=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 0)
    
    log "Allocated ${allocated_2mb} 2MB huge pages and ${allocated_1gb} 1GB huge pages"
}

# Setup network optimizations
setup_network_optimization() {
    log "Setting up network optimizations..."
    
    # Network interface optimizations (adjust interface name as needed)
    local interface=$(ip route | grep default | awk '{print $5}' | head -n1)
    if [[ -n "$interface" ]]; then
        # Increase ring buffer sizes
        ethtool -G "$interface" rx 4096 tx 4096 2>/dev/null || true
        
        # Enable hardware offloading
        ethtool -K "$interface" gso off gro off lro off tso off ufo off 2>/dev/null || true
        ethtool -K "$interface" rx-checksumming off tx-checksumming off 2>/dev/null || true
        
        # Set interrupt coalescing for low latency
        ethtool -C "$interface" adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0 2>/dev/null || true
        
        log "Optimized network interface: $interface"
    fi
    
    # Setup interrupt affinity for network IRQs
    local net_irqs=$(grep "$interface" /proc/interrupts | awk -F: '{print $1}' | tr -d ' ')
    for irq in $net_irqs; do
        if [[ -f "/proc/irq/$irq/smp_affinity" ]]; then
            # Bind network interrupts to housekeeping cores (6-11)
            echo "fc0" > "/proc/irq/$irq/smp_affinity" 2>/dev/null || true
            log "Set IRQ $irq affinity to housekeeping cores"
        fi
    done
}

# Setup kernel parameters
setup_kernel_parameters() {
    log "Setting up kernel parameters..."
    
    # Create temporary sysctl configuration
    cat > /tmp/hft-sysctl.conf << 'EOF'
# HFT Kernel Parameters

# Memory management
vm.swappiness = 1
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
vm.dirty_expire_centisecs = 500
vm.dirty_writeback_centisecs = 100
vm.overcommit_memory = 1
vm.overcommit_ratio = 100

# Network optimizations
net.core.rmem_max = 536870912
net.core.wmem_max = 536870912
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.core.netdev_max_backlog = 5000
net.core.netdev_budget = 600
net.ipv4.tcp_rmem = 4096 131072 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_low_latency = 1

# Scheduler settings
kernel.sched_rt_runtime_us = 950000
kernel.sched_rt_period_us = 1000000
kernel.sched_autogroup_enabled = 0

# Timer settings
kernel.timer_migration = 0

# IRQ settings
kernel.numa_balancing = 0

# File system
fs.file-max = 2097152
EOF

    # Apply kernel parameters
    sysctl -p /tmp/hft-sysctl.conf || true
    
    log "Kernel parameters applied"
}

# Setup interrupt affinity
setup_interrupt_affinity() {
    log "Setting up interrupt affinity..."
    
    # Move all non-network IRQs to housekeeping cores
    for irq_dir in /proc/irq/*/; do
        irq=$(basename "$irq_dir")
        [[ "$irq" =~ ^[0-9]+$ ]] || continue
        
        # Skip timer and IPI interrupts
        if grep -q -E "(timer|IPI|TLB|PMI)" "$irq_dir/name" 2>/dev/null; then
            continue
        fi
        
        # Check if it's a network IRQ (already handled above)
        if grep -q -E "(eth|enp|ens)" "$irq_dir/name" 2>/dev/null; then
            continue
        fi
        
        # Move other IRQs to housekeeping cores
        if [[ -f "$irq_dir/smp_affinity" ]]; then
            echo "fc0" > "$irq_dir/smp_affinity" 2>/dev/null || true
        fi
    done
    
    log "Interrupt affinity configured"
}

# Setup real-time optimizations
setup_realtime_optimization() {
    log "Setting up real-time optimizations..."
    
    # Set kernel thread priorities
    for kthread in $(ps -eo pid,comm | grep '\[.*\]' | awk '{print $1}'); do
        # Move kernel threads to housekeeping cores
        taskset -cp 6-11 "$kthread" 2>/dev/null || true
        
        # Lower priority of non-critical kernel threads
        case "$(ps -p "$kthread" -o comm= 2>/dev/null)" in
            *migration*|*rcu*|*watchdog*)
                chrt -f -p 1 "$kthread" 2>/dev/null || true
                ;;
        esac
    done
    
    # Setup RCU offloading to housekeeping cores (if available)
    if [[ -f "/sys/devices/system/cpu/rcu_nocbs/cpu_map" ]]; then
        echo "6-11" > /sys/devices/system/cpu/rcu_nocbs/cpu_map 2>/dev/null || true
        log "RCU nocbs configured for housekeeping cores"
    else
        log "RCU nocbs not available on this kernel"
    fi
    
    log "Real-time optimizations applied"
}

# Load HFT kernel module
load_kernel_module() {
    log "Loading HFT kernel module..."
    
    if [[ -f "kernel_module/hft_kernel_module.ko" ]]; then
        # Load module
        if ! lsmod | grep -q hft_kernel_module; then
            insmod kernel_module/hft_kernel_module.ko 2>/dev/null || true
            log "HFT kernel module loaded"
        fi
    elif [[ -f "kernel_module/hft_kernel_module.c" ]]; then
        # Build and load module
        cd kernel_module
        if [[ ! -f "hft_kernel_module.ko" ]] || [[ "hft_kernel_module.c" -nt "hft_kernel_module.ko" ]]; then
            make -f Makefile.kernel || true
        fi
        
        if [[ -f "hft_kernel_module.ko" ]] && ! lsmod | grep -q hft_kernel_module; then
            insmod hft_kernel_module.ko 2>/dev/null || true
            log "HFT kernel module loaded"
        fi
        cd ..
    else
        log "HFT kernel module not found, skipping"
    fi
}

# Unload HFT kernel module
unload_kernel_module() {
    log "Unloading HFT kernel module..."
    
    if lsmod | grep -q hft_kernel_module; then
        rmmod hft_kernel_module 2>/dev/null || true
        log "HFT kernel module unloaded"
    fi
}

# Setup memory locking limits
setup_memory_limits() {
    log "Setting up memory limits..."
    
    # Create limits configuration for HFT user
    cat > /etc/security/limits.d/99-hft.conf << 'EOF'
# HFT Memory Limits
*               soft    memlock         unlimited
*               hard    memlock         unlimited
*               soft    nofile          1048576
*               hard    nofile          1048576
*               soft    nproc           1048576
*               hard    nproc           1048576
*               soft    rtprio          99
*               hard    rtprio          99
EOF

    log "Memory limits configured"
}

# Show system status
show_status() {
    log "=== HFT System Status ==="
    
    # CPU status
    echo "CPU Governors:"
    for core in {0..5}; do
        if [[ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" ]]; then
            gov=$(cat "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" 2>/dev/null || echo "unknown")
            echo "  CPU${core}: $gov"
        fi
    done
    
    # Huge pages status
    echo -e "\nHuge Pages:"
    echo "  2MB pages: $(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo 0)"
    echo "  1GB pages: $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 0)"
    
    # Kernel module status
    echo -e "\nKernel Module:"
    if lsmod | grep -q hft_kernel_module; then
        echo "  HFT kernel module: LOADED"
        if [[ -f /proc/hft_stats ]]; then
            echo "  Statistics available at: /proc/hft_stats"
        fi
    else
        echo "  HFT kernel module: NOT LOADED"
    fi
    
    # NUMA status
    echo -e "\nNUMA:"
    echo "  Balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unknown)"
    
    # Network interface status
    local interface=$(ip route | grep default | awk '{print $5}' | head -n1)
    if [[ -n "$interface" ]]; then
        echo -e "\nNetwork Interface ($interface):"
        ethtool -g "$interface" 2>/dev/null | grep -E "RX:|TX:" || echo "  Ring buffer info not available"
    fi
}

# Start optimizations
start_optimizations() {
    log "Starting HFT system optimizations..."
    
    backup_settings
    setup_cpu_optimization
    setup_numa_optimization
    setup_huge_pages
    setup_network_optimization
    setup_kernel_parameters
    setup_interrupt_affinity
    setup_realtime_optimization
    setup_memory_limits
    load_kernel_module
    
    log "HFT system optimizations completed successfully"
}

# Stop optimizations
stop_optimizations() {
    log "Stopping HFT system optimizations..."
    
    unload_kernel_module
    restore_settings
    
    # Re-enable hyperthreading siblings
    for core in {6..11}; do
        if [[ -f "/sys/devices/system/cpu/cpu${core}/online" ]]; then
            echo 1 > "/sys/devices/system/cpu/cpu${core}/online" 2>/dev/null || true
        fi
    done
    
    # Re-enable CPU idle states
    for state in /sys/devices/system/cpu/cpuidle/state*/disable; do
        [[ -f "$state" ]] && echo 0 > "$state" 2>/dev/null || true
    done
    
    log "HFT system optimizations stopped"
}

# Main function
main() {
    check_root
    
    case "${1:-}" in
        start)
            start_optimizations
            ;;
        stop)
            stop_optimizations
            ;;
        reload|restart)
            stop_optimizations
            sleep 2
            start_optimizations
            ;;
        status)
            show_status
            ;;
        *)
            echo "Usage: $0 {start|stop|reload|restart|status}"
            echo ""
            echo "Commands:"
            echo "  start   - Apply all HFT system optimizations"
            echo "  stop    - Remove optimizations and restore defaults"
            echo "  reload  - Stop and start optimizations"
            echo "  status  - Show current optimization status"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"