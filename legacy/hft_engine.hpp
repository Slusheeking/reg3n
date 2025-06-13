#pragma once

// Production-optimized HFT Engine with dual WebSocket feeds, advanced online learning, and 95% capital allocation

#include "config.hpp"
#include "memory_manager.hpp"
#include "market_data.hpp"
#include "hft_models.hpp"
#include "order_engine.hpp"
#include "alpaca_client.hpp"
#include "websocket_client.hpp"
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <random>
#include <cstdio>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <iomanip>   // for std::fixed, std::setprecision
#include <cmath> // For std::abs, std::sqrt
#include <algorithm> // For std::min, std::max
#include <fstream>   // for safer file operations
#include <cstring>   // for memset
#include <sstream>   // for stringstream

// Platform-specific includes with guards
#ifdef __linux__
    #include <pthread.h>     // for pthread functions
    #include <sched.h>       // for SCHED_FIFO, sched_param
    #include <unistd.h>      // for sysconf
#endif

#include <cstdlib>      // for system()

namespace hft {

// Platform abstraction for CPU management
class PlatformCPUManager {
public:
    virtual ~PlatformCPUManager() = default;
    virtual bool set_cpu_affinity(std::thread& thread, int core) = 0;
    virtual bool set_realtime_priority(std::thread& thread, int priority) = 0;
    virtual bool isolate_cpu_cores(const std::vector<int>& cores) = 0;
    virtual int get_cpu_count() const = 0;
    virtual bool is_core_valid(int core) const = 0;
};

// Cross-platform CPU utilities
class CPUUtilities {
public:
    static int get_cpu_count() {
#ifdef __linux__
        return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#else
        return std::thread::hardware_concurrency();
#endif
    }
    
    static bool set_cpu_governor(int core, const std::string& governor) {
#ifdef __linux__
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_governor";
        
        // Check if cpufreq interface exists first
        std::ifstream check_file(path);
        if (!check_file.good()) {
            // cpufreq interface not available (common in VMs)
            return false;
        }
        check_file.close();
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << governor;
            return file.good();
        }
#endif
        return false;
    }
    
    static bool disable_cpu_idle_state(int state) {
#ifdef __linux__
        std::string path = "/sys/devices/system/cpu/cpuidle/state" + std::to_string(state) + "/disable";
        
        // Check if cpuidle interface exists first
        std::ifstream check_file(path);
        if (!check_file.good()) {
            // cpuidle interface not available (common in VMs)
            return false;
        }
        check_file.close();
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << "1";
            return file.good();
        }
#endif
        return false;
    }
    
    static bool set_no_turbo(bool disable) {
#ifdef __linux__
        std::string path = "/sys/devices/system/cpu/intel_pstate/no_turbo";
        
        // Check if intel_pstate interface exists first
        std::ifstream check_file(path);
        if (!check_file.good()) {
            // intel_pstate interface not available (common in VMs or AMD systems)
            return false;
        }
        check_file.close();
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << (disable ? "1" : "0");
            return file.good();
        }
#endif
        return false;
    }
};

// Production CPU manager for core isolation and real-time performance
class CPUManager : public PlatformCPUManager {
private:
    static constexpr int TRADING_CORE = 0;      // Core 0 for main trading loop
    static constexpr int LEARNING_CORE = 1;     // Core 1 for learning
    static constexpr int REGIME_CORE = 2;       // Core 2 for regime detection
    static constexpr int RISK_CORE = 3;         // Core 3 for risk monitoring
    static constexpr int WEBSOCKET_CORE = 4;    // Core 4 for WebSocket processing
    static constexpr int ORDER_CORE = 5;        // Core 5 for order execution
    
    std::vector<int> isolated_cores_;
    bool isolation_enabled_;
    int total_cpu_count_;

public:
    CPUManager() : isolation_enabled_(false) {
        total_cpu_count_ = CPUUtilities::get_cpu_count();
        
        // Validate and build isolated cores list
        std::vector<int> candidate_cores = {TRADING_CORE, LEARNING_CORE, REGIME_CORE, RISK_CORE, WEBSOCKET_CORE, ORDER_CORE};
        for (int core : candidate_cores) {
            if (is_core_valid(core)) {
                isolated_cores_.push_back(core);
            }
        }
    }
    
    bool is_core_valid(int core) const override {
        return core >= 0 && core < total_cpu_count_;
    }
    
    int get_cpu_count() const override {
        return total_cpu_count_;
    }
    
    // Initialize CPU isolation and performance settings
    bool initialize_cpu_isolation() {
        std::cout << "🖥️ Initializing CPU isolation and performance settings..." << std::endl;
        std::cout << "   Detected " << total_cpu_count_ << " CPU cores" << std::endl;
        
        if (isolated_cores_.empty()) {
            std::cerr << "❌ No valid CPU cores available for isolation" << std::endl;
            return false;
        }
        
        try {
#ifdef __linux__
            bool cpu_optimization_available = false;
            
            // Set CPU governor to performance mode for isolated cores
            int governor_successes = 0;
            for (int core : isolated_cores_) {
                if (CPUUtilities::set_cpu_governor(core, "performance")) {
                    governor_successes++;
                    cpu_optimization_available = true;
                }
            }
            if (governor_successes > 0) {
                std::cout << "✅ Set performance governor for " << governor_successes << " cores" << std::endl;
            } else {
                std::cout << "ℹ️ CPU frequency scaling not available (common in VMs)" << std::endl;
            }
            
            // Disable CPU idle states for isolated cores (C-states)
            int idle_state_successes = 0;
            for (int state = 1; state <= 3; ++state) {
                if (CPUUtilities::disable_cpu_idle_state(state)) {
                    idle_state_successes++;
                    cpu_optimization_available = true;
                }
            }
            if (idle_state_successes > 0) {
                std::cout << "✅ Disabled " << idle_state_successes << " CPU idle states" << std::endl;
            } else {
                std::cout << "ℹ️ CPU idle state control not available (common in VMs)" << std::endl;
            }
            
            // Disable turbo boost for consistent timing
            if (CPUUtilities::set_no_turbo(true)) {
                std::cout << "✅ Turbo boost disabled for consistent timing" << std::endl;
                cpu_optimization_available = true;
            } else {
                std::cout << "ℹ️ Turbo boost control not available (common in VMs or AMD systems)" << std::endl;
            }
            
            if (!cpu_optimization_available) {
                std::cout << "ℹ️ Hardware CPU optimizations not available - relying on thread pinning only" << std::endl;
            }
            
            // Set kernel thread affinity away from isolated cores
            set_kernel_thread_affinity();
            
            isolation_enabled_ = true;
            std::cout << "✅ CPU isolation initialized successfully" << std::endl;
            return true;
#else
            std::cout << "⚠️ CPU isolation not supported on this platform" << std::endl;
            return false;
#endif
            
        } catch (const std::exception& e) {
            std::cerr << "❌ CPU isolation failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Pin thread to specific core with real-time priority
    bool set_cpu_affinity(std::thread& thread, int core) override {
        return pin_thread_to_core(thread, core, 99, SCHED_FIFO);
    }
    
    bool set_realtime_priority(std::thread& thread, int priority) override {
#ifdef __linux__
        try {
            struct sched_param param;
            param.sched_priority = priority;
            int result = pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &param);
            return result == 0;
        } catch (const std::exception&) {
            return false;
        }
#else
        return false;
#endif
    }
    
    bool pin_thread_to_core(std::thread& thread, int core, int priority = 99, int policy = SCHED_FIFO) {
        // Validate core first
        if (!is_core_valid(core)) {
            std::cerr << "❌ Invalid core " << core << " (max: " << (total_cpu_count_ - 1) << ")" << std::endl;
            return false;
        }
        
#ifdef __linux__
        try {
            // Set CPU affinity (this usually works for regular users)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core, &cpuset);
            
            int result = pthread_setaffinity_np(thread.native_handle(), sizeof(cpuset), &cpuset);
            if (result != 0) {
                std::cerr << "❌ Failed to set CPU affinity for core " << core << std::endl;
                return false;
            }
            
            // Try to set real-time scheduling priority (graceful failure)
            struct sched_param param;
            param.sched_priority = priority;
            result = pthread_setschedparam(thread.native_handle(), policy, &param);
            if (result != 0) {
                std::cout << "⚠️ Real-time priority " << priority << " for core " << core
                          << " not available (requires elevated privileges)" << std::endl;
                std::cout << "✅ Thread pinned to core " << core << " with normal priority" << std::endl;
            } else {
                std::cout << "✅ Thread pinned to core " << core << " with real-time priority " << priority << std::endl;
            }
            
            return true; // Success even if priority setting failed
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Thread pinning failed: " << e.what() << std::endl;
            return false;
        }
#else
        std::cerr << "⚠️ Thread pinning not supported on this platform" << std::endl;
        return false;
#endif
    }
    
    // Get optimal core for specific thread type
    int get_core_for_thread_type(const std::string& thread_type) const {
        if (thread_type == "trading") return TRADING_CORE;
        if (thread_type == "learning") return LEARNING_CORE;
        if (thread_type == "regime") return REGIME_CORE;
        if (thread_type == "risk") return RISK_CORE;
        if (thread_type == "websocket") return WEBSOCKET_CORE;
        if (thread_type == "order") return ORDER_CORE;
        return -1; // Invalid thread type
    }
    
    // Restore CPU settings on shutdown
    void cleanup_cpu_settings() {
        if (!isolation_enabled_) return;
        
        std::cout << "🖥️ Restoring CPU settings..." << std::endl;
        
#ifdef __linux__
        // Restore CPU governor to powersave
        for (int core : isolated_cores_) {
            CPUUtilities::set_cpu_governor(core, "powersave");
        }
        
        // Re-enable CPU idle states
        for (int state = 1; state <= 3; ++state) {
            std::string path = "/sys/devices/system/cpu/cpuidle/state" + std::to_string(state) + "/disable";
            std::ofstream file(path);
            if (file.is_open()) {
                file << "0";
            }
        }
        
        // Re-enable turbo boost
        CPUUtilities::set_no_turbo(false);
#endif
        
        isolation_enabled_ = false;
        std::cout << "✅ CPU settings restored" << std::endl;
    }
    
    bool isolate_cpu_cores(const std::vector<int>& cores) override {
        // Implementation for isolating specific cores
        return initialize_cpu_isolation();
    }
    
    ~CPUManager() {
        cleanup_cpu_settings();
    }

private:
    void disable_hyperthreading_siblings() {
#ifdef __linux__
        // Disable hyperthreading siblings for isolated cores
        std::vector<int> siblings_to_disable;
        
        for (int core : isolated_cores_) {
            // Assume hyperthreading siblings are offset by number of physical cores
            int sibling = core + (total_cpu_count_ / 2);
            if (sibling < total_cpu_count_) {
                siblings_to_disable.push_back(sibling);
            }
        }
        
        for (int sibling : siblings_to_disable) {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(sibling) + "/online";
            std::ofstream file(path);
            if (file.is_open()) {
                file << "0";
            }
        }
#endif
    }
    
    void set_kernel_thread_affinity() {
#ifdef __linux__
        std::cout << "🔧 Configuring kernel thread affinity for HFT isolation..." << std::endl;
        
        // Calculate non-isolated cores mask (cores not used for HFT)
        // Assuming cores 0-5 are isolated for HFT, use remaining cores for kernel threads
        std::string non_isolated_mask = calculate_non_isolated_cores_mask();
        
        try {
            // 1. Set default IRQ affinity to non-isolated cores
            if (set_default_irq_affinity(non_isolated_mask)) {
                std::cout << "✅ Default IRQ affinity set to non-isolated cores" << std::endl;
            } else {
                std::cerr << "⚠️ Failed to set default IRQ affinity" << std::endl;
            }
            
            // 2. Move individual IRQs away from isolated cores
            move_individual_irqs(non_isolated_mask);
            
            // 3. Move kernel worker threads
            move_kernel_worker_threads(non_isolated_mask);
            
            // 4. Set RCU thread affinity if possible
            move_rcu_threads(non_isolated_mask);
            
            // 5. Move migration threads
            move_migration_threads(non_isolated_mask);
            
            std::cout << "✅ Kernel thread affinity configuration completed" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Kernel thread affinity configuration failed: " << e.what() << std::endl;
        }
        
        // Note: For complete isolation, also use kernel command line parameters:
        // isolcpus=0-5 nohz_full=0-5 rcu_nocbs=0-5 intel_pstate=disable
        std::cout << "💡 For complete isolation, add to kernel command line:" << std::endl;
        std::cout << "   isolcpus=0-5 nohz_full=0-5 rcu_nocbs=0-5 intel_pstate=disable" << std::endl;
#endif
    }
    
private:
    std::string calculate_non_isolated_cores_mask() const {
        // HFT cores: 0-5, so use cores 6+ for kernel threads
        // Create a bitmask for cores 6 and above
        int total_cores = total_cpu_count_;
        if (total_cores <= 6) {
            // If we have 6 or fewer cores, use core 0 for kernel threads
            return "1"; // Binary: 000001 (core 0)
        }
        
        // Calculate mask for cores 6+
        // For 8 cores: cores 6,7 = binary 11000000 = hex C0
        // For 16 cores: cores 6-15 = binary 1111111111000000 = hex FFC0
        uint64_t mask = 0;
        for (int core = 6; core < total_cores && core < 64; ++core) {
            mask |= (1ULL << core);
        }
        
        // Convert to hex string
        std::stringstream ss;
        ss << std::hex << mask;
        return ss.str();
    }
    
    bool set_default_irq_affinity(const std::string& mask) {
        std::ofstream irq_file("/proc/irq/default_smp_affinity");
        if (irq_file.is_open()) {
            irq_file << mask;
            return irq_file.good();
        }
        return false;
    }
    
    void move_individual_irqs(const std::string& mask) {
        // Move individual IRQs away from isolated cores
        std::cout << "🔧 Moving individual IRQs to non-isolated cores..." << std::endl;
        
        // Read /proc/interrupts to find active IRQs
        std::ifstream interrupts_file("/proc/interrupts");
        if (!interrupts_file.is_open()) {
            std::cerr << "⚠️ Cannot read /proc/interrupts" << std::endl;
            return;
        }
        
        std::string line;
        int moved_irqs = 0;
        
        while (std::getline(interrupts_file, line)) {
            // Parse IRQ number from the beginning of each line
            std::istringstream iss(line);
            std::string irq_str;
            iss >> irq_str;
            
            // Check if this is a valid IRQ line (starts with a number)
            if (!irq_str.empty() && std::isdigit(irq_str[0])) {
                // Remove the colon if present
                if (irq_str.back() == ':') {
                    irq_str.pop_back();
                }
                
                try {
                    int irq_num = std::stoi(irq_str);
                    
                    // Try to set affinity for this IRQ
                    std::string irq_affinity_path = "/proc/irq/" + std::to_string(irq_num) + "/smp_affinity";
                    std::ofstream irq_affinity_file(irq_affinity_path);
                    
                    if (irq_affinity_file.is_open()) {
                        irq_affinity_file << mask;
                        if (irq_affinity_file.good()) {
                            moved_irqs++;
                        }
                    }
                } catch (const std::exception&) {
                    // Skip invalid IRQ numbers
                    continue;
                }
            }
        }
        
        std::cout << "✅ Moved " << moved_irqs << " IRQs to non-isolated cores" << std::endl;
    }
    
    void move_kernel_worker_threads(const std::string& mask) {
        std::cout << "🔧 Moving kernel worker threads..." << std::endl;
        
        // List of kernel worker thread patterns to move
        std::vector<std::string> worker_patterns = {
            "kworker/",
            "ksoftirqd/",
            "migration/",
            "rcu_",
            "watchdog/"
        };
        
        int moved_threads = 0;
        
        for (const auto& pattern : worker_patterns) {
            moved_threads += move_threads_by_pattern(pattern, mask);
        }
        
        std::cout << "✅ Moved " << moved_threads << " kernel worker threads" << std::endl;
    }
    
    void move_rcu_threads(const std::string& mask) {
        std::cout << "🔧 Moving RCU threads..." << std::endl;
        
        int moved_rcu = 0;
        moved_rcu += move_threads_by_pattern("rcu_gp", mask);
        moved_rcu += move_threads_by_pattern("rcu_par_gp", mask);
        moved_rcu += move_threads_by_pattern("rcu_preempt", mask);
        moved_rcu += move_threads_by_pattern("rcu_sched", mask);
        moved_rcu += move_threads_by_pattern("rcu_bh", mask);
        
        std::cout << "✅ Moved " << moved_rcu << " RCU threads" << std::endl;
    }
    
    void move_migration_threads(const std::string& mask) {
        std::cout << "🔧 Moving migration threads..." << std::endl;
        
        int moved_migration = move_threads_by_pattern("migration/", mask);
        
        std::cout << "✅ Moved " << moved_migration << " migration threads" << std::endl;
    }
    
    int move_threads_by_pattern(const std::string& pattern, const std::string& mask) {
        int moved_count = 0;
        
        // Use pgrep to find processes matching the pattern
        std::string command = "pgrep -f '" + pattern + "' 2>/dev/null";
        FILE* pipe = popen(command.c_str(), "r");
        
        if (!pipe) {
            return 0;
        }
        
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            try {
                int pid = std::stoi(std::string(buffer));
                
                // Try to set CPU affinity for this thread
                std::string cpuset_path = "/proc/" + std::to_string(pid) + "/task/" + std::to_string(pid) + "/cpuset";
                
                // Check if this is a kernel thread (no memory maps)
                std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
                std::ifstream maps_file(maps_path);
                
                if (!maps_file.is_open() || maps_file.peek() == std::ifstream::traits_type::eof()) {
                    // This is likely a kernel thread, try to move it
                    if (set_thread_cpu_affinity(pid, mask)) {
                        moved_count++;
                    }
                }
                
            } catch (const std::exception&) {
                // Skip invalid PIDs
                continue;
            }
        }
        
        pclose(pipe);
        return moved_count;
    }
    
    bool set_thread_cpu_affinity(int pid, const std::string& mask) {
        // Convert hex mask to cpu_set_t
        uint64_t hex_mask;
        try {
            hex_mask = std::stoull(mask, nullptr, 16);
        } catch (const std::exception&) {
            return false;
        }
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        // Set bits in cpuset based on hex mask
        for (int i = 0; i < 64 && i < total_cpu_count_; ++i) {
            if (hex_mask & (1ULL << i)) {
                CPU_SET(i, &cpuset);
            }
        }
        
        // Try to set affinity using sched_setaffinity
        int result = sched_setaffinity(pid, sizeof(cpuset), &cpuset);
        return result == 0;
    }
};

// Performance targets for sub-200μs pipeline
struct PerformanceTargets {
    static constexpr double MARKET_DATA_TARGET_US = 2.0;
    static constexpr double MODEL_INFERENCE_TARGET_US = 20.0;
    static constexpr double ORDER_EXECUTION_TARGET_US = 80.0;
    static constexpr double TOTAL_PIPELINE_TARGET_US = 200.0;
};

// Forward declaration - actual definition is in hft_models.hpp
struct AdvancedLearningStats;

// Market regime detection system
class MarketRegimeDetector {
public:
    struct RegimeIndicators {
        double volatility_regime;      // 0=low, 1=high
        double trend_strength;         // 0=ranging, 1=trending
        double correlation_regime;     // 0=low correlation, 1=high correlation
        double volume_regime;          // 0=low volume, 1=high volume
        uint64_t last_update_ns;
    };

private:
    std::unordered_map<int32_t, RegimeIndicators> symbol_regimes_;
    mutable std::mutex regime_mutex_;
    
    // Market-wide regime
    std::atomic<double> market_volatility_{0.5};
    std::atomic<double> market_trend_{0.5};
    std::atomic<double> market_correlation_{0.5};
    
public:
    MarketRegimeDetector() = default;

    void update_regime(int32_t symbol_id, const float* features) {
        std::lock_guard<std::mutex> lock(regime_mutex_);
        
        auto& regime = symbol_regimes_[symbol_id];
        
        // Calculate regime indicators from features
        regime.volatility_regime = std::min(1.0, static_cast<double>(features[0] * 100.0f)); // Volatility feature
        regime.trend_strength = std::abs(static_cast<double>(features[1]) * 10.0);          // Price momentum
        regime.volume_regime = std::min(1.0, static_cast<double>(features[4] / 2.0f));       // Volume ratio
        regime.last_update_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
    RegimeIndicators get_regime(int32_t symbol_id) const {
        std::lock_guard<std::mutex> lock(regime_mutex_);
        auto it = symbol_regimes_.find(symbol_id);
        
        // FAIL-FAST: No mock/placeholder regime data - require real regime data
        if (it == symbol_regimes_.end()) {
            throw std::runtime_error("MarketRegimeDetector: No regime data available for symbol_id " + std::to_string(symbol_id) + " - cannot provide placeholder regime indicators");
        }
        
        return it->second;
    }
    
    struct MarketRegime {
        double volatility;
        double trend;
        double correlation;
        std::string regime_name;
    };
    
    MarketRegime get_market_regime() const {
        MarketRegime regime;
        regime.volatility = market_volatility_.load();
        regime.trend = market_trend_.load();
        regime.correlation = market_correlation_.load();
        
        // Classify regime
        if (regime.volatility > 0.7) {
            regime.regime_name = "HIGH_VOLATILITY";
        } else if (regime.trend > 0.7) {
            regime.regime_name = "TRENDING";
        } else if (regime.correlation > 0.7) {
            regime.regime_name = "HIGH_CORRELATION";
        } else {
            regime.regime_name = "NORMAL";
        }
        
        return regime;
    }

    void calculate_market_wide_indicators(const std::vector<int32_t>& active_stock_ids,
                                          const std::unordered_map<int32_t, RegimeIndicators>& current_symbol_regimes,
                                          int32_t vix_symbol_id = -1) {
        std::lock_guard<std::mutex> lock(regime_mutex_); // Protects symbol_regimes_ if accessed directly, though we take a copy

        // FAIL-FAST: No mock data or placeholders - require real active stocks
        if (active_stock_ids.empty()) {
            throw std::runtime_error("MarketRegimeDetector: No active stocks available - cannot calculate market regime without real data");
        }

        double sum_volatility = 0.0;
        double sum_trend_strength = 0.0;
        int count = 0;

        for (int32_t stock_id : active_stock_ids) {
            auto it = current_symbol_regimes.find(stock_id);
            if (it != current_symbol_regimes.end()) {
                sum_volatility += it->second.volatility_regime;
                sum_trend_strength += it->second.trend_strength;
                count++;
            }
        }

        // FAIL-FAST: No mock data - require real stock regime data
        if (count == 0) {
            throw std::runtime_error("MarketRegimeDetector: No valid stock regime data available - cannot calculate market indicators without real data");
        }

        // Calculate from real data only
        market_volatility_.store(sum_volatility / count);
        market_trend_.store(sum_trend_strength / count);

        // Market Correlation Logic - FAIL-FAST: No fallback calculations
        if (vix_symbol_id != -1) {
            auto vix_it = current_symbol_regimes.find(vix_symbol_id);
            if (vix_it != current_symbol_regimes.end()) {
                // Use real VIX data only
                double vix_proxy_correlation = vix_it->second.volatility_regime;
                market_correlation_.store(std::max(0.0, std::min(1.0, vix_proxy_correlation)));
            } else {
                // FAIL-FAST: No VIX fallback using volatility proxies - require real VIX data
                throw std::runtime_error("MarketRegimeDetector: VIX data unavailable - cannot calculate market correlation without real VIX data");
            }
        } else {
            // FAIL-FAST: No VIX symbol provided - require real VIX symbol for correlation calculation
            throw std::runtime_error("MarketRegimeDetector: VIX symbol not provided - cannot calculate market correlation without VIX reference");
        }
    }
    
    // Method to get a copy of symbol_regimes_ for safe iteration outside lock
    std::unordered_map<int32_t, RegimeIndicators> get_all_symbol_regimes() const {
        std::lock_guard<std::mutex> lock(regime_mutex_);
        return symbol_regimes_;
    }
};

// Risk management system with dynamic position sizing
class DynamicRiskManager {
private:
    // NO DEFAULTS - must be initialized from real Alpaca data
    std::atomic<double> portfolio_heat_{0.0};        // Current risk exposure
    std::atomic<double> max_portfolio_heat_{0.95};   // Maximum allowed risk
    std::atomic<double> daily_pnl_{0.0};             // Today's P&L
    std::atomic<double> max_daily_loss_{-1000.0};    // Maximum daily loss
    std::atomic<double> initial_portfolio_value_{0.0}; // Set from first real Alpaca data
    std::atomic<bool> initial_value_set_{false};    // Track if initial value was set

    // For max drawdown calculation - using atomics for thread safety
    std::atomic<double> current_peak_portfolio_value_{0.0}; // NO DEFAULT
    std::atomic<double> max_drawdown_value_{0.0};
    mutable std::mutex pnl_drawdown_mutex_; // Protects complex calculations
    
    // Symbol-specific risk limits
    std::unordered_map<int32_t, double> symbol_risk_limits_;
    std::mutex risk_mutex_; // Protects symbol_risk_limits_
    
public:
    DynamicRiskManager() = default; // NO hardcoded defaults

    bool can_open_position(int32_t symbol_id, double position_size, double confidence) {
        // Check portfolio heat
        if (portfolio_heat_.load() >= max_portfolio_heat_.load()) {
            return false;
        }
        
        // Check daily loss limit
        if (daily_pnl_.load() <= max_daily_loss_.load()) {
            return false;
        }
        
        // Check symbol-specific limits
        std::lock_guard<std::mutex> lock(risk_mutex_);
        auto it = symbol_risk_limits_.find(symbol_id);
        if (it != symbol_risk_limits_.end() && position_size > it->second) {
            return false;
        }
        
        // Confidence-based sizing
        return confidence > 0.6; // Minimum confidence threshold
    }
    
    double calculate_position_size(double confidence, double volatility, double portfolio_value) {
        // Kelly criterion with confidence adjustment
        double kelly_fraction = confidence * 0.25; // Conservative Kelly
        
        // Volatility adjustment
        double vol_adjustment = std::max(0.1, 1.0 - volatility * 2.0);
        
        // Portfolio heat adjustment
        double heat_adjustment = std::max(0.1, 1.0 - portfolio_heat_.load());
        
        double base_size = portfolio_value * 0.02; // 2% base position
        return base_size * kelly_fraction * vol_adjustment * heat_adjustment;
    }
    
    void update_portfolio_heat(double new_heat) {
        portfolio_heat_.store(new_heat);
    }
    
    // Thread-safe portfolio metrics update using atomics - REAL DATA ONLY
    void update_portfolio_metrics(double current_portfolio_value) {
        // Set initial value from first real Alpaca data
        if (!initial_value_set_.load()) {
            if (current_portfolio_value > 0 && !std::isnan(current_portfolio_value) && !std::isinf(current_portfolio_value)) {
                initial_portfolio_value_.store(current_portfolio_value);
                current_peak_portfolio_value_.store(current_portfolio_value);
                initial_value_set_.store(true);
                std::cout << "✅ Initial portfolio value set from real Alpaca data: $" << current_portfolio_value << std::endl;
            } else {
                // Don't log error on first attempt - portfolio data may not be ready yet
                return;
            }
        }
        
        // Calculate P&L from real initial value
        double initial_value = initial_portfolio_value_.load();
        daily_pnl_.store(current_portfolio_value - initial_value);
        
        // Thread-safe peak tracking using compare_exchange
        double current_peak = current_peak_portfolio_value_.load();
        while (current_portfolio_value > current_peak) {
            if (current_peak_portfolio_value_.compare_exchange_weak(current_peak, current_portfolio_value)) {
                break;
            }
        }
        
        // Calculate drawdown
        double peak = current_peak_portfolio_value_.load();
        double current_drawdown = peak - current_portfolio_value;
        if (current_drawdown > 0) {
            double current_max_dd = max_drawdown_value_.load();
            while (current_drawdown > current_max_dd) {
                if (max_drawdown_value_.compare_exchange_weak(current_max_dd, current_drawdown)) {
                    break;
                }
            }
        }
    }

    double get_max_drawdown() const {
        return max_drawdown_value_.load();
    }
    
    struct RiskMetrics {
        double portfolio_heat;
        double daily_pnl;
        double max_daily_loss;
        bool risk_limits_ok;
    };
    
    RiskMetrics get_risk_metrics() const {
        return {
            portfolio_heat_.load(),
            daily_pnl_.load(),
            max_daily_loss_.load(),
            portfolio_heat_.load() < max_portfolio_heat_.load() && 
            daily_pnl_.load() > max_daily_loss_.load()
        };
    }
};

// Main HFT engine coordinator with advanced features
class HFTEngine {
private:
    // Core components
    std::unique_ptr<UltraFastMemoryManager> memory_manager_;
    std::unique_ptr<MarketDataManager> market_data_;
    std::unique_ptr<TieredModelEnsemble> models_;
    std::unique_ptr<UltraFastOrderEngine> order_engine_;
    std::unique_ptr<::AlpacaClient> alpaca_client_;
    std::unique_ptr<UltraFastWebSocketClient> websocket_client_;
    
    // Advanced components
    std::unique_ptr<OnlineLearningEngine> learning_engine_; // From hft_models.hpp
    std::unique_ptr<MarketRegimeDetector> regime_detector_;
    std::unique_ptr<DynamicRiskManager> risk_manager_;
    std::unique_ptr<CPUManager> cpu_manager_;
    
    // Threading and control
    std::thread main_loop_thread_;
    std::thread learning_thread_;
    std::thread regime_thread_;
    std::thread risk_monitoring_thread_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> learning_enabled_{true};
    std::atomic<bool> regime_detection_enabled_{true};
    std::atomic<bool> risk_monitoring_enabled_{true};
    
    // Performance monitoring
    std::atomic<uint64_t> pipeline_cycles_{0};
    std::atomic<uint64_t> total_pipeline_time_ns_{0};
    std::atomic<uint64_t> signals_generated_{0};
    std::atomic<uint64_t> orders_executed_{0};
    std::atomic<uint64_t> regime_updates_{0};
    
    // Symbol management - now includes indices
    std::vector<std::string> stock_symbols_;
    std::vector<std::string> index_symbols_;
    std::vector<int32_t> stock_symbol_ids_;
    std::vector<int32_t> index_symbol_ids_;
    

public:
    explicit HFTEngine(int numa_node = -1) {
        // Load production symbols (stocks + indices)
        load_production_symbols();
        
        // Initialize CPU manager first for optimal thread placement
        cpu_manager_ = std::make_unique<CPUManager>();
        cpu_manager_->initialize_cpu_isolation();
        
        // Initialize core components with NUMA optimization
        memory_manager_ = std::make_unique<UltraFastMemoryManager>(numa_node);
        market_data_ = std::make_unique<MarketDataManager>(*memory_manager_);
        models_ = std::make_unique<TieredModelEnsemble>(*market_data_, *memory_manager_);
        
        // Initialize Alpaca client for REST API trading
        alpaca_client_ = std::make_unique<::AlpacaClient>(APIConfig{});
        if (!alpaca_client_->initialize()) {
            throw std::runtime_error("Failed to initialize Alpaca client");
        }
        
        // Initialize WebSocket client for real-time market data
        websocket_client_ = std::make_unique<UltraFastWebSocketClient>(*market_data_, *memory_manager_, APIConfig{});
        
        order_engine_ = std::make_unique<UltraFastOrderEngine>(*memory_manager_, alpaca_client_.get());
        
        // Initialize advanced components
        learning_engine_ = std::make_unique<OnlineLearningEngine>(10); // 10 features, as defined in hft_models.hpp
        regime_detector_ = std::make_unique<MarketRegimeDetector>();
        risk_manager_ = std::make_unique<DynamicRiskManager>();
        
        // Register and activate symbols
        initialize_symbols();
        
    }
    
    ~HFTEngine() {
        stop();
    }
    
    // Start the advanced HFT engine
    bool start() {
        if (running_.load()) {
            return true;
        }
        
        // Setup WebSocket market data callback to feed live data into market data manager
        websocket_client_->set_market_data_callback([this](const ParsedMessage& msg) {
            // Convert ParsedMessage to market data and update ring buffers
            if (msg.symbol_id >= 0) {
                // Feed real-time trade data
                if (msg.type == MessageType::TRADE && msg.price > 0) {
                    market_data_->update_trade(msg.symbol_id, msg.price, msg.volume, msg.timestamp_ns);
                }
                // Feed real-time quote data
                else if (msg.type == MessageType::QUOTE && msg.bid > 0 && msg.ask > 0) {
                    market_data_->update_quote(msg.symbol_id, msg.bid, msg.ask,
                                             msg.bid_size, msg.ask_size);
                }
                
                // Update regime detection with real features
                alignas(32) float features[10]; // MarketDataManager uses 10 features
                if (market_data_->get_features(msg.symbol_id, features)) {
                    regime_detector_->update_regime(msg.symbol_id, features);
                }
            }
        });
        
        // Start WebSocket client for real-time market data
        if (!websocket_client_->start()) {
            std::cerr << "❌ Failed to start WebSocket client for market data" << std::endl;
            return false;
        }
        
        // Start Alpaca account polling with 30-second intervals (rate limit compliance)
        if (!alpaca_client_->startAccountPolling([this](const ::AlpacaClient::AccountUpdate& update) {
            // Update risk management with real account data
            if (risk_manager_) {
                risk_manager_->update_portfolio_metrics(update.portfolio_value);
            }
            
            // Update order engine with live Alpaca data
            if (order_engine_) {
                order_engine_->update_portfolio_from_alpaca(update);
            }
        }, 30000)) { // 30 seconds = 2 requests/minute, well under 200/minute limit
            std::cerr << "❌ Failed to start Alpaca account polling" << std::endl;
            return false;
        }
        
        // Start order engine
        if (!order_engine_->start()) {
            std::cerr << "❌ Failed to start order engine" << std::endl;
            return false;
        }
        
        running_.store(true);
        
        // Start main trading loop with highest priority using CPUManager (with delay for data accumulation)
        try {
            main_loop_thread_ = std::thread([this]() {
                try {
                    // Wait for market data to accumulate before starting trading
                    std::this_thread::sleep_for(std::chrono::seconds(15));
                    this->main_trading_loop();
                } catch (const std::exception& e) {
                    std::cerr << "❌ CRITICAL: Exception in main trading loop thread: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "❌ CRITICAL: Unknown exception in main trading loop thread" << std::endl;
                }
            });
        } catch (const std::exception& e) {
            std::cerr << "❌ CRITICAL: Failed to create main trading loop thread: " << e.what() << std::endl;
            throw;
        }
        
        cpu_manager_->pin_thread_to_core(main_loop_thread_, cpu_manager_->get_core_for_thread_type("trading"), 99);
        
        // Start advanced subsystems with optimal core placement
        try {
            start_learning_loop();
            start_regime_detection_loop();
            start_risk_monitoring_loop();
        } catch (const std::exception& e) {
            std::cerr << "❌ CRITICAL: Exception starting advanced subsystems: " << e.what() << std::endl;
            throw;
        }
        
        return true;
    }
    
    // Stop the HFT engine
    void stop() {
        if (!running_.load()) {
            return;
        }
        
        
        running_.store(false);
        learning_enabled_.store(false);
        regime_detection_enabled_.store(false);
        risk_monitoring_enabled_.store(false);
        
        // Stop subsystems
        if (learning_thread_.joinable()) learning_thread_.join();
        if (regime_thread_.joinable()) regime_thread_.join();
        if (risk_monitoring_thread_.joinable()) risk_monitoring_thread_.join();
        
        // Stop main loop with exception safety
        if (main_loop_thread_.joinable()) {
            try {
                main_loop_thread_.join();
            } catch (const std::exception& e) {
                std::cerr << "⚠️ Exception during main loop join: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "⚠️ Unknown exception during main loop join" << std::endl;
            }
        }
        
        // Stop core components
        order_engine_->stop();
        
        // Stop WebSocket client
        if (websocket_client_) {
            websocket_client_->stop();
        }
        
        // Stop Alpaca account polling
        if (alpaca_client_) {
            alpaca_client_->stopAccountPolling();
        }
        
        std::cout << "✅ Advanced HFT Engine stopped" << std::endl;
    }
    
    // Enhanced performance statistics
    struct AdvancedPerformanceStats {
        uint64_t pipeline_cycles;
        double avg_pipeline_time_us;
        uint64_t signals_generated;
        uint64_t orders_executed;
        double signal_to_order_ratio;
        bool pipeline_target_met;
        
        // Component stats
        typename TieredModelEnsemble::ModelPerformanceStats model_stats;
        typename UltraFastOrderEngine::PerformanceStats order_stats;
        AdvancedLearningStats learning_stats; // Defined in hft_models.hpp
        typename DynamicRiskManager::RiskMetrics risk_metrics;
        typename MarketRegimeDetector::MarketRegime market_regime;
        
        
        // Memory stats
        double memory_avg_latency_us;
        uint64_t memory_operations;
        
        // Advanced metrics
        uint64_t regime_updates;
        double portfolio_sharpe_ratio;
        double max_drawdown;
    };
    
    AdvancedPerformanceStats get_performance_stats() const {
        uint64_t cycles = pipeline_cycles_.load();
        uint64_t total_time = total_pipeline_time_ns_.load();
        uint64_t signals = signals_generated_.load();
        uint64_t orders = orders_executed_.load();
        
        double avg_pipeline_us = cycles > 0 ?
            static_cast<double>(total_time) / (cycles * 1000.0) : 0.0;
        
        AdvancedPerformanceStats stats;
        stats.pipeline_cycles = cycles;
        stats.avg_pipeline_time_us = avg_pipeline_us;
        stats.signals_generated = signals;
        stats.orders_executed = orders;
        stats.signal_to_order_ratio = signals > 0 ? static_cast<double>(orders) / signals : 0.0;
        stats.pipeline_target_met = avg_pipeline_us <= PerformanceTargets::TOTAL_PIPELINE_TARGET_US;
        
        // Get component stats
        stats.model_stats = models_->get_performance_stats();
        stats.order_stats = order_engine_->get_performance_stats();
        stats.risk_metrics = risk_manager_->get_risk_metrics();
        stats.market_regime = regime_detector_->get_market_regime();
        
        // Get learning stats from both Tier 1 and Tier 2 models
        if (models_) {
            // Get stats from both tiers
            auto tier1_stats = models_->get_tier1_learning_stats();
            auto tier2_stats = models_->get_tier2_learning_stats();
            
            // Aggregate stats from Tier 1
            uint64_t tier1_total_trades = tier1_stats.momentum_perf.total_updates + tier1_stats.reversion_perf.total_updates;
            double tier1_momentum_pnl = static_cast<double>(tier1_stats.momentum_perf.cumulative_pnl);
            double tier1_reversion_pnl = static_cast<double>(tier1_stats.reversion_perf.cumulative_pnl);
            double tier1_total_pnl = tier1_momentum_pnl + tier1_reversion_pnl;
            uint64_t tier1_winning_momentum = static_cast<uint64_t>(tier1_stats.momentum_perf.win_rate * tier1_stats.momentum_perf.total_updates);
            uint64_t tier1_winning_reversion = static_cast<uint64_t>(tier1_stats.reversion_perf.win_rate * tier1_stats.reversion_perf.total_updates);
            uint64_t tier1_winning_trades = tier1_winning_momentum + tier1_winning_reversion;
            
            // Aggregate stats from Tier 2
            uint64_t tier2_total_trades = tier2_stats.momentum_perf.total_updates + tier2_stats.reversion_perf.total_updates;
            double tier2_momentum_pnl = static_cast<double>(tier2_stats.momentum_perf.cumulative_pnl);
            double tier2_reversion_pnl = static_cast<double>(tier2_stats.reversion_perf.cumulative_pnl);
            double tier2_total_pnl = tier2_momentum_pnl + tier2_reversion_pnl;
            uint64_t tier2_winning_momentum = static_cast<uint64_t>(tier2_stats.momentum_perf.win_rate * tier2_stats.momentum_perf.total_updates);
            uint64_t tier2_winning_reversion = static_cast<uint64_t>(tier2_stats.reversion_perf.win_rate * tier2_stats.reversion_perf.total_updates);
            uint64_t tier2_winning_trades = tier2_winning_momentum + tier2_winning_reversion;
            
            // Combine stats from both tiers
            stats.learning_stats.total_trades = tier1_total_trades + tier2_total_trades;
            stats.learning_stats.total_pnl = tier1_total_pnl + tier2_total_pnl;
            stats.learning_stats.winning_trades = tier1_winning_trades + tier2_winning_trades;
            
            if (stats.learning_stats.total_trades > 0) {
                stats.learning_stats.win_rate = static_cast<double>(stats.learning_stats.winning_trades) / stats.learning_stats.total_trades;
            } else {
                stats.learning_stats.win_rate = 0.0;
            }

            // Strategy-specific stats (combining both tiers)
            // Momentum strategy (both tiers)
            uint64_t total_momentum_trades = tier1_stats.momentum_perf.total_updates + tier2_stats.momentum_perf.total_updates;
            uint64_t total_momentum_wins = tier1_winning_momentum + tier2_winning_momentum;
            stats.learning_stats.strategy_win_rates[0] = total_momentum_trades > 0 ?
                static_cast<double>(total_momentum_wins) / total_momentum_trades : 0.0;
            stats.learning_stats.strategy_pnl[0] = tier1_momentum_pnl + tier2_momentum_pnl;
            
            // Mean Reversion strategy (both tiers)
            uint64_t total_reversion_trades = tier1_stats.reversion_perf.total_updates + tier2_stats.reversion_perf.total_updates;
            uint64_t total_reversion_wins = tier1_winning_reversion + tier2_winning_reversion;
            stats.learning_stats.strategy_win_rates[1] = total_reversion_trades > 0 ?
                static_cast<double>(total_reversion_wins) / total_reversion_trades : 0.0;
            stats.learning_stats.strategy_pnl[1] = tier1_reversion_pnl + tier2_reversion_pnl;
            
            // Breakout and Arbitrage strategies (not implemented)
            stats.learning_stats.strategy_win_rates[2] = 0.0; // Breakout
            stats.learning_stats.strategy_pnl[2] = 0.0;
            stats.learning_stats.strategy_win_rates[3] = 0.0; // Arbitrage
            stats.learning_stats.strategy_pnl[3] = 0.0;
        } else { // Default if models_ itself is not initialized
            stats.learning_stats.total_trades = 0;
            stats.learning_stats.winning_trades = 0;
            stats.learning_stats.win_rate = 0.0;
            stats.learning_stats.total_pnl = 0.0;
            for(int i=0; i<4; ++i) {
                stats.learning_stats.strategy_win_rates[i] = 0.0;
                stats.learning_stats.strategy_pnl[i] = 0.0;
            }
        }
        // Sharpe ratio is calculated using the aggregated stats from above.
        stats.learning_stats.sharpe_ratio = calculate_sharpe_ratio_internal(
            stats.learning_stats.total_pnl,
            stats.learning_stats.win_rate,
            stats.learning_stats.total_trades
        );
        
        // Memory stats
        const auto& memory_perf = memory_manager_->get_performance_stats();
        stats.memory_avg_latency_us = memory_perf.get_average_latency_us();
        stats.memory_operations = memory_perf.operation_count.load();
        
        // Advanced metrics
        stats.regime_updates = regime_updates_.load();
        stats.portfolio_sharpe_ratio = calculate_sharpe_ratio();
        stats.max_drawdown = calculate_max_drawdown();
        
        return stats;
    }
    
    // Get portfolio state
    auto get_portfolio_state() const {
        return order_engine_->get_portfolio_state();
    }

private:
    void load_production_symbols() {
        // Load stock symbols (Tier 1 + Tier 2)
        for (size_t i = 0; i < APIConfig::NUM_TIER_1_SYMBOLS; ++i) {
            stock_symbols_.push_back(APIConfig::TIER_1_SYMBOLS[i]);
        }
        for (size_t i = 0; i < APIConfig::NUM_TIER_2_SYMBOLS; ++i) {
            stock_symbols_.push_back(APIConfig::TIER_2_SYMBOLS[i]);
        }
        
        // Map index symbols to their liquid ETF proxies for WebSocket compatibility
        index_symbols_ = {
            "SPY",    // S&P 500 ETF (proxy for SPX)
            "QQQ",    // Nasdaq 100 ETF (proxy for NDX)
            "DIA",    // Dow Jones ETF (proxy for DJI)
            "IWM",    // Russell 2000 ETF (proxy for RUT)
            "VXX"     // VIX ETF (proxy for VIX)
        };
    }
    
    void initialize_symbols() {
        // Register stock symbols
        for (const auto& symbol : stock_symbols_) {
            int32_t symbol_id = memory_manager_->register_symbol(symbol);
            if (symbol_id >= 0) {
                stock_symbol_ids_.push_back(symbol_id);
                market_data_->activate_symbol(symbol_id);
            }
        }
        
        // Register index symbols
        for (const auto& symbol : index_symbols_) {
            int32_t symbol_id = memory_manager_->register_symbol(symbol);
            if (symbol_id >= 0) {
                index_symbol_ids_.push_back(symbol_id);
                market_data_->activate_symbol(symbol_id);
            }
        }
    }
    
    
    void set_thread_priority(std::thread& thread, int cpu_core, int priority) {
#ifdef __linux__
        // Validate core first
        if (!cpu_manager_->is_core_valid(cpu_core)) {
            std::cerr << "❌ Invalid CPU core " << cpu_core << " for thread priority setting" << std::endl;
            return;
        }
        
        // Set CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core, &cpuset);
        pthread_setaffinity_np(thread.native_handle(), sizeof(cpuset), &cpuset);
        
        // Set priority
        struct sched_param param;
        param.sched_priority = priority;
        pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &param);
#else
        (void)thread; (void)cpu_core; (void)priority; // Suppress unused parameter warnings
        std::cerr << "⚠️ Thread priority setting not supported on this platform" << std::endl;
#endif
    }
    
    // Calculate Sharpe ratio (internal helper using aggregated stats)
    double calculate_sharpe_ratio_internal(double total_pnl, double win_rate, uint64_t total_trades) const {
        if (total_trades < 10) return 0.0;
        
        double avg_return = total_trades > 0 ? total_pnl / total_trades : 0.0;
        double risk_free_rate = 0.02 / 252; // Daily risk-free rate (2% annual)
        
        // Estimate volatility from win rate (very rough estimate)
        double volatility_arg = win_rate * (1.0 - win_rate);
        double volatility = std::sqrt(std::max(0.0, volatility_arg)); // Ensure non-negative for sqrt
        if (volatility < 1e-9 && total_trades > 0) volatility = 0.01; // Avoid division by zero or near-zero
        
        return volatility > 1e-9 ? (avg_return - risk_free_rate) / volatility : 0.0;
    }

    // Public Sharpe ratio using aggregated learning stats from both tiers
    double calculate_sharpe_ratio() const {
        if (!models_) return 0.0;
        
        auto tier1_stats = models_->get_tier1_learning_stats();
        auto tier2_stats = models_->get_tier2_learning_stats();

        // Aggregate trades from both tiers
        uint64_t tier1_trades = tier1_stats.momentum_perf.total_updates + tier1_stats.reversion_perf.total_updates;
        uint64_t tier2_trades = tier2_stats.momentum_perf.total_updates + tier2_stats.reversion_perf.total_updates;
        uint64_t total_trades = tier1_trades + tier2_trades;
        
        if (total_trades < 10) return 0.0;

        // Aggregate P&L from both tiers
        double tier1_pnl = static_cast<double>(tier1_stats.momentum_perf.cumulative_pnl + tier1_stats.reversion_perf.cumulative_pnl);
        double tier2_pnl = static_cast<double>(tier2_stats.momentum_perf.cumulative_pnl + tier2_stats.reversion_perf.cumulative_pnl);
        double total_pnl = tier1_pnl + tier2_pnl;
        
        // Aggregate winning trades from both tiers
        uint64_t tier1_winning_momentum = static_cast<uint64_t>(tier1_stats.momentum_perf.win_rate * tier1_stats.momentum_perf.total_updates);
        uint64_t tier1_winning_reversion = static_cast<uint64_t>(tier1_stats.reversion_perf.win_rate * tier1_stats.reversion_perf.total_updates);
        uint64_t tier2_winning_momentum = static_cast<uint64_t>(tier2_stats.momentum_perf.win_rate * tier2_stats.momentum_perf.total_updates);
        uint64_t tier2_winning_reversion = static_cast<uint64_t>(tier2_stats.reversion_perf.win_rate * tier2_stats.reversion_perf.total_updates);
        uint64_t total_winning_trades = tier1_winning_momentum + tier1_winning_reversion + tier2_winning_momentum + tier2_winning_reversion;
        
        double overall_win_rate = (total_trades > 0) ? static_cast<double>(total_winning_trades) / total_trades : 0.0;
        
        return calculate_sharpe_ratio_internal(total_pnl, overall_win_rate, total_trades);
    }
    
    // Calculate maximum drawdown
    double calculate_max_drawdown() const {
        if (!risk_manager_) return 0.0; // Check if risk_manager_ is initialized
        return risk_manager_->get_max_drawdown();
    }
    
    // Ultra-optimized main trading loop with zero allocations and CPU pinning
    void main_trading_loop() {
        try {
            // Pre-allocate everything for zero-allocation operation - increased for both tiers
            alignas(64) float features_matrix[128 * 15]; // For all symbols (Tier 1 + Tier 2)
            alignas(64) TradingSignal signals[128];
            alignas(64) std::array<int32_t, 128> active_symbols;
        
        // Pin to CPU core 0 for deterministic performance
#ifdef __linux__
        if (cpu_manager_->is_core_valid(0)) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(0, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            
            // Set CPU governor to performance mode for this core
            CPUUtilities::set_cpu_governor(0, "performance");
        }
#endif
        
        // Get ALL symbol IDs (both Tier 1 and Tier 2)
        std::vector<int32_t> all_symbol_ids;
        
        try {
            // Add Tier 1 symbols
            for (size_t i = 0; i < APIConfig::NUM_TIER_1_SYMBOLS; ++i) {
                int32_t id = memory_manager_->get_symbol_index(APIConfig::TIER_1_SYMBOLS[i]);
                if (id >= 0) {
                    all_symbol_ids.push_back(id);
                }
            }
            
            // Add Tier 2 symbols
            for (size_t i = 0; i < APIConfig::NUM_TIER_2_SYMBOLS; ++i) {
                int32_t id = memory_manager_->get_symbol_index(APIConfig::TIER_2_SYMBOLS[i]);
                if (id >= 0) {
                    all_symbol_ids.push_back(id);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ CRITICAL: Exception in symbol ID gathering: " << e.what() << std::endl;
            throw;
        }
        
        while (running_.load(std::memory_order_relaxed)) {
            auto pipeline_start = std::chrono::high_resolution_clock::now();
            
            // Step 1: Batch fetch all features in parallel (no allocation) - now for ALL symbols
            int valid_features_count = 0;
            try {
                for (size_t i = 0; i < all_symbol_ids.size(); ++i) {
                    if (i >= 128) break; // Safety check for array bounds
                    
                    if (market_data_->get_features(all_symbol_ids[i], features_matrix + i * 15)) {
                        valid_features_count++;
                    } else {
                        // Zero out invalid features to prevent processing
                        std::memset(features_matrix + i * 15, 0, 15 * sizeof(float));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Feature calculation error: " << e.what() << std::endl;
                // Continue with zero features to prevent crash
                valid_features_count = 0;
            }
            
            // Market data status logging removed to reduce console spam
            
            // Step 2: Get regime without allocation (with error handling)
            auto regime = regime_detector_->get_market_regime();
            
            // Step 3: Batch inference using tiered model ensemble (both Tier 1 and Tier 2)
            uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                
            // Direct call to optimized batch inference for ALL symbols
            if (models_ && !all_symbol_ids.empty()) {
                // Use predict_batch method that routes to appropriate tier models
                models_->predict_batch(features_matrix, signals, all_symbol_ids, timestamp_ns);
                
                // Calculate signal count based on all_symbol_ids size
                size_t signal_count = std::min(all_symbol_ids.size(), static_cast<size_t>(128));
                
                // Process signals without allocation
                uint32_t executed_count = 0;
                uint32_t signals_with_direction = 0;
                uint32_t signals_above_confidence = 0;
                uint32_t signals_passed_risk = 0;
                
                for (size_t i = 0; i < signal_count; ++i) {
                    if (signals[i].direction != 0) {
                        signals_with_direction++;
                        
                        // Standard confidence threshold for signal filtering
                        if (signals[i].confidence > 0.6f) {
                            signals_above_confidence++;
                            
                            // Regime adjustment without allocation
                            double regime_adjustment = 1.0;
                            if (regime.regime_name == "HIGH_VOLATILITY") {
                                regime_adjustment = 0.7;
                            } else if (regime.regime_name == "TRENDING") {
                                regime_adjustment = 1.2;
                            }
                            
                            signals[i].confidence *= regime_adjustment;
                            
                            // Only proceed if still above minimum confidence after regime adjustment
                            if (signals[i].confidence > 0.55f) {
                                // Check risk limits
                                if (risk_manager_->can_open_position(signals[i].symbol_id,
                                                                   signals[i].position_size,
                                                                   signals[i].confidence)) {
                                    signals_passed_risk++;
                                    
                                    // Register and execute signal directly
                                    const float* signal_features = features_matrix + i * 15;
                                    order_engine_->register_signal(signals[i], signal_features);
                                    
                                    if (order_engine_->execute_signal(signals[i])) {
                                        executed_count++;
                                    }
                                    
                                    // Reasonable limit: Up to 5 trades per cycle for balanced execution
                                    if (executed_count >= 5) break;
                                }
                            }
                        }
                    }
                }
                
                // Signal filtering debug logging removed to reduce console spam
                
                signals_generated_.fetch_add(signal_count, std::memory_order_relaxed);
                orders_executed_.fetch_add(executed_count, std::memory_order_relaxed);
            }
            
            auto pipeline_end = std::chrono::high_resolution_clock::now();
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                pipeline_end - pipeline_start).count();
            
            // Update stats without division
            pipeline_cycles_.fetch_add(1, std::memory_order_relaxed);
            total_pipeline_time_ns_.fetch_add(duration_ns, std::memory_order_relaxed);
            
            // Log performance every 1000 cycles
            if (pipeline_cycles_.load() % 1000 == 0) {
                log_advanced_performance_stats();
            }
            
            // Adaptive sleep based on pipeline time for precision timing
            if (duration_ns < 100000) { // Less than 100μs
                // Busy wait for precision in ultra-fast cycles
                while ((std::chrono::high_resolution_clock::now() - pipeline_start).count() < 200000) {
                    _mm_pause(); // CPU pause instruction to reduce power consumption
                }
            } else {
                // Normal sleep for longer cycles
                std::this_thread::yield();
            }
        }
        
        // Restore CPU governor
#ifdef __linux__
        if (cpu_manager_->is_core_valid(0)) {
            CPUUtilities::set_cpu_governor(0, "powersave");
        }
#endif
        
        } catch (const std::exception& e) {
            std::cerr << "❌ CRITICAL: Exception in main_trading_loop: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Advanced learning feedback loop
    void start_learning_loop() {
        learning_thread_ = std::thread([this]() {
            
            while (learning_enabled_.load()) {
                try {
                    hft::SignalOutcome outcome; // Use SignalOutcome from hft_models.hpp
                    if (order_engine_->get_next_trade_outcome(outcome)) {
                        const float* features = order_engine_->get_signal_features(outcome.signal_id);
                        if (features) {
                            // The SignalOutcome and original features are passed to the TieredModelEnsemble.
                            // TieredModelEnsemble routes the outcome to the specific model based on tier:
                            //   - Tier 1: UltraFastLinearModel (momentum/reversion strategies)
                            //   - Tier 2: FastTreeModel (tree-based momentum/reversion with regime adaptation)
                            // Each model uses its internal OnlineLearningEngine instance(s) to update weights.
                            // Learning statistics are aggregated by querying each model's performance metrics.
                            models_->update_from_outcome(outcome, features);

                        }
                    } else {
                        // No new outcomes, sleep briefly
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "❌ Learning loop error: " << e.what() << std::endl;
                    // Avoid busy-looping on error
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            
        });
        
        cpu_manager_->pin_thread_to_core(learning_thread_, cpu_manager_->get_core_for_thread_type("learning"), 50);
    }
    
    // Market regime detection loop with safer error handling
    void start_regime_detection_loop() {
        regime_thread_ = std::thread([this]() {
            
            int consecutive_errors = 0;
            const int max_consecutive_errors = 5;
            
            while (regime_detection_enabled_.load()) {
                bool success = false;
                
                try {
                    if (regime_detector_ && memory_manager_) {
                        int32_t vix_id = memory_manager_->get_symbol_index("VIX");
                        
                        // Check if we have sufficient data before proceeding
                        if (stock_symbol_ids_.empty()) {
                            std::cerr << "⚠️ No active stock symbols for regime detection" << std::endl;
                            std::this_thread::sleep_for(std::chrono::seconds(30));
                            continue;
                        }
                        
                        auto current_symbol_regimes_snapshot = regime_detector_->get_all_symbol_regimes();
                        
                        // Validate we have some regime data
                        if (current_symbol_regimes_snapshot.empty()) {
                            std::cerr << "⚠️ No regime data available yet, waiting..." << std::endl;
                            std::this_thread::sleep_for(std::chrono::seconds(30));
                            continue;
                        }

                        regime_detector_->calculate_market_wide_indicators(
                            stock_symbol_ids_,
                            current_symbol_regimes_snapshot,
                            vix_id
                        );
                        regime_updates_.fetch_add(1, std::memory_order_relaxed);
                        success = true;
                        consecutive_errors = 0; // Reset error counter on success
                    }
                    
                } catch (const std::exception& e) {
                    consecutive_errors++;
                    std::cerr << "❌ Regime detection error (" << consecutive_errors << "/" << max_consecutive_errors << "): " << e.what() << std::endl;
                    
                    if (consecutive_errors >= max_consecutive_errors) {
                        std::cerr << "❌ Too many consecutive regime detection errors, disabling..." << std::endl;
                        regime_detection_enabled_.store(false);
                        break;
                    }
                }
                
                // Adaptive sleep based on success/failure
                if (success) {
                    std::this_thread::sleep_for(std::chrono::seconds(15));
                } else {
                    // Exponential backoff on errors
                    int sleep_time = std::min(15 * (1 << consecutive_errors), 300); // Max 5 minutes
                    std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
                }
            }
            
        });
        
        cpu_manager_->pin_thread_to_core(regime_thread_, cpu_manager_->get_core_for_thread_type("regime"), 30);
    }
    
    // Risk monitoring loop
    void start_risk_monitoring_loop() { // Renamed to avoid conflict
        risk_monitoring_thread_ = std::thread([this]() {
            
            while (risk_monitoring_enabled_.load()) {
                try {
                    // Update portfolio heat and risk metrics
                    auto portfolio_state = order_engine_->get_portfolio_state();
                    double portfolio_heat = portfolio_state.total_exposure;
                    risk_manager_->update_portfolio_heat(portfolio_heat);
                    
                    // Check risk limits
                    auto risk_metrics = risk_manager_->get_risk_metrics();
                    if (!risk_metrics.risk_limits_ok) {
                        std::cout << "⚠️ Risk limits breached! Portfolio heat: " 
                                  << risk_metrics.portfolio_heat << std::endl;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                    
                } catch (const std::exception& e) {
                    std::cerr << "❌ Risk monitoring error: " << e.what() << std::endl;
                }
            }
            
        });
        
        cpu_manager_->pin_thread_to_core(risk_monitoring_thread_, cpu_manager_->get_core_for_thread_type("risk"), 40);
    }
    
    void log_advanced_performance_stats() const {
        auto stats = get_performance_stats();
        
        std::cout << "\n=== ADVANCED HFT ENGINE PERFORMANCE ===" << std::endl;
        std::cout << "Pipeline cycles: " << stats.pipeline_cycles << std::endl;
        std::cout << "Avg pipeline time: " << stats.avg_pipeline_time_us 
                  << " μs (target: " << PerformanceTargets::TOTAL_PIPELINE_TARGET_US << " μs)" << std::endl;
        std::cout << "Signals generated: " << stats.signals_generated << std::endl;
        std::cout << "Orders executed: " << stats.orders_executed << std::endl;
        std::cout << "Signal->Order ratio: " << (stats.signal_to_order_ratio * 100.0) << "%" << std::endl;
        std::cout << "Pipeline target met: " << (stats.pipeline_target_met ? "YES" : "NO") << std::endl;
        
        std::cout << "\n--- REST API Performance ---" << std::endl;
        std::cout << "Account polling: Active" << std::endl;
        std::cout << "Rate limit: 200 req/min (30s intervals)" << std::endl;
        
        std::cout << "\n--- Model Performance ---" << std::endl;
        for (int tier = 0; tier < 2; ++tier) { // Adjusted for 2 tiers
            std::cout << "  Tier " << (tier + 1) << ": "
                      << stats.model_stats.avg_inference_time_us[tier] << " μs avg, "
                      << stats.model_stats.inference_count[tier] << " inferences, target met: "
                      << (stats.model_stats.target_met[tier] ? "YES" : "NO") << std::endl;
        }
        
        std::cout << "\n--- Online Learning ---" << std::endl;
        std::cout << "Total trades: " << stats.learning_stats.total_trades << std::endl;
        std::cout << "Win rate: " << (stats.learning_stats.win_rate * 100.0) << "%" << std::endl;
        std::cout << "Total P&L: $" << stats.learning_stats.total_pnl << std::endl;
        std::cout << "Sharpe ratio: " << stats.learning_stats.sharpe_ratio << std::endl;
        
        std::cout << "Strategy performance:" << std::endl;
        const char* strategy_names[] = {"Momentum", "Mean Reversion", "Breakout", "Arbitrage"};
        for (int i = 0; i < 4; ++i) {
            std::cout << "  " << strategy_names[i] << ": " 
                      << (stats.learning_stats.strategy_win_rates[i] * 100.0) << "% win rate, $"
                      << stats.learning_stats.strategy_pnl[i] << " P&L" << std::endl;
        }
        
        std::cout << "\n--- Market Regime ---" << std::endl;
        std::cout << "Current regime: " << stats.market_regime.regime_name << std::endl;
        std::cout << "Volatility: " << (stats.market_regime.volatility * 100.0) << "%" << std::endl;
        std::cout << "Trend strength: " << (stats.market_regime.trend * 100.0) << "%" << std::endl;
        std::cout << "Correlation: " << (stats.market_regime.correlation * 100.0) << "%" << std::endl;
        std::cout << "Regime updates: " << stats.regime_updates << std::endl;
        
        std::cout << "\n--- Risk Management ---" << std::endl;
        std::cout << "Portfolio heat: " << (stats.risk_metrics.portfolio_heat * 100.0) << "%" << std::endl;
        std::cout << "Daily P&L: $" << stats.risk_metrics.daily_pnl << std::endl;
        std::cout << "Risk limits OK: " << (stats.risk_metrics.risk_limits_ok ? "YES" : "NO") << std::endl;
        std::cout << "Portfolio Sharpe: " << stats.portfolio_sharpe_ratio << std::endl;
        std::cout << "Max drawdown: $" << stats.max_drawdown << std::endl;
        
        std::cout << "\n--- Order Engine ---" << std::endl;
        std::cout << "Avg processing time: " << stats.order_stats.avg_processing_time_us << " μs" << std::endl;
        std::cout << "Fill rate: " << (stats.order_stats.fill_rate * 100.0) << "%" << std::endl;
        
        std::cout << "\n--- Memory Manager ---" << std::endl;
        std::cout << "Avg latency: " << stats.memory_avg_latency_us << " μs" << std::endl;
        std::cout << "Operations: " << stats.memory_operations << std::endl;
        
        auto portfolio = get_portfolio_state();
        std::cout << "\n--- Portfolio State ---" << std::endl;
        std::cout << "Portfolio value: $" << std::fixed << std::setprecision(2) << portfolio.portfolio_value << std::endl;
        std::cout << "Cash balance: $" << std::fixed << std::setprecision(2) << portfolio.cash_balance << std::endl;
        std::cout << "Active positions: " << portfolio.active_positions << std::endl;
        std::cout << "Total exposure: " << (portfolio.total_exposure * 100.0) << "%" << std::endl;
        
        std::cout << "\n--- System Status ---" << std::endl;
        std::cout << "Online learning: " << (learning_enabled_.load() ? "ACTIVE" : "DISABLED") << std::endl;
        std::cout << "Regime detection: " << (regime_detection_enabled_.load() ? "ACTIVE" : "DISABLED") << std::endl;
        std::cout << "Risk monitoring: " << (risk_monitoring_enabled_.load() ? "ACTIVE" : "DISABLED") << std::endl;
        
        std::cout << "========================================\n" << std::endl;
    }
};

} // namespace hft
