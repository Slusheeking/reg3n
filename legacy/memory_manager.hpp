#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <immintrin.h>
#include <sys/mman.h>
#include <cstring>
#include <sched.h>
#include <numa.h>
#include "config.hpp"

// Compiler-specific force inline
#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define FORCE_INLINE inline
#endif

namespace hft {

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MAX_SYMBOLS = 1000;

// Lock-free SPSC queue for ultra-fast inter-thread communication
template<typename T, size_t Size>
class alignas(CACHE_LINE_SIZE) LockFreeQueue {
private:
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
    alignas(CACHE_LINE_SIZE) std::array<T, Size> buffer_;

public:
    bool try_push(const T& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % Size;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool try_pop(T& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }
        
        item = buffer_[current_head];
        head_.store((current_head + 1) % Size, std::memory_order_release);
        return true;
    }
    
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }
};

// Simplified aligned allocator without NUMA overhead
class FastAllocator {
public:
    template<typename T>
    T* allocate_aligned(size_t count, size_t alignment = CACHE_LINE_SIZE) {
        size_t size = count * sizeof(T);
        void* ptr = nullptr;
        
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        
        // Use huge pages for large allocations
        if (size >= 2 * 1024 * 1024) { // 2MB threshold
            madvise(ptr, size, MADV_HUGEPAGE);
        }
        
        return static_cast<T*>(ptr);
    }
    
    template<typename T>
    void deallocate(T* ptr, size_t count) {
        free(ptr);
    }
};

// NUMA-aware allocator with huge pages for production HFT
class NUMAOptimizedAllocator {
private:
    int numa_node_;
    void* huge_page_base_;
    size_t huge_page_size_;
    std::atomic<size_t> offset_{0};
    
public:
    explicit NUMAOptimizedAllocator(int numa_node = -1) : numa_node_(numa_node) {
        // Check NUMA availability
        if (numa_available() < 0) {
            numa_node_ = -1;
            std::cout << "⚠️ NUMA not available on this system" << std::endl;
        } else {
            int max_node = numa_max_node();
            if (numa_node_ > max_node) {
                std::cout << "⚠️ NUMA node " << numa_node_ << " invalid (max: "
                          << max_node << "), using default" << std::endl;
                numa_node_ = -1;
            }
        }
        
        // Allocate 1GB huge page if available, otherwise 2MB
        huge_page_size_ = (1ULL << 30); // Try 1GB first
        
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT);
        
        if (numa_node_ >= 0) {
            // Bind to specific NUMA node
            numa_set_preferred(numa_node_);
        }
        
        huge_page_base_ = mmap(nullptr, huge_page_size_,
                              PROT_READ | PROT_WRITE, flags, -1, 0);
        
        if (huge_page_base_ == MAP_FAILED) {
            // Fall back to 2MB pages
            huge_page_size_ = (2ULL << 20);
            flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT);
            huge_page_base_ = mmap(nullptr, huge_page_size_,
                                  PROT_READ | PROT_WRITE, flags, -1, 0);
            
            if (huge_page_base_ == MAP_FAILED) {
                throw std::runtime_error("Failed to allocate huge pages");
            }
            std::cout << "✅ Allocated 2MB huge pages" << std::endl;
        } else {
            std::cout << "✅ Allocated 1GB huge pages" << std::endl;
        }
        
        // Prefault pages
        memset(huge_page_base_, 0, huge_page_size_);
        
        // Advise kernel on usage pattern
        madvise(huge_page_base_, huge_page_size_, MADV_HUGEPAGE);
        madvise(huge_page_base_, huge_page_size_, MADV_DONTFORK);
    }
    
    ~NUMAOptimizedAllocator() {
        if (huge_page_base_ != MAP_FAILED) {
            munmap(huge_page_base_, huge_page_size_);
        }
    }
    
    template<typename T>
    T* allocate(size_t count) {
        size_t size = count * sizeof(T);
        size_t alignment = std::max(alignof(T), size_t(64)); // Cache line aligned
        
        // Atomic allocation from huge page
        size_t old_offset = offset_.fetch_add(size + alignment, std::memory_order_relaxed);
        
        if (old_offset + size + alignment > huge_page_size_) {
            throw std::bad_alloc(); // Out of huge page memory
        }
        
        // Align the pointer
        void* ptr = static_cast<char*>(huge_page_base_) + old_offset;
        size_t space = size + alignment;
        void* aligned = std::align(alignment, size, ptr, space);
        
        return static_cast<T*>(aligned);
    }
    
    size_t get_allocated() const noexcept {
        return offset_.load(std::memory_order_relaxed);
    }
};

// Ultra-fast symbol hash table with linear probing
class SymbolHashTable {
private:
    static constexpr size_t TABLE_SIZE = 65536;
    static constexpr int32_t EMPTY_SLOT = -1;
    
    alignas(CACHE_LINE_SIZE) std::array<int32_t, TABLE_SIZE> table_;
    alignas(CACHE_LINE_SIZE) std::array<std::string, MAX_SYMBOLS> symbol_names_;
    std::atomic<int32_t> next_index_{0};

    // Fast hash function optimized for symbol strings
    FORCE_INLINE uint32_t hash_symbol(const std::string& symbol) const noexcept {
        uint32_t hash = 2166136261u; // FNV offset basis
        for (char c : symbol) {
            hash ^= static_cast<uint32_t>(c);
            hash *= 16777619u; // FNV prime
        }
        return hash & (TABLE_SIZE - 1);
    }

public:
    SymbolHashTable() {
        table_.fill(EMPTY_SLOT);
    }
    
    // Register symbol and return index (thread-safe)
    int32_t register_symbol(const std::string& symbol) {
        uint32_t hash = hash_symbol(symbol);
        
        // Linear probing for collision resolution
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            uint32_t index = (hash + i) & (TABLE_SIZE - 1);
            int32_t current = table_[index];
            
            if (current == EMPTY_SLOT) {
                // Try to claim this slot
                int32_t new_index = next_index_.fetch_add(1, std::memory_order_acq_rel);
                if (new_index >= MAX_SYMBOLS) {
                    next_index_.store(MAX_SYMBOLS, std::memory_order_release);
                    return -1; // Table full
                }
                
                symbol_names_[new_index] = symbol;
                table_[index] = new_index;
                return new_index;
            } else if (symbol_names_[current] == symbol) {
                return current; // Already exists
            }
        }
        
        return -1; // Table full
    }
    
    // Fast symbol lookup (lock-free read)
    FORCE_INLINE int32_t get_symbol_index(const std::string& symbol) const noexcept {
        uint32_t hash = hash_symbol(symbol);
        
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            uint32_t index = (hash + i) & (TABLE_SIZE - 1);
            int32_t symbol_index = table_[index];
            
            if (symbol_index == EMPTY_SLOT) {
                return -1; // Not found
            }
            
            if (symbol_names_[symbol_index] == symbol) {
                return symbol_index;
            }
        }
        
        return -1; // Not found
    }
    
    const std::string& get_symbol_name(int32_t index) const {
        return symbol_names_[index];
    }
};

// Performance statistics with atomic counters
struct alignas(CACHE_LINE_SIZE) PerformanceStats {
    std::atomic<uint64_t> operation_count{0};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<uint64_t> max_latency_ns{0};
    std::atomic<uint64_t> error_count{0};
    
    // Default constructor
    PerformanceStats() = default;
    
    // Copy constructor
    PerformanceStats(const PerformanceStats& other) noexcept
        : operation_count(other.operation_count.load())
        , total_latency_ns(other.total_latency_ns.load())
        , max_latency_ns(other.max_latency_ns.load())
        , error_count(other.error_count.load()) {}
    
    // Assignment operator
    PerformanceStats& operator=(const PerformanceStats& other) noexcept {
        if (this != &other) {
            operation_count.store(other.operation_count.load());
            total_latency_ns.store(other.total_latency_ns.load());
            max_latency_ns.store(other.max_latency_ns.load());
            error_count.store(other.error_count.load());
        }
        return *this;
    }
    
    void record_operation(uint64_t latency_ns) noexcept {
        operation_count.fetch_add(1, std::memory_order_relaxed);
        total_latency_ns.fetch_add(latency_ns, std::memory_order_relaxed);
        
        uint64_t current_max = max_latency_ns.load(std::memory_order_relaxed);
        while (latency_ns > current_max && 
               !max_latency_ns.compare_exchange_weak(current_max, latency_ns, std::memory_order_relaxed)) {
            // Retry until successful or latency is no longer max
        }
    }
    
    double get_average_latency_us() const noexcept {
        uint64_t ops = operation_count.load(std::memory_order_relaxed);
        if (ops == 0) return 0.0;
        return static_cast<double>(total_latency_ns.load(std::memory_order_relaxed)) / (ops * 1000.0);
    }
};

// Zero-allocation memory pool for orders and signals
template<typename T, size_t N>
class alignas(CACHE_LINE_SIZE) ZeroAllocPool {
private:
    alignas(CACHE_LINE_SIZE) std::array<T, N> pool_;
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> tail_{0};
    
public:
    T* allocate() noexcept {
        uint32_t idx = head_.fetch_add(1, std::memory_order_relaxed) & (N-1);
        return &pool_[idx];
    }
    
    void deallocate(T* ptr) noexcept {
        // No-op for ring buffer pool
    }
    
    void reset() noexcept {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }
    
    // 1. Add pool size getters for monitoring
    size_t capacity() const noexcept { return N; }
    size_t allocated() const noexcept {
        return (head_.load() - tail_.load()) & (N-1);
    }
};

// Actual struct definitions for pool types
struct alignas(64) Order {
    uint64_t order_id;
    uint64_t signal_id;
    int32_t symbol_id;
    int8_t side;           // 1 = buy, -1 = sell
    float price;
    uint32_t quantity;
    uint64_t timestamp_ns;
    uint8_t type;          // 0 = market, 1 = limit
    uint8_t status;        // 0 = pending, 1 = filled, 2 = cancelled
    uint8_t strategy_id;   // 0=momentum, 1=reversion, 2=breakout
    uint8_t tier;          // 1, 2, 3
    float stop_price;
    float take_profit_price;
    uint64_t filled_timestamp_ns;
    float filled_price;
    uint32_t filled_quantity;
    uint8_t padding[4];
};

struct alignas(32) TradingSignal {
    uint64_t signal_id;
    int32_t symbol_id;
    int8_t direction;      // -1, 0, 1
    float confidence;      // 0.0 to 1.0
    float position_size;   // 0.0 to 1.0
    float stop_loss;       // percentage
    float take_profit;     // percentage
    uint32_t hold_time_ms; // milliseconds
    uint64_t timestamp_ns;
    uint8_t strategy_id;   // 0=momentum, 1=reversion, 2=breakout
    uint8_t tier;          // 1, 2, 3
    uint8_t padding[6];
};

struct alignas(CACHE_LINE_SIZE) MarketTick {
    uint64_t timestamp_ns;
    int32_t symbol_id;
    float price;
    uint32_t volume;
    float bid;
    float ask;
    uint16_t bid_size;
    uint16_t ask_size;
    uint8_t tier;
    uint8_t padding[7]; // Pad to cache line boundary
};

// Ultra-fast memory manager with zero-copy operations
class UltraFastMemoryManager {
private:
    FastAllocator allocator_;
    SymbolHashTable symbol_table_;
    PerformanceStats stats_;
    
    // Component references (set by register_component)
    std::unordered_map<std::string, void*> components_;
    
    // Zero-allocation memory pools
    ZeroAllocPool<Order, 4096> order_pool_;
    ZeroAllocPool<TradingSignal, 4096> signal_pool_;
    ZeroAllocPool<MarketTick, 8192> tick_pool_;
    
    // NUMA-optimized allocator for huge pages
    std::unique_ptr<NUMAOptimizedAllocator> numa_allocator_;
    
    // Tier 1 symbol classification
    std::unordered_set<int32_t> tier1_symbols_;

public:
    explicit UltraFastMemoryManager(int numa_node = -1) {
        try {
            numa_allocator_ = std::make_unique<NUMAOptimizedAllocator>(numa_node);
            std::cout << "✅ NUMA-optimized memory manager initialized on node " << numa_node << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "⚠️ NUMA allocation failed, falling back to standard allocator: " << e.what() << std::endl;
            numa_allocator_ = nullptr;
        }
        
        // Initialize tier 1 symbols (high-frequency trading symbols)
        initialize_tier1_symbols();
    }
    
    // Register symbol with O(1) average case
    int32_t register_symbol(const std::string& symbol) {
        auto start = std::chrono::high_resolution_clock::now();
        int32_t result = symbol_table_.register_symbol(symbol);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        stats_.record_operation(latency_ns);
        
        return result;
    }
    
    // Ultra-fast symbol lookup
    FORCE_INLINE int32_t get_symbol_index(const std::string& symbol) const noexcept {
        return symbol_table_.get_symbol_index(symbol);
    }
    
    // Get symbol name by index
    const std::string& get_symbol_name(int32_t index) const {
        return symbol_table_.get_symbol_name(index);
    }
    
    // Register component for coordination
    template<typename T>
    void register_component(const std::string& name, T* component) {
        components_[name] = static_cast<void*>(component);
    }
    
    // Get component reference
    template<typename T>
    T* get_component(const std::string& name) const {
        auto it = components_.find(name);
        return (it != components_.end()) ? static_cast<T*>(it->second) : nullptr;
    }
    
    // Allocate cache-aligned memory
    template<typename T>
    T* allocate_aligned(size_t count) {
        return allocator_.template allocate_aligned<T>(count);
    }
    
    template<typename T>
    void deallocate(T* ptr, size_t count) {
        allocator_.deallocate(ptr, count);
    }
    
    // Zero-allocation pool methods
    template<typename T>
    T* allocate_order() noexcept {
        if constexpr (std::is_same_v<T, Order>) {
            return order_pool_.allocate();
        }
        return nullptr;
    }
    
    template<typename T>
    T* allocate_signal() noexcept {
        if constexpr (std::is_same_v<T, TradingSignal>) {
            return signal_pool_.allocate();
        }
        return nullptr;
    }
    
    template<typename T>
    T* allocate_tick() noexcept {
        if constexpr (std::is_same_v<T, MarketTick>) {
            return tick_pool_.allocate();
        }
        return nullptr;
    }
    
    // Reset pools for clean state
    void reset_pools() noexcept {
        order_pool_.reset();
        signal_pool_.reset();
        tick_pool_.reset();
    }
    
    // Performance monitoring
    const PerformanceStats& get_performance_stats() const noexcept {
        return stats_;
    }
    
    bool validate_performance(double target_latency_us = 1.0) const noexcept {
        return stats_.get_average_latency_us() <= target_latency_us;
    }
    
    // 3. Add memory usage statistics
    struct MemoryUsageStats {
        size_t numa_allocated;
        size_t orders_allocated;
        size_t signals_allocated;
        size_t ticks_allocated;
    };
    
    MemoryUsageStats get_memory_usage() const noexcept {
        return {
            numa_allocator_ ? numa_allocator_->get_allocated() : 0,
            order_pool_.allocated(),
            signal_pool_.allocated(),
            tick_pool_.allocated()
        };
    }
    
    // Add pool health monitoring to detect if pools are getting exhausted
    void check_pool_health() const {
        auto usage = get_memory_usage();
        
        // Warn if pools are getting full
        if (usage.orders_allocated > order_pool_.capacity() * 0.8) {
            std::cerr << "⚠️ Order pool 80% full!" << std::endl;
        }
        if (usage.signals_allocated > signal_pool_.capacity() * 0.8) {
            std::cerr << "⚠️ Signal pool 80% full!" << std::endl;
        }
        if (usage.ticks_allocated > tick_pool_.capacity() * 0.8) {
            std::cerr << "⚠️ Tick pool 80% full!" << std::endl;
        }
    }
    
    // Tier classification methods
    bool is_tier1_symbol(int32_t symbol_id) const noexcept {
        return tier1_symbols_.find(symbol_id) != tier1_symbols_.end();
    }
    
    void add_tier1_symbol(int32_t symbol_id) {
        tier1_symbols_.insert(symbol_id);
    }
    
    void remove_tier1_symbol(int32_t symbol_id) {
        tier1_symbols_.erase(symbol_id);
    }
    
    size_t get_tier1_count() const noexcept {
        return tier1_symbols_.size();
    }

private:
    void initialize_tier1_symbols() {
        // Initialize Tier 1 symbols from configuration
        // These are the most liquid and actively traded symbols from config.hpp
        for (size_t i = 0; i < APIConfig::NUM_TIER_1_SYMBOLS; ++i) {
            const std::string symbol = APIConfig::TIER_1_SYMBOLS[i];
            
            int32_t symbol_id = symbol_table_.get_symbol_index(symbol);
            if (symbol_id != -1) {
                tier1_symbols_.insert(symbol_id);
            } else {
                // Register the symbol if it doesn't exist and add to tier 1
                symbol_id = symbol_table_.register_symbol(symbol);
                if (symbol_id != -1) {
                    tier1_symbols_.insert(symbol_id);
                }
            }
        }
        
        std::cout << "✅ Initialized " << tier1_symbols_.size() << " Tier 1 symbols from config" << std::endl;
    }
};

// Global memory manager instance
extern std::unique_ptr<UltraFastMemoryManager> g_memory_manager;

// Factory function
inline UltraFastMemoryManager& get_memory_manager() {
    if (!g_memory_manager) {
        g_memory_manager = std::make_unique<UltraFastMemoryManager>();
    }
    return *g_memory_manager;
}

// Convenience functions
inline int32_t register_symbol(const std::string& symbol) {
    return get_memory_manager().register_symbol(symbol);
}

template<typename T>
inline void register_component(const std::string& name, T* component) {
    get_memory_manager().register_component(name, component);
}

inline bool validate_performance() {
    return get_memory_manager().validate_performance();
}

} // namespace hft
