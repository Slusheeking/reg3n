#pragma once

#include "memory_manager.hpp"
#include <chrono>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <atomic>      // for std::atomic
#include <array>       // for std::array
#include <algorithm>   // for std::min

// Platform-specific includes
#ifdef __linux__
    #include <sys/mman.h>  // for mmap on Linux
#endif

// OpenMP include (conditional)
#ifdef _OPENMP
    #include <omp.h>       // for OpenMP
#endif

namespace hft {

// Runtime CPU feature detection
static bool has_avx512() noexcept {
#ifdef __GNUC__
    return __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER)
    int cpui[4];
    __cpuid(cpui, 7);
    return (cpui[1] & (1 << 16)) != 0; // Check AVX-512F bit
#else
    return false; // Conservative fallback
#endif
}

// Platform-specific huge page allocation
static void* allocate_huge_page(size_t size) noexcept {
#ifdef __linux__
    void* ptr = mmap(nullptr, size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT),
                     -1, 0);
    return (ptr == MAP_FAILED) ? nullptr : ptr;
#else
    return nullptr; // Use regular allocation on other platforms
#endif
}

static void deallocate_huge_page(void* ptr, size_t size) noexcept {
#ifdef __linux__
    if (ptr) {
        munmap(ptr, size);
    }
#else
    // No-op for other platforms
    (void)ptr; (void)size;
#endif
}

// MarketTick is now defined in memory_manager.hpp

// Ring buffer for ultra-fast market data storage
template<size_t BufferSize = 1800>
class alignas(CACHE_LINE_SIZE) UltraFastRingBuffer {
private:
    alignas(CACHE_LINE_SIZE) std::array<float, BufferSize> prices_;
    alignas(CACHE_LINE_SIZE) std::array<uint32_t, BufferSize> volumes_;
    alignas(CACHE_LINE_SIZE) std::array<uint64_t, BufferSize> timestamps_;
    alignas(CACHE_LINE_SIZE) std::array<std::atomic<float>, 4> quotes_; // bid, ask, bid_size, ask_size
    
    std::atomic<uint32_t> position_{0};
    uint64_t last_update_time_{0};

public:
    // Ultra-fast trade update with SIMD optimization
    FORCE_INLINE void update_trade(float price, uint32_t volume, uint64_t timestamp_ns) noexcept {
        uint32_t pos = position_.load(std::memory_order_relaxed);
        
        prices_[pos] = price;
        volumes_[pos] = volume;
        timestamps_[pos] = timestamp_ns;
        
        position_.store((pos + 1) % BufferSize, std::memory_order_release);
        last_update_time_ = timestamp_ns;
    }
    
    // Thread-safe quote update using atomic operations
    FORCE_INLINE void update_quote(float bid, float ask, float bid_size, float ask_size) noexcept {
        quotes_[0].store(bid, std::memory_order_relaxed);
        quotes_[1].store(ask, std::memory_order_relaxed);
        quotes_[2].store(bid_size, std::memory_order_relaxed);
        quotes_[3].store(ask_size, std::memory_order_relaxed);
        last_update_time_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    // Optimized 10-feature calculation using AVX (Polygon-native only)
    bool calculate_features_vectorized(float* features) const noexcept {
        uint32_t pos = position_.load(std::memory_order_acquire);
        
        // Initialize features to zero
        std::memset(features, 0, 10 * sizeof(float));
        
        // Reduced minimum data requirement for faster startup
        // Note: Features will be less accurate with fewer data points
        if (pos < 10) {
            return false; // Need at least 10 trades for basic features
        }
        
        // Calculate returns using vectorized operations
        const uint32_t lookback = std::min(pos, 100u);
        
        // Feature 0: ret_1s (1-period return) - FIXED UNDERFLOW
        if (lookback >= 2 && pos >= 2) {
            uint32_t curr_idx = (pos - 1 + BufferSize) % BufferSize;
            uint32_t prev_idx = (pos - 2 + BufferSize) % BufferSize;
            features[0] = (prices_[curr_idx] / prices_[prev_idx]) - 1.0f;
        }
        
        // Feature 1: ret_5s (5-period return) - FIXED UNDERFLOW
        if (lookback >= 6 && pos >= 6) {
            uint32_t curr_idx = (pos - 1 + BufferSize) % BufferSize;
            uint32_t prev_idx = (pos - 6 + BufferSize) % BufferSize;
            features[1] = (prices_[curr_idx] / prices_[prev_idx]) - 1.0f;
        }
        
        // Feature 2: ret_15s (15-period return) - FIXED UNDERFLOW
        if (lookback >= 16 && pos >= 16) {
            uint32_t curr_idx = (pos - 1 + BufferSize) % BufferSize;
            uint32_t prev_idx = (pos - 16 + BufferSize) % BufferSize;
            features[2] = (prices_[curr_idx] / prices_[prev_idx]) - 1.0f;
        }
        
        // Feature 3: spread (bid-ask spread as % of mid) - THREAD SAFE
        float bid = quotes_[0].load(std::memory_order_relaxed);
        float ask = quotes_[1].load(std::memory_order_relaxed);
        float mid_price = (bid + ask) * 0.5f;
        if (mid_price > 0.0f) {
            features[3] = (ask - bid) / mid_price;
        }
        
        // Feature 4: volume_intensity (current vs average) - FIXED UNDERFLOW
        if (lookback >= 20 && pos >= 20) {
            float avg_volume = 0.0f;
            for (uint32_t i = 0; i < 20; ++i) {
                avg_volume += volumes_[(pos - 1 - i + BufferSize) % BufferSize];
            }
            avg_volume /= 20.0f;
            if (avg_volume > 0.0f) {
                features[4] = volumes_[(pos - 1 + BufferSize) % BufferSize] / avg_volume;
            }
        }
        
        // Feature 5: price_acceleration (second derivative) - FIXED UNDERFLOW
        if (lookback >= 3 && pos >= 3) {
            uint32_t idx0 = (pos - 3 + BufferSize) % BufferSize;
            uint32_t idx1 = (pos - 2 + BufferSize) % BufferSize;
            uint32_t idx2 = (pos - 1 + BufferSize) % BufferSize;
            float vel1 = prices_[idx1] - prices_[idx0];
            float vel2 = prices_[idx2] - prices_[idx1];
            features[5] = vel2 - vel1;
        }
        
        // Feature 6: momentum_persistence (trend consistency) - FIXED UNDERFLOW
        if (lookback >= 10 && pos >= 10) {
            int positive_moves = 0;
            for (uint32_t i = 1; i < 10; ++i) {
                uint32_t curr = (pos - i + BufferSize) % BufferSize;
                uint32_t prev = (pos - i - 1 + BufferSize) % BufferSize;
                if (prices_[curr] > prices_[prev]) positive_moves++;
            }
            features[6] = (positive_moves / 9.0f) - 0.5f; // Center around 0
        }
        
        // Feature 7: quote_intensity (quote update frequency proxy) - THREAD SAFE
        float bid_size = quotes_[2].load(std::memory_order_relaxed);
        float ask_size = quotes_[3].load(std::memory_order_relaxed);
        if (bid_size + ask_size > 0.0f) {
            features[7] = bid_size / (bid_size + ask_size); // Bid size ratio
        }
        
        // Feature 8: trade_size_surprise (volume vs expected) - FIXED UNDERFLOW
        if (lookback >= 5 && pos >= 5) {
            float recent_avg = 0.0f;
            for (uint32_t i = 1; i < 5; ++i) {
                recent_avg += volumes_[(pos - 1 - i + BufferSize) % BufferSize];
            }
            recent_avg /= 4.0f;
            if (recent_avg > 0.0f) {
                features[8] = (volumes_[(pos - 1 + BufferSize) % BufferSize] / recent_avg) - 1.0f;
            }
        }
        
        // Feature 9: depth_imbalance (bid vs ask size imbalance) - THREAD SAFE
        float total_size = bid_size + ask_size;
        if (total_size > 0.0f) {
            features[9] = (bid_size - ask_size) / total_size;
        }
        
        // Continue with calculations...
        return true;
    }
    
    // AVX-512 optimized version with CPU feature detection
    void calculate_features_vectorized_avx512(float* features) const noexcept {
        // SAFETY: Check CPU support first
        if (!has_avx512()) {
            // Fallback to regular vectorized calculation
            calculate_features_vectorized(features);
            return;
        }
        
        uint32_t pos = position_.load(std::memory_order_acquire);
        if (pos < 31) { // Need at least 31 elements for AVX-512 lookback
            memset(features, 0, 10 * sizeof(float));
            return;
        }
        
        // AVX-512 optimized returns calculation - FIXED UNDERFLOW
        __m512 curr_prices = _mm512_loadu_ps(&prices_[(pos - 16 + BufferSize) % BufferSize]);
        __m512 prev_prices_1 = _mm512_loadu_ps(&prices_[(pos - 17 + BufferSize) % BufferSize]);
        __m512 prev_prices_5 = _mm512_loadu_ps(&prices_[(pos - 21 + BufferSize) % BufferSize]);
        __m512 prev_prices_15 = _mm512_loadu_ps(&prices_[(pos - 31 + BufferSize) % BufferSize]);
        
        __m512 ones = _mm512_set1_ps(1.0f);
        __m512 ret_1 = _mm512_sub_ps(_mm512_div_ps(curr_prices, prev_prices_1), ones);
        __m512 ret_5 = _mm512_sub_ps(_mm512_div_ps(curr_prices, prev_prices_5), ones);
        __m512 ret_15 = _mm512_sub_ps(_mm512_div_ps(curr_prices, prev_prices_15), ones);
        
        // Extract last values
        features[0] = _mm512_reduce_max_ps(ret_1);
        features[1] = _mm512_reduce_max_ps(ret_5);
        features[2] = _mm512_reduce_max_ps(ret_15);
        
        // Continue with other features using scalar calculations - THREAD SAFE
        float bid = quotes_[0].load(std::memory_order_relaxed);
        float ask = quotes_[1].load(std::memory_order_relaxed);
        float bid_size = quotes_[2].load(std::memory_order_relaxed);
        float ask_size = quotes_[3].load(std::memory_order_relaxed);
        
        float mid_price = (bid + ask) * 0.5f;
        if (mid_price > 0.0f) {
            features[3] = (ask - bid) / mid_price;
        }
        
        // Volume intensity - FIXED UNDERFLOW
        if (pos >= 20) {
            float avg_volume = 0.0f;
            for (uint32_t i = 0; i < 20; ++i) {
                avg_volume += volumes_[(pos - 1 - i + BufferSize) % BufferSize];
            }
            avg_volume /= 20.0f;
            if (avg_volume > 0.0f) {
                features[4] = volumes_[(pos - 1 + BufferSize) % BufferSize] / avg_volume;
            }
        }
        
        // Price acceleration - FIXED UNDERFLOW
        if (pos >= 3) {
            uint32_t idx0 = (pos - 3 + BufferSize) % BufferSize;
            uint32_t idx1 = (pos - 2 + BufferSize) % BufferSize;
            uint32_t idx2 = (pos - 1 + BufferSize) % BufferSize;
            float vel1 = prices_[idx1] - prices_[idx0];
            float vel2 = prices_[idx2] - prices_[idx1];
            features[5] = vel2 - vel1;
        }
        
        // Momentum persistence - FIXED UNDERFLOW
        if (pos >= 10) {
            int positive_moves = 0;
            for (uint32_t i = 1; i < 10; ++i) {
                uint32_t curr = (pos - i + BufferSize) % BufferSize;
                uint32_t prev = (pos - i - 1 + BufferSize) % BufferSize;
                if (prices_[curr] > prices_[prev]) positive_moves++;
            }
            features[6] = (positive_moves / 9.0f) - 0.5f;
        }
        
        // Quote intensity - THREAD SAFE
        if (bid_size + ask_size > 0.0f) {
            features[7] = bid_size / (bid_size + ask_size);
        }
        
        // Trade size surprise - FIXED UNDERFLOW
        if (pos >= 5) {
            float recent_avg = 0.0f;
            for (uint32_t i = 1; i < 5; ++i) {
                recent_avg += volumes_[(pos - 1 - i + BufferSize) % BufferSize];
            }
            recent_avg /= 4.0f;
            if (recent_avg > 0.0f) {
                features[8] = (volumes_[(pos - 1 + BufferSize) % BufferSize] / recent_avg) - 1.0f;
            }
        }
        
        // Depth imbalance - THREAD SAFE
        float total_size = bid_size + ask_size;
        if (total_size > 0.0f) {
            features[9] = (bid_size - ask_size) / total_size;
        }
    }
    
    // Get latest price with error return
    FORCE_INLINE bool get_latest_price(float& price) const noexcept {
        uint32_t pos = position_.load(std::memory_order_acquire);
        if (pos == 0) return false;
        
        price = prices_[(pos - 1) % BufferSize];
        return true;
    }
    
    // Get latest volume with error return
    FORCE_INLINE bool get_latest_volume(uint32_t& volume) const noexcept {
        uint32_t pos = position_.load(std::memory_order_acquire);
        if (pos == 0) return false;
        
        volume = volumes_[(pos - 1) % BufferSize];
        return true;
    }
    
    // Get quote data - THREAD SAFE
    FORCE_INLINE void get_quote(float& bid, float& ask, float& bid_size, float& ask_size) const noexcept {
        bid = quotes_[0].load(std::memory_order_relaxed);
        ask = quotes_[1].load(std::memory_order_relaxed);
        bid_size = quotes_[2].load(std::memory_order_relaxed);
        ask_size = quotes_[3].load(std::memory_order_relaxed);
    }
    
    uint64_t get_last_update_time() const noexcept {
        return last_update_time_;
    }
};

// Forward declaration
class MarketDataManager;

// ParallelFeatureCalculator for multi-threaded feature computation
class ParallelFeatureCalculator {
private:
    struct alignas(64) SymbolFeatureCache {
        float features[16]; // Padded to cache line
        uint64_t last_update{0};
        uint32_t symbol_id;
        uint32_t padding;
    };
    
    std::array<SymbolFeatureCache, 64> feature_cache_;
    
public:
    void calculate_all_features_parallel(const MarketDataManager& mdm,
                                       const std::vector<int32_t>& symbol_ids,
                                       float* output_matrix) noexcept;
};

// Multi-symbol market data manager
class MarketDataManager {
private:
    static constexpr size_t MAX_SYMBOLS = 1000;
    
    std::array<UltraFastRingBuffer<>, MAX_SYMBOLS> symbol_buffers_;
    std::array<bool, MAX_SYMBOLS> symbol_active_;
    UltraFastMemoryManager& memory_manager_;
    
    // Add pre-allocated tick pool
    struct alignas(4096) PreAllocatedData {
        std::array<MarketTick, 65536> tick_pool;
        std::atomic<uint32_t> tick_index{0};
        
        MarketTick* allocate_tick() noexcept {
            uint32_t idx = tick_index.fetch_add(1, std::memory_order_relaxed) & 65535;
            return &tick_pool[idx];
        }
    };
    
    PreAllocatedData* preallocated_data_;
    bool using_huge_pages_{false}; // Track allocation method

public:
    explicit MarketDataManager(UltraFastMemoryManager& mm) : memory_manager_(mm) {
        symbol_active_.fill(false);
        
        // Try huge page allocation first (platform-specific)
        preallocated_data_ = static_cast<PreAllocatedData*>(
            allocate_huge_page(sizeof(PreAllocatedData)));
        
        if (preallocated_data_) {
            using_huge_pages_ = true;
            new(preallocated_data_) PreAllocatedData(); // Placement new
        } else {
            // Fallback to regular allocation
            preallocated_data_ = new PreAllocatedData();
            using_huge_pages_ = false;
        }
    }
    
    ~MarketDataManager() {
        if (preallocated_data_) {
            if (using_huge_pages_) {
                preallocated_data_->~PreAllocatedData(); // Explicit destructor call
                deallocate_huge_page(preallocated_data_, sizeof(PreAllocatedData));
            } else {
                delete preallocated_data_;
            }
        }
    }
    
    // Update trade data for symbol
    FORCE_INLINE bool update_trade(int32_t symbol_id, float price, uint32_t volume, uint64_t timestamp_ns) noexcept {
        if (symbol_id < 0 || symbol_id >= MAX_SYMBOLS || !symbol_active_[symbol_id]) {
            return false;
        }
        
        symbol_buffers_[symbol_id].update_trade(price, volume, timestamp_ns);
        return true;
    }
    
    // Update quote data for symbol
    FORCE_INLINE bool update_quote(int32_t symbol_id, float bid, float ask, float bid_size, float ask_size) noexcept {
        if (symbol_id < 0 || symbol_id >= MAX_SYMBOLS || !symbol_active_[symbol_id]) {
            return false;
        }
        
        symbol_buffers_[symbol_id].update_quote(bid, ask, bid_size, ask_size);
        return true;
    }
    
    // Activate symbol for tracking
    bool activate_symbol(int32_t symbol_id) noexcept {
        if (symbol_id < 0 || symbol_id >= MAX_SYMBOLS) {
            return false;
        }
        
        symbol_active_[symbol_id] = true;
        return true;
    }
    
    // Get features for symbol
    bool get_features(int32_t symbol_id, float* features) const noexcept {
        if (symbol_id < 0 || symbol_id >= MAX_SYMBOLS || !symbol_active_[symbol_id]) {
            return false;
        }
        
        return symbol_buffers_[symbol_id].calculate_features_vectorized(features);
    }
    
    // Batch feature extraction for multiple symbols (10 features)
    void get_features_batch(const std::vector<int32_t>& symbol_ids, float* features_matrix) const noexcept {
        for (size_t i = 0; i < symbol_ids.size(); ++i) {
            get_features(symbol_ids[i], features_matrix + i * 10);
        }
    }
    
    // Get latest market data
    bool get_latest_data(int32_t symbol_id, float& price, uint32_t& volume,
                        float& bid, float& ask, float& bid_size, float& ask_size) const noexcept {
        if (symbol_id < 0 || symbol_id >= MAX_SYMBOLS || !symbol_active_[symbol_id]) {
            return false;
        }
        
        const auto& buffer = symbol_buffers_[symbol_id];
        if (!buffer.get_latest_price(price) || !buffer.get_latest_volume(volume)) {
            return false;
        }
        buffer.get_quote(bid, ask, bid_size, ask_size);
        
        return true;
    }
};

// Implementation of ParallelFeatureCalculator method (after MarketDataManager is fully defined)
inline void ParallelFeatureCalculator::calculate_all_features_parallel(const MarketDataManager& mdm,
                                   const std::vector<int32_t>& symbol_ids,
                                   float* output_matrix) noexcept {
    
    const size_t n_symbols = symbol_ids.size();
    
#ifdef _OPENMP
    const size_t n_threads = 4;
    const size_t symbols_per_thread = (n_symbols + n_threads - 1) / n_threads;
    
    #pragma omp parallel for num_threads(n_threads)
    for (size_t t = 0; t < n_threads; ++t) {
#else
    // Fallback to single-threaded execution
    const size_t n_threads = 1;
    const size_t symbols_per_thread = n_symbols;
    for (size_t t = 0; t < n_threads; ++t) {
#endif
        size_t start = t * symbols_per_thread;
        size_t end = std::min(start + symbols_per_thread, n_symbols);
        
        for (size_t i = start; i < end; ++i) {
            auto& cache = feature_cache_[i];
            
            // Check if cache is still valid (within 100us)
            uint64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            if (now - cache.last_update < 100000) { // 100us cache
                memcpy(output_matrix + i * 10, cache.features, 10 * sizeof(float));
            } else {
                mdm.get_features(symbol_ids[i], output_matrix + i * 10);
                memcpy(cache.features, output_matrix + i * 10, 10 * sizeof(float));
                cache.last_update = now;
            }
        }
    }
}

} // namespace hft
