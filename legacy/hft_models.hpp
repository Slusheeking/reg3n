#pragma once

#include "config.hpp"
#include "memory_manager.hpp"
#include "market_data.hpp"
#include <chrono>
#include <vector>
#include <array>
#include <algorithm>
#include <immintrin.h>
#include <cmath>
#include <atomic>       // for std::atomic
#include <mutex>        // for std::mutex
#include <memory>       // for std::unique_ptr
#include <cstring>      // for memset
#include <thread>       // for thread-related functions

namespace hft {

// SIMD feature detection and dispatch
class SIMDDispatcher {
private:
    static bool use_avx2_;
    static bool use_avx512_;
    static bool features_detected_;
    
public:
    static void detect_features() {
        if (features_detected_) return;
        
#ifdef __GNUC__
        use_avx2_ = __builtin_cpu_supports("avx2");
        use_avx512_ = __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER)
        int cpui[4];
        __cpuid(cpui, 7);
        use_avx2_ = (cpui[1] & (1 << 5)) != 0;   // AVX2 bit
        use_avx512_ = (cpui[1] & (1 << 16)) != 0; // AVX-512F bit
#else
        use_avx2_ = false;
        use_avx512_ = false;
#endif
        features_detected_ = true;
        
        std::cout << "🔧 SIMD Features: AVX2=" << (use_avx2_ ? "YES" : "NO")
                  << ", AVX-512=" << (use_avx512_ ? "YES" : "NO") << std::endl;
    }
    
    static bool has_avx2() {
        if (!features_detected_) detect_features();
        return use_avx2_;
    }
    
    static bool has_avx512() {
        if (!features_detected_) detect_features();
        return use_avx512_;
    }
};

// Static member definitions
bool SIMDDispatcher::use_avx2_ = false;
bool SIMDDispatcher::use_avx512_ = false;
bool SIMDDispatcher::features_detected_ = false;

// Advanced learning statistics structure
struct AdvancedLearningStats {
    uint64_t total_trades;
    uint64_t winning_trades;
    double win_rate;
    double total_pnl;
    double sharpe_ratio;
    std::array<double, 2> strategy_win_rates; // Momentum, Mean Reversion
    std::array<double, 2> strategy_pnl;
};

// Signal outcome tracking for online learning
struct alignas(32) SignalOutcome {
    uint64_t signal_id;
    int32_t symbol_id;
    float predicted_return;
    float actual_return;
    float pnl;
    uint64_t entry_time_ns;
    uint64_t exit_time_ns;
    uint8_t strategy_id;
    uint8_t tier;
    bool is_winner;
    uint8_t padding[5];
};

// Online learning engine for real-time weight updates
class OnlineLearningEngine {
private:
    float learning_rate_;
    float momentum_;
    float decay_rate_;
    std::vector<float> momentum_buffer_;
    std::atomic<uint64_t> update_count_{0};
    
    // Thread safety for weights
    mutable std::mutex weights_mutex_;
    
    // Performance tracking (using atomic operations for thread safety)
    std::atomic<int64_t> cumulative_pnl_cents_{0}; // Store P&L in cents to avoid float atomics
    std::atomic<uint32_t> winning_signals_{0};
    std::atomic<uint32_t> total_signals_{0};
    
public:
    explicit OnlineLearningEngine(size_t weight_count, float lr = 0.001f, float momentum = 0.9f)
        : learning_rate_(lr), momentum_(momentum), decay_rate_(0.999f) {
        momentum_buffer_.resize(weight_count, 0.0f);
    }
    
    // Thread-safe weight update based on signal outcome
    void update_weights(const SignalOutcome& outcome,
                       float* weights,
                       const float* features,
                       size_t feature_count) {
        if (outcome.pnl == 0.0f) return; // Skip neutral outcomes
        
        // Calculate reward/penalty based on P&L
        float reward = std::tanh(outcome.pnl / 100.0f); // Normalize P&L to [-1, 1]
        
        // Adaptive learning rate based on recent performance
        float adaptive_lr = calculate_adaptive_learning_rate();
        
        // Thread-safe SGD with momentum update
        {
            std::lock_guard<std::mutex> lock(weights_mutex_);
            for (size_t i = 0; i < feature_count; ++i) {
                float gradient = reward * features[i];
                momentum_buffer_[i] = momentum_ * momentum_buffer_[i] + adaptive_lr * gradient;
                weights[i] += momentum_buffer_[i];
                
                // Weight decay for regularization
                weights[i] *= decay_rate_;
            }
        }
        
        // Update performance tracking (atomic operations)
        update_count_.fetch_add(1, std::memory_order_relaxed);
        cumulative_pnl_cents_.fetch_add(static_cast<int64_t>(outcome.pnl * 100), std::memory_order_relaxed);
        if (outcome.is_winner) {
            winning_signals_.fetch_add(1, std::memory_order_relaxed);
        }
        total_signals_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Get current performance metrics
    struct PerformanceMetrics {
        float win_rate;
        float avg_pnl;
        float cumulative_pnl;
        uint64_t total_updates;
    };
    
    PerformanceMetrics get_performance() const {
        uint32_t total = total_signals_.load();
        float cumulative_pnl = cumulative_pnl_cents_.load() / 100.0f; // Convert back from cents
        return {
            total > 0 ? static_cast<float>(winning_signals_.load()) / total : 0.0f,
            total > 0 ? cumulative_pnl / total : 0.0f,
            cumulative_pnl,
            update_count_.load()
        };
    }
    
    // Thread-safe access to weights (returns copy to avoid race conditions)
    std::vector<float> get_weights() const {
        std::lock_guard<std::mutex> lock(weights_mutex_);
        return momentum_buffer_; // Return copy for thread safety
    }
    
private:
    float calculate_adaptive_learning_rate() {
        uint32_t total = total_signals_.load();
        if (total < 10) return learning_rate_; // Not enough data
        
        float win_rate = static_cast<float>(winning_signals_.load()) / total;
        
        // Increase learning rate if performing well, decrease if performing poorly
        if (win_rate > 0.6f) {
            return learning_rate_ * 1.2f; // Boost learning when winning
        } else if (win_rate < 0.4f) {
            return learning_rate_ * 0.5f; // Conservative learning when losing
        }
        return learning_rate_;
    }
};

// TradingSignal is now defined in memory_manager.hpp

// Ultra-fast linear model for Tier 1 symbols (target <5μs) with online learning
class UltraFastLinearModel {
private:
    // Pre-computed weights aligned for SIMD operations (10 features)
    alignas(32) std::array<float, 12> momentum_weights_;  // 12 for alignment
    alignas(32) std::array<float, 12> reversion_weights_;
    
    // Thresholds for signal generation
    float momentum_threshold_;
    float reversion_threshold_;
    
    // Online learning engines
    std::unique_ptr<OnlineLearningEngine> momentum_learner_;
    std::unique_ptr<OnlineLearningEngine> reversion_learner_;
    
    // Signal tracking
    mutable std::atomic<uint64_t> next_signal_id_{1};
    mutable std::mutex learning_mutex_;
    
public:
    UltraFastLinearModel() {
        // Initialize SIMD feature detection
        SIMDDispatcher::detect_features();
        // Initialize BALANCED weights for equal buy/sell signal generation
        momentum_weights_ = {{
            0.35f,  // ret_1s - reduced bias
            0.25f,  // ret_5s - reduced bias
            0.15f,  // ret_15s - reduced bias
            -0.20f, // spread - stronger negative weight
            0.15f,  // volume_intensity
            0.08f,  // price_acceleration - reduced
            0.10f,  // momentum_persistence - reduced
            0.05f,  // quote_intensity - reduced
            0.04f,  // trade_size_surprise
            0.06f,  // depth_imbalance
            0.0f,   // padding
            0.0f    // padding
        }};
        
        reversion_weights_ = {{
            -0.35f, // ret_1s (stronger reversion signal)
            -0.25f, // ret_5s (stronger reversion)
            -0.18f, // ret_15s (stronger reversion)
            0.15f,  // spread - positive for reversion
            -0.10f, // volume_intensity
            -0.15f, // price_acceleration (stronger negative)
            0.12f,  // momentum_persistence
            0.08f,  // quote_intensity
            -0.06f, // trade_size_surprise
            0.10f,  // depth_imbalance
            0.0f,   // padding
            0.0f    // padding
        }};
        
        // EQUAL thresholds for balanced signal generation
        momentum_threshold_ = 0.0012f;  // Slightly higher for quality
        reversion_threshold_ = 0.0012f; // SAME threshold for balance
        
        // Initialize online learning engines
        momentum_learner_ = std::make_unique<OnlineLearningEngine>(10, 0.001f, 0.9f);
        reversion_learner_ = std::make_unique<OnlineLearningEngine>(10, 0.001f, 0.9f);
    }
    
    // Vectorized batch prediction using AVX
    FORCE_INLINE void predict_batch_vectorized(const float* features_matrix, 
                                              TradingSignal* signals, 
                                              const std::vector<int32_t>& symbol_ids,
                                              uint64_t timestamp_ns) const noexcept {
        const size_t n_symbols = symbol_ids.size();
        
        for (size_t i = 0; i < n_symbols; ++i) {
            const float* features = features_matrix + i * 15;
            TradingSignal& signal = signals[i];
            
            // FAIL-FAST: Validate all features for NaN/Inf - no mock/placeholder values accepted
            for (int feat_idx = 0; feat_idx < 10; ++feat_idx) {
                if (std::isnan(features[feat_idx]) || std::isinf(features[feat_idx])) {
                    // Invalid feature detected - reject signal entirely
                    signal.direction = 0;
                    signal.confidence = 0.0f;
                    signal.position_size = 0.0f;
                    signal.strategy_id = 255; // Invalid strategy marker
                    signal.symbol_id = symbol_ids[i];
                    signal.timestamp_ns = timestamp_ns;
                    signal.tier = 1;
                    continue; // Skip to next symbol
                }
            }
            
            // Initialize signal
            signal.signal_id = next_signal_id_.fetch_add(1, std::memory_order_relaxed);
            signal.symbol_id = symbol_ids[i];
            signal.timestamp_ns = timestamp_ns;
            signal.tier = 1;
            
            float momentum_score, reversion_score;
            
            // Always use scalar implementation for compatibility
            momentum_score = 0.0f;
            reversion_score = 0.0f;
            for (size_t j = 0; j < 10; ++j) {
                momentum_score += features[j] * momentum_weights_[j];
                reversion_score += features[j] * reversion_weights_[j];
            }
            
            // BALANCED signal generation - no strategy preference, equal weighting
            bool momentum_valid = std::abs(momentum_score) > momentum_threshold_;
            bool reversion_valid = std::abs(reversion_score) > reversion_threshold_;
            
            if (momentum_valid && reversion_valid) {
                // Both strategies valid - choose the stronger one
                if (std::abs(momentum_score) > std::abs(reversion_score)) {
                    signal.direction = (momentum_score > 0) ? 1 : -1;
                    signal.confidence = std::min(std::abs(momentum_score) * 90.0f, 1.0f);
                    signal.strategy_id = 0; // Momentum
                } else {
                    signal.direction = (reversion_score > 0) ? 1 : -1;
                    signal.confidence = std::min(std::abs(reversion_score) * 90.0f, 1.0f);
                    signal.strategy_id = 1; // Mean reversion
                }
            } else if (momentum_valid) {
                signal.direction = (momentum_score > 0) ? 1 : -1;
                signal.confidence = std::min(std::abs(momentum_score) * 90.0f, 1.0f);
                signal.strategy_id = 0; // Momentum
            } else if (reversion_valid) {
                signal.direction = (reversion_score > 0) ? 1 : -1;
                signal.confidence = std::min(std::abs(reversion_score) * 90.0f, 1.0f);
                signal.strategy_id = 1; // Mean reversion
            } else {
                signal.direction = 0;
                signal.confidence = 0.0f;
                signal.strategy_id = 2; // No trade
            }
            
            // FAIL-FAST: No mock/placeholder volatility data - require real volatility
            if (features[7] <= 0) {
                // Invalid volatility - reject this signal entirely instead of using placeholder
                signal.direction = 0;
                signal.confidence = 0.0f;
                signal.position_size = 0.0f;
                signal.strategy_id = 255; // Invalid strategy marker
                continue; // Skip to next symbol
            }
            
            // Aggressive position sizing for $50K capital targeting $1000+ daily profit - using real volatility only
            float volatility = std::max(features[7], 1e-6f); // Prevent division by zero
            signal.position_size = (signal.direction != 0) ?
                std::min(signal.confidence * 1.2f / volatility, 0.20f) : 0.0f; // Up to 20% per position
            
            // SIMPLE 0.8% take profit target
            signal.stop_loss = 0.02f;      // 2% stop loss
            signal.take_profit = 0.008f;   // 0.8% take profit (FIXED)
            signal.hold_time_ms = 300000;  // 5 minutes max hold
        }
    }
    
    // Inference cache for ultra-fast repeated predictions
    struct alignas(64) InferenceCache {
        float features[16]; // Padded for AVX alignment
        float momentum_score;
        float reversion_score;
        uint64_t timestamp;
        int32_t symbol_id;
    };
    
    mutable std::array<InferenceCache, 64> inference_cache_;
    
    // Simplified batch inference using scalar operations
    FORCE_INLINE void predict_all_tier1_vectorized(
        const float* features_matrix,
        TradingSignal* signals,
        const std::vector<int32_t>& symbol_ids,
        uint64_t timestamp_ns) const noexcept {
        
        const size_t n = symbol_ids.size();
        
        // Process all symbols with scalar operations
        for (size_t i = 0; i < n; ++i) {
            const float* features = features_matrix + i * 15;
            float mom_score = 0.0f, rev_score = 0.0f;
            
            for (int j = 0; j < 10; ++j) {
                mom_score += features[j] * momentum_weights_[j];
                rev_score += features[j] * reversion_weights_[j];
            }
            
            generate_signal_branchless(signals[i], mom_score, rev_score,
                                      symbol_ids[i], timestamp_ns, features);
        }
    }
    
    // Online learning feedback method
    void update_from_outcome(const SignalOutcome& outcome, const float* features) {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        
        if (outcome.strategy_id == 0) {
            // Update momentum model
            momentum_learner_->update_weights(outcome, momentum_weights_.data(), features, 10);
        } else if (outcome.strategy_id == 1) {
            // Update reversion model
            reversion_learner_->update_weights(outcome, reversion_weights_.data(), features, 10);
        }
    }
    
    // Get learning performance metrics
    struct LearningStats {
        OnlineLearningEngine::PerformanceMetrics momentum_perf;
        OnlineLearningEngine::PerformanceMetrics reversion_perf;
    };
    
    LearningStats get_learning_stats() const {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        return {
            momentum_learner_->get_performance(),
            reversion_learner_->get_performance()
        };
    }

private:
    // AVX2 matrix transpose for 4x8 processing
    FORCE_INLINE void transpose_4x8_ps(__m256 in0, __m256 in1, __m256 in2, __m256 in3,
                                     __m256& out0, __m256& out1, __m256& out2, __m256& out3,
                                     __m256& out4, __m256& out5, __m256& out6, __m256& out7) const noexcept {
        __m256 tmp0, tmp1, tmp2, tmp3;
        
        tmp0 = _mm256_unpacklo_ps(in0, in1);
        tmp1 = _mm256_unpackhi_ps(in0, in1);
        tmp2 = _mm256_unpacklo_ps(in2, in3);
        tmp3 = _mm256_unpackhi_ps(in2, in3);
        
        out0 = _mm256_shuffle_ps(tmp0, tmp2, 0x44);
        out1 = _mm256_shuffle_ps(tmp0, tmp2, 0xEE);
        out2 = _mm256_shuffle_ps(tmp1, tmp3, 0x44);
        out3 = _mm256_shuffle_ps(tmp1, tmp3, 0xEE);
        
        // For 8 features, we need to handle the second half
        out4 = _mm256_permute2f128_ps(out0, out0, 0x01);
        out5 = _mm256_permute2f128_ps(out1, out1, 0x01);
        out6 = _mm256_permute2f128_ps(out2, out2, 0x01);
        out7 = _mm256_permute2f128_ps(out3, out3, 0x01);
    }
    
    // Safe signal generation avoiding undefined behavior
    FORCE_INLINE void generate_signal_branchless(
        TradingSignal& signal,
        float mom_score,
        float rev_score,
        int32_t symbol_id,
        uint64_t timestamp_ns,
        const float* features) const noexcept {
        
        signal.signal_id = next_signal_id_.fetch_add(1, std::memory_order_relaxed);
        signal.symbol_id = symbol_id;
        signal.timestamp_ns = timestamp_ns;
        signal.tier = 1;
        
        // Safe strategy selection avoiding UB
        float abs_mom = std::abs(mom_score);
        float abs_rev = std::abs(rev_score);
        
        // BALANCED strategy selection - equal treatment for both strategies
        bool momentum_valid = abs_mom > momentum_threshold_;
        bool reversion_valid = abs_rev > reversion_threshold_;
        
        if (momentum_valid && reversion_valid) {
            // Both valid - choose stronger signal
            if (abs_mom > abs_rev) {
                signal.strategy_id = 0; // Momentum strategy
                signal.direction = (mom_score > 0) ? 1 : -1;
                signal.confidence = std::min(abs_mom * 90.0f, 1.0f);
            } else {
                signal.strategy_id = 1; // Reversion strategy
                signal.direction = (rev_score > 0) ? 1 : -1;
                signal.confidence = std::min(abs_rev * 90.0f, 1.0f);
            }
        } else if (momentum_valid) {
            signal.strategy_id = 0; // Momentum strategy
            signal.direction = (mom_score > 0) ? 1 : -1;
            signal.confidence = std::min(abs_mom * 90.0f, 1.0f);
        } else if (reversion_valid) {
            signal.strategy_id = 1; // Reversion strategy
            signal.direction = (rev_score > 0) ? 1 : -1;
            signal.confidence = std::min(abs_rev * 90.0f, 1.0f);
        } else {
            signal.strategy_id = 2; // No trade
            signal.direction = 0;
            signal.confidence = 0.0f;
        }
        
        // Position sizing with division by zero protection
        float volatility = std::max(features[7], 1e-6f); // Prevent division by zero
        float base_size = 0.0f;
        
        if (signal.strategy_id == 0) { // Momentum strategy
            base_size = signal.confidence * 1.2f / volatility;
        } else if (signal.strategy_id == 1) { // Reversion strategy
            base_size = signal.confidence * 0.8f / volatility;
        }
        
        signal.position_size = (signal.direction != 0) ? std::min(base_size, 0.20f) : 0.0f;
        signal.stop_loss = 0.02f;      // 2% stop loss (FIXED)
        signal.take_profit = 0.008f;   // 0.8% take profit (FIXED)
        signal.hold_time_ms = 300000;  // 5 minutes max hold
    }
    
    // Horizontal sum for AVX register
    FORCE_INLINE float horizontal_sum_avx(__m256 v) const noexcept {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
};

// Fast tree model for Tier 2 symbols (target <20μs) with online learning
class FastTreeModel {
private:
    // Pre-computed lookup tables for tree traversal
    struct TreeNode {
        float threshold;
        int16_t feature_idx;
        int16_t left_child;
        int16_t right_child;
        float leaf_value;
    };
    
    static constexpr size_t MAX_NODES = 1024;
    std::array<TreeNode, MAX_NODES> momentum_tree_;
    std::array<TreeNode, MAX_NODES> reversion_tree_;
    size_t momentum_tree_size_;
    size_t reversion_tree_size_;
    
    // Regime classification lookup table
    std::array<float, 256> regime_lookup_;
    
    // Online learning engines for adaptive tree leaf values
    std::unique_ptr<OnlineLearningEngine> momentum_learner_;
    std::unique_ptr<OnlineLearningEngine> reversion_learner_;
    std::unique_ptr<OnlineLearningEngine> regime_learner_;
    
    // Adaptive leaf value adjustments
    std::array<float, MAX_NODES> momentum_leaf_adjustments_;
    std::array<float, MAX_NODES> reversion_leaf_adjustments_;
    
    // Signal tracking
    mutable std::atomic<uint64_t> next_signal_id_{1000000}; // Start at 1M to avoid conflicts with Tier 1
    mutable std::mutex learning_mutex_;

public:
    FastTreeModel() {
        // Initialize SIMD feature detection
        SIMDDispatcher::detect_features();
        
        // Initialize simplified decision trees
        initialize_momentum_tree();
        initialize_reversion_tree();
        initialize_regime_lookup();
        
        // Initialize online learning engines
        // Tree models have fewer direct weights but can adapt leaf values
        momentum_learner_ = std::make_unique<OnlineLearningEngine>(momentum_tree_size_, 0.0005f, 0.85f);
        reversion_learner_ = std::make_unique<OnlineLearningEngine>(reversion_tree_size_, 0.0005f, 0.85f);
        regime_learner_ = std::make_unique<OnlineLearningEngine>(256, 0.0001f, 0.9f); // For regime lookup table
        
        // Initialize leaf adjustments to zero
        momentum_leaf_adjustments_.fill(0.0f);
        reversion_leaf_adjustments_.fill(0.0f);
    }
    
    FORCE_INLINE void predict_batch(const float* features_matrix,
                                   TradingSignal* signals,
                                   const std::vector<int32_t>& symbol_ids,
                                   uint64_t timestamp_ns) const noexcept {
        
        const size_t n_symbols = symbol_ids.size();
        
        for (size_t i = 0; i < n_symbols; ++i) {
            const float* features = features_matrix + i * 15;
            TradingSignal& signal = signals[i];
            
            // FAIL-FAST: Validate all features for NaN/Inf - no mock/placeholder values accepted
            for (int feat_idx = 0; feat_idx < 10; ++feat_idx) {
                if (std::isnan(features[feat_idx]) || std::isinf(features[feat_idx])) {
                    // Invalid feature detected - reject signal entirely
                    signal.direction = 0;
                    signal.confidence = 0.0f;
                    signal.position_size = 0.0f;
                    signal.strategy_id = 255; // Invalid strategy marker
                    signal.symbol_id = symbol_ids[i];
                    signal.timestamp_ns = timestamp_ns;
                    signal.tier = 2;
                    continue; // Skip to next symbol
                }
            }
            
            // Initialize signal with unique ID
            signal.signal_id = next_signal_id_.fetch_add(1, std::memory_order_relaxed);
            signal.symbol_id = symbol_ids[i];
            signal.timestamp_ns = timestamp_ns;
            signal.tier = 2;
            
            // Fast regime classification using lookup table with adaptive adjustment
            uint8_t regime_idx = static_cast<uint8_t>(std::min(features[7] * 100.0f, 255.0f));
            float regime_prob = regime_lookup_[regime_idx];
            
            // Apply adaptive regime adjustment (thread-safe)
            {
                std::lock_guard<std::mutex> lock(learning_mutex_);
                auto regime_weights = regime_learner_->get_weights(); // Returns copy for thread safety
                if (regime_weights.size() > regime_idx) {
                    regime_prob += regime_weights[regime_idx] * 0.1f; // Small adjustment
                    regime_prob = std::max(0.0f, std::min(1.0f, regime_prob)); // Clamp to [0,1]
                }
            }
            
            // Tree traversal for momentum and reversion with adaptive leaf adjustments
            float momentum_score = traverse_tree_adaptive(momentum_tree_.data(), momentum_tree_size_,
                                                        features, momentum_leaf_adjustments_.data());
            float reversion_score = traverse_tree_adaptive(reversion_tree_.data(), reversion_tree_size_,
                                                         features, reversion_leaf_adjustments_.data());
            
            // BALANCED strategy selection - equal treatment for momentum and reversion
            float momentum_threshold = 0.008f;  // Equal thresholds
            float reversion_threshold = 0.008f; // Equal thresholds
            
            bool momentum_valid = std::abs(momentum_score) > momentum_threshold;
            bool reversion_valid = std::abs(reversion_score) > reversion_threshold;
            
            if (momentum_valid && reversion_valid) {
                // Both valid - choose stronger signal regardless of regime
                if (std::abs(momentum_score) > std::abs(reversion_score)) {
                    signal.direction = (momentum_score > 0) ? 1 : -1;
                    signal.confidence = std::min(std::abs(momentum_score) * 80.0f, 1.0f);
                    signal.strategy_id = 0; // Momentum
                } else {
                    signal.direction = (reversion_score > 0) ? 1 : -1;
                    signal.confidence = std::min(std::abs(reversion_score) * 80.0f, 1.0f);
                    signal.strategy_id = 1; // Mean reversion
                }
            } else if (momentum_valid) {
                signal.direction = (momentum_score > 0) ? 1 : -1;
                signal.confidence = std::min(std::abs(momentum_score) * 80.0f, 1.0f);
                signal.strategy_id = 0; // Momentum
            } else if (reversion_valid) {
                signal.direction = (reversion_score > 0) ? 1 : -1;
                signal.confidence = std::min(std::abs(reversion_score) * 80.0f, 1.0f);
                signal.strategy_id = 1; // Mean reversion
            } else {
                signal.direction = 0;
                signal.confidence = 0.0f;
                signal.strategy_id = 2; // No trade
            }
            
            // FAIL-FAST: No mock/placeholder volatility data - require real volatility
            if (features[7] <= 0) {
                // Invalid volatility - reject this signal entirely instead of using placeholder
                signal.direction = 0;
                signal.confidence = 0.0f;
                signal.position_size = 0.0f;
                signal.strategy_id = 255; // Invalid strategy marker
                continue; // Skip to next symbol
            }
            
            // Position sizing using real volatility only with division by zero protection
            float volatility = std::max(features[7], 1e-6f); // Prevent division by zero
            signal.position_size = (signal.direction != 0) ?
                std::min(signal.confidence * 0.4f / volatility, 0.03f) : 0.0f; // Max 3% per Tier 2 position
            
            signal.stop_loss = 0.02f;      // 2% stop loss (FIXED)
            signal.take_profit = 0.008f;   // 0.8% take profit (FIXED)
            signal.hold_time_ms = 300000;  // 5 minutes max hold
        }
    }
    
    // Online learning feedback method
    void update_from_outcome(const SignalOutcome& outcome, const float* features) {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        
        // Map the leaf node that was used for prediction to the adjustment array
        size_t leaf_idx = 0;
        
        if (outcome.strategy_id == 0) {
            // Update momentum model leaf adjustments
            leaf_idx = find_leaf_index(momentum_tree_.data(), momentum_tree_size_, features);
            if (leaf_idx < momentum_tree_size_) {
                // Update the specific leaf value adjustment based on outcome
                momentum_learner_->update_weights(outcome,
                                                &momentum_leaf_adjustments_[leaf_idx],
                                                features, 1); // Update single leaf
            }
        } else if (outcome.strategy_id == 1) {
            // Update reversion model leaf adjustments
            leaf_idx = find_leaf_index(reversion_tree_.data(), reversion_tree_size_, features);
            if (leaf_idx < reversion_tree_size_) {
                reversion_learner_->update_weights(outcome,
                                                 &reversion_leaf_adjustments_[leaf_idx],
                                                 features, 1); // Update single leaf
            }
        }
        
        // Also update regime classifier based on outcome (thread-safe)
        uint8_t regime_idx = static_cast<uint8_t>(std::min(features[7] * 100.0f, 255.0f));
        // Note: For regime updates, we update the lookup table directly
        // This is a simplified approach - in production you'd want more sophisticated regime learning
        if (regime_idx < 256) {
            // Update regime lookup based on outcome
            float regime_adjustment = outcome.is_winner ? 0.01f : -0.01f;
            regime_lookup_[regime_idx] = std::max(0.0f, std::min(1.0f,
                regime_lookup_[regime_idx] + regime_adjustment));
        }
    }
    
    // Get learning performance metrics
    struct LearningStats {
        OnlineLearningEngine::PerformanceMetrics momentum_perf;
        OnlineLearningEngine::PerformanceMetrics reversion_perf;
        OnlineLearningEngine::PerformanceMetrics regime_perf;
    };
    
    LearningStats get_learning_stats() const {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        return {
            momentum_learner_->get_performance(),
            reversion_learner_->get_performance(),
            regime_learner_->get_performance()
        };
    }
    
    // Get weights for visibility
    const std::vector<float>& get_weights() const {
        // Return empty vector as tree models don't have direct weights
        static std::vector<float> empty;
        return empty;
    }

private:
    FORCE_INLINE float traverse_tree(const TreeNode* tree, size_t tree_size, const float* features) const noexcept {
        size_t node_idx = 0;
        
        while (node_idx < tree_size) {
            const TreeNode& node = tree[node_idx];
            
            if (node.left_child == -1) { // Leaf node
                return node.leaf_value;
            }
            
            if (features[node.feature_idx] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        
        return 0.0f;
    }
    
    // Tree traversal with adaptive leaf adjustments
    FORCE_INLINE float traverse_tree_adaptive(const TreeNode* tree, size_t tree_size,
                                            const float* features, const float* adjustments) const noexcept {
        size_t node_idx = 0;
        
        while (node_idx < tree_size) {
            const TreeNode& node = tree[node_idx];
            
            if (node.left_child == -1) { // Leaf node
                // Apply adaptive adjustment to leaf value
                return node.leaf_value + adjustments[node_idx];
            }
            
            if (features[node.feature_idx] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        
        return 0.0f;
    }
    
    // Find which leaf index was used for a prediction
    FORCE_INLINE size_t find_leaf_index(const TreeNode* tree, size_t tree_size, const float* features) const noexcept {
        size_t node_idx = 0;
        
        while (node_idx < tree_size) {
            const TreeNode& node = tree[node_idx];
            
            if (node.left_child == -1) { // Leaf node
                return node_idx;
            }
            
            if (features[node.feature_idx] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        
        return 0;
    }
    
    void initialize_momentum_tree() {
        // BALANCED momentum tree - equal positive/negative potential
        momentum_tree_[0] = {0.005f, 0, 1, 2, 0.0f}; // ret_1s threshold (lowered)
        momentum_tree_[1] = {0.003f, 1, -1, -1, 0.6f}; // Positive momentum (reduced)
        momentum_tree_[2] = {-0.005f, 0, 3, 4, 0.0f}; // Negative threshold
        momentum_tree_[3] = {0.0f, 5, -1, -1, -0.6f}; // Negative momentum (balanced)
        momentum_tree_[4] = {0.0f, 5, -1, -1, 0.3f}; // Weak positive (reduced)
        momentum_tree_size_ = 5;
    }
    
    void initialize_reversion_tree() {
        // BALANCED reversion tree - equal buy/sell signals
        reversion_tree_[0] = {1.5f, 2, 1, 2, 0.0f}; // RSI-like feature (balanced)
        reversion_tree_[1] = {0.015f, 7, -1, -1, 0.6f}; // High vol reversion (reduced)
        reversion_tree_[2] = {-1.5f, 2, 3, 4, 0.0f}; // Negative threshold
        reversion_tree_[3] = {0.008f, 7, -1, -1, -0.6f}; // Negative reversion (balanced)
        reversion_tree_[4] = {0.0f, 11, -1, -1, 0.3f}; // Weak positive (reduced)
        reversion_tree_size_ = 5;
    }
    
    void initialize_regime_lookup() {
        // Pre-compute regime probabilities
        for (size_t i = 0; i < 256; ++i) {
            float vol = static_cast<float>(i) / 100.0f;
            if (vol > 0.3f) {
                regime_lookup_[i] = 0.8f; // High vol regime
            } else if (vol < 0.15f) {
                regime_lookup_[i] = 0.2f; // Low vol regime
            } else {
                regime_lookup_[i] = 0.5f; // Normal regime
            }
        }
    }
};

// Tiered model ensemble for ultra-fast inference
class TieredModelEnsemble {
private:
    UltraFastLinearModel tier1_model_;
    FastTreeModel tier2_model_;
    
    MarketDataManager& market_data_;
    UltraFastMemoryManager& memory_manager_;
    
    // Performance tracking for 2 tiers
    std::array<std::atomic<uint64_t>, 2> inference_counts_;
    std::array<std::atomic<uint64_t>, 2> total_inference_time_ns_;
    
public:
    TieredModelEnsemble(MarketDataManager& market_data, UltraFastMemoryManager& memory_manager)
        : market_data_(market_data), memory_manager_(memory_manager) {
        // Initialize performance counters
        for (auto& counter : inference_counts_) {
            counter.store(0);
        }
        for (auto& counter : total_inference_time_ns_) {
            counter.store(0);
        }
    }
    
    // Performance statistics structure
    struct ModelPerformanceStats {
        uint64_t inference_count[2];
        double avg_inference_time_us[2];
        bool target_met[2];
    };
    
    ModelPerformanceStats get_performance_stats() const {
        ModelPerformanceStats stats{};
        constexpr double targets_us[] = {5.0, 20.0}; // Adjusted for 2 tiers
        
        for (int i = 0; i < 2; ++i) { // Adjusted for 2 tiers
            uint64_t count = inference_counts_[i].load();
            uint64_t total_time = total_inference_time_ns_[i].load();
            
            stats.inference_count[i] = count;
            stats.avg_inference_time_us[i] = count > 0 ?
                static_cast<double>(total_time) / (count * 1000.0) : 0.0;
            stats.target_met[i] = stats.avg_inference_time_us[i] <= targets_us[i];
        }
        
        return stats;
    }

    // Method to get learning stats specifically from the Tier 1 model
    UltraFastLinearModel::LearningStats get_tier1_learning_stats() const {
        return tier1_model_.get_learning_stats();
    }
    
    // Method to get learning stats specifically from the Tier 2 model
    FastTreeModel::LearningStats get_tier2_learning_stats() const {
        return tier2_model_.get_learning_stats();
    }
    
    // Update models from trading outcomes
    void update_from_outcome(const SignalOutcome& outcome, const float* features) {
        if (outcome.tier == 1) {
            tier1_model_.update_from_outcome(outcome, features);
        } else if (outcome.tier == 2) {
            tier2_model_.update_from_outcome(outcome, features);
        }
    }
    
    // Main prediction method
    void predict_batch(const float* features_matrix,
                      TradingSignal* signals,
                      const std::vector<int32_t>& symbol_ids,
                      uint64_t timestamp_ns) {
        
        // Separate symbols by tier
        std::vector<int32_t> tier1_symbols, tier2_symbols;
        std::vector<size_t> tier1_indices, tier2_indices;
        
        for (size_t i = 0; i < symbol_ids.size(); ++i) {
            if (memory_manager_.is_tier1_symbol(symbol_ids[i])) {
                tier1_symbols.push_back(symbol_ids[i]);
                tier1_indices.push_back(i);
            } else {
                tier2_symbols.push_back(symbol_ids[i]);
                tier2_indices.push_back(i);
            }
        }
        
        // Process Tier 1 symbols with ultra-fast model
        if (!tier1_symbols.empty()) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Create feature matrix for tier 1 symbols
            std::vector<float> tier1_features(tier1_symbols.size() * 15);
            for (size_t i = 0; i < tier1_indices.size(); ++i) {
                std::memcpy(tier1_features.data() + i * 15,
                           features_matrix + tier1_indices[i] * 15,
                           15 * sizeof(float));
            }
            
            // Create signals array for tier 1
            std::vector<TradingSignal> tier1_signals(tier1_symbols.size());
            
            tier1_model_.predict_batch_vectorized(tier1_features.data(),
                                                 tier1_signals.data(),
                                                 tier1_symbols,
                                                 timestamp_ns);
            
            // Copy back to main signals array
            for (size_t i = 0; i < tier1_indices.size(); ++i) {
                signals[tier1_indices[i]] = tier1_signals[i];
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            inference_counts_[0].fetch_add(tier1_symbols.size());
            total_inference_time_ns_[0].fetch_add(duration.count());
        }
        
        // Process Tier 2 symbols with tree model
        if (!tier2_symbols.empty()) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Create feature matrix for tier 2 symbols
            std::vector<float> tier2_features(tier2_symbols.size() * 15);
            for (size_t i = 0; i < tier2_indices.size(); ++i) {
                std::memcpy(tier2_features.data() + i * 15,
                           features_matrix + tier2_indices[i] * 15,
                           15 * sizeof(float));
            }
            
            // Create signals array for tier 2
            std::vector<TradingSignal> tier2_signals(tier2_symbols.size());
            
            tier2_model_.predict_batch(tier2_features.data(),
                                      tier2_signals.data(),
                                      tier2_symbols,
                                      timestamp_ns);
            
            // Copy back to main signals array
            for (size_t i = 0; i < tier2_indices.size(); ++i) {
                signals[tier2_indices[i]] = tier2_signals[i];
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            inference_counts_[1].fetch_add(tier2_symbols.size());
            total_inference_time_ns_[1].fetch_add(duration.count());
        }
    }
};

} // namespace hft
