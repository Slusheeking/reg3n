"""
Model Caching System

Multi-level caching for model predictions and features.
"""

import time
import pickle
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        # Check if expired
        if key in self.timestamps:
            entry_time, ttl = self.timestamps[key]
            if ttl and time.time() - entry_time > ttl:
                self.delete(key)
                return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.timestamps[key] = (time.time(), ttl)
        return True
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        return True
    
    def clear(self) -> bool:
        self.cache.clear()
        self.timestamps.clear()
        return True
    
    def _evict_oldest(self):
        """Evict oldest entries to make space"""
        if not self.timestamps:
            return
        
        # Find oldest entry
        oldest_key = min(self.timestamps.keys(), 
                        key=lambda k: self.timestamps[k][0])
        self.delete(oldest_key)


class RedisCache(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, prefix: str = 'ai_trading'):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.prefix = prefix
        
        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError:
            raise ConnectionError(f"Cannot connect to Redis at {host}:{port}")
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(self._make_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            data = pickle.dumps(value)
            return self.client.set(self._make_key(key), data, ex=ttl)
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception:
            return False
    
    def clear(self) -> bool:
        try:
            keys = self.client.keys(f"{self.prefix}:*")
            if keys:
                return bool(self.client.delete(*keys))
            return True
        except Exception:
            return False


class ModelCache:
    """
    Multi-level caching system for model predictions
    
    Features:
    - Memory + Redis backends
    - Automatic cache key generation
    - TTL support
    - Hit/miss statistics
    """
    
    def __init__(self, use_redis: bool = False, redis_config: Optional[Dict] = None):
        # Primary cache (always memory)
        self.memory_cache = MemoryCache(max_size=5000)
        
        # Secondary cache (Redis if available)
        self.redis_cache = None
        if use_redis and REDIS_AVAILABLE:
            try:
                config = redis_config or {}
                self.redis_cache = RedisCache(**config)
            except Exception as e:
                print(f"Warning: Failed to initialize Redis cache: {e}")
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'total_requests': 0
        }
        
        # Default TTLs
        self.default_ttls = {
            'feature': 300,      # 5 minutes
            'prediction': 60,    # 1 minute
            'signal': 30,        # 30 seconds
            'model_state': 3600  # 1 hour
        }
    
    def generate_cache_key(self, symbol: str, data_type: str, 
                          context: Optional[Dict] = None) -> str:
        """Generate cache key from symbol and context"""
        
        key_parts = [symbol, data_type]
        
        if context:
            # Sort context for consistent keys
            context_str = '|'.join(f"{k}:{v}" for k, v in sorted(context.items()))
            key_parts.append(context_str)
        
        key_string = ':'.join(key_parts)
        
        # Hash for consistent length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, symbol: str, data_type: str, 
            context: Optional[Dict] = None) -> Optional[Any]:
        """Get cached value"""
        
        self.stats['total_requests'] += 1
        
        key = self.generate_cache_key(symbol, data_type, context)
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats['memory_hits'] += 1
            return value
        else:
            self.stats['memory_misses'] += 1
        
        # Try Redis cache
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                self.stats['redis_hits'] += 1
                # Store in memory cache for faster access
                self.memory_cache.set(key, value, ttl=60)
                return value
            else:
                self.stats['redis_misses'] += 1
        
        return None
    
    def set(self, symbol: str, data_type: str, value: Any,
            context: Optional[Dict] = None, ttl: Optional[int] = None) -> bool:
        """Set cached value"""
        
        key = self.generate_cache_key(symbol, data_type, context)
        
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttls.get(data_type, 300)
        
        # Store in memory cache
        success_memory = self.memory_cache.set(key, value, ttl=ttl)
        
        # Store in Redis cache
        success_redis = True
        if self.redis_cache:
            success_redis = self.redis_cache.set(key, value, ttl=ttl)
        
        return success_memory and success_redis
    
    def delete(self, symbol: str, data_type: str, 
              context: Optional[Dict] = None) -> bool:
        """Delete cached value"""
        
        key = self.generate_cache_key(symbol, data_type, context)
        
        success_memory = self.memory_cache.delete(key)
        
        success_redis = True
        if self.redis_cache:
            success_redis = self.redis_cache.delete(key)
        
        return success_memory and success_redis
    
    def clear_all(self) -> bool:
        """Clear all caches"""
        
        success_memory = self.memory_cache.clear()
        
        success_redis = True
        if self.redis_cache:
            success_redis = self.redis_cache.clear()
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
        
        return success_memory and success_redis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_hits = self.stats['memory_hits'] + self.stats['redis_hits']
        total_misses = self.stats['memory_misses'] + self.stats['redis_misses']
        total_requests = self.stats['total_requests']
        
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_requests': total_requests,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': hit_rate,
            'memory_hits': self.stats['memory_hits'],
            'memory_misses': self.stats['memory_misses'],
            'redis_hits': self.stats['redis_hits'],
            'redis_misses': self.stats['redis_misses'],
            'redis_available': self.redis_cache is not None
        }
    
    def cache_prediction(self, symbol: str, prediction: Dict[str, Any],
                        features_hash: str, ttl: int = 60) -> bool:
        """Cache model prediction with features context"""
        
        context = {
            'features_hash': features_hash,
            'timestamp': int(time.time())
        }
        
        return self.set(symbol, 'prediction', prediction, context, ttl)
    
    def get_cached_prediction(self, symbol: str, features_hash: str,
                             max_age: int = 60) -> Optional[Dict[str, Any]]:
        """Get cached prediction if still valid"""
        
        context = {
            'features_hash': features_hash,
            'timestamp': int(time.time())
        }
        
        # Try different recent timestamps
        current_time = int(time.time())
        for age in range(0, max_age, 10):  # Check every 10 seconds
            context['timestamp'] = current_time - age
            prediction = self.get(symbol, 'prediction', context)
            if prediction:
                return prediction
        
        return None
    
    def cache_features(self, symbol: str, features: Dict[str, Any],
                      price_data_hash: str, ttl: int = 300) -> bool:
        """Cache extracted features"""
        
        context = {
            'price_data_hash': price_data_hash,
            'timestamp': int(time.time() / 60) * 60  # Round to minute
        }
        
        return self.set(symbol, 'feature', features, context, ttl)
    
    def get_cached_features(self, symbol: str, price_data_hash: str,
                           max_age: int = 300) -> Optional[Dict[str, Any]]:
        """Get cached features if still valid"""
        
        current_minute = int(time.time() / 60) * 60
        
        for minutes_ago in range(0, max_age // 60 + 1):
            context = {
                'price_data_hash': price_data_hash,
                'timestamp': current_minute - (minutes_ago * 60)
            }
            
            features = self.get(symbol, 'feature', context)
            if features:
                return features
        
        return None


def create_model_cache(config: Dict[str, Any]) -> ModelCache:
    """Factory function to create model cache from config"""
    
    use_redis = config.get('use_redis', False)
    redis_config = config.get('redis', {})
    
    return ModelCache(use_redis=use_redis, redis_config=redis_config)
