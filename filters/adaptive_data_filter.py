#!/usr/bin/env python3

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

logger = get_system_logger("filters.adaptive_data_filter")

@dataclass
class MarketData:
    """Container for market data from Polygon"""
    symbol: str
    price: float
    volume: int
    market_cap: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    daily_change: Optional[float] = None
    volatility: Optional[float] = None
    momentum_score: Optional[float] = None
    market_condition: Optional[str] = None
    ml_score: Optional[float] = None
    strategy_type: Optional[str] = None

@dataclass
class MarketCondition:
    """Current market condition state"""
    condition: str
    vix_level: float
    spy_change: float
    volume_ratio: float
    timestamp: float
    confidence: float

class MarketConditionScanner:
    """Scans market conditions every 2 minutes"""
    
    def __init__(self):
        self.current_condition = "calm_range"
        self.last_scan_time = 0
        self.scan_interval = 120  # 2 minutes in seconds
        self.condition_history = []
        
        # Market condition thresholds
        self.thresholds = {
            'vix_high': 25,
            'vix_low': 15,
            'spy_bull_threshold': 0.005,  # 0.5% in 2 minutes
            'spy_bear_threshold': -0.005,  # -0.5% in 2 minutes
            'volume_high': 1.5  # 1.5x average volume
        }
    
    async def scan_market_conditions(self, market_data: Dict) -> MarketCondition:
        """Scan market conditions every 2 minutes"""
        current_time = time.time()
        
        if current_time - self.last_scan_time >= self.scan_interval:
            # Get fresh market indicators
            vix = market_data.get('vix', 20)
            spy_change = market_data.get('spy_2min_change', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Detect new condition
            new_condition = self._detect_condition(vix, spy_change, volume_ratio)
            confidence = self._calculate_confidence(vix, spy_change, volume_ratio)
            
            # Update if changed
            if new_condition != self.current_condition:
                logger.info(f"Market condition changed: {self.current_condition} → {new_condition}")
                self.current_condition = new_condition
            
            logger.log_filter_decision("market_condition_scan", "SYSTEM",
                                             decision=True,
                                             criteria={"old": self.current_condition, "new": new_condition})
            
            # Create condition object
            condition_obj = MarketCondition(
                condition=new_condition,
                vix_level=vix,
                spy_change=spy_change,
                volume_ratio=volume_ratio,
                timestamp=current_time,
                confidence=confidence
            )
            
            # Store in history
            self.condition_history.append(condition_obj)
            if len(self.condition_history) > 50:  # Keep last 50 scans
                self.condition_history = self.condition_history[-50:]
            
            self.last_scan_time = current_time
            return condition_obj
        
        # Return current condition if not time to scan
        return MarketCondition(
            condition=self.current_condition,
            vix_level=market_data.get('vix', 20),
            spy_change=market_data.get('spy_2min_change', 0),
            volume_ratio=market_data.get('volume_ratio', 1.0),
            timestamp=self.last_scan_time,
            confidence=0.8
        )
    
    def _detect_condition(self, vix: float, spy_change: float, volume_ratio: float) -> str:
        """Detect market condition based on indicators"""
        
        # Volatile market - high VIX
        if vix > self.thresholds['vix_high']:
            return "volatile"
        
        # Bull trending - positive SPY move with low VIX
        elif (spy_change > self.thresholds['spy_bull_threshold'] and 
              vix < self.thresholds['vix_low']):
            return "bull_trending"
        
        # Bear trending - negative SPY move
        elif spy_change < self.thresholds['spy_bear_threshold']:
            return "bear_trending"
        
        # Calm range - everything else
        else:
            return "calm_range"
    
    def _calculate_confidence(self, vix: float, spy_change: float, volume_ratio: float) -> float:
        """Calculate confidence in current condition detection"""
        confidence = 0.5
        
        # Higher confidence for extreme values
        if vix > 30 or vix < 12:
            confidence += 0.3
        
        if abs(spy_change) > 0.01:  # 1% move
            confidence += 0.2
        
        if volume_ratio > 2.0:  # High volume
            confidence += 0.1
        
        return min(confidence, 1.0)

class AdaptiveStockFilter:
    """Adaptive stock filter that changes rules based on market conditions"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'filters.yaml')
        self.config = self._load_config(config_path)
        self.condition_rules = self._setup_condition_rules()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load filter configuration"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(e, {"operation": "load_config", "path": config_path})
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default filter configuration matching YAML structure"""
        return {
            'market_conditions': {
                'bull_trending': {
                    'filter_rules': {
                        'min_price': 20,
                        'max_price': 500,
                        'min_volume': 2000000,
                        'min_market_cap': 1000000000,
                        'focus_sectors': ['tech', 'growth', 'consumer_discretionary'],
                        'min_momentum': 0.02,
                        'max_beta': 2.0
                    }
                },
                'bear_trending': {
                    'filter_rules': {
                        'min_price': 15,
                        'max_price': 300,
                        'min_volume': 5000000,
                        'min_market_cap': 500000000,
                        'focus_sectors': ['defensive', 'utilities', 'staples'],
                        'max_beta': 0.8,
                        'min_short_interest': 0.1
                    }
                },
                'volatile': {
                    'filter_rules': {
                        'min_price': 25,
                        'max_price': 400,
                        'min_volume': 10000000,
                        'min_market_cap': 2000000000,
                        'focus_sectors': ['any'],
                        'min_volatility': 0.03,
                        'min_options_volume': 1000
                    }
                },
                'calm_range': {
                    'filter_rules': {
                        'min_price': 15,
                        'max_price': 300,
                        'min_volume': 1000000,
                        'min_market_cap': 500000000,
                        'focus_sectors': ['any'],
                        'balanced_criteria': True
                    }
                }
            }
        }
    
    def _setup_condition_rules(self) -> Dict:
        """Setup filtering rules for each market condition"""
        market_conditions = self.config.get('market_conditions', {})
        
        # Extract filter_rules from each condition
        condition_rules = {}
        for condition, config in market_conditions.items():
            condition_rules[condition] = config.get('filter_rules', {})
        
        return condition_rules
    
    async def filter_stocks(self, stocks: List[MarketData], 
                          market_condition: MarketCondition) -> List[MarketData]:
        """Filter stocks based on current market condition"""
        
        condition = market_condition.condition
        rules = self.condition_rules.get(condition, self.condition_rules['calm_range'])
        
        filtered_stocks = []
        
        for stock in stocks:
            meets_criteria = await self._meets_condition_criteria(stock, rules, market_condition)
            logger.log_filter_decision("adaptive_stock_filter", stock.symbol,
                                             decision=meets_criteria,
                                             criteria={"condition": condition, "rules": rules})
            if meets_criteria:
                stock.market_condition = condition
                filtered_stocks.append(stock)
        
        logger.info(f"Filtered {len(stocks)} → {len(filtered_stocks)} stocks for {condition}")
        return filtered_stocks
    
    async def _meets_condition_criteria(self, stock: MarketData, rules: Dict, 
                                      market_condition: MarketCondition) -> bool:
        """Check if stock meets criteria for current market condition"""
        
        # Basic price and volume filters
        if not (rules['min_price'] <= stock.price <= rules['max_price']):
            return False
        
        if stock.volume < rules['min_volume']:
            return False
        
        if stock.market_cap < rules['min_market_cap']:
            return False
        
        # Condition-specific filters
        if market_condition.condition == 'bull_trending':
            return await self._check_bull_criteria(stock, rules)
        
        elif market_condition.condition == 'bear_trending':
            return await self._check_bear_criteria(stock, rules)
        
        elif market_condition.condition == 'volatile':
            return await self._check_volatile_criteria(stock, rules)
        
        else:  # calm_range
            return await self._check_calm_criteria(stock, rules)
    
    async def _check_bull_criteria(self, stock: MarketData, rules: Dict) -> bool:
        """Check criteria for bull trending market"""
        # Look for momentum stocks
        if stock.momentum_score and stock.momentum_score < rules.get('min_momentum', 0.02):
            return False
        
        # Positive daily change preferred
        if stock.daily_change and stock.daily_change < -0.02:  # Avoid stocks down >2%
            return False
        
        return True
    
    async def _check_bear_criteria(self, stock: MarketData, rules: Dict) -> bool:
        """Check criteria for bear trending market"""
        # Look for defensive characteristics or short candidates
        
        # For defensive: prefer lower beta, stable stocks
        # For shorts: prefer overvalued, weak stocks
        
        # Avoid momentum stocks in bear markets
        if stock.momentum_score and stock.momentum_score > 0.05:
            return False
        
        return True
    
    async def _check_volatile_criteria(self, stock: MarketData, rules: Dict) -> bool:
        """Check criteria for volatile market"""
        # Look for high volatility, high volume stocks
        
        if stock.volatility and stock.volatility < rules.get('min_volatility', 0.03):
            return False
        
        # Higher volume requirements in volatile markets
        if stock.volume < rules['min_volume']:
            return False
        
        return True
    
    async def _check_calm_criteria(self, stock: MarketData, rules: Dict) -> bool:
        """Check criteria for calm range market"""
        # Balanced approach - no extreme requirements
        return True

class MLReadyFilter:
    """Second stage filter to prepare stocks for ML processing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ml_strategies = self._setup_ml_strategies()
    
    def _setup_ml_strategies(self) -> Dict:
        """Setup ML strategies from config or defaults"""
        market_conditions = self.config.get('market_conditions', {})
        
        strategies = {}
        for condition, config in market_conditions.items():
            ml_strategy = config.get('ml_strategy', {})
            strategies[condition] = {
                'strategy': ml_strategy.get('strategy_name', f'{condition}_strategy'),
                'min_score': ml_strategy.get('min_score', 0.6),
                'features_focus': ml_strategy.get('features_focus', []),
                'max_candidates': ml_strategy.get('max_candidates', 80)
            }
        
        # Fallback defaults if no config
        if not strategies:
            strategies = {
                'bull_trending': {
                    'strategy': 'momentum_breakouts',
                    'min_score': 0.6,
                    'features_focus': ['momentum', 'volume_surge', 'breakout_patterns'],
                    'max_candidates': 100
                },
                'bear_trending': {
                    'strategy': 'short_setups',
                    'min_score': 0.6,
                    'features_focus': ['weakness', 'failed_bounces', 'overvaluation'],
                    'max_candidates': 50
                },
                'volatile': {
                    'strategy': 'volatility_trades',
                    'min_score': 0.7,
                    'features_focus': ['high_iv', 'news_reactive', 'options_flow'],
                    'max_candidates': 75
                },
                'calm_range': {
                    'strategy': 'mean_reversion',
                    'min_score': 0.5,
                    'features_focus': ['oversold', 'support_levels', 'value'],
                    'max_candidates': 80
                }
            }
        
        return strategies
    
    async def prepare_for_ml(self, filtered_stocks: List[MarketData], 
                           market_condition: MarketCondition) -> List[MarketData]:
        """Prepare filtered stocks for ML processing"""
        
        condition = market_condition.condition
        strategy = self.ml_strategies.get(condition, self.ml_strategies['calm_range'])
        
        # Score stocks for ML readiness
        scored_stocks = []
        for stock in filtered_stocks:
            score = await self._calculate_ml_score(stock, strategy, market_condition)
            if score >= strategy['min_score']:
                stock.ml_score = score
                stock.strategy_type = strategy['strategy']
                scored_stocks.append(stock)
        
        # Sort by score and limit candidates
        scored_stocks.sort(key=lambda x: x.ml_score, reverse=True)
        ml_ready = scored_stocks[:strategy['max_candidates']]
        
        logger.info(f"ML ready: {len(ml_ready)} stocks for {condition} ({strategy['strategy']})")
        logger.log_ml_prediction("ml_ready_filter",
                                       features={"condition": condition, "strategy": strategy['strategy']},
                                       prediction=len(ml_ready))
        return ml_ready
    
    async def _calculate_ml_score(self, stock: MarketData, strategy: Dict, 
                                market_condition: MarketCondition) -> float:
        """Calculate ML readiness score for stock"""
        
        base_score = 0.5
        
        # Volume score (higher volume = better for ML)
        if stock.volume > 5000000:
            base_score += 0.2
        elif stock.volume > 2000000:
            base_score += 0.1
        
        # Volatility score (depends on strategy)
        if stock.volatility:
            if market_condition.condition == 'volatile' and stock.volatility > 0.03:
                base_score += 0.2
            elif market_condition.condition in ['bull_trending', 'bear_trending'] and stock.volatility < 0.05:
                base_score += 0.1
        
        # Momentum score (depends on market condition)
        if stock.momentum_score:
            if market_condition.condition == 'bull_trending' and stock.momentum_score > 0.02:
                base_score += 0.2
            elif market_condition.condition == 'bear_trending' and stock.momentum_score < -0.02:
                base_score += 0.2
        
        # Market cap stability (larger caps generally better for ML)
        if stock.market_cap > 10000000000:  # >$10B
            base_score += 0.1
        
        return min(base_score, 1.0)

class AdaptiveDataFilter:
    """Main adaptive data filter combining all filtering stages"""
    
    def __init__(self, config_path: str = None):
        self.condition_scanner = MarketConditionScanner()
        self.stock_filter = AdaptiveStockFilter(config_path)
        
        # Load config for ML filter
        config = self.stock_filter.config if hasattr(self.stock_filter, 'config') else {}
        self.ml_filter = MLReadyFilter(config)
        
        logger.startup({
            "component": "adaptive_data_filter",
            "action": "initialization",
            "config_path": config_path
        })
        
        # Performance tracking
        self.filter_stats = {
            'total_processed': 0,
            'stage1_filtered': 0,
            'stage2_ml_ready': 0,
            'processing_times': []
        }
    
    async def process_polygon_data(self, polygon_data: List[Dict]) -> List[MarketData]:
        """Main processing method for Polygon data"""
        
        start_time = time.time()
        
        logger.log_data_flow("processing", "polygon_data", data_size=len(polygon_data))
        
        try:
            # Convert Polygon data to MarketData objects
            market_data_list = await self._convert_polygon_data(polygon_data)
            logger.log_data_flow("conversion", "market_data_objects", data_size=len(market_data_list))
            
            # Get market indicators for condition scanning
            market_indicators = await self._extract_market_indicators(polygon_data)
            logger.log_data_flow("extraction", "market_indicators", data_sample=market_indicators)
            
            # Stage 1: Scan market conditions (every 2 minutes)
            market_condition = await self.condition_scanner.scan_market_conditions(market_indicators)
            logger.log_data_flow("scan", "market_conditions", data_sample=market_condition.condition)
            
            # Stage 2: Adaptive stock filtering
            filtered_stocks = await self.stock_filter.filter_stocks(market_data_list, market_condition)
            logger.log_data_flow("filtering", "adaptive_stock", data_size=len(filtered_stocks))
            
            # Stage 3: ML readiness filtering
            ml_ready_stocks = await self.ml_filter.prepare_for_ml(filtered_stocks, market_condition)
            logger.log_data_flow("filtering", "ml_readiness", data_size=len(ml_ready_stocks))
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(market_data_list), len(filtered_stocks),
                             len(ml_ready_stocks), processing_time)
            
            logger.info(f"Adaptive filter: {len(market_data_list)} → {len(filtered_stocks)} → {len(ml_ready_stocks)} stocks in {processing_time:.3f}s")
            logger.log_performance("process_polygon_data", processing_time * 1000) # Convert to ms
            
            return ml_ready_stocks
            
        except Exception as e:
            logger.error(e, {"operation": "process_polygon_data"})
            return []
    
    async def _convert_polygon_data(self, polygon_data: List[Dict]) -> List[MarketData]:
        """Convert Polygon data format to MarketData objects"""
        market_data_list = []
        
        for data in polygon_data:
            try:
                market_data = MarketData(
                    symbol=data.get('symbol', ''),
                    price=float(data.get('price', 0)),
                    volume=int(data.get('volume', 0)),
                    market_cap=float(data.get('market_cap', 0)),
                    timestamp=float(data.get('timestamp', time.time())),
                    bid=data.get('bid'),
                    ask=data.get('ask'),
                    daily_change=data.get('daily_change'),
                    volatility=data.get('volatility'),
                    momentum_score=data.get('momentum_score')
                )
                market_data_list.append(market_data)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting data for {data.get('symbol', 'unknown')}: {e}")
                continue
        
        return market_data_list
    
    async def _extract_market_indicators(self, polygon_data: List[Dict]) -> Dict:
        """Extract market-wide indicators from Polygon data"""
        
        # Find VIX data
        vix_data = next((d for d in polygon_data if d.get('symbol') == 'VIX'), None)
        vix_level = float(vix_data.get('price', 20)) if vix_data else 20
        
        # Find SPY data for market direction
        spy_data = next((d for d in polygon_data if d.get('symbol') == 'SPY'), None)
        spy_change = float(spy_data.get('daily_change', 0)) if spy_data else 0
        
        # Calculate average volume ratio
        volume_ratios = [d.get('volume_ratio', 1.0) for d in polygon_data if d.get('volume_ratio')]
        avg_volume_ratio = np.mean(volume_ratios) if volume_ratios else 1.0
        
        return {
            'vix': vix_level,
            'spy_2min_change': spy_change,
            'volume_ratio': avg_volume_ratio
        }
    
    def _update_stats(self, total: int, filtered: int, ml_ready: int, processing_time: float):
        """Update filter performance statistics"""
        self.filter_stats['total_processed'] += total
        self.filter_stats['stage1_filtered'] += filtered
        self.filter_stats['stage2_ml_ready'] += ml_ready
        self.filter_stats['processing_times'].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.filter_stats['processing_times']) > 100:
            self.filter_stats['processing_times'] = self.filter_stats['processing_times'][-100:]
    
    def get_filter_stats(self) -> Dict:
        """Get filter performance statistics"""
        processing_times = self.filter_stats['processing_times']
        
        return {
            'total_processed': self.filter_stats['total_processed'],
            'stage1_filtered': self.filter_stats['stage1_filtered'],
            'stage2_ml_ready': self.filter_stats['stage2_ml_ready'],
            'current_condition': self.condition_scanner.current_condition,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'p95_processing_time': np.percentile(processing_times, 95) if processing_times else 0,
            'filter_efficiency': (self.filter_stats['stage2_ml_ready'] / 
                                max(self.filter_stats['total_processed'], 1)) * 100
        }

# Example usage
async def main():
    """Example usage of the adaptive data filter"""
    
    # Create filter
    filter_system = AdaptiveDataFilter()
    
    # Example Polygon data
    sample_data = [
        {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 50000000,
            'market_cap': 2500000000000,
            'daily_change': 0.02,
            'volatility': 0.025,
            'momentum_score': 0.03
        },
        {
            'symbol': 'VIX',
            'price': 18.5,
            'volume': 1000000,
            'market_cap': 0,
            'daily_change': -0.05
        },
        {
            'symbol': 'SPY',
            'price': 420.0,
            'volume': 100000000,
            'market_cap': 0,
            'daily_change': 0.008
        }
    ]
    
    # Process data
    ml_ready_stocks = await filter_system.process_polygon_data(sample_data)
    
    # Print results
    print(f"ML ready stocks: {len(ml_ready_stocks)}")
    for stock in ml_ready_stocks:
        print(f"  {stock.symbol}: {stock.strategy_type} (score: {stock.ml_score:.2f})")
    
    # Print stats
    stats = filter_system.get_filter_stats()
    print(f"Filter stats: {stats}")

if __name__ == "__main__":
    logger.info("Starting adaptive data filter example")
    asyncio.run(main())