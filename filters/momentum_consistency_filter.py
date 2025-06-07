#!/usr/bin/env python3

import sys
import os
import yaml
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

logger = get_system_logger("filters.momentum_consistency_filter")

@dataclass
class MomentumData:
    """Container for momentum calculation data"""
    symbol: str
    current_price: float
    price_6m_ago: float
    price_5m_ago: float
    momentum_6m: float
    momentum_5m: float
    momentum_rank_6m: int
    momentum_rank_5m: int
    is_consistent: bool

class MomentumConsistencyFilter:
    """
    Simple momentum consistency filter
    Tracks 6-month and 5-month momentum rankings
    Only passes stocks in top decile for BOTH periods
    """
    
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'filters.yaml')
        
        self.config = self._load_config(config_path)
        self.TOP_DECILE_THRESHOLD = self.config.get('momentum_filter', {}).get('top_decile_threshold', 0.1)
        self.ranking_interval = self.config.get('momentum_filter', {}).get('ranking_interval', 3600)
        
        logger.startup({
            "component": "momentum_consistency_filter",
            "action": "initialization",
            "top_decile_threshold": self.TOP_DECILE_THRESHOLD,
            "config_path": config_path
        })
        logger.log_data_flow("initialization", "momentum_filter")
        
        self.momentum_history = {}  # Store momentum data by symbol
        self.last_ranking_update = None
        
        logger.info("Momentum Consistency Filter initialized - targeting top decile consistency")
        logger.log_data_flow("initialization", "complete",
                                 data_sample={"threshold": self.TOP_DECILE_THRESHOLD})
    
    def _load_config(self, config_path: str) -> Dict:
        """Load filter configuration from YAML"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(e, {"operation": "load_config", "path": config_path})
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default momentum filter configuration"""
        return {
            'momentum_filter': {
                'top_decile_threshold': 0.1,
                'ranking_interval': 3600,
                'min_momentum_6m': 0.05,
                'min_momentum_5m': 0.04
            }
        }
    
    def calculate_momentum_rankings(self, market_data: List[Dict]) -> List[MomentumData]:
        """
        Calculate 6-month and 5-month momentum rankings for all stocks
        
        Args:
            market_data: List of stock data with price history
            
        Returns:
            List of MomentumData objects with rankings
        """
        try:
            momentum_data_list = []
            
            # Calculate momentum for each stock
            for stock in market_data:
                momentum_data = self._calculate_stock_momentum(stock)
                if momentum_data:
                    momentum_data_list.append(momentum_data)
            
            # Rank stocks by momentum
            ranked_data = self._rank_momentum_data(momentum_data_list)
            
            # Update momentum history
            self._update_momentum_history(ranked_data)
            
            logger.info(f"Calculated momentum rankings for {len(ranked_data)} stocks")
            logger.log_data_flow("calculation", "momentum_rankings", data_size=len(ranked_data))
            return ranked_data
            
        except Exception as e:
            logger.error(e, {"operation": "calculate_momentum_rankings"})
            return []
    
    def _calculate_stock_momentum(self, stock: Dict) -> Optional[MomentumData]:
        """Calculate 6-month and 5-month momentum for a single stock"""
        try:
            symbol = stock.get('symbol', '')
            current_price = float(stock.get('price', 0))
            
            logger.debug(f"Calculating momentum for {symbol}: current_price=${current_price:.2f}")
            
            # Get historical prices (simplified - in production, use actual historical data)
            price_6m_ago = self._get_historical_price(symbol, months=6, current_price=current_price)
            price_5m_ago = self._get_historical_price(symbol, months=5, current_price=current_price)
            
            if price_6m_ago <= 0 or price_5m_ago <= 0:
                logger.warning(f"Invalid historical prices for {symbol}: 6m=${price_6m_ago:.2f}, 5m=${price_5m_ago:.2f}")
                return None
            
            # Calculate momentum (total return)
            momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            momentum_5m = (current_price - price_5m_ago) / price_5m_ago
            
            logger.debug(f"Momentum calculated for {symbol}: 6m={momentum_6m:.2%}, 5m={momentum_5m:.2%}")
            
            return MomentumData(
                symbol=symbol,
                current_price=current_price,
                price_6m_ago=price_6m_ago,
                price_5m_ago=price_5m_ago,
                momentum_6m=momentum_6m,
                momentum_5m=momentum_5m,
                momentum_rank_6m=0,  # Will be set in ranking step
                momentum_rank_5m=0,  # Will be set in ranking step
                is_consistent=False  # Will be set in ranking step
            )
            
        except Exception as e:
            logger.error(e, {"operation": "_calculate_stock_momentum", "symbol": stock.get('symbol', 'unknown'), "stock_data": stock})
            return None
    
    def _get_historical_price(self, symbol: str, months: int, current_price: float) -> float:
        """
        Get historical price (simplified implementation)
        In production, this would query actual historical data
        """
        # Simplified: Use momentum score from current data if available
        # In production, replace with actual historical price lookup
        
        # For now, simulate historical prices based on current momentum
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            # Simulate strong momentum stocks
            if months == 6:
                return current_price * 0.85  # 15% gain over 6 months
            else:  # 5 months
                return current_price * 0.88  # 12% gain over 5 months
        else:
            # Simulate average momentum
            if months == 6:
                return current_price * 0.95  # 5% gain over 6 months
            else:  # 5 months
                return current_price * 0.96  # 4% gain over 5 months
    
    def _rank_momentum_data(self, momentum_data_list: List[MomentumData]) -> List[MomentumData]:
        """Rank stocks by 6-month and 5-month momentum"""
        try:
            total_stocks = len(momentum_data_list)
            logger.debug(f"Ranking {total_stocks} stocks by momentum")
            
            # Sort by 6-month momentum (descending)
            sorted_6m = sorted(momentum_data_list, key=lambda x: x.momentum_6m, reverse=True)
            for i, data in enumerate(sorted_6m):
                data.momentum_rank_6m = i + 1
            
            # Log top 6-month performers
            if sorted_6m:
                top_6m = sorted_6m[0]
                logger.debug(f"Top 6-month momentum: {top_6m.symbol} with {top_6m.momentum_6m:.2%}")
            
            # Sort by 5-month momentum (descending)
            sorted_5m = sorted(momentum_data_list, key=lambda x: x.momentum_5m, reverse=True)
            for i, data in enumerate(sorted_5m):
                data.momentum_rank_5m = i + 1
            
            # Log top 5-month performers
            if sorted_5m:
                top_5m = sorted_5m[0]
                logger.debug(f"Top 5-month momentum: {top_5m.symbol} with {top_5m.momentum_5m:.2%}")
            
            # Calculate top decile thresholds
            top_decile_cutoff = int(total_stocks * self.TOP_DECILE_THRESHOLD)
            
            # Mark consistent momentum stocks (top decile in BOTH periods)
            consistent_count = 0
            for data in momentum_data_list:
                data.is_consistent = (
                    data.momentum_rank_6m <= top_decile_cutoff and
                    data.momentum_rank_5m <= top_decile_cutoff
                )
                if data.is_consistent:
                    consistent_count += 1
                    logger.debug(f"Consistent momentum: {data.symbol} (6m rank: {data.momentum_rank_6m}, 5m rank: {data.momentum_rank_5m})")
            
            consistency_rate = (consistent_count / max(total_stocks, 1)) * 100
            logger.info(f"Momentum ranking complete: {total_stocks} stocks, {top_decile_cutoff} top decile cutoff, "
                       f"{consistent_count} consistent ({consistency_rate:.1f}%)")
            
            logger.log_data_flow("ranking", "momentum_data", data_size=len(momentum_data_list))
            return momentum_data_list
            
        except Exception as e:
            logger.error(e, {"operation": "rank_momentum_data", "total_stocks": len(momentum_data_list)})
            return momentum_data_list
    
    def _update_momentum_history(self, ranked_data: List[MomentumData]):
        """Update momentum history for tracking"""
        current_time = datetime.now()
        
        logger.debug(f"Updating momentum history for {len(ranked_data)} stocks at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        consistent_stocks = []
        for data in ranked_data:
            self.momentum_history[data.symbol] = {
                'timestamp': current_time,
                'momentum_6m': data.momentum_6m,
                'momentum_5m': data.momentum_5m,
                'rank_6m': data.momentum_rank_6m,
                'rank_5m': data.momentum_rank_5m,
                'is_consistent': data.is_consistent
            }
            
            if data.is_consistent:
                consistent_stocks.append(data.symbol)
        
        self.last_ranking_update = current_time
        
        logger.debug(f"Momentum history updated: {len(consistent_stocks)} consistent stocks: {', '.join(consistent_stocks[:10])}"
                    f"{'...' if len(consistent_stocks) > 10 else ''}")
    
    
    def filter_consistent_momentum_stocks(self, market_data: List[Dict]) -> List[Dict]:
        """
        Main filter method - returns only stocks with consistent momentum
        
        Args:
            market_data: List of stock data
            
        Returns:
            List of stocks that pass momentum consistency filter
        """
        logger.log_trading_signal("filter_start", "momentum_consistency", {
            "input_stocks": len(market_data)
        })
        
        try:
            # Calculate fresh momentum rankings
            momentum_data = self.calculate_momentum_rankings(market_data)
            
            logger.log_trading_signal("rankings_calculated", "momentum_consistency", {
                "total_stocks_analyzed": len(momentum_data),
                "consistent_stocks_found": sum(1 for data in momentum_data if data.is_consistent)
            })
            
            # Filter for consistent momentum stocks only
            consistent_stocks = []
            for data in momentum_data:
                if data.is_consistent:
                    logger.log_filter_decision("momentum_consistency", data.symbol, True, {
                        "momentum_6m": data.momentum_6m,
                        "momentum_5m": data.momentum_5m,
                        "rank_6m": data.momentum_rank_6m,
                        "rank_5m": data.momentum_rank_5m
                    })
                    
                    # Find original stock data and add momentum info
                    for stock in market_data:
                        if stock.get('symbol') == data.symbol:
                            # Add momentum data to stock
                            stock['momentum_6m'] = data.momentum_6m
                            stock['momentum_5m'] = data.momentum_5m
                            stock['momentum_rank_6m'] = data.momentum_rank_6m
                            stock['momentum_rank_5m'] = data.momentum_rank_5m
                            stock['momentum_consistent'] = True
                            consistent_stocks.append(stock)
                            break
            
            pass_rate = (len(consistent_stocks)/max(len(market_data), 1)*100)
            
            logger.log_trading_signal("filter_complete", "momentum_consistency", {
                "input_stocks": len(market_data),
                "output_stocks": len(consistent_stocks),
                "pass_rate_percent": pass_rate
            })
            
            logger.info(f"Momentum consistency filter: {len(market_data)} → {len(consistent_stocks)} stocks "
                       f"({pass_rate:.1f}% pass rate)")
            
            return consistent_stocks
            
        except Exception as e:
            logger.error(e, {"operation": "filter_consistent_momentum_stocks", "input_stocks": len(market_data)})
            return []
    
    def get_momentum_stats(self) -> Dict:
        """Get momentum filter statistics"""
        consistent_count = sum(1 for data in self.momentum_history.values() if data['is_consistent'])
        total_count = len(self.momentum_history)
        consistency_rate = (consistent_count / max(total_count, 1)) * 100
        
        logger.debug(f"Momentum stats requested: {total_count} tracked, {consistent_count} consistent ({consistency_rate:.1f}%)")
        
        stats_data = {
            'total_stocks_tracked': total_count,
            'consistent_momentum_stocks': consistent_count,
            'consistency_rate_pct': consistency_rate,
            'last_update': self.last_ranking_update.isoformat() if self.last_ranking_update else None,
            'top_decile_threshold': self.TOP_DECILE_THRESHOLD,
            'filter_criteria': 'Top decile in BOTH 6-month AND 5-month momentum'
        }
        
        # Log performance metrics
        logger.performance(stats_data)
        return stats_data
    
    def get_top_consistent_stocks(self, limit: int = 20) -> List[Dict]:
        """Get top consistent momentum stocks"""
        logger.debug(f"Getting top {limit} consistent momentum stocks")
        
        consistent_stocks = [
            {'symbol': symbol, **data}
            for symbol, data in self.momentum_history.items()
            if data['is_consistent']
        ]
        
        logger.debug(f"Found {len(consistent_stocks)} consistent stocks before sorting")
        
        # Sort by average momentum
        consistent_stocks.sort(
            key=lambda x: (x['momentum_6m'] + x['momentum_5m']) / 2,
            reverse=True
        )
        
        top_stocks = consistent_stocks[:limit]
        
        if top_stocks:
            top_symbols = [stock['symbol'] for stock in top_stocks[:5]]  # Log top 5
            logger.debug(f"Top consistent stocks: {', '.join(top_symbols)}")
        
        logger.debug(f"Returning {len(top_stocks)} top consistent momentum stocks")
        return top_stocks

# Example usage and testing
if __name__ == "__main__":
    # Create momentum filter
    momentum_filter = MomentumConsistencyFilter()
    logger.info("Starting Momentum Consistency Filter example")
    
    # Sample market data
    sample_data = [
        {'symbol': 'AAPL', 'price': 150.0, 'volume': 50000000},
        {'symbol': 'MSFT', 'price': 300.0, 'volume': 30000000},
        {'symbol': 'GOOGL', 'price': 2500.0, 'volume': 20000000},
        {'symbol': 'AMZN', 'price': 3000.0, 'volume': 25000000},
        {'symbol': 'TSLA', 'price': 200.0, 'volume': 40000000},
        {'symbol': 'META', 'price': 250.0, 'volume': 35000000},
        {'symbol': 'NVDA', 'price': 400.0, 'volume': 45000000},
        {'symbol': 'NFLX', 'price': 350.0, 'volume': 15000000},
        {'symbol': 'CRM', 'price': 180.0, 'volume': 10000000},
        {'symbol': 'ADBE', 'price': 450.0, 'volume': 8000000}
    ]
    
    # Test momentum filter
    consistent_stocks = momentum_filter.filter_consistent_momentum_stocks(sample_data)
    
    print(f"\nConsistent momentum stocks: {len(consistent_stocks)}")
    for stock in consistent_stocks:
        print(f"  {stock['symbol']}: 6M={stock['momentum_6m']:.1%}, 5M={stock['momentum_5m']:.1%}")
    
    # Print stats
    stats = momentum_filter.get_momentum_stats()
    print(f"\nMomentum filter stats: {stats}")