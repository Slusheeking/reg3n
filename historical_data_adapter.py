#!/usr/bin/env python3

"""
Historical Data Adapter for Production-Native Backtesting
Feeds historical data through existing production polygon_client.py without modifications
"""

import asyncio
import json
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # ERROR
    YELLOW = '\033[93m'   # WARNING
    BLUE = '\033[94m'     # DEBUG
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # Reset to default

class SystemLogger:
    def __init__(self, name="historical_data_adapter"):
        self.name = name
        self.color_map = {
            'ERROR': Colors.RED,
            'WARNING': Colors.YELLOW,
            'DEBUG': Colors.BLUE,
            'INFO': Colors.WHITE
        }
        
        # Create logs directory and file
        self.log_dir = '/home/ubuntu/reg3n-1/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, 'backtesting.log')
        
    def _format_message(self, level: str, message: str, colored: bool = True) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if colored:
            color = self.color_map.get(level, Colors.WHITE)
            return f"[{timestamp}] - {color}{level}{Colors.RESET} - [{self.name}]: {message}"
        else:
            return f"[{timestamp}] - {level} - [{self.name}]: {message}"
    
    def _write_to_file(self, level: str, message: str):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(self._format_message(level, str(message), colored=False) + '\n')
        except Exception:
            pass  # Fail silently
        
    def info(self, message, extra=None):
        print(self._format_message("INFO", str(message)))
        self._write_to_file("INFO", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("INFO", f"    Extra: {extra}")
    
    def debug(self, message, extra=None):
        print(self._format_message("DEBUG", str(message)))
        self._write_to_file("DEBUG", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("DEBUG", f"    Extra: {extra}")
    
    def warning(self, message, extra=None):
        print(self._format_message("WARNING", str(message)))
        self._write_to_file("WARNING", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("WARNING", f"    Extra: {extra}")
    
    def error(self, message, extra=None):
        print(self._format_message("ERROR", str(message)))
        self._write_to_file("ERROR", str(message))
        if extra:
            print(f"    Extra: {extra}")
            self._write_to_file("ERROR", f"    Extra: {extra}")

logger = SystemLogger()

class HistoricalDataAdapter:
    """
    Historical data adapter that feeds data to production polygon_client.py
    Simulates real-time WebSocket streams using historical REST API data
    """
    
    def __init__(self, polygon_api_key: str, cache_enabled: bool = True):
        self.polygon_api_key = polygon_api_key
        self.cache_enabled = cache_enabled
        self.data_cache = {}
        self.base_url = "https://api.polygon.io"
        
        # Rate limiting for Polygon API
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("Historical Data Adapter initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting for Polygon API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_polygon_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make rate-limited request to Polygon API"""
        try:
            self._rate_limit()
            
            params['apikey'] = self.polygon_api_key
            url = f"{self.base_url}{endpoint}"
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Polygon API request failed: {e}")
            return None
    
    def get_historical_aggregates(self, date: str, symbols: List[str], 
                                timespan: str = "minute", multiplier: int = 1) -> Dict[str, List[Dict]]:
        """
        Get historical aggregates for symbols on a specific date
        Returns data formatted exactly like production WebSocket AS.{symbol} stream
        """
        logger.info(f"Fetching historical aggregates for {len(symbols)} symbols on {date}")
        
        cache_key = f"aggs_{date}_{timespan}_{multiplier}_{len(symbols)}"
        if self.cache_enabled and cache_key in self.data_cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        aggregates_data = {}
        
        for symbol in symbols:
            try:
                # Polygon aggregates endpoint
                endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{date}/{date}"
                params = {"adjusted": "true", "sort": "asc", "limit": 50000}
                
                response = self._make_polygon_request(endpoint, params)
                
                if response and response.get('status') == 'OK' and response.get('results'):
                    # Format data to match production WebSocket structure
                    formatted_aggregates = []
                    for result in response['results']:
                        # Convert Polygon REST format to WebSocket format
                        aggregate = {
                            'symbol': symbol,
                            'open': result.get('o', 0),
                            'high': result.get('h', 0),
                            'low': result.get('l', 0),
                            'close': result.get('c', 0),
                            'volume': result.get('v', 0),
                            'vwap': result.get('vw', result.get('c', 0)),
                            'timestamp': result.get('t', 0),
                            'transactions': result.get('n', 0),
                            'otc': False
                        }
                        formatted_aggregates.append(aggregate)
                    
                    aggregates_data[symbol] = formatted_aggregates
                    logger.debug(f"Retrieved {len(formatted_aggregates)} aggregates for {symbol}")
                else:
                    logger.warning(f"No aggregate data found for {symbol} on {date}")
                    aggregates_data[symbol] = []
                    
            except Exception as e:
                logger.error(f"Error fetching aggregates for {symbol}: {e}")
                aggregates_data[symbol] = []
        
        # Cache the results
        if self.cache_enabled:
            self.data_cache[cache_key] = aggregates_data
        
        logger.info(f"Retrieved aggregates for {len(aggregates_data)} symbols")
        return aggregates_data
    
    def get_historical_quotes(self, date: str, symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        Get historical quotes for symbols on a specific date
        Returns data formatted exactly like production WebSocket Q.{symbol} stream
        """
        logger.info(f"Fetching historical quotes for {len(symbols)} symbols on {date}")
        
        cache_key = f"quotes_{date}_{len(symbols)}"
        if self.cache_enabled and cache_key in self.data_cache:
            logger.debug(f"Using cached quotes for {cache_key}")
            return self.data_cache[cache_key]
        
        quotes_data = {}
        
        for symbol in symbols:
            try:
                # Polygon quotes endpoint
                endpoint = f"/v3/quotes/{symbol}"
                params = {
                    "timestamp.gte": f"{date}T09:30:00.000Z",
                    "timestamp.lt": f"{date}T16:00:00.000Z",
                    "order": "asc",
                    "limit": 50000
                }
                
                response = self._make_polygon_request(endpoint, params)
                
                if response and response.get('status') == 'OK' and response.get('results'):
                    # Format data to match production WebSocket structure
                    formatted_quotes = []
                    for result in response['results']:
                        # Convert Polygon REST format to WebSocket format
                        quote = {
                            'symbol': symbol,
                            'bid': result.get('bid', 0),
                            'ask': result.get('ask', 0),
                            'bid_size': result.get('bid_size', 0),
                            'ask_size': result.get('ask_size', 0),
                            'timestamp': result.get('participant_timestamp', 0),
                            'exchange': result.get('exchange', 0),
                            'sip_timestamp': result.get('sip_timestamp', 0)
                        }
                        formatted_quotes.append(quote)
                    
                    quotes_data[symbol] = formatted_quotes
                    logger.debug(f"Retrieved {len(formatted_quotes)} quotes for {symbol}")
                else:
                    logger.warning(f"No quote data found for {symbol} on {date}")
                    quotes_data[symbol] = []
                    
            except Exception as e:
                logger.error(f"Error fetching quotes for {symbol}: {e}")
                quotes_data[symbol] = []
        
        # Cache the results
        if self.cache_enabled:
            self.data_cache[cache_key] = quotes_data
        
        logger.info(f"Retrieved quotes for {len(quotes_data)} symbols")
        return quotes_data
    
    def get_market_data(self, date: str, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get comprehensive market data for symbols on a specific date
        Returns data formatted for production polygon_client.py consumption
        """
        logger.info(f"Fetching comprehensive market data for {len(symbols)} symbols on {date}")
        
        # Get aggregates and quotes
        aggregates = self.get_historical_aggregates(date, symbols)
        quotes = self.get_historical_quotes(date, symbols)
        
        # Combine into format expected by production system
        market_data = {}
        
        for symbol in symbols:
            symbol_aggregates = aggregates.get(symbol, [])
            symbol_quotes = quotes.get(symbol, [])
            
            # Get latest data points
            latest_aggregate = symbol_aggregates[-1] if symbol_aggregates else None
            latest_quote = symbol_quotes[-1] if symbol_quotes else None
            
            # Format for production consumption
            symbol_data = {
                'symbol': symbol,
                'aggregates': symbol_aggregates,
                'quotes': symbol_quotes,
                'timestamp': time.time(),
                'data_type': 'historical'
            }
            
            # Add latest price/volume data
            if latest_aggregate:
                # Calculate total daily volume from all aggregates (not just last minute)
                total_daily_volume = sum(agg.get('volume', 0) for agg in symbol_aggregates)
                
                symbol_data.update({
                    'price': latest_aggregate['close'],
                    'volume': total_daily_volume,  # Use total daily volume instead of last minute
                    'high': latest_aggregate['high'],
                    'low': latest_aggregate['low'],
                    'open': latest_aggregate['open'],
                    'vwap': latest_aggregate['vwap'],
                    'market_cap': self._estimate_market_cap(symbol, latest_aggregate['close']),
                    'daily_change': (latest_aggregate['close'] - latest_aggregate['open']) / latest_aggregate['open'] if latest_aggregate['open'] > 0 else 0,
                    'volatility': self._estimate_volatility(symbol_aggregates),
                    'momentum_score': self._calculate_momentum_score(symbol_aggregates)
                })
            
            # Add latest quote data
            if latest_quote:
                symbol_data.update({
                    'bid': latest_quote['bid'],
                    'ask': latest_quote['ask'],
                    'bid_size': latest_quote['bid_size'],
                    'ask_size': latest_quote['ask_size']
                })
            
            market_data[symbol] = symbol_data
        
        logger.info(f"Prepared market data for {len(market_data)} symbols")
        return market_data
    
    async def simulate_realtime_feed(self, start_date: str, end_date: str, symbols: List[str]):
        """
        Async generator that yields historical data in chronological order
        Simulates real-time data feed for backtesting
        """
        logger.info(f"Starting realtime simulation from {start_date} to {end_date}")
        
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        current_date = start_dt
        while current_date <= end_dt:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                date_str = current_date.strftime("%Y-%m-%d")
                logger.info(f"Processing historical data for {date_str}")
                
                # Get market data for this date
                daily_data = self.get_market_data(date_str, symbols)
                
                # Yield data in time-ordered chunks (simulate minute-by-minute)
                yield {
                    'date': date_str,
                    'data': daily_data,
                    'timestamp': current_date.timestamp()
                }
            
            current_date += timedelta(days=1)
        
        logger.info("Realtime simulation completed")
    
    def get_vix_data(self, date: str) -> float:
        """Get VIX data for market condition detection"""
        try:
            # Cache key for VIX data
            cache_key = f"vix_{date}"
            if self.cache_enabled and cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Try multiple VIX-related symbols
            vix_symbols = ['VIX', 'VIXM', 'UVIX', 'VIXY']
            
            for symbol in vix_symbols:
                try:
                    endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{date}/{date}"
                    params = {"adjusted": "true"}
                    
                    response = self._make_polygon_request(endpoint, params)
                    
                    if response and response.get('status') == 'OK' and response.get('results'):
                        vix_data = response['results'][0]
                        vix_value = float(vix_data.get('c', 20.0))
                        
                        # Cache the result
                        if self.cache_enabled:
                            self.data_cache[cache_key] = vix_value
                        
                        logger.debug(f"Found VIX data using {symbol} for {date}: {vix_value}")
                        return vix_value
                        
                except Exception as symbol_error:
                    logger.debug(f"Failed to get VIX data from {symbol}: {symbol_error}")
                    continue
            
            # If no VIX data found, use a reasonable default based on historical context
            # For 2024 dates, use typical VIX values from that period
            default_vix = self._get_historical_vix_estimate(date)
            
            if self.cache_enabled:
                self.data_cache[cache_key] = default_vix
            
            logger.debug(f"No VIX data found for {date}, using historical estimate: {default_vix}")
            return default_vix
                
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return 20.0
    
    def _get_historical_vix_estimate(self, date: str) -> float:
        """Get historical VIX estimate based on date and market conditions"""
        try:
            # Parse the date
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            year = date_obj.year
            month = date_obj.month
            
            # Historical VIX estimates based on typical market conditions
            if year == 2024:
                if month in [1, 2]:  # Early 2024 - moderate volatility
                    return 16.5
                elif month in [3, 4, 5]:  # Spring 2024 - lower volatility
                    return 14.8
                elif month in [6, 7, 8]:  # Summer 2024 - mixed volatility
                    return 17.2
                elif month in [9, 10]:  # Fall 2024 - election volatility
                    return 19.5
                else:  # Late 2024
                    return 18.0
            elif year == 2023:
                return 18.5  # 2023 average
            elif year == 2022:
                return 25.0  # Higher volatility in 2022
            elif year == 2021:
                return 22.0  # Post-pandemic volatility
            else:
                return 20.0  # Long-term average
                
        except Exception:
            return 20.0  # Fallback to long-term average
    
    def get_spy_data(self, date: str) -> Dict:
        """Get SPY data for market trend analysis"""
        try:
            endpoint = f"/v2/aggs/ticker/SPY/range/1/day/{date}/{date}"
            params = {"adjusted": "true"}
            
            response = self._make_polygon_request(endpoint, params)
            
            if response and response.get('status') == 'OK' and response.get('results'):
                spy_data = response['results'][0]
                return {
                    'open': spy_data.get('o', 0),
                    'high': spy_data.get('h', 0),
                    'low': spy_data.get('l', 0),
                    'close': spy_data.get('c', 0),
                    'volume': spy_data.get('v', 0),
                    'daily_change': (spy_data.get('c', 0) - spy_data.get('o', 0)) / spy_data.get('o', 1)
                }
            else:
                logger.warning(f"No SPY data found for {date}")
                return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'daily_change': 0}
                
        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'daily_change': 0}
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap based on symbol and price"""
        # Rough estimates for common symbols (in billions)
        market_cap_estimates = {
            'AAPL': 3000000000000,  # ~$3T
            'MSFT': 2800000000000,  # ~$2.8T
            'GOOGL': 1700000000000, # ~$1.7T
            'AMZN': 1500000000000,  # ~$1.5T
            'TSLA': 800000000000,   # ~$800B
            'NVDA': 1800000000000,  # ~$1.8T
            'META': 800000000000,   # ~$800B
            'JPM': 500000000000,    # ~$500B
            'JNJ': 400000000000,    # ~$400B
            'V': 500000000000,      # ~$500B
            'PG': 350000000000,     # ~$350B
            'UNH': 500000000000,    # ~$500B
            'HD': 350000000000,     # ~$350B
            'MA': 400000000000,     # ~$400B
            'SPY': 500000000000,    # ETF
            'QQQ': 200000000000,    # ETF
            'IWM': 50000000000,     # ETF
            'VIX': 1000000000       # VIX instrument
        }
        
        # Use known estimate if available, otherwise estimate based on price
        if symbol in market_cap_estimates:
            return market_cap_estimates[symbol]
        else:
            # Rough estimation: assume 1B shares outstanding for unknown symbols
            return price * 1000000000  # price * 1B shares
    
    def _estimate_volatility(self, aggregates: List[Dict]) -> float:
        """Estimate volatility from price movements"""
        if len(aggregates) < 2:
            return 0.02  # Default 2% volatility
        
        # Calculate returns
        returns = []
        for i in range(1, len(aggregates)):
            prev_close = aggregates[i-1].get('close', 0)
            curr_close = aggregates[i].get('close', 0)
            if prev_close > 0:
                returns.append((curr_close - prev_close) / prev_close)
        
        if not returns:
            return 0.02
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return min(max(volatility, 0.005), 0.5)  # Clamp between 0.5% and 50%
    
    def _calculate_momentum_score(self, aggregates: List[Dict]) -> float:
        """Calculate momentum score from price movements"""
        if len(aggregates) < 10:
            return 0.0
        
        # Compare recent average to older average
        recent_prices = [agg.get('close', 0) for agg in aggregates[-5:]]
        older_prices = [agg.get('close', 0) for agg in aggregates[-10:-5]]
        
        if not recent_prices or not older_prices:
            return 0.0
        
        recent_avg = sum(recent_prices) / len(recent_prices)
        older_avg = sum(older_prices) / len(older_prices)
        
        if older_avg > 0:
            momentum = (recent_avg - older_avg) / older_avg
            return max(min(momentum, 0.2), -0.2)  # Clamp between -20% and +20%
        
        return 0.0
    
    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys())
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the historical data adapter
    import os
    
    api_key = os.getenv('POLYGON_API_KEY', 'Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57')
    adapter = HistoricalDataAdapter(api_key)
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'NVDA']
    test_date = '2024-01-15'
    
    print("Testing Historical Data Adapter...")
    
    # Test aggregates
    aggregates = adapter.get_historical_aggregates(test_date, test_symbols)
    print(f"Retrieved aggregates for {len(aggregates)} symbols")
    
    # Test quotes
    quotes = adapter.get_historical_quotes(test_date, test_symbols)
    print(f"Retrieved quotes for {len(quotes)} symbols")
    
    # Test market data
    market_data = adapter.get_market_data(test_date, test_symbols)
    print(f"Retrieved market data for {len(market_data)} symbols")
    
    # Test VIX and SPY
    vix = adapter.get_vix_data(test_date)
    spy = adapter.get_spy_data(test_date)
    print(f"VIX: {vix}, SPY close: {spy.get('close', 0)}")
    
    print("Historical Data Adapter test completed")