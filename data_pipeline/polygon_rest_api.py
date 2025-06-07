#!/usr/bin/env python3

import asyncio
import logging
import time
import os
import sys
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import enhanced logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_system_logger

# Load environment variables
load_dotenv()

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'data_pipeline.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

# Initialize logger
logger = get_system_logger("data_pipeline.polygon_rest")

class PolygonRESTClient:
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Polygon REST client with pure HTTP and YAML config"""
        logger.startup({
            "api_key_provided": bool(api_key),
            "config_loaded": True
        })
        
        logger.log_data_flow("initialization", "rest_client")
        
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        
        if not self.api_key:
            error = ValueError("POLYGON_API_KEY is required")
            logger.error(error, {"error_type": "configuration_error"})
            raise error
        
        # Load config
        api_config = CONFIG.get('api', {}).get('polygon', {})
        rate_config = CONFIG.get('rate_limiting', {})
        metrics_config = CONFIG.get('metrics', {})
        
        # Configuration from YAML
        self.base_url = api_config.get('base_url', "https://api.polygon.io")
        self.timeout = api_config.get('timeout', 15)
        self.max_retries = api_config.get('max_retries', 5)
        self.batch_size = rate_config.get('rest_batch_size', 50)
        self.min_request_interval = rate_config.get('min_request_interval', 0.01)
        
        # Initialize session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.rate_limiter = {"last_request": 0}
        
        # Statistics tracking
        self.stats = {
            'requests': {'total': 0, 'successful': 0, 'failed': 0},
            'performance': {'total_response_time_ms': 0.0},
            'config_loaded': True,
            'yaml_config': {
                'timeout': self.timeout,
                'max_retries': self.max_retries,
                'batch_size': self.batch_size
            }
        }
        
        # Metrics tracking
        self.metrics_enabled = metrics_config.get('enabled', True)
        
        logger.info("Polygon REST client initialized successfully")
        logger.info(f"Config: timeout={self.timeout}s, retries={self.max_retries}, batch_size={self.batch_size}")
        
        logger.log_data_flow("initialization", "complete", data_size=self.batch_size)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to Polygon API"""
        if not self.api_key:
            logger.error(Exception("No API key available"), {"endpoint": endpoint})
            return None
        
        url = f"{self.base_url}{endpoint}"
        request_params = {'apikey': self.api_key}
        if params:
            request_params.update(params)
        
        logger.log_api_request("GET", url,
                             params={k: v for k, v in request_params.items() if k != 'apikey'})
        
        start_time = time.time()
        self.stats['requests']['total'] += 1
        
        try:
            response = self.session.get(url, params=request_params, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            self.stats['performance']['total_response_time_ms'] += response_time
            
            if response.status_code == 200:
                self.stats['requests']['successful'] += 1
                data = response.json()
                
                logger.log_api_response(response.status_code,
                                      response_size=len(str(data)),
                                      response_time=response_time)
                
                return data
            else:
                self.stats['requests']['failed'] += 1
                error = Exception(f"API request failed: {response.status_code}")
                logger.error(error, {
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                    "response_time_ms": response_time
                })
                return None
                
        except Exception as e:
            self.stats['requests']['failed'] += 1
            logger.error(e, {
                "endpoint": endpoint,
                "response_time_ms": (time.time() - start_time) * 1000
            })
            return None
    
    async def _rate_limit_check(self):
        """Check rate limiting before making requests using YAML config"""
        now = time.time()
        time_since_last = now - self.rate_limiter["last_request"]
        
        if time_since_last < self.min_request_interval:
            logger.debug(f"Rate limiting: waiting {self.min_request_interval - time_since_last:.3f}s")
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.rate_limiting("sleep", sleep_time * 1000)
            await asyncio.sleep(sleep_time)
        
        self.rate_limiter["last_request"] = time.time()
    
    def _track_request(self, success: bool, response_time_ms: float = 0):
        """Track request statistics with YAML config metrics"""
        if not self.metrics_enabled:
            return
            
        self.stats['requests']['total'] += 1
        if success:
            self.stats['requests']['successful'] += 1
        else:
            self.stats['requests']['failed'] += 1
        self.stats['performance']['total_response_time_ms'] += response_time_ms
        
        # Log performance if enabled
        if response_time_ms > (self.timeout * 1000 * 0.8):  # 80% of timeout
            logger.warning(f"Slow request detected: {response_time_ms:.2f}ms (timeout: {self.timeout}s)")
    
    # Market Status and Basic Data
    def get_market_status(self) -> Optional[Dict]:
        """Get current market status"""
        try:
            response = self._make_request("/v1/marketstatus/now")
            if response:
                self._track_request(True)
                return response
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_market_status"})
            return None
    
    # Snapshot Endpoints
    def get_single_snapshot(self, ticker: str) -> Optional[Dict]:
        """Get snapshot for a single ticker"""
        try:
            response = self._make_request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
            if response:
                self._track_request(True)
                return response.get('results', response)
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_single_snapshot", "ticker": ticker})
            return None
    
    def get_market_movers(self, direction: str = "gainers") -> Optional[List[Dict]]:
        """Get market movers (gainers/losers)"""
        try:
            if direction == "gainers":
                endpoint = "/v2/snapshot/locale/us/markets/stocks/gainers"
            elif direction == "losers":
                endpoint = "/v2/snapshot/locale/us/markets/stocks/losers"
            else:
                logger.error(Exception(f"Invalid direction: {direction}"), {"direction": direction})
                return None
            
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_market_movers", "direction": direction})
            return None
    
    def get_full_market_snapshot(self, limit: int = 50) -> Optional[Dict]:
        """Get full market snapshot"""
        try:
            response = self._make_request("/v2/snapshot/locale/us/markets/stocks/tickers")
            if response:
                self._track_request(True)
                
                # Convert to dict with ticker as key
                snapshot_dict = {}
                results = response.get('results', [])
                for i, item in enumerate(results):
                    if i >= limit:
                        break
                    ticker = item.get('ticker')
                    if ticker:
                        snapshot_dict[ticker] = item
                
                return snapshot_dict
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_full_market_snapshot"})
            return None
    
    # VIX and Market Breadth
    def get_vix_data(self) -> Optional[Dict]:
        """Get VIX data"""
        try:
            response = self._make_request("/v2/snapshot/locale/us/markets/indices/tickers/I:VIX")
            if response:
                self._track_request(True)
                return response.get('results', response)
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_vix_data"})
            return None
    
    def get_market_breadth_data(self, symbols: List[str] = None) -> Optional[Dict]:
        """Get market breadth data for regime detection"""
        try:
            if not symbols:
                symbols = ["SPY", "QQQ", "IWM"]
            
            breadth_data = {}
            for symbol in symbols:
                snapshot = self.get_single_snapshot(symbol)
                if snapshot:
                    breadth_data[symbol] = snapshot
            
            return breadth_data
        except Exception as e:
            logger.error(e, {"operation": "get_market_breadth_data"})
            return None
    
    # Aggregates (OHLCV) Data
    def get_aggregates(self, ticker: str, multiplier: int = 1, timespan: str = "day",
                      from_date: str = None, to_date: str = None, limit: int = 5000) -> Optional[List[Dict]]:
        """Get aggregates (OHLCV) data"""
        try:
            if not from_date:
                from_date = "2023-01-01"
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            
            params = {
                'multiplier': multiplier,
                'timespan': timespan,
                'from': from_date,
                'to': to_date,
                'limit': limit
            }
            
            endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            response = self._make_request(endpoint, params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_aggregates", "ticker": ticker})
            return None
    
    def get_previous_close(self, ticker: str) -> Optional[Dict]:
        """Get previous close for a ticker"""
        try:
            endpoint = f"/v2/aggs/ticker/{ticker}/prev"
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                results = response.get('results', [])
                return results[0] if results else None
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_previous_close", "ticker": ticker})
            return None
    
    def get_grouped_daily_aggs(self, date: str = None) -> Optional[List[Dict]]:
        """Get grouped daily aggregates for all tickers"""
        try:
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_grouped_daily_aggregates"})
            return None
    
    # Last Trade and Quote
    def get_last_trade(self, ticker: str) -> Optional[Dict]:
        """Get last trade for a ticker"""
        try:
            endpoint = f"/v2/last/trade/{ticker}"
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                return response.get('results', response)
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_last_trade", "ticker": ticker})
            return None
    
    def get_last_quote(self, ticker: str) -> Optional[Dict]:
        """Get last quote for a ticker"""
        try:
            endpoint = f"/v2/last/nbbo/{ticker}"
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                return response.get('results', response)
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_last_quote", "ticker": ticker})
            return None
    
    # News Endpoints
    def get_ticker_news(self, ticker: str, limit: int = 10) -> Optional[List[Dict]]:
        """Get news for a specific ticker"""
        try:
            params = {
                'ticker': ticker,
                'limit': limit,
                'order': 'desc'
            }
            response = self._make_request("/v2/reference/news", params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_ticker_news", "ticker": ticker})
            return None
    
    def get_market_news(self, limit: int = 100, days_back: int = 3) -> Optional[List[Dict]]:
        """Get general market news"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'limit': limit,
                'order': 'desc',
                'published_utc.gte': start_date.strftime('%Y-%m-%d'),
                'published_utc.lte': end_date.strftime('%Y-%m-%d')
            }
            response = self._make_request("/v2/reference/news", params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_market_news"})
            return None
    
    # Reference Data
    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get detailed information about a ticker"""
        try:
            endpoint = f"/v3/reference/tickers/{ticker}"
            response = self._make_request(endpoint)
            if response:
                self._track_request(True)
                return response.get('results', response)
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_ticker_details", "ticker": ticker})
            return None
    
    # Financials
    def get_ticker_financials(self, ticker: str, timeframe: str = "annual", limit: int = 1) -> Optional[List[Dict]]:
        """Get financials for a specific ticker"""
        try:
            params = {
                'timeframe': timeframe,
                'limit': limit,
                'order': 'desc'
            }
            endpoint = "/vX/reference/financials"
            params['ticker'] = ticker
            response = self._make_request(endpoint, params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_ticker_financials", "ticker": ticker})
            return None
    
    # Dividends
    def get_dividends(self, ticker: str = None, ex_dividend_date: str = None,
                     record_date: str = None, declaration_date: str = None,
                     pay_date: str = None, frequency: int = None,
                     cash_amount: float = None, dividend_type: str = None,
                     order: str = "desc", limit: int = 10) -> Optional[List[Dict]]:
        """Get dividend data from Polygon API"""
        try:
            time.time()
            
            # Build parameters dict
            params = {
                'order': order,
                'limit': limit
            }
            
            if ticker:
                params['ticker'] = ticker
            if ex_dividend_date:
                params['ex_dividend_date'] = ex_dividend_date
            if record_date:
                params['record_date'] = record_date
            if declaration_date:
                params['declaration_date'] = declaration_date
            if pay_date:
                params['pay_date'] = pay_date
            if frequency is not None:
                params['frequency'] = frequency
            if cash_amount is not None:
                params['cash_amount'] = cash_amount
            if dividend_type:
                params['dividend_type'] = dividend_type
            
            response = self._make_request("/v3/reference/dividends", params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_dividends", "ticker": ticker})
            return None
    
    # Stock Splits
    def get_stock_splits(self, ticker: str = None, execution_date: str = None,
                        reverse_split: bool = None, order: str = "desc",
                        limit: int = 10, sort: str = "execution_date") -> Optional[List[Dict]]:
        """Get stock splits data from Polygon API"""
        try:
            time.time()
            
            # Build parameters dict
            params = {
                'order': order,
                'limit': limit,
                'sort': sort
            }
            
            if ticker:
                params['ticker'] = ticker
            if execution_date:
                params['execution_date'] = execution_date
            if reverse_split is not None:
                params['reverse_split'] = reverse_split
            
            response = self._make_request("/v3/reference/splits", params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_stock_splits", "ticker": ticker})
            return None
    
    def get_stock_splits_history(self, ticker: str, years_back: int = 5) -> Optional[List[Dict]]:
        """Get historical stock splits for a ticker"""
        try:
            from datetime import datetime, timedelta
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            params = {
                'ticker': ticker,
                'execution_date.gte': start_date.strftime('%Y-%m-%d'),
                'execution_date.lte': end_date.strftime('%Y-%m-%d'),
                'order': 'desc',
                'limit': 100
            }
            response = self._make_request("/v3/reference/splits", params)
            if response:
                self._track_request(True)
                return response.get('results', [])
            else:
                self._track_request(False)
                return None
        except Exception as e:
            self._track_request(False)
            logger.error(e, {"operation": "get_stock_splits_history", "ticker": ticker})
            return None

    # Utility Methods
    def get_stats(self) -> Dict:
        """Get client statistics"""
        total_requests = self.stats['requests']['total']
        if total_requests > 0:
            success_rate = (self.stats['requests']['successful'] / total_requests) * 100
            avg_response_time = self.stats['performance']['total_response_time_ms'] / total_requests
        else:
            success_rate = 0
            avg_response_time = 0
        
        return {
            'requests': {
                'total': total_requests,
                'successful': self.stats['requests']['successful'],
                'failed': self.stats['requests']['failed'],
                'success_rate_pct': success_rate
            },
            'performance': {
                'avg_response_time_ms': avg_response_time
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        stats = self.get_stats()
        if stats['requests']['total'] == 0:
            return True  # No requests made yet
        
        return stats['requests']['success_rate_pct'] > 50


# Example usage
async def main():
    """Example usage of the Polygon REST client with pure HTTP"""
    
    # Create client (uses environment variables)
    client = PolygonRESTClient()
    
    try:
        # Test market status
        market_status = client.get_market_status()
        print(f"Market Status: {market_status}")
        
        # Test single snapshot
        aapl_snapshot = client.get_single_snapshot("AAPL")
        print(f"AAPL Snapshot: {aapl_snapshot}")
        
        # Test market movers
        gainers = client.get_market_movers("gainers")
        print(f"Top Gainers: {len(gainers) if gainers else 0}")
        
        # Test VIX data
        vix_data = client.get_vix_data()
        print(f"VIX Data: {vix_data}")
        
        # Test aggregates
        aapl_aggs = client.get_aggregates("AAPL", timespan="minute", limit=10)
        print(f"AAPL Aggregates: {len(aapl_aggs) if aapl_aggs else 0}")
        
        # Print stats
        stats = client.get_stats()
        print(f"Client Stats: {stats}")
        
    except Exception as e:
        logger.error(e, {"operation": "main"})


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())