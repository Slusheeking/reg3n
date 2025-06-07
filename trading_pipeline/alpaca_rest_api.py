#!/usr/bin/env python3

import asyncio
import time
import os
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Import unified system logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_logger

# Load environment variables
load_dotenv()

# Load YAML configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml', 'trading_pipeline.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

# Initialize component logger
logger = get_logger("alpaca_rest_api")


@dataclass
class OrderRequest:
    """Order request data structure"""
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    type: str = 'market'  # 'market', 'limit', 'stop', 'stop_limit'
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    extended_hours: bool = False
    client_order_id: Optional[str] = None


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    qty: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    lastday_price: float
    change_today: float


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    created_at: str
    updated_at: str
    submitted_at: Optional[str]
    filled_at: Optional[str]


class AlpacaRESTClient:
    """
    Alpaca REST API client for trading operations
    Handles account management, orders, and positions using YAML config
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Initialize Alpaca REST client with YAML config"""
        logger.info("Initializing Alpaca REST client", extra={
            "component": "alpaca_rest_api",
            "action": "initialization_start"
        })
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            logger.error("Missing required Alpaca API credentials", extra={
                "component": "alpaca_rest_api",
                "action": "initialization_error",
                "error_type": "configuration_error",
                "has_api_key": bool(self.api_key),
                "has_secret_key": bool(self.secret_key)
            })
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        
        # Load config
        alpaca_config = CONFIG.get('api', {}).get('alpaca', {})
        rate_config = CONFIG.get('rate_limiting', {})
        
        self.base_url = alpaca_config.get('base_url', "https://paper-api.alpaca.markets")
        self.environment = alpaca_config.get('environment', 'paper')
        self.timeout = 30  # Alpaca timeout
        
        # Rate limiting
        self.rate_limiter = {
            "last_request": 0,
            "min_interval": rate_config.get('min_request_interval', 0.01)
        }
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        })
        
        # Statistics tracking
        self.stats = {
            'requests': {'total': 0, 'successful': 0, 'failed': 0},
            'orders': {'submitted': 0, 'filled': 0, 'cancelled': 0, 'rejected': 0},
            'performance': {'total_response_time_ms': 0.0}
        }
        
        logger.info("Alpaca REST client initialized successfully", extra={
            "component": "alpaca_rest_api",
            "action": "initialization_complete",
            "environment": self.environment,
            "base_url": self.base_url,
            "rate_limiting": True,
            "timeout": self.timeout,
            "retry_strategy": True
        })
    
    async def _rate_limit_check(self):
        """Check rate limiting before making requests"""
        now = time.time()
        time_since_last = now - self.rate_limiter["last_request"]
        
        if time_since_last < self.rate_limiter["min_interval"]:
            sleep_time = self.rate_limiter["min_interval"] - time_since_last
            await asyncio.sleep(sleep_time)
            logger.debug("Rate limiting delay applied", extra={
                "component": "alpaca_rest_api",
                "action": "rate_limiting",
                "delay_ms": sleep_time * 1000
            })
        
        self.rate_limiter["last_request"] = time.time()
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to Alpaca API"""
        url = f"{self.base_url}{endpoint}"
        
        logger.debug("API request initiated", extra={
            "component": "alpaca_rest_api",
            "action": "api_request",
            "method": method.upper(),
            "url": url,
            "has_params": bool(data)
        })
        
        start_time = time.time()
        self.stats['requests']['total'] += 1
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=data, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, timeout=self.timeout)
            elif method.upper() == 'PATCH':
                response = self.session.patch(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = (time.time() - start_time) * 1000
            self.stats['performance']['total_response_time_ms'] += response_time
            
            if response.status_code in [200, 201, 204]:
                self.stats['requests']['successful'] += 1
                result = response.json() if response.content else {}
                
                logger.log_api_response(response.status_code, response_size=len(str(result)), response_time=response_time)
                
                return result
            else:
                self.stats['requests']['failed'] += 1
                Exception(f"API request failed: {response.status_code} - {response.text}")
                logger.error(Exception(f"API request failed: {response.status_code} - {response.text}"), {
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
                "method": method,
                "response_time_ms": (time.time() - start_time) * 1000
            })
            return None
    
    # Account Management
    def get_account(self) -> Optional[Dict]:
        """Get account information"""
        try:
            response = self._make_request('GET', '/v2/account')
            if response:
                logger.debug(f"Account info retrieved: ${response.get('equity', 'N/A')} equity", extra={
                    "component": "alpaca_rest_api",
                    "equity": response.get('equity', 'N/A')
                })
            return response
        except Exception as e:
            logger.error(f"Account info error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/account",
                "error": str(e)
            })
            return None
    
    def get_portfolio_history(self, period: str = '1D', timeframe: str = '1Min') -> Optional[Dict]:
        """Get portfolio history"""
        try:
            params = {'period': period, 'timeframe': timeframe}
            response = self._make_request('GET', '/v2/account/portfolio/history', params)
            return response
        except Exception as e:
            logger.error(f"Portfolio history error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/account/portfolio/history",
                "error": str(e)
            })
            return None
    
    # Position Management
    def get_positions(self) -> Optional[List[Position]]:
        """Get all positions"""
        try:
            response = self._make_request('GET', '/v2/positions')
            if response:
                positions = []
                for pos_data in response:
                    position = Position(
                        symbol=pos_data['symbol'],
                        qty=float(pos_data['qty']),
                        side=pos_data['side'],
                        market_value=float(pos_data['market_value']),
                        cost_basis=float(pos_data['cost_basis']),
                        unrealized_pl=float(pos_data['unrealized_pl']),
                        unrealized_plpc=float(pos_data['unrealized_plpc']),
                        current_price=float(pos_data['current_price']),
                        lastday_price=float(pos_data['lastday_price']),
                        change_today=float(pos_data['change_today'])
                    )
                    positions.append(position)
                
                logger.debug(f"Retrieved {len(positions)} positions", extra={
                    "component": "alpaca_rest_api",
                    "position_count": len(positions)
                })
                return positions
            return []
        except Exception as e:
            logger.error(f"Positions retrieval error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/positions",
                "error": str(e)
            })
            return None
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            response = self._make_request('GET', f'/v2/positions/{symbol}')
            if response:
                position = Position(
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    market_value=float(response['market_value']),
                    cost_basis=float(response['cost_basis']),
                    unrealized_pl=float(response['unrealized_pl']),
                    unrealized_plpc=float(response['unrealized_plpc']),
                    current_price=float(response['current_price']),
                    lastday_price=float(response['lastday_price']),
                    change_today=float(response['change_today'])
                )
                return position
            return None
        except Exception as e:
            logger.error(f"Position retrieval error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": f"/v2/positions/{symbol}",
                "symbol": symbol,
                "error": str(e)
            })
            return None
    
    def close_position(self, symbol: str, qty: Optional[float] = None, percentage: Optional[float] = None) -> Optional[Dict]:
        """Close position (partial or full)"""
        try:
            data = {}
            if qty:
                data['qty'] = str(qty)
            if percentage:
                data['percentage'] = str(percentage)
            
            response = self._make_request('DELETE', f'/v2/positions/{symbol}', data)
            if response:
                logger.info(f"Position closed for {symbol}", extra={
                    "component": "alpaca_rest_api",
                    "symbol": symbol,
                    "action": "position_closed"
                })
            return response
        except Exception as e:
            logger.error(f"Position close error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": f"/v2/positions/{symbol}",
                "symbol": symbol,
                "action": "close",
                "error": str(e)
            })
            return None
    
    def close_all_positions(self, cancel_orders: bool = True) -> Optional[List[Dict]]:
        """Close all positions"""
        try:
            params = {'cancel_orders': cancel_orders}
            response = self._make_request('DELETE', '/v2/positions', params)
            if response:
                logger.info(f"All positions closed: {len(response)} positions", extra={
                    "component": "alpaca_rest_api",
                    "closed_count": len(response),
                    "action": "all_positions_closed"
                })
            return response
        except Exception as e:
            logger.error(f"Close all positions error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/positions",
                "action": "close_all",
                "error": str(e)
            })
            return None
    
    # Order Management
    def submit_order(self, order_request: OrderRequest) -> Optional[Order]:
        """Submit a new order"""
        try:
            data = {
                'symbol': order_request.symbol,
                'qty': str(order_request.qty),
                'side': order_request.side,
                'type': order_request.type,
                'time_in_force': order_request.time_in_force,
                'extended_hours': order_request.extended_hours
            }
            
            if order_request.limit_price:
                data['limit_price'] = str(order_request.limit_price)
            if order_request.stop_price:
                data['stop_price'] = str(order_request.stop_price)
            if order_request.client_order_id:
                data['client_order_id'] = order_request.client_order_id
            
            response = self._make_request('POST', '/v2/orders', data)
            if response:
                self.stats['orders']['submitted'] += 1
                order = Order(
                    id=response['id'],
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    order_type=response['type'],
                    status=response['status'],
                    filled_qty=float(response.get('filled_qty', 0)),
                    filled_avg_price=float(response['filled_avg_price']) if response.get('filled_avg_price') else None,
                    created_at=response['created_at'],
                    updated_at=response['updated_at'],
                    submitted_at=response.get('submitted_at'),
                    filled_at=response.get('filled_at')
                )
                
                logger.info(f"Order submitted: {order_request.side} {order_request.qty} {order_request.symbol} @ {order_request.type}", extra={
                    "component": "alpaca_rest_api",
                    "side": order_request.side,
                    "qty": order_request.qty,
                    "symbol": order_request.symbol,
                    "order_type": order_request.type,
                    "action": "order_submitted"
                })
                return order
            return None
        except Exception as e:
            self.stats['orders']['rejected'] += 1
            logger.error(f"Order submission error for {order_request.symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "action": "submit_order",
                "symbol": order_request.symbol,
                "error": str(e)
            })
            return None
    
    def get_orders(self, status: str = 'all', limit: int = 50, direction: str = 'desc') -> Optional[List[Order]]:
        """Get orders"""
        try:
            params = {
                'status': status,
                'limit': limit,
                'direction': direction
            }
            response = self._make_request('GET', '/v2/orders', params)
            if response:
                orders = []
                for order_data in response:
                    order = Order(
                        id=order_data['id'],
                        symbol=order_data['symbol'],
                        qty=float(order_data['qty']),
                        side=order_data['side'],
                        order_type=order_data['type'],
                        status=order_data['status'],
                        filled_qty=float(order_data.get('filled_qty', 0)),
                        filled_avg_price=float(order_data['filled_avg_price']) if order_data.get('filled_avg_price') else None,
                        created_at=order_data['created_at'],
                        updated_at=order_data['updated_at'],
                        submitted_at=order_data.get('submitted_at'),
                        filled_at=order_data.get('filled_at')
                    )
                    orders.append(order)
                
                logger.debug(f"Retrieved {len(orders)} orders", extra={
                    "component": "alpaca_rest_api",
                    "order_count": len(orders)
                })
                return orders
            return []
        except Exception as e:
            logger.error(f"Orders retrieval error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/orders",
                "error": str(e)
            })
            return None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get specific order"""
        try:
            response = self._make_request('GET', f'/v2/orders/{order_id}')
            if response:
                order = Order(
                    id=response['id'],
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    order_type=response['type'],
                    status=response['status'],
                    filled_qty=float(response.get('filled_qty', 0)),
                    filled_avg_price=float(response['filled_avg_price']) if response.get('filled_avg_price') else None,
                    created_at=response['created_at'],
                    updated_at=response['updated_at'],
                    submitted_at=response.get('submitted_at'),
                    filled_at=response.get('filled_at')
                )
                return order
            return None
        except Exception as e:
            logger.error(f"Order retrieval error for {order_id}: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": f"/v2/orders/{order_id}",
                "order_id": order_id,
                "error": str(e)
            })
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            response = self._make_request('DELETE', f'/v2/orders/{order_id}')
            if response is not None:  # 204 returns empty response
                self.stats['orders']['cancelled'] += 1
                logger.info(f"Order cancelled: {order_id}", extra={
                    "component": "alpaca_rest_api",
                    "order_id": order_id,
                    "action": "order_cancelled"
                })
                return True
            return False
        except Exception as e:
            logger.error(f"Order cancellation error for {order_id}: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": f"/v2/orders/{order_id}",
                "order_id": order_id,
                "action": "cancel",
                "error": str(e)
            })
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all orders"""
        try:
            response = self._make_request('DELETE', '/v2/orders')
            if response is not None:
                cancelled_count = len(response) if response else 0
                self.stats['orders']['cancelled'] += cancelled_count
                logger.info(f"All orders cancelled: {cancelled_count} orders", extra={
                    "component": "alpaca_rest_api",
                    "cancelled_count": cancelled_count,
                    "action": "all_orders_cancelled"
                })
                return True
            return False
        except Exception as e:
            logger.error(f"Cancel all orders error: {e}", extra={
                "component": "alpaca_rest_api",
                "endpoint": "/v2/orders",
                "action": "cancel_all",
                "error": str(e)
            })
            return False
    
    # Advanced Order Types for Momentum Trading
    def submit_bracket_order(self, symbol: str, qty: int, side: str = 'buy',
                           stop_loss_price: Optional[float] = None,
                           take_profit_price: Optional[float] = None) -> Optional[Order]:
        """Submit bracket order with automatic stop loss and take profit"""
        try:
            data = {
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': 'market',
                'time_in_force': 'day',
                'order_class': 'bracket'
            }
            
            if stop_loss_price:
                data['stop_loss'] = {
                    'stop_price': str(round(stop_loss_price, 2))
                }
            
            if take_profit_price:
                data['take_profit'] = {
                    'limit_price': str(round(take_profit_price, 2))
                }
            
            response = self._make_request('POST', '/v2/orders', data)
            if response:
                self.stats['orders']['submitted'] += 1
                order = Order(
                    id=response['id'],
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    order_type=response['type'],
                    status=response['status'],
                    filled_qty=float(response.get('filled_qty', 0)),
                    filled_avg_price=float(response['filled_avg_price']) if response.get('filled_avg_price') else None,
                    created_at=response['created_at'],
                    updated_at=response['updated_at'],
                    submitted_at=response.get('submitted_at'),
                    filled_at=response.get('filled_at')
                )
                
                logger.info(f"Bracket order submitted: {side} {qty} {symbol} "
                           f"Stop: ${stop_loss_price:.2f} TP: ${take_profit_price:.2f}", extra={
                    "component": "alpaca_rest_api",
                    "side": side,
                    "qty": qty,
                    "symbol": symbol,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "action": "bracket_order_submitted"
                })
                return order
            return None
            
        except Exception as e:
            self.stats['orders']['rejected'] += 1
            logger.error(f"Bracket order error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "symbol": symbol,
                "error": str(e),
                "action": "bracket_order_error"
            })
            return None
    
    def submit_conditional_limit_order(self, symbol: str, qty: int, limit_price: float,
                                     side: str = 'sell') -> Optional[Order]:
        """Submit conditional limit order for tier 2 profit taking"""
        try:
            data = {
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': 'limit',
                'limit_price': str(round(limit_price, 2)),
                'time_in_force': 'gtc'
            }
            
            response = self._make_request('POST', '/v2/orders', data)
            if response:
                self.stats['orders']['submitted'] += 1
                order = Order(
                    id=response['id'],
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    order_type=response['type'],
                    status=response['status'],
                    filled_qty=float(response.get('filled_qty', 0)),
                    filled_avg_price=float(response['filled_avg_price']) if response.get('filled_avg_price') else None,
                    created_at=response['created_at'],
                    updated_at=response['updated_at'],
                    submitted_at=response.get('submitted_at'),
                    filled_at=response.get('filled_at')
                )
                
                logger.info(f"Conditional limit order: {side} {qty} {symbol} @ ${limit_price:.2f}", extra={
                    "component": "alpaca_rest_api",
                    "side": side,
                    "qty": qty,
                    "symbol": symbol,
                    "limit_price": limit_price,
                    "action": "conditional_order_submitted"
                })
                return order
            return None
            
        except Exception as e:
            self.stats['orders']['rejected'] += 1
            logger.error(f"Conditional order error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "symbol": symbol,
                "error": str(e),
                "action": "conditional_order_error"
            })
            return None
    
    def submit_trailing_stop_order(self, symbol: str, qty: int, trail_percent: float,
                                 side: str = 'sell') -> Optional[Order]:
        """Submit trailing stop order for tier 3"""
        try:
            data = {
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': 'trailing_stop',
                'trail_percent': str(trail_percent),
                'time_in_force': 'gtc'
            }
            
            response = self._make_request('POST', '/v2/orders', data)
            if response:
                self.stats['orders']['submitted'] += 1
                order = Order(
                    id=response['id'],
                    symbol=response['symbol'],
                    qty=float(response['qty']),
                    side=response['side'],
                    order_type=response['type'],
                    status=response['status'],
                    filled_qty=float(response.get('filled_qty', 0)),
                    filled_avg_price=float(response['filled_avg_price']) if response.get('filled_avg_price') else None,
                    created_at=response['created_at'],
                    updated_at=response['updated_at'],
                    submitted_at=response.get('submitted_at'),
                    filled_at=response.get('filled_at')
                )
                
                logger.info(f"Trailing stop order: {side} {qty} {symbol} trail {trail_percent}%", extra={
                    "component": "alpaca_rest_api",
                    "side": side,
                    "qty": qty,
                    "symbol": symbol,
                    "trail_percent": trail_percent,
                    "action": "trailing_stop_submitted"
                })
                return order
            return None
            
        except Exception as e:
            self.stats['orders']['rejected'] += 1
            logger.error(f"Trailing stop error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "symbol": symbol,
                "error": str(e),
                "action": "trailing_stop_error"
            })
            return None
    
    def submit_momentum_trade_package(self, order_package: Dict) -> Optional[Dict]:
        """Submit complete momentum trade package - set and forget"""
        symbol = order_package['symbol']
        
        logger.info("Momentum trade package submission initiated", extra={
            "component": "alpaca_rest_api",
            "action": "submit_momentum_package",
            "symbol": symbol,
            "total_qty": order_package.get('total_qty', 0),
            "tier_quantities": order_package.get('tier_quantities', {}),
            "prices": order_package.get('prices', {})
        })
        
        try:
            submitted_orders = {}
            
            logger.info(f"Submitting momentum trade package for {symbol}", extra={
                "component": "alpaca_rest_api",
                "symbol": symbol,
                "action": "momentum_package_submission"
            })
            
            # 1. Main bracket order (Tier 1: Entry + Stop + TP1)
            main_order = self.submit_bracket_order(
                symbol=symbol,
                qty=order_package['tier_quantities']['tier1'],
                stop_loss_price=order_package['prices']['stop_loss'],
                take_profit_price=order_package['prices']['tp1_target']
            )
            submitted_orders['tier1_bracket'] = main_order
            
            # 2. Tier 2 limit order (+3% target)
            if order_package['tier_quantities']['tier2'] > 0:
                tier2_order = self.submit_conditional_limit_order(
                    symbol=symbol,
                    qty=order_package['tier_quantities']['tier2'],
                    limit_price=order_package['prices']['tp2_target']
                )
                submitted_orders['tier2_limit'] = tier2_order
            
            # 3. Tier 3 trailing stop
            if order_package['tier_quantities']['tier3'] > 0:
                tier3_order = self.submit_trailing_stop_order(
                    symbol=symbol,
                    qty=order_package['tier_quantities']['tier3'],
                    trail_percent=order_package['prices']['trail_percent']
                )
                submitted_orders['tier3_trailing'] = tier3_order
            
            # Log successful submission
            logger.info(f"Complete momentum package submitted for {symbol}: "
                       f"Total {order_package['total_qty']} shares, "
                       f"Stop: -1.5%, Targets: +1%, +3%, trailing, Time Exit: 3:45 PM")
            
            return submitted_orders
            
        except Exception as e:
            logger.error(f"Momentum package submission error for {symbol}: {e}", extra={
                "component": "alpaca_rest_api",
                "symbol": symbol,
                "error": str(e),
                "action": "momentum_package_error"
            })
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
            'orders': self.stats['orders'],
            'performance': {
                'avg_response_time_ms': avg_response_time
            },
            'environment': self.environment
        }
    
    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        stats = self.get_stats()
        if stats['requests']['total'] == 0:
            return True  # No requests made yet
        
        return stats['requests']['success_rate_pct'] > 80  # 80% success rate threshold


# Example usage
async def main():
    """Example usage of Alpaca REST client"""
    
    try:
        # Create client
        client = AlpacaRESTClient()
        
        # Get account info
        account = client.get_account()
        if account:
            logger.info(f"Account equity: ${account.get('equity', 'N/A')}", extra={
                "component": "alpaca_rest_api",
                "equity": account.get('equity', 'N/A')
            })
        
        # Get positions
        positions = client.get_positions()
        if positions:
            logger.info(f"Current positions: {len(positions)}", extra={
                "component": "alpaca_rest_api",
                "position_count": len(positions)
            })
            for pos in positions[:5]:  # Show first 5
                logger.info(f"  {pos.symbol}: {pos.qty} shares, P&L: ${pos.unrealized_pl:.2f}", extra={
                    "component": "alpaca_rest_api",
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "unrealized_pl": pos.unrealized_pl
                })
        
        # Get recent orders
        orders = client.get_orders(limit=10)
        if orders:
            logger.info(f"Recent orders: {len(orders)}", extra={
                "component": "alpaca_rest_api",
                "order_count": len(orders)
            })
            for order in orders[:3]:  # Show first 3
                logger.info(f"  {order.symbol}: {order.side} {order.qty} @ {order.status}", extra={
                    "component": "alpaca_rest_api",
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": order.qty,
                    "status": order.status
                })
        
        # Print stats
        stats = client.get_stats()
        logger.info("Client performance statistics", extra={
            "component": "alpaca_rest_api",
            "action": "performance_stats",
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Main execution error: {e}", extra={
            "component": "alpaca_rest_api",
            "error": str(e)
        })


if __name__ == "__main__":
    asyncio.run(main())