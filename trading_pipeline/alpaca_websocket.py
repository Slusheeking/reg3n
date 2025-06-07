#!/usr/bin/env python3

import asyncio
import json
import time
import os
import yaml
import websockets
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

# Import enhanced logging
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

# Initialize logger
logger = get_logger("alpaca_websocket")


@dataclass
class TradeUpdate:
    """Trade update data structure"""
    event: str  # 'new', 'fill', 'partial_fill', 'canceled', 'expired', 'done_for_day', 'replaced'
    order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    price: Optional[float]
    timestamp: str
    position_qty: Optional[float] = None


@dataclass
class AccountUpdate:
    """Account update data structure"""
    event: str  # 'account_update'
    buying_power: float
    cash: float
    portfolio_value: float
    timestamp: str


class ConnectionHealthMonitor:
    """Monitor WebSocket connection health for Alpaca"""
    
    def __init__(self):
        health_config = CONFIG.get('health_monitoring', {})
        self.heartbeat_interval = health_config.get('heartbeat_interval', 30)
        self.max_reconnect_attempts = health_config.get('max_reconnect_attempts', 10)
        self.data_timeout_seconds = health_config.get('data_timeout_seconds', 60)
        self.last_heartbeat = time.time()
        self.last_data_received = time.time()
        self.connection_status = "disconnected"
        self.reconnect_count = 0
        self.total_messages_received = 0
        
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = time.time()
        
    def update_data_received(self):
        """Update data received timestamp"""
        self.last_data_received = time.time()
        self.total_messages_received += 1
        
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        now = time.time()
        heartbeat_ok = (now - self.last_heartbeat) < (self.heartbeat_interval * 2)
        data_flow_ok = (now - self.last_data_received) < self.data_timeout_seconds
        return heartbeat_ok and data_flow_ok and self.connection_status == "connected"
    
    def get_status(self) -> Dict:
        """Get connection status"""
        now = time.time()
        return {
            "status": self.connection_status,
            "last_heartbeat_seconds_ago": now - self.last_heartbeat,
            "last_data_seconds_ago": now - self.last_data_received,
            "reconnect_count": self.reconnect_count,
            "total_messages": self.total_messages_received,
            "is_healthy": self.is_healthy()
        }


class AlpacaWebSocketClient:
    """
    Alpaca WebSocket client for real-time trading updates
    Handles trade confirmations, account updates, and order status changes
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 trade_callback: Optional[Callable] = None, account_callback: Optional[Callable] = None):
        """Initialize Alpaca WebSocket client"""
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        
        # Load config
        alpaca_config = CONFIG.get('api', {}).get('alpaca', {})
        health_config = CONFIG.get('health_monitoring', {})
        
        self.websocket_url = alpaca_config.get('websocket_url', "wss://stream.data.alpaca.markets/v2")
        self.environment = alpaca_config.get('environment', 'paper')
        
        # Use paper trading URL if in paper mode
        if self.environment == 'paper':
            self.websocket_url = "wss://paper-api.alpaca.markets/stream"
        
        self.websocket = None
        self.health_monitor = ConnectionHealthMonitor()
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = health_config.get('max_reconnect_attempts', 10)
        self.reconnect_backoff_base = health_config.get('reconnect_backoff_base', 1.0)
        self.reconnect_backoff_max = health_config.get('reconnect_backoff_max', 60.0)
        
        # Callbacks
        self.trade_callback = trade_callback
        self.account_callback = account_callback
        
        # Statistics
        self.stats = {
            'messages': {'total': 0, 'trade_updates': 0, 'account_updates': 0, 'errors': 0},
            'connection': {'connects': 0, 'disconnects': 0, 'reconnects': 0}
        }
        
        logger.info("Alpaca WebSocket client initialized", extra={
            "component": "alpaca_websocket",
            "action": "initialization_complete",
            "environment": self.environment,
            "websocket_url": self.websocket_url,
            "trade_callback": trade_callback is not None,
            "account_callback": account_callback is not None
        })
    
    async def connect(self):
        """Connect to Alpaca WebSocket"""
        try:
            logger.info("Connecting to Alpaca WebSocket", extra={
                "component": "alpaca_websocket",
                "action": "connection_attempt",
                "websocket_url": self.websocket_url
            })
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.websocket_url)
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            
            if auth_data.get("T") == "success":
                self.is_connected = True
                self.health_monitor.connection_status = "connected"
                self.health_monitor.reconnect_count = self.reconnect_attempts
                self.reconnect_attempts = 0
                self.stats['connection']['connects'] += 1
                
                logger.info("WebSocket connected successfully", extra={
                    "component": "alpaca_websocket",
                    "action": "connection_established",
                    "environment": self.environment,
                    "auth_status": "success"
                })
                
                # Subscribe to trading updates
                await self._subscribe_to_updates()
                
                # Start message handling
                asyncio.create_task(self._message_handler())
                asyncio.create_task(self._heartbeat_handler())
                
            else:
                Exception(f"Authentication failed: {auth_data}")
                logger.error(Exception(f"Authentication failed: {auth_data}"), {"auth_data": auth_data})
                await self._handle_reconnection()
            
        except Exception as e:
            logger.error(f"Connection error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "connect",
                "error": str(e)
            })
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()
    
    async def _subscribe_to_updates(self):
        """Subscribe to trading and account updates"""
        try:
            # Subscribe to trade updates and account updates
            subscribe_message = {
                "action": "listen",
                "data": {
                    "streams": ["trade_updates", "account_updates"]
                }
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to trade_updates and account_updates", extra={
                "component": "alpaca_websocket",
                "action": "subscription_complete",
                "streams": ["trade_updates", "account_updates"]
            })
            
        except Exception as e:
            logger.error(f"Subscription error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "subscribe",
                "error": str(e)
            })
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                self.health_monitor.update_data_received()
                self.stats['messages']['total'] += 1
                
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}", extra={
                        "component": "alpaca_websocket",
                        "operation": "parse_json_message",
                        "error": str(e)
                    })
                    self.stats['messages']['errors'] += 1
                except Exception as e:
                    logger.error(f"Message processing error: {e}", extra={
                        "component": "alpaca_websocket",
                        "operation": "process_message",
                        "error": str(e)
                    })
                    self.stats['messages']['errors'] += 1
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed", extra={
                "component": "alpaca_websocket",
                "action": "connection_closed"
            })
            self.is_connected = False
            self.health_monitor.connection_status = "disconnected"
            self.stats['connection']['disconnects'] += 1
            await self._handle_reconnection()
        except Exception as e:
            logger.error(f"Message handler error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "message_handler",
                "error": str(e)
            })
            self.is_connected = False
            self.health_monitor.connection_status = "error"
            await self._handle_reconnection()
    
    async def _process_message(self, data: Dict):
        """Process individual message"""
        try:
            message_type = data.get('T')  # Message type
            
            if message_type == 'trade_update':
                await self._handle_trade_update(data)
            elif message_type == 'account_update':
                await self._handle_account_update(data)
            elif message_type == 'listening':
                logger.debug(f"Listening confirmation: {data.get('data', {}).get('streams', [])}", extra={
                    "component": "alpaca_websocket",
                    "streams": data.get('data', {}).get('streams', [])
                })
            elif message_type == 'success':
                logger.debug("Success message received", extra={
                    "component": "alpaca_websocket",
                    "message_type": "success"
                })
            elif message_type == 'error':
                Exception(f"Error message: {data.get('msg', 'Unknown error')}")
                logger.error(f"Error message: {data.get('msg', 'Unknown error')}", extra={
                    "component": "alpaca_websocket",
                    "message_data": data,
                    "error": "websocket_error_message"
                })
                self.stats['messages']['errors'] += 1
            else:
                logger.debug(f"Unknown message type: {message_type}", extra={
                    "component": "alpaca_websocket",
                    "message_type": message_type
                })
                
        except Exception as e:
            logger.error(f"Process message error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "process_message",
                "data": data,
                "error": str(e)
            })
    
    async def _handle_trade_update(self, data: Dict):
        """Handle trade update messages"""
        try:
            self.stats['messages']['trade_updates'] += 1
            
            trade_update = TradeUpdate(
                event=data.get('event', ''),
                order_id=data.get('order', {}).get('id', ''),
                symbol=data.get('order', {}).get('symbol', ''),
                side=data.get('order', {}).get('side', ''),
                qty=float(data.get('order', {}).get('qty', 0)),
                filled_qty=float(data.get('order', {}).get('filled_qty', 0)),
                price=float(data.get('price', 0)) if data.get('price') else None,
                timestamp=data.get('timestamp', ''),
                position_qty=float(data.get('position_qty', 0)) if data.get('position_qty') else None
            )
            
            logger.info(f"Trade update: {trade_update.event} - {trade_update.symbol} {trade_update.side} {trade_update.qty}", extra={
                "component": "alpaca_websocket",
                "action": "trade_update",
                "event": trade_update.event,
                "symbol": trade_update.symbol,
                "side": trade_update.side,
                "qty": trade_update.qty,
                "filled_qty": trade_update.filled_qty,
                "price": trade_update.price
            })
            
            # Call user callback if provided
            if self.trade_callback:
                try:
                    await self.trade_callback(trade_update)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}", extra={
                        "component": "alpaca_websocket",
                        "action": "trade_callback",
                        "order_id": trade_update.order_id,
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Trade update handling error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "handle_trade_update",
                "data": data,
                "error": str(e)
            })
    
    async def _handle_account_update(self, data: Dict):
        """Handle account update messages"""
        try:
            self.stats['messages']['account_updates'] += 1
            
            account_update = AccountUpdate(
                event=data.get('event', ''),
                buying_power=float(data.get('buying_power', 0)),
                cash=float(data.get('cash', 0)),
                portfolio_value=float(data.get('portfolio_value', 0)),
                timestamp=data.get('timestamp', '')
            )
            
            logger.debug(f"Account update: Portfolio ${account_update.portfolio_value:.2f}, Cash ${account_update.cash:.2f}", extra={
                "component": "alpaca_websocket",
                "portfolio_value": account_update.portfolio_value,
                "cash": account_update.cash
            })
            
            # Call user callback if provided
            if self.account_callback:
                try:
                    await self.account_callback(account_update)
                except Exception as e:
                    logger.error(f"Account callback error: {e}", extra={
                        "component": "alpaca_websocket",
                        "action": "account_callback",
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Account update handling error: {e}", extra={
                "component": "alpaca_websocket",
                "action": "handle_account_update",
                "data": data,
                "error": str(e)
            })
    
    async def _heartbeat_handler(self):
        """Handle heartbeat to maintain connection health"""
        while self.is_connected:
            try:
                # Send ping to keep connection alive
                if self.websocket and not self.websocket.closed:
                    await self.websocket.ping()
                    self.health_monitor.update_heartbeat()
                    
                await asyncio.sleep(self.health_monitor.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}", extra={
                    "component": "alpaca_websocket",
                    "action": "heartbeat",
                    "error": str(e)
                })
                break
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            Exception(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.", extra={
                "component": "alpaca_websocket",
                "max_attempts": self.max_reconnect_attempts,
                "error": "max_reconnect_attempts_reached"
            })
            return
        
        self.reconnect_attempts += 1
        self.stats['connection']['reconnects'] += 1
        
        # Calculate backoff time
        wait_time = min(
            self.reconnect_backoff_base * (2 ** self.reconnect_attempts),
            self.reconnect_backoff_max
        )
        
        logger.warning(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time:.1f}s", extra={
            "component": "alpaca_websocket",
            "reconnect_attempts": self.reconnect_attempts,
            "max_attempts": self.max_reconnect_attempts,
            "wait_time": wait_time
        })
        await asyncio.sleep(wait_time)
        await self.connect()
    
    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.is_connected = False
            self.health_monitor.connection_status = "disconnected"
            self.stats['connection']['disconnects'] += 1
            logger.info("WebSocket disconnected", extra={
                "component": "alpaca_websocket",
                "action": "disconnected",
                "reason": "manual"
            })
    
    def get_health_status(self) -> Dict:
        """Get connection health status"""
        return self.health_monitor.get_status()
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        health_status = self.get_health_status()
        return {
            "health": health_status,
            "messages": self.stats['messages'],
            "connection": self.stats['connection'],
            "environment": self.environment,
            "is_connected": self.is_connected,
            "websocket_url": self.websocket_url
        }


# Example usage and callbacks
async def example_trade_callback(trade_update: TradeUpdate):
    """Example trade update callback"""
    logger.info(f"TRADE CALLBACK: {trade_update.event} - {trade_update.symbol} {trade_update.side} {trade_update.qty}", extra={
        "component": "alpaca_websocket",
        "event": trade_update.event,
        "symbol": trade_update.symbol,
        "side": trade_update.side,
        "qty": trade_update.qty
    })
    
    if trade_update.event == 'fill':
        logger.info(f"Order filled: {trade_update.symbol} at ${trade_update.price}", extra={
            "component": "alpaca_websocket",
            "symbol": trade_update.symbol,
            "price": trade_update.price,
            "event": "fill"
        })
    elif trade_update.event == 'partial_fill':
        logger.info(f"Partial fill: {trade_update.filled_qty}/{trade_update.qty} shares", extra={
            "component": "alpaca_websocket",
            "filled_qty": trade_update.filled_qty,
            "total_qty": trade_update.qty,
            "event": "partial_fill"
        })
    elif trade_update.event == 'canceled':
        logger.warning(f"Order canceled: {trade_update.order_id}", extra={
            "component": "alpaca_websocket",
            "order_id": trade_update.order_id,
            "event": "canceled"
        })


async def example_account_callback(account_update: AccountUpdate):
    """Example account update callback"""
    logger.info(f"ACCOUNT CALLBACK: Portfolio ${account_update.portfolio_value:.2f}, Cash ${account_update.cash:.2f}", extra={
        "component": "alpaca_websocket",
        "portfolio_value": account_update.portfolio_value,
        "cash": account_update.cash
    })


async def main():
    """Example usage of Alpaca WebSocket client"""
    
    try:
        # Create client with callbacks
        client = AlpacaWebSocketClient(
            trade_callback=example_trade_callback,
            account_callback=example_account_callback
        )
        
        # Connect
        await client.connect()
        
        # Run for a while to receive updates
        logger.info("Listening for trading updates... (Press Ctrl+C to stop)", extra={
            "component": "alpaca_websocket",
            "action": "listening_for_updates"
        })
        await asyncio.sleep(300)  # Run for 5 minutes
        
    except KeyboardInterrupt:
        logger.info("Shutting down...", extra={
            "component": "alpaca_websocket",
            "action": "shutdown"
        })
    except Exception as e:
        logger.error(f"Main execution error: {e}", extra={
            "component": "alpaca_websocket",
            "error": str(e)
        })
    finally:
        if 'client' in locals():
            await client.disconnect()
            
            # Print final stats
            stats = client.get_stats()
            logger.info("WebSocket client performance stats", extra={
                "component": "alpaca_websocket",
                "action": "performance_stats",
                "stats": stats
            })


if __name__ == "__main__":
    asyncio.run(main())