"""
TimescaleDB Database Client for Trading System
High-performance time-series database for financial data persistence
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import orjson
from contextlib import asynccontextmanager

from settings import config

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "trading_user"
    password: str = "secure_password"
    pool_min_size: int = 10
    pool_max_size: int = 50
    command_timeout: int = 30
    
    def get_dsn(self) -> str:
        """Get database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class TimescaleDBClient:
    """High-performance TimescaleDB client for trading data"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.config = db_config
        self.pool: Optional[asyncpg.Pool] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize database connection pool and schema"""
        try:
            logger.info("Initializing TimescaleDB connection...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                dsn=self.config.get_dsn(),
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'jit': 'off',  # Disable JIT for consistent performance
                    'application_name': 'trading_system'
                }
            )
            
            # Initialize schema
            await self._initialize_schema()
            
            self.initialized = True
            logger.info("TimescaleDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    async def _initialize_schema(self):
        """Initialize database schema and hypertables"""
        
        # Check if tables already exist to avoid conflicts
        async with self.pool.acquire() as conn:
            existing_tables = await conn.fetch("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """)
            existing_table_names = {row['tablename'] for row in existing_tables}
        
        # Skip schema creation if key tables already exist
        if 'model_versions' in existing_table_names and 'lag_llama_forecasts' in existing_table_names:
            logger.info("Database schema already exists - skipping initialization")
            return
        
        schema_sql = """
        -- Enable TimescaleDB extension
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        
        -- Real-time bars (second-level aggregates)
        CREATE TABLE IF NOT EXISTS realtime_bars (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open NUMERIC(12,4) NOT NULL,
            high NUMERIC(12,4) NOT NULL,
            low NUMERIC(12,4) NOT NULL,
            close NUMERIC(12,4) NOT NULL,
            volume BIGINT NOT NULL,
            vwap NUMERIC(12,4),
            trade_count INTEGER,
            PRIMARY KEY (time, symbol)
        );
        
        -- Create hypertable for realtime_bars
        SELECT create_hypertable('realtime_bars', 'time', 
                                chunk_time_interval => INTERVAL '1 hour',
                                if_not_exists => TRUE);
        
        -- Daily bars (end-of-day summaries)
        CREATE TABLE IF NOT EXISTS daily_bars (
            date DATE NOT NULL,
            symbol TEXT NOT NULL,
            open NUMERIC(12,4) NOT NULL,
            high NUMERIC(12,4) NOT NULL,
            low NUMERIC(12,4) NOT NULL,
            close NUMERIC(12,4) NOT NULL,
            volume BIGINT NOT NULL,
            vwap NUMERIC(12,4),
            prev_close NUMERIC(12,4),
            gap_percent NUMERIC(8,4),
            volume_ratio NUMERIC(8,4),
            PRIMARY KEY (date, symbol)
        );
        
        -- Trade executions
        CREATE TABLE IF NOT EXISTS trade_executions (
            time TIMESTAMPTZ NOT NULL,
            trade_id UUID PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
            quantity INTEGER NOT NULL,
            price NUMERIC(12,4) NOT NULL,
            strategy TEXT NOT NULL,
            signal_confidence NUMERIC(4,3),
            execution_latency_ms INTEGER,
            pnl NUMERIC(12,4),
            commission NUMERIC(8,4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create hypertable for trade_executions
        SELECT create_hypertable('trade_executions', 'time',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE);
        
        -- Strategy performance metrics
        CREATE TABLE IF NOT EXISTS strategy_performance (
            date DATE NOT NULL,
            strategy TEXT NOT NULL,
            trades_count INTEGER NOT NULL DEFAULT 0,
            winning_trades INTEGER NOT NULL DEFAULT 0,
            losing_trades INTEGER NOT NULL DEFAULT 0,
            total_pnl NUMERIC(12,4) NOT NULL DEFAULT 0,
            gross_profit NUMERIC(12,4) NOT NULL DEFAULT 0,
            gross_loss NUMERIC(12,4) NOT NULL DEFAULT 0,
            max_win NUMERIC(12,4) NOT NULL DEFAULT 0,
            max_loss NUMERIC(12,4) NOT NULL DEFAULT 0,
            avg_win NUMERIC(12,4) NOT NULL DEFAULT 0,
            avg_loss NUMERIC(12,4) NOT NULL DEFAULT 0,
            win_rate NUMERIC(5,4) NOT NULL DEFAULT 0,
            profit_factor NUMERIC(8,4) NOT NULL DEFAULT 0,
            sharpe_ratio NUMERIC(8,4),
            max_drawdown NUMERIC(8,4),
            PRIMARY KEY (date, strategy)
        );
        
        -- Market regime data
        CREATE TABLE IF NOT EXISTS market_regime (
            time TIMESTAMPTZ NOT NULL,
            regime_type TEXT NOT NULL,
            spy_price NUMERIC(12,4),
            spy_macd_line NUMERIC(8,4),
            spy_macd_signal NUMERIC(8,4),
            spy_macd_histogram NUMERIC(8,4),
            vix_level NUMERIC(8,4),
            vix_change NUMERIC(8,4),
            qqq_price NUMERIC(12,4),
            confidence NUMERIC(4,3),
            bullish_indices INTEGER,
            total_indices INTEGER,
            PRIMARY KEY (time)
        );
        
        -- Create hypertable for market_regime
        SELECT create_hypertable('market_regime', 'time',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE);
        
        -- Professional indicators cache
        CREATE TABLE IF NOT EXISTS indicators_cache (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            timespan TEXT NOT NULL DEFAULT 'minute',
            rsi_14 NUMERIC(6,3),
            macd_line NUMERIC(8,4),
            macd_signal NUMERIC(8,4),
            macd_histogram NUMERIC(8,4),
            sma_20 NUMERIC(12,4),
            sma_50 NUMERIC(12,4),
            sma_200 NUMERIC(12,4),
            ema_12 NUMERIC(12,4),
            ema_26 NUMERIC(12,4),
            ema_50 NUMERIC(12,4),
            bollinger_upper NUMERIC(12,4),
            bollinger_lower NUMERIC(12,4),
            bollinger_middle NUMERIC(12,4),
            PRIMARY KEY (time, symbol, timespan)
        );
        
        -- Create hypertable for indicators_cache
        SELECT create_hypertable('indicators_cache', 'time',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE);
        
        -- Gap candidates tracking
        CREATE TABLE IF NOT EXISTS gap_candidates (
            date DATE NOT NULL,
            symbol TEXT NOT NULL,
            gap_percent NUMERIC(8,4) NOT NULL,
            volume_ratio NUMERIC(8,4) NOT NULL,
            prev_close NUMERIC(12,4) NOT NULL,
            open_price NUMERIC(12,4),
            current_price NUMERIC(12,4),
            direction TEXT NOT NULL CHECK (direction IN ('up', 'down')),
            volume BIGINT,
            market_cap BIGINT,
            sector TEXT,
            detected_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (date, symbol)
        );
        
        -- System performance metrics
        CREATE TABLE IF NOT EXISTS system_metrics (
            time TIMESTAMPTZ NOT NULL,
            metric_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value NUMERIC(12,4) NOT NULL,
            metadata JSONB,
            PRIMARY KEY (time, metric_type, metric_name)
        );
        
        -- Create hypertable for system_metrics
        SELECT create_hypertable('system_metrics', 'time',
                                chunk_time_interval => INTERVAL '1 hour',
                                if_not_exists => TRUE);
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_realtime_bars_symbol_time 
            ON realtime_bars (symbol, time DESC);
        CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol_date 
            ON daily_bars (symbol, date DESC);
        CREATE INDEX IF NOT EXISTS idx_trade_executions_symbol_time 
            ON trade_executions (symbol, time DESC);
        CREATE INDEX IF NOT EXISTS idx_trade_executions_strategy 
            ON trade_executions (strategy, time DESC);
        CREATE INDEX IF NOT EXISTS idx_indicators_cache_symbol_time 
            ON indicators_cache (symbol, time DESC);
        CREATE INDEX IF NOT EXISTS idx_gap_candidates_date_gap 
            ON gap_candidates (date, ABS(gap_percent) DESC);
        
        -- Compression policies (compress data older than 7 days)
        SELECT add_compression_policy('realtime_bars', INTERVAL '7 days', if_not_exists => TRUE);
        SELECT add_compression_policy('trade_executions', INTERVAL '7 days', if_not_exists => TRUE);
        SELECT add_compression_policy('market_regime', INTERVAL '7 days', if_not_exists => TRUE);
        SELECT add_compression_policy('indicators_cache', INTERVAL '7 days', if_not_exists => TRUE);
        SELECT add_compression_policy('system_metrics', INTERVAL '7 days', if_not_exists => TRUE);
        
        -- Retention policies (keep data for 2 years)
        SELECT add_retention_policy('realtime_bars', INTERVAL '2 years', if_not_exists => TRUE);
        SELECT add_retention_policy('system_metrics', INTERVAL '1 year', if_not_exists => TRUE);
        
        -- Model versions table (regular table, not hypertable)
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            model_path TEXT NOT NULL,
            training_config JSONB,
            performance_metrics JSONB,
            training_data_hash TEXT,
            validation_metrics JSONB,
            is_active BOOLEAN DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            created_by TEXT DEFAULT 'system',
            UNIQUE(model_name, version)
        );
        
        -- Lag-Llama forecasts table (hypertable with timestamp in primary key)
        CREATE TABLE IF NOT EXISTS lag_llama_forecasts (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            forecast_horizon INTEGER NOT NULL,
            confidence_score NUMERIC(4,3),
            mean_forecast NUMERIC(12,4),
            median_forecast NUMERIC(12,4),
            quantile_10 NUMERIC(12,4),
            quantile_25 NUMERIC(12,4),
            quantile_75 NUMERIC(12,4),
            quantile_90 NUMERIC(12,4),
            volatility_forecast NUMERIC(8,4),
            trend_direction TEXT,
            optimal_hold_minutes INTEGER,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, symbol, forecast_horizon)
        );
        
        -- Create hypertable for lag_llama_forecasts
        SELECT create_hypertable('lag_llama_forecasts', 'timestamp',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE);
        
        -- Multi-timeframe forecasts table (hypertable with timestamp in primary key)
        CREATE TABLE IF NOT EXISTS multi_timeframe_forecasts (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            model_version_id INTEGER REFERENCES model_versions(version_id),
            horizon_5min JSONB,
            horizon_15min JSONB,
            horizon_30min JSONB,
            horizon_60min JSONB,
            horizon_120min JSONB,
            confidence_scores JSONB,
            volatility_forecasts JSONB,
            trend_directions JSONB,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, symbol, model_version_id)
        );
        
        -- Create hypertable for multi_timeframe_forecasts
        SELECT create_hypertable('multi_timeframe_forecasts', 'timestamp',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE);
        
        -- Additional indexes for model versioning
        CREATE INDEX IF NOT EXISTS idx_model_versions_name_active
            ON model_versions (model_name, is_active, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_lag_llama_forecasts_symbol_time
            ON lag_llama_forecasts (symbol, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_multi_timeframe_forecasts_symbol_time
            ON multi_timeframe_forecasts (symbol, timestamp DESC);
        
        -- Compression policies for new tables
        SELECT add_compression_policy('lag_llama_forecasts', INTERVAL '7 days', if_not_exists => TRUE);
        SELECT add_compression_policy('multi_timeframe_forecasts', INTERVAL '7 days', if_not_exists => TRUE);
        
        -- Retention policies for new tables
        SELECT add_retention_policy('lag_llama_forecasts', INTERVAL '1 year', if_not_exists => TRUE);
        SELECT add_retention_policy('multi_timeframe_forecasts', INTERVAL '1 year', if_not_exists => TRUE);
        """
        
        async with self.pool.acquire() as conn:
            # Execute schema creation in transaction
            async with conn.transaction():
                await conn.execute(schema_sql)
        
        logger.info("Database schema initialized successfully")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            yield conn
    
    # ========== REALTIME BARS ==========
    
    async def insert_realtime_bars(self, bars: List[Dict[str, Any]]):
        """Insert real-time bars (second-level aggregates)"""
        if not bars:
            return
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany("""
                    INSERT INTO realtime_bars 
                    (time, symbol, open, high, low, close, volume, vwap, trade_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        trade_count = EXCLUDED.trade_count
                """, [
                    (
                        bar['timestamp'], bar['symbol'], bar['open'], bar['high'],
                        bar['low'], bar['close'], bar['volume'], 
                        bar.get('vwap'), bar.get('trade_count')
                    ) for bar in bars
                ])
                
        except Exception as e:
            logger.error(f"Error inserting realtime bars: {e}")
            raise
    
    async def get_realtime_bars(self, symbol: str, hours: int = 1) -> List[Dict]:
        """Get recent real-time bars for symbol"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT time, symbol, open, high, low, close, volume, vwap, trade_count
                    FROM realtime_bars
                    WHERE symbol = $1 AND time >= NOW() - INTERVAL '%s hours'
                    ORDER BY time DESC
                    LIMIT 3600
                """ % hours, symbol)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching realtime bars for {symbol}: {e}")
            return []
    
    # ========== DAILY BARS ==========
    
    async def insert_daily_bars(self, bars: List[Dict[str, Any]]):
        """Insert daily bar summaries"""
        if not bars:
            return
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany("""
                    INSERT INTO daily_bars 
                    (date, symbol, open, high, low, close, volume, vwap, 
                     prev_close, gap_percent, volume_ratio)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        prev_close = EXCLUDED.prev_close,
                        gap_percent = EXCLUDED.gap_percent,
                        volume_ratio = EXCLUDED.volume_ratio
                """, [
                    (
                        bar['date'], bar['symbol'], bar['open'], bar['high'],
                        bar['low'], bar['close'], bar['volume'], bar.get('vwap'),
                        bar.get('prev_close'), bar.get('gap_percent'), 
                        bar.get('volume_ratio')
                    ) for bar in bars
                ])
                
        except Exception as e:
            logger.error(f"Error inserting daily bars: {e}")
            raise
    
    async def get_daily_bars(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get daily bars for symbol"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT date, symbol, open, high, low, close, volume, vwap,
                           prev_close, gap_percent, volume_ratio
                    FROM daily_bars
                    WHERE symbol = $1 AND date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY date DESC
                    LIMIT $2
                """ % days, symbol, days)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching daily bars for {symbol}: {e}")
            return []
    
    # ========== TRADE EXECUTIONS ==========
    
    async def insert_trade_execution(self, trade: Dict[str, Any]):
        """Insert trade execution record"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO trade_executions 
                    (time, trade_id, symbol, side, quantity, price, strategy,
                     signal_confidence, execution_latency_ms, pnl, commission)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    trade['timestamp'], trade['trade_id'], trade['symbol'],
                    trade['side'], trade['quantity'], trade['price'], 
                    trade['strategy'], trade.get('signal_confidence'),
                    trade.get('execution_latency_ms'), trade.get('pnl'),
                    trade.get('commission')
                )
                
        except Exception as e:
            logger.error(f"Error inserting trade execution: {e}")
            raise
    
    async def get_trade_executions(self, symbol: Optional[str] = None, 
                                 strategy: Optional[str] = None,
                                 days: int = 7) -> List[Dict]:
        """Get trade execution history"""
        try:
            where_clauses = ["time >= NOW() - INTERVAL '%s days'" % days]
            params = []
            param_count = 0
            
            if symbol:
                param_count += 1
                where_clauses.append(f"symbol = ${param_count}")
                params.append(symbol)
            
            if strategy:
                param_count += 1
                where_clauses.append(f"strategy = ${param_count}")
                params.append(strategy)
            
            where_clause = " AND ".join(where_clauses)
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT time, trade_id, symbol, side, quantity, price, strategy,
                           signal_confidence, execution_latency_ms, pnl, commission
                    FROM trade_executions
                    WHERE {where_clause}
                    ORDER BY time DESC
                    LIMIT 1000
                """, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching trade executions: {e}")
            return []
    
    # ========== STRATEGY PERFORMANCE ==========
    
    async def update_strategy_performance(self, date: date, strategy: str, 
                                        metrics: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO strategy_performance 
                    (date, strategy, trades_count, winning_trades, losing_trades,
                     total_pnl, gross_profit, gross_loss, max_win, max_loss,
                     avg_win, avg_loss, win_rate, profit_factor, sharpe_ratio, max_drawdown)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (date, strategy) DO UPDATE SET
                        trades_count = EXCLUDED.trades_count,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        total_pnl = EXCLUDED.total_pnl,
                        gross_profit = EXCLUDED.gross_profit,
                        gross_loss = EXCLUDED.gross_loss,
                        max_win = EXCLUDED.max_win,
                        max_loss = EXCLUDED.max_loss,
                        avg_win = EXCLUDED.avg_win,
                        avg_loss = EXCLUDED.avg_loss,
                        win_rate = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown
                """,
                    date, strategy, metrics.get('trades_count', 0),
                    metrics.get('winning_trades', 0), metrics.get('losing_trades', 0),
                    metrics.get('total_pnl', 0), metrics.get('gross_profit', 0),
                    metrics.get('gross_loss', 0), metrics.get('max_win', 0),
                    metrics.get('max_loss', 0), metrics.get('avg_win', 0),
                    metrics.get('avg_loss', 0), metrics.get('win_rate', 0),
                    metrics.get('profit_factor', 0), metrics.get('sharpe_ratio'),
                    metrics.get('max_drawdown')
                )
                
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
            raise
    
    # ========== MARKET REGIME ==========
    
    async def insert_market_regime(self, regime_data: Dict[str, Any]):
        """Insert market regime data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO market_regime 
                    (time, regime_type, spy_price, spy_macd_line, spy_macd_signal,
                     spy_macd_histogram, vix_level, vix_change, qqq_price,
                     confidence, bullish_indices, total_indices)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (time) DO UPDATE SET
                        regime_type = EXCLUDED.regime_type,
                        spy_price = EXCLUDED.spy_price,
                        spy_macd_line = EXCLUDED.spy_macd_line,
                        spy_macd_signal = EXCLUDED.spy_macd_signal,
                        spy_macd_histogram = EXCLUDED.spy_macd_histogram,
                        vix_level = EXCLUDED.vix_level,
                        vix_change = EXCLUDED.vix_change,
                        qqq_price = EXCLUDED.qqq_price,
                        confidence = EXCLUDED.confidence,
                        bullish_indices = EXCLUDED.bullish_indices,
                        total_indices = EXCLUDED.total_indices
                """,
                    regime_data['timestamp'], regime_data['regime_type'],
                    regime_data.get('spy_price'), regime_data.get('spy_macd_line'),
                    regime_data.get('spy_macd_signal'), regime_data.get('spy_macd_histogram'),
                    regime_data.get('vix_level'), regime_data.get('vix_change'),
                    regime_data.get('qqq_price'), regime_data.get('confidence'),
                    regime_data.get('bullish_indices'), regime_data.get('total_indices')
                )
                
        except Exception as e:
            logger.error(f"Error inserting market regime: {e}")
            raise
    
    async def get_latest_market_regime(self) -> Optional[Dict]:
        """Get latest market regime"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT time, regime_type, spy_price, spy_macd_line, spy_macd_signal,
                           spy_macd_histogram, vix_level, vix_change, qqq_price,
                           confidence, bullish_indices, total_indices
                    FROM market_regime
                    ORDER BY time DESC
                    LIMIT 1
                """)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching latest market regime: {e}")
            return None
    
    # ========== INDICATORS CACHE ==========
    
    async def insert_indicators(self, indicators: List[Dict[str, Any]]):
        """Insert professional indicators"""
        if not indicators:
            return
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany("""
                    INSERT INTO indicators_cache 
                    (time, symbol, timespan, rsi_14, macd_line, macd_signal, macd_histogram,
                     sma_20, sma_50, sma_200, ema_12, ema_26, ema_50,
                     bollinger_upper, bollinger_lower, bollinger_middle)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (time, symbol, timespan) DO UPDATE SET
                        rsi_14 = EXCLUDED.rsi_14,
                        macd_line = EXCLUDED.macd_line,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram,
                        sma_20 = EXCLUDED.sma_20,
                        sma_50 = EXCLUDED.sma_50,
                        sma_200 = EXCLUDED.sma_200,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        ema_50 = EXCLUDED.ema_50,
                        bollinger_upper = EXCLUDED.bollinger_upper,
                        bollinger_lower = EXCLUDED.bollinger_lower,
                        bollinger_middle = EXCLUDED.bollinger_middle
                """, [
                    (
                        ind['timestamp'], ind['symbol'], ind.get('timespan', 'minute'),
                        ind.get('rsi_14'), ind.get('macd_line'), ind.get('macd_signal'),
                        ind.get('macd_histogram'), ind.get('sma_20'), ind.get('sma_50'),
                        ind.get('sma_200'), ind.get('ema_12'), ind.get('ema_26'),
                        ind.get('ema_50'), ind.get('bollinger_upper'), 
                        ind.get('bollinger_lower'), ind.get('bollinger_middle')
                    ) for ind in indicators
                ])
                
        except Exception as e:
            logger.error(f"Error inserting indicators: {e}")
            raise
    
    async def get_latest_indicators(self, symbol: str, 
                                  timespan: str = 'minute') -> Optional[Dict]:
        """Get latest indicators for symbol"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT time, symbol, timespan, rsi_14, macd_line, macd_signal, macd_histogram,
                           sma_20, sma_50, sma_200, ema_12, ema_26, ema_50,
                           bollinger_upper, bollinger_lower, bollinger_middle
                    FROM indicators_cache
                    WHERE symbol = $1 AND timespan = $2
                    ORDER BY time DESC
                    LIMIT 1
                """, symbol, timespan)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching indicators for {symbol}: {e}")
            return None
    
    # ========== GAP CANDIDATES ==========
    
    async def insert_gap_candidate(self, symbol: str, gap_type: str, gap_percent: float,
                                 previous_close: float, current_price: float, volume: int,
                                 volume_ratio: float, date: Optional[date] = None):
        """Insert single gap candidate"""
        if date is None:
            date = datetime.now().date()
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO gap_candidates 
                    (date, symbol, gap_percent, volume_ratio, prev_close, current_price,
                     direction, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        gap_percent = EXCLUDED.gap_percent,
                        volume_ratio = EXCLUDED.volume_ratio,
                        prev_close = EXCLUDED.prev_close,
                        current_price = EXCLUDED.current_price,
                        direction = EXCLUDED.direction,
                        volume = EXCLUDED.volume,
                        detected_at = NOW()
                """, date, symbol, gap_percent, volume_ratio, previous_close,
                     current_price, 'up' if gap_percent > 0 else 'down', volume)
                
        except Exception as e:
            logger.error(f"Error inserting gap candidate: {e}")
            raise
    
    async def insert_gap_candidates(self, candidates: List[Dict[str, Any]]):
        """Insert gap candidates"""
        if not candidates:
            return
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany("""
                    INSERT INTO gap_candidates 
                    (date, symbol, gap_percent, volume_ratio, prev_close, open_price,
                     current_price, direction, volume, market_cap, sector)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        gap_percent = EXCLUDED.gap_percent,
                        volume_ratio = EXCLUDED.volume_ratio,
                        prev_close = EXCLUDED.prev_close,
                        open_price = EXCLUDED.open_price,
                        current_price = EXCLUDED.current_price,
                        direction = EXCLUDED.direction,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap,
                        sector = EXCLUDED.sector,
                        detected_at = NOW()
                """, [
                    (
                        cand['date'], cand['symbol'], cand['gap_percent'],
                        cand['volume_ratio'], cand['prev_close'], cand.get('open_price'),
                        cand['current_price'], cand['direction'], cand.get('volume'),
                        cand.get('market_cap'), cand.get('sector')
                    ) for cand in candidates
                ])
                
        except Exception as e:
            logger.error(f"Error inserting gap candidates: {e}")
            raise
    
    async def get_gap_candidates(self, date: Optional[date] = None, 
                               min_gap_percent: float = 2.0) -> List[Dict]:
        """Get gap candidates for date"""
        if date is None:
            date = datetime.now().date()
        
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT date, symbol, gap_percent, volume_ratio, prev_close,
                           open_price, current_price, direction, volume, market_cap, sector
                    FROM gap_candidates
                    WHERE date = $1 AND ABS(gap_percent) >= $2
                    ORDER BY ABS(gap_percent) DESC
                    LIMIT 100
                """, date, min_gap_percent)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching gap candidates: {e}")
            return []
    
    # ========== SYSTEM METRICS ==========
    
    async def insert_system_metrics(self, metrics: List[Dict[str, Any]]):
        """Insert system performance metrics"""
        if not metrics:
            return
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany("""
                    INSERT INTO system_metrics (time, metric_type, metric_name, metric_value, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (time, metric_type, metric_name) DO UPDATE SET
                        metric_value = EXCLUDED.metric_value,
                        metadata = EXCLUDED.metadata
                """, [
                    (
                        metric['timestamp'], metric['metric_type'], metric['metric_name'],
                        metric['metric_value'], orjson.dumps(metric.get('metadata', {})).decode()
                    ) for metric in metrics
                ])
                
        except Exception as e:
            logger.error(f"Error inserting system metrics: {e}")
            raise
    
    # ========== UTILITY METHODS ==========
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.get_connection() as conn:
                # Get table sizes
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)
                
                # Get connection stats
                connection_stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                return {
                    'table_stats': [dict(row) for row in table_stats],
                    'connection_stats': dict(connection_stats) if connection_stats else {},
                    'pool_stats': {
                        'size': self.pool.get_size() if self.pool else 0,
                        'min_size': self.config.pool_min_size,
                        'max_size': self.config.pool_max_size
                    }
                }
                
        except Exception as e:
            logger.error(f"Error fetching database stats: {e}")
            return {}
    
    # ========== EARNINGS CALENDAR ==========
    
    async def insert_earnings_calendar(self, symbol: str, earnings_date: date,
                                     announcement_time: str, title: str = "",
                                     description: str = ""):
        """Insert earnings calendar entry"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO earnings_calendar 
                    (symbol, earnings_date, announcement_time, title, description)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (symbol, earnings_date) DO UPDATE SET
                        announcement_time = EXCLUDED.announcement_time,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """, symbol, earnings_date, announcement_time, title, description)
                
        except Exception as e:
            logger.error(f"Error inserting earnings calendar: {e}")
            raise
    
    async def get_earnings_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming earnings"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, earnings_date, announcement_time, title, description
                    FROM earnings_calendar
                    WHERE earnings_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '%s days'
                    ORDER BY earnings_date, symbol
                """ % days_ahead)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    # ========== SYMBOLS MANAGEMENT ==========
    
    async def update_symbol_market_cap(self, symbol: str, market_cap: Optional[float],
                                     sector: Optional[str] = None):
        """Update symbol market cap and sector"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO symbols (symbol, market_cap, sector, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        market_cap = EXCLUDED.market_cap,
                        sector = EXCLUDED.sector,
                        updated_at = NOW()
                """, symbol, market_cap, sector)
                
        except Exception as e:
            logger.error(f"Error updating symbol market cap: {e}")
            raise
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT symbol, name, exchange, sector, industry, market_cap,
                           shares_outstanding, is_active
                    FROM symbols
                    WHERE symbol = $1
                """, symbol)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None
    
    # ========== TRADING SIGNALS ==========
    
    async def insert_trading_signal(self, signal: Dict[str, Any]):
        """Insert trading signal"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO trading_signals 
                    (time, symbol, strategy, signal_type, confidence, price, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                    signal['timestamp'], signal['symbol'], signal['strategy'],
                    signal['signal_type'], signal.get('confidence'),
                    signal.get('price'), orjson.dumps(signal.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting trading signal: {e}")
            raise
    
    async def get_trading_signals(self, symbol: Optional[str] = None,
                                strategy: Optional[str] = None,
                                hours: int = 24) -> List[Dict]:
        """Get recent trading signals"""
        try:
            where_clauses = ["time >= NOW() - INTERVAL '%s hours'" % hours]
            params = []
            param_count = 0
            
            if symbol:
                param_count += 1
                where_clauses.append(f"symbol = ${param_count}")
                params.append(symbol)
            
            if strategy:
                param_count += 1
                where_clauses.append(f"strategy = ${param_count}")
                params.append(strategy)
            
            where_clause = " AND ".join(where_clauses)
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT time, symbol, strategy, signal_type, confidence, price, metadata
                    FROM trading_signals
                    WHERE {where_clause}
                    ORDER BY time DESC
                    LIMIT 1000
                """, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching trading signals: {e}")
            return []
    
    # ========== POSITIONS ==========
    
    async def insert_position(self, position: Dict[str, Any]):
        """Insert or update position"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO positions 
                    (symbol, strategy, side, quantity, entry_price, current_price,
                     unrealized_pnl, realized_pnl, status, opened_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    position['symbol'], position['strategy'], position['side'],
                    position['quantity'], position['entry_price'], position.get('current_price'),
                    position.get('unrealized_pnl', 0), position.get('realized_pnl', 0),
                    position.get('status', 'OPEN'), position['opened_at'],
                    orjson.dumps(position.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting position: {e}")
            raise
    
    async def update_position(self, position_id: int, **kwargs):
        """Update position"""
        try:
            set_clauses = []
            params = []
            param_count = 0
            
            for key, value in kwargs.items():
                if key in ['current_price', 'unrealized_pnl', 'realized_pnl', 'status', 'closed_at']:
                    param_count += 1
                    set_clauses.append(f"{key} = ${param_count}")
                    params.append(value)
            
            if not set_clauses:
                return
            
            param_count += 1
            params.append(position_id)
            
            set_clause = ", ".join(set_clauses)
            
            async with self.get_connection() as conn:
                await conn.execute(f"""
                    UPDATE positions 
                    SET {set_clause}, updated_at = NOW()
                    WHERE id = ${param_count}
                """, *params)
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            raise
    
    async def get_open_positions(self, strategy: Optional[str] = None) -> List[Dict]:
        """Get open positions"""
        try:
            where_clauses = ["status = 'OPEN'"]
            params = []
            param_count = 0
            
            if strategy:
                param_count += 1
                where_clauses.append(f"strategy = ${param_count}")
                params.append(strategy)
            
            where_clause = " AND ".join(where_clauses)
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT id, symbol, strategy, side, quantity, entry_price, current_price,
                           unrealized_pnl, realized_pnl, status, opened_at, metadata
                    FROM positions
                    WHERE {where_clause}
                    ORDER BY opened_at DESC
                """, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []
    
    # ========== ORDERS ==========
    
    async def insert_order(self, order: Dict[str, Any]):
        """Insert order"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO orders 
                    (symbol, strategy, order_type, side, quantity, price, stop_price,
                     filled_quantity, avg_fill_price, status, broker_order_id, submitted_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, 
                    order['symbol'], order['strategy'], order['order_type'], order['side'],
                    order['quantity'], order.get('price'), order.get('stop_price'),
                    order.get('filled_quantity', 0), order.get('avg_fill_price'),
                    order.get('status', 'PENDING'), order.get('broker_order_id'),
                    order['submitted_at'], orjson.dumps(order.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting order: {e}")
            raise
    
    async def update_order_status(self, order_id: int, status: str, filled_quantity: int = 0,
                                avg_fill_price: Optional[float] = None, filled_at: Optional[datetime] = None):
        """Update order status"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE orders 
                    SET status = $1, filled_quantity = $2, avg_fill_price = $3, 
                        filled_at = $4, updated_at = NOW()
                    WHERE id = $5
                """, status, filled_quantity, avg_fill_price, filled_at, order_id)
                
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            raise
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None,
                       days: int = 7) -> List[Dict]:
        """Get orders"""
        try:
            where_clauses = ["submitted_at >= NOW() - INTERVAL '%s days'" % days]
            params = []
            param_count = 0
            
            if symbol:
                param_count += 1
                where_clauses.append(f"symbol = ${param_count}")
                params.append(symbol)
            
            if status:
                param_count += 1
                where_clauses.append(f"status = ${param_count}")
                params.append(status)
            
            where_clause = " AND ".join(where_clauses)
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT id, symbol, strategy, order_type, side, quantity, price, stop_price,
                           filled_quantity, avg_fill_price, status, broker_order_id, 
                           submitted_at, filled_at, metadata
                    FROM orders
                    WHERE {where_clause}
                    ORDER BY submitted_at DESC
                    LIMIT 1000
                """, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self.initialized = False
        logger.info("Database connections cleaned up")
    
    # ========== MODEL VERSIONING METHODS ==========
    
    async def insert_model_version(self, model_data: Dict[str, Any]) -> int:
        """Insert new model version and return id (using existing schema)"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO model_versions
                    (version_id, model_type, checkpoint_path, training_start, training_end,
                     training_data_period, performance_metrics, validation_loss, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                    model_data.get('version', 'v1.0.0'),
                    model_data.get('model_name', 'lag_llama'),
                    model_data.get('model_path', ''),
                    model_data.get('training_start', datetime.now()),
                    model_data.get('training_end', datetime.now()),
                    model_data.get('training_data_period', 30),
                    orjson.dumps(model_data.get('performance_metrics', {})).decode(),
                    model_data.get('validation_loss', 0.0),
                    model_data.get('is_active', False)
                )
                
                return row['id']
                
        except Exception as e:
            logger.error(f"Error inserting model version: {e}")
            raise
    
    async def get_active_model_version(self, model_type: str) -> Optional[Dict]:
        """Get active model version (using existing schema)"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT id, version_id, model_type, checkpoint_path, training_start,
                           training_end, training_data_period, performance_metrics,
                           validation_loss, is_active, created_at, updated_at
                    FROM model_versions
                    WHERE model_type = $1 AND is_active = true
                    ORDER BY created_at DESC
                    LIMIT 1
                """, model_type)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching active model version: {e}")
            return None
    
    async def get_model_versions(self, model_type: str, limit: int = 10) -> List[Dict]:
        """Get model version history (using existing schema)"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT id, version_id, model_type, checkpoint_path, training_start,
                           training_end, training_data_period, performance_metrics,
                           validation_loss, is_active, created_at, updated_at
                    FROM model_versions
                    WHERE model_type = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, model_type, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching model versions: {e}")
            return []
    
    async def activate_model_version(self, model_id: int):
        """Activate a model version (deactivates others, using existing schema)"""
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Get model type for this version
                    model_row = await conn.fetchrow("""
                        SELECT model_type FROM model_versions WHERE id = $1
                    """, model_id)
                    
                    if not model_row:
                        raise ValueError(f"Model version {model_id} not found")
                    
                    model_type = model_row['model_type']
                    
                    # Deactivate all versions for this model type
                    await conn.execute("""
                        UPDATE model_versions
                        SET is_active = false, updated_at = NOW()
                        WHERE model_type = $1
                    """, model_type)
                    
                    # Activate the specified version
                    await conn.execute("""
                        UPDATE model_versions
                        SET is_active = true, updated_at = NOW()
                        WHERE id = $1
                    """, model_id)
                
        except Exception as e:
            logger.error(f"Error activating model version: {e}")
            raise
    
    async def update_model_performance(self, model_id: int, performance_metrics: Dict[str, Any]):
        """Update model performance metrics (using existing schema)"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE model_versions
                    SET performance_metrics = $1, updated_at = NOW()
                    WHERE id = $2
                """, orjson.dumps(performance_metrics).decode(), model_id)
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            raise
    
    # ========== MULTI-TIMEFRAME FORECASTS ==========
    
    async def insert_multi_timeframe_forecast(self, forecast_data: Dict[str, Any]):
        """Insert multi-timeframe Lag-Llama forecast"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO multi_timeframe_forecasts
                    (timestamp, symbol, model_version_id, horizon_5min, horizon_15min,
                     horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                     volatility_forecasts, trend_directions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (timestamp, symbol, model_version_id) DO UPDATE SET
                        horizon_5min = EXCLUDED.horizon_5min,
                        horizon_15min = EXCLUDED.horizon_15min,
                        horizon_30min = EXCLUDED.horizon_30min,
                        horizon_60min = EXCLUDED.horizon_60min,
                        horizon_120min = EXCLUDED.horizon_120min,
                        confidence_scores = EXCLUDED.confidence_scores,
                        volatility_forecasts = EXCLUDED.volatility_forecasts,
                        trend_directions = EXCLUDED.trend_directions,
                        metadata = EXCLUDED.metadata
                """,
                    forecast_data['timestamp'], forecast_data['symbol'],
                    forecast_data.get('model_version_id'),
                    orjson.dumps(forecast_data.get('horizon_5min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_15min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_30min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_60min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_120min', {})).decode(),
                    orjson.dumps(forecast_data.get('confidence_scores', {})).decode(),
                    orjson.dumps(forecast_data.get('volatility_forecasts', {})).decode(),
                    orjson.dumps(forecast_data.get('trend_directions', {})).decode(),
                    orjson.dumps(forecast_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting multi-timeframe forecast: {e}")
            raise
    
    async def get_latest_multi_timeframe_forecast(self, symbol: str, model_version_id: Optional[int] = None) -> Optional[Dict]:
        """Get latest multi-timeframe forecast for symbol"""
        try:
            async with self.get_connection() as conn:
                if model_version_id:
                    row = await conn.fetchrow("""
                        SELECT timestamp, symbol, model_version_id, horizon_5min, horizon_15min,
                               horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                               volatility_forecasts, trend_directions, metadata
                        FROM multi_timeframe_forecasts
                        WHERE symbol = $1 AND model_version_id = $2
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, symbol, model_version_id)
                else:
                    row = await conn.fetchrow("""
                        SELECT timestamp, symbol, model_version_id, horizon_5min, horizon_15min,
                               horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                               volatility_forecasts, trend_directions, metadata
                        FROM multi_timeframe_forecasts
                        WHERE symbol = $1
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, symbol)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe forecast: {e}")
            return None
    
    async def get_multi_timeframe_forecasts(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent multi-timeframe forecasts"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, symbol, model_version_id, horizon_5min, horizon_15min,
                           horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                           volatility_forecasts, trend_directions, metadata
                    FROM multi_timeframe_forecasts
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """ % hours, symbol)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe forecasts: {e}")
            return []

# Global database client instance
db_client = None

def get_database_client() -> TimescaleDBClient:
    """Get or create the global database client"""
    global db_client
    if db_client is None:
        db_config = DatabaseConfig(
            host=config.database.host if hasattr(config, 'database') else "localhost",
            port=config.database.port if hasattr(config, 'database') else 5432,
            database=config.database.database if hasattr(config, 'database') else "trading_system",
            username=config.database.username if hasattr(config, 'database') else "trading_user",
            password=config.database.password if hasattr(config, 'database') else "secure_password"
        )
        db_client = TimescaleDBClient(db_config)
    return db_client

    # ========== NEW SCHEMA METHODS ==========
    
    async def insert_system_event(self, event_data: Dict[str, Any]):
        """Insert system event"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO system_events (timestamp, event_type, metadata)
                    VALUES ($1, $2, $3)
                """, 
                    event_data['timestamp'], event_data['event_type'],
                    orjson.dumps(event_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting system event: {e}")
            raise
    
    async def insert_trading_session(self, session_data: Dict[str, Any]):
        """Insert trading session data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO trading_sessions (timestamp, session_type, data)
                    VALUES ($1, $2, $3)
                """, 
                    session_data['timestamp'], session_data['session_type'],
                    orjson.dumps(session_data.get('data', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting trading session: {e}")
            raise
    
    async def insert_strategy_performance(self, performance_data: Dict[str, Any]):
        """Insert strategy performance data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO strategy_performance 
                    (timestamp, strategy, trades_today, winning_trades, total_pnl, 
                     win_rate, active_signals, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                    performance_data['timestamp'], performance_data['strategy'],
                    performance_data.get('trades_today', 0), performance_data.get('winning_trades', 0),
                    performance_data.get('total_pnl', 0), performance_data.get('win_rate', 0),
                    performance_data.get('active_signals', 0),
                    orjson.dumps(performance_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting strategy performance: {e}")
            raise
    
    async def insert_orb_range(self, range_data: Dict[str, Any]):
        """Insert ORB range data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO orb_ranges 
                    (symbol, timestamp, high, low, open, close, volume, range_size, 
                     range_percent, quality, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    range_data['symbol'], range_data['timestamp'], range_data['high'],
                    range_data['low'], range_data['open'], range_data['close'],
                    range_data.get('volume'), range_data.get('range_size'),
                    range_data.get('range_percent'), range_data.get('quality'),
                    range_data.get('vwap')
                )
                
        except Exception as e:
            logger.error(f"Error inserting ORB range: {e}")
            raise
    
    async def insert_breakout_analysis(self, analysis_data: Dict[str, Any]):
        """Insert breakout analysis data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO breakout_analysis 
                    (symbol, timestamp, breakout_direction, breakout_price, current_price,
                     sustainability_probability, false_breakout_probability, volume_confirmation,
                     volume_ratio, target_projections, resistance_levels, support_levels)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                    analysis_data['symbol'], analysis_data['timestamp'], 
                    analysis_data['breakout_direction'], analysis_data.get('breakout_price'),
                    analysis_data.get('current_price'), analysis_data.get('sustainability_probability'),
                    analysis_data.get('false_breakout_probability'), analysis_data.get('volume_confirmation'),
                    analysis_data.get('volume_ratio'), analysis_data.get('target_projections', []),
                    analysis_data.get('resistance_levels', []), analysis_data.get('support_levels', [])
                )
                
        except Exception as e:
            logger.error(f"Error inserting breakout analysis: {e}")
            raise
    
    async def insert_mean_reversion_metrics(self, metrics_data: Dict[str, Any]):
        """Insert mean reversion metrics"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO mean_reversion_metrics 
                    (symbol, timestamp, current_price, sma_20, sma_50, sma_200, vwap,
                     deviation_sma_20, deviation_sma_50, deviation_sma_200, deviation_vwap,
                     z_score_20, bb_upper, bb_lower, bb_middle, bb_position, rsi,
                     momentum_5, momentum_10, volume_ratio, volume_trend, 
                     realized_volatility, volatility_percentile)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                """, 
                    metrics_data['symbol'], metrics_data['timestamp'], metrics_data['current_price'],
                    metrics_data.get('sma_20'), metrics_data.get('sma_50'), metrics_data.get('sma_200'),
                    metrics_data.get('vwap'), metrics_data.get('deviation_sma_20'), 
                    metrics_data.get('deviation_sma_50'), metrics_data.get('deviation_sma_200'),
                    metrics_data.get('deviation_vwap'), metrics_data.get('z_score_20'),
                    metrics_data.get('bb_upper'), metrics_data.get('bb_lower'), 
                    metrics_data.get('bb_middle'), metrics_data.get('bb_position'),
                    metrics_data.get('rsi'), metrics_data.get('momentum_5'), 
                    metrics_data.get('momentum_10'), metrics_data.get('volume_ratio'),
                    metrics_data.get('volume_trend'), metrics_data.get('realized_volatility'),
                    metrics_data.get('volatility_percentile')
                )
                
        except Exception as e:
            logger.error(f"Error inserting mean reversion metrics: {e}")
            raise
    
    async def insert_reversion_analysis(self, analysis_data: Dict[str, Any]):
        """Insert reversion analysis data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO reversion_analysis 
                    (symbol, timestamp, reversion_type, mean_type, reversion_probability,
                     mean_reversion_timeframe, oversold_bounce_prob, overbought_fade_prob,
                     entry_price, target_mean, target_price, stop_loss_price, confidence_score,
                     max_hold_time, trend_alignment, volatility_regime)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, 
                    analysis_data['symbol'], analysis_data['timestamp'], analysis_data['reversion_type'],
                    analysis_data.get('mean_type'), analysis_data.get('reversion_probability'),
                    analysis_data.get('mean_reversion_timeframe'), analysis_data.get('oversold_bounce_prob'),
                    analysis_data.get('overbought_fade_prob'), analysis_data.get('entry_price'),
                    analysis_data.get('target_mean'), analysis_data.get('target_price'),
                    analysis_data.get('stop_loss_price'), analysis_data.get('confidence_score'),
                    analysis_data.get('max_hold_time'), analysis_data.get('trend_alignment'),
                    analysis_data.get('volatility_regime')
                )
                
        except Exception as e:
            logger.error(f"Error inserting reversion analysis: {e}")
            raise
    
    async def insert_lag_llama_forecast(self, forecast_data: Dict[str, Any]):
        """Insert Lag-Llama forecast data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO lag_llama_forecasts 
                    (symbol, timestamp, forecast_horizon, confidence_score, mean_forecast,
                     median_forecast, quantile_10, quantile_25, quantile_75, quantile_90,
                     volatility_forecast, trend_direction, optimal_hold_minutes, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, 
                    forecast_data['symbol'], forecast_data['timestamp'], 
                    forecast_data.get('forecast_horizon'), forecast_data.get('confidence_score'),
                    forecast_data.get('mean_forecast'), forecast_data.get('median_forecast'),
                    forecast_data.get('quantile_10'), forecast_data.get('quantile_25'),
                    forecast_data.get('quantile_75'), forecast_data.get('quantile_90'),
                    forecast_data.get('volatility_forecast'), forecast_data.get('trend_direction'),
                    forecast_data.get('optimal_hold_minutes'),
                    orjson.dumps(forecast_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting Lag-Llama forecast: {e}")
            raise
    
    async def insert_polygon_indicator(self, indicator_data: Dict[str, Any]):
        """Insert Polygon indicator data"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO polygon_indicators 
                    (symbol, timestamp, indicator_type, timespan, window_size, value, additional_values)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, timestamp, indicator_type, timespan, window_size) DO UPDATE SET
                        value = EXCLUDED.value,
                        additional_values = EXCLUDED.additional_values
                """, 
                    indicator_data['symbol'], indicator_data['timestamp'], 
                    indicator_data['indicator_type'], indicator_data.get('timespan', 'minute'),
                    indicator_data.get('window_size'), indicator_data.get('value'),
                    orjson.dumps(indicator_data.get('additional_values', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting Polygon indicator: {e}")
            raise
    
    async def get_polygon_indicator(self, symbol: str, indicator_type: str, 
                                  timespan: str = 'minute', window_size: Optional[int] = None) -> Optional[Dict]:
        """Get latest Polygon indicator"""
        try:
            async with self.get_connection() as conn:
                where_clauses = ["symbol = $1", "indicator_type = $2", "timespan = $3"]
                params = [symbol, indicator_type, timespan]
                
                if window_size is not None:
                    where_clauses.append("window_size = $4")
                    params.append(window_size)
                
                where_clause = " AND ".join(where_clauses)
                
                row = await conn.fetchrow(f"""
                    SELECT symbol, timestamp, indicator_type, timespan, window_size, 
                           value, additional_values
                    FROM polygon_indicators
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, *params)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching Polygon indicator: {e}")
            return None
    
    async def insert_market_regime_analysis(self, regime_data: Dict[str, Any]):
        """Insert market regime analysis"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO market_regime 
                    (timestamp, regime_type, confidence, vix_level, spy_trend, qqq_trend,
                     iwm_trend, sector_rotation, volatility_regime)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        regime_type = EXCLUDED.regime_type,
                        confidence = EXCLUDED.confidence,
                        vix_level = EXCLUDED.vix_level,
                        spy_trend = EXCLUDED.spy_trend,
                        qqq_trend = EXCLUDED.qqq_trend,
                        iwm_trend = EXCLUDED.iwm_trend,
                        sector_rotation = EXCLUDED.sector_rotation,
                        volatility_regime = EXCLUDED.volatility_regime
                """, 
                    regime_data['timestamp'], regime_data['regime_type'], 
                    regime_data.get('confidence'), regime_data.get('vix_level'),
                    regime_data.get('spy_trend'), regime_data.get('qqq_trend'),
                    regime_data.get('iwm_trend'), 
                    orjson.dumps(regime_data.get('sector_rotation', {})).decode(),
                    regime_data.get('volatility_regime')
                )
                
        except Exception as e:
            logger.error(f"Error inserting market regime analysis: {e}")
            raise
    
    async def get_latest_market_regime_analysis(self) -> Optional[Dict]:
        """Get latest market regime analysis"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT timestamp, regime_type, confidence, vix_level, spy_trend, 
                           qqq_trend, iwm_trend, sector_rotation, volatility_regime
                    FROM market_regime
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching latest market regime analysis: {e}")
            return None
    
    # ========== ENHANCED SYSTEM METRICS ==========
    
    async def insert_system_metrics_enhanced(self, metrics_data: Dict[str, Any]):
        """Insert enhanced system metrics"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO system_metrics 
                    (time, uptime_seconds, total_trades, total_pnl, data_messages_processed,
                     forecasts_generated, errors_encountered, memory_usage, cpu_usage, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (time) DO UPDATE SET
                        uptime_seconds = EXCLUDED.uptime_seconds,
                        total_trades = EXCLUDED.total_trades,
                        total_pnl = EXCLUDED.total_pnl,
                        data_messages_processed = EXCLUDED.data_messages_processed,
                        forecasts_generated = EXCLUDED.forecasts_generated,
                        errors_encountered = EXCLUDED.errors_encountered,
                        memory_usage = EXCLUDED.memory_usage,
                        cpu_usage = EXCLUDED.cpu_usage,
                        metadata = EXCLUDED.metadata
                """, 
                    metrics_data['timestamp'], metrics_data.get('uptime_seconds'),
                    metrics_data.get('total_trades'), metrics_data.get('total_pnl'),
                    metrics_data.get('data_messages_processed'), metrics_data.get('forecasts_generated'),
                    metrics_data.get('errors_encountered'), metrics_data.get('memory_usage'),
                    metrics_data.get('cpu_usage'), 
                    orjson.dumps(metrics_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting enhanced system metrics: {e}")
            raise
    
    # ========== MODEL VERSIONING METHODS ==========
    
    async def insert_model_version(self, model_data: Dict[str, Any]) -> int:
        """Insert new model version and return version_id"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO model_versions
                    (model_name, version, model_path, training_config, performance_metrics,
                     training_data_hash, validation_metrics, is_active, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING version_id
                """,
                    model_data['model_name'], model_data['version'], model_data['model_path'],
                    orjson.dumps(model_data.get('training_config', {})).decode(),
                    orjson.dumps(model_data.get('performance_metrics', {})).decode(),
                    model_data.get('training_data_hash'),
                    orjson.dumps(model_data.get('validation_metrics', {})).decode(),
                    model_data.get('is_active', False), model_data.get('created_by', 'system')
                )
                
                return row['version_id']
                
        except Exception as e:
            logger.error(f"Error inserting model version: {e}")
            raise
    
    async def get_active_model_version(self, model_name: str) -> Optional[Dict]:
        """Get active model version"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT version_id, model_name, version, model_path, training_config,
                           performance_metrics, training_data_hash, validation_metrics,
                           is_active, created_at, created_by
                    FROM model_versions
                    WHERE model_name = $1 AND is_active = true
                    ORDER BY created_at DESC
                    LIMIT 1
                """, model_name)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching active model version: {e}")
            return None
    
    async def get_model_versions(self, model_name: str, limit: int = 10) -> List[Dict]:
        """Get model version history"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT version_id, model_name, version, model_path, training_config,
                           performance_metrics, training_data_hash, validation_metrics,
                           is_active, created_at, created_by
                    FROM model_versions
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, model_name, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching model versions: {e}")
            return []
    
    async def activate_model_version(self, version_id: int):
        """Activate a model version (deactivates others)"""
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Get model name for this version
                    model_name_row = await conn.fetchrow("""
                        SELECT model_name FROM model_versions WHERE version_id = $1
                    """, version_id)
                    
                    if not model_name_row:
                        raise ValueError(f"Model version {version_id} not found")
                    
                    model_name = model_name_row['model_name']
                    
                    # Deactivate all versions for this model
                    await conn.execute("""
                        UPDATE model_versions
                        SET is_active = false, updated_at = NOW()
                        WHERE model_name = $1
                    """, model_name)
                    
                    # Activate the specified version
                    await conn.execute("""
                        UPDATE model_versions
                        SET is_active = true, updated_at = NOW()
                        WHERE version_id = $1
                    """, version_id)
                
        except Exception as e:
            logger.error(f"Error activating model version: {e}")
            raise
    
    async def update_model_performance(self, version_id: int, performance_metrics: Dict[str, Any]):
        """Update model performance metrics"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE model_versions
                    SET performance_metrics = $1, updated_at = NOW()
                    WHERE version_id = $2
                """, orjson.dumps(performance_metrics).decode(), version_id)
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            raise
    
    # ========== MULTI-TIMEFRAME FORECASTS ==========
    
    async def insert_multi_timeframe_forecast(self, forecast_data: Dict[str, Any]):
        """Insert multi-timeframe Lag-Llama forecast"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO multi_timeframe_forecasts
                    (symbol, timestamp, model_version_id, horizon_5min, horizon_15min,
                     horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                     volatility_forecasts, trend_directions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (symbol, timestamp, model_version_id) DO UPDATE SET
                        horizon_5min = EXCLUDED.horizon_5min,
                        horizon_15min = EXCLUDED.horizon_15min,
                        horizon_30min = EXCLUDED.horizon_30min,
                        horizon_60min = EXCLUDED.horizon_60min,
                        horizon_120min = EXCLUDED.horizon_120min,
                        confidence_scores = EXCLUDED.confidence_scores,
                        volatility_forecasts = EXCLUDED.volatility_forecasts,
                        trend_directions = EXCLUDED.trend_directions,
                        metadata = EXCLUDED.metadata
                """,
                    forecast_data['symbol'], forecast_data['timestamp'],
                    forecast_data.get('model_version_id'),
                    orjson.dumps(forecast_data.get('horizon_5min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_15min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_30min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_60min', {})).decode(),
                    orjson.dumps(forecast_data.get('horizon_120min', {})).decode(),
                    orjson.dumps(forecast_data.get('confidence_scores', {})).decode(),
                    orjson.dumps(forecast_data.get('volatility_forecasts', {})).decode(),
                    orjson.dumps(forecast_data.get('trend_directions', {})).decode(),
                    orjson.dumps(forecast_data.get('metadata', {})).decode()
                )
                
        except Exception as e:
            logger.error(f"Error inserting multi-timeframe forecast: {e}")
            raise
    
    async def get_latest_multi_timeframe_forecast(self, symbol: str, model_version_id: Optional[int] = None) -> Optional[Dict]:
        """Get latest multi-timeframe forecast for symbol"""
        try:
            async with self.get_connection() as conn:
                if model_version_id:
                    row = await conn.fetchrow("""
                        SELECT symbol, timestamp, model_version_id, horizon_5min, horizon_15min,
                               horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                               volatility_forecasts, trend_directions, metadata
                        FROM multi_timeframe_forecasts
                        WHERE symbol = $1 AND model_version_id = $2
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, symbol, model_version_id)
                else:
                    row = await conn.fetchrow("""
                        SELECT symbol, timestamp, model_version_id, horizon_5min, horizon_15min,
                               horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                               volatility_forecasts, trend_directions, metadata
                        FROM multi_timeframe_forecasts
                        WHERE symbol = $1
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, symbol)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe forecast: {e}")
            return None
    
    async def get_multi_timeframe_forecasts(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent multi-timeframe forecasts"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, timestamp, model_version_id, horizon_5min, horizon_15min,
                           horizon_30min, horizon_60min, horizon_120min, confidence_scores,
                           volatility_forecasts, trend_directions, metadata
                    FROM multi_timeframe_forecasts
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """ % hours, symbol)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe forecasts: {e}")
            return []
    
    # ========== QUERY METHODS ==========
    
    async def get_strategy_performance_summary(self, strategy: str, days: int = 30) -> Optional[Dict]:
        """Get strategy performance summary"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        strategy,
                        COUNT(*) as total_records,
                        AVG(win_rate) as avg_win_rate,
                        SUM(trades_today) as total_trades,
                        SUM(total_pnl) as total_pnl,
                        AVG(active_signals) as avg_active_signals,
                        MAX(timestamp) as last_update
                    FROM strategy_performance
                    WHERE strategy = $1 AND timestamp >= NOW() - INTERVAL '%s days'
                    GROUP BY strategy
                """ % days, strategy)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching strategy performance summary: {e}")
            return None
    
    async def get_recent_orb_ranges(self, symbol: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Get recent ORB ranges"""
        try:
            where_clauses = ["timestamp >= NOW() - INTERVAL '%s hours'" % hours]
            params = []
            param_count = 0
            
            if symbol:
                param_count += 1
                where_clauses.append(f"symbol = ${param_count}")
                params.append(symbol)
            
            where_clause = " AND ".join(where_clauses)
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT symbol, timestamp, high, low, open, close, volume, 
                           range_size, range_percent, quality, vwap
                    FROM orb_ranges
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching recent ORB ranges: {e}")
            return []
    
    async def get_recent_forecasts(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent Lag-Llama forecasts"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, timestamp, forecast_horizon, confidence_score, 
                           mean_forecast, median_forecast, quantile_10, quantile_25,
                           quantile_75, quantile_90, volatility_forecast, trend_direction,
                           optimal_hold_minutes, metadata
                    FROM lag_llama_forecasts
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """ % hours, symbol)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching recent forecasts: {e}")
            return []

# Convenience function for getting database manager
def get_database_manager():
    """Alias for get_database_client for compatibility"""
    return get_database_client()
