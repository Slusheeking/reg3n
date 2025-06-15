-- Trading System Database Schema for TimescaleDB
-- Optimized for high-frequency trading data

-- Enable TimescaleDB extension (already done)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Symbols table for stock metadata
CREATE TABLE IF NOT EXISTS symbols (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    shares_outstanding BIGINT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market data table (OHLCV data) - This will be a hypertable
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    vwap DECIMAL(12,4),
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '1d', etc.
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert market_data to hypertable partitioned by time
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Tick data table for real-time price updates
CREATE TABLE IF NOT EXISTS tick_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    size INTEGER,
    bid DECIMAL(12,4),
    ask DECIMAL(12,4),
    bid_size INTEGER,
    ask_size INTEGER,
    PRIMARY KEY (time, symbol)
);

-- Convert tick_data to hypertable
SELECT create_hypertable('tick_data', 'time', if_not_exists => TRUE);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,4), -- 0.0 to 1.0
    price DECIMAL(12,4),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert trading_signals to hypertable
SELECT create_hypertable('trading_signals', 'time', if_not_exists => TRUE);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'LONG', 'SHORT'
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4),
    realized_pnl DECIMAL(12,4) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'PARTIAL'
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    order_type VARCHAR(20) NOT NULL, -- 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
    side VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4),
    stop_price DECIMAL(12,4),
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(12,4),
    status VARCHAR(20) DEFAULT 'PENDING', -- 'PENDING', 'FILLED', 'PARTIAL', 'CANCELLED'
    broker_order_id VARCHAR(100),
    submitted_at TIMESTAMPTZ NOT NULL,
    filled_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    total_pnl DECIMAL(12,4),
    daily_pnl DECIMAL(12,4),
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),
    metadata JSONB,
    PRIMARY KEY (time, strategy, COALESCE(symbol, ''))
);

-- Convert performance_metrics to hypertable
SELECT create_hypertable('performance_metrics', 'time', if_not_exists => TRUE);

-- Earnings calendar table
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    fiscal_quarter VARCHAR(10),
    fiscal_year INTEGER,
    estimated_eps DECIMAL(8,4),
    actual_eps DECIMAL(8,4),
    surprise_percent DECIMAL(8,4),
    revenue_estimate BIGINT,
    actual_revenue BIGINT,
    announcement_time VARCHAR(20), -- 'BMO', 'AMC', 'DMT'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, earnings_date)
);

-- Gap candidates table for gap trading strategy
CREATE TABLE IF NOT EXISTS gap_candidates (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    scan_date DATE NOT NULL,
    gap_type VARCHAR(20) NOT NULL, -- 'GAP_UP', 'GAP_DOWN'
    gap_percent DECIMAL(8,4) NOT NULL,
    pre_market_high DECIMAL(12,4),
    pre_market_low DECIMAL(12,4),
    pre_market_volume BIGINT,
    previous_close DECIMAL(12,4) NOT NULL,
    market_open DECIMAL(12,4),
    catalyst VARCHAR(255),
    float_shares BIGINT,
    avg_volume_20d BIGINT,
    relative_volume DECIMAL(8,4),
    is_tradeable BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, scan_date)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data (timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time ON tick_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_strategy_time ON trading_signals (strategy, time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_time ON trading_signals (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions (strategy, status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders (strategy, status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, status);
CREATE INDEX IF NOT EXISTS idx_performance_strategy_time ON performance_metrics (strategy, time DESC);
CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date ON earnings_calendar (symbol, earnings_date);
CREATE INDEX IF NOT EXISTS idx_gap_candidates_date ON gap_candidates (scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_gap_candidates_symbol ON gap_candidates (symbol, scan_date DESC);

-- Create compression policies for older data (compress data older than 7 days)
SELECT add_compression_policy('market_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('tick_data', INTERVAL '3 days', if_not_exists => TRUE);
SELECT add_compression_policy('trading_signals', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('performance_metrics', INTERVAL '90 days', if_not_exists => TRUE);

-- Create retention policies (keep data for specified periods)
-- Keep tick data for 30 days, market data for 2 years
SELECT add_retention_policy('tick_data', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('market_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('trading_signals', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('performance_metrics', INTERVAL '5 years', if_not_exists => TRUE);

-- Create continuous aggregates for common queries
-- Daily OHLCV aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_market_data
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    symbol,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    AVG(vwap) AS avg_vwap
FROM market_data 
WHERE timeframe = '1m'
GROUP BY day, symbol;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('daily_market_data',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Hourly performance metrics aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    strategy,
    symbol,
    LAST(total_pnl, time) AS total_pnl,
    SUM(daily_pnl) AS hourly_pnl,
    AVG(win_rate) AS avg_win_rate,
    AVG(sharpe_ratio) AS avg_sharpe_ratio
FROM performance_metrics
GROUP BY hour, strategy, symbol;

-- Add refresh policy for hourly performance
SELECT add_continuous_aggregate_policy('hourly_performance',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE);

-- Grant permissions to trading_user
GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- Grant permissions on materialized views
GRANT SELECT ON daily_market_data TO trading_user;
GRANT SELECT ON hourly_performance TO trading_user;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at columns
CREATE TRIGGER update_symbols_updated_at BEFORE UPDATE ON symbols
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_earnings_updated_at BEFORE UPDATE ON earnings_calendar
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some initial data for testing
INSERT INTO symbols (symbol, name, exchange, sector, industry, market_cap, is_active) VALUES
('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology', 'Consumer Electronics', 3000000000000, true),
('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology', 'Internet Services', 2000000000000, true),
('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology', 'Software', 2800000000000, true),
('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Cyclical', 'Auto Manufacturers', 800000000000, true),
('NVDA', 'NVIDIA Corporation', 'NASDAQ', 'Technology', 'Semiconductors', 1500000000000, true)
ON CONFLICT (symbol) DO NOTHING;

-- Create a view for active trading opportunities
CREATE OR REPLACE VIEW active_trading_opportunities AS
SELECT 
    gc.symbol,
    gc.gap_type,
    gc.gap_percent,
    gc.scan_date,
    s.name,
    s.sector,
    s.market_cap,
    gc.pre_market_volume,
    gc.relative_volume,
    gc.catalyst
FROM gap_candidates gc
JOIN symbols s ON gc.symbol = s.symbol
WHERE gc.is_tradeable = true 
    AND gc.scan_date >= CURRENT_DATE - INTERVAL '7 days'
    AND s.is_active = true
ORDER BY gc.scan_date DESC, ABS(gc.gap_percent) DESC;

GRANT SELECT ON active_trading_opportunities TO trading_user;

-- Create a function to get latest market data for a symbol
CREATE OR REPLACE FUNCTION get_latest_market_data(symbol_param VARCHAR(10), timeframe_param VARCHAR(10) DEFAULT '1m')
RETURNS TABLE(
    time TIMESTAMPTZ,
    symbol VARCHAR(10),
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    vwap DECIMAL(12,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT md.time, md.symbol, md.open, md.high, md.low, md.close, md.volume, md.vwap
    FROM market_data md
    WHERE md.symbol = symbol_param 
        AND md.timeframe = timeframe_param
    ORDER BY md.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION get_latest_market_data TO trading_user;

-- Create a function to calculate strategy performance
CREATE OR REPLACE FUNCTION calculate_strategy_performance(strategy_param VARCHAR(50), days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    strategy VARCHAR(50),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5,4),
    total_pnl DECIMAL(12,4),
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),
    profit_factor DECIMAL(8,4),
    max_drawdown DECIMAL(8,4)
) AS $$
BEGIN
    RETURN QUERY
    WITH strategy_stats AS (
        SELECT 
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as winning_trades,
            COUNT(*) FILTER (WHERE realized_pnl < 0) as losing_trades,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) FILTER (WHERE realized_pnl > 0) as avg_win,
            AVG(realized_pnl) FILTER (WHERE realized_pnl < 0) as avg_loss
        FROM positions 
        WHERE strategy = strategy_param 
            AND status = 'CLOSED'
            AND closed_at >= NOW() - (days_back || ' days')::INTERVAL
    )
    SELECT 
        strategy_param,
        ss.total_trades::INTEGER,
        ss.winning_trades::INTEGER,
        ss.losing_trades::INTEGER,
        CASE WHEN ss.total_trades > 0 THEN ss.winning_trades::DECIMAL / ss.total_trades ELSE 0 END,
        COALESCE(ss.total_pnl, 0),
        COALESCE(ss.avg_win, 0),
        COALESCE(ss.avg_loss, 0),
        CASE WHEN ss.avg_loss < 0 THEN ABS(ss.avg_win / ss.avg_loss) ELSE 0 END,
        0::DECIMAL(8,4) -- Placeholder for max_drawdown calculation
    FROM strategy_stats ss;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION calculate_strategy_performance TO trading_user;

-- System metrics table for monitoring system performance
CREATE TABLE IF NOT EXISTS system_metrics (
    time TIMESTAMPTZ NOT NULL,
    uptime_seconds INTEGER,
    total_trades INTEGER,
    total_pnl DECIMAL(12,4),
    data_messages_processed INTEGER,
    forecasts_generated INTEGER,
    errors_encountered INTEGER,
    memory_usage INTEGER,
    cpu_usage DECIMAL(5,2),
    metadata JSONB,
    PRIMARY KEY (time)
);

-- Convert system_metrics to hypertable
SELECT create_hypertable('system_metrics', 'time', if_not_exists => TRUE);

-- System events table for logging important events
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert system_events to hypertable
SELECT create_hypertable('system_events', 'timestamp', if_not_exists => TRUE);

-- Trading sessions table for session-based tracking
CREATE TABLE IF NOT EXISTS trading_sessions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    session_type VARCHAR(50) NOT NULL, -- 'premarket_analysis', 'market_open', 'market_close'
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert trading_sessions to hypertable
SELECT create_hypertable('trading_sessions', 'timestamp', if_not_exists => TRUE);

-- Strategy performance table for detailed strategy tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    trades_today INTEGER,
    winning_trades INTEGER,
    total_pnl DECIMAL(12,4),
    win_rate DECIMAL(5,4),
    active_signals INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert strategy_performance to hypertable
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

-- ORB ranges table for opening range breakout data
CREATE TABLE IF NOT EXISTS orb_ranges (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT,
    range_size DECIMAL(12,4),
    range_percent DECIMAL(8,4),
    quality VARCHAR(20),
    vwap DECIMAL(12,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert orb_ranges to hypertable
SELECT create_hypertable('orb_ranges', 'timestamp', if_not_exists => TRUE);

-- Breakout analysis table for ORB breakout data
CREATE TABLE IF NOT EXISTS breakout_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    breakout_direction VARCHAR(20) NOT NULL,
    breakout_price DECIMAL(12,4),
    current_price DECIMAL(12,4),
    sustainability_probability DECIMAL(5,4),
    false_breakout_probability DECIMAL(5,4),
    volume_confirmation BOOLEAN,
    volume_ratio DECIMAL(8,4),
    target_projections DECIMAL(12,4)[],
    resistance_levels DECIMAL(12,4)[],
    support_levels DECIMAL(12,4)[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert breakout_analysis to hypertable
SELECT create_hypertable('breakout_analysis', 'timestamp', if_not_exists => TRUE);

-- Mean reversion metrics table
CREATE TABLE IF NOT EXISTS mean_reversion_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    current_price DECIMAL(12,4),
    sma_20 DECIMAL(12,4),
    sma_50 DECIMAL(12,4),
    sma_200 DECIMAL(12,4),
    vwap DECIMAL(12,4),
    deviation_sma_20 DECIMAL(8,4),
    deviation_sma_50 DECIMAL(8,4),
    deviation_sma_200 DECIMAL(8,4),
    deviation_vwap DECIMAL(8,4),
    z_score_20 DECIMAL(8,4),
    bb_upper DECIMAL(12,4),
    bb_lower DECIMAL(12,4),
    bb_middle DECIMAL(12,4),
    bb_position DECIMAL(5,4),
    rsi DECIMAL(5,2),
    momentum_5 DECIMAL(8,4),
    momentum_10 DECIMAL(8,4),
    volume_ratio DECIMAL(8,4),
    volume_trend VARCHAR(20),
    realized_volatility DECIMAL(8,4),
    volatility_percentile DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert mean_reversion_metrics to hypertable
SELECT create_hypertable('mean_reversion_metrics', 'timestamp', if_not_exists => TRUE);

-- Reversion analysis table
CREATE TABLE IF NOT EXISTS reversion_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    reversion_type VARCHAR(50) NOT NULL,
    mean_type VARCHAR(50),
    reversion_probability DECIMAL(5,4),
    mean_reversion_timeframe INTEGER,
    oversold_bounce_prob DECIMAL(5,4),
    overbought_fade_prob DECIMAL(5,4),
    entry_price DECIMAL(12,4),
    target_mean DECIMAL(12,4),
    target_price DECIMAL(12,4),
    stop_loss_price DECIMAL(12,4),
    confidence_score DECIMAL(5,4),
    max_hold_time INTEGER,
    trend_alignment BOOLEAN,
    volatility_regime VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert reversion_analysis to hypertable
SELECT create_hypertable('reversion_analysis', 'timestamp', if_not_exists => TRUE);

-- Lag-Llama forecasts table
CREATE TABLE IF NOT EXISTS lag_llama_forecasts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    forecast_horizon INTEGER,
    confidence_score DECIMAL(5,4),
    mean_forecast DECIMAL(12,4),
    median_forecast DECIMAL(12,4),
    quantile_10 DECIMAL(12,4),
    quantile_25 DECIMAL(12,4),
    quantile_75 DECIMAL(12,4),
    quantile_90 DECIMAL(12,4),
    volatility_forecast DECIMAL(8,4),
    trend_direction VARCHAR(20),
    optimal_hold_minutes INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert lag_llama_forecasts to hypertable
SELECT create_hypertable('lag_llama_forecasts', 'timestamp', if_not_exists => TRUE);

-- Polygon indicators cache table
CREATE TABLE IF NOT EXISTS polygon_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_type VARCHAR(50) NOT NULL, -- 'RSI', 'MACD', 'SMA', 'EMA', 'BB'
    timespan VARCHAR(10) NOT NULL, -- 'minute', 'hour', 'day'
    window_size INTEGER,
    value DECIMAL(12,4),
    additional_values JSONB, -- For complex indicators like MACD, BB
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, indicator_type, timespan, window_size)
);

-- Convert polygon_indicators to hypertable
SELECT create_hypertable('polygon_indicators', 'timestamp', if_not_exists => TRUE);

-- Market regime analysis table
CREATE TABLE IF NOT EXISTS market_regime (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    regime_type VARCHAR(50) NOT NULL, -- 'bull', 'bear', 'sideways', 'volatile'
    confidence DECIMAL(5,4),
    vix_level DECIMAL(8,4),
    spy_trend VARCHAR(20),
    qqq_trend VARCHAR(20),
    iwm_trend VARCHAR(20),
    sector_rotation JSONB,
    volatility_regime VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert market_regime to hypertable
SELECT create_hypertable('market_regime', 'timestamp', if_not_exists => TRUE);

-- Enhanced gap candidates with more detailed analysis
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS volume_ratio DECIMAL(8,4);
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS price_target DECIMAL(12,4);
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20);
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS technical_score DECIMAL(5,4);
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS momentum_score DECIMAL(5,4);
ALTER TABLE gap_candidates ADD COLUMN IF NOT EXISTS news_sentiment VARCHAR(20);

-- Create additional indexes for new tables
CREATE INDEX IF NOT EXISTS idx_system_metrics_time ON system_metrics (time DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_type_time ON system_events (event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_type_time ON trading_sessions (session_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_time ON strategy_performance (strategy, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orb_ranges_symbol_time ON orb_ranges (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_analysis_symbol_time ON breakout_analysis (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_mean_reversion_metrics_symbol_time ON mean_reversion_metrics (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_reversion_analysis_symbol_time ON reversion_analysis (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lag_llama_forecasts_symbol_time ON lag_llama_forecasts (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_polygon_indicators_symbol_type_time ON polygon_indicators (symbol, indicator_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_regime_time ON market_regime (timestamp DESC);

-- Create indexes on JSONB metadata columns for better performance
CREATE INDEX IF NOT EXISTS idx_trading_signals_metadata_gin ON trading_signals USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_positions_metadata_gin ON positions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_orders_metadata_gin ON orders USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_performance_metadata_gin ON performance_metrics USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_system_metrics_metadata_gin ON system_metrics USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_system_events_metadata_gin ON system_events USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_lag_llama_forecasts_metadata_gin ON lag_llama_forecasts USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_polygon_indicators_additional_gin ON polygon_indicators USING GIN (additional_values);

-- Add compression policies for new tables
SELECT add_compression_policy('system_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('system_events', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('trading_sessions', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('strategy_performance', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('orb_ranges', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('breakout_analysis', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('mean_reversion_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('reversion_analysis', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('lag_llama_forecasts', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('polygon_indicators', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('market_regime', INTERVAL '30 days', if_not_exists => TRUE);

-- Add retention policies for new tables
SELECT add_retention_policy('system_metrics', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('system_events', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('trading_sessions', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('strategy_performance', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('orb_ranges', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('breakout_analysis', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('mean_reversion_metrics', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('reversion_analysis', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('lag_llama_forecasts', INTERVAL '3 months', if_not_exists => TRUE);
SELECT add_retention_policy('polygon_indicators', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('market_regime', INTERVAL '2 years', if_not_exists => TRUE);

-- Grant permissions on new tables
GRANT ALL ON system_metrics TO trading_user;
GRANT ALL ON system_events TO trading_user;
GRANT ALL ON trading_sessions TO trading_user;
GRANT ALL ON strategy_performance TO trading_user;
GRANT ALL ON orb_ranges TO trading_user;
GRANT ALL ON breakout_analysis TO trading_user;
GRANT ALL ON mean_reversion_metrics TO trading_user;
GRANT ALL ON reversion_analysis TO trading_user;
GRANT ALL ON lag_llama_forecasts TO trading_user;
GRANT ALL ON polygon_indicators TO trading_user;
GRANT ALL ON market_regime TO trading_user;

-- Create materialized view for strategy performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS strategy_performance_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    strategy,
    AVG(win_rate) AS avg_win_rate,
    SUM(trades_today) AS total_trades,
    SUM(total_pnl) AS total_pnl,
    AVG(active_signals) AS avg_active_signals
FROM strategy_performance
GROUP BY hour, strategy;

-- Add refresh policy for strategy performance summary
SELECT add_continuous_aggregate_policy('strategy_performance_summary',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE);

GRANT SELECT ON strategy_performance_summary TO trading_user;

-- Create function to get latest system metrics
CREATE OR REPLACE FUNCTION get_latest_system_metrics()
RETURNS TABLE(
    time TIMESTAMPTZ,
    uptime_seconds INTEGER,
    total_trades INTEGER,
    total_pnl DECIMAL(12,4),
    data_messages_processed INTEGER,
    forecasts_generated INTEGER,
    errors_encountered INTEGER,
    memory_usage INTEGER,
    cpu_usage DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT sm.time, sm.uptime_seconds, sm.total_trades, sm.total_pnl, 
           sm.data_messages_processed, sm.forecasts_generated, sm.errors_encountered,
           sm.memory_usage, sm.cpu_usage
    FROM system_metrics sm
    ORDER BY sm.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION get_latest_system_metrics TO trading_user;

-- Create function to get strategy performance for a specific strategy
CREATE OR REPLACE FUNCTION get_strategy_performance_latest(strategy_name VARCHAR(50))
RETURNS TABLE(
    timestamp TIMESTAMPTZ,
    strategy VARCHAR(50),
    trades_today INTEGER,
    winning_trades INTEGER,
    total_pnl DECIMAL(12,4),
    win_rate DECIMAL(5,4),
    active_signals INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT sp.timestamp, sp.strategy, sp.trades_today, sp.winning_trades,
           sp.total_pnl, sp.win_rate, sp.active_signals
    FROM strategy_performance sp
    WHERE sp.strategy = strategy_name
    ORDER BY sp.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION get_strategy_performance_latest TO trading_user;

COMMIT;
