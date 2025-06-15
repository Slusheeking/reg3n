-- Model versioning table for fine-tuned Lag-Llama models
-- Execute this SQL manually in your PostgreSQL database

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_id VARCHAR(100) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    checkpoint_path VARCHAR(255) NOT NULL,
    training_start TIMESTAMPTZ NOT NULL,
    training_end TIMESTAMPTZ NOT NULL,
    training_data_period INTEGER NOT NULL,
    performance_metrics JSONB,
    validation_loss DECIMAL(12,8) NOT NULL,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_model_versions_type_loss ON model_versions (model_type, validation_loss ASC);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions (is_active, model_type);
CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions (created_at DESC);

-- Create trigger for updated_at
CREATE TRIGGER update_model_versions_updated_at 
    BEFORE UPDATE ON model_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON model_versions TO trading_user;
GRANT ALL ON SEQUENCE model_versions_id_seq TO trading_user;

-- Multi-timeframe forecasts extension
ALTER TABLE lag_llama_forecasts ADD COLUMN IF NOT EXISTS forecast_timeframe INTEGER DEFAULT 15;
CREATE INDEX IF NOT EXISTS idx_forecasts_timeframe ON lag_llama_forecasts (forecast_timeframe, timestamp DESC);

COMMIT;