import asyncio
import numpy as np
import os
import pickle
from datetime import datetime, timedelta, date
import pandas as pd # For BDay

# --- Configuration ---
# WARNING: Hardcoding API keys is generally not recommended for production.
# Consider using environment variables or a secure config file.
POLYGON_API_KEY_FETCH = "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57"

SYMBOLS = ['NVDA', 'TSLA', 'AMD', 'SPY', 'QQQ']
TIMESPAN = "minute"
MULTIPLIER = 1
OUTPUT_FILE = "raw_market_aggregates.pkl"
# Number of recent trading days to fetch data for
NUM_TRADING_DAYS = 5
# Max aggregates to fetch per symbol per day (Polygon limit is 50000, 1-min bars in a day is 390)
AGGREGATES_LIMIT_PER_DAY = 500 

# Attempt to import from reg3n_hft_run.py
# This assumes fetch_polygon_data.py is in the same directory as reg3n_hft_run.py
# or reg3n_hft_run.py is in the Python path.
try:
    from reg3n_hft_run import SymbolManager, UltraFastLogger, LOG_LEVEL
except ImportError:
    print("ERROR: Could not import SymbolManager, UltraFastLogger, LOG_LEVEL from reg3n_hft_run.py.")
    print("Ensure reg3n_hft_run.py is in the same directory or PYTHONPATH.")
    # Define dummy classes/variables if import fails, so script can be partially reviewed
    # This is primarily for static analysis if the main script isn't runnable standalone here.
    class UltraFastLogger:
        def __init__(self, name, level="INFO"): self.name = name; self.level = level
        def info(self, msg): print(f"INFO [{self.name}]: {msg}")
        def error(self, msg): print(f"ERROR [{self.name}]: {msg}")
        def warning(self, msg): print(f"WARNING [{self.name}]: {msg}")
        def debug(self, msg): print(f"DEBUG [{self.name}]: {msg}")
    LOG_LEVEL = "INFO"
    class SymbolManager:
        def __init__(self, api_key): self.api_key = api_key; self.logger = UltraFastLogger("DummySymbolManager", LOG_LEVEL)
        async def fetch_historical_aggregates(self, symbol, timespan, date_from, date_to, multiplier, limit):
            self.logger.warning("Using DUMMY SymbolManager.fetch_historical_aggregates. No real data will be fetched.")
            return {'symbol': symbol, 'results': []} # Return empty structure

logger = UltraFastLogger("FetchPolygonData", level=LOG_LEVEL)

def get_last_n_trading_days(n: int) -> list[str]:
    """Gets the last N trading days in YYYY-MM-DD format."""
    trading_days = []
    current_date = date.today()
    while len(trading_days) < n:
        # pandas BDay to check for business days (approximates trading days, excluding weekends)
        # For more accuracy, a proper trading calendar considering holidays would be needed.
        # This simple version subtracts calendar days and checks if it's a weekday.
        if current_date.weekday() < 5: # Monday to Friday
            trading_days.append(current_date.strftime("%Y-%m-%d"))
        current_date -= timedelta(days=1)
    trading_days.reverse() # Get them in chronological order
    logger.info(f"Target trading days for fetching: {trading_days}")
    return trading_days

async def fetch_and_save_data():
    if not POLYGON_API_KEY_FETCH or POLYGON_API_KEY_FETCH == "YOUR_POLYGON_API_KEY_HERE":
        logger.error("POLYGON_API_KEY_FETCH is not set or is a placeholder. Please update it in this script.")
        return

    # Ensure SymbolManager was imported correctly
    if "SymbolManager" not in globals() or globals()["SymbolManager"].__module__ == __name__: # Check if it's the dummy
        logger.error("Actual SymbolManager from reg3n_hft_run.py could not be imported. Exiting.")
        return

    symbol_manager = SymbolManager(api_key=POLYGON_API_KEY_FETCH)
    
    all_aggregates_data = {} # Store data per symbol
    
    target_dates = get_last_n_trading_days(NUM_TRADING_DAYS)

    for symbol in SYMBOLS:
        all_aggregates_data[symbol] = []
        logger.info(f"--- Processing symbol: {symbol} ---")
        for date_str in target_dates:
            logger.info(f"Fetching {TIMESPAN} aggregates for {symbol} on {date_str}...")
            try:
                aggregates_result = await symbol_manager.fetch_historical_aggregates(
                    symbol=symbol,
                    timespan=TIMESPAN,
                    from_date=date_str,
                    to_date=date_str,
                    multiplier=MULTIPLIER,
                    limit=AGGREGATES_LIMIT_PER_DAY 
                )
                
                if aggregates_result and aggregates_result.get('bars') and aggregates_result.get('count', 0) > 0:
                    num_bars = len(aggregates_result['bars'])
                    logger.info(f"Fetched {num_bars} aggregate bar(s) for {symbol} on {date_str}.")
                    all_aggregates_data[symbol].extend(aggregates_result['bars'])
                elif aggregates_result and aggregates_result.get('count', -1) == 0: # Check if 'count' is explicitly 0
                     logger.info(f"No aggregates found for {symbol} on {date_str} (API returned count: 0).")
                else: # Handles other cases, including API errors that might not have 'bars' or 'count'
                    logger.warning(f"No 'bars' key, zero count, or other issue for {symbol} on {date_str}. Response: {aggregates_result}")
            except Exception as e:
                logger.error(f"Error fetching/processing aggregates for {symbol} on {date_str}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        logger.info(f"Total aggregates fetched for {symbol}: {len(all_aggregates_data[symbol])}")

    # Save the raw data
    if any(all_aggregates_data.values()): # Check if any data was fetched
        try:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(all_aggregates_data, f)
            logger.info(f"Successfully saved raw market aggregates to {OUTPUT_FILE}")
            total_points = sum(len(v) for v in all_aggregates_data.values())
            logger.info(f"Total aggregate data points saved: {total_points}")
        except Exception as e:
            logger.error(f"Error saving data to pickle file: {e}")
    else:
        logger.warning("No data fetched for any symbol. Output file not saved.")

if __name__ == "__main__":
    logger.info("Starting Polygon data fetching script...")
    # This script uses asyncio for SymbolManager's async methods
    asyncio.run(fetch_and_save_data())
    logger.info("Polygon data fetching script finished.")