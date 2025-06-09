import asyncio
import pickle
import numpy as np
import os
from typing import List, Dict, Any

# Attempt to import from reg3n_hft_run.py
try:
    from reg3n_hft_run import FeatureEngineer, MarketData, UltraFastLogger, LOG_LEVEL, FEATURE_COUNT
except ImportError:
    print("ERROR: Could not import FeatureEngineer, MarketData, UltraFastLogger, LOG_LEVEL, FEATURE_COUNT from reg3n_hft_run.py.")
    print("Ensure reg3n_hft_run.py is in the same directory or PYTHONPATH.")
    # Define dummy classes/variables if import fails
    FEATURE_COUNT = 12 
    LOG_LEVEL = "INFO"
    class UltraFastLogger:
        def __init__(self, name, level="INFO"): self.name = name; self.level = level
        def info(self, msg): print(f"INFO [{self.name}]: {msg}")
        def error(self, msg): print(f"ERROR [{self.name}]: {msg}")
        def warning(self, msg): print(f"WARNING [{self.name}]: {msg}")
        def debug(self, msg): print(f"DEBUG [{self.name}]: {msg}")

    class MarketData: # Simplified dummy
        def __init__(self, symbol: str, timestamp: float, price: float, volume: int, ohlcv: Dict = None, **kwargs):
            self.symbol = symbol
            self.timestamp = timestamp
            self.price = price
            self.volume = volume
            self.ohlcv = ohlcv if ohlcv is not None else {}
            # Add other fields with defaults if FeatureEngineer expects them
            self.bid = kwargs.get('bid', 0.0)
            self.ask = kwargs.get('ask', 0.0)
            self.bid_size = kwargs.get('bid_size', 0)
            self.ask_size = kwargs.get('ask_size', 0)
            self.data_type = kwargs.get('data_type', "trade")
            self.volatility = kwargs.get('volatility', 0.0)
            self.momentum_score = kwargs.get('momentum_score', 0.0)


    class FeatureEngineer: # Simplified dummy
        def __init__(self, *args, **kwargs):
            self.logger = UltraFastLogger("DummyFeatureEngineer", LOG_LEVEL)
            self.total_feature_count = FEATURE_COUNT
            self.logger.info(f"Dummy FeatureEngineer initialized for {self.total_feature_count} features.")

        async def engineer_features_batch(self, market_data_list: List[MarketData]) -> np.ndarray:
            self.logger.warning("Using DUMMY FeatureEngineer.engineer_features_batch. No real features will be generated.")
            if not market_data_list:
                return np.array([], dtype=np.float32).reshape(0, self.total_feature_count)
            # Simulate feature generation
            num_samples = len(market_data_list)
            return np.random.rand(num_samples, self.total_feature_count).astype(np.float32)

logger = UltraFastLogger("ProcessCalibrationData", level=LOG_LEVEL)

INPUT_RAW_AGGREGATES_FILE = "raw_market_aggregates.pkl"
OUTPUT_CALIBRATION_FEATURES_FILE = "real_calibration_features.npy"
LOOKBACK_WINDOW = 60  # Number of historical bars to provide for feature calculation
SEQUENCE_LENGTH = 50  # Length of feature sequences for the model

def create_market_data_item_with_history(symbol_str: str, current_bar: Dict[str, Any], historical_bars: List[Dict[str, Any]]) -> MarketData:
    """
    Creates a MarketData object for the current_bar, populating its ohlcv field
    with data from historical_bars.
    """
    ohlcv_history = {
        'open': [float(b.get('open', 0)) for b in historical_bars],
        'high': [float(b.get('high', 0)) for b in historical_bars],
        'low': [float(b.get('low', 0)) for b in historical_bars],
        'close': [float(b.get('close', 0)) for b in historical_bars],
        'volume': [int(b.get('volume', 0)) for b in historical_bars]
    }

    # The primary attributes of MarketData should reflect the *current* bar
    md = MarketData(
        symbol=symbol_str,
        timestamp=current_bar.get("timestamp", 0), # Already in seconds from fetch_polygon_data
        price=float(current_bar.get("close", 0)),
        volume=int(current_bar.get("volume", 0)),
        ohlcv=ohlcv_history,
        data_type="aggregate" # Mark as aggregate since it's from bars
    )
    # Add other fields if necessary, e.g. if FeatureEngineer uses them directly from MarketData item
    # For now, assuming FeatureEngineer primarily uses price, volume, timestamp and the ohlcv history
    return md

async def main():
    if "FeatureEngineer" not in globals() or globals()["FeatureEngineer"].__module__ == __name__:
        logger.error("Actual FeatureEngineer from reg3n_hft_run.py could not be imported. Exiting.")
        return

    if not os.path.exists(INPUT_RAW_AGGREGATES_FILE):
        logger.error(f"Input file {INPUT_RAW_AGGREGATES_FILE} not found. Please run fetch_polygon_data.py first.")
        return

    logger.info(f"Loading raw aggregates from {INPUT_RAW_AGGREGATES_FILE}...")
    with open(INPUT_RAW_AGGREGATES_FILE, 'rb') as f:
        raw_aggregates_data: Dict[str, List[Dict[str, Any]]] = pickle.load(f)

    if not raw_aggregates_data:
        logger.warning("No data found in raw_market_aggregates.pkl. Exiting.")
        return

    all_engineered_features_list: List[np.ndarray] = []
    
    # Instantiate FeatureEngineer once if it handles internal state per symbol appropriately,
    # or inside the loop if state needs to be reset per symbol.
    # Based on _price_buffer and _volume_buffer, it seems to maintain state.
    # However, engineer_features_batch processes a list for one symbol at a time.
    # The compute_all_features_for_item is called for each item in that list.
    # The _get_ohlcv_from_market_data uses the ohlcv field of the *current* item.
    # So, instantiating once should be fine.
    feature_engineer = FeatureEngineer()

    total_feature_vectors = 0

    for symbol, symbol_bars in raw_aggregates_data.items():
        if not symbol_bars:
            logger.info(f"No bars found for symbol {symbol}. Skipping.")
            continue
        
        logger.info(f"Processing {len(symbol_bars)} bars for symbol {symbol}...")
        
        market_data_list_for_symbol: List[MarketData] = []
        for i in range(len(symbol_bars)):
            if i < LOOKBACK_WINDOW - 1:
                # Not enough history for a full lookback window for the earliest bars
                # These initial bars won't be used to generate features that require full history.
                # However, we still create MarketData items for them to be part of the history
                # for later bars.
                # The `compute_all_features_for_item` will handle partial history gracefully.
                # Or, we can choose to only generate features for bars where full history is available.
                # For calibration, it's better to have features from consistent history length.
                pass # We will only create MD items that have full lookback

            # Define the slice for historical bars for the current bar at index `i`
            # History includes the current bar itself as the most recent point.
            history_start_idx = max(0, i - LOOKBACK_WINDOW + 1)
            current_historical_slice = symbol_bars[history_start_idx : i + 1]
            
            # Only create a MarketData item for feature generation if we have enough history
            if len(current_historical_slice) < LOOKBACK_WINDOW : # Ensure full lookback for the features
                 if i >= LOOKBACK_WINDOW -1: # Should not happen if logic is correct
                      logger.warning(f"Skipping bar {i} for {symbol} due to insufficient history slice len {len(current_historical_slice)} despite index.")
                 continue


            md_item = create_market_data_item_with_history(symbol, symbol_bars[i], current_historical_slice)
            market_data_list_for_symbol.append(md_item)

        if not market_data_list_for_symbol:
            logger.info(f"No processable data points (after lookback consideration) for symbol {symbol}. Skipping.")
            continue

        logger.info(f"Engineering features for {len(market_data_list_for_symbol)} data points for {symbol}...")
        # engineer_features_batch expects a list of MarketData items, where each item
        # has its relevant history already embedded in its .ohlcv field.
        try:
            features_for_symbol_array: np.ndarray = await feature_engineer.engineer_features_batch(market_data_list_for_symbol)
            
            if features_for_symbol_array is not None and features_for_symbol_array.ndim == 2 and features_for_symbol_array.shape[0] >= SEQUENCE_LENGTH:
                num_features_actual = features_for_symbol_array.shape[1]
                if num_features_actual != FEATURE_COUNT:
                    logger.warning(f"Feature count mismatch for {symbol}! Expected {FEATURE_COUNT}, got {num_features_actual}. Skipping symbol.")
                    continue

                # Create sequences for this symbol
                symbol_sequences_list = []
                num_possible_sequences = features_for_symbol_array.shape[0] - SEQUENCE_LENGTH + 1
                for i in range(num_possible_sequences):
                    sequence = features_for_symbol_array[i : i + SEQUENCE_LENGTH, :]
                    symbol_sequences_list.append(sequence)
                
                if symbol_sequences_list:
                    stacked_symbol_sequences = np.stack(symbol_sequences_list, axis=0) # Shape: (num_seq_for_symbol, SEQUENCE_LENGTH, FEATURE_COUNT)
                    all_engineered_features_list.append(stacked_symbol_sequences)
                    total_feature_vectors += stacked_symbol_sequences.shape[0] # Counting sequences now
                    logger.info(f"Successfully created {stacked_symbol_sequences.shape[0]} sequences of shape {stacked_symbol_sequences.shape[1:]} for {symbol}.")
                else:
                    logger.warning(f"No sequences generated for {symbol} despite having enough feature vectors initially.")
            else:
                fv_shape_str = str(features_for_symbol_array.shape) if features_for_symbol_array is not None else "None"
                min_req_len = SEQUENCE_LENGTH
                logger.warning(f"Not enough feature vectors (shape: {fv_shape_str}, min required: {min_req_len}) to form any sequence for {symbol}, or feature engineering returned invalid data.")
        except Exception as e:
            logger.error(f"Error during feature engineering for symbol {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())


    if not all_engineered_features_list:
        logger.warning("No features were engineered for any symbol. Output file will not be saved.")
        return

    logger.info(f"Concatenating sequences from all symbols. Total sequences: {total_feature_vectors}")
    # all_engineered_features_list contains 3D arrays [(num_seq_sym1, SL, FC), (num_seq_sym2, SL, FC), ...]
    final_features_array = np.concatenate(all_engineered_features_list, axis=0) # axis=0 to stack sequences
    
    logger.info(f"Saving {final_features_array.shape[0]} sequences, each of shape ({final_features_array.shape[1]}, {final_features_array.shape[2]}) to {OUTPUT_CALIBRATION_FEATURES_FILE}...")
    np.save(OUTPUT_CALIBRATION_FEATURES_FILE, final_features_array)
    logger.info(f"Successfully saved calibration sequences. Final array shape: {final_features_array.shape}")

if __name__ == "__main__":
    asyncio.run(main())