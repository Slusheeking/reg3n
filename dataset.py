"""
Dataset Generator for Lag-Llama Testing
Fetches and prepares real historical data from Polygon API
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict

from polygon import PolygonHTTPClient
from settings import config

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a generated dataset"""
    symbol: str
    total_points: int
    start_date: datetime
    end_date: datetime
    price_range: Tuple[float, float]
    avg_volume: float
    data_quality_score: float
    file_path: str

class PolygonDatasetGenerator:
    """Generate datasets from real Polygon historical data"""
    
    def __init__(self, custom_symbols=None):
        self.polygon_client = PolygonHTTPClient(config.polygon.api_key)
        
        # Use custom symbols if provided, otherwise use default
        if custom_symbols:
            self.test_symbols = custom_symbols
        else:
            self.test_symbols = ['PLTR', 'AMD', 'CRCL', 'SPY', 'QQQ']
            
        self.required_data_points = 600  # Minimum for Lag-Llama + testing
        self.datasets_dir = "datasets"
        
        # Ensure datasets directory exists
        os.makedirs(self.datasets_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the Polygon client"""
        await self.polygon_client.initialize()
        
    async def cleanup(self):
        """Cleanup the Polygon client"""
        await self.polygon_client.close()
        
    async def fetch_historical_data(self, symbol: str, days: int = 5) -> List[Dict]:
        """Fetch historical minute-level data from Polygon"""
        logger.info(f"📊 Fetching historical data for {symbol} ({days} days)...")
        
        try:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            # Adjust for weekends - go back further if needed
            while start_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                start_date -= timedelta(days=1)
            
            # Fetch minute-level aggregates
            response = await self.polygon_client.get_aggregates(
                symbol=symbol,
                multiplier=1,
                timespan="minute",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            data_points = []
            if response.get('results'):
                for bar in response['results']:
                    # Convert timestamp from milliseconds to datetime
                    timestamp = datetime.fromtimestamp(bar.get('t', 0) / 1000)
                    
                    # Only include market hours data (9:30 AM - 4:00 PM ET)
                    if 9.5 <= timestamp.hour + timestamp.minute/60 <= 16:
                        data_point = {
                            'timestamp': timestamp,
                            'open': float(bar.get('o', 0.0)),
                            'high': float(bar.get('h', 0.0)),
                            'low': float(bar.get('l', 0.0)),
                            'close': float(bar.get('c', 0.0)),
                            'volume': int(bar.get('v', 0)),
                            'vwap': float(bar.get('vw', 0.0)),
                            'transactions': int(bar.get('n', 0))
                        }
                        data_points.append(data_point)
            
            logger.info(f"✅ Fetched {len(data_points)} data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return []
    
    async def fetch_extended_data_if_needed(self, symbol: str, current_data: List[Dict]) -> List[Dict]:
        """Fetch additional historical data if we don't have enough points"""
        if len(current_data) >= self.required_data_points:
            return current_data
            
        logger.info(f"🔄 Need more data for {symbol}. Current: {len(current_data)}, Required: {self.required_data_points}")
        
        # Try fetching more days
        for days in [10, 15, 20, 30]:
            extended_data = await self.fetch_historical_data(symbol, days)
            if len(extended_data) >= self.required_data_points:
                logger.info(f"✅ Got sufficient data for {symbol} with {days} days: {len(extended_data)} points")
                return extended_data
            
        logger.warning(f"⚠️ Could only get {len(current_data)} points for {symbol}, proceeding anyway")
        return current_data
    
    def calculate_data_quality_score(self, data: List[Dict]) -> float:
        """Calculate data quality score based on various metrics"""
        if not data:
            return 0.0
            
        # Convert to arrays for analysis
        prices = [d['close'] for d in data]
        volumes = [d['volume'] for d in data]
        
        # Completeness score (no missing data since we filter)
        completeness_score = 1.0
        
        # Price continuity (check for reasonable price movements)
        if len(prices) > 1:
            returns = np.diff(prices) / np.array(prices[:-1])
            # Check for extreme movements (> 20% in one minute)
            extreme_moves = np.abs(returns) > 0.20
            continuity_score = max(0.5, 1.0 - (np.sum(extreme_moves) / len(returns)))
        else:
            continuity_score = 1.0
            
        # Volume consistency (check for reasonable volume patterns)
        if len(volumes) > 1:
            volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
            # Lower coefficient of variation is better (more consistent)
            volume_score = max(0.3, 1.0 - min(volume_cv / 2.0, 1.0))
        else:
            volume_score = 0.5
            
        # Data density score (more data points = better)
        density_score = min(len(data) / self.required_data_points, 1.0)
        
        # Combined score
        quality_score = (
            completeness_score * 0.25 +
            continuity_score * 0.25 +
            volume_score * 0.25 +
            density_score * 0.25
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def prepare_dataset_for_cache(self, symbol: str, data: List[Dict]) -> List[Tuple[float, int, float]]:
        """Prepare dataset for cache loading (price, volume, timestamp)"""
        cache_data = []
        
        for point in data:
            price = point['close']
            volume = point['volume']
            timestamp = point['timestamp'].timestamp()
            cache_data.append((price, volume, timestamp))
            
        # Sort by timestamp to ensure chronological order
        cache_data.sort(key=lambda x: x[2])
        
        return cache_data
    
    def save_dataset(self, symbol: str, data: List[Dict], dataset_info: DatasetInfo) -> str:
        """Save dataset to file"""
        filename = f"{symbol}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.datasets_dir, filename)
        
        dataset = {
            'info': asdict(dataset_info),
            'data': []
        }
        
        # Convert datetime objects to ISO strings for JSON serialization
        for point in data:
            point_copy = point.copy()
            point_copy['timestamp'] = point['timestamp'].isoformat()
            dataset['data'].append(point_copy)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
            
        logger.info(f"💾 Saved dataset for {symbol} to {filepath}")
        return filepath
    
    def load_dataset(self, filepath: str) -> Tuple[DatasetInfo, List[Dict]]:
        """Load dataset from file"""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
            
        # Reconstruct DatasetInfo
        info_dict = dataset['info']
        info_dict['start_date'] = datetime.fromisoformat(info_dict['start_date'])
        info_dict['end_date'] = datetime.fromisoformat(info_dict['end_date'])
        dataset_info = DatasetInfo(**info_dict)
        
        # Reconstruct data with datetime objects
        data = []
        for point in dataset['data']:
            point_copy = point.copy()
            point_copy['timestamp'] = datetime.fromisoformat(point['timestamp'])
            data.append(point_copy)
            
        return dataset_info, data
    
    async def generate_all_datasets(self) -> Dict[str, DatasetInfo]:
        """Generate datasets for all test symbols"""
        logger.info("🚀 Starting dataset generation for all symbols...")
        
        await self.initialize()
        
        try:
            datasets_info = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📈 Processing {symbol}...")
                
                # Fetch initial data
                data = await self.fetch_historical_data(symbol, days=5)
                
                # Fetch more data if needed
                data = await self.fetch_extended_data_if_needed(symbol, data)
                
                if not data:
                    logger.error(f"❌ No data available for {symbol}")
                    continue
                
                # Calculate metrics
                prices = [d['close'] for d in data]
                volumes = [d['volume'] for d in data]
                
                price_range = (min(prices), max(prices))
                avg_volume = np.mean(volumes)
                data_quality_score = self.calculate_data_quality_score(data)
                
                # Create dataset info
                dataset_info = DatasetInfo(
                    symbol=symbol,
                    total_points=len(data),
                    start_date=data[0]['timestamp'],
                    end_date=data[-1]['timestamp'],
                    price_range=price_range,
                    avg_volume=avg_volume,
                    data_quality_score=data_quality_score,
                    file_path=""  # Will be set after saving
                )
                
                # Save dataset
                filepath = self.save_dataset(symbol, data, dataset_info)
                dataset_info.file_path = filepath
                
                datasets_info[symbol] = dataset_info
                
                logger.info(f"✅ Generated dataset for {symbol}:")
                logger.info(f"   📊 Data points: {len(data)}")
                logger.info(f"   💰 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
                logger.info(f"   📈 Avg volume: {avg_volume:,.0f}")
                logger.info(f"   🎯 Quality score: {data_quality_score:.3f}")
                
                # Rate limiting
                await asyncio.sleep(1)
            
            # Save summary
            self.save_datasets_summary(datasets_info)
            
            logger.info("🎉 Dataset generation completed successfully!")
            return datasets_info
            
        finally:
            await self.cleanup()
    
    def save_datasets_summary(self, datasets_info: Dict[str, DatasetInfo]):
        """Save summary of all generated datasets"""
        summary_file = os.path.join(self.datasets_dir, "datasets_summary.json")
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_symbols': len(datasets_info),
            'symbols': list(datasets_info.keys()),
            'datasets': {}
        }
        
        for symbol, info in datasets_info.items():
            summary['datasets'][symbol] = asdict(info)
            # Convert datetime objects to strings
            summary['datasets'][symbol]['start_date'] = info.start_date.isoformat()
            summary['datasets'][symbol]['end_date'] = info.end_date.isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"📋 Saved datasets summary to {summary_file}")
    
    def load_datasets_summary(self) -> Optional[Dict]:
        """Load datasets summary"""
        summary_file = os.path.join(self.datasets_dir, "datasets_summary.json")
        
        if not os.path.exists(summary_file):
            return None
            
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset files"""
        if not os.path.exists(self.datasets_dir):
            return []
            
        dataset_files = []
        for filename in os.listdir(self.datasets_dir):
            if filename.endswith('_dataset_*.json') and filename != 'datasets_summary.json':
                dataset_files.append(os.path.join(self.datasets_dir, filename))
                
        return sorted(dataset_files)


async def main():
    """Main function to generate datasets"""
    generator = PolygonDatasetGenerator()
    
    try:
        datasets_info = await generator.generate_all_datasets()
        
        print("\n🎯 Dataset Generation Summary:")
        print("=" * 50)
        
        for symbol, info in datasets_info.items():
            print(f"\n📈 {symbol}:")
            print(f"   📊 Data points: {info.total_points}")
            print(f"   📅 Date range: {info.start_date.strftime('%Y-%m-%d')} to {info.end_date.strftime('%Y-%m-%d')}")
            print(f"   💰 Price range: ${info.price_range[0]:.2f} - ${info.price_range[1]:.2f}")
            print(f"   📈 Avg volume: {info.avg_volume:,.0f}")
            print(f"   🎯 Quality score: {info.data_quality_score:.3f}")
            print(f"   📁 File: {info.file_path}")
        
        print(f"\n✅ Successfully generated datasets for {len(datasets_info)} symbols!")
        
    except Exception as e:
        logger.error(f"❌ Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())