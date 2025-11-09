"""
DATA INGESTION MODULE - Crypto Volatility Forecasting
=====================================================

This module fetches cryptocurrency data from Yahoo Finance.

What it does (in simple terms):
- Connects to Yahoo Finance (like visiting a website)
- Downloads historical price data (Open, High, Low, Close, Volume)
- Saves it to a CSV file for later use
- Provides easy functions to get data whenever you need it

"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

# Set up logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """
    A class to fetch cryptocurrency data from Yahoo Finance.
        
    Example:
        fetcher = CryptoDataFetcher(symbol='BTC-USD')
        data = fetcher.fetch_historical_data(days_back=365)
    """
    
    # Supported cryptocurrencies on Yahoo Finance
    SUPPORTED_SYMBOLS = {
        'BTC': 'BTC-USD',      # Bitcoin
        'ETH': 'ETH-USD',      # Ethereum
        'BNB': 'BNB-USD',      # Binance Coin
        'SOL': 'SOL-USD',      # Solana
        'ADA': 'ADA-USD',      # Cardano
        'XRP': 'XRP-USD',      # Ripple
        'DOT': 'DOT-USD',      # Polkadot
        'DOGE': 'DOGE-USD',    # Dogecoin
    }
    
    # Valid time intervals
    VALID_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
    
    def __init__(self, symbol: str = 'BTC-USD', data_dir: str = 'data/raw'):
        """
        Initialize the data fetcher.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            data_dir: Directory to save downloaded data
            
        Example:
            # For Bitcoin
            fetcher = CryptoDataFetcher('BTC-USD')
            
            # For Ethereum
            fetcher = CryptoDataFetcher('ETH-USD')
        """
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        
        logger.info(f"Initialized CryptoDataFetcher for {symbol}")
    
    def fetch_historical_data(
        self,
        days_back: int = 365,
        interval: str = '1h',
        save_to_csv: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data.

        Args:
            days_back: How many days of history to fetch (e.g., 365 = 1 year)
            interval: Time between data points ('1h' = hourly, '1d' = daily)
            save_to_csv: Whether to save the data to a CSV file
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            
        Example:
            data = fetcher.fetch_historical_data(days_back=365, interval='1h')
            print(f"Got {len(data)} data points")
        """
        
        # Validate interval
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Must be one of: {self.VALID_INTERVALS}"
            )
        
        logger.info(f"Fetching {days_back} days of {interval} data for {self.symbol}...")
        
        try:
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Download data from Yahoo Finance
            # This is where the magic happens!
            df = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False  # Hide progress bar for cleaner output
            )
            
            if df.empty:
                logger.error(f"No data retrieved for {self.symbol}")
                return pd.DataFrame()
            
            # Clean up the data
            df = self._clean_data(df)
            
            logger.info(f"Successfully fetched {len(df)} data points")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Save to CSV if requested
            if save_to_csv:
                self._save_to_csv(df, interval)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_latest_data(
        self,
        period: str = '7d',
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch the most recent data (for real-time predictions).
       
        Args:
            period: Time period ('1d', '7d', '1mo', etc.)
            interval: Time between data points
            
        Returns:
            DataFrame with recent data
            
        Example:
            # Get last 7 days of hourly data
            recent_data = fetcher.fetch_latest_data(period='7d', interval='1h')
        """
        logger.info(f"Fetching latest {period} of data for {self.symbol}...")
        
        try:
            df = yf.download(
                self.symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                logger.error(f"No data retrieved")
                return pd.DataFrame()
            
            df = self._clean_data(df)
            logger.info(f"Fetched {len(df)} recent data points")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the data.
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove rows with missing values
        df.dropna(inplace=True)
        
        # Rename columns to be consistent (remove any extra spaces or weird formatting)
        new_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                # If it's a tuple, take the first element
                new_columns.append(col[0])
            elif isinstance(col, str):
                # If it's a string, just strip whitespace
                new_columns.append(col.strip())
            else:
                # If it's something else, convert to string
                new_columns.append(str(col))
        
        df.columns = new_columns
        
        # Ensure index is datetime and sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        return df
    
    def _save_to_csv(self, df: pd.DataFrame, interval: str) -> None:
        """
        Save data to a CSV file.
        
        The file will be named like: BTC-USD_1h_20241104.csv
        """
        filename = f"{self.symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f" Data saved to: {filepath}")
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load previously saved data from CSV.
        Useful if you already downloaded data and don't want to fetch it again.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with the loaded data
            
        Example:
            data = fetcher.load_from_csv('BTC-USD_1h_20241104.csv')
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        logger.info(f"Loaded {len(df)} data points")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the data.
        
        Returns useful statistics about your data:
        - How many data points
        - Date range
        - Average price
        - Min/Max prices
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'symbol': self.symbol,
            'num_points': len(df),
            'start_date': str(df.index[0]),
            'end_date': str(df.index[-1]),
            'avg_close': df['Close'].mean(),
            'min_close': df['Close'].min(),
            'max_close': df['Close'].max(),
            'avg_volume': df['Volume'].mean(),
        }
        
        return summary


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

def example_usage():
    """
    Example showing how to use the CryptoDataFetcher.
    
    Run this file directly to test: python src/data_ingestion.py
    """
    print("=" * 70)
    print("CRYPTO DATA INGESTION - EXAMPLE")
    print("=" * 70)
    
    # Example 1: Fetch Bitcoin data
    print("\n[Example 1] Fetching 1 year of hourly Bitcoin data...")
    btc_fetcher = CryptoDataFetcher(symbol='BTC-USD')
    btc_data = btc_fetcher.fetch_historical_data(
        days_back=365,
        interval='1h',
        save_to_csv=True
    )
    
    print("\nFirst 5 rows:")
    print(btc_data.head())
    
    print("\nLast 5 rows:")
    print(btc_data.tail())
    
    # Example 2: Get data summary
    print("\n[Example 2] Data Summary:")
    summary = btc_fetcher.get_data_summary(btc_data)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Example 3: Fetch latest data (for predictions)
    print("\n[Example 3] Fetching latest 7 days...")
    latest_data = btc_fetcher.fetch_latest_data(period='7d', interval='1h')
    print(f"Got {len(latest_data)} recent data points")
    
    # Example 4: Try Ethereum
    print("\n[Example 4] Fetching Ethereum data...")
    eth_fetcher = CryptoDataFetcher(symbol='ETH-USD')
    eth_data = eth_fetcher.fetch_historical_data(
        days_back=30,
        interval='1h'
    )
    print(f"Got {len(eth_data)} Ethereum data points")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)


# This runs when you execute the file directly
if __name__ == "__main__":
    example_usage()