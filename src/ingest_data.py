"""
Data Ingestion Module for Cryptocurrency Market Data

Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance API.
Supports multiple cryptocurrencies and timeframes with built-in data validation
and persistence to CSV format.

"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

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
        Initialize the data fetcher for specified cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            data_dir: Output directory for CSV files
            
        """
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        
        logger.info(f"Initialized CryptoDataFetcher for {symbol}")
    
    def fetch_historical_data(
        self,
        days_back: int = 730,
        timeframe: str = '1d',
        save_to_csv: bool = True,
        use_ticker_object: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic Yahoo Finance limit handling.
        
        Yahoo Finance imposes data availability limits:
        - Minute intervals: 7 days maximum
        - Hourly intervals: 60 days maximum  
        - Daily intervals: Full historical data available
        
        Args:
            timeframe: Candle interval ('1m', '5m', '1h', '1d', etc.)
            days_back: Number of days to fetch (default: 730 = 2 years)
            save_to_csv: Persist data to disk
            use_ticker_object: Use yf.Ticker() instead of yf.download()
            
        Returns:
            DataFrame with OHLCV columns and datetime index
            
        Raises:
            ValueError: If invalid timeframe specified
            
        Example:
            >>> df = fetcher.fetch_historical_data('1d', 730)
            >>> df.columns
            Index(['Open', 'High', 'Low', 'Close', 'Volume'], dtype='object')
        """
        if timeframe not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{timeframe}'. "
                f"Must be one of: {self.VALID_INTERVALS}"
            )
        
        # Enforce Yahoo Finance API limits
        if timeframe in ['1m', '2m', '5m', '15m', '30m'] and days_back > 7:
            logger.warning(
                f"Minute data limited to 7 days. Adjusting from {days_back} to 7."
            )
            days_back = 7
        elif timeframe in ['1h', '90m', '60m'] and days_back > 60:
            logger.warning(
                f"Hourly data limited to 60 days. Adjusting from {days_back} to 60."
            )
            days_back = 60
        
        logger.info(f"Fetching {days_back} days of {timeframe} data for {self.symbol}")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Two API approaches: functional vs OOP
            # Functional is faster for batch downloads, OOP provides metadata access
            if use_ticker_object:
                logger.info("Using yf.Ticker().history() method")
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            else:
                logger.info("Using yf.download() method")
                df = yf.download(
                    self.symbol,
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    progress=False
                )
            
            if df.empty:
                logger.error(f"No data retrieved for {self.symbol}")
                return pd.DataFrame()
            
            df = self._clean_data(df)
            
            logger.info(f"Successfully fetched {len(df)} candles")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            if save_to_csv:
                self._save_to_csv(df, timeframe)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_latest_data(
        self,
        period: str = '30d',
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch the most recent data (for real-time predictions).
       
        Args:
            period: Time period ('1d', '7d', '1mo', etc.)
            interval: Time between data points
            
        Returns:
            DataFrame with recent data
            
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

        df = df.copy()
        initial_len = len(df)     

        ## Drop incomplete candles   
        df.dropna(inplace=True)
        
        # Filter zero/negative volume (data quality issue)
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]
            removed = initial_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with invalid volume")
        
        # Standardize column names (Yahoo sometimes returns tuples)
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

        logger.info(f"Cleaned data: {len(df)} valid rows")
        return df
    
    def _save_to_csv(self, df: pd.DataFrame, interval: str) -> None:
        """
        Save data to a CSV file.
        Format: {symbol}_{interval}_{date}.csv
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
    print("\n[Example 1] Fetching 2 year of daily Bitcoin data...")
    btc_fetcher = CryptoDataFetcher(symbol='BTC-USD')
    btc_data = btc_fetcher.fetch_historical_data(
        days_back=730,
        timeframe='1d',
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
    print("\n[Example 3] Fetching latest 30 days...")
    latest_data = btc_fetcher.fetch_latest_data(period='30d', interval='1d')
    print(f"Got {len(latest_data)} recent data points")
    
    # Example 4: Try Ethereum
    print("\n[Example 4] Fetching Ethereum data...")
    eth_fetcher = CryptoDataFetcher(symbol='ETH-USD')
    eth_data = eth_fetcher.fetch_historical_data(
        days_back=30,
        timeframe='1d'
    )
    print(f"Got {len(eth_data)} Ethereum data points")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)


# This runs when you execute the file directly
if __name__ == "__main__":
    example_usage()