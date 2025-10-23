
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """Fetches cryptocurrency OHLCV data from exchanges."""
    
    def __init__(self, exchange_name: str = 'binance', symbol: str = 'BTC/USDT'):
        """
        Initialize the data fetcher.
        
        Args:
            exchange_name: Name of the exchange (binance, coinbase, etc.)
            symbol: Trading pair symbol
        """
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,  # Respect rate limits
        })
        self.symbol = symbol
        logger.info(f"Initialized {exchange_name} fetcher for {symbol}")
    
    def fetch_historical_data(
        self, 
        timeframe: str = '1h',
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            timeframe: Candlestick timeframe ('1m', '5m', '1h', '1d')
            days_back: Number of days to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {days_back} days of {timeframe} data...")
        
        since = self.exchange.parse8601(
            (datetime.now() - timedelta(days=days_back)).isoformat()
        )
        
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, 
                    timeframe, 
                    since,
                    limit=1000  # Most exchanges limit to 1000 candles per request
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Move to next timestamp
                
                logger.info(f"Fetched {len(ohlcv)} candles, total: {len(all_ohlcv)}")
                time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df
    
    def fetch_latest_data(self, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch the most recent OHLCV data for prediction.
        
        Args:
            timeframe: Candlestick timeframe
            limit: Number of recent candles to fetch
            
        Returns:
            DataFrame with recent OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            raise