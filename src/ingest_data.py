import ccxt
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables (API keys)
load_dotenv()

def fetch_ohlcv_data(exchange_id, symbol='BTC/USDT', timeframe='1d', since_months=12):
    """Fetches historical OHLCV data for a given exchange and symbol."""

    # Instantiate the exchange client dynamically
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': os.getenv('CCXT_API_KEY'),
        'secret': os.getenv('CCXT_SECRET'),
        'enableRateLimit': True, # Important for production
    })

    # Calculate start timestamp (in milliseconds)
    since_date = datetime.utcnow() - timedelta(days=30 * since_months)
    since_ms = exchange.parse8601(since_date.isoformat() + 'Z')

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=None)

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # Save raw data
    filename = f"data/raw_{exchange_id}_{symbol.replace('/', '')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename)

    print(f"Successfully fetched {len(df)} candles and saved to {filename}")
    return df

if __name__ == '__main__':
    exchange_id = os.getenv('CCXT_EXCHANGE', 'coinbasepro') # Default to coinbasepro if not set
    fetch_ohlcv_data(exchange_id, since_months=36) # Fetch 3 years of daily data