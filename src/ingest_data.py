# -*- coding: utf-8 -*-
"""
Data Ingestion for Bitcoin Volatility Forecasting

ANALOGY: This is like your weather station that collects temperature data every day.
We're downloading Bitcoin price history so we can measure how "wild" the price swings are.

What this does:
1. Downloads BTC-USD price data from Yahoo Finance
2. Gets daily data for the past 2-3 years
3. Saves it to a CSV file for later use
"""

import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
import os
import numpy as np


def fetch_btc_data(years_back=2, save_path="data/btc_usd_raw.csv"):
    """
    Fetch Bitcoin (BTC-USD) price data from Yahoo Finance

    ANALOGY: Like asking the weather bureau for "give me temperature readings
    for the past 2 years, measured every day"

    Parameters:
    -----------
    years_back : int
        How many years of historical data to fetch (default: 2)
        Think: "How many years of weather history do I need?"

    save_path : str
        Where to save the downloaded data
        Think: "What file cabinet drawer to store this in?"

    Returns:
    --------
    pandas DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    print(f"\n>>> Downloading BTC-USD data...")
    print(f"    From: {start_date.date()} to {end_date.date()}")
    print(f"    That's {years_back} years of Bitcoin price history!")

    try:
        # Download the data using pandas_datareader
        # This connects to Yahoo Finance and gets the data
        print("    Connecting to Yahoo Finance...")
        df = pdr.get_data_yahoo('BTC-USD', start=start_date, end=end_date)

        # Check if we got data
        if df.empty:
            raise ValueError("No data received! Check your internet connection.")

        print(f"    SUCCESS! Downloaded {len(df)} daily price points")
        print(f"    Date range: {df.index[0].date()} to {df.index[-1].date()}")

    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        print("    Creating sample data for demonstration...")
        # Fallback: create sample data
        df = _create_sample_data(years_back)

    # Create directory if it doesn't exist (like creating a folder)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV file (like saving to a spreadsheet)
    df.to_csv(save_path)
    print(f"    Saved to: {save_path}\n")

    return df


def _create_sample_data(years_back=2):
    """
    Create sample Bitcoin data for demonstration

    ANALOGY: If we can't get real weather data, generate realistic fake data
    based on known patterns (hot in summer, cold in winter)
    """
    print("    Generating realistic sample Bitcoin data...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Simulate Bitcoin price with random walk
    # Starting price around $30,000
    np.random.seed(42)  # For reproducibility
    n_days = len(dates)

    # Simulate realistic Bitcoin prices with volatility
    returns = np.random.normal(0.001, 0.03, n_days)  # Daily returns
    prices = 30000 * np.exp(np.cumsum(returns))  # Geometric Brownian motion

    # Create OHLC data (Open, High, Low, Close)
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.02, 0.02, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.05, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.05, 0, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e9, 5e9, n_days),
        'Adj Close': prices
    }, index=dates)

    print(f"    Created {len(df)} days of sample data")
    return df


def load_btc_data(file_path="data/btc_usd_raw.csv"):
    """
    Load previously downloaded Bitcoin data from CSV

    ANALOGY: Instead of calling the weather bureau again, just read from
    the file we saved before (faster!)

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pandas DataFrame with Bitcoin price data
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}\nRun fetch_btc_data() first!")

    print(f"\n>>> Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"    Loaded {len(df)} rows\n")

    return df


# Main execution - runs when you call this script directly
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 20 + "BITCOIN DATA INGESTION")
    print("=" * 70)

    # Fetch and save data
    df = fetch_btc_data(years_back=2, save_path="data/btc_usd_raw.csv")

    # Show a preview (like peeking at the first few rows of a spreadsheet)
    print(">>> Data Preview (first 5 rows):")
    print(df.head())

    print("\n>>> Data Preview (last 5 rows):")
    print(df.tail())

    # Show summary statistics
    print("\n>>> Summary Statistics:")
    print(df.describe())

    print("\n" + "=" * 70)
    print("    Data ingestion complete!")
    print("    Next step: Calculate Realized Volatility from this data")
    print("=" * 70 + "\n")
