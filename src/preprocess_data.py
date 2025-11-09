# -*- coding: utf-8 -*-
"""
Data Preprocessing for Bitcoin Volatility Forecasting

ANALOGY: This is like a chef preparing ingredients before cooking.
We take raw price data and transform it into "Realized Volatility" - our main ingredient!

What this does:
1. Loads raw Bitcoin price data
2. Calculates daily returns (how much price changed each day)
3. Calculates Realized Volatility (how "wild" the price swings are)
4. Prepares data for machine learning models
"""

import pandas as pd
import numpy as np
import os


def calculate_returns(df, price_col='Close'):
    """
    Calculate daily returns from price data

    ANALOGY: If temperature was 70F yesterday and 75F today,
    the "return" is (75-70)/70 = 7.14% increase

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with price data
    price_col : str
        Which column to use for calculations (default: 'Close')

    Returns:
    --------
    pandas Series with daily returns
    """

    # Calculate percentage change from one day to the next
    # Formula: (Today - Yesterday) / Yesterday
    returns = df[price_col].pct_change()

    # The first value will be NaN (no yesterday to compare to)
    # So we drop it
    returns = returns.dropna()

    return returns


def calculate_realized_volatility(returns, window=1):
    """
    Calculate Realized Volatility from returns

    ANALOGY: Measuring how bumpy a road is.
    We look at all the bumps (price changes) in a window and calculate
    the average "bumpiness"

    Parameters:
    -----------
    returns : pandas Series
        Daily returns
    window : int
        How many days to look back (default: 1 = daily volatility)

    Returns:
    --------
    pandas Series with realized volatility
    """

    # Realized Volatility = Standard deviation of returns
    # We use rolling window to calculate it over time

    if window == 1:
        # Daily volatility = absolute value of daily returns
        # (simplified version for single-day)
        rv = np.abs(returns)
    else:
        # Rolling standard deviation
        rv = returns.rolling(window=window).std()

    return rv


def prepare_volatility_data(raw_data_path="data/btc_usd_raw.csv",
                            output_path="data/btc_volatility.csv",
                            window=7):
    """
    Full preprocessing pipeline: Load data -> Calculate returns -> Calculate RV

    ANALOGY: This is the complete "prep kitchen" process:
    1. Get ingredients from storage (load data)
    2. Wash and cut them (calculate returns)
    3. Prepare the final dish (calculate volatility)

    Parameters:
    -----------
    raw_data_path : str
        Path to raw Bitcoin price data CSV
    output_path : str
        Where to save the processed volatility data
    window : int
        Rolling window for volatility calculation (default: 7 days)

    Returns:
    --------
    pandas DataFrame with date, price, returns, and realized volatility
    """

    print("\n" + "=" * 70)
    print(" " * 20 + "DATA PREPROCESSING")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n>>> Step 1: Loading raw Bitcoin data...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found! Run ingest_data.py first.")

    df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    print(f"    Loaded {len(df)} days of price data")

    # Step 2: Calculate returns
    print("\n>>> Step 2: Calculating daily returns...")
    print("    Formula: Return = (Today's Price - Yesterday's Price) / Yesterday's Price")
    returns = calculate_returns(df, price_col='Close')
    print(f"    Calculated returns for {len(returns)} days")
    print(f"    Average daily return: {returns.mean():.4f} ({returns.mean()*100:.2f}%)")
    print(f"    Daily return std dev: {returns.std():.4f} ({returns.std()*100:.2f}%)")

    # Step 3: Calculate Realized Volatility
    print(f"\n>>> Step 3: Calculating Realized Volatility (RV)...")
    print(f"    Using {window}-day rolling window")
    rv = calculate_realized_volatility(returns, window=window)
    print(f"    Calculated RV for {len(rv.dropna())} days")
    print(f"    Average RV: {rv.mean():.4f}")
    print(f"    Max RV: {rv.max():.4f} (most volatile day!)")
    print(f"    Min RV: {rv.min():.4f} (calmest day!)")

    # Step 4: Combine everything into one DataFrame
    print("\n>>> Step 4: Combining all features...")
    result_df = pd.DataFrame({
        'Price': df['Close'],
        'Returns': returns,
        'Realized_Volatility': rv
    })

    # Drop rows with NaN values (from rolling calculations)
    result_df = result_df.dropna()
    print(f"    Final dataset: {len(result_df)} days")

    # Step 5: Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path)
    print(f"\n>>> Saved processed data to: {output_path}")

    # Show preview
    print("\n>>> Data Preview:")
    print(result_df.head(10))

    print("\n>>> Statistical Summary:")
    print(result_df.describe())

    print("\n" + "=" * 70)
    print("    Preprocessing complete!")
    print("    Next step: Train GARCH and LSTM models on this data")
    print("=" * 70 + "\n")

    return result_df


def create_features_for_forecasting(df, lookback=30):
    """
    Create lagged features for forecasting

    ANALOGY: Like a weather forecaster looking at the past 30 days
    to predict tomorrow. We create "features" = things the model can learn from

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe with Realized_Volatility column
    lookback : int
        How many days of history to use as features (default: 30)

    Returns:
    --------
    X (features), y (target) for machine learning
    """

    print(f"\n>>> Creating features for forecasting...")
    print(f"    Lookback window: {lookback} days")
    print(f"    Target: Next 7 days average RV")

    # Create lagged features (past volatility values)
    X_list = []
    y_list = []

    rv = df['Realized_Volatility'].values

    # For each point in time, create a feature vector
    for i in range(lookback, len(rv) - 7):
        # Features: past 'lookback' days of RV
        X_list.append(rv[i-lookback:i])

        # Target: average RV for next 7 days
        y_list.append(np.mean(rv[i:i+7]))

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"    Created {len(X)} training samples")
    print(f"    Feature shape: {X.shape} (samples, lookback_days)")
    print(f"    Target shape: {y.shape}")

    return X, y


# Main execution
if __name__ == "__main__":
    # Run the full preprocessing pipeline
    df = prepare_volatility_data(
        raw_data_path="data/btc_usd_raw.csv",
        output_path="data/btc_volatility.csv",
        window=7  # 7-day rolling volatility
    )

    # Create features for machine learning
    X, y = create_features_for_forecasting(df, lookback=30)

    print("\n>>> Sample features (first 5):")
    print(X[:5])
    print("\n>>> Sample targets (first 5):")
    print(y[:5])
