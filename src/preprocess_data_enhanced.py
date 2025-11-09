# -*- coding: utf-8 -*-
"""
Enhanced Data Preprocessing for Bitcoin Volatility Forecasting

IMPROVEMENTS based on best practices:
1. Log returns instead of simple returns (better statistical properties)
2. Feature engineering: High-Low spread, Open-Close spread, Volume
3. Multiple features for multivariate LSTM
4. Proper temporal train-test-validation split

ANALOGY: Like a master chef preparing multiple ingredients (not just one!)
to create a complex, delicious dish.
"""

import pandas as pd
import numpy as np
import os


def calculate_log_returns(df, price_col='Close'):
    """
    Calculate log returns from price data

    ANALOGY: Instead of saying "temperature went from 70F to 75F = +7.1%",
    we use logarithms: log(75/70) = 0.069 (more mathematically stable!)

    WHY LOG RETURNS?
    - Better statistical properties (more normally distributed)
    - Time-additive: log return over 2 days = sum of daily log returns
    - Handle large price changes better
    - Standard in financial mathematics

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with price data
    price_col : str
        Which column to use (default: 'Close')

    Returns:
    --------
    pandas Series with log returns
    """

    # Log returns formula: ln(Price_today / Price_yesterday)
    log_returns = np.log(df[price_col] / df[price_col].shift(1))

    # Drop NaN values
    log_returns = log_returns.dropna()

    return log_returns


def engineer_features(df):
    """
    Create engineered features from OHLC data

    ANALOGY: Like extracting different nutrients from the same food.
    Instead of just "price", we extract:
    - How wide the daily range was (High-Low)
    - How much it moved from open to close (Open-Close)
    - How much was traded (Volume)

    These features capture different aspects of market behavior!

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with OHLC + Volume data

    Returns:
    --------
    pandas DataFrame with engineered features
    """

    features = pd.DataFrame(index=df.index)

    # 1. High-Low Spread (intraday volatility indicator)
    # ANALOGY: "How much did the temperature vary during the day?"
    features['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

    # 2. Open-Close Spread (directional movement)
    # ANALOGY: "Did the day end warmer or colder than it started?"
    features['OC_Spread'] = (df['Close'] - df['Open']) / df['Open']

    # 3. Log Volume (trading activity)
    # ANALOGY: "How many people were outside today?" (market participation)
    # Use log to normalize (volumes can have huge ranges)
    features['Log_Volume'] = np.log(df['Volume'] + 1)  # +1 to avoid log(0)

    # 4. Price (Close price)
    features['Close'] = df['Close']

    return features


def calculate_realized_volatility_rolling(log_returns, window=30):
    """
    Calculate Realized Volatility using rolling window

    ANALOGY: Instead of looking at just today's bumpiness, we look at
    the past 30 days to get a "smoothed" average bumpiness.

    Parameters:
    -----------
    log_returns : pandas Series
        Log returns series
    window : int
        Rolling window size (default: 30 days, as used in research)

    Returns:
    --------
    pandas Series with realized volatility
    """

    # Realized Volatility = Rolling standard deviation of log returns
    # This is the industry-standard approach
    rv = log_returns.rolling(window=window).std()

    return rv


def prepare_enhanced_dataset(raw_data_path="data/btc_usd_raw.csv",
                             output_path="data/btc_features_enhanced.csv",
                             rv_window=30):
    """
    Full enhanced preprocessing pipeline

    ANALOGY: The complete kitchen workflow:
    1. Get raw ingredients
    2. Extract multiple nutrients/features
    3. Calculate volatility metric
    4. Package everything together

    Parameters:
    -----------
    raw_data_path : str
        Path to raw Bitcoin OHLC data
    output_path : str
        Where to save processed features
    rv_window : int
        Window for realized volatility calculation (default: 30 days)

    Returns:
    --------
    pandas DataFrame with all features ready for modeling
    """

    print("\n" + "=" * 70)
    print(" " * 15 + "ENHANCED DATA PREPROCESSING")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n>>> Step 1: Loading raw Bitcoin data...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found! Run ingest_data.py first.")

    df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    print(f"    Loaded {len(df)} days of OHLC data")
    print(f"    Columns: {', '.join(df.columns)}")

    # Step 2: Calculate log returns
    print("\n>>> Step 2: Calculating log returns...")
    print("    Formula: ln(Price_today / Price_yesterday)")
    print("    Why? Better statistical properties than simple returns")

    log_returns = calculate_log_returns(df, price_col='Close')
    print(f"    Calculated {len(log_returns)} log returns")
    print(f"    Mean log return: {log_returns.mean():.6f}")
    print(f"    Std dev: {log_returns.std():.6f}")

    # Step 3: Engineer features
    print("\n>>> Step 3: Engineering features from OHLC data...")
    print("    Creating:")
    print("      - High-Low Spread (intraday volatility)")
    print("      - Open-Close Spread (directional movement)")
    print("      - Log Volume (trading activity)")

    features = engineer_features(df)
    print(f"    Created {len(features.columns)} features")

    # Step 4: Calculate Realized Volatility
    print(f"\n>>> Step 4: Calculating Realized Volatility...")
    print(f"    Using {rv_window}-day rolling window")
    print("    Formula: Rolling std dev of log returns")

    rv = calculate_realized_volatility_rolling(log_returns, window=rv_window)
    print(f"    Calculated RV for {len(rv.dropna())} days")
    print(f"    Average RV: {rv.mean():.6f}")
    print(f"    Max RV: {rv.max():.6f} (most volatile period!)")
    print(f"    Min RV: {rv.dropna().min():.6f} (calmest period!)")

    # Step 5: Combine everything
    print("\n>>> Step 5: Combining all features...")

    result_df = pd.DataFrame({
        'Close': features['Close'],
        'Log_Returns': log_returns,
        'Realized_Volatility': rv,
        'HL_Spread': features['HL_Spread'],
        'OC_Spread': features['OC_Spread'],
        'Log_Volume': features['Log_Volume']
    })

    # Drop rows with NaN (from rolling calculations)
    result_df = result_df.dropna()
    print(f"    Final dataset: {len(result_df)} days")
    print(f"    Features: {', '.join(result_df.columns)}")

    # Step 6: Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path)
    print(f"\n>>> Saved enhanced features to: {output_path}")

    # Show preview
    print("\n>>> Data Preview (first 5 rows):")
    print(result_df.head())

    print("\n>>> Statistical Summary:")
    print(result_df.describe())

    print("\n" + "=" * 70)
    print("    Enhanced preprocessing complete!")
    print(f"    Ready for multivariate LSTM training with {len(result_df.columns)} features")
    print("=" * 70 + "\n")

    return result_df


def create_multivariate_sequences(df, feature_cols, target_col='Realized_Volatility',
                                  lookback=30, forecast_horizon=7):
    """
    Create sequences for multivariate LSTM

    ANALOGY: Creating study flashcards with MULTIPLE pieces of information.
    Instead of just "past 30 days of temperature", we include:
    - Temperature
    - Humidity
    - Wind speed
    - Atmospheric pressure
    All together to predict future weather!

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with all features
    feature_cols : list
        List of column names to use as features
    target_col : str
        Target variable to predict
    lookback : int
        How many days to look back (default: 30)
    forecast_horizon : int
        How many days ahead to predict average (default: 7)

    Returns:
    --------
    X (features), y (targets), feature_names
    """

    print(f"\n>>> Creating multivariate sequences...")
    print(f"    Input features: {', '.join(feature_cols)}")
    print(f"    Target: {target_col}")
    print(f"    Lookback: {lookback} days")
    print(f"    Forecast horizon: {forecast_horizon} days")

    X_list = []
    y_list = []

    # Extract feature data
    feature_data = df[feature_cols].values
    target_data = df[target_col].values

    # Create sequences
    for i in range(lookback, len(df) - forecast_horizon):
        # Features: past 'lookback' days of ALL features
        X_list.append(feature_data[i-lookback:i, :])

        # Target: average volatility for next 'forecast_horizon' days
        y_list.append(np.mean(target_data[i:i+forecast_horizon]))

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"    Created {len(X)} training samples")
    print(f"    Input shape: {X.shape} (samples, timesteps, features)")
    print(f"    Target shape: {y.shape}")

    return X, y, feature_cols


# Main execution
if __name__ == "__main__":
    # Run enhanced preprocessing
    df = prepare_enhanced_dataset(
        raw_data_path="data/btc_usd_raw.csv",
        output_path="data/btc_features_enhanced.csv",
        rv_window=30  # 30-day rolling window (industry standard)
    )

    # Create multivariate sequences for LSTM
    feature_cols = ['Realized_Volatility', 'HL_Spread', 'OC_Spread', 'Log_Volume']

    X, y, feature_names = create_multivariate_sequences(
        df,
        feature_cols=feature_cols,
        target_col='Realized_Volatility',
        lookback=30,
        forecast_horizon=7
    )

    print("\n>>> Sample input (first sample, first 3 timesteps):")
    print(X[0, :3, :])
    print(f"\n>>> Feature order: {feature_names}")
    print(f"\n>>> Sample target (first 5): {y[:5]}")

    # Save sequences for later use
    np.save('data/X_multivariate.npy', X)
    np.save('data/y_multivariate.npy', y)
    print("\n>>> Sequences saved to data/ directory")
