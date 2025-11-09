import pandas as pd
import numpy as np

def calculate_realized_volatility(df):
    """
    Calculates the daily log returns and Realized Volatility (RV).
    RV is typically the standard deviation of returns over a lookback period.

    Args:
        df (pd.DataFrame): DataFrame with 'close' prices indexed by timestamp.

    Returns:
        pd.DataFrame: DataFrame with 'log_return' and 'realized_vol' columns.
    """

    # 1. Calculate Log Returns
    # log_return(t) = log(Price(t) / Price(t-1))
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Calculate Realized Volatility (RV)
    # We'll use a 30-day rolling standard deviation of log returns 
    # as a proxy for the chibui191 guideline's realized volatility calculation.
    LOOKBACK_WINDOW = 30

    # Realized Volatility (RV) is the annualized standard deviation of returns
    # using the lookback window. Annualization factor for daily data is sqrt(252)
    # However, for a volatility time series, we often just use the period std dev.
    # We will use the standard deviation of returns * 100 for a percentage value.
    df['realized_vol'] = df['log_return'].rolling(window=LOOKBACK_WINDOW).std() * 100

    # 3. Drop rows with NaN values created by the shift(1) and rolling window
    df.dropna(inplace=True)

    # Optional: Save processed data
    # For simplicity, we'll return the DataFrame instead of saving it.
    return df[['log_return', 'realized_vol']]


if __name__ == '__main__':
    # --- Example Usage ---
    # 1. Find the latest raw data file (assuming you ran the ingestion script)
    import glob
    latest_file = max(glob.glob('data/raw_*.csv'), key=os.path.getctime)
    print(f"Loading raw data from: {latest_file}")

    # 2. Load the data
    raw_df = pd.read_csv(latest_file, index_col='timestamp', parse_dates=True)

    # 3. Process the data
    processed_df = calculate_realized_volatility(raw_df)

    print("\nProcessed Data Head:")
    print(processed_df.head())
    print(f"\nProcessed Data Shape: {processed_df.shape}")

    # Save the processed data for the training scripts to use
    processed_df.to_csv("data/processed_data.csv")
    print("\nProcessed data saved to data/processed_data.csv")