"""
FEATURE ENGINEERING MODULE - Crypto Volatility Forecasting
==========================================================

This module transforms raw OHLCV data into features for volatility forecasting.

What it does:
1. Calculates log returns (standardized price changes)
2. Calculates realized volatility (our target variable)
3. Adds technical indicators (volume, momentum, etc.)
4. Creates lagged features (past values for time series)
5. Prepares data for model training
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilityFeatureEngineer:
    """
    Calculates volatility and creates features for forecasting.    
    Example:
        engineer = VolatilityFeatureEngineer()
        df = engineer.calculate_log_returns(df)
        df = engineer.calculate_realized_volatility(df, window=24)
    """
    
    def __init__(self, save_dir: str = 'data/processed'):
        """
        Initialize the feature engineer.
        
        Args:
            save_dir: Directory to save processed features
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized VolatilityFeatureEngineer")
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from close prices.
        
        ================
        Log returns have nice mathematical properties:
        - They're symmetric (10% up, then 10% down gets you back to start)
        - They're additive over time
        - They're more normally distributed 

        FORMULA: log(Price_today / Price_yesterday)
        
        Example:
            Price went from $100 to $110
            Regular return: (110-100)/100 = 10%
            Log return: log(110/100) = 0.0953 ≈ 9.53%
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            DataFrame with added 'log_return' column
        """
        df = df.copy()
        
        # Calculate log returns
        # np.log() is natural logarithm (ln)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # First row will be NaN (no previous price to compare to)
        logger.info(f"✅ Calculated log returns (first value is NaN)")
        
        return df
    
    def calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        window: int = 24,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate realized volatility using rolling standard deviation.
        1. Take log returns over a window (e.g., last 24 hours)
        2. Calculate standard deviation of those returns
        3. Annualize it (scale to yearly volatility)
        
        Args:
            df: DataFrame with 'log_return' column
            window: Rolling window size (24 = 24 hours for hourly data)
            annualize: Whether to annualize the volatility
            
        Returns:
            DataFrame with added 'realized_vol' column
            
        Example:
            If window=24 and interval is 1 hour:
            - Each volatility value uses the last 24 hours of returns
            - For daily data with window=30, uses last 30 days
        """
        df = df.copy()
        
        if 'log_return' not in df.columns:
            raise ValueError("Must calculate log_returns first!")
        
        # Calculate rolling standard deviation
        vol = df['log_return'].rolling(window=window).std()
        
        # Annualize the volatility
        # Multiply by sqrt(window) due tovolatility scales with square root of time
        if annualize:
            vol = vol * np.sqrt(window)
        
        df['realized_vol'] = vol
        
        logger.info(
            f"✅ Calculated realized volatility "
            f"(window={window}, annualized={annualize})"
        )
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        These give the model additional context about price movements:
        - Price momentum (trend direction)
        - High-Low range (intraday volatility proxy)
        - Price changes over different periods
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added price features
        """
        df = df.copy()
        
        # 1. Price momentum (% change over last 24 periods)
        df['price_momentum_24'] = df['Close'].pct_change(periods=24)
        
        # 2. High-Low range as % of close (intraday volatility)
        df['hl_range_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # 3. Close vs Open (direction of candle)
        df['close_open_diff'] = (df['Close'] - df['Open']) / df['Open']
        
        # 4. Rolling max and min over 24 periods
        df['rolling_max_24'] = df['Close'].rolling(window=24).max()
        df['rolling_min_24'] = df['Close'].rolling(window=24).min()
        
        # 5. Distance from rolling max/min
        df['dist_from_max'] = (df['rolling_max_24'] - df['Close']) / df['Close']
        df['dist_from_min'] = (df['Close'] - df['rolling_min_24']) / df['Close']
        
        logger.info("✅ Added price-based features")
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Volume indicates market activity and can signal volatility changes:
        - High volume = lots of trading (often during volatile periods)
        - Low volume = quiet market (typically less volatile)
        
        Args:
            df: DataFrame with 'Volume' column
            
        Returns:
            DataFrame with added volume features
        """
        df = df.copy()
        
        # 1. Volume moving average
        df['volume_ma_24'] = df['Volume'].rolling(window=24).mean()
        
        # 2. Volume standard deviation
        df['volume_std_24'] = df['Volume'].rolling(window=24).std()
        
        # 3. Volume ratio (current vs average)
        # Ratio > 1 means above-average volume
        df['volume_ratio'] = df['Volume'] / df['volume_ma_24']
        
        # 4. Volume change
        df['volume_change'] = df['Volume'].pct_change()
        
        logger.info("✅ Added volume-based features")
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional volatility-based features.
        
        These help the model understand recent volatility patterns:
        - Short-term volatility (last 6 hours)
        - Medium-term volatility (last 12 hours)
        - Volatility changes (is it increasing or decreasing?)
        
        Args:
            df: DataFrame with 'log_return' column
            
        Returns:
            DataFrame with added volatility features
        """
        df = df.copy()
        
        if 'log_return' not in df.columns:
            raise ValueError("Must calculate log_returns first!")
        
        # Calculate volatility at different time horizons
        for window in [6, 12, 24, 48]:
            col_name = f'realized_vol_{window}h'
            df[col_name] = df['log_return'].rolling(window=window).std() * np.sqrt(window)
        
        # Volatility of volatility (how stable is volatility itself?)
        df['vol_of_vol'] = df['realized_vol'].rolling(window=24).std()
        
        # Volatility change (is volatility increasing or decreasing?)
        df['vol_change'] = df['realized_vol'].pct_change()
        
        logger.info("✅ Added volatility-based features")
        
        return df
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'realized_vol',
        lags: List[int] = [1, 2, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting(past values of our target variable).
        
        Example:
            If we want to predict volatility at 3pm, we might use:
            - Volatility at 2pm (lag 1)
            - Volatility at 1pm (lag 2)
            - Volatility at 9am (lag 6)
            - Volatility at 3am (lag 12)
            - Volatility yesterday at 3pm (lag 24)
        
        WHY?
        Time series often have patterns:
        - High volatility now → likely high volatility soon (persistence)
        - Daily/weekly cycles
        
        Args:
            df: DataFrame with target column
            target_col: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with added lagged features
        """
        df = df.copy()
        
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame!")
        
        for lag in lags:
            col_name = f'{target_col}_lag{lag}'
            df[col_name] = df[target_col].shift(lag)
        
        logger.info(f"✅ Created {len(lags)} lagged features for '{target_col}'")
        
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_horizon: int = 24,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final dataset for model training.
        
        This creates X (features) and y (target) for machine learning:
        - X: All features we calculated
        - y: What we want to predict (future volatility)
        
        FORECASTING SETUP:
        ==================
        We want to predict volatility 24 hours ahead.
        
        Current time: 3pm today
        Features (X): Everything up to 3pm today
        Target (y): Volatility at 3pm tomorrow
        
        Args:
            df: DataFrame with all features
            target_horizon: Hours ahead to predict (24 = predict tomorrow)
            drop_na: Whether to remove rows with missing values
            
        Returns:
            X: Features (independent variables)
            y: Target (dependent variable - what we want to predict)
        """
        df = df.copy()
        
        # Create target variable (future volatility)
        # We shift backwards (negative) because we want future values
        df['target_vol'] = df['realized_vol'].shift(-target_horizon)
        
        # Define which columns are features (X)
        # We exclude: target, original OHLCV, and intermediate calculations
        exclude_cols = [
            'target_vol', 'Open', 'High', 'Low', 'Close', 'Volume',
            'rolling_max_24', 'rolling_min_24'  # These are helpers, not direct features
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Drop rows with NaN if requested
        if drop_na:
            df.dropna(inplace=True)
        
        X = df[feature_cols]
        y = df['target_vol']
        
        logger.info(f"✅ Prepared training data:")
        logger.info(f"   Features (X): {X.shape} ({len(feature_cols)} features)")
        logger.info(f"   Target (y): {y.shape}")
        logger.info(f"   Feature columns: {feature_cols}")
        
        return X, y
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        volatility_window: int = 24,
        lags: List[int] = [1, 2, 6, 12, 24],
        save_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete feature engineering pipeline.
        
        This is a convenience function that runs all steps in order:
        1. Calculate returns
        2. Calculate volatility
        3. Add all features
        4. Create lags
        5. Prepare training data
        
        Args:
            df: Raw OHLCV DataFrame
            volatility_window: Window for volatility calculation
            lags: Lag periods to create
            save_features: Whether to save processed data
            
        Returns:
            X: Features
            y: Target variable
        """
        logger.info("🏭 Starting full feature engineering pipeline...")
        
        # Step 1: Calculate returns
        df = self.calculate_log_returns(df)
        
        # Step 2: Calculate realized volatility
        df = self.calculate_realized_volatility(df, window=volatility_window)
        
        # Step 3: Add price features
        df = self.add_price_features(df)
        
        # Step 4: Add volume features
        df = self.add_volume_features(df)
        
        # Step 5: Add additional volatility features
        df = self.add_volatility_features(df)
        
        # Step 6: Create lagged features
        df = self.create_lagged_features(df, target_col='realized_vol', lags=lags)
        
        # Step 7: Prepare training data
        X, y = self.prepare_training_data(df, target_horizon=24)
        
        # Save processed data if requested
        if save_features:
            self._save_processed_data(df, X, y)
        
        logger.info("✅ Feature engineering pipeline completed!")
        
        return X, y
    
    def _save_processed_data(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Save processed data to CSV files."""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d')
        
        # Save full processed dataframe
        df_path = self.save_dir / f'processed_data_{timestamp}.csv'
        df.to_csv(df_path)
        logger.info(f"💾 Saved processed data to: {df_path}")
        
        # Save features
        X_path = self.save_dir / f'features_X_{timestamp}.csv'
        X.to_csv(X_path)
        logger.info(f"💾 Saved features (X) to: {X_path}")
        
        # Save target
        y_path = self.save_dir / f'target_y_{timestamp}.csv'
        y.to_csv(y_path)
        logger.info(f"💾 Saved target (y) to: {y_path}")


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

def example_usage():
    """
    Example showing how to use the VolatilityFeatureEngineer.
    
    Run this file directly to test: python src/feature_engineering.py
    """
    print("=" * 70)
    print("FEATURE ENGINEERING - EXAMPLE")
    print("=" * 70)
    
    # Import data fetcher
    import sys
    sys.path.append('.')
    from src.ingest_data import CryptoDataFetcher
    
    # Step 1: Load some data
    print("\n[Step 1] Loading Bitcoin data...")
    fetcher = CryptoDataFetcher('BTC-USD')
    
    # Try to load existing data, or fetch new data
    try:
        # List files in data/raw
        from pathlib import Path
        raw_files = list(Path('data/raw').glob('BTC-USD*.csv'))
        if raw_files:
            latest_file = sorted(raw_files)[-1]
            df = fetcher.load_from_csv(latest_file.name)
            print(f"Loaded existing data: {latest_file.name}")
        else:
            raise FileNotFoundError
    except:
        print("No existing data found, fetching new data...")
        df = fetcher.fetch_historical_data(days_back=90, interval='1h')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Initialize feature engineer
    print("\n[Step 2] Initializing feature engineer...")
    engineer = VolatilityFeatureEngineer()
    
    # Step 3: Run full pipeline
    print("\n[Step 3] Running feature engineering pipeline...")
    X, y = engineer.full_pipeline(
        df,
        volatility_window=24,
        lags=[1, 2, 6, 12, 24],
        save_features=True
    )
    
    # Step 4: Display results
    print("\n[Step 4] Results:")
    print(f"\nFeatures (X):")
    print(f"  Shape: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    print("\nFirst few rows:")
    print(X.head())
    
    print(f"\nTarget (y):")
    print(f"  Shape: {y.shape}")
    print(f"  Mean volatility: {y.mean():.4f}")
    print(f"  Min volatility: {y.min():.4f}")
    print(f"  Max volatility: {y.max():.4f}")
    
    # Step 5: Summary statistics
    print("\n[Step 5] Feature Summary:")
    print(X.describe())
    
    print("\n" + "=" * 70)
    print("✅ Feature engineering completed successfully!")
    print("=" * 70)
    print(f"\nYou now have {X.shape[0]} training samples with {X.shape[1]} features each!")
    print(f"Ready for model training!")


if __name__ == "__main__":
    example_usage()