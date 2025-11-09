# -*- coding: utf-8 -*-
"""
Model Training for Bitcoin Volatility Forecasting

This script trains TWO different models:
1. GARCH - The experienced meteorologist (traditional approach)
2. LSTM - The AI supercomputer (deep learning approach)

Then we compare their forecasts!
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# GARCH modeling
from arch import arch_model

# LSTM modeling
# Note: We'll handle TensorFlow separately since it's large

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def load_volatility_data(file_path="data/btc_volatility.csv"):
    """Load preprocessed volatility data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found! Run preprocess_data.py first.")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def train_garch_model(returns, forecast_horizon=7):
    """
    Train a GARCH model for volatility forecasting

    ANALOGY: Training the experienced fisherman to predict storms.
    He learns patterns like "big waves yesterday = big waves today"

    GARCH captures "volatility clustering" - the tendency for high volatility
    periods to cluster together (like storms coming in groups)

    Parameters:
    -----------
    returns : pandas Series
        Daily returns data
    forecast_horizon : int
        How many days ahead to forecast (default: 7)

    Returns:
    --------
    Trained GARCH model and forecast
    """

    print("\n" + "=" * 70)
    print(" " * 15 + "TRAINING GARCH MODEL")
    print("=" * 70)

    print("\n>>> What is GARCH?")
    print("    GARCH = Generalized AutoRegressive Conditional Heteroskedasticity")
    print("    (Don't worry about the name!)")
    print("")
    print("    Think of it as: 'The Experienced Storm Tracker'")
    print("    - Uses recent volatility to predict future volatility")
    print("    - Captures 'volatility clustering' (wild days follow wild days)")
    print("    - Industry standard for financial volatility forecasting")

    # Convert returns to percentage (GARCH works better with this)
    returns_pct = returns * 100

    print(f"\n>>> Training GARCH(1,1) model...")
    print("    (1,1) means: use 1 lag of volatility + 1 lag of squared returns")

    # Define GARCH(1,1) model
    # p=1, q=1 are the standard settings that work well for most financial data
    model = arch_model(returns_pct, vol='Garch', p=1, q=1, rescale=False)

    # Fit the model
    print("    Fitting model to data...")
    model_fit = model.fit(disp='off')  # disp='off' = don't show iteration details

    print("    Model fitted successfully!")
    print("\n>>> Model Summary:")
    print(model_fit.summary())

    # Make forecast
    print(f"\n>>> Generating {forecast_horizon}-day ahead forecast...")
    forecast = model_fit.forecast(horizon=forecast_horizon)

    # Extract volatility forecast
    # GARCH predicts variance, so we take square root to get volatility
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])

    print(f"    7-Day Volatility Forecast (in %):")
    for i, vol in enumerate(volatility_forecast, 1):
        print(f"      Day {i}: {vol:.4f}%")

    avg_forecast = np.mean(volatility_forecast) / 100  # Convert back to decimal
    print(f"\n    Average 7-day forecast: {avg_forecast:.4f} ({avg_forecast*100:.2f}%)")

    print("\n" + "=" * 70)
    print("    GARCH model training complete!")
    print("=" * 70 + "\n")

    return model_fit, volatility_forecast / 100  # Return in decimal form


def evaluate_garch_rolling(df, train_size=0.8, forecast_horizon=7):
    """
    Evaluate GARCH using rolling window forecast

    ANALOGY: Testing the fisherman's storm predictions over many days.
    Each day, he makes a prediction, then we check if he was right.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with returns
    train_size : float
        Fraction of data to use for initial training
    forecast_horizon : int
        Days ahead to forecast

    Returns:
    --------
    Dictionary with predictions and actual values
    """

    print("\n>>> Evaluating GARCH with rolling forecasts...")

    returns = df['Returns'] * 100  # Convert to percentage
    rv_actual = df['Realized_Volatility']

    split_idx = int(len(df) * train_size)

    predictions = []
    actuals = []
    dates = []

    # Rolling window evaluation
    print(f"    Generating forecasts for test period...")

    for i in range(split_idx, len(df) - forecast_horizon, forecast_horizon):
        # Train on data up to current point
        train_returns = returns.iloc[:i]

        try:
            # Fit GARCH model
            model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
            model_fit = model.fit(disp='off')

            # Forecast next 7 days
            forecast = model_fit.forecast(horizon=forecast_horizon)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :])

            # Average forecast
            avg_forecast = np.mean(vol_forecast) / 100

            # Actual average volatility for next 7 days
            actual_avg = rv_actual.iloc[i:i+forecast_horizon].mean()

            predictions.append(avg_forecast)
            actuals.append(actual_avg)
            dates.append(df.index[i])

        except:
            continue

    print(f"    Generated {len(predictions)} rolling forecasts")

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"\n>>> GARCH Performance Metrics:")
    print(f"    MAE (Mean Absolute Error):  {mae:.6f}")
    print(f"    RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"    Interpretation: On average, off by {mae*100:.2f}% volatility")

    return {
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates,
        'mae': mae,
        'rmse': rmse
    }


def save_garch_model(model_fit, save_path="models/garch_model.pkl"):
    """Save trained GARCH model to disk"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_fit, save_path)
    print(f"\n>>> GARCH model saved to: {save_path}")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "BITCOIN VOLATILITY FORECASTING")
    print(" " * 20 + "Model Training Pipeline")
    print("=" * 70)

    # Load data
    print("\n>>> Loading preprocessed data...")
    df = load_volatility_data("data/btc_volatility.csv")
    print(f"    Loaded {len(df)} days of data")

    # Train GARCH model on full data
    print("\n>>> Training GARCH model on full dataset...")
    garch_model, garch_forecast = train_garch_model(
        df['Returns'],
        forecast_horizon=7
    )

    # Evaluate GARCH with rolling forecasts
    print("\n>>> Evaluating GARCH model performance...")
    garch_results = evaluate_garch_rolling(
        df,
        train_size=0.8,
        forecast_horizon=7
    )

    # Save results
    save_garch_model(garch_model, "models/garch_model.pkl")

    # Save evaluation results
    results_df = pd.DataFrame({
        'Date': garch_results['dates'],
        'GARCH_Prediction': garch_results['predictions'],
        'Actual_RV': garch_results['actuals']
    })
    results_df.to_csv('models/garch_predictions.csv', index=False)
    print(">>> Predictions saved to: models/garch_predictions.csv")

    print("\n" + "=" * 70)
    print("    GARCH training complete!")
    print(f"    Next 7-day average volatility forecast: {np.mean(garch_forecast):.4f}")
    print("=" * 70 + "\n")

    print("\n>>> NEXT STEP: Train LSTM model (coming next!)")
