# -*- coding: utf-8 -*-
"""
LSTM Model Training for Bitcoin Volatility Forecasting

ANALOGY: Training an AI supercomputer to recognize complex patterns
in volatility that humans might miss.

LSTM = Long Short-Term Memory
Think: A robot with a good memory that can learn from the past
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def load_volatility_data(file_path="data/btc_volatility.csv"):
    """Load preprocessed volatility data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found! Run preprocess_data.py first.")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def create_lstm_dataset(data, lookback=30, forecast_horizon=7):
    """
    Create dataset for LSTM training

    ANALOGY: Creating flashcards for the AI to study.
    Each flashcard shows 30 days of history and asks:
    "What will the next 7 days look like?"

    Parameters:
    -----------
    data : array
        Realized Volatility values
    lookback : int
        How many days of history to look at (default: 30)
    forecast_horizon : int
        How many days ahead to predict (default: 7)

    Returns:
    --------
    X (features), y (targets)
    """

    X, y = [], []

    for i in range(lookback, len(data) - forecast_horizon):
        # Features: past 'lookback' days
        X.append(data[i-lookback:i])

        # Target: average volatility for next 'forecast_horizon' days
        y.append(np.mean(data[i:i+forecast_horizon]))

    return np.array(X), np.array(y)


def build_lstm_model(lookback=30):
    """
    Build LSTM neural network architecture

    ANALOGY: Designing the AI's "brain"
    - Input layer: Receives 30 days of history
    - LSTM layers: The "memory" cells that remember patterns
    - Dense layers: The "decision maker" that outputs prediction

    Parameters:
    -----------
    lookback : int
        Number of time steps to look back

    Returns:
    --------
    Compiled Keras model
    """

    print("\n>>> Building LSTM Neural Network...")
    print("    Architecture:")

    model = models.Sequential([
        # First LSTM layer - learns high-level patterns
        # 64 units = 64 "memory cells"
        layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
        print("      Layer 1: LSTM with 64 memory cells (learns patterns)") or None,

        # Dropout for regularization (prevents overfitting)
        # ANALOGY: Like not memorizing the textbook word-for-word
        layers.Dropout(0.2),
        print("      Layer 2: Dropout 20% (prevents memorization)") or None,

        # Second LSTM layer - learns deeper patterns
        layers.LSTM(32, activation='relu'),
        print("      Layer 3: LSTM with 32 memory cells (deeper patterns)") or None,

        # Dropout again
        layers.Dropout(0.2),
        print("      Layer 4: Dropout 20%") or None,

        # Dense layer - combines everything
        layers.Dense(16, activation='relu'),
        print("      Layer 5: Dense layer with 16 neurons") or None,

        # Output layer - single prediction (average 7-day volatility)
        layers.Dense(1)
        print("      Layer 6: Output (1 prediction value)") or None,
    ])

    # Remove None entries
    model = models.Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    # Compile model
    # ANALOGY: Teaching the AI how to learn
    # - Optimizer (Adam): The learning strategy
    # - Loss (MSE): How to measure mistakes
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    print("\n>>> Model Summary:")
    model.summary()

    return model


def train_lstm_model(df, lookback=30, forecast_horizon=7, epochs=50, batch_size=32):
    """
    Train LSTM model for volatility forecasting

    ANALOGY: Like sending the AI to school.
    It studies flashcards (epochs) over and over until it gets good at predictions

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with Realized_Volatility column
    lookback : int
        Days of history to use
    forecast_horizon : int
        Days ahead to forecast
    epochs : int
        How many times to study the entire dataset
    batch_size : int
        How many examples to learn from at once

    Returns:
    --------
    Trained model, scaler, history
    """

    print("\n" + "=" * 70)
    print(" " * 15 + "TRAINING LSTM MODEL")
    print("=" * 70)

    print("\n>>> What is LSTM?")
    print("    LSTM = Long Short-Term Memory Neural Network")
    print("")
    print("    Think of it as: 'The AI Supercomputer'")
    print("    - Learns complex patterns automatically")
    print("    - Has 'memory' to remember important past events")
    print("    - Can capture non-linear relationships")
    print("    - State-of-the-art for time series forecasting")

    # Extract volatility data
    rv_data = df['Realized_Volatility'].values

    # Normalize data (LSTM works better with values between 0 and 1)
    # ANALOGY: Converting all measurements to the same scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    rv_scaled = scaler.fit_transform(rv_data.reshape(-1, 1))

    print(f"\n>>> Preparing data...")
    print(f"    Lookback window: {lookback} days")
    print(f"    Forecast horizon: {forecast_horizon} days")

    # Create dataset
    X, y = create_lstm_dataset(rv_scaled, lookback, forecast_horizon)

    print(f"    Created {len(X)} training samples")
    print(f"    Feature shape: {X.shape}")

    # Reshape for LSTM (needs 3D: samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
    )

    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")

    # Build model
    model = build_lstm_model(lookback)

    # Early stopping (stops training if not improving)
    # ANALOGY: "If you're not getting better after 10 study sessions, take a break"
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print(f"\n>>> Training model...")
    print(f"    Epochs: {epochs} (training cycles)")
    print(f"    Batch size: {batch_size}")
    print("    This may take a few minutes...")

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    print("\n>>> Evaluating model...")
    y_pred = model.predict(X_test)

    # Inverse transform (convert back from 0-1 scale to original scale)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)

    print(f"\n>>> LSTM Performance Metrics:")
    print(f"    MAE (Mean Absolute Error): {mae:.6f}")
    print(f"    RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"    Interpretation: On average, off by {mae*100:.2f}% volatility")

    print("\n" + "=" * 70)
    print("    LSTM model training complete!")
    print("=" * 70 + "\n")

    return model, scaler, history, (y_test_original, y_pred_original)


def save_lstm_model(model, scaler, save_dir="models"):
    """Save trained LSTM model and scaler"""
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(save_dir, 'lstm_scaler.pkl'))

    print(f"\n>>> LSTM model saved to: {save_dir}/lstm_model.h5")
    print(f">>> Scaler saved to: {save_dir}/lstm_scaler.pkl")


# Main execution
if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        exit(1)

    print("\n" + "=" * 70)
    print(" " * 15 + "LSTM VOLATILITY FORECASTING")
    print(" " * 20 + "Deep Learning Pipeline")
    print("=" * 70)

    # Load data
    print("\n>>> Loading preprocessed data...")
    df = load_volatility_data("data/btc_volatility.csv")
    print(f"    Loaded {len(df)} days of data")

    # Train LSTM
    model, scaler, history, (y_test, y_pred) = train_lstm_model(
        df,
        lookback=30,
        forecast_horizon=7,
        epochs=50,
        batch_size=32
    )

    # Save model
    save_lstm_model(model, scaler)

    # Save predictions
    results_df = pd.DataFrame({
        'Actual_RV': y_test.flatten(),
        'LSTM_Prediction': y_pred.flatten()
    })
    results_df.to_csv('models/lstm_predictions.csv', index=False)
    print(">>> Predictions saved to: models/lstm_predictions.csv")

    print("\n" + "=" * 70)
    print("    All models trained successfully!")
    print("    NEXT: Build Streamlit dashboard to visualize predictions")
    print("=" * 70 + "\n")
