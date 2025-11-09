# -*- coding: utf-8 -*-
"""
Enhanced Multivariate LSTM Model for Bitcoin Volatility Forecasting

IMPROVEMENTS:
1. Multivariate input (4 features instead of 1)
2. Bidirectional LSTM (learns patterns both forward and backward)
3. RMSPE metric (Root Mean Squared Percentage Error)
4. Baseline model comparisons
5. Proper temporal train-test split (no shuffling!)

ANALOGY: Upgraded from a student with 1 textbook to a genius with 4 textbooks
who can read them both forward and backward!
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# Utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def load_multivariate_data(X_path="data/X_multivariate.npy",
                           y_path="data/y_multivariate.npy"):
    """Load preprocessed multivariate sequences"""

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Multivariate data not found! Run preprocess_data_enhanced.py first.")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"\n>>> Loaded multivariate data:")
    print(f"    X shape: {X.shape} (samples, timesteps, features)")
    print(f"    y shape: {y.shape}")

    return X, y


def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Squared Percentage Error

    ANALOGY: Instead of "you were off by $5", we say "you were off by 10%"
    Percentage errors are better for comparing across different scales!

    Formula: sqrt(mean((y_true - y_pred) / y_true)^2) * 100

    Parameters:
    -----------
    y_true : array
        Actual values
    y_pred : array
        Predicted values

    Returns:
    --------
    RMSPE value (in percentage)
    """

    # Avoid division by zero
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100


def create_baseline_predictions(y_train, y_test):
    """
    Create baseline predictions for comparison

    ANALOGY: Before using advanced AI, try simple guesses:
    1. "Tomorrow will be like today" (Naive)
    2. "Tomorrow will be the average" (Mean)

    These baselines help us know if our fancy model is actually useful!

    Parameters:
    -----------
    y_train : array
        Training targets
    y_test : array
        Test targets

    Returns:
    --------
    Dictionary with baseline predictions and metrics
    """

    print("\n>>> Creating baseline models for comparison...")

    baselines = {}

    # Baseline 1: Mean (predict the training average every time)
    mean_pred = np.full_like(y_test, y_train.mean())
    baselines['mean'] = {
        'predictions': mean_pred,
        'mae': mean_absolute_error(y_test, mean_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, mean_pred)),
        'rmspe': calculate_rmspe(y_test, mean_pred)
    }

    # Baseline 2: Naive (predict last known value)
    naive_pred = np.full_like(y_test, y_train[-1])
    baselines['naive'] = {
        'predictions': naive_pred,
        'mae': mean_absolute_error(y_test, naive_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, naive_pred)),
        'rmspe': calculate_rmspe(y_test, naive_pred)
    }

    print(f"    Mean Baseline  - MAE: {baselines['mean']['mae']:.6f}, RMSPE: {baselines['mean']['rmspe']:.2f}%")
    print(f"    Naive Baseline - MAE: {baselines['naive']['mae']:.6f}, RMSPE: {baselines['naive']['rmspe']:.2f}%")

    return baselines


def build_multivariate_bidirectional_lstm(n_timesteps=30, n_features=4):
    """
    Build enhanced Bidirectional LSTM architecture

    ANALOGY: Like a detective who investigates a case by:
    1. Reading evidence forward (what happened before leads to what?)
    2. Reading evidence backward (what happened after explains what?)
    Then combining both perspectives!

    Architecture based on best practices from research:
    - Bidirectional LSTM layers (learn patterns both ways)
    - Dropout for regularization (prevent memorization)
    - Smaller layer sizes (32, 16) to prevent overfitting

    Parameters:
    -----------
    n_timesteps : int
        Lookback window (default: 30 days)
    n_features : int
        Number of input features (default: 4)

    Returns:
    --------
    Compiled Keras model
    """

    print("\n>>> Building Enhanced Multivariate Bidirectional LSTM...")
    print("    Architecture:")
    print("      Layer 1: Bidirectional LSTM (32 units forward + 32 backward = 64 total)")
    print("      Layer 2: Dropout (10% to prevent overfitting)")
    print("      Layer 3: Bidirectional LSTM (16 units forward + 16 backward = 32 total)")
    print("      Layer 4: Dropout (10%)")
    print("      Layer 5: Dense (16 neurons)")
    print("      Layer 6: Output (1 prediction)")

    model = models.Sequential([
        # First Bidirectional LSTM layer
        # Learns patterns by reading sequence both forward and backward
        layers.Bidirectional(
            layers.LSTM(32, activation='relu', return_sequences=True),
            input_shape=(n_timesteps, n_features)
        ),

        # Dropout - randomly ignore 10% of connections during training
        # ANALOGY: Like not relying on the same study notes every time
        layers.Dropout(0.1),

        # Second Bidirectional LSTM layer (no return_sequences = output only final state)
        layers.Bidirectional(
            layers.LSTM(16, activation='relu')
        ),

        # Another dropout
        layers.Dropout(0.1),

        # Dense layer - combines learned patterns
        layers.Dense(16, activation='relu'),

        # Output layer - single prediction (7-day average volatility)
        layers.Dense(1)
    ])

    # Compile with Adam optimizer and MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    print("\n>>> Model Summary:")
    model.summary()

    return model


def train_enhanced_lstm(X, y, test_size=0.2, epochs=100, batch_size=32):
    """
    Train enhanced multivariate LSTM model

    ANALOGY: Sending the AI detective to school with 4 textbooks.
    Study sessions (epochs) continue until performance stops improving.

    Parameters:
    -----------
    X : numpy array
        Input features (samples, timesteps, features)
    y : numpy array
        Target values
    test_size : float
        Fraction for testing (default: 0.2 = 20%)
    epochs : int
        Maximum training cycles
    batch_size : int
        Samples per training batch

    Returns:
    --------
    model, scaler, history, test results
    """

    print("\n" + "=" * 70)
    print(" " * 10 + "ENHANCED MULTIVARIATE BIDIRECTIONAL LSTM")
    print("=" * 70)

    print("\n>>> What makes this LSTM special?")
    print("    1. MULTIVARIATE: Uses 4 features (not just 1!)")
    print("       - Realized Volatility")
    print("       - High-Low Spread")
    print("       - Open-Close Spread")
    print("       - Log Volume")
    print("    2. BIDIRECTIONAL: Reads patterns forward AND backward")
    print("    3. REGULARIZED: Dropout layers prevent overfitting")
    print("    4. OPTIMIZED: Architecture based on financial research")

    # Normalize features (LSTM works better with values between 0 and 1)
    print("\n>>> Normalizing features...")
    scaler = MinMaxScaler()

    # Reshape for scaling: (samples * timesteps, features)
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

    # Also scale targets
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split (NO SHUFFLING - respect time order!)
    print(f"\n>>> Splitting data (test_size={test_size})...")
    print("    IMPORTANT: NO shuffling (time series must stay in order!)")

    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_scaled[:split_idx]
    y_test = y_scaled[split_idx:]

    print(f"    Train: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"    Test:  {len(X_test)} samples ({test_size*100:.0f}%)")

    # Create baseline predictions
    baselines = create_baseline_predictions(y, y[split_idx:])

    # Build model
    model = build_multivariate_bidirectional_lstm(
        n_timesteps=n_timesteps,
        n_features=n_features
    )

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Stop if no improvement for 15 epochs
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce learning rate by half
        patience=7,
        min_lr=0.00001,
        verbose=1
    )

    print(f"\n>>> Training model...")
    print(f"    Max epochs: {epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Early stopping: patience=15")
    print("    This may take a few minutes...\n")

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Predictions
    print("\n>>> Evaluating model...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform to original scale
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    rmspe = calculate_rmspe(y_test_original, y_pred_original)

    print("\n" + "=" * 70)
    print(">>> PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nBaseline Models:")
    print(f"  Mean Baseline:  MAE={baselines['mean']['mae']:.6f}, RMSPE={baselines['mean']['rmspe']:.2f}%")
    print(f"  Naive Baseline: MAE={baselines['naive']['mae']:.6f}, RMSPE={baselines['naive']['rmspe']:.2f}%")

    print(f"\nEnhanced LSTM:")
    print(f"  MAE:   {mae:.6f}")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  RMSPE: {rmspe:.2f}%")

    # Calculate improvement
    improvement_vs_mean = ((baselines['mean']['rmspe'] - rmspe) / baselines['mean']['rmspe']) * 100
    print(f"\n  Improvement vs Mean Baseline: {improvement_vs_mean:.2f}%")

    print("=" * 70 + "\n")

    return model, (scaler, y_scaler), history, (y_test_original, y_pred_original), baselines


def save_enhanced_lstm(model, scalers, save_dir="models"):
    """Save trained model and scalers"""
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, 'lstm_multivariate_model.h5'))
    joblib.dump(scalers, os.path.join(save_dir, 'lstm_scalers.pkl'))

    print(f"\n>>> Model saved to: {save_dir}/lstm_multivariate_model.h5")
    print(f">>> Scalers saved to: {save_dir}/lstm_scalers.pkl")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 10 + "ENHANCED LSTM VOLATILITY FORECASTING")
    print(" " * 15 + "Multivariate Deep Learning")
    print("=" * 70)

    # Load data
    X, y = load_multivariate_data()

    # Train model
    model, scalers, history, (y_test, y_pred), baselines = train_enhanced_lstm(
        X, y,
        test_size=0.2,
        epochs=100,
        batch_size=32
    )

    # Save model
    save_enhanced_lstm(model, scalers)

    # Save predictions
    results_df = pd.DataFrame({
        'Actual_RV': y_test,
        'LSTM_Prediction': y_pred,
        'Mean_Baseline': baselines['mean']['predictions'],
        'Naive_Baseline': baselines['naive']['predictions']
    })
    results_df.to_csv('models/lstm_enhanced_predictions.csv', index=False)
    print(">>> Predictions saved to: models/lstm_enhanced_predictions.csv")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/lstm_training_history.csv', index=False)
    print(">>> Training history saved to: models/lstm_training_history.csv")

    print("\n" + "=" * 70)
    print("    Enhanced LSTM training complete!")
    print("    NEXT: Build Streamlit dashboard for visualization")
    print("=" * 70 + "\n")
