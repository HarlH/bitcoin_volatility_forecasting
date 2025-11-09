"""
LSTM MODEL MODULE - Deep Learning Volatility Forecasting
========================================================

LSTM = Long Short-Term Memory (a type of neural network)

What it does:
- Learns complex patterns from sequential data
- Can use multiple features (not just returns)
- No assumptions about data distribution

LSTM is good at:
- Remembering long-term patterns
- Handling multiple input features
- Capturing non-linear relationships

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMVolatilityModel:
    """
    LSTM neural network for volatility forecasting.
    
    WHAT IS LSTM?
    =============
    LSTM is a type of neural network designed for sequences:
    - It has "memory" to remember past information
    - It can learn which past info is important and which to forget
    - Think of it like a smart assistant that remembers relevant context
    
    ARCHITECTURE:
    Input → LSTM Layer 1 → Dropout → LSTM Layer 2 → Dropout → Dense → Output
    
    WHY THIS STRUCTURE?
    - Multiple LSTM layers: Learn hierarchical patterns
    - Dropout: Prevents overfitting (randomly ignore some neurons)
    - Dense layer: Combines features for final prediction
    
    Example:
        model = LSTMVolatilityModel(lookback=24, lstm_units=64)
        model.train(X, y, epochs=50)
        prediction = model.predict(new_data)
    """
    
    def __init__(
        self,
        lookback: int = 24,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Number of past time steps to use (24 = last 24 hours)
            lstm_units: Number of LSTM units (neurons) in each layer
            dropout_rate: Fraction of neurons to drop (0.2 = drop 20%)
            learning_rate: How fast the model learns
            
        PARAMETER GUIDANCE:
        - lookback: 24 is good for hourly data (1 day of history)
        - lstm_units: 64 is a good starting point (32-128 typical range)
        - dropout_rate: 0.2-0.3 prevents overfitting
        - learning_rate: 0.001 is standard (0.0001-0.01 range)
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        logger.info(
            f"🧠 Initialized LSTM model "
            f"(lookback={lookback}, units={lstm_units}, dropout={dropout_rate})"
        )
    
    def build_model(self, n_features: int):
        """
        Build the LSTM neural network architecture.
        
        LAYER EXPLANATION:
        ==================
        1. LSTM(units, return_sequences=True): First LSTM layer
           - Processes sequence and outputs to next LSTM layer
           - return_sequences=True means output sequence, not just final state
        
        2. Dropout(rate): Randomly drops neurons during training
           - Prevents overfitting (memorizing training data)
           - Like studying with randomness to learn concepts, not just answers
        
        3. LSTM(units//2): Second LSTM layer
           - return_sequences=False means output only final state
           - Fewer units (units//2) creates information bottleneck
        
        4. Dense(1): Output layer
           - Single neuron for single volatility prediction
        
        Args:
            n_features: Number of input features
        """
        logger.info(f"Building LSTM model with {n_features} features...")
        
        self.model = Sequential([
            # First LSTM layer
            LSTM(
                units=self.lstm_units,
                return_sequences=True,
                input_shape=(self.lookback, n_features),
                name='lstm_1'
            ),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer  
            LSTM(
                units=self.lstm_units // 2,
                return_sequences=False,
                name='lstm_2'
            ),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Output layer
            Dense(1, name='output')
        ])
        
        # Compile model
        # - Adam optimizer: Adaptive learning (adjusts learning rate automatically)
        # - MSE loss: Mean Squared Error (standard for regression)
        # - MAE metric: Mean Absolute Error (easier to interpret)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("✅ Model built successfully!")
        
        # Print model summary
        print("\n" + "=" * 70)
        print("LSTM MODEL ARCHITECTURE")
        print("=" * 70)
        self.model.summary()
        print("=" * 70 + "\n")
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data into sequences for LSTM.
        
        WHAT ARE SEQUENCES?
        ===================
        LSTMs need data in sequence format:
        
        Instead of: Single row of features → prediction
        We need: Multiple rows (sequence) → prediction
        
        Example with lookback=3:
        ```
        Sequence 1: [hour1, hour2, hour3] → predict hour4
        Sequence 2: [hour2, hour3, hour4] → predict hour5
        Sequence 3: [hour3, hour4, hour5] → predict hour6
        ```
        
        This sliding window approach gives model temporal context.
        
        Args:
            X: Features array (samples, features)
            y: Target array (samples,)
            
        Returns:
            X_seq: Sequenced features (samples, lookback, features)
            y_seq: Corresponding targets (samples,)
        """
        logger.info(f"Preparing sequences (lookback={self.lookback})...")
        
        # Scale features to 0-1 range (helps neural network training)
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq, y_seq = [], []
        
        # Create sliding windows
        for i in range(len(X_scaled) - self.lookback):
            # Get sequence of past 'lookback' time steps
            X_seq.append(X_scaled[i:i + self.lookback])
            # Get target value at end of sequence
            y_seq.append(y[i + self.lookback])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"✅ Created {len(X_seq)} sequences")
        logger.info(f"   X_seq shape: {X_seq.shape}")
        logger.info(f"   y_seq shape: {y_seq.shape}")
        
        return X_seq, y_seq
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the LSTM model.
        
        TRAINING PROCESS:
        =================
        1. Prepare sequences from data
        2. Split into train/validation sets
        3. Train model in epochs (full passes through data)
        4. Use early stopping (stop if not improving)
        
        Args:
            X: Features (samples, features)
            y: Target values (samples,)
            validation_split: Fraction of data for validation (0.2 = 20%)
            epochs: Maximum training iterations
            batch_size: Samples per gradient update (32 is standard)
            verbose: 0=silent, 1=progress bar, 2=one line per epoch
            
        PARAMETER GUIDANCE:
        - validation_split: 0.2 is standard (20% for validation)
        - epochs: 50-100 typical (early stopping will stop earlier if needed)
        - batch_size: 32 is standard (16-128 range)
        """
        logger.info("🏋️ Starting LSTM training...")
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(n_features=X.shape[1])
        
        # Define callbacks
        # - EarlyStopping: Stop if validation loss doesn't improve
        # - ModelCheckpoint: Save best model during training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Stop if no improvement for 10 epochs
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'data/models/lstm_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("✅ Training completed!")
        
        # Print final metrics
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_mae = self.history.history['mae'][-1]
        final_val_mae = self.history.history['val_mae'][-1]
        
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)
        print(f"Final Training Loss (MSE): {final_loss:.6f}")
        print(f"Final Validation Loss (MSE): {final_val_loss:.6f}")
        print(f"Final Training MAE: {final_mae:.6f}")
        print(f"Final Validation MAE: {final_val_mae:.6f}")
        print("=" * 70 + "\n")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make volatility predictions.
        
        Args:
            X: Features to predict on (samples, features)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction!")
        
        # Prepare sequences
        X_scaled = self.scaler.transform(X)
        
        # For prediction, we need at least lookback samples
        if len(X_scaled) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} samples for prediction")
        
        # Take last lookback samples
        X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, X.shape[1])
        
        # Make prediction
        prediction = self.model.predict(X_seq, verbose=0)
        
        return prediction[0, 0]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Dictionary of metrics
        """
        logger.info("📊 Evaluating model...")
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Get predictions
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Calculate metrics
        mae = np.mean(np.abs(y_seq - predictions))
        rmse = np.sqrt(np.mean((y_seq - predictions) ** 2))
        mape = np.mean(np.abs((y_seq - predictions) / y_seq)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'n_samples': len(y_seq)
        }
        
        logger.info(f"✅ Evaluation complete: MAE={mae:.6f}, RMSE={rmse:.6f}")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Shows loss and MAE over epochs for both training and validation sets.
        """
        if self.history is None:
            raise ValueError("No training history available!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Model Loss Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Model MAE Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training history saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No trained model to save!")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save scaler
        scaler_path = filepath.replace('.keras', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"💾 Model saved to: {filepath}")
        logger.info(f"💾 Scaler saved to: {scaler_path}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        self.model = keras.models.load_model(filepath)
        
        scaler_path = filepath.replace('.keras', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"📂 Model loaded from: {filepath}")


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

def example_usage():
    """
    Example showing how to use the LSTM model.
    
    Run this file directly to test: python src/models/lstm_model.py
    """
    print("=" * 70)
    print("LSTM MODEL - EXAMPLE")
    print("=" * 70)
    
    # Load processed features
    print("\n[Step 1] Loading processed features...")
    from pathlib import Path
    
    # Find most recent feature files
    processed_files = list(Path('data/processed').glob('features_X_*.csv'))
    if not processed_files:
        print("❌ No processed features found!")
        print("Please run feature_engineering.py first")
        return
    
    latest_X = sorted(processed_files)[-1]
    latest_y = str(latest_X).replace('features_X', 'target_y')
    
    X = pd.read_csv(latest_X, index_col=0)
    y = pd.read_csv(latest_y, index_col=0).squeeze()
    
    print(f"Loaded: {latest_X.name}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    print("\n[Step 2] Splitting data...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Train LSTM
    print("\n[Step 3] Training LSTM model...")
    lstm = LSTMVolatilityModel(
        lookback=24,
        lstm_units=64,
        dropout_rate=0.2
    )
    
    lstm.train(
        X_train.values,
        y_train.values,
        epochs=20,  # Use 50+ for real training
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate
    print("\n[Step 4] Evaluating on test set...")
    metrics = lstm.evaluate(X_test.values, y_test.values)
    
    print("\nTest Set Metrics:")
    for key, value in metrics.items():
        if key == 'n_samples':
            print(f"  {key}: {value}")
        elif key == 'MAPE':
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.6f}")
    
    # Plot training history
    print("\n[Step 5] Plotting training history...")
    lstm.plot_training_history('data/models/lstm_training_history.png')
    
    # Save model
    print("\n[Step 6] Saving model...")
    lstm.save_model('data/models/lstm_model.keras')
    
    print("\n" + "=" * 70)
    print("✅ LSTM model training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()