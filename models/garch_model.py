"""
GARCH MODEL MODULE - Traditional Volatility Forecasting
=======================================================

GARCH = Generalized Autoregressive Conditional Heteroskedasticity

- Models volatility clustering (high vol follows high vol)
- Specifically designed for financial time series
- Fast and interpretable

GARCH(1,1) is the most common specification:
- 1 lag of squared returns (ARCH component)
- 1 lag of past volatility (GARCH component)
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GARCHVolatilityModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    WHAT IS GARCH?
    ==============
    GARCH models volatility clustering - the tendency for:
    - Large price changes to follow large changes
    - Small price changes to follow small changes
        
    THE MODEL:
    σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
    
    Where:
    - σ²(t) = today's volatility (what we're predicting)
    - ω = long-run average volatility (constant)
    - α = how much yesterday's shock matters (ARCH effect)
    - β = how much yesterday's volatility matters (GARCH effect)
    - ε²(t-1) = yesterday's squared return (surprise)
    
    Example:
        model = GARCHVolatilityModel()
        model.train(returns_data)
        forecast = model.forecast(horizon=24)
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.
        
        Args:
            p: GARCH order (how many past volatility lags, usually 1)
            q: ARCH order (how many past return lags, usually 1)
            
        GARCH(1,1) is most common and usually sufficient!
        """
        self.p = p  # GARCH order
        self.q = q  # ARCH order
        self.model = None
        self.result = None
        self.returns_data = None
        
        logger.info(f"📊 Initialized GARCH({p},{q}) model")
    
    def train(
        self,
        returns: pd.Series,
        rescale: bool = True,
        show_summary: bool = True
    ):
        """
        Train the GARCH model on returns data.
        
        WHY USE RETURNS INSTEAD OF PRICES?
        ==================================
        - Returns are stationary (prices are not)
        - Returns have consistent statistical properties
        - GARCH is designed for returns, not prices
        
        Args:
            returns: Log returns series (from feature engineering)
            rescale: Scale returns to percentages (helps convergence)
            show_summary: Whether to print model summary
            
        Example:
            returns = df['log_return'].dropna()
            model.train(returns)
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) < 100:
            raise ValueError("Need at least 100 observations for GARCH")
        
        # Store original returns for later
        self.returns_data = returns.copy()
        
        # Rescale to percentages (helps numerical stability)
        if rescale:
            returns_scaled = returns * 100
        else:
            returns_scaled = returns
        
        logger.info(f"Training GARCH({self.p},{self.q}) on {len(returns)} observations...")
        
        try:
            # Define GARCH model
            # mean='Zero' assumes returns have zero mean (common for crypto)
            # vol='Garch' specifies GARCH volatility model
            # dist='Normal' assumes normally distributed errors
            self.model = arch_model(
                returns_scaled,
                mean='Zero',
                vol='Garch',
                p=self.p,
                q=self.q,
                dist='Normal',
                rescale=False
            )
            
            # Fit the model
            # disp='off' suppresses iteration output
            self.result = self.model.fit(disp='off', show_warning=False)
            
            logger.info("✅ GARCH model trained successfully!")
            
            # Print summary if requested
            if show_summary:
                print("\n" + "=" * 70)
                print("GARCH MODEL SUMMARY")
                print("=" * 70)
                print(self.result.summary())
                print("=" * 70)
                
                # Explain the parameters
                self._explain_parameters()
            
        except Exception as e:
            logger.error(f"❌ Error training GARCH model: {e}")
            raise
    
    def _explain_parameters(self):
        """Explain what the fitted parameters mean."""
        params = self.result.params
        
        print("\n📊 PARAMETER INTERPRETATION:")
        print("-" * 70)
        
        if 'omega' in params:
            omega = params['omega']
            print(f"ω (omega) = {omega:.6f}")
            print("  → Long-run average volatility level")
            print()
        
        if 'alpha[1]' in params:
            alpha = params['alpha[1]']
            print(f"α (alpha) = {alpha:.6f}")
            print("  → Impact of yesterday's shock on today's volatility")
            print(f"  → {alpha:.1%} of yesterday's surprise affects today")
            print()
        
        if 'beta[1]' in params:
            beta = params['beta[1]']
            print(f"β (beta) = {beta:.6f}")
            print("  → Persistence of volatility")
            print(f"  → {beta:.1%} of yesterday's volatility carries to today")
            print()
        
        # Persistence measure
        if 'alpha[1]' in params and 'beta[1]' in params:
            persistence = params['alpha[1]'] + params['beta[1]']
            print(f"Persistence (α + β) = {persistence:.6f}")
            if persistence > 0.99:
                print("  → Very high persistence! Volatility shocks last a long time")
            elif persistence > 0.95:
                print("  → High persistence. Volatility is quite sticky")
            else:
                print("  → Moderate persistence. Volatility mean-reverts")
    
    def forecast(
        self,
        horizon: int = 24,
        method: str = 'analytic'
    ) -> Tuple[float, pd.DataFrame]:
        """
        Forecast future volatility.
        
        Args:
            horizon: Number of steps ahead to forecast (24 = 24 hours)
            method: Forecasting method ('analytic' or 'simulation')
            
        Returns:
            mean_forecast: Average forecasted volatility
            full_forecast: DataFrame with forecast for each horizon
            
        Example:
            # Predict volatility for next 24 hours
            avg_vol, forecast_df = model.forecast(horizon=24)
            print(f"Expected volatility: {avg_vol:.4f}")
        """
        if self.result is None:
            raise ValueError("Model must be trained before forecasting!")
        
        logger.info(f"Forecasting {horizon} steps ahead...")
        
        # Generate forecast
        forecast = self.result.forecast(horizon=horizon, method=method)
        
        # Extract variance forecasts and convert to volatility (std dev)
        variance_forecast = forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back from %
        
        # Calculate mean forecast
        mean_forecast = volatility_forecast.mean()
        
        # Create DataFrame with forecasts
        forecast_df = pd.DataFrame({
            'horizon': range(1, horizon + 1),
            'volatility': volatility_forecast
        })
        
        logger.info(f"✅ Forecast complete. Mean volatility: {mean_forecast:.4f}")
        
        return mean_forecast, forecast_df
    
    def get_fitted_volatility(self) -> pd.Series:
        """
        Get the fitted volatility (in-sample estimates).
        
        This shows how well the model fits historical data.
        
        Returns:
            Series of fitted volatility values
        """
        if self.result is None:
            raise ValueError("Model must be trained first!")
        
        # Get conditional volatility from model
        fitted_vol = self.result.conditional_volatility / 100  # Convert from %
        
        return fitted_vol
    
    def calculate_metrics(self, actual_vol: pd.Series) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            actual_vol: True realized volatility
            
        Returns:
            Dictionary of metrics (MAE, RMSE, etc.)
        """
        fitted_vol = self.get_fitted_volatility()
        
        # Align the series
        common_index = actual_vol.index.intersection(fitted_vol.index)
        actual = actual_vol.loc[common_index]
        fitted = fitted_vol.loc[common_index]
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - fitted))
        rmse = np.sqrt(np.mean((actual - fitted) ** 2))
        mape = np.mean(np.abs((actual - fitted) / actual)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'n_obs': len(actual)
        }
        
        logger.info(f"📊 Model Metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def plot_diagnostics(self, save_path: Optional[str] = None):
        """
        Plot model diagnostics.
        
        Shows:
        1. Standardized residuals (should be random)
        2. Conditional volatility (fitted values)
        3. ACF of squared standardized residuals
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.result is None:
            raise ValueError("Model must be trained first!")
        
        fig = self.result.plot(annualize='D')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Diagnostic plots saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Where to save the model
        """
        if self.result is None:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'result': self.result,
            'p': self.p,
            'q': self.q,
            'returns_data': self.returns_data
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        
        logger.info(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        model_data = joblib.load(filepath)
        
        self.result = model_data['result']
        self.p = model_data['p']
        self.q = model_data['q']
        self.returns_data = model_data['returns_data']
        
        logger.info(f"📂 Model loaded from: {filepath}")


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

def example_usage():
    """
    Example showing how to use the GARCH model.
    
    Run this file directly to test: python src/models/garch_model.py
    """
    print("=" * 70)
    print("GARCH MODEL - EXAMPLE")
    print("=" * 70)
    
    # Import required modules
    import sys
    sys.path.append('.')
    from src.ingest_data import CryptoDataFetcher
    from src.feature_engineering import VolatilityFeatureEngineer
    
    # Step 1: Load data
    print("\n[Step 1] Loading Bitcoin data...")
    fetcher = CryptoDataFetcher('BTC-USD')
    
    try:
        from pathlib import Path
        raw_files = list(Path('data/raw').glob('BTC-USD*.csv'))
        if raw_files:
            latest_file = sorted(raw_files)[-1]
            df = fetcher.load_from_csv(latest_file.name)
        else:
            df = fetcher.fetch_historical_data(days_back=90, interval='1h')
    except:
        df = fetcher.fetch_historical_data(days_back=90, interval='1h')
    
    # Step 2: Calculate returns
    print("\n[Step 2] Calculating returns...")
    engineer = VolatilityFeatureEngineer()
    df = engineer.calculate_log_returns(df)
    df = engineer.calculate_realized_volatility(df, window=24)
    
    returns = df['log_return'].dropna()
    actual_vol = df['realized_vol'].dropna()
    
    print(f"Data: {len(returns)} observations")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # Step 3: Train GARCH model
    print("\n[Step 3] Training GARCH(1,1) model...")
    garch = GARCHVolatilityModel(p=1, q=1)
    garch.train(returns, show_summary=True)
    
    # Step 4: Make forecast
    print("\n[Step 4] Forecasting 24 hours ahead...")
    mean_forecast, forecast_df = garch.forecast(horizon=24)
    
    print(f"\nForecast Results:")
    print(f"  Mean 24h volatility: {mean_forecast:.4f} ({mean_forecast*100:.2f}%)")
    print(f"\nFirst few forecasts:")
    print(forecast_df.head(10))
    
    # Step 5: Calculate metrics
    print("\n[Step 5] Calculating performance metrics...")
    fitted_vol = garch.get_fitted_volatility()
    
    # Align data
    common_idx = actual_vol.index.intersection(fitted_vol.index)
    metrics = garch.calculate_metrics(actual_vol.loc[common_idx])
    
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if key == 'n_obs':
            print(f"  {key}: {value}")
        elif key == 'MAPE':
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.6f}")
    
    # Step 6: Save model
    print("\n[Step 6] Saving model...")
    garch.save_model('data/models/garch_model.pkl')
    
    print("\n" + "=" * 70)
    print("✅ GARCH model training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()