from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import sys
from arch import arch_model
from notebooks.logging.logger import logging
from notebooks.exception.Exception import CryptoException
from notebooks.feature_engineering import FeatureEngineering
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


class ModelBaselines:
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.feature_cols = ['past_vol_15', 'volume_avg_30', 'high_low_ratio_5', 'past_return_7']
        self.scaler = None  # Add scaling if needed later

    def training_data(self, df: pd.DataFrame):
        """Split data into train/test sets"""
        try:
            split_date = '2024-01-01'
            # Ensure index is datetime for proper comparison
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            train = df[df.index < split_date].copy()
            test = df[df.index >= split_date].copy()
            
            X_train = train[self.feature_cols]
            y_train = train['target_volatility']
            X_test = test[self.feature_cols]
            y_test = test['target_volatility']
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CryptoException(f"Error in training_data: {e}", sys)

    def Naive_Forecast(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Naive persistence: forecast = past_vol_15"""
        try:
            y_pred_naive = X_test['past_vol_15']
            rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
            logging.info("RMSE Naive Forecast: %.4f", rmse_naive)
            return rmse_naive
        except Exception as e:
            raise CryptoException(f"Error in Naive_Forecast: {e}", sys)

    def XGBoost_Forecast(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate XGBoost model"""
        try:
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            y_pred_xgb = model.predict(X_test)
            rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
            logging.info("RMSE XGBoost Forecast: %.4f", rmse_xgb)
            return rmse_xgb, model
        except Exception as e:
            raise CryptoException(f"Error in XGBoost_Forecast: {e}", sys)

    def GARCH_Forecast(self, df: pd.DataFrame):
        """
        Fit GARCH model on log returns.
        Returns RMSE comparing in-sample conditional volatility to target.
        """
        try:
            # Use log returns for GARCH
            if 'log_return' not in df.columns:
                df = df.copy()
                df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            train_returns = df[df.index <= '2023-12-31']['log_return'].dropna()
            
            if len(train_returns) < 100:
                logging.warning("Insufficient data for GARCH, skipping")
                return np.inf
            
            # ✅ FIX: Use actual returns, not ellipsis (...)
            model = arch_model(train_returns, vol='GARCH', p=1, q=1, rescale=True)
            model_fit = model.fit(disp='off')
            
            # Get in-sample conditional volatility (annualized)
            in_sample_vol = model_fit.conditional_volatility * np.sqrt(252)
            
            # Align with target volatility for comparison
            common_idx = in_sample_vol.index.intersection(df.index)
            if len(common_idx) == 0:
                logging.warning("No overlap for GARCH evaluation")
                return np.inf
                
            y_pred_garch = in_sample_vol.loc[common_idx]
            y_true_garch = df.loc[common_idx, 'target_volatility']
            
            # Remove NaN values
            valid = y_pred_garch.notna() & y_true_garch.notna()
            if valid.sum() < 10:
                logging.warning("Too few valid points for GARCH RMSE")
                return np.inf
                
            rmse_garch = np.sqrt(mean_squared_error(y_true_garch[valid], y_pred_garch[valid]))
            logging.info("RMSE GARCH Forecast: %.4f", rmse_garch)
            return rmse_garch, model_fit
            
        except Exception as e:
            logging.error(f"GARCH error: {e}")
            return np.inf, None

    def Initiate_ModelBaselines(self):
        """Main method: train all models and return artifacts for saving"""
        try:
            # 1. Get engineered data
            df = self.feature_engineering.initiate_feature_engineering()
            if df is None or df.empty:
                raise ValueError("Feature engineering returned empty DataFrame")
            
            # 2. Prepare train/test splits
            X_train, y_train, X_test, y_test = self.training_data(df)
            logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # 3. Train Naive (no model object needed)
            logging.info(" Training Naive Persistence...")
            rmse_naive = self.Naive_Forecast(X_test, y_test)
            
            # 4. Train XGBoost
            logging.info(" Training XGBoost...")
            rmse_xgb, xgb_model = self.XGBoost_Forecast(X_train, y_train, X_test, y_test)
            
            # 5. Train GARCH
            logging.info(" Training GARCH...")
            rmse_garch, garch_model_fit = self.GARCH_Forecast(df)
            
            # 6. Plot Feature Importance (XGBoost)
            try:
                feat_imp = pd.Series(xgb_model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
                plt.figure(figsize=(8, 5))
                feat_imp.plot(kind='barh')
                plt.title('XGBoost Feature Importances')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                
                # Save plot instead of showing (better for headless/server environments)
                plot_path = Path('artifacts/feature_importance.png')
                plot_path.parent.mkdir(exist_ok=True)
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logging.info(f" Saved feature importance plot to {plot_path}")
            except Exception as plot_err:
                logging.warning(f"Could not save feature importance plot: {plot_err}")
            
            # 7. Plot RMSE Comparison
            try:
                models = ['Naive', 'XGBoost', 'GARCH']
                rmse_values = [rmse_naive, rmse_xgb, rmse_garch if np.isfinite(rmse_garch) else 999]
                
                plt.figure(figsize=(8, 5))
                bars = plt.bar(models, rmse_values, color=['#28a745', '#007bff', '#6f42c1'])
                plt.title('Model Comparison: RMSE (Lower is Better)')
                plt.ylabel('RMSE')
                
                # Add value labels on bars
                for bar, val in zip(bars, rmse_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plot_path = Path('artifacts/rmse_comparison.png')
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logging.info(f" Saved RMSE comparison plot to {plot_path}")
            except Exception as plot_err:
                logging.warning(f"Could not save RMSE plot: {plot_err}")
            
            # 8. Print summary
            print(f"\n Baselines Summary:")
            print(f"  Naive:    {rmse_naive:.4f} {'✅ Best' if rmse_naive <= min(rmse_xgb, rmse_garch) else ''}")
            print(f"  XGBoost:  {rmse_xgb:.4f} {'✅ Best' if rmse_xgb <= min(rmse_naive, rmse_garch) else ''}")
            print(f"  GARCH:    {rmse_garch:.4f} {'✅ Best' if (np.isfinite(rmse_garch) and rmse_garch <= min(rmse_naive, rmse_xgb)) else ''}\n")
            
            # 9. Return dictionary for serialization
            return {
                'models': {
                    'xgb': xgb_model,
                    'garch': garch_model_fit if garch_model_fit is not None else None,
                    # Naive doesn't need a model object - it just uses past_vol_15 column
                },
                'feature_names': self.feature_cols,
                'scaler': self.scaler,
                'scores': {
                    'naive_rmse': float(rmse_naive),
                    'xgb_rmse': float(rmse_xgb),
                    'garch_rmse': float(rmse_garch) if np.isfinite(rmse_garch) else None
                },
                'best_model': 'naive' if rmse_naive <= min(rmse_xgb, rmse_garch) else ('xgb' if rmse_xgb <= rmse_garch else 'garch'),
                'trained_at': pd.Timestamp.now().isoformat()
            }
            
        except CryptoException:
            raise
        except Exception as e:
            logging.error(f"Unexpected error in Initiate_ModelBaselines: {e}", exc_info=True)
            raise CryptoException(f"Training failed: {type(e).__name__} - {str(e)}", sys)