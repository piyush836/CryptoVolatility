


"""
CLI Entry Point for Crypto Volatility Forecaster
Routes execution between data pipeline, ML training, and Flask web server.
"""
import argparse
import sys
import os
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# PATH CONFIGURATION (Matches advanced_app.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "volatility_models.pkl"

# Ensure imports work regardless of execution directory
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CUSTOM MODULE IMPORTS
# =============================================================================
try:
    from src.data_loader import DataLoader
    from notebooks.feature_engineering import FeatureEngineering
    from notebooks.model_baselines import ModelBaselines
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Ensure your project structure matches: src/, notebooks/, artifacts/")
    sys.exit(1)

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================
def download_data():
    """Download raw crypto data"""
    print("📥 Downloading Bitcoin data...")
    DataLoader().initiateDataloader()
    print("✅ Data saved to data/raw_data/raw_data.csv\n")


def engineer_features():
    """Run feature engineering and return processed DataFrame"""
    print(" Engineering features...")
    df = FeatureEngineering().initiate_feature_engineering()
    print(f"✅ Features created! ({len(df)} rows)\n")
    return df


def train_and_save_models(df):
    """
    Train models and serialize them + metadata to artifacts/volatility_models.pkl
    Expects ModelBaselines to return a dict like:
    {'models': {'xgb': model, 'garch': model}, 'feature_names': [...], 'scaler': scaler}
    """
    print("🏋️ Training models...")
    baseline = ModelBaselines()
    results = baseline.Initiate_ModelBaselines()
    
    # Safely extract components (adapt if your class returns differently)
    if isinstance(results, dict):
        models_dict = results.get('models', results)
        feature_names = results.get('feature_names', list(df.columns[:-1]))
        scaler = results.get('scaler')
    else:
        # Fallback if class returns just the model(s)
        models_dict = {'model': results}
        feature_names = list(df.columns[:-1])
        scaler = None
        
    if not models_dict:
        raise ValueError("Training failed: No models returned.")
        
    # Create artifacts directory
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Package for serialization
    artifacts = {
        'models': models_dict,
        'feature_names': feature_names,
        'scaler': scaler,
        'trained_at': datetime.now().isoformat()
    }
    
    # Save to disk
    joblib.dump(artifacts, MODEL_FILE)
    print(f"💾 Models saved to {MODEL_FILE}")
    print(f"📦 Contains: {list(models_dict.keys())}")
    print("✅ Training complete!\n")
    return artifacts


def predict(df=None, use_saved_model=True):
    """
    Run inference. Prioritizes saved ML model, falls back to naive persistence.
    """
    if use_saved_model and MODEL_FILE.exists():
        print("🤖 Loading trained model for prediction...")
        artifacts = joblib.load(MODEL_FILE)
        model = next(iter(artifacts['models'].values()))
        feature_names = artifacts['feature_names']
        
        # Get fresh data if not provided
        if df is None:
            print(" Fetching latest data for prediction...")
            df = engineer_features()
            
        # Prepare latest row
        X_latest = df[feature_names].iloc[[-1]]
        if artifacts.get('scaler') is not None:
            X_latest = artifacts['scaler'].transform(X_latest)
            
        # Predict
        pred = float(model.predict(X_latest)[0])
        vol_pct = pred if pred > 1 else pred * 100
        
        # Safely get date
        idx = df.index[-1]
        date_str = idx.date() if hasattr(idx, 'date') else str(idx)
        
        print(f"📅 Forecast Date: {date_str}")
        print(f"📊 15-day Volatility: {vol_pct:.2f}%")
        print(f"💡 Model Used: {list(artifacts['models'].keys())[0]}")
    else:
        # Fallback to Naive Persistence (past_vol_15)
        print("💡 Fallback: Using Naive Persistence (past_vol_15)")
        if df is None:
            df = engineer_features()
            
        latest_row = df.iloc[-1]
        forecast_vol = latest_row.get('past_vol_15', 0)
        idx = df.index[-1]
        date_str = idx.date() if hasattr(idx, 'date') else str(idx)
        
        print(f"📅 Forecast Date: {date_str}")
        print(f"📊 15-day Volatility: {forecast_vol:.2%}")
        print("💡 Model: Naive Persistence Baseline")
    print()


def serve():
    """Start Flask web server with pre-loaded models"""
    print("🌐 Starting Flask server with ML integration...")
    print("📍 Visit http://localhost:5000")
    print("🚀 Multi-coin volatility dashboard is running!\n")
    
    # Pre-warm model loading (optional but recommended for faster first request)
    try:
        from src.advanced_app import load_trained_models
        load_trained_models()
        print("✅ Models pre-loaded for Flask app.")
    except ImportError:
        print("⚠️ Could not pre-load models. Flask will load them on first request.")
        
    from src.advanced_app import app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Crypto Volatility Forecaster - CLI & Web Dashboard'
    )
    parser.add_argument(
        '--mode',
        choices=['download', 'full', 'predict', 'train', 'serve'],
        default='full',
        help='Operation mode: download, full, train, predict, or serve'
    )
    args = parser.parse_args()

    if args.mode == 'download':
        download_data()
        
    elif args.mode == 'full':
        download_data()
        df = engineer_features()
        train_and_save_models(df)
        predict(df, use_saved_model=True)
        
    elif args.mode == 'train':
        df = engineer_features()
        train_and_save_models(df)
        
    elif args.mode == 'predict':
        predict(df=None, use_saved_model=True)
        
    elif args.mode == 'serve':
        serve()


if __name__ == '__main__':
    main()