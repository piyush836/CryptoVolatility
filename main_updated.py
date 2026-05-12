import argparse
import sys
# ✅ REMOVED: from flask import app (invalid)

from src.data_loader import DataLoader
from notebooks.feature_engineering import FeatureEngineering
from notebooks.model_baselines import ModelBaselines


def download_data():
    print("📥 Downloading Bitcoin data...")
    DataLoader().initiateDataloader()
    print("✅ Data saved to data/raw_data/raw_data.csv\n")


def engineer_features():
    print("🔧 Engineering features...")
    df = FeatureEngineering().initiate_feature_engineering()
    print(f"✅ Features created! ({len(df)} rows)\n")
    return df


def train_models(df):
    print("🏋️ Training models...")
    ModelBaselines().Initiate_ModelBaselines()
    print("✅ Training complete!\n")


def predict(df):
    latest_row = df.iloc[-1]
    forecast_vol = latest_row['past_vol_15']
    print(f"📅 Forecast Date: {latest_row.name.date()}")
    print(f"📊 15-day Volatility: {forecast_vol:.2%}")
    print("💡 Model: Naive Persistence\n")


def serve():
    print("🌐 Starting Flask server...")
    print("📍 Visit http://localhost:5000")
    print("🚀 Multi-coin volatility dashboard is running!\n")
    from src.advanced_app import app
    app.run(host='0.0.0.0', port=5000, debug=True)


def main():
    parser = argparse.ArgumentParser(
        description='Crypto Volatility Forecaster - CLI and Web Dashboard'
    )
    parser.add_argument(
        '--mode',
        choices=['download', 'full', 'predict', 'train', 'serve'],
        default='full',
        help='Operation mode'
    )
    args = parser.parse_args()

    if args.mode == 'download':
        download_data()
    elif args.mode == 'full':
        download_data()
        df = engineer_features()
        train_models(df)  # ✅ Fixed: was "tra in_models"
        predict(df)
    elif args.mode == 'predict':
        df = engineer_features()
        predict(df)
    elif args.mode == 'train':
        df = engineer_features()  # ✅ Fixed: was "engineer_feature s"
        train_models(df)
    elif args.mode == 'serve':
        serve()


# ✅ Fixed entry point
if __name__ == '__main__':
    main()