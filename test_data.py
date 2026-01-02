# test_features.py
from notebooks.feature_engineering import FeatureEngineering
from src.data_loader import DataLoader
from notebooks.model_baselines import ModelBaselines
if __name__ == "__main__":
    df = FeatureEngineering().initiate_feature_engineering()
    print("Feature engineering completed successfully!")
    load = DataLoader().initiateDataloader()
    print("Data loading completed successfully!")
    # Create the model baselines object
    mb = ModelBaselines()
    
    # Run the full pipeline (this computes all models and plots results)
   
    latest_row = df.iloc[-1]
    forecast_vol = latest_row['past_vol_15']  # Your best forecast (naïve baseline)
    print(f"📅 Today's date: {latest_row.name}")
    print(f"🔮 Forecasted 15-day annualized volatility: {forecast_vol:.2%}")
 

    
    
