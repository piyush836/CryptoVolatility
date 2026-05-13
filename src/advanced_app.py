"""
Minimal working Flask app for Crypto Volatility Forecaster
Integrates trained ML models with statistical fallback.
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import yfinance as yf
import joblib
import traceback

# =============================================================================
# SAFE EXCEPTION HANDLING (fallback if custom module missing)
# =============================================================================
try:
    from notebooks.exception.Exception import CryptoException
except ImportError:
    class CryptoException(Exception):
        """Fallback exception class for crypto-related errors"""
        pass

# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================
# Get the project root directory (parent of 'src')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
SUPPORTED_CRYPTOS = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'XRP': 'Ripple', 'ADA': 'Cardano',
    'SOL': 'Solana', 'DOGE': 'Dogecoin', 'MATIC': 'Polygon', 'LINK': 'Chainlink',
    'LTC': 'Litecoin', 'BCH': 'Bitcoin Cash',
}

VALID_HORIZONS = [7, 15, 30, 60]
DEFAULT_WINDOW = 15
ANNUALIZATION_FACTOR = np.sqrt(365)

# Model path for ML integration
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "volatility_models.pkl")
_cached_models = None

# =============================================================================
# MODEL LOADING & CACHING
# =============================================================================
def load_trained_models():
    """Loads trained models from disk with in-memory caching."""
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f" Loading models from: {MODEL_PATH}")
            artifacts = joblib.load(MODEL_PATH)
            
            _cached_models = {
                'models': artifacts.get('models', {}),
                'feature_names': artifacts.get('feature_names', []),
                'scaler': artifacts.get('scaler'),
                'trained_at': artifacts.get('trained_at'),
                'method': 'ml'
            }
            print(f"✅ Loaded {len(_cached_models['models'])} ML models.")
        else:
            print(f"⚠️ Model file not found at {MODEL_PATH}. Using statistical fallback.")
            _cached_models = {'method': 'statistical'}
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        _cached_models = {'method': 'statistical', 'error': str(e)}
        
    return _cached_models

# =============================================================================
# DATA LOADING
# =============================================================================
def download_crypto_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download crypto data from yfinance (compatible with yfinance >= 0.2.40)"""
    try:
        ticker = f"{symbol}-USD"
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 2:
            raise CryptoException(f"No data retrieved for {ticker}. Yahoo Finance may be blocking requests.")
        
        # Handle yfinance multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize column names
        df.columns = df.columns.str.strip().str.title()
        
        # Reset index if Date is the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            
        # Ensure critical columns exist
        if 'Date' not in df.columns:
            date_candidates = [c for c in df.columns if str(c).lower() == 'date']
            if date_candidates:
                df.rename(columns={date_candidates[0]: 'Date'}, inplace=True)
            else:
                raise CryptoException(f"Could not identify Date column in {ticker} data")
                
        if 'Close' not in df.columns:
            close_candidates = [c for c in df.columns if str(c).lower() == 'close']
            if close_candidates:
                df.rename(columns={close_candidates[0]: 'Close'}, inplace=True)
            else:
                raise CryptoException(f"Could not identify Close column. Available: {list(df.columns)}")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Symbol'] = symbol
        return df
        
    except CryptoException:
        raise
    except Exception as e:
        raise CryptoException(f"Failed to download {symbol}: {type(e).__name__} - {str(e)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _classify_risk(vol_pct: float) -> str:
    """Classify volatility into risk levels."""
    if vol_pct >= 80: return 'very_high'
    elif vol_pct >= 60: return 'high'
    elif vol_pct >= 40: return 'medium'
    elif vol_pct >= 20: return 'low'
    else: return 'very_low'

def get_portfolio_risk_score(forecasts: dict) -> dict:
    """Aggregate individual forecasts into a portfolio risk score."""
    try:
        valid = [f for f in forecasts.values() if f.get('success')]
        if not valid:
            return {'error': 'No valid forecast data', 'portfolio_risk': 'UNKNOWN'}
        
        vols = [f['forecast_vol'] for f in valid]
        avg_vol = float(np.mean(vols))
        high_risk = [c for c, f in forecasts.items() if f.get('success') and f.get('risk_level') in ['high', 'very_high']]
        
        return {
            'average_volatility': round(avg_vol, 2),
            'max_volatility': round(float(np.max(vols)), 2),
            'min_volatility': round(float(np.min(vols)), 2),
            'high_risk_count': len(high_risk),
            'high_risk_coins': high_risk,
            'portfolio_risk': 'HIGH' if avg_vol > 60 else ('MEDIUM' if avg_vol > 40 else 'LOW'),
        }
    except Exception as e:
        return {'error': f"{type(e).__name__}: {str(e)}", 'portfolio_risk': 'ERROR'}

# =============================================================================
# VOLATILITY FORECASTING LOGIC (ML + Statistical Fallback)
# =============================================================================
def engineer_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features on live data to match training data structure.
    Must match the features used in notebooks/feature_engineering.py
    """
    try:
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            raise ValueError("Missing Close or Volume columns")
        
        df = df.copy()
        
        # 1. Log Returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Past Volatility (15-day rolling, annualized)
        df['past_vol_15'] = df['log_return'].rolling(window=15, min_periods=1).std() * ANNUALIZATION_FACTOR
        
        # 3. Volume Average (30-day)
        df['volume_avg_30'] = df['Volume'].rolling(window=30, min_periods=1).mean()
        
        # 4. High-Low Ratio (5-day intraday range)
        df['high_low_ratio_5'] = (df['High'] - df['Low']).rolling(window=5, min_periods=1).mean() / df['Close']
        
        # 5. Past Return (7-day)
        df['past_return_7'] = df['log_return'].rolling(window=7, min_periods=1).sum()
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 1:
            raise ValueError("No valid rows after feature engineering")
        
        print(f"✅ Engineered {len(df)} rows with features: {['past_vol_15', 'volume_avg_30', 'high_low_ratio_5', 'past_return_7']}")
        return df
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return df  # Return original - will fall back to statistical method


def forecast_volatility(df: pd.DataFrame, window: int = DEFAULT_WINDOW, use_ml: bool = True) -> dict:
    """
    Forecast cryptocurrency volatility using ML models (with statistical fallback).
    Always returns a complete dict to prevent frontend 'undefined' errors.
    """
    # Default safe response structure
    safe_response = {
        'current_vol': 0.0,
        'forecast_vol': 0.0,
        'trend': 'unknown',
        'risk_level': 'unknown',
        'data_points': 0,
        'latest_price': 0.0,
        'latest_date': datetime.now().strftime('%Y-%m-%d'),
        'method': 'error',
        'success': False,
        'error': None
    }

    try:
        # 🔍 Validate input data
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided")
        if 'Close' not in df.columns:
            raise ValueError(f"Missing 'Close' column. Available: {list(df.columns)}")
        if len(df) < 2:
            raise ValueError(f"Insufficient data points (got {len(df)}, need ≥2)")

        print(f"📊 forecast_volatility: {len(df)} rows, columns={list(df.columns)}")
        
        # Clean price data
        close_prices = df['Close'].replace(0, np.nan).dropna()
        if len(close_prices) < 2:
            raise ValueError(f"Insufficient valid prices (got {len(close_prices)})")
        
        # Calculate log returns for statistical method
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_returns) < 1:
            raise ValueError("Could not calculate log returns")
        
        # Determine trend from latest return
        last_return = log_returns.iloc[-1]
        trend = 'up' if last_return > 0 else 'down' if last_return < 0 else 'flat'
        latest_date = pd.Timestamp(df['Date'].iloc[-1]).strftime('%Y-%m-%d') if 'Date' in df.columns else datetime.now().strftime('%Y-%m-%d')
        latest_price = round(float(df['Close'].iloc[-1]), 2)

        # =====================================================================
        # OPTION 1: ML Prediction (with feature engineering)
        # =====================================================================
        if use_ml:
            try:
                # Engineer features for live prediction
                df_with_features = engineer_features_for_prediction(df)
                
                artifacts = load_trained_models()
                
                if artifacts and artifacts.get('method') == 'ml':
                    required_features = artifacts.get('feature_names', [])
                    
                    # Check if we have the required engineered features
                    has_features = all(feat in df_with_features.columns for feat in required_features)
                    
                    if has_features and len(df_with_features) >= 1:
                        print("🤖 Attempting ML prediction with engineered features...")
                        
                        # Prepare features from latest row
                        X_latest = df_with_features[required_features].iloc[[-1]]
                        
                        # Apply scaler if available
                        if artifacts.get('scaler') is not None:
                            X_latest = artifacts['scaler'].transform(X_latest)
                        
                        # Get best model (prioritize XGBoost)
                        models = artifacts['models']
                        if 'xgb' in models and models['xgb'] is not None:
                            model = models['xgb']
                            model_name = 'XGBoost'
                        elif 'garch' in models and models['garch'] is not None:
                            # GARCH requires special handling - skip for now
                            print("⚠️ GARCH model requires special forecasting logic - using XGBoost fallback")
                            model = None
                            model_name = None
                        else:
                            model = next((m for m in models.values() if m is not None), None)
                            model_name = list(models.keys())[0] if model else None
                        
                        if model is not None and hasattr(model, 'predict'):
                            # Make prediction
                            pred = float(model.predict(X_latest)[0])
                            
                            # Validate prediction
                            if not np.isfinite(pred):
                                raise ValueError(f"Invalid prediction value: {pred}")
                            
                            # Convert to percentage if needed
                            vol_pct = pred if pred > 1 else pred * 100
                            
                            print(f"✅ ML prediction successful: {vol_pct:.2f}% ({model_name})")
                            
                            # Classify risk level
                            risk_level = _classify_risk(vol_pct)
                            
                            return {
                                **safe_response,
                                'current_vol': round(vol_pct, 2),
                                'forecast_vol': round(vol_pct, 2),
                                'trend': trend,
                                'risk_level': risk_level,
                                'data_points': len(df),
                                'latest_price': latest_price,
                                'latest_date': latest_date,
                                'method': 'ml',
                                'model_used': model_name,
                                'success': True
                            }
                        else:
                            print("⚠️ No valid ML model found for prediction")
                            
            except Exception as ml_err:
                print(f"⚠️ ML prediction failed: {ml_err}. Falling back to statistical method.")
                # Continue to statistical method below

        # =====================================================================
        # OPTION 2: Statistical Method (Rolling Volatility)
        # =====================================================================
        print(f"📐 Using statistical method (window={window})")
        
        # Calculate rolling volatility
        rolling_vol = log_returns.rolling(window=window, min_periods=1).std() * ANNUALIZATION_FACTOR
        current_vol = float(rolling_vol.iloc[-1])
        
        # Validate volatility
        if not np.isfinite(current_vol):
            raise ValueError(f"Invalid volatility calculation: {current_vol}")
        
        vol_pct = current_vol * 100
        print(f"✅ Statistical forecast: {vol_pct:.2f}%")
        
        # Classify risk level
        risk_level = _classify_risk(vol_pct)
        
        return {
            **safe_response,
            'current_vol': round(vol_pct, 2),
            'forecast_vol': round(vol_pct, 2),
            'trend': trend,
            'risk_level': risk_level,
            'data_points': len(df),
            'latest_price': latest_price,
            'latest_date': latest_date,
            'method': 'statistical',
            'window_used': window,
            'success': True
        }
        
    except Exception as e:
        # =====================================================================
        # ERROR HANDLING: Always return safe defaults
        # =====================================================================
        print(f"❌ forecast_volatility ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        safe_response['error'] = f"{type(e).__name__}: {str(e)}"
        return safe_response


def _classify_risk(vol_pct: float) -> str:
    """Classify volatility into risk levels."""
    if vol_pct >= 80:
        return 'very_high'
    elif vol_pct >= 60:
        return 'high'
    elif vol_pct >= 40:
        return 'medium'
    elif vol_pct >= 20:
        return 'low'
    else:
        return 'very_low'

       

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    try:
        return render_template('advanced_index.html')
    except Exception as e:
        return jsonify({
            "status": "template_error",
            "message": str(e),
            "hint": f"Ensure 'advanced_index.html' exists in: {TEMPLATE_DIR}"
        }), 500

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    try:
        data = request.get_json(force=True, silent=True) or {}
        coins = data.get('coins', ['BTC', 'ETH'])
        horizon = int(data.get('horizon', 15))
        if horizon not in VALID_HORIZONS:
            horizon = 15
            
        days_back = max(100, horizon * 4)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        forecasts, errors = {}, {}
        for coin in coins:
            if coin not in SUPPORTED_CRYPTOS:
                errors[coin] = f"Unsupported: {coin}"
                continue
            try:
                # 1. Download raw data
                df = download_crypto_data(coin, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if len(df) < horizon:
                    errors[coin] = f"Insufficient data (got {len(df)}, need ≥{horizon})"
                    continue
                
                # ✅ 2. Engineer features BEFORE forecasting
                df = engineer_features_for_prediction(df)
                
                # 3. Now forecast with features available
                forecasts[coin] = forecast_volatility(df, window=DEFAULT_WINDOW, use_ml=True)
                
            except CryptoException as e:
                errors[coin] = str(e)
            except Exception as e:
                errors[coin] = f"Unexpected: {type(e).__name__} - {str(e)}"
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'parameters': {'coins_requested': coins, 'horizon_days': horizon, 'lookback_days': days_back},
            'forecasts': forecasts,
            'portfolio_risk': get_portfolio_risk_score(forecasts),
            'errors': errors if errors else None,
        })
    except Exception as e:
        app.logger.error(f"Forecast API error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Server error: {type(e).__name__}"}), 500

@app.route('/api/chart-data', methods=['POST'])
def api_chart_data():
    try:
        data = request.get_json(force=True, silent=True) or {}
        coin = data.get('coin', 'BTC')
        days = min(int(data.get('days', 100)), 365)
        if coin not in SUPPORTED_CRYPTOS:
            return jsonify({'error': f"Invalid coin: {coin}"}), 400
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = download_crypto_data(coin, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        rolling_vol = log_returns.rolling(window=15).std() * ANNUALIZATION_FACTOR * 100
        
        return jsonify({
            'coin': coin, 'name': SUPPORTED_CRYPTOS[coin],
            'data': {
                'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].round(2).tolist(),
                'volatility': rolling_vol.round(2).fillna(0).tolist(),
            },
            'metadata': {
                'points': len(df),
                'latest_price': float(df['Close'].iloc[-1]),
                'latest_vol': float(rolling_vol.iloc[-1]) if not pd.isna(rolling_vol.iloc[-1]) else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(), 
        'version': '1.0.0',
        'ml_model_loaded': _cached_models is not None and _cached_models.get('method') == 'ml'
    }), 200

@app.route('/about')
def about():
    return jsonify({
        'project': 'Crypto Volatility Forecaster',
        'description': 'Predicts crypto volatility using causal ML & statistical baselines',
        'endpoints': {
            '/': 'Dashboard', 
            '/api/forecast': 'POST forecasts', 
            '/health': 'Health check',
            '/api/chart-data': 'POST historical chart data'
        }
    }), 200

# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(error):
    app.logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error', 'details': str(error) if app.debug else None}), 500

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    os.makedirs('data/raw_data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    print("🚀 Crypto Volatility Dashboard starting...")
    print(f"📁 Templates: {TEMPLATE_DIR}")
    print(f"📦 Models: {MODEL_PATH}")
    print("📍 API Docs: http://localhost:5000/about")
    print("💡 Tip: Run 'python main.py --mode full' to train models first.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)