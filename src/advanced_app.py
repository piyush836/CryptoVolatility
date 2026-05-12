"""
Minimal working Flask app for Crypto Volatility Forecaster
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import yfinance as yf
import requests  # ✅ For session-based yfinance workaround
import templates  # ✅ Ensure template is imported for Flask to find it

# =============================================================================
# SAFE EXCEPTION HANDLING (fallback if custom module missing)
# =============================================================================
try:
    from notebooks.exception.Exception import CryptoException
except ImportError:
    class CryptoException(Exception):
        """Fallback exception class for crypto-related errors"""
        def __init__(self, message: str, *args):
            super().__init__(message)

# =============================================================================
# FLASK APP INITIALIZATION - FIXED PATH & SINGLE IMPORT
# =============================================================================
# ✅ Get the project root directory (parent of 'src')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')

# ✅ Initialize Flask with explicit template folder
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')

# Debug print (remove in production)
print(f"🔍 Flask templates path: {TEMPLATE_DIR}")

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

# =============================================================================
# DATA LOADING - FIXED: SESSION-BASED HEADERS (works on all yfinance versions)
# =============================================================================
def download_crypto_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download crypto data from yfinance (compatible with yfinance >= 0.2.40)"""
    try:
        ticker = f"{symbol}-USD"
        
        # ✅ FIX: Remove custom session - yfinance 0.2.40+ handles headers internally
        # If blocked, yfinance will auto-retry with appropriate headers
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 2:
            raise CryptoException(f"No data retrieved for {ticker}. Yahoo Finance may be blocking requests.")
        
        # Handle yfinance multi-level columns (new in 0.2.40+)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Flatten to single level
        # After flattening MultiIndex columns, normalize column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ✅ NEW: Strip whitespace and standardize casing for critical columns
        df.columns = df.columns.str.strip().str.title()  # "close" → "Close", " close " → "Close"

        # Handle index variations
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        
        
        # Ensure 'Date' column exists and is datetime
        if 'Date' not in df.columns:
            date_col = [c for c in df.columns if str(c).lower() == 'date']
            if date_col:
                df.rename(columns={date_col[0]: 'Date'}, inplace=True)
            else:
                raise CryptoException(f"Could not identify Date column in {ticker} data")
            
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Symbol'] = symbol
        return df
        
    except CryptoException:
        raise
    except Exception as e:
        raise CryptoException(f"Failed to download {symbol}: {type(e).__name__} - {str(e)}")

def download_crypto_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        ticker = f"{symbol}-USD"
        
        # Download data (yfinance 0.2.40+ returns MultiIndex columns)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 2:
            raise CryptoException(f"No data retrieved for {ticker}")
        
        # ✅ CRITICAL FIX: Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            # Extract just the metric name (e.g., 'Close' from ('Close', 'BTC-USD'))
            df.columns = df.columns.get_level_values(0)
        
        # ✅ Also strip whitespace and standardize casing
        df.columns = df.columns.str.strip().str.title()
        
        # Reset index if Date is the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Ensure 'Date' column exists
        if 'Date' not in df.columns:
            date_candidates = [c for c in df.columns if str(c).lower() == 'date']
            if date_candidates:
                df.rename(columns={date_candidates[0]: 'Date'}, inplace=True)
            else:
                raise CryptoException(f"Could not identify Date column in {ticker} data")
        
        # Ensure 'Close' column exists (required for volatility calculation)
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
# VOLATILITY FORECASTING LOGIC (unchanged - working correctly)
# =============================================================================
def forecast_volatility(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> dict:
    try:
        # 🔍 DEBUG: Print what we received
        print(f"🔍 forecast_volatility: columns={list(df.columns)}, rows={len(df)}")
        print(f"🔍 df['Close'] sample: {df['Close'].head(3).tolist()}")
        
        close_prices = df['Close'].replace(0, np.nan).dropna()
        if len(close_prices) < 2:
            raise ValueError(f"Insufficient price data (got {len(close_prices)} valid prices)")
            
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        print(f"🔍 log_returns: {len(log_returns)} valid, first 3: {log_returns.head(3).tolist()}")
        
        rolling_vol = log_returns.rolling(window=window, min_periods=1).std() * ANNUALIZATION_FACTOR
        current_vol = rolling_vol.iloc[-1]
        vol_pct = current_vol * 100
        
        print(f"🔍 Volatility result: {vol_pct:.2f}%")  # 🔍 DEBUG LINE
        
        if log_returns.empty:
            trend = 'flat'
        else:
            last_return = log_returns.iloc[-1]
            trend = 'up' if last_return > 0 else 'down' if last_return < 0 else 'flat'

        if vol_pct >= 80:
            risk_level = 'very_high'
        elif vol_pct >= 60:
            risk_level = 'high'
        elif vol_pct >= 40:
            risk_level = 'medium'
        elif vol_pct >= 20:
            risk_level = 'low'
        else:
            risk_level = 'very_low'

        result = {
            'current_vol': round(vol_pct, 2),
            'forecast_vol': round(vol_pct, 2),
            'trend': trend,
            'risk_level': risk_level,
            'data_points': len(df),
            'latest_price': round(float(df['Close'].iloc[-1]), 2),
            'latest_date': pd.Timestamp(df['Date'].iloc[-1]).strftime('%Y-%m-%d'),
            'success': True,
        }
        print(f"✅ Returning forecast: {result}")  # 🔍 DEBUG LINE
        return result
        
    except Exception as e:
        print(f"❌ forecast_volatility ERROR: {e}")  # 🔍 DEBUG LINE
        import traceback
        traceback.print_exc()
        return {'error': f"{type(e).__name__}: {str(e)}", 'success': False}
def get_portfolio_risk_score(forecasts: dict) -> dict:
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
        if horizon not in VALID_HORIZONS: horizon = 15
        
        days_back = max(100, horizon * 4)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        forecasts, errors = {}, {}
        for coin in coins:
            if coin not in SUPPORTED_CRYPTOS:
                errors[coin] = f"Unsupported: {coin}"
                continue
            try:
                df = download_crypto_data(coin, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if len(df) < horizon:
                    errors[coin] = f"Insufficient data (got {len(df)}, need ≥{horizon})"
                    continue
                forecasts[coin] = forecast_volatility(df, window=DEFAULT_WINDOW)
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
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0'}), 200

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception:
        return jsonify({
            'project': 'Crypto Volatility Forecaster',
            'description': 'Predicts 15-day Bitcoin volatility using causal ML',
            'endpoints': {'/': 'Dashboard', '/api/forecast': 'POST forecasts', '/health': 'Health check'}
        }), 200

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
    
    print("🚀 Crypto Volatility Dashboard starting...")
    print(f"📁 Templates: {TEMPLATE_DIR}")
    print("📍 API Docs: http://localhost:5000/about")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)