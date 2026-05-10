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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notebooks.logging.logger import logging
from notebooks.exception.Exception import CryptoException

app = Flask(__name__)

SUPPORTED_CRYPTOS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'SOL': 'Solana',
    'DOGE': 'Dogecoin',
    'MATIC': 'Polygon',
    'LINK': 'Chainlink',
    'LTC': 'Litecoin',
    'BCH': 'Bitcoin Cash',
}

HORIZONS = {
    '7': '7-day (1 week)',
    '15': '15-day (3 weeks)',
    '30': '30-day (1 month)',
    '60': '60-day (2 months)',
}


def download_crypto_data(symbol: str, start_date: str, end_date: str):
    """Download crypto data from yfinance"""
    try:
        ticker = f"{symbol}-USD"
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        
        df.reset_index(inplace=True)
        df['Symbol'] = symbol
        return df
    except Exception as e:
        raise CryptoException(f"Failed to download {symbol}: {str(e)}", sys)


def forecast_volatility(df: pd.DataFrame, window: int, horizon: int) -> dict:
    """Forecast future volatility using naïve persistence"""
    try:
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(365)
        
        current_vol = rolling_vol.iloc[-1]
        forecast_vol = current_vol
        
        avg_vol = rolling_vol.iloc[-60:].mean() if len(rolling_vol) >= 60 else rolling_vol.mean()
        
        if current_vol > avg_vol * 1.2:
            trend = 'increasing'
        elif current_vol < avg_vol * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        if forecast_vol < 0.30:
            risk_level = 'low'
        elif forecast_vol < 0.50:
            risk_level = 'medium'
        elif forecast_vol < 0.80:
            risk_level = 'high'
        else:
            risk_level = 'very_high'
        
        return {
            'current_vol': round(current_vol * 100, 2),
            'forecast_vol': round(forecast_vol * 100, 2),
            'trend': trend,
            'risk_level': risk_level,
            'data_points': len(df),
            'latest_price': round(df['Close'].iloc[-1], 2),
            'latest_date': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
            'success': True,
        }
    except Exception as e:
        return {'error': str(e), 'success': False}


def get_portfolio_risk_score(forecasts: dict) -> dict:
    """Calculate portfolio-level risk metrics"""
    try:
        volatilities = [f['forecast_vol'] for f in forecasts.values() if f.get('success')]
        
        if not volatilities:
            return {'error': 'No valid data'}
        
        avg_vol = np.mean(volatilities)
        max_vol = np.max(volatilities)
        min_vol = np.min(volatilities)
        risk_coins = [coin for coin, f in forecasts.items() 
                     if f.get('success') and f['risk_level'] in ['high', 'very_high']]
        
        return {
            'average_volatility': round(avg_vol, 2),
            'max_volatility': round(max_vol, 2),
            'min_volatility': round(min_vol, 2),
            'high_risk_count': len(risk_coins),
            'high_risk_coins': risk_coins,
            'portfolio_risk': 'HIGH' if avg_vol > 0.60 else ('MEDIUM' if avg_vol > 0.40 else 'LOW'),
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """POST forecast request"""
    try:
        data = request.get_json()
        coins = data.get('coins', ['BTC', 'ETH'])
        horizon = int(data.get('horizon', 15))
        days_back = max(100, horizon * 2)
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if horizon not in [7, 15, 30, 60]:
            horizon = 15
        
        forecasts = {}
        errors = {}
        
        for coin in coins:
            if coin not in SUPPORTED_CRYPTOS:
                errors[coin] = 'Unsupported cryptocurrency'
                continue
            
            try:
                df = download_crypto_data(coin, start_date, end_date)
                
                if df.empty or len(df) < horizon:
                    errors[coin] = 'Insufficient data'
                    continue
                
                result = forecast_volatility(df, window=15, horizon=horizon)
                forecasts[coin] = result
                
            except Exception as e:
                errors[coin] = str(e)
        
        portfolio_risk = get_portfolio_risk_score(forecasts)
        
        return jsonify({
            'status': 'success',
            'horizon_days': horizon,
            'forecast_date': datetime.now().strftime('%Y-%m-%d'),
            'forecasts': forecasts,
            'portfolio_risk': portfolio_risk,
            'errors': errors,
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/chart-data', methods=['POST'])
def api_chart_data():
    """Get historical data for charting"""
    try:
        data = request.get_json()
        coin = data.get('coin', 'BTC')
        days = int(data.get('days', 100))
        
        if coin not in SUPPORTED_CRYPTOS:
            return jsonify({'error': 'Invalid coin'}), 400
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        df = download_crypto_data(coin, start_date, end_date)
        
        if df.empty:
            return jsonify({'error': 'No data'}), 400
        
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        rolling_vol = log_returns.rolling(window=15).std() * np.sqrt(365) * 100
        
        chart_data = {
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].round(2).tolist(),
            'volatility': rolling_vol.round(2).tolist(),
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/about')
def about():
    return render_template('about.html')


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error', 'details': str(error)}), 500


if __name__ == '__main__':
    os.makedirs('data/raw_data', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)