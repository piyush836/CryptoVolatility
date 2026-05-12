# 📈 Crypto Volatility Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Predict Bitcoin volatility 15 days ahead using time-series ML and financial econometrics.**

## 🎯 Problem

Bitcoin traders need reliable volatility forecasts to manage risk. This project forecasts 15-day annualized volatility using historical price data and machine learning.

**Example**: On Dec 11, 2025, the model forecasted **37.62% volatility** — useful for options traders, portfolio managers, and risk analysts.

---

## 🏆 Results

| Model | RMSE | Status |
|-------|------|--------|
| **Naïve Volatility Persistence** | **0.1604** | ✅ **Best** |
| XGBoost | 0.3139 | Overfit |
| GARCH | 5.5352 | Misaligned |

### Key Finding
**Short-horizon volatility exhibits strong persistence.** Recent volatility is the best predictor of near-term volatility. This simple baseline beats complex models because:
- ML models overfit on engineered features
- GARCH assumes specific distributional assumptions that don't hold
- Market microstructure means yesterday's vol = today's vol (inertia)

---

## 🛠 Tech Stack

- **Data**: yfinance (Yahoo Finance API)
- **Feature Engineering**: pandas, numpy
- **ML Models**: scikit-learn, XGBoost, ARCH (GARCH)
- **Evaluation**: sklearn.metrics
- **Visualization**: matplotlib
- **API**: Flask (optional deployment)
- **Containerization**: Docker (optional)

---

## 📦 Installation & Setup

### Local Setup (No Docker)

```bash
# Clone repository
git clone https://github.com/yourname/crypto-volatility
cd crypto-volatility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python main.py --mode download
```

This downloads 5+ years of daily BTC-USD data from Yahoo Finance and saves to `data/raw_data/raw_data.csv`.

---

## 🚀 Usage

### Quick Forecast

```bash
python main.py --mode predict
```

Output:
