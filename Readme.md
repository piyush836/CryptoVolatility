# 📈 CryptoVolatility: Bitcoin Volatility Forecasting

Predicting 15-day future volatility of Bitcoin using causal machine learning and financial econometrics.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## Project Overview

**Problem**: Forecast 15-day future volatility of Bitcoin price to support trading and risk decisions.

**Who cares**: Traders, risk analysts, quant researchers.

**Data Source**: Historical BTC price data from Yahoo Finance CSVs.

**Final Output**: Numeric volatility forecast (percentage) and comparison to baseline models.

**Where this fits**: This project covers data ingestion, feature engineering, modeling, and evaluation stages of a DS pipeline.


## 🎯 Goal
Forecast **annualized 15-day Bitcoin volatility** using only historical data — with **no future leakage** — to support risk-aware trading and portfolio management.

## Key Insights

- The Naïve model outperformed XGBoost and GARCH, suggesting persistence in short-term volatility.
- This indicates that simple baselines can be strong benchmarks in financial time series forecasting.

## 📊 Results (as of Jan 2026)
| Model          | RMSE     | Performance |
|----------------|----------|-------------|
| **Naïve**      | 0.1604   | ✅ Best     |
| XGBoost        | 0.3139   | ❌ Worse    |
| GARCH          | 5.5352   | ❌ Misaligned |

✅ **Real-world validation**:  
- Forecast on **2025-12-11**: **37.62%**  
- Actual (Dec 12–26): **22.97%**  
- Interpretation: Model correctly sensed elevated risk, though market calmed faster than expected.

## 🧠 Features
- ✅ **Causal feature engineering** (no look-ahead bias)
- ✅ Temporal train/test split (past → future)
- ✅ Log returns, annualized volatility (√252)
- ✅ Baseline comparison: Naïve, GARCH, XGBoost
- ✅ Config-driven pipeline (`config.yaml`)

## 📂 Project Structure
- 📁 data/raw – raw CSV files
- 📁 notebooks – EDA and model notebooks
- 📁 src – processing and model code

## Data Science Pipeline

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Modeling
5. Evaluation
6. Insights/Output

## SQL Feature Extraction (Conceptual)

The following SQL queries illustrate how volatility-related features could be extracted from a relational database before modeling.

## Assumptions

- Future price behavior is suitable for statistical forecasting.
- External events are not explicitly modeled.


## Next Steps

- Try LSTM / RNN for time-series.
- Add macroeconomic features.
- Deploy a simple API.


