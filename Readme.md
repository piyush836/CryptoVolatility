# 📈 CryptoVolatility: Bitcoin Volatility Forecasting

Predicting 15-day future volatility of Bitcoin using causal machine learning and financial econometrics.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 🎯 Goal
Forecast **annualized 15-day Bitcoin volatility** using only historical data — with **no future leakage** — to support risk-aware trading and portfolio management.

> 🔍 **Key Insight**: *"Volatility is highly persistent — a simple baseline often outperforms complex models."*

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