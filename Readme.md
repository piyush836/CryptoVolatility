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

# 1. Fetch historical BTC prices

```sql
SELECT timestamp, close_price
FROM crypto_prices
WHERE symbol = 'BTC'
ORDER BY timestamp;

# 2. Calculate daily returns

```sql
SELECT
    timestamp,
    (close_price - LAG(close_price) OVER (ORDER BY timestamp)) /
     LAG(close_price) OVER (ORDER BY timestamp) AS daily_return
FROM crypto_prices
WHERE symbol = 'BTC';

### 3. Calculate 15-day rolling volatility

```sql
SELECT
    timestamp,
    STDDEV(daily_return) OVER (
        ORDER BY timestamp
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS rolling_volatility_15d
FROM (
    SELECT
        timestamp,
        (close_price - LAG(close_price) OVER (ORDER BY timestamp)) /
         LAG(close_price) OVER (ORDER BY timestamp) AS daily_return
    FROM crypto_prices
    WHERE symbol = 'BTC'
) t;

### 4. Select training data (no future leakage)

```sql
SELECT *
FROM crypto_prices
WHERE symbol = 'BTC'
AND timestamp <= '2025-11-30';
```
## Assumptions

- Future price behavior is suitable for statistical forecasting.
- External events are not explicitly modeled.


## Next Steps

- Try LSTM / RNN for time-series.
- Add macroeconomic features.
- Deploy a simple API.

## SQL vs Pandas for Feature Engineering

SQL is used for filtering, aggregations, and window-based computations directly on stored data, reducing data transfer and ensuring consistent feature extraction across systems. Pandas is then used for more flexible transformations, exploratory analysis, and modeling workflows once the required features are extracted.

**Used SQL for:**
- Filtering historical BTC data by time range
- Computing daily returns
- Calculating rolling 15-day volatility
- Enforcing train/test split without future leakage

**Used pandas for:**
- Exploratory data analysis and visualization
- Feature inspection and validation
- Model training and evaluation

