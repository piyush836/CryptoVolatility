"""
Model evaluation result

Naïve model RMSE: 0.1604
XGBoost RMSE: 0.3139
GARCH RMSE: 5.5352

"""

"""
Error Interpretation 

The naïve volatility model performed best because short-horizon volatility
exhibits strong persistence. Recent realized volatility already captures most
of the information needed for near-term forecasts, while more complex models
introduced additional variance without sufficient new signal.
"""


"""
GARCH Interpretation 

GARCH models estimate conditional volatility under strong parametric assumptions
and are better suited for modeling volatility dynamics rather than direct
forward-looking forecasts. In this setup, the mismatch between in-sample
conditional variance and the future volatility target led to weaker performance.
"""

"""
Failure Scenarios 

Volatility forecasts are less reliable during sudden regime changes such as
market crashes, major macroeconomic events, or unexpected news, where historical
patterns break down and recent volatility is no longer representative.
"""

"""
Failure Detection 

Model failure can be detected by monitoring forecast errors over time.
Sustained increases in RMSE or large deviations between predicted and realized
volatility may indicate regime changes that require model retraining or review.
"""

"""
Mitigation Strategies 

To mitigate model failure, the system can retrain models on more recent data,
shorten the forecasting horizon, or fall back to simpler baselines during periods
of extreme uncertainty. Incorporating regime detection or external signals may
also improve robustness.
"""
