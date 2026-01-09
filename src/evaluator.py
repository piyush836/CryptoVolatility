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
GARCH Interpretation (Day 12)

GARCH models estimate conditional volatility under strong parametric assumptions
and are better suited for modeling volatility dynamics rather than direct
forward-looking forecasts. In this setup, the mismatch between in-sample
conditional variance and the future volatility target led to weaker performance.
"""
