from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

import sys
from arch import arch_model
from notebooks.logging.logger import logging
from notebooks.exception.Exception import CryptoException
from notebooks.feature_engineering import FeatureEngineering
import matplotlib.pyplot as plt



class ModelBaselines:
    def __init__(self):
        self.feature_engineering=FeatureEngineering()
        
    def traing_data(self,df:pd.DataFrame):
        
        try:
            
            split_date='2024-01-01'
            test=df[df.index>=split_date]
            train=df[df.index<split_date]
            feature_cols=['past_vol_15','volume_avg_30','high_low_ratio_5','past_return_7']
            X_train=train[feature_cols]
            y_train=train['target_volatility']
            X_test=test[feature_cols]
            y_test=test['target_volatility']
            return X_train,y_train,X_test,y_test
        except Exception as e:
            raise CryptoException(str(e),sys)
    def Niive_Forcast(self,X_test:pd.Series,y_test:pd.Series):
        try:
            y_pred_naive=X_test['past_vol_15']
            rmse_naive=np.sqrt(mean_squared_error(y_test,y_pred_naive))
            logging.info("RMSE Naive Forecast: %.4f", rmse_naive)

            return rmse_naive
        except Exception as e:
            raise CryptoException(str(e),sys)
    def XGBOOST_Forcast(self,X_train:pd.Series,y_train:pd.Series,X_test:pd.Series,y_test:pd.Series):
        try:
            model=XGBRegressor()
            model.fit(X_train,y_train)
            y_pred_xgb=model.predict(X_test)
            rmse_xgb=np.sqrt(mean_squared_error(y_test,y_pred_xgb))
            logging.info("RMSE XGBOOST Forecast: %.4f", rmse_xgb)
           
            return rmse_xgb,model
           
        except Exception as e:
            raise CryptoException(str(e),sys)
    def GARCH_Forcast(self, df: pd.DataFrame):
        # NOTE:
        # GARCH volatility is evaluated using in-sample conditional variance
        # and is not directly comparable to forward-looking ML forecasts.
        # Results are reported for reference only.

        try:

            # Train on 2020-2023 log returns
            train_end = '2023-12-31'
            train_returns = df[df.index <= train_end]['log_return'].dropna()
            
            # Fit GARCH
            model = arch_model(train_returns, vol='GARCH', p=1, q=1, rescale=True)
            model_fit = model.fit(disp='off')
            
            # Get in-sample conditional volatility (for 2020-2023)
            in_sample_vol = model_fit.conditional_volatility * np.sqrt(252)  # Annualized
            
            # Align with overlapping part of target
            # Note: target_volatility for 2020-2023 is future-looking, but we'll compare in-sample vol as proxy
            common_index = in_sample_vol.index.intersection(df.index)
            y_pred_garch = in_sample_vol[common_index]
            y_true_garch = df.loc[common_index, 'target_volatility']
            
            # Remove NaN
            valid = y_pred_garch.notna() & y_true_garch.notna()
            rmse_garch = np.sqrt(mean_squared_error(y_true_garch[valid], y_pred_garch[valid]))
            logging.info("RMSE GARCH Forecast: %.4f", rmse_garch)
            return rmse_garch
            
        except Exception as e:
            raise CryptoException(str(e), sys)
    def Initiate_ModelBaselines(self):
        try:
            df = self.feature_engineering.initiate_feature_engineering()
            X_train, y_train, X_test, y_test = self.traing_data(df)
            rmse_naive = self.Niive_Forcast(X_test, y_test)
            rmse_xgb, xgb_model = self.XGBOOST_Forcast(X_train, y_train, X_test, y_test)
            rmse_garch = self.GARCH_Forcast(df)
            feature_cols=['past_vol_15',
                          'volume_avg_30',
                          'high_low_ratio_5',
                          'past_return_7']
            feat_imp=pd.Series(xgb_model.feature_importances_,index=feature_cols).sort_values(ascending=False)
            plt.figure(figsize=(10,6))
            feat_imp.plot(kind='bar')
            plt.title('XGBoost Feature Importances')
            plt.ylabel('Importance Score')
            plt.show()

            models=['Naive Forecast','XGBOOST Forecast','GARCH Forecast']
            rmse_values=[rmse_naive,rmse_xgb,rmse_garch]
            plt.figure(figsize=(10,6))
            plt.bar(models,rmse_values,color=['blue','orange','green'])
            plt.title('RMSE Comparison of Models')
            plt.ylabel('RMSE')
            plt.show()
            print("Baselines: Naive={:.4f}, XGBoost={:.4f}, GARCH={:.4f}".format(
            rmse_naive, rmse_xgb, rmse_garch))

            return rmse_naive
        except Exception as e:
            raise CryptoException(str(e),sys)