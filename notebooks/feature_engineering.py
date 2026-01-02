from notebooks.logging.logger import logging
from notebooks.exception.Exception import CryptoException
import pandas as pd
import numpy as np
import sys
class FeatureEngineering:
    
        
    
    def load_raw_data(self):
        try:
            
            #load data
            file_path='E:/CryptoVolatility/data/raw_data/raw_data.csv'
            rdata: pd.DataFrame = pd.read_csv(file_path)
                # Must have 'Date' column
            if 'Date' not in rdata.columns:
                raise ValueError(f"'Date' column missing. Available: {list(rdata.columns)}")
        
                # Convert and set index
            rdata['Date'] = pd.to_datetime(rdata['Date'])
            rdata.set_index('Date', inplace=True)
                # Ensure UTC timezone
            if isinstance(rdata.index, pd.DatetimeIndex):
                    
                if rdata.index.tz is None:
                    rdata.index = rdata.index.tz_localize('UTC')
                else:
                    rdata.index = rdata.index.tz_convert('UTC')
            else:
                # coerce index to datetime then localize to UTC
                rdata.index = pd.to_datetime(rdata.index).tz_localize('UTC')
        
            return rdata
        except Exception as e:
            raise CryptoException(str(e),sys)

    def Compute_log_Returns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Computing log returns")
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # 1. Create log_return FIRST
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            print('close, log returns head:\n', df[['Close', 'log_return']].head())
            logging.info("Log returns computed successfully")
            
            # 2. Compute features using df (not df_clean!)
            df['past_vol_15'] = df['log_return'].rolling(window=15).std() * np.sqrt(252)
            
            # 3. Compute target volatility
            logging.info("computing target volatility")
            df['future_vol'] = df['log_return'].rolling(window=15).std().shift(-15)
            df['target_volatility'] = df['future_vol'] * np.sqrt(252)
            
            # 4. Compute other features
            logging.info("Creating lagged features")
            df['volume_avg_30'] = df['Volume'].rolling(window=30).mean()
            df['high_low_ratio_5'] = ((df['High'] - df['Low']) / df['Close']).rolling(window=5).mean()
            df['past_return_7'] = df['log_return'].rolling(window=7).sum()
            
            # 5. Final clean: drop rows where features OR target are NaN
            feature_cols = ['past_vol_15', 'volume_avg_30', 'high_low_ratio_5', 'past_return_7']
            df_final = df.dropna(subset=feature_cols + ['target_volatility'])
            
            logging.info("Lagged features created successfully")
            return df_final
            
        except Exception as e:
            raise CryptoException(str(e), sys)


    def initiate_feature_engineering(self):
        try:
            df=self.load_raw_data()
            df_final=self.Compute_log_Returns(df)
            return df_final
        except Exception as e:
            raise CryptoException(str(e),sys)