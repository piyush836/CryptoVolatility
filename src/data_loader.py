
from notebooks.logging.logger import logging
from notebooks.exception.Exception import CryptoException
import yaml
import pandas as pd
import sys
import yfinance as f  # type: ignore
import os
class DataLoader:
    def read_ymal(self):
        config_path = 'config.yaml'
        with open(config_path,'r')as file:
            data=yaml.safe_load(file)
            return data
    def read_yahoo(self):
        try:
            data = self.read_ymal()
            # download data from yfinance 
            tickers=data['data']['symbol']
            start=data['data']['start_date']
            end=data['data']['end_date']
            interval=data['data']['interval']
            btc=f.download(tickers,start,end,interval,auto_adjust=False,progress=False)
            print("Columns before flatting:",btc.columns.to_list())
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns= btc.columns.get_level_values(0)
            if len(btc.columns) > 1 and all(str(col) == str(btc.columns[0]) for col in btc.columns):
                standard_names = ['Open', 'High', 'Low', 'Close', 'Volume']
                btc.columns = standard_names[:len(btc.columns)]
        
            print("Columns after flatting:",btc.columns.to_list())
            if 'Close' not in btc.columns:
                raise ValueError(f"Still no 'Close' column. Final columns: {btc.columns.tolist()}")
            #reset the index to have Date as a column
            btc.reset_index(inplace=True)
            btc=btc.dropna(subset=['Close'])
            price_cols = ['Open', 'High', 'Low', 'Adj Close']
            for col in price_cols:
                if col in btc.columns:
                    btc[col] = pd.to_numeric(btc[col], errors='coerce')

            # Final drop of any remaining bad rows
            btc = btc.dropna(subset=['Close'])
                    

            #save to csv
            
            btc.to_csv(os.path.join('data/raw_data','raw_data.csv'),index=False)
            logging.info("Data downloaded and saved successfully")

        except Exception as e:
            raise CryptoException(str(e),sys)    

    def initiateDataloader(self):
        try:
            self.read_yahoo()
           
        except Exception as e:
            raise CryptoException(str(e),sys)



