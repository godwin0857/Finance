# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 18:19:46 2025

@author: jauha
"""


#importing libraries

from alpha_vantage.timeseries import TimeSeries
import pandas as pd

#python library to keep track of system time
import time

key_path = "C:\\Users\\jauha\\anaconda3\\envs\\algoTrading\\Projects\\Alpha vantage API key.txt"

#Extracting data from single ticker
#ts=TimeSeries(key=open(key_path,'r').read(),output_format='pandas')
#data=ts.get_daily(symbol="EURUSD",outputsize="full")[0]
#data.columns = ["open","high","low","close","vol"]

#
data=data.iloc[::-1]


all_tickers= ["ACHR","NVDA","RKLB","TSMC"]
close_price= pd.DataFrame()

# to keep track of api calls
api_call_count=0

for ticker in all_tickers:
    #capturing the system time
    start_time = time.time()
    ts=TimeSeries(key=open(key_path,'r').read(),output_format='pandas')
    data1=ts.get_intraday(symbol=ticker,interval='1min',outputsize='compact')[0]
    api_call_count+=1
    data1.columns=["open","high","low","close","volume"]
    close_price[ticker]=data1["close"]
    # wait for atleast 60s from first api call to stay within the 5 request per minute limit
    if api_call_count==5:
        api_call_count=0
        time.sleep(60-(time.time() - start_time))