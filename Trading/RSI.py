# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:47:14 2025

@author: jauha
"""



import yfinance as yf
import numpy as np

tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]

ohlcv_data={}

for ticker in tickers:
    temp = yf.download(ticker,period='1mo',interval='15m',multi_level_index=False)
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker]= temp

df = ohlcv_data["RELIANCE.NS"]

print(ohlcv_data["RELIANCE.NS"].index.tz)  # None means naive; otherwise shows timezone

for ticker in ohlcv_data:
    ohlcv_data[ticker].index = ohlcv_data[ticker].index.tz_convert('Asia/Kolkata')


#function to calculate macd
#true range = max of high/low, 
def RSI(DF, n=14):
    df = DF.copy()
    #Get the daily change vs previous period
    df["change"]=df["Close"]-df["Close"].shift(1)
    #Get the gains; np.where is similar to if function of excel
    df["gains"] = np.where(df["change"]>=0,df["change"],0)
    df["loss"] = np.where(df["change"]<0,-1*df["change"],0)
    df["avg_g"] = df["gains"].ewm(alpha=1/n,min_periods=n).mean()
    df["avg_l"] = df["loss"].ewm(alpha=1/n,min_periods=n).mean()
    df["rs"]=df["avg_g"]/df["avg_l"]
    df["rsi"]=100-(100/(1+df["rs"]))
    return df["rsi"]


#Adding a new column in the main dataframe
for ticker in ohlcv_data:
    ohlcv_data[ticker]["RSI"] = RSI(ohlcv_data[ticker])
  