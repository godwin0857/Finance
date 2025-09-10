# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:12:20 2025

@author: jauha
"""


import yfinance as yf

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
def ATR(DF, n=14):
    df = DF.copy()
    df["H-L"] = df["High"]-df["Low"]
    df["H-PC"] = df["High"]-df["Close"].shift(1)
    df["L-PC"] = df["Low"]-df["Close"].shift(1)
    #Set axis to calculate the max across rows
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1,skipna=False)
    #Can use com instead of span in case of high divergence vs charting tool - kite, tradingview
    df["ATR"] = df["TR"].ewm(span=n,min_periods=n).mean()
    return df.loc[:,["ATR"]]


for ticker in ohlcv_data:
    ohlcv_data[ticker][["ATR"]] = ATR(ohlcv_data[ticker])
    
    #Bollinger-Bands
    
def BB(DF,n=20):
    df= DF.copy()
    df["Mid_Band"]=df["Close"].rolling(n).mean()
    #To calculate the STD across entire population (n) and not sample (n-1), hence degrees of freedom = 0
    df["Up_Band"]=df["Mid_Band"]+ 2* df["Close"].rolling(n).std(ddof=0)
    df["Low_Band"]=df["Mid_Band"]- 2* df["Close"].rolling(n).std(ddof=0)
    df["Width"]=df["Up_Band"]-df["Low_Band"]
    return df[["Mid_Band","Low_Band","Up_Band","Width"]]
    
for ticker in ohlcv_data:
    ohlcv_data[ticker][["Mid_Band","Low_Band","Up_Band","Width"]] = BB(ohlcv_data[ticker])
