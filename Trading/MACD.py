# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:37:43 2025

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
def MACD(DF, a=12, b=26, c=9):
    df = DF.copy()
    df["ma_fast"] = df["Close"].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df["Close"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    return df.loc[:,["macd","signal"]]


for ticker in ohlcv_data:
    ohlcv_data[ticker][["MACD","SIGNAL"]] = MACD(ohlcv_data[ticker])