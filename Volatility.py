# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 02:23:42 2025

@author: jauha
"""

import yfinance as yf
import numpy as np

tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]

ohlcv_data={}

for ticker in tickers:
    temp = yf.download(ticker,period='10y',interval='1d',multi_level_index=False)
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker]= temp
    

def volatility(DF):
    df=DF.copy()
    # df = ohlcv_data["RELIANCE.NS"].copy()
    df["ret"]=df["Close"].pct_change()
    vol = df["ret"].std() * np.sqrt(252)
    return vol

for ticker in ohlcv_data:
    print("Volatility of {} - {}".format(ticker, volatility(ohlcv_data[ticker])))