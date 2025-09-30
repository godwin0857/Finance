# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 23:37:28 2025

@author: jauha
"""

import yfinance as yf

tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]

ohlcv_data={}

for ticker in tickers:
    temp = yf.download(ticker,period='10y',interval='1d',multi_level_index=False)
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker]= temp
    


def CAGR(DF):
    df=DF.copy()
    
    # df=ohlcv_data["RELIANCE.NS"].copy()
    
    # % change of closing prices
    df["return"]=df["Close"].pct_change()
    
    #cumulative return
    df["cum_ret"]=(1+df["return"]).cumprod()
    
    
    # CAGR = (End/Start)^(1/years) - 1
    print(len(df))
    
    # n is years; if the data is hourly then divide further by 6.25 to get daily numbers
    n = len(df)/247.7
    # get last value of this series
    CAGR = (df["cum_ret"].iloc[-1])**(1/n)-1
    return CAGR

for ticker in ohlcv_data:
    print("CAGR for {} - {}".format(ticker, CAGR(ohlcv_data[ticker])))