# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:38:19 2025

@author: jauha
"""

import yfinance as yf
import numpy as np
import pandas as pd

tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]

ohlcv_data={}

for ticker in tickers:
    temp = yf.download(ticker,period='1y',interval='1d',multi_level_index=False)
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker]= temp
  
DF = ohlcv_data["RELIANCE.NS"]
ticker="RELIANCE.NS"


def maxDD(DF):
    df=DF.copy()
    df["return"]=df["Close"].pct_change()
    df["cum_ret"]=(1+df["return"]).cumprod()
    # max=df["cum_ret"].max()
#take a cummulative max to compare with each cum_ret and get the difference    
    df["cum_rolling_max"]=df["cum_ret"].cummax()

    df["DD"]=df["cum_rolling_max"]-df["cum_ret"]
    maxDD = (df["DD"]/df["cum_rolling_max"]).max()
    return maxDD
        
def CAGR(DF):
    df=DF.copy()
    df["ret"]=df["Close"].pct_change()
    df["cum_ret"]=(1+df["ret"]).cumprod()
    n=len(df)/247.7
    cagr=((df["cum_ret"].iloc[-1])**(1/n))-1
    return cagr

def calmar(DF):
    df=DF.copy()
    return CAGR(df)/maxDD(df)

for ticker in ohlcv_data:
    print("Max drawdown of {} is {:.2%}".format(ticker, maxDD(ohlcv_data[ticker])))
    print("Calmar of {} is {:.2%}".format(ticker, calmar(ohlcv_data[ticker])))
    