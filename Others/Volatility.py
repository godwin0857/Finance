# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 02:23:42 2025

@author: jauha
"""

import yfinance as yf
import numpy as np
import pandas as pd

tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]

ohlcv_data={}

for ticker in tickers:
    temp = yf.download(ticker,period='10y',interval='1d',multi_level_index=False)
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker]= temp
  
DF = ohlcv_data["RELIANCE.NS"]
ticker="RELIANCE.NS"

def CAGR(DF):
    df=DF.copy()
    df["ret"]=df["Close"].pct_change()
    df["cum_ret"]=(1+df["ret"]).cumprod()
    n=len(df)/247.7
    cagr=((df["cum_ret"].iloc[-1])**(1/n))-1
    return cagr


def volatility(DF):
    df=DF.copy()
    # df = ohlcv_data["RELIANCE.NS"].copy()
    df["ret"]=df["Close"].pct_change()
    df["ret"].dropna()                          #remove nan from ret
    vol = df["ret"].std() * np.sqrt(247.7)
    return vol

def sharpe(DF,rf):
    df=DF.copy()
    sharpe=(CAGR(df)-rf)/volatility(df)
    return sharpe

def sortino(DF,rf):
    df=DF.copy()
    df["return"]=df["Close"].pct_change()
    neg_ret=np.where(df["return"]>0,0,df["return"])
    #remove nan from neg_ret
    neg_ret=neg_ret[~np.isnan(neg_ret)]
    neg_volatility= pd.Series(neg_ret[neg_ret!=0]).std()*np.sqrt(247.7)         #exclude 0 values from neg_ret to calculate neg_volatility
    sortino = (CAGR(DF)-rf)/neg_volatility
    return sortino

for ticker in ohlcv_data:
    # print("CAGR ratio of {} = {:.2%}".format(ticker, CAGR(ohlcv_data[ticker])))
    # print("Volatility ratio of {} = {:.2%}".format(ticker, CAGR(ohlcv_data[ticker])))
    print("Sharpe ratio of {} = {:.2%}".format(ticker, sharpe(ohlcv_data[ticker], .065)))
    print("Sortino ratio of {} = {:.2%}".format(ticker, sortino(ohlcv_data[ticker], .065)))
    
    