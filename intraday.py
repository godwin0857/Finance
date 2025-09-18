# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 03:18:50 2025

@author: jauha
"""

import yfinance as yf
import numpy as np
import copy
import time
import quandl as qd
import eodhd as ed

def ATR(DF, n=14):
    df = DF.copy()
    df["H-L"] = df["High"]-df["Low"]
    df["H-PC"] = df["High"]-df["Close"].shift(1)
    df["L-PC"] = df["Low"]-df["Close"].shift(1)
    #Set axis to calculate the max across rows
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1,skipna=False)
    #Can use com instead of span in case of high divergence vs charting tool - kite, tradingview
    # df["ATR"] = df["TR"].ewm(span=n,min_periods=n).mean()
    df["ATR"] = df["TR"].rolling(n).mean()
    df2=df.drop(["H-L","H-PC","L-PC"],axis=1)
    return df2["ATR"]

#Calculate the cumulative annual growth rate of the strategy
def CAGR(DF):
    df=DF.copy()
    df.dropna(inplace=True)
    df["cum_ret"]=(1+df["ret"]).cumprod()
    n=len(df)/(247.7*75) #75 5min candles in a day
    cagr=(df["cum_ret"].iloc[-1]**(1/n))-1
    return cagr

def volatility(DF):
    df=DF.copy()
    # df = ohlcv_data["RELIANCE.NS"].copy()
    df["ret"]=df["Close"].pct_change()
    df["ret"].dropna()                          #remove nan from ret
    vol = df["ret"].std() * np.sqrt(247.7*75)
    return vol

def sharpe(DF,rf):
    df=DF.copy()
    sharpe=(CAGR(df)-rf)/volatility(df)
    return sharpe

def maxDD(DF):
    df=DF.copy()
    df["cum_ret"]=(1+df["ret"]).cumprod()
    #take a cummulative max to compare with each cum_ret and get the difference    
    df["cum_rolling_max"]=df["cum_ret"].cummax()
    df["DD"]=df["cum_rolling_max"]-df["cum_ret"]
    maxDD = (df["DD"]/df["cum_rolling_max"]).max()
    return maxDD

tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]



ohlcv_intra={}
request_counter=0
start_time=time.time()
for ticker in tickers:
    temp = yf.download(ticker,period='20d',interval='5m',multi_level_index=False)
    temp.index = temp.index.tz_convert('Asia/Kolkata')
    #Dataframe function to filter by time duration
    temp=temp.between_time("9:20","15:30")
    request_counter+=1
    temp.dropna(how="any",inplace=True)
    ohlcv_intra[ticker]= temp
    # if request_counter==5:
    #     request_counter=1
    #     time.sleep(60-(time.time()-start_time))

# for ticker in ohlcv_data:
#     ohlcv_data[ticker].index = ohlcv_data[ticker].index.tz_convert('Asia/Kolkata')


# for ticker in ohlcv_data:
#     ohlcv_data[ticker][["ATR"]] = ATR(ohlcv_data[ticker])
    
ohlcv_dict={}

#calculating the atr adn rolling max price
ohlcv_dict=copy.deepcopy(ohlcv_intra)
tickers_signal={}
tickers_ret={}

for ticker in tickers:
    print("Calculating ATR and rolling max price for ",ticker)
    ohlcv_dict[ticker]["ATR"]=ATR(ohlcv_dict[ticker],20)
    #max of last 20 periods' rolling high
    ohlcv_dict[ticker]["roll_max_close"]=ohlcv_dict[ticker]["High"].rolling(20).max()
    #min of last 20 periods' rolling low
    ohlcv_dict[ticker]["roll_min_close"]=ohlcv_dict[ticker]["Low"].rolling(20).min()
    ohlcv_dict[ticker]["roll_max_vol"]=ohlcv_dict[ticker]["Volume"].rolling(20).max()
    ohlcv_dict[ticker].dropna(inplace=True)
    
    tickers_signal[ticker]="" # add this ticker in signal; will store the buy or sell signal here
    tickers_ret[ticker]=[]     # add this ticker in return

sell_price=0
buy_price=0

for ticker in tickers:
    print("Calsulating returns for ", ticker)
    for i in range (len(ohlcv_dict[ticker])): # running for all 5min candles
        if tickers_signal[ticker]=="":
            tickers_ret[ticker].append(0) # if there is no signal then return 0 in return
            if ohlcv_dict[ticker]["High"][i]>=ohlcv_dict[ticker]["roll_max_close"][i] \
               and ohlcv_dict[ticker]["Volume"][i]>1.5*ohlcv_dict[ticker]["roll_max_vol"][i-1]:
                   tickers_signal[ticker]="Buy"
            elif ohlcv_dict[ticker]["Low"][i]<=ohlcv_dict[ticker]["roll_min_close"][i] \
               and ohlcv_dict[ticker]["Volume"][i]>1.5*ohlcv_dict[ticker]["roll_max_vol"][i-1]:
                   tickers_signal[ticker]="Sell"
            
        elif tickers_signal=="Buy":
            #Stop Loss being hit
            #SL is relative with 1 ATR below the last closing i.e. trailing as per last candle
            if ohlcv_dict[ticker]["Low"][i]<ohlcv_dict[ticker]["Close"][i-1] - ohlcv_dict[ticker]["ATR"][i-1]:
                tickers_signal=""
                sell_price = ohlcv_dict[ticker]["Close"][i-1] - ohlcv_dict[ticker]["ATR"][i-1]
                buy_price = ohlcv_dict[ticker]["Close"][i-1]
                tickers_ret[ticker].append((ohlcv_dict[ticker]["Close"][i-1] - ohlcv_dict[ticker]["ATR"][i-1])/ohlcv_dict[ticker]["Close"][i-1])
            
            if ohlcv_dict[ticker]["Low"][i]<ohlcv_dict[ticker]["Close"][i-1] - ohlcv_dict[ticker]["ATR"][i-1]:
                tickers_signal=""    




