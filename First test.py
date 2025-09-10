# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:14:14 2025
Getting data from yfinance

periodstr
Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max Default: 1mo Either Use period parameter or use start and end

intervalstr
Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days

start: str
Download start date string (YYYY-MM-DD) or _datetime, inclusive. Default is 99 years ago E.g. for start=”2020-01-01”, the first data point will be on “2020-01-01”

end: str
Download end date string (YYYY-MM-DD) or _datetime, exclusive. Default is now E.g. for end=”2023-01-01”, the last data point will be on “2022-12-31”

group_by: str
Group by ‘ticker’ or ‘column’ (default)

@author: jauha
"""
import datetime as dt
import yfinance as yf
import pandas as pd


stocks = ["INFY.NS", "RELIANCE.NS","SBIN.NS"]
start= dt.datetime.today()-dt.timedelta(360)
end=dt.datetime.today()

cl_price=pd.DataFrame()
ohlcv_data ={}


for ticker in stocks:
    cl_price[ticker]=yf.download(ticker,start,end,multi_level_index=False)["Close"]

for ticker in stocks:
    ohlcv_data[ticker]=yf.download(ticker,start,end,multi_level_index=False)
    
# Lookup data by row and column label ==> .loc[<row#>,<col name>]


print(ohlcv_data["SBIN.NS"]["Low"])


# Older code for testing
#data1 = yf.download("GOOG",period="1d",interval="15m",multi_level_index=False)
#data1["Adj Close"]=data1["Close"]

#NAN handling
stocks1=["META","GOOG"]
start1 = dt.datetime.today()-dt.timedelta(3650)
end1 = dt.datetime.today()
cl_price1=pd.DataFrame()
for ticker in stocks1:
    cl_price1[ticker]=yf.download(ticker,start1,end1,multi_level_index=False)["Close"]

#filling nan values
#bfill is for backfilling data from last valid value in column or in row
# axis specifies if bfill has to be across col(0), rows (1)
cl_price1.fillna(method='bfill', axis=0)