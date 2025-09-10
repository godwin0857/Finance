# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:13:44 2025

@author: jauha
"""

import datetime as dt
import yfinance as yf
import pandas as pd


stocks = ["INFY.NS", "RELIANCE.NS","SBIN.NS"]
start= dt.datetime.today()-dt.timedelta(3650)
end=dt.datetime.today()

cl_price=pd.DataFrame()

for ticker in stocks:
    cl_price[ticker]=yf.download(ticker,start,end,multi_level_index=False)["Close"]

#filling nan values
#bfill is for backfilling data from last valid value in column or in row
# axis specifies if bfill has to be across col(0), rows (1)
cl_price.dropna(axis=0,how='any',inplace=True)

cl_price.mean()
cl_price.std()
cl_price.median()
cl_price.describe()

#display first/last 5 rows of dataframe
cl_price.head()
cl_price.tail()

#get DoD % change
daily_return=cl_price.pct_change()

#shift moves the data by 1 position
#cl_price/cl_price.shift(1)-1

daily_return.mean(axis=1)


#moving values with rolling window; min period is to calculate the rolling number even with few nans
daily_return.rolling(window=10).mean()
daily_return.rolling(window=10).median()
daily_return.rolling(window=10).max()
daily_return.rolling(window=10).sum()

#exponential weighted operations
#weightage/value decays as you move from latest to oldest
daily_return.ewm(com=10, min_periods=10).mean()

#plotting the graph
#sharing the axis - x(default True)/y (default False)
cl_price.plot(subplots=True, layout=(2,2))
daily_return.plot(subplots=True, layout=(2,2))

# cumulative return by multiplying the 1+daily return with cumulative product
(1 + daily_return).cumprod().plot()