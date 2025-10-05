# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:49:40 2025

@author: jauha
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# tickers=["RELIANCE.NS","TCS.NS","SBIN.NS"]
# tickers = ['ALLCARGO.NS', 'AXISBANK.NS', 'BANKBEES.NS', 'DELTACORP.NS', 'ENGINERSIN.NS', 'GAIL.NS', 'HDFCBANK.NS', 'INDUSINDBK.NS', 'LT.NS', 'MANALIPETC.NS', 'NATCOPHARM.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'SAIL.NS', 'SETFNIF50.NS', 'SRHHYPOLTD.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'VEDL.NS', 'ZYDUSLIFE.NS']

tickers = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BHARTIARTL.NS", "CIPLA.NS",
    "COALINDIA.NS", "DRREDDY.NS", "HCLTECH.NS", "HDFCBANK.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",  "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "NESTLEIND.NS", "IOC.NS",
    "BPCL.NS", "GAIL.NS", "VEDL.NS", "JSWSTEEL.NS", "BOSCHLTD.NS",
    "ZEEL.NS", "PFC.NS", "AMBUJACEM.NS", "ACC.NS", "BANKBARODA.NS",
    "PNB.NS", "NMDC.NS", "BHEL.NS"
]


stock_data={}

for ticker in tickers:
    stock_data[ticker]=yf.download(ticker,period='5y',interval='1d',multi_level_index=False)

    
for ticker in tickers:
    stock_data[ticker]["ret"]=stock_data[ticker]['Close']/stock_data[ticker]['Close'].shift(1)-1
    #Cum Ret of atocks
    stock_data[ticker]["log_ret"]=np.log(stock_data[ticker]['Close']/stock_data[ticker]['Close'].shift(1))
    
    # stock_data[ticker]["return"]=stock_data[ticker]['Close'].pct_change()
    # stock_data[ticker]['return_rolling_std_10'] = stock_data[ticker]['return'].rolling(window=10).std()
    # stock_data[ticker]['ret/10d_sd']=  stock_data[ticker]["return"] / stock_data[ticker]['return_rolling_std_10']
    
    # for i in range(len(stock_data[ticker])):
    #     stock_data[ticker]["10d_sd"] = stock_data[ticker]["return"].ewm(span=a, min_periods=a).sd
    
    # stock_data[ticker]['ret/10d_sd'].plot()
    # plt.show()
    
    
# days_above_4pct = {ticker: (stock_data[ticker]['return'] > 0.025 ).sum() for ticker in stock_data}
# days_below_4pct = {ticker: (stock_data[ticker]['return'] < -0.025 ).sum() for ticker in stock_data}



#EfficiencyRatio or PriceNoise
#Ratio of 8 1d ret and 8th day ret

def efficiencyRatio(df,n):
    #n'th day difference
    df['n_diff'] = abs(df['Close']-df['Close'].shift(n))
    df['1_diff'] = abs(df['Close']-df['Close'].shift(1))
    df['sum_1_diff'] = abs(df['1_diff'].rolling(window=n).sum())
    df['effRatio'] = df['n_diff']/df['1_diff'].rolling(window=n).sum()
    return df

#Plot price trend with efficiencyRatio trend
def showEffRatio(df,ticker):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis: Efficiency Ratio
    color1 = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Efficiency Ratio', color=color1)
    ax1.plot(df.index, df['effRatio'], color=color1, label='Efficiency Ratio')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Right y-axis: Closing Price
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Closing Price', color=color2)
    ax2.plot(df.index, df['Close'], color=color2, label='Closing Price')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    plt.title('{} - Efficiency Ratio vs Closing Price'.format(ticker))
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()



num_of_trading_days=247.7
tempDF=pd.DataFrame()

for ticker in tickers:
    tempDF[ticker]=stock_data[ticker]["Close"]

def calculate_return(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

log_ret=calculate_return(tempDF)

def show_stats(returns):
    print("mean of annual returns",returns.mean()*num_of_trading_days)
    print("correlation of annual returns",returns.corr())
    return returns.corr()

tempCov=show_stats(log_ret)

#Stocks with correlation>0.7
positions = np.where((tempCov > 0.7) & (tempCov<1))
labels = [(tempCov.index[i], tempCov.columns[j]) for i, j in zip(*positions)]


def plotGraph(df1,df2,ticker1,ticker2):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis: Efficiency Ratio
    color1 = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(ticker1, color=color1)
    ax1.plot(df1.index, df1['Close'], color=color1, label=ticker1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Right y-axis: Closing Price
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel(ticker2, color=color2)
    ax2.plot(df2.index, df2['Close'], color=color2, label=ticker2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    plt.title('{} vs {} '.format(ticker1,ticker2))
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()    


for label in labels:
    plotGraph(stock_data[label[0]], stock_data[label[1]], label[0], label[1])





for ticker in tickers:
    efficiencyRatio(stock_data[ticker], 10)
    # plt.figure(figsize=(10,6))
    # plt.grid(True)
    # stock_data[ticker]['effRatio'].plot()
    # plt.show()
    # print("Avg EffRatio for {} is {:.2%}".format(ticker,stock_data[ticker]['effRatio'].mean()))
    
    showEffRatio(stock_data[ticker], ticker)
    