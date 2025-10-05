# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 20:14:56 2025

@author: jauha
"""


import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates




# =============================================================================
# def calmar(DF):
#     df=DF.copy()
#     return CAGR(df)/maxDD(df)
# =============================================================================

def volatility(DF):
    df=DF.copy()
    vol = df["mon_ret"].std() * np.sqrt(12)
    return vol

def sharpe(DF,rf):
    df=DF.copy()
    sharpe=(CAGR(df)-rf)/volatility(df)
    return sharpe

# =============================================================================
# def sortino(DF,rf):
#     df=DF.copy()
#     neg_ret=np.where(df["mon_ret"]>0,0,df["mon_ret"])
#     #remove nan from neg_ret
#     neg_ret=neg_ret[~np.isnan(neg_ret)]
#     neg_volatility= pd.Series(neg_ret[neg_ret!=0]).std()*np.sqrt(12)         #exclude 0 values from neg_ret to calculate neg_volatility
#     sortino = (CAGR(DF)-rf)/neg_volatility
#     return sortino
# =============================================================================

def maxDD(DF):
    df=DF.copy()
    df["cum_ret"]=(1+df["mon_ret"]).cumprod()
    #take a cummulative max to compare with each cum_ret and get the difference    
    df["cum_rolling_max"]=df["cum_ret"].cummax()
    df["DD"]=df["cum_rolling_max"]-df["cum_ret"]
    maxDD = (df["DD"]/df["cum_rolling_max"]).max()
    return maxDD


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

ohlcv_mdata={}
start=dt.datetime.today()-dt.timedelta(730)
end=dt.datetime.today()

for ticker in tickers:
    ohlcv_mdata[ticker]=yf.download(ticker,start,end,interval='1mo',multi_level_index=False)
    ohlcv_mdata[ticker].dropna(inplace=True,how="all")

# =============================================================================
# attempt = 0
# drop=[]
# 
# 
# while len(tickers)!=0 and attempt < 2:
#     tickers = [j for j in tickers if j not in drop]
#     for i in range(len(tickers)):
#         try:
#             ohlcv_mdata[tickers[i]] = yf.download(tickers[i],period='10y',interval='1mo',multi_level_index=False)
#             ohlcv_mdata[tickers[i]].dropna(how="any",inplace=True)
#             drop.append(tickers[i])
#             print("Try executed for {}  i = {} ".format((tickers[i]),i))
#         except:
#             print(tickers[i]," :failed to fetch data...retrying  i = {} ".format(i))
#             continue
#     attempt+=1
# =============================================================================


#Only store the stocks which are available in ohlcv mdata
tickers = ohlcv_mdata.keys()


ohlcv_dict=ohlcv_mdata.copy()
return_df=pd.DataFrame()
for ticker in tickers:
    print("Calculating monthly return for ",ticker)
    ohlcv_dict[ticker]["mon_ret"]=ohlcv_dict[ticker]["Close"].pct_change() #Added mon_ret column in the ohlcv_dict>Stocks
    return_df[ticker] = ohlcv_dict[ticker]["mon_ret"] #Storing the Ticker and Monthly Returns in Rows and Columns of this dataframe 
    return_df.dropna(inplace=True)
    
    
#Buy top 10, sell bottom 3 with unique stocks each month
def portfolio(DF,n,x):
    """This function returns cumulative portfolio return
        DF = dataframe with monthly returns of stocks
        n = number of stocks in the portfolio
        x= number or underperforming stocks to tbe removed from portfolio monthly"""
    df = DF.copy()
    df.dropna(inplace=True)
    folio =[]
    monthly_ret = []
    # print(len(df))
    for i in range (len(df)): #i is month
        # print(" i = {} and len(folio) is {}".format(i, len(folio)))
        if len(folio)>0:
            monthly_ret.append(df[folio].iloc[i,:].mean()) #get the mean of all stocks in this month
            bad_stocks = df[folio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            folio = [t for t in folio if t not in bad_stocks]
        fill = n-len(folio)
        # df[[]] to get the dataframe consisting of all columns as per t condition
        # iloc[i,:] returns the row at i'th position
        # [:fill] returns the top fill'th items
        #index.value to get their index value and finally store it in the new picks list
        new_picks = df[[t for t in tickers if t not in folio]].iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        folio = folio + new_picks
        # print(folio)
    
        # np.array(monthly_ret): Converts monthly_ret (which might be a Python list or another sequence) into a NumPy array. This makes it compatible and efficient for use with Pandas
        # pd.DataFrame(...): Builds a new Pandas DataFrame from the NumPy array. The data in monthly_ret becomes the values of the DataFrame.
        # columns=["mon_ret"]: Sets the column name for the new DataFrame to "mon_ret". If the NumPy array is one-dimensional, the DataFrame will have one column called "mon_ret"; if it’s two-dimensional, "mon_ret" names the only or first column (more column names would be needed if there are multiple columns).
        monthly_ret_df=pd.DataFrame(np.array(monthly_ret), columns=["mon_ret"])
    return monthly_ret_df

#Buy top 10, sell bottom 3 with no limitations on unique stocks each month
def portfolio1(DF,n,x):
    df = DF.copy()
    df.dropna(inplace=True)
    folio =[]
    monthly_ret = []
    i=10
    # print(len(df))
    for i in range (len(df)): #i is month
        # print(" i = {} and len(folio) is {}".format(i, len(folio)))
        if len(folio)>0:
            monthly_ret.append(df[folio].iloc[i,:].mean()) #get the mean of all stocks in this month
            bad_stocks = df[folio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            folio = [t for t in folio if t not in bad_stocks]
        fill = n-len(folio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        folio = folio + new_picks
        # print(folio)
    
        monthly_ret_df=pd.DataFrame(np.array(monthly_ret), columns=["mon_ret"])
    return monthly_ret_df

#Calculate the cumulative annual growth rate of the strategy
def CAGR(DF):
    df=DF.copy()
    df.dropna(inplace=True)
    df["cum_ret"]=(1+df["mon_ret"]).cumprod()
    n=len(df)/12
    cagr=(df["cum_ret"].iloc[-1]**(1/n))-1
    return cagr


#How did NIFTY ^NSEI perform in this period
nifty_data=yf.download(("^NSEI"),period='10y',interval='1mo',multi_level_index=False)
nifty_data["mon_ret"]=nifty_data["Close"].pct_change()

print("CAGR for the portfolio1 is {:.2%}".format(CAGR(portfolio1(return_df,10,3))))
print("Sharpe for the portfolio1 is {:.2%}".format(sharpe(portfolio1(return_df,10,3), 0.06)))
print("MaxDD for the portfolio1 is {:.2%}".format(maxDD(portfolio1(return_df,10,3))))

print("CAGR for the portfolio is {:.2%}".format(CAGR(portfolio(return_df,10,3))))
print("Sharpe for the portfolio is {:.2%}".format(sharpe(portfolio(return_df,10,3), 0.06)))
print("MaxDD for the portfolio is {:.2%}".format(maxDD(portfolio(return_df,10,3))))



print("CAGR of NIFTY is {:.2%}".format(CAGR(nifty_data)))
print("Sharpe of NIFTY is {:.2%}".format(sharpe(nifty_data, 0.06)))
print("Max DD of NIFTY is {:.2%}".format(maxDD(nifty_data)))


#Charting

fig, ax = plt.subplots()
ax.plot((1 + portfolio1(return_df, 10, 3)).cumprod())
ax.plot((1 + portfolio(return_df, 10, 3)).cumprod())
ax.plot((1+nifty_data["mon_ret"][2:].reset_index(drop=True)).cumprod())

plt.xticks(rotation=45)  # Makes labels more readable
plt.title("Index Returns vs Strategy Returns")
plt.ylabel("Cumulative Return")
plt.xlabel("Months")
ax.legend(["Strategy1 Returns","Strategy Return","Index Returns"])
plt.show()

