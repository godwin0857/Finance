# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 21:00:38 2025

@author: jauha
"""

import MetaTrader5 as mt5
import os
import datetime as dt
import pandas as pd

os.chdir(r"C:\Projects\algoTrading") # change working directory
key=open("MT_key.txt","r").read().split() #split the content from the file into list of strings
path=r"C:\Program Files\MetaTrader 5\terminal64.exe"

# establish MetaTrader 5 connection to a specified trading account
if not mt5.initialize(path=path, login=int(key[0]), server=key[2], password=key[1]):
    print("initialize() failed, error code =",mt5.last_error())
else:
    print("Connection established...")


symbol="EURUSD"

hist_data= mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, dt.datetime(2025,9,28), 100) #Get the 15min data for the symbol

ohlc_data=pd.DataFrame(hist_data) #Convert the ohlc response to dataframe

ohlc_data.time=pd.to_datetime(ohlc_data.time,unit="s") # change the format of time column

ohlc_data.set_index("time", inplace=True) # set the datatime as the index and reflect the changes instantly with inplace


##placing market order
def market_order(symbol, vol):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": vol,
        "price": mt5.symbol_info_tick(symbol).ask,
        "type": mt5.ORDER_TYPE_BUY,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    order_status=mt5.order_send(request)
    return order_status

# market_order(symbol, .01)

def limit_order(symbol, vol, buysell, pip_away): #buy sell to pass the trade type and pip away to set the limit order price
# =============================================================================
#     symbol="EURUSD"
#     buysell="buy"
#     pip_away=10
#     vol=0.01
# =============================================================================
    pip_unit = 10 * mt5.symbol_info(symbol).point # get the pip value of the symbol
    price=0
    
    if buysell.capitalize()[0]=="B":
        direction = mt5.ORDER_TYPE_BUY_LIMIT #Type for LIMIT order
        price = mt5.symbol_info_tick(symbol).ask - pip_unit * pip_away
        
    else:
        direction = mt5.ORDER_TYPE_SELL_LIMIT #Type for LIMIT order
        price = mt5.symbol_info_tick(symbol).ask + pip_unit * pip_away
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING, #Action for LIMIT order
        "symbol": symbol,
        "volume": vol,
        "price": price,
        "type": direction,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    print(request)
    
    order_status=mt5.order_send(request)
    return order_status

limit_order(symbol, .01, "buy", 10)