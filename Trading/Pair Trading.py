# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 11:23:38 2025

@author: jauha
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

stock_data={}
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
for ticker in tickers:
    stock_data[ticker]=yf.download(ticker,period='10y',interval='1d',multi_level_index=False)
    stock_data[ticker].bfill()
    # stock_data[ticker].index = stock_data[ticker].index.tz_convert('Asia/Kolkata')
    #Dataframe function to filter by time duration - USE ONLY WHEN HANDLING INTRADAY DATA
    # stock_data[ticker]=stock_data[ticker].between_time("9:20","15:30")



#Align the indices of both the stocks
def align_stock_data(df1, df2):
    # Find common timestamps
    common_idx = df1.index.intersection(df2.index)

    # Reindex both DataFrames with common timestamps
    df1_aligned = df1.reindex(common_idx)
    df2_aligned = df2.reindex(common_idx)

    # Optional: drop rows with missing data after reindexing
    df1_aligned = df1_aligned.dropna(subset=['Close'])
    df2_aligned = df2_aligned.dropna(subset=['Close'])

    # Re-align once more if dropping NaNs removed different rows
    common_idx_post = df1_aligned.index.intersection(df2_aligned.index)
    df1_aligned = df1_aligned.reindex(common_idx_post)
    df2_aligned = df2_aligned.reindex(common_idx_post)

    return df1_aligned, df2_aligned



def getCov(stock_dict,corr_threshold):
    tempDF=pd.DataFrame()
    stock_dict=stock_data
    for ticker in tickers:
        tempDF[ticker]=stock_dict[ticker]["Close"]


    log_return = np.log(tempDF/tempDF.shift(1))
    log_return = log_return[1:]

    tempCov=log_return.corr()

    #Stocks with correlation>0.7
    positions = np.where((tempCov > corr_threshold) & (tempCov<1))
    labels = [(tempCov.index[i], tempCov.columns[j]) for i, j in zip(*positions)]
    return labels

def calc_raw_stochastic(close, n):
    min_L = close.rolling(window=n).min()
    max_H = close.rolling(window=n).max()
    return ((close - min_L) / (max_H - min_L)) * 100

def max_drawdown(pnl_curve):
    cummax = np.maximum.accumulate(pnl_curve)
    drawdown = cummax - pnl_curve
    return np.max(drawdown)

def calculate_entry_threshold(spread, percentile):
    # Calculate the absolute spread excluding NaNs
    abs_spread = spread.abs().dropna()
    # Entry threshold is the specified percentile of absolute spread values
    entry_threshold = np.percentile(abs_spread, percentile)
    return entry_threshold

def calculate_ATR(df, n=14):
    """
    Calculate the Average True Range (ATR) for a stock DataFrame.
    Assumes df has columns ['High', 'Low', 'Close'].
    n: period for ATR calculation
    Returns a Series of ATR values aligned with df index.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (close.shift(1) - high).abs()
    tr3 = (close.shift(1) - low).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=n, min_periods=1).mean()
    return atr



def plot_stochastic_spread(spread, entry_thr, exit_thr=0, title="Stochastic Spread with Entry/Exit Thresholds"):
    plt.figure(figsize=(14,6))
    plt.plot(spread.index, spread, label='Stochastic Spread (Stock1 - Stock2)')
    plt.axhline(entry_thr, color='red', linestyle='--', label=f'Entry Threshold (+{entry_thr:.2f})')
    plt.axhline(-entry_thr, color='red', linestyle='--', label=f'Entry Threshold (-{entry_thr:.2f})')
    plt.axhline(exit_thr, color='green', linestyle='-', label=f'Exit Threshold ({exit_thr:.2f})')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Spread Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def pair_trading_backtest_adaptive_threshold(stock_dict, stock1, stock2,
                                             sto_n,entry_thr,exit_thr,
                                             stop_loss=0.01, profit_take=0.05,
                                             threshold_percentile=80,atr_period=10):

    df1, df2 = align_stock_data(stock_dict[stock1], stock_dict[stock2])
    
    sto1 = calc_raw_stochastic(df1['Close'], n=sto_n)
    sto2 = calc_raw_stochastic(df2['Close'], n=sto_n)
    spread = sto1 - sto2

    # Calculate adaptive entry threshold based on the relative trend distribution
    # entry_thr = calculate_entry_threshold(spread, percentile=threshold_percentile)
    # print(f"Adaptive entry threshold (at {threshold_percentile} percentile): {entry_thr:.2f}")
    
    plot_stochastic_spread(spread, entry_thr, 0, title=f"Stochastic Spread for {stock1} vs {stock2}")

    # Calculate ATR for both stocks
    atr1 = calculate_ATR(df1, n=atr_period)
    atr2 = calculate_ATR(df2, n=atr_period)


    trades = []
    open_trade = None
    pnl_series = []

    for i in range(sto_n, len(spread)):
        if open_trade is None:
            if spread.iloc[i] > entry_thr:
                qty1 = atr2.iloc[i]  # qty proportional to other stock's ATR
                qty2 = atr1.iloc[i]
                open_trade = {
                    'entry_date': df1.index[i],
                    'side': 'short1_long2',
                    'entry_price1': df1['Close'].iloc[i],
                    'entry_price2': df2['Close'].iloc[i],
                    'qty1': qty1,
                    'qty2': qty2
                }
            elif spread.iloc[i] < -entry_thr:
                qty1 = atr2.iloc[i]
                qty2 = atr1.iloc[i]
                open_trade = {
                   'entry_date': df1.index[i],
                   'side': 'long1_short2',
                   'entry_price1': df1['Close'].iloc[i],
                   'entry_price2': df2['Close'].iloc[i],
                   'qty1': qty1,
                   'qty2': qty2
                }
        else:
            price1_now = df1['Close'].iloc[i]
            price2_now = df2['Close'].iloc[i]
            if open_trade['side'] == 'short1_long2':
                pnl = open_trade['qty1'] * (open_trade['entry_price1'] - price1_now) + \
                      open_trade['qty2'] * (price2_now - open_trade['entry_price2'])
            else:# long1_short2
                pnl = open_trade['qty1'] * (price1_now - open_trade['entry_price1']) + \
                      open_trade['qty2'] * (open_trade['entry_price2'] - price2_now)

            entry_spread = abs(open_trade['entry_price1'] - open_trade['entry_price2'])
            pnl_pct = pnl / entry_spread if entry_spread != 0 else 0
            exit_signal = (
                (open_trade['side'] == 'short1_long2' and spread.iloc[i] <= exit_thr) or
                (open_trade['side'] == 'long1_short2' and spread.iloc[i] >= -exit_thr)
            )
            stop_signal = pnl_pct <= -stop_loss
            profit_signal = pnl_pct >= profit_take
            if exit_signal or stop_signal or profit_signal or i == len(spread) - 1:
                trades.append({
                    'entry_date': open_trade['entry_date'],
                    'exit_date': df1.index[i],
                    'side': open_trade['side'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                pnl_series.append(pnl)
                open_trade = None
    
    trades_df = pd.DataFrame(trades)
    avg_return = trades_df['pnl_pct'].mean()
    gross_profits = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
    gross_losses = trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()
    net_gains = gross_profits + gross_losses
    max_dd = max_drawdown(pnl_series)
    
    print("Trading pairs of {} - {}".format(stock1,stock2))
    print("Total trades  : ",len(trades_df))
    print(f"Avg return per trade: {avg_return:.4f}")
    # print(f"Gross profits: {gross_profits:.2f}")
    # print(f"Gross losses : {gross_losses:.2f}")
    print(f"Net gains    : {net_gains:.2f}")
    # print(f"Max drawdown: {max_dd:.2f}")

    return trades_df

# Usage example:
# trades = pair_trading_backtest_adaptive_threshold(stock_dict, 'AAPL', 'MSFT')

def _total_net_gains(trades_df: pd.DataFrame) -> float:
    # Sum of pnl column; empty DataFrame yields 0.0
    if trades_df is None or trades_df.empty:
        return 0.0
    return float(trades_df['pnl'].sum())

def _pair_correlation_and_years(stock_dict, stock1: str, stock2: str):
    # Align and compute correlation on log returns; also return span in years
    df1, df2 = align_stock_data(stock_dict[stock1], stock_dict[stock2])
    close_df = pd.DataFrame({stock1: df1['Close'], stock2: df2['Close']})
    log_ret = np.log(close_df / close_df.shift(1)).dropna()
    corr = float(log_ret[stock1].corr(log_ret[stock2])) if not log_ret.empty else np.nan
    if len(close_df.index) >= 2:
        days_span = (close_df.index[-1] - close_df.index[0]).days
        years_span = days_span / 247.7 if days_span > 0 else 0.0
    else:
        years_span = 0.0
    return corr, years_span

def _information_ratio(trades_df: pd.DataFrame, years_span: float) -> float:
    # IR = annualized return / annualized volatility using trade returns approximation
    if trades_df is None or trades_df.empty or years_span <= 0:
        return 0.0
    trade_returns = trades_df['pnl_pct'].dropna()
    if trade_returns.empty:
        return 0.0
    trades_per_year = len(trade_returns) / years_span
    mean_per_trade = trade_returns.mean()
    std_per_trade = trade_returns.std(ddof=1)
    if std_per_trade == 0 or trades_per_year <= 0:
        return 0.0
    ann_return = mean_per_trade * trades_per_year
    ann_vol = std_per_trade * np.sqrt(trades_per_year)
    return float(ann_return / ann_vol) if ann_vol != 0 else 0.0


allTrades20_80_0=pd.DataFrame()
# allTradesIn20_80_0={}
# allTrades["HINDALCO.NS - TATASTEEL.NS"]=pair_trading_backtest_adaptive_threshold(stock_data, 'HINDALCO.NS', 'TATASTEEL.NS')
# allTrades['JSWSTEEL.NS - TATASTEEL.NS'] = pair_trading_backtest_adaptive_threshold(stock_data, 'JSWSTEEL.NS', 'TATASTEEL.NS')
# allTrades['BANKBARODA.NS - SBIN.NS'] = pair_trading_backtest_adaptive_threshold(stock_data, 'BANKBARODA.NS', 'SBIN.NS')
# allTrades['BANKBARODA.NS - PNB.NS'] = pair_trading_backtest_adaptive_threshold(stock_data, 'BANKBARODA.NS', 'PNB.NS')
# trades = pair_trading_backtest_adaptive_threshold(stock_data, 'ACC.NS', 'AMBUJACEM.NS')
# trades = pair_trading_backtest_adaptive_threshold(stock_data, 'BPCL.NS', 'IOC.NS')

highCovarPair = getCov(stock_data, 0.7)
processed = set()

# Single dictionary keyed by pair_key with requested columns
pair_comparison = {}
sto_periods = [20, 14, 10, 7]

for pair in highCovarPair:
    pair_f = frozenset(pair)
    if (pair_f in processed):
        continue
    pair_l=list(pair_f)
    pair_key = "{} - {}".format(pair_l[0],pair_l[1])

    corr, years_span = _pair_correlation_and_years(stock_data, pair_l[0], pair_l[1])

    # Initialize row dict with correlation
    row = {
        'correlation': corr
    }

    # Per-period stats
    for sto_n in sto_periods:
        trades_df = pair_trading_backtest_adaptive_threshold(stock_data, pair_l[0], pair_l[1], sto_n, 80, 0)
        row[f'net_gains_{sto_n}'] = _total_net_gains(trades_df)
        row[f'num_trades_{sto_n}'] = int(0 if trades_df is None else len(trades_df))
        row[f'information_ratio_{sto_n}'] = _information_ratio(trades_df, years_span)

    pair_comparison[pair_key] = row

    # Concise summary to console
    print(f"Summary for {pair_key} (entry 80, exit 0): corr={corr:.3f}")
    for sto_n in sto_periods:
        print(
            f"  sto_n={sto_n}: net_gains={row[f'net_gains_{sto_n}']:.2f}, "
            f"trades={row[f'num_trades_{sto_n}']}, IR={row[f'information_ratio_{sto_n}']:.3f}"
        )

    processed.add(pair_f)