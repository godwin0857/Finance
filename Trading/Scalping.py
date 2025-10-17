# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:31:37 2025

@author: jauha
"""

# -*- coding: utf-8 -*-
"""
Intraday Scalping Strategy
Combines multiple indicators for high-probability scalping setups
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== TECHNICAL INDICATORS ====================

def ATR(DF, n=14):
    """Calculate Average True Range"""
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].rolling(n).mean()
    return df["ATR"]

def MACD(DF, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    df = DF.copy()
    df["ma_fast"] = df["Close"].ewm(span=fast, min_periods=fast).mean()
    df["ma_slow"] = df["Close"].ewm(span=slow, min_periods=slow).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=signal, min_periods=signal).mean()
    df["histogram"] = df["macd"] - df["signal"]
    return df[["macd", "signal", "histogram"]]

def RSI(DF, n=14):
    """Calculate Relative Strength Index"""
    df = DF.copy()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def VWAP(DF):
    """Calculate Volume Weighted Average Price"""
    df = DF.copy()
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap'] = (df['typical_price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df['vwap']

def SuperTrend(DF, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    df = DF.copy()
    atr = ATR(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            if df['Close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend, direction

# ==================== SCALPING SIGNAL DETECTION ====================

def identify_scalping_signals(df):
    """
    Identify scalping opportunities based on multiple criteria:
    1. Price crossing VWAP
    2. MACD histogram momentum
    3. RSI for overbought/oversold
    4. Volume surge
    5. SuperTrend direction
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
    signals['signal_strength'] = 0.0
    signals['stop_loss'] = 0.0
    signals['target'] = 0.0
    signals['reason'] = ''
    
    # Calculate all indicators
    df['ATR'] = ATR(df, 14)
    df['RSI'] = RSI(df, 14)
    df['VWAP'] = VWAP(df)
    macd_df = MACD(df)
    df['MACD'] = macd_df['macd']
    df['MACD_signal'] = macd_df['signal']
    df['MACD_hist'] = macd_df['histogram']
    df['SuperTrend'], df['ST_direction'] = SuperTrend(df, 10, 3)
    
    # Volume analysis
    df['vol_sma'] = df['Volume'].rolling(20).mean()
    df['vol_surge'] = df['Volume'] > (df['vol_sma'] * 1.5)
    
    # Price momentum
    df['price_momentum'] = df['Close'].pct_change(3) * 100
    
    for i in range(30, len(df)):
        score = 0
        reasons = []
        
        # Bullish conditions
        bullish_conditions = 0
        if df['Close'].iloc[i] > df['VWAP'].iloc[i] and df['Close'].iloc[i-1] <= df['VWAP'].iloc[i-1]:
            bullish_conditions += 1
            reasons.append("VWAP_cross_up")
        
        if df['MACD_hist'].iloc[i] > 0 and df['MACD_hist'].iloc[i] > df['MACD_hist'].iloc[i-1]:
            bullish_conditions += 1
            reasons.append("MACD_momentum")
        
        if df['RSI'].iloc[i] > 50 and df['RSI'].iloc[i] < 70:
            bullish_conditions += 1
            reasons.append("RSI_bullish")
        
        if df['vol_surge'].iloc[i]:
            bullish_conditions += 1
            reasons.append("volume_surge")
        
        if df['ST_direction'].iloc[i] == 1:
            bullish_conditions += 1
            reasons.append("SuperTrend_up")
        
        # Bearish conditions
        bearish_conditions = 0
        if df['Close'].iloc[i] < df['VWAP'].iloc[i] and df['Close'].iloc[i-1] >= df['VWAP'].iloc[i-1]:
            bearish_conditions += 1
            reasons.append("VWAP_cross_down")
        
        if df['MACD_hist'].iloc[i] < 0 and df['MACD_hist'].iloc[i] < df['MACD_hist'].iloc[i-1]:
            bearish_conditions += 1
            reasons.append("MACD_bearish")
        
        if df['RSI'].iloc[i] < 50 and df['RSI'].iloc[i] > 30:
            bearish_conditions += 1
            reasons.append("RSI_bearish")
        
        if df['vol_surge'].iloc[i]:
            bearish_conditions += 1
        
        if df['ST_direction'].iloc[i] == -1:
            bearish_conditions += 1
            reasons.append("SuperTrend_down")
        
        # Generate signals (require at least 3 conditions)
        if bullish_conditions >= 3:
            signals.loc[df.index[i], 'signal'] = 1
            signals.loc[df.index[i], 'signal_strength'] = bullish_conditions / 5
            signals.loc[df.index[i], 'stop_loss'] = df['Close'].iloc[i] - (1.5 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'target'] = df['Close'].iloc[i] + (2 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'reason'] = ', '.join(reasons)
        
        elif bearish_conditions >= 3:
            signals.loc[df.index[i], 'signal'] = -1
            signals.loc[df.index[i], 'signal_strength'] = bearish_conditions / 5
            signals.loc[df.index[i], 'stop_loss'] = df['Close'].iloc[i] + (1.5 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'target'] = df['Close'].iloc[i] - (2 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'reason'] = ', '.join(reasons)
    
    return signals, df

# ==================== BACKTESTING ====================

def backtest_scalping(df, signals, initial_capital=100000):
    """Backtest the scalping strategy"""
    portfolio = pd.DataFrame(index=df.index)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    portfolio['returns'] = 0.0
    
    position = 0
    entry_price = 0
    stop_loss = 0
    target = 0
    trades = []
    
    for i in range(len(signals)):
        if position == 0:  # No position
            if signals['signal'].iloc[i] == 1:  # Buy signal
                position = 1
                entry_price = signals['price'].iloc[i]
                stop_loss = signals['stop_loss'].iloc[i]
                target = signals['target'].iloc[i]
                shares = int((initial_capital * 0.95) / entry_price)
                portfolio.loc[signals.index[i], 'cash'] = initial_capital - (shares * entry_price)
                portfolio.loc[signals.index[i], 'holdings'] = shares
                
            elif signals['signal'].iloc[i] == -1:  # Sell signal
                position = -1
                entry_price = signals['price'].iloc[i]
                stop_loss = signals['stop_loss'].iloc[i]
                target = signals['target'].iloc[i]
                shares = int((initial_capital * 0.95) / entry_price)
                portfolio.loc[signals.index[i], 'cash'] = initial_capital + (shares * entry_price)
                portfolio.loc[signals.index[i], 'holdings'] = -shares
        
        elif position == 1:  # Long position
            current_price = signals['price'].iloc[i]
            if current_price <= stop_loss or current_price >= target:
                # Close position
                exit_price = current_price
                shares = portfolio['holdings'].iloc[i-1]
                pnl = (exit_price - entry_price) * shares
                portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + (shares * exit_price)
                portfolio.loc[signals.index[i], 'holdings'] = 0
                
                trades.append({
                    'entry_time': signals.index[i-1],
                    'exit_time': signals.index[i],
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': (exit_price / entry_price - 1) * 100,
                    'reason': 'SL' if current_price <= stop_loss else 'Target'
                })
                position = 0
            else:
                portfolio.loc[signals.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
                portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        elif position == -1:  # Short position
            current_price = signals['price'].iloc[i]
            if current_price >= stop_loss or current_price <= target:
                # Close position
                exit_price = current_price
                shares = abs(portfolio['holdings'].iloc[i-1])
                pnl = (entry_price - exit_price) * shares
                portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - (shares * exit_price)
                portfolio.loc[signals.index[i], 'holdings'] = 0
                
                trades.append({
                    'entry_time': signals.index[i-1],
                    'exit_time': signals.index[i],
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': (entry_price / exit_price - 1) * 100,
                    'reason': 'SL' if current_price >= stop_loss else 'Target'
                })
                position = 0
            else:
                portfolio.loc[signals.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
                portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        # Calculate total portfolio value
        current_price = signals['price'].iloc[i]
        portfolio.loc[signals.index[i], 'total'] = (
            portfolio.loc[signals.index[i], 'cash'] + 
            portfolio.loc[signals.index[i], 'holdings'] * current_price
        )
    
    portfolio['returns'] = portfolio['total'].pct_change()
    
    return portfolio, pd.DataFrame(trades)

# ==================== PERFORMANCE METRICS ====================

def calculate_metrics(portfolio, trades_df):
    """Calculate strategy performance metrics"""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades_df) * 100
    
    avg_return = trades_df['return'].mean()
    
    returns = portfolio['returns'].dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 78) if returns.std() != 0 else 0
    
    cummax = portfolio['total'].cummax()
    drawdown = (cummax - portfolio['total']) / cummax
    max_dd = drawdown.max() * 100
    
    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_win': winning_trades['return'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl'] < 0]['return'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    }

# ==================== MAIN EXECUTION ====================

def run_scalping_strategy(ticker, period='6m', interval='1d'):
    """Run the complete scalping strategy"""
    print(f"\n{'='*60}")
    print(f"Running Scalping Strategy for {ticker}")
    print(f"{'='*60}\n")
    
    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False,multi_level_index=False)
    
    if df.empty:
        print(f"No data available for {ticker}")
        return None, None, None
    
    # Convert timezone and filter trading hours
    df.index = df.index.tz_convert('Asia/Kolkata')
    df = df.between_time("9:20", "15:25")
    df.dropna(inplace=True)
    
    if len(df) < 50:
        print(f"Insufficient data for {ticker}")
        return None, None, None
    
    # Identify signals
    signals, df_with_indicators = identify_scalping_signals(df)
    
    # Backtest
    portfolio, trades = backtest_scalping(df, signals)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio, trades)
    
    # Print results
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Return per Trade: {metrics['avg_return']:.2f}%")
    print(f"Average Win: {metrics['avg_win']:.2f}%")
    print(f"Average Loss: {metrics['avg_loss']:.2f}%")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    if len(trades) > 0:
        print(f"\nRecent Signals:")
        recent_signals = signals[signals['signal'] != 0].tail(5)
        for idx, row in recent_signals.iterrows():
            signal_type = "BUY" if row['signal'] == 1 else "SELL"
            print(f"{idx.strftime('%Y-%m-%d %H:%M')} | {signal_type} | Price: ₹{row['price']:.2f} | "
                  f"SL: ₹{row['stop_loss']:.2f} | Target: ₹{row['target']:.2f}")
            print(f"  Reason: {row['reason']}")
    
    return df_with_indicators, signals, trades

# ==================== RUN FOR MULTIPLE TICKERS ====================

if __name__ == "__main__":
    # Top liquid stocks for scalping
    tickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
        "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
    ]
    
    results = {}
    
    for ticker in tickers[:3]:  # Test with first 3 stocks
        try:
            df, signals, trades = run_scalping_strategy(ticker, period='5d', interval='5m')
            if df is not None:
                results[ticker] = {'df': df, 'signals': signals, 'trades': trades}
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Strategy execution completed!")
    print(f"{'='*60}")