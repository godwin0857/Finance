# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:42:32 2025

@author: jauha
"""

# -*- coding: utf-8 -*-
"""
Swing Trading Strategy - Daily Timeframe
Multi-indicator strategy for medium-term position trading
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

def Bollinger_Bands(DF, n=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df = DF.copy()
    df['BB_middle'] = df['Close'].rolling(window=n).mean()
    df['BB_std'] = df['Close'].rolling(window=n).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (std_dev * df['BB_std'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    return df[['BB_upper', 'BB_middle', 'BB_lower', 'BB_width']]

def ADX(DF, n=14):
    """Calculate Average Directional Index (trend strength)"""
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                             df['High'] - df['High'].shift(1), 0)
    df['DMplus'] = np.where(df['DMplus'] < 0, 0, df['DMplus'])
    
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                              df['Low'].shift(1) - df['Low'], 0)
    df['DMminus'] = np.where(df['DMminus'] < 0, 0, df['DMminus'])
    
    df['TR_smooth'] = df['TR'].rolling(window=n).sum()
    df['DMplus_smooth'] = df['DMplus'].rolling(window=n).sum()
    df['DMminus_smooth'] = df['DMminus'].rolling(window=n).sum()
    
    df['DIplus'] = 100 * (df['DMplus_smooth'] / df['TR_smooth'])
    df['DIminus'] = 100 * (df['DMminus_smooth'] / df['TR_smooth'])
    
    df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
    df['ADX'] = df['DX'].rolling(window=n).mean()
    
    return df['ADX']

def Stochastic(DF, k=14, d=3):
    """Calculate Stochastic Oscillator"""
    df = DF.copy()
    df['L14'] = df['Low'].rolling(window=k).min()
    df['H14'] = df['High'].rolling(window=k).max()
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=d).mean()
    return df[['%K', '%D']]

def OBV(DF):
    """Calculate On-Balance Volume"""
    df = DF.copy()
    df['price_change'] = df['Close'].diff()
    df['direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))
    df['obv'] = (df['Volume'] * df['direction']).cumsum()
    return df['obv']

# ==================== SWING TRADING SIGNAL DETECTION ====================

def identify_swing_signals(df):
    """
    Identify swing trading opportunities based on daily data:
    1. MACD crossovers and divergence
    2. RSI extremes and divergence
    3. Bollinger Band squeeze/breakout
    4. ADX for trend strength
    5. Volume confirmation
    6. Moving average alignment
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
    
    macd_df = MACD(df)
    df['MACD'] = macd_df['macd']
    df['MACD_signal'] = macd_df['signal']
    df['MACD_hist'] = macd_df['histogram']
    
    bb_df = Bollinger_Bands(df, 20, 2)
    df['BB_upper'] = bb_df['BB_upper']
    df['BB_middle'] = bb_df['BB_middle']
    df['BB_lower'] = bb_df['BB_lower']
    df['BB_width'] = bb_df['BB_width']
    
    df['ADX'] = ADX(df, 14)
    
    stoch_df = Stochastic(df, 14, 3)
    df['Stoch_K'] = stoch_df['%K']
    df['Stoch_D'] = stoch_df['%D']
    
    df['OBV'] = OBV(df)
    
    # Moving averages for trend
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # Volume analysis
    df['vol_sma'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_sma']
    
    # Price momentum
    df['momentum_5'] = df['Close'].pct_change(5) * 100
    df['momentum_10'] = df['Close'].pct_change(10) * 100
    
    for i in range(200, len(df)):  # Need 200 days for SMA_200
        score = 0
        reasons = []
        
        # === BULLISH CONDITIONS ===
        bullish_conditions = 0
        
        # 1. MACD bullish crossover
        if (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and 
            df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]):
            bullish_conditions += 2  # Strong signal
            reasons.append("MACD_bullish_crossover")
        elif df['MACD_hist'].iloc[i] > 0 and df['MACD_hist'].iloc[i] > df['MACD_hist'].iloc[i-1]:
            bullish_conditions += 1
            reasons.append("MACD_momentum_up")
        
        # 2. RSI oversold recovery
        if df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] > 30:
            bullish_conditions += 2
            reasons.append("RSI_oversold_recovery")
        elif 40 < df['RSI'].iloc[i] < 60:
            bullish_conditions += 1
            reasons.append("RSI_neutral_bullish")
        
        # 3. Bollinger Band bounce
        if df['Close'].iloc[i-1] <= df['BB_lower'].iloc[i-1] and df['Close'].iloc[i] > df['BB_lower'].iloc[i]:
            bullish_conditions += 2
            reasons.append("BB_lower_bounce")
        elif df['Close'].iloc[i] > df['BB_middle'].iloc[i]:
            bullish_conditions += 1
            reasons.append("Above_BB_middle")
        
        # 4. Strong trend (ADX)
        if df['ADX'].iloc[i] > 25:
            if df['Close'].iloc[i] > df['SMA_20'].iloc[i]:
                bullish_conditions += 1
                reasons.append("Strong_uptrend")
        
        # 5. Moving average alignment (bullish)
        if (df['Close'].iloc[i] > df['EMA_9'].iloc[i] > df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i]):
            bullish_conditions += 2
            reasons.append("MA_alignment_bullish")
        elif df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i]:
            bullish_conditions += 1
            reasons.append("Golden_cross_20_50")
        
        # 6. Stochastic oversold
        if df['Stoch_K'].iloc[i-1] < 20 and df['Stoch_K'].iloc[i] > 20:
            bullish_conditions += 1
            reasons.append("Stoch_oversold_exit")
        
        # 7. Volume confirmation
        if df['vol_ratio'].iloc[i] > 1.2:
            bullish_conditions += 1
            reasons.append("High_volume")
        
        # 8. OBV trend
        if df['OBV'].iloc[i] > df['OBV'].iloc[i-5]:
            bullish_conditions += 1
            reasons.append("OBV_rising")
        
        # === BEARISH CONDITIONS ===
        bearish_conditions = 0
        
        # 1. MACD bearish crossover
        if (df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and 
            df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]):
            bearish_conditions += 2
            reasons.append("MACD_bearish_crossover")
        elif df['MACD_hist'].iloc[i] < 0 and df['MACD_hist'].iloc[i] < df['MACD_hist'].iloc[i-1]:
            bearish_conditions += 1
            reasons.append("MACD_momentum_down")
        
        # 2. RSI overbought reversal
        if df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] < 70:
            bearish_conditions += 2
            reasons.append("RSI_overbought_reversal")
        elif 40 < df['RSI'].iloc[i] < 60:
            bearish_conditions += 1
            reasons.append("RSI_neutral_bearish")
        
        # 3. Bollinger Band rejection
        if df['Close'].iloc[i-1] >= df['BB_upper'].iloc[i-1] and df['Close'].iloc[i] < df['BB_upper'].iloc[i]:
            bearish_conditions += 2
            reasons.append("BB_upper_rejection")
        elif df['Close'].iloc[i] < df['BB_middle'].iloc[i]:
            bearish_conditions += 1
            reasons.append("Below_BB_middle")
        
        # 4. Strong downtrend (ADX)
        if df['ADX'].iloc[i] > 25:
            if df['Close'].iloc[i] < df['SMA_20'].iloc[i]:
                bearish_conditions += 1
                reasons.append("Strong_downtrend")
        
        # 5. Moving average alignment (bearish)
        if (df['Close'].iloc[i] < df['EMA_9'].iloc[i] < df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]):
            bearish_conditions += 2
            reasons.append("MA_alignment_bearish")
        elif df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]:
            bearish_conditions += 1
            reasons.append("Death_cross_20_50")
        
        # 6. Stochastic overbought
        if df['Stoch_K'].iloc[i-1] > 80 and df['Stoch_K'].iloc[i] < 80:
            bearish_conditions += 1
            reasons.append("Stoch_overbought_exit")
        
        # 7. Volume confirmation
        if df['vol_ratio'].iloc[i] > 1.2:
            bearish_conditions += 1
        
        # 8. OBV trend
        if df['OBV'].iloc[i] < df['OBV'].iloc[i-5]:
            bearish_conditions += 1
            reasons.append("OBV_falling")
        
        # === GENERATE SIGNALS ===
        # For swing trading, we need stronger confirmation (higher threshold)
        if bullish_conditions >= 5:
            signals.loc[df.index[i], 'signal'] = 1
            signals.loc[df.index[i], 'signal_strength'] = bullish_conditions / 10
            # Wider stops for daily timeframe
            signals.loc[df.index[i], 'stop_loss'] = df['Close'].iloc[i] - (2.5 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'target'] = df['Close'].iloc[i] + (4 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'reason'] = ', '.join(reasons)
        
        elif bearish_conditions >= 5:
            signals.loc[df.index[i], 'signal'] = -1
            signals.loc[df.index[i], 'signal_strength'] = bearish_conditions / 10
            signals.loc[df.index[i], 'stop_loss'] = df['Close'].iloc[i] + (2.5 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'target'] = df['Close'].iloc[i] - (4 * df['ATR'].iloc[i])
            signals.loc[df.index[i], 'reason'] = ', '.join(reasons)
    
    return signals, df

# ==================== BACKTESTING ====================

def backtest_swing_trading(df, signals, initial_capital=100000):
    """Backtest the swing trading strategy"""
    portfolio = pd.DataFrame(index=df.index)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    portfolio['returns'] = 0.0
    
    position = 0
    entry_price = 0
    stop_loss = 0
    target = 0
    entry_date = None
    trades = []
    
    for i in range(len(signals)):
        if position == 0:  # No position
            if signals['signal'].iloc[i] == 1:  # Buy signal
                position = 1
                entry_price = signals['price'].iloc[i]
                stop_loss = signals['stop_loss'].iloc[i]
                target = signals['target'].iloc[i]
                entry_date = signals.index[i]
                shares = int((initial_capital * 0.95) / entry_price)
                portfolio.loc[signals.index[i], 'cash'] = initial_capital - (shares * entry_price)
                portfolio.loc[signals.index[i], 'holdings'] = shares
                
            elif signals['signal'].iloc[i] == -1:  # Sell signal
                position = -1
                entry_price = signals['price'].iloc[i]
                stop_loss = signals['stop_loss'].iloc[i]
                target = signals['target'].iloc[i]
                entry_date = signals.index[i]
                shares = int((initial_capital * 0.95) / entry_price)
                portfolio.loc[signals.index[i], 'cash'] = initial_capital + (shares * entry_price)
                portfolio.loc[signals.index[i], 'holdings'] = -shares
        
        elif position == 1:  # Long position
            current_price = signals['price'].iloc[i]
            
            # Check for exit conditions
            if current_price <= stop_loss or current_price >= target:
                # Close position
                exit_price = current_price
                shares = portfolio['holdings'].iloc[i-1]
                pnl = (exit_price - entry_price) * shares
                portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + (shares * exit_price)
                portfolio.loc[signals.index[i], 'holdings'] = 0
                
                holding_days = (signals.index[i] - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': signals.index[i],
                    'holding_days': holding_days,
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': (exit_price / entry_price - 1) * 100,
                    'reason': 'SL' if current_price <= stop_loss else 'Target'
                })
                position = 0
            else:
                # Hold position
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
                
                holding_days = (signals.index[i] - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': signals.index[i],
                    'holding_days': holding_days,
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
            'max_drawdown': 0,
            'avg_holding_days': 0
        }
    
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades_df) * 100
    
    avg_return = trades_df['return'].mean()
    avg_holding = trades_df['holding_days'].mean()
    
    returns = portfolio['returns'].dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    cummax = portfolio['total'].cummax()
    drawdown = (cummax - portfolio['total']) / cummax
    max_dd = drawdown.max() * 100
    
    profit_factor = (winning_trades['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    
    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_holding_days': avg_holding,
        'profit_factor': profit_factor,
        'avg_win': winning_trades['return'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl'] < 0]['return'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    }

# ==================== VISUALIZATION ====================

def plot_signals(df, signals, ticker):
    """Plot price with signals and indicators"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Price and signals
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.plot(df.index, df['BB_upper'], 'g--', alpha=0.5, label='BB Upper')
    ax1.plot(df.index, df['BB_lower'], 'r--', alpha=0.5, label='BB Lower')
    
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'{ticker} - Swing Trading Signals')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # MACD
    ax2.plot(df.index, df['MACD'], label='MACD', linewidth=1.5)
    ax2.plot(df.index, df['MACD_signal'], label='Signal', linewidth=1.5)
    ax2.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax3.axhline(y=70, color='r', linestyle='--', linewidth=0.8, label='Overbought')
    ax3.axhline(y=30, color='g', linestyle='--', linewidth=0.8, label='Oversold')
    ax3.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================

def run_swing_strategy(ticker, period='2y', interval='1d'):
    """Run the complete swing trading strategy"""
    print(f"\n{'='*60}")
    print(f"Running Swing Trading Strategy for {ticker}")
    print(f"{'='*60}\n")
    
    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
    
    if df.empty:
        print(f"No data available for {ticker}")
        return None, None, None
    
    # Flatten column names if they are still multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)
    
    if len(df) < 250:
        print(f"Insufficient data for {ticker} (need at least 250 days)")
        return None, None, None
    
    # Identify signals
    signals, df_with_indicators = identify_swing_signals(df)
    
    # Backtest
    portfolio, trades = backtest_swing_trading(df, signals)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio, trades)
    
    # Print results
    print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Trading Days: {len(df)}")
    print(f"\n--- Performance Metrics ---")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Return per Trade: {metrics['avg_return']:.2f}%")
    print(f"Average Win: {metrics['avg_win']:.2f}%")
    print(f"Average Loss: {metrics['avg_loss']:.2f}%")
    print(f"Average Holding Period: {metrics['avg_holding_days']:.1f} days")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    if len(trades) > 0:
        print(f"\n--- Recent Trades ---")
        recent_trades = trades.tail(5)
        for idx, trade in recent_trades.iterrows():
            print(f"{trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')} | "
                  f"{trade['type']} | Entry: ₹{trade['entry_price']:.2f} | Exit: ₹{trade['exit_price']:.2f} | "
                  f"Return: {trade['return']:.2f}% | Days: {trade['holding_days']}")
        
        print(f"\n--- Active Signals (Last 3) ---")
        recent_signals = signals[signals['signal'] != 0].tail(3)
        for idx, row in recent_signals.iterrows():
            signal_type = "BUY" if row['signal'] == 1 else "SELL"
            print(f"{idx.strftime('%Y-%m-%d')} | {signal_type} | Price: ₹{row['price']:.2f} | "
                  f"SL: ₹{row['stop_loss']:.2f} | Target: ₹{row['target']:.2f}")
            print(f"  Strength: {row['signal_strength']:.2f} | Reason: {row['reason']}")
    
    # Plot
    # plot_signals(df_with_indicators.tail(200), signals.tail(200), ticker)
    
    return df_with_indicators, signals, trades

# ==================== RUN FOR MULTIPLE TICKERS ====================

if __name__ == "__main__":
    # Top liquid stocks for swing trading
    tickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", 
        "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", 
        "KOTAKBANK.NS", "LT.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS",
        "TATAMOTORS.NS", "TATASTEEL.NS", "WIPRO.NS"
    ]
    
    results = {}
    summary_data = []
    
    for ticker in tickers:  # Test with first 5 stocks
        try:
            df, signals, trades = run_swing_strategy(ticker, period='2y', interval='1d')
            if df is not None and trades is not None:
                results[ticker] = {'df': df, 'signals': signals, 'trades': trades}
                
                # Calculate summary metrics
                if len(trades) > 0:
                    winning_trades = trades[trades['pnl'] > 0]
                    summary_data.append({
                        'Ticker': ticker,
                        'Total Trades': len(trades),
                        'Win Rate %': len(winning_trades) / len(trades) * 100,
                        'Avg Return %': trades['return'].mean(),
                        'Total PnL': trades['pnl'].sum(),
                        'Avg Days': trades['holding_days'].mean()
                    })
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary comparison
    if summary_data:
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}")
    
    print("\nStrategy execution completed!")
    print(f"{'='*80}")