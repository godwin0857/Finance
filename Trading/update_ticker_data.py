# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 13:18:50 2025

@author: jauha
"""
import yfinance as yf
import pandas as pd
import sqlite3
import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DB_NAME = "nifty_data.db"
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "TITAN.NS", "BAJFINANCE.NS", "MARUTI.NS",
    "ULTRACEMCO.NS", "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS",
    "ASIANPAINT.NS", "NTPC.NS", "M&M.NS", "POWERGRID.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "LTIM.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "ONGC.NS", "GRASIM.NS", "BAJAJFINSV.NS", "TECHM.NS", "HINDALCO.NS",
    "EICHERMOT.NS", "NESTLEIND.NS", "DRREDDY.NS", "CIPLA.NS", "TATACONSUM.NS",
    "BRITANNIA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "BPCL.NS", "HEROMOTOCO.NS", "UPL.NS", "INDUSINDBK.NS", "BEL.NS"
]

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create tables if they don't exist
    for table in ['data_5m', 'data_15m']:
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table} (
                Datetime TEXT,
                Ticker TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                PRIMARY KEY (Ticker, Datetime)
            )
        ''')
    conn.commit()
    conn.close()

def get_last_timestamp(ticker, table_name, conn):
    query = f"SELECT MAX(Datetime) FROM {table_name} WHERE Ticker = '{ticker}'"
    try:
        result = pd.read_sql(query, conn)
        last_date = result.iloc[0, 0]
        if last_date:
            return pd.to_datetime(last_date)
    except:
        pass
    return None

def store_data(ticker, interval):
    table_name = "data_5m" if interval == "5m" else "data_15m"
    conn = sqlite3.connect(DB_NAME)
    
    try:
        # 1. Fetch Data (YF returns UTC-aware usually)
        df = yf.download(ticker, period="59d", interval=interval, progress=False)
        
        if df.empty:
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        
        # 2. Standardize Column Name
        if 'Datetime' not in df.columns:
             df.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True)
        
        # 3. TIMEZONE CONVERSION LOGIC (The Fix)
        # Check if timezone aware
        if df['Datetime'].dt.tz is not None:
            # Convert to IST
            df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')
        else:
            # If naive, assume UTC first then convert
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            
        # Remove timezone info (make naive) so SQLite stores clean strings like "2023-10-27 09:15:00"
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        
        # Format as string for storage
        df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Ticker'] = ticker
        
        # Keep relevant columns
        cols = ['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]
        
        # 4. Incremental Update
        last_ts = get_last_timestamp(ticker, table_name, conn)
        
        if last_ts:
            # Ensure last_ts is string for comparison
            last_ts_str = last_ts.strftime('%Y-%m-%d %H:%M:%S')
            df = df[df['Datetime'] > last_ts_str]
            
        if not df.empty:
            df.to_sql(table_name, conn, if_exists='append', index=False)
            
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    finally:
        conn.close()

def update_database():
    print(f"--- Updating {DB_NAME} (Converting to IST) ---")
    init_db()
    
    with tqdm(total=len(NIFTY_50_TICKERS)) as pbar:
        for ticker in NIFTY_50_TICKERS:
            store_data(ticker, "5m")
            store_data(ticker, "15m")
            pbar.update(1)
            pbar.set_description(f"Processing {ticker}")
            
    print("\n--- Update Complete! All times are now IST. ---")

if __name__ == "__main__":
    update_database()
    
    # Verification Step
    conn = sqlite3.connect(DB_NAME)
    print("\n--- Verifying Timezone (Top 3 Rows for RELIANCE) ---")
    df_verify = pd.read_sql("SELECT * FROM data_5m WHERE Ticker='RELIANCE.NS' ORDER BY Datetime DESC LIMIT 3", conn)
    print(df_verify)
    conn.close()