# algoTrade
Quantitative Trading & Technical Analysis Toolkit
This repository contains a collection of Python scripts designed for end-to-end quantitative trading research, financial data extraction, technical indicator calculation, and automated data scraping for both Indian (NSE) and international markets. The toolkit is engineered for flexibility, with modular scripts optimized for both exploratory analysis and scalable workflow integration.

Contents
Financial Data Extraction:

- Fetch daily, intraday, and historical OHLCV data for equities and currencies using yfinance and Alpha Vantage.
- Scripts handle time zone conversion, NaN handling, multi-ticker download, and backfilling.

Technical Indicator Calculations:

- ATR, Bollinger Bands, MACD, RSI: Robust and efficient implementations for batch multi-ticker processing.
- Renko Chart Automation: Leverages stocktrends for automated Renko chart creation, with dynamic ATR-based brick sizing.

Web Scraping Utilities:
- BeautifulSoup Extractors: Extract financial tables from Moneycontrol and NASDAQ for custom datasets.
- Selenium Automation: Fully browser-automated scraping of Yahoo Finance financial statements, with dynamic element interaction for structured data output.

Statistical Analysis & Visualization:
- Rolling mean, median, max, sum, and exponential weighted stats on returns and prices.
- Sample scripts for clean data visualization using pandas and matplotlib.

Sample Scripts Included
| Script Name        | Description           |
| ------------- |:-------------:|
| First-test.py | Downloads historical data, manages missing values, illustrates data access, and works with long time series.| 
| alpha-vantage.py | Fetches intraday and daily close prices for multiple tickers from Alpha Vantage. Handles API rate limiting.| 
| Scraping-via-beautifulsoup.py | Uses requests and BeautifulSoup to extract financial data tables from Nasdaq or Moneycontrol.| 
| Selenium-scraping.py | Automates browser navigation and scraping to pull financial statements from Yahoo Finance. Handles dynamic page elements.| 
| Stats.py | Performs statistical calculations—mean, std, median, rolling stats—on daily returns and price series; includes sample plotting code.| 
| MACD.py | Calculates MACD and signal line for multiple tickers with adjustable periods.| 
| ATR-and-BB.py | Downloads OHLCV data for NSE stocks and calculates ATR & Bollinger Bands with timezone conversion.| 
| RSI.py | Computes the Relative Strength Index (RSI) using a vectorized pandas approach for multiple tickers.| 
| Renko.py | Creates Renko charts for tickers using ATR-based brick sizing and multi-resolution OHLCV data.| 

Usage
- Each script is standalone for rapid prototyping—modify tickers, time windows, and parameters as required.
- Optimized for use with Jupyter Notebook/Spyder for iterative analysis, or command-line execution for workflow automation.
- Step-by-step comments and function docstrings are included to facilitate understanding, modification, and extension.

Requirements\
Python 3.7+

Key Libraries\
pandas, numpy, yfinance, alpha_vantage, stocktrends, matplotlib, beautiful soup, selenium

Author\
Built and maintained by a product guy dabbling with programming, financial market analysis, and data automation workflows.

