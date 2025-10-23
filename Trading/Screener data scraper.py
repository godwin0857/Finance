# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 18:35:59 2025

@author: jauha
Screener.in Peer Comparison Data Scraper
Extracts peer comparison metrics for multiple stock tickers
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time

# Configuration
CHROMEDRIVER_PATH = r"C:\Users\jauha\anaconda3\envs\algoTrading\Web Driver\chromedriver-win64\chromedriver-win64\chromedriver.exe"

TICKER_CSV_FILE = "tickers.csv"  # CSV file with tickers (no headers, one ticker per row)


# Stock tickers to scrape
# TICKERS = [
#     "BEACON", "GAYAHWS", "MGEL", "MUNISH", "ROCKINGDCE", "SANGINITA", "WHITEFORCE",
#     "3IINFO-RE", "CHANDAN", "MVKAGRO", "NRVANDANA", "POLYSIL", "BAGDIGITAL",
#     "TECHERA", "CARTRADE", "GREENLEAF", "IZMO", "JYOTIGLOBL", "KAVDEFENCE",
#     "MPEL", "WEWORK", "ESFL", "GRMOVER", "KANDARP", "NIKITA", "SHANKARA",
#     "TARMAT", "ADVANCE", "INFIBEAM", "OMFREIGHT", "RBLBANK", "SHEEL",
#     "EIMCOELECO", "FABTECH", "SUBAHOTELS", "VIGOR", "VIJAYPD", "BMWVENTLTD",
#     "NIRMAN", "PACEDIGITK", "ATALREAL", "ECOLINE", "GML", "JKIPL", "SAMMAANCAP",
#     "GLOBECIVIL", "VINEETLAB", "ACTIVEINFR", "AGIIL", "MGSL", "ONDOOR",
#     "PANACHE", "PROZONER", "TECHD", "URAVIDEF", "MOBIKWIK", "NIBE", "PRIMECAB",
#     "SHANTI", "SHIVASHRIT", "SIDDHICOTS", "AERON", "BHARATGEAR", "ESCONET",
#     "SHRADHA-RE"
# ]

# Fields to extract
FIELDS = [
    "CMP Rs.", "P/E", "Mar Cap Rs.Cr.", "Free Cash Flow Rs.Cr.", "CMP / FCF",
    "Sales Qtr Rs.Cr.", "NP Qtr Rs.Cr.", "Qtr Profit Var %",
    "Qtr Sales Var %", "ROCE %", "ROE %", "Debt / Eq", "PEG", "CMP / Sales"
]


def load_tickers_from_csv(csv_file):
    """
    Load tickers from CSV file (no headers, one ticker per row)
    Removes duplicates and returns unique tickers
    """
    try:
        # Read CSV without headers
        df = pd.read_csv(csv_file, header=None, names=['ticker'])
        
        # Strip whitespace from ticker values
        df['ticker'] = df['ticker'].str.strip()
        
        # Remove any empty rows
        df = df[df['ticker'].notna() & (df['ticker'] != '')]
        
        # Get unique tickers (removes duplicates)
        unique_tickers = df['ticker'].unique().tolist()
        
        print(f"Loaded {len(df)} tickers from CSV ({len(unique_tickers)} unique)")
        if len(df) != len(unique_tickers):
            print(f"  → Removed {len(df) - len(unique_tickers)} duplicate(s)")
        
        return unique_tickers
    
    except FileNotFoundError:
        print(f" CSV file '{csv_file}' not found!")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return []

def initialize_driver(chromedriver_path=CHROMEDRIVER_PATH):
    """Initialize and return Chrome WebDriver"""
    service = Service(chromedriver_path)
    service.start()
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--start-maximized')
    options.add_argument(r"--user-data-dir=C:\Users\jauha\AppData\Local\Google\Chrome\User Data\Profile 3")
    options.add_argument(r"--profile-directory=Profile 3")  # e.g., 'Default', 'Profile 1', 'Profile 2'
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scrape_peer_comparison(driver, ticker):
    """
    Scrape peer comparison data for a given ticker
    Returns a dictionary with the ticker's peer comparison metrics
    """
    url = f"https://www.screener.in/company/{ticker}/"
    # print(f"\nScraping {ticker}...")
    
    try:
        driver.get(url)
        time.sleep(3)  # Initial wait for page load
        
        # Wait for peer comparison section to load
        try:
            WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.ID, "peers-table-placeholder")))
        except TimeoutException:
            print(f"Peer comparison section not found for {ticker}")
            return None
        
        peer_data = {"Ticker": ticker}
        
        try:
           # Step 1: Get the company ID from main body > main > second div's data-company-id attribute
            main_element = driver.find_element(By.TAG_NAME, "main")
            divs = main_element.find_elements(By.XPATH, "./div")
            peers_placeholder = driver.find_element(By.ID, "peers-table-placeholder")
            second_div = divs[1]
            company_id = second_div.get_attribute("data-company-id")            
            if not company_id:
                print(f"Could not extract company ID for {ticker}")
                return None
            
            # print(f"  → Company ID: {company_id}")
            
            # Step 2: Locate table with specific class within peers section
            table = peers_placeholder.find_element(By.XPATH, "//section[@id='peers']//table[@class='data-table text-nowrap striped mark-visited no-scroll-right']")
            
            # Get tbody
            tbody = table.find_element(By.XPATH, ".//tbody")
            
            # Get the first <tr> within tbody for headers (skip first 2 td elements)
            first_row = tbody.find_element(By.XPATH, ".//tr[1]")
            header_cells = first_row.find_elements(By.TAG_NAME, "th")
            header_texts = [h.text.strip() for h in header_cells[2:]]  # Start from index 2 (3rd element)

            
            # Step 3: Get the specific row where data-row-company-id matches the company_id
            target_row = table.find_element(
                By.XPATH, 
                f".//tbody//tr[@data-row-company-id='{company_id}']"
            )
            
            cells = target_row.find_elements(By.TAG_NAME, "td")
            
            # Get cell values from 3rd td onwards (index 2 onwards)
            cell_values = [cell.text.strip() for cell in cells[2:]]
            cells[1].text.strip()
            
            i=0
            # Map headers to cell values
            for i, field in enumerate(header_texts):
                if i < len(cell_values):
                    # print("Value of i = ",i)
                    value = cell_values[i]
                    peer_data[field] = value
            
            # Extract only the required fields
            result = {"Ticker": ticker}
            for field in FIELDS:
                result[field] = peer_data.get(field, "N/A")
            
            # print(f"  ✓ Successfully scraped {ticker}")
            return result
            
        except NoSuchElementException as e:
            print(f"Table structure not found for {ticker}: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error scraping {ticker}: {str(e)}")
        return None



"""Main execution function"""
print("=" * 70)
print("Screener.in Peer Comparison Data Scraper")
print("=" * 70)

# Load tickers from CSV
TICKERS = load_tickers_from_csv(TICKER_CSV_FILE)

if not TICKERS:
    print("No tickers to process. Exiting.")

print("=" * 70)

# Initialize driver
driver = initialize_driver(CHROMEDRIVER_PATH)

successful = 0
failed = 0
fin_data={}
fin_list=[]

try:
    for ticker in TICKERS:
        data = scrape_peer_comparison(driver, ticker)
        if data:
            fin_list.append(data)
            # fin_data[ticker]=data
            successful += 1
        else:
            failed += 1
        
        # Small delay between requests to avoid rate limiting
        time.sleep(1)
        
        print (successful+failed,"/",len(TICKERS))
    
    # fin_list = [ticker for ticker in fin_data.values()]
    ticker = ('')
    
    
    
    
    # Create DataFrame
    if fin_list:
        df = pd.DataFrame(fin_list)
        
        # Save to CSV
        output_file = "screener_peer_comparison_data.csv"
        df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 70)
        print("Scraping Complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Data saved to: {output_file}")
        print("=" * 70)
        
        # Display preview
        print("\nData Preview:")
        print(df.head().to_string())
    else:
        print("\n No data was successfully scraped.")

finally:
    driver.quit()
    print("\nDriver closed successfully.")
    


# =============================================================================
# ticker="CARTRADE"
# url = "https://www.screener.in/company/CARTRADE/"
# CHROMEDRIVER_PATH = r"C:\Users\jauha\anaconda3\envs\algoTrading\Web Driver\chromedriver-win64\chromedriver-win64\chromedriver.exe"
# 
# # driver.get("https://www.google.com")  # Open desired URL in new tab
# 
# service = Service(CHROMEDRIVER_PATH)
# service.start()
# options = webdriver.ChromeOptions()
# options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_argument('--start-maximized')
# options.add_argument(r"--user-data-dir=C:\Users\jauha\AppData\Local\Google\Chrome\User Data\Profile 3")
# options.add_argument(r"--profile-directory=Profile 3")  # e.g., 'Default', 'Profile 1', 'Profile 2'
# driver = webdriver.Chrome(service=service, options=options)
# driver.get(url)
# #Wait for peer comparison section to load
# WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//section[@id='peers']")))
# 
# all_data = []
# successful = 0
# failed = 0
# fin_data={}
# 
# time.sleep(2)  # Initial wait for page load
# 
# 
# 
# print("\nScraping {cartrade}...in ticker for loop")
# #url = f"https://www.screener.in/company/{ticker}/"
#     
# 
# 
# fin_data["Ticker"] = ticker
# 
# #Step 1: Get the company ID from main body > main > second div's data-company-id attribute
# main_element = driver.find_element(By.TAG_NAME, "main")
# divs = main_element.find_elements(By.XPATH, "./div")
# peers_placeholder = driver.find_element(By.ID, "peers-table-placeholder")
# 
# if len(divs) < 2:
#     print(f"  ⚠️  Could not find second div under main for {ticker}")
# 
# 
# second_div = divs[1]
# company_id = second_div.get_attribute("data-company-id")
# 
# if not company_id:
#     print(f"  ⚠️  Could not extract company ID for {ticker}")
# 
# 
# print(f"  → Company ID: {company_id}")
# 
# # Step 2: Locate table with specific class within peers section
# table = driver.find_element(
#     By.XPATH, 
#     "//section[@id='peers']//table[@class='data-table text-nowrap striped mark-visited no-scroll-right']"
# )
# 
# # Get tbody
# tbody = table.find_element(By.XPATH, ".//tbody")
# 
# # Get the first <tr> within tbody for headers (skip first 2 td elements)
# first_row = tbody.find_element(By.XPATH, ".//tr[1]")
# header_cells = first_row.find_elements(By.TAG_NAME, "th")
# header_texts = [h.text.strip() for h in header_cells[2:]]  # Start from index 2 (3rd element)
# 
# # Step 3: Get the specific row where data-row-company-id matches the company_id
# target_row = table.find_element(
#     By.XPATH, 
#     f".//tbody//tr[@data-row-company-id='{company_id}']"
# )
# 
# cells = target_row.find_elements(By.TAG_NAME, "td")
# 
# # Get cell values from 3rd td onwards (index 2 onwards)
# cell_values = [cell.text.strip() for cell in cells[2:]]
# cells[1].text.strip()
# 
# i=0
# # Map headers to cell values
# for i, field in enumerate(header_texts):
#     if i < len(cell_values):
#         print("Value of i = ",i)
#         value = cell_values[i]
#         fin_data[field] = value
# 
# # Extract only the required fields
# result = {"Ticker": ticker}
# for field in FIELDS:
#     result[field] = fin_data.get(field, "N/A")
# 
# print(f"  ✓ Successfully scraped {ticker}")
# =============================================================================
