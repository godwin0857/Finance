# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:17:19 2025

@author: jauha
"""

import requests
from bs4 import BeautifulSoup

#url="https://www.moneycontrol.com/india/stockpricequote/miningminerals/vedanta/SG"
#page=requests.get(url)
url="https://www.nasdaq.com/market-activity/stocks/achr/financials"

#mimicking actual user requests via browser
headers={"User-Agent":"Chrome/139.0.7258.155"}
#passing user agent values in the request header
page=requests.get(url,headers=headers)
#getting the page's HTML content
page_content=page.content
soup=BeautifulSoup(page_content,"html.parser")
#tab1 = soup.find("div",{"class":"table yf-yuwun0"})
tab1=soup.find("div", {"class": "jupiter22-c-financials-table__table_container"})
