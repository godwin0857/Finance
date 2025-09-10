# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:33:14 2025

@author: jauha
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
# Optionally, wait for the page or element to load
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

path=r"C:\\Users\\jauha\\anaconda3\\envs\\algoTrading\\Web Driver\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"

#instantiate the service
service=Service(path)
service.start()

ticker = "AAPL"
url="https://finance.yahoo.com/quote/{}/financials/".format(ticker)
driver=webdriver.Chrome(service=service)
driver.get(url)
driver.implicitly_wait(2)


#get button data - all buttons within the section with data testid - qsp financials
buttons=driver.find_elements(By.XPATH,"//section[@data-testid ='qsp-financials']//button")

#get button data from article level
#buttons=driver.find_elements(By.XPATH,"//article[@class ='yf-m6gtul']//button")

# for button in buttons:
#    print(button.accessible_name) #button label is available in accessible name
#    if button.accessible_name in ["Quarterly","Expand All"]:
#        pass
#    else:
#        WebDriverWait(driver,2).until(EC.element_to_be_clickable(button)).click()


#To avoid DOM being changed after page load
buttons_count = len(driver.find_elements(By.XPATH, "//section[@data-testid ='qsp-financials']//button"))
i=0
for i in range(buttons_count):
    # Relocate buttons in each iteration to avoid stale reference
    buttons = driver.find_elements(By.XPATH, "//section[@data-testid ='qsp-financials']//button")
    button = buttons[i]
    print(button.accessible_name)  # Button label is available in accessible name

    if button.accessible_name not in ["Quarterly", "Expand All"]:
        WebDriverWait(driver, 2).until(EC.element_to_be_clickable(button)).click()

#to get the data in string and store it in table
#table=driver.find_element(By.XPATH, '//*[@id="main-content-wrapper"]/article/section/div/div/div[2]').text


incomeSt={}
table_head=driver.find_elements(By.XPATH, "//*[contains(@class, 'column yf-1yyu1pc')]")
#hidden rows have lvl changed in class name
table=driver.find_elements(By.XPATH, "//*[@class='row lv-0 yf-t22klz' or @class='row lv-1 yf-t22klz']")
headings=[]
for cell in table_head:
    headings.append(cell.text)


#transform the table data into key value pairs for heading and values
table_data = {}
for cell in table:
    #split the cell data at the first new line occurrence
    lines = cell.text.split('\n', 1)  # Split at first newline only
    key = lines[0]
    #replace the new line in the values with a space
    value = lines[1] if len(lines) > 1 else ''
   # table_data[key] = value
    #convert the value string into a list
    incomeSt[key] = value.split("\n")
    