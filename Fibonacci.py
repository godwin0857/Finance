# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:27:48 2025

@author: jauha
"""

import time
import numpy as np

def fibonacci(n):
    if n<=1:
        return n
    else:
        return(fibonacci(n-1)+fibonacci(n-2)) # Recursive function
    
    
def main():
    num=np.random.randint(1,25)
    print(f"{num} fibonnaci number is : {fibonacci(num)}")


#Continuous Time execution
starttime=time.time()
timeout = time.time()+60*2 #timeout of 2mins
while time.time()<timeout:
    try:
        main()
        print("startime is {}, current time is {}".format(starttime,time.time()-starttime))
        # check for time elapsed from the start; get the remainder from the last 5th second
        time.sleep(5-((time.time()-starttime)%5.0))
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
    