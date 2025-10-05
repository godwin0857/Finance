# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:38:43 2025

@author: jauha
"""

from math import exp as e


def future_value(sum, rate, time):
    return sum*(1+rate)**time

def present_value(sum, rate, time):
    return sum*(1+rate)**-time

def future_continuous_value(sum, rate, time):
    return sum*e(-rate*time)

def present_continuous_value(sum, rate, time):
    return sum*e(rate*time)


sum = 100
rate = 0.08
time = 2

#Present and Future values

print("Future discrete value is %s"% future_value(sum, rate, time))
print("Present discrete value is %s"% present_value(sum, rate, time))
print("Future continuous value is %s"% future_continuous_value(sum, rate, time))
print("Present continuous value is %s"% present_continuous_value(sum, rate, time))


##Bond prices
#Zero coupon bond

class ZeroCouponBond:
    
    #constructor
    def __init__(self, principal,interest_rate,duration):
        self.principal=principal
        self.interest_rate=interest_rate / 100
        self.duration=duration
    
    #calculate the present value
    def presentVal(self,p,n):
        return p*(1+self.interest_rate)**-n
    
    def price(self):
        return self.presentVal(self.principal,self.duration)

class CouponBond:
    
    def __init__(self,principal, couponrate, maturity, interest_rate, paymentType):
        self.principal = principal
        self.interest_rate = interest_rate/100
        self.couponrate = couponrate/100 #annual coupon rate
        self.maturity = maturity
        self.paymentType = paymentType
    
    
    
    def presentVal(self,p,n):
        return p*((1+self.interest_rate)**-n)
    
    
    def calculatePrice(self,p,couponrate,maturity,paymentType):
        price = 0.0
        
        if str(paymentType).capitalize()=="M" and maturity > 0:
            durationFactor = 12
        if str(paymentType).capitalize()=="Y" and maturity > 0:
            durationFactor = 1
        else:
            print ("Enter correct payment type (Monthly/Yearly) and Maturity > 0...")
           # return 0
            
        
        #Calculating the sum of PV of all coupon payments
        for n in range(1,maturity+1):
            price = price + self.presentVal(self.principal * self.couponrate/durationFactor, n) #(couponrate*p/durationFactor)*(1+(self.interest_rate/durationFactor))**-n
            print("PV of {}th interest is {}".format(n,self.presentVal(self.principal * self.couponrate/durationFactor, n)))
        
        #Calculating the PV of the principal
        
        
        
        return price + self.presentVal(self.principal,self.maturity)
            
    
        
# =============================================================================
# couponrate = .10
# maturity=2
# p=100000
# paymentType="yearly"
# interest_rate=.06
# price = 0
# durationFactor=0
# def presentVal(p,n):
#     return p*((1+interest_rate)**-n)    
#     
# if str(paymentType).capitalize()[0]=="M" and maturity > 0:
#     durationFactor = 12
# if str(paymentType).capitalize()[0]=="Y" and maturity > 0:
#     durationFactor = 1
# else:
#     print ("Enter correct payment type (Monthly/Yearly) and Maturity > 0...")
#    # return 0
#     
# for n in range(1,maturity+1):
#     price = price + presentVal(p * couponrate/durationFactor, n) #(couponrate*p/durationFactor)*(1+(self.interest_rate/durationFactor))**-n
#     print("PV of {}th interest is {}".format(n,presentVal(p * couponrate/durationFactor, n)))
# =============================================================================
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    principal = 10000
    rate = 4
    duration = 2
    Bond1 = ZeroCouponBond(principal, rate, duration)
    
    print ("Price of the bond is %s"% Bond1.price())
    