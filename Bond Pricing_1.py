# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:49:01 2025

@author: jauha
"""

class CouponBond:
    """
    A class to represent and calculate the price of a coupon bond.
    """
    
    def __init__(self, principal, coupon_rate, duration, frequency, risk_free_rate):
        """
        Initialize a CouponBond object.
        
        Args:
            principal (float): Face value/principal amount of the bond
            coupon_rate (float): Annual coupon rate (as decimal, e.g., 0.06 for 6%)
            duration (str): Duration of the bond (e.g., '6m', '1y', '5y')
            frequency (str): Payment frequency - 'monthly' or 'yearly'
            risk_free_rate (float): Annual risk-free interest rate for discounting (as decimal)
        """
        self.principal = principal
        self.coupon_rate = coupon_rate
        self.duration = duration
        self.frequency = frequency.lower().strip()
        self.risk_free_rate = risk_free_rate
        
        # Validate inputs
        self._validate_inputs()
        
        # Parse and calculate bond parameters
        self.duration_years = self._parse_duration(duration)
        self.payments_per_year = self._get_payments_per_year()
        self.total_payments = int(self.duration_years * self.payments_per_year)
        self.coupon_per_period = (self.principal * self.coupon_rate) / self.payments_per_year
        self.discount_rate_per_period = self.risk_free_rate / self.payments_per_year
    
    def _validate_inputs(self):
        """Validate all input parameters."""
        if self.principal <= 0:
            raise ValueError("Principal must be positive")
        if self.coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
        if self.frequency not in ['monthly', 'yearly']:
            raise ValueError("Frequency must be 'monthly' or 'yearly'")
    
    def _parse_duration(self, duration_str):
        """
        Parse duration string (e.g., '1m', '6m', '1y', '5y') and return duration in years.
        
        Args:
            duration_str (str): Duration in format like '1m', '6m', '1y', '5y'
        
        Returns:
            float: Duration in years
        """
        duration_str = duration_str.lower().strip()
        
        if duration_str.endswith('m'):
            months = int(duration_str[:-1])
            return months / 12
        elif duration_str.endswith('y'):
            years = int(duration_str[:-1])
            return years
        else:
            raise ValueError("Duration must end with 'm' for months or 'y' for years (e.g., '6m', '2y')")
    
    def _get_payments_per_year(self):
        """Get the number of payments per year based on frequency."""
        if self.frequency == 'monthly':
            return 12
        elif self.frequency == 'yearly':
            return 1
    
    @staticmethod
    def calculate_present_value(future_value, rate, time_periods):
        """
        Calculate the present value of a future cash flow.
        
        Args:
            future_value (float): The future cash flow amount
            rate (float): Discount rate per period (as decimal, e.g., 0.05 for 5%)
            time_periods (float): Number of periods until the cash flow
        
        Returns:
            float: Present value of the future cash flow
        """
        if rate < 0:
            raise ValueError("Discount rate cannot be negative")
        
        pv = future_value / ((1 + rate) ** time_periods)
        return pv
    
    def get_pv_of_coupons(self):
        """
        Calculate the present value of all coupon payments.
        
        Returns:
            float: Present value of all coupon payments
        """
        if self.total_payments == 0:
            raise ValueError("Duration too short for any coupon payments")
        
        pv_coupons = 0
        for period in range(1, self.total_payments + 1):
            pv_coupon = self.calculate_present_value(
                self.coupon_per_period, 
                self.discount_rate_per_period, 
                period
            )
            pv_coupons += pv_coupon
        
        return pv_coupons
    
    def get_pv_of_principal(self):
        """
        Calculate the present value of the principal (paid at maturity).
        
        Returns:
            float: Present value of principal
        """
        pv_principal = self.calculate_present_value(
            self.principal,
            self.discount_rate_per_period,
            self.total_payments
        )
        return pv_principal
    
    def calculate_price(self):
        """
        Calculate the total price of the coupon bond.
        
        Returns:
            float: Bond price
        """
        pv_coupons = self.get_pv_of_coupons()
        pv_principal = self.get_pv_of_principal()
        return pv_coupons + pv_principal
    
    def get_bond_details(self):
        """
        Get detailed breakdown of bond pricing.
        
        Returns:
            dict: Dictionary containing bond price and breakdown of components
        """
        pv_coupons = self.get_pv_of_coupons()
        pv_principal = self.get_pv_of_principal()
        bond_price = pv_coupons + pv_principal
        
        return {
            'bond_price': round(bond_price, 2),
            'pv_coupons': round(pv_coupons, 2),
            'pv_principal': round(pv_principal, 2),
            'total_coupon_payments': self.total_payments,
            'coupon_per_period': round(self.coupon_per_period, 2),
            'duration_years': self.duration_years,
            'principal': self.principal,
            'coupon_rate': self.coupon_rate,
            'risk_free_rate': self.risk_free_rate,
            'frequency': self.frequency
        }
    
    def __str__(self):
        """String representation of the bond."""
        details = self.get_bond_details()
        return f"""
Coupon Bond Details:
{'=' * 50}
Principal: ${self.principal:,.2f}
Coupon Rate: {self.coupon_rate * 100:.2f}%
Duration: {self.duration} ({self.duration_years} years)
Payment Frequency: {self.frequency.capitalize()}
Risk-Free Rate: {self.risk_free_rate * 100:.2f}%

Valuation:
{'=' * 50}
Bond Price: ${details['bond_price']:,.2f}
PV of Coupons: ${details['pv_coupons']:,.2f}
PV of Principal: ${details['pv_principal']:,.2f}
Total Coupon Payments: {details['total_coupon_payments']}
Coupon per Period: ${details['coupon_per_period']:,.2f}
"""


# Example usage
if __name__ == "__main__":
    # Example 1: 5-year bond with yearly payments
    print("=" * 60)
    print("Example 1: 5-year bond with yearly coupon payments")
    print("=" * 60)
    
    bond1 = CouponBond(
        principal=1000,           # $1000 face value
        coupon_rate=0.06,         # 6% annual coupon rate
        duration='5y',            # 5 years
        frequency='yearly',       # Yearly payments
        risk_free_rate=0.05       # 5% risk-free rate
    )
    
    print(bond1)
    
    # Example 2: 18-month bond with monthly payments
    print("=" * 60)
    print("Example 2: 18-month bond with monthly coupon payments")
    print("=" * 60)
    
    bond2 = CouponBond(
        principal=5000,           # $5000 face value
        coupon_rate=0.08,         # 8% annual coupon rate
        duration='18m',           # 18 months
        frequency='monthly',      # Monthly payments
        risk_free_rate=0.06       # 6% risk-free rate
    )
    
    print(bond2)
    
    # Example 3: Using individual methods
    print("=" * 60)
    print("Example 3: 3-year bond - Using individual methods")
    print("=" * 60)
    
    bond3 = CouponBond(
        principal=10000,
        coupon_rate=0.07,
        duration='3y',
        frequency='yearly',
        risk_free_rate=0.05
    )
    
    print(f"Bond Price: ${bond3.calculate_price():,.2f}")
    print(f"PV of Coupons: ${bond3.get_pv_of_coupons():,.2f}")
    print(f"PV of Principal: ${bond3.get_pv_of_principal():,.2f}")
    
    # Example 4: Using static method for custom PV calculation
    print("\n" + "=" * 60)
    print("Example 4: Using static PV method for custom calculation")
    print("=" * 60)
    
    future_payment = 1000
    discount_rate = 0.05
    periods = 3
    
    pv = CouponBond.calculate_present_value(future_payment, discount_rate, periods)
    print(f"PV of ${future_payment} received in {periods} years at {discount_rate*100}% rate: ${pv:.2f}")