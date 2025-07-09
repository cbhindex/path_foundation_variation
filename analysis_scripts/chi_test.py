#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 01:54:31 2025

@author: digitalpathology
"""

from scipy.stats import chi2_contingency
from math import floor

a = 0.782
b = 0.803

a_1 = floor(280 * a)
b_1 = floor(813 * b)


# Path Foundation example
table = [
    [a_1, 280 - a_1],  # Cohort 1
    [b_1, 813 - b_1],  # Cohort 4
]

chi2, p, dof, expected = chi2_contingency(table)
print(f"Chi-squared p-value: {p:.4e}")