#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 01:54:31 2025

@author: digitalpathology
"""

from scipy.stats import chi2_contingency
from math import floor

a = 0.273
b = 0.515

a_1 = floor(33 * a)
b_1 = floor(33 * b)


# Path Foundation example
table = [
    [a_1, 17 - a_1],  # Cohort 1
    [b_1, 17 - b_1],  # Cohort 4
]

chi2, p, dof, expected = chi2_contingency(table)
print(f"Chi-squared p-value: {p:.4e}")