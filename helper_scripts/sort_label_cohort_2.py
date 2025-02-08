#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:17:53 2025

@@author: Dr Binghao Chai
@institute: University College London (UCL)

This script sorts out cohort 2 label in a csv spreadsheet.
"""

import os
import pandas as pd

# Define source and output paths
source_folder = "/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_2"
output_csv = "/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_2.csv"

# Initialize a list to store extracted information
records = []

# Loop through the files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".csv"):
        parts = filename.split("_")
        if len(parts) >= 2:
            ground_truth = int(parts[0])  # Convert XX to int
            case_id = "_".join(parts[:3]).replace(".csv", "")  # Extract XX_YY_ZZ without .csv
            records.append([case_id, ground_truth])

# Create a DataFrame
df = pd.DataFrame(records, columns=["case_id", "ground_truth"])

# Save the DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f"CSV file saved successfully at: {output_csv}")