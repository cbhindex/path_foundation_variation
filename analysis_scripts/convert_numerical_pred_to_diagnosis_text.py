#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:56:43 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script convert the numerical label in the case-level prediction result 
(csv files) to diagnosis in the text for pathologists to read.

"""

import pandas as pd

# Define the mapping dictionary for numerical values to diagnosis text
diagnosis_mapping = {
    1: "MPNST",
    2: "Dermatofibrosarcoma protuberans",
    3: "Neurofibroma",
    4: "Nodular fasciitis",
    5: "Desmoid Fibromatosis",
    6: "Synovial sarcoma",
    7: "Lymphoma",
    8: "Glomus tumour",
    9: "Intramuscular myxoma",
    10: "Ewing",
    11: "Schwannoma",
    12: "Myxoid liposarcoma",
    13: "Leiomyosarcoma",
    14: "Solitary fibrous tumour",
    15: "Low grade fibromyxoid sarcoma"
}

# Define the input and output file paths (Change these when processing another file)
input_file = "/home/digitalpathology/workspace/path_foundation_stain_variation/output/cohort_4/cohort_4_individual_results.csv"  # Change this for each file
output_file = "/home/digitalpathology/workspace/path_foundation_stain_variation/output/cohort_4/cohort_4_individual_results_text.csv"  # Change accordingly

# Load the CSV file
df = pd.read_csv(input_file)

# Columns to replace numerical values with diagnosis text
columns_to_replace = ["ground_truth", "top_1_pred", "top_2_pred", "top_3_pred", "top_4_pred", "top_5_pred"]

# Replace numerical values with diagnosis text
for col in columns_to_replace:
    if col in df.columns:
        df[col] = df[col].map(diagnosis_mapping)

# Save the modified file
df.to_csv(output_file, index=False)

print(f"Modified file saved at: {output_file}")