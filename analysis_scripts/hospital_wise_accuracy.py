#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:53:07 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script calculate the hospital wise accuracy using the original metadata
and the case-level prediction results (csv) as input.

"""

import pandas as pd
import os

# Load the CSV file
csv_file_path = "/home/digitalpathology/workspace/path_foundation_stain_variation/output/cohort_4/cohort_4_individual_results.csv"
csv_df = pd.read_csv(csv_file_path)

# Load the Excel file
excel_file_path = "/home/digitalpathology/workspace/path_foundation_stain_variation/metadata/cohort_4/external_staining_variation data_refined_BC_23JAN2025.xlsx"
xls = pd.ExcelFile(excel_file_path)

# Load the "Images" sheet
images_df = pd.read_excel(xls, sheet_name="Images")

# Extract the relevant columns and drop NaN values
images_df = images_df[['File ID', 'Staining Institute']].dropna()

# Remove file extensions from 'File ID' to match 'slide_id' in csv_df
images_df['slide_id'] = images_df['File ID'].apply(lambda x: os.path.splitext(str(x))[0])

# Merge the CSV file with images_df on 'slide_id'
merged_df = csv_df.merge(images_df[['slide_id', 'Staining Institute']], on='slide_id', how='left')

# Rename columns for clarity
merged_df.rename(columns={'Staining Institute': 'staining_institute'}, inplace=True)

# Trim whitespace and standardize naming to avoid inconsistencies
merged_df['staining_institute'] = merged_df['staining_institute'].str.strip()
merged_df['staining_institute'] = merged_df['staining_institute'].str.replace(" +", " ", regex=True)

# Compute statistics per staining institute
staining_stats = merged_df.groupby('staining_institute').agg(
    total_case=('slide_id', 'count'),
    correctly_predicted_case=('top_1_correct', 'sum')
).reset_index()

# Compute accuracy
staining_stats['accuracy'] = staining_stats['correctly_predicted_case'] / staining_stats['total_case']

# Sort the staining institute accuracy summary by accuracy (descending), total cases (descending), and alphabetically (ascending)
staining_stats_sorted = staining_stats.sort_values(
    by=['accuracy', 'total_case', 'staining_institute'], ascending=[False, False, True]
)

# Save the final sorted CSV file
staining_stats_sorted_path = "/home/digitalpathology/workspace/path_foundation_stain_variation/output/cohort_4/hospital_accuracy/staining_institute_accuracy_sorted_final.csv"
staining_stats_sorted.to_csv(staining_stats_sorted_path, index=False)

# Create the second CSV file with only three columns
filtered_df = merged_df[['slide_id', 'top_1_correct', 'staining_institute']]
filtered_csv_path = "/home/digitalpathology/workspace/path_foundation_stain_variation/output/cohort_4/hospital_accuracy/slide_staining_info.csv"
filtered_df.to_csv(filtered_csv_path, index=False)

print("Processing complete. Files saved:")
print(f"1. Staining Institute Accuracy Summary (Sorted): {staining_stats_sorted_path}")
print(f"2. Slide Staining Information: {filtered_csv_path}")

