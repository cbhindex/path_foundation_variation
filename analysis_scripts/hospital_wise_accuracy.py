#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:53:07 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script calculates hospital-wise accuracy using the original metadata
and case-level prediction results (CSV) as input.

Parameters
----------
csv_file: str
    Path to the prediction results CSV file.
    
excel_file: str
    PPath to the metadata Excel file.
    
csv_file: str
    Path to the prediction results CSV file.
    
output_dir: str
    Directory to save the output CSV files. 
   
"""

import pandas as pd
import os
import argparse

# define the key function
def calculate_hospital_accuracy(csv_file_path, excel_file_path, output_dir):
    # Load the CSV file
    csv_df = pd.read_csv(csv_file_path)

    # Load the Excel file and "Images" sheet
    xls = pd.ExcelFile(excel_file_path)
    images_df = pd.read_excel(xls, sheet_name="Images")

    # Extract relevant columns and clean
    images_df = images_df[['File ID', 'Staining Institute']].dropna()
    images_df['slide_id'] = images_df['File ID'].apply(lambda x: os.path.splitext(str(x))[0])

    # Merge with prediction CSV
    merged_df = csv_df.merge(images_df[['slide_id', 'Staining Institute']], on='slide_id', how='left')
    merged_df.rename(columns={'Staining Institute': 'staining_institute'}, inplace=True)
    merged_df['staining_institute'] = merged_df['staining_institute'].str.strip().str.replace(" +", " ", regex=True)

    # Compute statistics per staining institute
    staining_stats = merged_df.groupby('staining_institute').agg(
        total_case=('slide_id', 'count'),
        correctly_predicted_case=('top_1_correct', 'sum')
    ).reset_index()
    staining_stats['accuracy'] = staining_stats['correctly_predicted_case'] / staining_stats['total_case']

    # Sort the statistics
    staining_stats_sorted = staining_stats.sort_values(
        by=['accuracy', 'total_case', 'staining_institute'], ascending=[False, False, True]
    )

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    stats_output_path = os.path.join(output_dir, "staining_institute_accuracy_sorted_final.csv")
    slide_info_output_path = os.path.join(output_dir, "slide_staining_info.csv")

    # Save outputs
    staining_stats_sorted.to_csv(stats_output_path, index=False)
    merged_df[['slide_id', 'top_1_correct', 'staining_institute']].to_csv(slide_info_output_path, index=False)

    print("Processing complete. Files saved:")
    print(f"1. Staining Institute Accuracy Summary (Sorted): {stats_output_path}")
    print(f"2. Slide Staining Information: {slide_info_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate hospital-wise accuracy from prediction results and metadata.")
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to the prediction results CSV file.')
    parser.add_argument('--excel_file', type=str, required=True,
                        help='Path to the metadata Excel file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output CSV files.')

    args = parser.parse_args()
    
    calculate_hospital_accuracy(args.csv_file, args.excel_file, args.output_dir)
