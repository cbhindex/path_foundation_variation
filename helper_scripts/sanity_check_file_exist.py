#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:41:19 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This Python script checks for consistency between the files in a given folder 
and the entries in a CSV file. It compares the basenames (without extensions) 
of all files in the folder with the case_id column in the CSV. If all files are 
accounted for and matched correctly, the script confirms that everything is 
consistent. If discrepancies exist, it reports which files are present in the 
folder but missing from the CSV, and which entries in the CSV do not have a 
corresponding file in the folder. This ensures that all expected files are 
properly tracked and documented.

"""

import os
import pandas as pd

# ==== User Input ====
folder_path = '/media/digitalpathology/b_chai/trident_outputs/scan_exp_embeddings_all_in_one/3dhistech/uni_v2'  # Replace with the actual folder path
csv_path = '/home/digitalpathology/workspace/path_foundation_stain_variation/labels_14classes/scan_exp/scan_exp_8_slides.csv'   # Replace with the actual CSV file path
# =====================

# Step 1: Get file basenames (without extension) from the folder
folder_files = set(
    os.path.splitext(f)[0] for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
)

# Step 2: Read case_ids from CSV
df = pd.read_csv(csv_path)
csv_case_ids = set(df['case_id'].astype(str))

# Step 3: Compare
files_not_in_csv = folder_files - csv_case_ids
csv_not_in_folder = csv_case_ids - folder_files

# Step 4: Output results
if not files_not_in_csv and not csv_not_in_folder:
    print("All file basenames in the folder are consistent with the 'case_id' column in the CSV.")
else:
    print("Inconsistencies found:")
    if files_not_in_csv:
        print("\nFiles in folder but not in CSV:")
        for f in sorted(files_not_in_csv):
            print(f"  - {f}")
    if csv_not_in_folder:
        print("\nRecords in CSV but file not found in folder:")
        for c in sorted(csv_not_in_folder):
            print(f"  - {c}")