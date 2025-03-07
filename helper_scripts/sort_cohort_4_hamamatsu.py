#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:31:17 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script is to sort the additional hammamatsu slides for Cohort 4: file copy,
re-name as well as label generation (a csv file with two columns "case_id" and "ground_truth")

Parameters
----------
source_folder: str
    Path to the source folder containing hospital subfolders.

target_folder: str
    Path to the target folder where files will be copied.

csv_output: str
    Path to save the CSV file containing label information.

"""

import os
import shutil
import argparse
import pandas as pd

# Name mapping dictionary
name_mapping = {
    "01": "S00140809",
    "02": "S00140830",
    "03": "S00140833",
    "04": "S00140900",
    "05": "S00140897",
    "06": "S00140922",
    "07": "S00140812",
    "08": "S00140817",
    "09": "S00140904",
    "10": "S00140814",
    "11": "S00140836",
    "12": "S00140802",
    "13": "S00140916",
    "14": "S00140825",
    "15": "S00141224"
}

# Reverse mapping for quick lookup
case_id_set = set(name_mapping.values())


def copy_and_rename_files(source_folder, target_folder, csv_output):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    csv_data = []

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Extract hospital ID from subfolder name (expecting format XXX_MM)
        parts = subfolder.split("_")
        if len(parts) < 2 or not parts[-1].isdigit():
            continue
        
        hospital_id = int(parts[-1])
        if hospital_id == 1:  # Skip subfolders with _1
            continue
        
        # Process files inside the subfolder
        for file_name in os.listdir(subfolder_path):
            if not file_name.endswith(".ndpi"):
                continue
            
            file_base, ext = os.path.splitext(file_name)
            if file_base not in case_id_set:
                continue  # Skip files not in the mapping
            
            # Find corresponding mapping key
            kk = next(k for k, v in name_mapping.items() if v == file_base)
            
            # Construct new file name
            new_file_name = f"{kk}_{file_base}_h{hospital_id}_hamamatsu{ext}"
            
            # Copy and rename file
            src_path = os.path.join(subfolder_path, file_name)
            dst_path = os.path.join(target_folder, new_file_name)
            shutil.copy2(src_path, dst_path)
            
            # Append to CSV data
            csv_data.append([new_file_name.replace(".ndpi", ""), int(kk)])
    
    # Convert to DataFrame and save CSV
    df = pd.DataFrame(csv_data, columns=["case_id", "ground_truth"])
    df.to_csv(csv_output, index=False)
    
    print(f"Processing complete. Files copied to {target_folder}, and CSV saved to {csv_output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch copy and rename files based on hospital folders")
    parser.add_argument("--source_folder", type=str, 
                        default="/media/digitalpathology/hamamatsu/Hamamatsu_ALL",
                        help="Path to the source folder containing hospital subfolders")
    parser.add_argument("--target_folder", type=str, 
                        default="/media/digitalpathology/variation_project/cohort_4_hamamatsu",
                        help="Path to the target folder where files will be copied")
    parser.add_argument("--csv_output", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_4_hamamatsu.csv",
                        help="Path to save the CSV file containing label information")
    
    args = parser.parse_args()
    copy_and_rename_files(args.source_folder, args.target_folder, args.csv_output)
