#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:15:04 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script is to sort the additional 3d-histech slides for Cohort 4: file copy,
re-name as well as label generation (a csv file with two columns "case_id" and "ground_truth")
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
            if not file_name.endswith(".mrxs"):
                continue
            
            file_base, ext = os.path.splitext(file_name)
            file_parts = file_base.split("_")
            if len(file_parts) < 2:
                continue
            
            case_id = file_parts[0]
            something_else = "_".join(file_parts[1:])
            
            if case_id not in case_id_set:
                continue  # Skip files not in the mapping
            
            # Find corresponding mapping key
            class_id = next(k for k, v in name_mapping.items() if v == case_id)
            
            # Construct new file name
            new_file_name = f"{class_id}_{case_id}_{something_else}_3dhistect{ext}"
            
            # Copy and rename file
            src_path = os.path.join(subfolder_path, file_name)
            dst_path = os.path.join(target_folder, new_file_name)
            shutil.copy2(src_path, dst_path)
            
            # Copy corresponding subfolder
            subsubfolder_path = os.path.join(subfolder_path, file_base)
            dst_subsubfolder_path = os.path.join(target_folder, new_file_name.replace(".mrxs", ""))
            if os.path.exists(subsubfolder_path) and os.path.isdir(subsubfolder_path):
                shutil.copytree(subsubfolder_path, dst_subsubfolder_path, dirs_exist_ok=True)
            
            # Append to CSV data
            csv_data.append([new_file_name.replace(".mrxs", ""), int(class_id)])
    
    # Convert to DataFrame and save CSV
    df = pd.DataFrame(csv_data, columns=["case_id", "ground_truth"])
    df.to_csv(csv_output, index=False)
    
    print(f"Processing complete. Files copied to {target_folder}, and CSV saved to {csv_output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch copy and rename files with associated subfolders")
    parser.add_argument("--source_folder", type=str, 
                        default="/media/digitalpathology2/My Passport/3d histech/HCA_Batch1",
                        help="Path to the source folder containing hospital subfolders")
    parser.add_argument("--target_folder", type=str, 
                        default="/media/digitalpathology2/My Passport/cohort_4_3dhistech",
                        help="Path to the target folder where files will be copied")
    parser.add_argument("--csv_output", type=str, 
                        default="/media/digitalpathology2/My Passport/cohort_4_3dhistech.csv",
                        help="Path to save the CSV file containing label information")
    
    args = parser.parse_args()
    copy_and_rename_files(args.source_folder, args.target_folder, args.csv_output)
