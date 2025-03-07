#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:58:22 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script sorts out cohort 3 data for re-naming.

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

# Define source and target paths
source_root = "/home/digitalpathology/workspace/path_foundation_stain_variation/source_wsi"
target_folder = os.path.join(source_root, "cohort_3")
csv_output = os.path.join(source_root, "label_info.csv")

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

def copy_files_with_rename(source_subfolder, file_ext, rename_suffix, copy_folder=False):
    subfolder_path = os.path.join(source_root, source_subfolder)
    
    csv_data = []
    
    for file_name in os.listdir(subfolder_path):
        if not file_name.endswith(file_ext):
            continue
        
        file_base, ext = os.path.splitext(file_name)
        file_parts = file_base.split("_")
        
        # Find case_id in file name
        case_id = next((c for c in case_id_set if c in file_parts), None)
        if not case_id:
            continue
        
        # Find corresponding class ID
        class_id = next(k for k, v in name_mapping.items() if v == case_id)
        
        # Construct new file name
        new_file_name = f"{class_id}_{case_id}_h1_{rename_suffix}{ext}"
        
        # Copy and rename file
        src_path = os.path.join(subfolder_path, file_name)
        dst_path = os.path.join(target_folder, new_file_name)
        shutil.copy2(src_path, dst_path)
        
        # Copy corresponding folder if required
        if copy_folder:
            subsubfolder_path = os.path.join(subfolder_path, file_base)
            dst_subsubfolder_path = os.path.join(target_folder, new_file_name.replace(ext, ""))
            if os.path.exists(subsubfolder_path) and os.path.isdir(subsubfolder_path):
                shutil.copytree(subsubfolder_path, dst_subsubfolder_path, dirs_exist_ok=True)
        
        # Append to CSV data
        csv_data.append([new_file_name.replace(ext, ""), int(class_id)])
    
    return csv_data

# Process all three cohorts
csv_data = []
csv_data.extend(copy_files_with_rename("Cohort_3_3dhistech", ".mrxs", "3dhistect", copy_folder=True))
csv_data.extend(copy_files_with_rename("Cohort_3_hamamatsu", ".ndpi", "hamamatsu"))
csv_data.extend(copy_files_with_rename("Cohort_3_aperio", ".svs", "aperio"))

# Save CSV
if csv_data:
    df = pd.DataFrame(csv_data, columns=["case_id", "ground_truth"])
    df.to_csv(csv_output, index=False)
    print(f"Processing complete. Files copied to {target_folder}, and CSV saved to {csv_output}.")
else:
    print("No valid files found for processing.")
