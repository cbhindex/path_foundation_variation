#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:40:09 2025

@@author: Dr Binghao Chai
@institute: University College London (UCL)

This script sorts out cohort 2 data for re-naming.

"""

import os

# Define the root directory
root_dir = "/home/digitalpathology/workspace/path_foundation_stain_variation/source_wsi"

# Mapping of old names to new identifiers
name_mapping = {
    "01.svs": "S00140809",
    "02.svs": "S00140830",
    "03.svs": "S00140833",
    "04.svs": "S00140900",
    "05.svs": "S00140897",
    "06.svs": "S00140922",
    "07.svs": "S00140812",
    "08.svs": "S00140817",
    "09.svs": "S00140904",
    "10.svs": "S00140814",
    "11.svs": "S00140836",
    "12.svs": "S00140802",
    "13.svs": "S00140916",
    "14.svs": "S00140825",
    "15.svs": "S00141224"
}

# Iterate over subfolders (h2, h3, h4, ...)
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    
    # Ensure it's a directory
    if os.path.isdir(subfolder_path):
        
        # Iterate over files in the subfolder
        for file_name in os.listdir(subfolder_path):
            
            # Check if the file matches one in the mapping
            if file_name in name_mapping:
                
                # Construct the new file name
                new_file_name = f"{file_name.split('.')[0]}_{name_mapping[file_name]}_{subfolder}.svs"
                
                # Full paths for renaming
                old_file_path = os.path.join(subfolder_path, file_name)
                new_file_path = os.path.join(subfolder_path, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")

print("Renaming process completed.")
