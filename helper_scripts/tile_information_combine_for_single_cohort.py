#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 01:04:19 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Previously, the tile generation for each cohort was applied on each subtype basis,
this script combines the tile information for each cohort.

"""

import os
import shutil
import pandas as pd

def merge_folders(root_folder, target_folder):
    """
    Merges subfolders (masks, patches, stitches) and process_list_autogen.csv from multiple subfolders
    inside root_folder into a single target_folder.
    """
    # Ensure target subfolders exist
    subfolders = ["masks", "patches", "stitches"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(target_folder, subfolder), exist_ok=True)

    # Collect and merge CSV files
    csv_files = []
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # Copy files from masks, patches, stitches
            for sub in subfolders:
                src_path = os.path.join(subfolder_path, sub)
                if os.path.exists(src_path):
                    for file in os.listdir(src_path):
                        src_file = os.path.join(src_path, file)
                        dst_file = os.path.join(target_folder, sub, file)
                        if not os.path.exists(dst_file):  # Avoid overwriting files
                            shutil.copy2(src_file, dst_file)

            # Process CSV file
            csv_path = os.path.join(subfolder_path, "process_list_autogen.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                csv_files.append(df)

    # Merge all CSV files, keeping only the first header
    if csv_files:
        merged_df = pd.concat(csv_files, ignore_index=True)
        csv_target_path = os.path.join(target_folder, "process_list_autogen.csv")
        merged_df.to_csv(csv_target_path, index=False)

    print(f"Merge completed. Files copied to: {target_folder}")

# Example usage:
root_folder = "/home/digitalpathology/workspace/path_foundation_stain_variation/tiles/cohort_4_no_scan_variation_by_subtype"  # Change to your root folder
target_folder = "/home/digitalpathology/workspace/path_foundation_stain_variation/tiles/cohort_4_no_scan_variation"  # Change to your target folder
merge_folders(root_folder, target_folder)

