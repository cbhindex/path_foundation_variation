#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 00:35:29 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script provides two utility functions to assist with quality control and 
data consolidation of .h5 or .csv feature files used in computational pathology 
workflows. It uses an Excel spreadsheet containing file_id references to manage 
and verify file consistency across different data folders.

1. find_missing_files_by_file_id(folder_path, spreadsheet_path)
This function checks which file_id entries from the given spreadsheet do not 
have a corresponding file in the specified folder. It assumes that filenames 
(excluding extensions) should exactly match the file_id. It prints a list of 
missing file_ids, helping identify incomplete datasets.

2. copy_matched_files_to_output(folder_path, spreadsheet_path, output_path)
This function scans all files in the source folder and copies those whose 
basenames match the file_id entries in the spreadsheet to a specified output 
directory. It preserves the original file extension and logs the number of 
matched files copied. This is particularly useful for merging validated files 
into a clean, centralized folder for downstream processing or analysis.

"""

import os
from pathlib import Path
import pandas as pd
import shutil

def find_missing_files_by_file_id(folder_path, spreadsheet_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not os.path.isfile(spreadsheet_path):
        raise FileNotFoundError(f"Spreadsheet not found: {spreadsheet_path}")

    # Load spreadsheet and extract file_id column
    df = pd.read_excel(spreadsheet_path, dtype=str)
    df = df[['file_id']].dropna()
    file_ids = df['file_id'].astype(str).tolist()

    # Determine file extension from folder contents
    all_files = list(folder.glob("*"))
    if not all_files:
        print("The folder is empty.")
        return
    extension = all_files[0].suffix  # assume all files share the same extension

    # Create a set of basenames (without extension)
    folder_basenames = {f.stem for f in all_files if f.is_file()}

    # Find file_ids that are missing in folder
    missing_file_ids = [fid for fid in file_ids if fid not in folder_basenames]

    print("\n=== Missing file_ids (not found in folder) ===")
    if missing_file_ids:
        for fid in missing_file_ids:
            print(fid)
    else:
        print("All file_id entries in the spreadsheet exist in the folder.")
        
def copy_matched_files_to_output(folder_path, spreadsheet_path, output_path):
    folder = Path(folder_path)
    output = Path(output_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not os.path.isfile(spreadsheet_path):
        raise FileNotFoundError(f"Spreadsheet not found: {spreadsheet_path}")
    if not output.exists():
        output.mkdir(parents=True)

    # Load spreadsheet and extract file_id
    df = pd.read_excel(spreadsheet_path, dtype=str)
    df = df[['file_id']].dropna()
    file_ids = df['file_id'].astype(str).tolist()
    file_ids_set = set(file_ids)

    # Get all files in folder and copy matched ones
    matched_files = []
    for file in folder.iterdir():
        if file.is_file() and file.stem in file_ids_set:
            shutil.copy2(file, output / file.name)
            matched_files.append(file.name)

    print(f"\n=== {len(matched_files)} matched files copied to: {output_path} ===")
    if not matched_files:
        print("No matching files were found and copied.")
    else:
        for fname in matched_files:
            print(fname)
   
# # First check missing ones
# find_missing_files_by_file_id(
#     "/media/digitalpathology/b_chai/trident_outputs/cohort_1/20x_224px_0px_overlap/features_uni_v2", 
#     "/home/digitalpathology/workspace/Scanning_variation_BC.xlsx"
#     )

# Then copy matched files
copy_matched_files_to_output(
#     "/media/digitalpathology/b_chai/trident_outputs/cohort_1/20x_224px_0px_overlap/slide_features_prism",
    "/home/digitalpathology/temp",
    "/home/digitalpathology/workspace/path_foundation_stain_variation/metadata/scan_exp/Scanning_variation_BC_discrepancy.xlsx",
    "/media/digitalpathology/b_chai/trident_outputs/scan_exp/scan_exp_embeddings_all_in_one/3dhistech/prism"
)

