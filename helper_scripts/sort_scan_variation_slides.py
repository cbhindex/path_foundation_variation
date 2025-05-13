#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 22:44:32 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This Python script is designed to standardize the filenames of .h5 or .csv files 
within a specified folder based on metadata provided in an Excel spreadsheet. 
Each file in the folder may have a filename that exactly matches or contains 
identifiers such as the “ML number on MF” or the “Slide brady ID on MF” found in 
the spreadsheet. The script scans each file in the folder and, if a match is 
found with one of these identifiers, renames the file to {file_id}.h5 or 
{file_id}.csv, using the corresponding file_id from the spreadsheet. In 
addition to renaming files, the script outputs a list of files in the folder 
that could not be matched to any spreadsheet entry, as well as a list of file_ids 
from the spreadsheet that were not used (i.e., had no corresponding file in the 
folder). This ensures consistent file naming and helps identify missing or 
unmatched data.

"""

import os
import pandas as pd
from pathlib import Path

def standardize_filenames(folder_path, spreadsheet_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not os.path.isfile(spreadsheet_path):
        raise FileNotFoundError(f"Spreadsheet not found: {spreadsheet_path}")
    
    # Load spreadsheet
    df = pd.read_excel(spreadsheet_path, dtype=str)
    df = df[['file_id', 'ML number on MF', 'Slide brady ID on MF']].dropna()

    # Create mapping from possible patterns to file_id
    match_dict = {}
    for _, row in df.iterrows():
        file_id = str(row['file_id'])
        ml_number = str(row['ML number on MF'])
        brady_id = str(row['Slide brady ID on MF'])

        match_dict[ml_number] = file_id
        match_dict[brady_id] = file_id

    # Track matches
    matched_file_ids = set()
    unmatched_files = []
    renamed_files = []

    # Determine file extension
    all_files = list(folder.glob("*"))
    if not all_files:
        print("The folder is empty.")
        return
    extension = all_files[0].suffix  # assume all files share the same extension

    for file in all_files:
        name = file.name
        matched = False
        for key, file_id in match_dict.items():
            if key in name:
                new_filename = f"{file_id}{extension}"
                new_path = folder / new_filename

                if file.name != new_filename:
                    os.rename(file, new_path)
                    renamed_files.append((file.name, new_filename))
                matched = True
                matched_file_ids.add(file_id)
                break
        if not matched:
            unmatched_files.append(file.name)

    # Report results
    print("\n=== Renamed Files ===")
    for old, new in renamed_files:
        print(f"{old} -> {new}")

    print("\n=== Unmatched Files in Folder ===")
    if unmatched_files:
        for f in unmatched_files:
            print(f)
    else:
        print("All files matched with spreadsheet entries.")

    print("\n=== Unmatched file_id in Spreadsheet ===")
    spreadsheet_file_ids = set(df['file_id'].astype(str))
    unmatched_file_ids = spreadsheet_file_ids - matched_file_ids
    if unmatched_file_ids:
        for fid in unmatched_file_ids:
            print(fid)
    else:
        print("All file_id entries in spreadsheet matched a file in the folder.")

standardize_filenames(
    "/media/digitalpathology/b_chai/trident_outputs/scan_exp_aperio_synovial_sarcoma_cleaned/20x_512px_0px_overlap/slide_features_titan", 
    "/home/digitalpathology/workspace/Scanning_variation_BC_discrepancy.xlsx"
    )


