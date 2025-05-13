#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 00:35:29 2025

@author: digitalpathology
"""

import os
from pathlib import Path
import pandas as pd

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


find_missing_files_by_file_id(
    "/media/digitalpathology/b_chai/trident_outputs/cohort_1/20x_224px_0px_overlap/features_uni_v2", 
    "/home/digitalpathology/workspace/Scanning_variation_BC.xlsx"
    )