#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:18:39 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Split a case-level label table into train/validation/test CSV files.

Expected input columns are:
1. ``case_id``
2. ``ground_truth``

The split is performed class-by-class to keep class balance approximately
consistent across train/validation/test subsets. Split ratios are expressed as
percentages and must sum to 100.

CLI Arguments
-------------
--source_csv : str
    Input CSV with columns ``case_id`` and ``ground_truth``.
--train_size : int
    Training split percentage.
--val_size : int
    Validation split percentage.
--test_size : int
    Test split percentage.
--output_folder : str
    Output directory for generated split CSV files.
"""

import os
import argparse
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Data Split Function
# -----------------------------------------------------------------------------

def split_data(source_csv, train_size, val_size, test_size):    
    """
    Split a labeled case table into train/validation/test subsets.

    Parameters
    ----------
    source_csv : str
        Input CSV path with columns ``case_id`` and ``ground_truth``.
    train_size : int
        Training split percentage.
    val_size : int
        Validation split percentage.
    test_size : int
        Test split percentage.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """
    # Check if split percentages sum to 100
    if train_size + val_size + test_size != 100:
        raise ValueError("Train, validation, and test sizes must sum to 100.")
    
    # Load the CSV file
    df = pd.read_csv(source_csv)
    
    if "case_id" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("CSV file must contain 'case_id' and 'ground_truth' columns.")
    
    train_records = []
    val_records = []
    test_records = []
    
    # Group by ground truth to split per class
    for class_label, group in df.groupby("ground_truth"):
        train, temp = train_test_split(group, train_size=train_size/100, stratify=group["ground_truth"], random_state=42)
        val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), stratify=temp["ground_truth"], random_state=42)
        
        train_records.append(train)
        val_records.append(val)
        test_records.append(test)
    
    # Concatenate results
    train_df = pd.concat(train_records)
    val_df = pd.concat(val_records)
    test_df = pd.concat(test_records)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # define argument parser
    parser = argparse.ArgumentParser(
        description="Split a labeled case CSV into train/validation/test CSV files."
    )
    parser.add_argument("--source_csv", type=str, required=True,
                        help="Input CSV with columns: case_id, ground_truth.")
    parser.add_argument("--train_size", type=int, default=60,
                        help="Training split percentage (0-100).")
    parser.add_argument("--val_size", type=int, default=20,
                        help="Validation split percentage (0-100).")
    parser.add_argument("--test_size", type=int, default=20,
                        help="Test split percentage (0-100).")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output directory for generated split CSV files.")
    
    args = parser.parse_args()
    
    since = time.time()
    
    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Perform data split
    train_df, val_df, test_df = split_data(args.source_csv, args.train_size, args.val_size, args.test_size)
    
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(args.source_csv))[0]
    
    # Save to output files
    train_path = os.path.join(args.output_folder, f"{filename}_train.csv")
    val_path = os.path.join(args.output_folder, f"{filename}_val.csv")
    test_path = os.path.join(args.output_folder, f"{filename}_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train, validation, and test splits saved in {args.output_folder}")
    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
