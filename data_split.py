#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:18:39 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script performs a train-validation-test split on a CSV file containing 
case IDs and ground truth labels. The split is applied to each class individually 
to maintain balance. The user provides a source CSV file, which contains two 
columns: 'case_id' and 'ground_truth'. The train, validation, and test sizes 
are specified as percentages, ensuring they sum to 100. The script processes 
each class separately, applying the requested split to preserve class distribution. 
The resulting split datasets are returned as DataFrames.

Parameters
----------
source_csv: str
    Path to the input CSV file containing case IDs and ground truths.

train_size: int
    Percentage of training size (0-100).

val_size: int
    Percentage of validation size (0-100).

test_size: int
    Percentage of testing size (0-100).
    
output_folder: str
    Path to the output folder where split files will be stored.

"""

import os
import argparse
import time
import pandas as pd
from sklearn.model_selection import train_test_split

#################### define data processing function ####################

def split_data(source_csv, train_size, val_size, test_size):    
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
    parser = argparse.ArgumentParser(description="Train-validation-test split for dataset.")
    parser.add_argument("--source_csv", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_1.csv",
                        help="Path to the input CSV file containing case IDs and ground truths.")
    parser.add_argument("--train_size", type=int, default=60,
                        help="Percentage of training size (0-100).")
    parser.add_argument("--val_size", type=int, default=20,
                        help="Percentage of validation size (0-100).")
    parser.add_argument("--test_size", type=int, default=20,
                        help="Percentage of testing size (0-100).")
    parser.add_argument("--output_folder", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels",
                        help="Path to the output folder where split files will be stored.")
    
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