#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:23:53 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script organises CSV files (embeddings) stored within subfolders. It scans 
an input directory containing multiple subfolders, each named in the format 
'XX_xxx', where 'XX' represents the ground truth label. It then extracts CSV 
files from these subfolders, preserves their original filenames (including spaces), 
and copies them into a specified output folder. Additionally, it generates a 
summary CSV file that maps each CSV file's name (case ID) to its corresponding 
ground truth label and saves it in a designated output directory with a 
user-specified filename.

Parameters
----------
input_folder: str
    Path to the input folder containing subfolders with CSV files.

output_folder: str
    Path to the output folder where CSV files will be copied.

csv_output_folder: str
    Path to the folder where the summary CSV will be stored.

csv_filename: str
    Filename for the output CSV storing case IDs and ground truths.

"""

import os
import shutil
import time
import argparse
import pandas as pd

#################### define data processing function ####################

def process_data(input_folder, output_folder, csv_output_folder, csv_filename):
    
    # Ensure output folders exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_output_folder, exist_ok=True)
    
    # Prepare the CSV output path
    csv_output_path = os.path.join(csv_output_folder, csv_filename)
    records = []
    
    # Iterate through subfolders in the input folder
    for subfolder in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder)
        
        # Check if it's a directory and matches the pattern XX_xxx
        if os.path.isdir(subfolder_path) and "_" in subfolder:
            try:
                ground_truth = int(subfolder.split("_")[0])  # Extract ground truth from subfolder name
            except ValueError:
                print(f"Skipping invalid folder: {subfolder}")
                continue
            
            # Process each CSV file in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file)
                    case_id = os.path.splitext(file)[0]  # Extract case ID (retain spaces if present)
                    
                    # Copy file to output folder
                    dest_path = os.path.join(output_folder, file)
                    shutil.copy(file_path, dest_path)
                    
                    # Append record
                    records.append([case_id, ground_truth])
    
    # Create and save the DataFrame
    df = pd.DataFrame(records, columns=["case_id", "ground_truth"])
    df.to_csv(csv_output_path + ".csv", index=False)
    print(f"Processing complete. CSV saved at {csv_output_path}")

if __name__ == "__main__":
    # define argument parser
    parser = argparse.ArgumentParser(description="Sort and organise embeddings.")
    parser.add_argument("--input_folder", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1_by_class",
                        help="Path to the input folder containing subfolders with CSV files.")
    parser.add_argument("--output_folder", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1",
                        help="Path to the output folder where CSV files will be copied.")
    parser.add_argument("--csv_output_folder", type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels",
                        help="Path to the folder where the summary CSV will be stored.")
    parser.add_argument("--csv_filename", type=str, 
                        default="cohort_1",
                        help="Filename for the output CSV storing case IDs and ground truths.")
    args = parser.parse_args()
    
    since = time.time()
    
    process_data(args.input_folder, args.output_folder, args.csv_output_folder, args.csv_filename)
    
    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

