#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:16:35 2024

@author: Dr Binghao Chai
@institute: University College London (UCL)

The function of this script is to copy files from a source folder to a target 
folder in batch using a csv file as a reference. In the csv file, there are two 
columns, where the first column shows the primary key, and the second column 
shows the file name.

For the source folder, it may contain subfolders, and each of the subfolders 
may contain subfolders, and so on. Each file can be located in any subfolder.
For each file name, there might be duplicated files located in different subfolders,
but in the case the file name is the same, the file is the same. In this case, 
only one copy of the duplicated file will be copied to the target folder. If a 
file name is not exist in any of the source subfolders, then this file name will
be skipped for copy.


In the file_name column of the csv file, not all files contain an extension, so
a sanity check of extensions is also integrated into this script. The script 
will attempt to find files with the given name followed by one of the expected 
extensions (.svs, .ndpi, .tiff).


Parameters
----------
source_path: str
    The source folder path.

target_path: str
    The target folder path.
    
csv_path: str
    The path for the csv file containing the file names, the first row of the 
    csv file should be the header.
    
exclude: str
    Indicate if any subfolder should not be traversed, if all the subfolders are to be traversed, then enter 'None'.

"""

# package import
import time
import argparse
import warnings
import os
import csv
import shutil

warnings.filterwarnings("ignore") # ignore warnings
since = time.time()


# arguments definition
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--source_path",
        type=str, 
        default="/home/digitalpathology/Desktop/RDS/digital_pathology/WSI", 
        help="The source folder path." 
        )
    parser.add_argument(
        "--target_path",
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/source_wsi/01_external", 
        help="The target folder path." 
        )
    parser.add_argument(
        "--csv_path",
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/metadata/01_external.csv", 
        help="The path for the csv file containing the file names, the first row \
            of the csv file should be the header." 
        )
    parser.add_argument(
        "--exclude",
        type=str, 
        default="external_paul,B_Chai", 
        help="Indicate if any subfolder should not be traversed, if all the subfolders \
            are to be traversed, then enter 'None'." 
        )
        
    opt = parser.parse_args()
    print(opt)


# Parse the exclude argument into a list
if opt.exclude is None or opt.exclude.lower() == 'none':
    exclude_folders = []
else:
    exclude_folders = opt.exclude.split(',')


# Chech if the target folder exists
if not os.path.exists(opt.target_path):
    os.makedirs(opt.target_path)


# Define a function to traverse and copy files recursively in all subdirectories
def copy_files(local_folder, remote_folder, csv_file, exclude_folders):
    
    # Read the CSV file
    with open(csv_file, newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if there is one
        file_names = [row[1] for row in reader]

    # Define possible extensions
    possible_extensions = ['.svs', '.ndpi', '.tiff']

    # Create a set to keep track of copied files to avoid duplication
    copied_files = set()

    # Traverse the remote folder and its subfolders
    for root, dirs, files in os.walk(remote_folder):
        
        for exclude_folder in exclude_folders:
            if exclude_folder in dirs:
                dirs.remove(exclude_folder)  # Don't traverse these directories
        
        for name in files:
            
            # Check if the file is in the list with or without an extension
            base_name, ext = os.path.splitext(name)
            
            if name in file_names and name not in copied_files:
                copy_file(root, name, local_folder, copied_files)
                
            elif ext and base_name in file_names and name not in copied_files:
                copy_file(root, name, local_folder, copied_files)
                
            elif not ext and base_name in file_names and name not in copied_files:
                for extension in possible_extensions:
                    extended_name = base_name + extension
                    if extended_name in files and extended_name not in copied_files:
                        copy_file(root, extended_name, local_folder, copied_files)
                        break


# Define the actual function for copying files
def copy_file(root, name, local_folder, copied_files):
    
    # Construct full file paths
    source = os.path.join(root, name)
    destination = os.path.join(local_folder, name)

    # Check if the file already exists in the target folder
    if not os.path.exists(destination):  
        print(f"Processing: {source}")
        # Copy file to the local folder
        shutil.copy2(source, destination)
        copied_files.add(name)
        print(f"Copied: {name}")
    else:
        print(f"Skipped (already exists): {name}")  


copy_files(opt.target_path, opt.source_path, opt.csv_path, exclude_folders=exclude_folders)

# debug print    
time_elapsed = time.time() - since
print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)) 

