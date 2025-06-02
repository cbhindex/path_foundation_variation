#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:19:38 2025

@@author: Dr Binghao Chai
@institute: University College London (UCL)

This Python script checks whether all .h5 files in a specified folder share the 
same internal data structure. Specifically, it verifies that each file contains 
the same dataset keys and that the shape of each dataset (excluding the first 
dimension, which may vary in size) is consistent across files. The script 
iterates through each .h5 file, compares its structure to a reference, and 
reports any files that deviate from this structure. If inconsistencies are 
found, the names of the problematic files are printed; otherwise, it confirms 
that all files are structurally consistent.
"""

import os
import h5py

def check_h5_structure_consistency(folder_path):
    reference_structure = None
    inconsistent_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    structure = {}
                    for key in f.keys():
                        data_shape = f[key].shape
                        # Only store the shape after the first dimension (number of records can vary)
                        structure[key] = data_shape[1:] if len(data_shape) > 1 else ()

                    if reference_structure is None:
                        reference_structure = structure
                    else:
                        if structure != reference_structure:
                            inconsistent_files.append(filename)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                inconsistent_files.append(filename)

    if inconsistent_files:
        print("Inconsistent HDF5 files found:")
        for fname in inconsistent_files:
            print(f"- {fname}")
    else:
        print("All .h5 files have consistent structure (except record count).")

# Example usage
if __name__ == '__main__':
    folder = '/media/digitalpathology/b_chai/trident_outputs/scan_exp_embeddings_all_in_one/3dhistech/resnet50'  # <- replace with your actual path
    check_h5_structure_consistency(folder)