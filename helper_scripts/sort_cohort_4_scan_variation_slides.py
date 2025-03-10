#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 19:27:36 2025

@author: digitalpathology
"""

import os
import shutil

def copy_missing_files(folder_a, folder_b, folder_c):
    """
    Copies files that are in folder B but not in folder A to folder C.
    """
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
    
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))
    
    missing_files = files_b - files_a
    
    for file in missing_files:
        src = os.path.join(folder_b, file)
        dst = os.path.join(folder_c, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied: {file}")
        
if __name__ == "__main__":
    folder_a = "/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_4"  
    folder_b = "/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_4_mix_need_to_be_refined" 
    folder_c = "/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_4_scan_variation_part" 
    
    copy_missing_files(folder_a, folder_b, folder_c)
