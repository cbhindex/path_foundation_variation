#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:01:16 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Helper functions for pytorch env (this is the dominant env) for the training and 
testing of classifier.

"""

# package import
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.helper_class_pytorch import SlideBagDataset

# Define a custom collate function for variable-size patch features,
# This is to handle the variable number of patches across slides (bags).
def collate_fn_variable_size(batch):
    # Extract patch features and labels from the batch
    batch_patches = [
        torch.tensor(item[0], dtype=torch.float32) if isinstance(item[0], np.ndarray) 
        else item[0] for item in batch
        ]  # List of patch features (varying sizes)
    labels = torch.tensor([item[1] for item in batch])  # List of slide-level labels (same size)
    
    return batch_patches, labels

# # Example patch features for 3 slides (variable number of patches per slide)
# slide_patch_features = [
#     torch.randn(120, 384),  # Slide 1: 120 patches, 384 features
#     torch.randn(200, 384),  # Slide 2: 200 patches, 384 features
#     torch.randn(90, 384)    # Slide 3: 90 patches, 384 features
#     ]
# slide_labels = [3, 7, 1]  # Example labels for the 3 slides

# dataset = SlideBagDataset(slide_patch_features, slide_labels) # Create the dataset

# # Check the number of slides
# print(f'Number of slides: {len(dataset)}')
# # Access the first slide's patches and label as example
# patches, label = dataset[0]
# print(f'Number of patches in slide 1: {patches.shape[0]}, Label: {label}')

# # Example of creating a DataLoader with the custom collate function
# data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_variable_size)

# # Iterate through the DataLoader (example)
# for batch_patches, batch_labels in data_loader:
#     print(f"Batch labels: {batch_labels}")
#     for i, patches in enumerate(batch_patches):
#         print(f"Slide {i+1} has {patches.shape[0]} patches")

# Define a custom collate function for random instance (patch) selection from 
# a bag (slide)
def collate_fn_random_sampling(batch, k_tiles=100):
    """
    Custom collate function to handle variable-size patch features and randomly
    select K patches from each bag (slide). If the bag has fewer than or equal to K patches,
    all patches are selected.
    
    Parameters
    ----------
    batch: list of tuples (patch_features, label)
        patch_features is a tensor of shape [n_patches_in_bag, 384] representing 
        the patch feature vectors, and label is the slide-level label.
        
    K: int 
        number of patches to sample from each bag. If the bag has fewer than or 
        equal to K patches, all patches are selected.
    
    Returns
    -------
    batch_patches: list of tensors 
        list of tensors with randomly selected patches from each slide.
        
    labels: tensor 
        tensor of slide-level labels.
    """
    batch_patches = []
    labels = torch.tensor([item[1] for item in batch])  # List of slide-level labels

    for item in batch:
        patch_features = item[0]  # Patch features for the current slide
        n_patches_in_bag = patch_features.shape[0]  # Get number of patches
        
        # If the bag has fewer than or equal to K patches, select all patches
        if n_patches_in_bag <= k_tiles:
            selected_patches = patch_features  # Select all patches
        else:
            # Randomly select K patches
            selected_indices = random.sample(range(n_patches_in_bag), k_tiles)
            selected_patches = patch_features[selected_indices]
        
        # Convert to tensor if it's still a NumPy array
        if isinstance(selected_patches, np.ndarray):
            selected_patches = torch.tensor(selected_patches, dtype=torch.float32)
        
        # Append the selected patches to the batch
        batch_patches.append(selected_patches)
    
    return batch_patches, labels

