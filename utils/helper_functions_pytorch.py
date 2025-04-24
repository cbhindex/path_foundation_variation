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
import os
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import DataLoader

from utils.helper_class_pytorch import SlideBagDataset

#################### define collate functions ####################

def collate_fn_variable_size(batch):
    """
    This collate function is to handle the variable number of patches across s
    lides (bags).
    
    """
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
        
    k_tiles: int 
        number of patches to sample from each bag. If the bag has fewer than or 
        equal to K patches, all patches are selected.
    
    Returns
    -------
    batch_patches: list of tensors 
        list of tensors with randomly selected patches from each slide.
        
    labels: tensor 
        tensor of slide-level labels.
    """
    random.seed(42)
    
    batch_patches = []
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)  # Ensure correct dtype for labels

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

#################### define data loading function ####################

def load_data(folder_path, label_csv):
    """
    This function is mainly used to load data. There are two inputs for this function,
    they are: (1) a folder path containing many csv files, and (2) a path to a
    single csv file serves as the metadata and contains ground truth of each case.
    
    Parameters
    ----------
    folder_path: str
        Path to a folder containing many csv files, the name of these csv files
        are there case IDs. The csv files are generally the embeddings for each 
        of the slides, where the header is x_coord, y_coord, and emb_1, emb_2 ...
        emb_384. The first columns reflects the location of each tile, and the rest
        384 columns shows the feature vector.

    label_csv: str
        Path to the metadate (a single cvsv file). There are two columns for the
        csv file, and the header to be "case_id" and "ground_truth". case_id matches
        the files within {folder_path} folder, and "ground_truth" is an int value
        showing the class/diagnosis.

    Returns
    -------
    patch_features: list of ndarrays
    
    labels: list of ints
    
    slide_filenames: list of str
        List of slide filenames corresponding to the loaded cases.

    """
    # read the metadata containing case_id and their ground_truth
    label_df = pd.read_csv(label_csv)
    case_ids = label_df['case_id'].tolist()
    labels_dict = label_df.set_index('case_id')['ground_truth'].to_dict()
    
    patch_features, labels, slide_filenames = [], [], []
    
    for case_id in case_ids:
        csv_path = os.path.join(folder_path, f"{case_id}.csv")
        # check if embedding csv file exist
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Ignore x_coord, y_coord and keep the rest as feature vectors
            features = df.iloc[:, 2:].values  
            patch_features.append(features)
            labels.append(labels_dict[case_id])
            slide_filenames.append(case_id)  # Store the case_id as slide filename
        # else:
        #     print(f"Warning: {case_id}.csv not found. Skipping this case.")
    
    return patch_features, labels, slide_filenames

def load_data_h5(folder_path, label_csv):
    """
    Loads patch-level feature embeddings and labels from .h5 files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .h5 files. Each file is named by its case ID
        and includes two datasets: 'coords' (N x 2) and 'features' (N x m). Here
        m is the size of embedding vector, for example Path Foundation is 384, and
        UNI-V2 is 1536 etc.
    
    label_csv : str
        Path to the metadata CSV with columns: 'case_id' and 'ground_truth'.

    Returns
    -------
    patch_features : list of np.ndarray
        List of feature arrays per slide, each of shape (N_patches, m_features).
    
    labels : list of int
        List of class labels (ground truth) corresponding to each slide.
    
    slide_filenames : list of str
        List of case IDs corresponding to the slides loaded.
    """
    # Load metadata CSV
    label_df = pd.read_csv(label_csv)
    case_ids = label_df['case_id'].tolist()
    labels_dict = label_df.set_index('case_id')['ground_truth'].to_dict()

    patch_features, labels, slide_filenames = [], [], []

    for case_id in case_ids:
        h5_path = os.path.join(folder_path, f"{case_id}.h5")
        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    if 'features' not in hf or 'coords' not in hf:
                        print(f"Warning: Missing datasets in {case_id}.h5. Skipping.")
                        continue

                    features = hf['features'][:]
                    coords = hf['coords'][:]
                    
                    if features.shape[0] != coords.shape[0]:
                        print(f"Warning: Mismatch in coords/features length for {case_id}. Skipping.")
                        continue

                    patch_features.append(features)  # (N_patches, 1536)
                    labels.append(labels_dict[case_id])
                    slide_filenames.append(case_id)

            except Exception as e:
                print(f"Error reading {case_id}.h5: {e}. Skipping this case.")
        # else:
        #     print(f"Warning: {case_id}.h5 not found. Skipping this case.")

    return patch_features, labels, slide_filenames

#################### generate slide-level embeddings using trained model ####################

def extract_slide_embeddings(model, test_loader, slide_filenames, device):
    model.eval()
    slide_embeddings = []
    slide_ids = []

    with tqdm(total=len(slide_filenames), desc="Extracting Embeddings", unit="slide") as pbar:
        with torch.no_grad():
            for batch_idx, (batch_patches, _) in enumerate(test_loader):
                for i, patches in enumerate(batch_patches):
                    patches = patches.to(device)
                    slide_id = slide_filenames[batch_idx * test_loader.batch_size + i]

                    slide_embedding = model(patches)

                    slide_embeddings.append(slide_embedding.cpu().numpy())
                    slide_ids.append(slide_id)
                    pbar.update(1)

    return np.array(slide_embeddings), slide_ids