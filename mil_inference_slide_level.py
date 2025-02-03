#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:47:01 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script is to run the MIL classifier (with two different strategies for tile-
selection within slides) on embeddings on the hold-out testing slides. (Pipeline 2)

Parameters
----------
embedding_path: str
    The root folder containing embeddings. This folder contains multiple subfolders
    where each subfolder represents a class (ground truth). Each CSV file in these
    subfolders represents a slide and contains tile embeddings.

ground_truth: int
    The ground truth label for this batch of embeddings.

model: str
    The path for storing trained model, should load the state dictionary only.

output_folder: str
    The output folder path for slide-level results.

k_tiles: int
    K tiles selected from each of the slides for the multi-instance classifier.

"""

# Package Import
import os
from tqdm import tqdm
import time
import argparse
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

since = time.time()

# Arguments definition
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--embedding_path",
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/embeddings/15_external",
        help="The root folder containing embeddings. This folder contains subfolders "
             "for each class, and CSV files inside represent slide embeddings."
    )    
    parser.add_argument(
        "--ground_truth",
        type=int, 
        default="15", 
        help="The ground truth label for all slides in the folder."
    )
    parser.add_argument(
        "--model",
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation/model/mil_best_model_state_dict_random.pth",
        help="The path for storing trained model, should load the state dictionary only." 
    )
    parser.add_argument(
        "--output_folder",
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/output",
        help="The output folder path for slide-level results." 
    )
    parser.add_argument(
        "--k_tiles",
        type=int, 
        default=500, 
        help="K tiles selected from each of the slides for the multi-instance classifier." 
    )  

    opt = parser.parse_args()
    print(opt)

# Ensure output folder exists
os.makedirs(opt.output_folder, exist_ok=True)

#################### Define helper classes and functions ####################

# Define the dataset class (Restored original version)
class SlideBagDataset(Dataset):
    def __init__(self, slide_patch_features, slide_labels):
        """
        slide_patch_features: list of torch.Tensor, where each element is a tensor
                              of shape [n_patches_in_bag, 384] for a slide.
        slide_labels: list of int, where each element is the slide-level label.
        """
        self.slide_patch_features = slide_patch_features
        self.slide_labels = slide_labels

    def __len__(self):
        return len(self.slide_patch_features)  # Number of slides (bags)

    def __getitem__(self, idx):
        patch_features = self.slide_patch_features[idx]  # Shape: [n_patches_in_bag, 384]
        label = self.slide_labels[idx]  # Scalar label (e.g., subtype ID)
        return patch_features, label

# Random tile selection
def random_tile_selection(patch_features, K=opt.k_tiles):
    if patch_features.shape[0] <= K:
        return patch_features  # Return all patches if fewer than K
    else:
        selected_indices = random.sample(range(patch_features.shape[0]), K)
        return patch_features[selected_indices]

#################### Define the model and load the trained model ####################

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, attention_dim, num_classes):
        super(AttentionMIL, self).__init__()
        self.attention = nn.Linear(input_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        attention_weights = torch.tanh(self.attention(x))
        attention_weights = F.softmax(self.attention_score(attention_weights), dim=0)
        
        # Weighted sum of patch features
        slide_representation = torch.sum(attention_weights * x, dim=0)
        
        # Slide-level classification
        output = self.classifier(slide_representation)
        return output, attention_weights

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionMIL(input_dim=384, attention_dim=128, num_classes=15).to(device)
model.load_state_dict(torch.load(opt.model))
model.eval()

#################### Load the external test data ####################

def load_test_data(embedding_path, ground_truth, K=opt.k_tiles):
    slide_patch_features = []
    slide_labels = []
    slide_filenames = []  # Store filenames for correct slide_id mapping

    # Sort filenames to maintain order
    slide_files = sorted([f for f in os.listdir(embedding_path) if f.endswith(".csv")])

    with tqdm(total=len(slide_files), desc="Loading test data", unit="slide") as pbar:
        for filename in slide_files:
            file_path = os.path.join(embedding_path, filename)
            df_csv = pd.read_csv(file_path)

            # Extract features (skip first two columns x_coord, y_coord)
            patch_features = df_csv.iloc[:, 2:].values  

            # Convert to PyTorch tensor
            patch_features = torch.tensor(patch_features, dtype=torch.float32)

            # Apply tile selection
            selected_patches = random_tile_selection(patch_features, K)

            # Store patches, labels, and slide filenames
            slide_patch_features.append(selected_patches)
            slide_labels.append(ground_truth)
            slide_filenames.append(filename)  # Store actual filename

            pbar.update(1)

    return slide_patch_features, slide_labels, slide_filenames

# Load and sort data
slide_patch_features, slide_labels, slide_filenames = load_test_data(opt.embedding_path, opt.ground_truth)
test_dataset = SlideBagDataset(slide_patch_features, slide_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # No shuffle to maintain order

#################### Model Evaluation ####################

def evaluate_model(model, test_loader, slide_filenames):
    results = []

    with torch.no_grad():
        for idx, (batch_patches, batch_labels) in enumerate(test_loader):
            slide_id = slide_filenames[idx]  # Use actual filename as slide_id
            batch_patches = batch_patches[0].to(device)  # Extract single slide
            batch_labels = batch_labels.item()  # Convert tensor to integer

            output, _ = model(batch_patches)
            probabilities = F.softmax(output, dim=0)
            top3_probs, top3_classes = torch.topk(probabilities, 3)

            top1_correct = int(top3_classes[0].item() == batch_labels)
            top3_correct = int(batch_labels in top3_classes.cpu().numpy())

            results.append([
                slide_id, batch_labels,
                top3_classes[0].item(), top3_probs[0].item(),
                top3_classes[1].item(), top3_probs[1].item(),
                top3_classes[2].item(), top3_probs[2].item(),
                top1_correct, top3_correct
            ])

    df_results = pd.DataFrame(results, columns=[
        "slide_id", "ground_truth",
        "top_1_pred", "top_1_prob",
        "top_2_pred", "top_2_prob",
        "top_3_pred", "top_3_prob",
        "top_1_correct?", "top_3_correct?"
    ])
    
    df_results.to_csv(f"{opt.output_folder}/{opt.ground_truth}.csv", index=False)

# Run evaluation
evaluate_model(model, test_loader, slide_filenames)

# Print runtime
print(f"Task complete in {int((time.time() - since) // 60)}m {int((time.time() - since) % 60)}s")