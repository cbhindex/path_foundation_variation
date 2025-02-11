#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:56:28 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Helper classes for pytorch env (this is the dominant env) for the training and 
testing of classifier.

"""

# package loading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Define the slide-level bag data structure for the slide-level MIL
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
        # Return the number of slides (bags)
        return len(self.slide_patch_features)

    def __getitem__(self, idx):
        # Return the patch features and label for the slide at index `idx`
        patch_features = self.slide_patch_features[idx]  # Shape: [n_patches_in_bag, 384]
        label = self.slide_labels[idx]  # Scalar label (e.g., subtype ID)
        return patch_features, label
    
# Define the attention mechanism and model
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, attention_dim, num_classes):
        super(AttentionMIL, self).__init__()
        self.attention = nn.Linear(input_dim, attention_dim)  # Attention mechanism
        self.attention_score = nn.Linear(attention_dim, 1)  # Attention score layer
        self.classifier = nn.Linear(input_dim, num_classes)  # Slide-level classifier

    def forward(self, x):
        # x is of shape [n_patches_in_bag, input_dim]
        attention_weights = torch.tanh(self.attention(x))  # Shape: [n_patches_in_bag, attention_dim]
        attention_weights = F.softmax(self.attention_score(attention_weights), dim=0)  # Shape: [n_patches_in_bag, 1]

        # Weighted sum of patch features
        slide_representation = torch.sum(attention_weights * x, dim=0)  # Shape: [input_dim]
        
        # Slide-level classification
        output = self.classifier(slide_representation)  # Shape: [num_classes]
        
        return output, attention_weights
    
# Define a new class to extract slide-level embeddings
class AttentionMIL_EmbeddingExtractor(AttentionMIL):
    def forward(self, x):
        """
        Modified forward method to return only slide-level embeddings.
        """
        with torch.no_grad():
            attention_weights = torch.tanh(self.attention(x))
            attention_weights = F.softmax(self.attention_score(attention_weights), dim=0)
            slide_representation = torch.sum(attention_weights * x, dim=0)  # Slide embedding
        return slide_representation
    
    