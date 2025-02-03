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