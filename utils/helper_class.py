#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:05:26 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Helper classes.

"""

# packages import
import h5py
import numpy as np
import tensorflow as tf

# define the dataset class as a bridge to align the .h5 format tiles
class WholeSlideBagTF:
    def __init__(self, file_path, wsi, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (object): Whole Slide Image object (e.g., from OpenSlide).
            img_transforms (callable, optional): Optional transform to be applied 
            on a sample.
        """
        self.file_path = file_path
        self.wsi = wsi
        self.img_transforms = img_transforms

        # Read metadata from the .h5 file
        with h5py.File(self.file_path, "r") as f:
            # dset = f["coords"]
            # self.length = len(dset)
            self.coords = f["coords"][:]  # Load all coordinates into memory
            self.patch_level = f["coords"].attrs["patch_level"]
            self.patch_size = f["coords"].attrs["patch_size"]
        
        self.length = len(self.coords)  # Number of patches

        self.summary()

    def __len__(self):
        """Return the total number of patches."""
        return self.length

    def summary(self):
        """Print summary information about the dataset."""
        with h5py.File(self.file_path, "r") as f:
            dset = f["coords"]
            for name, value in dset.attrs.items():
                print(f"{name}: {value}")
        
        print("\nFeature extraction settings:")
        print(f"Transformations: {self.img_transforms}")

    def _get_patch(self, coord):
        """Extract a patch from the WSI using the given coordinate."""
        # Extract the patch using the WSI object
        img = self.wsi.read_region(coord, self.patch_level, 
                                   (self.patch_size, self.patch_size)).convert("RGB")
        return img

    def _process_sample(self, idx):
        """Process a single sample: extract a patch and apply transformations."""
        coord = self.coords[idx]  # Get coordinate
        img = self._get_patch(coord)  # Extract patch
        if self.img_transforms:
            img = self.img_transforms(img)  # Apply transformations
        
        # Convert PIL image to TensorFlow tensor
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
        coord = tf.convert_to_tensor(coord, dtype=tf.int32)
        return img, coord

    def get_tf_dataset(self):
        """Return a TensorFlow dataset."""
        dataset = tf.data.Dataset.range(self.length)  # Create dataset of indices
        dataset = dataset.map(
            lambda idx: tf.py_function(
                func=self._process_sample,
                inp=[idx],
                Tout=(tf.float32, tf.int32)  # Image as uint8 tensor, coord as int32 tensor
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset