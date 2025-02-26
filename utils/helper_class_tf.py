#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:05:26 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Helper classes for tensorflow env for embedding generation.

"""

# packages import
import h5py
import numpy as np
import tensorflow as tf
import cv2
import staintools
from PIL import Image
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor

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

# Below Class is very slow, rewrite
class WholeSlideBagTFNorm:
    def __init__(self, file_path, wsi, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (object): Whole Slide Image object (e.g., from OpenSlide).
            img_transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_path = file_path
        self.wsi = wsi
        self.img_transforms = img_transforms

        # Read metadata from the .h5 file
        with h5py.File(self.file_path, "r") as f:
            self.coords = f["coords"][:]  # Load all coordinates into memory
            self.patch_level = f["coords"].attrs["patch_level"]
            self.patch_size = f["coords"].attrs["patch_size"]
        
        self.length = len(self.coords)  # Number of patches

        # Initialize stain normalizer (Vahadane)
        self._stain_normalizer = staintools.StainNormalizer(method='vahadane')
        target_img = cv2.imread("/home/digitalpathology/workspace/path_foundation_stain_variation/assets/485819_colour_sample.tif")
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        self._stain_normalizer.fit(target_img)

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
        x, y = coord
        img = self.wsi.read_region((x, y), self.patch_level, (self.patch_size, self.patch_size)).convert("RGB")
        return np.array(img)  # Convert PIL Image to NumPy

    def _is_tissue_present(self, img, threshold=200, min_tissue_ratio=0.3):
        """
        Check if the tile contains tissue (not mostly background).
        - `threshold`: Pixel intensity above which a pixel is considered background.
        - `min_tissue_ratio`: Minimum proportion of pixels that should be tissue.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tissue_pixels = np.sum(gray < threshold)
        total_pixels = img.shape[0] * img.shape[1]
        return (tissue_pixels / total_pixels) > min_tissue_ratio

    def _normalize_tile(self, img):
        """Apply stain normalization if the tile has sufficient tissue."""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self._is_tissue_present(img):  # Only normalize if there's enough tissue
            try:
                img = self._stain_normalizer.transform(img)
            except Exception as e:
                print(f"Normalization failed: {e}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _process_sample(self, idx):
        """Process a single sample: extract a patch, normalize, and apply transformations."""
        idx = idx.numpy()  # Ensure idx is a NumPy integer
        coord = tuple(self.coords[idx])  # Convert to tuple

        img = self._get_patch(coord)  # Extract patch
        img = self._normalize_tile(img)  # Apply stain normalization

        # Convert NumPy array to PIL image for further transformations
        img = Image.fromarray(img)

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
                Tout=(tf.float32, tf.int32)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return dataset

# class WholeSlideBagTFNorm:
#     def __init__(self, file_path, wsi, img_transforms=None):
#         """
#         Args:
#             file_path (string): Path to the .h5 file containing patched data.
#             wsi (object): Whole Slide Image object (e.g., from OpenSlide).
#             img_transforms (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.file_path = file_path
#         self.wsi = wsi
#         self.img_transforms = img_transforms

#         # Read metadata from the .h5 file
#         with h5py.File(self.file_path, "r") as f:
#             self.coords = f["coords"][:]  # Load all coordinates into memory
#             self.patch_level = f["coords"].attrs["patch_level"]
#             self.patch_size = f["coords"].attrs["patch_size"]
        
#         self.length = len(self.coords)  # Number of patches

#         # Initialize stain normalizer (Vahadane)
#         self._stain_normalizer = staintools.StainNormalizer(method='vahadane')
#         target_img = cv2.imread("/home/digitalpathology/workspace/path_foundation_stain_variation/assets/485819_colour_sample.tif")
#         target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
#         self._stain_normalizer.fit(target_img)

#         self.summary()

#     def summary(self):
#         """Print summary information about the dataset."""
#         with h5py.File(self.file_path, "r") as f:
#             dset = f["coords"]
#             for name, value in dset.attrs.items():
#                 print(f"{name}: {value}")
#         print("\nFeature extraction settings:")
#         print(f"Transformations: {self.img_transforms}")

#     def _get_patch(self, coord):
#         """Extract a patch from the WSI using the given coordinate."""
#         x, y = coord
#         img = self.wsi.read_region((x, y), self.patch_level, (self.patch_size, self.patch_size)).convert("RGB")
#         return np.array(img)  # Convert PIL Image to NumPy

#     def _is_tissue_present(self, img, threshold=200, min_tissue_ratio=0.3):
#         """
#         Check if the tile contains tissue (not mostly background).
#         - `threshold`: Pixel intensity above which a pixel is considered background.
#         - `min_tissue_ratio`: Minimum proportion of pixels that should be tissue.
#         """
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         tissue_pixels = np.sum(gray < threshold)
#         total_pixels = img.shape[0] * img.shape[1]
#         return (tissue_pixels / total_pixels) > min_tissue_ratio

#     def _normalize_tile(self, img):
#         """Apply stain normalization if the tile has sufficient tissue."""
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         if self._is_tissue_present(img):  # Only normalize if there's enough tissue
#             try:
#                 img = self._stain_normalizer.transform(img)
#             except Exception as e:
#                 print(f"Normalization failed: {e}")
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     def _data_generator(self):
#         """
#         A Python generator that yields preprocessed (normalized) patches and their coordinates.
#         This function performs the image extraction, normalization, and any transformation in Python.
#         """
#         for idx in range(self.length):
#             coord = tuple(self.coords[idx])
#             img = self._get_patch(coord)       # Extract patch
#             img = self._normalize_tile(img)      # Apply stain normalization

#             # Apply any additional image transformations (if provided)
#             pil_img = Image.fromarray(img)
#             if self.img_transforms:
#                 pil_img = self.img_transforms(pil_img)
#             # Convert back to NumPy array with float32 dtype
#             img = np.array(pil_img, dtype=np.float32)

#             # Yield the processed image and its coordinate as NumPy arrays.
#             yield img, np.array(coord, dtype=np.int32)

#     def get_tf_dataset(self):
#         if self.img_transforms:
#             dummy_img = Image.new("RGB", (self.patch_size, self.patch_size))
#             dummy_img = self.img_transforms(dummy_img)
#             dummy_img = np.array(dummy_img, dtype=np.float32)
#             out_img_shape = dummy_img.shape
#         else:
#             out_img_shape = (self.patch_size, self.patch_size, 3)
    
#         output_types = (tf.float32, tf.int32)
#         output_shapes = (tf.TensorShape(out_img_shape), tf.TensorShape([2]))
    
#         dataset = tf.data.Dataset.from_generator(
#             self._data_generator,
#             output_types=output_types,
#             output_shapes=output_shapes
#         )
#         # Force the dataset to have a known cardinality.
#         dataset = dataset.apply(tf.data.experimental.assert_cardinality(self.length))
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)
#         return dataset