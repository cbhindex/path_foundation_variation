#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:05:35 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Helper functions for tensorflow env for embedding generation.

"""

# package import
import tensorflow as tf
# import cv2
# import staintools
import numpy as np

# Define a custom transformation pipeline with TensorFlow
def resize_transforms(img):
    """
    this function resize an input to [224, 224]

    """
    # Resize to 224x224
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    # Normalize pixel values to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    return img

# def load_ref_image():
#     # Precompute and set up a global stain normalizer (do this once)
#     target_path = "/home/digitalpathology/workspace/path_foundation_stain_variation/assets/485819_colour_sample.tif"
#     target_img = cv2.imread(target_path)
#     target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
#     global_normalizer = staintools.StainNormalizer(method='vahadane')
#     global_normalizer.fit(target_img)
    
#     return target_img, global_normalizer

# def _norm_and_resize(image_input):
    
#     target_img, global_normalizer = load_ref_image()
    
#     # Ensure image_input is a NumPy array.
#     image_np = np.array(image_input)
#     image_np = image_np.astype(np.uint8)
#     # Convert from RGB to BGR for stain normalization.
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     # Apply stain normalization.
#     normalized_bgr = global_normalizer.transform(image_bgr)
#     # Convert back to RGB.
#     normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
#     # Resize to 224x224.
#     resized_img = cv2.resize(normalized_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
#     # Normalize pixel values to [0, 1].
#     resized_img = resized_img.astype(np.float32) / 255.0
#     return resized_img

# def norm_and_resize(img):
#     output = tf.py_function(func=_norm_and_resize, inp=[img], Tout=tf.float32)
#     output.set_shape([224, 224, 3])
#     return output
