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

# Define a custom transformation pipeline with TensorFlow
def resize_transforms(img):
    # Resize to 224x224
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    # Normalize pixel values to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    return img