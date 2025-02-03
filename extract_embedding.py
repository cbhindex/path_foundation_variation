#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:19:56 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

The function of this script is to extract embeddings for a folder of whole slide
images using Google Path Foundation model under Tensorflow (GPU) framework.

There are two inputs to put together:
    (1) the raw whole slide images in .svs or .ndpi, 
    (2) their tiling coordinates information in .h5, this is generated under 
    CLAM preprocessing pipeline.
All the raw WSIs should be stored in a single folder, and their tiling information
should be stored in another folder, where the file name should match.

The output of this script should be a series of CSV files (one file per slide) 
storing the embeddings.

Parameters
----------
wsi_path: str
    The folder path for storing oroginal whole slide images. The slides should 
    be stored in .svs or .ndpi format.

tile_path: str
    The folder path for storing the tiling information. The format of tiling 
    should be .h5, generated with CLAM pipeline.

model_path: str
    The path for Path Foundation model.

output_path: str
    The output folder path for generated embeddings.

"""

# package import
import time
import argparse
from tqdm import tqdm
import os

import numpy as np
import pandas as pd
from PIL import Image
import h5py

from openslide import OpenSlide

import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

from utils.helper_class import WholeSlideBagTF
from utils.helper_functions import resize_transforms

# define argument parser
parser = argparse.ArgumentParser(description='extract embeddings')
parser.add_argument('--wsi_path', type=str,
                    default='/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/source_wsi/01_external',
					help='path to folder for oroginal slides')
parser.add_argument('--tile_path', type=str, 
                    default='/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/tiles/01_external/patches',
					help='path to folder for tiling information in .h5')
parser.add_argument('--model_path', type=str, 
                    default='/home/digitalpathology/.cache/huggingface/hub/models--google--path-foundation/snapshots/fd6a835ceaae15be80db6abd8dcfeb86a9287e72',
					help='path to Path Foundation model')
parser.add_argument('--output_path', type=str,
                    default='/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/embeddings/01_external',
                    help='path to folder for outputs')

#################### define slide processing function ####################

def process_slide(tile_path, wsi_path, model, batch_size=32):
    """Process a single WSI and extract embeddings."""
    
    wsi = OpenSlide(wsi_path)
    
    # load tiles
    dataset = WholeSlideBagTF(file_path=tile_path, wsi=wsi, img_transforms=resize_transforms)
    tf_dataset = dataset.get_tf_dataset().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # # Iterate over the dataset - debug print
    # # Take one batch from the tf_dataset
    # for img_batch, coord_batch in tf_dataset.take(1):  # Take one batch
    #     # Print the shape and data type of the batch
    #     print(f"Image batch shape: {img_batch.shape}")  # Should be (batch_size, 224, 224, 3)
    #     print(f"Image batch dtype: {img_batch.dtype}")  # Should be float32
    #     print(f"Coordinate batch shape: {coord_batch.shape}")  # Should be (batch_size, 2)
    #     print(f"Coordinate batch dtype: {coord_batch.dtype}")  # Should be int32
    
    #     # Check the pixel value range
    #     print(f"Image pixel value range: {tf.reduce_min(img_batch).numpy()} to {tf.reduce_max(img_batch).numpy()}")
    
    #     # Optionally inspect a specific image and its coordinates
    #     print("\nFirst image and its coordinates:")
    #     print(img_batch[0].numpy())  # Convert to NumPy for inspection
    #     print(coord_batch[0].numpy())  # Convert to NumPy for inspection
    
    # Initialize an empty list to collect data for the DataFrame    
    outputs = []
    
    for img_batch, coord_batch in tqdm(tf_dataset, desc="Processing Batches"):
        result = model(img_batch)
        batch_output = result['output_0'].numpy()
        coords_np = coord_batch.numpy()
        
        for i in range(batch_output.shape[0]):
            outputs.append([coords_np[i][0], coords_np[i][1]] + batch_output[i].tolist())
            
    nr_features = batch_output.shape[1]
    
    return outputs, nr_features

#################### main part for inference ####################
if __name__ == "__main__":
    args = parser.parse_args()

    # Print GPU availability
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    
    since = time.time()
    
    # Path to the pre-trained model (TFSMLayer)
    model_path = args.model_path
    path_foundation = TFSMLayer(model_path, call_endpoint="serving_default")
    
    
    # List all WSI and tile files
    wsi_files = sorted([f for f in os.listdir(args.wsi_path) if f.endswith((".svs", ".ndpi", "tif"))])
    tile_files = sorted([f for f in os.listdir(args.tile_path) if f.endswith(".h5")])
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    for wsi_file, tile_file in tqdm(zip(wsi_files, tile_files), desc="Processing Slides", total=len(wsi_files)):
        # Ensure the filenames match (excluding extensions)
        if os.path.splitext(wsi_file)[0] != os.path.splitext(tile_file)[0]:
            print(f"Skipping {wsi_file} and {tile_file}: filenames do not match.")
            continue
    
        # Construct full paths
        wsi_path = os.path.join(args.wsi_path, wsi_file)
        tile_path = os.path.join(args.tile_path, tile_file)
    
        print(f"Processing slide: {wsi_file}")
    
        # Process the slide
        outputs, nr_features = process_slide(
            tile_path=tile_path,
            wsi_path=wsi_path,
            model=path_foundation,
            batch_size=32,
        )
    
        # Create a DataFrame with the required columns
        columns = ['x_coord', 'y_coord'] + [f'emb_{i+1}' for i in range(nr_features)]
        df = pd.DataFrame(outputs, columns=columns)
    
        # Save the DataFrame to a CSV file
        slide_name = os.path.splitext(wsi_file)[0]
        output_file = os.path.join(args.output_path, f"{slide_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")
    
    print("All slides processed!")
        
    # outputs, nr_features = process_slide(
    #     tile_path="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/tiles/01_external/patches/RNOH_1_3_S00140422_131427.h5",
    #     wsi_path="/home/digitalpathology/workspace/path_foundation_stain_variation_refactoring/source_wsi/01_external/RNOH_1_3_S00140422_131427.svs",
    #     model=path_foundation,
    #     batch_size=32,
    #     )
    
    # # Create a DataFrame with the required columns
    # columns = ['x_coord', 'y_coord'] + [f'emb_{i+1}' for i in range(nr_features)]
    # df = pd.DataFrame(outputs, columns=columns)
    
    # # Save the DataFrame to a single CSV file
    # df.to_csv("output_features_with_coords.csv", index=False)
    
    # print("Feature extraction and saving completed!")
    
    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    
    