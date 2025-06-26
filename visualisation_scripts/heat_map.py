#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:48:41 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script generates a static, slide-level heatmap in pdf format, visualising 
the tile-level attention scores learned by a MLP model. It takes as input a 
tile embedding `.h5` file (containing patch coordinates and feature embeddings) 
and a trained MIL model, applies the model to compute attention weights for 
each tile, and renders the attention map using matplotlib.

The attention scores are normalised and arranged on a spatial grid according to 
tile coordinates. The final heatmap highlights tiles that contributed most to 
the slide-level prediction.

Parameters
----------
embedding_path: str
    Path to .h5 file with embeddings and coords.
    
model_path: str
    Path to trained MLP model.
    
output_dir: str
    Output directory to save heatmap files.
                        )

device: str, 
    Device to use (default: cuda).

"""


import argparse
import h5py
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from utils.helper_class_pytorch import AttentionMIL

def load_model(model_path, input_dim, num_classes, device):
    model = AttentionMIL(input_dim=input_dim, attention_dim=128, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def normalize_scores(scores):
    min_s, max_s = np.min(scores), np.max(scores)
    return (scores - min_s) / (max_s - min_s + 1e-6)

def generate_heatmap(coords, scores, slide_id, output_dir, tile_size=10):
    norm_scores = normalize_scores(scores)

    # Create a dataframe of tile positions and scores
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    df = pd.DataFrame({
        'x_idx': ((coords[:, 0] - x_min) // tile_size).astype(int),
        'y_idx': ((coords[:, 1] - y_min) // tile_size).astype(int),
        'score': norm_scores
    })

    # Pivot to form 2D grid
    grid = df.pivot(index='y_idx', columns='x_idx', values='score')

    if grid.isna().all().all():
        print(f"[ERROR] Slide {slide_id}: All values are NaN in heatmap. Nothing to render.")
        return

    # Plot with imshow
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid.values, cmap=cmap, origin='upper', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'Tile-level attention heatmap: {slide_id}')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pdf_path = os.path.join(output_dir, f"{slide_id}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate heatmap from tile embeddings and MLP model')
    parser.add_argument('--embedding_path', 
                        type=str, 
                        default='/media/digitalpathology/b_chai/trident_outputs/cohort_4/20x_224px_0px_overlap/features_uni_v2/RNOH_4_21_S00128702_171212.h5', 
                        help='Path to .h5 file with embeddings and coords'
                        )
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/home/digitalpathology/workspace/path_foundation_stain_variation/models/uni_v2/trained_on_cohort_1_train/mil_best_model_state_dict_epoch_8.pth', 
                        help='Path to trained MLP model'
                        )
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='/home/digitalpathology/temp', 
                        help='Output directory to save heatmap files'
                        )
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda', 
                        help='Device to use (default: cuda)'
                        )
    
    args = parser.parse_args() 
    
    since = time.time()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    slide_id = os.path.splitext(os.path.basename(args.embedding_path))[0]

    with h5py.File(args.embedding_path, "r") as f:
        coords = f['coords'][()]
        features = f['features'][()]

    input_dim = features.shape[1]
    model = load_model(args.model_path, input_dim, num_classes=14, device=device)

    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        logits, attention_weights = model(inputs)
        probs = torch.softmax(logits, dim=0)
        scores = attention_weights.squeeze(1).cpu().numpy()

    generate_heatmap(coords, scores, slide_id, args.output_dir)

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
