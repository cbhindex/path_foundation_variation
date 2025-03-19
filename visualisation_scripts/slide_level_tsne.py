#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:40:00 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script computes the slide-level embeddings using the trained model
and then visualises the slide-level embeddings with t-SNE.

Now it also adds prediction and confidence score information to the interactive t-SNE plot.

Parameters
----------
test_folder: str
    Path to validation data folder.
    
test_labels: str
    Path to validation label CSV.

prediction_csv: str
    Path to prediction results CSV.
    
output: str
    Path to output folder.

"""

# Package Import
import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import random

import plotly.express as px
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from utils.helper_class_pytorch import SlideBagDataset
from utils.helper_functions_pytorch import load_data, collate_fn_variable_size

# Diagnosis mapping
# DIAGNOSIS_MAPPING_15CLASSES = {
#     1: "MPNST",
#     2: "Dermatofibrosarcoma protuberans",
#     3: "Neurofibroma",
#     4: "Nodular fasciitis",
#     5: "Desmoid Fibromatosis",
#     6: "Synovial sarcoma",
#     7: "Lymphoma",
#     8: "Glomus tumour",
#     9: "Intramuscular myxoma",
#     10: "Ewing",
#     11: "Schwannoma",
#     12: "Myxoid liposarcoma",
#     13: "Leiomyosarcoma",
#     14: "Solitary fibrous tumour",
#     15: "Low grade fibromyxoid sarcoma"
# }
DIAGNOSIS_MAPPING = {
    1: "Dermatofibrosarcoma protuberans",
    2: "Neurofibroma",
    3: "Nodular fasciitis",
    4: "Desmoid Fibromatosis",
    5: "Synovial sarcoma",
    6: "Lymphoma",
    7: "Glomus tumour",
    8: "Intramuscular myxoma",
    9: "Ewing",
    10: "Schwannoma",
    11: "Myxoid liposarcoma",
    12: "Leiomyosarcoma",
    13: "Solitary fibrous tumour",
    14: "Low grade fibromyxoid sarcoma"
}

# TODO: maybe put this function to utils?
def extract_random_slide_embeddings(test_loader, test_slide_filenames, nr_patch=3):
    """
    Randomly selects `nr_patch` patches for each slide instead of using a model-based approach.

    Parameters:
    - test_loader (DataLoader): DataLoader containing patch-level embeddings.
    - test_slide_filenames (list): List of slide filenames corresponding to test_loader data.
    - nr_patch (int): Number of patches to randomly select per slide. Default is 5.

    Returns:
    - slide_embeddings (np.array): Randomly selected patch embeddings.
    - slide_ids (list): Corresponding slide IDs (not unique, as each slide has `nr_patch` entries).
    """
    slide_embeddings = []
    slide_ids = []

    with tqdm(total=len(test_slide_filenames), desc="Selecting Random Patches", unit="slide") as pbar:
        for batch_idx, (batch_patches, _) in enumerate(test_loader):
            for i, patches in enumerate(batch_patches):
                slide_id = test_slide_filenames[batch_idx * test_loader.batch_size + i]
                
                # Convert to NumPy array (if it's a tensor)
                patches_np = patches.cpu().numpy() if isinstance(patches, torch.Tensor) else patches
                
                # Ensure there are at least `nr_patch` samples to choose from
                num_available_patches = patches_np.shape[0]
                if num_available_patches < nr_patch:
                    selected_patches = patches_np  # Take all available patches if less than `nr_patch`
                else:
                    selected_patches = patches_np[random.sample(range(num_available_patches), nr_patch)]
                
                for patch in selected_patches:
                    slide_embeddings.append(patch)
                    slide_ids.append(slide_id)
                
                pbar.update(1)

    return np.array(slide_embeddings), slide_ids

# Function to perform t-SNE visualization
def plot_tsne(embeddings, slide_ids, labels, predictions, confidences, output_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2D = tsne.fit_transform(embeddings)

    # Create DataFrame for visualization
    df_tsne = pd.DataFrame({
        'tSNE1': embeddings_2D[:, 0],
        'tSNE2': embeddings_2D[:, 1],
        'slide_id': slide_ids,
        'Diagnosis': [DIAGNOSIS_MAPPING[label] for label in labels],
        'Prediction': [DIAGNOSIS_MAPPING[pred] for pred in predictions],
        'Score': [f"{conf:.6f}" for conf in confidences]  # Ensure 6 decimal places
    })

    # Define color palette
    unique_labels = df_tsne['Diagnosis'].unique()
    color_map = plt.get_cmap("tab20").colors  # Get Tab20 colormap
    color_map = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in color_map]  # Convert to Plotly format
    #color_map = px.colors.qualitative.Plotly
    label_to_color = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}

    # Save interactive plot
    fig = px.scatter(
        df_tsne,
        x='tSNE1',
        y='tSNE2',
        color=df_tsne['Diagnosis'],
        hover_data=['slide_id', 'Prediction', 'Score'],
        title="Interactive t-SNE Visualization of Slide-Level Embeddings",
        labels={'color': 'Diagnosis'},
        width=1200, height=800,
        color_discrete_map=label_to_color
    )
    output_html = os.path.join(output_path, "tsne_visualisation_interactive.html")
    fig.write_html(output_html)
    print(f"Interactive t-SNE visualisation saved at: {output_html}")

    # Save static plot
    plt.figure(figsize=(12, 8))
    
    #colors = [label_to_color[label] for label in df_tsne['Diagnosis']]
    tab20_cmap = plt.get_cmap("tab20")
    unique_labels_list = list(unique_labels)
    label_to_color_static = {label: tab20_cmap(i / max(1, len(unique_labels_list)-1)) for i, label in enumerate(unique_labels_list)}
    colors = [label_to_color_static[label] for label in df_tsne['Diagnosis']]
    
    plt.scatter(df_tsne['tSNE1'], df_tsne['tSNE2'], c=colors, s=30, alpha=0.9)

    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.title("t-SNE Visualisation of Slide-Level Embeddings")

    plt.tight_layout()
    output_png = os.path.join(output_path, "tsne_visualisation.png")
    plt.savefig(output_png)
    print(f"Static t-SNE plot saved at: {output_png}")

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/new_cohort_2",
                        help='Path to validation data folder')
    parser.add_argument('--test_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/new_cohort_2.csv", 
                        help='Path to validation label CSV')
    parser.add_argument('--prediction_csv', type=str, 
                        default='/home/digitalpathology/workspace/path_foundation_stain_variation/output/trained_without_variation/new_cohort_2/new_cohort_2_individual_results.csv', 
                        help='Path to prediction results CSV')
    parser.add_argument('--output', type=str, 
                        default='/home/digitalpathology/workspace/path_foundation_stain_variation/visualisation/cohort_2', 
                        help='Path to output folder')

    args = parser.parse_args()
    
    since = time.time()
    os.makedirs(args.output, exist_ok=True)

    # Load test data
    test_patch_features, test_labels, test_slide_filenames = load_data(args.test_folder, args.test_labels)
    test_dataset = SlideBagDataset(test_patch_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_variable_size)

    print("Data loading completed.")

    # Extract slide-level embeddings
    embeddings, slide_ids = extract_random_slide_embeddings(test_loader, test_slide_filenames, nr_patch=3)

    # Load labels
    labels_df = pd.read_csv(args.test_labels)
    label_dict = dict(zip(labels_df['case_id'], labels_df['ground_truth']))

    # Load predictions
    predictions_df = pd.read_csv(args.prediction_csv)
    predictions_dict = dict(zip(predictions_df['slide_id'], predictions_df['top_1_pred']))
    confidence_dict = dict(zip(predictions_df['slide_id'], predictions_df['top_1_prob']))

    # Assign ground truth labels, predictions, and confidence scores
    ground_truths = [label_dict.get(slide_id, "Unknown") for slide_id in slide_ids]
    predictions = [predictions_dict.get(slide_id, "Unknown") for slide_id in slide_ids]
    confidences = [confidence_dict.get(slide_id, 0.0) for slide_id in slide_ids]

    # Save embeddings to CSV
    df_embeddings = pd.DataFrame(embeddings, index=slide_ids)
    df_embeddings.insert(0, 'ground_truth', ground_truths)
    df_embeddings.insert(1, 'prediction', predictions)
    df_embeddings.insert(2, 'confidence_score', confidences)
    df_embeddings.to_csv(os.path.join(args.output, "slide_embeddings_with_labels.csv"))

    # Generate t-SNE visualization
    plot_tsne(embeddings, slide_ids, ground_truths, predictions, confidences, args.output)
    
    ########## below JavaScript code is provided by Jianan Chen for improve user experience ##########
    
    # Read the existing HTML file
    with open(os.path.join(args.output, "tsne_visualisation_interactive.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # JavaScript code
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
    
        plot.on('plotly_click', function(data) {
            if (data.points.length > 0) {
                var filename = data.points[0].customdata[0]; 
                navigator.clipboard.writeText(filename).then(function() {
                    alert("Copied to clipboard: " + filename);
                }).catch(function(err) {
                    console.error("Failed to copy: ", err);
                });
            }
        });
    });
    </script>
    """
    
    # Insert the JavaScript code before </body>
    html_content = html_content.replace("</body>", js_code + "\n</body>")
    
    # Save the modified HTML
    with open(os.path.join(args.output, "tsne_visualisation_interactive.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("Embedding extraction and visualisation completed.")

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))