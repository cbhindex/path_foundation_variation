#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:40:00 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script generally compute the slide-level embeddings using the trained model
and then visualise the slide-level embeddings with t-SNE.

"""

# Package Import
import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from utils.helper_class_pytorch import SlideBagDataset, AttentionMIL_EmbeddingExtractor
from utils.helper_functions_pytorch import load_data, collate_fn_variable_size

# Function to extract embeddings
def extract_slide_embeddings(model, test_loader, slide_filenames, device):
    model.eval()  # Set to evaluation mode
    slide_embeddings = []
    slide_ids = []
    
    with tqdm(total=len(slide_filenames), desc="Extracting Embeddings", unit="slide") as pbar:
        with torch.no_grad():
            for batch_idx, (batch_patches, _) in enumerate(test_loader):
                
                for i, patches in enumerate(batch_patches):
                    patches = patches.to(device)
                    slide_id = slide_filenames[batch_idx * test_loader.batch_size + i]  # Get actual case ID
                    
                    # Extract slide-level embedding
                    slide_embedding = model(patches)
                    
                    slide_embeddings.append(slide_embedding.cpu().numpy())
                    slide_ids.append(slide_id)
                    pbar.update(1)
    
    return np.array(slide_embeddings), slide_ids

# Function to perform both interactive and static t-SNE visualisation
def plot_tsne(embeddings, slide_ids, labels, output_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2D = tsne.fit_transform(embeddings)
    
    # Diagnosis mapping
    diagnosis_mapping = {
        1: "MPNST",
        2: "Dermatofibrosarcoma protuberans",
        3: "Neurofibroma",
        4: "Nodular fasciitis",
        5: "Desmoid Fibromatosis",
        6: "Synovial sarcoma",
        7: "Lymphoma",
        8: "Glomus tumour",
        9: "Intramuscular myxoma",
        10: "Ewing",
        11: "Schwannoma",
        12: "Myxoid liposarcoma",
        13: "Leiomyosarcoma",
        14: "Solitary fibrous tumour",
        15: "Low grade fibromyxoid sarcoma"
    }
    
    # Create DataFrame for visualisation
    df_tsne = pd.DataFrame({
        'tSNE1': embeddings_2D[:, 0],
        'tSNE2': embeddings_2D[:, 1],
        'slide_id': slide_ids,
        'Diagnosis': [diagnosis_mapping[label] for label in labels]  # Map numerical labels to diagnosis names
    })
    
    # Define color palette using plotly's categorical colors
    unique_labels = df_tsne['Diagnosis'].unique()
    color_map = px.colors.qualitative.Plotly
    label_to_color = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}
    
    # Save interactive plot
    fig = px.scatter(
        df_tsne, 
        x='tSNE1', 
        y='tSNE2', 
        color=df_tsne['Diagnosis'],  # Color by Diagnosis names
        hover_data=['slide_id'],  # Show case ID on hover
        title="Interactive t-SNE Visualisation of Slide-Level Embeddings (Grouped by Ground Truth)",
        labels={'color': 'Diagnosis'},  # Change legend title to 'Diagnosis'
        width=1200, height=800,
        color_discrete_map=label_to_color  # Apply the same color scheme
    )
    output_html = os.path.join(output_path, "tsne_visualisation_interactive.html")
    fig.write_html(output_html)
    print(f"Interactive t-SNE visualisation saved at: {output_html}")
    
    # Save static plot (using the same color scheme as interactive plot)
    plt.figure(figsize=(12, 8))
    colors = [label_to_color[label] for label in df_tsne['Diagnosis']]
    scatter = plt.scatter(
        df_tsne['tSNE1'], df_tsne['tSNE2'], 
        c=colors, s=30, alpha=0.9
    )
    
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.title("t-SNE Visualisation of Slide-Level Embeddings (Grouped by Ground Truth)")
    
    # Create legend outside the main figure (upper right-hand side)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label], markersize=8, label=label)
               for label in unique_labels]
    legend = plt.legend(handles=handles, title="Diagnosis", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.tight_layout()
    output_png = os.path.join(output_path, "tsne_visualisation.png")
    plt.savefig(output_png, bbox_extra_artists=(legend,), bbox_inches='tight')
    print(f"Static t-SNE plot saved at: {output_png}")

# Main function
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_4",
                        help='Path to validation data folder')
    parser.add_argument('--test_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_4.csv", 
                        help='Path to validation label CSV')
    parser.add_argument('--model', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/models/mil_best_model_state_dict_epoch_37.pth", 
                        help='Path to saved model')
    parser.add_argument('--output', type=str, 
                        default='/home/digitalpathology/workspace/path_foundation_stain_variation/visualisation/cohort_4', 
                        help='Path to output folder')

    args = parser.parse_args()
    
    since = time.time()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load test data
    test_patch_features, test_labels, test_slide_filenames = load_data(args.test_folder, args.test_labels)
    test_dataset = SlideBagDataset(test_patch_features, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        collate_fn=collate_fn_variable_size
    )
    
    # debug print
    print("Data loading completed.")
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL_EmbeddingExtractor(input_dim=384, attention_dim=128, num_classes=15).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    # Extract slide-level embeddings
    embeddings, slide_ids = extract_slide_embeddings(model, test_loader, test_slide_filenames, device)
    
    # Load labels
    labels_df = pd.read_csv(args.test_labels)
    label_dict = dict(zip(labels_df['case_id'], labels_df['ground_truth']))
    
    # Create DataFrame with embeddings
    column_names = ['emb_' + str(i+1) for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings, index=slide_ids, columns=column_names)
    
    # Extend df_embeddings with labels by checking label_dict and placing it after case_id
    df_embeddings.insert(0, 'ground_truth', df_embeddings.index.map(lambda slide_id: label_dict.get(slide_id, "Unknown")))
    
    # Save embeddings to CSV
    df_embeddings.to_csv(os.path.join(args.output, "slide_embeddings_with_labels.csv"))
    
    # # Generate t-SNE visualisation - both the interactive and still image version
    plot_tsne(embeddings, slide_ids, df_embeddings['ground_truth'].values, args.output)
    
    # Read the existing HTML file
    with open(os.path.join(args.output, "tsne_visualisation_interactive.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    
    ########## below JavaScript code is provided by Jianan Chen for improve user experience ##########
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
