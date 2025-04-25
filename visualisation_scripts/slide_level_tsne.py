#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:50:18 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script computes the slide-level embeddings using the trained model
and then visualises the slide-level embeddings with t-SNE. The interactive t-sne
shows also the prediction and confidence score information. The t-sne plot has
three colouring strategy, they are: dots coloured by scan_device, by staining_institute,
and by their ground truth diagnosis.

Parameters
----------
input_csv: str
    Path to input CSV file with embeddings and labels.
    
output: str
    Path to output folder.

"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# def generate_color_palette(unique_values):
#     """Generate a distinct color palette for unique values, optimized for visual clarity."""
#     num_colors = len(unique_values)
    
#     # Prioritize strong contrast and visually distinct colors first
#     all_colors = (
#         px.colors.qualitative.Bold +          # High contrast, good for few categories
#         px.colors.qualitative.Dark2 +         # Still strong and good for up to ~12
#         px.colors.qualitative.Set3 +          # More colorful, medium contrast
#         px.colors.qualitative.Safe +          # Designed for accessibility
#         px.colors.qualitative.Pastel +        # Soft backup colors
#         px.colors.qualitative.Light24         # Large pool, lower contrast
#     )

#     # Deduplicate while preserving order
#     seen = set()
#     deduped_colors = []
#     for color in all_colors:
#         if color not in seen:
#             deduped_colors.append(color)
#             seen.add(color)

#     # Repeat colors only if necessary, but warn if uniqueness can't be preserved
#     if num_colors > len(deduped_colors):
#         print(f"Warning: Not enough unique colors for {num_colors} values, some colors will repeat.")

#     colors = deduped_colors * ((num_colors // len(deduped_colors)) + 1)
#     color_map = {str(val): colors[i] for i, val in enumerate(unique_values)}
    
#     return color_map

def generate_color_palette(unique_values):
    """Generate a perceptually distinct, print-safe palette with good early contrast."""
    num_colors = len(unique_values)

    # Reordered with first N colors being maximally distinct (for scan device / diagnosis)
    curated_palette = [
        "#E41A1C",  # vivid red
        "#377EB8",  # strong blue
        "#FFD700",  # golden yellow

        "#4DAF4A",  # vibrant green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#999999",  # gray
        "#66C2A5",  # teal
        "#FC8D62",  # salmon
        "#8DA0CB",  # steel blue
        "#E78AC3",  # pink/magenta
        "#A6D854",  # lime green

        # Round 2 - Medium contrast
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
        "#E6AB02", "#A6761D", "#666666",

        # Round 3 - Tol and Okabe-Ito blends
        "#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677",
        "#882255", "#661100", "#999933", "#6699CC", "#44AA99",

        # Final backup colors
        "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
        "#CC79A7", "#B3B3B3", "#888888", "#E5C494", "#FFD92F"
    ]

    # Deduplicate and preserve order
    seen = set()
    deduped_colors = []
    for color in curated_palette:
        if color not in seen:
            deduped_colors.append(color)
            seen.add(color)

    if num_colors > len(deduped_colors):
        print(f"Warning: Only {len(deduped_colors)} colors. Some will repeat for {num_colors} categories.")
    
    colors = deduped_colors * ((num_colors // len(deduped_colors)) + 1)

    color_map = {str(val): colors[i] for i, val in enumerate(unique_values)}
    return color_map

def plot_tsne(embeddings, slide_ids, labels, predictions, confidences, staining_institutes, scan_devices, color_by, output_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2D = tsne.fit_transform(embeddings)

    df_tsne = pd.DataFrame({
        'tSNE1': embeddings_2D[:, 0],
        'tSNE2': embeddings_2D[:, 1],
        'slide_id': slide_ids,
        # 'Diagnosis': [str(label) for label in labels],
        'Diagnosis': [
            DIAGNOSIS_MAPPING[int(label)] if color_by == 'subtype' else str(label) 
            for label in labels
            ],
        'Prediction': predictions,
        'Score': [f"{conf:.6f}" for conf in confidences],
        'Staining Institute': staining_institutes,
        'Scan Device': scan_devices
    })

    if color_by == 'staining_institute':
        color_column = 'Staining Institute'
    elif color_by == 'scan_device':
        color_column = 'Scan Device'
    else:
        color_column = 'Diagnosis'  # Default to subtype classification

    if color_by == 'staining_institute':
        unique_values = sorted(df_tsne[color_column].unique(), key=lambda x: int(x[1:]))
    elif color_by == 'subtype':
        inverse_mapping = {v: k for k, v in DIAGNOSIS_MAPPING.items()}
        unique_values = sorted(df_tsne[color_column].unique(), key=lambda x: inverse_mapping[x])
    else:
        unique_values = sorted(df_tsne[color_column].unique())

    color_map = generate_color_palette(unique_values)

    fig = px.scatter(
        df_tsne,
        x='tSNE1',
        y='tSNE2',
        color=df_tsne[color_column],
        hover_data=['slide_id', 'Prediction', 'Score'],
        # title=f"Interactive t-SNE Visualization (Color by {color_column})",
        labels={'color': color_column},
        width=1200, height=800,
        category_orders={color_column: unique_values},  # Maintain consistent legend order
        color_discrete_map=color_map
    )
    
    # Generate div with CDN JS, not full HTML
    plot_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    # Add the custom JavaScript for click-to-copy
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
    # Wrap everything into a complete HTML page
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>t-SNE Visualization</title>
    </head>
    <body>
        {plot_div}
        {js_code}
    </body>
    </html>
    """
    output_html = os.path.join(output_path, f"tsne_visualisation_interactive_{color_by}.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Interactive t-SNE visualization saved at: {output_html}")
    
    # Save static PDF version
    pdf_output = os.path.join(output_path, f"tsne_visualisation_static_{color_by}.pdf")
    fig.write_image(pdf_output, format="pdf", width=1200, height=800)
    print(f"Static t-SNE PDF saved at: {pdf_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/visualisation_scripts/tsne_source/cohort_2_path_foundation.csv",
                        help='Path to input CSV file with embeddings and labels')
    parser.add_argument('--output', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/visualisation/tsne_plots/cohort_2",
                        help='Path to output folder')

    args = parser.parse_args()

    since = time.time()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    embeddings = df.iloc[:, 4:-2].values  # Assuming embeddings are all columns after the first four and before last two columns
    slide_ids = df.iloc[:, 0].tolist()
    labels = [str(label) for label in df['ground_truth'].tolist()]  # Ensure categorical labels
    predictions = df['prediction'].tolist()
    confidences = df['confidence_score'].tolist()
    staining_institutes = df['staining_institute'].tolist()
    scan_devices = df['scan_device'].tolist()

    plot_tsne(
        embeddings, slide_ids, labels, predictions, confidences, 
        staining_institutes, scan_devices, 'subtype', args.output)
    plot_tsne(
        embeddings, slide_ids, labels, predictions, confidences, 
        staining_institutes, scan_devices, 'scan_device', args.output)
    plot_tsne(
        embeddings, slide_ids, labels, predictions, confidences, 
        staining_institutes, scan_devices, 'staining_institute', args.output)
    
    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    
