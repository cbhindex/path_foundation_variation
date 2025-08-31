#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:33:22 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script generates a radar plot showing performance of each subtype for each 
model using a csv file as an input. 

The headers of the output csv files are "subtype", "path_foundation", "uni_v2", 
"conch_v15", "titan", "resnet50". The csv file have multiple columns where 
first column is the subtypes (writing in text), and the rest 5 columns are 
accuracies for each subtype for each models. Each row of this csv file stands
for a subtype.

In this radar plot, the colours are (sequentially): 4169E1, FF6347, FFD700, 
708090, 87CEEB, which aligns with other plot in figures.

Parameters
----------
input: str
    Path to input csv file.
    
output: str
    Path to output folder.

"""

import os
import argparse
import time

import pandas as pd
import plotly.graph_objects as go

# Custom colors with 0.2 alpha transparency
custom_colors = ["#4169E1", "#FF6347", "#FFD700", "#708090", "#87CEEB"]

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

# Main function
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input csv file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder')
    
    args = parser.parse_args()
    
    since = time.time()
    
    # Make sure output folder exists
    os.makedirs(f"{args.output}", exist_ok=True)

    # CSV loading
    df = pd.read_csv(f"{args.input}")
    
    # Data extraction
    subtypes = df['subtype'].tolist()
    models = df.columns[1:]
    # Add first value at the end to "close the radar loop"
    theta = subtypes + [subtypes[0]] 
    
    # Build the radar chart
    fig = go.Figure()
    
    for i, model in enumerate(models):
        r_values = df[model].tolist() + [df[model].tolist()[0]]
        rgba_fill = hex_to_rgba(custom_colors[i], 0.2)
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta,
            name=model,
            fill='toself',
            line=dict(color=custom_colors[i]),
            fillcolor=rgba_fill
        ))
    
    # Layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        # title="Subtype-level Accuracy Across Models"
    )
    
    # Save interactive HTML and static figure
    # fig.write_html(f"{args.output}/interactive_radar_plot.h1tml")
    fig.write_html(f"{args.output}/interactive_radar_plot.html", include_mathjax=False)
    fig.write_image(f"{args.output}/interactive_radar_plot.pdf", width=800, height=600)
    
    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))