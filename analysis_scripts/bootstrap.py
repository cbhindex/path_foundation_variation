#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 22:09:50 2025

@author: digitalpathology
"""

import pandas as pd
import numpy as np
import argparse
import os

def compute_ci(values, alpha=0.05):
    lower = np.percentile(values, 100 * (alpha / 2))
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Bootstrap Top-1 and Top-3 accuracy with confidence intervals.")
    
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="/home/digitalpathology/workspace/path_foundation_stain_variation/output/scan_exp/3dhistech/resnet50/8_slides/cohort_1/cohort_1_individual_results.csv", 
        help="Path to the CSV result file."
        )
    parser.add_argument(
        "--n_bootstrap", 
        type=int, 
        default=1000, 
        help="Number of bootstrap resamples."
        )
    
    args = parser.parse_args()
    csv_path = args.csv_path
    B = args.n_bootstrap

    # Load CSV
    df = pd.read_csv(csv_path)
    n = len(df)

    # Compute overall accuracies
    overall_top1_acc = df['top_1_correct'].sum() / n
    overall_top3_acc = df['top_3_correct'].sum() / n

    # Bootstrap resampling
    top1_accs = []
    top3_accs = []

    for _ in range(B):
        sample = df.sample(n=n, replace=True)
        top1_accs.append(sample['top_1_correct'].mean())
        top3_accs.append(sample['top_3_correct'].mean())

    # Confidence intervals
    top1_lower, top1_upper = compute_ci(top1_accs)
    top3_lower, top3_upper = compute_ci(top3_accs)

    # Print to console
    print(f"Top-1 Accuracy: {overall_top1_acc:.4f} (95% CI: {top1_lower:.4f} – {top1_upper:.4f})")
    print(f"Top-3 Accuracy: {overall_top3_acc:.4f} (95% CI: {top3_lower:.4f} – {top3_upper:.4f})")

    # Prepare output DataFrame
    result_df = pd.DataFrame({
        "metric": ["top1_accuracy", "top3_accuracy"],
        "accuracy": [overall_top1_acc, overall_top3_acc],
        "ci_lower": [top1_lower, top3_lower],
        "ci_upper": [top1_upper, top3_upper],
        "n_bootstrap": [B, B]
    })

    # Determine output path (same folder as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "bootstrap_accuracy_results.csv")

    # Save CSV
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")