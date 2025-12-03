#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:45:20 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script is for the slide-level inference using logistic regression on the
slide-level embeddings (for TITAN and PRISM only)
"""

# Import packages
import h5py
import os
import argparse

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from collections import defaultdict

# Define diagnosis and class_ID mapping
diagnosis_mapping = {
    0: "Dermatofibrosarcoma protuberans",
    1: "Neurofibroma",
    2: "Nodular fasciitis",
    3: "Desmoid Fibromatosis",
    4: "Synovial sarcoma",
    5: "Lymphoma",
    6: "Glomus tumour",
    7: "Intramuscular myxoma",
    8: "Ewing",
    9: "Schwannoma",
    10: "Myxoid liposarcoma",
    11: "Leiomyosarcoma",
    12: "Solitary fibrous tumour",
    13: "Low grade fibromyxoid sarcoma"
}

# Define helper functions
def train_and_evaluate(train_data, train_labels, test_data, test_labels):
    # Initialize and train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_data, train_labels)

    # Predictions
    test_preds = model.predict(test_data)

    # Compute accuracy
    micro_acc = accuracy_score(test_labels, test_preds)
    macro_acc = accuracy_score(test_labels, test_preds, normalize=True)

    print(f"Micro Accuracy: {micro_acc:.4f}")
    print(f"Macro Accuracy: {macro_acc:.4f}")

    return model

def load_embedding(slide_id, embedding_dir):
    feats_path = os.path.join(embedding_dir, f"{slide_id}.h5")
    if os.path.exists(feats_path):
        with h5py.File(feats_path, 'r') as h5_file:
            return np.array(h5_file['features'])
    else:
        print(f"Warning: Embedding file not found for slide_id {slide_id}")
        return None

def construct_datasets(metadata_file, embedding_dir):
    df = pd.read_csv(metadata_file)
    datasets = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    case_ids = {"train": [], "val": [], "test": []}

    for _, row in df.iterrows():
        label = row['ground_truth'] - 1
        split = row['split'].lower()
        case_id = row['case_id']

        embedding = load_embedding(case_id, embedding_dir)
        if embedding is not None:
            datasets[split].append(embedding)
            labels[split].append(label)
            case_ids[split].append(case_id)

    for split in datasets:
        datasets[split] = np.array(datasets[split])
        labels[split] = np.array(labels[split])

    return datasets, labels, case_ids

def train_and_evaluate(train_data, train_labels, test_data, test_labels, test_case_ids):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_data, train_labels)

    test_preds = model.predict(test_data)
    test_probs = model.predict_proba(test_data)

    metrics = {
        "Micro Accuracy": accuracy_score(test_labels, test_preds),
        "Macro Accuracy": accuracy_score(test_labels, test_preds),
        "Top-1 Accuracy": top_k_accuracy_score(test_labels, test_probs, k=1),
        "Top-3 Accuracy": top_k_accuracy_score(test_labels, test_probs, k=3),
        "Top-5 Accuracy": top_k_accuracy_score(test_labels, test_probs, k=5)
    }
    pd.DataFrame([metrics]).to_csv("accuracy_metrics.csv", index=False)

    class_counts = defaultdict(int)
    class_correct = defaultdict(int)

    for true_label, pred_label in zip(test_labels, test_preds):
        class_counts[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1

    class_accuracy_data = []
    for c in sorted(class_counts.keys()):
        class_accuracy_data.append([
            diagnosis_mapping[c],
            class_correct[c] / class_counts[c] if class_counts[c] > 0 else 0,
            class_counts[c]
        ])
    pd.DataFrame(class_accuracy_data, columns=["Type of Sarcoma", "Accuracy", "Number of Cases"]).to_csv(
        "classwise_accuracy.csv", index=False)

    prediction_data = []
    for i, probs in enumerate(test_probs):
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3_indices]
        top3_correct = test_labels[i] in top3_indices
        prediction_data.append([
            test_case_ids[i],
            i,
            diagnosis_mapping[test_labels[i]],
            diagnosis_mapping[top3_indices[0]], f"{top3_probs[0]:.3f}",
            diagnosis_mapping[top3_indices[1]], f"{top3_probs[1]:.3f}",
            diagnosis_mapping[top3_indices[2]], f"{top3_probs[2]:.3f}",
            test_labels[i] == top3_indices[0], top3_correct
        ])
    pd.DataFrame(prediction_data, columns=[
        "Case ID", "Sample ID", "True Label", "Predicted (Top1) Label", "Top1 Probability",
        "Top2 Prediction", "Top2 Probability", "Top3 Prediction", "Top3 Probability", "Top1 Correct?", "Top3 Correct?"
    ]).to_csv("top3_predictions.csv", index=False)

    print("Results saved to accuracy_metrics.csv, classwise_accuracy.csv, and top3_predictions.csv")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    
    parser.add_argument(
        "--metadata_file", type=str, required=True, help="a CSV file containing metadata.")
    parser.add_argument(
        "--embedding_dir", type=str, required=True, help="Directory with slide-level embeddings files.")
                            
    args = parser.parse_args()

    datasets, labels, case_ids = construct_datasets(args.metadata_file, args.embedding_dir)
    
    # Print dataset sizes
    for split in datasets:
        print(f"{split}: {datasets[split].shape}, labels: {labels[split].shape}")
    
    model = train_and_evaluate(datasets['train'], labels['train'], datasets['test'], labels['test'], case_ids['test'])
    print(1)
