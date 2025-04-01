#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:47:01 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script is to run the MIL classifier on embeddings for testing datasets. It 
takes WSIs that have been converted into patch-level 384-dimensional feature embeddings 
as the input, together with a metadata spreadsheet containin case_id and ground_truth.

Parameters
----------
test_folder: str
    Path to validation data folder.
    
test_labels: str
    Path to validation label CSV.

model: str
    Path to saved model.
    
output: str
    Path to output folder.
    
"""

# Package Import
import os
import time
import argparse
from tqdm import tqdm

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.helper_class_pytorch import SlideBagDataset, AttentionMIL
from utils.helper_functions_pytorch import load_data, collate_fn_variable_size


#################### define function for model inference ####################

def evaluate_model(model, test_loader, slide_filenames, device, num_class):
    """
    Function to evaluate the model on a test dataset.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained AttentionMIL model.
        
    test_loader : DataLoader
        DataLoader containing the test dataset.
        
    slide_filenames : list
        List of actual case IDs corresponding to the test slides.
        
    device : torch.device
        Device to run the evaluation on (CPU or GPU).
        
    num_class : int, optional
        Number of classes.

    Returns
    -------
    dict
        Overall accuracy and per-class accuracy.
        
    dict
        Individual predictions and probabilities for each case.
        
    dict
        Top-3 accuracy overall and per-class.
    """
    model.eval()  # Set model to evaluation mode
    
    correct, total = 0, 0
    class_correct = torch.zeros(num_class).to(device)
    class_total = torch.zeros(num_class).to(device)
    top3_correct, total_cases = 0, 0
    class_top3_correct = torch.zeros(num_class).to(device)
    
    individual_results = []
    
    # Initialize tqdm progress bar (updates per slide)
    with tqdm(total=len(slide_filenames), desc="Processing Slides", unit="slide") as pbar:
        with torch.no_grad():  # No need to calculate gradients during validation
            for batch_idx, (batch_patches, batch_labels) in enumerate(test_loader):
                
                # Adjust label range from [1, X] to [0, X - 1]
                batch_labels = batch_labels - 1
                batch_labels = batch_labels.to(device)
                
                for i, patches in enumerate(batch_patches):
                    patches = patches.to(device)
                    slide_id = slide_filenames[batch_idx * test_loader.batch_size + i]  # Get actual case ID
                    
                    # Forward pass
                    output, _ = model(patches)
                    probabilities = F.softmax(output, dim=0)
                    # Extract top-3 and top-5 predictions and probabilities
                    top3_probs, top3_classes = torch.topk(probabilities, 3)
                    top5_probs, top5_classes = torch.topk(probabilities, 5)
                    
                    # As the labels in the dataset is ranged from 1 to X, but CrossEntropyLoss 
                    # expects the labels to be in the range 0 to X - 1, so we need to 
                    # adjust the labels to be in the range [0, X - 1] instead of [1, X]
                    # Map predictions back to range [1, X]
                    top5_preds = (top5_classes.cpu().numpy() + 1).tolist()
                    predicted_class = top5_preds[0]
                    true_label = batch_labels[i].item() + 1  # Map back to [1, X]
                    
                    top1_correct = int(predicted_class == true_label)
                    top3_correct_case = int(true_label in (top3_classes.cpu().numpy() + 1)) # Map back to [1, X]
                    # Check if ground truth is in top-5 predictions
                    top5_correct_case = int(true_label in top5_preds)
                    
                    total += 1
                    correct += top1_correct
                    
                    # Calculate per-class accuracy
                    label_idx = batch_labels[i]
                    class_correct[label_idx] += top1_correct
                    class_total[label_idx] += 1
                    
                    # Calculate top-3 accuracy
                    top3_correct += top3_correct_case
                    class_top3_correct[label_idx] += top3_correct_case
                    total_cases += 1
                    
                    # Store individual results
                    individual_results.append({
                        "slide_id": slide_id,  # Actual case ID
                        "ground_truth": true_label,
                        "top_1_pred": top5_preds[0],
                        "top_1_prob": top5_probs[0].item(),
                        "top_2_pred": top5_preds[1],
                        "top_2_prob": top5_probs[1].item(),
                        "top_3_pred": top5_preds[2],
                        "top_3_prob": top5_probs[2].item(),
                        "top_4_pred": top5_preds[3],
                        "top_4_prob": top5_probs[3].item(),
                        "top_5_pred": top5_preds[4],
                        "top_5_prob": top5_probs[4].item(),
                        "top_1_correct": top1_correct,
                        "top_3_correct": top3_correct_case,
                        "top_5_correct": top5_correct_case
                    })
                    
                    # Update progress bar after processing each slide
                    pbar.update(1)
    
    # Compute overall accuracy
    overall_top1_accuracy = 100 * correct / total
    overall_top3_accuracy = 100 * top3_correct / total_cases
    
    # Compute per-class accuracy
    top1_results = {"overall_top_1_accuracy": overall_top1_accuracy}
    top3_results = {"overall_top_3_accuracy": overall_top3_accuracy}
    
    for i in range(num_class):
        class_key = f"top_1_class_{i+1}_accuracy"
        class_top3_key = f"top_3_class_{i+1}_accuracy"
    
        if class_total[i] > 0:
            top1_results[class_key] = (100 * class_correct[i] / class_total[i]).cpu().numpy().item()
            top3_results[class_top3_key] = (100 * class_top3_correct[i] / class_total[i]).cpu().numpy().item()
        else:
            top1_results[class_key] = None
            top3_results[class_top3_key] = None
    
    return top1_results, top3_results, individual_results

# Define the mapping dictionary for numerical values to diagnosis text
# diagnosis_mapping_15class = {
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
diagnosis_mapping = {
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

# Main function
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1",
                        help='Path to validation data folder')
    parser.add_argument('--test_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels_14classes/cohort_1_test.csv", 
                        help='Path to validation label CSV')
    parser.add_argument('--model', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/models/trained_on_14slides_3dhistech/mil_best_model_state_dict_epoch_8.pth", 
                        help='Path to saved model')
    parser.add_argument('--output', type=str, 
                        default='/home/digitalpathology/workspace/path_foundation_stain_variation/output/trained_on_14slides_3dhistech', 
                        help='Path to output folder')

    args = parser.parse_args()
    
    since = time.time()
    
    cohort_id = os.path.basename(args.test_folder)
    # make sure output folder exists
    os.makedirs(f"{args.output}/{cohort_id}", exist_ok=True)
    
    # testing data preparation
    test_patch_features, test_labels, test_slide_filenames = load_data(args.test_folder, args.test_labels)
    test_dataset = SlideBagDataset(test_patch_features, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        collate_fn=collate_fn_variable_size
        )

    # debug print
    print("Data loading completed.")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL(input_dim=384, attention_dim=128, num_classes=14).to(device)
    model.load_state_dict(torch.load(args.model))
    
    # Run model evaluation
    top1_results, top3_results, individual_results = evaluate_model(
        model, test_loader, test_slide_filenames, device, num_class=14)
    
    df_top1_results = pd.DataFrame(list(top1_results.items()))
    df_top1_results.to_csv(
        f"{args.output}/{cohort_id}/{cohort_id}_top1_accuracy.csv", 
        index=False, header=False
        )
    
    df_top3_results = pd.DataFrame(list(top3_results.items()))
    df_top3_results.to_csv(
        f"{args.output}/{cohort_id}/{cohort_id}_top3_accuracy.csv", 
        index=False, header=False
        )
    
    df_individual_results = pd.DataFrame(individual_results)
    df_individual_results.to_csv(
        f"{args.output}/{cohort_id}/{cohort_id}_individual_results.csv", 
        index=False
        )
    
    # convert numerical diagnostic labels to text for pathologist case review
    # Define the input and output file paths (Change these when processing another file)
    text_csv_input_file = f"{args.output}/{cohort_id}/{cohort_id}_individual_results.csv"
    text_csv_output_file = f"{args.output}/{cohort_id}/{cohort_id}_individual_results_diagnosis_in_text.csv"

    # Load the CSV file
    text_csv_df = pd.read_csv(text_csv_input_file)
    
    # Columns to replace numerical values with diagnosis text
    columns_to_replace = ["ground_truth", "top_1_pred", "top_2_pred", "top_3_pred", "top_4_pred", "top_5_pred"]
    
    # Replace numerical values with diagnosis text
    for col in columns_to_replace:
        if col in text_csv_df.columns:
            text_csv_df[col] = text_csv_df[col].map(diagnosis_mapping)
    
    # Save the modified file
    text_csv_df.to_csv(text_csv_output_file, index=False)

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))