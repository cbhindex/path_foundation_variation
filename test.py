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
# TODO

"""

# Package Import
from tqdm import tqdm
import time
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from utils.helper_class_pytorch import SlideBagDataset, AttentionMIL
from utils.helper_functions_pytorch import load_data, collate_fn_random_sampling


#################### define function for model inference ####################

def evaluate_model(model, test_loader, slide_filenames, device, num_class=15):
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
        Number of classes, by default 15.

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
                
                # Adjust label range from [1, 15] to [0, 14]
                batch_labels = batch_labels - 1
                batch_labels = batch_labels.to(device)
                
                for i, patches in enumerate(batch_patches):
                    patches = patches.to(device)
                    slide_id = slide_filenames[batch_idx * test_loader.batch_size + i]  # Get actual case ID
                    
                    # Forward pass
                    output, _ = model(patches)
                    probabilities = F.softmax(output, dim=0)
                    top3_probs, top3_classes = torch.topk(probabilities, 3)
                    
                    # As the labels in the dataset is ranged from 1 to 15, but CrossEntropyLoss 
                    # expects the labels to be in the range 0 to num_classes - 1, so we need to 
                    # adjust the labels to be in the range [0, 14] instead of [1, 15]
                    predicted_class = top3_classes[0].item() + 1  # Map back to [1, 15]
                    true_label = batch_labels[i].item() + 1  # Map back to [1, 15]
                    
                    top1_correct = int(predicted_class == true_label)
                    top3_correct_case = int(true_label in (top3_classes.cpu().numpy() + 1))
                    
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
                        "top_1_pred": predicted_class, # already mapped back to [1, 15] so no need to +1
                        "top_1_prob": top3_probs[0].item(),
                        "top_2_pred": top3_classes[1].item() + 1,
                        "top_2_prob": top3_probs[1].item(),
                        "top_3_pred": top3_classes[2].item() + 1,
                        "top_3_prob": top3_probs[2].item(),
                        "top_1_correct": top1_correct,
                        "top_3_correct": top3_correct_case
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

# Main function
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1",
                        help='Path to validation data folder')
    parser.add_argument('--test_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_1_test.csv", 
                        help='Path to validation label CSV')
    parser.add_argument('--model', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/models/mil_best_model_state_dict_epoch_37.pth", 
                        help='Path to saved model')
    parser.add_argument('--k_instances', type=int, default=500, help='Number of instances per bag')
    parser.add_argument('--output', type=str, 
                        default='.', 
                        help='Path to output folder')

    args = parser.parse_args()
    
    since = time.time()
    
    # testing data preparation
    test_patch_features, test_labels, test_slide_filenames = load_data(args.test_folder, args.test_labels)
    test_dataset = SlideBagDataset(test_patch_features, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        collate_fn=lambda x: collate_fn_random_sampling(x, args.k_instances)
        )

    # debug print
    print("Data loading completed.")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL(input_dim=384, attention_dim=128, num_classes=15).to(device)
    model.load_state_dict(torch.load(args.model))
    
    # Run model evaluation
    top1_results, top3_results, individual_results = evaluate_model(
        model, test_loader, test_slide_filenames, device)

    # TODO 1: run script for 10 times (from test_loader to model inference)
    # TODO 2: save outputs - top1_results, top3_results, individual_results

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))