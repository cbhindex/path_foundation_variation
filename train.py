#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:42:45 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

This script trains an attention-based multi-instance learning (MIL) model for 
digital pathology classification. It processes whole slide images (WSIs) that 
have been converted into patch-level 384-dimensional feature embeddings. 
The script loads training and validation data from CSV files, where each slide (bag) 
contains a variable number of patches (instances). The model uses an attention 
mechanism to aggregate patch features into a slide-level representation for 
classification.

The training process includes data loading, model training, validation, and 
early stopping. The train_model() function iterates through multiple epochs, 
adjusting labels, computing loss, updating model parameters, and tracking accuracy 
per class. Validation is performed after each epoch, and the best model is saved 
based on validation accuracy. If the validation loss does not improve for a specified 
number of epochs (patience), early stopping is triggered to prevent overfitting. 
This ensures efficient and optimal training while maintaining generalisation 
performance.

Parameters
----------
train_folder: str
    Path to training data folder.

train_labels: str
    Path to training label CSV.

val_folder: str
    Path to validation data folder.

val_labels: str
    Path to validation label CSV.
    
model_folder: str
    Path to saved model folder, this is also the output folder.

k_instances: int
    Number of instances per bag.
    
epochs: int
    Number of training epochs.
    
lr: float
    Learning rate.

patience: int
    Number of patient epochs for early stop

"""

import os
import argparse
import time

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.helper_class_pytorch import SlideBagDataset, AttentionMIL
from utils.helper_functions_pytorch import collate_fn_random_sampling, load_data

#################### define the training loop ####################

# Train function
def train_model(
        train_loader, val_loader, model, criterion, optimizer, device, 
        model_folder, num_class=15, epochs=50, patience=10
        ):
    
    # Ensure the output folder (the folder to save model) exists
    os.makedirs(model_folder, exist_ok=True)
    
    model.to(device)
    
    best_val_accuracy = 0.0
    best_epoch = 0
    best_val_loss = float('inf')  # Track best validation loss
    epochs_no_improve = 0  # Track epochs without improvement
    
    for epoch in range(epochs):
        # Training Phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct, total = 0, 0
        
        for batch_patches, batch_labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            
            # As the labels in the dataset is ranged from 1 to 15, but CrossEntropyLoss 
            # expects the labels to be in the range 0 to num_classes - 1, so we need to 
            # adjust the labels to be in the range [0, 14] instead of [1, 15]
            batch_labels = batch_labels - 1  # Adjust label range from [1, 15] to [0, 14] for CrossEntropyLoss
            batch_labels = batch_labels.to(device)

            # Process each slide (bag) separately since each has variable patches
            for i, patches in enumerate(batch_patches):
                patches = patches.to(device)  # Move patches to the device (e.g., GPU)
                
                # Forward pass through the model
                output, attention_weights = model(patches)
                
                # Compute the loss
                loss = criterion(output.unsqueeze(0), batch_labels[i].unsqueeze(0))  # Ensure dimensions match
                
                # Backward pass and update the model parameters
                loss.backward()  # Backpropagation
                optimizer.step()  # Gradient descent step
                
                running_loss += loss.item()
        
        # Store the average training loss for this epoch
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        
        # Validation Phase
        model.eval()  # Set model to evaluation mode
        val_loss, correct, total = 0.0, 0, 0
        class_correct = torch.zeros(num_class).to(device)
        class_total = torch.zeros(num_class).to(device)
        
        with torch.no_grad():  # No need to calculate gradients during validation
            for batch_patches, batch_labels in val_loader:
                
                # As the labels in the dataset is ranged from 1 to 15, but CrossEntropyLoss 
                # expects the labels to be in the range 0 to num_classes - 1, so we need to 
                # adjust the labels to be in the range [0, 14] instead of [1, 15]
                batch_labels = batch_labels - 1 # Adjust label range from [1, 15] to [0, 14]
                batch_labels = batch_labels.to(device)
                
                for i, patches in enumerate(batch_patches):
                    patches = patches.to(device)
                    
                    # Forward pass
                    output, attention_weights = model(patches)
                    
                    # Compute validation loss
                    loss = criterion(output.unsqueeze(0), batch_labels[i].unsqueeze(0))
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output, 0)  # Get predicted class
                    total += 1
                    correct += (predicted == batch_labels[i]).sum().item()
                    
                    # Calculate per-class accuracy
                    label = batch_labels[i]
                    class_correct[label] += (predicted == label).item()
                    class_total[label] += 1
        
        # Compute validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Overall Val Acc: {val_accuracy:.2f}%")
        
        # Reduce learning rate if validation loss stops improving
        scheduler.step(val_loss) 
        
        # Print per-class accuracy
        for i in range(15):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
                print(f"Class {i+1} Val Acc: {class_accuracy:.2f}%")
            else:
                print(f"Class {i+1} Val Acc: N/A (No samples)")
        
        # Save the best model if it achieves the best validation accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1  # Save the current epoch (1-based index)
            epochs_no_improve = 0  # Reset counter since we have improvement
            
            # Save the model's state dictionary (weights)
            torch.save(model.state_dict(), f"{model_folder}/mil_best_model_state_dict_epoch_{best_epoch}.pth")
            
            # Optionally, save the entire model (architecture + weights)
            torch.save(model, f"{model_folder}/mil_best_model_full_epoch_{best_epoch}.pth")
            
            print(f"Best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}%")
        else:
            epochs_no_improve += 1  # Increase counter if no improvement

        # **EARLY STOPPING CHECK**: Stop training if no improvement for `patience` epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            break  # Stop training
    
    # After training, print information about the best model
    print("Training completed.")
    print(f"The best model was saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}%")

# Main function
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1", 
                        help='Path to training data folder')
    parser.add_argument('--train_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_1_train.csv",
                        help='Path to training label CSV')
    parser.add_argument('--val_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/embeddings/cohort_1",
                        help='Path to validation data folder')
    parser.add_argument('--val_labels', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/labels/cohort_1_val.csv", 
                        help='Path to validation label CSV')
    parser.add_argument('--model_folder', type=str, 
                        default="/home/digitalpathology/workspace/path_foundation_stain_variation/models", 
                        help='Path to saved model folder')
    parser.add_argument('--k_instances', type=int, default=500, help='Number of instances per bag')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Number of patient epochs for early stop')
    
    args = parser.parse_args()
    
    since = time.time()
    
    # training and validation data preparation
    train_patch_features, train_labels, _ = load_data(args.train_folder, args.train_labels)
    val_patch_features, val_labels, _ = load_data(args.val_folder, args.val_labels)
    
    train_dataset = SlideBagDataset(train_patch_features, train_labels)
    val_dataset = SlideBagDataset(val_patch_features, val_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        collate_fn=lambda x: collate_fn_random_sampling(x, args.k_instances)
        )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
        collate_fn=lambda x: collate_fn_random_sampling(x, args.k_instances)
        )
    
    # debug print
    print("Data loading completed.")
    
    # define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL(input_dim=384, attention_dim=128, num_classes=15)
    
    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # launch training process
    train_model(
        train_loader, val_loader, model, criterion, optimizer, device, 
        model_folder=args.model_folder,
        num_class=15, epochs=args.epochs, patience=args.patience
        )

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))