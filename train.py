#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:42:45 2025

@author: Dr Binghao Chai
@institute: University College London (UCL)

Train an attention-based MIL classifier on slide-level bags of patch embeddings.

Each slide is represented as a variable-length bag of patch embeddings loaded from
``.csv`` or ``.h5`` files. Labels in metadata are expected to start from 1 and
are shifted to 0-based indices internally for ``CrossEntropyLoss``.

Training includes:
1. Random patch sampling per bag (``k_instances``).
2. Validation at each epoch.
3. Learning-rate scheduling on validation loss.
4. Early stopping and checkpointing of the best model.

CLI Arguments
-------------
--train_folder : str
    Primary training embedding directory.
--train_folder_2 : str, optional
    Optional second training embedding directory.
--train_folder_3 : str, optional
    Optional third training embedding directory.
--train_labels : str
    Training label CSV with columns ``case_id`` and ``ground_truth``.
--val_folder : str
    Validation embedding directory.
--val_labels : str
    Validation label CSV with columns ``case_id`` and ``ground_truth``.
--model_folder : str
    Output directory for checkpoints and logs.
--k_instances : int
    Number of patch instances randomly sampled per slide.
--epochs : int
    Maximum number of training epochs.
--lr : float
    Initial learning rate.
--patience : int
    Early-stopping patience (epochs).
--num_class : int
    Number of target classes.
--emb_type : {"h5", "csv"}
    Embedding file format.
"""

import os
import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.helper_class_pytorch import SlideBagDataset, AttentionMIL
from utils.helper_functions_pytorch import collate_fn_random_sampling, load_data, load_data_h5

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

# Train function
def train_model(
        train_loader, val_loader, model, criterion, optimizer, device, 
        model_folder, scheduler=None, num_class=14, epochs=50, patience=10
        ):
    """
    Train and validate an MIL model with checkpointing and early stopping.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training dataloader yielding ``(batch_patches, batch_labels)``.
    val_loader : torch.utils.data.DataLoader
        Validation dataloader yielding ``(batch_patches, batch_labels)``.
    model : torch.nn.Module
        MIL model that returns ``(logits, attention_weights)`` for one slide bag.
    criterion : torch.nn.Module
        Loss function (typically ``CrossEntropyLoss``).
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device for training/inference (CPU or CUDA).
    model_folder : str
        Output directory for model checkpoints.
    scheduler : torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau, optional
        Learning-rate scheduler stepped by validation loss when provided.
    num_class : int, default=14
        Number of target classes.
    epochs : int, default=50
        Maximum number of training epochs.
    patience : int, default=10
        Early-stopping patience measured in epochs without validation improvement.
    """
    
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
            # As the labels in the dataset is ranged from 1 to N, but CrossEntropyLoss 
            # expects the labels to be in the range 0 to num_classes - 1, so we need to 
            # adjust the labels to be in the range [0, 14] instead of [1, N]
            batch_labels = batch_labels - 1  # Adjust label range from [1, N] to [0, N-1] for CrossEntropyLoss
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()  # Zero gradients for the current batch update
            batch_loss = 0.0

            # Process each slide (bag) separately since each has variable patches
            for i, patches in enumerate(batch_patches):
                patches = patches.to(device)  # Move patches to the device (e.g., GPU)
                
                # Forward pass through the model
                output, attention_weights = model(patches)
                
                # Compute the loss
                loss = criterion(output.unsqueeze(0), batch_labels[i].unsqueeze(0))  # Ensure dimensions match
                batch_loss += loss

            # Single optimizer update per batch
            batch_loss = batch_loss / len(batch_patches)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
        
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
                
                # As the labels in the dataset is ranged from 1 to N, but CrossEntropyLoss 
                # expects the labels to be in the range 0 to num_classes - 1, so we need to 
                # adjust the labels to be in the range [0, 14] instead of [1, N]
                batch_labels = batch_labels - 1 # Adjust label range from [1, N] to [0, N-1]
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
        if scheduler is not None:
            scheduler.step(val_loss) 
        
        # Print per-class accuracy
        for i in range(num_class):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
                print(f"Class {i+1} Val Acc: {class_accuracy:.2f}%")
            else:
                print(f"Class {i+1} Val Acc: N/A (No samples)")
        
        # Save the best model if it achieves the best validation accuracy / best loss
        #if val_loss <= best_val_loss or val_accuracy >= best_val_accuracy:
        if val_loss <= best_val_loss:
        # if val_accuracy >= best_val_accuracy:
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
    parser = argparse.ArgumentParser(
        description="Train an attention-based MIL classifier from patch embeddings."
    )
    parser.add_argument('--train_folder', type=str, required=True,
                        help='Primary training embedding directory.')
    parser.add_argument('--train_folder_2', type=str, default=None,
                        help='Optional second training embedding directory.')
    parser.add_argument('--train_folder_3', type=str, default=None,
                        help='Optional third training embedding directory.')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Training label CSV with columns case_id, ground_truth.')
    parser.add_argument('--val_folder', type=str, required=True,
                        help='Validation embedding directory.')
    parser.add_argument('--val_labels', type=str, required=True,
                        help='Validation label CSV with columns case_id, ground_truth.')
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Output directory for checkpoints and logs.')
    parser.add_argument('--k_instances', type=int, default=500,
                        help='Number of patch instances randomly sampled per slide.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early-stopping patience (epochs).')
    parser.add_argument('--num_class', type=int, default=14,
                        help='Number of target classes.')
    parser.add_argument('--emb_type', type=str, default='h5', choices=['h5', 'csv'],
                        help='Embedding file format: h5 or csv.')
    
    args = parser.parse_args()
    
    since = time.time()
    
    # make sure output folder exists
    os.makedirs(f"{args.model_folder}", exist_ok=True)
    
    # # training and validation data preparation        
    if args.emb_type == 'csv':
        train_patch_features, train_labels, train_slide_ids = [], [], []
    
        for all_folder in [args.train_folder, args.train_folder_2, args.train_folder_3]:
            if all_folder:
                features, labels, slide_ids = load_data(all_folder, args.train_labels)
                train_patch_features += features
                train_labels += labels
                train_slide_ids += slide_ids
        val_patch_features, val_labels, _ = load_data(args.val_folder, args.val_labels)
    
    elif args.emb_type == 'h5':
        train_patch_features, train_labels, train_slide_ids = [], [], []
    
        for all_folder in [args.train_folder, args.train_folder_2, args.train_folder_3]:
            if all_folder:
                features, labels, slide_ids = load_data_h5(all_folder, args.train_labels)
                train_patch_features += features
                train_labels += labels
                train_slide_ids += slide_ids
        val_patch_features, val_labels, _ = load_data_h5(args.val_folder, args.val_labels)
    
    # get train-validation size (number of slides)
    print(f"Number of training slides: {len(train_labels)}")
    print(f"Number of validation slides: {len(val_labels)}")
    
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
    
    # different foundation model have different dimensions
    emb_dim = np.shape(val_patch_features[0])[1]
    
    # define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL(input_dim=emb_dim, attention_dim=128, num_classes=args.num_class)
    
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
        scheduler=scheduler,
        num_class=args.num_class, epochs=args.epochs, patience=args.patience
        )

    # Print the total runtime
    time_elapsed = time.time() - since
    print("Task complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
