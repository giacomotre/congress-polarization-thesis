import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set the model to training mode
    total_loss = 0
    start_time = time.time()

    for step, batch in enumerate(data_loader):
        # Move batch data to the appropriate device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero out any accumulated gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        total_loss += loss.item()

        # Optional: Print training progress
        # if step % 100 == 0: # Print every 100 batches
        #     elapsed = time.time() - start_time
        #     print(f"  Batch {step}/{len(data_loader)} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        device (torch.device): The device to evaluate on (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing evaluation metrics (loss, accuracy, f1, auc, confusion_matrix).
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = [] # To calculate AUC

    start_time = time.time()

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # Move batch data to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Get predictions and probabilities
            # Logits are raw scores. Apply softmax to get probabilities.
            # For binary classification, we only need the probability of the positive class (label 1).
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of the positive class (index 1)
            preds = torch.argmax(logits, dim=1) # Get the index of the max logit as prediction


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # Collect probabilities for AUC

            # Optional: Print evaluation progress
            # if step % 100 == 0: # Print every 100 batches
            #     elapsed = time.time() - start_time
            #     print(f"  Eval Batch {step}/{len(data_loader)} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")


    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    # Convert lists to numpy arrays for scikit-learn metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)


    accuracy = accuracy_score(all_labels, all_preds)
    # Use 'weighted' average for f1_score as recommended for potential class imbalance
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC only if there are at least two unique labels in the batch
    auc = "NA" # Default to NA if AUC can't be calculated
    if len(np.unique(all_labels)) == 2:
        # For AUC, we need the probability of the positive class (label 1)
        auc = roc_auc_score(all_labels, all_probs)
    else:
        print("Skipping AUC calculation: Only one class present in evaluation labels.")


    # Calculate confusion matrix
    # Pass unique labels to confusion_matrix to ensure correct shape even if a class is missing in a small batch
    unique_labels = np.unique(np.concatenate((all_labels, all_preds))) # Get all unique labels from both true and predicted
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels).tolist() # Convert to list for JSON logging


    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'labels': unique_labels.tolist() # Include labels for the confusion matrix plot
    }

    return metrics