import csv
import math
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from model import GNN
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from custom_logging import setup_logging
from utils import set_seeds_and_reproducibility
from data_processing import convert_output

set_seeds_and_reproducibility()

setup_logging()


def initialize_model(device: torch.device, params: dict, steps_per_epoch=None) -> Tuple[GNN, optim.Optimizer, ReduceLROnPlateau, torch.nn.Module]:
    """
    Initialize the model, optimizer, scheduler, and loss criterion.

    Args:
        device (torch.device): Device to run the model on.
        params (dict): Dictionary of model parameters.

    Returns:
        model (GNN): Initialized model.
        optimizer (optim.Optimizer): Optimizer for the model.
        scheduler (ReduceLROnPlateau): Learning rate scheduler.
        criterion (torch.nn.Module): Loss criterion.
    """
    logging.info("Initializing model...")
    model = GNN().to(device)
    # model = GNN(num_heads_first_two=4, num_heads_final=6, features_per_head_first_two=256, features_per_head_final=121).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    # optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, verbose=True)
    print("steps per epoch one cycle lr: ", steps_per_epoch)
    scheduler = OneCycleLR(optimizer, max_lr=params['learning_rate'], epochs=params['max_epochs'], steps_per_epoch=steps_per_epoch, pct_start=0.2, anneal_strategy='linear')
    criterion = torch.nn.MSELoss()
    return model, optimizer, scheduler, criterion


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, steps_per_epoch: int, scheduler) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss criterion.
        device (torch.device): Device to run the model on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_steps = 0
    # steps_per_epoch = None
    for data in train_loader:
        if steps_per_epoch is not None and num_steps >= steps_per_epoch:
            break
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output = convert_output(output, device)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        num_steps += 1
        total_loss += loss.item() * data.num_graphs
    # Clear CUDA cache
    torch.cuda.empty_cache()
    return total_loss / len(train_loader.dataset)


def validate(model: torch.nn.Module, validate_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device) -> float:
    """
    Validate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        validate_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss criterion.
        device (torch.device): Device to run the model on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in validate_loader:
            data = data.to(device)
            output = model(data)
            # output = convert_output(output, device)
            loss = criterion(output, data.y)
            total_loss += data.num_graphs * loss.item()
    return total_loss / len(validate_loader.dataset)


def predict_and_evaluate(model, loader, device):
    """
    Evaluate the model and compute performance metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader providing the dataset for evaluation.
        device (torch.device): The device to perform the computations on (e.g., 'cpu' or 'cuda').
        scaler (object): The scaler used to normalize the data, with an inverse_transform method to denormalize it.

    Returns:
        tuple: A tuple containing two lists:
            - predictions (list): The predicted values after denormalization.
            - actuals (list): The actual values after denormalization.
    """
    # need to save that data in new file to plot as well after 3 trials
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            # Apply inverse transformation to the model output
            predicted_coords = convert_output(output.cpu().numpy(), device)
            actual_coords = convert_output(data.y.cpu().numpy(), device)

            # predicted_coords = scaler.inverse_transform(output.cpu().numpy())
            # actual_coords = scaler.inverse_transform(data.y.cpu().numpy())

            predictions.extend(predicted_coords)
            actuals.extend(actual_coords)

    predictions = np.concatenate([np.array(pred).flatten() for pred in predictions])
    actuals = np.concatenate([np.array(act).flatten() for act in actuals])

    # calculate metrics MSE, RMSE using predictions and actuals
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    print(f'Mean Squared Error: {mse}')
    rmse = math.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')

    err_metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

    return predictions, actuals, err_metrics


def save_err_metrics(data, filename: str = 'results/error_metrics.csv') -> None:
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        # Write the header only if the file didn't exist before
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(data)
