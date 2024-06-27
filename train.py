import csv
import json
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
from data_processing import convert_output, convert_output_eval

set_seeds_and_reproducibility()

setup_logging()

# Load midpoints once
with open('midpoints.json', 'r') as f:
    midpoints = json.load(f)


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
        steps_per_epoch (int): Max number of steps per epoch to run training for.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0  # Use this to correctly compute average loss
    for data in train_loader:
        if steps_per_epoch is not None and num_batches >= steps_per_epoch:
            break
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output * 2
        # output[:, 0] = output[:, 0] * torch.sqrt(torch.tensor(3.0))
        # Multiply all the radii (1st column) by sqrt(3)
        # print(output.shape)
        # quit()
        output = convert_output(output, 'prediction')  # Ensure output conversion uses PyTorch and remains on the GPU
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * data.num_graphs  # Ensure data.num_graphs or equivalent is valid
        num_batches += 1

    # Clear CUDA cache if necessary (usually not required every batch)
    torch.cuda.empty_cache()

    return total_loss / sum(data.num_graphs for data in train_loader)  # This assumes each batch might have a different size


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
    total_graphs = 0  # Use this to correctly compute average loss if batch sizes vary
    with torch.no_grad():
        for data in validate_loader:
            data = data.to(device)
            output = model(data)
            output = output * 2
            # output[:, 0] = output[:, 0] * torch.sqrt(torch.tensor(3.0))
            output = convert_output(output, 'prediction')  # Ensure this function is suitable for validation context
            loss = criterion(output, data.y)
            total_loss += data.num_graphs * loss.item()
            total_graphs += data.num_graphs  # Accumulate the total number of graphs or samples processed

    return total_loss / total_graphs  # Use total_graphs for a more accurate average if batch sizes are not uniform


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
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for data in loader:
            ids = data.id  # This will be a tensor with batch_size elements

            for i in range(len(ids)):
                id_ = ids[i].item()  # Extract the id for each item in the batch

                # Process individual data items
                data_item = data[i].to(device)
                output = model(data_item)
                # output[:, 0] = output[:, 0] * torch.sqrt(torch.tensor(3.0))
                output = output * 2
                # print("output: ", output)

                # Convert and uncenter using the index to retrieve the correct midpoint
                predicted_coords = convert_output_eval(output, 'prediction', device, id_, midpoints)
                actual_coords = data_item.y  # convert_output_eval(data_item.y, 'target', device, id_, midpoints)

                predictions.append(predicted_coords.cpu().numpy())
                print("prediction: ", predicted_coords.cpu().numpy())
                actuals.append(actual_coords.cpu().numpy())

    # quit()
    predictions = np.concatenate([np.array(pred).flatten() for pred in predictions])
    actuals = np.concatenate([np.array(act).flatten() for act in actuals])
    print("predictions: ", predictions)
    print("actuals: ", actuals)

    import matplotlib.pyplot as plt

    # Assuming `predictions` and `actuals` are your arrays of predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.title('Predictions vs. Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)  # Line showing perfect predictions
    # plt.savefig(f'results/graphs/polar_knn_minmax_trial2.png')
    plt.show()

    # calculate metrics MSE, RMSE using predictions and actuals
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    print(f'Mean Squared Error: {mse}')
    rmse = math.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')

    err_metrics = {
        'actuals': actuals,
        'predictions': predictions,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

    return predictions, actuals, err_metrics


def save_err_metrics(data, filename: str = 'results/error_metrics_converted.csv') -> None:
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        # Write the header only if the file didn't exist before
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(data)
