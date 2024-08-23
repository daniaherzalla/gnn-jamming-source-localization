import csv
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
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
from config import params

# if params['reproduce']:
#     set_seeds_and_reproducibility()

setup_logging()


def initialize_model(device: torch.device, params: dict, steps_per_epoch=None) -> Tuple[GNN, optim.Optimizer, OneCycleLR, torch.nn.Module]:
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
    if params['coords'] == 'cartesian':
        in_channels = len(params['additional_features']) + len(params['required_features']) + 1  # Add one (two for 3D) since position data is considered separately for each coordinate
    elif params['coords'] == 'polar':
        in_channels = len(params['additional_features']) + len(params['required_features']) + 2
    else:
        raise "Unknown coordinate system"
    # print('params: ', params)
    print('in_channels: ', in_channels)
    # print("params['additional_features']: ", params['additional_features'])
    model = GNN(dropout_rate=params['dropout_rate'], num_heads=params['num_heads'], model_type=params['model'], in_channels=in_channels, hidden_channels=params['hidden_channels'], out_channels=params['out_channels'], num_layers=params['num_layers']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
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
            # output = convert_output(output, device)  # Ensure this function is suitable for validation context
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
    predictions, actuals, rmse_list = [], [], []

    with torch.no_grad():
        for data in loader:
            # Process individual data items
            data = data.to(device)
            output = model(data)
            # print("output: ", output)

            # Convert and uncenter using the index to retrieve the correct midpoint
            predicted_coords = convert_output_eval(output, data, 'prediction', device)
            actual_coords = convert_output_eval(data.y, data, 'target', device)

            predictions.append(predicted_coords.cpu().numpy())
            # print("prediction: ", predicted_coords.cpu().numpy())
            actuals.append(actual_coords.cpu().numpy())
            mse = mean_squared_error(actual_coords.cpu().numpy(), predicted_coords.cpu().numpy())
            rmse = math.sqrt(mse)
            rmse_list.append(rmse)

    predictions = np.concatenate([np.array(pred).flatten() for pred in predictions])
    actuals = np.concatenate([np.array(act).flatten() for act in actuals])
    rmse_list = np.concatenate([np.array(rmse_list).flatten() for rmse in rmse_list])
    print("predictions: ", predictions)
    print("actuals: ", actuals)

    # Assuming `predictions` and `actuals` are your arrays of predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.title('Predictions vs. Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)  # Line showing perfect predictions
    # plt.savefig(f'results/graphs/polar_knn_minmax_trial2.png')
    # plt.show()

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

    return predictions, actuals, err_metrics, rmse_list


def predict_and_evaluate_full(loader, model, device, original_dataset):
    """
    Extended evaluation function to gather all required details for plotting, including
    fetching details using IDs from the data DataFrame.

    Args:
        loader: DataLoader providing the dataset for evaluation.
        model: Trained model for evaluation.
        device: Device to perform computations on.
        original_dataset: Original DataFrame with additional information.

    Returns:
        Predictions, actuals, and node details including RSSI and other metrics.
    """
    model.eval()
    predictions, actuals, node_details = [], [], []

    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            output = model(data_batch)

            # Convert and uncenter using the provided conversion function
            predicted_coords = convert_output_eval(output, data_batch, 'prediction', device)
            actual_coords = convert_output_eval(data_batch.y, data_batch, 'target', device)

            # Collect predictions and actuals
            predictions.append(predicted_coords.cpu().numpy())
            actuals.append(actual_coords.cpu().numpy())

            # # Fetch additional data using ID from the original DataFrame
            # ids = data_batch.id.cpu().numpy()
            # batch_details = original_dataset.loc[original_dataset['id'].isin(ids)]
            # for id in ids:
            #     row = batch_details[batch_details['id'] == id].iloc[0]
            #     node_details.append({
            #         'node_positions': row['node_positions'],
            #         'node_rssi': row['node_rssi'],
            #         'node_noise': row['node_noise'],
            #         'jammer_position': row['jammer_position'],
            #         'node_states': row['node_states']
            #     })

    # Flatten predictions and actuals if they are nested lists
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

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

    return predictions, actuals, node_details, err_metrics


def predict(loader, model, device):
    """
    Extended evaluation function to gather all required details for plotting, including
    fetching details using IDs from the data DataFrame.

    Args:
        loader: DataLoader providing the dataset for evaluation.
        model: Trained model for evaluation.
        device: Device to perform computations on.

    Returns:
        Predictions, actuals, and node details including RSSI and other metrics.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            output = model(data_batch)

            # Convert and uncenter using the provided conversion function
            predicted_coords = convert_output_eval(output, data_batch, 'prediction', device)

            # Collect predictions and actuals
            predictions.append(predicted_coords.cpu().numpy())

    # Flatten predictions and actuals if they are nested lists
    predictions = np.concatenate(predictions)

    return predictions


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


def plot_network_with_rssi(node_positions, final_rssi, jammer_position, noise_floor_db, jammed, prediction):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes
    for idx, pos in enumerate(node_positions):
        color = 'red' if jammed[idx] else 'blue'
        node_info = f"Node {idx}\nRSSI: {final_rssi[idx]:.2f} dB\nNoise: {noise_floor_db[idx]:.2f} dB"
        ax.plot(pos[0], pos[1], 'o', color=color)  # Nodes in blue or red depending on jamming status
        ax.text(pos[0], pos[1], node_info, fontsize=9, ha='right')

    # Plot jammer
    ax.plot(jammer_position[0], jammer_position[1], 'r^', markersize=15)  # Jammer in red
    ax.text(jammer_position[0], jammer_position[1], ' Jammer', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=15)

    # Plot prediction
    ax.plot(prediction[0], prediction[1], 'gx', markersize=15)  # Prediction in green
    ax.text(prediction[0], prediction[1], ' Prediction', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=15)

    # Calculate and plot the line between jammer position and prediction
    line = np.array([jammer_position, prediction])
    ax.plot(line[:, 0], line[:, 1], 'k--')

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(jammer_position) - np.array(prediction)) ** 2))

    # Annotate the line with the RMSE
    mid_point = np.mean(line, axis=0)
    ax.text(mid_point[0], mid_point[1]-20, f'RMSE: {rmse:.2f}m', fontsize=12, color='black')

    coord_system = params['coords']
    # ax.set_title(f'Network Topology with RSSI, Noise Floor, and {coord_system} Jammer Prediction', fontsize=11)
    ax.set_title(f'Network Topology with Actual and GNN Predicted Jammer Position', fontsize=11)
    ax.set_xlabel('X position (m)', fontsize=14)
    ax.set_ylabel('Y position (m)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
