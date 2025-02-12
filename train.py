import csv
import logging
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch
from tqdm import tqdm

from config import params
from custom_logging import setup_logging
from data_processing import convert_output_eval
from model import GNN
from utils import AverageMeter

setup_logging()


def initialize_model(device: torch.device, params: dict, steps_per_epoch=None, deg_histogram=None) -> Tuple[GNN, optim.Optimizer, OneCycleLR, torch.nn.Module]:
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
    if 'angle_of_arrival' in params['required_features']:
        feature_dims = 2
    else:
        feature_dims = 1
    if 'moving_avg_aoa' in params['additional_features']:
        feature_dims = 3
    if params['coords'] == 'cartesian':
        in_channels = len(params['additional_features']) + len(
            params['required_features']) + feature_dims  # Add one (two for 3D) since position data is considered separately for each coordinate and one more for sin cos of aoa
    elif params['coords'] == 'polar':
        in_channels = len(params['additional_features']) + len(params['required_features']) + 2  # r, sin cos theta, sin cos aoa
    else:
        raise "Unknown coordinate system"

    print('in_channels: ', in_channels)
    model = GNN(in_channels=in_channels, dropout_rate=params['dropout_rate'], num_heads=params['num_heads'], model_type=params['model'], hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'], num_layers=params['num_layers'], deg=deg_histogram).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    # optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=params['learning_rate'], epochs=params['max_epochs'], steps_per_epoch=steps_per_epoch, pct_start=0.2, anneal_strategy='linear') # 10 epochs warmup (sgd momentum 0.9) #(10/params['max_epochs'])
    regression_criterion = torch.nn.MSELoss()
    classification_criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, regression_criterion, classification_criterion


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, regression_criterion: torch.nn.Module, classification_criterion: torch.nn.Module,  device: torch.device, steps_per_epoch: int,
          scheduler) -> float:
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
    total_loss = AverageMeter()
    regression_loss_meter = AverageMeter()
    classification_loss_meter = AverageMeter()
    model.train()
    progress_bar = tqdm(train_loader, total=steps_per_epoch, desc="Training", leave=True)

    # Dictionary to store the results by graph index and epoch
    detailed_metrics = []

    for num_batches, data in enumerate(progress_bar):
        if steps_per_epoch is not None and num_batches >= steps_per_epoch:
            break

        data = data.to(device)
        optimizer.zero_grad()
        regression_output, classification_output = model(data)
        # print('regression_output: ', regression_output)
        # print('classification_output: ', classification_output)
        regression_loss = regression_criterion(regression_output, data.y[:, :-1])  # Assuming data.y contains the regression labels
        classification_loss = classification_criterion(classification_output, data.y[:, -1].long())  # Assuming data.y_class contains the classification labels
        loss = regression_loss + classification_loss

        # output = model(data)
        # loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update AverageMeter with the current batch loss
        total_loss.update(loss.item(), data.num_graphs)
        regression_loss_meter.update(regression_loss.item(), data.num_graphs)
        classification_loss_meter.update(classification_loss.item(), data.num_graphs)

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        # Log the average loss so far and the current learning rate in the progress bar
        progress_bar.set_postfix({
            "Train Loss (MSE)": regression_loss_meter.avg,
            "Learning Rate": current_lr
        })

        # Dictionary to store individual graph details
        graph_details = {}

        # Calculate RMSE for each graph in the batch
        for idx in range(data.num_graphs):
            prediction = convert_output_eval(regression_output[idx], data[idx], 'prediction', device)
            actual = convert_output_eval(data.y[idx, :-1], data[idx], 'target', device)

            mse = mean_squared_error(actual.cpu().numpy(), prediction.cpu().numpy())
            rmse = math.sqrt(mse)
            perc_completion = data.perc_completion[idx].item()

            # Storing the metrics in the dictionary with graph id as key
            graph_details[idx] = {'rmse': rmse, 'perc_completion': perc_completion}

        # Append to the detailed metrics dict
        detailed_metrics.append(graph_details)

    # Return the average loss tracked by AverageMeter
    return total_loss.avg, detailed_metrics


def validate(model: torch.nn.Module, validate_loader: torch.utils.data.DataLoader, regression_criterion: torch.nn.Module, classification_criterion: torch.nn.Module, device: torch.device, test_loader=False) -> float:
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
    predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list = [], [], [], [], [], [], []
    class_predictions, class_actuals = [], []
    # total_loss = AverageMeter()
    regression_loss_meter = AverageMeter()
    classification_loss_meter = AverageMeter()
    progress_bar = tqdm(validate_loader, desc="Validating", leave=True)

    # Dictionary to store the results by graph index and epoch
    detailed_metrics = []

    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)
            # output = model(data)
            regression_output, classification_output = model(data)
            regression_loss = regression_criterion(regression_output, data.y[:, :-1])
            classification_loss = classification_criterion(classification_output, data.y[:, -1].long())

            # combined_loss = regression_loss + classification_loss
            # total_loss.update(combined_loss.item(), data.num_graphs)
            regression_loss_meter.update(regression_loss.item(), data.num_graphs)
            classification_loss_meter.update(classification_loss.item(), data.num_graphs)

            # Compute predicted class labels
            class_pred_labels = torch.argmax(classification_output, dim=1)
            class_true_labels = data.y[:, -1]

            # Store predictions and actuals for accuracy calculation
            # class_predictions.extend(class_pred_labels.cpu().numpy())
            # class_actuals.extend(class_true_labels.cpu().numpy())

            # Ensure class_true_labels and class_pred_labels are flat arrays
            class_true_labels = class_true_labels.flatten().cpu().numpy()  # Flatten and convert to numpy
            class_pred_labels = class_pred_labels.flatten().cpu().numpy()  # Flatten and convert to numpy

            # Store predictions and actuals for accuracy calculation
            class_predictions.extend(class_pred_labels)
            class_actuals.extend(class_true_labels)

            if test_loader:
                predicted_coords = convert_output_eval(regression_output, data, 'prediction', device)
                actual_coords = convert_output_eval(data.y[:, :-1], data, 'target', device)

                predictions.append(predicted_coords.cpu().numpy())
                actuals.append(actual_coords.cpu().numpy())

                # Storing detailed metrics for further analysis
                graph_details = {
                    'regression_loss': regression_loss.item(),
                    'classification_loss': classification_loss.item(),
                    # 'combined_loss': combined_loss.item()
                }

                perc_completion_list.append(data.perc_completion.cpu().numpy())
                pl_exp_list.append(data.pl_exp.cpu().numpy())
                sigma_list.append(data.sigma.cpu().numpy())
                jtx_list.append(data.jtx.cpu().numpy())
                num_samples_list.append(data.num_samples.cpu().numpy())
            else:
                graph_details = {}

                # Calculate RMSE for each graph in the batch
                for idx in range(data.num_graphs):
                    # prediction = convert_output_eval(output[idx], data[idx], 'prediction', device)
                    # actual = convert_output_eval(data.y[idx], data[idx], 'target', device)

                    prediction = convert_output_eval(regression_output[idx], data[idx], 'prediction', device)
                    actual = convert_output_eval(data.y[idx, :-1], data[idx], 'target', device)

                    mse = mean_squared_error(actual.cpu().numpy(), prediction.cpu().numpy())
                    rmse = math.sqrt(mse)
                    perc_completion = data.perc_completion[idx].item()
                    pl_exp = data.pl_exp[idx].item()
                    sigma = data.sigma[idx].item()
                    jtx = data.jtx[idx].item()
                    num_samples = data.num_samples[idx].item()

                    # Storing the metrics in the dictionary with graph id as key
                    graph_details[idx] = {'rmse': rmse, 'perc_completion': perc_completion, 'pl_exp': pl_exp, 'sigma': sigma, 'jtx': jtx, 'num_samples': num_samples}

                # Append to the detailed metrics dict
                detailed_metrics.append(graph_details)

            # Update the progress bar with the running average RMSE
            progress_bar.set_postfix({"Validation Loss (MSE)": regression_loss_meter.avg})

    # Calculate classification accuracy
    class_predictions = np.array(class_predictions)
    class_actuals = np.array(class_actuals)
    accuracy = np.mean(class_predictions == class_actuals) * 100  # Accuracy in percentage

    if test_loader:
        # Flatten predictions and actuals if they are nested lists
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        perc_completion_list = np.concatenate(perc_completion_list)
        pl_exp_list = np.concatenate(pl_exp_list)
        sigma_list = np.concatenate(sigma_list)
        jtx_list = np.concatenate(jtx_list)
        num_samples_list = np.concatenate(num_samples_list)

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = math.sqrt(mse)
        print("MAE: ", mae)
        print("MSE: ", mse)
        print("loss_meter.avg: ", regression_loss_meter.avg)
        print("RMSE: ", rmse)
        print("Classification Accuracy: ", accuracy)

        err_metrics = {
            'actuals': actuals,
            'predictions': predictions,
            'perc_completion': perc_completion_list,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'classification_accuracy': accuracy
        }
        return predictions, actuals, err_metrics, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list

    return regression_loss_meter.avg, detailed_metrics


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
    ax.text(mid_point[0], mid_point[1] - 20, f'RMSE: {rmse:.2f}m', fontsize=12, color='black')

    coord_system = params['coords']
    # ax.set_title(f'Network Topology with RSSI, Noise Floor, and {coord_system} Jammer Prediction', fontsize=11)
    ax.set_title(f'Network Topology with Actual and GNN Predicted Jammer Position', fontsize=11)
    ax.set_xlabel('X position (m)', fontsize=14)
    ax.set_ylabel('Y position (m)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
