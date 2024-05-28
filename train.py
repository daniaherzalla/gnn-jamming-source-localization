import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import GraphAttentionNetwork
from typing import Tuple
from sklearn.metrics import mean_squared_error
import logging
from custom_logging import setup_logging
from utils import set_random_seeds

set_random_seeds()

# Setup custom logging
setup_logging()


def initialize_model(device: torch.device, params: dict) -> Tuple[GraphAttentionNetwork, optim.Optimizer, ReduceLROnPlateau, torch.nn.Module]:
    """
    Initialize the model, optimizer, scheduler, and loss criterion.

    Args:
        device (torch.device): Device to run the model on.
        params (dict): Dictionary of model parameters.

    Returns:
        model (GraphAttentionNetwork): Initialized model.
        optimizer (optim.Optimizer): Optimizer for the model.
        scheduler (ReduceLROnPlateau): Learning rate scheduler.
        criterion (torch.nn.Module): Loss criterion.
    """
    logging.info("Initializing model...")
    model = GraphAttentionNetwork(params["dropout_rate"], params["num_heads"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=params['patience'], verbose=True)
    criterion = torch.nn.MSELoss()
    return model, optimizer, scheduler, criterion


def train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device) -> float:
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
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += data.num_graphs * loss.item()
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
            loss = criterion(output, data.y)
            total_loss += data.num_graphs * loss.item()
    return total_loss / len(validate_loader.dataset)


def predict_and_evaluate(model, loader, device, scaler):
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
            data = data.to(device)
            output = model(data)
            # Apply inverse transformation to the model output (denormalize)
            predicted_coords = scaler.inverse_transform(output.cpu().numpy())
            actual_coords = scaler.inverse_transform(data.y.cpu().numpy())

            predictions.extend(predicted_coords)
            actuals.extend(actual_coords)

    # calculate metrics MSE, RMSE using predictions and actuals
    mse = mean_squared_error(actuals, predictions)
    print(f'Mean Squared Error: {mse}')
    rmse = math.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')
    return predictions, actuals
