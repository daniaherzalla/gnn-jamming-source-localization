import os
import pandas as pd
import numpy as np
import ast
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Tuple, List, Any
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import logging
import pickle
from utils import set_random_seeds
from custom_logging import setup_logging

setup_logging()

# Set random seed for reproducibility
set_random_seeds()


def fit_and_save_scaler(data, path='scaler.pkl'):
    """
    Fits a MinMaxScaler to the specified features of the data and saves the scaler to a file.

    Args:
        data (DataFrame): Pandas DataFrame containing the features to scale.
        path (str): Path to save the scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Prepare the features by exploding and then vertically stacking them
    # Ensure the columns you want to explode are actually lists of lists if not, adjust preprocessing
    drone_positions = np.vstack(data['drone_positions'].explode().tolist())
    jammer_positions = np.vstack(data['jammer_position'].apply(lambda x: [x]).explode().tolist())
    combined_features = np.vstack([drone_positions, jammer_positions])
    scaler.fit(combined_features)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler


def load_scaler(path='scaler.pkl'):
    """
    Loads a MinMaxScaler from a file.

    Args:
        path (str): Path from which to load the scaler object.

    Returns:
        MinMaxScaler: The loaded scaler object.
    """
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def preprocess_data(data, scaler_path='scaler.pkl'):
    logging.info("Preprocessing data...")
    for col in ['drone_positions', 'states', 'drones_rssi']:
        data[col] = data[col].apply(safe_convert_list)

    data['jammer_position'] = data['jammer_position'].apply(lambda x: [float(i) for i in x.strip('[]').split()])

    if not os.path.exists(scaler_path):
        scaler = fit_and_save_scaler(data, ['drone_positions', 'jammer_position'], scaler_path)
    else:
        scaler = load_scaler(scaler_path)

    data['drone_positions'] = data['drone_positions'].apply(lambda x: scaler.transform(x).tolist())
    data['jammer_position'] = data['jammer_position'].apply(lambda x: scaler.transform([x])[0].tolist())

    data['centroid'] = data['drone_positions'].apply(lambda positions: np.mean(positions, axis=0))
    data['distance_to_centroid'] = data.apply(lambda row: [np.linalg.norm(np.array(pos) - row['centroid']) for pos in row['drone_positions']], axis=1)

    rssi_values = np.concatenate(data['drones_rssi'].tolist())
    min_rssi, max_rssi = rssi_values.min(), rssi_values.max()
    data['drones_rssi'] = data['drones_rssi'].apply(lambda x: [(val - min_rssi) / (max_rssi - min_rssi) for val in x])

    return data


def save_datasets(data, train_path, val_path, test_path):
    torch_geo_dataset = [create_torch_geo_data(row) for _, row in data.iterrows()]
    train_size = int(0.6 * len(torch_geo_dataset))
    val_size = int(0.2 * len(torch_geo_dataset))
    test_size = len(torch_geo_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(torch_geo_dataset, [train_size, val_size, test_size])

    logging.info("Saving preprocessed data...")
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_dataset, val_dataset, test_dataset


def load_data(dataset_path: str, train_path: str, val_path: str, test_path: str):
    if all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        logging.info("Loading preprocessed data...")
        with open(train_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_dataset = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        data = pd.read_csv(dataset_path)
        data.drop(columns=['random_seed', 'num_drones', 'num_jammed_drones', 'num_rssi_vals_with_noise', 'drones_rssi_sans_noise'], inplace=True)
        data = preprocess_data(data)
        train_dataset, val_dataset, test_dataset = save_datasets(data, train_path, val_path, test_path)
    return train_dataset, val_dataset, test_dataset


def create_data_loader(train_dataset, val_dataset, test_dataset, batch_size: int):
    """

    Args:
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
    """
    logging.info("Creating DataLoader objects...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def safe_convert_list(row: str) -> List[Any]:
    """
    Safely convert a string representation of a list to an actual list.

    Args:
        row (str): String representation of a list.

    Returns:
        List: Converted list or an empty list if conversion fails.
    """
    try:
        return ast.literal_eval(row)
    except (ValueError, SyntaxError):
        return []


# def create_torch_geo_data(row: pd.Series) -> Data:
#     """
#     Create a PyTorch Geometric data object from a row of the dataset.
#
#     Args:
#         row (pd.Series): A row of the DataFrame.
#
#     Returns:
#         Data: A PyTorch Geometric data object.
#     """
#     node_features = [np.array(pos + [1 if state == 'jammed' else 0, rssi, dist], dtype=float) for pos, state, rssi, dist in zip(row['drone_positions'], row['states'], row['drones_rssi'], row['distance_to_centroid'])]
#     node_features = np.stack(node_features)
#     node_features = torch.tensor(node_features, dtype=torch.float)
#
#     edge_index = []
#     edge_weight = []
#     num_drones = len(row['drone_positions'])
#     for i in range(num_drones):
#         for j in range(i + 1, num_drones):
#             dist = np.linalg.norm(np.array(row['drone_positions'][i]) - np.array(row['drone_positions'][j]))
#             proximity_threshold = 10
#             if dist < proximity_threshold:
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])
#                 edge_weight.append((row['drones_rssi'][i] + row['drones_rssi'][j]) / 2)
#                 edge_weight.append((row['drones_rssi'][i] + row['drones_rssi'][j]) / 2)
#
#     if not edge_index:  # Check if the edge_index list is still empty
#         edge_index = [[i, i] for i in range(num_drones)]  # Add a self-loop to each node
#         edge_weight = [1.0 for _ in range(num_drones)]  # Use a default edge weight of 1
#
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
#     y = torch.tensor(row['jammer_position'], dtype=torch.float).unsqueeze(0)
#     return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)


# def create_torch_geo_data(row: pd.Series) -> Data:
#     """
#     Create a PyTorch Geometric data object from a row of the dataset.
#
#     Args:
#         row (pd.Series): A row of the DataFrame.
#
#     Returns:
#         Data: A PyTorch Geometric data object.
#     """
#     node_features = [np.array(pos + [1 if state == 'jammed' else 0, rssi, dist], dtype=float) for pos, state, rssi, dist in zip(row['drone_positions'], row['states'], row['drones_rssi'], row['distance_to_centroid'])]
#     node_features = torch.tensor(np.stack(node_features), dtype=torch.float)
#
#     edge_index = []
#     edge_weight = []
#     num_drones = len(row['drone_positions'])
#     for i in range(num_drones):
#         for j in range(i + 1, num_drones):
#             dist = np.linalg.norm(np.array(row['drone_positions'][i]) - np.array(row['drone_positions'][j]))
#             if dist < 10:  # Proximity threshold
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])
#                 edge_weight.extend([(row['drones_rssi'][i] + row['drones_rssi'][j]) / 2] * 2)
#
#     if not edge_index:  # Ensuring at least self-loops if no edges exist
#         edge_index = [[i, i] for i in range(num_drones)]
#         edge_weight = [1.0] * num_drones
#
#     data = Data(
#         x=node_features,
#         edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
#         edge_attr=torch.tensor(edge_weight, dtype=torch.float),
#         y=torch.tensor(row['jammer_position'], dtype=torch.float).unsqueeze(0)
#     )
#     return data

def create_torch_geo_data(row: pd.Series) -> Data:
    # Efficiently prepare node features and convert to Tensor
    node_features = [pos + [1 if state == 'jammed' else 0, rssi, dist] for pos, state, rssi, dist in zip(row['drone_positions'], row['states'], row['drones_rssi'], row['distance_to_centroid'])]
    node_features = torch.tensor(node_features, dtype=torch.float32)  # Use float32 directly

    # Preparing edges and weights
    edge_index, edge_weight = [], []
    num_nodes = len(row['drone_positions'])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(np.array(row['drone_positions'][i]) - np.array(row['drone_positions'][j]))
            if dist < 0.1:  # Proximity threshold
                edge_index.extend([[i, j], [j, i]])
                weight = (row['drones_rssi'][i] + row['drones_rssi'][j]) / 2
                edge_weight.extend([weight, weight])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight else torch.empty(0, dtype=torch.float)

    # Target variable preparation
    y = torch.tensor(row['jammer_position'], dtype=torch.float).unsqueeze(0)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)
