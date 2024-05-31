import gzip
import os
import pandas as pd
import numpy as np
import ast
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Tuple, List, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging
import pickle
from utils import set_random_seeds
from custom_logging import setup_logging

setup_logging()

# Set random seed for reproducibility
set_random_seeds()


def fit_and_save_scaler(data, path='data/scaler.pkl'):
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


def load_scaler(path='data/scaler.pkl'):
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


def preprocess_data(data, scaler_path='data/scaler.pkl', inference=False):
    """
   Preprocess the input data by converting lists, scaling features, and normalizing RSSI values.

   Args:
       data (pd.DataFrame): The input data containing columns to be processed.
       scaler_path (str): The path to save/load the scaler for normalization.

   Returns:
       pd.DataFrame: The preprocessed data with transformed features.
   """
    # TODO: add more graph related node features
    logging.info("Preprocessing data...")
    for col in ['drone_positions', 'states', 'drones_rssi']:
        data[col] = data[col].apply(safe_convert_list)

    if not inference:
        data['jammer_position'] = data['jammer_position'].apply(lambda x: [float(i) for i in x.strip('[]').split()])

    if not os.path.exists(scaler_path):
        scaler = fit_and_save_scaler(data, ['drone_positions', 'jammer_position'], scaler_path)
    else:
        scaler = load_scaler(scaler_path)

    data['drone_positions'] = data['drone_positions'].apply(lambda x: scaler.transform(x).tolist())
    if not inference:
        data['jammer_position'] = data['jammer_position'].apply(lambda x: scaler.transform([x])[0].tolist())

    data['centroid'] = data['drone_positions'].apply(lambda positions: np.mean(positions, axis=0))
    data['distance_to_centroid'] = data.apply(lambda row: [np.linalg.norm(np.array(pos) - row['centroid']) for pos in row['drone_positions']], axis=1)

    rssi_values = np.concatenate(data['drones_rssi'].tolist())
    min_rssi, max_rssi = rssi_values.min(), rssi_values.max()
    data['drones_rssi'] = data['drones_rssi'].apply(lambda x: [(val - min_rssi) / (max_rssi - min_rssi) for val in x])

    return data


def save_datasets(data, train_path, val_path, test_path):
    """
    Save the preprocessed data into train, validation, and test datasets.

    Args:
        data (pd.DataFrame): The preprocessed data to be split and saved.
        train_path (str): The file path to save the training dataset.
        val_path (str): The file path to save the validation dataset.
        test_path (str): The file path to save the test dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    torch_geo_dataset = [create_torch_geo_data(row) for _, row in data.iterrows()]
    train_size = int(0.6 * len(torch_geo_dataset))
    val_size = int(0.2 * len(torch_geo_dataset))
    test_size = len(torch_geo_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(torch_geo_dataset, [train_size, val_size, test_size])

    logging.info("Saving preprocessed data...")
    with gzip.open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with gzip.open(val_path, 'wb') as f:
        pickle.dump(val_dataset, f)
    with gzip.open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_dataset, val_dataset, test_dataset


def load_data(dataset_path: str, train_path: str, val_path: str, test_path: str):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.
        train_path (str): The file path of the saved training dataset.
        val_path (str): The file path of the saved validation dataset.
        test_path (str): The file path of the saved test dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    if all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        logging.info("Loading preprocessed data...")
        # Load compressed datasets
        with gzip.open(train_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with gzip.open(val_path, 'rb') as f:
            val_dataset = pickle.load(f)
        with gzip.open(test_path, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        data = pd.read_csv(dataset_path)
        data.drop(columns=['random_seed', 'num_drones', 'num_jammed_drones', 'num_rssi_vals_with_noise', 'drones_rssi_sans_noise'], inplace=True)
        data = preprocess_data(data)
        train_dataset, val_dataset, test_dataset = save_datasets(data, train_path, val_path, test_path)
    return train_dataset, val_dataset, test_dataset


def create_data_loader(train_dataset, val_dataset, test_dataset, batch_size: int):
    """
    Create data loader objects.
    Args:
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
    """
    logging.info("Creating DataLoader objects...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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


def create_torch_geo_data(row: pd.Series) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """

    # prepare node features and convert to Tensor
    node_features = [pos + [1 if state == 'jammed' else 0, rssi, dist] for pos, state, rssi, dist in
                     zip(row['drone_positions'], row['states'], row['drones_rssi'], row['distance_to_centroid'])]
    node_features = torch.tensor(node_features, dtype=torch.float32)  # Use float32 directly

    # Preparing edges and weights using KNN
    # positions = np.array(row['drone_positions'])
    # k = 5  # num of neighbors
    # nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
    # distances, indices = nbrs.kneighbors(positions)
    # edge_index, edge_weight = [], []
    #
    # for i in range(indices.shape[0]):
    #     # Add self-loop
    #     edge_index.extend([[i, i]])
    #     edge_weight.extend([0.0])
    #
    #     for j in range(1, indices.shape[1]):
    #         edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
    #         dist = distances[i, j]
    #         edge_weight.extend([dist, dist])

    # Preparing edges and weights using geographical proximity
    edge_index, edge_weight = [], []
    num_nodes = len(row['drone_positions'])

    # Add self-loops
    for i in range(num_nodes):
        edge_index.append([i, i])
        edge_weight.append(row['drones_rssi'][i])

    # Add edges based on proximity
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
