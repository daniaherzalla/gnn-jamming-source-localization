import gzip
import os
import pandas as pd
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Tuple, List, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging
import pickle
from utils import set_seeds_and_reproducibility, cartesian_to_polar
from custom_logging import setup_logging
from config import params

setup_logging()

set_seeds_and_reproducibility()


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


# def preprocess_data(data, inference, scaler_path='data/scaler.pkl'):
#     """
#    Preprocess the input data by converting lists, scaling features, and normalizing RSSI values.
#
#    Args:
#        inference (bool): Performing hyperparameter tuning or inference
#        data (pd.DataFrame): The input data containing columns to be processed.
#        scaler_path (str): The path to save/load the scaler for normalization.
#
#    Returns:
#        pd.DataFrame: The preprocessed data with transformed features.
#    """
#     logging.info("Preprocessing data...")
#
#     # Apply the function to each column with its specific data type
#     data['drone_positions'] = data['drone_positions'].apply(lambda x: safe_convert_list(x, 'drones_pos'))
#     data['jammer_position'] = data['jammer_position'].apply(lambda x: safe_convert_list(x, 'jammer_pos'))
#     data['states'] = data['states'].apply(lambda x: safe_convert_list(x, 'states'))
#     data['drones_rssi'] = data['drones_rssi'].apply(lambda x: safe_convert_list(x, 'drones_rssi'))
#
#     logging.info("Fitting scaler")
#     if not os.path.exists(scaler_path):
#         scaler = fit_and_save_scaler(data, scaler_path)
#     else:
#         scaler = load_scaler(scaler_path)
#
#     # Apply scaler
#     if not inference:
#         data['jammer_position'] = data['jammer_position'].apply(lambda x: scaler.transform([x])[0].tolist())
#     data['drone_positions'] = data['drone_positions'].apply(lambda x: scaler.transform(x).tolist())
#
#     logging.info("Calculating node features")
#     # Calculate centroid and other features
#     data['centroid'] = data['drone_positions'].apply(lambda positions: np.mean(positions, axis=0))
#     data['distance_to_centroid'] = data.apply(lambda row: [np.linalg.norm(pos - row['centroid']) for pos in row['drone_positions']], axis=1)
#
#     # Including 3D angle calculations for azimuth and elevation
#     data['azimuth_angle'] = data.apply(lambda row: [np.arctan2(pos[1] - row['centroid'][1], pos[0] - row['centroid'][0]) for pos in row['drone_positions']], axis=1)
#     data['elevation_angle'] = data.apply(lambda row: [np.arcsin((pos[2] - row['centroid'][2]) / np.linalg.norm(pos - row['centroid'])) for pos in row['drone_positions']], axis=1)
#
#     # Sample connectivity
#     data['sample_connectivity'] = data['drone_positions'].apply(lambda positions: euclidean_distances(positions, positions))
#
#     # Normalizing RSSI values
#     rssi_values = np.concatenate(data['drones_rssi'].tolist())
#     min_rssi, max_rssi = rssi_values.min(), rssi_values.max()
#     data['drones_rssi'] = data['drones_rssi'].apply(lambda x: [(val - min_rssi) / (max_rssi - min_rssi) for val in x])
#
#     # Relative RSSI calculation
#     data['relative_rssi'] = data.apply(lambda row: [rssi - np.mean(row['drones_rssi']) for rssi in row['drones_rssi']], axis=1)
#
#     return data


def preprocess_data(data, inference, scaler_path='data/scaler.pkl'):
    """
   Preprocess the input data by converting lists, scaling features, and normalizing RSSI values.

   Args:
       inference (bool): Performing hyperparameter tuning or inference
       data (pd.DataFrame): The input data containing columns to be processed.
       scaler_path (str): The path to save/load the scaler for normalization.

   Returns:
       pd.DataFrame: The preprocessed data with transformed features.
   """
    logging.info("Preprocessing data...")

    # Apply the function to each column with its specific data type
    data['drone_positions'] = data['drone_positions'].apply(lambda x: safe_convert_list(x, 'drones_pos'))
    data['jammer_position'] = data['jammer_position'].apply(lambda x: safe_convert_list(x, 'jammer_pos'))
    data['drones_rssi'] = data['drones_rssi'].apply(lambda x: safe_convert_list(x, 'drones_rssi'))

    logging.info("Fitting scaler")
    if not os.path.exists(scaler_path):
        scaler = fit_and_save_scaler(data, scaler_path)
    else:
        scaler = load_scaler(scaler_path)

    # Apply scaler
    if not inference:
        data['jammer_position'] = data['jammer_position'].apply(lambda x: scaler.transform([x])[0].tolist())
    data['drone_positions'] = data['drone_positions'].apply(lambda x: scaler.transform(x).tolist())

    # Convert drone positions from Cartesian to polar coordinates
    data['polar_coordinates'] = data['drone_positions'].apply(cartesian_to_polar)

    # Normalizing RSSI values
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

    # Shuffle the dataset
    random.seed(params['seed'])
    random.shuffle(torch_geo_dataset)

    train_size = int(0.7 * len(torch_geo_dataset))
    val_size = int(0.1 * len(torch_geo_dataset))
    test_size = len(torch_geo_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(torch_geo_dataset, [train_size, val_size, test_size])

    # logging.info("Saving preprocessed data...")
    # with gzip.open(train_path, 'wb') as f:
    #     pickle.dump(train_dataset, f)
    # with gzip.open(val_path, 'wb') as f:
    #     pickle.dump(val_dataset, f)
    # with gzip.open(test_path, 'wb') as f:
    #     pickle.dump(test_dataset, f)

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
    # if all(os.path.exists(path) for path in [train_path, val_path, test_path]):
    #     logging.info("Loading preprocessed data...")
    #     # Load compressed datasets
    #     with gzip.open(train_path, 'rb') as f:
    #         train_dataset = pickle.load(f)
    #     with gzip.open(val_path, 'rb') as f:
    #         val_dataset = pickle.load(f)
    #     with gzip.open(test_path, 'rb') as f:
    #         test_dataset = pickle.load(f)
    # else:
    data = pd.read_csv(dataset_path)
    data.drop(columns=['random_seed', 'num_drones', 'num_jammed_drones', 'num_rssi_vals_with_noise', 'drones_rssi_sans_noise', 'jammer_type', 'jammer_power', 'pl_exp', 'sigma'], inplace=True)
    data = preprocess_data(data, inference=params['inference'])
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


def safe_convert_list(row: str, data_type: str) -> List[Any]:
    """
    Safely convert a string representation of a list to an actual list,
    with type conversion tailored to specific data types including handling
    for 'states' which are extracted and stripped of surrounding quotes.

    Args:
        row (str): String representation of a list.
        data_type (str): The type of data to convert ('jammer_pos', 'drones_pos', 'drones_rssi', 'states').

    Returns:
        List: Converted list or an empty list if conversion fails.
    """
    try:
        if data_type == 'jammer_pos':
            result = row.strip('[').strip(']').split()
            return [float(pos) for pos in result]
        elif data_type == 'drones_pos':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        elif data_type == 'drones_rssi':
            result = row.strip('[').strip(']').split(', ')
            return [float(rssi) for rssi in result]
        elif data_type == 'states':
            result = row.strip('][').split(', ')
            return [state.strip("'") for state in result]
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error


def create_torch_geo_data(row: pd.Series) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """

    # prepare node features and convert to Tensor
    # node_features = [
    #     pos + [1 if state == 'jammed' else 0, rssi, dist, azi, ele, rel_rssi]
    #     for pos, state, rssi, dist, azi, ele, rel_rssi in
    #     zip(
    #         row['drone_positions'],
    #         row['states'],
    #         row['drones_rssi'],
    #         row['distance_to_centroid'],
    #         row['azimuth_angle'],
    #         row['elevation_angle'],
    #         row['relative_rssi']
    #     )
    # ]
    if params['feats'] == 'polar':
        node_features = [pos + [rssi] for pos, rssi in zip(row['polar_coordinates'], row['drones_rssi'])]
    elif params['feats'] == 'cartesian':
        node_features = [pos + [rssi] for pos, rssi in zip(row['drone_positions'], row['drones_rssi'])]
    elif params['feats'] == 'polar_cartesian':
        node_features = [pos_cartesian + pos_polar + [rssi] for pos_cartesian, pos_polar, rssi in zip(row['drone_positions'], row['polar_coordinates'], row['drones_rssi'])]
    else:
        raise ValueError

    node_features = torch.tensor(node_features, dtype=torch.float32)  # Use float32 directly

    # Preparing edges and weights using KNN
    if params['edges'] == 'knn':
        positions = np.array(row['drone_positions'])
        num_samples = positions.shape[0]
        k = min(5, num_samples - 1)  # num of neighbors, ensuring k < num_samples
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        edge_index, edge_weight = [], []

        for i in range(indices.shape[0]):
            # Add self-loop
            edge_index.extend([[i, i]])
            edge_weight.extend([0.0])

            for j in range(1, indices.shape[1]):
                edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                dist = distances[i, j]
                edge_weight.extend([dist, dist])

    elif params['edges'] == 'proximity':
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
                    # weight = (row['drones_rssi'][i] + row['drones_rssi'][j]) / 2
                    # edge_weight.extend([weight, weight])
                    edge_weight.extend([dist, dist])
    else:
        raise ValueError

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight else torch.empty(0, dtype=torch.float)

    # Target variable preparation
    y = torch.tensor(row['jammer_position'], dtype=torch.float).unsqueeze(0)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)
