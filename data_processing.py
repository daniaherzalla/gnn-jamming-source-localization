import gzip
import json
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


def fit_and_save_scaler(data, feature, path='data/scaler.pkl'):
    """
    Fits a MinMaxScaler to the specified features of the data and saves the scaler to a file.

    Args:
        data (DataFrame): Pandas DataFrame containing the features to scale.
        feature (str): The feature type ('rssi' or 'coords') to apply scaling to.
        path (str): Path to save the scaler object.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    if feature == 'rssi':
        # Extract all RSSI values from 'drones_rssi'
        all_rssi = np.concatenate(data['drones_rssi'].values)
        # Fit the scaler to the RSSI values, reshaping for compatibility
        scaler.fit(all_rssi.reshape(-1, 1))
    elif feature == 'coords':
        if params['feats'] == 'polar':
            # Extract radius values from 'polar_coordinates'
            drone_radii = np.array([pos[0] for sublist in data['polar_coordinates'] for pos in sublist])
            jammer_radii = np.array([pos[0] for sublist in data['jammer_position'] for pos in sublist])
            all_radii = np.concatenate([drone_radii, jammer_radii])
            # Fit the scaler to the radius values, reshaping for compatibility
            scaler.fit(all_radii.reshape(-1, 1))
        else:
            drone_positions = np.vstack(data['drone_positions'].explode().tolist())
            jammer_positions = np.vstack(data['jammer_position'])
            all_positions = np.vstack([drone_positions, jammer_positions])
            scaler.fit(all_positions)

    # Save the scaler object to a file
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

    return scaler


def load_scaler(path):
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


def angle_to_cyclical(positions):
    """Convert a list of positions from polar to cyclical coordinates."""
    transformed_positions = []
    for position in positions:
        r, theta, phi = position
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        transformed_positions.append([r, sin_theta, cos_theta, sin_phi, cos_phi])
    return transformed_positions


def cyclical_to_angular(output):
    """
    Convert cyclical coordinates (sin and cos) back to angular coordinates (theta and phi).

    Args:
        output (list): List of lists containing [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.

    Returns:
        list: Updated list with [r, theta, phi] for each point.
    """
    result = []
    # print("output: ", output)
    for point in output:
        r = point[0]
        sin_theta = point[1]
        cos_theta = point[2]
        sin_phi = point[3]
        cos_phi = point[4]

        # Compute the original angles using arctan2
        theta = np.arctan2(sin_theta, cos_theta)
        phi = np.arctan2(sin_phi, cos_phi)

        # Append the original coordinates [r, theta, phi]
        result.append([r, theta, phi])
    return result


def standardize_values(values):
    """Standardize values using z-score normalization."""
    # values = np.array(values)  # Convert values to numpy array
    if len(values) == 0:
        raise ValueError("Cannot calculate standard deviation for an empty array")
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    standardized_values = (values - mean) / std
    return standardized_values


def reverse_standardization(standardized_values, mean, std):
    """Reverse standardization of values using stored mean and standard deviation."""
    original_values = standardized_values * std + mean
    return original_values


def center_coordinates(coords):
    """Center coordinates within a specified bounding box."""

    # Compute the lower and upper bounds for the given coordinates
    lower_bound = np.min(coords, axis=0)
    upper_bound = np.max(coords, axis=0)

    midpoint = (lower_bound + upper_bound) / 2
    centered_coords = coords - midpoint

    data_to_save = {
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist()
    }

    # Write the dictionary to a JSON file
    with open('centering_vars.json', 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    return centered_coords


def center_and_convert_coordinates(data):
    """Center and convert drone positions."""
    logging.info("Centering coords")
    data['drone_positions'] = data['drone_positions'].apply(center_coordinates)
    data['jammer_position'] = data['jammer_position'].apply(center_coordinates)
    if params['feats'] == 'polar':
        data['polar_coordinates'] = data['drone_positions'].apply(cartesian_to_polar)
        data['polar_coordinates'] = data['polar_coordinates'].apply(angle_to_cyclical)

        data['jammer_position'] = data['jammer_position'].apply(cartesian_to_polar)
        data['jammer_position'] = data['jammer_position'].apply(angle_to_cyclical)


def standardize_data(data):
    """Apply z-score or min-max normalization based on the feature type."""
    if params['norm'] == 'zscore':
        apply_z_score_normalization(data)
    elif params['norm'] == 'minmax':
        apply_min_max_normalization(data)


def save_mean_std_json(data, file_path='zscore_mean_std.json'):
    # Write the dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Data saved to JSON successfully.")


def apply_z_score_normalization(data):
    """Apply Z-score normalization."""
    logging.info("Applying z-score normalization")

    # Drone Positions
    if params['feats'] == 'polar':
        all_radii = []
        drones_radii = []
        jammer_radii = []
        for positions in data['polar_coordinates']:
            for position in positions:
                drones_radii.append(position[0])  # Extract all radii

        for positions in data['jammer_position']:
            for position in positions:
                jammer_radii.append(position[0])

        all_radii = drones_radii + jammer_radii

        # Compute mean and standard deviation for each coordinate (x, y, z) separately
        radii_mean = np.mean(all_radii, axis=0)
        radii_std = np.std(all_radii, axis=0)

        standardized_drones_radii = (np.array(drones_radii) - radii_mean) / radii_std
        standardized_jammer_radii = (np.array(jammer_radii) - radii_mean) / radii_std

        # Structure the data into a dictionary
        data_to_save = {
            "radii_means": radii_mean.tolist(),
            "radii_stds": radii_std.tolist()
        }
        save_mean_std_json(data_to_save)

        # Replace original radii with standardized ones for drones
        radius_index = 0
        for positions in data['polar_coordinates']:
            for position in positions:
                position[0] = standardized_drones_radii[radius_index]
                radius_index += 1

        # Replace original radii with standardized ones for jammer
        radius_index = 0
        for positions in data['jammer_position']:
            for position in positions:
                position[0] = standardized_jammer_radii[radius_index]
                radius_index += 1
    else:
        # Concatenate all drone positions from all scenarios
        all_drone_positions = np.concatenate(data['drone_positions'].tolist())
        all_jammer_positions = np.array(data['jammer_position'].tolist())
        all_positions = np.vstack([all_drone_positions, all_jammer_positions])

        # Compute mean and standard deviation for each coordinate (x, y, z) separately
        position_means = np.mean(all_positions, axis=0)
        position_stds = np.std(all_positions, axis=0)

        # Structure the data into a dictionary
        data_to_save = {
            "position_means": position_means.tolist(),
            "position_stds": position_stds.tolist()
        }
        save_mean_std_json(data_to_save)

        # Standardize the drone positions for each scenario
        standardized_positions = []

        for positions_scenario in data['drone_positions']:
            standardized_positions_scenario = []
            for position in positions_scenario:
                standardized_position = (np.array(position) - position_means) / position_stds
                standardized_positions_scenario.append(standardized_position)
            standardized_positions.append(standardized_positions_scenario)
        data['drone_positions'] = standardized_positions  # Replace the original drone positions in each row with the standardized values

        # Standardize the jammer positions for each scenario
        standardized_positions = []

        for position in data['jammer_position']:
            standardized_positions_scenario = []
            standardized_position = (np.array(position) - position_means) / position_stds
            standardized_positions_scenario.append(standardized_position)
            standardized_positions.append(standardized_positions_scenario)
        data['jammer_position'] = standardized_positions

    # Normalizing RSSI
    all_rssi_values = np.concatenate(data['drones_rssi'].tolist())

    # Compute mean and standard deviation using all RSSI values
    rssi_mean = np.mean(all_rssi_values)
    rssi_std = np.std(all_rssi_values)

    # Standardize the RSSI values using the mean and standard deviation computed from all scenarios combined
    standardized_rssi_values = []

    for rssi_scenario in data['drones_rssi']:
        standardized_rssi_scenario = (np.array(rssi_scenario) - rssi_mean) / rssi_std
        standardized_rssi_values.append(standardized_rssi_scenario)

    # Replace the original RSSI values in each row of the DataFrame with the standardized values
    data['drones_rssi'] = standardized_rssi_values


def apply_min_max_normalization(data):
    """Apply Min-Max normalization."""
    logging.info("Fitting min-max scaler")
    coords_scaler = fit_and_save_scaler(data, 'coords', 'data/coords_scaler.pkl')
    if params['feats'] == 'polar':
        data['polar_coordinates'] = data['polar_coordinates'].apply(lambda x: [[coords_scaler.transform([[pos[0]]])[0][0]] + pos[1:] for pos in x])
        # Normalize jammer_position using the same scaler
        data['jammer_position'] = data['jammer_position'].apply(lambda x: [[coords_scaler.transform([[pos[0]]])[0][0]] + pos[1:] for pos in x])
        # data['jammer_position'] = data['jammer_position'].apply(lambda x: [coords_scaler.transform([[x[0]]])[0][0]] + x[1:])
    else:
        data['drone_positions'] = data['drone_positions'].apply(lambda x: coords_scaler.transform(x).tolist())
        # Normalize jammer_position using the same scaler
        data['jammer_position'] = data['jammer_position'].apply(lambda x: coords_scaler.transform(np.array(x).reshape(1, -1)).tolist())

    # Apply normalization to RSSI values
    rssi_scaler = fit_and_save_scaler(data, 'rssi', 'data/rssi_scaler.pkl')
    data['drones_rssi'] = [np.squeeze(rssi_scaler.transform(np.array(rssi).reshape(-1, 1))).tolist() for rssi in data['drones_rssi']]


def convert_data_type(data):
    # Convert from str to required data type
    data['drone_positions'] = data['drone_positions'].apply(lambda x: safe_convert_list(x, 'drones_pos'))
    data['jammer_position'] = data['jammer_position'].apply(lambda x: safe_convert_list(x, 'jammer_pos'))
    data['drones_rssi'] = data['drones_rssi'].apply(lambda x: safe_convert_list(x, 'drones_rssi'))


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

    # Conversion from string to list type
    convert_data_type(data)
    center_and_convert_coordinates(data)
    standardize_data(data)
    return data


def polar_to_cartesian(data):
    """
    Convert polar coordinates back to Cartesian coordinates.

    Args:
        data (list): List of lists containing [r, theta, phi] for each point.

    Returns:
        list: Updated list with [x, y, z] for each point.
    """
    result = []
    for point in data:
        r = point[0]
        theta = point[1]
        phi = point[2]

        # Compute the Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        # Append the Cartesian coordinates [x, y, z]
        result.append([x, y, z])
    return result


def undo_center_coordinates(centered_coords, lower_bound, upper_bound):
    """Revert the centering of coordinates to their original location."""
    # print("centered_coords: ", centered_coords)
    midpoint = (lower_bound + upper_bound) / 2
    # print("midpoint: ", midpoint)
    original_coords = [coords + midpoint for coords in centered_coords]
    return original_coords


def convert_output(output, device):
    with open('centering_vars.json', 'r') as json_file:
        data_loaded = json.load(json_file)
        lower_bound = np.array(data_loaded['lower_bound'])
        upper_bound = np.array(data_loaded['upper_bound'])

    if params['feats'] == 'cartesian':
        if params['norm'] == 'zscore':
            with open('zscore_mean_std.json', 'r') as json_file:
                data_loaded = json.load(json_file)
                position_means = np.array(data_loaded['position_means'])
                position_stds = np.array(data_loaded['position_stds'])

            converted_output = output * position_stds + position_means
            converted_output = undo_center_coordinates(converted_output, lower_bound, upper_bound)

        elif params['norm'] == 'minmax':
            scaler = load_scaler('data/coords_scaler.pkl')
            converted_output = scaler.inverse_transform(output)
            converted_output = undo_center_coordinates(converted_output, lower_bound, upper_bound)

    elif params['feats'] == 'polar':
        polar_coords = cyclical_to_angular(output)
        if params['norm'] == 'zscore':
            with open('zscore_mean_std.json', 'r') as json_file:
                data_loaded = json.load(json_file)
                radii_mean = np.array(data_loaded['radii_means'])
                radii_std = np.array(data_loaded['radii_stds'])
            print("position_means: ", radii_mean)
            print("position_std: ", radii_std)
            polar_coords[0] = polar_coords[0] * radii_std + radii_mean
            converted_output = polar_to_cartesian(polar_coords)
            converted_output = undo_center_coordinates(converted_output, lower_bound, upper_bound)

        elif params['norm'] == 'minmax':
            scaler = load_scaler('data/coords_scaler.pkl')
            # Convert list to numpy array and reshape
            radii_array = np.array(polar_coords[0]).reshape(-1, 1)
            polar_coords[0] = scaler.inverse_transform(radii_array)
            converted_output = polar_to_cartesian(polar_coords)
            converted_output = undo_center_coordinates(converted_output, lower_bound, upper_bound)

    return converted_output


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
    logging.info('Creating edges')
    torch_geo_dataset = [create_torch_geo_data(row) for _, row in data.iterrows()]

    # Shuffle the dataset
    random.shuffle(torch_geo_dataset)

    logging.info('Creating train-test split')
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

    print("train_dataset[0]: ", train_dataset[0])

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
    # logging.info(f"row['drone_positions']: {row['drone_positions'][0]}")
    # logging.info(f"row['drones_rssi']: {row['drones_rssi'][0]}")
    # logging.info(f"row['jammer_position']: {row['jammer_position'][0]}")
    # quit()

    if params['feats'] == 'polar':
        node_features = [list(pos) + [rssi] for pos, rssi in zip(row['polar_coordinates'], row['drones_rssi'])]
    elif params['feats'] == 'cartesian':
        node_features = [list(pos) + [rssi] for pos, rssi in zip(row['drone_positions'], row['drones_rssi'])]
    else:
        raise ValueError

    # print("node features: ", node_features)
    # quit()
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

        # # Get distance of 1 m normalized
        # scaler = load_scaler()
        # proximity_threshold = np.array([[1.0, 1.0, 1.0]])  # Distance of 1 meter
        # normalized_prox_threshold = scaler.transform(proximity_threshold)
        # print("normalized_prox_threshold: ", normalized_prox_threshold)
        # quit()

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
    # print("row['jammer_position']: ", row['jammer_position'])
    # quit()
    y = torch.tensor(row['jammer_position'][0], dtype=torch.float).unsqueeze(0)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)
