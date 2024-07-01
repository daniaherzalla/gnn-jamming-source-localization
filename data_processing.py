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

        # Define the theoretical min and max RSSI values
        min_rssi = -100  # Adjust this value as needed
        max_rssi = -30  # Adjust this value as needed

        # Normalize the RSSI values
        normalized_rssi = (all_rssi - min_rssi) / (max_rssi - min_rssi)

        # Ensure the values are between 0 and 1
        normalized_rssi = np.clip(normalized_rssi, 0, 1)

        # # Fit the scaler to the RSSI values, reshaping for compatibility
        # scaler.fit(all_rssi.reshape(-1, 1))
    elif feature == 'coords':
        # if params['feats'] == 'polar':
        #     # Extract radius values from 'polar_coordinates'
        #     drone_radii = np.array([pos[0] for sublist in data['polar_coordinates'] for pos in sublist])
        #     jammer_radii = np.array([pos[0] for sublist in data['jammer_position'] for pos in sublist])
        #     all_radii = np.concatenate([drone_radii, jammer_radii])
        #     # Fit the scaler to the radius values, reshaping for compatibility
        #     scaler.fit(all_radii.reshape(-1, 1))
        # else:
        node_positions = np.vstack(data['node_positions'].explode().tolist())
        # jammer_positions = np.vstack(data['jammer_position'])
        # all_positions = np.vstack([node_positions, jammer_positions])
        scaler.fit(node_positions)

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
    """
    Convert a list of positions from polar to cyclical coordinates.

    Args:
        positions (list): List of polar coordinates [r, theta, phi] for each point.
                          r is the radial distance,
                          theta is the polar angle from the positive z-axis (colatitude),
                          phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        list: List of cyclical coordinates [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
    """
    transformed_positions = []
    for position in positions:
        r, theta, phi = position
        sin_theta = np.sin(theta)  # Sine of the polar angle
        cos_theta = np.cos(theta)  # Cosine of the polar angle
        sin_phi = np.sin(phi)  # Sine of the azimuthal angle
        cos_phi = np.cos(phi)  # Cosine of the azimuthal angle
        transformed_positions.append([r, sin_theta, cos_theta, sin_phi, cos_phi])
    return transformed_positions


# def cyclical_to_angular(output):
#     """
#     Convert cyclical coordinates (sin and cos) back to angular coordinates (theta and phi).
#
#     Args:
#         output (list): List of lists containing [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
#
#     Returns:
#         list: Updated list with [r, theta, phi] for each point.
#     """
#     result = []
#     for point in output:
#         r = point[0]
#         sin_theta = point[1]
#         cos_theta = point[2]
#         sin_phi = point[3]
#         cos_phi = point[4]
#
#         # Compute the original angles using arctan2
#         theta = np.arctan2(sin_theta, cos_theta)
#         phi = np.arctan2(sin_phi, cos_phi)
#
#         # Append the original coordinates [r, theta, phi]
#         result.append([r, theta, phi])
#     return result


def cyclical_to_angular(output):
    """
    Convert cyclical coordinates (sin and cos) back to angular coordinates (theta and phi) using PyTorch.

    Args:
        output (Tensor): Tensor containing [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
                         r is the radial distance,
                         theta is the polar angle from the positive z-axis (colatitude),
                         phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        Tensor: Updated tensor with [r, theta, phi] for each point.
    """
    print('output: ', output)
    print('r: ', output[:, 0])
    r = output[:, 0]
    sin_theta = output[:, 1]
    cos_theta = output[:, 2]
    sin_phi = output[:, 3]
    cos_phi = output[:, 4]

    theta = torch.atan2(sin_theta, cos_theta)  # Polar angle calculation from sin and cos
    phi = torch.atan2(sin_phi, cos_phi)  # Azimuthal angle calculation from sin and cos

    return torch.stack([r, theta, phi], dim=1)


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


def center_coordinates(coords, center=None):
    """Center coordinates using given or calculated bounds."""
    # Calculate the geometric center
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    # print("cetnered coords: ", centered_coords)
    return centered_coords, center


def center_and_convert_coordinates(data):
    """Center and convert drone and jammer positions using shared bounds for each row, and save midpoints."""
    logging.info("Centering coordinates")

    midpoints = {}
    # Initialize columns for storing min and max coordinates
    data['min_x'] = np.nan
    data['max_x'] = np.nan
    data['min_y'] = np.nan
    data['max_y'] = np.nan
    data['min_z'] = np.nan
    data['max_z'] = np.nan

    data['drones_pos_center'] = None

    for idx, row in data.iterrows():
        node_positions = np.vstack(row['node_positions'])

        # Calculate bounds and midpoints
        _, center = center_coordinates(node_positions)
        midpoints[row['id']] = center.tolist()  # Save center for this index

        # Save min and max coordinates for x, y, and z
        data.at[idx, 'min_x'] = np.min(node_positions[:, 0])
        data.at[idx, 'max_x'] = np.max(node_positions[:, 0])
        data.at[idx, 'min_y'] = np.min(node_positions[:, 1])
        data.at[idx, 'max_y'] = np.max(node_positions[:, 1])
        data.at[idx, 'min_z'] = np.min(node_positions[:, 2])
        data.at[idx, 'max_z'] = np.max(node_positions[:, 2])

        # Center coordinates using the calculated midpoint
        centered_node_positions = node_positions - center
        data.at[idx, 'node_positions'] = centered_node_positions.tolist()

        # Optionally, you can store the center in the DataFrame as well
        # print("center: ", center)
        data.at[idx, 'drones_pos_center'] = center

        # # Similar centering for jammer position
        jammer_pos = np.array(row['jammer_position'])
        centered_jammer_position = jammer_pos - center
        data.at[idx, 'jammer_position'] = centered_jammer_position.tolist()

    # Save midpoints to a JSON file for later use during uncentering
    with open('midpoints.json', 'w') as f:
        json.dump(midpoints, f)

    if params['feats'] == 'polar':
        data['polar_coordinates'] = data['node_positions'].apply(cartesian_to_polar)
        # # Extract all 'r' values from 'polar_coordinates'
        # all_r_values = []
        # for polar_coords in data['polar_coordinates']:
        #     r_values = [coord[0] for coord in polar_coords]  # Extract 'r' from each [r, theta, phi]
        #     all_r_values.extend(r_values)
        #
        # # Find the maximum 'r' value
        # r_max_bound = max(all_r_values)
        # print("r_max_bound: ", r_max_bound)
        # quit()
        # data['polar_coordinates'] = data['polar_coordinates'].apply(angle_to_cyclical)

    return data

    # data['jammer_position'] = data['jammer_position'].apply(cartesian_to_polar) # !! keep in normalized centered cartesian form
        # data['jammer_position'] = data['jammer_position'].apply(angle_to_cyclical)


def standardize_data(data):
    """Apply z-score or min-max normalization based on the feature type."""
    if params['norm'] == 'zscore':
        apply_z_score_normalization(data)
    elif params['norm'] == 'minmax':
        # print("Range before scaling:", np.min(data['jammer_position']), np.max(data['jammer_position']))
        apply_min_max_normalization(data)
        # After applying the scaler
        # print("Range after scaling:", np.min(data['jammer_position']), np.max(data['jammer_position']))


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
        all_node_positions = np.concatenate(data['node_positions'].tolist())
        all_jammer_positions = np.array(data['jammer_position'].tolist())
        all_positions = np.vstack([all_node_positions, all_jammer_positions])

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

        for positions_scenario in data['node_positions']:
            standardized_positions_scenario = []
            for position in positions_scenario:
                standardized_position = (np.array(position) - position_means) / position_stds
                standardized_positions_scenario.append(standardized_position)
            standardized_positions.append(standardized_positions_scenario)
        data['node_positions'] = standardized_positions  # Replace the original drone positions in each row with the standardized values

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
    experiment_path = 'experiments/' + params['feats'] + '_' + params['edges'] + '_' + params['norm'] + '/' + 'trial' + str(params['trial_num'])
    coords_scaler = fit_and_save_scaler(data, 'coords', f'{experiment_path}/coords_scaler.pkl')
    if params['feats'] == 'polar':
        data['polar_coordinates'] = data['polar_coordinates'].apply(lambda x: [[coords_scaler.transform([[pos[0]]])[0][0]] + pos[1:] for pos in x])
        # Normalize jammer_position using the same scaler
        data['jammer_position'] = data['jammer_position'].apply(lambda x: [[coords_scaler.transform([[pos[0]]])[0][0]] + pos[1:] for pos in x])
        # data['jammer_position'] = data['jammer_position'].apply(lambda x: [coords_scaler.transform([[x[0]]])[0][0]] + x[1:])
    # else:
    data['node_positions'] = data['node_positions'].apply(lambda x: coords_scaler.transform(x).tolist())
    # Normalize jammer_position using the same scaler
    # print("data['jammer_position']: ", data['jammer_position'])
    data['jammer_position'] = data['jammer_position'].apply(lambda x: coords_scaler.transform(np.array(x).reshape(1, -1)).tolist())

    # Apply normalization to RSSI values
    # rssi_scaler = fit_and_save_scaler(data, 'rssi', f'{experiment_path}/rssi_scaler.pkl')
    # data['drones_rssi'] = [np.squeeze(rssi_scaler.transform(np.array(rssi).reshape(-1, 1))).tolist() for rssi in data['drones_rssi']]
    # Extract all RSSI values from 'drones_rssi'
    # all_rssi = np.concatenate(data['drones_rssi'].values)

    # Theoretical RSSI range
    rssi_min = -100  # Theoretical minimum RSSI value
    rssi_max = -30  # Theoretical maximum RSSI value

    # Function to normalize RSSI values
    def normalize_rssi(rssi_list):
        return [(value - rssi_min) / (rssi_max - rssi_min) for value in rssi_list]

    # Apply normalization
    data['drones_rssi'] = data['drones_rssi'].apply(normalize_rssi)



def convert_data_type(data):
    # Convert from str to required data type
    data['node_positions'] = data['node_positions'].apply(lambda x: safe_convert_list(x, 'drones_pos'))
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
    standardize_data(data)
    center_and_convert_coordinates(data)
    return data


# def polar_to_cartesian(data):
#     """
#     Convert polar coordinates back to Cartesian coordinates.
#
#     Args:
#         data (list): List of lists containing [r, theta, phi] for each point.
#
#     Returns:
#         list: Updated list with [x, y, z] for each point.
#     """
#     result = []
#     for point in data:
#         r = point[0]
#         theta = point[1]
#         phi = point[2]
#
#         # Compute the Cartesian coordinates
#         x = r * np.sin(phi) * np.cos(theta)
#         y = r * np.sin(phi) * np.sin(theta)
#         z = r * np.cos(phi)
#
#         # Append the Cartesian coordinates [x, y, z]
#         result.append([x, y, z])
#     return result


def polar_to_cartesian(data):
    """
    Convert polar coordinates back to Cartesian coordinates using PyTorch.

    Args:
        data (Tensor): Tensor containing [r, theta, phi] for each point.
                       r is the radial distance,
                       theta is the polar angle from the positive z-axis (colatitude),
                       phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        Tensor: Updated tensor with [x, y, z] for each point.
    """
    r = data[:, 0]
    theta = data[:, 1]  # Polar angle (colatitude)
    phi = data[:, 2]  # Azimuthal angle

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack([x, y, z], dim=1)


def undo_center_coordinates(centered_coords, id, midpoints):
    """
    Adjust coordinates by adding the midpoint, calculated from stored midpoints for each index.

    Args:
        centered_coords (np.ndarray): The centered coordinates to be adjusted.
        id (int): The unique identifier for the data sample.
        midpoints (dict): The dictionary containing midpoints for each ID.

    Returns:
        np.ndarray: The uncentered coordinates.
    """
    midpoint = np.array(midpoints[str(id)])  # Convert index to string if it's an integer
    return centered_coords + midpoint

def scaler_inverse_transform_torch(scaled, scaler, norm_type='minmax'):
    """
    Reverse the scaling operation in PyTorch.

    Args:
        scaled (torch.Tensor): The scaled data.
        scaler (dict): Dictionary containing scaler parameters ('min', 'max', 'mean', 'std').
        norm_type (str): Type of normalization ('minmax' or 'zscore').

    Returns:
        torch.Tensor: The data in its original scale.
    """
    if norm_type == 'minmax':
        # Assuming 'min' and 'max' are stored in the scaler dict
        min_val = scaler['min']
        max_val = scaler['max']
        return min_val + (scaled - 0) * (max_val - min_val) / (1 - 0)
    elif norm_type == 'zscore':
        # Assuming 'mean' and 'std' are stored in the scaler dict
        mean = scaler['mean']
        std = scaler['std']
        return scaled * std + mean
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def convert_output_eval(output, data, device, id, midpoints):
    """
    Convert and evaluate the output coordinates by uncentering them using the stored midpoints.

    Args:
        output (torch.Tensor): The model output tensor.
        data (str): The type of data, either 'prediction' or 'target'.
        device (torch.device): The device on which the computation is performed.
        id (int): The unique identifier for the data sample.
        midpoints (dict): The dictionary containing midpoints for each ID.

    Returns:
        torch.Tensor: The converted coordinates after uncentering.
    """
    output = output.to(device)  # Ensure the output tensor is on the right device

    if params['feats'] == 'cartesian' or data == 'target':
        # output = output.cpu().numpy()
        converted_output = undo_center_coordinates(output.cpu().numpy(), id, midpoints)
    if params['feats'] == 'polar' and data == 'prediction':
        # print("output: ", output)
        # polar_coords = cyclical_to_angular(output)
        # print("polar_coords: ", polar_coords)
        cartesian_output = polar_to_cartesian(output)
        # output = cartesian_output.cpu().numpy()
        # print("cartesian_output: ", output)
        converted_output = undo_center_coordinates(cartesian_output.cpu().numpy(), id, midpoints)
        # print("undo_center_coordinates: ", converted_output)

    # Reverse standardization
    if params['norm'] == 'zscore':
        with open('zscore_mean_std.json', 'r') as json_file:
            data_loaded = json.load(json_file)
            position_means = np.array(data_loaded['position_means'])
            position_stds = np.array(data_loaded['position_stds'])
        converted_output = converted_output * position_stds + position_means

    # Reverse min-max normalization
    elif params['norm'] == 'minmax':
        experiment_path = 'experiments/' + params['feats'] + '_' + params['edges'] + '_' + params['norm'] + '/' + 'trial' + str(params['trial_num'])
        scaler = load_scaler(f'{experiment_path}/coords_scaler.pkl')
        # print("Output before inverse transform:", output.min(), output.max(), output.mean())
        converted_output = scaler.inverse_transform(converted_output)
        # print("Output after inverse transform:", converted_output.min(), converted_output.max(), converted_output.mean())

        # Assuming 'scaler' is already loaded with the training set's parameters
        # test_points = np.array([[0, 0, 0], [100, 100, 100], [50, 50, 50]])
        # scaled_points = scaler.transform(test_points)
        # reversed_points = scaler.inverse_transform(scaled_points)
        # print("Original points:", test_points)
        # print("Scaled points:", scaled_points)
        # print("Reversed points:", reversed_points)
        # quit()

        # print("unormalized: ", converted_output)
        # print('\n')

    return torch.tensor(converted_output, device=device)



def convert_output(output, device):
    output = output.to(device)  # Ensure the output is on the correct device
    if params['feats'] == 'polar':
        # polar_coords = cyclical_to_angular(output)

        converted_output = polar_to_cartesian(output)
        return converted_output
    return output  # If not polar, just pass the output through


def save_datasets(preprocessed_data, data, train_path, val_path, test_path):
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
    torch_geo_dataset = [create_torch_geo_data(row) for _, row in preprocessed_data.iterrows()]

    # Shuffle the dataset
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Reorder both datasets based on shuffled indices
    data = data.iloc[indices].reset_index(drop=True)
    torch_geo_dataset = [torch_geo_dataset[i] for i in indices]

    logging.info('Creating train-test split...')
    train_size = int(0.7 * len(torch_geo_dataset))
    val_size = int(0.1 * len(torch_geo_dataset))
    test_size = len(torch_geo_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(torch_geo_dataset, [train_size, val_size, test_size])

    # Function to extract indices from a Subset
    def get_indices(dataset_subset):
        return dataset_subset.indices

    # Extract indices for each split
    train_indices = get_indices(train_dataset)
    val_indices = get_indices(val_dataset)
    test_indices = get_indices(test_dataset)

    # Use these indices to split the DataFrame
    train_df = data.iloc[train_indices].reset_index(drop=True)
    val_df = data.iloc[val_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)

    # print("test df: ", test_df)
    # print("test dataset: ", test_dataset)
    #
    # # Extract 'id' from the PyTorch Geometric test_dataset (ensuring it's based on the actual order in the subset)
    # test_dataset_ids = [test_dataset.dataset[data_idx].id for data_idx in test_dataset.indices]
    #
    # # Extract 'id' from the test_data DataFrame directly
    # test_data_ids = test_df['id'].tolist()
    #
    # # Directly compare the lists
    # ids_in_same_order = test_data_ids == test_dataset_ids
    # print("Are the IDs in the same order across both test datasets?:", ids_in_same_order)
    #
    # quit()

    # During the dataset creation and after splitting:
    # print("Example Jammer Position in dataset before split:", data.iloc[0]['jammer_position'])
    # print("Example Jammer Position in training dataset:", train_dataset[0].y)
    # print("Example Jammer Position in validation dataset:", val_dataset[0].y)
    # print("Example Jammer Position in test dataset:", test_dataset[0].y)

    # logging.info("Saving preprocessed data...")
    # with gzip.open(train_path, 'wb') as f:
    #     pickle.dump(train_dataset, f)
    # with gzip.open(val_path, 'wb') as f:
    #     pickle.dump(val_dataset, f)
    # with gzip.open(test_path, 'wb') as f:
    #     pickle.dump(test_dataset, f)

    logging.info("Saving preprocessed data...")
    experiments_path = 'experiments/' + params['feats'] + '_' + params['edges'] + '_' + params['norm'] + '/' + 'trial' + str(params['trial_num']) + '/'

    # train_path = experiments_path + 'train_df.gzip'
    # val_path = experiments_path + 'validation_df.gzip'
    # test_path = experiments_path + 'test_df.gzip'
    # with gzip.open(train_path, 'wb') as f:
    #     pickle.dump(train_df, f)
    # with gzip.open(val_path, 'wb') as f:
    #     pickle.dump(val_df, f)
    # with gzip.open(test_path, 'wb') as f:
    #     pickle.dump(test_df, f)
    #
    # train_path = experiments_path + 'train_torch_geo_dataset.gzip'
    # val_path = experiments_path + 'validation_torch_geo_dataset.gzip'
    # test_path = experiments_path + 'test_torch_geo_dataset.gzip'
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
    data['id'] = range(1, len(data) + 1)
    # print("Unique Jammer Positions Initial:", data['jammer_position'].unique())
    # data.drop(columns=['jammer_type', 'jammer_power', 'pl_exp', 'sigma'], inplace=True)
    data.drop(columns=['jammer_power', 'pl_exp', 'sigma'], inplace=True)

    # Create a deep copy of the DataFrame
    data_to_preprocess = data.copy(deep=True)

    # Before and after preprocessing:
    # print("Jammer Positions before processing:", data['jammer_position'].head().apply(lambda x: np.array2string(np.array(x), precision=20)))
    preprocessed_data = preprocess_data(data_to_preprocess, inference=params['inference'])
    # print("Jammer Positions after processing:", data['jammer_position'].head().apply(lambda x: np.array2string(np.array(x), precision=20)))

    train_dataset, val_dataset, test_dataset = save_datasets(preprocessed_data, data, train_path, val_path, test_path)

    # print("train_dataset[0]: ", train_dataset[0])

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=2)
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
            result = row.strip('[[').strip(']]').split()
            return [float(pos) for pos in result]
        elif data_type == 'drones_pos':
            result = row.strip('[').strip(']').split('], [')
            result = [[float(num) for num in elem.split(', ')] for elem in result]
            print('result: ', result)
            quit()
            return result
        elif data_type == 'drones_rssi':
            print('row: ', row)
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

    # Selecting features based on configuration
    if params['feats'] == 'polar':
        node_features = [list(pos) + [rssi] for pos, rssi in zip(row['polar_coordinates'], row['drones_rssi'])]
    elif params['feats'] == 'cartesian':
        node_features = [list(pos) + [rssi] for pos, rssi in zip(row['node_positions'], row['drones_rssi'])]
    else:
        raise ValueError("Unsupported feature specification")
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Preparing edges and weights
    positions = np.array(row['node_positions'])
    if params['edges'] == 'knn':
        num_samples = positions.shape[0]
        k = min(5, num_samples - 1)  # num of neighbors, ensuring k < num_samples
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        edge_index, edge_weight = [], []

        for i in range(indices.shape[0]):
            edge_index.extend([[i, i]])
            edge_weight.extend([0.0])
            for j in range(1, indices.shape[1]):
                edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                edge_weight.extend([distances[i, j], distances[i, j]])
    elif params['edges'] == 'proximity':
        edge_index, edge_weight = [], []
        num_nodes = len(row['node_positions'])
        for i in range(num_nodes):
            edge_index.append([i, i])
            edge_weight.append(0)  # Placeholder for self-loop weight
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 0.1:  # Proximity threshold
                    edge_index.extend([[i, j], [j, i]])
                    edge_weight.extend([dist, dist])
    else:
        raise ValueError("Unsupported edge specification")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Target variable
    y = torch.tensor(row['jammer_position'], dtype=torch.float)

    # Create the Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)
    data.id = row['id']  # Assign the id from the row to the Data object

    # Store additional geometric information
    data.min_coords = [row['min_x'], row['min_y'], row['min_z']]
    data.max_coords = [row['max_x'], row['max_y'], row['max_z']]
    data.center_coords = row['drones_pos_center']

    return data