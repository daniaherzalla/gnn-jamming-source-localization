import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset, Dataset
from typing import Tuple, List, Any
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import logging
from utils import cartesian_to_polar
from custom_logging import setup_logging
from config import params

from torch_geometric.utils import to_networkx

setup_logging()


class TemporalGraphDataset(Dataset):
    def __init__(self, data):
        """
        Initialize the dataset with preprocessed DataFrame.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        graph = create_torch_geo_data(row)  # Create PyTorch Geometric Data object
        graph = engineer_node_features(graph)
        return graph



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
    if params['3d']:
        for position in positions:
            r, theta, phi = position
            sin_theta = np.sin(theta)  # Sine of the polar angle
            cos_theta = np.cos(theta)  # Cosine of the polar angle
            sin_phi = np.sin(phi)  # Sine of the azimuthal angle
            cos_phi = np.cos(phi)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta, sin_phi, cos_phi])
    else:
        for position in positions:
            r, theta = position
            sin_theta = np.sin(theta)  # Sine of the azimuthal angle
            cos_theta = np.cos(theta)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta])
    return transformed_positions


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
    r = output[:, 0]
    sin_theta = output[:, 1]
    cos_theta = output[:, 2]
    sin_phi = output[:, 3]
    cos_phi = output[:, 4]

    theta = torch.atan2(sin_theta, cos_theta)  # Polar angle calculation from sin and cos
    phi = torch.atan2(sin_phi, cos_phi)  # Azimuthal angle calculation from sin and cos

    return torch.stack([r, theta, phi], dim=1)


def mean_centering(coords):
    """Center coordinates using given or calculated bounds."""
    # Calculate the geometric center
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    return centered_coords, center


def center_coordinates(data):
    """Center and convert drone and jammer positions using shared bounds for each row, and save midpoints."""
    logging.info("Centering coordinates")

    # Initialize columns for storing min and max coordinates
    data['node_positions_center'] = None

    for idx, row in data.iterrows():
        # Center coordinates using the calculated mean
        node_positions = np.vstack(row['node_positions'])
        centered_node_positions, center = mean_centering(node_positions)

        # Convert to list and check structure
        data.at[idx, 'node_positions'] = centered_node_positions.tolist()

        # Similar centering for jammer position
        jammer_pos = np.array(row['jammer_position'])
        centered_jammer_position = jammer_pos - center
        data.at[idx, 'jammer_position'] = centered_jammer_position.tolist()

        # Save center for this row index
        data.at[idx, 'node_positions_center'] = center


def standardize_data(data):
    """Apply z-score or min-max normalization based on the feature type."""
    if params['norm'] == 'minmax':
        apply_min_max_normalization(data)  # For RSSI and coordinates
    elif params['norm'] == 'unit_sphere':
        apply_min_max_normalization(data)  # For RSSI
        apply_unit_sphere_normalization(data)  # For coordinates


def apply_min_max_normalization(data):
    """Apply custom normalization to position and RSSI data."""
    logging.info("Applying min-max normalization")

    # Initialize columns for storing min and max coordinates
    data['min_coords'] = None
    data['max_coords'] = None

    # Iterate over each row to apply normalization individually
    for idx, row in data.iterrows():
        # Normalize RSSI values to range [0, 1]
        node_noise = np.array(row['node_noise'])
        min_rssi = np.min(node_noise)
        max_rssi = np.max(node_noise)

        range_rssi = max_rssi - min_rssi if max_rssi != min_rssi else 1
        normalized_rssi = (node_noise - min_rssi) / range_rssi
        data.at[idx, 'node_noise'] = normalized_rssi.tolist()

        if params['norm'] == 'minmax':
            # Normalize node positions to range [-1, 1]
            node_positions = np.vstack(row['node_positions'])
            min_coords = np.min(node_positions, axis=0)
            max_coords = np.max(node_positions, axis=0)

            # Save min and max coordinates to the dataframe
            if params['3d']:
                data.at[idx, 'min_coords'] = (min_coords[0], min_coords[1], min_coords[2])
                data.at[idx, 'max_coords'] = (max_coords[0], max_coords[1], max_coords[2])
            else:
                data.at[idx, 'min_coords'] = (min_coords[0], min_coords[1])
                data.at[idx, 'max_coords'] = (max_coords[0], max_coords[1])

            range_coords = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
            normalized_positions = 2 * ((node_positions - min_coords) / range_coords) - 1
            data.at[idx, 'node_positions'] = normalized_positions.tolist()

            # Normalize jammer position similarly if present
            if 'jammer_position' in row:
                jammer_position = np.array(row['jammer_position']).reshape(1, -1)
                jammer_position = 2 * ((jammer_position - min_coords) / range_coords) - 1
                data.at[idx, 'jammer_position'] = jammer_position.flatten().tolist()


def apply_unit_sphere_normalization(data):
    """
    Apply unit sphere normalization to position data.

    Parameters:
    data (dict): A dictionary containing 'node_positions', an array of positions.

    Returns:
    tuple: A tuple containing the normalized positions and the maximum radius.
    """
    logging.info("Applying unit sphere normalization")
    # Initialize a column for maximum radius
    data['max_radius'] = None

    for idx, row in data.iterrows():
        # Extract positions from the current row
        positions = np.array(row['node_positions'])
        jammer_position = np.array(row['jammer_position'])

        # Calculate the maximum radius from the centroid
        max_radius = np.max(np.linalg.norm(positions, axis=1))

        # Check for zero radius to prevent division by zero
        if max_radius == 0:
            raise ValueError("Max radius is zero, normalization cannot be performed.")

        # Normalize the positions uniformly
        normalized_positions = positions / max_radius
        normalized_jammer_position = jammer_position / max_radius

        # Update the DataFrame with normalized positions and maximum radius
        data.at[idx, 'node_positions'] = normalized_positions.tolist()
        data.at[idx, 'jammer_position'] = normalized_jammer_position.tolist()
        data.at[idx, 'max_radius'] = max_radius


def convert_data_type(data):
    # Convert from str to required data type for specified features
    dataset_features = params['required_features'] + ['jammer_position']

    # Apply conversion to each feature directly
    for feature in dataset_features:
        data[feature] = data[feature].apply(lambda x: safe_convert_list(x, feature))


def add_cyclical_features(data):
    """Convert azimuth angles to cyclical coordinates."""
    data['azimuth_angle'] = data.apply(lambda row: [np.arctan2(pos[1] - row['centroid'][1], pos[0] - row['centroid'][0]) for pos in row['node_positions']], axis=1)
    data['sin_azimuth'] = data['azimuth_angle'].apply(lambda angles: [np.sin(angle) for angle in angles])
    data['cos_azimuth'] = data['azimuth_angle'].apply(lambda angles: [np.cos(angle) for angle in angles])


def calculate_proximity_metric(positions, threshold=0.2):
    """Calculate the number of nearby nodes within a given threshold distance."""
    nbrs = NearestNeighbors(radius=threshold).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions)
    return [len(idx) - 1 for idx in indices]  # subtract 1 to exclude the node itself


def add_proximity_count(data):
    """Add proximity feature based on a threshold distance."""
    data['proximity_count'] = data['node_positions'].apply(
        lambda positions: calculate_proximity_metric(np.array(positions))
    )


def calculate_noise_statistics(subgraphs, stats_to_compute):
    all_graph_stats = []

    for subgraph in subgraphs:
        node_stats = []
        edge_index = subgraph.edge_index
        num_nodes = subgraph.num_nodes

        for node_id in range(num_nodes):
            # Identifying the neighbors of the current node
            neighbors = torch.cat([
                edge_index[1][edge_index[0] == node_id],
                edge_index[0][edge_index[1] == node_id]
            ], dim=0).unique()

            curr_node_noise = subgraph.x[node_id][2]  # Assuming noise is the third feature
            neighbor_noises = subgraph.x[neighbors, 2]

            # Combine current node's noise with neighbor noises
            all_noises = torch.cat([neighbor_noises, curr_node_noise.unsqueeze(0)], dim=0)

            # Handle case with fewer than two neighbors safely
            if all_noises.size(0) > 1:
                std_noise = all_noises.std().item()
                range_noise = (all_noises.max() - all_noises.min()).item()
            else:
                std_noise = 0
                range_noise = 0

            temp_stats = {
                'mean_noise': all_noises.mean().item(),
                'median_noise': all_noises.median().item(),
                'std_noise': std_noise,
                'range_noise': range_noise,
                'relative_noise': (curr_node_noise - neighbor_noises.mean()).item() if neighbors.size(0) > 0 else 0,
            }

            # Compute WCL
            if 'wcl_coefficient' in stats_to_compute:
                weights = torch.pow(10, neighbor_noises / 10)
                weighted_positions = weights.unsqueeze(1) * subgraph.x[neighbors, :2]
                wcl_estimation = weighted_positions.sum(0) / weights.sum() if weights.sum() > 0 else subgraph.x[node_id, :2]
                temp_stats['wcl_coefficient'] = wcl_estimation.tolist()

            node_stats.append(temp_stats)

        all_graph_stats.append(node_stats)

    return all_graph_stats


def add_clustering_coefficients(graphs):
    """
    Compute the clustering coefficient for each node in each graph.

    Args:
        graphs (list): List of NetworkX graph objects.

    Returns:
        list: A list of lists, where each sublist contains the clustering coefficients for nodes in a graph.
    """
    all_graphs_clustering_coeffs = []  # This will hold a list of lists, each sublist for a graph

    for graph in graphs:
        graph_clustering_coeffs = []  # Initialize an empty list for current graph's node clustering coefficients

        if len(graph.nodes()) > 0:
            clustering_coeffs = nx.clustering(graph)
            nodes = list(graph.nodes())

            # Populate the clustering coefficients for each node, maintaining the order
            for node in nodes:
                graph_clustering_coeffs.append(clustering_coeffs[node])
        else:
            graph_clustering_coeffs = []

        all_graphs_clustering_coeffs.append(graph_clustering_coeffs)  # Append the completed list for this graph

    return all_graphs_clustering_coeffs

def dynamic_moving_average(x, max_window_size=10):
    num_nodes = x.size(0)
    window_sizes = torch.clamp(num_nodes - torch.arange(num_nodes), min=1, max=max_window_size)
    averages = torch.zeros_like(x)

    for i in range(num_nodes):
        start = max(i - window_sizes[i] // 2, 0)
        end = min(i + window_sizes[i] // 2 + 1, num_nodes)
        averages[i] = x[start:end].mean(dim=0)

    return averages


def engineer_node_features(subgraph):
    if subgraph.x.size(0) == 0:
        raise ValueError("Empty subgraph encountered")

    new_features = []

    # Calculating centroid
    centroid = torch.mean(subgraph.x[:, :2], dim=0)  # Select only x and y

    if 'dist_to_centroid' in params['additional_features']:
        distances = torch.norm(subgraph.x[:, :2] - centroid, dim=1, keepdim=True)
        # print("dist_to_centroid: ", distances)
        new_features.append(distances)

    if 'sin_azimuth' in params['additional_features']:
        azimuth_angles = torch.atan2(subgraph.x[:, 1] - centroid[1], subgraph.x[:, 0] - centroid[0])
        new_features.append(torch.sin(azimuth_angles).unsqueeze(1))
        new_features.append(torch.cos(azimuth_angles).unsqueeze(1))

    # Graph-based noise stats
    graph_stats = ['mean_noise', 'median_noise', 'std_noise', 'range_noise', 'relative_noise', 'wcl_coefficient']
    noise_stats_to_compute = [stat for stat in graph_stats if stat in params['additional_features']]

    if noise_stats_to_compute:
        noise_stats = calculate_noise_statistics([subgraph], noise_stats_to_compute)
        # Save each stat as a separate tensor
        for stat in noise_stats_to_compute:
            stat_values = torch.tensor([node_stat[stat] for node_stat in noise_stats[0]], dtype=torch.float32)
            new_features.append(stat_values.unsqueeze(1))  # Unsqueeze to maintain the correct dimension

    # Moving Average for node noise with adjusted padding
    if 'moving_avg_noise' in params['additional_features']:
        node_noise = subgraph.x[:, 2]  # noise is at position 2
        moving_avg_noise = dynamic_moving_average(node_noise)
        new_features.append(moving_avg_noise.unsqueeze(1))

    # Example of using dynamic moving average for AoA
    if 'moving_avg_aoa' in params['additional_features']:
        sin_aoa = subgraph.x[:, 4]  # sin(AoA) is at position 4
        cos_aoa = subgraph.x[:, 5]  # cos(AoA) is at position 5
        aoa = torch.atan2(sin_aoa, cos_aoa)
        moving_avg_aoa = dynamic_moving_average(aoa)
        new_features.append(torch.sin(moving_avg_aoa).unsqueeze(1))
        new_features.append(torch.cos(moving_avg_aoa).unsqueeze(1))

    if new_features:
        try:
            new_features_tensor = torch.cat(new_features, dim=1)
            subgraph.x = torch.cat((subgraph.x, new_features_tensor), dim=1)
        except RuntimeError as e:
            raise e

    return subgraph


def preprocess_data(data, params):
    """
   Preprocess the input data by converting lists, scaling features, and normalizing RSSI values.

   Args:
       data (pd.DataFrame): The input data containing columns to be processed.

   Returns:
       pd.DataFrame: The preprocessed data with transformed features.
   """
    logging.info("Preprocessing data...")
    # Conversion from string to list type
    center_coordinates(data)
    standardize_data(data)
    if params['coords'] == 'polar':
        convert_to_polar(data)
    return data


def convert_to_polar(data):
    data['polar_coordinates'] = data['node_positions'].apply(cartesian_to_polar)
    data['polar_coordinates'] = data['polar_coordinates'].apply(angle_to_cyclical)


def polar_to_cartesian(data):
    """
    Convert polar coordinates to Cartesian coordinates using only PyTorch operations.

    Args:
        data (Tensor): Tensor on the appropriate device (GPU or CPU) containing
                       [r, theta, phi] for each point.
                       r is the radial distance,
                       theta is the polar angle from the positive z-axis (colatitude),
                       phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        Tensor: Updated tensor with [x, y, z] for each point.
    """
    r = data[:, 0]
    theta = data[:, 1]  # Polar angle (colatitude)

    if params['3d']:
        phi = data[:, 2]  # Azimuthal angle
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        cartesian_coords = torch.stack([x, y, z], dim=1)
    else:
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        cartesian_coords = torch.stack([x, y], dim=1)

    return cartesian_coords



# Original!
def convert_output_eval(output, data_batch, data_type, device):
    """
    Convert and evaluate the output coordinates by uncentering them using the stored midpoints.

    Args:
        output (torch.Tensor): The model output tensor.
        data_batch (torch.Tensor): Data batch.
        data_type (str): The type of data, either 'prediction' or 'target'.
        device (torch.device): The device on which the computation is performed.

    Returns:
        torch.Tensor: The converted coordinates after uncentering.
    """
    output = output.to(device)  # Ensure the output tensor is on the right device

    if params['norm'] == 'minmax':
        # 1. Reverse normalization using min_coords and max_coords
        min_coords = data_batch.min_coords.to(device).view(-1, 2)
        max_coords = data_batch.max_coords.to(device).view(-1, 2)

        range_coords = max_coords - min_coords
        converted_output = (output + 1) / 2 * range_coords + min_coords


    elif params['norm'] == 'unit_sphere':
        # 1. Reverse unit sphere normalization using max_radius
        max_radius = data_batch.max_radius.to(device).view(-1, 1)
        converted_output = output * max_radius

    # 2. Reverse centering using the stored node_positions_center
    centers = data_batch.node_positions_center.to(device).view(-1, 2)
    converted_output += centers

    # return torch.tensor(converted_output, device=device)
    return converted_output.clone().detach().to(device)



def convert_output(output, device):  # for training to compute val loss
    output = output.to(device)  # Ensure the output is on the correct device
    if params['coords'] == 'polar':
        output = cyclical_to_angular(output)
        converted_output = polar_to_cartesian(output)
        return converted_output
    return output  # If not polar, just pass the output through


def save_reduced_dataset(dataset, indices, path):
    """
    Saves only the necessary data from the original dataset at specified indices,
    effectively reducing the file size by excluding unnecessary data.
    """
    reduced_data = [dataset[i] for i in indices]  # Extract only the relevant data
    torch.save(reduced_data, path)  # Save the truly reduced dataset


def split_datasets(data):
    """
    Save the preprocessed data into train, validation, and test datasets.

    Args:
        preprocessed_data (pd.DataFrame): The preprocessed data to be split and saved.
        params (str): The GNN project parameters.
        experiments_path (str): The file path to save the train, test, validation datasets.

    Returns:
        Tuple[list, list, list, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        The train, validation, and test datasets and their corresponding DataFrames.
    """

    logging.info('Creating graphs...')

    # Stratified split using scikit-learn
    train_idx, test_idx, train_test_y, test_y = train_test_split(
        np.arange(len(data)),
        data['dataset'],
        test_size=0.3,
        stratify=data['dataset'],
        random_state=100
    )

    # Now split the test into validation and test
    val_idx, test_idx, _, _ = train_test_split(
        test_idx,
        test_y,
        test_size=len(data) - len(train_idx) - int(0.1 * len(data)),
        stratify=test_y,
        random_state=100
    )

    # Convert indices back to DataFrame subsets
    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)

    # return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, raw_test_df
    return train_df, val_df, test_df




def save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path):
    """
    Process the combined train, validation, and test data, and save them to disk.

    Args:
        combined_train_data (list): List of training data samples.
        combined_val_data (list): List of validation data samples.
        combined_test_data (list): List of test data samples.
        combined_train_df (pd.DataFrame): DataFrame containing combined training data.
        combined_val_df (pd.DataFrame): DataFrame containing combined validation data.
        combined_test_df (pd.DataFrame): DataFrame containing combined test data.
        experiments_path (str): The path where the processed data will be saved.
    """
    logging.info("Saving data...")

    # Save the combined DataFrame subsets
    combined_train_df.to_csv(os.path.join(experiments_path, 'train_dataset.csv'), index=False)
    combined_val_df.to_csv(os.path.join(experiments_path, 'val_dataset.csv'), index=False)
    combined_test_df.to_csv(os.path.join(experiments_path, 'test_dataset.csv'), index=False)

    # Dataset types for specific filtering
    dataset_types = ['circle', 'triangle', 'rectangle', 'random', 'circle_jammer_outside_region',
                     'triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
                     'random_jammer_outside_region', 'all_jammed', 'all_jammed_jammer_outside_region',
                     'dynamic_guided_path', 'dynamic_linear_path']

    for dataset in dataset_types:
        train_subset = combined_train_df[combined_train_df['dataset'] == dataset]
        val_subset = combined_val_df[combined_val_df['dataset'] == dataset]
        test_subset = combined_test_df[combined_test_df['dataset'] == dataset]

        if not train_subset.empty:
            train_subset.to_csv(os.path.join(experiments_path, f'{dataset}_train_set.csv'), index=False)
        if not val_subset.empty:
            val_subset.to_csv(os.path.join(experiments_path, f'{dataset}_val_set.csv'), index=False)
        if not test_subset.empty:
            test_subset.to_csv(os.path.join(experiments_path, f'{dataset}_test_set.csv'), index=False)


# Sample function definition
def calculate_perc_completion(data):
    # Loop through each row in the dataframe
    perc_completion_list = []
    for _, row in data.iterrows():
        timestamps = np.array(row['timestamps'])
        min_time = np.min(timestamps)
        max_time = np.max(timestamps)
        perc_completion = (timestamps - min_time) / (max_time - min_time) if max_time != min_time else np.zeros_like(timestamps)
        perc_completion_list.append(perc_completion)

    # Add the new column to the dataframe
    data['perc_completion_full'] = perc_completion_list
    return data


def downsample_data(data):
    """
    Apply downsampling using time window averaging to reduce the number of nodes to a fixed `max_nodes`.

    Args:
        data (pd.DataFrame): The input data containing columns to be downsampled.
        max_nodes (int): The desired number of nodes to retain after downsampling.

    Returns:
        pd.DataFrame: The downsampled dataset with the specified number of nodes.
    """
    max_nodes = params['max_nodes']
    logging.info(f"Downsampling to {max_nodes} nodes...")

    for idx, row in data.iterrows():
        # Convert timestamps to numpy array
        timestamps = np.array(row['timestamps'])
        num_original_nodes = len(timestamps)

        # Calculate the window size to downsample to `max_nodes`
        if num_original_nodes <= max_nodes:
            # If the number of original nodes is already less than or equal to max_nodes, skip downsampling
            continue

        window_size = num_original_nodes // max_nodes

        # Calculate the number of windows based on the calculated window size
        num_windows = max_nodes
        downsampled_timestamps = []
        downsampled_positions = []
        downsampled_noise_values = []
        downsampled_angles = []

        for i in range(num_windows):
            # Define the window range
            start_idx = i * window_size
            end_idx = start_idx + window_size

            # Average values within the window
            window_timestamps = timestamps[start_idx:end_idx]
            downsampled_timestamps.append(np.mean(window_timestamps))

            # For node positions, calculate the mean of x, y coordinates
            node_positions = np.array(row['node_positions'][start_idx:end_idx])
            downsampled_positions.append(np.mean(node_positions, axis=0).tolist())

            # For noise values and angles, calculate the mean
            noise_values = np.array(row['node_noise'][start_idx:end_idx])
            downsampled_noise_values.append(np.mean(noise_values))

            if 'angle_of_arrival' in params['required_features']:
                angles = np.array(row['angle_of_arrival'][start_idx:end_idx])
                downsampled_angles.append(np.mean(angles))

        # Replace the original data with downsampled data
        data.at[idx, 'timestamps'] = downsampled_timestamps
        data.at[idx, 'node_positions'] = downsampled_positions
        data.at[idx, 'node_noise'] = downsampled_noise_values
        if 'angle_of_arrival' in params['required_features']:
            data.at[idx, 'angle_of_arrival'] = downsampled_angles

    return data



def load_data(params, test_set_name, experiments_path=None):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    logging.info("Loading data...")
    if params['inference']:
        # load raw data csv
        for test_data in test_set_name:
            # TODO: add downsampling, perc completion
            print(f"dataset: {test_data}")
            print(f"experiments_path: {experiments_path}")
            file_path = experiments_path + test_data
            test_df = pd.read_csv(file_path)
            test_df['id'] = range(1, len(test_df) + 1)
            convert_data_type(test_df)
            # calculate_perc_completion()

            # Apply transformations to graphs
            if params['dynamic']:
                test_df = apply_processing(test_df, 'test')
                test_df = test_df.reset_index()

            print("TEST")
            test_dataset = preprocess_data(test_df, params)
            print(test_dataset.columns)

            return None, None, test_dataset
    else:
        combined_train_df = pd.DataFrame()
        combined_val_df = pd.DataFrame()
        combined_test_df = pd.DataFrame()

        if params['all_env_data']:
            datasets = ['data/train_test_data/log_distance/urban_area/combined_urban_area.csv', 'data/train_test_data/log_distance/shadowed_urban_area/combined_shadowed_urban_area.csv']
        else:
            datasets = [params['dataset_path']]
        for dataset in datasets:
            print(f"dataset: {dataset}")
            data = pd.read_csv(dataset)
            data['id'] = range(1, len(data) + 1)
            convert_data_type(data)

            # Apply downsampling
            data = downsample_data(data)

            # Calculate normalized time on full instance
            data = calculate_perc_completion(data)

            # Create train test splits
            train_df, val_df, test_df = split_datasets(data)

            # Apply random_crop to training and test datasets
            if params['dynamic']:
                train_df = apply_processing(train_df, 'train')
                val_df = apply_processing(val_df, 'val')
                test_df = apply_processing(test_df, 'test')
                val_df = val_df.reset_index()
                test_df = test_df.reset_index()

            test_df.to_csv(os.path.join(experiments_path, 'raw_test_data.csv'), index=False)

            train_dataset = preprocess_data(train_df, params)
            val_dataset = preprocess_data(val_df, params)
            test_dataset = preprocess_data(test_df, params)

            combined_train_df = pd.concat([combined_train_df, train_dataset], ignore_index=True)
            combined_val_df = pd.concat([combined_val_df, val_dataset], ignore_index=True)
            combined_test_df = pd.concat([combined_test_df, test_dataset], ignore_index=True)

        # Process and save the combined data
        save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path)

        return combined_train_df, combined_val_df, combined_test_df


def random_crop(row, min_nodes=3):
    """
    Perform a random crop of node samples for one row.
    Args:
        row (pd.Series): A single row from a DataFrame.
        min_nodes (int): Minimum number of nodes to keep, default is 3.
    Returns:
        pd.Series: Modified row with cropped data.
    """
    # Determine the features to crop
    if 'timestamps' in params['required_features']:
        node_features = params['required_features'] + ['perc_completion_full']
    else:
        node_features = params['required_features']

    total_nodes = len(row['node_positions'])

    if total_nodes > min_nodes:
        # Fix the start index at 0
        start = 0
        # Randomize the end index, ensuring at least `min_nodes` nodes are kept
        end = np.random.randint(min_nodes, total_nodes)

        # Crop all the relevant features
        for key in node_features:
            if isinstance(row[key], list) or isinstance(row[key], np.ndarray):  # Ensure it's a list or ndarray
                row[key] = row[key][start:end]

    return row


def incremental_node_addition(row):
    """
    Generate incremental additions of node samples for one row.
    Args:
        row (pd.Series): A single row from a DataFrame.
    Returns:
        List[pd.Series]: List of new rows each incrementing node samples by one.
    """
    if 'timestamps' in params['required_features']:
        node_features = params['required_features'] + ['perc_completion_full']
    else:
        node_features = params['required_features']

    min_nodes = 3
    total_nodes = len(row['node_positions'])
    new_rows = []

    if total_nodes >= min_nodes:
        for i in range(min_nodes, total_nodes + 1):
            new_row = row.copy()
            for key in node_features:
                new_row[key] = row[key][:i]
            new_rows.append(new_row)
    return new_rows


def batch_node_addition(row, granularity=25):
    """
    Generate incremental additions of node samples for one row based on a percentage granularity.
    Args:
        row (pd.Series): A single row from a DataFrame.
        granularity (int): Percentage of total nodes to increment at each step.
    Returns:
        List[pd.Series]: List of new rows, each incrementing node samples by the specified percentage.
    """
    if 'timestamps' in params['required_features']:
        node_features = params['required_features'] + ['perc_completion_full']
    else:
        node_features = params['required_features']

    min_nodes = max(3, int(len(row['node_positions']) * (granularity / 100)))  # Ensuring at least 3 nodes
    total_nodes = len(row['node_positions'])
    new_rows = []

    if total_nodes >= min_nodes:
        step_size = max(1, int(total_nodes * (granularity / 100)))  # Calculate step size as a percentage of total nodes
        steps = range(min_nodes, total_nodes + 1, step_size)  # Create steps from min_nodes to total_nodes with step_size
        for i in steps:
            new_row = row.copy()
            for key in node_features:
                new_row[key] = row[key][:i]
            new_rows.append(new_row)

        # Make sure to include the full node set if the last step isn't exactly total_nodes
        if steps[-1] != total_nodes:
            new_row = row.copy()
            for key in node_features:
                new_row[key] = row[key][:total_nodes]
            new_rows.append(new_row)

    return new_rows


def apply_processing(df, mode):
    """
    Apply the specified processing mode to each row of the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data to process.
        mode (str): Processing mode, either 'train_val' or 'test'.
    Returns:
        pd.DataFrame: DataFrame containing all processed rows.
    """
    processed_rows = []
    for _, row in df.iterrows():
        processed = process_row(row, mode)
        processed_rows.extend(processed)  # Extend to flatten list of Series into one list

    # Convert list of pd.Series to DataFrame
    return pd.DataFrame(processed_rows)

def process_row(row, mode):
    """
    Process a single row based on the specified mode.
    """
    if mode == 'train':
        return [random_crop(row)]
    elif mode == 'val':
        return batch_node_addition(row)
    elif mode == 'test':
        return incremental_node_addition(row)


def create_data_loader(train_data, val_data, test_data, batch_size):
    """
    Create data loaders using the TemporalGraphDataset instances for training, validation, and testing sets.
    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        val_data (pd.DataFrame): DataFrame containing the validation data.
        test_data (pd.DataFrame): DataFrame containing the testing data.
        batch_size (int): The size of batches.
    Returns:
        tuple: Three DataLoaders for the training, validation, and testing datasets.
    """
    if params['inference']:
        # Instantiate the dataset classes for train, val, and test
        test_dataset = TemporalGraphDataset(test_data)

        # Create DataLoaders for each dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return None, None, test_loader
    else:
        # Instantiate the dataset classes for train, val, and test
        train_dataset = TemporalGraphDataset(train_data)
        val_dataset = TemporalGraphDataset(val_data)
        test_dataset = TemporalGraphDataset(test_data)

        # Create DataLoaders for each dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return train_loader, val_loader, test_loader


def safe_convert_list(row: str, data_type: str) -> List[Any]:
    """
    Safely convert a string representation of a list to an actual list,
    with type conversion tailored to specific data types including handling
    for 'states' which are extracted and stripped of surrounding quotes.

    Args:
        row (str): String representation of a list.
        data_type (str): The type of data to convert ('jammer_pos', 'drones_pos', 'node_noise', 'states').

    Returns:
        List: Converted list or an empty list if conversion fails.
    """
    try:
        if data_type == 'jammer_position':
            result = row.strip('[').strip(']').split(', ')
            return [float(pos) for pos in result]
        elif data_type == 'node_positions':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        elif data_type == 'node_noise':
            result = row.strip('[').strip(']').split(', ')
            return [float(noise) for noise in result]
        elif data_type == 'node_rssi':
            result = row.strip('[').strip(']').split(', ')
            return [float(rssi) for rssi in result]
        elif data_type == 'node_states':
            result = row.strip('[').strip(']').split(', ')
            return [int(state) for state in result]
        elif data_type == 'timestamps':
            result = row.strip('[').strip(']').split(', ')
            return [float(time) for time in result]
        elif data_type == 'angle_of_arrival':
            result = row.strip('[').strip(']').split(', ')
            return [float(aoa) for aoa in result]
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error


def plot_graph_temporal(subgraph):
    G = to_networkx(subgraph, to_undirected=True)
    pos = nx.spring_layout(G)  # Layout for visual clarity
    nx.draw(G, pos, node_size=70, node_color='skyblue', with_labels=True, font_weight='bold')
    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()


def plot_graph(positions, edge_index, node_features, edge_weights=None, jammer_positions=None, show_weights=False):
    G = nx.Graph()

    # Ensure positions and features are numpy arrays for easier handling
    positions = np.array(positions)
    node_features = np.array(node_features)
    jammer_positions = np.array(jammer_positions)

    # Add nodes with features and positions
    for i, pos in enumerate(positions):
        # assuming RSSI is the last feature in node_features array
        if params['dynamic']:
            G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2],
                       timestamp=node_features[i][-1], sin_aoa=node_features[i][-3], cos_aoa=node_features[i][-2])
        else:
            G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2])

    # Convert edge_index to a usable format if it's a tensor or similar
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    # Add edges
    if edge_weights is not None:
        edge_weights = edge_weights.numpy() if isinstance(edge_weights, torch.Tensor) else edge_weights
        for idx, (start, end) in enumerate(edge_index.T):  # Ensure edge_index is transposed correctly
            weight = edge_weights[idx]
            if weight != 0:  # Check if weight is not zero
                G.add_edge(start, end, weight=weight)
    else:
        for start, end in edge_index.T:
            G.add_edge(start, end)

    # Position for drawing
    pos = {i: (p[0], p[1]) for i, p in enumerate(positions)}

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=50)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Node labels including timestamp, sin and cos of AoA
    if params['dynamic']:
        node_labels = {i: f"ID:{i}\nNoise:{G.nodes[i]['noise']:.2f}\nTimestamp:{G.nodes[i]['timestamp']:.2f}\nSin AoA:{G.nodes[i]['sin_aoa']:.2f}\nCos AoA:{G.nodes[i]['cos_aoa']:.2f}"
                       for i in G.nodes()}
    else:
        node_labels = {i: f"ID:{i}\nNoise:{G.nodes[i]['noise']:.2f}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Optionally draw edge weights
    if show_weights and edge_weights is not None:
        edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw jammer position
    if jammer_positions is not None:
        for jammer_pos in jammer_positions:
            plt.scatter(*jammer_pos, color='red', s=100, label='Jammer')  # Add jammers to the plot
            plt.annotate('Jammer', xy=jammer_pos, xytext=(5, 5), textcoords='offset points')

    plt.title("Network Graph with Node Features")
    plt.axis('off')  # Turn off the axis
    plt.show()



def ensure_complementary_features(params):
    """
    Ensure that if either sin_azimuth or cos_azimuth is included, then both are included.

    Args:
        params (dict): Dictionary containing 'required_features' and 'additional_features'.

    Returns:
        list: Updated list of features including both sin_azimuth and cos_azimuth if either is present.
    """
    required_features = params['required_features']
    additional_features = [] #params['additional_features']

    if isinstance(additional_features, tuple):
        additional_features = list(additional_features)

    if isinstance(required_features, tuple):
        required_features = list(required_features)

    # Combine required and additional features
    all_features = required_features + additional_features
    return all_features


def create_torch_geo_data(row: pd.Series) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """
    # Select and combine features
    # all_features = ensure_complementary_features(params) #params['required_features'] + params['additional_features']
    all_features = params['required_features'].copy()  # Copy the list

    # Handling Angle of Arrival (AoA)
    if 'angle_of_arrival' in params['required_features']:
        # print('AOA')
        aoa = np.array(row['angle_of_arrival'])
        sin_aoa = np.sin(aoa)
        cos_aoa = np.cos(aoa)
        all_features.remove('angle_of_arrival')  # Remove original AoA feature

    # Convert timestamps to percentage completion
    if 'timestamps' in params['required_features']:
        timestamps = np.array(row['timestamps'])
        min_time = np.min(timestamps)
        max_time = np.max(timestamps)
        perc_completion = (timestamps - min_time) / (max_time - min_time) if max_time != min_time else np.zeros_like(timestamps)
        all_features.remove('timestamps')  # Remove original timestamps to replace with normalized ones

    node_features = [
        sum(([feature_value] if not isinstance(feature_value, list) else feature_value
             for feature_value in node_data), [])
        for node_data in zip(*(row[feature] for feature in all_features))
    ]

    # Append AoA and percentage completion as new features
    for i, feature_list in enumerate(node_features):
        if 'angle_of_arrival' in params['required_features']:
            feature_list.extend([sin_aoa[i], cos_aoa[i]])
        if 'timestamps' in params['required_features']:
            feature_list.extend([perc_completion[i]])

    # Convert to PyTorch tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Preparing edges and weights
    positions = np.array(row['node_positions'])
    if params['edges'] == 'knn':
        num_samples = positions.shape[0]
        k = min(params['num_neighbors'], num_samples - 1)  # num of neighbors, ensuring k < num_samples
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        edge_index, edge_weight = [], []

        # Add self loop
        for i in range(indices.shape[0]):
            edge_index.extend([[i, i]])
            edge_weight.extend([0.0])
            for j in range(1, indices.shape[1]):
                edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                edge_weight.extend([distances[i, j], distances[i, j]])
    else:
        raise ValueError("Unsupported edge specification")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    jammer_positions = np.array(row['jammer_position']).reshape(-1, 2)  # Assuming this reshaping is valid based on your data structure
    y = torch.tensor(jammer_positions, dtype=torch.float)

    # Plot
    # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True)

    # Create the Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # Convert geometric information to tensors
    data.id = row['id']  # Assign the id from the row to the Data object
    data.node_positions_center = torch.tensor(row['node_positions_center'], dtype=torch.float)
    data.sigma = torch.tensor(row['sigma'], dtype=torch.float)
    if params['norm'] == 'minmax':
        data.min_coords = torch.tensor(row['min_coords'], dtype=torch.float)
        data.max_coords = torch.tensor(row['max_coords'], dtype=torch.float)
    elif params['norm'] == 'unit_sphere':
        data.max_radius = torch.tensor(row['max_radius'], dtype=torch.float)

    # Store the perc_completion as part of the Data object
    if 'timestamps' in params['required_features']:
        data.perc_completion = torch.tensor(perc_completion, dtype=torch.float32)
        data.perc_completion_full = torch.tensor(row['perc_completion_full'], dtype=torch.float)

    return data
