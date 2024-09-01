import gzip
import json
import os
import pandas as pd
import numpy as np
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Tuple, List, Any
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import logging
import pickle
from utils import set_seeds_and_reproducibility, cartesian_to_polar
from custom_logging import setup_logging
from config import params

setup_logging()

# if params['reproduce']:
#     set_seeds_and_reproducibility()


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
        data.at[idx, 'node_positions'] = centered_node_positions.tolist()

        # Similar centering for jammer position
        jammer_pos = np.array(row['jammer_position'])
        # print("jammer_pos: ", jammer_pos)
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
        # print('normalized_jammer_position: ', normalized_jammer_position)

        # Update the DataFrame with normalized positions and maximum radius
        data.at[idx, 'node_positions'] = normalized_positions.tolist()
        data.at[idx, 'jammer_position'] = normalized_jammer_position.tolist()
        data.at[idx, 'max_radius'] = max_radius


def convert_data_type(data):
    # Convert from str to required data type for specified features
    dataset_features = ['jammer_position', 'node_positions', 'node_noise', 'node_rssi', 'node_states']

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


def create_graphs(data):
    graphs = []
    for index, row in data.iterrows():
        G = nx.Graph()
        positions = np.array(row['node_positions'])
        num_nodes = positions.shape[0]

        if num_nodes > 1:  # Ensure there are enough nodes to form a graph
            k = min(params['num_neighbors'], num_nodes - 1)  # num of neighbors
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
            distances, indices = nbrs.kneighbors(positions)

            for i in range(num_nodes):
                G.add_node(i, position=positions[i], noise=row['node_noise'][i])  # Changed 'pos' to 'position'
                for j in indices[i]:  # Skip the first index since it is the point itself
                    G.add_edge(i, j)

        graphs.append(G)

    return graphs




def calculate_noise_statistics(graphs, stats_to_compute):
    all_graph_stats = []  # This will hold a list of lists, each sublist for a graph

    for G in graphs:
        graph_stats = []  # Initialize an empty list for current graph's node stats

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            curr_node_noise = G.nodes[node]['noise']
            neighbor_noises = [G.nodes[neighbor]['noise'] for neighbor in neighbors]  # Neighbors' noise
            neighbor_positions = [G.nodes[neighbor]['position'] for neighbor in neighbors]  # Neighbors' positions

            node_stats = {}  # Dictionary to store stats for the current node

            # Compute mean noise for neighbors excluding the node itself
            if neighbors:  # Ensure there are neighbors
                mean_neighbor_noise = np.mean(neighbor_noises)
            else:
                mean_neighbor_noise = curr_node_noise  # If no neighbors, fallback to own noise

            if 'mean_noise' in stats_to_compute:
                node_stats['mean_noise'] = mean_neighbor_noise
            if 'median_noise' in stats_to_compute:
                node_stats['median_noise'] = np.median(neighbor_noises + [curr_node_noise])  # Include self for median
            if 'std_noise' in stats_to_compute:
                node_stats['std_noise'] = np.std(neighbor_noises + [curr_node_noise])  # Include self for std
            if 'range_noise' in stats_to_compute:
                node_stats['range_noise'] = np.max(neighbor_noises + [curr_node_noise]) - np.min(neighbor_noises + [curr_node_noise])

            # Compute relative noise
            node_stats['relative_noise'] = curr_node_noise - mean_neighbor_noise

            # Compute the weighted centroid local (WCL)
            if 'wcl_coefficient' in stats_to_compute:
                total_weight = 0
                weighted_sum = np.zeros(2)  # 2D positions
                for pos, noise in zip(neighbor_positions, neighbor_noises):
                    weight = 10 ** (noise / 10)
                    weighted_coords = weight * np.array(pos)
                    weighted_sum += weighted_coords
                    total_weight += weight
                if total_weight != 0:
                    wcl_estimation = weighted_sum / total_weight
                    node_stats['wcl_coefficient'] = wcl_estimation.tolist()  # Store as a list if necessary

            graph_stats.append(node_stats)  # Append the current node's stats to the graph's list

        all_graph_stats.append(graph_stats)  # Append the completed list of node stats for this graph

    return all_graph_stats


# def calculate_noise_statistics(graphs, jammer_positions, stats_to_compute):
#     all_graph_stats = []  # list of lists, each sublist for a graph
#     count = 0
#
#     for G in graphs:
#         graph_stats = []  # current graph's node stats
#         count += 1
#
#         for node in G.nodes:
#             plt.figure(figsize=(12, 10))  # new fig for each node's WCL calculation
#
#             # Plot all nodes and edges
#             for n in G.nodes:
#                 plt.scatter(*G.nodes[n]['position'], color='grey', s=50, edgecolor='black', zorder=1)  # Normal nodes
#             for n1, n2 in G.edges:
#                 plt.plot(*zip(G.nodes[n1]['position'], G.nodes[n2]['position']), color='grey', linewidth=0.3, zorder=0)  # Edges
#
#             # Plot jammer position
#             jammer_pos = jammer_positions[count]
#             print("jammer_pos: ", jammer_pos)
#             plt.scatter(*jammer_pos, color='magenta', s=100, marker='^', label='Jammer', zorder=4)
#
#             neighbors = list(G.neighbors(node))
#             curr_node_noise = G.nodes[node]['noise']
#             neighbor_noises = [G.nodes[neighbor]['noise'] for neighbor in neighbors]
#             neighbor_positions = [G.nodes[neighbor]['position'] for neighbor in neighbors]
#
#             node_stats = {}
#
#             if 'wcl_coefficient' in stats_to_compute:
#                 total_weight = 0
#                 weighted_sum = np.zeros(2)  # Assuming 2D positions
#                 weights = []
#                 for pos, noise in zip(neighbor_positions, neighbor_noises):
#                     weight = 10 ** (noise / 10)
#                     weights.append(weight)
#                     weighted_coords = weight * np.array(pos)
#                     weighted_sum += weighted_coords
#                     total_weight += weight
#                 if total_weight != 0:
#                     wcl_estimation = weighted_sum / total_weight
#                     node_stats['wcl_coefficient'] = wcl_estimation.tolist()
#
#                     # Plot WCL point on top
#                     plt.scatter(*wcl_estimation, color='red', marker='x', s=200, zorder=3)
#
#             # Highlight the current node on top
#             plt.scatter(*G.nodes[node]['position'], color='blue', s=100, zorder=2)
#
#             # Annotate neighbor nodes with weights
#             for neighbor, weight in zip(neighbors, weights):
#                 plt.annotate(f'{weight:.2f}', xy=G.nodes[neighbor]['position'], textcoords="offset points", xytext=(0,10), ha='center')
#
#             plt.title(f'Graph with WCL for Node {node}')
#             plt.xlabel('X coordinate')
#             plt.ylabel('Y coordinate')
#             plt.grid(True)
#
#             graph_stats.append(node_stats)  # Append the current node's stats to the graph's list
#             plt.show()
#             # if dataset_type == 'random_all_jammed_jammer_outside_region':
#             #     plt.show()
#             # else:
#             #     plt.close()
#
#         all_graph_stats.append(graph_stats)  # Append the completed list of node stats for this graph
#
#     return all_graph_stats


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


def engineer_node_features(data, params):
    logging.info('Calculating node features')
    data['centroid'] = data['node_positions'].apply(lambda positions: np.mean(positions, axis=0))

    if 'dist_to_centroid' in params.get('additional_features', []):
        data['dist_to_centroid'] = data.apply(lambda row: [np.linalg.norm(pos - row['centroid']) for pos in row['node_positions']], axis=1)

    # Cyclical features
    if 'sin_azimuth' in params.get('additional_features', []) or 'cos_azimuth' in params.get('additional_features', []):
        add_cyclical_features(data)

    # Proximity counts
    if 'proximity_count' in params.get('additional_features', []):
        add_proximity_count(data)

    # Clustering coefficients
    if 'clustering_coefficient' in params.get('additional_features', []):
        graphs = create_graphs(data)
        clustering_coeffs = add_clustering_coefficients(graphs)
        data['clustering_coefficient'] = clustering_coeffs  # Assign the coefficients directly

    # Graph-based noise stats
    graph_stats = ['mean_noise', 'median_noise', 'std_noise', 'range_noise', 'relative_noise', 'wcl_coefficient']
    noise_stats_to_compute = [stat for stat in graph_stats if stat in params.get('additional_features', [])]

    if noise_stats_to_compute:
        jammer_positions = data['jammer_position'].tolist()
        # dataset_list = data['dataset'].tolist()
        graphs = create_graphs(data)
        # node_noise_stats = calculate_noise_statistics(graphs, jammer_positions, noise_stats_to_compute)
        node_noise_stats = calculate_noise_statistics(graphs, noise_stats_to_compute)

        for stat in noise_stats_to_compute:
            # Initialize the column for the statistic
            data[stat] = pd.NA

            # Assign the stats for each graph to the DataFrame
            for idx, graph_stats in enumerate(node_noise_stats):
                current_stat_list = [node_stats.get(stat) for node_stats in graph_stats if stat in node_stats]
                data.at[idx, stat] = current_stat_list

    return data

    # if params['3d']:
    #     if 'elevation_angle' in params.get('additional_features', []):
    #         data['elevation_angle'] = data.apply(
    #             lambda row: [np.arcsin((pos[2] - row['centroid'][2]) / np.linalg.norm(pos - row['centroid'])) for pos in row['node_positions']], axis=1)


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
    convert_data_type(data)
    center_coordinates(data)
    standardize_data(data)
    engineer_node_features(data, params)
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


# def undo_center_coordinates(centered_coords, id, midpoints):
#     """
#     Adjust coordinates by adding the midpoint, calculated from stored midpoints for each index.
#
#     Args:
#         centered_coords (np.ndarray): The centered coordinates to be adjusted.
#         id (int): The unique identifier for the data sample.
#         midpoints (dict): The dictionary containing midpoints for each ID.
#
#     Returns:
#         np.ndarray: The uncentered coordinates.
#     """
#     midpoint = np.array(midpoints[str(id)])  # Convert index to string if it's an integer
#     return centered_coords + midpoint


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
        # # 0. Reverse to cartesian
        # if params['coords'] == 'polar' and data_type == 'prediction':
        #     output = cyclical_to_angular(output)
        #     output = polar_to_cartesian(output)

        # 1. Reverse normalization using min_coords and max_coords
        min_coords = data_batch.min_coords.to(device).view(-1, 2)  # Reshape from [32] to [16, 2]
        max_coords = data_batch.max_coords.to(device).view(-1, 2)

        range_coords = max_coords - min_coords
        converted_output = (output + 1) / 2 * range_coords + min_coords

    elif params['norm'] == 'unit_sphere':
        # # 0. Reverse to cartesian
        # if params['coords'] == 'polar' and data_type == 'prediction':
        #     output = cyclical_to_angular(output)
        #     output = polar_to_cartesian(output)

        # 1. Reverse unit sphere normalization using max_radius
        # print("output: ", output)
        max_radius = data_batch.max_radius.to(device).view(-1, 1)
        # print("max radius: ", max_radius)
        converted_output = output * max_radius
        # print("converted output: ", converted_output)
        # quit()

    # 2. Reverse centering using the stored node_positions_center
    centers = data_batch.node_positions_center.to(device).view(-1, 2)
    converted_output += centers

    return torch.tensor(converted_output, device=device)


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


def split_datasets(preprocessed_data, data, params, experiments_path):
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
    torch_geo_dataset = [create_torch_geo_data(row, params) for _, row in preprocessed_data.iterrows()]

    # Stratified split using scikit-learn
    train_idx, test_idx, train_test_y, test_y = train_test_split(
        np.arange(len(torch_geo_dataset)),
        preprocessed_data['dataset'],
        test_size=0.3,  # Split 70% train, 30% test
        stratify=preprocessed_data['dataset'],
        random_state=100  # For reproducibility
    )

    # Now split the test into validation and test
    val_idx, test_idx, _, _ = train_test_split(
        test_idx,
        test_y,
        test_size=len(torch_geo_dataset) - len(train_idx) - int(0.1 * len(torch_geo_dataset)),
        stratify=test_y,
        random_state=100
    )

    train_dataset = [torch_geo_dataset[i] for i in train_idx]
    val_dataset = [torch_geo_dataset[i] for i in val_idx]
    test_dataset = [torch_geo_dataset[i] for i in test_idx]

    # Convert indices back to DataFrame subsets
    train_df = preprocessed_data.iloc[train_idx].reset_index(drop=True)
    val_df = preprocessed_data.iloc[val_idx].reset_index(drop=True)
    test_df = preprocessed_data.iloc[test_idx].reset_index(drop=True)

    # Apply the same indices to the raw data
    raw_test_df = data.iloc[test_idx].reset_index(drop=True)

    return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, raw_test_df



def save_datasets(combined_train_data, combined_val_data, combined_test_data, combined_raw_test_data, combined_train_df, combined_val_df, combined_test_df, combined_raw_test_df, experiments_path):
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

    # Save the combined datasets
    save_reduced_dataset(combined_train_data, list(range(len(combined_train_data))), os.path.join(experiments_path, 'train_dataset.pt'))
    save_reduced_dataset(combined_val_data, list(range(len(combined_val_data))), os.path.join(experiments_path, 'val_dataset.pt'))
    save_reduced_dataset(combined_test_data, list(range(len(combined_test_data))), os.path.join(experiments_path, 'test_dataset.pt'))

    # Save the combined DataFrame subsets
    combined_train_df.to_csv(os.path.join(experiments_path, 'train_dataset.csv'), index=False)
    combined_val_df.to_csv(os.path.join(experiments_path, 'val_dataset.csv'), index=False)
    combined_test_df.to_csv(os.path.join(experiments_path, 'test_dataset.csv'), index=False)
    combined_raw_test_df.to_csv(os.path.join(experiments_path, 'raw_test_data.csv'), index=False)

    # Dataset types for specific filtering
    dataset_types = ['circle', 'triangle', 'rectangle', 'random', 'circle_jammer_outside_region',
                     'triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
                     'random_jammer_outside_region']

    for dataset in dataset_types:
        train_indices = combined_train_df[combined_train_df['dataset'] == dataset].index.tolist()
        val_indices = combined_val_df[combined_val_df['dataset'] == dataset].index.tolist()
        test_indices = combined_test_df[combined_test_df['dataset'] == dataset].index.tolist()

        if train_indices:
            save_reduced_dataset(combined_train_data, train_indices, os.path.join(experiments_path, f'{dataset}_train_set.pt'))
        if val_indices:
            save_reduced_dataset(combined_val_data, val_indices, os.path.join(experiments_path, f'{dataset}_val_set.pt'))
        if test_indices:
            save_reduced_dataset(combined_test_data, test_indices, os.path.join(experiments_path, f'{dataset}_test_set.pt'))

    # Special cases for "all_jammed" and "all_jammed_jammer_outside_region"
    all_jammed_train_indices = combined_train_df[(combined_train_df['dataset'].str.contains('all_jammed')) & (~combined_train_df['dataset'].str.contains('jammer_outside_region'))].index.tolist()
    all_jammed_val_indices = combined_val_df[(combined_val_df['dataset'].str.contains('all_jammed')) & (~combined_val_df['dataset'].str.contains('jammer_outside_region'))].index.tolist()
    all_jammed_test_indices = combined_test_df[(combined_test_df['dataset'].str.contains('all_jammed')) & (~combined_test_df['dataset'].str.contains('jammer_outside_region'))].index.tolist()

    all_jammed_jammer_outside_region_train_indices = combined_train_df[combined_train_df['dataset'].str.contains('all_jammed_jammer_outside_region')].index.tolist()
    all_jammed_jammer_outside_region_val_indices = combined_val_df[combined_val_df['dataset'].str.contains('all_jammed_jammer_outside_region')].index.tolist()
    all_jammed_jammer_outside_region_test_indices = combined_test_df[combined_test_df['dataset'].str.contains('all_jammed_jammer_outside_region')].index.tolist()

    # TODO:
    if all_jammed_train_indices:
        save_reduced_dataset(combined_train_data, all_jammed_train_indices, os.path.join(experiments_path, 'all_jammed_train_set.pt'))
    if all_jammed_val_indices:
        save_reduced_dataset(combined_val_data, all_jammed_val_indices, os.path.join(experiments_path, 'all_jammed_val_set.pt'))
    if all_jammed_test_indices:
        save_reduced_dataset(combined_test_data, all_jammed_test_indices, os.path.join(experiments_path, 'all_jammed_test_set.pt'))

    if all_jammed_jammer_outside_region_train_indices:
        save_reduced_dataset(combined_train_data, all_jammed_jammer_outside_region_train_indices, os.path.join(experiments_path, 'all_jammed_jammer_outside_region_train_set.pt'))
    if all_jammed_jammer_outside_region_val_indices:
        save_reduced_dataset(combined_val_data, all_jammed_jammer_outside_region_val_indices, os.path.join(experiments_path, 'all_jammed_jammer_outside_region_val_set.pt'))
    if all_jammed_jammer_outside_region_test_indices:
        save_reduced_dataset(combined_test_data, all_jammed_jammer_outside_region_test_indices, os.path.join(experiments_path, 'all_jammed_jammer_outside_region_test_set.pt'))

    quit()


def load_data(params, train_set_name, val_set_name, test_set_name, experiments_path=None, data=None):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    logging.info("Loading data...")

    if params['save_data']:
        combined_train_data = []
        combined_val_data = []
        combined_test_data = []
        combined_raw_test_data = []
        combined_train_df = pd.DataFrame()
        combined_val_df = pd.DataFrame()
        combined_test_df = pd.DataFrame()
        combined_raw_test_df = pd.DataFrame()

        if params['all_env_data']:
            datasets = ['data/train_test_data/log_distance/urban_area/combined_urban_area.csv', 'data/train_test_data/log_distance/shadowed_urban_area/combined_shadowed_urban_area.csv']
            # datasets = ['data/train_test_data/log_distance/urban_area/rectangle.csv'] # circle_jammer_outside_region
        else:
            datasets = [params['dataset_path']]
        for dataset in datasets:
            print(f"dataset: {dataset}")
            data = pd.read_csv(dataset)
            data['id'] = range(1, len(data) + 1)

            # Create a deep copy of the DataFrame
            data_to_preprocess = data.copy(deep=True)
            preprocessed_data = preprocess_data(data_to_preprocess, params)
            train_data, val_data, test_data, train_df, val_df, test_df, raw_test_df = split_datasets(preprocessed_data, data, params, experiments_path)

            combined_train_data.extend(train_data)
            combined_val_data.extend(val_data)
            combined_test_data.extend(test_data)
            combined_raw_test_data.extend(raw_test_df)
            combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
            combined_val_df = pd.concat([combined_val_df, val_df], ignore_index=True)
            combined_test_df = pd.concat([combined_test_df, test_df], ignore_index=True)
            combined_raw_test_df = pd.concat([combined_raw_test_df, raw_test_df], ignore_index=True)

        # Process and save the combined data
        save_datasets(combined_train_data, combined_val_data, combined_test_data, combined_raw_test_data,
                                       combined_train_df, combined_val_df, combined_test_df, combined_raw_test_df,
                                       experiments_path)
    else:
        # TODO: check what to be returned for the original dataset for plotting
        train_dataset = torch.load(os.path.join(experiments_path, train_set_name))
        val_dataset = torch.load(os.path.join(experiments_path, val_set_name))
        test_dataset = torch.load(os.path.join(experiments_path, test_set_name))
        # test_dataset_csv = pd.read_csv(experiments_path + test_set)

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
    if not params['inference']:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return None, None, test_loader


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
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error


def plot_graph(positions, edge_index, node_features, edge_weights=None, show_weights=False):
    G = nx.Graph()

    # Ensure positions and features are numpy arrays for easier handling
    positions = np.array(positions)
    node_features = np.array(node_features)

    print("node features 0: ", node_features[0])
    print("node features 1: ", node_features[1])
    print("node features 2: ", node_features[2])
    print("node features 3: ", node_features[3])

    # Add nodes with features and positions
    for i, pos in enumerate(positions):
        # Example feature: assuming RSSI is the last feature in node_features array
        print("pos[0]: ", pos[0])
        print("pos[1]: ", pos[1])
        print("node_features: ", node_features)
        G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2])

    # Convert edge_index to a usable format if it's a tensor or similar
    print("edge_index: ", edge_index)
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

    # Node labels
    node_labels = {i: f"{i}\nNoise:{G.nodes[i]['noise']:.2f}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Optionally draw edge weights
    if show_weights and edge_weights is not None:
        edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

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
    additional_features = params['additional_features']

    if isinstance(additional_features, tuple):
        additional_features = list(additional_features)

    if isinstance(required_features, tuple):
        required_features = list(required_features)

    # # if additional_features:
    # # Check for the presence of sin_azimuth or cos_azimuth
    # has_sin = 'sin_azimuth' in additional_features
    # has_cos = 'cos_azimuth' in additional_features
    #
    # # If one is present and not the other, add the missing one
    # if has_sin and not has_cos:
    #     additional_features.append('cos_azimuth')
    # elif has_cos and not has_sin:
    #     additional_features.append('sin_azimuth')

    # Combine required and additional features
    all_features = required_features + additional_features
    return all_features
    # else:
    #     return required_features


def create_torch_geo_data(row: pd.Series, params) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """
    # Selecting features based on configuration
    all_features = ensure_complementary_features(params)

    # Combining features from each node into a single list
    node_features = [
        sum(([feature_value] if not isinstance(feature_value, list) else feature_value
             for feature_value in node_data), [])
        for node_data in zip(*(row[feature] for feature in all_features))
    ]

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
    # if row['dataset'] == 'triangle':
    #     plot_graph(positions=positions, edge_index=edge_index, node_features=node_features, edge_weights=edge_weight, show_weights=True)

    # Create the Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # Convert geometric information to tensors
    data.id = row['id']  # Assign the id from the row to the Data object
    data.node_positions_center = torch.tensor(row['node_positions_center'], dtype=torch.float)
    if params['norm'] == 'minmax':
        data.min_coords = torch.tensor(row['min_coords'], dtype=torch.float)
        data.max_coords = torch.tensor(row['max_coords'], dtype=torch.float)
    elif params['norm'] == 'unit_sphere':
        data.max_radius = torch.tensor(row['max_radius'], dtype=torch.float)

    return data
