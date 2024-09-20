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
import pickle
from utils import cartesian_to_polar
from custom_logging import setup_logging
from config import params

from torch_geometric.utils import to_networkx

setup_logging()


from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(self, data, mode='train'):
        """
        Initialize the dataset with preprocessed DataFrame.
        """
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # print("row: ", row)
        graph = create_torch_geo_data(row)  # Create PyTorch Geometric Data object
        graph = engineer_node_features(graph)
        # print("graph: ", graph)
        return graph


import random
#
#
# class BufferedDataLoader:
#     def __init__(self, dataset, batch_size, shuffle=False, num_workers=1, collate_fn=None):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_workers = num_workers
#         self.collate_fn = collate_fn
#         self.buffer = []
#
#     def __iter__(self):
#         dataset_indices = list(range(len(self.dataset)))
#
#         if self.shuffle:
#             random.shuffle(dataset_indices)
#
#         # Create a ThreadPoolExecutor to parallelize data loading
#         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             futures = []
#
#             for idx in dataset_indices:
#                 if len(futures) >= self.num_workers:
#                     for future in as_completed(futures):
#                         data = future.result()
#                         processed_data = self.collate_fn(data) if self.collate_fn else data
#                         self.buffer.extend(processed_data)
#
#                         while len(self.buffer) >= self.batch_size:
#                             yield self.buffer[:self.batch_size]
#                             self.buffer = self.buffer[self.batch_size:]
#
#                         futures.remove(future)
#
#                 # Submit new loading task
#                 futures.append(executor.submit(self.dataset.__getitem__, idx))
#
#             # Process remaining futures
#             for future in as_completed(futures):
#                 data = future.result()
#                 processed_data = self.collate_fn(data) if self.collate_fn else data
#                 self.buffer.extend(processed_data)
#
#                 while len(self.buffer) >= self.batch_size:
#                     yield self.buffer[:self.batch_size]
#                     self.buffer = self.buffer[self.batch_size:]
#
#         if self.buffer:
#             yield self.buffer
#             self.buffer = []  # Clear buffer after last batch
#
#     def __len__(self):
#         # This might not be exact if data expands unpredictably
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# old version with buffering
# class TemporalGraphDataset(torch.utils.data.Dataset):
#     def __init__(self, train_dataset, val_dataset, test_dataset):
#         self.datasets = {
#             'train': train_dataset,
#             'val': val_dataset,
#             'test': test_dataset
#         }
#         self.lengths = {key: len(dataset) for key, dataset in self.datasets.items()}
#
#     def __len__(self):
#         return sum(self.lengths.values())
#
#     def get_dataset(self, dataset_type):
#         return self.datasets[dataset_type], self.lengths[dataset_type]
#
#     def load_data(self, index, dataset_type='test'):
#         dataset, _ = self.get_dataset(dataset_type)
#         return dataset[index]
#
#     def transform(self, graph, evaluate=False):
#         if not evaluate:
#             return self.random_crop(graph)
#         else:
#             return self.incremental_node_addition(graph)
#
#     def __getitem__(self, index, dataset_type='test', evaluate=False):
#         graph = self.load_data(index, dataset_type)
#         transformed_graph = self.transform(graph, evaluate)
#         return transformed_graph
#
#     def random_crop(self, graph):
#         print("graph: ", graph)
#         num_nodes = graph.x.size(0)
#         start_index = np.random.randint(0, num_nodes)
#         end_index = np.random.randint(start_index + 1, num_nodes + 1)
#         return self.extract_subgraph(graph, start_index, end_index)
#
#     def incremental_node_addition(self, graph):
#         # This will generate a sequence of subgraphs, each with one more node than the last
#         subgraphs = []
#         for end_index in range(1, graph.x.size(0) + 1):
#             subgraph = self.extract_subgraph(graph, 0, end_index)
#             subgraphs.append(subgraph)
#         return subgraphs
#
#     def extract_subgraph(self, graph, start_index, end_index):
#         node_slice = torch.arange(start_index, end_index)
#         edge_mask = ((graph.edge_index[0] >= start_index) & (graph.edge_index[0] < end_index) &
#                      (graph.edge_index[1] >= start_index) & (graph.edge_index[1] < end_index))
#         edge_index = graph.edge_index[:, edge_mask] - start_index
#         subgraph = Data(x=graph.x[node_slice], edge_index=edge_index, edge_attr=graph.edge_attr[edge_mask] if graph.edge_attr is not None else None, y=graph.y, min_coords=graph.min_coords, max_coords=graph.max_coords,node_positions_center=graph.node_positions_center)
#         # plot_graph_temporal(subgraph)
#
#         # here, before engineering node features, need to perform preprocessing
#         # cant create graphs before data preprocessing because otherwise will create edges/node features with unprocessed data (no centering norm etc)
#         # maybe instead can create slices then call data preprocessing then create torch geo data then call the rest of the func
#         # change increment function to increment on csv data not graph data
#
#         subgraph = engineer_node_features(subgraph, params)
#         return subgraph


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
        # print("idx: ", idx)
        # print("row: ", row)
        # Center coordinates using the calculated mean
        node_positions = np.vstack(row['node_positions'])
        centered_node_positions, center = mean_centering(node_positions)

        # Convert to list and check structure
        centered_list = centered_node_positions.tolist()
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
    dataset_features = ['jammer_position', 'node_positions', 'node_noise', 'timestamps', 'angle_of_arrival']

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



# Original!
# def calculate_noise_statistics(graphs, stats_to_compute):
#     all_graph_stats = []  # This will hold a list of lists, each sublist for a graph
#
#     for G in graphs:
#         graph_stats = []  # Initialize an empty list for current graph's node stats
#
#         for node in G.nodes:
#             neighbors = list(G.neighbors(node))
#             curr_node_noise = G.nodes[node]['noise']
#             neighbor_noises = [G.nodes[neighbor]['noise'] for neighbor in neighbors]  # Neighbors' noise
#             neighbor_positions = [G.nodes[neighbor]['position'] for neighbor in neighbors]  # Neighbors' positions
#
#             node_stats = {}  # Dictionary to store stats for the current node
#
#             # Compute mean noise for neighbors excluding the node itself
#             if neighbors:  # Ensure there are neighbors
#                 mean_neighbor_noise = np.mean(neighbor_noises)
#             else:
#                 mean_neighbor_noise = curr_node_noise  # If no neighbors, fallback to own noise
#
#             if 'mean_noise' in stats_to_compute:
#                 node_stats['mean_noise'] = mean_neighbor_noise
#             if 'median_noise' in stats_to_compute:
#                 node_stats['median_noise'] = np.median(neighbor_noises + [curr_node_noise])  # Include self for median
#             if 'std_noise' in stats_to_compute:
#                 node_stats['std_noise'] = np.std(neighbor_noises + [curr_node_noise])  # Include self for std
#             if 'range_noise' in stats_to_compute:
#                 node_stats['range_noise'] = np.max(neighbor_noises + [curr_node_noise]) - np.min(neighbor_noises + [curr_node_noise])
#
#             # Compute relative noise
#             node_stats['relative_noise'] = curr_node_noise - mean_neighbor_noise
#
#             # Compute the weighted centroid local (WCL)
#             if 'wcl_coefficient' in stats_to_compute:
#                 total_weight = 0
#                 weighted_sum = np.zeros(2)  # 2D positions
#                 for pos, noise in zip(neighbor_positions, neighbor_noises):
#                     weight = 10 ** (noise / 10)
#                     weighted_coords = weight * np.array(pos)
#                     weighted_sum += weighted_coords
#                     total_weight += weight
#                 if total_weight != 0:
#                     wcl_estimation = weighted_sum / total_weight
#                     node_stats['wcl_coefficient'] = wcl_estimation.tolist()  # Store as a list if necessary
#
#             graph_stats.append(node_stats)  # Append the current node's stats to the graph's list
#
#         all_graph_stats.append(graph_stats)  # Append the completed list of node stats for this graph
#
#     return all_graph_stats


def calculate_noise_statistics(subgraphs, stats_to_compute):
    all_graph_stats = []

    for subgraph in subgraphs:
        node_stats = []
        edge_index = subgraph.edge_index
        num_nodes = subgraph.num_nodes

        for node_id in range(num_nodes):
            # Identifying the neighbors of the current node
            mask = (edge_index[0] == node_id) | (edge_index[1] == node_id)
            neighbors = edge_index[1][mask] if edge_index[0][mask].eq(node_id).any() else edge_index[0][mask]

            curr_node_noise = subgraph.x[node_id][2]  # Assuming noise is the third feature
            neighbor_noises = subgraph.x[neighbors, 2]

            # Handle case with fewer than two neighbors safely
            if neighbors.size(0) > 1:
                std_noise = neighbor_noises.std().item()
                range_noise = (neighbor_noises.max() - neighbor_noises.min()).item()
            else:
                std_noise = 0
                range_noise = 0

            temp_stats = {
                'mean_noise': neighbor_noises.mean().item() if neighbors.size(0) > 0 else curr_node_noise.item(),
                'median_noise': neighbor_noises.median().item() if neighbors.size(0) > 0 else curr_node_noise.item(),
                'std_noise': std_noise,
                'range_noise': range_noise,
                'relative_noise': (curr_node_noise - neighbor_noises.mean()).item() if neighbors.size(0) > 0 else 0,
            }

            # Compute WCL if needed
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


def engineer_node_features(subgraph):
    if subgraph.x.size(0) == 0:
        raise ValueError("Empty subgraph encountered")

    new_features = []

    # Calculating centroid
    centroid = torch.mean(subgraph.x, dim=0)

    if 'dist_to_centroid' in params['additional_features']:
        distances = torch.norm(subgraph.x - centroid, dim=1, keepdim=True)
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

    if new_features:
        new_features_tensor = torch.cat(new_features, dim=1)
        subgraph.x = torch.cat((subgraph.x, new_features_tensor), dim=1)

    return subgraph


# # Original!
# def engineer_node_features(data, params):
#     logging.info('Calculating node features')
#     data['centroid'] = data['node_positions'].apply(lambda positions: np.mean(positions, axis=0))
#
#     if 'dist_to_centroid' in params.get('additional_features', []):
#         data['dist_to_centroid'] = data.apply(lambda row: [np.linalg.norm(pos - row['centroid']) for pos in row['node_positions']], axis=1)
#
#     # Cyclical features
#     if 'sin_azimuth' in params.get('additional_features', []) or 'cos_azimuth' in params.get('additional_features', []):
#         add_cyclical_features(data)
#
#     # Proximity counts
#     if 'proximity_count' in params.get('additional_features', []):
#         add_proximity_count(data)
#
#     # Clustering coefficients
#     if 'clustering_coefficient' in params.get('additional_features', []):
#         graphs = create_graphs(data)
#         clustering_coeffs = add_clustering_coefficients(graphs)
#         data['clustering_coefficient'] = clustering_coeffs  # Assign the coefficients directly
#
#     # Graph-based noise stats
#     graph_stats = ['mean_noise', 'median_noise', 'std_noise', 'range_noise', 'relative_noise', 'wcl_coefficient']
#     noise_stats_to_compute = [stat for stat in graph_stats if stat in params.get('additional_features', [])]
#
#     if noise_stats_to_compute:
#         jammer_positions = data['jammer_position'].tolist()
#         # dataset_list = data['dataset'].tolist()
#         graphs = create_graphs(data)
#         # node_noise_stats = calculate_noise_statistics(graphs, jammer_positions, noise_stats_to_compute)
#         node_noise_stats = calculate_noise_statistics(graphs, noise_stats_to_compute)
#
#         for stat in noise_stats_to_compute:
#             # Initialize the column for the statistic
#             data[stat] = pd.NA
#
#             # Assign the stats for each graph to the DataFrame
#             for idx, graph_stats in enumerate(node_noise_stats):
#                 current_stat_list = [node_stats.get(stat) for node_stats in graph_stats if stat in node_stats]
#                 data.at[idx, stat] = current_stat_list
#
#     return data
#
#     # if params['3d']:
#     #     if 'elevation_angle' in params.get('additional_features', []):
#     #         data['elevation_angle'] = data.apply(
#     #             lambda row: [np.arcsin((pos[2] - row['centroid'][2]) / np.linalg.norm(pos - row['centroid'])) for pos in row['node_positions']], axis=1)

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
    # engineer_node_features(data, params)  # if engineering node features on temporal subgraphs before passing to dataloader
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
    # torch_geo_dataset = [create_torch_geo_data(row, params) for _, row in preprocessed_data.iterrows()]

    # Stratified split using scikit-learn
    train_idx, test_idx, train_test_y, test_y = train_test_split(
        np.arange(len(preprocessed_data)),
        preprocessed_data['dataset'],
        test_size=0.3,  # Split 70% train, 30% test
        stratify=preprocessed_data['dataset'],
        random_state=100  # For reproducibility
    )

    # Now split the test into validation and test
    val_idx, test_idx, _, _ = train_test_split(
        test_idx,
        test_y,
        test_size=len(preprocessed_data) - len(train_idx) - int(0.1 * len(preprocessed_data)),
        stratify=test_y,
        random_state=100
    )

    # Convert indices back to DataFrame subsets
    train_df = preprocessed_data.iloc[train_idx].reset_index(drop=True)
    val_df = preprocessed_data.iloc[val_idx].reset_index(drop=True)
    test_df = preprocessed_data.iloc[test_idx].reset_index(drop=True)

    # Apply the same indices to the raw data
    raw_test_df = data.iloc[test_idx].reset_index(drop=True)

    # return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, raw_test_df
    return train_df, val_df, test_df, raw_test_df




def save_datasets(combined_train_df, combined_val_df, combined_test_df, combined_raw_test_df, experiments_path):
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
    combined_raw_test_df.to_csv(os.path.join(experiments_path, 'raw_test_data.csv'), index=False)

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
            convert_data_type(data)

            # Create train test splits
            train_df, val_df, test_df, raw_test_df = split_datasets(data, data, params, experiments_path)

            # Apply random_crop to training and test datasets
            train_df = apply_processing(train_df, 'train_val')
            val_df = apply_processing(val_df, 'train_val')
            test_df = apply_processing(test_df, 'test')
            test_df = test_df.reset_index()

            # print("train_df['jammer_position']: ", train_df['jammer_position'])

            print("TRAIN\n\n")
            train_dataset = preprocess_data(train_df, params)
            print("VAL\n\n")
            val_dataset = preprocess_data(val_df, params)
            print("TEST\n\n")
            test_dataset = preprocess_data(test_df, params)

            # print("test_dataset: ", test_dataset.iloc[0])

            # DONt create graphs yet! we will do that in the dataloader because otherwise it will take up too much space to do it beforehand

            combined_raw_test_data.extend(raw_test_df)
            combined_train_df = pd.concat([combined_train_df, train_dataset], ignore_index=True)
            combined_val_df = pd.concat([combined_val_df, val_dataset], ignore_index=True)
            combined_test_df = pd.concat([combined_test_df, test_dataset], ignore_index=True)
            combined_raw_test_df = pd.concat([combined_raw_test_df, raw_test_df], ignore_index=True)

        # Process and save the combined data
        save_datasets(combined_train_df, combined_val_df, combined_test_df, combined_raw_test_df,
                                       experiments_path)
    else:
        # Load train, val and test sets that were saved
        train_dataset = pd.read_csv(os.path.join(experiments_path, train_set_name))
        val_dataset = pd.read_csv(os.path.join(experiments_path, val_set_name))
        test_dataset = pd.read_csv(os.path.join(experiments_path, test_set_name))

    return train_dataset, val_dataset, test_dataset


def random_crop(row, min_nodes=3):
    """
    Perform a random crop of node samples for one row.
    Args:
        row (pd.Series): A single row from a DataFrame.
        min_nodes (int): Minimum number of nodes to keep, default is 3.
    Returns:
        pd.Series: Modified row with cropped data.
    """
    total_nodes = len(row['node_positions'])
    if total_nodes > min_nodes:
        start = np.random.randint(0, total_nodes - min_nodes)
        end = np.random.randint(start + min_nodes, total_nodes)
        for key in ['timestamps', 'node_positions', 'node_noise', 'angle_of_arrival']:
            if key == 'node_positions':
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
    min_nodes = 3
    total_nodes = len(row['node_positions'])
    new_rows = []
    if total_nodes >= min_nodes:
        for i in range(min_nodes, total_nodes + 1):
            new_row = row.copy()
            for key in ['timestamps', 'node_positions', 'node_noise', 'angle_of_arrival']:
                new_row[key] = row[key][:i]
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
    if mode == 'train_val':
        return [random_crop(row)]
    elif mode == 'test':
        return incremental_node_addition(row)


# def create_data_loader(train_dataset, val_dataset, test_dataset, batch_size: int):
#     """
#     Create data loader objects.
#     Args:
#         batch_size (int): Batch size for the DataLoader.
#
#     Returns:
#         train_loader (DataLoader): DataLoader for the training dataset.
#         val_loader (DataLoader): DataLoader for the validation dataset.
#         test_loader (DataLoader): DataLoader for the testing dataset.
#     """
#     logging.info("Creating DataLoader objects...")
#     if not params['inference']:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#
#         return train_loader, val_loader, test_loader
#     else:
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#
#         return None, None, test_loader

from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.data import Batch


# class CustomDataLoader(TorchDataLoader):
#     def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None, **kwargs):
#         super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)


def custom_collate(batch):
    data_list = [item if isinstance(item, Data) else item[0] for item in batch]
    batch = Batch.from_data_list(data_list)
    return batch
#
# def custom_collate_eval(batch):
#     # Assuming each item in 'batch' is a single Data object
#     return batch  # Directly return the list of Data objects, no batching


# def custom_collate_fn(batch):
#     # Flatten out the lists of graphs into a single list
#     batch_graphs = [graph for sublist in batch for graph in sublist]
#
#     # Ensure the batch size remains constant
#     while len(batch_graphs) > params['batch_size']:
#         yield batch_graphs[:params['batch_size']]
#         batch_graphs = batch_graphs[params['batch_size']:]
#
#     if batch_graphs:
#         yield batch_graphs


# def tensor_to_list(tensor):
#     return tensor.detach().cpu().numpy().tolist()

# def process_and_save_data(processed_data, filename='processed_test_data.csv'):
#     data_list = []
#
#     # Flatten the list of lists if necessary
#     flat_list = [item for sublist in processed_data for item in sublist]
#
#     for data in flat_list:
#         data_dict = {
#             'x': tensor_to_list(data.x),
#             'edge_index': tensor_to_list(data.edge_index),
#             'edge_attr': tensor_to_list(data.edge_attr),
#             'y': tensor_to_list(data.y),
#             'min_coords': tensor_to_list(data.min_coords),
#             'max_coords': tensor_to_list(data.max_coords),
#             'node_positions_center': tensor_to_list(data.node_positions_center)
#         }
#         data_list.append(data_dict)
#
#     # Convert list of dictionaries to DataFrame
#     df = pd.DataFrame(data_list)
#     df.to_csv(filename, index=False)
#     print(f"Data saved to {filename}")
#

# def create_data_loader(temporal_dataset, batch_size: int):
#     """
#     Create data loaders for training, validation, and testing sets with custom preprocessing using a custom DataLoader.
#     Args:
#         temporal_dataset (TemporalGraphDataset): The dataset containing train, val, and test sets.
#         batch_size (int): Batch size for the DataLoader.
#     Returns:
#         Tuple[CustomDataLoader, CustomDataLoader, CustomDataLoader]: Custom DataLoader for the training, validation, and testing datasets.
#     """
#     # Fetch datasets
#     train_data, train_length = temporal_dataset.get_dataset('train')
#     val_data, val_length = temporal_dataset.get_dataset('val')
#     test_data, test_length = temporal_dataset.get_dataset('test')
#
#     # Process training and validation data
#     processed_train_data = [temporal_dataset.__getitem__(idx, 'train', evaluate=False) for idx in range(train_length)]
#     processed_val_data = [temporal_dataset.__getitem__(idx, 'val', evaluate=False) for idx in range(val_length)]
#
#     # # Save processed test data to CSV
#     # process_and_save_data(processed_test_data, 'processed_test_data.csv')
#     #
#     # quit()
#
#     # Create DataLoaders for training and validation with preprocessed data
#     train_loader = CustomDataLoader(processed_train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4)
#     val_loader = CustomDataLoader(processed_val_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=4)
#
#     # Create DataLoader for the test set without preprocessing
#     test_loader = CustomDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_eval, num_workers=4)
#
#     return train_loader, val_loader, test_loader


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
    # Instantiate the dataset classes for train, val, and test
    train_dataset = TemporalGraphDataset(train_data, mode='train')
    val_dataset = TemporalGraphDataset(val_data, mode='val')
    test_dataset = TemporalGraphDataset(test_data, mode='test')

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # TODO: do we need to add drop_last=True??
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



# def create_data_loader(temporal_dataset, batch_size: int):
#     """
#     Create data loader objects for train, validation, and test datasets using temporal slicing.
#     Args:
#         temporal_dataset (TemporalGraphDataset): The dataset containing train, val, and test sets.
#         batch_size (int): Batch size for the DataLoader.
#
#     Returns:
#         train_loader (DataLoader): DataLoader for the training dataset.
#         val_loader (DataLoader): DataLoader for the validation dataset.
#         test_loader (DataLoader): DataLoader for the testing dataset.
#     """
#     logging.info("Creating DataLoader objects...")
#     train_data, train_length = temporal_dataset.get_dataset('train')
#     val_data, val_length = temporal_dataset.get_dataset('val')
#     test_data, test_length = temporal_dataset.get_dataset('test')
#
#     # Create data loaders with evaluation mode specified for validation and test sets
#     train_loader = DataLoader([temporal_dataset.__getitem__(idx, dataset_type='train', evaluate=False) for idx in range(train_length)], batch_size=params['batch_size'], shuffle=True, collate_fn=custom_collate, num_workers=4)
#     val_loader = DataLoader([temporal_dataset.__getitem__(idx, dataset_type='val', evaluate=False) for idx in range(val_length)], batch_size=params['batch_size'],shuffle=False, collate_fn=custom_collate, num_workers=4)
#     test_loader = DataLoader([temporal_dataset.__getitem__(idx, dataset_type='test', evaluate=True) for idx in range(test_length)], batch_size=params['batch_size'],shuffle=False, collate_fn=custom_collate, num_workers=4)
#
#     # Plot
#     # for batch_data in train_loader:
#     #     plot_graph_temporal(batch_data)
#
#     return train_loader, val_loader, test_loader


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
        # Example feature: assuming RSSI is the last feature in node_features array
        G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2],
                   timestamp=node_features[i][-1], sin_aoa=node_features[i][-3], cos_aoa=node_features[i][-2])

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
    node_labels = {i: f"ID:{i}\nNoise:{G.nodes[i]['noise']:.2f}\nTimestamp:{G.nodes[i]['timestamp']:.2f}\nSin AoA:{G.nodes[i]['sin_aoa']:.2f}\nCos AoA:{G.nodes[i]['cos_aoa']:.2f}"
                   for i in G.nodes()}
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
    # print("row in torch geo data: ", row)
    # Handling Angle of Arrival (AoA)
    aoa = np.array(row['angle_of_arrival'])
    sin_aoa = np.sin(aoa)
    cos_aoa = np.cos(aoa)

    # Convert timestamps to percentage completion
    timestamps = np.array(row['timestamps'])
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    perc_completion = (timestamps - min_time) / (max_time - min_time) if max_time != min_time else np.zeros_like(timestamps)

    # Select and combine features
    all_features = ensure_complementary_features(params)
    all_features.remove('angle_of_arrival')  # Remove original AoA feature
    all_features.remove('timestamps')  # Remove original timestamps to replace with normalized ones

    node_features = [
        sum(([feature_value] if not isinstance(feature_value, list) else feature_value
             for feature_value in node_data), [])
        for node_data in zip(*(row[feature] for feature in all_features))
    ]

    # Append AoA and percentage completion as new features
    for i, feature_list in enumerate(node_features):
        feature_list.extend([sin_aoa[i], cos_aoa[i], perc_completion[i]])

    # print("all_features: ", all_features)
    # print("node_features: ", node_features)
    # quit()

    # Convert to PyTorch tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Preparing edges and weights
    positions = np.array(row['node_positions'])
    # print("positions: ", positions)
    # print("len(positions): ", len(positions), '\n')
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
    # print("jammer_positions: ", jammer_positions)
    y = torch.tensor(jammer_positions, dtype=torch.float)

    # Plot
    # if row['dataset'] == 'triangle':
    # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True)

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

    # Store the perc_completion as part of the Data object
    data.perc_completion = torch.tensor(perc_completion, dtype=torch.float32)

    return data
