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
import logging
import pickle
from utils import set_seeds_and_reproducibility, cartesian_to_polar
from custom_logging import setup_logging
from config import params

setup_logging()

if params['reproduce']:
    set_seeds_and_reproducibility()


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


def add_node_noise_statistical_features(data):
    """Calculate statistical features for node noise."""
    data['mean_noise'] = data['node_noise'].apply(np.mean)
    data['median_noise'] = data['node_noise'].apply(np.median)
    data['std_noise'] = data['node_noise'].apply(np.std)
    data['range_noise'] = data['node_noise'].apply(lambda x: np.max(x) - np.min(x))


def calculate_proximity_metric(positions, threshold=0.1):
    """Calculate the number of nearby nodes within a given threshold distance."""
    nbrs = NearestNeighbors(radius=threshold).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions)
    return [len(idx) - 1 for idx in indices]  # subtract 1 to exclude the node itself


def add_proximity_count(data):
    """Add proximity feature based on a threshold distance."""
    data['proximity_count'] = data['node_positions'].apply(
        lambda positions: calculate_proximity_metric(np.array(positions))
    )


def create_graphs(data, threshold=0.2):
    graphs = []

    for index, row in data.iterrows():
        G = nx.Graph()
        positions = np.array(row['node_positions'])
        num_nodes = positions.shape[0]

        # First, add all nodes to the graph
        for i in range(num_nodes):
            G.add_node(i, pos=positions[i])  # Optional: Store positions as node attributes

        # Then, evaluate and add edges based on the proximity threshold
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < threshold:
                    G.add_edge(i, j)

        graphs.append(G)

        # # Visualization and debugging output
        # pos = nx.get_node_attributes(G, 'pos')  # Get positions for drawing
        # nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='#FF5733', node_size=500)
        # plt.title(f"Graph Visualization for Row {index}")
        # plt.show()

    return graphs


def add_clustering_coefficients(data, graphs):
    """
    Compute and add the clustering coefficient for each graph, mapping them to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame where each row corresponds to a graph.
        graphs (list): List of NetworkX graph objects, each corresponding to a row in 'data'.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column for clustering coefficients.
    """
    # Initialize a list to hold clustering coefficients for each node in each graph
    all_clustering_coeffs = []

    # Iterate through each graph and compute clustering coefficients
    for graph in graphs:
        if len(graph.nodes()) > 0:  # Ensure the graph has nodes
            clustering_coeffs = nx.clustering(graph)
            # Collect the clustering coefficients in order of nodes
            # Assuming node labels in the graph correspond to their positions in the node list
            graph_clustering_coeffs = [clustering_coeffs.get(node) for node in graph.nodes()]
        else:
            graph_clustering_coeffs = []

        all_clustering_coeffs.append(graph_clustering_coeffs)

    # Assign the list of clustering coefficients to the corresponding row in the DataFrame
    data['clustering_coefficient'] = all_clustering_coeffs

    return data


def engineer_node_features(data, params):
    # Check if features are in additional features and calculate accordingly
    data['centroid'] = data['node_positions'].apply(lambda positions: np.mean(positions, axis=0))

    if 'dist_to_centroid' in params.get('additional_features', []):
        data['dist_to_centroid'] = data.apply(lambda row: [np.linalg.norm(pos - row['centroid']) for pos in row['node_positions']], axis=1)

    if 'relative_noise' in params.get('additional_features', []):
        data['relative_noise'] = data.apply(lambda row: [noise - np.mean(row['node_noise']) for noise in row['node_noise']], axis=1)

    if 'mean_noise' in params.get('additional_features', []):
        # Calculate the mean and repeat it num_samples times
        data['mean_noise'] = data.apply(lambda row: [np.mean(row['node_noise'])] * row['num_samples'], axis=1)

    if 'median_noise' in params.get('additional_features', []):
        # Calculate the median and repeat it num_samples times
        data['median_noise'] = data.apply(lambda row: [np.median(row['node_noise'])] * row['num_samples'], axis=1)

    if 'std_noise' in params.get('additional_features', []):
        # Calculate the standard deviation and repeat it num_samples times
        data['std_noise'] = data.apply(lambda row: [np.std(row['node_noise'])] * row['num_samples'], axis=1)

    if 'range_noise' in params.get('additional_features', []):
        # Calculate the range (max - min) and repeat it num_samples times
        data['range_noise'] = data.apply(lambda row: [(np.max(row['node_noise']) - np.min(row['node_noise']))] * row['num_samples'], axis=1)

    if 'sin_azimuth' or 'cos_azimuth' in params.get('additional_features', []):
        add_cyclical_features(data)

    if 'proximity_count' in params.get('additional_features', []):
        add_proximity_count(data)

    if 'clustering_coefficient' in params.get('additional_features', []):
        # Adjust 'data' according to your actual data structure
        graphs = create_graphs(data)
        add_clustering_coefficients(data, graphs)

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

    # 1. Reverse normalization using min_coords and max_coords
    if params['norm'] == 'minmax':
        min_coords = data_batch.min_coords.to(device).view(-1, 2)  # Reshape from [32] to [16, 2]
        max_coords = data_batch.max_coords.to(device).view(-1, 2)

        range_coords = max_coords - min_coords
        converted_output = (output + 1) / 2 * range_coords + min_coords

    elif params['norm'] == 'unit_sphere':
        # 0. Reverse to cartesian
        if params['coords'] == 'polar' and data_type == 'prediction':
            output = polar_to_cartesian(output)

        # 1. Reverse unit sphere normalization using max_radius
        max_radius = data_batch.max_radius.to(device).view(-1, 1)
        converted_output = output * max_radius

    # 2. Reverse centering using the stored node_positions_center
    centers = data_batch.node_positions_center.to(device).view(-1, 2)
    converted_output += centers

    return torch.tensor(converted_output, device=device)


def convert_output(output, device):
    output = output.to(device)  # Ensure the output is on the correct device
    if params['coords'] == 'polar':
        converted_output = polar_to_cartesian(output)
        return converted_output
    return output  # If not polar, just pass the output through


def save_datasets(preprocessed_data, data, params):
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

    logging.info('Creating graphs...')
    torch_geo_dataset = [create_torch_geo_data(row, params) for _, row in preprocessed_data.iterrows()]

    if params['inference']:
        logging.info("Inference mode: creating test dataset...")
        test_dataset = torch.utils.data.Subset(torch_geo_dataset, np.arange(len(torch_geo_dataset)))
        return None, None, test_dataset

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

    # Saving data
    if params['save_data']:
        logging.info("Saving data...")
        # Extract indices for each split
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices
        experiments_path = 'experiments_datasets/' + params['coords'] + '_' + params['edges'] + '_' + params['norm'] + '/' + params['dataset'] + '/' + 'trial' + str(params['trial_num']) + '/'

        # Use these indices to split the DataFrame
        train_df = data.iloc[train_indices].reset_index(drop=True)
        val_df = data.iloc[val_indices].reset_index(drop=True)
        test_df = data.iloc[test_indices].reset_index(drop=True)

        # Save raw dataframes before preprocessing
        train_path = experiments_path + 'train_df.gzip'
        val_path = experiments_path + 'validation_df.gzip'
        test_path = experiments_path + 'test_df.gzip'
        with gzip.open(train_path, 'wb') as f:
            pickle.dump(train_df, f)
        with gzip.open(val_path, 'wb') as f:
            pickle.dump(val_df, f)
        with gzip.open(test_path, 'wb') as f:
            pickle.dump(test_df, f)

        # Save test dataframe as CSV
        test_path = experiments_path + 'test_df.csv'  # Change file extension to .csv
        test_df.to_csv(test_path, index=False)  # Save without row indices

        # # Save graphs
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


def load_data(dataset_path: str, params, data=None):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    logging.info("Loading data...")
    if data is None:
        data = pd.read_csv(dataset_path)
    data['id'] = range(1, len(data) + 1)
    # data.drop(columns=['jammer_power', 'pl_exp', 'sigma'], inplace=True)

    # Create a deep copy of the DataFrame
    data_to_preprocess = data.copy(deep=True)
    preprocessed_data = preprocess_data(data_to_preprocess, params)
    train_dataset, val_dataset, test_dataset = save_datasets(preprocessed_data, data, params)

    return train_dataset, val_dataset, test_dataset, data


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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Check for the presence of sin_azimuth or cos_azimuth
    has_sin = 'sin_azimuth' in additional_features
    has_cos = 'cos_azimuth' in additional_features

    # If one is present and not the other, add the missing one
    if has_sin and not has_cos:
        additional_features.append('cos_azimuth')
    elif has_cos and not has_sin:
        additional_features.append('sin_azimuth')

    # Combine required and additional features
    all_features = required_features + additional_features
    return all_features


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
    # additional_features = list(params['additional_features'])
    # all_features = ['node_positions', 'node_noise'] + additional_features

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
    elif params['edges'] == 'proximity':
        edge_index, edge_weight = [], []
        num_nodes = len(row['node_positions'])
        for i in range(num_nodes):
            edge_index.append([i, i])
            edge_weight.append(0)
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 0.1:  # Proximity threshold
                    edge_index.extend([[i, j], [j, i]])
                    edge_weight.extend([dist, dist])
    else:
        raise ValueError("Unsupported edge specification")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    jammer_positions = np.array(row['jammer_position']).reshape(-1, 2)  # Assuming this reshaping is valid based on your data structure
    y = torch.tensor(jammer_positions, dtype=torch.float)

    # Plot
    # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features, edge_weights=edge_weight, show_weights=True)

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
