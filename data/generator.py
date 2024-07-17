import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
LIGHT_SPEED = 3e8  # Speed of light (m/s), used in path loss calculations
FREQUENCY = 2.4e9  # Operating frequency in Hz (e.g., 2.4 GHz for WiFi)
NOISE_LEVEL = -100  # Ambient noise level at the receiver in dBm [-100, -90]
REFERENCE_SINR = 10  # Typical SINR in dB under normal conditions
NOISE_THRESHOLD = -55  # A jamming attack is deemed successful if the floor noise > NOISE_THRESHOLD
RSSI_THRESHOLD = -80  # Minimum RSSI threshold (represents the point where communication is effectively non-functional)
MESH_NETWORK = True  # Set to True for mesh network, False for AP-client network
PLOT = True
POS_SAMPLING_STRAT = 'random'


def dbm_to_linear(dbm):
    """Convert dBm to linear scale (milliwatts)."""
    return 10 ** (dbm / 10)


def linear_to_db(linear):
    """Convert linear scale (milliwatts) to dB."""
    return 10 * np.log10(linear)


def log_distance_path_loss(d, pl0=32, d0=1, n=2.1, sigma=2):
    # Prevent log of zero if distance is zero by replacing it with a very small positive number
    d = np.where(d == 0, np.finfo(float).eps, d)
    # Calculate the path loss
    path_loss = pl0 + 10 * n * np.log10(d / d0)
    # Add shadow fading if sigma is not zero
    if sigma != 0:
        path_loss += np.random.normal(0, sigma, size=d.shape)
    return path_loss


def free_space_path_loss(d):
    """Calculate free space path loss given distance d in meters."""
    # Replace zero distances with a very small positive number
    d = np.where(d == 0, np.finfo(float).eps, d)
    return 20 * np.log10(d) + 20 * np.log10(FREQUENCY) + 20 * np.log10(4 * np.pi / LIGHT_SPEED)  # FSPL formula


def sample_path_jamming(node_pos_i, node_pos_j, jammer_pos, loss_func, n, sigma, num_samples=10):
    # Linearly interpolate points between node_pos_i and node_pos_j
    line_points = np.linspace(node_pos_i, node_pos_j, num=num_samples, endpoint=True)
    max_jamming_power_dbm = -np.inf
    for point in line_points:
        # Calculate distance from this point to the jammer
        dist_to_jammer = np.linalg.norm(point - jammer_pos)
        latitude = jammer_pos[1]  # Assuming second coordinate is latitude
        degrees_lat, degrees_lon = meters_to_degrees(dist_to_jammer, latitude)

        # Compute jamming power at this point using path loss model
        if loss_func == log_distance_path_loss:
            jamming_power_dbm = P_tx_jammer + G_tx_jammer + G_rx - loss_func(dist_to_jammer, n=n, sigma=sigma)
        else:
            jamming_power_dbm = P_tx_jammer + G_tx_jammer + G_rx - loss_func(dist_to_jammer)
        max_jamming_power_dbm = max(max_jamming_power_dbm, jamming_power_dbm)
    return max_jamming_power_dbm


def calculate_node_bounds(arena_size):
    # Constants
    node_range = 200.0  # average comm. range in meters

    # Calculate number of nodes required for horizontal/vertical coverage
    min_nodes_h = math.ceil(arena_size / node_range)

    # Calculate the diagonal of the arena
    diagonal = arena_size * math.sqrt(2)

    # Calculate number of nodes required for diagonal coverage
    min_nodes_diagonal = math.ceil(diagonal / node_range)

    return min_nodes_h + min_nodes_diagonal


def plot_network_with_rssi(node_positions, final_rssi, jammer_position, sinr_db, noise_floor_db, jammed):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes
    for idx, pos in enumerate(node_positions):
        color = 'red' if jammed[idx] else 'blue'

        # Check if the node is effectively disconnected based on RSSI
        node_info = f"Node {idx}\nRSSI: {final_rssi[idx]:.2f} dB\nSINR: {sinr_db[idx]:.2f} dB\nNoise: {noise_floor_db[idx]:.2f} dB"
        ax.plot(pos[0], pos[1], 'o', color=color)  # Nodes in blue or red depending on jamming status
        ax.text(pos[0], pos[1], node_info, fontsize=9, ha='right')

    # Plot jammer
    ax.plot(jammer_position[0][0], jammer_position[0][1], 'r^', markersize=10)  # Jammer in red
    ax.text(jammer_position[0][0], jammer_position[0][1], ' Jammer', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=10)

    ax.set_title('Network Topology with RSSI, SINR, and Noise Floor', fontsize=11)
    ax.set_xlabel('X position (m)', fontsize=14)
    ax.set_ylabel('Y position (m)', fontsize=14)
    plt.grid(True)
    plt.show()


def get_position(n_nodes, size, placement='random'):
    if placement == 'random':
        positions = np.random.rand(n_nodes, 2) * size
    elif placement == 'circle':
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        x = size / 2 * np.cos(angles) + size / 2
        y = size / 2 * np.sin(angles) + size / 2
        positions = np.column_stack((x, y))
    elif placement == 'triangle':
        positions = np.zeros((n_nodes, 2))
        sides = 3
        # Calculate points per side and determine extra points
        points_per_side, additional_points = divmod(n_nodes, sides)
        # Generate a random angle for rotation (in radians)
        random_rotation = np.random.uniform(0, 2 * np.pi)
        index = 0  # Initialize index for position assignment
        for i in range(sides):
            # Compute start and end points for each side of the triangle with random rotation
            angle_start = i * 2 * np.pi / sides + random_rotation
            angle_end = (i + 1) * 2 * np.pi / sides + random_rotation
            start = np.array([np.cos(angle_start), np.sin(angle_start)]) * size / 2 + size / 2
            end = np.array([np.cos(angle_end), np.sin(angle_end)]) * size / 2 + size / 2
            # Assign points to the current side, adjust for additional points
            num_points_this_side = points_per_side + (1 if additional_points > 0 else 0)
            additional_points -= 1 if additional_points > 0 else 0
            # Distribute points linearly between start and end points
            positions[index:index + num_points_this_side] = np.linspace(start, end, num_points_this_side, endpoint=False)
            index += num_points_this_side
        positions = positions[:n_nodes]
    elif placement == 'rectangle':
        x = np.linspace(0, size, int(np.sqrt(n_nodes) + 1))
        y = np.linspace(0, size, int(np.sqrt(n_nodes) + 1))
        xv, yv = np.meshgrid(x, y)
        positions = np.column_stack((xv.ravel(), yv.ravel()))[:n_nodes]
    else:
        raise ValueError("Unsupported placement type.")
    return positions


def meters_to_degrees(distance_meters, latitude):
    """Convert distance in meters to degrees latitude and longitude."""
    # Convert meters to degrees latitude
    degrees_latitude = distance_meters / 111320
    # Convert meters to degrees longitude, considering the latitude
    degrees_longitude = distance_meters / (111320 * np.cos(np.radians(latitude)))
    return degrees_latitude, degrees_longitude


# np.random.seed(42)

# Initialize DataFrame to collect data
columns = ["num_samples", "node_positions", "node_rssi", "node_noise", "node_states", "jammer_position", "jammer_power", "jammer_gain", "pl_exp", "sigma"]
data_collection = pd.DataFrame(columns=columns)

# Node information
instance_count, num_instances = 0, 333
loss_func = log_distance_path_loss

for instance_count in tqdm(range(num_instances)):
    # Path loss variables for simulation of environment where the conditions are predominantly open with minimal obstructions
    n = np.random.uniform(2.0, 3.5)  # Random path loss exponent between 2.0 and 2.5
    sigma = np.random.uniform(2, 6)  # Random shadow fading between 2 dB and 6 dB

    size = np.random.randint(500, 1500)  # Area size in meters [500, 1500]
    lb_nodes = calculate_node_bounds(size)
    ub_nodes = 8 * lb_nodes
    beta_values = np.random.beta(2, 8)
    n_nodes = math.ceil(beta_values * (ub_nodes - lb_nodes) + lb_nodes)

    # Radio parammeters
    P_tx = np.random.randint(15, 30)  # Transmit power in dBm [15, 30]
    G_tx = 0  # Transmitting antenna gain in dBi [0, 5]
    G_rx = 0  # Receiving antenna gain in dBi [0, 5]
    P_tx_jammer = np.random.randint(20, 50)  # Jammer transmit power in dBm [25, 50]
    G_tx_jammer = np.random.randint(0, 5)  # Jammer transmitting antenna gain in dBi [0, 5]

    # Node positions
    node_positions = get_position(n_nodes, size, placement=POS_SAMPLING_STRAT)

    # Random jammer position
    jammer_position = np.random.rand(1, 2) * size

    # Convert positions to degrees relative to a reference point (e.g., equator for simplicity)
    ref_latitude = 0  # Assuming a reference latitude, adjust as necessary
    node_positions_deg = np.array([meters_to_degrees(pos, ref_latitude) for pos in node_positions])
    jammer_position_deg = meters_to_degrees(jammer_position[0], ref_latitude)

    config = {
        'size': size,
        'n_nodes': n_nodes,
        'P_tx': P_tx,
        'G_tx': G_tx,
        'G_rx': G_rx,
        'P_tx_jammer': P_tx_jammer,
        'G_tx_jammer': G_tx_jammer
    }

    # Distance calculations
    dist_matrix = np.linalg.norm(node_positions_deg[:, np.newaxis, :] - node_positions_deg[np.newaxis, :, :], axis=2)
    jammer_dist = np.linalg.norm(node_positions_deg - jammer_position_deg, axis=1)

    # Path loss calculations
    if loss_func == log_distance_path_loss:
        path_loss = loss_func(dist_matrix, n=n, sigma=sigma)
        path_loss_jammer = loss_func(jammer_dist, n=n, sigma=sigma)
    else:
        path_loss = loss_func(dist_matrix)
        path_loss_jammer = loss_func(jammer_dist)

    # RSSI calculations
    rssi_matrix = P_tx + G_tx + G_rx - path_loss
    rssi_jammer = P_tx_jammer + G_tx_jammer + G_rx - path_loss_jammer

    # Precompute maximum jammer RSSI for each pair using broadcasting
    # Sample alongwith path (i,j) and take the highest jamming power from those points to represent power at (i,j)
    max_jammer_rssi_matrix = np.full((n_nodes, n_nodes), -np.inf)  # Initialize with low values
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only fill upper triangle, the matrix is symmetric
            # Sample path and compute maximum jamming power
            max_jamming_power_dbm = sample_path_jamming(node_positions[i], node_positions[j], jammer_position[0], loss_func, n, sigma)
            max_jammer_rssi_matrix[i, j] = max_jamming_power_dbm
            max_jammer_rssi_matrix[j, i] = max_jamming_power_dbm

    # Convert maximum jammer RSSI to linear scale for the entire matrix
    jammer_power_linear_matrix = dbm_to_linear(max_jammer_rssi_matrix)

    # Compute the noise floor for all pairs
    noise_floor_matrix = dbm_to_linear(NOISE_LEVEL) + jammer_power_linear_matrix

    # Convert RSSI values of the original matrix to linear scale
    rssi_linear_matrix = dbm_to_linear(rssi_matrix)

    # Compute SINR for all pairs
    sinr_matrix = rssi_linear_matrix / noise_floor_matrix

    # Convert SINR back to dB for the entire matrix
    sinr_db_matrix = linear_to_db(sinr_matrix)

    # Adjust the RSSI matrix
    rssi_affected_matrix = np.minimum(rssi_matrix, rssi_matrix - (REFERENCE_SINR - sinr_db_matrix))
    rssi_affected_matrix = np.maximum(rssi_affected_matrix, RSSI_THRESHOLD)

    # Ignore self-comparisons
    np.fill_diagonal(rssi_matrix, np.nan)
    np.fill_diagonal(rssi_affected_matrix, np.nan)

    # Calculate normal RSSI as the mean/max of the valid signals
    affected_rssi = np.nanmax(rssi_affected_matrix, axis=1)

    # Reference to disconnected nodes
    disconnected = affected_rssi == np.nan
    # Calculate normal RSSI as the mean/max of the valid signals
    affected_rssi = np.nan_to_num(affected_rssi, nan=RSSI_THRESHOLD)  # Replace NaN with rssi_threshold or any suitable default (isolated samples)

    # Compute linear powers for jammer and normal signals
    P_rx_linear = dbm_to_linear(affected_rssi)
    N_linear = dbm_to_linear(NOISE_LEVEL)
    jammer_linear = dbm_to_linear(rssi_jammer)

    # Compute SINR in linear scale and convert SINR from linear scale to dB
    noise_floor = jammer_linear + N_linear
    noise_floor_dB = linear_to_db(noise_floor)

    SINR_linear = P_rx_linear / noise_floor
    SINR_dB = linear_to_db(SINR_linear)

    # Compute detection threshold for SINR
    jammed = noise_floor_dB > NOISE_THRESHOLD

    # Condition checks
    jammed_nodes = np.sum(jammed) >= 4  # At least 4 nodes are jammed
    not_jammed_nodes = np.sum(~jammed) >= 1  # At least 2 nodes are not jammed
    high_rssi_nodes = np.sum(affected_rssi > -80) >= 1  # At least 2 nodes have RSSI greater than -80

    # Plot only if all conditions are met
    if jammed_nodes and not_jammed_nodes and high_rssi_nodes:
        if PLOT:
            plot_network_with_rssi(node_positions, affected_rssi, jammer_position, SINR_dB, noise_floor_dB, jammed)
        instance_count += 1

        # Data to be collected
        data = {
            "num_samples": n_nodes,
            "node_positions": [node_positions.tolist()],  # Convert positions to list for storage
            "node_rssi": [affected_rssi.tolist()],  # RSSI values converted to list
            "node_noise": [noise_floor_dB.tolist()],  # RSSI values converted to list
            "node_states": [jammed.astype(int).tolist()],  # Convert boolean array to int and then to list
            "jammer_position": [jammer_position[0].tolist()],  # Jammer position
            "jammer_power": P_tx_jammer,  # Jammer transmit power
            "jammer_gain": G_tx_jammer,  # Jammer gain
            "pl_exp": n,  # Path loss exponent
            "sigma": sigma  # Shadow fading
        }

        # Append the new row to the data collection DataFrame
        data_collection = pd.concat([data_collection, pd.DataFrame(data, index=[0])], ignore_index=True)

# After the loop, you can save the DataFrame to a CSV file or use it for further analysis
data_collection.to_csv(f"{POS_SAMPLING_STRAT}.csv", index=False)
