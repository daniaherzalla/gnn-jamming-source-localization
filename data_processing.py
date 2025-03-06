import os
import pickle
import hashlib
import json

import pandas as pd
import numpy as np
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset, Dataset
from torch_geometric.transforms import AddRandomWalkPE
from typing import Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import logging
from utils import cartesian_to_polar
from custom_logging import setup_logging
from config import params

from torch_geometric.utils import to_networkx


setup_logging()


class Instance:
    def __init__(self, row):
        if params['inference']:
            # Initialize attributes from the pandas row and convert appropriate fields to numpy arrays only if not already arrays
            self.node_positions = row['node_positions'] if isinstance(row['node_positions'], np.ndarray) else np.array(row['node_positions'])
            self.node_positions_cart = row['node_positions_cart'] if isinstance(row['node_positions_cart'], np.ndarray) else np.array(row['node_positions_cart'])
            self.node_noise = row['node_noise'] if isinstance(row['node_noise'], np.ndarray) else np.array(row['node_noise'])
            if params['dynamic']:
                self.jammed_at = row['jammed_at']
        else:
            # Initialize attributes from the pandas row and convert appropriate fields to numpy arrays only if not already arrays
            self.num_samples = row['num_samples']
            self.node_positions = row['node_positions'] if isinstance(row['node_positions'], np.ndarray) else np.array(row['node_positions'])
            self.node_positions_cart = row['node_positions_cart'] if isinstance(row['node_positions_cart'], np.ndarray) else np.array(row['node_positions_cart'])
            self.node_noise = row['node_noise'] if isinstance(row['node_noise'], np.ndarray) else np.array(row['node_noise'])
            self.pl_exp = row['pl_exp']
            self.sigma = row['sigma']
            self.jammer_power = row['jammer_power']
            self.jammer_position = row['jammer_position'] if isinstance(row['jammer_position'], np.ndarray) else np.array(row['jammer_position'])
            self.jammer_gain = row['jammer_gain']
            self.id = row['id']
            self.dataset = row['dataset']
            # self.jammer_placement_bin = row['jammer_placement_bin']
            # self.jammer_direction = row['jammer_direction']
            if params['dynamic']:
                self.jammed_at = row['jammed_at']

    def get_crop(self, start, end):
        if params['dynamic']:
            if params['inference']:
                cropped_instance = Instance({
                    'node_positions': self.node_positions[start:end],
                    'node_positions_cart': self.node_positions_cart[start:end],
                    'node_noise': self.node_noise[start:end],
                    'jammed_at': self.jammed_at  # Jammed index remains the same, can adjust logic if needed
                })
            else:
                cropped_instance = Instance({
                    'num_samples': end - start,
                    'node_positions': self.node_positions[start:end],
                    'node_positions_cart': self.node_positions_cart[start:end],
                    'node_noise': self.node_noise[start:end],
                    'pl_exp': self.pl_exp,
                    'sigma': self.sigma,
                    'jammer_power': self.jammer_power,
                    'jammer_position': self.jammer_position,
                    'jammer_gain': self.jammer_gain,
                    'id': self.id,
                    'dataset': self.dataset,
                    'jammed_at': self.jammed_at  # Jammed index remains the same, can adjust logic if needed
                })
        else:
            cropped_instance = Instance({
                'num_samples': end - start,
                'node_positions': self.node_positions[start:end],
                'node_positions_cart': self.node_positions_cart[start:end],
                'node_noise': self.node_noise[start:end],
                'pl_exp': self.pl_exp,
                'sigma': self.sigma,
                'jammer_power': self.jammer_power,
                'jammer_position': self.jammer_position,
                'jammer_gain': self.jammer_gain,
                'id': self.id,
                'dataset': self.dataset
            })
        return cropped_instance

    def apply_rotation(self, degrees):
        if params['coords'] == 'polar':
            radians = np.deg2rad(degrees)

            # Update node_positions
            new_node_positions = []
            for r, sin_theta, cos_theta in self.node_positions:
                # Convert sin(theta) and cos(theta) to theta
                theta = np.arctan2(sin_theta, cos_theta)
                # Add rotation
                new_theta = theta + radians

                # Calculate new sin and cos
                new_sin_theta = np.sin(new_theta)
                new_cos_theta = np.cos(new_theta)

                # Append new values preserving the radius
                new_node_positions.append([r, new_sin_theta, new_cos_theta])

            self.node_positions = np.array(new_node_positions)

            # Update jammer_position
            r, sin_theta, cos_theta = self.jammer_position[0]

            theta = np.arctan2(sin_theta, cos_theta)
            new_theta = theta + radians  # Add rotation directly to the angle

            # Calculate new sin and cos
            new_sin_theta = np.sin(new_theta)
            new_cos_theta = np.cos(new_theta)

            self.jammer_position = np.array([r, new_sin_theta, new_cos_theta])

            # Rotate node_positions_cart (cartesian)
            new_node_positions_cart = []
            for x, y in self.node_positions_cart:
                new_x = x * np.cos(radians) - y * np.sin(radians)
                new_y = x * np.sin(radians) + y * np.cos(radians)
                new_node_positions_cart.append([new_x, new_y])

            self.node_positions_cart = np.array(new_node_positions_cart)

    def drop_node(self, drop_rate=0.2, min_nodes=3):
        """
        Apply NodeDrop augmentation by randomly dropping nodes' feature vectors.

        Args:
            drop_rate (float): Probability of dropping a node's feature vector. Default is 0.2.
        """
        # Get the number of nodes
        num_nodes = len(self.node_positions)

        # Generate a binary mask for dropping nodes
        # 1 = keep the node, 0 = drop the node
        mask = np.random.binomial(1, 1 - drop_rate, size=num_nodes)

        # Ensure that at least `min_nodes` are not dropped
        while sum(mask) < min_nodes:
            mask[np.random.choice(num_nodes, min_nodes - sum(mask), replace=False)] = 1

        # Apply the mask to each feature array
        self.node_positions = self.node_positions[mask == 1]
        self.node_positions_cart = self.node_positions_cart[mask == 1]
        self.node_noise = self.node_noise[mask == 1]

        # If angle_of_arrival is part of the features, drop it as well
        if hasattr(self, 'angle_of_arrival'):
            self.angle_of_arrival = self.angle_of_arrival[mask == 1]

        # Update the number of samples after dropping nodes
        self.num_samples = len(self.node_positions)

    def zoom_in(self, max_crop_size=None):
        """
        Creates a cropped instance by selecting a subset of nodes based on their noise levels.

        Args:
            max_crop_size (int, optional): Maximum number of nodes to include in the cropped instance.
                                          If not provided, defaults to the length of `self.node_positions`.

        Returns:
            Instance: A new cropped instance with selected nodes and copied attributes.
        """
        # Set max_crop_size based on the length of node_positions if not provided
        if max_crop_size is None:
            max_crop_size = len(self.node_positions) + 1

        # Ensure a minimum of 3 nodes are included
        min_nodes_with_highest_noise = 3

        # Randomly choose crop size between min_nodes_with_highest_noise and max_crop_size
        crop_size = random.randint(min_nodes_with_highest_noise, max_crop_size)

        # Sort nodes by noise in descending order
        indices_sorted_by_noise = np.argsort(-self.node_noise)

        # Always include the top 3 nodes with the highest noise
        mandatory_indices = indices_sorted_by_noise[:min_nodes_with_highest_noise]

        # Select additional nodes if crop_size > 3
        additional_indices = []
        if crop_size > min_nodes_with_highest_noise:
            remaining_indices = indices_sorted_by_noise[min_nodes_with_highest_noise:]
            additional_size = min(crop_size - min_nodes_with_highest_noise, len(remaining_indices))

            if additional_size > 0:
                additional_indices = np.random.choice(remaining_indices, size=additional_size, replace=False)

        # Combine mandatory and additional indices
        final_indices = np.concatenate((mandatory_indices, additional_indices)).astype(int)

        # Ensure at least 3 nodes are selected
        if len(final_indices) < 3:
            final_indices = np.concatenate((mandatory_indices, indices_sorted_by_noise[1:2])).astype(int)

        self.num_samples = len(final_indices)
        self.node_positions = self.node_positions[final_indices]
        self.node_positions_cart = self.node_positions_cart[final_indices]
        self.node_noise = self.node_noise[final_indices]


    def downsample(self):
        if params['ds_method'] == 'noise':

            # # Plot the original Cartesian positions
            # cart_pos = np.array(self.node_positions_cart)
            # plt.figure(figsize=(10, 6))
            # plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', label='Original Positions')
            # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', s=1, label='Jammer Position')
            # plt.title('Original Node Positions (Cartesian)')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            # print("og num nodes: ", len(self.node_positions))
            # Convert positions to a DataFrame to use bin_nodes
            node_df = pd.DataFrame({
                'r': self.node_positions[:, 0],
                'sin_theta': self.node_positions[:, 1],
                'cos_theta': self.node_positions[:, 2],
                'sin_phi': self.node_positions[:, 3],
                'cos_phi': self.node_positions[:, 4],
                'x': self.node_positions_cart[:, 0],
                'y': self.node_positions_cart[:, 1],
                'z': self.node_positions_cart[:, 2],
                'noise_level': self.node_noise
            })

            binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])
            self.node_positions = binned_nodes[['r', 'sin_theta', 'cos_theta', 'sin_phi', 'cos_phi']].to_numpy()
            self.node_positions_cart = binned_nodes[['x', 'y', 'z']].to_numpy()
            self.node_noise = binned_nodes['noise_level'].to_numpy()

            # if len(self.node_positions) < 3:
            #     print('downsampling err: ', len(self.node_positions))
            #     quit()

            # # Plot the original Cartesian positions
            # cart_pos = np.array(self.node_positions_cart)
            # plt.figure(figsize=(10, 6))
            # plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', label='DS Positions')
            # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', s=1, label='Jammer Position')
            # plt.title('DS Node Positions (Cartesian)')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.legend()
            # plt.grid(True)
            # plt.show()


        # elif params['ds_method'] == 'time_window_avg':
        #     # # Plot the original Cartesian positions
        #     # cart_pos = np.array(self.node_positions_cart)
        #     # plt.figure(figsize=(10, 6))
        #     # plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', s=0.5, label='Original Positions')
        #     # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', label='Jammer Position')
        #     # plt.title('Original Node Positions (Cartesian)')
        #     # plt.xlabel('X Coordinate')
        #     # plt.ylabel('Y Coordinate')
        #     # plt.legend()
        #     # plt.grid(True)
        #     # plt.show()
        #
        #     max_nodes_per_segment = params['max_nodes']
        #     distance_threshold = 100  # meters
        #     overlap = 50  # meters overlap between consecutive windows
        #
        #     # Compute cumulative distances
        #     cumulative_distances = [0]
        #     for i in range(1, len(self.node_positions_cart)):
        #         distance = np.linalg.norm(self.node_positions_cart[i] - self.node_positions_cart[i - 1])
        #         cumulative_distances.append(cumulative_distances[-1] + distance)
        #
        #     cumulative_distances = np.array(cumulative_distances)
        #
        #     # Create downsampled attributes
        #     downsampled_positions = []
        #     downsampled_noise_values = []
        #     downsampled_cart_positions = []
        #     start_idx = 0
        #
        #     while start_idx < len(self.node_positions_cart):
        #
        #         # Find the end of the current window
        #         end_idx = np.searchsorted(cumulative_distances, cumulative_distances[start_idx] + distance_threshold, side='right')
        #         end_idx = min(end_idx, len(self.node_positions_cart))  # Ensure it does not go out of bounds
        #
        #         # Segmenting the data within the current window
        #         segment_length = end_idx - start_idx
        #
        #         if segment_length > 0:
        #             sample_indices = np.arange(start_idx, end_idx, dtype=int)
        #
        #             # If segment exceeds max_nodes_per_segment, downsample by selecting nodes
        #             if len(sample_indices) > max_nodes_per_segment:
        #
        #                 # Calculate the step size to select max_nodes_per_segment nodes
        #                 step = len(sample_indices) // max_nodes_per_segment
        #
        #                 # Select nodes at regular intervals
        #                 selected_indices = sample_indices[::step][:max_nodes_per_segment]
        #
        #                 # Append the selected nodes to the downsampled lists
        #                 downsampled_positions.extend(self.node_positions[selected_indices])
        #                 downsampled_noise_values.extend(self.node_noise[selected_indices])
        #                 downsampled_cart_positions.extend(self.node_positions_cart[selected_indices])
        #
        #             else:
        #                 # If segment does not exceed max_nodes_per_segment, include all nodes
        #                 downsampled_positions.extend(self.node_positions[sample_indices])
        #                 downsampled_noise_values.extend(self.node_noise[sample_indices])
        #                 downsampled_cart_positions.extend(self.node_positions_cart[sample_indices])
        #
        #         # Move to the next window with overlap
        #         start_idx = np.searchsorted(cumulative_distances, cumulative_distances[start_idx] + distance_threshold - overlap, side='right')
        #
        #     # Step 2: Noise Filtering
        #     num_filtered_nodes = max(1, int(params['max_nodes'] * params['filtering_proportion']))
        #     high_noise_indices = np.argsort(self.node_noise)[-num_filtered_nodes:]
        #     self.node_positions = self.node_positions[high_noise_indices]
        #     self.node_positions_cart = self.node_positions_cart[high_noise_indices]
        #     self.node_noise = self.node_noise[high_noise_indices]
        #
        #     if len(downsampled_positions) >= 3:
        #         self.node_positions = np.array(downsampled_positions)
        #         self.node_noise = np.array(downsampled_noise_values)
        #         self.node_positions_cart = np.array(downsampled_cart_positions)
        #
        #         # Plot the downsampled Cartesian positions
        #         downsampled_cart_positions = np.array(downsampled_cart_positions)
        #         plt.figure(figsize=(10, 6))
        #         plt.scatter(downsampled_cart_positions[:, 0], downsampled_cart_positions[:, 1], c='blue', marker='o', s=0.5, label='Downsampled Positions')
        #         plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', label='Jammer Position')
        #         plt.title('Downsampled Node Positions (Cartesian)')
        #         plt.xlabel('X Coordinate')
        #         plt.ylabel('Y Coordinate')
        #         plt.legend()
        #         plt.grid(True)
        #         plt.show()

        # elif params['ds_method'] == 'time_window_avg':
        #     max_nodes = params['max_nodes']
        #     num_original_nodes = len(self.node_positions)
        #
        #     if num_original_nodes > max_nodes:
        #         indices = np.linspace(0, num_original_nodes - 1, num=max_nodes, dtype=int)
        #
        #         # Directly select downsampled attributes
        #         self.node_positions = self.node_positions[indices]
        #         self.node_noise = self.node_noise[indices]
        #         self.node_positions_cart = self.node_positions_cart[indices]
        #
        #         # # Step 2: Noise Filtering
        #         # num_filtered_nodes = max(1, int(params['max_nodes'] * params['filtering_proportion']))
        #         # high_noise_indices = np.argsort(self.node_noise)[-num_filtered_nodes:]
        #         # self.node_positions = self.node_positions[high_noise_indices]
        #         # self.node_positions_cart = self.node_positions_cart[high_noise_indices]
        #         # self.node_noise = self.node_noise[high_noise_indices]

        # elif params['ds_method'] == 'time_window_avg':
        #     # # Plot the original Cartesian positions
        #     # cart_pos = np.array(self.node_positions_cart)
        #     # plt.figure(figsize=(10, 6))
        #     # plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', label='Original Positions')
        #     # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', s=1, label='Jammer Position')
        #     # plt.title('Original Node Positions (Cartesian)')
        #     # plt.xlabel('X Coordinate')
        #     # plt.ylabel('Y Coordinate')
        #     # plt.legend()
        #     # plt.grid(True)
        #     # plt.show()
        #
        #     max_nodes = params['max_nodes']
        #     num_original_nodes = len(self.node_positions)
        #
        #     if num_original_nodes > max_nodes:
        #         window_size = num_original_nodes // max_nodes
        #         num_windows = max_nodes
        #
        #         # Create downsampled attributes
        #         downsampled_positions = []
        #         downsampled_noise_values = []
        #         downsampled_cart_positions = []
        #         downsampled_angles = []
        #
        #         for i in range(num_windows):
        #             start_idx = i * window_size
        #             end_idx = start_idx + window_size
        #
        #             downsampled_positions.append(np.mean(self.node_positions[start_idx:end_idx], axis=0))
        #             downsampled_noise_values.append(np.mean(self.node_noise[start_idx:end_idx]))
        #             downsampled_cart_positions.append(np.mean(self.node_positions_cart[start_idx:end_idx], axis=0))
        #
        #             if 'angle_of_arrival' in params['required_features']:
        #                 downsampled_angles.append(np.mean(self.angle_of_arrival[start_idx:end_idx]))
        #
        #         # Update self with downsampled data
        #         self.node_positions = np.array(downsampled_positions)
        #         self.node_noise = np.array(downsampled_noise_values)
        #         self.node_positions_cart = np.array(downsampled_cart_positions)
        #
        #         # Plot the downsampled Cartesian positions
        #         downsampled_cart_positions = np.array(downsampled_cart_positions)
        #         plt.figure(figsize=(10, 6))
        #         plt.scatter(downsampled_cart_positions[:, 0], downsampled_cart_positions[:, 1], c='blue', marker='o', s=0.5, label='Downsampled Positions')
        #         plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', label='Jammer Position')
        #         plt.title('Downsampled Node Positions (Cartesian)')
        #         plt.xlabel('X Coordinate')
        #         plt.ylabel('Y Coordinate')
        #         plt.legend()
        #         plt.grid(True)
        #         plt.show()

        elif params['ds_method'] == 'time_window_avg':
            # Plot the original Cartesian positions
            cart_pos = np.array(self.node_positions_cart)
            print(cart_pos)
            plt.figure(figsize=(10, 6))
            plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', label='Original Positions')
            # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', s=0.5, label='Jammer Position')
            plt.title('Original Node Positions (Cartesian)')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()

            max_nodes = params['max_nodes']
            num_original_nodes = len(self.node_positions)

            def spherical_to_cartesian(spherical_coords):
                radius, sin_theta, cos_theta, sin_azimuth, cos_azimuth = spherical_coords.T
                # theta = np.arctan2(sin_theta, cos_theta)
                # azimuth = np.arctan2(sin_azimuth, cos_azimuth)

                x = radius * sin_theta * cos_azimuth
                y = radius * sin_theta * sin_azimuth
                z = radius * cos_theta
                return np.stack([x, y, z], axis=1)

            def cartesian_to_spherical(cartesian_coords):
                x, y, z = cartesian_coords.T
                radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
                azimuth = np.arctan2(y, x)

                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                sin_azimuth = np.sin(azimuth)
                cos_azimuth = np.cos(azimuth)

                return np.stack([radius, sin_theta, cos_theta, sin_azimuth, cos_azimuth], axis=1)

            # Downsampling with Cartesian averaging
            if num_original_nodes > max_nodes:
                window_size = num_original_nodes // max_nodes
                num_windows = max_nodes

                downsampled_cart_positions = []
                downsampled_noise_values = []

                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size

                    # Convert spherical to Cartesian
                    cart_coords = spherical_to_cartesian(self.node_positions[start_idx:end_idx])

                    # Average Cartesian coordinates
                    avg_cart_coords = np.mean(cart_coords, axis=0)
                    downsampled_cart_positions.append(avg_cart_coords)

                    # Average other attributes (e.g., noise)
                    downsampled_noise_values.append(np.mean(self.node_noise[start_idx:end_idx]))

                # Convert averaged Cartesian coordinates back to spherical
                self.node_positions = cartesian_to_spherical(np.array(downsampled_cart_positions))
                self.node_noise = np.array(downsampled_noise_values)
                self.node_positions_cart = np.array(downsampled_cart_positions)

                # Plot the downsampled Cartesian positions
                downsampled_cart_positions = np.array(downsampled_cart_positions)
                plt.figure(figsize=(10, 6))
                print(downsampled_cart_positions)
                plt.scatter(downsampled_cart_positions[:, 0], downsampled_cart_positions[:, 1], c='blue', marker='o', s=0.5, label='Downsampled Positions')
                # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', label='Jammer Position')
                plt.title('Downsampled Node Positions (Cartesian)')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.grid(True)
                plt.show()

        # elif params['ds_method'] == 'time_window_avg':
        #     max_nodes = params['max_nodes']
        #     num_original_nodes = len(self.node_positions)
        #
        #     if num_original_nodes > max_nodes:
        #         indices = np.linspace(0, num_original_nodes - 1, num=max_nodes, dtype=int)
        #
        #         # Directly select downsampled attributes for positions
        #         self.node_positions = self.node_positions[indices]
        #         self.node_positions_cart = self.node_positions_cart[indices]
        #
        #         # Compute the average for node noise
        #         bin_size = num_original_nodes // max_nodes
        #         self.node_noise = np.array([np.mean(self.node_noise[i * bin_size:(i + 1) * bin_size]) for i in range(max_nodes)])

        # elif params['ds_method'] == 'hybrid':
        #     max_nodes = params['max_nodes']
        #
        #     # Step 2: Noise Filtering
        #     num_filtered_nodes = max(1, int(max_nodes * params['filtering_proportion']))
        #     high_noise_indices = np.argsort(self.node_noise)[-num_filtered_nodes:]
        #     self.node_positions = self.node_positions[high_noise_indices]
        #     self.node_positions_cart = self.node_positions_cart[high_noise_indices]
        #     self.node_noise = self.node_noise[high_noise_indices]
        #     if 'angle_of_arrival' in params['required_features']:
        #         self.angle_of_arrival = self.angle_of_arrival[high_noise_indices]
        #
        #     # Step 1: Time Window Averaging
        #     num_original_nodes = len(self.node_positions)
        #
        #     if num_original_nodes > max_nodes:
        #         window_size = num_original_nodes // max_nodes
        #         num_windows = max_nodes
        #
        #         # Create downsampled attributes
        #         downsampled_positions = []
        #         downsampled_noise_values = []
        #         downsampled_cart_positions = []
        #         downsampled_angles = []
        #
        #         for i in range(num_windows):
        #             start_idx = i * window_size
        #             end_idx = start_idx + window_size
        #
        #             downsampled_positions.append(np.mean(self.node_positions[start_idx:end_idx], axis=0))
        #             downsampled_noise_values.append(np.mean(self.node_noise[start_idx:end_idx]))
        #             downsampled_cart_positions.append(np.mean(self.node_positions_cart[start_idx:end_idx], axis=0))
        #             if 'angle_of_arrival' in params['required_features']:
        #                 downsampled_angles.append(np.mean(self.angle_of_arrival[start_idx:end_idx]))
        #
        #         # Update self with downsampled data
        #         self.node_positions = np.array(downsampled_positions)
        #         self.node_noise = np.array(downsampled_noise_values)
        #         self.node_positions_cart = np.array(downsampled_cart_positions)
        elif params['ds_method'] == 'hybrid':
            # Plot the original Cartesian positions
            cart_pos = np.array(self.node_positions_cart)
            plt.figure(figsize=(10, 6))
            plt.scatter(cart_pos[:, 0], cart_pos[:, 1], c='blue', marker='o', label='Original Positions')
            # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', s=0.5, label='Jammer Position')
            plt.title('Original Node Positions (Cartesian)')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()

            max_nodes = params['max_nodes']
            num_original_nodes = len(self.node_positions)

            def spherical_to_cartesian(spherical_coords):
                radius, sin_theta, cos_theta, sin_azimuth, cos_azimuth = spherical_coords.T
                theta = np.arctan2(sin_theta, cos_theta)
                azimuth = np.arctan2(sin_azimuth, cos_azimuth)

                x = radius * np.sin(theta) * np.cos(azimuth)
                y = radius * np.sin(theta) * np.sin(azimuth)
                z = radius * np.cos(theta)
                return np.stack([x, y, z], axis=1)

            def cartesian_to_spherical(cartesian_coords):
                x, y, z = cartesian_coords.T
                radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
                azimuth = np.arctan2(y, x)

                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                sin_azimuth = np.sin(azimuth)
                cos_azimuth = np.cos(azimuth)

                return np.stack([radius, sin_theta, cos_theta, sin_azimuth, cos_azimuth], axis=1)

            # Downsampling with Cartesian averaging
            if num_original_nodes > max_nodes:
                window_size = num_original_nodes // max_nodes
                num_windows = max_nodes

                downsampled_cart_positions = []
                downsampled_noise_values = []

                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size

                    # Convert spherical to Cartesian
                    cart_coords = spherical_to_cartesian(self.node_positions[start_idx:end_idx])

                    # Average Cartesian coordinates
                    avg_cart_coords = np.mean(cart_coords, axis=0)
                    downsampled_cart_positions.append(avg_cart_coords)

                    # Average other attributes (e.g., noise)
                    downsampled_noise_values.append(np.mean(self.node_noise[start_idx:end_idx]))

                # Convert averaged Cartesian coordinates back to spherical
                self.node_positions = cartesian_to_spherical(np.array(downsampled_cart_positions))
                self.node_noise = np.array(downsampled_noise_values)
                self.node_positions_cart = np.array(downsampled_cart_positions)

                # Step 2: Noise Filtering
                num_filtered_nodes = max(3, int(params['max_nodes'] * params['filtering_proportion']))
                high_noise_indices = np.argsort(downsampled_noise_values)[-num_filtered_nodes:]
                print(high_noise_indices)
                self.node_positions = self.node_positions[high_noise_indices]
                self.node_positions_cart = self.node_positions_cart[high_noise_indices]
                self.node_noise = self.node_noise[high_noise_indices]

                # Plot the downsampled Cartesian positions
                downsampled_cart_positions = np.array(self.node_positions_cart)
                plt.figure(figsize=(10, 6))
                plt.scatter(downsampled_cart_positions[:, 0], downsampled_cart_positions[:, 1], c='blue', marker='o', s=0.5, label='Downsampled Positions')
                # plt.scatter(self.jammer_position[0][0], self.jammer_position[0][1], c='red', marker='x', label='Jammer Position')
                plt.title('Downsampled Node Positions (Cartesian)')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.grid(True)
                plt.show()

        else:
            raise ValueError("Undefined downsampling method")


class TemporalGraphDataset(Dataset):
    def __init__(self, data, test=False, dynamic=True, discretization_coeff=0.25):
        self.data = data
        self.test = test  # for test set
        self.dynamic = dynamic
        self.discretization_coeff = discretization_coeff

        if self.test:
            # Precompute the graphs during dataset initialization for the test set
            self.samples = self.expand_samples()
            self.precomputed_graphs = [self.precompute_graph(instance) for instance in self.samples]
        else:
            self.samples = [Instance(row) for _, row in data.iterrows()]

    def expand_samples(self):
        expanded_samples = []
        for _, row in self.data.iterrows():
            if params['inference']:
                instance = Instance(row)
                instance.perc_completion = 1
                expanded_samples.append(instance)
                return expanded_samples
            if params['dynamic']:
                # lb_end = max(int(row['jammed_at']), min(10, len(row['node_positions'])))
                lb_end = max(int(row['jammed_at']), min(params['max_nodes'], len(row['node_positions'])))
                ub_end = len(row['node_positions'])
                # lb_end = int((ub_end - lb_end) / 2)

                # Define step size
                if self.discretization_coeff == -1:
                    step_size = 1
                elif isinstance(self.discretization_coeff, float):
                    step_size = max(1, int(self.discretization_coeff * (ub_end - lb_end)))
                else:
                    raise ValueError("Invalid discretization coefficient type")

                # Generate instances for various end points with the step size
                for i in range(lb_end, ub_end + 1, step_size):
                    instance = Instance(row).get_crop(0, i)
                    instance.perc_completion = i/ub_end
                    expanded_samples.append(instance)
            else:
                instance = Instance(row)
                instance.perc_completion = 1
                expanded_samples.append(instance)
        print("len expanded samples: ", len(expanded_samples))
        return expanded_samples


    def precompute_graph(self, instance):
        # Create the graph once and engineer the node features
        if params['downsample']:
            instance.downsample()
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)
        return graph

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, start_crop=0):
        if self.test:
            # Return the precomputed graph for test set
            return self.precomputed_graphs[idx]

        # For non-test set, perform random cropping
        instance = self.samples[idx]
        if params['dynamic']:
            # Check if jammed_at is not NaN and set the lower bound for random selection
            if np.isnan(instance.jammed_at):
                raise ValueError("No jammed instance")
            lb_end = max(int(instance.jammed_at), min(params['max_nodes'], len(instance.node_positions)))
            # lb_end = max(int(instance.jammed_at), min(10, len(instance.node_positions)))
            ub_end = len(instance.node_positions)  # The upper bound is always the length of node_positions
            end = random.randint(lb_end, ub_end)
            instance = instance.get_crop(start_crop, end)
            instance.perc_completion = end / ub_end
        else:
            instance.perc_completion = 1.0

        if params['downsample']:
            instance.downsample()

        if 'crop' in params['aug']:
            instance.zoom_in()

        if 'rot' in params['aug']:
            instance.apply_rotation(random.randint(0, 360))

        if 'drop_node' in params['aug']:
            instance.drop_node()

        # Create and engineer the graph on the fly for training
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)

        if 'noise' in params['aug']:
            # Add noise to the feature vector during training
            graph.x = add_random_noise_to_features(graph.x, noise_std=1e-4)

        return graph


def add_random_noise_to_features(features, noise_std=1e-4, noise_prob=0.2):
    """
    Adds Gaussian noise to a random subset of features in the feature vector.
    Applies noise to a RANDOM SUBSET of values for the selected features.

    Args:
        features (torch.Tensor): The feature vector of shape (num_nodes, num_features).
        noise_std (float): Standard deviation of the Gaussian noise. Default is 1e-4.
        noise_prob (float): Probability of adding noise to each value of the selected features. Default is 0.5.

    Returns:
        torch.Tensor: The feature vector with noise added to a random subset of features.
    """
    num_features = features.size(1)
    # Randomly select the number of features to add noise to (between 0 and half of the features)
    num_features_to_noise = random.randint(0, num_features // 2)
    # Randomly select the indices of features to add noise to
    features_to_noise = random.sample(range(num_features), num_features_to_noise)

    # Add Gaussian noise to a RANDOM SUBSET of values for the selected features
    for feature_idx in features_to_noise:
        # Create a mask to randomly select values to add noise to
        mask = torch.rand_like(features[:, feature_idx]) < noise_prob
        noise = torch.randn_like(features[:, feature_idx]) * noise_std
        features[:, feature_idx][mask] += noise[mask]

    return features


# def add_random_noise_to_all_features(features, noise_std=1e-4, noise_prob=0.2):
#     """
#     Adds Gaussian noise to all features in the feature vector.
#     Applies noise to a random subset of values for each feature.
#
#     Args:
#         features (torch.Tensor): The feature vector of shape (num_nodes, num_features).
#         noise_std (float): Standard deviation of the Gaussian noise. Default is 1e-4.
#         noise_prob (float): Probability of adding noise to each value of the features. Default is 0.5.
#
#     Returns:
#         torch.Tensor: The feature vector with noise added to a random subset of all features.
#     """
#     num_features = features.size(1)
#
#     # Iterate over all features
#     for feature_idx in range(num_features):
#         # Create a mask to randomly select values to add noise to
#         mask = torch.rand_like(features[:, feature_idx]) < noise_prob
#         noise = torch.randn_like(features[:, feature_idx]) * noise_std
#         features[:, feature_idx][mask] += noise[mask]
#
#     return features

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
    try:
        # convert to a numpy array with dtype float
        numpy_coords = np.array(coords, dtype=float)
        center = np.mean(numpy_coords, axis=0)
        centered_coords = numpy_coords - center
        return centered_coords, center
    except Exception as e:
        return None, None


def apply_min_max_normalization_instance(instance):
    """Apply min-max normalization to position and RSSI data for an instance."""
    # logging.info("Applying min-max normalization for instance")

    # Normalize Noise values to range [0, 1]
    min_noise = np.min(instance.node_noise)
    max_noise = np.max(instance.node_noise)
    range_noise = max_noise - min_noise if max_noise != min_noise else 1
    normalized_noise = (instance.node_noise - min_noise) / range_noise
    instance.node_noise = normalized_noise

    # Normalize node positions to range [-1, 1]
    min_coords = np.min(instance.node_positions, axis=0)
    max_coords = np.max(instance.node_positions, axis=0)
    range_coords = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
    normalized_positions = 2 * ((instance.node_positions - min_coords) / range_coords) - 1
    instance.min_coords = min_coords
    instance.max_coords = max_coords
    instance.node_positions = normalized_positions

    # Normalize jammer position similarly
    jammer_position = 2 * ((instance.jammer_position - min_coords) / range_coords) - 1
    instance.jammer_position = jammer_position


def apply_min_max_normalization_instance_noise(instance):
    """Apply min-max normalization to position and RSSI data for an instance."""
    # logging.info("Applying min-max normalization for instance")

    # Normalize Noise values to range [0, 1]
    instance.node_noise_original = instance.node_noise.copy()
    min_noise = 0 #np.min(instance.node_noise)
    max_noise = 80 #np.max(instance.node_noise)
    range_noise = max_noise - min_noise if max_noise != min_noise else 1
    normalized_noise = (instance.node_noise - min_noise) / range_noise
    instance.node_noise = normalized_noise

    # Normalize node positions cartesian for weights to range [0, 1]
    instance.node_positions_cart_original = instance.node_positions_cart.copy()
    min_coords = np.min(instance.node_positions_cart, axis=0)
    max_coords = np.max(instance.node_positions_cart, axis=0)
    range_coords = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
    normalized_positions = (instance.node_positions_cart - min_coords) / range_coords
    instance.node_positions_cart = normalized_positions


# def apply_unit_sphere_normalization(instance):
#     """
#     Apply unit sphere normalization to position data, normalizing only the radius.
#
#     Parameters:
#     instance: An object with attributes 'node_positions' and 'jammer_position',
#               each an array of positions where the first index is the radius.
#
#     Modifies:
#     instance.node_positions: Normalized positions.
#     instance.jammer_position: Normalized jammer position.
#     instance.max_radius: Maximum radius used for normalization.
#     """
#
#     # Extract the radius component from each position
#     # Assuming the radius is at index 0 of each sub-array in node_positions
#     radii = instance.node_positions[:, 0]  # Extracts the first column from the positions array
#
#     # Calculate the maximum radius from the radii
#     max_radius = np.max(radii)
#
#     # Normalize only the radius component of the positions
#     normalized_positions = instance.node_positions.copy()  # Create a copy to avoid modifying the original data
#     normalized_positions[:, 0] /= max_radius  # Normalize only the radius component
#
#     # Assuming jammer_position is a single array [radius, sin, cos]
#     if not params['inference']:
#         normalized_jammer_position = instance.jammer_position.copy()
#         normalized_jammer_position[0][0] /= max_radius  # Normalize only the radius component of the jammer position
#         instance.jammer_position = normalized_jammer_position
#
#     # Update the instance variables
#     instance.node_positions = normalized_positions
#     instance.max_radius = max_radius



# def apply_unit_sphere_normalization_cart(instance):
#     """
#     Apply unit sphere normalization to position data in Cartesian coordinates (x, y).
#
#     Parameters:
#     instance: An object with attributes 'node_positions' and 'jammer_position',
#               where 'node_positions' is an array of [x, y] coordinates.
#
#     Modifies:
#     instance.node_positions: Normalized positions.
#     instance.jammer_position: Normalized jammer position.
#     instance.max_radius: Maximum radius used for normalization.
#     """
#
#     # Extract x and y coordinates from node positions
#     x = instance.node_positions[:, 0]  # x coordinates
#     y = instance.node_positions[:, 1]  # y coordinates
#
#     # Compute the radius for each node position
#     radii = np.sqrt(x**2 + y**2)
#
#     # Calculate the maximum radius from the radii
#     max_radius = np.max(radii)
#
#     # Normalize the x and y coordinates by dividing by the maximum radius
#     normalized_positions = instance.node_positions.copy()  # Create a copy to avoid modifying the original data
#     normalized_positions[:, 0] /= max_radius  # Normalize x coordinates
#     normalized_positions[:, 1] /= max_radius  # Normalize y coordinates
#
#     # Normalize the jammer position (if it exists)
#     if not params['inference']:
#         jammer_x, jammer_y = instance.jammer_position[0][0], instance.jammer_position[0][1]
#         normalized_jammer_x = jammer_x / max_radius
#         normalized_jammer_y = jammer_y / max_radius
#         instance.jammer_position = np.array([[normalized_jammer_x, normalized_jammer_y]])
#
#     # Update the instance variables
#     print(normalized_positions)
#     instance.node_positions = normalized_positions
#     instance.max_radius = max_radius


def apply_unit_sphere_normalization(instance):
    """
    Apply unit sphere normalization to position data, normalizing only the radius.

    Parameters:
    instance: An object with attributes 'node_positions' and 'jammer_position',
              each an array of positions where the first index is the radius.

    Modifies:
    instance.node_positions: Normalized positions.
    instance.jammer_position: Normalized jammer position.
    instance.max_radius: Maximum radius used for normalization.
    """

    # Extract the radius component from each position
    # Assuming the radius is at index 0 of each sub-array in node_positions
    radii = instance.node_positions[:, 0]  # Extracts the first column from the positions array

    # Calculate the maximum radius from the radii
    max_radius = np.max(radii)

    # Normalize only the radius component of the positions
    normalized_positions = instance.node_positions.copy()  # Create a copy to avoid modifying the original data
    # print(normalized_positions)
    normalized_positions[:, 0] /= max_radius  # Normalize only the radius component

    if not params['inference']:
        # Assuming jammer_position is a single array [radius, sin, cos]
        normalized_jammer_position = instance.jammer_position.copy()
        # print(normalized_jammer_position)
        normalized_jammer_position[0][0] /= max_radius  # Normalize only the radius component of the jammer position

        # Update the instance variables
        instance.jammer_position = normalized_jammer_position

    # Update the instance variables
    instance.node_positions = normalized_positions
    instance.max_radius = max_radius


def apply_unit_sphere_normalization_cartesian(instance):
    """
    Apply unit sphere normalization to position data.

    Parameters:
    data (dict): A dictionary containing 'node_positions', an array of positions.

    Returns:
    tuple: A tuple containing the normalized positions and the maximum radius.
    """
    # Extract positions from the current row
    cart_positions = instance.node_positions_cart

    # Calculate the maximum radius from the centroid
    max_radius_cart = np.max(np.linalg.norm(cart_positions, axis=1))

    # Check for zero radius to prevent division by zero
    if max_radius_cart == 0:
        raise ValueError("Max radius is zero, normalization cannot be performed.")

    # Normalize the positions uniformly
    normalized_cart_positions = cart_positions / max_radius_cart
    instance.node_positions_cart = normalized_cart_positions


def apply_z_score_normalization_instance(instance):
    """Apply Z-score normalization to position and RSSI data for an instance."""
    # logging.info("Applying Z-score normalization for instance")

    # Normalize Noise values to mean = 0 and std = 1
    mean_noise = np.mean(instance.node_noise)
    std_noise = np.std(instance.node_noise)
    if std_noise == 0:
        std_noise = 1
    normalized_noise = (instance.node_noise - mean_noise) / std_noise
    instance.node_noise = normalized_noise

    # Normalize node positions to mean = 0 and std = 1
    mean_coords = np.mean(instance.node_positions, axis=0)
    std_coords = np.std(instance.node_positions, axis=0, ddof=1)
    std_coords[std_coords == 0] = 1  # Prevent division by zero
    normalized_positions = (instance.node_positions - mean_coords) / std_coords
    instance.mean_coords = mean_coords
    instance.std_coords = std_coords
    instance.node_positions = normalized_positions

    # Normalize jammer position similarly
    jammer_position = (instance.jammer_position - mean_coords) / std_coords
    instance.jammer_position = jammer_position

    return instance


def convert_data_type(data, load_saved_data):
    if load_saved_data:
        if params['dynamic']:
            dataset_features = params['required_features'] + ['jammer_position', 'jammed_at', 'jammer_power',  'num_samples',  'sigma', 'jammer_power', 'id']
        else:
            dataset_features = params['required_features'] + ['jammer_position', 'jammer_power', 'num_samples', 'sigma', 'jammer_power', 'id']
    else:
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


def dynamic_moving_average(x, max_window_size=10):
    num_nodes = x.size(0)
    window_sizes = torch.clamp(num_nodes - torch.arange(num_nodes), min=1, max=max_window_size)
    averages = torch.zeros_like(x)

    for i in range(num_nodes):
        start = max(i - window_sizes[i] // 2, 0)
        end = min(i + window_sizes[i] // 2 + 1, num_nodes)
        averages[i] = x[start:end].mean(dim=0)

    return averages

# Vectorized
def calculate_noise_statistics(subgraphs, noise_stats_to_compute):
    subgraph = subgraphs[0]
    edge_index = subgraph.edge_index
    if params['3d']:
        node_positions = subgraph.x[:, :5]  # radius, sin(theta), cos(theta), sin(az), cos(az)
        node_noises = subgraph.x[:, 5]
    else:
        node_positions = subgraph.x[:, :3]  # radius, sin(theta), cos(theta)
        node_noises = subgraph.x[:, 3]

    # Create an adjacency matrix from edge_index and include self-loops
    num_nodes = node_noises.size(0)
    adjacency = torch.zeros(num_nodes, num_nodes, device=node_noises.device)
    adjacency[edge_index[0], edge_index[1]] = 1
    torch.diagonal(adjacency).fill_(1)  # Add self-loops

    # Calculate the sum and count of neighbor noises
    neighbor_sum = torch.mm(adjacency, node_noises.unsqueeze(1)).squeeze()
    neighbor_count = adjacency.sum(1)

    # Avoid division by zero for mean calculation
    neighbor_count = torch.where(neighbor_count == 0, torch.ones_like(neighbor_count), neighbor_count)
    mean_neighbor_noise = neighbor_sum / neighbor_count

    # Standard deviation
    neighbor_variance = torch.mm(adjacency, (node_noises ** 2).unsqueeze(1)).squeeze() / neighbor_count - (mean_neighbor_noise ** 2)
    std_noise = torch.sqrt(neighbor_variance)

    # Range: max - min for each node's neighbors
    expanded_noises = node_noises.unsqueeze(0).repeat(num_nodes, 1)
    max_noise = torch.where(adjacency == 1, expanded_noises, torch.full_like(expanded_noises, float('-inf'))).max(1).values
    min_noise = torch.where(adjacency == 1, expanded_noises, torch.full_like(expanded_noises, float('inf'))).min(1).values
    range_noise = max_noise - min_noise

    # Median noise calculation
    median_noise = torch.full_like(mean_neighbor_noise, float('nan'))  # Initial fill
    for i in range(num_nodes):
        neighbors = node_noises[adjacency[i] == 1]
        if neighbors.numel() > 0:
            median_noise[i] = neighbors.median()

    # Replace inf values that appear if a node has no neighbors
    range_noise[range_noise == float('inf')] = 0
    range_noise[range_noise == float('-inf')] = 0

    # Noise Differential
    noise_differential = node_noises - mean_neighbor_noise

    # Convert from polar to Cartesian coordinates
    r = node_positions[:, 0]  # Radius
    sin_theta = node_positions[:, 1]  # sin(theta)
    cos_theta = node_positions[:, 2]  # cos(theta)
    if params['3d']:
        sin_az = node_positions[:, 3]  # sin(azimuth)
        cos_az = node_positions[:, 4]  # cos(azimuth)
        # Compute x, y, z in Cartesian coordinates for 3D
        x = r * cos_theta * cos_az
        y = r * sin_theta * cos_az
        z = r * sin_az
        node_positions_cart = torch.stack((x, y, z), dim=1)  # 3D Cartesian coordinates

        path_distance = torch.cumsum(torch.norm(torch.diff(node_positions_cart, dim=0, prepend=torch.zeros(1, 3)), dim=1), dim=0)
        vector_from_previous_position = torch.diff(node_positions_cart, dim=0, prepend=torch.zeros(1, node_positions_cart.size(1)))
        vector_x = vector_from_previous_position[:, 0]
        vector_y = vector_from_previous_position[:, 1]
        vector_z = vector_from_previous_position[:, 2]
        moving_avg_noise = torch.conv1d(node_noises.view(1, 1, -1), weight=torch.ones(1, 1, 5) / 5, padding=2).view(-1)

        # Calculate rate of change of signal strength (first derivative)
        rate_of_change_signal_strength = torch.diff(node_noises, prepend=torch.tensor([node_noises[0]]))

        # Calculate Distance-Weighted Signal Strength
        distance_weighted_signal_strength = node_noises / (path_distance + 1)  # +1 to avoid division by zero
    else:
        # Compute x, y in Cartesian coordinates for 2D
        x = r * cos_theta
        y = r * sin_theta
        node_positions_cart = torch.stack((x, y), dim=1)  # 2D Cartesian coordinates

    # Weighted Centroid Localization (WCL) calculation in Cartesian space
    weighted_centroid_radius = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_sin_theta = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_cos_theta = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_positions = torch.zeros_like(node_positions_cart)

    if params['3d']:
        weighted_centroid_sin_az = torch.zeros(num_nodes, device=node_positions.device)
        weighted_centroid_cos_az = torch.zeros(num_nodes, device=node_positions.device)

    for i in range(num_nodes):
        weights = torch.pow(10, node_noises[adjacency[i] == 1] / 10)
        valid_neighbor_positions = node_positions_cart[adjacency[i] == 1]

        if weights.sum() > 0:
            centroid_cartesian = (weights.unsqueeze(1) * valid_neighbor_positions).sum(0) / weights.sum()
            radius = torch.norm(centroid_cartesian, p=2)
            sin_theta = centroid_cartesian[1] / radius if radius != 0 else 0
            cos_theta = centroid_cartesian[0] / radius if radius != 0 else 0

            weighted_centroid_radius[i] = radius
            weighted_centroid_sin_theta[i] = sin_theta
            weighted_centroid_cos_theta[i] = cos_theta
            weighted_centroid_positions[i] = centroid_cartesian

            if params['3d']:
                sin_az = centroid_cartesian[2] / radius if radius != 0 else 0
                cos_az = torch.sqrt(centroid_cartesian[0] ** 2 + centroid_cartesian[1] ** 2) / radius if radius != 0 else 1
                weighted_centroid_sin_az[i] = sin_az
                weighted_centroid_cos_az[i] = cos_az


    # Distance from Weighted Centroid
    distances_to_wcl = torch.norm(node_positions_cart - weighted_centroid_positions, dim=1)

    # Sin and Cos of azimuth angle to weighted centroid
    delta_positions = node_positions_cart - weighted_centroid_positions
    # azimuth_angles = torch.atan2(delta_positions[:, 1], delta_positions[:, 0])
    # sin_azimuth_to_wcl = torch.sin(azimuth_angles)
    # cos_azimuth_to_wcl = torch.cos(azimuth_angles)

    # Adding azimuth angles directly
    # azimuth_to_wcl = azimuth_angles  # This tensor now holds the azimuth angles

    # Generate random features for each node
    random_range = 1000
    random_features = torch.randint(random_range, size=(num_nodes,), device=node_noises.device).float() / random_range

    # Base dictionary with entries that are common to both 2D and 3D cases
    noise_stats = {
        'mean_noise': mean_neighbor_noise,
        'std_noise': std_noise,
        'range_noise': range_noise,
        'max_noise': max_noise,
        'median_noise': median_noise,
        'noise_differential': noise_differential,
        'weighted_centroid_radius': weighted_centroid_radius,
        'weighted_centroid_sin_theta': weighted_centroid_sin_theta,
        'weighted_centroid_cos_theta': weighted_centroid_cos_theta,
        'dist_to_wcl': distances_to_wcl,
        'random_feature': random_features
    }
    # Conditionally add 3D specific elements
    if params['3d']:
        noise_stats.update({
            'weighted_centroid_sin_az': weighted_centroid_sin_az,
            'weighted_centroid_cos_az': weighted_centroid_cos_az,
            'path_distance': path_distance,
            'vector_x': vector_x,
            'vector_y': vector_y,
            'vector_z': vector_z,
            'moving_avg_noise': moving_avg_noise,
            'rate_of_change_signal_strength': rate_of_change_signal_strength,
            'distance_weighted_signal_strength': distance_weighted_signal_strength
        })

    return noise_stats

def engineer_node_features(subgraph):
    if subgraph.x.size(0) == 0:
        raise ValueError("Empty subgraph encountered")

    # print("ENGINEERING NODE FEATS")
    new_features = []

    # Extract components
    r = subgraph.x[:, 0]  # Radii
    sin_theta = subgraph.x[:, 1]  # Sin of angles
    cos_theta = subgraph.x[:, 2]  # Cos of angles
    if params['3d']:
        sin_az = subgraph.x[:, 3]  # Sin of azimuth
        cos_az = subgraph.x[:, 4]  # Cos of azimuth
        # Convert to 3D Cartesian coordinates
        x = r * cos_theta * cos_az
        y = r * sin_theta * cos_az
        z = r * sin_az
        cartesian_coords = torch.stack((x, y, z), dim=1)
    else:
        # Convert to 2D Cartesian coordinates
        x = r * cos_theta
        y = r * sin_theta
        cartesian_coords = torch.stack((x, y), dim=1)

    # Calculate centroid
    centroid = torch.mean(cartesian_coords, dim=0)

    if 'dist_to_centroid' in params['additional_features']:
        distances = torch.norm(cartesian_coords - centroid, dim=1, keepdim=True)
        new_features.append(distances)

    if 'sin_azimuth' in params['additional_features']:
        azimuth_angles = torch.atan2(y - centroid[1], x - centroid[0])
        new_features.append(torch.sin(azimuth_angles).unsqueeze(1))
        new_features.append(torch.cos(azimuth_angles).unsqueeze(1))

    # Graph-based noise stats
    graph_stats = [
        'mean_noise', 'median_noise', 'std_noise', 'range_noise',
        'relative_noise', 'max_noise', 'weighted_centroid_radius',
        'weighted_centroid_sin_theta', 'weighted_centroid_cos_theta',
        'noise_differential','dist_to_wcl', 'sin_azimuth_to_wcl',
        'cos_azimuth_to_wcl', 'azimuth_to_wcl', 'random_feature',
        'weighted_centroid_sin_az', 'weighted_centroid_cos_az',
        'path_distance', 'vector_x', 'vector_y', 'vector_z', 'moving_avg_noise',
        'distance_weighted_signal_strength', 'rate_of_change_signal_strength'
    ]

    noise_stats_to_compute = [stat for stat in graph_stats if stat in params['additional_features']]
    if noise_stats_to_compute:
        noise_stats = calculate_noise_statistics([subgraph], noise_stats_to_compute)

        # Add calculated statistics directly to the features list
        for stat in noise_stats_to_compute:
            if stat in noise_stats:
                new_features.append(noise_stats[stat].unsqueeze(1))

    if new_features:
        try:
            new_features_tensor = torch.cat(new_features, dim=1)
            subgraph.x = torch.cat((subgraph.x, new_features_tensor), dim=1)
        except RuntimeError as e:
            raise e

    return subgraph


def convert_to_polar(data):
    #data['polar_coordinates'] = data['node_positions'].apply(cartesian_to_polar)
    #data['polar_coordinates'] = data['polar_coordinates'].apply(angle_to_cyclical)
    # Convert Cartesian coordinates to polar coordinates and then to cyclical coordinates
    #print(data['node_positions'])
    #print(data['jammer_position'])
    data['node_positions_cart'] = data['node_positions'].copy()
    data['node_positions'] = data['node_positions'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))
    # print(data['node_positions'])
    if not params['inference']:
        if params['dynamic']:
           data['jammer_position'] = data['jammer_position'].apply(lambda x: [x])
           data['jammer_position'] = data['jammer_position'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))
        else:
           data['jammer_position'] = data['jammer_position'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))
    #data['jammer_position'] = data['jammer_position'].apply(cartesian_to_polar)
    #print(data['node_positions'])
    #print(data['jammer_position'])



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
# def convert_output_eval(output, data_batch, data_type, device):
#     """
#     Convert and evaluate the output coordinates by uncentering them using the stored midpoints.
#
#     Args:
#         output (torch.Tensor): The model output tensor.
#         data_batch (torch.Tensor): Data batch.
#         data_type (str): The type of data, either 'prediction' or 'target'.
#         device (torch.device): The device on which the computation is performed.
#
#     Returns:
#         torch.Tensor: The converted coordinates after uncentering.
#     """
#     output = output.to(device)  # Ensure the output tensor is on the right device
#
#     if params['norm'] == 'minmax':
#         # 1. Reverse normalization using min_coords and max_coords
#         min_coords = data_batch.min_coords.to(device).view(-1, 2)
#         max_coords = data_batch.max_coords.to(device).view(-1, 2)
#
#         range_coords = max_coords - min_coords
#         converted_output = (output + 1) / 2 * range_coords + min_coords
#
#
#     elif params['norm'] == 'unit_sphere':
#         # 1. Reverse unit sphere normalization using max_radius
#         max_radius = data_batch.max_radius.to(device).view(-1, 1)
#         converted_output = output * max_radius
#
#     # 2. Reverse centering using the stored node_positions_center
#     centers = data_batch.node_positions_center.to(device).view(-1, 2)
#     converted_output += centers
#
#     # return torch.tensor(converted_output, device=device)
#     return converted_output.clone().detach().to(device)

# def convert_output_eval(output, data_batch, data_type, device, training=False):
#     """
#     Convert and evaluate the model output or target coordinates by reversing the preprocessing steps:
#     normalization, centering, and conversion from cyclical to angular coordinates.
#
#     Args:
#         output (torch.Tensor): The model output tensor or target tensor.
#         data_batch (dict): Dictionary containing data batch with necessary meta data.
#         data_type (str): The type of data, either 'prediction' or 'target'.
#         device (torch.device): The device on which the computation is performed.
#
#     Returns:
#         torch.Tensor: The converted coordinates after reversing preprocessing steps.
#     """
#     output = output.to(device)  # Ensure the tensor is on the right device
#
#     # Ensure output always has a batch dimension
#     if output.ndim == 1:
#         output = output.unsqueeze(0)  # Add batch dimension if missing
#
#     # Step 1: Reverse unit sphere normalization
#     if params['norm'] == 'unit_sphere':
#         if 'max_radius' in data_batch:
#             max_radius = data_batch['max_radius'].to(device).view(-1, 1)
#             print("Output shape:", output.shape)  # Check the shape of output
#             print("Max radius shape:", max_radius.shape)  # Check the shape of max_radius
#
#             output *= max_radius #.squeeze()
#     else:
#         return ValueError
#
#     return output.clone().detach().to(device)

def convert_output_eval(output, data_batch, data_type, device):
    """
    Convert and evaluate the model output or target coordinates by reversing the preprocessing steps:
    normalization, centering, and converting polar coordinates (radius, sin(theta), cos(theta)) to Cartesian coordinates.

    Args:
        output (torch.Tensor): The model output tensor or target tensor.
        data_batch (dict): Dictionary containing data batch with necessary metadata.
        data_type (str): The type of data, either 'prediction' or 'target'.
        device (torch.device): The device on which the computation is performed.
        training (bool): Flag indicating whether the operation is for training or not.

    Returns:
        torch.Tensor: The converted Cartesian coordinates after reversing preprocessing steps.
    """
    # This function is called on both prediction and actual jammer pos
    output = output.to(device)  # Ensure the tensor is on the right device

    # Ensure output always has at least two dimensions [batch_size, features]
    if output.ndim == 1:
        output = output.unsqueeze(0)  # Add batch dimension if missing

    # Step 1: Reverse unit sphere normalization
    if params['norm'] == 'unit_sphere':
        max_radius = data_batch['max_radius'].to(device)
        if max_radius.ndim == 0:
            max_radius = max_radius.unsqueeze(0)  # Ensure max_radius is at least 1D for broadcasting

        # Apply normalization reversal safely using broadcasting
        output[:, 0] = output[:, 0] * max_radius

    # Step 2: Convert from polar (radius, sin(theta), cos(theta)) to Cartesian coordinates
    radius = output[:, 0]
    sin_theta = output[:, 1]
    cos_theta = output[:, 2]

    if params['3d']:
        # # Assume additional columns for 3D coordinates: sin_azimuth and cos_azimuth
        sin_phi = output[:, 3]
        cos_phi = output[:, 4]

        # Calculate theta from sin_theta and cos_theta for clarity in following the formulas
        # theta = torch.atan2(sin_theta, cos_theta)

        # # Correct conversion for 3D
        # x = radius * cos_theta * cos_phi  # Note the correction in angle usage
        # y = radius * sin_theta * cos_phi
        # z = radius * sin_phi  # cos(theta) is directly used

        # Correct conversion for 3D
        x = radius * sin_theta * cos_phi  # Note the correction in angle usage
        y = radius * sin_theta * sin_phi
        z = radius * cos_theta  # cos(theta) is directly used


        # Stack x, y, and z coordinates horizontally to form the Cartesian coordinate triplets
        cartesian_coords = torch.stack((x, y, z), dim=1)
    else:
        # Calculate 2D Cartesian coordinates
        x = radius * cos_theta
        y = radius * sin_theta

        # Stack x and y coordinates horizontally to form the Cartesian coordinate pairs
        cartesian_coords = torch.stack((x, y), dim=1)

    return cartesian_coords.clone().detach()



def save_reduced_dataset(dataset, indices, path):
    """
    Saves only the necessary data from the original dataset at specified indices,
    effectively reducing the file size by excluding unnecessary data.
    """
    reduced_data = [dataset[i] for i in indices]  # Extract only the relevant data
    torch.save(reduced_data, path)  # Save the truly reduced dataset


def calculate_centroid(node_positions):
    """
    Calculate the centroid of a list of node positions.

    Args:
        node_positions (list of lists): List of node positions, where each position is a list of [x, y] coordinates.

    Returns:
        list: The centroid [x, y] of the node positions.
    """
    if not node_positions:
        return [0, 0]  # Default centroid if no nodes are present

    # Convert list of lists to a numpy array for easier calculations
    nodes_array = np.array(node_positions)

    # Calculate the mean of the x and y coordinates
    centroid = np.mean(nodes_array, axis=0)

    return centroid.tolist()


from math import atan2, degrees
def determine_direction(jammer_position, centroid):
    """
    Determine the quadrant of the jammer relative to the centroid of node positions,
    adjusted for zero-indexing suitable for PyTorch.

    Parameters:
    - centroid: Tuple or list containing the x and y coordinates of the centroid of the node positions.
    - jammer_position: Tuple or list containing the x and y coordinates of the jammer position.

    Returns:
    - int: Adjusted quadrant number for zero-indexing (0, 1, 2, or 3).
    """
    # print('jammer_position: ', jammer_position)
    # print('centroid: ', centroid)
    dy = jammer_position[0][1] - centroid[1]
    dx = jammer_position[0][0] - centroid[0]
    angle = degrees(atan2(dy, dx))

    if angle > 0:
        return 0 if angle < 90 else 1
    else:
        return 3 if angle > -90 else 2


def determine_direction(jammer_position, centroid):
    """
    Determine the cardinal direction of the jammer relative to the centroid.

    Args:
        jammer_position (list): The [x, y] position of the jammer.
        centroid (list): The [x, y] position of the centroid.

    Returns:
        str: The cardinal direction (e.g., 'North', 'South-East', etc.).
    """
    x_jammer, y_jammer = jammer_position[0]
    x_centroid, y_centroid = centroid

    # Calculate the difference between jammer and centroid
    dx = x_jammer - x_centroid
    dy = y_jammer - y_centroid

    # Determine the direction based on the angle
    angle = np.arctan2(dy, dx)  # Angle in radians
    angle_deg = np.degrees(angle)  # Convert to degrees

    # Map the angle to a cardinal direction
    if -22.5 <= angle_deg < 22.5:
        return 2 #'East'
    elif 22.5 <= angle_deg < 67.5:
        return 1 #'North-East'
    elif 67.5 <= angle_deg < 112.5:
        return 0 #'North'
    elif 112.5 <= angle_deg < 157.5:
        return 7 #'North-West'
    elif 157.5 <= angle_deg <= 180 or -180 <= angle_deg < -157.5:
        return 6 #'West'
    elif -157.5 <= angle_deg < -112.5:
        return 5 #'South-West'
    elif -112.5 <= angle_deg < -67.5:
        return 4 #'South'
    elif -67.5 <= angle_deg < -22.5:
        return 3 #'South-East'
    else:
        return ValueError

def add_jammer_direction(data):
    """
    Add a column 'jammer_direction' to the DataFrame indicating the cardinal direction of the jammer.

    Args:
        data (pd.DataFrame): The DataFrame containing 'jammer_position' and 'node_positions'.

    Returns:
        pd.DataFrame: The DataFrame with the added 'jammer_direction' column.
    """
    # Calculate the centroid for each row and determine the jammer direction
    data['centroid'] = data['node_positions'].apply(calculate_centroid)
    data['jammer_direction'] = data.apply(lambda row: determine_direction(row['jammer_position'], row['centroid']), axis=1)

    # Drop the centroid column as it's no longer needed
    data.drop(columns=['centroid'], inplace=True)

    return data


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def filter_data(data, base_shapes, experiments_path):
    """
    Filter the data based on the dataset column and save filtered data to disk.

    Args:
        data (pd.DataFrame): The input data.
        base_shapes (list): List of base shapes to filter by.
        experiments_path (str): Path to save the filtered data.

    Returns:
        dict: A dictionary of filtered DataFrames.
    """
    filtered_data = {}

    for base_shape in base_shapes:
        # Exact matching for base shape and base shape all jammed
        exact_base = data['dataset'] == base_shape
        exact_base_all_jammed = data['dataset'] == f"{base_shape}_all_jammed"
        filtered_base = data[exact_base | exact_base_all_jammed]

        if not filtered_base.empty:
            filtered_data[f'{base_shape}'] = filtered_base
            filtered_base.to_pickle(os.path.join(experiments_path, f'{base_shape}_dataset.pkl'))

        # Exact matching for base shape jammer outside region and base shape all jammed jammer outside region
        exact_jammer_outside = data['dataset'] == f"{base_shape}_jammer_outside_region"
        exact_jammer_outside_all_jammed = data['dataset'] == f"{base_shape}_all_jammed_jammer_outside_region"
        filtered_jammer_outside = data[exact_jammer_outside | exact_jammer_outside_all_jammed]

        if not filtered_jammer_outside.empty:
            filtered_data[f'{base_shape}_jammer_outside'] = filtered_jammer_outside
            filtered_jammer_outside.to_pickle(os.path.join(experiments_path, f'{base_shape}_jammer_outside_dataset.pkl'))

    return filtered_data

def split_datasets(data, experiments_path):
    """
    Preprocess and split the dataset into train, validation, and test sets with stratification based on dynamically created jammer distance categories.

    Args:
        data (pd.DataFrame): The input data.
        experiments_path (str): Path to save the split datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
    """
    if params['dynamic']:
        # Stratify on the distance category
        train_idx, val_test_idx = train_test_split(
            data.index,  # Use DataFrame index to ensure correct referencing
            test_size=0.3,
            random_state=100
        )

        # Split the test set into validation and test
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=0.6667,  # 20% test / 30% total = 0.6667
            random_state=100
        )
    else:
        # Work on a copy to avoid SettingWithCopyWarning when modifying data
        data = data.copy()

        # Calculate centroid for each entry by processing the node_positions list of lists
        data.loc[:, 'centroid_x'] = data['node_positions'].apply(lambda positions: np.mean([pos[0] for pos in positions]))
        data.loc[:, 'centroid_y'] = data['node_positions'].apply(lambda positions: np.mean([pos[1] for pos in positions]))

        # Extract x and y coordinates from jammer_position
        data.loc[:, 'jammer_x'] = data['jammer_position'].apply(lambda pos: pos[0][0])
        data.loc[:, 'jammer_y'] = data['jammer_position'].apply(lambda pos: pos[0][1])

        # Calculate distance between jammer and centroid
        data.loc[:, 'jammer_distance'] = data.apply(
            lambda row: calculate_distance(
                row['jammer_x'], row['jammer_y'], row['centroid_x'], row['centroid_y']
            ),
            axis=1
        )

        # Create dynamic bins based on min and max jammer distances
        num_bins = 7
        min_distance = data['jammer_distance'].min()
        max_distance = data['jammer_distance'].max()
        bin_edges = np.linspace(min_distance, max_distance, num=num_bins + 1)  # Create bin edges
        bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}m' for i in range(num_bins)]  # Create bin labels

        # Bin distances into categories
        data['distance_category'] = pd.cut(data['jammer_distance'], bins=bin_edges, labels=bin_labels, include_lowest=True)

        # Stratify on the distance category
        train_idx, val_test_idx = train_test_split(
            data.index,  # Use DataFrame index to ensure correct referencing
            test_size=0.3,
            stratify=data['distance_category'],  # Stratify on distance categories
            random_state=100
        )

        # Split the test set into validation and test
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=0.6667,  # 20% test / 30% total = 0.6667
            stratify=data.loc[val_test_idx, 'distance_category'],  # Stratify on distance categories
            random_state=100
        )

    print("Overlap immediately post-split:", set(train_idx) & set(val_test_idx))

    # Convert indices back to DataFrame subsets
    train_df = data.loc[train_idx]
    val_df = data.loc[val_idx]
    test_df = data.loc[test_idx]

    # Print sizes to debug
    print("Train size:", len(train_df), "Val size:", len(val_df), "Test size:", len(test_df))
    print("Overlap train-val:", set(train_df.index) & set(val_df.index))
    print("Overlap train-test:", set(train_df.index) & set(test_df.index))
    print("Overlap val-test:", set(val_df.index) & set(test_df.index))

    assert set(train_df.index).isdisjoint(set(val_df.index)), "Train and validation sets overlap!"
    assert set(train_df.index).isdisjoint(set(test_df.index)), "Train and test sets overlap!"
    assert set(val_df.index).isdisjoint(set(test_df.index)), "Validation and test sets overlap!"

    # # confirm the split ratio
    # print("Original Data Size:", data.shape[0])
    # print("Training Data Size:", train_df.shape[0], "=>", (train_df.shape[0] / data.shape[0]) * 100, "%")
    # print("Validation Data Size:", val_df.shape[0], "=>", (val_df.shape[0] / data.shape[0]) * 100, "%")
    # print("Test Data Size:", test_df.shape[0], "=>", (test_df.shape[0] / data.shape[0]) * 100, "%")

    # Check for overlapping indices
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)

    assert train_indices.isdisjoint(val_indices), "Train and validation sets overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and test sets overlap!"
    assert val_indices.isdisjoint(test_indices), "Validation and test sets overlap!"

    print("All sets are independent.")

    # # Save split datasets to disk
    # train_df.to_pickle(os.path.join(experiments_path, 'train_dataset.pkl'))
    # val_df.to_pickle(os.path.join(experiments_path, 'validation_dataset.pkl'))
    # test_df.to_pickle(os.path.join(experiments_path, 'test_dataset.pkl'))

    return train_df, val_df, test_df

def process_data(data, experiments_path):
    """
    Process the data by filtering and splitting it.

    Args:
        data (pd.DataFrame): The input data.
        experiments_path (str): Path to save the processed data.

    Returns:
        dict: A dictionary of filtered and split datasets.
    """
    if params['dynamic']:
        split_datasets_dict = {}
        train_df, val_df, test_df = split_datasets(data, experiments_path)
        split_datasets_dict['dynamic'] = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    else:
        # Define base shapes for filtering
        base_shapes = ['circle', 'triangle', 'rectangle', 'random']

        # Step 1: Filter the data
        filtered_data = filter_data(data, base_shapes, experiments_path)

        # Step 2: Split each filtered dataset
        split_datasets_dict = {}
        for key, filtered_df in filtered_data.items():
            train_df, val_df, test_df = split_datasets(filtered_df, experiments_path)
            split_datasets_dict[key] = {
                'train': train_df,
                'validation': val_df,
                'test': test_df
            }

    return split_datasets_dict

#
# #
# #
# #
# def calculate_distance(x1, y1, x2, y2):
#     """Calculate Euclidean distance between two points."""
#     return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
#
# def split_datasets(data, data_class):
#     """
#     Split the dataset into train, validation, and test sets with stratification based on jammer distance and dataset column.
#
#     Args:
#         data (pd.DataFrame): The input data.
#
#     Returns:
#         Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
#     """
#     # Calculate centroid for each entry by processing the node_positions list of lists
#     data['centroid_x'] = data['node_positions'].apply(lambda positions: np.mean([pos[0] for pos in positions]))
#     data['centroid_y'] = data['node_positions'].apply(lambda positions: np.mean([pos[1] for pos in positions]))
#
#     # Extract x and y coordinates from jammer_position
#     data['jammer_x'] = data['jammer_position'].apply(lambda pos: pos[0][0])
#     data['jammer_y'] = data['jammer_position'].apply(lambda pos: pos[0][1])
#
#     # Calculate distance between jammer and centroid
#     data['jammer_distance'] = data.apply(
#         lambda row: calculate_distance(
#             row['jammer_x'], row['jammer_y'], row['centroid_x'], row['centroid_y']
#         ),
#         axis=1
#     )
#
#     # Bin distances into categories
#     bins = [0, 500, 1000, 1500, np.inf]  # Define distance bins
#     labels = ['0-500m', '500-1000m', '1000-1500m', '>1500m']  # Define bin labels
#     data['distance_category'] = pd.cut(data['jammer_distance'], bins=bins, labels=labels)
#
#     # Stratify on the compound key
#     train_idx, val_test_idx = train_test_split(
#         np.arange(len(data)),
#         test_size=0.3,
#         stratify=data['distance_category'],  # Stratify on compound key
#         random_state=100
#     )
#
#     # Split the test set into validation and test
#     val_idx, test_idx = train_test_split(
#         val_test_idx,
#         test_size=0.6667,  # 20% test / 30% total = 0.6667
#         stratify=data.iloc[val_test_idx]['distance_category'],  # Stratify on compound key
#         random_state=100
#     )
#
#     # Convert indices back to DataFrame subsets
#     train_df = data.iloc[train_idx].reset_index(drop=True)
#     val_df = data.iloc[val_idx].reset_index(drop=True)
#     test_df = data.iloc[test_idx].reset_index(drop=True)
#
#     return train_df, val_df, test_df

# def save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path):
#     """
#     Process the combined train, validation, and test data, and save them to disk as .pkl files.
#
#     Args:
#         combined_train_df (pd.DataFrame): DataFrame containing combined training data.
#         combined_val_df (pd.DataFrame): DataFrame containing combined validation data.
#         combined_test_df (pd.DataFrame): DataFrame containing combined test data.
#         experiments_path (str): The path where the processed data will be saved.
#     """
#     logging.info("Saving data")
#
#     # Define file paths
#     train_file_path = os.path.join(experiments_path, 'train_dataset.pkl')
#     val_file_path = os.path.join(experiments_path, 'val_dataset.pkl')
#     test_file_path = os.path.join(experiments_path, 'test_dataset.pkl')
#
#     # Save the combined DataFrame subsets as .pkl files
#     with open(train_file_path, 'wb') as f:
#         pickle.dump(combined_train_df, f)
#     with open(val_file_path, 'wb') as f:
#         pickle.dump(combined_val_df, f)
#     with open(test_file_path, 'wb') as f:
#         pickle.dump(combined_test_df, f)
#
#     # Dataset types for specific filtering
#     if params['dynamic']:
#         dataset_types = ['guided_path_data', 'linear_path_data']
#     else:
#         dataset_types = ['circle', 'triangle', 'rectangle', 'random', 'circle_jammer_outside_region',
#                          'triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
#                          'random_jammer_outside_region', 'all_jammed', 'all_jammed_jammer_outside_region']
#
#     for dataset in dataset_types:
#         # Create filtered subsets based on dataset type
#         train_subset = combined_train_df[combined_train_df['dataset'] == dataset]
#         val_subset = combined_val_df[combined_val_df['dataset'] == dataset]
#         test_subset = combined_test_df[combined_test_df['dataset'] == dataset]
#
#         # Save each subset as .pkl if it is not empty
#         if not train_subset.empty:
#             train_subset_path = os.path.join(experiments_path, f'{dataset}_train_dataset.pkl')
#             with open(train_subset_path, 'wb') as f:
#                 pickle.dump(train_subset, f)
#
#         if not val_subset.empty:
#             val_subset_path = os.path.join(experiments_path, f'{dataset}_val_dataset.pkl')
#             with open(val_subset_path, 'wb') as f:
#                 pickle.dump(val_subset, f)
#
#         if not test_subset.empty:
#             test_subset_path = os.path.join(experiments_path, f'{dataset}_test_dataset.pkl')
#             with open(test_subset_path, 'wb') as f:
#                 pickle.dump(test_subset, f)


def save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path, data_class):
    """
    Process the combined train, validation, and test data, and save them to disk as .pkl files.

    Args:
        combined_train_df (pd.DataFrame): DataFrame containing combined training data.
        combined_val_df (pd.DataFrame): DataFrame containing combined validation data.
        combined_test_df (pd.DataFrame): DataFrame containing combined test data.
        experiments_path (str): The path where the processed data will be saved.
    """
    logging.info("Saving data")

    # Define file paths
    train_file_path = os.path.join(experiments_path, f'{data_class}_train_dataset.pkl')
    val_file_path = os.path.join(experiments_path, f'{data_class}_val_dataset.pkl')
    test_file_path = os.path.join(experiments_path, f'{data_class}_test_dataset.pkl')

    # Save the combined DataFrame subsets as .pkl files
    with open(train_file_path, 'wb') as f:
        pickle.dump(combined_train_df, f)
    with open(val_file_path, 'wb') as f:
        pickle.dump(combined_val_df, f)
    with open(test_file_path, 'wb') as f:
        pickle.dump(combined_test_df, f)

    # # Define base shapes
    # base_shapes = ['circle', 'triangle', 'rectangle', 'random']
    #
    # # Define dataframes for each type
    # dataframes = {'train': combined_train_df, 'val': combined_val_df, 'test': combined_test_df}
    #
    # # Filtering and saving subsets based on dataset types
    # for base_shape in base_shapes:
    #     for dtype, df in dataframes.items():
    #         # Add jammer_placement_bin column with default as None for safety
    #         # df['jammer_placement_bin'] = None
    #
    #         # Exact matching for base shape and base shape all jammed
    #         exact_base = df['dataset'] == base_shape
    #         print(df[exact_base]['dataset'])
    #         exact_base_all_jammed = df['dataset'] == f"{base_shape}_all_jammed"
    #         print(df[exact_base_all_jammed]['dataset'])
    #         # df.loc[exact_base | exact_base_all_jammed, 'jammer_placement_bin'] = 0
    #         filtered_base = df[exact_base | exact_base_all_jammed]
    #         if not filtered_base.empty:
    #             filtered_base.to_pickle(os.path.join(experiments_path, f'{base_shape}_{dtype}_dataset.pkl'))
    #
    #         # Exact matching for base shape jammer outside region and base shape all jammed jammer outside region
    #         exact_jammer_outside = df['dataset'] == f"{base_shape}_jammer_outside_region"
    #         exact_jammer_outside_all_jammed = df['dataset'] == f"{base_shape}_all_jammed_jammer_outside_region"
    #         # df.loc[exact_jammer_outside | exact_jammer_outside_all_jammed, 'jammer_placement_bin'] = 1
    #         # print(df[exact_jammer_outside_all_jammed]['dataset'])
    #         filtered_jammer_outside = df[exact_jammer_outside | exact_jammer_outside_all_jammed]
    #         if not filtered_jammer_outside.empty:
    #             filtered_jammer_outside.to_pickle(os.path.join(experiments_path, f'{base_shape}_jammer_outside_region_{dtype}_dataset.pkl'))


def downsample_data(instance):
    """
    Downsamples the data of an instance object based on a fixed number of maximum nodes.

    Args:
        instance (Instance): The instance to downsample.
    """
    max_nodes = params['max_nodes']
    num_original_nodes = len(instance.node_positions)

    if num_original_nodes <= max_nodes:
        return instance  # No downsampling needed

    window_size = num_original_nodes // max_nodes
    num_windows = max_nodes

    # Create downsampled attributes
    downsampled_positions = []
    downsampled_noise_values = []
    downsampled_angles = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        downsampled_positions.append(np.mean(instance.node_positions[start_idx:end_idx], axis=0))
        downsampled_noise_values.append(np.mean(instance.node_noise[start_idx:end_idx]))
        if 'angle_of_arrival' in params['required_features']:
            downsampled_angles.append(np.mean(instance.angle_of_arrival[start_idx:end_idx]))

    # Update instance with downsampled data
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise_values)
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = np.array(downsampled_angles)

    return instance

# Noise based downsampling
def downsample_data_by_highest_noise(instance):
    """
    Downsamples the data of an instance object by keeping the nodes with the highest noise values.

    Args:
        instance (Instance): The instance to downsample.
        num_nodes_to_keep (int): The number of nodes to retain based on the highest noise values.

    Returns:
        Instance: The downsampled instance.
    """
    # Get the indices of the nodes with the highest noise values
    top_indices = np.argsort(instance.node_noise)[-params['max_nodes']:]

    # Sort the indices to maintain the order in the dataset
    top_indices = np.sort(top_indices)

    # Extract the corresponding positions, noise values, and angles (if applicable)
    downsampled_positions = instance.node_positions[top_indices]
    downsampled_noise_values = instance.node_noise[top_indices]

    if 'angle_of_arrival' in params['required_features']:
        downsampled_angles = instance.angle_of_arrival[top_indices]
        instance.angle_of_arrival = np.array(downsampled_angles)

    # Update instance with the downsampled data
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise_values)

    return instance


# def bin_nodes(nodes, grid_meters):
#     """Bin nodes by averaging positions, noise levels, and angle of arrival within each grid cell."""
#     max_nodes = params['max_nodes']
#     nodes['x_bin'] = (nodes['x'] // grid_meters).astype(int)
#     nodes['y_bin'] = (nodes['y'] // grid_meters).astype(int)
#     binned = nodes.groupby(['x_bin', 'y_bin']).mean().reset_index()
#     binned['x'] = (binned['x_bin'] + 0.5) * grid_meters
#     binned['y'] = (binned['y_bin'] + 0.5) * grid_meters
#     # Sort by noise_level and keep the top max_nodes
#     binned = binned.sort_values(by='noise_level', ascending=False).head(max_nodes)
#     return binned


def bin_nodes(node_df, grid_meters):
    """Bin nodes by averaging positions, noise levels, and angle of arrival within each grid cell for both polar and Cartesian coordinates,
    and merge results into a single DataFrame containing both coordinate systems.

    Args:
        node_df (pd.DataFrame): DataFrame containing node data in both polar and Cartesian coordinates.
        grid_meters (int): The size of each grid cell for binning.
        max_nodes (int): Maximum number of nodes to keep after binning.

    Returns:
        pd.DataFrame: Binned nodes with averaged positions and other features for Cartesian and polar coordinates.
    """
    # Handle Cartesian coordinates
    node_df['x_bin'] = (node_df['x'] // grid_meters).astype(int)
    node_df['y_bin'] = (node_df['y'] // grid_meters).astype(int)
    node_df['z_bin'] = (node_df['z'] // grid_meters).astype(int)

    binned_cartesian = node_df.groupby(['x_bin', 'y_bin', 'z_bin']).agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'noise_level': 'mean'
    }).reset_index()

    # Drop the bin columns as they are no longer needed
    binned_cartesian.drop(columns=['x_bin', 'y_bin', 'z_bin'], inplace=True)

    # Handle Polar coordinates by converting to Cartesian first
    node_df['x_polar'] = node_df['r'] * node_df['sin_theta'] * node_df['cos_phi']
    node_df['y_polar'] = node_df['r'] * node_df['sin_theta'] * node_df['sin_phi']
    node_df['z_polar'] = node_df['r'] * node_df['cos_theta']

    node_df['x_polar_bin'] = (node_df['x_polar'] // grid_meters).astype(int)
    node_df['y_polar_bin'] = (node_df['y_polar'] // grid_meters).astype(int)
    node_df['z_polar_bin'] = (node_df['z_polar'] // grid_meters).astype(int)

    binned_polar = node_df.groupby(['x_polar_bin', 'y_polar_bin', 'z_polar_bin']).agg({
        'x_polar': 'mean',
        'y_polar': 'mean',
        'z_polar': 'mean'
    }).reset_index()

    # Convert averaged Cartesian coordinates back to polar
    # Correct calculation of r, theta, and phi
    binned_polar['r'] = np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2 + binned_polar['z_polar'] ** 2)
    # binned_polar['theta'] = np.arccos(binned_polar['z_polar'] / binned_polar['r'])
    binned_polar['theta'] = np.arctan2(np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2), binned_polar['z_polar'])
    binned_polar['phi'] = np.arctan2(binned_polar['y_polar'], binned_polar['x_polar'])

    # calculate sin and cos of theta and phi if needed for further processing
    binned_polar['sin_theta'] = np.sin(binned_polar['theta'])
    binned_polar['cos_theta'] = np.cos(binned_polar['theta'])
    binned_polar['sin_phi'] = np.sin(binned_polar['phi'])
    binned_polar['cos_phi'] = np.cos(binned_polar['phi'])

    # binned_polar['r'] = np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2 + binned_polar['z_polar'] ** 2)
    # binned_polar['sin_theta'] = binned_polar['y_polar'] / np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2)
    # binned_polar['cos_theta'] = binned_polar['x_polar'] / np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2)
    # binned_polar['sin_az'] = binned_polar['z_polar'] / binned_polar['r']
    # binned_polar['cos_az'] = np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2) / binned_polar['r']

    # Drop unnecessary columns
    binned_polar.drop(columns=['x_polar_bin', 'y_polar_bin', 'z_polar_bin', 'x_polar', 'y_polar', 'z_polar'], inplace=True)

    # Merge the two DataFrames
    binned = pd.concat([binned_cartesian, binned_polar], axis=1)

    # Drop rows with NaN values
    binned = binned.dropna()

    # Sort by noise_level and keep the top max_nodes
    binned = binned.sort_values(by='noise_level', ascending=False).head(params['max_nodes'])

    # Ensure at least 3 nodes are returned
    if len(binned) < 3:
        # print(node_df)
        # print("Warning: Binning resulted in fewer than 3 nodes. Returning original dataset.")
        return node_df  # Fall back to the original dataset
    else:
        return binned


def hybrid_downsampling_pipeline(instance):
    """
    Combines spatial binning, time window averaging, and noise filtering in sequence.

    Args:
        instance: The data instance to downsample.
        filtering_proportion: Proportion of nodes to retain after noise filtering (e.g., 0.6 for 60%).

    Returns:
        instance: Downsampled instance.
    """
    # Step 1: Spatial Binning
    node_df = pd.DataFrame({
        'x': instance.node_positions[:, 0],
        'y': instance.node_positions[:, 1],
        'noise_level': instance.node_noise
    })
    if 'angle_of_arrival' in params['required_features']:
        node_df['angle_of_arrival'] = instance.angle_of_arrival

    binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])

    # Step 2: Time Window Averaging
    downsampled_positions, downsampled_noise, downsampled_angles = [], [], []
    max_nodes = params['max_nodes']
    num_binned_nodes = len(binned_nodes)

    window_size = max(1, num_binned_nodes // max_nodes)
    for i in range(0, num_binned_nodes, window_size):
        batch = binned_nodes.iloc[i:i + window_size]
        downsampled_positions.append(batch[['x', 'y']].mean().to_numpy())
        downsampled_noise.append(batch['noise_level'].mean())
        if 'angle_of_arrival' in params['required_features']:
            downsampled_angles.append(batch['angle_of_arrival'].mean())

    # Update instance after time window averaging
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise)
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = np.array(downsampled_angles)

    # Step 3: Noise Filtering
    num_filtered_nodes = max(1, int(max_nodes * params['filtering_proportion']))
    high_noise_indices = np.argsort(instance.node_noise)[-num_filtered_nodes:]
    instance.node_positions = instance.node_positions[high_noise_indices]
    instance.node_noise = instance.node_noise[high_noise_indices]
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = instance.angle_of_arrival[high_noise_indices]

    return instance


def add_jammed_column(data, threshold=-55):
    data['jammed_at'] = None
    for i, noise_list in enumerate(data['node_noise']):
        # print("noise list: ", noise_list)
        # Check if noise_list is a valid non-empty list
        if not isinstance(noise_list, list) or len(noise_list) == 0:
            raise ValueError(f"Invalid or empty node_noise list at row {i}")

        count = 0
        jammed_index = None  # Store the index of the third noise > threshold

        for idx, noise in enumerate(noise_list):
            if noise > threshold:
                count += 1
                # Save the index of the third noise sample that exceeds the threshold
                if count == 3:
                    jammed_index = idx + 1
                    break

        # Save the index of the third "jammed" sample or handle no jamming detected
        if jammed_index is not None:
            data.at[i, 'jammed_at'] = jammed_index
        else:
            # print(data['node_noise'][710])
            raise ValueError(f"No sufficient jammed noise samples found for row {i}")

    return data


def load_data(params, data_class, experiments_path=None):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    if params['inference']:
        # Load the test data only for inference mode
        test_set_name = [data_class]
        for test_data in test_set_name:
            print(f"dataset: {test_data}")
            file_path = os.path.join(experiments_path, f'{test_data}.pkl')
            with open(file_path, 'rb') as f:
                test_df = pickle.load(f)

            print(test_df.columns)
            return None, None, test_df
    else:
        datasets = [params['dataset_path']]
        for dataset in datasets:
            print(f"dataset: {dataset}")

            # Load the interpolated dataset from the pickle file
            with open(dataset, "rb") as f:
                data_list = []
                try:
                    # Read each dictionary entry in the list and add to data_list
                    while True:
                        data_list.append(pickle.load(f))
                except EOFError:
                    pass  # End of file reached

            # Convert the list of dictionaries to a DataFrame
            if isinstance(data_list[0], pd.DataFrame):
                data = data_list[0]
            else:
                data = pd.DataFrame(data_list)

            print(len(data))
            print("COLUMNS: ", data.columns)

            # Add additional columns required for processing
            data['id'] = range(1, len(data) + 1)

            if not params['dynamic']:
                convert_data_type(data, load_saved_data=False)

            # Add jammed column
            if params['dynamic']:
                data = add_jammed_column(data, threshold=-55)

            # Create train test splits
            # data = add_jammer_direction(data)
            data.reset_index(inplace=True)
            print(data.index.is_unique)

            # # Add average of node_positions and node_noise to each row
            # data['node_positions'] = data['node_positions'].apply(
            #     lambda positions: positions + [np.mean(positions, axis=0).tolist()]
            # )
            # data['node_noise'] = data['node_noise'].apply(
            #     lambda noise: noise + [np.mean(noise)]
            # )

            split_datasets_dict = process_data(data, experiments_path)

            # Access the train, validation, and test DataFrame for a specific data class
            train_df = split_datasets_dict[data_class]['train']
            val_df = split_datasets_dict[data_class]['validation']
            test_df = split_datasets_dict[data_class]['test']

            # Process and save the combined data
            save_datasets(train_df, val_df, test_df, experiments_path, data_class)

            # Convert rows to tuples
            train_tuples = train_df.apply(tuple, axis=1)
            val_tuples = val_df.apply(tuple, axis=1)
            test_tuples = test_df.apply(tuple, axis=1)

            # Check for duplicates across all DataFrames
            duplicates_in_val_from_train = val_tuples.isin(train_tuples)
            duplicates_in_val_from_test = val_tuples.isin(test_tuples)

            # Print results
            print(f"Rows in val_df duplicated in train_df: {duplicates_in_val_from_train.sum()}")
            print(f"Rows in val_df duplicated in test_df: {duplicates_in_val_from_test.sum()}")

            if duplicates_in_val_from_train.any():
                print("Duplicated rows in val_df from train_df:")
                print(val_df[duplicates_in_val_from_train])

            if duplicates_in_val_from_test.any():
                print("Duplicated rows in val_df from test_df:")
                print(val_df[duplicates_in_val_from_test])

            convert_to_polar(train_df)
            convert_to_polar(val_df)
            convert_to_polar(test_df)

        # print(train_df['dataset'])
        return train_df, val_df, test_df

def shuffle_positions_and_noise(data):
    # Iterate over each row and shuffle the positions and noise
    for idx, row in data.iterrows():
        paired_list = list(zip(row['node_positions'], row['node_noise']))
        random.shuffle(paired_list)  # Shuffle the paired list
        # Unzip the paired list back into node_positions and node_noise
        positions, noises = zip(*paired_list)
        # Update the DataFrame with shuffled data
        data.at[idx, 'node_positions'] = list(positions)
        data.at[idx, 'node_noise'] = list(noises)

    return data

def get_params_hash(params):
    # Create a copy of the params dictionary.
    params_copy = params.copy()
    # Remove the 'num_workers' key from the copy.
    if 'num_workers' in params_copy:
        del params_copy['num_workers']
    elif 'max_epochs' in params_copy:
        del params_copy['epochs']
    # Serialize the dictionary with sorted keys to ensure consistent order.
    params_str = json.dumps(params_copy, sort_keys=True)
    # Compute and return the MD5 hash of the serialized string.
    return hashlib.md5(params_str.encode()).hexdigest()

def create_data_loader(params, train_data, val_data, test_data, experiment_path):
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
    deg_histogram = None

    if params['inference']:
        logging.info('Computing testing data')
        test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=params['test_discrite_coeff'])
        test_loader = DataLoader(test_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, num_workers=0)

        return None, None, test_loader
    else:
        # # Generate a unique identifier for the current params
        # params_hash = get_params_hash(params)
        # cache_path = os.path.join(experiment_path, f"data_loader_{params_hash}.pkl")
        # os.makedirs(experiment_path, exist_ok=True)
        #
        # if os.path.exists(cache_path):
        #     # Load cached data loaders
        #     with open(cache_path, 'rb') as f:
        #         train_loader, val_loader, test_loader = pickle.load(f)
        #     logging.info("Loaded cached data loaders")
        # else:
        # Create data loaders and save them if cache doesn't exist
        logging.info("Creating data loaders")
        train_loader, val_loader, test_loader = generate_data_loaders(params, train_data, val_data, test_data)

            # # Save data loaders to cache
            # with open(cache_path, 'wb') as f:
            #     pickle.dump((train_loader, val_loader, test_loader), f)
            # logging.info("Saved data loaders")

        if params['model'] == 'PNA':
            deg_histogram = compute_degree_histogram(train_loader)

        return train_loader, val_loader, test_loader, deg_histogram


def generate_data_loaders(params, train_data, val_data, test_data):
    train_dataset = TemporalGraphDataset(train_data, test=False)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=params['num_workers'])

    val_dataset = TemporalGraphDataset(val_data, test=True, discretization_coeff=params['val_discrite_coeff'])
    val_loader = DataLoader(val_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=params['test_discrite_coeff'])
    test_loader = DataLoader(test_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader, test_loader


from torch_geometric.utils import degree
def compute_degree_histogram(data_loader):
    max_degree = 0
    deg_histogram = None
    for data in data_loader:
        d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
        batch_max_degree = d.max().item()
        max_degree = max(max_degree, batch_max_degree)
        if deg_histogram is None:
            deg_histogram = torch.bincount(d, minlength=max_degree + 1)
        else:
            if batch_max_degree > deg_histogram.numel() - 1:
                new_histogram = torch.zeros(batch_max_degree + 1, dtype=deg_histogram.dtype)
                new_histogram[:deg_histogram.numel()] = deg_histogram
                deg_histogram = new_histogram
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())
    return deg_histogram


def safe_convert_list(row: str, data_type: str):
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
            return [[float(pos) for pos in result]]
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
        elif data_type == 'jammed_at':
            return int(row)
        elif data_type == 'jammer_power':
            return float(row)
        elif data_type == 'num_samples':
            return float(row)
        elif data_type == 'sigma':
            return float(row)
        elif data_type == 'jammer_power':
            return float(row)
        elif data_type == 'id':
            return int(row)
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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph(positions, edge_index, node_features, edge_weights=None, jammer_positions=None, show_weights=False, perc_completion=None, id=None, jammer_power=None):
    G = nx.Graph()

    # Check if positions are tensors and convert them
    if isinstance(positions, torch.Tensor):
        positions = positions.numpy()
    elif isinstance(positions, list) and isinstance(positions[0], torch.Tensor):
        positions = [pos.numpy() for pos in positions]

    # Ensure node_features is a numpy array for easier handling
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.numpy()

    # Add nodes with features and positions
    for i, pos in enumerate(positions):
        # Assuming RSSI is the last feature in node_features array
        G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2])

    # Convert edge_index to numpy array if it's a tensor
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    edge_index = edge_index.T  # Ensure edge_index is transposed if necessary

    # Position for drawing, ensuring positions are tuples
    pos = {i: (p[0], p[1]) for i, p in enumerate(positions)}

    # Add edges to the graph
    if edge_weights is not None:
        edge_weights = edge_weights.numpy() if isinstance(edge_weights, torch.Tensor) else edge_weights
        for idx, (start, end) in enumerate(edge_index):
            G.add_edge(start, end, weight=edge_weights[idx])
    else:
        for start, end in edge_index:
            G.add_edge(start, end)

    # Draw the graph nodes
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=20)  # Use a default color

    # Draw the last index node with a different color (e.g., red)
    last_node_index = len(positions) - 1  # Assuming last index is what you're referring to
    nx.draw_networkx_nodes(G, pos, nodelist=[last_node_index], node_color='red', node_size=60)

    # Draw the edges
    if show_weights and edge_weights is not None:
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        weights = 10 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)  # Normalize and scale
        nx.draw_networkx_edges(G, pos, width=0.5)
    else:
        nx.draw_networkx_edges(G, pos)

    # Optionally draw jammer positions
    if jammer_positions is not None:
        if isinstance(jammer_positions, torch.Tensor):
            jammer_positions = jammer_positions.numpy()
        for jp in jammer_positions:
            plt.scatter(jp[0], jp[1], color='red', s=50, marker='x', label='Jammer')

    # Additional plot settings
    if perc_completion is not None:
        plt.title(f"Graph {id} - {perc_completion:.2f}% trajectory since start", fontsize=15)
    else:
        plt.title(f"Graph {id}", fontsize=15)

    plt.axis('off')
    plt.show()


# def plot_graph(positions, node_features, edge_weights=None, jammer_positions=None, show_weights=False, perc_completion=None, id=None, jammer_power=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Ensure positions and features are numpy arrays for easier handling
#     positions = np.array(positions)
#     node_features = np.array(node_features)
#     if jammer_positions is not None:
#         jammer_positions = np.array(jammer_positions)
#
#     # Extract positions for x, y, z
#     x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
#
#     # Node color based on noise value, assuming RSSI is at index 2
#     noise_values = node_features[:, 2]
#     node_colors = ['red' if noise > -55 else 'blue' for noise in noise_values]
#
#     # Draw the graph nodes without edges and annotations
#     sc = ax.scatter(x, y, z, c=node_colors, depthshade=True)
#
#     # Optionally draw jammer positions
#     if jammer_positions is not None:
#         ax.scatter(jammer_positions[0][0], jammer_positions[0][1], jammer_positions[0][2], color='red', marker='x', label='Jammer')
#
#         # Assuming jammer_power is available in an array, where each entry corresponds to a jammer
#         if jammer_power is not None:
#             ax.text(jammer_positions[0][0], jammer_positions[0][1], jammer_positions[0][2], f'Power: {jammer_power:.1f} dB', color='black')
#
#     # Title setup with optional percentage completion
#     if perc_completion is not None and id is not None:
#         perc_completion_title = f"Graph {id} {perc_completion:.2f}% trajectory since start"
#         plt.title(perc_completion_title)
#
#     plt.axis('off')  # Turn off the axis
#     ax.legend()
#     plt.show()


# ORIGINAL
# def create_torch_geo_data(instance: Instance) -> Data:
#     """
#     Create a PyTorch Geometric Data object from a row of the dataset.
#
#     Args:
#         row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.
#
#     Returns:
#         Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
#     """
#     # Downsample (binning and highest noise)
#     if params['downsampling']:
#         if params['ds_method'] == 'noise':
#             # Convert positions to a DataFrame to use bin_nodes
#             if 'angle_of_arrival' in params['required_features']:
#                 node_df = pd.DataFrame({
#                     'x': instance.node_positions[:, 0],
#                     'y': instance.node_positions[:, 1],
#                     'noise_level': instance.node_noise,
#                     'angle_of_arrival': instance.angle_of_arrival  # Include angle of arrival
#                 })
#                 binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])
#                 instance.node_positions = binned_nodes[['x', 'y']].to_numpy()
#                 instance.node_noise = binned_nodes['noise_level'].to_numpy()
#                 instance.angle_of_arrival = binned_nodes['angle_of_arrival'].to_numpy()  # Update angle of arrival
#             else:
#                 # Convert positions to a DataFrame to use bin_nodes
#                 node_df = pd.DataFrame({
#                     'x': instance.node_positions[:, 0],
#                     'y': instance.node_positions[:, 1],
#                     'noise_level': instance.node_noise
#                 })
#                 binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])
#                 instance.node_positions = binned_nodes[['x', 'y']].to_numpy()
#                 instance.node_noise = binned_nodes['noise_level'].to_numpy()
#
#             # instance = downsample_data_by_highest_noise(instance)
#         elif params['ds_method'] == 'time_window_avg':
#             instance = downsample_data(instance)
#         elif params['ds_method'] == 'hybrid':
#             instance = hybrid_downsampling_pipeline(instance)
#         else:
#             raise ValueError("Undefined downsampling method")
#
#     # Preprocess instance data
#     center_coordinates_instance(instance)
#     if params['norm'] == 'minmax':
#         apply_min_max_normalization_instance(instance)
#     elif params['norm'] == 'unit_sphere':
#         apply_min_max_normalization_instance_noise(instance)
#         apply_unit_sphere_normalization(instance)
#
#     if 'angle_of_arrival' in params['required_features']:
#         # Convert AoA from degrees to radians
#         aoa_radians = np.radians(instance.angle_of_arrival)
#
#         # Create node features without adding an extra list around the numpy array
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None],  # Ensure node_noise is reshaped to (n, 1)
#             np.sin(aoa_radians[:, None]),
#             np.cos(aoa_radians[:, None])
#         ], axis=1)
#     else:
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None]
#         ], axis=1)
#
#
#     # Convert to 2D tensor
#     node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
#
    # # Preparing edges and weights
    # positions = instance.node_positions_cart
    # if params['num_neighbors'] == 'fc':
    #     num_neighbors = 10000000
    # else:
    #     num_neighbors = params['num_neighbors']
    # if params['edges'] == 'knn':
    #     num_samples = positions.shape[0]
    #     k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
    #     nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
    #     distances, indices = nbrs.kneighbors(positions)
    #     edge_index, edge_weight = [], []
    #
    #     # Add self loop
    #     # for i in range(indices.shape[0]):
    #     #     edge_index.extend([[i, i]])
    #     #     edge_weight.extend([1.0])  # Self-loops can have a weight of 0 or another meaningful value
    #     #     for j in range(1, indices.shape[1]):
    #     #         edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
    #     #         # Inverse the distance, adding a small epsilon to avoid division by zero
    #     #         inv_distance = 1 / (distances[i, j] + 1e-5)
    #     #         edge_weight.extend([inv_distance, inv_distance])
    #
    #     # Define the scaling parameter alpha for the Gaussian decay function
    #     # Define the scaling parameter alpha for the Gaussian decay function
    #     alpha = 1.0  # Adjust this based on your experimentation to find the best fit
    #
    #     for i in range(indices.shape[0]):
    #         for j in range(1, indices.shape[1]):  # Skip the first index as it's the node itself
    #             edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
    #             # Apply Gaussian decay to the distance
    #             gaussian_weight = np.exp(-alpha * distances[i, j])
    #             edge_weight.extend([gaussian_weight, gaussian_weight])
    #
    # else:
    #     raise ValueError("Unsupported edge specification")
    #
    # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    #
    # # Apply DropEdge
    # if 'drop_edge' in params['aug']:
    #     drop_edge_rate = 0.8
    #     if drop_edge_rate > 0:
    #         num_edges = edge_index.size(1)
    #         num_edges_to_drop = int(drop_edge_rate * num_edges)
    #
    #         # Randomly select edges to drop
    #         edges_to_drop = torch.randperm(num_edges)[:num_edges_to_drop]
    #         mask = torch.ones(num_edges, dtype=torch.bool)
    #         mask[edges_to_drop] = False
    #
    #         # Apply mask to edge_index and edge_weight
    #         edge_index = edge_index[:, mask]
    #         edge_weight = edge_weight[mask]
    #
    # # Add self-loops for renormalization
    # num_nodes = positions.shape[0]
    # self_loop_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
    # self_loop_weights = torch.ones(num_nodes, dtype=torch.float)  # Weight of 1 for self-loops
    #
    # # Combine original edges with self-loops
    # edge_index = torch.cat([edge_index, self_loop_indices], dim=1)
    # edge_weight = torch.cat([edge_weight, self_loop_weights])
    #
    # # Apply renormalization trick
    # row, col = edge_index
    # deg = torch.zeros(num_nodes, dtype=torch.float)
    # deg = deg.scatter_add(0, row, edge_weight)  # Compute degree for each node
    # deg_inv_sqrt = deg.pow(-0.5)
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes (degree 0)
    # norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # Normalize edge weights
    #
    # # Update edge weights with normalized values
    # edge_weight = norm
#
#
#     # Assuming instance.jammer_position is a list or array that can be reshaped
#     jammer_positions = np.array(instance.jammer_position).reshape(-1, params['out_features'])
#
#     # Convert jammer_positions to a tensor
#     y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Convert instance.jammer_power to a tensor and reshape it to match the dimensions of y
#     jammer_power = torch.tensor(instance.jammer_power, dtype=torch.float).reshape(-1, 1)
#
#     # Concatenate jammer_power to y along the appropriate dimension
#     y = torch.cat((y, jammer_power), dim=1)
#
#
#     # jammer_positions = np.array(instance.jammer_position).reshape(-1, params['out_features'])  # Assuming this reshaping is valid based on your data structure
#     # y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Plot
#     # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
#
#     # Create the Data object
#     data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y)
#
#     # Convert geometric information to tensors
#     # data.id = instance.id
#     data.node_positions_center = torch.tensor(instance.node_positions_center, dtype=torch.float)
#     if params['norm'] == 'minmax':
#         data.min_coords = torch.tensor(instance.min_coords, dtype=torch.float)
#         data.max_coords = torch.tensor(instance.max_coords, dtype=torch.float)
#     elif params['norm'] == 'unit_sphere':
#         data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)
#
#     # Store the perc_completion as part of the Data object
#     data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)
#     data.pl_exp = torch.tensor(instance.pl_exp, dtype=torch.float)
#     data.sigma = torch.tensor(instance.sigma, dtype=torch.float)
#     data.jtx = torch.tensor(instance.jammer_power, dtype=torch.float)
#     data.num_samples = torch.tensor(instance.num_samples, dtype=torch.float)
#
#     # Apply pos encoding transform
#     if params['model'] == 'GPS':
#         transform = AddRandomWalkPE(walk_length=20, attr_name='pe')
#         data = transform(data)
#
#     return data


# def create_torch_geo_data(instance: Instance) -> Data:
#     """
#     Create a PyTorch Geometric Data object from a row of the dataset.
#
#     Args:
#         row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.
#
#     Returns:
#         Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
#     """
#     if len(instance.node_positions) < 3:
#         print(instance.node_positions_cart)
#         print('ERROR')
#         quit()
#     # Preprocess instance data
#     if params['norm'] == 'minmax':
#         apply_min_max_normalization_instance(instance)
#     elif params['norm'] == 'unit_sphere':
#         apply_min_max_normalization_instance_noise(instance)
#         apply_unit_sphere_normalization(instance)
#
#     if 'angle_of_arrival' in params['required_features']:
#         # Convert AoA from degrees to radians
#         aoa_radians = np.radians(instance.angle_of_arrival)
#
#         # Create node features without adding an extra list around the numpy array
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None],  # Ensure node_noise is reshaped to (n, 1)
#             np.sin(aoa_radians[:, None]),
#             np.cos(aoa_radians[:, None])
#         ], axis=1)
#     else:
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None],
#             instance.node_positions_cart
#         ], axis=1)
#
#
#     # Convert to 2D tensor
#     node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
#
#     # Preparing edges and weights
#     if params['aug'] == 'drop_edge':
#         # Preparing edges and weights
#         positions = instance.node_positions_cart
#         if params['num_neighbors'] == 'fc':
#             num_neighbors = 10000000
#         else:
#             num_neighbors = params['num_neighbors']
#         if params['edges'] == 'knn':
#             num_samples = positions.shape[0]
#             k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
#             nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
#             distances, indices = nbrs.kneighbors(positions)
#             edge_index, edge_weight = [], []
#
#             # Define the scaling parameter alpha for the Gaussian decay function
#             alpha = 1.0  # Adjust this based on your experimentation to find the best fit
#
#             for i in range(indices.shape[0]):
#                 for j in range(1, indices.shape[1]):  # Skip the first index as it's the node itself
#                     edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
#                     # Apply Gaussian decay to the distance
#                     gaussian_weight = np.exp(-alpha * distances[i, j])
#                     edge_weight.extend([gaussian_weight, gaussian_weight])
#
#         else:
#             raise ValueError("Unsupported edge specification")
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
#         # Apply DropEdge
#         drop_edge_rate = 0.8
#         if drop_edge_rate > 0:
#             num_edges = edge_index.size(1)
#             num_edges_to_drop = int(drop_edge_rate * num_edges)
#
#             # Randomly select edges to drop
#             edges_to_drop = torch.randperm(num_edges)[:num_edges_to_drop]
#             mask = torch.ones(num_edges, dtype=torch.bool)
#             mask[edges_to_drop] = False
#
#             # Apply mask to edge_index and edge_weight
#             edge_index = edge_index[:, mask]
#             edge_weight = edge_weight[mask]
#
#         # Add self-loops for renormalization
#         num_nodes = positions.shape[0]
#         self_loop_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
#         self_loop_weights = torch.ones(num_nodes, dtype=torch.float)  # Weight of 1 for self-loops
#
#         # Combine original edges with self-loops
#         edge_index = torch.cat([edge_index, self_loop_indices], dim=1)
#         edge_weight = torch.cat([edge_weight, self_loop_weights])
#
#         # Apply renormalization trick
#         row, col = edge_index
#         deg = torch.zeros(num_nodes, dtype=torch.float)
#         deg = deg.scatter_add(0, row, edge_weight)  # Compute degree for each node
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes (degree 0)
#         norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # Normalize edge weights
#
#         # Update edge weights with normalized values
#         edge_weight = norm
#     else:
#         positions = instance.node_positions_cart
#         if params['num_neighbors'] == 'fc':
#             num_neighbors = positions.shape[0] - 1
#         else:
#             num_neighbors = params['num_neighbors']
#         if params['edges'] == 'knn':
#             num_samples = positions.shape[0]
#             num_edges = params['num_neighbors']  # Expected that this is set outside this function, e.g., params['num_edges'] = 3
#
#             # Initialize NearestNeighbors with more neighbors than needed to calculate full connections later
#             nbrs = NearestNeighbors(n_neighbors=num_edges + 1, algorithm='auto').fit(positions)
#             distances, indices = nbrs.kneighbors(positions)
#
#             edge_index, edge_weight = [], []
#             alpha = 1.0  # Gaussian decay parameter
#
#             # Add edges based on KNN for all nodes except the last
#             for i in range(num_samples - 1):  # Exclude the last node in this loop
#                 for j in range(1, num_edges + 1):  # Skip the first index (self-loop), limit to num_edges
#                     gaussian_weight = np.exp(-alpha * distances[i, j])
#                     edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
#                     edge_weight.extend([gaussian_weight, gaussian_weight])
#
#             # Handling the last node separately
#             last_node_index = num_samples - 1
#             # Add an edge between the last node and every other node
#             for i in range(num_samples):
#                 if i != last_node_index:
#                     distance = np.linalg.norm(positions[last_node_index] - positions[i])  # Calculate the Euclidean distance
#                     gaussian_weight = np.exp(-alpha * distance)
#                     edge_index.extend([[last_node_index, i], [i, last_node_index]])
#                     edge_weight.extend([gaussian_weight, gaussian_weight])
#
#             # Add self-loops for all nodes
#             for i in range(num_samples):
#                 edge_index.append([i, i])
#                 edge_weight.append(1.0)  # Uniform weight for self-loops
#
#         else:
#             raise ValueError("Unsupported edge specification")
#
#         # Convert edge_index and edge_weight to tensors
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
#     # # Assuming instance.jammer_position is a list or array that can be reshaped
#     # jammer_positions = np.array(instance.jammer_position).reshape(-1, 3)
#     #
#     # # Convert jammer_positions to a tensor
#     # y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Prepare target variables
#     # jammer_positions = np.array(instance.jammer_position).reshape(-1, 2)
#     # print('jammer_positions: ', jammer_positions)
#     # y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Assume instance.jammer_placement_bin is accessible
#     # Assuming `instance.jammer_placement_bin` is a scalar representing class index
#     # jammer_placement_bin = torch.tensor([instance.jammer_placement_bin], dtype=torch.long)
#     # print('jammer_placement_bin: ', jammer_placement_bin)
#
#     # Assuming instance.jammer_placement_bin is already an array with the same length as jammer_positions
#     # jammer_dir = torch.tensor(instance.jammer_direction_labels, dtype=torch.long)
#
#     # Ensure jammer_placement_bin is a 2D tensor with the second dimension being 1 to match y for concatenation
#     # jammer_direction_tensor = jammer_dir.view(-1, 1)
#
#     # Concatenate position and placement classification into a single tensor
#     # y_tensor = torch.cat([y, jammer_direction_tensor], dim=1)
#
#     # Create the Data object
#     # data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y_tensor)
#
#     if not params['inference']:
#         jammer_positions = np.array(instance.jammer_position).reshape(-1, params['out_features'])  # Assuming this reshaping is valid based on your data structure
#         y = torch.tensor(jammer_positions, dtype=torch.float)
#     else:
#         y = None
#
#     # Plot
#     # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
#     # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
#
#     # Create the Data object
#     data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y)
#
#     # Convert geometric information to tensors
#     # data.id = instance.id
#     if params['norm'] == 'minmax':
#         data.min_coords = torch.tensor(instance.min_coords, dtype=torch.float)
#         data.max_coords = torch.tensor(instance.max_coords, dtype=torch.float)
#     elif params['norm'] == 'unit_sphere':
#         data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)
#
#     # Store the perc_completion as part of the Data object
#     if not params['inference']:
#         data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)
#         data.pl_exp = torch.tensor(instance.pl_exp, dtype=torch.float)
#         data.sigma = torch.tensor(instance.sigma, dtype=torch.float)
#         data.jtx = torch.tensor(instance.jammer_power, dtype=torch.float)
#         data.num_samples = torch.tensor(instance.num_samples, dtype=torch.float)
#
#     # Apply pos encoding transform
#     if params['model'] == 'GPS':
#         transform = AddRandomWalkPE(walk_length=20, attr_name='pe')
#         data = transform(data)
#
#     return data


def weighted_centroid_localization(drones_pos, drones_noise):
    weights = np.array([10 ** (noise / 10) for noise in drones_noise])
    weighted_sum = np.dot(weights, drones_pos) / np.sum(weights)
    return weighted_sum.tolist()


def create_torch_geo_data(instance: Instance) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """
    # Calculate Weighted Centroid Localization
    centroid_cartesian = weighted_centroid_localization(instance.node_positions_cart, instance.node_noise)
    # print(f"Calculated WCL Centroid in Cartesian Coordinates: {centroid_cartesian}")

    # Wrap the centroid in a list to match the expected input format for cartesian_to_polar
    centroid_polar = cartesian_to_polar([centroid_cartesian])

    # Convert the polar coordinates to cyclical coordinates
    centroid_spherical = angle_to_cyclical(centroid_polar)

    # print(f"Calculated WCL Centroid in Spherical Coordinates: {centroid_spherical}")

    # Append centroid to node positions
    instance.node_positions_cart = np.vstack([instance.node_positions_cart, centroid_cartesian])
    instance.node_positions = np.vstack([instance.node_positions, centroid_spherical])

    weighted_noise = weighted_centroid_localization(instance.node_noise, instance.node_noise)
    instance.node_noise = np.append(instance.node_noise, weighted_noise)  # Append high noise value for the centroid node

    # Preprocess instance data
    if params['norm'] == 'minmax':
        apply_min_max_normalization_instance(instance)
    elif params['norm'] == 'unit_sphere':
        apply_min_max_normalization_instance_noise(instance)
        apply_unit_sphere_normalization(instance)

    print(instance.node_positions[-1])
    quit()
    #
    #
    # weighted_noise = weighted_centroid_localization(instance.node_noise, instance.node_noise)
    # instance.node_noise = np.append(instance.node_noise, 100000)  # Append high noise value for the centroid node

    if 'angle_of_arrival' in params['required_features']:
        # Convert AoA from degrees to radians
        aoa_radians = np.radians(instance.angle_of_arrival)

        # Create node features without adding an extra list around the numpy array
        node_features = np.concatenate([
            instance.node_positions,
            instance.node_noise[:, None],  # Ensure node_noise is reshaped to (n, 1)
            np.sin(aoa_radians[:, None]),
            np.cos(aoa_radians[:, None])
        ], axis=1)
    else:
        node_features = np.concatenate([
            instance.node_positions,
            instance.node_noise[:, None],
            instance.node_positions_cart
        ], axis=1)


    # Convert to 2D tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    # Preparing edges and weights
    if params['aug'] == 'drop_edge':
        # Preparing edges and weights
        positions = instance.node_positions_cart
        if params['num_neighbors'] == 'fc':
            num_neighbors = 10000000
        else:
            num_neighbors = params['num_neighbors']
        if params['edges'] == 'knn':
            num_samples = positions.shape[0]
            k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
            distances, indices = nbrs.kneighbors(positions)
            edge_index, edge_weight = [], []

            # Define the scaling parameter alpha for the Gaussian decay function
            alpha = 1.0  # Adjust this based on your experimentation to find the best fit

            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):  # Skip the first index as it's the node itself
                    edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                    # Apply Gaussian decay to the distance
                    gaussian_weight = np.exp(-alpha * distances[i, j])
                    edge_weight.extend([gaussian_weight, gaussian_weight])

        else:
            raise ValueError("Unsupported edge specification")

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Apply DropEdge
        drop_edge_rate = 0.8
        if drop_edge_rate > 0:
            num_edges = edge_index.size(1)
            num_edges_to_drop = int(drop_edge_rate * num_edges)

            # Randomly select edges to drop
            edges_to_drop = torch.randperm(num_edges)[:num_edges_to_drop]
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[edges_to_drop] = False

            # Apply mask to edge_index and edge_weight
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]

        # Add self-loops for renormalization
        num_nodes = positions.shape[0]
        self_loop_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
        self_loop_weights = torch.ones(num_nodes, dtype=torch.float)  # Weight of 1 for self-loops

        # Combine original edges with self-loops
        edge_index = torch.cat([edge_index, self_loop_indices], dim=1)
        edge_weight = torch.cat([edge_weight, self_loop_weights])

        # Apply renormalization trick
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float)
        deg = deg.scatter_add(0, row, edge_weight)  # Compute degree for each node
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes (degree 0)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # Normalize edge weights

        # Update edge weights with normalized values
        edge_weight = norm
    else:
        positions = instance.node_positions_cart
        if params['num_neighbors'] == 'fc':
            num_neighbors = positions.shape[0] - 1
        else:
            num_neighbors = params['num_neighbors']
        if params['edges'] == 'knn':
            num_samples = positions.shape[0]
            num_edges = params['num_neighbors']  # Expected that this is set outside this function, e.g., params['num_edges'] = 3

            # Initialize NearestNeighbors with more neighbors than needed to calculate full connections later
            nbrs = NearestNeighbors(n_neighbors=num_edges + 1, algorithm='auto').fit(positions)
            distances, indices = nbrs.kneighbors(positions)

            edge_index, edge_weight = [], []
            alpha = 1.0  # Gaussian decay parameter

            # Add edges based on KNN for all nodes except the last
            for i in range(num_samples - 1):  # Exclude the last node in this loop
                for j in range(1, num_edges + 1):  # Skip the first index (self-loop), limit to num_edges
                    gaussian_weight = np.exp(-alpha * distances[i, j])
                    edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                    edge_weight.extend([gaussian_weight, gaussian_weight])

            # Handling the last node separately
            last_node_index = num_samples - 1
            # Add an edge between the last node and every other node
            for i in range(num_samples):
                if i != last_node_index:
                    distance = np.linalg.norm(positions[last_node_index] - positions[i])  # Calculate the Euclidean distance
                    gaussian_weight = np.exp(-alpha * distance)
                    edge_index.extend([[last_node_index, i], [i, last_node_index]])
                    edge_weight.extend([gaussian_weight, gaussian_weight])

            # Add self-loops for all nodes
            for i in range(num_samples):
                edge_index.append([i, i])
                edge_weight.append(1.0)  # Uniform weight for self-loops

        else:
            raise ValueError("Unsupported edge specification")

        # Convert edge_index and edge_weight to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # # Assuming instance.jammer_position is a list or array that can be reshaped
    # jammer_positions = np.array(instance.jammer_position).reshape(-1, 3)
    #
    # # Convert jammer_positions to a tensor
    # y = torch.tensor(jammer_positions, dtype=torch.float)

    # Prepare target variables
    # jammer_positions = np.array(instance.jammer_position).reshape(-1, 2)
    # print('jammer_positions: ', jammer_positions)
    # y = torch.tensor(jammer_positions, dtype=torch.float)

    # Assume instance.jammer_placement_bin is accessible
    # Assuming instance.jammer_placement_bin is a scalar representing class index
    # jammer_placement_bin = torch.tensor([instance.jammer_placement_bin], dtype=torch.long)
    # print('jammer_placement_bin: ', jammer_placement_bin)

    # Assuming instance.jammer_placement_bin is already an array with the same length as jammer_positions
    # jammer_dir = torch.tensor(instance.jammer_direction_labels, dtype=torch.long)

    # Ensure jammer_placement_bin is a 2D tensor with the second dimension being 1 to match y for concatenation
    # jammer_direction_tensor = jammer_dir.view(-1, 1)

    # Concatenate position and placement classification into a single tensor
    # y_tensor = torch.cat([y, jammer_direction_tensor], dim=1)

    # Create the Data object
    # data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y_tensor)

    if not params['inference']:
        jammer_positions = np.array(instance.jammer_position).reshape(-1, params['out_features'])  # Assuming this reshaping is valid based on your data structure
        y = torch.tensor(jammer_positions, dtype=torch.float)
    else:
        y = None

    # Plot
    # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
    # plot_graph(positions=instance.node_positions, node_features=node_features_tensor,  edge_index=edge_index, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)

    # Create the Data object
    data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y, wcl_pred=instance.node_positions[-1])

    # Convert geometric information to tensors
    # data.id = instance.id
    if params['norm'] == 'minmax':
        data.min_coords = torch.tensor(instance.min_coords, dtype=torch.float)
        data.max_coords = torch.tensor(instance.max_coords, dtype=torch.float)
    elif params['norm'] == 'unit_sphere':
        data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)

    # Store the perc_completion as part of the Data object
    if not params['inference']:
        data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)
        data.pl_exp = torch.tensor(instance.pl_exp, dtype=torch.float)
        data.sigma = torch.tensor(instance.sigma, dtype=torch.float)
        data.jtx = torch.tensor(instance.jammer_power, dtype=torch.float)
        data.num_samples = torch.tensor(instance.num_samples, dtype=torch.float)

    # Apply pos encoding transform
    if params['model'] == 'GPS':
        transform = AddRandomWalkPE(walk_length=20, attr_name='pe')
        data = transform(data)

    return data


# def create_torch_geo_data(instance: Instance) -> Data:
#     """
#     Create a PyTorch Geometric Data object from a row of the dataset.
#
#     Args:
#         row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.
#
#     Returns:
#         Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
#     """
#     if len(instance.node_positions) < 3:
#         print(instance.node_positions_cart)
#         print('ERROR')
#         quit()
#     # Preprocess instance data
#     if params['norm'] == 'minmax':
#         apply_min_max_normalization_instance(instance)
#     elif params['norm'] == 'unit_sphere':
#         apply_min_max_normalization_instance_noise(instance)
#         apply_unit_sphere_normalization(instance)
#
#     if 'angle_of_arrival' in params['required_features']:
#         # Convert AoA from degrees to radians
#         aoa_radians = np.radians(instance.angle_of_arrival)
#
#         # Create node features without adding an extra list around the numpy array
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None],  # Ensure node_noise is reshaped to (n, 1)
#             np.sin(aoa_radians[:, None]),
#             np.cos(aoa_radians[:, None])
#         ], axis=1)
#     else:
#         node_features = np.concatenate([
#             instance.node_positions,
#             instance.node_noise[:, None],
#             instance.node_positions_cart
#         ], axis=1)
#
#
#     # Convert to 2D tensor
#     node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
#
#     # Preparing edges and weights
#     if params['aug'] == 'drop_edge':
#         # Preparing edges and weights
#         positions = instance.node_positions_cart
#         if params['num_neighbors'] == 'fc':
#             num_neighbors = 10000000
#         else:
#             num_neighbors = params['num_neighbors']
#         if params['edges'] == 'knn':
#             num_samples = positions.shape[0]
#             k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
#             nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
#             distances, indices = nbrs.kneighbors(positions)
#             edge_index, edge_weight = [], []
#
#             # Define the scaling parameter alpha for the Gaussian decay function
#             alpha = 1.0  # Adjust this based on your experimentation to find the best fit
#
#             for i in range(indices.shape[0]):
#                 for j in range(1, indices.shape[1]):  # Skip the first index as it's the node itself
#                     edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
#                     # Apply Gaussian decay to the distance
#                     gaussian_weight = np.exp(-alpha * distances[i, j])
#                     edge_weight.extend([gaussian_weight, gaussian_weight])
#
#         else:
#             raise ValueError("Unsupported edge specification")
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
#         # Apply DropEdge
#         drop_edge_rate = 0.8
#         if drop_edge_rate > 0:
#             num_edges = edge_index.size(1)
#             num_edges_to_drop = int(drop_edge_rate * num_edges)
#
#             # Randomly select edges to drop
#             edges_to_drop = torch.randperm(num_edges)[:num_edges_to_drop]
#             mask = torch.ones(num_edges, dtype=torch.bool)
#             mask[edges_to_drop] = False
#
#             # Apply mask to edge_index and edge_weight
#             edge_index = edge_index[:, mask]
#             edge_weight = edge_weight[mask]
#
#         # Add self-loops for renormalization
#         num_nodes = positions.shape[0]
#         self_loop_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
#         self_loop_weights = torch.ones(num_nodes, dtype=torch.float)  # Weight of 1 for self-loops
#
#         # Combine original edges with self-loops
#         edge_index = torch.cat([edge_index, self_loop_indices], dim=1)
#         edge_weight = torch.cat([edge_weight, self_loop_weights])
#
#         # Apply renormalization trick
#         row, col = edge_index
#         deg = torch.zeros(num_nodes, dtype=torch.float)
#         deg = deg.scatter_add(0, row, edge_weight)  # Compute degree for each node
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes (degree 0)
#         norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # Normalize edge weights
#
#         # Update edge weights with normalized values
#         edge_weight = norm
#     else:
#         positions = instance.node_positions_cart
#         if params['num_neighbors'] == 'fc':
#             num_neighbors = 10000000
#         else:
#             num_neighbors = params['num_neighbors']
#         if params['edges'] == 'knn':
#             num_samples = positions.shape[0]
#             k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
#             nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
#             distances, indices = nbrs.kneighbors(positions)
#             edge_index, edge_weight = [], []
#
#             # Define the scaling parameter alpha for the Gaussian decay function
#             alpha = 1.0
#
#             # Add self-loops
#             for i in range(indices.shape[0]):
#                 edge_index.extend([[i, i]])
#                 edge_weight.extend([1.0])  # Self-loops can have a weight of 1
#
#                 for j in range(1, indices.shape[1]):
#                     edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
#
#                     # Apply Gaussian decay to the distance
#                     gaussian_weight = np.exp(-alpha * distances[i, j])
#                     edge_weight.extend([gaussian_weight, gaussian_weight])
#
#         else:
#             raise ValueError("Unsupported edge specification")
#
#         # Convert edge_index and edge_weight to tensors
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
#     # # Assuming instance.jammer_position is a list or array that can be reshaped
#     # jammer_positions = np.array(instance.jammer_position).reshape(-1, 3)
#     #
#     # # Convert jammer_positions to a tensor
#     # y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Prepare target variables
#     # jammer_positions = np.array(instance.jammer_position).reshape(-1, 2)
#     # print('jammer_positions: ', jammer_positions)
#     # y = torch.tensor(jammer_positions, dtype=torch.float)
#
#     # Assume instance.jammer_placement_bin is accessible
#     # Assuming instance.jammer_placement_bin is a scalar representing class index
#     # jammer_placement_bin = torch.tensor([instance.jammer_placement_bin], dtype=torch.long)
#     # print('jammer_placement_bin: ', jammer_placement_bin)
#
#     # Assuming instance.jammer_placement_bin is already an array with the same length as jammer_positions
#     # jammer_dir = torch.tensor(instance.jammer_direction_labels, dtype=torch.long)
#
#     # Ensure jammer_placement_bin is a 2D tensor with the second dimension being 1 to match y for concatenation
#     # jammer_direction_tensor = jammer_dir.view(-1, 1)
#
#     # Concatenate position and placement classification into a single tensor
#     # y_tensor = torch.cat([y, jammer_direction_tensor], dim=1)
#
#     # Create the Data object
#     # data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y_tensor)
#
#     if not params['inference']:
#         jammer_positions = np.array(instance.jammer_position).reshape(-1, params['out_features'])  # Assuming this reshaping is valid based on your data structure
#         y = torch.tensor(jammer_positions, dtype=torch.float)
#     else:
#         y = None
#
#     # Plot
#     # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
#     # plot_graph(positions=positions, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)
#
#     # Create the Data object
#     data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y)
#
#     # Convert geometric information to tensors
#     # data.id = instance.id
#     if params['norm'] == 'minmax':
#         data.min_coords = torch.tensor(instance.min_coords, dtype=torch.float)
#         data.max_coords = torch.tensor(instance.max_coords, dtype=torch.float)
#     elif params['norm'] == 'unit_sphere':
#         data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)
#
#     # Store the perc_completion as part of the Data object
#     if not params['inference']:
#         data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)
#         data.pl_exp = torch.tensor(instance.pl_exp, dtype=torch.float)
#         data.sigma = torch.tensor(instance.sigma, dtype=torch.float)
#         data.jtx = torch.tensor(instance.jammer_power, dtype=torch.float)
#         data.num_samples = torch.tensor(instance.num_samples, dtype=torch.float)
#
#     # Apply pos encoding transform
#     if params['model'] == 'GPS':
#         transform = AddRandomWalkPE(walk_length=20, attr_name='pe')
#         data = transform(data)
#
#     return data