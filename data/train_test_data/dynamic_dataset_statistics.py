import pandas as pd
import os
import numpy as np
from typing import List, Any


# Define the paths to the CSV files
file_paths = [
    'dynamic/guided_path_data.csv',
    'dynamic/linear_path_data.csv'
]

stat_columns = ['num_samples', 'node_noise', 'jammer_power', 'jammer_gain', 'pl_exp']#, 'sigma']


# Conversion function
def safe_convert_list(row: str, data_type: str) -> List[Any]:
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
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error


# Function to compute the distance between two points
def compute_distance(node_position, jammer_position):
    return np.sqrt((node_position[0] - jammer_position[0]) ** 2 + (node_position[1] - jammer_position[1]) ** 2)


# Function to convert the necessary columns and calculate distances
def convert_and_compute_distances(df):
    df['num_samples'] = df['num_samples'].astype(float)
    df['node_noise'] = df['node_noise'].apply(lambda x: safe_convert_list(x, 'node_noise'))
    df['jammer_power'] = df['jammer_power'].astype(float)
    df['jammer_gain'] = df['jammer_gain'].astype(float)
    df['pl_exp'] = df['pl_exp'].astype(float)
    # df['sigma'] = df['sigma'].astype(float)

    # Convert node_positions and jammer_position, then compute distances
    df['node_positions'] = df['node_positions'].apply(lambda x: safe_convert_list(x, 'node_positions'))
    df['jammer_position'] = df['jammer_position'].apply(lambda x: safe_convert_list(x, 'jammer_position'))

    df['node_distances'] = df.apply(lambda row: [compute_distance(node, row['jammer_position'])
                                                 for node in row['node_positions']], axis=1)

    return df


# Function to compute per-row statistics for node distances, node_rssi, and node_noise
def compute_row_statistics(df):
    # Node distances
    df['node_distances_mean'] = df['node_distances'].apply(np.mean)
    df['node_distances_std'] = df['node_distances'].apply(np.std)
    df['node_distances_var'] = df['node_distances'].apply(np.var)
    df['node_distances_min'] = df['node_distances'].apply(np.min)
    df['node_distances_max'] = df['node_distances'].apply(np.max)

    # Node Noise
    df['node_noise_mean'] = df['node_noise'].apply(np.mean)
    df['node_noise_std'] = df['node_noise'].apply(np.std)
    df['node_noise_min'] = df['node_noise'].apply(np.min)
    df['node_noise_max'] = df['node_noise'].apply(np.max)

    return df


# Function to compute statistics across rows
def compute_statistics(df, group_col, stat_cols):
    results = []

    grouped_df = df.groupby(group_col)

    for name, group in grouped_df:
        result = {'dataset': name}

        for col in stat_cols:
            if col in ['num_samples', 'jammer_power', 'jammer_gain', 'pl_exp']: # 'sigma'
                stats = group[col].agg(['mean', 'std', 'min', 'max'])
                result[f'{col}_mean'] = stats['mean']
                result[f'{col}_std'] = stats['std']
                result[f'{col}_min'] = stats['min']
                result[f'{col}_max'] = stats['max']

        # Aggregate statistics across rows for node distances
        result['node_distances_mean_of_means'] = group['node_distances_mean'].mean()
        result['node_distances_mean_std'] = group['node_distances_mean'].std()
        result['node_distances_var_of_vars'] = group['node_distances_var'].var()
        result['node_distances_min_of_mins'] = group['node_distances_min'].mean()
        result['node_distances_max_of_maxs'] = group['node_distances_max'].mean()

        # Aggregate statistics across rows for node Noise
        result['node_noise_mean_of_means'] = group['node_noise_mean'].mean()
        result['node_noise_mean_std'] = group['node_noise_mean'].std()
        result['node_noise_std_of_stds'] = group['node_noise_std'].mean()
        result['node_noise_min_of_mins'] = group['node_noise_min'].mean()
        result['node_noise_max_of_maxs'] = group['node_noise_max'].mean()

        results.append(result)

    return pd.DataFrame(results)


# Function to group datasets
def group_datasets(df):
    def assign_group(dataset):
        # Replace "jammer_outside_region" with "outside"
        dataset = dataset.replace("jammer_outside_region", "outside")

        if "all_jammed_outside" in dataset:
            return "all_jammed_outside"
        elif "all_jammed" in dataset and "_outside" not in dataset:
            return "all_jammed"
        else:
            return dataset

    df['dataset_group'] = df['dataset'].apply(assign_group)
    return df


# Process each file
for file_path in file_paths:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert columns to appropriate data types and compute distances
    df = convert_and_compute_distances(df)

    # Compute per-row statistics for node distances, node_rssi, and node_noise
    df = compute_row_statistics(df)

    # Group datasets based on the specified conditions
    df = group_datasets(df)

    # Compute statistics grouped by the new 'dataset_group' column
    stats_df = compute_statistics(df, 'dataset_group', stat_columns)

    # Save the result to a new CSV file
    output_file = os.path.join(os.path.dirname(file_path), 'statistics_' + os.path.basename(file_path))
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics saved to {output_file}")

    # Print formatted statistics
    print("Formatted Statistics:")
    for _, row in stats_df.iterrows():
        dataset_name = row['dataset']
        for feature in stat_columns:
            if f'{feature}_mean' in row and not pd.isna(row[f'{feature}_mean']):
                mean_value = row.get(f'{feature}_mean', '--')
                std_value = row.get(f'{feature}_std', '--')
                min_value = row.get(f'{feature}_min', '--')
                max_value = row.get(f'{feature}_max', '--')
                print(f"{dataset_name} & {feature} & {round(mean_value, 1)}\\sd{{{round(std_value, 1)}}} & {round(min_value, 1)}/{round(max_value, 1)}")

        # Print aggregated statistics for node RSSI
        print(f"{dataset_name}")

        # Print aggregated statistics for node Noise
        print(f"& Noise & {round(row['node_noise_mean_of_means'], 1)}\\sd{{{round(row['node_noise_mean_std'], 1)}}} & {round(row['node_noise_min_of_mins'], 1)}/{round(row['node_noise_max_of_maxs'], 1)}")

        # Print aggregated statistics for node distances
        print(f"& Dist to Jam. & {round(row['node_distances_mean_of_means'], 1)}\\sd{{{round(row['node_distances_mean_std'], 1)}}} & {round(row['node_distances_min_of_mins'], 1)}/{round(row['node_distances_max_of_maxs'], 1)}")

        # Aggregated stats for node distance from jammer variance
        print(f"& node_distances_var & {round(row['node_distances_var_of_vars'], 1)}\n")


