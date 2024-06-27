import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


def validate_drone_positions(positions):
    """Validate that the positions list contains valid [x, y, z] coordinates and provide detailed error output."""
    if isinstance(positions, str):
        try:
            positions = positions.strip('[]')
            positions = positions.split('], [')
            positions = [list(map(float, pos.split(', '))) for pos in positions]
        except Exception as e:
            print(f"Failed to convert positions from string: {e}")
            return None

    if not isinstance(positions, list) or len(positions) == 0 or any(len(pos) != 3 for pos in positions):
        print(f"Positions after processing: {positions}")
        print("Positions are not properly formatted.")
        return None

    return positions


def calculate_centroid(positions):
    positions_array = np.array(positions)
    centroid = np.mean(positions_array, axis=0)
    return centroid


def calculate_weighted_centroid(rssi_values, positions):
    weighted_sum = np.zeros(3)
    total_weight = 0

    for rssi, coords in zip(rssi_values, positions):
        weight = 10 ** (rssi / 10)
        weighted_coords = np.array(coords) * weight
        weighted_sum += weighted_coords
        total_weight += weight

    if total_weight != 0:
        return weighted_sum / total_weight
    else:
        return np.zeros(3)


def calculate_metrics(centroid, actual_pos):
    mse = np.mean((centroid - actual_pos) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(centroid - actual_pos))
    return mse, rmse, mae


def convert_str_to_list(pos_str):
    try:
        pos_str = pos_str.strip('[]').strip()
        coords = pos_str.split()
        return [float(coord.strip()) for coord in coords]
    except Exception as e:
        print(f"Failed to convert position from string: {e}")
        return None


def convert_str_to_list_rssi(pos_str):
    try:
        pos_str = pos_str.strip('[]').strip()
        coords = pos_str.split(',')
        return [float(coord.strip()) for coord in coords]
    except Exception as e:
        print(f"Failed to convert position from string: {e}")
        return None


def process_test_sets(test_path):
    trials = ['trial1', 'trial2', 'trial3']
    all_metrics = []
    wcl_metrics = []

    for trial in trials:
        file_path = os.path.join(test_path, trial, 'test_df.gzip')
        # try:
        test_data = pd.read_pickle(file_path, compression='gzip')
        # print("test data cols: ", test_data.columns)

        # TODO: check if test_df in trial 1 for cartesian for eg is same as test_df in polar - Done
        mse_values = []
        rmse_values = []
        mae_values = []
        wcl_mse_values = []
        wcl_rmse_values = []
        wcl_mae_values = []

        for idx, row in test_data.iterrows():
            drone_positions = row.get('drone_positions', [])
            # print("row['rssi']: ", row['drone_rssi'])
            rssi_values = row.get('drones_rssi', [])
            rssi_converted = convert_str_to_list_rssi(rssi_values)
            # print("rssi_values: ", rssi_values)

            validated_positions = validate_drone_positions(drone_positions)
            centroid = calculate_centroid(validated_positions)
            # print("centroid: ", centroid)
            actual_pos = row['jammer_position']
            actual_pos = convert_str_to_list(actual_pos)
            # print("rssi_values: ", rssi_values)
            # print("validated_positions: ", validated_positions)
            wcl_centroid = calculate_weighted_centroid(rssi_converted, validated_positions)
            # print("wcl_centroid: ", wcl_centroid)

            mse, rmse, mae = calculate_metrics(centroid, actual_pos)
            mse_values.append(mse)
            rmse_values.append(rmse)
            mae_values.append(mae)

            wcl_mse, wcl_rmse, wcl_mae = calculate_metrics(wcl_centroid, actual_pos)
            wcl_mse_values.append(wcl_mse)
            wcl_rmse_values.append(wcl_rmse)
            wcl_mae_values.append(wcl_mae)

        trial_metrics = {
            'Trial': trial,
            'MSE': np.mean(mse_values),
            'RMSE': np.mean(rmse_values),
            'MAE': np.mean(mae_values)
        }
        wcl_trial_metrics = {
            'Trial': trial,
            'MSE': np.mean(wcl_mse_values),
            'RMSE': np.mean(wcl_rmse_values),
            'MAE': np.mean(wcl_mae_values)
        }
        all_metrics.append(trial_metrics)
        wcl_metrics.append(wcl_trial_metrics)
        # except Exception as e:
        #     print(f"Failed to process {file_path}: {e}")

    return pd.DataFrame(all_metrics), pd.DataFrame(wcl_metrics)


# Example path
polar_path = 'experiments/polar_knn_minmax'
cartesian_path = 'experiments/cartesian_knn_minmax'
centroid_metrics, wcl_metrics = process_test_sets(polar_path)
centroid_metrics_test, wcl_metrics_test = process_test_sets(cartesian_path)
# print("centroid_metrics: ", centroid_metrics)
# print("wcl_metrics: ", wcl_metrics)

# Plotting epoch metrics
data = pd.read_csv('results/epoch_metrics_converted.csv')
plot_data = pd.DataFrame(columns=['Trial', 'Combination', 'Epoch', 'Val Loss'])
temp_data = []
for index, row in data.iterrows():
    epoch_list = ast.literal_eval(row['epochs'])
    for epoch_info in epoch_list:
        parts = epoch_info.split(', ')
        epoch_number = int(parts[0].split(': ')[1])
        val_loss = float(parts[2].split(': ')[1])
        temp_data.append({
            'Trial': row['trial'],
            'Combination': row['combination'],
            'Epoch': epoch_number,
            'Val Loss': val_loss
        })

plot_data = pd.DataFrame(temp_data)
fig, ax = plt.subplots()
stats_data = plot_data.groupby(['Combination', 'Epoch'])['Val Loss'].agg(['mean', 'std']).reset_index()
for label, grp in stats_data.groupby('Combination'):
    ax.plot(grp['Epoch'], grp['mean'], label=label)
    ax.fill_between(grp['Epoch'], grp['mean'] - grp['std'], grp['mean'] + grp['std'], alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.legend(title='Combination')
plt.xlim(0, 200)
plt.savefig('results/epoch_metrics_converted.png', dpi=300)
plt.show()

# Calculate mean and standard deviation for each metric across trials
metrics_agg = centroid_metrics[['MSE', 'RMSE', 'MAE']].agg(['mean', 'std']).T
metrics_agg.reset_index(inplace=True)
metrics_agg.rename(columns={'index': 'Metric'}, inplace=True)

wcl_metrics_agg = wcl_metrics[['MSE', 'RMSE', 'MAE']].agg(['mean', 'std']).T
wcl_metrics_agg.reset_index(inplace=True)
wcl_metrics_agg.rename(columns={'index': 'Metric'}, inplace=True)

# # Test with cartesian
# metrics_agg_test = centroid_metrics_test[['MSE', 'RMSE', 'MAE']].agg(['mean', 'std']).T
# metrics_agg_test.reset_index(inplace=True)
# metrics_agg_test.rename(columns={'index': 'Metric'}, inplace=True)
#
# wcl_metrics_agg_test = wcl_metrics_test[['MSE', 'RMSE', 'MAE']].agg(['mean', 'std']).T
# wcl_metrics_agg_test.reset_index(inplace=True)
# wcl_metrics_agg_test.rename(columns={'index': 'Metric'}, inplace=True)

# print("metrics_agg: \n", metrics_agg)
# print("metrics_agg_test: \n", metrics_agg_test)
#
# print("wcl_metrics_agg: \n", wcl_metrics_agg)
# print("wcl_metrics_agg_test: \n", wcl_metrics_agg_test)
#
# quit()

# Plotting mean RMSE, MAE, and MSE comparison with standard deviations for combinations
stats_metrics = data.groupby('combination')[['mae', 'mse', 'rmse']].agg(['mean', 'std']).reset_index()
stats_metrics.columns = ['combination', 'mae_mean', 'mae_std', 'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std']

combinations = stats_metrics['combination'].unique()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['mae', 'mse', 'rmse']

# Plot each metric
for i, metric in enumerate(metrics):
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'

    for j, comb in enumerate(combinations):
        mean = stats_metrics.loc[stats_metrics['combination'] == comb, mean_col].values[0]
        std = stats_metrics.loc[stats_metrics['combination'] == comb, std_col].values[0]

        # Create the bar
        bar = axs[i].bar(comb, mean, yerr=std, capsize=4, label=comb if i == 0 else "")

        # Add text for the mean value on the bar
        axs[i].text(bar[0].get_x() + bar[0].get_width() / 2, mean, f'{mean:.2f}',
                    ha='center', va='bottom', fontsize=10, color='black')

    # Add the centroid metrics bar
    mean = metrics_agg.loc[metrics_agg['Metric'] == metric.upper(), 'mean'].values[0]
    std = metrics_agg.loc[metrics_agg['Metric'] == metric.upper(), 'std'].values[0]

    # Create the bar
    bar = axs[i].bar('CL', mean, yerr=std, capsize=4, color='orange', label='CL' if i == 0 else "")

    # Add text for the mean value on the bar
    axs[i].text(bar[0].get_x() + bar[0].get_width() / 2, mean, f'{mean:.2f}',
                ha='center', va='bottom', fontsize=10, color='black')

    # Add the weighted centroid metrics bar
    mean = wcl_metrics_agg.loc[wcl_metrics_agg['Metric'] == metric.upper(), 'mean'].values[0]
    std = wcl_metrics_agg.loc[wcl_metrics_agg['Metric'] == metric.upper(), 'std'].values[0]

    # Create the bar
    bar = axs[i].bar('WCL', mean, yerr=std, capsize=4, color='green', label='WCL' if i == 0 else "")

    # Add text for the mean value on the bar
    axs[i].text(bar[0].get_x() + bar[0].get_width() / 2, mean, f'{mean:.2f}',
                ha='center', va='bottom', fontsize=10, color='black')

    # Add title and labels
    axs[i].set_title(f'Mean {metric.upper()}')
    axs[i].set_ylabel(metric.upper())
    axs[i].set_xticklabels(list(combinations) + ['CL', 'WCL'], rotation=45, ha="right")

# Add legend for the first subplot
axs[0].legend(title='Combination')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('results/combined_metrics_comparison.png', dpi=300)
plt.show()