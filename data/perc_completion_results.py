# import pandas as pd
# import numpy as np
# import regex as re
#
# # Function to fix the list format using regex
# def fix_list_format(s):
#     return [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
#
# # Function to calculate RMSE for the 'Prediction' and 'Actual' columns
# def calculate_rmse(pred, actual):
#     return np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(pred, actual)]))
#
# folder_path = '/home/dania/gnn_clone_test/gnn-jamming-source-localization/experiments_datasets/datasets/dynamic/controlled/'
# predictions_seed1_path = folder_path + 'predictions_200epoch_offic.csv'
# predictions_df = pd.read_csv(predictions_seed1_path)
#
# # Fix the list format for Prediction and Actual columns
# predictions_df['Prediction'] = predictions_df['Prediction'].apply(fix_list_format)
# predictions_df['Actual'] = predictions_df['Actual'].apply(fix_list_format)
# predictions_df['Percentage Completion'] = predictions_df['Percentage Completion'].astype(float)
#
# # Calculate RMSE for each row
# predictions_df['RMSE'] = predictions_df.apply(lambda row: calculate_rmse(row['Prediction'], row['Actual']), axis=1)
#
# print("predictions_df['RMSE'] mean: ", predictions_df['RMSE'].mean())
#
# # Create categories based on 'Percentage Completion'
# conditions = [
#     (predictions_df['Percentage Completion'] <= 0.25),
#     (predictions_df['Percentage Completion'] > 0.25) & (predictions_df['Percentage Completion'] <= 0.5),
#     (predictions_df['Percentage Completion'] > 0.5) & (predictions_df['Percentage Completion'] <= 0.75),
#     (predictions_df['Percentage Completion'] > 0.75) & (predictions_df['Percentage Completion'] <= 1)
# ]
# categories = ['<= 0.25', '> 0.25 and <= 0.5', '> 0.5 and <= 0.75', '> 0.75 and <= 1']
#
# # Apply categories to the DataFrame
# predictions_df['Completion Category'] = np.select(conditions, categories, default=np.nan)
#
# # Calculate average RMSE for each completion category
# average_rmse_df = predictions_df.groupby('Completion Category')['RMSE'].mean().reset_index()
#
# # Add a row for the overall average RMSE using pd.concat
# overall_rmse = pd.DataFrame({'Completion Category': ['All'], 'RMSE': [predictions_df['RMSE'].mean()]})
# average_rmse_df = pd.concat([average_rmse_df, overall_rmse], ignore_index=True)
#
# print(average_rmse_df)


import pandas as pd
import numpy as np
import regex as re

# Function to fix the list format using regex
def fix_list_format(s):
    try:
        if 'e+' in s or 'e-' in s or 'E+' in s or 'E-' in s:
            result = [float(num) for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
        else:
            result = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
        return result
    except Exception as e:
        print(f"Error processing row: {s}, error: {e}")
        return []

def clean_list(lst):
    return [x for x in lst if x != '' and x is not None and not np.isnan(x)]

def calculate_overall_rmse(predictions, actuals):
    pred_flat = np.concatenate([clean_list(pred) for pred in predictions])
    actual_flat = np.concatenate([clean_list(act) for act in actuals])
    return np.sqrt(np.mean((pred_flat - actual_flat) ** 2))

# folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/downsampling/dynamic/interpolat/'  # downsampling
# folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/engineered_feats/dynamic/moving_avg_noise/'  # engineered features
# folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/downsampling/dynamic/controlled_path/hybrid/60perc/'  # hybrid downsampling (time wind avg noise filtering) (create
# folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/engineered_feats/dynamic/moving_avg_aoa/'  # hybrid downsampling (time wind avg noise filtering) (create
# folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/datasets/controlled_path/GCN/'  # hybrid downsampling (time wind avg noise filtering) (create
folder_path = '/home/dania/gnn-jamming-source-localization/data/'  # hybrid downsampling (time wind avg noise filtering) (create
predictions_path = folder_path + 'predictions_WCL.csv'
predictions_df = pd.read_csv(predictions_path)

predictions_df['Prediction'] = predictions_df['Prediction'].apply(fix_list_format)
predictions_df['Actual'] = predictions_df['Actual'].apply(fix_list_format)
predictions_df['Percentage Completion'] = predictions_df['Percentage Completion'].astype(float)

predictions_df = predictions_df[predictions_df['Prediction'].apply(len) == predictions_df['Actual'].apply(len)]

# conditions = [
#     (predictions_df['Percentage Completion'] <= 0.25),
#     (predictions_df['Percentage Completion'] > 0.25) & (predictions_df['Percentage Completion'] <= 0.5),
#     (predictions_df['Percentage Completion'] > 0.5) & (predictions_df['Percentage Completion'] <= 0.75),
#     (predictions_df['Percentage Completion'] > 0.75) & (predictions_df['Percentage Completion'] <= 1)
# ]
#
# categories = ['<= 0.25', '> 0.25 and <= 0.5', '> 0.5 and <= 0.75', '> 0.75 and <= 1']
# predictions_df['Completion Category'] = np.select(conditions, categories, default=np.nan)

# Define conditions for cumulative ranges
conditions = [
    (predictions_df['Percentage Completion'] <= 0.25),
    (predictions_df['Percentage Completion'] <= 0.5),
    (predictions_df['Percentage Completion'] <= 0.75),
    (predictions_df['Percentage Completion'] <= 1.0)
]
categories = ['Up to 0.25', 'Up to 0.5', 'Up to 0.75', 'Up to 1.0']
predictions_df['Completion Category'] = np.select(conditions, categories, default=np.nan)

# Calculate RMSE per seed, then calculate STD across these RMSEs
def calculate_rmse_per_seed(group):
    return calculate_overall_rmse(group['Prediction'].tolist(), group['Actual'].tolist())

# Calculate average RMSE and STD across all seeds for each completion category
def calculate_metrics(group):
    rmse_per_seed = group.groupby('Seed').apply(calculate_rmse_per_seed)
    overall_rmse = rmse_per_seed.mean()
    std = rmse_per_seed.std()
    return pd.Series({'RMSE': overall_rmse, 'STD': std})

results = predictions_df.groupby('Completion Category').apply(calculate_metrics).reset_index()

# Calculate overall metrics across all data without category distinction
overall_rmse_per_seed = predictions_df.groupby('Seed').apply(lambda g: calculate_overall_rmse(g['Prediction'].tolist(), g['Actual'].tolist()))
overall_rmse = overall_rmse_per_seed.mean()
overall_std = overall_rmse_per_seed.std()

# Add a row for the overall average RMSE and STD
overall_results = pd.DataFrame({'Completion Category': ['All'], 'RMSE': [overall_rmse], 'STD': [overall_std]})
results = pd.concat([results, overall_results], ignore_index=True)

# # Print results with RMSE and STD
# print(results)

# Format and print results with RMSE and STD in the specified format
for index, row in results.iterrows():
    formatted_output = f"{row['Completion Category']}: {row['RMSE']:.1f}\\sd{{{row['STD']:.1f}}}"
    print(formatted_output)

import matplotlib.pyplot as plt
import numpy as np

# Function to calculate RMSE for each unique percentage completion value
def calculate_rmse_continuous(df):
    percentage_completion_values = sorted(df['Percentage Completion'].unique())
    rmse_values = []
    for perc in percentage_completion_values:
        subset = df[df['Percentage Completion'] == perc]
        rmse = calculate_overall_rmse(subset['Prediction'].tolist(), subset['Actual'].tolist())
        rmse_values.append(round(rmse, 1))
    return percentage_completion_values, rmse_values

# Function to calculate a moving average for smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Get continuous RMSE values
x_perc_completion, y_rmse = calculate_rmse_continuous(predictions_df)

# Smooth RMSE values using a moving average
window_size = 5  # Adjust the window size for smoothing as needed
y_rmse_smoothed = moving_average(y_rmse, window_size)

# Adjust x-axis values to match the smoothed data
x_perc_completion_smoothed = x_perc_completion[window_size - 1:]

# Plot original and smoothed RMSE
plt.figure(figsize=(10, 6))
# plt.plot(x_perc_completion, y_rmse, marker='o', markersize=1, linestyle='--', linewidth=1, alpha=0.5)
plt.plot(x_perc_completion, y_rmse, marker='o', markersize=1, linestyle='-', linewidth=1, alpha=0.7)
# plt.plot(x_perc_completion_smoothed, y_rmse_smoothed, marker='o', markersize=1, linestyle='-', linewidth=1)
plt.xlabel('Percentage Completion')
plt.ylabel('RMSE')
plt.xlim(0, 1)
# plt.title('Percentage Completion vs RMSE (Smoothed with Moving Average)')
plt.grid(True, alpha=0.5, linewidth=0.5)  # Adjust alpha for transparency and linewidth for thickness
plt.legend()
plt.show()

# import matplotlib.pyplot as plt
#
# # Visualize the average prediction vs. actual values by completion category
# grouped = predictions_df.groupby('Completion Category').apply(lambda g: pd.DataFrame({
#     'Average Prediction': [np.mean(np.concatenate(g['Prediction'].tolist()))],
#     'Average Actual': [np.mean(np.concatenate(g['Actual'].tolist()))]
# })).reset_index()
#
# # Plotting
# fig, ax = plt.subplots()
# grouped.plot(x='Completion Category', y=['Average Prediction', 'Average Actual'], kind='bar', ax=ax)
# plt.title('Average Prediction and Actual Values by Completion Category')
# plt.ylabel('Value')
# plt.show()
#
#
# # Calculate errors and variance within each category
# predictions_df['Errors'] = predictions_df.apply(lambda row: np.array(row['Prediction']) - np.array(row['Actual']), axis=1)
# predictions_df['Error Variance'] = predictions_df['Errors'].apply(lambda errors: np.var(errors))
#
# # Average error variance by category
# error_variance_by_category = predictions_df.groupby('Completion Category')['Error Variance'].mean()
# print(error_variance_by_category)
#
#
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# axes = axes.flatten()
#
# for ax, (category, group) in zip(axes, predictions_df.groupby('Completion Category')):
#     errors = np.concatenate(group['Errors'].tolist())
#     ax.hist(errors, bins=30, alpha=0.75)
#     ax.set_title(f'Error Distribution: {category}')
#     ax.set_xlabel('Error')
#     ax.set_ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()
