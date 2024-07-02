import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('epoch_metrics_converted.csv')

# Prepare a new DataFrame for plotting epoch metrics
plot_data = pd.DataFrame(columns=['Trial', 'Combination', 'Epoch', 'Val Loss'])

# Parsing the epochs data
temp_data = []  # List to store temporary data dictionaries
for index, row in data.iterrows():
    # Convert the string representation of the list to an actual list
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

# Convert list of dicts to DataFrame
plot_data = pd.DataFrame(temp_data)

# Group by 'Combination' and 'Epoch' and calculate the mean and std of 'Val Loss'
stats_data = plot_data.groupby(['Combination', 'Epoch'])['Val Loss'].agg(['mean', 'std']).reset_index()

# Plotting epoch metrics
fig, ax = plt.subplots()
for label, grp in stats_data.groupby('Combination'):
    ax.plot(grp['Epoch'], grp['mean'], label=label)  # Plot the mean
    ax.fill_between(grp['Epoch'], grp['mean'] - grp['std'], grp['mean'] + grp['std'], alpha=0.3)

ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
ax.legend(title='Combination')
# plt.ylim(0, 0.7)
plt.xlim(0, 200)
plt.savefig('epoch_metrics_converted.png', dpi=300)
plt.show()

# Plotting mean RMSE, MAE, and MSE comparison with standard deviations
stats_metrics = data.groupby('combination')[['mae', 'mse', 'rmse']].agg(['mean', 'std']).reset_index()
stats_metrics.columns = ['combination', 'mae_mean', 'mae_std', 'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std']

# Unique combinations for color differentiation
combinations = stats_metrics['combination'].unique()

# Bar plots for each metric with error bars
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['mae', 'mse', 'rmse']

for i, metric in enumerate(metrics):
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    for j, comb in enumerate(combinations):
        mean = stats_metrics.loc[stats_metrics['combination'] == comb, mean_col].values[0]
        std = stats_metrics.loc[stats_metrics['combination'] == comb, std_col].values[0]
        axs[i].bar(comb, mean, yerr=std, capsize=4, label=comb if i == 0 else "")
    axs[i].set_title(f'Mean {metric.upper()} with Std Dev')
    axs[i].set_ylabel(metric.upper())
    axs[i].set_xticklabels(stats_metrics['combination'], rotation=45, ha="right")

axs[0].legend(title='Combination')

plt.tight_layout()
plt.savefig('err_metrics_comparison_converted.png', dpi=300)
plt.show()
