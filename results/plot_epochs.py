import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re

# Read the data from the CSV file
folder_path = '/home/dania/gnn_clone_test/gnn-jamming-source-localization/experiments_datasets/datasets/dynamic/controlled/'
file_path = folder_path + 'epoch_metrics.csv'
data = pd.read_csv(file_path)

# Regular expression to capture valid epoch data
pattern = re.compile(r"Epoch: (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)")

# Parsing the epochs data
temp_data = []  # List to store temporary data dictionaries
for index, row in data.iterrows():
    # Find all matches using regex and ignore malformed endings
    matches = pattern.findall(row['epochs'])
    for match in matches:
        epoch_number, train_loss, val_loss = match
        temp_data.append({
            'Trial': row['trial'],
            'Combination': row['combination'],
            'Epoch': int(epoch_number),
            'Val Loss': float(val_loss)
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
# plt.ylim(0, 0.4)
plt.xlim(0)
fig_path = folder_path + 'epoch_metrics.png'
plt.savefig(fig_path, dpi=300)
plt.show()

# Plotting mean RMSE, MAE, and MSE comparison with standard deviations
stats_metrics = data.groupby('combination')[['mae', 'mse', 'rmse']].agg(['mean', 'std']).reset_index()
stats_metrics.columns = ['combination', 'mae_mean', 'mae_std', 'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std']

# Unique combinations for color differentiation
combinations = stats_metrics['combination'].unique()

# Bar plots for each metric with error bars and value annotations
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['mae', 'mse', 'rmse']

for i, metric in enumerate(metrics):
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    for j, comb in enumerate(combinations):
        mean = stats_metrics.loc[stats_metrics['combination'] == comb, mean_col].values[0]
        std = stats_metrics.loc[stats_metrics['combination'] == comb, std_col].values[0]
        bar = axs[i].bar(comb, mean, yerr=std, capsize=4, label=comb if i == 0 else "")

        # Add annotation for each bar
        axs[i].annotate(f'{mean:.2f}',
                        xy=(comb, mean),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    axs[i].set_title(f'Mean {metric.upper()} with Std Dev')
    axs[i].set_ylabel(metric.upper())
    axs[i].set_xticklabels(stats_metrics['combination'], rotation=45, ha="right")

axs[0].legend(title='Combination')

plt.tight_layout()
fig_path = folder_path + 'err_metrics_comparison.png'
plt.savefig(fig_path, dpi=300)
plt.show()
