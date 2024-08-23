import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Function to create subplots for each dataset with various plot types
def create_subplots_for_each(df_list, metric, plot_type, ylabel='Value'):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'{metric.capitalize()}')

    file_names = ['FSPL', 'Urban Area', 'Shadowed Urban Area']  # Titles for subplots

    for i, (df, ax) in enumerate(zip(df_list, axes)):
        if plot_type == 'bar':
            sns.barplot(x='dataset', y=metric, data=df, errorbar=None, ax=ax)
            ax.errorbar(df['dataset'], df[metric], yerr=df[f'{metric}_std'], fmt='none', c='black', capsize=5)
        elif plot_type == 'box':
            sns.boxplot(x='dataset', y=metric, data=df, ax=ax)
        elif plot_type == 'line':
            sns.lineplot(x='dataset', y=metric, data=df, marker='o', ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(x='dataset', y=metric, data=df, ax=ax)
        elif plot_type == 'scatter':
            sns.scatterplot(x=metric, y=f'{metric}_std', hue='dataset', data=df, ax=ax)
        elif plot_type == 'density':
            sns.kdeplot(df, x=metric, hue='dataset', ax=ax)
            ax.set_ylim(0, 0.2)  # Adjust y-axis limit for KDE plots

        ax.set_title(file_names[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        # Adjust x-axis label rotation, alignment, and spacing
        ax.tick_params(axis='x', rotation=30, pad=2)
        for label in ax.get_xticklabels():
            # Replace 'jammer_outside_region' with 'outside'
            if 'jammer_outside_region' in label.get_text():
                label.set_text('outside')
            label.set_horizontalalignment('right')

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Load statistics CSV files
file_paths = [
    'fspl/statistics_combined_fspl.csv',
    'log_distance/urban_area/statistics_combined_urban_area.csv',
    'log_distance/shadowed_urban_area/statistics_combined_shadowed_urban_area.csv'
]
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Rename columns to use '_mean' for node_noise
dfs = [df.rename(columns={'node_noise_mean': 'node_noise', 'node_distances_mean': 'node_distances', 'jammer_power_mean': 'jammer_power', 'jammer_gain_mean': 'jammer_gain', 'pl_exp_mean': 'pl_exp'}) for df in dfs]


# Create subplots for various metrics
metrics = ['node_noise', 'node_distances', 'jammer_power', 'jammer_gain', 'pl_exp', 'sigma']

# Create plots for each metric
for metric in metrics:
    create_subplots_for_each(dfs, metric, plot_type='bar')
    create_subplots_for_each(dfs, metric, plot_type='line')
    create_subplots_for_each(dfs, metric, plot_type='scatter')
    create_subplots_for_each(dfs, metric, plot_type='density')

