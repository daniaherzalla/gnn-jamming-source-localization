import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'Times New Roman'

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Function to fix the list format using regex
def fix_list_format(s):
    try:
        if 'e+' in s or 'e-' in s or 'E+' in s or 'E-' in s:
            return [float(num) for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
        else:
            return [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
    except Exception as e:
        print(f"Error processing row: {s}, error: {e}")
        return []

def clean_list(lst):
    return [x for x in lst if x != '' and x is not None and not np.isnan(x)]

def calculate_overall_rmse(predictions, actuals):
    pred_flat = np.concatenate([clean_list(pred) for pred in predictions])
    actual_flat = np.concatenate([clean_list(act) for act in actuals])
    return np.sqrt(np.mean((pred_flat - actual_flat) ** 2))

# Function to calculate RMSE for each unique percentage completion value
def calculate_rmse_continuous(df):
    percentage_completion_values = sorted(df['Percentage Completion'].unique())
    rmse_values = []
    for perc in percentage_completion_values:
        subset = df[df['Percentage Completion'] == perc]
        rmse = calculate_overall_rmse(subset['Prediction'].tolist(), subset['Actual'].tolist())
        rmse_values.append(round(rmse, 1))  # Round RMSE to one decimal place
    return percentage_completion_values, rmse_values

# Function to calculate a moving average for smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Noise
files = [
    ('/home/dania/gnn-jamming-source-localization/experiments_datasets/datasets/controlled_path/GCN/noise/predictions_400_hybrid_GCN.csv', 'GCN'),
    ('/home/dania/gnn-jamming-source-localization/experiments_datasets/datasets/controlled_path/MLP/noise/predictions_400_hybrid_MLP.csv', 'MLP'),
    ('/home/dania/gnn-jamming-source-localization/experiments_datasets/downsampling/dynamic/controlled_path/hybrid/60perc/predictions_400_hybrid.csv', 'Sage'),
    ('/home/dania/gnn-jamming-source-localization/data/predictions_CL.csv', 'CL'),
    ('/home/dania/gnn-jamming-source-localization/data/predictions_WCL.csv', 'WCL'),
    ('/home/dania/gnn-jamming-source-localization/data/predictions_LSQ.csv', 'LSQ'),
    ('/home/dania/gnn-jamming-source-localization/data/predictions_PL.csv', 'PL'),
    ('/home/dania/gnn-jamming-source-localization/data/predictions_MJ.csv', 'MJ')
]

folder_path = '/home/dania/gnn-jamming-source-localization/experiments_datasets/'

# Use seaborn muted palette
palette = sns.color_palette('colorblind', n_colors=len(files))

plt.figure(figsize=(10, 6))

# Loop over files and assign a color for each model
for i, (file_name, label) in enumerate(files):
    predictions_path = file_name
    predictions_df = pd.read_csv(predictions_path)
    predictions_df['Prediction'] = predictions_df['Prediction'].apply(fix_list_format)
    predictions_df['Actual'] = predictions_df['Actual'].apply(fix_list_format)
    predictions_df['Percentage Completion'] = predictions_df['Percentage Completion'].astype(float)

    # Ensure predictions and actuals are the same length
    predictions_df = predictions_df[predictions_df['Prediction'].apply(len) == predictions_df['Actual'].apply(len)]

    x_perc_completion, y_rmse = calculate_rmse_continuous(predictions_df)

    # Apply moving average smoothing
    window_size = 10
    y_rmse_smoothed = moving_average(y_rmse, window_size)
    x_perc_completion_smoothed = x_perc_completion[len(x_perc_completion) - len(y_rmse_smoothed):]

    plt.plot(
        x_perc_completion_smoothed,
        y_rmse_smoothed,
        linestyle='-',
        linewidth=1,
        color=palette[i],  # Use seaborn muted palette
        label=label
    )

plt.xlabel('Percentage Completion', fontsize=16, fontname='Times New Roman')
plt.ylabel('RMSE (m)', fontsize=16, fontname='Times New Roman')
plt.xlim(0, 1)
plt.ylim(-50, 2000)
plt.grid(True, alpha=0.5, linewidth=0.5)
# Move the legend outside the plot
plt.legend(
    fontsize=16,
    prop={'family': 'Times New Roman'},
    loc='center left',  # Position legend to the left
    bbox_to_anchor=(1, 0.5)  # Move it outside the plot, centered vertically
)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(folder_path + 'controlledpath_models.pdf', bbox_inches='tight', dpi=300)
plt.show()
