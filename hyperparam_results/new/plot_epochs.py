import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON data
with open('GAT_optuna_results.json', 'r') as file:
    data = json.load(file)

# Prepare a DataFrame for plotting epoch metrics
plot_data = pd.DataFrame(columns=['Trial', 'Combination', 'Epoch', 'Val Loss'])

# Parsing the epochs data
temp_data = []  # List to store temporary data dictionaries
trial_counter = 0
for trial in data['trials']:
    trial_counter += 1
    combination = f"{trial['hyperparameters']['dropout_rate']}_{trial['hyperparameters']['num_heads']}"
    for epoch, val_loss in trial['intermediate_values'].items():
        temp_data.append({
            'Trial': trial_counter,
            'Combination': combination,
            'Epoch': int(epoch),
            'Val Loss': val_loss
        })

# Convert list of dicts to DataFrame
plot_data = pd.DataFrame(temp_data)

# Group by 'Combination' and 'Epoch' and calculate the mean and std of 'Val Loss'
stats_data = plot_data.groupby(['Combination', 'Epoch'])['Val Loss'].agg(['mean', 'std']).reset_index()

# Plotting epoch metrics
fig, ax = plt.subplots(figsize=(15, 7))
for label, grp in stats_data.groupby('Combination'):
    ax.plot(grp['Epoch'], grp['mean'], label=label)  # Plot the mean
    ax.fill_between(grp['Epoch'], grp['mean'] - grp['std'], grp['mean'] + grp['std'], alpha=0.3)

ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
# ax.legend(title='Combination')
plt.xlim(0, max(plot_data['Epoch']))
plt.savefig('epoch_metrics_from_json.png', dpi=300)
plt.show()
