import json
import matplotlib.pyplot as plt

# Load the JSON data from the first file
with open('trial_results_GAT_TRIAL2.json', 'r') as file1:
    data1 = json.load(file1)

# Load the JSON data from the second file
with open('GAT_optuna_results.json', 'r') as file2:
    data2 = json.load(file2)

# Extract the loss values from the first 50 trials for both files
loss_values1 = [trial['result']['loss'] for trial in data1['trials'][:200]]
loss_values2 = [trial['value'] for trial in data2['trials'][:200]]

# Find the index of the minimum loss values
min_loss_idx1 = loss_values1.index(min(loss_values1))
min_loss_idx2 = loss_values2.index(min(loss_values2))

# Plot the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values1, marker='o', linestyle='-', color='b', linewidth=1, markersize=4, label='GAT_hyperopt_results')
plt.plot(loss_values2, marker='x', linestyle='-', color='k', linewidth=1, markersize=4, label='GAT_optuna_results')

# Highlight the point with the lowest loss value
plt.plot(min_loss_idx1, loss_values1[min_loss_idx1], marker='o', color='r', markersize=7)
plt.plot(min_loss_idx2, loss_values2[min_loss_idx2], marker='x', color='r', markersize=7)

plt.annotate(f'{loss_values1[min_loss_idx1]:.4f}', (min_loss_idx1, loss_values1[min_loss_idx1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate(f'{loss_values2[min_loss_idx2]:.4f}', (min_loss_idx2, loss_values2[min_loss_idx2]), textcoords="offset points", xytext=(0,10), ha='center')


plt.title('Loss')
plt.xlabel('Trial Number')
plt.ylabel('Val Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_comparison_optuna_hyperopt.png', dpi=300)
plt.show()
