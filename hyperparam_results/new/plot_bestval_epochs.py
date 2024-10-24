import json
import matplotlib.pyplot as plt

# Names of the JSON files and corresponding model names
files_and_models = {
    'GAT_optuna_results.json': 'GAT',
    'GATv2_optuna_results.json': 'GATv2',
    'Sage_optuna_results.json': 'Sage',
    'GCN_optuna_results.json': 'GCN',
    'GIN_optuna_results.json': 'GIN'
}

# Initialize the figure for individual plots
fig, axs = plt.subplots(len(files_and_models), 1, figsize=(16, 20), sharex=True)
axs = axs.flatten()  # Flatten in case of a single row of plots

# Prepare a figure for the combined plot
fig_combined, ax_combined = plt.subplots(figsize=(20, 8))

# Loop through each file, extract epoch data, plot, and print trial count
for i, (file, model) in enumerate(files_and_models.items()):
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        best_dropout = data['best_hyperparameters']['dropout_rate']

        # Print out the count of number of trials
        num_trials = len(data['trials'])
        print(f"Model {model} has {num_trials} trials.")

        # Find the trial with the matching dropout rate
        for trial in data['trials']:
            if trial['hyperparameters']['dropout_rate'] == best_dropout:
                epochs = trial['intermediate_values']
                epochs = {int(k): v for k, v in epochs.items()}  # Convert keys to integers
                epochs_sorted = sorted(epochs.items())  # Sort epochs by key (epoch number)
                epochs, val_losses = zip(*epochs_sorted)  # Unzip into separate lists

                # Determine the epoch with the minimum validation loss
                min_loss_index = val_losses.index(min(val_losses))
                min_epoch = epochs[min_loss_index]
                min_val_loss = val_losses[min_loss_index]

                # Plot in individual subplots
                axs[i].plot(epochs, val_losses, marker='o', linestyle='-', linewidth=1, markersize=4)
                axs[i].scatter([min_epoch], [min_val_loss], color='red', s=90)  # Highlight the best epoch in red
                axs[i].set_title(f"{model}")
                axs[i].set_ylabel('Validation Loss')
                axs[i].set_ylim(0, 1)  # Ensure y-axis is limited from 0 to 1
                axs[i].set_xlim(0, 200)  # Set x-axis limit to match across all plots

                # Plot on combined plot
                ax_combined.plot(epochs, val_losses, label=model, marker='o', linestyle='-', markersize=4)
                ax_combined.scatter([min_epoch], [min_val_loss], color='grey', s=100)  # Highlight the best epoch in red

                break  # Stop after finding the first match

axs[-1].set_xlabel('Epoch')  # Set x-label on the last subplot

# Customize the combined plot
ax_combined.set_title('Validation Loss Progression Across Models')
ax_combined.set_xlabel('Epoch')
ax_combined.set_ylabel('Validation Loss')
ax_combined.set_ylim(0, 1)  # Set the y-axis limit to 1 for clarity
ax_combined.set_xlim(0, 200)
ax_combined.legend()

plt.tight_layout()
fig.savefig('individual_val_loss_progression.png', dpi=300)  # Save the figure with individual subplots
fig_combined.savefig('combined_val_loss_progression.png', dpi=300)  # Save the combined plot

plt.show()  # Display plots
plt.close(fig)  # Close the individual figure to free up memory
plt.close(fig_combined)  # Close the combined figure to free up memory
