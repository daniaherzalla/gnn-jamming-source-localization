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

# Dictionary to hold the best losses for each model
best_losses = {}

# Loop through each file and extract the best loss
for file, model in files_and_models.items():
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        best_losses[model] = data['best_loss']

# Plotting the best losses
models = list(best_losses.keys())
losses = [best_losses[model] for model in models]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(models, losses)  # Add color for better visual appeal
ax.set_xlabel('Model')
ax.set_ylabel('Best Validation Loss')
ax.set_title('Comparison of Best Validation Loss by Model')
plt.xticks(rotation=45)  # Rotate for better label visibility

# Annotate each bar with the value of the loss
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005,  # Adjust the position slightly above the bar
            round(yval, 4),  # The value to be displayed
            ha='center', va='bottom')  # Center the text horizontally and align it from the bottom

plt.tight_layout()
plt.savefig('best_val_loss_comparison.png', dpi=300)
plt.show()
