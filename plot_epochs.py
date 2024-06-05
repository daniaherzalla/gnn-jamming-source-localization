import ast
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('results/epoch_metrics.csv')
df = pd.DataFrame(data)

# Prepare a new DataFrame for plotting
plot_data = pd.DataFrame(columns=['Trial', 'Combination', 'Epoch', 'Val Loss'])

# Parsing the epochs data
temp_data = []  # List to store temporary data dictionaries
for index, row in df.iterrows():
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

# Plotting
fig, ax = plt.subplots()
for label, grp in stats_data.groupby('Combination'):
    ax.plot(grp['Epoch'], grp['mean'], label=label)  # Plot the mean
    ax.fill_between(grp['Epoch'], grp['mean'] - grp['std'], grp['mean'] + grp['std'], alpha=0.3)

# ax.set_title('Average Validation Loss with Standard Deviation for Each Combination')
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
ax.legend(title='Combination')
plt.ylim(0, 0.1)
plt.xlim(0, 200)
plt.savefig('results/epoch_metrics.png', dpi=300)
plt.show()
