import pandas as pd
import os

# Define the folder containing the CSV files
# folder_path = 'log_distance'
folder_path = 'fspl'

# List to hold individual DataFrames
dataframes = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Full path to the CSV file
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Add a column for the filename
        df['source_file'] = filename

        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
combined_df.to_csv(f'{folder_path}/combined_{folder_path}.csv', index=False)

print("CSV files have been concatenated and saved as 'combined.csv'.")
