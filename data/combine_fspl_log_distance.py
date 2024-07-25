
import pandas as pd
import os

# Define the paths to the CSV files
fspl_file_path = os.path.join('fspl', 'combined_fspl.csv')
log_distance_file_path = os.path.join('log_distance', 'combined_log_distance.csv')

# Read the CSV files into DataFrames
df_fspl = pd.read_csv(fspl_file_path)
df_log_distance = pd.read_csv(log_distance_file_path)

# Concatenate the DataFrames
combined_df = pd.concat([df_fspl, df_log_distance], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
combined_df.to_csv('combined_fspl_log_distance.csv', index=False)

print("CSV files have been concatenated and saved as 'combined_fspl_log_distance.csv'.")
