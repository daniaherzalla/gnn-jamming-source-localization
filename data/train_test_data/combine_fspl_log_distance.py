import pandas as pd
import os

# Define the paths to the CSV files
fspl_file_path = os.path.join('fspl', 'combined_fspl.csv')
urban_area_file_path = os.path.join('log_distance/urban_area', 'combined_urban_area.csv')
shadowed_urban_area_file_path = os.path.join('log_distance/shadowed_urban_area', 'combined_shadowed_urban_area.csv')

# Read the CSV files into DataFrames
df_fspl = pd.read_csv(fspl_file_path)
df_urban_area = pd.read_csv(urban_area_file_path)
df_shadowed_urban_area = pd.read_csv(shadowed_urban_area_file_path)

# Concatenate the DataFrames
combined_df = pd.concat([df_fspl, df_urban_area, df_shadowed_urban_area], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
combined_df.to_csv('combined_fspl_log_distance.csv', index=False)

print("CSV files have been concatenated and saved as 'combined_fspl_log_distance.csv'.")
