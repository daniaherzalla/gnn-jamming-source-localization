import pandas as pd
import os

# Define the folder containing the CSV files
folder_name = 'urban_area'
folder_path = 'log_distance/urban_area'
# folder_path = 'fspl'

# List to hold individual DataFrames
dataframes = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Full path to the CSV file
        file_path = os.path.join(folder_path, filename)

        # Remove the .csv extension from the filename
        base_filename = os.path.splitext(filename)[0]

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if filename starts with "all_jammed"
        if filename.startswith("all_jammed"):
            # Add the node_placement value to the base filename and assign it to the dataset column
            df['dataset'] = df['node_placement'].apply(lambda x: f"{x}_{base_filename}")

            # Append the DataFrame to the list
            dataframes.append(df)
        else:
            # Add a column for the filename
            df['dataset'] = base_filename

            # Append the DataFrame to the list
            dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
combined_df.to_csv(f'{folder_path}/combined_{folder_name}.csv', index=False)

print("CSV files have been concatenated and saved.")



# import pandas as pd
# import os
#
# # Define the folder containing the CSV files
# folder_name = 'urban_area'
# folder_path = 'log_distance/urban_area'
# # folder_path = 'fspl'
#
# # List to hold individual DataFrames
# dataframes = []
#
# # Iterate over all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         # Full path to the CSV file
#         file_path = os.path.join(folder_path, filename)
#
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(file_path)
#
#         # Add a column for the filename
#         df['dataset'] = filename
#
#         # Append the DataFrame to the list
#         dataframes.append(df)
#
# # Concatenate all DataFrames
# combined_df = pd.concat(dataframes, ignore_index=True)
#
# # Save the concatenated DataFrame to a new CSV file
# combined_df.to_csv(f'{folder_path}/combined_{folder_name}.csv', index=False)
#
# print("CSV files have been concatenated and saved.")


