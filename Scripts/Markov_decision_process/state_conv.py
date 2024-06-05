"""
This script reads CSV files from a specified folder, extracts numerical data from their filenames,
and combines them into a single DataFrame. The CSV files are expected to have a column named "Value".
The numerical part of the filename is used as a column header in the combined DataFrame.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Define the pattern for file names
file_prefix = "optimal_state_value_"
file_suffix = ".csv"

# Initialize a list to store DataFrames
dfs = []

# Loop through the tables in the folder
folder_path = r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_6\Outputs"

for filename in os.listdir(folder_path):
    if filename.startswith(file_prefix) and filename.endswith(file_suffix):
        # Extract the number part from the file name and convert it to an integer
        file_number = filename[len(file_prefix):-len(file_suffix)]
        try:
            file_number = int(file_number)
        except ValueError:
            continue  # Skip non-numeric file names
        
        # Read each table
        table_df = pd.read_csv(os.path.join(folder_path, filename))
        
        # Rename the "Value" column with the file number
        table_df.rename(columns={"Value": file_number}, inplace=True)
        
        # Append DataFrame to the list
        dfs.append(table_df)

# Concatenate all DataFrames horizontally based on their indices
combined_df = pd.concat(dfs, axis=1)
# Drop duplicate columns
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Ensure 'State' column is the first column
combined_df = combined_df.reindex(['State'] + sorted([col for col in combined_df.columns if col != 'State'], key=lambda x: int(x)), axis=1)


# Set the 'State' column as the index for easier plotting
combined_df.set_index('State', inplace=True)

# Get unique first elements of the tuples in the 'State' column
unique_first_elements = combined_df.index.str.split(',').str[0].unique()

# Generate a color map with unique colors for each unique first element
color_map = plt.cm.get_cmap('tab10', len(unique_first_elements))

# Plot each series with colors based on the first element of the tuple
for i, (state, series) in enumerate(combined_df.iterrows()):
    first_element = state.split(',')[0][1:]  # Extract the first element of the tuple
    color = color_map(i % len(unique_first_elements))  # Get color from the colormap
    series.plot(marker='o', figsize=(10, 6), label=state, color=color)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Optimal State Values')

# Show the legend
plt.grid(True)
plt.tight_layout()
plt.show()