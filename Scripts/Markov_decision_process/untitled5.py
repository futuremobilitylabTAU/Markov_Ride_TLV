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

# Visualize the initial points
initial_values = combined_df.iloc[:, 0]  # Use the first iteration to determine the clusters


# Manually define clusters based on visual inspection of initial values
# Adjust these ranges based on visual inspection of the initial points
cluster_assignment = {}
for state in combined_df.index:
    value = initial_values.loc[state]  # Get the initial value
    if -2<=value < 20:
        cluster_assignment[state] = 1  # Bottom area (Green)
    elif 20 <= value < 150:
        cluster_assignment[state] = 2  # Middle area (Red)
    else:
        cluster_assignment[state] = 3  # Top area (Blue)

# Define the colors for the three clusters
area_colors = {1: 'tab:green', 2: 'tab:red', 3: 'tab:blue'}

# Create a new figure with a larger size for the final plot
plt.figure(figsize=(14, 8))

# Plot each series with colors based on the cluster assignment
for state, series in combined_df.iterrows():
    cluster = cluster_assignment[state]  # Get the cluster
    color = area_colors[cluster]  # Get the color for the cluster
    series.plot(marker='o', linestyle='-', markersize=4, linewidth=1, label=state, color=color)

# Add labels and title
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('Optimal State Values', fontsize=16)

# Show grid
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
