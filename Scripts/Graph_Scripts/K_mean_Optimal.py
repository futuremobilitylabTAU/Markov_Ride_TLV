import pandas as pd 
import matplotlib.pyplot as plt

# Load data
delta = pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Clustering\delta.csv", index_col=0)
sum_of_squared = pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Clustering\sum_of_squared.csv", index_col=0)
sum_of_squared = sum_of_squared.rename({'Sum of Squared Distances (AM)': 'AM', 'Sum of Squared Distances (OP)': 'OP', 'Sum of Squared Distances (PM)': 'PM'}, axis=1)

# Plot
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Plotting the three lines on the same graph
sum_of_squared['AM'].plot(ax=axes, color='b', marker='x', linestyle='--', label='AM', linewidth=3)
sum_of_squared['OP'].plot(ax=axes, color='r', marker='x', linestyle='--', label='OP', linewidth=3)
sum_of_squared['PM'].plot(ax=axes, color='g', marker='x', linestyle='--', label='PM', linewidth=3)

# Mark specific points on each line
axes.scatter(11, sum_of_squared['AM'].iloc[11]+50000000000, color='b', s=500, label='')
axes.scatter(13, sum_of_squared['OP'].iloc[13]+80000000000, color='r', s=500, label='')
axes.scatter(20, sum_of_squared['PM'].iloc[20]+70000000000, color='g', s=500, label='')

# Customize plot labels and appearance
axes.set_xlabel('Number of Clusters', fontsize=30, labelpad=10)
axes.set_ylabel('Sum of Squared Distances', fontsize=30, labelpad=10)
axes.grid(linewidth=1.5)
axes.legend(prop={"size": 40})
axes.tick_params(axis='both', which='major', labelsize=20)
axes.set_xticks(sum_of_squared.index)
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

# Adjust x-axis limits to center the scatter points
axes.set_xlim([min(sum_of_squared.index), max(sum_of_squared.index)])

# Display the plot
plt.tight_layout()
plt.show()
