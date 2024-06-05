
"""
Author: Gabriel Dadashev
Date: 11-4-24

Description: This script visualizes trip demand data over time using a line plot. 
             It reads trip demand data from a CSV file, plots multiple series of 
             data representing different modes of transportation, and customizes 
             the plot appearance.

"""

import pandas as pd 
import matplotlib.pyplot as plt

# Read trip demand data from CSV file
data = pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Demand_by_Time\Demand_by_Time.csv',index_col=0)
# Create a figure and axes object for plotting
fig, axes = plt.subplots(nrows=1, ncols=1)
# Plot data for different transportation modes
data['Car - Simulated by Simmobility'].plot(ax=axes, color='teal', linewidth=5, label=r'Car - Simulated by $\bf{Simmobility}$')
data['Car - External Volume based Cellular Survey'].plot(ax=axes, color='orchid', linewidth=5, label=r'Car - External Volume based $\bf{Cellular Survey}$')
data['AMoD - Simulated by Simmobility'].plot(ax=axes, color='gold', linewidth=10, label=r'AMoD - Simulated by $\bf{Simmobility}$')

# Customize plot labels and appearance
axes.set_xlabel('Time' , fontsize=40,labelpad=20)
axes.set_ylabel('Trips', fontsize=40,labelpad=20)
axes.grid(linewidth=1.5)
axes.legend( prop={"size":30})
axes.tick_params(axis='x', rotation=0,labelsize=25)
axes.tick_params(axis='y', rotation=0,labelsize=25)
axes.set_xticks(data.index)
# Display the plot

fig.show()


