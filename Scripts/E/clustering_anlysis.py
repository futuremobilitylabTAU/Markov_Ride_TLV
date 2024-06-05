"""
This Python script conducts k-means clustering analysis on Autonomous Mobility on Demand (AmoD) demand data for Tel Aviv Yaffo in AM,OP,PM.

Author: Gabriel Dadashev
Date: 7-12-2023
"""

# Importing necessary libraries
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
np.random.seed(0)
# Read AmoD demand data for different time periods
requst_am = pd.read_json(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Amod_schedule\requst_am.json")
amod_am = requst_am.copy()

requst_op = pd.read_json(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Amod_schedule\requst_op.json")
amod_op = requst_op.copy()

requst_pm = pd.read_json(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Amod_schedule\requst_pm.json")
amod_pm = requst_pm.copy()

# Extract 'x' and 'y' coordinates from the AmoD data for AM
amod_am['or_x'] = amod_am.apply(lambda x: x['origin']['x'], axis=1)
amod_am['or_y'] = amod_am.apply(lambda x: x['origin']['y'], axis=1)

# Extract 'x' and 'y' coordinates from the AmoD data for OP
amod_op['or_x'] = amod_op.apply(lambda x: x['origin']['x'], axis=1)
amod_op['or_y'] = amod_op.apply(lambda x: x['origin']['y'], axis=1)

# Extract 'x' and 'y' coordinates from the AmoD data for PM
amod_pm['or_x'] = amod_pm.apply(lambda x: x['origin']['x'], axis=1)
amod_pm['or_y'] = amod_pm.apply(lambda x: x['origin']['y'], axis=1)

# Create arrays for k-means clustering for 2017
x_amod_am = amod_am['or_x'].to_numpy()
y_amod_am = amod_am['or_y'].to_numpy()
data_amod_am = np.vstack((x_amod_am, y_amod_am)).T

# Create arrays for k-means clustering for 2040
x_amod_op = amod_op['or_x'].to_numpy()
y_amod_op = amod_op['or_y'].to_numpy()
data_amod_op = np.vstack((x_amod_op, y_amod_op)).T

# Create arrays for k-means clustering for 2017
x_amod_pm = amod_pm['or_x'].to_numpy()
y_amod_pm = amod_pm['or_y'].to_numpy()
data_amod_pm = np.vstack((x_amod_pm, y_amod_pm)).T

# Calculate sum of squared distances for different numbers of clusters (k) for each time period
Sum_of_squared_distances_am = []
K = range(2, 35)
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data_amod_am)
    Sum_of_squared_distances_am.append(kmeans.inertia_)

Sum_of_squared_distances_op = []
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data_amod_op)
    Sum_of_squared_distances_op.append(kmeans.inertia_)

Sum_of_squared_distances_pm = []
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data_amod_pm)
    Sum_of_squared_distances_pm.append(kmeans.inertia_)

# Store sum of squared distances and delta values in dataframes
am = pd.DataFrame(Sum_of_squared_distances_am, columns=['Sum_of_squared'])
op = pd.DataFrame(Sum_of_squared_distances_op, columns=['Sum_of_squared'])
pm = pd.DataFrame(Sum_of_squared_distances_pm, columns=['Sum_of_squared'])

am['delta'] = am['Sum_of_squared'].diff().fillna(0)
op['delta'] = op['Sum_of_squared'].diff().fillna(0)
pm['delta'] = pm['Sum_of_squared'].diff().fillna(0)

# Creating dataframes for sum of squared distances
sum_of_squared_df = pd.DataFrame({
    'Number of Clusters': range(2, 35),
    'Sum of Squared Distances (AM)': Sum_of_squared_distances_am,
    'Sum of Squared Distances (OP)': Sum_of_squared_distances_op,
    'Sum of Squared Distances (PM)': Sum_of_squared_distances_pm
})
sum_of_squared_df = sum_of_squared_df.set_index('Number of Clusters')

# Creating dataframes for delta values
delta_df = pd.DataFrame({
    'Number of Clusters': range(2, 35),
    'Delta (AM)': am['delta'],
    'Delta (OP)': op['delta'],
    'Delta (PM)': pm['delta']
})
delta_df = delta_df.set_index('Number of Clusters')

# Write dataframes to CSV files for further analysis
delta_df.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Clustering\delta.csv')
sum_of_squared_df.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Clustering\sum_of_squared.csv')













