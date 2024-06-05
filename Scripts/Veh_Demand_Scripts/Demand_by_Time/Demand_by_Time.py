
"""
Author: Gabriel Dadashev
Date: 11-4-24

Description:
This script processes mobility data related to demand for different modes of transportation for Aimsun Simulation. 
It performs data cleaning, aggregation, and analysis on activity schedule data and external demand matrices. 
The processed data is then exported to a CSV file for further analysis or visualization.
"""


import pandas as pd 

# Read activity schedule data from CSV into a DataFrame
das = pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Activity_schedule\Das.csv")

# Filter out rows where either the previous stop zone or the stop zone is in [41, 42, 43]
das = das[(das['prev_stop_zone'].isin([41, 42, 43]) == False) & 
          (das['stop_zone'].isin([41, 42, 43]) == False)]

# Remove rows where the previous stop zone is the same as the stop zone
das = das[das['prev_stop_zone'] != das['stop_zone']]

# Keep only rows where the stop mode is either 'Car' or 'AMOD'
das = das[das['stop_mode'].isin(['Car', 'AMOD'])]

# Keep only rows where the previous stop departure time is between 5:45 AM and 9:15 PM
das = das[(das['prev_stop_departure_time'] > 5.75) & (das['prev_stop_departure_time'] < 21.25)]

# Convert the previous stop departure time to integer
das['prev_stop_departure_time'] = das['prev_stop_departure_time'].astype(int)

# Group by stop mode and previous stop departure time and count the number of occurrences
time_series_demand = das.groupby(['stop_mode', 'prev_stop_departure_time'])['prev_stop_zone'].count().reset_index()

# Pivot the data to have stop mode as columns and previous stop departure time as index
pivot_demand = time_series_demand.pivot(index='prev_stop_departure_time', columns='stop_mode', values='prev_stop_zone')

# Calculate external demand for each CSV file and store it in a dictionary
external_demand = {}
for csv in range(6, 21): 
    matrix = pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\External_Val_Matrixs\External_matrix_aimsun_%s.csv" % csv, index_col=0)
    demand = int(matrix.sum().sum())
    external_demand.update({csv: demand})

# Convert the dictionary to a DataFrame and set 'csv' as index
external_demand = pd.DataFrame(list(external_demand.items()), columns=['csv', 'demand']).set_index('csv')

# Rename the column to indicate it's external demand
external_demand = external_demand.rename(columns={'demand': 'Car - External Volume based Cellular Survey'})

# Concatenate external demand DataFrame with pivot_demand DataFrame
pivot_demand = pd.concat([pivot_demand, external_demand], ignore_index=False, axis=1)

# Rename columns for clarity
pivot_demand = pivot_demand.rename(columns={'Car': 'Car - Simulated by Simmobility', 'AMOD': 'AMoD - Simulated by Simmobility'})

# Print the sum of demand for each mode
print(pivot_demand.sum())

# Export pivot_demand DataFrame to a CSV file
pivot_demand.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Demand_by_Time\Demand_by_Time.csv')