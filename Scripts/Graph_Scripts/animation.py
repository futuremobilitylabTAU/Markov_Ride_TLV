import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg as ffmpeg
import matplotlib

# Define the minx, maxx, miny, maxy values
minx, miny = 172321, 639893
maxx, maxy = 197634, 685928

# Load the new data
file_path = 'C:/Users/dadashev/Dropbox/Optimizing_Mobility_with_Markovian_Model_for_AMoD/data.csv'

new_data = pd.read_csv(file_path)

# Update the Markov_state based on the 'State' column
new_data.loc[new_data['State'].isin(['Travelling to origin', 'At pickup', 'PickupDone']), 'Markov_state'] = 'Travelling to origin'
new_data.loc[new_data['State'].isin(['Travelling to destination', 'At delivery', 'DeliveryDone']), 'Markov_state'] = 'Travelling to destination'
new_data.loc[new_data['State'].isin(['At reposition point', 'Repositioning']), 'Markov_state'] = 'Rebalance'
new_data.loc[new_data['State'].isin(['Idle'])&(new_data['Markov_state'].isin(['Nothing'])), 'Markov_state'] = 'Searching a customer'

# Assign numerical IDs to Veh_ID for simplicity
new_data['Veh_ID'] = new_data['Veh_ID'].astype('category').cat.codes

# Convert Time to datetime, handling different formats
new_data['Time_full'] = pd.to_datetime(new_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Identify the entries that only contain time
time_only_mask = new_data['Time_full'].isna()

# Handle the time-only entries by adding a default date (e.g., 2022-01-01)
default_date = '2022-01-01 '
new_data.loc[time_only_mask, 'Time'] = default_date + new_data.loc[time_only_mask, 'Time']

# Convert the time-only entries (now with the default date) to datetime
new_data['Time'] = pd.to_datetime(new_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Combine the two parsed datetime columns, giving priority to the full datetime entries
new_data['Time'] = new_data['Time_full'].combine_first(new_data['Time'])

# Drop the auxiliary column used for parsing
new_data.drop(columns=['Time_full'], inplace=True)

# Rename columns
new_data.rename(columns={'x': 'longitude', 'y': 'latitude', 'Veh_ID': 'vehicle_id', 'Time': 'time'}, inplace=True)

# Filter out rows with invalid dates
new_data = new_data.dropna(subset=['time'])
new_data_sort = new_data.sort_values(by=['vehicle_id', 'time'])

# Convert DataFrame to GeoDataFrame with EPSG 2039
gdf = gpd.GeoDataFrame(new_data, geometry=gpd.points_from_xy(new_data.longitude, new_data.latitude), crs='EPSG:2039')

# Read the shapefile and transform it to EPSG 2039
shapefile_path = r"C:/Users/dadashev/Dropbox/Optimizing_Mobility_with_Markovian_Model_for_AMoD/Data/Network_Shape/sections.shp"
sections = gpd.read_file(shapefile_path)
sections = sections.to_crs(epsg=2039)

# Create a color mapping for different road types
road_type_colors = {
    179: '#5D737E',
    180: '#4A5362',
    177: '#4b5565',
    175: '#39474E',
    17454573: '#39474E',
    178: '#39474E',
    17453004: '#39474E'
}

# Get unique times and sort them
unique_times = np.sort(gdf['time'].unique())

# Get the start time for each vehicle
vehicle_start_times = gdf.groupby('vehicle_id')['time'].min().reset_index()

# Merge the start times back to the original GeoDataFrame
gdf = gdf.merge(vehicle_start_times, on='vehicle_id', suffixes=('', '_start'))

# Define city names and coordinates
cities = {
    'Rishon LeTsiyon': (177700, 650000),
    'Tel Aviv': (177300, 670000),
    'Kefar Sava': (194000, 677000)
}

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Function to update the plot
def update(frame):
    ax.clear()
    ax.set_facecolor('#928B96')
    
    current_time = unique_times[frame]
    
    # Plot the shapefile layers first with a lower z-order, using the color mapping for road types
    for road_type, color in road_type_colors.items():
        sections[sections['rd_type'] == road_type].plot(ax=ax, color=color, zorder=1)
    
    # Filter the data to get the latest location of each vehicle up to the current time
    filtered_gdf = gdf[(gdf['time_start'] <= current_time) & (gdf['time'] <= current_time)]
    latest_gdf = filtered_gdf.sort_values('time').groupby('vehicle_id').tail(1)
    
    print(f"Time: {current_time}, Number of points: {len(latest_gdf)}")

    
    # Plot points with different colors based on the 'Markov_state' column with a higher z-order
    state_colors = {
        'Nothing': 'red',
        'Idle': 'blue',
        'Rebalance': 'green',
        'Searching a customer': 'yellow',
        'Travelling to origin': 'orange',  
        'Travelling to destination': 'purple'
    }
    for state, color in state_colors.items():
        state_gdf = latest_gdf[latest_gdf['Markov_state'] == state]
        if not state_gdf.empty:
            print(f"Plotting {len(state_gdf)} points for state {state}")
            state_gdf.plot(ax=ax, color=color, markersize=10, label=state, zorder=2)
            for x, y, label in zip(state_gdf.geometry.x, state_gdf.geometry.y, state_gdf['vehicle_id']):
                ax.text(x + 50, y, label, fontsize=10, weight='bold', ha='left', zorder=3)
    
    # Convert numpy.datetime64 to pandas.Timestamp and format the time
    formatted_time = pd.Timestamp(current_time).strftime('%H:%M:%S')
    ax.text(minx + 500, maxy - 1000, formatted_time, fontsize=12, color='white', weight='bold', ha='left')
    
    # Annotate cities
    for city, (x, y) in cities.items():
        ax.text(x, y, city, fontsize=10, color='white', weight='bold', ha='center', va='center', zorder=4, alpha=0.6)
    
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    
    # Create a legend only for the state colors
    handles, labels = ax.get_legend_handles_labels()
    state_handles = [handles[labels.index(state)] for state in state_colors.keys() if state in labels]
    state_labels = [state for state in state_colors.keys() if state in labels]
    ax.legend(state_handles, state_labels, loc='upper right')
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])

# Create the animation with a slower interval (in milliseconds)
ani = animation.FuncAnimation(fig, update, frames=len(unique_times), repeat=False, interval=1)

# Get the path to the ffmpeg executable using imageio_ffmpeg
ffmpeg_path = ffmpeg.get_ffmpeg_exe()

# Set the ffmpeg path in matplotlib rcParams
matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path

# Save the animation as an MP4 file using the path to ffmpeg
writer = animation.FFMpegWriter(fps=30, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
animation_file_path = 'updated_vehicle_locations_2.mp4'
ani.save(animation_file_path, writer=writer)

plt.show()
