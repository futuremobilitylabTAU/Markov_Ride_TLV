import geopandas as gpd
from shapely.geometry import Point

# Read polygon layer
polygons = gpd.read_file(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_6\Shape_file\clusters.shp')

# Read point layer
points = gpd.read_file(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_6\Shape_file\points_origins.shp')

# Perform spatial join
spatial_join = gpd.sjoin(points, polygons, how="left", op='within')
spatial_join['Name_right']=spatial_join['Name_right'].astype(int)
# Group points by polygon identifier
grouped = spatial_join.groupby('Name_right')

# Calculate average point for each group
average_points = []
for group_name, group_data in grouped:
    # Calculate average coordinates
    avg_x = group_data.geometry.x.mean()
    avg_y = group_data.geometry.y.mean()
    # Create average point
    avg_point = Point(avg_x, avg_y)
    # Store average point along with polygon identifier
    average_points.append({'Name_right': group_name, 'geometry': avg_point})

# Create GeoDataFrame from average points
average_points_gdf = gpd.GeoDataFrame(average_points, geometry='geometry')
average_points_gdf=average_points_gdf.rename({'Name_right':'Name'},axis=1)
# Output or visualize average points
print(average_points_gdf)
# Define output file path for average points shapefile
output_shapefile = r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_pm\Shape_file\center_of_demand.shp'

# Save average points GeoDataFrame to shapefile
average_points_gdf.to_file(output_shapefile)
