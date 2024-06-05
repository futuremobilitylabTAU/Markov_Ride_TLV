import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

# Load the GeoPackage file
gpkg_am = gpd.read_file(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\AMoD Demand Map\am.gpkg")
gpkg_op = gpd.read_file(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\AMoD Demand Map\op.gpkg")
gpkg_pm= gpd.read_file(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\AMoD Demand Map\pm.gpkg")

gpkg_am['Name']=gpkg_am['Name'].astype(str)
gpkg_op['Name']=gpkg_op['Name'].astype(str)
gpkg_pm['Name']=gpkg_pm['Name'].astype(str)


gpkg_am['center'] = gpkg_am['geometry'].apply(lambda x: x.representative_point().coords[:])
gpkg_am['center'] = [coords[0] for coords in gpkg_am['center']]

gpkg_op['center'] = gpkg_op['geometry'].apply(lambda x: x.representative_point().coords[:])
gpkg_op['center'] = [coords[0] for coords in gpkg_op['center']]

gpkg_pm['center'] = gpkg_pm['geometry'].apply(lambda x: x.representative_point().coords[:])
gpkg_pm['center'] = [coords[0] for coords in gpkg_pm['center']]




# Plotting
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

# Plot the first map with OSM background
#ax = gpkg_am.plot(aspect=1,ax=axes[0], alpha=0.5, edgecolor='black')

ax = gpkg_am.plot(column='NUMPOINTS', cmap='Purples', legend=True,  aspect=1, ax=axes[0], alpha=0.5, edgecolor='black')


ctx.add_basemap(ax, crs='EPSG:2039', zoom=12,attribution='')




for idx, row in gpkg_am.iterrows():
    axes[0].annotate(xy=row['center'], horizontalalignment='center', text=row['Name'])



axes[0].set_title('AM - 11 Clusters')

# Plot the second map with OSM background
ax = gpkg_op.plot(column='NUMPOINTS', cmap='Blues', legend=True,  aspect=1, ax=axes[1], alpha=0.5, edgecolor='black')
ctx.add_basemap(ax, crs='EPSG:2039' , zoom=12,attribution='')



for idx, row in gpkg_op.iterrows():
    axes[1].annotate(xy=row['center'], horizontalalignment='center', text=row['Name'])



axes[1].set_title('OP - 13 Clusters')

# Plot the third map with OSM background
ax = gpkg_pm.plot(column='NUMPOINTS', cmap='Greens', legend=True,  aspect=1, ax=axes[2], alpha=0.5, edgecolor='black')
ctx.add_basemap(ax, crs='EPSG:2039', zoom=12,attribution='')



for idx, row in gpkg_pm.iterrows():
    axes[2].annotate(xy=row['center'], horizontalalignment='center', text=row['Name'])




axes[2].set_title('PM - 20 Clusters')

# Show the plot
plt.show()