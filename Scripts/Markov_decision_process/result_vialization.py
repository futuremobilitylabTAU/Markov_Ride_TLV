"""
Python Script for Processing Markov Decision Process Outputs

Author: Gabriel Dadashev  14-3-24

This script processes the outputs of a Markov Decision Process (MDP) model,
including the best policy, maximum state value, and maximum state value for pick-up.
It reads CSV files containing the MDP outputs, applies styling and color mapping to the data,
and generates HTML files to visualize the results.

Dependencies:
- pandas
- numpy
-random

Usage:
1. Set the value of the `model` variable to the desired model number.
2. Run the script to generate HTML files with the processed data.

Note: Ensure that the paths to the input CSV files and output HTML files are correctly specified.

"""


import pandas as pd 
import numpy as np
import random


###Model Number on the Folder

model=6


# Define function to apply different configurations based on rules
def apply_config(value):
    if value[0] =='R':
        return 'background-color: %s'%(color_dict[int(value[1:])])
    elif value[0:3]=='INR':
        return 'background-color: %s'%(color_dict[int(value[3:])])
    # Example: If value is less than 0, apply red background color
    elif value[0]=='C':
        return 'background-color: %s'%(color_dict[int(value[1:])])
    elif value[0]=='T':
        return 'background-color: %s'%(color_dict[int(value[1:])])

 
    
# Define color map functions
def color_map_1(value, max_value):
    # Define the color range (lighter to darker)
   # color_range = ['#feffeb', '#fcfcc5', '#fcf99e','#f5e08f', '#f9cc60', '#ffb62e']
    color_range = ['#feffeb', '#fcfcc5', '#fcf99e','#f5e08f', '#f9cc60', '#ffb62e']

    # Calculate bin index based on value and maximum value in the column
    bin_index = min(int(value) / max_value * (len(color_range) - 1), len(color_range) - 1)
    print("bin_index:", int(bin_index))
    print("length of color_range:", len(color_range))
    return f'background-color: {color_range[int(bin_index)]}'

# Define color map functions
def color_map_2(value, max_value):
    # Define the color range (lighter to darker)
    color_range = ['#e6ffff', '#d0f7fd', '#bbeffd','#a9e5ff', '#9cdaff', '#96ceff', '#98c1ff', '#a1b2ff']
    # Calculate bin index based on value and maximum value in the column
    bin_index = min(int(value) / max_value * (len(color_range) - 1), len(color_range) - 1)
    return f'background-color: {color_range[int(bin_index)]}'


# Apply style to DataFrame
def apply_style_1(df):
    styled_df = df.style
    for col in df.columns:
        max_value = df[col].max()
        styled_df = styled_df.apply(lambda x: [color_map_1(v, max_value) if pd.notnull(v) else '' for v in x], axis=0)
    return styled_df
# Apply style to DataFrame
def apply_style_2(df):
    styled_df = df.style
    for col in df.columns:
        styled_df = df.style.apply(lambda x: [color_map_2(v, df.max(axis=1).max()) if pd.notnull(v) else '' for v in x], axis=1)
    return styled_df

# Define  rule for updating the index
def update_index_rule(old_index):
    global global_variable
    interval_up=round(old_index*(57.5/global_variable),1)
    interval_down=round(interval_up-(57.5/global_variable),1)
    interval_up=str(interval_up)
    interval_down=str(interval_down)
    return interval_down+'-'+interval_up


# Process VI CSV file
resullt=pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Outputs\optimal_VI_40.csv"%(model))
resullt[['Cluster', 'Energy']] = resullt['State'].str.split(',', expand=True)
resullt['Cluster']=resullt['Cluster'].str[1:]
resullt['Energy']=resullt['Energy'].str[1:-1]
resullt=resullt[resullt['Action']!='P'].copy()
resullt=resullt[['Energy','Cluster','Action']].copy()
resullt['Energy']=resullt['Energy'].astype(int)
resullt['Cluster']=resullt['Cluster'].astype(int)
resullt=resullt.rename({'Energy':'Kwh'},axis=1)
resullt=resullt.set_index(['Kwh','Cluster'])
resullt=resullt.unstack(level=1)
resullt.columns=[resullt.columns[i][1] for i in range(len(resullt.columns))]
color_dict = {i+1: '#' + '%06X' % random.randint(0, 0xFFFFFF) for i in range(len(resullt.columns))}
for i in resullt.columns: resullt[i] = resullt[i].replace('T', 'T%s'%(i))
global_variable=len(resullt)
new_index = resullt.index.map(update_index_rule)
resullt.index = new_index








resullt_bp=resullt.copy()



# Process state value CSV file
resullt=pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Outputs\optimal_state_value_40.csv"%(model))
resullt[['Cluster', 'Energy']] = resullt['State'].str.split(',', expand=True)
resullt['Cluster']=resullt['Cluster'].str[1:]
resullt['Energy']=resullt['Energy'].str[1:-1]
resullt=resullt[['Energy','Cluster','Value']].copy()
resullt['Cluster']=resullt['Cluster'].astype(int)
resullt_e=resullt[resullt['Energy']!=''].copy()
resullt_e['Energy']=resullt_e['Energy'].astype(int)
resullt_p=resullt[resullt['Energy']==''].copy()
resullt_p['Energy']='P'
resullt_p=resullt_p.rename({'Energy':'NIS'},axis=1)
resullt_e=resullt_e.set_index(['Energy','Cluster'])
resullt_p=resullt_p.set_index(['NIS','Cluster'])
resullt_e=resullt_e.unstack(level=1)
resullt_p=resullt_p.unstack(level=1)
resullt_p.columns=[resullt_p.columns[i][1] for i in range(len(resullt_p.columns))]
resullt_e.columns=[resullt_e.columns[i][1] for i in range(len(resullt_e.columns))]
global_variable=len(resullt_e)
new_index = resullt_e.index.map(update_index_rule)
resullt_e.index = new_index



# Apply the function to the DataFrame element-wise
resullt_bp = resullt_bp.style.map(apply_config)
resullt_e = apply_style_1(resullt_e)
resullt_p = apply_style_2(resullt_p)

# Write HTML file
   
html_title_1 = "<h1>Best Policy</h1>\n"
html_title_2 = "<h1>Maxiumum State Value</h1>\n"
html_title_3 = "<h1>Maxiumum State Value For Pick-Up</h1>\n"

 

# Combine tables horizontally in HTML
html_combined_tables = (
    "<div style='display:flex;'>\n"
    "<div style='flex:1;'>\n"
    + html_title_1
    + resullt_bp.to_html()
    + "</div>\n"
    "<div style='flex:1;'>\n"
    + html_title_2
    + resullt_e.to_html()
    + "</div>\n"
    "</div>"
    "<div style='flex:1;'>\n"
    + html_title_3
    + resullt_p.to_html()
    + "</div>\n"
    "</div>"
)

# Write the combined tables to the HTML file
with open(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Outputs\combined_tables_0.html"%(model), "w") as f:
    f.write(html_combined_tables)
