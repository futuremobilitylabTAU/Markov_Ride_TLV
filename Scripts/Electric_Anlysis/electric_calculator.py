import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import ast
path_database=r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Simulation\Full_Day_Simulation\model.sqlite"
path_section_data=r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\section_data.csv"

##Tesla 3 
road_kwh_res=104 ###KWh/km
road_kwh_sec=104 
road_kwh_motor=160 
road_kwh_pri=129

#ELectric 
#electric_car_type=[18398069,18398062]

#%%% Connect to Data Base Result Of Aimsun Next
con = sqlite3.connect(path_database)
trajectory = pd.read_sql_query("SELECT * from MEVEHTRAJECTORY", con)
con.close()

#sample
#%%% trajectory order and time cancoulation

trajectory=trajectory[['origin', 'destination','oid','sid','exitTime','entranceTime','speed','travelledDistance']]
#trajectory['time']=trajectory['exitTime']-trajectory['generationTime']
print('flag')
#%%% only cars that trvael in the simulation time

trajectory=trajectory[trajectory['travelledDistance']>0]
#trajectory=trajectory[trajectory['sid'].isin(electric_car_type)==True]

#%%%

con = sqlite3.connect(path_database)
trajectory_by_section = pd.read_sql_query("SELECT * from MEVEHSECTTRAJECTORY", con)
con.close()
print('flag')

#%%
print('flag')

section_data=pd.read_csv(path_section_data)

trajectory_by_section=trajectory_by_section.merge(section_data,left_on='sectionId',right_on='Section-ID',how='left')

t1=trajectory_by_section.sample(1000)
print('flag')


trajectory_by_section['Electric_sum']=0
trajectory_by_section.loc[trajectory_by_section['Road-Type']=='Motorway','Electric_sum']=(road_kwh_motor*trajectory_by_section['Length']).astype(int)
trajectory_by_section.loc[trajectory_by_section['Road-Type']=='Secondary','Electric_sum']=(road_kwh_sec*trajectory_by_section['Length']).astype(int)
trajectory_by_section.loc[trajectory_by_section['Road-Type']=='Residential','Electric_sum']=(road_kwh_res*trajectory_by_section['Length']).astype(int)
trajectory_by_section.loc[trajectory_by_section['Road-Type']=='Primary','Electric_sum']=(road_kwh_pri*trajectory_by_section['Length']).astype(int)
trajectory_by_section.loc[trajectory_by_section['Road-Type']=='Suburban','Electric_sum']=(road_kwh_res*trajectory_by_section['Length']).astype(int)
elctric_vactor=trajectory_by_section.groupby('oid')['Electric_sum'].sum()
elctric_vactor=elctric_vactor.reset_index()
trajectory=trajectory.merge(elctric_vactor,on='oid',how='left')


#%%

key_zones=pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\zones_numbering_without_cord.csv')
trajectory=trajectory.merge(key_zones,left_on='origin',right_on='Aimsun',how='left')
trajectory=trajectory.rename({'Ex':'Origin-External-ID'},axis=1)
trajectory=trajectory.merge(key_zones,left_on='destination',right_on='Aimsun',how='left')
trajectory=trajectory.rename({'Ex':'Destenation-External-ID'},axis=1)
trajectory.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Supply\path_with_kwh_all.csv')