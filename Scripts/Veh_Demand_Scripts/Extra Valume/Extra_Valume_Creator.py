

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#%%
pd.set_option('future.no_silent_downcasting', True)

path_das=r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Cellular Demand\AvgDayHourlyTrips201819_1270_weekday_v1.csv'
celular_data = pd.read_csv(path_das)

path_key=r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\Key_Area33_to_Area1270.csv'
key_big_taz = pd.read_csv(path_key)

path_key=r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\Key_Celular_Aimsun.csv'
key = pd.read_csv(path_key)


path_cellular=r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\AimsunAreasList.csv'
transformation_list = pd.read_csv(path_cellular)


#%%

number_of_zones=key.groupby('Taz 1270')['TAZV41'].count()

keys_list=key.groupby('Taz 1270')['TAZV41'].apply(list)

#%%

keys_list=keys_list.to_frame()
number_of_zones=number_of_zones.to_frame()

#%%

celular_data=celular_data.merge(keys_list,left_on='fromZone',right_on='Taz 1270',how='left')
celular_data=celular_data.rename({'TAZV41':'From_List'},axis=1)

celular_data=celular_data.merge(number_of_zones,left_on='fromZone',right_on='Taz 1270',how='left')
celular_data=celular_data.rename({'TAZV41':'From_Number_Zones'},axis=1)
######
celular_data=celular_data.merge(keys_list,left_on='ToZone',right_on='Taz 1270',how='left')
celular_data=celular_data.rename({'TAZV41':'To_List'},axis=1)

celular_data=celular_data.merge(number_of_zones,left_on='ToZone',right_on='Taz 1270',how='left')
celular_data=celular_data.rename({'TAZV41':'To_Number_Zones'},axis=1)


celular_data=celular_data.merge(key_big_taz[['TAZ_1270','TAZ_33']],left_on='fromZone',right_on='TAZ_1270',how='left')
celular_data=celular_data.rename({'TAZ_33':'from_Taz_33'},axis=1)

celular_data=celular_data.merge(key_big_taz[['TAZ_1270','TAZ_33']],left_on='ToZone',right_on='TAZ_1270',how='left')
celular_data=celular_data.rename({'TAZ_33':'To_Taz_33'},axis=1)


celular_data=celular_data.drop(columns=['TAZ_1270_x','TAZ_1270_y'])
#%% 

celular_data.loc[celular_data['from_Taz_33']<=300010,'From_List']=51
celular_data.loc[celular_data['from_Taz_33']<=300010,'From_Number_Zones']=1

celular_data.loc[celular_data['To_Taz_33']<=300010,'To_List']=51
celular_data.loc[celular_data['To_Taz_33']<=300010,'To_Number_Zones']=1



celular_data.loc[(celular_data['from_Taz_33']==300028)|(celular_data['from_Taz_33']==300027)|(celular_data['from_Taz_33']==300011),'From_List']=52
celular_data.loc[(celular_data['from_Taz_33']==300028)|(celular_data['from_Taz_33']==300027)|(celular_data['from_Taz_33']==300011),'From_Number_Zones']=1

celular_data.loc[(celular_data['To_Taz_33']==300028)|(celular_data['To_Taz_33']==300027)|(celular_data['To_Taz_33']==300011),'To_List']=52
celular_data.loc[(celular_data['To_Taz_33']==300028)|(celular_data['To_Taz_33']==300027)|(celular_data['To_Taz_33']==300011),'To_Number_Zones']=1


celular_data.loc[celular_data['from_Taz_33']>=300029,'From_List']=53
celular_data.loc[celular_data['from_Taz_33']>=300029,'From_Number_Zones']=1

celular_data.loc[celular_data['To_Taz_33']>=300029,'To_List']=53
celular_data.loc[celular_data['To_Taz_33']>=300029,'To_Number_Zones']=1


key=key[key['TAZV41']<9000]

key.loc[(key['TAZV41']<7000)&(key['TAZV41']>5999),'TAZV41']=41
key.loc[(key['TAZV41']<8000)&(key['TAZV41']>6999),'TAZV41']=42
key.loc[(key['TAZV41']<9000)&(key['TAZV41']>7999),'TAZV41']=43




#%% Tnuha Hoza

celular_data_hutza=celular_data[['h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_hutza = celular_data_hutza[celular_data_hutza['From_List'].isnull()==False]
celular_data_hutza = celular_data_hutza[celular_data_hutza['To_List'].isnull()==False]


celular_data_hutza=celular_data_hutza[celular_data_hutza['From_List'].isin([51,52,53])==True]

celular_data_hutza=celular_data_hutza[celular_data_hutza['To_List'].isin([51,52,53])==True]


celular_data_hutza=celular_data_hutza.drop(columns=['From_Number_Zones','To_Number_Zones'])

#%%
# Assuming celular_data_hutza is your DataFrame

for i in range(6, 21):
    matrix_name = f"matrix_huza_{i}"
    column_name = f"h{i}"
    
    matrix = celular_data_hutza[['From_List', 'To_List', column_name]]
    matrix = matrix.groupby(['From_List', 'To_List'])[column_name].sum()
    matrix = matrix.unstack(level=-1)
    matrix = matrix * 0.6
    matrix.loc[52, 53] = 0
    matrix.loc[53, 52] = 0
    matrix.loc[52, 52] = 0
    matrix.loc[51, 51] = 0
    matrix.loc[53, 53] = 0
    
    globals()[matrix_name] = matrix


#%% Tnuha from zafom to mercaz
celular_data_north=celular_data[['ToZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_north = celular_data_north[celular_data_north['From_List'].isnull()==False]
celular_data_north = celular_data_north[celular_data_north['To_List'].isnull()==False]

#%%
celular_data_north=celular_data_north[celular_data_north['From_List'].isin([51])==True]
celular_data_from_north=celular_data_north[(celular_data_north['To_List'].isin([51,52,53])==False)]

celular_data_from_north=celular_data_from_north.copy()



for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_from_north[column_name] = celular_data_from_north[column_name] / celular_data_from_north['To_Number_Zones']


celular_data_from_north=celular_data_from_north.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_from_north_2=celular_data_from_north.merge(key,left_on='ToZone',right_on='Taz 1270',how='left')
celular_data_from_north_2=celular_data_from_north_2.drop(columns=['Taz 1270','ToZone' ,'To_List'])

#%%

for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_from_north_2"
    matrix_name = f"matrix_from_north_{i}"
    
    data = globals()[data_name][['TAZV41', 'From_List', column_name]]
    matrix = data.groupby(['From_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    
    globals()[matrix_name] = matrix


#%% Tnuha mercaz to zafom



celular_data_north=celular_data[['fromZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_north=celular_data_north[celular_data_north['To_List'].isin([51])==True]
celular_data_to_north=celular_data_north[(celular_data_north['From_List'].isin([51,52,53])==False)]
celular_data_to_north=celular_data_to_north.copy()

for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_to_north[column_name] = celular_data_to_north[column_name] / celular_data_to_north['From_Number_Zones']

#%%
celular_data_to_north=celular_data_to_north.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_to_north_2=celular_data_to_north.merge(key,left_on='fromZone',right_on='Taz 1270',how='left')

celular_data_to_north_2=celular_data_to_north_2.drop(columns=['Taz 1270','fromZone' ,'From_List'])

#%%



for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_to_north_2"
    matrix_name = f"matrix_to_north_{i}"
    
    data = globals()[data_name][['TAZV41', 'To_List', column_name]]
    matrix = data.groupby(['To_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    matrix = matrix.T
    
    globals()[matrix_name] = matrix










#%% Tnuha from darom to mercaz
celular_data_south=celular_data[['ToZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_south = celular_data_south[celular_data_south['From_List'].isnull()==False]
celular_data_south = celular_data_south[celular_data_south['To_List'].isnull()==False]

#%%
celular_data_south=celular_data_south[celular_data_south['From_List'].isin([53])==True]
celular_data_from_south=celular_data_south[(celular_data_south['To_List'].isin([51,52,53])==False)]

celular_data_from_south=celular_data_from_south.copy()

for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_from_south[column_name] = celular_data_from_south[column_name] / celular_data_from_south['To_Number_Zones']




celular_data_from_south=celular_data_from_south.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_from_south_2=celular_data_from_south.merge(key,left_on='ToZone',right_on='Taz 1270',how='left')
celular_data_from_south_2=celular_data_from_south_2.drop(columns=['Taz 1270','ToZone' ,'To_List'])

#%%



for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_from_south_2"
    matrix_name = f"matrix_from_south_{i}"
    
    data = globals()[data_name][['TAZV41', 'From_List', column_name]]
    matrix = data.groupby(['From_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    
    globals()[matrix_name] = matrix







#%% Tnuha mercaz to darom
celular_data_south=celular_data[['fromZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_south=celular_data_south[celular_data_south['To_List'].isin([53])==True]
celular_data_to_south=celular_data_south[(celular_data_south['From_List'].isin([51,52,53])==False)]
celular_data_to_south=celular_data_to_south.copy()


for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_from_south[column_name] = celular_data_to_south[column_name] / celular_data_to_south['From_Number_Zones']





#%%
celular_data_to_south=celular_data_to_south.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_to_south_2=celular_data_to_south.merge(key,left_on='fromZone',right_on='Taz 1270',how='left')

celular_data_to_south_2=celular_data_to_south_2.drop(columns=['Taz 1270','fromZone' ,'From_List'])


#%%
for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_to_south_2"
    matrix_name = f"matrix_to_south_{i}"
    
    data = globals()[data_name][['TAZV41', 'To_List', column_name]]
    matrix = data.groupby(['To_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    matrix = matrix.T
    
    globals()[matrix_name] = matrix















#%% Tnuha from mizrah to mercaz
celular_data_east=celular_data[['ToZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_east = celular_data_east[celular_data_east['From_List'].isnull()==False]
celular_data_east = celular_data_east[celular_data_east['To_List'].isnull()==False]

#%%
celular_data_east=celular_data_east[celular_data_east['From_List'].isin([52])==True]
celular_data_from_east=celular_data_east[(celular_data_east['To_List'].isin([51,52,53])==False)]

celular_data_from_east=celular_data_from_east.copy()



for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_from_east[column_name] = celular_data_from_east[column_name] / celular_data_from_east['To_Number_Zones']






celular_data_from_east=celular_data_from_east.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_from_east_2=celular_data_from_east.merge(key,left_on='ToZone',right_on='Taz 1270',how='left')
celular_data_from_east_2=celular_data_from_east_2.drop(columns=['Taz 1270','ToZone' ,'To_List'])

#%%



for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_from_east_2"
    matrix_name = f"matrix_from_east_{i}"
    
    data = globals()[data_name][['TAZV41', 'From_List', column_name]]
    matrix = data.groupby(['From_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    
    globals()[matrix_name] = matrix





#%% Tnuha mercaz to miztah
celular_data_east=celular_data[['fromZone','h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17','h18', 'h19', 'h20', 'From_List','From_Number_Zones', 'To_List', 'To_Number_Zones']]
celular_data_east=celular_data_east[celular_data_east['To_List'].isin([52])==True]
celular_data_to_east=celular_data_east[(celular_data_east['From_List'].isin([51,52,53])==False)]
celular_data_to_east=celular_data_to_east.copy()




for i in range(6, 21):
    column_name = f"h{i}"
    celular_data_to_east[column_name] = celular_data_to_east[column_name] / celular_data_to_east['From_Number_Zones']




#%%
celular_data_to_east=celular_data_to_east.drop(columns=['From_Number_Zones','To_Number_Zones'])

celular_data_to_east_2=celular_data_to_east.merge(key,left_on='fromZone',right_on='Taz 1270',how='left')

celular_data_to_east_2=celular_data_to_east_2.drop(columns=['Taz 1270','fromZone' ,'From_List'])

#%%


for i in range(6, 21):
    column_name = f"h{i}"
    data_name = f"celular_data_to_east_2"
    matrix_name = f"matrix_to_east_{i}"
    
    data = globals()[data_name][['TAZV41', 'To_List', column_name]]
    matrix = data.groupby(['To_List', 'TAZV41'])[column_name].sum().unstack(level=-1) * 0.6
    matrix = matrix.T
    
    globals()[matrix_name] = matrix





#%% connect all


for i in range(6, 21):
    matrix_to = globals()[f"matrix_to_north_{i}"]
    matrix_from = globals()[f"matrix_from_north_{i}"]
    matrix_to_s = globals()[f"matrix_to_south_{i}"]
    matrix_from_s = globals()[f"matrix_from_south_{i}"]
    matrix_to_e = globals()[f"matrix_to_east_{i}"]
    matrix_from_e = globals()[f"matrix_from_east_{i}"]
    matrix_h = globals()[f"matrix_huza_{i}"]
    
    matrix = pd.concat([matrix_to, matrix_to_s, matrix_to_e], axis=1)
    matrix = pd.concat([matrix, matrix_from, matrix_from_s, matrix_from_e], axis=0)
    
    matrix.loc[51, 52] = matrix_h.loc[51, 52]
    matrix.loc[51, 53] = matrix_h.loc[51, 53]
    matrix.loc[52, 51] = matrix_h.loc[52, 51]
    matrix.loc[53, 51] = matrix_h.loc[53, 51]
    
    globals()[f"matrix_{i}"] = matrix




#%%%



matrix_empty=pd.DataFrame(columns=transformation_list['Aimsun'],index=transformation_list['Aimsun'])
matrix_empty=matrix_empty.fillna(0)




#%%


for i in range(6, 21):
    globals()[f"matrix_{i}"] = globals()[f"matrix_{i}"].combine_first(matrix_empty).fillna(0)





#%%


for i in range(6, 21):
    matrix = globals()[f"matrix_{i}"]
    filename = f'C:/Users/dadashev/Dropbox/Optimizing_Mobility_with_Markovian_Model_for_AMoD/Data/Demand/External_Val_Matrixs/External_matrix_{i}.csv'
    matrix.to_csv(filename)








#%%%

idd=pd.read_csv(r"C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Scripts\Aimsun_Script\id.csv")
index_mapping = dict(zip(idd['Ex'], idd['Aimsun']))




for i in range(6, 21):
    matrix = globals()[f"matrix_{i}"]
    matrix = matrix.rename(index=index_mapping)
    matrix = matrix.rename(columns=index_mapping)
    matrix = matrix.sort_index(axis=1)
    matrix = matrix.sort_index()
    filename = f'C:/Users/dadashev/Dropbox/Optimizing_Mobility_with_Markovian_Model_for_AMoD/Data/Demand/External_Val_Matrixs/External_matrix_aimsun_{i}.csv'
    matrix.to_csv(filename)



























