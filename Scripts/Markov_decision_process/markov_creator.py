# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pandas import DataFrame
import random
import datetime
import uuid
import math
import utm
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.colors as mcolors
import sqlite3
from scipy.stats import norm
from scipy.stats import lognorm
import statistics
import fiona
import geopandas as gpd
from shapely.geometry import Point
import scipy.integrate as integrat
from scipy.integrate import quad
import scipy

#%%%
model=7


class Demand:
    def __init__(self, requst,taz_info,number_of_cluster):

        

        self.requst=requst
        self.number_of_cluster = number_of_cluster    
        self.cluster_centers = None
        self.regions=None
        self.voronoi_vertices_inner=None
        self.number_requst = len(requst)
        self.KMeans(requst)
        self.geo_data_anlysis()
        self.taz_info = taz_info 
         
        self.add_clusters_to_oringins()
        
        self.demad_probbility=self.Demand_Anlysis()

        
        self.add_clusters_to_centroids()
        self.add_clusters_to_detantion()
          
        
 
    def get_request(self):
        return self.requst 
    def get_number_request(self):
        return self.number_requst 
    def get_cluster_centers(self):
        return self.cluster_centers 
    def get_demand_probbility(self):
        return self.demad_probbility 
    
    
    def KMeans (self,requst):
        print('KMeans')
        origin_list = requst.origin.values.tolist()
        number_of_cluster=self.number_of_cluster
        requst['x']=requst['origin'].apply(lambda x: x['x'])
        requst['y']=requst['origin'].apply(lambda x: x['y'])
        x=requst['x'].to_numpy()
        y=requst['y'].to_numpy()
        data=np.vstack((x,y)).T
        ### here  db = my KMeans  Algo
        db = KMeans(n_clusters=self.number_of_cluster, random_state=0).fit(data)
        labels = db.labels_
        labels_df = pd.DataFrame(labels, columns = ['cluster'])
        
        requst=requst.reset_index()
        
                 
    
        
        self.cluster_centers=db.cluster_centers_
        ### Make Redions
    
        dec=self.veroni(db.cluster_centers_)
        self.regions=dec 
        section_cord = pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\section_coordination_by_taz.csv')
        #Voronoi Lines
        centers=self.get_cluster_centers()
        vor = Voronoi(centers)     
        voronoi_plot_2d(vor, show_vertices=True, line_colors='orange',line_width=4.5, line_alpha=1, point_size=2)
        #Points
        plt.scatter(section_cord['X'].to_numpy(), section_cord['Y'].to_numpy(),c='black', alpha=0.5,marker='.')
        plt.scatter(x, y, c=labels, alpha=1)
        plt.scatter(self.cluster_centers[:,0],self.cluster_centers[:,1],c='red', alpha=0.7,marker='o',s=90)
        ##Plot Options: Limitis and Text
        plt.subplots_adjust(top=0.94,bottom=0.04,left=0.505,right=0.77,hspace=0.2,wspace=0.2)
        plt.grid(color='black', linestyle='-', linewidth=0.1)
        plt.ylim(640000,680000)
        plt.xlim(173000,200000)
        plt.title('%s'%(self.number_of_cluster)+' clusters', fontsize=30)
        plt.show()
        
        
    
        
    
    def Demand_Anlysis (self):
        
        print('Demand Anlysis')
        request=self.get_request()
        sumerry=request.groupby('cluster')['id'].count()
        sumerry=sumerry.reset_index()
        sumerry['prob']=sumerry['id']/sumerry['id'].sum()
        sumerry.cluster=sumerry.cluster.astype(int)
        area_dict = dict(zip(sumerry.cluster+1, sumerry.prob))
        
    
     
        
        return area_dict
    def geo_data_anlysis(self):
    
        print('geo data anlysis')
        schema= {'geometry':'Point','properties' : [('Name','str')]}
        pointShp=fiona.open(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Shape_file\points_origins.shp'%(model),mode='w',driver='ESRI Shapefile',schema=schema,crs="EPSG:2039")
        self.requst['shape_origin']=self.requst.apply(lambda x: {'geometry': {'type':'Point', 'coordinates': (x['x'],x['y'])},'properties':{'Name': x['name']}},axis=1)
        for index, row in self.requst.iterrows():
            pointShp.write(row.shape_origin)
        pointShp.close()
        
        schema= {'geometry':'Polygon','properties' : [('Name','str')]}
        polyShp=fiona.open(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Shape_file\clusters.shp'%(model),mode='w',driver='ESRI Shapefile',schema=schema,crs="EPSG:2039")
        for i in range(len(self.regions)):
            d=self.regions[i]
            d['Name']=i
            xyList = []
            rowName = ''
            for index, row in d.iterrows():
                xyList.append((row.x,row.y))
                rowName = row.Name
            rowDict = {'geometry' : {'type':'Polygon','coordinates': [xyList]}, 'properties': {'Name' : rowName}}
            polyShp.write(rowDict)
        polyShp.close()
        
    def veroni(self,towers):
        print('veroni')
        import numpy as np
        import scipy as sp
        import scipy.spatial
        import sys
        eps = sys.float_info.epsilon
        bounding_box = np.array([173000, 200000, 640000, 680000]) # [x_min, x_max, y_min, y_max]
        def in_box(towers, bounding_box):
            return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                                 towers[:, 0] <= bounding_box[1]),
                                  np.logical_and(bounding_box[2] <= towers[:, 1],
                                                 towers[:, 1] <= bounding_box[3]))
        def voronoi(towers, bounding_box):
            print('voronoi')
            i = in_box(towers, bounding_box)
            points_center = towers[i, :]
            points_left = np.copy(points_center)
            points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
            points_right = np.copy(points_center)
            points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
            points_down = np.copy(points_center)
            points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
            points_up = np.copy(points_center)
            points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
            points = np.append(points_center, np.append(np.append(points_left,points_right,axis=0), np.append(points_down, points_up,    axis=0),      axis=0), axis=0)
            vor = sp.spatial.Voronoi(points)
            regions = []
            for region in vor.regions:
                flag = True
                for index in region:
                    if index == -1:
                        flag = False
                        break
                    else:
                        x = vor.vertices[index, 0]
                        y = vor.vertices[index, 1]
                        if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                               bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                            flag = False
                            break
                if region != [] and flag:
                    regions.append(region)
            vor.filtered_points = points_center
            vor.filtered_regions = regions
            return vor

        def centroid_region(vertices):
            A = 0
            C_x = 0
            C_y = 0
            for i in range(0, len(vertices) - 1):
                s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
                A = A + s
                C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
                C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
            A = 0.5 * A
            C_x = (1.0 / (6.0 * A)) * C_x
            C_y = (1.0 / (6.0 * A)) * C_y
            return np.array([[C_x, C_y]])

        vor = voronoi(towers, bounding_box)
        dic={}
        ii=-1
        for region in vor.filtered_regions:
            ii=ii+1
            vertices = vor.vertices[region, :]
            dic.update({ii:  pd.DataFrame({'x':vertices[:, 0],'y':vertices[:, 1]})})
        return dic
    
    
      
    def add_clusters_to_oringins(self):
        
           self.requst['coords'] = list(zip(  self.requst['x'],  self.requst['y']))
           self.requst['coords'] = self.requst['coords'].apply(Point)
           cluster_shape = gpd.GeoDataFrame.from_file(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Shape_file\clusters.shp'%(model))
           points = gpd.GeoDataFrame(self.requst, geometry='coords', crs=cluster_shape.crs)
           pointInPolys = gpd.tools.sjoin(points, cluster_shape, predicate="within", how='left')
           pointInPolys=pointInPolys.rename({'Name':'cluster'},axis=1)
           pointInPolys=pointInPolys.drop(['index_right'],axis=1)
           self.requst=pointInPolys
           print('*')
           
    
    def add_clusters_to_centroids(self):
        
           self.taz_info['coords'] = list(zip(  self.taz_info['X'],  self.taz_info['Y']))
           self.taz_info['coords'] = self.taz_info['coords'].apply(Point)
           cluster_shape = gpd.GeoDataFrame.from_file(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Shape_file\clusters.shp'%(model))
           points = gpd.GeoDataFrame(self.taz_info, geometry='coords', crs=cluster_shape.crs)
           pointInPolys = gpd.tools.sjoin(points, cluster_shape, predicate="within", how='left')
           self.taz_info=pointInPolys
           print('*')
           
    def add_clusters_to_detantion(self):

    
        self.requst['coords2']=self.requst.apply(lambda x: [x['destination']['x'],x['destination']['y']],axis=1)
        self.requst['coords2'] = self.requst['coords2'].apply(Point)
        cluster_shape = gpd.GeoDataFrame.from_file(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Shape_file\clusters.shp'%(model))
        points = gpd.GeoDataFrame(self.requst, geometry='coords2', crs=cluster_shape.crs)
        pointInPolys = gpd.tools.sjoin(points, cluster_shape, predicate="within", how='left')
        pointInPolys=pointInPolys.rename({'Name':'destination_cluster'},axis=1)
        self.requst=pointInPolys
        



        
class Energy:
    def __init__(self,number_of_energy_levels=3):
        if number_of_energy_levels>2:
            self.number_of_energy_levels=number_of_energy_levels
        else: print(" the energy level can be only bigger from 2 , defult value 3 is updaeted")


class MDP_Data:
    def __init__(self,number_of_energy_levels,number_of_clusters, requst,taz_info,supply_info):
        
        
        
        ##Set Up Time 6-11
        simulation_starting=16 #11   #16
        simulation_ending=21  #16  #21

    
        markov_start_time=(simulation_starting-6)*3600
        markov_end_time=(simulation_ending-6)*3600
        
        t=16 ## 6 +5 = 11 , 11+5 =16 , 16+5 =21 
        hours = int(t)
        minutes = (t*60) % 60
        seconds = (t*3600) % 60
        time_start="%d:%02d:%02d" % (hours, minutes, seconds)
        time_start = datetime.strptime(time_start, '%H:%M:%S')

        
        hours_2 = int(t+5)
        minutes_2 = ((t+5)*60) % 60
        seconds_2 = ((t+5)*3600) % 60
        time_ending="%d:%02d:%02d" % (hours_2, minutes_2, seconds_2)  
        time_ending = datetime.strptime(time_ending, '%H:%M:%S')
        
                      
        requst['time']=(requst['time']/1000)
        requst['time'] = requst['time'].apply(lambda x:datetime.fromtimestamp(x))
        requst['time'] = requst['time'].apply(lambda x:x.strftime('%H:%M:%S')) 
        requst['time'] = requst['time'].apply(lambda x:datetime.strptime(x, '%H:%M:%S'))  
        requst=requst[(requst['time']<=time_ending)&(requst['time']>=time_start)]
        requst=requst.copy()
    
   
 
        
        
        ##time_filter needed here
        self.demand=Demand(requst,taz_info,number_of_clusters)
        self.supply_info=supply_info[(supply_info['entranceTime']>=markov_start_time)&(supply_info['entranceTime']<=markov_end_time)].copy()
        

  

        self.supply_info= self.supply_info.merge(self.demand.taz_info[['Ex','Name']],left_on='Origin-External-ID',right_on='Ex',how='left')
        self.supply_info= self.supply_info.rename({'Name':'Origin Cluster'},axis=1)
        self.supply_info= self.supply_info.merge(self.demand.taz_info[['Ex','Name']],left_on='Destenation-External-ID',right_on='Ex',how='left')
        self.supply_info= self.supply_info.rename({'Name':'Destenation Cluster'},axis=1)
        
        self.supply_info=self.supply_info.dropna() ####### לחזור לזה
        self.supply_info['Origin Cluster']=  self.supply_info['Origin Cluster'].astype(int)
        self.supply_info['Origin Cluster']=  self.supply_info['Origin Cluster']+1 
        self.supply_info['Destenation Cluster']=  self.supply_info['Destenation Cluster'].astype(int)
        self.supply_info['Destenation Cluster']=  self.supply_info['Destenation Cluster']+1 
        self.supply_info['Electric_sum']=self.supply_info['Electric_sum']/1000
        
        self.supply_info['Reward_Empty']=-0.62*self.supply_info['Electric_sum']
        self.supply_info['Taxi_Profit']=16.3+1.72*supply_info['travelledDistance']/1000+1.72*(supply_info['exitTime']-supply_info['entranceTime'])/3600
        self.supply_info['Reward_Paseenger']=0.5* self.supply_info['Taxi_Profit']-self.supply_info['Reward_Empty']
     

        self.enregy=Energy(number_of_energy_levels)
        self.time=t
        self.state_set=self.state_geneetor(number_of_energy_levels,number_of_clusters)#[]
        self.action_set=self.action_geneetor(number_of_energy_levels,number_of_clusters)#[]
        self.state_action_subset=self.state_action_subset_geneetor(number_of_energy_levels,number_of_clusters)#[]
        self.next_state_from_state=self.next_state_from_state_geneetor(number_of_energy_levels,number_of_clusters)#[]
        ### 1 KWh price is 0.62 Shekel - Fule Tesla 3 Buttery is 57.5 KWh (Regular Model) and 75 kWh to Long Range Model 
        ### Forfet price - price of make decision to go out from the markov chain or stay.   
        self.risk_var={}
        
    
        self.reward=self.set_up_reward(number_of_energy_levels,number_of_clusters,-35.65,0) #{}'
        self.transition=self.set_up_transition()
        self.make_data_frames()
        print(self.risk_var)
        
        

    def get_states(self):
        return self.state_set
    def get_actions(self):
        return self.action_set
    def get_state_action_subset(self):
        return self.state_action_subset
    def get_transition(self):
        return self.transition
    def get_next_state_from_state(self):
        return self.next_state_from_state
    def get_reward(self):
        return self.reward
    def get_demand_table(self):
        return self.demand.get_request()        
    def state_geneetor(self,number_of_energy_levels,number_of_clusters):
        ### Pick_Up states
        state=[]
        for iterator in range(1,number_of_clusters+1):
            state_pick_up='('+'%s'%(iterator)+','+'P'+')'
            state.append(state_pick_up)
        ### Location States
        for iterator_cluster in range(1,number_of_clusters+1):
            for iterator_energy in range(1,number_of_energy_levels+1):
                state_location='('+'%s'%(iterator_cluster)+','+'E'+'%s'%(iterator_energy)+')'
                state.append(state_location)
        return state
    def action_geneetor(self,number_of_energy_levels,number_of_clusters):
        print('action_geneetor')
        action=['T','P']
        ### Rebalance Actions
        for iterator_cluster in range(1,number_of_clusters+1):
            action_inr='INR'+'%s'%(iterator_cluster)
            action.append(action_inr)
            action_reb='R'+'%s'%(iterator_cluster)
            action.append(action_reb) 
            action_charge='C'+'%s'%(iterator_cluster)
            action.append(action_charge) 
        return action
    def state_action_subset_geneetor(self,number_of_energy_levels,number_of_clusters) :
        print('state action subset geneetor')
        sub_set=[]
        for state in self.state_set:
            for action in self.action_set:
                cluster=int(state[state.find('(')+1 : state.find(',')]) 
                energy=state[state.find(',')+1 : state.find(')')]
                tup=(state,action)
                if action=='P' and energy=='P' and tup not in sub_set :
                    sub_set.append(tup)
                if action=='T' and energy[0]=='E' and int(energy[1:])>1 and tup not in sub_set :
                    sub_set.append(tup)
                if action[0]=='R' and int(action[1:])!=cluster and energy[0]=='E' and int(energy[1:])>1 and tup not in sub_set :
                        sub_set.append(tup)
                if action[0:3]=='INR' and int(action[3:])==cluster and energy[0]=='E' and int(energy[1:])>1 and tup not in sub_set :
                        sub_set.append(tup)
                if action[0]=='C' and energy[0]=='E' and int(energy[1:])==1 and int(action[1:])==cluster and tup not in sub_set :
                        sub_set.append(tup) 
        return sub_set
    def next_state_from_state_geneetor(self,number_of_energy_levels,number_of_clusters):
        print('next_state_from_state_geneetor')
        next_states=[]
        for state_action in self.state_action_subset:
            if state_action[1]=='P':
                next_states.append((state_action,state_action[0])) 
            if state_action[1][0]=='C':
                cluster=int(state_action[0][state_action[0].find('(')+1 : state_action[0].find(',')]) 
                next_states.append((state_action,(cluster,'E%s'%(number_of_energy_levels))))
            if state_action[1][0]=='R':
                cluster=int(state_action[0][state_action[0].find('(')+1 : state_action[0].find(',')]) 
                energy=state_action[0][state_action[0].find(',')+1 : state_action[0].find(')')]
                action=state_action[1]
                for i in reversed(range(int(energy[1:]))):
                    if(i==0):break
                    next_states.append((state_action,(int(action[1:]),'E%s'%(i))))       
            if state_action[1][0:3]=='INR':
                cluster=int(state_action[0][state_action[0].find('(')+1 : state_action[0].find(',')]) 
                energy=state_action[0][state_action[0].find(',')+1 : state_action[0].find(')')]
                action=state_action[1]
                for i in reversed(range(int(energy[1:]))):
                    if(i==0):break
                    next_states.append((state_action,(cluster,'E%s'%(i)))) 
            if state_action[1]=='T':
                next_states.append((state_action,state_action[0])) 
                cluster=int(state_action[0][state_action[0].find('(')+1 : state_action[0].find(',')]) 
                next_states.append((state_action,(cluster,'P')))
        return next_states
    
   
    def set_up_reward(self,number_of_energy_levels,number_of_clusters,charging_price,forfeit_price):
        print('set up reward')
        reward={}
        for state_actions in self.state_action_subset:
            if state_actions[1][0]=='C':
                reward.update({state_actions:charging_price}) 
         
            if state_actions[1][0]=='R':
                area_dest=int(state_actions[1][1:])
                area_or=int(state_actions[0][state_actions[0].find('(')+1 : state_actions[0].find(',')])
                ###ניתן לדייק על ידי אזורים קרובים בהמשך
                mean_value=self.supply_info['Reward_Empty'].mean()
                d_value=self.supply_info['travelledDistance'].mean()/1000

                supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)&(self.supply_info['Destenation Cluster']==area_dest)]
                rew=supply['Reward_Empty'].mean()
                if str(rew)=='nan' : rew=mean_value
                
                ## פונקציית סיכון 
                eng=state_actions[0][state_actions[0].find(',')+1 : state_actions[0].find(')')]
                eng=int(eng[1:])
                d=supply['travelledDistance'].mean()/1000
                if str(d)=='nan' : d=d_value
                print(d)
               # risk=57.5/eng+0.000001*np.exp(d)
                risk=0
                if d>20: risk=100000
               
                

        
          #      risk=0.00001*np.exp(d)
          
          
                self.risk_var.update({(eng,area_or,area_dest):risk})
                rew=rew-risk
                
                
                reward.update({state_actions:rew})  
            if state_actions[1][0:3]=='INR':
                area_dest=int(state_actions[1][3:])
                area_or=int(state_actions[1][3:])
                mean_value=self.supply_info['Reward_Empty'].mean()
                supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)&(self.supply_info['Destenation Cluster']==area_dest)]
                rew=supply['Reward_Empty'].mean()
                if str(rew)=='nan' : rew=mean_value
                
           
                
                
                reward.update({state_actions:rew}) 
            if state_actions[1][0]=='T':
                 area_dest=int(state_actions[0][state_actions[0].find('(')+1 : state_actions[0].find(',')])
                 area_or=area_dest
                 mean_value=self.supply_info['Reward_Empty'].mean()
                 supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)&(self.supply_info['Destenation Cluster']==area_dest)]
                 rew=supply['Reward_Empty'].mean()
                 if str(rew)=='nan' : rew=mean_value
                 reward.update({state_actions:rew+forfeit_price})
            if state_actions[1][0]=='P':
                area_or=int(state_actions[0][state_actions[0].find('(')+1 : state_actions[0].find(',')])
                supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)]
                supply_group=supply.groupby(['Origin Cluster','Destenation Cluster'])['Reward_Paseenger'].mean()
                req=self.demand.requst
                supply_group=supply_group.reset_index()
                supply_group['Origin Cluster']=supply_group['Origin Cluster'].astype(int)
                supply_group['Destenation Cluster']=supply_group['Destenation Cluster'].astype(int)
                req['cluster']=req['cluster'].astype(int)
                req['destination_cluster']=req['destination_cluster'].astype(int)
                
                req['destination_cluster']=req['destination_cluster']+1
                req['cluster']=req['cluster']+1

                ################WTF?
                req=req.merge(supply_group,left_on=['cluster','destination_cluster'],right_on=['Origin Cluster','Destenation Cluster'],how='left')
                req=req[req['cluster']==area_or]
                rew=req['Reward_Paseenger'].mean()
                if str(rew)=='nan' : rew=0
                reward.update({state_actions:rew}) 

   #             print(area_dest,area_or)
                ####Suply Table Make Global Anlysis By Time and 
        return reward ## Update
 
    def set_up_transition(self):
        print('set_up_transition')
        p=self.demand.get_demand_probbility()
        transition={}
        for trans in self.next_state_from_state:
            if trans[0][1][0]=='C':
                transition.update({trans:1})
            if trans[0][1][0]=='P':
                transition.update({trans:1})
            if trans[0][1][0]=='T':  
                area_or=int(trans[0][0][trans[0][0].find('(')+1 : trans[0][0].find(',')])
                p_area=p[area_or]
                energy=trans[0][0][trans[0][0].find(',')+1 : trans[0][0].find(')')]
                j=int(energy[1:])
                m=self.enregy.number_of_energy_levels
                supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)]
                supply=supply[supply['Electric_sum']>0].copy()
                mu, loc, sigma = scipy.stats.lognorm.fit(supply['Electric_sum'], floc=0)
                sigma=abs(sigma)       
                def integrand(x, mu, sigma):
                    f=1/(x*sigma*np.sqrt(2*np.pi))
                    f=f*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
                    return f
                inte = quad(integrand, 0, (j/m)*57.5, args=(mu,sigma))
                if inte[0]>1: i=1
                else: i=inte[0]
                if(trans[1]==trans[0][0]):
                    transition.update({trans:1-p_area*i})
                else:
                    transition.update({trans:p_area*i})
            if trans[0][1][0]=='R':
                area_or=int(trans[0][0][trans[0][0].find('(')+1 : trans[0][0].find(',')])
                area_dest=int(trans[0][1][1:])
                
                energy_or=trans[0][0][trans[0][0].find(',')+1 : trans[0][0].find(')')]
                energy_or=int(energy_or[1:])
                energy_dest=int(trans[1][1][1:])
                energy_dest=energy_or-energy_dest
                
                supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)&(self.supply_info['Destenation Cluster']==area_dest)]
                supply=supply[supply['Electric_sum']>0].copy()
                mu, loc, sigma = scipy.stats.lognorm.fit(supply['Electric_sum'], floc=0)
                sigma=abs(sigma)       
                m=self.enregy.number_of_energy_levels
                inte_all_interval = quad(integrand, 0, ((energy_or-1)/m)*57.5, args=(mu,sigma))

                def integrand(x, mu, sigma):
                    f=1/(x*sigma*np.sqrt(2*np.pi))
                    f=f*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
                    return f
                inte = quad(integrand, ((energy_dest-1)/m)*57.5, (energy_dest/m)*57.5, args=(mu,sigma))
                i=inte[0]/inte_all_interval[0]
                if i<0.00001: i=0
                transition.update({trans:i})
            if trans[0][1][0:3]=='INR':
                 area_or=int(trans[0][0][trans[0][0].find('(')+1 : trans[0][0].find(',')])
                 area_dest=area_or
                 energy_or=trans[0][0][trans[0][0].find(',')+1 : trans[0][0].find(')')]
                 energy_or=int(energy_or[1:])
                 energy_dest=int(trans[1][1][1:])
                 energy_dest=energy_or-energy_dest
                 supply=self.supply_info[(self.supply_info['Origin Cluster']==area_or)&(self.supply_info['Destenation Cluster']==area_dest)]
                 supply=supply[supply['Electric_sum']>0].copy()
                 mu, loc, sigma = scipy.stats.lognorm.fit(supply['Electric_sum'], floc=0)
                 sigma=abs(sigma)       
                 m=self.enregy.number_of_energy_levels
                 inte_all_interval = quad(integrand, 0, ((energy_or-1)/m)*57.5, args=(mu,sigma))
                 def integrand(x, mu, sigma):
                     f=1/(x*sigma*np.sqrt(2*np.pi))
                     f=f*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
                     return f
                 inte = quad(integrand, ((energy_dest-1)/m)*57.5, (energy_dest/m)*57.5, args=(mu,sigma))
                 i=inte[0]/inte_all_interval[0]
                 if i<0.00001: i=0

                 transition.update({trans:i})
            print(trans)
        return transition
    
    def check_transition(self):
        df=pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Transitions\trans_df.csv'%(model))
        df['Transitions']= round(df['Transitions'],6)
        grop=df.groupby(['State','Action']).agg({'Transitions':'sum','Next State':'count'}) 
        grop['delta']=1-grop['Transitions']
        grop=grop[grop['delta']!=0]
        grop['delta_relative']=grop['delta']/grop['Next State']
        grop=grop.reset_index()
        grop=grop[['State', 'Action','delta_relative']]
        df_3=df.merge(grop,on=['State', 'Action'],how='left')
        df_3=df_3.fillna(0)
        df_3['Transitions']=df_3['Transitions']+df_3['delta_relative']
        df_3=df_3[['State', 'Action','Next State','Transitions']]
        check=df_3.groupby(['State','Action']).agg({'Transitions':'sum'}) 
        if check['Transitions'].sum()==len(check): print('tansition are ok')
        
        
        
        
        df_3.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Transitions\trans_df.csv'%(model),index=False)

        


        
 
        


            
    def make_data_frames(self):
        trans=self.get_transition()
        rew=self.get_reward()
        
        
        sub_state_actions=self.get_state_action_subset()
        state_action_state=self.get_next_state_from_state()
 
        a = np.zeros(shape=(len(sub_state_actions),3))
        rew_df=pd.DataFrame(a,columns=['State','Action','Reword'])
        rew_df['State']=rew_df.apply(lambda x: sub_state_actions[x.name][0], axis=1)
        rew_df['Action']=rew_df.apply(lambda x: sub_state_actions[x.name][1], axis=1)
        rew_df['State']=rew_df['State'].astype(str)
        rew_df['Action']=rew_df['Action'].astype(str)
        rew_df['Reword']=rew_df.apply(lambda x: rew[(x['State'],x['Action'])], axis=1)
        rew_df.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Reword\rew_df.csv'%(model),index=False)
   
        a = np.zeros(shape=(len(state_action_state),4))
        trans_df=pd.DataFrame(data=a,columns=['State','Action','Next State','Transitions'])
        trans_df['State']=trans_df.apply(lambda x: state_action_state[x.name][0][0], axis=1)
        trans_df['Action']=trans_df.apply(lambda x: state_action_state[x.name][0][1], axis=1)
        trans_df['Next State']=trans_df.apply(lambda x: state_action_state[x.name][1], axis=1)
        trans_df['Transitions']=trans_df.apply(lambda x: trans[((x['State'],x['Action']),x['Next State'])], axis=1)
      
        trans_df['State']=trans_df['State'].astype(str)
        trans_df['Action']=trans_df['Action'].astype(str)
        trans_df['Next State']=trans_df['Next State'].astype(str)
        trans_df['Next State'] = trans_df['Next State'].str.replace("'", '')
        trans_df['Next State'] = trans_df['Next State'].str.replace(" ", '')
        trans_df.to_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Markov_decision_process\model_%s\Transitions\trans_df.csv'%(model),index=False)
        self.check_transition()

#%%

taz_info = pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Keys\zones_numbering.csv')
requst=pd.read_json(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Demand\Amod_schedule\requst_pm.json')
#%%

trajectory=pd.read_csv(r'C:\Users\dadashev\Dropbox\Optimizing_Mobility_with_Markovian_Model_for_AMoD\Data\Supply\path_with_kwh_all.csv')
#%%

data=MDP_Data(4,5,requst,taz_info,trajectory)
states=data.get_states()
actions=data.get_actions()
sub_state_actions=data.get_state_action_subset()
req=data.get_demand_table()
rew=data.get_reward()
state_action_state=data.get_next_state_from_state()
trans=data.get_transition()
